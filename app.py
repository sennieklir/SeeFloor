from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
# override=True: this project's own .env always wins over any stray
# environment variable left set elsewhere on the machine (e.g. an old key
# someone set globally in a shell profile). Without override=True,
# load_dotenv() silently keeps whatever was already in os.environ and
# ignores .env -- which is exactly the kind of "I changed .env but nothing
# happened" confusion this is meant to prevent.
load_dotenv(override=True)
import re
import uuid
import math
import time
import logging
import subprocess
import glob
import tempfile
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
import base64
import json
import io
import networkx as nx
from PIL import Image as PILImage
from wallgrid import (
    build_grid_from_segments, shortest_path_distance, path_to_pct_waypoints,
    path_to_exact_pct_points, render_deterministic_route_overlay,
)
from roboflow_cv import (
    call_roboflow_wall_door_detection,
    convert_predictions_to_wallgrid_input,
    merge_wall_door_sources,
    RoboflowResponseError,
)

app = Flask(__name__)
# Anchored to this file's own directory, NOT the process's current working
# directory. Flask always *serves* static files from the folder next to
# app.py (its root_path) regardless of where you launched `python app.py`
# from -- but a bare relative string like 'static/uploads' resolves against
# the cwd instead. If those two ever differ (different terminal, IDE run
# button, a script that cd's elsewhere first), uploads get written to the
# wrong folder: /upload still returns 200 (the write itself succeeds), but
# every later GET /static/uploads/<file> 404s because Flask is looking in a
# different directory than the one the file was actually saved to.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ── SAVED ANALYSES DATABASE (backs the "history" and "reports" sidebar panels) ──
# A saved analysis is a full snapshot of one /analyze run (rooms, clusters,
# evacuation paths, rule-based recommendations, sprinkler basis, etc.) plus
# whatever AI recommendations were generated for it. History shows the list
# of snapshots; Reports flattens every snapshot's recommendations into one
# feed so they can be reviewed without re-running an analysis.
DB_PATH = os.path.join(BASE_DIR, 'seefloor_history.db')


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS saved_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            building_name TEXT,
            building_risk_index REAL,
            building_risk_label TEXT,
            room_count INTEGER,
            high_risk_count INTEGER,
            cluster_count INTEGER,
            analysis_json TEXT NOT NULL,
            ai_recommendations_json TEXT,
            floorplan_images_json TEXT
        )
    ''')
    conn.commit()
    conn.close()


init_db()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("seefloor_debug.log")]
)
logger = logging.getLogger("seefloor")


def resolve_static_path(url_path):
    """
    Converts a URL like '/static/uploads/abc.jpg' (what /upload hands back to
    the browser, and what the browser later sends back to us) into an
    absolute filesystem path anchored to BASE_DIR.

    This replaces the old `filepath.lstrip('/')` pattern used across the
    analyze/recommend routes. That pattern produced a *relative* path
    ('static/uploads/abc.jpg'), which -- exactly like the old UPLOAD_FOLDER
    bug -- only resolves correctly if the process's current working
    directory happens to match app.py's own folder. When it doesn't, the
    file genuinely exists and is even viewable in the browser (Flask serves
    static/ from BASE_DIR just fine), but os.path.exists()/open() here would
    look in the wrong place and fail as if the file were missing -- which is
    exactly what "Analysis failed for all floors" looks like: every floor's
    file "not found" before the code ever gets a chance to call GPT-4o.
    """
    if not url_path:
        return ''
    relative = url_path.lstrip('/')
    return os.path.join(BASE_DIR, relative)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": f"Bad request: {str(e)}"}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": f"Not found: {str(e)}"}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.errorhandler(Exception)
def unhandled_exception(e):
    # Previously this returned a generic message with no logging at all --
    # any bug outside the per-GPT-call try/excepts (which do log) would
    # crash silently: nothing in seefloor_debug.log to tell you what broke,
    # just a bare 500 in the request log. Now it's always logged with a
    # full traceback so a future "why did this fail" has an actual answer.
    logger.exception("Unhandled exception on %s %s", request.method, request.path)
    return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def scale_confidence_from_source(scale_source):
    return scale_source == "estimated"


def _sanitize_markers(raw_markers):
    """Validates and clips user-placed exit/stair/elevator/door markers
    coming from the frontend's pre-analysis marker tool. Exit/stair/
    elevator markers are used two ways downstream: (1) as hints in GPT's
    Step-1 extraction prompt (see _marker_hint_block), and (2) merged
    directly into exits_list / stairwells_list / elevators_list as
    authoritative CV/Dijkstra targets (see _merge_marker_exits /
    _merge_marker_stairs / _merge_marker_elevators), so a room still
    routes to a user-marked exit, stairwell, or elevator even if GPT fails
    to echo it back in its own "exits"/"stairwells"/"elevators" list. Door
    markers remain hints only. We only guard against bad input here, not
    compute anything with it."""
    if not isinstance(raw_markers, list):
        return []
    cleaned = []
    for m in raw_markers[:30]:  # generous cap, well past what anyone would place by hand
        if not isinstance(m, dict):
            continue
        try:
            x_pct = max(0.0, min(100.0, float(m.get("x_pct"))))
            y_pct = max(0.0, min(100.0, float(m.get("y_pct"))))
        except (TypeError, ValueError):
            continue
        marker_type = m.get("type") if m.get("type") in ("exit", "stair", "elevator", "door") else None
        if marker_type is None:
            continue
        cleaned.append({"x_pct": round(x_pct, 2), "y_pct": round(y_pct, 2), "type": marker_type})
    return cleaned


def _sanitize_extinguisher_markers_for_overlay(raw_markers):
    """Validates user-placed fire-extinguisher markers submitted to
    /wallgrid-evacuation-plan specifically for drawing on the deterministic
    route-overlay image (see render_deterministic_route_overlay's
    extinguisher_markers param in wallgrid.py). Deliberately separate from
    _sanitize_markers (exit/stair/elevator/door -- authoritative Dijkstra
    hints) and from _sanitize_safety_markers (the /analyze-wide, floor_level
    -tagged version feeding generate_marker_recommendations()): this one is
    scoped to a single floor's request and is purely a drawing input -- it
    never touches routing, distance, or recommendation logic."""
    if not isinstance(raw_markers, list):
        return []
    cleaned = []
    for m in raw_markers[:60]:
        if not isinstance(m, dict):
            continue
        try:
            x_pct = max(0.0, min(100.0, float(m.get("x_pct"))))
            y_pct = max(0.0, min(100.0, float(m.get("y_pct"))))
        except (TypeError, ValueError):
            continue
        cleaned.append({"x_pct": round(x_pct, 2), "y_pct": round(y_pct, 2)})
    return cleaned


def _sanitize_wall_segments(raw_walls):
    """Validates wall line segments coming from the frontend's geometry
    review/repair tool (see /detect-geometry and the geometry_override
    branch of /ai-analyze-multi below). Same shape GPT's own Step-1 trace
    already produces ({x1_pct,y1_pct,x2_pct,y2_pct}), so a user-edited wall
    list is indistinguishable, downstream, from an AI-detected one -- it
    flows into build_grid_from_segments() exactly the same way. We only
    guard against bad input here, not compute anything with it."""
    if not isinstance(raw_walls, list):
        return []
    cleaned = []
    for w in raw_walls[:600]:  # generous cap, well past any real floor plan's wall count
        if not isinstance(w, dict):
            continue
        try:
            x1 = max(0.0, min(100.0, float(w.get("x1_pct"))))
            y1 = max(0.0, min(100.0, float(w.get("y1_pct"))))
            x2 = max(0.0, min(100.0, float(w.get("x2_pct"))))
            y2 = max(0.0, min(100.0, float(w.get("y2_pct"))))
        except (TypeError, ValueError):
            continue
        cleaned.append({
            "x1_pct": round(x1, 2), "y1_pct": round(y1, 2),
            "x2_pct": round(x2, 2), "y2_pct": round(y2, 2),
        })
    return cleaned


def _sanitize_door_points(raw_doors):
    """Validates door points coming from the frontend's geometry
    review/repair tool -- see _sanitize_wall_segments above, same rationale."""
    if not isinstance(raw_doors, list):
        return []
    cleaned = []
    for d in raw_doors[:200]:
        if not isinstance(d, dict):
            continue
        try:
            x_pct = max(0.0, min(100.0, float(d.get("x_pct"))))
            y_pct = max(0.0, min(100.0, float(d.get("y_pct"))))
        except (TypeError, ValueError):
            continue
        orientation = d.get("orientation") if d.get("orientation") in ("horizontal", "vertical") else "horizontal"
        cleaned.append({"x_pct": round(x_pct, 2), "y_pct": round(y_pct, 2), "orientation": orientation})
    return cleaned


def _sanitize_stair_footprints(raw_footprints):
    """Validates staircase footprint rectangles -- same rationale as
    _sanitize_wall_segments. Entry point fields are optional (a footprint
    without one just doesn't get an entry gap punched, same as GPT
    omitting it today)."""
    if not isinstance(raw_footprints, list):
        return []
    cleaned = []
    for sf in raw_footprints[:20]:
        if not isinstance(sf, dict):
            continue
        try:
            x_min = max(0.0, min(100.0, float(sf.get("x_min_pct"))))
            y_min = max(0.0, min(100.0, float(sf.get("y_min_pct"))))
            x_max = max(0.0, min(100.0, float(sf.get("x_max_pct"))))
            y_max = max(0.0, min(100.0, float(sf.get("y_max_pct"))))
        except (TypeError, ValueError):
            continue
        entry = {}
        try:
            entry["entry_x_pct"] = round(max(0.0, min(100.0, float(sf.get("entry_x_pct")))), 2)
            entry["entry_y_pct"] = round(max(0.0, min(100.0, float(sf.get("entry_y_pct")))), 2)
        except (TypeError, ValueError):
            pass
        cleaned.append({
            "name": str(sf.get("name") or "stairwell")[:60],
            "x_min_pct": round(x_min, 2), "y_min_pct": round(y_min, 2),
            "x_max_pct": round(x_max, 2), "y_max_pct": round(y_max, 2),
            **entry,
        })
    return cleaned


def _sanitize_named_points(raw_points):
    """Validates named point lists (exits/stairwells/elevators) coming
    back from the geometry review tool unchanged by the user -- same
    {name,x_pct,y_pct} shape scale_info already carries."""
    if not isinstance(raw_points, list):
        return []
    cleaned = []
    for p in raw_points[:30]:
        if not isinstance(p, dict):
            continue
        try:
            x_pct = max(0.0, min(100.0, float(p.get("x_pct"))))
            y_pct = max(0.0, min(100.0, float(p.get("y_pct"))))
        except (TypeError, ValueError):
            continue
        cleaned.append({"name": str(p.get("name") or "")[:60], "x_pct": round(x_pct, 2), "y_pct": round(y_pct, 2)})
    return cleaned


def _marker_hint_block(markers):
    """Builds the prompt fragment that tells GPT where the user has already
    pointed at exits/stairs/elevators/doors, without touching anything else
    in the prompt. Exit/stair/elevator markers hint at WHAT's there; door
    markers hint at WHERE an opening in a wall is -- the most common
    failure modes seen in practice are a missed exit/stairwell/elevator
    (fixed by exit/stair/elevator markers) and a missed or misplaced door
    forcing Dijkstra into a bogus detour or leaving a room unreachable
    (fixed by door markers)."""
    if not markers:
        return ""
    exit_stair_elevator = [m for m in markers if m["type"] in ("exit", "stair", "elevator")]
    doors = [m for m in markers if m["type"] == "door"]
    block = ""

    if exit_stair_elevator:
        lines = "\n".join(
            f'  - {m["type"]} near x_pct={m["x_pct"]}, y_pct={m["y_pct"]}'
            for m in exit_stair_elevator
        )
        block += f"""

The user has manually marked the following point(s) on this image as exits,
staircases, or elevators before analysis, to help you find them:
{lines}

Treat each marked point as a high-confidence hint, not a final answer: look at
the actual image near that point, confirm there really is an exit door, a
staircase, or an elevator there, and report its precise x_pct/y_pct (which
may differ slightly from the marker) in your "exits", "stairwells", or
"elevators" list, matching the marker's type. If a marker doesn't line up
with anything real nearby, use your own visual judgement instead of forcing a
match. These markers are hints to reduce missed or misidentified exits/
stairs/elevators -- they are not an exhaustive list, so still report any
other exits, stairwells, or elevators you can see that weren't marked."""

    if doors:
        lines = "\n".join(
            f'  - door/opening near x_pct={m["x_pct"]}, y_pct={m["y_pct"]}'
            for m in doors
        )
        block += f"""

The user has also marked the following point(s) as a door or wall opening
that your wall-tracing needs to get right:
{lines}

At each marked point, look closely at the wall(s) immediately around it.
There should be a gap in the wall there (a door swing arc, or a visible break
in the line). Make sure your "doors" list includes an entry at that gap, and
double check the "walls" segments on either side of it don't accidentally
seal it shut (e.g. one continuous wall line drawn straight through where the
gap actually is). A wall that's missing its door, or a door that's missing
entirely, is the single most common cause of a room appearing unreachable or
of a walking path being forced into a long, wrong detour -- these markers
exist specifically to catch that."""

    return block


def _merge_marker_points(target_list, floor_markers, marker_type, label, dedup_pct_radius=3.0):
    """Merges the user's manually-placed markers of `marker_type` directly
    into `target_list` as first-class CV/Dijkstra targets, instead of
    relying on GPT to have echoed them back correctly in its own
    exits/stairwells list.

    Why this is needed: _marker_hint_block already tells GPT "here's where
    the user says an exit/staircase is", but that's just a prompt hint --
    GPT can still omit the point, misjudge what it leads to, or report
    slightly-off coordinates that miss the actual doorway/landing on the
    wallgrid. When that happens, downstream distance lookups only ever see
    GPT's (incomplete) list, so a room right next to a real, user-marked
    exit or stairwell can end up routed to a much farther one that GPT DID
    report -- e.g. the "Service returned 14.0 even though it's already
    outside" failure mode on the ground floor, and the same thing can
    happen to a stairwell on an upper floor (a room right next to the
    marked stairs getting routed the long way to a different one GPT
    found). Feeding marker points in directly makes them authoritative
    targets regardless of what GPT did with the hint, and since the
    nearest-target lookup already picks the nearest REACHABLE point out of
    a list, this also gives "route to nearest of several exits/stairwells"
    support for free -- works the same whether there's 1 marked point or
    10.

    Marker points that land within dedup_pct_radius percent of a point GPT
    already reported are skipped, so the same real-world exit/stairwell
    doesn't show up twice under two names/coordinates.
    """
    merged = list(target_list or [])
    if not floor_markers:
        return merged

    existing_pts = [
        (e.get("x_pct"), e.get("y_pct")) for e in merged
        if e.get("x_pct") is not None and e.get("y_pct") is not None
    ]

    matching_markers = [m for m in floor_markers if m.get("type") == marker_type]
    for i, m in enumerate(matching_markers, start=1):
        mx, my = m["x_pct"], m["y_pct"]
        is_dup = any(
            ((mx - ex) ** 2 + (my - ey) ** 2) ** 0.5 <= dedup_pct_radius
            for ex, ey in existing_pts
        )
        if is_dup:
            continue
        merged.append({
            "name": f"user-marked {label} {i}" if len(matching_markers) > 1 else f"user-marked {label}",
            "x_pct": mx,
            "y_pct": my,
            "source": "user_marker",
        })
        existing_pts.append((mx, my))

    return merged


def _merge_marker_exits(exits_list, floor_markers, dedup_pct_radius=3.0):
    """Exit-specific wrapper around _merge_marker_points -- see that
    docstring for the full rationale. Used on the ground floor (and any
    floor with its own direct-to-outside doors)."""
    return _merge_marker_points(exits_list, floor_markers, "exit", "exit", dedup_pct_radius)


def _merge_marker_stairs(stairwells_list, floor_markers, dedup_pct_radius=3.0):
    """Stairwell-specific wrapper around _merge_marker_points -- see that
    docstring for the full rationale. Used on upper floors, where rooms
    route to the nearest stairwell/elevator rather than directly to an
    exit; the exact same "GPT missed/mislabeled the marked point" failure
    mode applies here too (a room can get routed the long way to a
    different stairwell GPT reported instead of the one right next to it
    that the user marked)."""
    return _merge_marker_points(stairwells_list, floor_markers, "stair", "stairwell", dedup_pct_radius)


def _merge_marker_elevators(elevators_list, floor_markers, dedup_pct_radius=3.0):
    """Elevator-specific wrapper around _merge_marker_points -- mirrors
    _merge_marker_stairs exactly, just targeting elevators_list instead of
    stairwells_list. Used on upper floors alongside stairwells: rooms route
    to whichever vertical transition (stairwell or elevator) gives the
    shorter navigable path (see verify_room_distance's transition_sets),
    and the same "GPT missed/mislabeled the marked point" failure mode
    applies here too -- a room can end up routed to a farther stairwell
    GPT reported instead of the elevator right next to it that the user
    marked."""
    return _merge_marker_points(elevators_list, floor_markers, "elevator", "elevator", dedup_pct_radius)


def _nearest_room_name(x_pct, y_pct, rooms, max_pct_distance=25.0):
    """Finds the room_name of whichever room's centroid sits closest (in
    pct-space) to (x_pct, y_pct). Used to turn a generic user-marked
    exit/stairwell/elevator label into something a person can actually
    picture, e.g. "user-marked exit 1 (near Foyer)" instead of just
    "user-marked exit 1" -- naming a nearby room reads far more naturally
    in the plain-English evacuation guide than a bare compass direction.

    Returns None (leaving the caller's label untouched) if there's no room
    data to compare against, or if the closest room is implausibly far away
    (max_pct_distance, in the same 0-100 image-pct units as x_pct/y_pct) --
    that guards against confidently naming the wrong room when a marker
    sits well outside every known room's area.
    """
    best_name, best_dist = None, None
    for r in rooms or []:
        rx, ry = r.get("centroid_x_pct"), r.get("centroid_y_pct")
        rname = r.get("room_name")
        if rx is None or ry is None or not rname:
            continue
        dist = ((x_pct - rx) ** 2 + (y_pct - ry) ** 2) ** 0.5
        if best_dist is None or dist < best_dist:
            best_dist, best_name = dist, rname
    if best_name is None or (max_pct_distance is not None and best_dist > max_pct_distance):
        return None
    return best_name


def _label_markers_with_nearest_room(points_list, rooms):
    """Appends "(near <RoomName>)" to a user-marked exit/stairwell/
    elevator's name, based on the closest room's centroid on this same
    floor -- see _nearest_room_name(). Only touches entries whose name
    starts with "user-marked" (i.e. came from _merge_marker_points); GPT's
    own named exits (e.g. "front door") are left as-is since they're
    already descriptive. Mutates and returns points_list; safe to call with
    an empty/None rooms list (no-op)."""
    if not points_list or not rooms:
        return points_list
    for p in points_list:
        name = p.get("name") or ""
        if not name.startswith("user-marked") or "(near " in name:
            continue
        nearest = _nearest_room_name(p.get("x_pct"), p.get("y_pct"), rooms)
        if nearest:
            p["name"] = f"{name} (near {nearest})"
    return points_list


# ---------------------------------------------------------------------------
# FIRE EXTINGUISHER / WINDOW MARKERS (recommendation-only)
#
# Deliberately kept separate from _sanitize_markers/_marker_hint_block/
# _merge_marker_points above: those exist/stair/door markers feed GPT's
# Step-1 extraction prompt and get merged into exits_list/stairwells_list as
# authoritative Dijkstra targets. Fire extinguisher and window markers do
# neither -- they never touch GPT's prompt, wall-tracing, or distance/graph
# math. They're submitted alongside the already-computed room list to
# /analyze and only ever influence generate_marker_recommendations() below.
# ---------------------------------------------------------------------------

SAFETY_MARKER_TYPES = ("fire_extinguisher", "window")

# Same high-hazard occupancy list generate_recommendations() already checks
# for the "install automatic fire suppression" recommendation -- reused here
# rather than redefined, so the two stay in sync.
_HAZARD_ROOM_TYPES = ('electrical_room', 'kitchen', 'gas_storage', 'dirty_kitchen')


def _sanitize_safety_markers(raw_markers):
    """Validates user-placed fire-extinguisher/window markers submitted with
    the /analyze payload (one flat list across all floors, each tagged with
    its own floor_level -- unlike the per-floor floor_markers used by
    _sanitize_markers). Only guards against bad input; computes nothing."""
    if not isinstance(raw_markers, list):
        return []
    cleaned = []
    for m in raw_markers[:60]:  # generous cap, well past what anyone would place by hand
        if not isinstance(m, dict):
            continue
        try:
            x_pct = max(0.0, min(100.0, float(m.get("x_pct"))))
            y_pct = max(0.0, min(100.0, float(m.get("y_pct"))))
            floor_level = int(m.get("floor_level"))
        except (TypeError, ValueError):
            continue
        marker_type = m.get("type") if m.get("type") in SAFETY_MARKER_TYPES else None
        if marker_type is None:
            continue
        cleaned.append({
            "x_pct": round(x_pct, 2),
            "y_pct": round(y_pct, 2),
            "floor_level": floor_level,
            "type": marker_type,
        })
    return cleaned


def _nearest_room_for_marker(marker, rooms_analyzed, max_pct_radius=12.0):
    """Finds the room whose centroid_x_pct/centroid_y_pct is closest to a
    safety marker, restricted to the same floor and to within
    max_pct_radius percent of the image. Rooms with no centroid on file
    (e.g. a manually-typed room with no AI-detected coordinates) simply
    can't be matched and are skipped -- this never raises, it returns None
    if nothing on that floor is close enough.

    On a miss, logs WHY at info level (DIAG marker-match) instead of just
    silently returning None -- floor-level mismatches between the marker
    tool and the room table are the single most common cause of "I marked
    it and it's still not showing up," and without this there was no way
    to tell that apart from "it really is just too far away" short of
    guessing. Check seefloor_debug.log after a miss to see which one it was."""
    best_room = None
    best_dist = None
    same_floor_candidates = []
    for room in rooms_analyzed:
        if room.get('floor_level') != marker['floor_level']:
            continue
        same_floor_candidates.append(room)
        cx, cy = room.get('centroid_x_pct'), room.get('centroid_y_pct')
        if cx is None or cy is None:
            continue
        try:
            dist = ((marker['x_pct'] - float(cx)) ** 2 + (marker['y_pct'] - float(cy)) ** 2) ** 0.5
        except (TypeError, ValueError):
            continue
        if dist <= max_pct_radius and (best_dist is None or dist < best_dist):
            best_dist = dist
            best_room = room

    if best_room is None:
        if not same_floor_candidates:
            all_floors = sorted({room.get('floor_level') for room in rooms_analyzed})
            logger.info(
                "DIAG marker-match MISS: %s marker at floor_level=%r (%.1f,%.1f) -- "
                "ZERO rooms exist on that floor. Rooms only exist on floor_level(s)=%s. "
                "This is a floor-numbering mismatch between the marker tool and the room "
                "table, not a distance/radius problem.",
                marker.get('type'), marker.get('floor_level'),
                marker['x_pct'], marker['y_pct'], all_floors
            )
        else:
            missing_centroid = [
                r['room_name'] for r in same_floor_candidates
                if r.get('centroid_x_pct') is None or r.get('centroid_y_pct') is None
            ]
            logger.info(
                "DIAG marker-match MISS: %s marker at floor_level=%r (%.1f,%.1f) -- "
                "%d room(s) exist on that floor but NONE within %.1f%% radius. "
                "Candidates (name, centroid_x_pct, centroid_y_pct): %s. "
                "Of those, missing centroid entirely (unmatchable regardless of distance): %s",
                marker.get('type'), marker.get('floor_level'),
                marker['x_pct'], marker['y_pct'], len(same_floor_candidates), max_pct_radius,
                [(r['room_name'], r.get('centroid_x_pct'), r.get('centroid_y_pct')) for r in same_floor_candidates],
                missing_centroid or "none"
            )

    return best_room


def generate_marker_recommendations(rooms_analyzed, markers):
    """Purely additive companion to generate_recommendations(): folds the
    user's placed fire-extinguisher/window markers into the recommendations
    list without touching generate_recommendations() itself or anything it
    already computes. Returns [] (no-op) when no markers were submitted, so
    existing callers that never send markers see no change in behavior."""
    recs = []
    if not markers:
        return recs

    extinguisher_markers = [m for m in markers if m['type'] == 'fire_extinguisher']
    window_markers = [m for m in markers if m['type'] == 'window']

    if extinguisher_markers:
        covered_rooms = set()
        orphan_marker_floors = set()  # floors with an extinguisher marker but ZERO rooms recorded there
        room_floors_present = {r.get('floor_level') for r in rooms_analyzed}
        for m in extinguisher_markers:
            room = _nearest_room_for_marker(m, rooms_analyzed)
            if room:
                covered_rooms.add(room['room_name'])
            elif m['floor_level'] not in room_floors_present:
                orphan_marker_floors.add(m['floor_level'])

        if covered_rooms:
            recs.append({
                "priority": "Compliant",
                "message": (
                    f"Fire extinguisher marker(s) placed near: {', '.join(sorted(covered_rooms))}. "
                    "This supports compliance with Fire Code PD 1185's portable fire "
                    "extinguisher requirements for these spaces."
                ),
                "icon": "🧯"
            })

        hazard_rooms = [r for r in rooms_analyzed if r['room_type'] in _HAZARD_ROOM_TYPES]
        uncovered = sorted({
            r['room_name'] for r in hazard_rooms if r['room_name'] not in covered_rooms
        })
        if uncovered:
            # Surface the floor-mismatch case explicitly (marker placed on a
            # floor number that no room in the table is tagged with) instead
            # of letting it look identical to "genuinely too far away" --
            # this is the single most common cause of a marker silently not
            # counting, especially right after adding/reordering floors.
            mismatch_hint = ""
            if orphan_marker_floors:
                mismatch_hint = (
                    f" ⚠️ {len(orphan_marker_floors)} extinguisher marker(s) are placed on "
                    f"floor(s) {sorted(orphan_marker_floors)}, but no room in the table is "
                    "tagged with that floor number -- double-check the floor number field "
                    "matches between where you placed the marker and where the room is listed."
                )
            recs.append({
                "priority": "High",
                "message": (
                    f"No fire extinguisher marker found near {len(uncovered)} high-hazard "
                    f"room(s): {', '.join(uncovered)}. Fire Code PD 1185 requires portable "
                    "fire extinguishers in high-hazard occupancies such as kitchens, "
                    "electrical rooms, and gas storage areas."
                    f"{mismatch_hint}"
                ),
                "icon": "🧯"
            })

    if window_markers:
        matched_rooms = sorted({
            room['room_name']
            for room in (_nearest_room_for_marker(m, rooms_analyzed) for m in window_markers)
            if room
        })
        recs.append({
            "priority": "Informational",
            "message": (
                f"{len(window_markers)} window(s) marked"
                + (f" in: {', '.join(matched_rooms)}" if matched_rooms else "")
                + ". Per NBC PD 1096, a window only counts toward required egress if it's "
                "clearly labeled and sized as an emergency egress window -- confirm these "
                "aren't being relied on in place of a required exit or stairwell."
            ),
            "icon": "🪟"
        })

    return recs


# ---------------------------------------------------------------------------
# WALL / CONSTRUCTION TYPE (per-room, recommendation-only)
#
# NBC PD 1096 Sections 401-403 classify buildings into Types I-IV by their
# structural/wall material, each with its own fire-resistive requirement:
#   Type I   - wood, no fire-resistive requirement
#   Type II  - wood w/ fire-retardant treatment, 1-hour fire-resistive
#   Type III - masonry + wood, 1-hour fire-resistive, incombustible exterior
#   Type IV  - steel/iron/concrete/masonry
# A building can legitimately mix these room-to-room (e.g. a masonry
# firewall around a kitchen in an otherwise wood-frame house), so this is
# captured per room rather than once per building -- same reasoning as the
# fire-extinguisher/window markers above. Entirely separate from hazard
# scoring: construction_type never touches compute_hazard_index/classify_risk,
# it only ever feeds generate_construction_recommendations() below.
# ---------------------------------------------------------------------------

CONSTRUCTION_TYPES = ("type_1", "type_2", "type_3", "type_4", "not_sure")

_CONSTRUCTION_LABELS = {
    "type_1": "Type I (wood)",
    "type_2": "Type II (wood, fire-retardant treated)",
    "type_3": "Type III (masonry + wood)",
    "type_4": "Type IV (steel/iron/concrete/masonry)",
}


def _sanitize_construction_type(value):
    """Falls back to 'not_sure' for anything missing/unrecognized, so a room
    with no construction_type set (e.g. every room submitted by an old
    client that predates this field) is simply skipped by
    generate_construction_recommendations() rather than guessed at."""
    return value if value in CONSTRUCTION_TYPES else "not_sure"


def generate_construction_recommendations(rooms_analyzed):
    """Companion helper, called from inside generate_recommendations() below
    (see the call near its end) so wall-construction data actually reaches
    the recommendations list. Flags high-hazard rooms (same
    _HAZARD_ROOM_TYPES list the fire-suppression check already uses)
    sitting in wood-based construction that hasn't been upgraded to a
    fire-resistive rating, per NBC PD 1096 Sections 401-403. This is a code
    requirement independent of the computed hazard/risk score, so it is NOT
    gated on risk_color -- a short-travel-distance kitchen with wood walls
    still needs to be flagged. Rooms with construction_type == 'not_sure'
    (the default -- see _sanitize_construction_type) are skipped rather
    than assumed compliant or non-compliant. Returns [] when nothing is
    flaggable, so callers that never set construction_type see no change
    in behavior."""
    recs = []

    wood_no_rating = sorted({
        r['room_name'] for r in rooms_analyzed
        if r['room_type'] in _HAZARD_ROOM_TYPES
        and r.get('construction_type') == 'type_1'
    })
    if wood_no_rating:
        recs.append({
            "priority": "Critical",
            "message": (
                f"{len(wood_no_rating)} high-hazard room(s) -- {', '.join(wood_no_rating)} -- "
                "are in Type I (wood) construction with no fire-resistive rating. "
                "NBC PD 1096 Sections 401-403 require this occupancy type to be upgraded "
                "to at least Type III (1-hour fire-resistive) construction, or the hazard "
                "isolated behind fire-resistive walls."
            ),
            "icon": "🪵"
        })

    wood_treated = sorted({
        r['room_name'] for r in rooms_analyzed
        if r['room_type'] in _HAZARD_ROOM_TYPES
        and r.get('construction_type') == 'type_2'
    })
    if wood_treated:
        recs.append({
            "priority": "Moderate",
            "message": (
                f"{len(wood_treated)} high-hazard room(s) -- {', '.join(wood_treated)} -- "
                "are in Type II (fire-retardant treated wood) construction. This meets the "
                "1-hour fire-resistive minimum per NBC PD 1096 Sections 401-403, but confirm "
                "the walls enclosing this specific room weren't part of a non-bearing "
                "partition exempted from that treatment."
            ),
            "icon": "🪵"
        })

    masonry_wood = sorted({
        r['room_name'] for r in rooms_analyzed
        if r['room_type'] in _HAZARD_ROOM_TYPES
        and r.get('construction_type') == 'type_3'
    })
    if masonry_wood:
        recs.append({
            "priority": "Informational",
            "message": (
                f"{len(masonry_wood)} high-hazard room(s) -- {', '.join(masonry_wood)} -- "
                "are in Type III (masonry + wood) construction. NBC PD 1096 Sections 401-403 "
                "require this type to be 1-hour fire-resistive throughout with incombustible "
                "exterior walls -- verify this room's walls meet that rating."
            ),
            "icon": "🧱"
        })

    return recs


# ---------------------------------------------------------------------------
# CV pathfinding fallback/validator (wallgrid.py)
# ---------------------------------------------------------------------------
# GPT-4o supplies wall/door geometry (semantic recognition, its strength);
# wallgrid.py's Dijkstra-over-rasterized-grid computes the actual walking
# distance (deterministic geometry, not LLM arithmetic). See
# seefloor_pathfinding_module/INTEGRATION.md for the full rationale and the
# earlier pixel-threshold attempt that this replaced.
#
# These constants are specified in meters and converted below to whichever
# unit the request is using, so they stay consistent with building_width_m/
# building_depth_m (which the rest of this file already treats as being in
# the request's chosen unit despite the "_m" suffix -- see the existing
# fallback_w/fallback_d conversion a few lines below).
CV_WALL_THICKNESS_M = 0.15
CV_DOOR_WIDTH_M = 0.9
CV_CELL_SIZE_M = 0.1
CV_STAIR_DESCENT_M = 3.5          # one flight, matches the GPT prompt's constant
CV_ELEVATOR_DESCENT_M = 2.5       # shorter and more realistic than a stair descent
CV_DISCREPANCY_THRESHOLD_M = 2.0  # gap between GPT estimate and CV distance worth flagging
CV_MAX_DETOUR_RATIO = 3.0         # a real walking path rarely exceeds ~3x straight-line
CV_DETOUR_SLACK_M = 3.0           # + flat slack so short straight-lines aren't over-penalized


def _straight_line_dist(pct_a, pct_b, bw, bd):
    """Straight-line (not walking) distance between two x_pct/y_pct points,
    scaled into real units via the building's width/depth. Used only as a
    sanity ceiling on the CV walking distance, not as a distance itself --
    a doorless direct line is never the actual path."""
    if None in pct_a or None in pct_b or bw is None or bd is None:
        return None
    dx = (pct_b[0] - pct_a[0]) / 100.0 * bw
    dy = (pct_b[1] - pct_a[1]) / 100.0 * bd
    return (dx ** 2 + dy ** 2) ** 0.5


def _normalize_pct_point(point):
    """Clamps percentage coordinates to the valid 0-100 range and rejects invalid values."""
    if point is None:
        return None
    try:
        x_pct, y_pct = point
    except (TypeError, ValueError):
        return None
    try:
        x_pct = float(x_pct)
        y_pct = float(y_pct)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x_pct) and math.isfinite(y_pct)):
        return None
    return (max(0.0, min(100.0, x_pct)), max(0.0, min(100.0, y_pct)))


def _build_floor_grid(scale_info, bw, bd, unit_is_ft, floor_label, logger_):
    """Builds the wallgrid occupancy grid for one floor from GPT-supplied
    wall/door segments. Returns None (rather than raising) if there isn't
    enough geometry to build a meaningful grid, so the caller can fall back
    to GPT's own estimate without the CV layer for that floor."""
    walls = scale_info.get("walls") or []
    doors = scale_info.get("doors") or []
    stair_footprints = scale_info.get("stair_footprints") or []
    if not walls:
        logger_.info("Floor %s: no wall segments extracted, skipping CV validation.", floor_label)
        return None

    unit_factor = (1.0 / FEET_TO_METERS) if unit_is_ft else 1.0
    try:
        return build_grid_from_segments(
            walls, doors,
            building_width_m=bw,
            building_depth_m=bd,
            cell_size_m=CV_CELL_SIZE_M * unit_factor,
            wall_thickness_m=CV_WALL_THICKNESS_M * unit_factor,
            door_width_m=CV_DOOR_WIDTH_M * unit_factor,
            stair_footprints=stair_footprints,
            entry_width_m=CV_DOOR_WIDTH_M * unit_factor,
        )
    except Exception as e:
        logger_.warning("Floor %s: wallgrid build failed (%s), skipping CV validation.", floor_label, e)
        return None


def _nearest_cv_distance(grid_result, start_pct, targets):
    """Returns (distance, target, path) for the closest of `targets` (a
    list of dicts with x_pct/y_pct) reachable from start_pct on
    grid_result, or (None, None, None) if none are reachable or no
    targets/start are available. `path` is the raw grid-cell Dijkstra path
    to that target (see wallgrid.path_to_pct_waypoints for turning it into
    overlay-friendly pct waypoints) -- kept here mainly so the deterministic
    evacuation-route overlay can show roughly where this route actually
    goes, not just how long it is."""
    if grid_result is None or not targets or start_pct[0] is None or start_pct[1] is None:
        return None, None, None
    best_dist, best_target, best_path = None, None, None
    for t in targets:
        if t.get("x_pct") is None or t.get("y_pct") is None:
            continue
        dist, path = shortest_path_distance(grid_result, start_pct, (t["x_pct"], t["y_pct"]))
        if dist is not None and (best_dist is None or dist < best_dist):
            best_dist, best_target, best_path = dist, t, path
    return best_dist, best_target, best_path


def verify_room_distance(room, floor_label, grid_result, exits_list, stairwells_list,
                          ground_floor_grid, unit_is_ft, bw, bd, floor1_bw=None, floor1_bd=None,
                          sanity_max_distance=None, elevators_list=None, ground_floor_exits_list=None):
    """Computes the deterministic (wallgrid + Dijkstra) distance-to-exit for
    one room and attaches it to the room dict alongside GPT's own estimate,
    following the two-source validation pattern in INTEGRATION.md:

      - room['distance_to_exit_gpt']  -> GPT-4o's own arithmetic estimate
      - room['distance_to_exit_cv']   -> deterministic graph distance (or None)
      - room['distance_source']       -> 'graph_verified' | 'gpt_estimate_only'
                                          | 'cv_rejected_implausible'
      - room['distance_discrepancy_flag'] -> True if the two disagree by more
        than CV_DISCREPANCY_THRESHOLD_M (converted to the active unit)

    When a CV distance is available AND passes a sanity check, it REPLACES
    distance_to_exit -- the deterministic graph computation is trusted over
    the LLM's mental arithmetic, per INTEGRATION.md Step 4.

    IMPORTANT CAVEAT (see INTEGRATION.md's "Known limitations" #1): that
    trust assumes GPT's wall/door TRACING was accurate. It sometimes isn't
    -- a missed or misplaced door can force Dijkstra into a long, "valid but
    wrong" detour, and that detour can still be well under the building's
    outer diagonal (so a diagonal-only check misses it) while still being
    obviously too long for the room's actual position. The real sanity
    check here is against the STRAIGHT-LINE distance to the target: normal
    walking paths through doors/hallways are rarely more than
    ~CV_MAX_DETOUR_RATIO x the straight-line distance (plus a flat slack for
    short hops). A path that blows past that ratio is far more likely to be
    a broken graph (missed door forcing a loop around the house) than a
    genuinely convoluted floor plan, so it is NOT trusted: it's surfaced as
    'cv_rejected_implausible' for visibility, and distance_to_exit falls
    back to GPT's own estimate instead of being overwritten with a number
    that's provably a detour artifact.
    """
    gpt_distance = room.get("distance_to_exit")
    room["distance_to_exit_gpt"] = gpt_distance

    start_pct = _normalize_pct_point((room.get("centroid_x_pct"), room.get("centroid_y_pct")))
    if start_pct is None:
        room["distance_to_exit_cv"] = None
        room["cv_target_used"] = None
        room["nearest_exit_used"] = None
        room["distance_source"] = "gpt_estimate_only"
        room["route_waypoints_pct"] = None
        room["route_path_exact_pct"] = None
        return room

    unit_factor = (1.0 / FEET_TO_METERS) if unit_is_ft else 1.0
    threshold = CV_DISCREPANCY_THRESHOLD_M * unit_factor
    detour_slack = CV_DETOUR_SLACK_M * unit_factor

    cv_distance = None
    cv_target_name = None
    straight_line_total = None
    best_transition_penalty = None
    cv_route_path = None       # raw grid-cell path(s) for the winning route, for waypoint hints
    cv_route_grids = None      # matching WallGridResult(s) to convert cv_route_path with

    if floor_label == 1:
        cv_distance, target, cv_path = _nearest_cv_distance(grid_result, start_pct, exits_list)
        if target is not None:
            cv_target_name = target.get("name")
            straight_line_total = _straight_line_dist(
                start_pct, (target.get("x_pct"), target.get("y_pct")), bw, bd
            )
            cv_route_path = [cv_path]
            cv_route_grids = [grid_result]
    else:
        # Upper floor: compare every available vertical transition (stairs or
        # elevator) and choose the route with the shortest navigable distance.
        transition_sets = []
        if stairwells_list:
            transition_sets.append(("stairwell", stairwells_list, CV_STAIR_DESCENT_M * unit_factor))
        if elevators_list:
            transition_sets.append(("elevator", elevators_list, CV_ELEVATOR_DESCENT_M * unit_factor))

        landing_targets = ground_floor_exits_list or exits_list
        logger.info(
            "DIAG floor %s room %r: transition_sets=%s ground_floor_grid=%s landing_targets=%d",
            floor_label, room.get("room_name"),
            [(k, len(v)) for k, v, _ in transition_sets],
            ground_floor_grid is not None, len(landing_targets or []),
        )
        if ground_floor_grid is not None and landing_targets and transition_sets:
            best_total = None
            best_target_name = None
            best_exit_name = None
            best_penalty = None
            best_straight_line_total = None
            best_transit_path = None
            best_landing_path = None

            for transition_kind, transitions, transition_penalty in transition_sets:
                for transition in transitions:
                    transition_pct = _normalize_pct_point((transition.get("x_pct"), transition.get("y_pct")))
                    if transition_pct is None:
                        continue

                    transit_dist, transit_path = shortest_path_distance(grid_result, start_pct, transition_pct)
                    if transit_dist is None:
                        logger.info(
                            "DIAG floor %s room %r: UNREACHABLE to %s %r at (%.1f,%.1f) on this floor's grid",
                            floor_label, room.get("room_name"), transition_kind,
                            transition.get("name"), transition_pct[0], transition_pct[1],
                        )
                        continue

                    landing_targets = ground_floor_exits_list or exits_list
                    landing_dist, exit_, landing_path = _nearest_cv_distance(
                        ground_floor_grid, transition_pct, landing_targets
                    )
                    if landing_dist is None or exit_ is None:
                        logger.info(
                            "DIAG floor %s room %r: %s %r landing UNREACHABLE to any ground-floor exit "
                            "(landing_targets=%s)",
                            floor_label, room.get("room_name"), transition_kind,
                            transition.get("name"), landing_targets,
                        )
                        continue

                    candidate_total = round(transit_dist + transition_penalty + landing_dist, 2)
                    if best_total is None or candidate_total < best_total:
                        best_total = candidate_total
                        best_target_name = transition.get("name", transition_kind)
                        best_exit_name = exit_.get("name", "exit")
                        best_penalty = transition_penalty
                        best_transit_path = transit_path
                        best_landing_path = landing_path
                        leg1 = _straight_line_dist(start_pct, transition_pct, bw, bd)
                        leg2 = _straight_line_dist(
                            transition_pct, (exit_.get("x_pct"), exit_.get("y_pct")),
                            floor1_bw or bw, floor1_bd or bd
                        )
                        if leg1 is not None and leg2 is not None:
                            best_straight_line_total = leg1 + leg2

            if best_total is not None:
                cv_distance = best_total
                cv_target_name = f"{best_target_name} -> {best_exit_name}"
                straight_line_total = best_straight_line_total
                best_transition_penalty = best_penalty
                # Two legs on two different grids (this floor, then ground
                # floor after the stair/elevator transition) -- kept as
                # separate (path, grid) pairs since each needs its own
                # grid's dimensions to convert cells back to pct.
                cv_route_path = [best_transit_path, best_landing_path]
                cv_route_grids = [grid_result, ground_floor_grid]

    room["distance_to_exit_cv"] = cv_distance
    room["cv_target_used"] = cv_target_name
    room["nearest_exit_used"] = cv_target_name

    ratio_bound = (
        straight_line_total * CV_MAX_DETOUR_RATIO + detour_slack + (best_transition_penalty if floor_label > 1 else 0)
        if straight_line_total is not None else None
    )
    implausible = cv_distance is not None and (
        (sanity_max_distance is not None and cv_distance > sanity_max_distance)
        or (ratio_bound is not None and cv_distance > ratio_bound)
    )

    if cv_distance is not None and not implausible:
        room["distance_source"] = "graph_verified"
        room["distance_to_exit"] = cv_distance
        if gpt_distance is not None:
            try:
                if abs(float(gpt_distance) - cv_distance) > threshold:
                    room["distance_discrepancy_flag"] = True
            except (TypeError, ValueError):
                pass
        # Only offer waypoint hints for routes we actually trust (graph_
        # verified, i.e. passed the same detour sanity check the distance
        # itself did) -- an implausible/rejected path would just be a bad
        # prior instead of no prior, which is worse than nothing.
        waypoints = []
        exact_points = []
        for p, g in zip(cv_route_path or [], cv_route_grids or []):
            if p and g is not None:
                waypoints.extend(path_to_pct_waypoints(p, g))
                # Lossless version of the same path, used by the
                # deterministic renderer (see render_deterministic_route_
                # overlay / path_to_exact_pct_points) so the drawn line
                # can't cut through a wall the way a straight chord
                # between two RDP-simplified corners sometimes could.
                exact_points.extend(path_to_exact_pct_points(p, g))
        room["route_waypoints_pct"] = waypoints or None
        room["route_path_exact_pct"] = exact_points or None
    elif implausible:
        # Dijkstra "succeeded" but the resulting path is far longer than the
        # straight-line distance would justify -- almost certainly a
        # missed/misplaced door forcing a bogus detour, not a real walking
        # distance. Don't let it overwrite GPT's estimate; keep GPT's
        # number as distance_to_exit and surface the rejected CV number for
        # review.
        room["distance_source"] = "cv_rejected_implausible"
        room["route_waypoints_pct"] = None
        room["route_path_exact_pct"] = None
    else:
        room["distance_source"] = "gpt_estimate_only"
        room["route_waypoints_pct"] = None
        room["route_path_exact_pct"] = None

    return room


def strip_markdown(text):
    if text.startswith("```"):
        text = '\n'.join([l for l in text.split('\n') if not l.startswith("```")])
    return text.strip()


class GPTResponseError(Exception):
    """Raised when GPT-4o returns something that cannot be turned into valid JSON."""
    pass


def _repair_truncated_json(raw_text):
    """
    Best-effort repair of JSON that got cut off mid-object because the
    response hit max_tokens (finish_reason == 'length'). Walks the string,
    tracks how many strings/arrays/objects are still open, and closes them
    so json.loads() has a chance of succeeding on a partial-but-valid object
    instead of failing outright. Ported/adapted from the THESIS DEMO branch.
    """
    text = raw_text.strip()
    depth_brace = 0
    depth_bracket = 0
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == '\\' and in_string:
            i += 2
            continue
        if c == '"':
            in_string = not in_string
        elif not in_string:
            if c == '{':
                depth_brace += 1
            elif c == '}':
                depth_brace -= 1
            elif c == '[':
                depth_bracket += 1
            elif c == ']':
                depth_bracket -= 1
        i += 1

    if in_string:
        text += '"'
    text = text.rstrip().rstrip(',')
    text += ']' * max(0, depth_bracket)
    text += '}' * max(0, depth_brace)
    return text


def extract_json(raw_text):
    """
    Robustly pull a JSON object out of an LLM text response.

    The old code only handled a "```json ... ```" fence, and only when the
    fence was the very first character of the string. In practice GPT-4o
    frequently prepends a sentence of commentary ("Here is the analysis for
    Floor 2:") before the fence, especially on more complex, multi-image
    prompts -- that leading text made json.loads() fail immediately with
    "Expecting value: line 1 column 1 (char 0)" because the parser never even
    reached the fenced block.

    Tries, in order:
      1. Direct json.loads on the stripped text (fast path).
      2. A fenced ```json ... ``` block found ANYWHERE in the text.
      3. The substring between the first "{" and the last "}" (handles stray
         prose before/after the object with no fences at all).
    Raises GPTResponseError with the offending text attached for logging if
    none of these produce valid JSON.
    """
    if raw_text is None:
        raise GPTResponseError("GPT returned no content (empty/None message.content)")

    text = raw_text.strip()
    if not text:
        raise GPTResponseError("GPT returned an empty string")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            raise GPTResponseError(
                f"Could not parse JSON even after extraction. Error: {e}. "
                f"Raw response (truncated): {text[:500]}"
            )

    raise GPTResponseError(f"No JSON object found in GPT response. Raw (truncated): {text[:500]}")


def call_gpt4o_json(content, max_tokens=2000, max_retries=2, temperature=0, label="request"):
    """
    Calls GPT_MODEL with response_format forced to JSON, retries on transient
    API errors and on malformed output, and always logs the raw text on
    failure so a bad response is debuggable instead of a bare 500.

    Also detects responses cut off by the token limit (finish_reason ==
    "length"): a mid-object cutoff makes json.loads() fail even though the
    model's *reasoning* was fine, it just ran out of room. When that happens
    this tries `_repair_truncated_json` to recover a partial-but-valid
    result, and grows max_tokens on the next attempt instead of resending
    the same too-small budget.

    Returns (parsed_dict, raw_text_of_last_successful_or_final_attempt, truncated).
    `truncated` is True if the JSON we returned came from a cut-off response
    (even if repair succeeded) so callers can warn the user that the result
    may be incomplete rather than silently trusting a patched object.
    """
    last_error = None
    current_max_tokens = max_tokens
    for attempt in range(1, max_retries + 2):  # max_retries=2 -> 3 total attempts
        try:
            response = client.chat.completions.create(
                **_gpt_completion_kwargs(
                    model=GPT_MODEL,
                    max_tokens=current_max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": content}],
                )
            )
            choice = response.choices[0]

            if getattr(choice, "finish_reason", None) == "content_filter":
                raise GPTResponseError(f"{label}: response blocked by content filter")

            raw_content = choice.message.content
            if raw_content is None:
                refusal = getattr(choice.message, "refusal", None)
                raise GPTResponseError(f"{label}: model returned no content (refusal={refusal!r})")

            truncated = getattr(choice, "finish_reason", None) == "length"

            try:
                return extract_json(raw_content), raw_content, truncated
            except GPTResponseError as parse_err:
                if truncated:
                    logger.warning(
                        "%s: response was cut off at %d tokens (finish_reason=length); "
                        "attempting repair before giving up.",
                        label, current_max_tokens
                    )
                    try:
                        repaired = extract_json(_repair_truncated_json(raw_content))
                        logger.warning("%s: recovered a partial result from a truncated response.", label)
                        return repaired, raw_content, True
                    except GPTResponseError:
                        pass  # fall through to normal retry-with-bigger-budget logic below

                logger.warning("%s: JSON parse failed on attempt %d/%d: %s",
                                label, attempt, max_retries + 1, parse_err)
                last_error = parse_err
                if truncated:
                    # sending the same max_tokens would just get cut off again
                    current_max_tokens = min(int(current_max_tokens * 1.5), 8000)
                elif isinstance(content, list):
                    content = content + [{
                        "type": "text",
                        "text": "Your previous reply was not valid JSON. Reply with ONLY the JSON "
                                "object. No prose, no markdown fences, no explanation before or after it."
                    }]

        except (APIError, APIConnectionError, RateLimitError) as api_err:
            logger.warning("%s: OpenAI API error on attempt %d/%d: %s",
                            label, attempt, max_retries + 1, api_err)
            last_error = api_err
            time.sleep(min(2 ** attempt, 8))

    logger.error("%s: exhausted all retries. Last error: %s", label, last_error)
    raise GPTResponseError(f"{label} failed after {max_retries + 1} attempts: {last_error}")


# ─────────────────────────────────────────────
# DPWH-Based Algorithm
# National Building Code of the Philippines PD 1096
# ─────────────────────────────────────────────

ROOM_TYPE_WEIGHTS = {
    # ── Institutional / commercial (original set) ──
    "classroom": 1.2,
    "laboratory": 1.8,
    "office": 1.0,
    "restroom": 0.8,
    "corridor": 0.6,
    "stairwell": 0.4,
    "storage": 1.5,
    "kitchen": 2.0,
    "electrical_room": 2.5,
    "server_room": 2.2,
    "conference": 1.1,
    "lobby": 0.7,

    # ── Residential (small-scale residential/mixed-use buildings) ──
    "bedroom": 1.4,        # sleeping area -- occupants are often unaware or
                           # slower to respond during a fire event, especially at night
    "living_room": 0.85,   # common/gathering area, usually closer to the main entrance
    "dining_room": 0.85,   # similar occupancy/egress profile to living_room
    "bathroom": 0.8,       # residential equivalent of restroom; low fuel load
    "garage": 1.4,         # stores vehicles, fuel, paint, and other flammables
    "carport": 0.9,        # open-sided vehicle parking; less enclosed than a garage
    "laundry_room": 1.6,   # dryers/heating elements are a common residential fire origin
    "utility_room": 1.5,   # water heaters, breaker panels, mechanical equipment
    "attic": 1.7,          # confined roof space, poor egress, wiring/insulation hazards
    "basement": 1.6,       # below-grade, limited egress routes, often flammable storage
    "closet": 1.0,         # small enclosed space, moderate fuel load (fabrics/textiles)
    "home_office": 1.0,    # residential equivalent of office
    "nursery": 1.6,        # children's room -- occupants can't self-evacuate, needs
                           # faster response than a typical bedroom
    "dirty_kitchen": 2.0,  # common PH secondary/outdoor cooking area; same open-flame
                           # risk profile as kitchen
    "gas_storage": 2.5,    # LPG tank/cylinder storage; explosion & flammable-gas hazard
    "balcony": 0.6,        # open-air, minimal enclosure
    "porch": 0.6,          # open-air, minimal enclosure

    "other": 1.0
}

# Built from ROOM_TYPE_WEIGHTS so the AI prompts below can never drift out of
# sync with the room types the hazard-scoring algorithm actually knows about.
ROOM_TYPE_ENUM_STR = "|".join(ROOM_TYPE_WEIGHTS.keys())

DPWH_MAX_DISTANCE = 45.0
DPWH_SPRINKLERED_MAX = 60.0


def occupant_load_factor(occupant_load):
    if occupant_load <= 10:
        return 1.0
    elif occupant_load <= 50:
        return 1.1
    elif occupant_load <= 499:
        return 1.3
    elif occupant_load <= 999:
        return 1.5
    else:
        return 1.8


def floor_level_multiplier(floor):
    if floor <= 1:
        return 1.0
    elif floor <= 3:
        return 1.2
    elif floor <= 4:
        return 1.5
    else:
        return 2.0


def compute_hazard_index(room, max_distance=DPWH_MAX_DISTANCE):
    distance = float(room['distance_to_exit'])
    room_type = room['room_type'].lower()
    floor = int(room['floor_level'])
    adjacency = room.get('adjacency', 'none')
    occupant_load = int(room.get('occupant_load', 10))
    room_weight = ROOM_TYPE_WEIGHTS.get(room_type, 1.0)
    floor_mult = floor_level_multiplier(floor)
    occ_factor = occupant_load_factor(occupant_load)
    adjacency_factor = 1.0
    if adjacency in ['electrical_room', 'kitchen', 'storage', 'laboratory',
                      'garage', 'laundry_room', 'utility_room', 'attic',
                      'basement', 'dirty_kitchen', 'gas_storage']:
        adjacency_factor = 1.3
    elif adjacency in ['stairwell', 'lobby', 'corridor', 'balcony', 'porch', 'carport']:
        adjacency_factor = 0.8
    # max_distance is the PD 1096 Section 1207(b)(4) travel-distance ceiling:
    # 45.0m for non-sprinklered buildings, 60.0m for fully sprinklered ones.
    # It's the caller's job to pass the right one -- see the 'sprinklered'
    # flag handled in /analyze.
    distance_ratio = distance / max_distance
    hi = distance_ratio * room_weight * floor_mult * adjacency_factor * occ_factor
    return round(hi, 4)


def classify_risk(hazard_index):
    if hazard_index < 0.5:
        return "green", "Low Risk"
    elif hazard_index < 0.9:
        return "orange", "Moderate Risk"
    else:
        return "red", "High Risk"


# ─────────────────────────────────────────────
# SPATIAL GRAPH MODULE
# ─────────────────────────────────────────────

# Fallback connection distance for build_spatial_graph(): when two rooms on
# the same floor don't share an 'adjacency' match (e.g. AI-detected rooms
# whose adjacency field wasn't set/matched), but both have AI-detected
# centroid coordinates that sit within this many percentage-points of each
# other on the floor plan image (0-100 scale, same units as
# centroid_x_pct/y_pct), they're still treated as spatially adjacent for
# clustering purposes. Purely geometric -- no GPT call involved.
CLUSTER_PROXIMITY_PCT = 12.0


def _rooms_spatially_close(room_a, room_b, threshold_pct=CLUSTER_PROXIMITY_PCT):
    """True if both rooms have AI-detected centroid coordinates and those
    centroids sit within threshold_pct of each other (straight-line, in the
    same 0-100 pct space as centroid_x_pct/y_pct). Returns False whenever
    either room lacks centroid data (e.g. manually-typed rooms) -- this is
    a fallback ON TOP OF the adjacency-field check in build_spatial_graph,
    not a replacement for it.
    """
    ax, ay = room_a.get('centroid_x_pct'), room_a.get('centroid_y_pct')
    bx, by = room_b.get('centroid_x_pct'), room_b.get('centroid_y_pct')
    if ax is None or ay is None or bx is None or by is None:
        return False
    try:
        dist = ((float(ax) - float(bx)) ** 2 + (float(ay) - float(by)) ** 2) ** 0.5
    except (TypeError, ValueError):
        return False
    return dist <= threshold_pct


def build_spatial_graph(rooms_analyzed):
    """
    Constructs a spatial graph where:
    - Each room is a NODE with its hazard attributes
    - Edges connect spatially adjacent rooms based on adjacency field
    - Edge weight = average HI of the two connected rooms
    """
    G = nx.Graph()

    for i, room in enumerate(rooms_analyzed):
        G.add_node(i,
            room_name=room['room_name'],
            room_type=room['room_type'],
            floor_level=room['floor_level'],
            distance_to_exit=room['distance_to_exit'],
            hazard_index=room['hazard_index'],
            risk_color=room['risk_color'],
            risk_label=room['risk_label'],
            adjacency=room.get('adjacency', 'none')
        )

    for i, room_a in enumerate(rooms_analyzed):
        for j, room_b in enumerate(rooms_analyzed):
            if i >= j:
                continue
            if room_a['floor_level'] != room_b['floor_level']:
                continue
            a_adj = room_a.get('adjacency', 'none').lower()
            b_adj = room_b.get('adjacency', 'none').lower()
            a_type = room_a['room_type'].lower()
            b_type = room_b['room_type'].lower()
            connected = (
                a_adj == b_type or
                b_adj == a_type or
                (a_adj == b_adj and a_adj not in ['none', 'other'])
            )
            connection_type = 'adjacency' if connected else None
            # Fallback: no adjacency-field match, but the AI-detected
            # positions on the floor plan put these two rooms right next to
            # each other -- still worth linking for clustering purposes.
            if not connected and _rooms_spatially_close(room_a, room_b):
                connected = True
                connection_type = 'proximity'
            if connected:
                edge_weight = round((room_a['hazard_index'] + room_b['hazard_index']) / 2, 4)
                G.add_edge(i, j, weight=edge_weight, connection_type=connection_type)

    return G


CLUSTER_HI_THRESHOLD = 0.5


def detect_clusters_graph(G, rooms_analyzed, hi_threshold=CLUSTER_HI_THRESHOLD):
    """
    Uses graph traversal (connected components on high-risk subgraph)
    to detect clusters of high-risk zones — replaces the old list-index method.
    """
    high_risk_nodes = [
        n for n, data in G.nodes(data=True)
        if data['hazard_index'] >= hi_threshold
    ]
    subgraph = G.subgraph(high_risk_nodes)
    clusters = []
    for component in nx.connected_components(subgraph):
        if len(component) >= 2:
            cluster_rooms = [rooms_analyzed[n] for n in component]
            avg_hi = round(sum(r['hazard_index'] for r in cluster_rooms) / len(cluster_rooms), 4)
            clusters.append({
                'rooms': cluster_rooms,
                'avg_hi': avg_hi,
                'floor': cluster_rooms[0]['floor_level'],
                'node_ids': list(component)
            })
    return clusters


def cluster_detection_diagnostics(G, rooms_analyzed, clusters, hi_threshold=CLUSTER_HI_THRESHOLD):
    """Explains, in plain language, why detect_clusters_graph() did or
    didn't find any clusters -- meant to replace a bare "no clusters
    detected" message with something the person can actually act on.
    Entirely deterministic: reads the same graph/room data already computed
    by build_spatial_graph()/detect_clusters_graph() above, no GPT
    call involved.
    """
    high_risk = [
        (n, data) for n, data in G.nodes(data=True)
        if data['hazard_index'] >= hi_threshold
    ]

    if clusters:
        return {'high_risk_room_count': len(high_risk), 'reason': None}

    if not high_risk:
        return {
            'high_risk_room_count': 0,
            'reason': (
                f"No room's hazard index reached the clustering threshold "
                f"(HI \u2265 {hi_threshold}) -- every room analyzed is "
                f"currently Low or Moderate risk, so there's nothing to "
                f"group into a cluster."
            ),
        }

    if len(high_risk) == 1:
        room_name = high_risk[0][1].get('room_name', 'that room')
        return {
            'high_risk_room_count': 1,
            'reason': (
                f"Only one room (\"{room_name}\") is at/above the "
                f"clustering threshold (HI \u2265 {hi_threshold}). "
                f"Clustering needs at least two high-risk rooms on the "
                f"same floor that are adjacent -- or, for AI-detected "
                f"rooms, close together on the floor plan."
            ),
        }

    # 2+ high-risk rooms exist, but none of them landed in the same
    # connected component -- i.e. nothing links them (adjacency field
    # mismatch, or too far apart for the proximity fallback).
    room_names = ", ".join(f'"{data.get("room_name", "?")}"' for _, data in high_risk[:6])
    if len(high_risk) > 6:
        room_names += f", and {len(high_risk) - 6} more"
    return {
        'high_risk_room_count': len(high_risk),
        'reason': (
            f"{len(high_risk)} rooms are at/above the clustering threshold "
            f"(HI \u2265 {hi_threshold}) -- {room_names} -- but none of "
            f"them are linked to each other. Check that each room's "
            f"\"adjacent room type\" field actually names a neighboring "
            f"room's type, or upload a floor plan image so AI-detected "
            f"positions can link nearby rooms automatically."
        ),
    }


def compute_evacuation_paths(G, rooms_analyzed):
    """
    Uses Dijkstra shortest path algorithm to find the safest evacuation
    route from each high-risk room through the spatial graph to the
    nearest exit node (lobby, corridor, or stairwell).
    """
    exit_node_ids = [
        n for n, data in G.nodes(data=True)
        if data['room_type'] in ['lobby', 'corridor', 'stairwell']
    ]

    evacuation_paths = []
    for node_id, data in G.nodes(data=True):
        if data['risk_color'] != 'red':
            continue
        if not exit_node_ids:
            continue
        best_path = None
        best_cost = float('inf')
        best_exit = None
        for exit_id in exit_node_ids:
            if exit_id == node_id:
                continue
            try:
                path = nx.dijkstra_path(G, node_id, exit_id, weight='weight')
                cost = nx.dijkstra_path_length(G, node_id, exit_id, weight='weight')
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
                    best_exit = rooms_analyzed[exit_id]['room_name']
            except nx.NetworkXNoPath:
                continue
        if best_path:
            evacuation_paths.append({
                'from_room': data['room_name'],
                'to_exit': best_exit,
                'path': [rooms_analyzed[n]['room_name'] for n in best_path],
                'path_cost': round(best_cost, 4)
            })
    return evacuation_paths


def compute_all_evacuation_paths(G, rooms_analyzed):
    """Like compute_evacuation_paths(), but returns the Dijkstra shortest
    path for EVERY room on the graph (not just high-risk/red ones), and
    every room gets *some* result (empty path/None exit if unreachable)
    instead of being silently skipped.

    This exists specifically to feed render_deterministic_route_overlay()
    real routing ground truth -- the wall/door-aware Dijkstra path drawn
    straight onto the image, with no image-generation model or vision
    guesswork involved in the route itself
    this file already computes, which is what produced visibly wrong routes
    (wrong nearest exit, lines implying a path through a wall, etc.).
    """
    exit_node_ids = [
        n for n, data in G.nodes(data=True)
        if data['room_type'] in ('lobby', 'corridor', 'stairwell', 'porch', 'foyer')
    ]

    paths = []
    for node_id, data in G.nodes(data=True):
        if node_id in exit_node_ids:
            # This room IS a landing/egress-adjacent space -- trivially "at" the exit.
            paths.append({
                'from_room': data['room_name'],
                'to_exit': data['room_name'],
                'path': [data['room_name']],
                'path_cost': 0.0,
            })
            continue

        if not exit_node_ids:
            paths.append({'from_room': data['room_name'], 'to_exit': None, 'path': [], 'path_cost': None})
            continue

        best_path, best_cost, best_exit = None, float('inf'), None
        for exit_id in exit_node_ids:
            try:
                path = nx.dijkstra_path(G, node_id, exit_id, weight='weight')
                cost = nx.dijkstra_path_length(G, node_id, exit_id, weight='weight')
            except nx.NetworkXNoPath:
                continue
            if cost < best_cost:
                best_cost, best_path, best_exit = cost, path, rooms_analyzed[exit_id]['room_name']

        if best_path is not None:
            paths.append({
                'from_room': data['room_name'],
                'to_exit': best_exit,
                'path': [rooms_analyzed[n]['room_name'] for n in best_path],
                'path_cost': round(best_cost, 4),
            })
        else:
            paths.append({'from_room': data['room_name'], 'to_exit': None, 'path': [], 'path_cost': None})

    return paths


def graph_to_dict(G):
    """Serialize graph to JSON-safe dict for API response."""
    return {
        'nodes': [
            {
                'id': n,
                'room_name': data['room_name'],
                'room_type': data['room_type'],
                'floor_level': data['floor_level'],
                'hazard_index': data['hazard_index'],
                'risk_color': data['risk_color']
            }
            for n, data in G.nodes(data=True)
        ],
        'edges': [
            {'source': u, 'target': v, 'weight': data['weight']}
            for u, v, data in G.edges(data=True)
        ],
        'node_count': G.number_of_nodes(),
        'edge_count': G.number_of_edges()
    }


def compute_floor_risk_index(rooms_on_floor):
    if not rooms_on_floor:
        return 0
    avg = sum(r['hazard_index'] for r in rooms_on_floor) / len(rooms_on_floor)
    return round(avg, 4)


def compute_building_risk_index(all_rooms):
    if not all_rooms:
        return 0
    avg = sum(r['hazard_index'] for r in all_rooms) / len(all_rooms)
    return round(avg, 4)


# ---------------------------------------------------------------------------
# DISASTER SCENARIO (recommendation-only, supplementary to fire)
#
# "fire" is the system's default/baseline scenario -- it's what
# generate_recommendations() below and the hazard index itself are built
# around (NBC PD 1096 travel-distance thresholds, PD 1185 fire code). This
# section adds "earthquake" and "flood" as OPTIONAL, SUPPLEMENTARY advisory
# scenarios the user can pick from a dropdown: when one is selected, this
# layer reads the exact same already-computed room/cluster data (floor
# level, room type, construction type, occupant load) and produces extra
# recommendations specific to that hazard. It never runs a second hazard
# calculation and never touches compute_hazard_index/classify_risk/
# compute_floor_risk_index/compute_building_risk_index -- those stay fire-
# evacuation-distance-based (PD 1096) no matter what's selected here.
#
# Citation basis is deliberately different per scenario, because PD 1096
# and PD 1185 are fire/life-safety egress codes and don't speak to
# earthquake or flood behavior:
#   - earthquake: NSCP 2015 (National Structural Code of the Philippines),
#     particularly its seismic design provisions, plus general PHIVOLCS/
#     NDRRMC "duck-cover-hold" public guidance under RA 10121.
#   - flood: NDRRMC/RA 10121 (Philippine Disaster Risk Reduction and
#     Management Act) flood-response guidance -- there's no equivalent
#     structural code for flood the way NSCP covers seismic loads, so this
#     stays procedural/advisory rather than code-cited.
# These are advisory only, phrased with that uncertainty in mind, and are
# clearly weaker/less code-grounded than the PD 1096 recommendations above
# -- that's expected and should be disclosed as such in the defense.
# ---------------------------------------------------------------------------

DISASTER_TYPES = ("fire", "earthquake", "flood")

# Rooms whose contents become a secondary hazard (gas rupture, electrical
# short, water contact) once shaking or flooding starts -- reuses the same
# list generate_recommendations()/generate_construction_recommendations()
# already check, so all three stay in sync.
_UTILITY_RISK_ROOM_TYPES = ('electrical_room', 'kitchen', 'gas_storage', 'dirty_kitchen', 'server_room')


def _sanitize_disaster_type(value):
    """Falls back to 'fire' -- the system's default/baseline scenario --
    for anything missing or unrecognized, so an old client that never sends
    disaster_type (or sends garbage) sees exactly today's behavior."""
    return value if value in DISASTER_TYPES else "fire"


def generate_earthquake_recommendations(rooms_analyzed, clusters):
    """Supplementary earthquake advisories layered on top of the fire-based
    recommendations, built only from data the system already has. Grounded
    in NSCP 2015 seismic provisions where structural, and general PHIVOLCS/
    NDRRMC public guidance (RA 10121) where behavioral -- see the section
    comment above for why this can't be PD 1096-cited the way fire recs are."""
    recs = []

    upper_floor_rooms = sorted({
        r['room_name'] for r in rooms_analyzed if r['floor_level'] >= 3
    })
    if upper_floor_rooms:
        recs.append({
            "priority": "High",
            "message": (
                f"{len(upper_floor_rooms)} room(s) on floor 3 or above -- "
                f"{', '.join(upper_floor_rooms)}. NSCP 2015's seismic design "
                "provisions treat upper floors as higher lateral-displacement "
                "risk during shaking. Do not evacuate mid-shake: drop-cover-hold "
                "first, then use stairwells only (never elevators) once shaking "
                "stops and the stairwell is confirmed structurally intact."
            ),
            "icon": "🌏",
            "category": "earthquake"
        })

    masonry_rooms = sorted({
        r['room_name'] for r in rooms_analyzed
        if r.get('construction_type') == 'type_3'
    })
    if masonry_rooms:
        recs.append({
            "priority": "Moderate",
            "message": (
                f"{len(masonry_rooms)} room(s) -- {', '.join(masonry_rooms)} -- "
                "are Type III (masonry + wood) construction. Unreinforced masonry "
                "is a well-documented seismic vulnerability; NSCP 2015 requires "
                "these elements to meet its seismic reinforcement provisions -- "
                "have a structural engineer confirm this room's masonry is tied "
                "and reinforced, not just fire-rated."
            ),
            "icon": "🧱",
            "category": "earthquake"
        })

    utility_rooms = sorted({
        r['room_name'] for r in rooms_analyzed
        if r['room_type'] in _UTILITY_RISK_ROOM_TYPES
    })
    if utility_rooms:
        recs.append({
            "priority": "High",
            "message": (
                f"{len(utility_rooms)} room(s) -- {', '.join(utility_rooms)} -- "
                "carry gas or electrical utilities. Post-earthquake fires from "
                "ruptured gas lines or shorted wiring are a leading secondary "
                "cause of casualties (PHIVOLCS/NDRRMC guidance). Ensure gas "
                "shutoff valves and electrical breakers for these rooms are "
                "clearly labeled and reachable without entering the room itself."
            ),
            "icon": "⚡",
            "category": "earthquake"
        })

    if clusters:
        recs.append({
            "priority": "Moderate",
            "message": (
                f"{len(clusters)} high-risk cluster(s) were already flagged for "
                "fire risk (spatial graph traversal). The same clustering "
                "concentrates occupant load during an earthquake evacuation too "
                "-- treat these zones as bottlenecks in an earthquake drill, not "
                "just a fire drill."
            ),
            "icon": "🌏",
            "category": "earthquake"
        })

    return recs


def generate_flood_recommendations(rooms_analyzed, clusters):
    """Supplementary flood advisories layered on top of the fire-based
    recommendations, built only from data the system already has. There's
    no NSCP/PD 1096 equivalent for flood response, so this stays procedural
    and is cited to NDRRMC/RA 10121 general flood-response guidance rather
    than a specific building-code section -- see the section comment above."""
    recs = []

    ground_floor_rooms = sorted({
        r['room_name'] for r in rooms_analyzed if r['floor_level'] <= 1
    })
    upper_floor_exists = any(r['floor_level'] >= 2 for r in rooms_analyzed)

    if ground_floor_rooms:
        if upper_floor_exists:
            recs.append({
                "priority": "High",
                "message": (
                    f"{len(ground_floor_rooms)} room(s) on the ground floor -- "
                    f"{', '.join(ground_floor_rooms)}. Per NDRRMC flood-response "
                    "guidance, ground-floor exits can become impassable before "
                    "occupants realize it. Identify and post a vertical "
                    "evacuation route to an upper floor as a backup to the "
                    "ground-floor exit routes this system already computed."
                ),
                "icon": "🌊",
                "category": "flood"
            })
        else:
            recs.append({
                "priority": "Critical",
                "message": (
                    "All rooms are on the ground floor with no upper floor to "
                    "evacuate to. Per NDRRMC flood-response guidance, a "
                    "single-story building has no built-in vertical refuge -- "
                    "identify a nearby elevated relocation site in the LGU's "
                    "flood contingency plan before a flood event, since this "
                    "building can't provide one on its own."
                ),
                "icon": "🌊",
                "category": "flood"
            })

    basement_rooms = sorted({
        r['room_name'] for r in rooms_analyzed if r['room_type'] == 'basement'
    })
    if basement_rooms:
        recs.append({
            "priority": "Critical",
            "message": (
                f"{len(basement_rooms)} basement room(s) -- {', '.join(basement_rooms)}. "
                "Basements flood first and drain last -- per NDRRMC guidance "
                "these should never be a designated refuge point or relied on "
                "as part of an evacuation route during a flood event."
            ),
            "icon": "🌊",
            "category": "flood"
        })

    ground_utility_rooms = sorted({
        r['room_name'] for r in rooms_analyzed
        if r['floor_level'] <= 1 and r['room_type'] in _UTILITY_RISK_ROOM_TYPES
    })
    if ground_utility_rooms:
        recs.append({
            "priority": "High",
            "message": (
                f"{len(ground_utility_rooms)} ground-floor utility room(s) -- "
                f"{', '.join(ground_utility_rooms)} -- carry gas or electrical "
                "service. Electrocution and gas leaks are leading flood-related "
                "hazards (NDRRMC guidance); cut power at the main breaker and "
                "shut off gas supply to these rooms before floodwater reaches "
                "the building, not after."
            ),
            "icon": "⚡",
            "category": "flood"
        })

    stairwell_rooms = sorted({
        r['room_name'] for r in rooms_analyzed if r['room_type'] == 'stairwell'
    })
    if stairwell_rooms and upper_floor_exists:
        recs.append({
            "priority": "Informational",
            "message": (
                f"{len(stairwell_rooms)} stairwell(s) -- {', '.join(stairwell_rooms)} "
                "-- are available as vertical evacuation routes. Keep these "
                "unobstructed and clearly signed as flood refuge access, in "
                "addition to their normal role as fire egress routes."
            ),
            "icon": "🪜",
            "category": "flood"
        })

    return recs


def generate_disaster_recommendations(rooms_analyzed, clusters, disaster_type):
    """Dispatcher for the supplementary disaster-scenario layer above.
    disaster_type == 'fire' (the default) returns [] -- fire is already the
    baseline generate_recommendations() below is built around, so there's
    nothing supplementary to add. Only 'earthquake'/'flood' add anything,
    and both are purely additive: this never removes or modifies anything
    generate_recommendations() produced, and never touches hazard scoring."""
    if disaster_type == "earthquake":
        return generate_earthquake_recommendations(rooms_analyzed, clusters)
    if disaster_type == "flood":
        return generate_flood_recommendations(rooms_analyzed, clusters)
    return []


def generate_recommendations(rooms_analyzed, clusters, max_distance=DPWH_MAX_DISTANCE, sprinklered=False):
    recs = []
    high_risk = [r for r in rooms_analyzed if r['risk_color'] == 'red']
    moderate_risk = [r for r in rooms_analyzed if r['risk_color'] == 'orange']

    if high_risk:
        sprinkler_note = (
            " This building is marked as fully sprinklered, so the extended "
            "60.0m threshold already applies -- these rooms exceed even that."
            if sprinklered else
            " Installing an automatic fire sprinkler system would raise this "
            f"threshold to {DPWH_SPRINKLERED_MAX}m per NBC PD 1096 Section 1207(b)(4)."
        )
        recs.append({
            "priority": "Critical",
            "message": (
                f"{len(high_risk)} room(s) exceed the safe evacuation travel distance "
                f"of {max_distance}m per NBC PD 1096 Section 1207(b)(4). "
                "Immediate relocation or addition of exit routes is required."
                f"{sprinkler_note}"
            ),
            "icon": "🚨"
        })

    if clusters:
        recs.append({
            "priority": "Critical",
            "message": (
                f"{len(clusters)} high-risk cluster(s) detected via spatial graph traversal. "
                "Clustered hazard zones require dedicated automatic fire-extinguishing "
                "systems per NBC PD 1096 Section 1212(a)."
            ),
            "icon": "🔥"
        })

    for room in rooms_analyzed:
        if room['room_type'] in ['electrical_room', 'kitchen', 'gas_storage', 'dirty_kitchen'] and room['risk_color'] != 'green':
            recs.append({
                "priority": "High",
                "message": (
                    f"Room '{room['room_name']}' (Floor {room['floor_level']}) is a "
                    "high-hazard occupancy type. Install automatic fire suppression per "
                    "NBC PD 1096 Section 1212(a) and Fire Code PD 1185."
                ),
                "icon": "⚡"
            })
            break

    if moderate_risk:
        recs.append({
            "priority": "Moderate",
            "message": (
                f"{len(moderate_risk)} room(s) are in the moderate-risk range. "
                "Review corridor widths — NBC PD 1096 Section 1207(d)(1) "
                "requires a minimum of 1.10 meters."
            ),
            "icon": "⚠️"
        })

    if not sprinklered:
        for room in rooms_analyzed:
            if float(room['distance_to_exit']) > DPWH_SPRINKLERED_MAX:
                recs.append({
                    "priority": "High",
                    "message": (
                        f"Room '{room['room_name']}' exceeds even the sprinklered maximum "
                        f"of {DPWH_SPRINKLERED_MAX}m per NBC PD 1096 Section 1207(b)(4). "
                        "Sprinklering this building would not be enough on its own -- "
                        "an additional exit point must be added."
                    ),
                    "icon": "🚪"
                })

    high_floors = [r for r in rooms_analyzed if r['floor_level'] >= 5]
    if high_floors:
        recs.append({
            "priority": "High",
            "message": (
                f"{len(high_floors)} room(s) are on floor 5 or above. "
                "NBC PD 1096 Section 1207(i) requires at least one smokeproof enclosure."
            ),
            "icon": "🏢"
        })

    for room in rooms_analyzed:
        occ = int(room.get('occupant_load', 0))
        if occ >= 500 and room['risk_color'] != 'green':
            recs.append({
                "priority": "High",
                "message": (
                    f"Room '{room['room_name']}' has an occupant load of {occ}. "
                    "NBC PD 1096 Section 1207(b)(1) requires at least 3 exits "
                    "for occupant loads of 500-999."
                ),
                "icon": "👥"
            })
            break

    # Wall/construction-type compliance (NBC PD 1096 Sections 401-403) --
    # see generate_construction_recommendations() above. Folded in here,
    # inside generate_recommendations() itself, rather than left as a
    # separate call the caller has to remember to make, so wall-material
    # data reliably reaches the recommendations list.
    recs.extend(generate_construction_recommendations(rooms_analyzed))

    if not recs:
        recs.append({
            "priority": "Compliant",
            "message": (
                "All rooms are within NBC PD 1096 safe evacuation distance thresholds "
                "and occupant load requirements. Building appears compliant with the "
                "National Building Code of the Philippines."
            ),
            "icon": "✅"
        })

    return recs


# ---------------------------------------------------------------------------
# OCCUPANCY TYPE (building-wide, recommendation-only)
#
# NBC PD 1096 Rule VII (Section 701) classifies every building into
# occupancy groups by use -- Group A for single-family, exclusive-use
# residential dwellings; Group B for hotels, apartments, and multi-unit
# residential buildings; Group C for education and recreation (schools,
# libraries, recreation centers); Group E for business/mercantile (offices,
# retail, dining establishments); and Group F for industrial (factories,
# workshops, manufacturing). Each group carries its own life-safety
# expectations under the Code -- Group B in particular carries corridor/exit requirements
# Group A does not, which is why residential is split into two distinct
# selections here rather than one combined "Group A/B" option. This is
# captured once per building (unlike the per-room construction_type above)
# because occupancy classification is a whole-building determination under
# Section 701, not a per-room one. Modeled on the disaster_type dropdown
# above: a simple user-selected value that layers OPTIONAL, SUPPLEMENTARY
# advisory recommendations on top of whatever generate_recommendations()
# already produced. It never runs a second hazard calculation and never
# touches compute_hazard_index/classify_risk/compute_floor_risk_index/
# compute_building_risk_index -- those stay travel-distance-based (PD 1096
# Section 1207(b)(4)) no matter which occupancy type is selected here.
# 'not_sure' (the default) is a deliberate no-op, exactly like
# disaster_type's 'fire' baseline, so a client that never sends
# occupancy_type sees no change in behavior.
# ---------------------------------------------------------------------------

OCCUPANCY_TYPES = ("residential_a", "residential_b", "educational", "commercial", "industrial", "not_sure")

_OCCUPANCY_LABELS = {
    "residential_a": "Group A (Residential -- single-family, exclusive-use dwellings)",
    "residential_b": "Group B (Residential -- hotels, apartments, multi-unit residential)",
    "educational": "Group C (Education and Recreation -- schools, libraries, recreation centers)",
    "commercial": "Group E (Business/Mercantile -- offices, retail, dining)",
    "industrial": "Group F (Industrial -- factories, workshops, manufacturing)",
}


def _sanitize_occupancy_type(value):
    """Falls back to 'not_sure' for anything missing/unrecognized, so an old
    client that never sends occupancy_type (or sends garbage) sees exactly
    today's behavior -- generate_occupancy_recommendations() below returns
    [] for 'not_sure' rather than guessing at a classification the user
    never actually declared."""
    return value if value in OCCUPANCY_TYPES else "not_sure"


def generate_occupancy_recommendations(rooms_analyzed, occupancy_type, clusters):
    """Supplementary occupancy-type advisories layered on top of the
    rule-based recommendations, built only from data the system already has
    (room_type, occupant_load, risk clusters). Grounded in NBC PD 1096
    Section 701's occupancy-group classification -- see the section comment
    above. Returns [] for 'not_sure'/unrecognized values, so this is purely
    additive and never required."""
    recs = []
    if occupancy_type not in _OCCUPANCY_LABELS:
        return recs

    recs.append({
        "priority": "Informational",
        "message": (
            f"Building declared as {_OCCUPANCY_LABELS[occupancy_type]} per "
            "NBC PD 1096 Section 701. The recommendations below are tailored "
            "to this occupancy classification -- re-run the analysis if the "
            "declared use changes, since exit, construction, and fire-safety "
            "requirements differ by occupancy group."
        ),
        "icon": "🏷️"
    })

    if occupancy_type == "industrial":
        hazard_rooms = sorted({
            r['room_name'] for r in rooms_analyzed
            if r['room_type'] in _HAZARD_ROOM_TYPES
        })
        if hazard_rooms:
            recs.append({
                "priority": "High",
                "message": (
                    f"{len(hazard_rooms)} high-hazard room(s) -- {', '.join(hazard_rooms)} -- "
                    "in a Group F (Industrial) building per NBC PD 1096 Section 701. "
                    "Industrial occupancies housing process equipment, gas, or "
                    "electrical hazards should be held to the stricter end of the "
                    "construction and fire-suppression requirements in Sections "
                    "401-403 and 1212(a), not the minimum -- confirm with the "
                    "local Building Official whether this qualifies as a "
                    "high-hazard industrial use requiring additional safeguards."
                ),
                "icon": "🏭"
            })
        if clusters:
            recs.append({
                "priority": "Moderate",
                "message": (
                    "Industrial floor plans concentrate equipment and process "
                    "hazards -- the high-risk cluster(s) already flagged above "
                    "warrant a dedicated industrial fire-safety review under "
                    "Fire Code PD 1185, in addition to the standard NBC PD 1096 "
                    "evacuation check."
                ),
                "icon": "🏭"
            })

    elif occupancy_type == "commercial":
        high_occupant_rooms = sorted({
            r['room_name'] for r in rooms_analyzed
            if int(r.get('occupant_load', 0)) >= 50
        })
        if high_occupant_rooms:
            recs.append({
                "priority": "Moderate",
                "message": (
                    f"{len(high_occupant_rooms)} room(s) -- {', '.join(high_occupant_rooms)} -- "
                    "carry meaningful occupant loads in a Group E (Business/"
                    "Mercantile) building per NBC PD 1096 Section 701. Confirm "
                    "exit signage, emergency lighting, and panic-hardware "
                    "requirements for public/commercial use are met on every "
                    "exit door serving these rooms, on top of the occupant-load "
                    "exit-count check under Section 1207(b)(1)."
                ),
                "icon": "🏬"
            })
        else:
            recs.append({
                "priority": "Informational",
                "message": (
                    "No rooms currently show a large occupant load. Group E "
                    "(Business/Mercantile) occupancies under NBC PD 1096 Section "
                    "701 should still confirm accessible-egress and exit-signage "
                    "requirements even at low occupancy, since public access "
                    "applies regardless of headcount."
                ),
                "icon": "🏬"
            })

    elif occupancy_type == "residential_a":
        oversized_a = sorted({
            r['room_name'] for r in rooms_analyzed
            if int(r.get('occupant_load', 0)) >= 10
        })
        if oversized_a:
            recs.append({
                "priority": "Moderate",
                "message": (
                    f"{len(oversized_a)} room(s) -- {', '.join(oversized_a)} -- "
                    "show occupant loads beyond a typical single-family "
                    "household. NBC PD 1096 Section 701 limits Group A to "
                    "single-family, exclusive-use dwellings -- if this building "
                    "actually houses multiple unrelated units or guests (hotel, "
                    "apartment, boarding house), it should be re-declared as "
                    "Group B, which carries additional corridor and exit "
                    "requirements Group A does not."
                ),
                "icon": "🏠"
            })
        else:
            recs.append({
                "priority": "Compliant",
                "message": (
                    "Occupant loads are consistent with a single-family Group A "
                    "residential occupancy per NBC PD 1096 Section 701, which "
                    "carries the Code's baseline exit and construction "
                    "requirements rather than the stricter thresholds applied "
                    "to Group B, commercial, or industrial occupancies."
                ),
                "icon": "🏠"
            })

    elif occupancy_type == "residential_b":
        recs.append({
            "priority": "Moderate",
            "message": (
                "Group B (hotels, apartments, multi-unit residential) carries "
                "corridor width, exit-count, and fire-separation requirements "
                "beyond the Group A baseline under NBC PD 1096 Section 701 -- "
                "confirm each unit has an independent path to an exit that "
                "doesn't pass through another unit, and that shared corridors "
                "meet the minimum 1.10m width already checked under Section "
                "1207(d)(1) above."
            ),
            "icon": "🏨"
        })
        high_occupant_b = sorted({
            r['room_name'] for r in rooms_analyzed
            if int(r.get('occupant_load', 0)) >= 50
        })
        if high_occupant_b:
            recs.append({
                "priority": "High",
                "message": (
                    f"{len(high_occupant_b)} room(s) -- {', '.join(high_occupant_b)} -- "
                    "carry occupant loads typical of shared/common areas (lobby, "
                    "function hall) in a Group B building. NBC PD 1096 Section "
                    "1207(b)(1)'s exit-count thresholds apply here just as they "
                    "would in a commercial building -- don't assume the "
                    "residential label exempts these spaces."
                ),
                "icon": "🏨"
            })

    elif occupancy_type == "educational":
        high_occupant_edu = sorted({
            r['room_name'] for r in rooms_analyzed
            if int(r.get('occupant_load', 0)) >= 50
        })
        if high_occupant_edu:
            recs.append({
                "priority": "High",
                "message": (
                    f"{len(high_occupant_edu)} room(s) -- {', '.join(high_occupant_edu)} -- "
                    "carry occupant loads typical of classrooms, assembly, or "
                    "recreation areas in a Group C (Education and Recreation) "
                    "building per NBC PD 1096 Section 701. Confirm outward-"
                    "swinging exit doors, panic hardware, and unobstructed "
                    "corridor widths are provided for these rooms, on top of "
                    "the occupant-load exit-count check under Section "
                    "1207(b)(1) -- educational occupancies concentrate large "
                    "numbers of occupants, many of them children, who may "
                    "need adult-assisted evacuation."
                ),
                "icon": "🏫"
            })
        else:
            recs.append({
                "priority": "Informational",
                "message": (
                    "No rooms currently show a large occupant load. Group C "
                    "(Education and Recreation) occupancies under NBC PD 1096 "
                    "Section 701 should still confirm exit-signage, emergency "
                    "lighting, and assembly-area egress requirements even at "
                    "low occupancy, since classroom and activity spaces can "
                    "fill quickly during scheduled use."
                ),
                "icon": "🏫"
            })

    return recs


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/app')
def app_home():
    return render_template('app.html')


@app.route('/upload', methods=['POST'])
def upload_floor_plan():
    if 'floorplan' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['floorplan']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return jsonify({'success': True, 'filepath': f'/static/uploads/{filename}'})


@app.route('/builtin-pd', methods=['GET'])
def builtin_pd():
    builtin = os.path.join(app.config['UPLOAD_FOLDER'], 'pd_builtin_pd1096.pdf')
    if os.path.exists(builtin):
        return jsonify({'success': True, 'filepath': '/static/uploads/pd_builtin_pd1096.pdf', 'filename': 'PD 1096 - National Building Code of the Philippines'})
    return jsonify({'success': False, 'error': 'Built-in PD not found'}), 404


@app.route('/upload-pd', methods=['POST'])
def upload_pd():
    if 'pd_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['pd_file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    filename = f"pd_{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return jsonify({'success': True, 'filepath': f'/static/uploads/{filename}', 'filename': file.filename})


FEET_TO_METERS = 0.3048


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    rooms = data.get('rooms', [])
    building_name = data.get('building_name', 'Unnamed Building')
    floorplan_path = data.get('floorplan_path', None)
    distance_unit = data.get('distance_unit', 'm')
    unit_is_ft = distance_unit == 'ft'
    # Purely a display/reporting figure -- AI-derived when scale info is
    # confident, or hand-entered by the user. Never fed into hazard scoring.
    building_size_sqm = data.get('building_size_sqm', None)
    building_length_m = data.get('building_length_m', None)
    building_width_m = data.get('building_width_m', None)
    # Whole-building sprinkler status per NBC PD 1096 Section 1207(b)(4):
    # a fully sprinklered building gets a 60.0m max travel distance instead
    # of the default 45.0m. Applies building-wide, not per-room.
    sprinklered = bool(data.get('sprinklered', False))
    max_travel_distance = DPWH_SPRINKLERED_MAX if sprinklered else DPWH_MAX_DISTANCE
    # Fire-extinguisher/window markers, if the frontend sent any -- see the
    # "FIRE EXTINGUISHER / WINDOW MARKERS" section above. Never required:
    # an empty/missing list just means generate_marker_recommendations()
    # below returns [] and nothing changes for existing callers.
    safety_markers = _sanitize_safety_markers(data.get('markers', []))
    # Disaster scenario dropdown -- 'fire' is the default/baseline the hazard
    # index and generate_recommendations() are already built around. Only
    # 'earthquake'/'flood' add anything, via the purely-additive supplementary
    # layer in generate_disaster_recommendations() -- see that section's
    # comment block for the full rationale. Never affects hazard scoring.
    disaster_type = _sanitize_disaster_type(data.get('disaster_type', 'fire'))
    # Building-wide occupancy classification per NBC PD 1096 Section 701 --
    # 'not_sure' (the default) is a no-op, exactly like disaster_type's
    # 'fire' baseline. See the "OCCUPANCY TYPE" section comment above.
    occupancy_type = _sanitize_occupancy_type(data.get('occupancy_type', 'not_sure'))

    if not rooms:
        return jsonify({'error': 'No room data provided'}), 400

    # Step 1: Compute hazard index per room
    # NBC PD 1096's 45.0m / 60.0m travel-distance thresholds are defined in
    # METERS. If the UI is set to feet, the raw input value must be converted
    # to meters BEFORE it reaches compute_hazard_index/classify_risk, or the
    # hazard index and every PD 1096 compliance check downstream is wrong by
    # a factor of ~3.28. This conversion is enforced here, server-side, so it
    # can't be skipped by a frontend bug or a bypassed client.
    rooms_analyzed = []
    for room in rooms:
        raw_distance = float(room['distance_to_exit'])
        distance_m = round(raw_distance * FEET_TO_METERS, 4) if unit_is_ft else raw_distance
        room_for_calc = dict(room)
        room_for_calc['distance_to_exit'] = distance_m

        hi = compute_hazard_index(room_for_calc, max_distance=max_travel_distance)
        color, label = classify_risk(hi)
        rooms_analyzed.append({
            'room_name': room.get('room_name', 'Unknown Room'),
            'room_type': room['room_type'],
            'floor_level': int(room['floor_level']),
            'distance_to_exit': round(distance_m, 4),
            'distance_input_value': raw_distance,
            'distance_input_unit': distance_unit,
            'adjacency': room.get('adjacency', 'none'),
            'occupant_load': int(room.get('occupant_load', 10)),
            'measurement_method': room.get('measurement_method', 'manual_entry'),
            # Which exit/stairwell the wallgrid+Dijkstra graph actually
            # routed this room to (e.g. "corridor -> service exit"), so the
            # results table can tell someone not just HOW FAR their nearest
            # exit is but WHICH one to head for. None for manually-typed
            # rooms or rooms where the CV distance wasn't trusted -- see
            # verify_room_distance()'s room['nearest_exit_used'].
            'nearest_exit_used': room.get('nearest_exit_used'),
            'hazard_index': hi,
            'risk_color': color,
            'risk_label': label,
            # Carried through only for _nearest_room_for_marker() to match
            # fire-extinguisher/window markers to a room -- unused by, and
            # never touching, the hazard/risk computation above. Absent for
            # manually-typed rooms with no AI-detected coordinates, which
            # simply won't be matchable (see that function's docstring).
            'centroid_x_pct': room.get('centroid_x_pct'),
            'centroid_y_pct': room.get('centroid_y_pct'),
            # Deterministic wallgrid+Dijkstra route data computed earlier by
            # verify_room_distance() (during /ai-analyze-multi) -- carried
            # through here rather than dropped, since render_deterministic_
            # route_overlay() (the "generate evacuation routes (deterministic)"
            # button) reads these straight off lastAnalysisData.rooms, which
            # is populated from THIS endpoint's response, not ai-analyze-multi's.
            'distance_source': room.get('distance_source'),
            'route_waypoints_pct': room.get('route_waypoints_pct'),
            # Lossless companion to route_waypoints_pct -- see
            # path_to_exact_pct_points() in wallgrid.py. This is what
            # render_deterministic_route_overlay() actually draws;
            # route_waypoints_pct stays around only as the RDP-simplified
            # shape hint used elsewhere.
            'route_path_exact_pct': room.get('route_path_exact_pct'),
            # Per-room wall/construction type, used only by
            # generate_construction_recommendations() -- see that section's
            # comment block. Never touches hazard_index/risk_color above.
            'construction_type': _sanitize_construction_type(room.get('construction_type')),
        })

    # Step 2: Build spatial graph
    G = build_spatial_graph(rooms_analyzed)

    # Step 3: Detect clusters via graph traversal
    clusters = detect_clusters_graph(G, rooms_analyzed)
    # Plain-language explanation of why clustering did/didn't find anything
    # -- see cluster_detection_diagnostics()'s docstring. Deterministic,
    # same graph/room data as above, no GPT call.
    cluster_diagnostics = cluster_detection_diagnostics(G, rooms_analyzed, clusters)

    # Step 4: Compute evacuation paths via Dijkstra
    evacuation_paths = compute_evacuation_paths(G, rooms_analyzed)

    # Step 5: Floor and building risk indices
    floors = {}
    for r in rooms_analyzed:
        floors.setdefault(r['floor_level'], []).append(r)

    floor_risk_labels = {}
    for fl, rms in floors.items():
        fi = compute_floor_risk_index(rms)
        _, label = classify_risk(fi)
        floor_risk_labels[fl] = {'index': fi, 'label': label}

    building_index = compute_building_risk_index(rooms_analyzed)
    _, building_label = classify_risk(building_index)

    # Step 6: Recommendations
    recommendations = generate_recommendations(
        rooms_analyzed, clusters,
        max_distance=max_travel_distance,
        sprinklered=sprinklered
    )
    recommendations += generate_marker_recommendations(rooms_analyzed, safety_markers)
    # Supplementary earthquake/flood advisories -- no-op ([]) when
    # disaster_type is 'fire' (the default), so existing callers that never
    # send disaster_type see no change in behavior or recommendation count.
    recommendations += generate_disaster_recommendations(rooms_analyzed, clusters, disaster_type)
    # Supplementary occupancy-type advisories -- no-op ([]) when
    # occupancy_type is 'not_sure' (the default), so existing callers that
    # never send occupancy_type see no change in recommendation count.
    recommendations += generate_occupancy_recommendations(rooms_analyzed, occupancy_type, clusters)

    # Format cluster data for response
    cluster_data = []
    for i, cluster in enumerate(clusters):
        cluster_data.append({
            'cluster_id': i + 1,
            'floor': cluster['floor'],
            'rooms': [r['room_name'] for r in cluster['rooms']],
            'avg_hi': cluster['avg_hi']
        })

    return jsonify({
        'building_name': building_name,
        'floorplan_path': floorplan_path,
        'building_size_sqm': building_size_sqm,
        'building_length_m': building_length_m,
        'building_width_m': building_width_m,
        'rooms': rooms_analyzed,
        'floor_risk': floor_risk_labels,
        'building_risk_index': building_index,
        'building_risk_label': building_label,
        'clusters': cluster_data,
        'cluster_diagnostics': cluster_diagnostics,
        'evacuation_paths': evacuation_paths,
        'spatial_graph': graph_to_dict(G),
        'recommendations': recommendations,
        'sprinklered': sprinklered,
        'dpwh_max_distance': max_travel_distance,
        'disaster_type': disaster_type,
        'occupancy_type': occupancy_type
    })


# Swapped from gpt-4o (2026-08-09): gpt-4o still works today, but
# OpenAI's own model guidance (developers.openai.com/api/docs/guides/
# latest-model) is steering everyone to the GPT-5.6 family, and the
# original GPT-5 snapshot this project could have used instead is ALREADY
# scheduled for API shutdown (Dec 11, 2026) -- so it's not a safe target
# either. gpt-5.6-terra is the mid tier: full vision support, much cheaper
# than -sol, and this is a bounded extraction task (walls/doors/rooms as
# JSON), not something that needs frontier reasoning.
GPT_MODEL = "gpt-5.6-sol"


def _gpt_completion_kwargs(model, max_tokens, temperature, response_format, messages):
    """Builds the kwargs for client.chat.completions.create() in a way that
    works for BOTH gpt-4o-family and gpt-5-family models, so GPT_MODEL can
    be swapped back to "gpt-4o" without this call site breaking.

    Two concrete differences that matter here:
    - max_tokens vs max_completion_tokens: gpt-5-family models reject
      `max_tokens` outright (400 error) and require `max_completion_tokens`
      instead -- gpt-4o-family models are the opposite.
    - temperature: gpt-5-family models only support the default value (1)
      and 400 on anything else, INCLUDING temperature=0 -- which is what
      this project used deliberately for deterministic JSON extraction. So
      for gpt-5-family, `temperature` is omitted entirely rather than sent
      as 0 (or even sent as 1 explicitly, in case a future snapshot is
      stricter about that too). Practical effect: gpt-5.6-terra's JSON
      extraction is less deterministic run-to-run than gpt-4o's was --
      worth knowing if you see room/wall counts wobble slightly between
      identical runs. gpt-4o-family models keep getting `temperature` as
      passed in, unchanged.
    - reasoning_effort: gpt-5-family models default to spending some of
      their token budget on invisible "reasoning" tokens before writing
      any visible output -- and those reasoning tokens count AGAINST
      max_completion_tokens. On a tight budget (e.g. 2000, sized for
      gpt-4o which had no such overhead) this can burn the entire budget
      on reasoning and return finish_reason="length" with an EMPTY
      message.content, which looks like a truncation bug but isn't one --
      it's the model never getting to the answer at all. Since this
      project's extraction calls are bounded lookups (walls/doors/rooms
      from an image), not problems that benefit from step-by-step
      reasoning, gpt-5-family calls set reasoning_effort="none" to disable
      that entirely. Not applied to o-series (o3/o4-mini etc.) since
      "none" isn't confirmed supported there.
    """
    kwargs = {
        "model": model,
        "response_format": response_format,
        "messages": messages,
    }
    if model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max_tokens
        kwargs["reasoning_effort"] = "none"
    elif model.startswith("o"):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature
    return kwargs


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def strip_markdown(text):
    if text.startswith("```"):
        text = '\n'.join([l for l in text.split('\n') if not l.startswith("```")])
    return text.strip()


@app.route('/ai-recommend', methods=['POST'])
def ai_recommend():
    data = request.json
    analysis_data = data.get('analysis', {})
    floorplan_paths = data.get('floorplan_paths', [])
    pd_filepath = data.get('pd_filepath', '')

    if not analysis_data:
        return jsonify({'error': 'No analysis data provided'}), 400

    content_parts = []
    mime_map = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'gif': 'image/gif', 'pdf': 'application/pdf'}

    if pd_filepath:
        actual_pd = resolve_static_path(pd_filepath)
        if os.path.exists(actual_pd):
            ext = actual_pd.rsplit('.', 1)[-1].lower()
            if ext == 'pdf':
                with tempfile.TemporaryDirectory() as tmpdir:
                    try:
                        subprocess.run(
                            subprocess.run(["pdftoppm", "-png", "-r", "150", "-f", "1", "-l", "25", pd_filepath, tmpdir + "/page"], check=True),
                            capture_output=True, timeout=30
                        )
                        page_files = sorted(glob.glob(f'{tmpdir}/page-*.ppm'))[:8]
                        if page_files:
                            content_parts.append({"type": "text", "text": f"REFERENCE DOCUMENT - PD 1096 (first {len(page_files)} pages):"})
                            for pf in page_files:
                                img = PILImage.open(pf).convert('RGB')
                                buf = io.BytesIO()
                                img.save(buf, format='JPEG', quality=85)
                                pg_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pg_b64}"}})
                    except Exception:
                        pass
            else:
                with open(actual_pd, 'rb') as f:
                    pd_b64 = base64.b64encode(f.read()).decode('utf-8')
                pd_mime = mime_map.get(ext, 'image/jpeg')
                content_parts.append({"type": "text", "text": "REFERENCE DOCUMENT - PD 1096:"})
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:{pd_mime};base64,{pd_b64}"}})

    valid_fps = [resolve_static_path(fp) for fp in floorplan_paths if os.path.exists(resolve_static_path(fp))]
    if valid_fps:
        content_parts.append({"type": "text", "text": f"FLOOR PLAN IMAGE(S) - {len(valid_fps)} floor(s):"})
        for fp in valid_fps:
            ext = fp.rsplit('.', 1)[-1].lower()
            with open(fp, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
            img_mime = mime_map.get(ext, 'image/jpeg')
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:{img_mime};base64,{img_b64}"}})

    rooms = analysis_data.get('rooms', [])
    building_name = analysis_data.get('building_name', 'Building')
    bri = analysis_data.get('building_risk_index', 0)
    brl = analysis_data.get('building_risk_label', '')
    clusters = analysis_data.get('clusters', [])
    existing_recs = analysis_data.get('recommendations', [])
    evacuation_paths = analysis_data.get('evacuation_paths', [])
    graph_data = analysis_data.get('spatial_graph', {})

    high_rooms = [r for r in rooms if r['risk_color'] == 'red']
    moderate_rooms = [r for r in rooms if r['risk_color'] == 'orange']
    low_rooms = [r for r in rooms if r['risk_color'] == 'green']

    room_summary = "\n".join([
        f"  - {r['room_name']} | Type: {r['room_type']} | Floor {r['floor_level']} | "
        f"Dist to Exit: {r['distance_to_exit']}m | HI: {r['hazard_index']} | "
        f"Risk: {r['risk_label']} | Adjacent to: {r['adjacency']} | Occupants: {r.get('occupant_load', '?')}"
        for r in rooms
    ])

    cluster_summary = ""
    if clusters:
        cluster_summary = "\nGraph-detected high-risk clusters:\n" + "\n".join([
            f"  - Cluster {c['cluster_id']} on Floor {c['floor']}: {', '.join(c['rooms'])} (avg HI: {c['avg_hi']})"
            for c in clusters
        ])

    path_summary = ""
    if evacuation_paths:
        path_summary = "\nDijkstra evacuation paths computed:\n" + "\n".join([
            f"  - {p['from_room']} -> {' -> '.join(p['path'])} -> {p['to_exit']} (cost: {p['path_cost']})"
            for p in evacuation_paths
        ])

    graph_summary = f"\nSpatial graph: {graph_data.get('node_count', 0)} nodes, {graph_data.get('edge_count', 0)} edges"

    existing_recs_text = "\n".join([f"  [{r['priority']}] {r['message']}" for r in existing_recs])

    prompt = f"""You are a licensed fire safety engineer specializing in PD 1096 and PD 1185.

=== BUILDING ANALYSIS ===
Building: {building_name}
Building Risk Index: {bri} - {brl}
Total Rooms: {len(rooms)} ({len(high_rooms)} high, {len(moderate_rooms)} moderate, {len(low_rooms)} low)
{graph_summary}

Room Details:
{room_summary}
{cluster_summary}
{path_summary}

=== EXISTING RECOMMENDATIONS ===
{existing_recs_text}

=== YOUR TASK ===
Generate ADDITIONAL specific AI-powered recommendations beyond the rule-based ones above.

Respond with ONLY this JSON object (note: recommendations is wrapped in an object,
not a bare array, so the response is guaranteed valid JSON):
{{
  "recommendations": [
    {{
      "priority": "Critical | High | Moderate | Informational",
      "icon": "emoji",
      "room_ref": "room name(s) or Building-wide",
      "pd_section": "PD 1096 Section X",
      "message": "specific actionable recommendation",
      "action": "exact corrective action"
      "pd_section": "Section X: X"
    }}
  ]
}}

Generate 3-6 recommendations. Return ONLY the JSON object."""

    content_parts.append({"type": "text", "text": prompt})

    try:
        parsed, raw, truncated = call_gpt4o_json(
            content=content_parts,
            max_tokens=3000,
            label="ai-recommend"
        )
        recs = parsed.get('recommendations', [])
        if not recs:
            return jsonify({'error': 'GPT returned no recommendations. Please try again.'}), 422
        response_payload = {'success': True, 'recommendations': recs}
        if truncated:
            response_payload['warning'] = (
                "The AI response was cut off before it finished (ran out of tokens). "
                "The recommendations shown may be incomplete -- try again for a full list."
            )
        return jsonify(response_payload)
    except GPTResponseError as e:
        logger.error("ai_recommend failed: %s", e)
        return jsonify({'error': f'GPT returned invalid JSON: {str(e)}'}), 500
    except Exception as e:
        logger.exception("ai_recommend unexpected error")
        return jsonify({'error': str(e)}), 500


@app.route('/ai-analyze', methods=['POST'])
def ai_analyze():
    data = request.json
    filepath = data.get('filepath', '')

    if not filepath:
        return jsonify({'error': 'No filepath provided'}), 400

    actual_path = resolve_static_path(filepath)
    if not os.path.exists(actual_path):
        return jsonify({'error': f'File not found: {actual_path}'}), 404

    with open(actual_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    ext = actual_path.rsplit('.', 1)[-1].lower()
    mime_map = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'gif': 'image/gif'}
    mime_type = mime_map.get(ext, 'image/jpeg')

    step1_prompt = """You are analyzing a building floor plan image.
STEP 1 - Extract scale information only.
Look for dimension lines, scale bars, building dimensions, room area labels in sqm.
Confidence: high=dimension lines found, medium=only area labels, low=nothing found.

An EXIT is a door/opening that leads DIRECTLY to the outside of the building
(front door, side door to a yard/street, garage door to a driveway). A door
between two interior rooms is NOT an exit, even if it looks prominent -- only
list openings that lead outside in "exits_visible".

Return ONLY valid JSON, no explanation, no markdown:

{
  "scale_source": "dimension_lines | scale_bar | text_label | estimated",
  "confidence": "high | medium | low",
  "confidence_reason": "explanation",
  "building_width_m": 8.0,
  "building_depth_m": 6.5,
  "dimension_annotations": [{"label": "8.00", "location": "bottom", "direction": "horizontal"}],
  "room_areas_sqm": [{"label": "27.90", "room_hint": "largest room"}],
  "exits_visible": ["front door"],
  "notes": ""
}"""

    try:
        scale_info, _raw_s1, _s1_truncated = call_gpt4o_json(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}", "detail": "high"}},
                {"type": "text", "text": step1_prompt}
            ],
            max_tokens=1000,
            label="single-floor scale-extraction"
        )
    except GPTResponseError as e:
        logger.warning("Scale extraction failed, using fallback defaults: %s", e)
        scale_info = {
            "scale_source": "estimated", "confidence": "low",
            "confidence_reason": "Could not extract scale from image",
            "building_width_m": 10.0, "building_depth_m": 8.0,
            "dimension_annotations": [], "room_areas_sqm": [], "exits_visible": []
        }

    bw = float(scale_info.get("building_width_m") or 10.0)
    bd = float(scale_info.get("building_depth_m") or 8.0)
    scale_source = scale_info.get("scale_source", "estimated")
    low_confidence = scale_confidence_from_source(scale_source)
    dim_notes = json.dumps(scale_info.get("dimension_annotations", []))
    area_notes = json.dumps(scale_info.get("room_areas_sqm", []))
    max_diag = round((bw**2 + bd**2)**0.5, 1)

    prompt = f"""You are analyzing a building floor plan image.

SCALE CONTEXT:
- Building width: {bw}m, depth: {bd}m
- Scale source: {scale_source}
- Dimension annotations: {dim_notes}
- Room area labels (sqm): {area_notes}
- Max diagonal: {max_diag}m

TASK: Identify every room and calculate walking distance to nearest exit.
Trace actual walking path, not straight-line. No distance can exceed {max_diag}m.

Return ONLY valid JSON, no explanation, no markdown:

{{
  "building_name": "name",
  "exits_identified": ["main front door"],
  "rooms": [
    {{
      "room_name": "Room Name",
      "room_type": "{ROOM_TYPE_ENUM_STR}",
      "floor_level": 1,
      "area_sqm": 0.0,
      "distance_to_exit": 0.0,
      "distance_calculation": "3m to door + 4m corridor = 7m",
      "adjacency": "{ROOM_TYPE_ENUM_STR}",
      "occupant_load": 20
    }}
  ]
}}

- occupant_load: use area_sqm / 4.6 per NBC PD 1096
- Include ALL visible rooms
- Return ONLY the JSON"""

    try:
        parsed, raw, truncated = call_gpt4o_json(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}", "detail": "high"}},
                {"type": "text", "text": prompt}
            ],
            max_tokens=5000,
            label="single-floor room-analysis"
        )
        if not parsed.get('rooms'):
            return jsonify({'error': 'Floor plan image is too blurry or unclear. Please upload a clearer image.', 'low_confidence': True}), 422
        parsed['scale_info'] = scale_info
        parsed['low_confidence'] = low_confidence
        if truncated:
            parsed['warning'] = (
                "The AI's response was cut off before it finished (ran out of tokens). "
                "Some rooms may be missing or incomplete -- try again, or split a "
                "complex floor plan into smaller sections."
            )
        return jsonify({'success': True, 'data': parsed})
    except GPTResponseError as e:
        logger.error("ai_analyze failed: %s", e)
        return jsonify({'error': f'GPT returned invalid JSON: {str(e)}'}), 500
    except Exception as e:
        logger.exception("ai_analyze unexpected error")
        return jsonify({'error': str(e)}), 500


@app.route('/detect-geometry', methods=['POST'])
def detect_geometry():
    """Phase 1 of the two-phase AI analysis flow (see /ai-analyze-multi's
    geometry_override branch for phase 2). Runs ONLY GPT's Step-1 wall/
    door/exit/stairwell trace plus the Roboflow hybrid merge -- the exact
    same code /ai-analyze-multi itself runs before building each floor's
    wallgrid -- and returns that raw geometry for the frontend's review/
    repair screen, WITHOUT going on to detect rooms or compute any
    distance-to-exit/hazard numbers.

    The user reviews (and can edit: delete/add walls, add doors) the
    returned geometry in the UI, then submits it back as each floor's
    geometry_override in the /ai-analyze-multi call that follows. That
    downstream call treats the reviewed geometry as ground truth and skips
    re-running GPT's trace and Roboflow for that floor -- see the
    "Geometry review/repair override" comment there. Nothing about the
    hazard index, Dijkstra routing, or distance calculation is touched by
    this endpoint or by that override path: both still flow through the
    exact same build_grid_from_segments()/verify_room_distance() code a
    fully-automatic run would use, just fed geometry a human already
    looked at instead of GPT's first, unreviewed trace.
    """
    data = request.json
    filepaths = data.get('filepaths', [])
    distance_unit = data.get('distance_unit', 'm')
    unit_is_ft = distance_unit == 'ft'

    if not filepaths:
        return jsonify({'error': 'No filepaths provided'}), 400

    def _parse_positive_meters(raw):
        if raw is None or raw == '':
            return None
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        return val if val > 0 else None

    user_width_m = _parse_positive_meters(data.get('building_width_m'))
    user_length_m = _parse_positive_meters(data.get('building_length_m'))
    user_provided_dims = user_width_m is not None and user_length_m is not None
    if user_provided_dims:
        unit_factor = (1.0 / FEET_TO_METERS) if unit_is_ft else 1.0
        user_bw = round(user_width_m * unit_factor, 2)
        user_bd = round(user_length_m * unit_factor, 2)

    mime_map = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'gif': 'image/gif'}
    floors_out = []
    floor_errors = []

    def load_image_b64(path):
        try:
            with PILImage.open(path) as img:
                img.verify()
        except Exception as e:
            raise GPTResponseError(f"'{path}' is not a valid/readable image: {e}")
        with open(path, 'rb') as f:
            data_bytes = f.read()
        ext = path.rsplit('.', 1)[-1].lower()
        return base64.b64encode(data_bytes).decode('utf-8'), mime_map.get(ext, 'image/jpeg')

    for floor_data in filepaths:
        filepath = resolve_static_path(floor_data.get('filepath', ''))
        floor_label = floor_data.get('floor_label', 1)
        floor_markers = _sanitize_markers(floor_data.get('markers'))

        if not os.path.exists(filepath):
            floor_errors.append({'floor': floor_label, 'error': f'File not found: {filepath}'})
            continue

        try:
            image_data, mime_type = load_image_b64(filepath)
            with PILImage.open(filepath) as _img:
                img_w, img_h = _img.size
        except GPTResponseError as e:
            floor_errors.append({'floor': floor_label, 'error': str(e)})
            continue

        # ---------- STEP 1: scale + exit/stairwell extraction ----------
        # Identical prompt to /ai-analyze-multi's own Step 1 -- kept as a
        # verbatim copy (not a shared function) so this endpoint's
        # behavior can never drift out of sync with the fallback GPT-only
        # path a caller takes if it skips the review screen entirely.
        step1_prompt = f"""You are analyzing Floor {floor_label} of a building floor plan image.

STEP 1 - Extract scale and egress reference points only. Do not analyze rooms yet.

Look for: dimension lines, scale bars, printed building dimensions, room area
labels (sqm/sqft), every exit door, every stairwell, and every elevator.

WHAT COUNTS AS AN EXIT (be strict about this):
An EXIT is a door or opening that leads DIRECTLY to the OUTSIDE of the
building -- a main entrance, a side door to a yard/street, a garage door to a
driveway, or a fire exit to the exterior. It is NOT an exit if it only leads
to another interior room or hallway, no matter how prominent that door looks.
A staircase counts as an egress point on its own (list it under "stairwells"),
not as an "exit", since it still requires descending to reach the outside.

List EVERY exit, stairwell, and elevator you can see, even if there are
several -- do not assume there is only one. For each, give an approximate
pixel location (x_pct, y_pct as percentages of image width/height, 0-100) so
positions can be cross-checked against other floors of the same building.

WALLS AND DOORS -- this is the most common mistake, watch for it carefully:
Trace every WALL as a straight line segment in the same percentage
coordinates, and every DOOR opening (a gap in a wall where a door swing arc,
threshold line, or opening symbol is drawn). Approximate curved or angled
walls as several short straight segments.
- In OPEN-CONCEPT areas (e.g. a living room flowing into a dining or kitchen
  area with no full wall between them), do NOT draw a solid wall where the
  plan actually shows an open threshold or partial wall -- that will
  incorrectly block off a room that is really walkable.
- Conversely, do not assume two spaces are connected just because they look
  close together -- only mark a door/opening where the drawing actually shows
  one (an arc, a gap, or a labeled doorway).
- A missed door here is the single most common failure mode: it makes a real
  room look "unreachable" and forces the path-finder into a bogus long
  detour around the building. Missing a wall has the opposite problem: it
  lets a path cut straight through a solid barrier. Look twice at every
  opening before deciding whether it's a wall or a door.

This wall/door trace is used to build a deterministic walking-distance graph
as a cross-check on your own distance estimate below -- please be as
complete and careful as you can.

STAIRCASE FOOTPRINT -- separate from the single stairwell point above:
For every staircase you listed under "stairwells", also report the
rectangular floor-space it physically occupies (the tread/riser hatching,
the run of steps, the landing) as x_pct/y_pct min/max bounds. A staircase
run is NOT open floor -- a route on this level must go AROUND it to the
stairwell's actual entry point, never straight across the steps. If the
staircase is angled, give the bounds of a rectangle that fully contains it
(slightly oversized is fine; slightly undersized lets a route cut a corner
through the steps, which is worse)."""
        step1_prompt += _marker_hint_block(floor_markers)
        step1_prompt += """

Respond with ONLY this JSON object, no commentary:
{
  "scale_source": "dimension_lines | scale_bar | text_label | estimated",
  "confidence": "high | medium | low",
  "confidence_reason": "one short sentence",
  "building_width_m": 8.0,
  "building_depth_m": 6.5,
  "dimension_annotations": [{"label": "8.00", "location": "bottom", "direction": "horizontal"}],
  "room_areas_sqm": [{"label": "27.90", "room_hint": "largest room"}],
  "exits": [{"name": "front door", "x_pct": 50.0, "y_pct": 95.0}],
  "stairwells": [{"name": "main stairwell", "x_pct": 45.0, "y_pct": 60.0}],
  "elevators": [{"name": "main elevator", "x_pct": 70.0, "y_pct": 40.0}],
  "walls": [{"x1_pct": 8.0, "y1_pct": 8.0, "x2_pct": 92.0, "y2_pct": 8.0}],
  "doors": [{"x_pct": 45.0, "y_pct": 32.0, "orientation": "horizontal"}],
  "stair_footprints": [{"name": "main stairwell", "x_min_pct": 40.0, "y_min_pct": 55.0, "x_max_pct": 50.0, "y_max_pct": 70.0, "entry_x_pct": 45.0, "entry_y_pct": 55.0}],
  "notes": ""
}

Remember: trace EVERY wall and door in the image exhaustively, not just the
points mentioned above -- the markers are additional hints on top of that
full trace, not a replacement for it."""

        try:
            scale_info, _raw_s1, _s1_truncated = call_gpt4o_json(
                content=[
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}", "detail": "high"}},
                    {"type": "text", "text": step1_prompt}
                ],
                max_tokens=5000,
                label=f"floor {floor_label} scale-extraction (geometry review)"
            )
        except GPTResponseError as e:
            floor_errors.append({'floor': floor_label, 'error': f'Wall/door detection failed for this floor: {e}'})
            continue

        bw = float(scale_info.get("building_width_m") or 10.0)
        bd = float(scale_info.get("building_depth_m") or 8.0)
        scale_source = scale_info.get("scale_source", "estimated")
        if user_provided_dims:
            bw, bd = user_bw, user_bd
            scale_source = "user_provided"
            scale_info["confidence"] = "high"
            scale_info["confidence_reason"] = "Building width and length were entered by the user."

        user_exit_markers = [m for m in floor_markers if m.get("type") == "exit"]
        exits_list = (
            _merge_marker_exits([], floor_markers) if user_exit_markers
            else (scale_info.get("exits") or [])
        )
        stairwells_list = _merge_marker_stairs(scale_info.get("stairwells", []), floor_markers)
        elevators_list = _merge_marker_elevators(scale_info.get("elevators", []), floor_markers)

        # ---------- Roboflow hybrid: second, independent wall/door source ----------
        # Same best-effort merge /ai-analyze-multi runs -- see that
        # endpoint's comment block for the full rationale. Any failure
        # here just leaves the floor with GPT-only walls/doors.
        roboflow_walls_added = 0
        roboflow_doors_added = 0
        try:
            rf_predictions = call_roboflow_wall_door_detection(filepath)
            rf_geometry = convert_predictions_to_wallgrid_input(rf_predictions, img_w, img_h)
            merged_walls, merged_doors, roboflow_walls_added, roboflow_doors_added, _rf_conflicts = merge_wall_door_sources(
                scale_info.get("walls"), scale_info.get("doors"),
                rf_geometry["walls"], rf_geometry["doors"],
            )
            scale_info["walls"] = merged_walls
            scale_info["doors"] = merged_doors
        except RoboflowResponseError as e:
            logger.info("Floor %s: Roboflow hybrid detection skipped (%s) during geometry review.", floor_label, e)
        except Exception as e:
            logger.warning("Floor %s: Roboflow hybrid detection failed unexpectedly (%s) during geometry review.", floor_label, e)

        walls = _sanitize_wall_segments(scale_info.get("walls"))
        doors = _sanitize_door_points(scale_info.get("doors"))
        if not walls:
            floor_errors.append({
                'floor': floor_label,
                'warning': (
                    "No wall segments could be traced from this floor's image -- "
                    "the review screen will be empty. You can still add walls "
                    "manually, or continue without them (distances will fall "
                    "back to GPT's own estimate, unverified against a wall/door graph)."
                )
            })

        floors_out.append({
            'floor_label': floor_label,
            'filepath': floor_data.get('filepath', ''),
            'image_width_px': img_w,
            'image_height_px': img_h,
            'building_width_m': round(bw, 2),
            'building_depth_m': round(bd, 2),
            'scale_source': scale_source,
            'confidence': scale_info.get('confidence', 'low'),
            'confidence_reason': scale_info.get('confidence_reason', ''),
            'walls': walls,
            'doors': doors,
            'stair_footprints': _sanitize_stair_footprints(scale_info.get('stair_footprints')),
            'exits': _sanitize_named_points(exits_list),
            'stairwells': _sanitize_named_points(stairwells_list),
            'elevators': _sanitize_named_points(elevators_list),
            'walls_detected_count': len(walls),
            'doors_detected_count': len(doors),
            'roboflow_walls_added': roboflow_walls_added,
            'roboflow_doors_added': roboflow_doors_added,
        })

    if not floors_out:
        return jsonify({'error': 'Geometry detection failed for all floors.', 'floor_errors': floor_errors}), 500

    response_payload = {'success': True, 'data': {'floors': floors_out}}
    if floor_errors:
        response_payload['warnings'] = floor_errors
    return jsonify(response_payload)


@app.route('/ai-analyze-multi', methods=['POST'])
def ai_analyze_multi():
    data = request.json
    filepaths = data.get('filepaths', [])
    distance_unit = data.get('distance_unit', 'm')
    unit_is_ft = distance_unit == 'ft'

    if not filepaths:
        return jsonify({'error': 'No filepaths provided'}), 400

    # If the user already typed in the building's width/length (always
    # entered in meters in the UI), use those instead of letting GPT guess
    # scale from the image -- only skip GPT's guess when BOTH are given and
    # positive, since a lopsided override (only one dimension) would leave
    # the other one as GPT's guess paired with the user's real measurement,
    # which is worse than either alone.
    def _parse_positive_meters(raw):
        if raw is None or raw == '':
            return None
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        return val if val > 0 else None

    user_width_m = _parse_positive_meters(data.get('building_width_m'))
    user_length_m = _parse_positive_meters(data.get('building_length_m'))
    user_provided_dims = user_width_m is not None and user_length_m is not None
    if user_provided_dims:
        # bw/bd are treated as being in the request's chosen unit throughout
        # this file (see the CV-constants comment above), despite the "_m"
        # suffix -- convert the user's meters input the same way the
        # existing fallback_w/fallback_d defaults do.
        unit_factor = (1.0 / FEET_TO_METERS) if unit_is_ft else 1.0
        user_bw = round(user_width_m * unit_factor, 2)
        user_bd = round(user_length_m * unit_factor, 2)

    all_rooms = []
    building_name = "Multi-Floor Building"
    floor_errors = []  # collect per-floor failures instead of aborting everything
    mime_map = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'gif': 'image/gif'}
    floor_grid_results = {}  # floor_label -> WallGridResult, built once per floor, reused per room
    floor_dims = {}          # floor_label -> (bw, bd), for cross-floor straight-line sanity checks
    floor_scale_sources = {} # floor_label -> scale_source string, so we know whether a footprint
                             # estimate came from real dimension lines or a guess
    ground_floor_exit_targets = None

    def load_image_b64(path):
        """Validates the file is actually a readable image (catches corrupt
        uploads / wrong file types before they ever reach the OpenAI API)."""
        try:
            with PILImage.open(path) as img:
                img.verify()
        except Exception as e:
            raise GPTResponseError(f"'{path}' is not a valid/readable image: {e}")
        with open(path, 'rb') as f:
            data_bytes = f.read()
        ext = path.rsplit('.', 1)[-1].lower()
        return base64.b64encode(data_bytes).decode('utf-8'), mime_map.get(ext, 'image/jpeg')

    for floor_data in filepaths:
        filepath = resolve_static_path(floor_data.get('filepath', ''))
        floor_label = floor_data.get('floor_label', 1)
        floor_markers = _sanitize_markers(floor_data.get('markers'))

        if not os.path.exists(filepath):
            floor_errors.append({'floor': floor_label, 'error': f'File not found: {filepath}'})
            continue

        try:
            image_data, mime_type = load_image_b64(filepath)
        except GPTResponseError as e:
            floor_errors.append({'floor': floor_label, 'error': str(e)})
            continue

        # ---------- Geometry review/repair override ----------
        # If the frontend already ran /detect-geometry for this floor and
        # the user reviewed (and possibly edited) the wall/door trace on
        # the review screen, that edited geometry is sent back here as
        # geometry_override and is treated as ground truth for this floor:
        # GPT's own Step-1 wall/door/exit trace AND the Roboflow hybrid
        # merge are both skipped entirely (skipping Roboflow too matters --
        # otherwise a wall the user deliberately erased would just get
        # silently re-added by Roboflow's own detection on the next pass).
        # Nothing downstream of this (grid build, room detection,
        # verify_room_distance, hazard scoring) changes at all: it already
        # only reads from scale_info/exits_list/stairwells_list/
        # elevators_list, which this branch populates in the exact same
        # shape GPT's own Step-1 response would have.
        geometry_override = floor_data.get('geometry_override')
        if geometry_override and isinstance(geometry_override, dict):
            scale_info = {
                "scale_source": "user_reviewed",
                "confidence": "high",
                "confidence_reason": "Wall/door detection was reviewed (and possibly corrected) by the user before analysis.",
                "building_width_m": geometry_override.get("building_width_m"),
                "building_depth_m": geometry_override.get("building_depth_m"),
                "dimension_annotations": [],
                "room_areas_sqm": [],
                "exits": _sanitize_named_points(geometry_override.get("exits")),
                "stairwells": _sanitize_named_points(geometry_override.get("stairwells")),
                "elevators": _sanitize_named_points(geometry_override.get("elevators")),
                "walls": _sanitize_wall_segments(geometry_override.get("walls")),
                "doors": _sanitize_door_points(geometry_override.get("doors")),
                "stair_footprints": _sanitize_stair_footprints(geometry_override.get("stair_footprints")),
            }
            skip_step1_and_roboflow = True
        else:
            skip_step1_and_roboflow = False

        # ---------- STEP 1: scale + exit/stairwell extraction ----------
        step1_prompt = f"""You are analyzing Floor {floor_label} of a building floor plan image.

STEP 1 - Extract scale and egress reference points only. Do not analyze rooms yet.

Look for: dimension lines, scale bars, printed building dimensions, room area
labels (sqm/sqft), every exit door, every stairwell, and every elevator.

WHAT COUNTS AS AN EXIT (be strict about this):
An EXIT is a door or opening that leads DIRECTLY to the OUTSIDE of the
building -- a main entrance, a side door to a yard/street, a garage door to a
driveway, or a fire exit to the exterior. It is NOT an exit if it only leads
to another interior room or hallway, no matter how prominent that door looks.
A staircase counts as an egress point on its own (list it under "stairwells"),
not as an "exit", since it still requires descending to reach the outside.

List EVERY exit, stairwell, and elevator you can see, even if there are
several -- do not assume there is only one. For each, give an approximate
pixel location (x_pct, y_pct as percentages of image width/height, 0-100) so
positions can be cross-checked against other floors of the same building.

WALLS AND DOORS -- this is the most common mistake, watch for it carefully:
Trace every WALL as a straight line segment in the same percentage
coordinates, and every DOOR opening (a gap in a wall where a door swing arc,
threshold line, or opening symbol is drawn). Approximate curved or angled
walls as several short straight segments.
- In OPEN-CONCEPT areas (e.g. a living room flowing into a dining or kitchen
  area with no full wall between them), do NOT draw a solid wall where the
  plan actually shows an open threshold or partial wall -- that will
  incorrectly block off a room that is really walkable.
- Conversely, do not assume two spaces are connected just because they look
  close together -- only mark a door/opening where the drawing actually shows
  one (an arc, a gap, or a labeled doorway).
- A missed door here is the single most common failure mode: it makes a real
  room look "unreachable" and forces the path-finder into a bogus long
  detour around the building. Missing a wall has the opposite problem: it
  lets a path cut straight through a solid barrier. Look twice at every
  opening before deciding whether it's a wall or a door.

This wall/door trace is used to build a deterministic walking-distance graph
as a cross-check on your own distance estimate below -- please be as
complete and careful as you can.

STAIRCASE FOOTPRINT -- separate from the single stairwell point above:
For every staircase you listed under "stairwells", also report the
rectangular floor-space it physically occupies (the tread/riser hatching,
the run of steps, the landing) as x_pct/y_pct min/max bounds. A staircase
run is NOT open floor -- a route on this level must go AROUND it to the
stairwell's actual entry point, never straight across the steps. If the
staircase is angled, give the bounds of a rectangle that fully contains it
(slightly oversized is fine; slightly undersized lets a route cut a corner
through the steps, which is worse)."""
        step1_prompt += _marker_hint_block(floor_markers)
        step1_prompt += """

Respond with ONLY this JSON object, no commentary:
{
  "scale_source": "dimension_lines | scale_bar | text_label | estimated",
  "confidence": "high | medium | low",
  "confidence_reason": "one short sentence",
  "building_width_m": 8.0,
  "building_depth_m": 6.5,
  "dimension_annotations": [{"label": "8.00", "location": "bottom", "direction": "horizontal"}],
  "room_areas_sqm": [{"label": "27.90", "room_hint": "largest room"}],
  "exits": [{"name": "front door", "x_pct": 50.0, "y_pct": 95.0}],
  "stairwells": [{"name": "main stairwell", "x_pct": 45.0, "y_pct": 60.0}],
  "elevators": [{"name": "main elevator", "x_pct": 70.0, "y_pct": 40.0}],
  "walls": [{"x1_pct": 8.0, "y1_pct": 8.0, "x2_pct": 92.0, "y2_pct": 8.0}],
  "doors": [{"x_pct": 45.0, "y_pct": 32.0, "orientation": "horizontal"}],
  "stair_footprints": [{"name": "main stairwell", "x_min_pct": 40.0, "y_min_pct": 55.0, "x_max_pct": 50.0, "y_max_pct": 70.0, "entry_x_pct": 45.0, "entry_y_pct": 55.0}],
  "notes": ""
}

Remember: trace EVERY wall and door in the image exhaustively, not just the
points mentioned above -- the markers are additional hints on top of that
full trace, not a replacement for it."""

        if not skip_step1_and_roboflow:
            try:
                scale_info, _raw_s1, _s1_truncated = call_gpt4o_json(
                    content=[
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}", "detail": "high"}},
                        {"type": "text", "text": step1_prompt}
                    ],
                    max_tokens=5000,
                    label=f"floor {floor_label} scale-extraction"
                )
            except GPTResponseError as e:
                logger.warning("Floor %s scale extraction failed, using fallback defaults: %s", floor_label, e)
                fallback_w = round(10.0 / 0.3048, 1) if unit_is_ft else 10.0
                fallback_d = round(8.0 / 0.3048, 1) if unit_is_ft else 8.0
                scale_info = {
                    "scale_source": "estimated", "confidence": "low",
                    "confidence_reason": "Could not extract scale from image",
                    "building_width_m": fallback_w, "building_depth_m": fallback_d,
                    "dimension_annotations": [], "room_areas_sqm": [], "exits": [], "stairwells": [],
                    "walls": [], "doors": []
                }

        bw = float(scale_info.get("building_width_m") or 10.0)
        bd = float(scale_info.get("building_depth_m") or 8.0)
        max_diag = round((bw**2 + bd**2)**0.5, 1)
        user_exit_markers = [m for m in floor_markers if m.get("type") == "exit"]
        if user_exit_markers:
            # The user has manually marked exit(s) on this floor -- those
            # markers are now the ONLY exits used for routing/distance
            # calculations, full stop. GPT-4o's own "exits"/"exits_visible"
            # guess for this floor is discarded entirely rather than merged
            # in alongside them: previously it was ADDED to the user's
            # markers, so a GPT-hallucinated or mislabeled "exit" (e.g. an
            # interior doorway that doesn't actually lead outside) could
            # still sneak into exits_list and pull a room's route toward a
            # fake exit even when the user had already placed the real
            # one(s) by hand. _merge_marker_exits still does the actual
            # marker -> exits_list conversion (name/x_pct/y_pct/source), it
            # just starts from an empty list instead of GPT's guesses.
            exits_list = _merge_marker_exits([], floor_markers)
        else:
            # No user-placed exit markers on this floor -- fall back to
            # GPT-4o's own detected exits so unmarked floors keep working.
            exits_list = scale_info.get("exits") or [{"name": n} for n in scale_info.get("exits_visible", [])]
        stairwells_list = scale_info.get("stairwells", [])
        stairwells_list = _merge_marker_stairs(stairwells_list, floor_markers)
        elevators_list = scale_info.get("elevators", [])
        elevators_list = _merge_marker_elevators(elevators_list, floor_markers)
        if floor_label == 1 and exits_list:
            ground_floor_exit_targets = exits_list
        exits_notes = json.dumps(exits_list)
        stairwell_notes = json.dumps(stairwells_list)
        elevator_notes = json.dumps(elevators_list)
        dim_notes = json.dumps(scale_info.get("dimension_annotations", []))
        area_notes = json.dumps(scale_info.get("room_areas_sqm", []))
        scale_source = scale_info.get("scale_source", "estimated")
        if user_provided_dims:
            # The user measured and typed in the building's width/length
            # themselves -- trust that over GPT's own visual guess at scale.
            # Wall/door tracing (used for the CV pathfinding grid below) still
            # comes from GPT's read of the image; only the width/depth scale
            # factor that converts that tracing into real-world meters/feet
            # is overridden here, for every floor of this building.
            bw, bd = user_bw, user_bd
            max_diag = round((bw**2 + bd**2)**0.5, 1)
            scale_source = "user_provided"
            scale_info["confidence"] = "high"
            scale_info["confidence_reason"] = "Building width and length were entered by the user."
        low_confidence = scale_confidence_from_source(scale_source)
        is_upper = floor_label > 1

        # ---------- Roboflow hybrid: second, independent wall/door source ----------
        # Runs a pretrained Roboflow Universe instance-segmentation model
        # (see roboflow_cv.py) on this floor's image and merges its
        # filtered wall/door detections INTO scale_info's walls/doors
        # (GPT's own trace is kept as-is; Roboflow only ADDS geometry GPT's
        # trace didn't already cover -- see merge_wall_door_sources()).
        # This is best-effort: any failure here (bad/missing API key,
        # network, model down) falls straight back to GPT-only walls/doors
        # for this floor and never breaks the rest of the analysis.
        roboflow_walls_added = 0
        roboflow_doors_added = 0
        if not skip_step1_and_roboflow:
            try:
                with PILImage.open(filepath) as _img:
                    img_w, img_h = _img.size
                rf_predictions = call_roboflow_wall_door_detection(filepath)
                rf_geometry = convert_predictions_to_wallgrid_input(rf_predictions, img_w, img_h)
                merged_walls, merged_doors, roboflow_walls_added, roboflow_doors_added, _rf_conflicts = merge_wall_door_sources(
                    scale_info.get("walls"), scale_info.get("doors"),
                    rf_geometry["walls"], rf_geometry["doors"],
                )
                scale_info["walls"] = merged_walls
                scale_info["doors"] = merged_doors
                logger.info(
                    "DIAG floor %s: Roboflow hybrid added %d wall(s) / %d door(s) "
                    "(raw detections=%d) on top of GPT's own trace",
                    floor_label, roboflow_walls_added, roboflow_doors_added, len(rf_predictions),
                )
            except RoboflowResponseError as e:
                logger.info(
                    "Floor %s: Roboflow hybrid detection skipped (%s) -- using GPT-only walls/doors.",
                    floor_label, e,
                )
            except Exception as e:
                # Never let a CV-source failure take down the whole analysis.
                logger.warning(
                    "Floor %s: Roboflow hybrid detection failed unexpectedly (%s) -- "
                    "using GPT-only walls/doors.", floor_label, e,
                )
        else:
            logger.info(
                "DIAG floor %s: using user-reviewed geometry override -- "
                "GPT Step-1 trace and Roboflow hybrid both skipped for this floor.",
                floor_label,
            )

        # ---------- CV fallback/validator: build this floor's wallgrid ----------
        # One deterministic grid per floor, built from GPT's own wall/door
        # tracing (now possibly supplemented by the Roboflow hybrid step
        # above), reused below for every room on this floor (and for upper
        # floors' stairwell-landing-to-exit leg, via floor_grid_results[1]).
        grid_result = _build_floor_grid(scale_info, bw, bd, unit_is_ft, floor_label, logger)
        floor_grid_results[floor_label] = grid_result
        floor_dims[floor_label] = (bw, bd)
        floor_scale_sources[floor_label] = scale_source
        walls_extracted = len(scale_info.get("walls") or [])   # post-Roboflow-merge total
        doors_extracted = len(scale_info.get("doors") or [])   # post-Roboflow-merge total
        logger.info(
            "DIAG floor %s: scale_source=%r low_confidence=%s walls=%d (of which +%d from Roboflow) "
            "doors=%d (of which +%d from Roboflow) exits=%d stairwells=%d elevators=%d "
            "markers_sent=%d (types=%s) grid_built=%s",
            floor_label, scale_source, low_confidence, walls_extracted, roboflow_walls_added,
            doors_extracted, roboflow_doors_added,
            len(exits_list or []), len(stairwells_list or []), len(elevators_list or []),
            len(floor_markers), [m["type"] for m in floor_markers], grid_result is not None,
        )


        upper_note = (
            f"This is Floor {floor_label} (upper floor). For each room: "
            f"distance_to_exit = (walking distance to the NEAREST stairwell) + 3.5m (one flight descent) "
            f"+ (walking distance from that stairwell's Floor-1 landing to the NEAREST Floor-1 exit, "
            f"using IMAGE 1 to locate that exit). If there are multiple stairwells, use whichever gives "
            f"the shortest total distance."
            if is_upper else
            "This is the GROUND floor. For each room, measure the walking path (through doors and "
            "corridors, never through walls) to the NEAREST exit. If there are multiple exits, use "
            "whichever is closer for that room."
        )

        room_scope_note = ""
        if is_upper:
            room_scope_note = f"""
ROOM LIST SCOPE -- READ CAREFULLY: you were shown TWO images. IMAGE 1 was
Floor 1, given ONLY so you can trace the path from a stairwell landing to
the nearest ground-floor exit for the distance_to_exit calculation above.
The "rooms" list below must contain ONLY rooms that are physically visible
and labeled in IMAGE 2 (Floor {floor_label}). Do NOT list a room because you
saw it in IMAGE 1 -- even if Floor 1 and Floor {floor_label} share a similar
layout or repeated room names (e.g. both floors having a "T&B" or
"Bedroom"), each room in your list must come from something you can
actually see drawn in IMAGE 2, with its centroid_x_pct/centroid_y_pct
measured against IMAGE 2's own pixel dimensions -- never copy a position
from IMAGE 1. If IMAGE 2 shows fewer rooms than IMAGE 1, that is expected
and correct -- report only what IMAGE 2 actually contains.
"""

        prompt = f"""You are analyzing Floor {floor_label} of a building floor plan image.

SCALE CONTEXT:
- Building width: {bw}m, depth: {bd}m (source: {scale_source})
- Dimension annotations: {dim_notes}
- Room area labels (sqm): {area_notes}
- Exits on this building: {exits_notes}
- Stairwells: {stairwell_notes}
- Elevators: {elevator_notes}
- Max plausible diagonal walking distance: {max_diag}m

FLOOR NOTE: {upper_note}
HARD LIMIT: distance_to_exit must be <= {max_diag + (floor_label - 1) * 4}m. If your estimate exceeds
this, re-check your path -- you have likely traced through a wall.
{room_scope_note}
Respond with ONLY this JSON object, no commentary, no markdown fences:
{{
  "building_name": "name or 'Unnamed Building'",
  "rooms": [
    {{
      "room_name": "Room Name",
      "room_type": "{ROOM_TYPE_ENUM_STR}",
      "floor_level": {floor_label},
      "area_sqm": 0.0,
      "centroid_x_pct": 0.0,
      "centroid_y_pct": 0.0,
      "distance_to_exit": 0.0,
      "distance_calculation": "2m to stairwell + 3.5m descent + 2.5m to exit = 8m",
      "nearest_exit_used": "name of the exit/stairwell this distance is measured to",
      "adjacency": "{ROOM_TYPE_ENUM_STR}",
      "occupant_load": 20
    }}
  ]
}}

Rules:
- occupant_load = round(area_sqm / 4.6) per NBC PD 1096
- Set ALL floor_level to exactly {floor_label}
- Include every visibly labeled room, including small ones (T&B, storage, foyer, etc.)
- centroid_x_pct/centroid_y_pct = the approximate center point of the room, in the
  same x_pct/y_pct percentage coordinates used for exits/stairwells/walls/doors above
- Every room MUST come from the floor plan you were asked to analyze -- see
  ROOM LIST SCOPE above if this is an upper floor
- distance_calculation must show the arithmetic that produces distance_to_exit
- Return ONLY the JSON object above"""

        content_parts = []
        if is_upper:
            floor1_filepath = resolve_static_path(filepaths[0].get('filepath', ''))
            if os.path.exists(floor1_filepath):
                try:
                    floor1_data, floor1_mime = load_image_b64(floor1_filepath)
                    content_parts.append({"type": "text", "text": "IMAGE 1 - Floor 1 (ground floor): use this ONLY to locate the main exit(s) relative to the stairwell landing."})
                    content_parts.append({"type": "image_url", "image_url": {"url": f"data:{floor1_mime};base64,{floor1_data}"}})
                    content_parts.append({"type": "text", "text": f"IMAGE 2 - Floor {floor_label}: this is the floor to analyze."})
                except GPTResponseError as e:
                    logger.warning("Floor %s: could not attach Floor 1 reference image: %s", floor_label, e)

        content_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}", "detail": "high"}})
        content_parts.append({"type": "text", "text": prompt})

        try:
            parsed, raw, truncated = call_gpt4o_json(
                content=content_parts,
                max_tokens=4500,
                label=f"floor {floor_label} room-analysis"
            )
        except GPTResponseError as e:
            logger.error("Floor %s room analysis permanently failed: %s", floor_label, e)
            floor_errors.append({'floor': floor_label, 'error': str(e)})
            continue

        if parsed.get('building_name'):
            building_name = parsed['building_name']

        rooms = parsed.get('rooms', [])
        if not rooms:
            floor_errors.append({'floor': floor_label, 'error': 'GPT returned valid JSON but zero rooms -- image may be too blurry or unclear.'})
            continue

        if truncated:
            # the response was cut off by the token limit -- the rooms we did
            # get are real, but the list itself may be incomplete (a room cut
            # off mid-object) so this is a warning, not a hard failure.
            floor_errors.append({
                'floor': floor_label,
                'warning': (
                    "The AI's response for this floor was cut off before it finished "
                    "(ran out of tokens). Some rooms may be missing -- double-check "
                    "the room count against the floor plan, or try re-analyzing this "
                    "floor on its own."
                )
            })

        if low_confidence or (walls_extracted == 0 and doors_extracted == 0):
            # GPT still returned rooms even though it couldn't confidently read
            # scale/wall/door info from this image (e.g. it's blurry, low-res,
            # or hand-drawn) -- those room distances are GPT's own guesses with
            # no geometry cross-check possible (confirmed by "no wall segments
            # extracted, skipping CV validation" in the log). Previously this
            # was only ever tagged per-room (room['low_confidence']) and never
            # actually shown anywhere in the UI, so a blurry upload looked
            # exactly like a clean, verified result. Surfacing it here instead.
            floor_errors.append({
                'floor': floor_label,
                'warning': (
                    f"This floor's image didn't give the AI enough to work with "
                    f"({scale_info.get('confidence_reason') or 'low confidence, no wall/door data extracted'}). "
                    f"Room distances for this floor are GPT's own estimate only -- "
                    f"they could NOT be cross-checked against the wall/door geometry. "
                    f"Try a clearer or higher-resolution image if these numbers look off."
                )
            })

        # ---------- Floor-1 bleed-through guard ----------
        # Upper-floor requests attach Floor 1's image as a second reference
        # image (see content_parts above, used ONLY so GPT can trace the
        # stairwell-landing-to-exit leg of distance_to_exit). The prompt now
        # tells GPT not to list rooms from that reference image, but prompt
        # instructions aren't guaranteed -- GPT has been observed echoing
        # Floor 1's rooms straight into an upper floor's room list (still
        # tagged floor_level={floor_label} because it *does* follow that
        # instruction), which shows up as Floor 1's hazard-overlay dots
        # appearing on Floor 2's floor plan. This is a deterministic net
        # that doesn't depend on GPT behaving: a room is treated as
        # bleed-through only if BOTH its name matches a Floor 1 room AND its
        # centroid is within ~1.5 percentage points of that Floor 1 room's
        # centroid -- coordinates that close together on two independently-
        # measured images essentially can't happen by coincidence, so this
        # won't false-positive on two floors that legitimately share a
        # common room name (e.g. "T&B" on every floor).
        if is_upper:
            floor1_rooms = [r for r in all_rooms if r.get('floor_level') == 1]
            if floor1_rooms:
                deduped_rooms = []
                bled_through = []
                for room in rooms:
                    rx, ry = room.get('centroid_x_pct'), room.get('centroid_y_pct')
                    rname = (room.get('room_name') or '').strip().lower()
                    is_bleed = False
                    if rx is not None and ry is not None and rname:
                        for f1 in floor1_rooms:
                            fx, fy = f1.get('centroid_x_pct'), f1.get('centroid_y_pct')
                            fname = (f1.get('room_name') or '').strip().lower()
                            if fx is None or fy is None or fname != rname:
                                continue
                            if abs(rx - fx) <= 1.5 and abs(ry - fy) <= 1.5:
                                is_bleed = True
                                break
                    if is_bleed:
                        bled_through.append(room.get('room_name'))
                    else:
                        deduped_rooms.append(room)
                if bled_through:
                    logger.warning(
                        "Floor %s: dropped %d room(s) that look like Floor 1 bleed-through "
                        "(same name + near-identical centroid as a Floor 1 room): %s",
                        floor_label, len(bled_through), bled_through,
                    )
                    floor_errors.append({
                        'floor': floor_label,
                        'warning': (
                            f"{len(bled_through)} room(s) the AI initially listed for this floor "
                            f"-- {', '.join(bled_through)} -- looked identical to Floor 1 rooms "
                            f"(same name, same plotted position) shown only as a reference image, "
                            f"and were dropped as likely misattributed. If any of these genuinely "
                            f"exist on this floor too, add them manually."
                        )
                    })
                rooms = deduped_rooms
                if not rooms:
                    floor_errors.append({
                        'floor': floor_label,
                        'error': (
                            'Every room the AI detected for this floor looked like Floor 1 '
                            'bleed-through and was dropped. Try re-analyzing this floor on its '
                            'own, or add its rooms manually.'
                        )
                    })
                    continue

        # Now that this floor's room list is final, upgrade any user-marked
        # exit/stairwell/elevator's generic label ("user-marked exit 1")
        # with the name of whichever room sits closest to it ("user-marked
        # exit 1 (near Foyer)") -- see _label_markers_with_nearest_room().
        # Done once per floor, in place, so every room's verify_room_distance()
        # call below (and therefore the plain-English evacuation guide, which
        # reads room['nearest_exit_used'] straight off this) automatically
        # picks up the improved label with no other changes needed.
        exits_list = _label_markers_with_nearest_room(exits_list, rooms)
        stairwells_list = _label_markers_with_nearest_room(stairwells_list, rooms)
        elevators_list = _label_markers_with_nearest_room(elevators_list, rooms)

        ground_floor_grid = floor_grid_results.get(1)
        floor1_bw, floor1_bd = floor_dims.get(1, (bw, bd))
        sanity_max_distance = max_diag + (floor_label - 1) * 4
        for room in rooms:
            room['scale_source'] = scale_source
            room['confidence'] = scale_info.get('confidence', 'low' if low_confidence else 'medium')
            room['confidence_reason'] = scale_info.get('confidence_reason', '')
            room['low_confidence'] = low_confidence
            room['building_width_m'] = bw
            room['building_depth_m'] = bd
            room['cv_walls_extracted'] = walls_extracted
            room['cv_doors_extracted'] = doors_extracted
            room['roboflow_walls_added'] = roboflow_walls_added
            room['roboflow_doors_added'] = roboflow_doors_added
            room['wall_door_source'] = (
                'gpt+roboflow_hybrid' if (roboflow_walls_added or roboflow_doors_added) else 'gpt_only'
            )
            verify_room_distance(
                room, floor_label, grid_result,
                exits_list, stairwells_list,
                ground_floor_grid, unit_is_ft,
                bw, bd, floor1_bw, floor1_bd,
                sanity_max_distance=sanity_max_distance,
                elevators_list=elevators_list,
                ground_floor_exits_list=ground_floor_exit_targets or scale_info.get("ground_floor_exits") or scale_info.get("exits") or exits_list
            )
        all_rooms.extend(rooms)
        if grid_result is not None and grid_result.auto_repaired_cells:
            logger.info(
                "DIAG floor %s: auto-repaired %d wall cell(s) at %s -- "
                "GPT's wall trace likely missed a door here (no user marker needed)",
                floor_label, len(grid_result.auto_repaired_cells), grid_result.auto_repaired_cells,
            )

    if not all_rooms:
        # every floor failed -- this IS a hard error
        return jsonify({'error': 'Analysis failed for all floors.', 'floor_errors': floor_errors}), 500

    # Building footprint (sqm) estimate for the "building size" display field --
    # uses Floor 1's scale info if available (falls back to whichever floor
    # analyzed first). Purely informational: never fed back into hazard scoring.
    footprint_floor = 1 if 1 in floor_dims else next(iter(floor_dims), None)
    building_footprint_sqm = None
    building_footprint_width_m = None
    building_footprint_length_m = None
    footprint_scale_source = None
    if footprint_floor is not None:
        fp_bw, fp_bd = floor_dims[footprint_floor]
        building_footprint_sqm = round(fp_bw * fp_bd, 1)
        building_footprint_width_m = round(fp_bw, 1)
        building_footprint_length_m = round(fp_bd, 1)
        footprint_scale_source = floor_scale_sources.get(footprint_floor, 'estimated')

    response_payload = {
        'success': True,
        'data': {
            'building_name': building_name,
            'rooms': all_rooms,
            'building_footprint_sqm': building_footprint_sqm,
            'building_footprint_width_m': building_footprint_width_m,
            'building_footprint_length_m': building_footprint_length_m,
            'building_footprint_scale_source': footprint_scale_source,
        }
    }
    if floor_errors:
        # partial success: tell the caller which floors are missing instead of
        # silently dropping them or failing the entire multi-floor batch
        response_payload['warnings'] = floor_errors
    return jsonify(response_payload)



@app.route('/wallgrid-evacuation-plan', methods=['POST'])
def wallgrid_evacuation_plan():
    """Draws each room's already-computed route_waypoints_pct straight
    onto the floor plan image (see render_deterministic_route_overlay in
    wallgrid.py). Fully deterministic -- no image-generation model
    involved.

    Rooms whose distance_source isn't 'graph_verified' (no trustworthy
    wall/door-aware path) are reported in `skipped_rooms` instead of being
    guessed at -- the caller decides how to show that (e.g. "no verified
    route for these rooms yet" instead of a silently wrong line).
    """
    data = request.json or {}
    filepath = resolve_static_path(data.get('filepath', ''))
    floor_label = data.get('floor_label', 1)
    rooms_for_floor = data.get('rooms') or []
    markers_for_floor = _sanitize_markers(data.get('markers'))
    # Fire-extinguisher markers for this floor, drawing-only -- see
    # _sanitize_extinguisher_markers_for_overlay's docstring.
    extinguisher_markers_for_floor = _sanitize_extinguisher_markers_for_overlay(
        data.get('extinguisher_markers')
    )

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': f'File not found: {filepath}'}), 400

    try:
        image_bytes, drawn_rooms, skipped_rooms = render_deterministic_route_overlay(
            filepath, rooms_for_floor, markers=markers_for_floor,
            extinguisher_markers=extinguisher_markers_for_floor,
            floor_label=floor_label,
        )
    except Exception as e:
        logger.error("Floor %s deterministic route render failed: %s", floor_label, e)
        return jsonify({'error': str(e)}), 500

    out_name = f"wallgrid_routes_floor{floor_label}_{uuid.uuid4().hex[:12]}.png"
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
    with open(out_path, "wb") as f:
        f.write(image_bytes)

    return jsonify({
        'success': True,
        'floor_label': floor_label,
        'filepath': f"/static/uploads/{out_name}",
        'drawn_rooms': drawn_rooms,
        'skipped_rooms': skipped_rooms,
        'notes': (
            f"Drew verified routes for {len(drawn_rooms)} room(s)."
            + (f" No verified route yet for: {', '.join(skipped_rooms)}." if skipped_rooms else "")
        ),
    })


# ─────────────────────────────────────────────
# GPT TEXT-BASED EVACUATION GUIDE (simple, non-adjacency)
# ─────────────────────────────────────────────
# Deliberately separate from build_spatial_graph()/compute_evacuation_paths()
# (the Dijkstra/adjacency-graph machinery /analyze and the deterministic
# route overlay both lean on). That graph is only as good as each room's
# 'adjacency' field -- if a room's adjacency wasn't set/matched during
# analysis, the graph has no edge for it and any route built from it comes
# back empty ("no connected route ... check that its adjacency field is set
# correctly"). This endpoint sidesteps that entirely: GPT_MODEL is just
# asked to write a short, plain-English evacuation paragraph per room from
# the same hazard-analysis data (room type, hazard index, risk label) plus
# any labeled/directional marked exits (see _label_markers_by_floor), with
# no dependency on adjacency/graph connectivity at all -- a simple guide
# naming a specific exit, not a graph-routed path.

def _gpt_text_route_room_line(room):
    """One line per room for the plain-text guide prompt. Intentionally
    leaves out 'adjacency' -- this feeds the non-adjacency-based guide, so
    that field shouldn't factor into what GPT writes. Includes
    'nearest_exit_used' -- the deterministic, per-room nearest-exit label
    computed by verify_room_distance() (the same ground truth shown in the
    room-by-room breakdown table's "Dist. to Exit" column, e.g. "near front
    exterior door from foyer") -- so GPT names THAT specific exit instead of
    guessing among the generic "Exit 1"/"Exit 2" marker labels."""
    return (
        f"- \"{room.get('room_name', 'Unknown Room')}\" | floor {room.get('floor_level')} "
        f"| type: {room.get('room_type', 'other')} | risk: {room.get('risk_label', 'Unknown')} "
        f"| hazard index: {room.get('hazard_index')} | distance to exit: {room.get('distance_to_exit')}m "
        f"| nearest exit: {room.get('nearest_exit_used') or 'unknown'}"
    )


def _sanitize_markers_with_floor(raw_markers):
    """Same validation as _sanitize_markers, but keeps floor_level -- this
    endpoint aggregates every floor into a single GPT call, so each marker
    needs to say which floor it belongs to (unlike the deterministic
    per-floor endpoint, which calls _sanitize_markers once per floor and
    doesn't need that field)."""
    if not isinstance(raw_markers, list):
        return []
    cleaned = []
    for m in raw_markers[:120]:  # generous cap across all floors combined
        if not isinstance(m, dict):
            continue
        try:
            x_pct = max(0.0, min(100.0, float(m.get("x_pct"))))
            y_pct = max(0.0, min(100.0, float(m.get("y_pct"))))
            floor_level = int(m.get("floor_level"))
        except (TypeError, ValueError):
            continue
        marker_type = m.get("type") if m.get("type") in ("exit", "stair", "elevator") else None
        if marker_type is None:
            continue
        cleaned.append({
            "x_pct": round(x_pct, 2), "y_pct": round(y_pct, 2),
            "type": marker_type, "floor_level": floor_level,
        })
    return cleaned


def _direction_label(x_pct, y_pct):
    """Coarse compass-style position label from a marker's x_pct/y_pct
    (0-100, image space), e.g. "north-east corner" or "west side" -- just
    enough for GPT to describe WHERE a labeled exit is without needing
    exact pixel coordinates."""
    x_zone = "west" if x_pct < 33 else ("east" if x_pct > 66 else "")
    y_zone = "north" if y_pct < 33 else ("south" if y_pct > 66 else "")
    if x_zone and y_zone:
        return f"{y_zone}-{x_zone} corner"
    if y_zone:
        return f"{y_zone} side"
    if x_zone:
        return f"{x_zone} side"
    return "center of the floor"


_MARKER_LABEL_NAME = {"exit": "Exit", "stair": "Stairwell", "elevator": "Elevator"}


def _label_markers_by_floor(markers):
    """Groups sanitized markers by floor and gives each one a stable,
    human-readable label ("Exit 1", "Stairwell 2", ...) plus a coarse
    compass position -- so the prompt (and GPT's answer) can name a
    SPECIFIC exit/stairwell instead of just saying "a marked exit"."""
    by_floor = {}
    for m in markers:
        by_floor.setdefault(m["floor_level"], []).append(m)

    labeled = {}
    for floor_level, ms in by_floor.items():
        counters = {"exit": 0, "stair": 0, "elevator": 0}
        entries = []
        for m in ms:
            t = m["type"]
            counters[t] = counters.get(t, 0) + 1
            entries.append({
                "label": f"{_MARKER_LABEL_NAME.get(t, t.title())} {counters[t]}",
                "type": t,
                "direction": _direction_label(m["x_pct"], m["y_pct"]),
            })
        labeled[floor_level] = entries
    return labeled


def _gpt_text_route_marker_lines(labeled_markers_by_floor):
    """Turns the output of _label_markers_by_floor() into prompt lines
    naming each marked exit/stairwell and its rough position, e.g.
    "Floor 1 marked exits/stairwells: Exit 1 (north side), Stairwell 1
    (south-east corner)." -- so GPT can point a room toward a SPECIFIC
    named exit rather than a generic "marked exit"."""
    lines = []
    for floor_label, entries in sorted(
        (labeled_markers_by_floor or {}).items(), key=lambda kv: str(kv[0])
    ):
        exits_stairs = [e for e in entries if e["type"] in ("exit", "stair")]
        elevators = [e for e in entries if e["type"] == "elevator"]
        if not exits_stairs and not elevators:
            continue
        line = f"- Floor {floor_label} marked exits/stairwells: "
        line += (", ".join(f"{e['label']} ({e['direction']})" for e in exits_stairs)
                 if exits_stairs else "none marked")
        if elevators:
            line += (". Also marked: " +
                     ", ".join(f"{e['label']} ({e['direction']})" for e in elevators) +
                     " -- do NOT direct anyone to use these during a fire evacuation.")
        lines.append(line)
    return lines


_FALLBACK_HAZARD_CAUTIONS = [
    (("kitchen", "service", "utility"),
     "Watch for a grease or gas-fed fire flaring back up near the stove or "
     "gas line -- don't linger there, and don't use water on a grease fire."),
    (("electrical", "server", "panel"),
     "Stay away from any exposed wiring, switches, or panels -- risk of "
     "shock or arc flash, especially if the area is wet."),
    (("storage",),
     "This room can hold flammable or combustible clutter that fuels a "
     "fire fast -- don't stop to gather stored items, and watch your "
     "footing over anything fallen."),
    (("stair",),
     "Smoke and heat rise and pool near the top of stairwells -- stay low, "
     "keep a hand on the railing, and don't fight the flow of people."),
    (("bedroom",),
     "Check the door or handle for heat before opening it -- if it's hot, "
     "keep it shut and use another way out."),
]


def _fallback_hazard_caution(room_type):
    """Deterministic, room-type-aware caution used only in the fallback path
    (when GPT skipped a room from its response) -- mirrors the kind of
    concrete, reason-attached hazard guidance the main prompt now asks GPT
    to generate for every room, so a fallback row doesn't read as noticeably
    thinner than the rest of the guide."""
    rt = (room_type or "").strip().lower()
    for keywords, caution in _FALLBACK_HAZARD_CAUTIONS:
        if any(kw in rt for kw in keywords):
            return caution
    return (
        "Watch for furniture, clutter, or a narrow doorway that could slow "
        "you down in a rush, and keep the path to the door clear for "
        "others."
    )


def build_gpt_text_route_prompt(rooms, has_markers, has_high_risk_rooms=True):
    lines = [
        "You are writing a plain-English, per-room evacuation guide for a "
        "building, based on a hazard analysis already computed by this "
        "system (grounded in Philippine building codes PD 1096, PD 1185, "
        "RA 9514, NSCP 2015).",
        "",
        "For EVERY room listed below, write a evacuation instruction of "
        "5-7 sentences (roughly 80-130 words), plain layperson language, no "
        "jargon, that a person could read once and act on immediately. This "
        "should read like real, substantive safety guidance -- not a "
        "generic template repeated with the room name swapped out.",
        "",
        "Each instruction should cover, in this order:",
        "1. WHICH SPECIFIC exit or door to head to -- use that room's own "
        "\"nearest exit\" value given below (this is the actual nearest exit "
        "already computed for that room, e.g. \"front exterior door from "
        "foyer\" or \"service exterior door\") as the named destination. "
        "Describe its rough position (e.g. \"on the north side of the "
        "floor\") using the MARKED EXITS/STAIRWELLS list below only to help "
        "phrase that position -- do NOT substitute a different exit/"
        "stairwell from that list, and do NOT pick or guess among them; "
        "always point the room toward its own given nearest exit.",
        "2. A short sense of DIRECTION/ROUTE from that room toward it (e.g. "
        "\"exit the room and turn toward the north side of the building\").",
        (
            "3. An explicit reminder to AVOID walking through or lingering in "
            "any room on the list marked risk: High Risk, going around it "
            "instead if it would normally sit on the direct path."
            if has_high_risk_rooms else
            "3. (Skip this step -- no room in this building is currently "
            "marked High Risk, so do NOT tell anyone to avoid or route "
            "around a High Risk room; that instruction would be false and "
            "confusing. Go straight to step 4.)"
        ),
        "4. A SPECIFIC, PRACTICAL caution tied to the actual hazard(s) "
        "involved -- not a generic filler line. Reason about this room's own "
        "\"type\"" + (
            " (and, if the direct path would normally pass near a High "
            "Risk room, that room's type too)" if has_high_risk_rooms else ""
        ) + " and name the real-world danger "
        "and how to handle it, e.g.:",
        "   - kitchen/service/utility room nearby: grease or gas-fed fire "
        "can flare or reignite suddenly -- don't linger near the stove/gas "
        "line, and if a pan fire is small and you're trained to, smother it "
        "with a lid rather than using water.",
        "   - electrical/server/panel room or exposed wiring: risk of shock "
        "or arc flash, especially if wet -- do not touch switches, panels, "
        "or metal fixtures, and route around it if there's visible smoke or "
        "sparking.",
        "   - storage room or anything described as cluttered: often holds "
        "flammable or combustible material that can fuel a fire fast and "
        "block a clear path -- don't stop to protect stored items, and "
        "watch your footing over anything that may have fallen.",
        "   - stairwell/stairs: smoke and heat rise and pool at the top, so "
        "stay low, keep a hand on the railing, and don't fight the main "
        "flow of people.",
        "   - bedroom or a room with a closed door: check if the door or "
        "handle feels hot before opening it; if it does, keep it shut and "
        "use another way out.",
        "   - long corridor or dead-end area: note that smoke/fire can cut "
        "off a route quickly, so have the general direction of a second way "
        "out in mind if the first is blocked.",
        "   For rooms with no obvious special hazard, still give a concrete, "
        "situational caution (e.g. furniture or clutter likely to be in the "
        "path in a rush, a narrow doorway that could bottleneck if several "
        "people leave at once) rather than a vague \"be careful\" with no "
        "reason attached.",
        "5. One brief, concrete general safety reminder relevant to the "
        "situation (e.g. stay low under smoke, don't use elevators, move "
        "calmly but quickly, help others nearby if safe to do so, do a "
        "quick headcount once outside).",
        "",
        "RULES:",
        "- Do NOT reason about or reference any room-adjacency/connectivity "
        "graph -- this is a general safety guide, not a precise routed path. "
        "Use your own judgment about room type and position, not a graph.",
        "- If a room is itself an exit, stairwell, lobby, corridor, porch, or "
        "foyer (i.e. already exit-adjacent), say so plainly, still naming "
        "that room's own nearest exit value if one is given, e.g. \"You are "
        "already near the front exterior door from foyer -- proceed "
        "directly outside.\"",
        (
            "- Every room below has its own \"nearest exit\" value -- always "
            "name that specific exit/door, never say just \"a marked exit\" "
            "or \"the nearest exit\" without naming it."
            if has_markers else
            "- No exits/stairwells were marked on the floor plan for this "
            "building. Use each room's own \"nearest exit\" value as the "
            "named destination where available; if it's \"unknown\", "
            "describe heading toward the building's main entrance or the "
            "most likely exterior wall/corridor in general spatial terms "
            "instead."
        ),
        "- Rooms not on the ground floor (floor > 1) must be told to use the "
        "stairwell, never the elevator, during a fire evacuation.",
        "- Keep every caution PLAUSIBLE and grounded in the room's actual "
        "type/risk data given below -- never invent specific equipment, "
        "brand names, or hazards that weren't implied by the room type or "
        "risk label.",
        (
            "- Do NOT mention \"High Risk rooms\", \"avoid any High Risk "
            "room\", or anything similar anywhere in ANY instruction -- "
            "every room in this building is currently Low Risk or Moderate "
            "Risk, so that warning would be false and just adds noise."
            if not has_high_risk_rooms else
            "- Only tell someone to avoid/route around a room if it is "
            "actually marked High Risk below -- do not invent High Risk "
            "rooms that aren't in the list."
        ),
        "",
        "ROOMS (from this building's hazard analysis):",
    ]
    lines.extend(_gpt_text_route_room_line(r) for r in rooms)
    lines.append("")
    lines.append(
        "Return ONLY a JSON object of this exact shape, one entry per room "
        "above, in the same order, matching room_name and floor_level "
        "exactly as given:"
    )
    lines.append(
        '{"routes": [{"room_name": "...", "floor_level": 1, "instruction": "..."}]}'
    )
    return "\n".join(lines)


@app.route('/gpt-evacuation-routes-text', methods=['POST'])
def gpt_evacuation_routes_text():
    """Generates a simple, plain-English, per-room evacuation guide using
    GPT_MODEL -- see the module comment above this section for why it's
    kept separate from the adjacency-graph-based routing used elsewhere.

    Expects the client's cached /analyze 'rooms' list (lastAnalysisData,
    across however many floors were analyzed -- same shape /analyze
    returns) and, optionally, 'markers' -- a flat list of that floor's
    exit/stair/elevator markers from the marker tool, each carrying its own
    floor_level (see _sanitize_markers_with_floor). Each marker is given a
    stable label ("Exit 1", "Stairwell 2", ...) and a rough compass
    position (see _label_markers_by_floor/_direction_label) so GPT can name
    a SPECIFIC exit per room instead of a generic "marked exit".
    """
    data = request.json or {}
    rooms = [r for r in (data.get('rooms') or []) if r.get('room_name')]
    markers = _sanitize_markers_with_floor(data.get('markers'))

    if not rooms:
        return jsonify({'error': 'No room data provided -- run analysis first.'}), 400

    # Cap defensively -- this is a per-room guide, not meant for an
    # unbounded room list, and keeps the prompt/response comfortably within
    # budget regardless of how large a building someone throws at it.
    rooms = rooms[:150]

    labeled_markers = _label_markers_by_floor(markers)
    marker_lines = _gpt_text_route_marker_lines(labeled_markers)
    has_high_risk_rooms = any(r.get('risk_label') == 'High Risk' for r in rooms)

    prompt_lines = [build_gpt_text_route_prompt(
        rooms, has_markers=bool(marker_lines), has_high_risk_rooms=has_high_risk_rooms
    )]
    if marker_lines:
        prompt_lines.insert(1, "\nMARKED EXITS/STAIRWELLS (from the marker tool):\n" +
                             "\n".join(marker_lines))
    prompt = "\n".join(prompt_lines)

    # Longer per-room instructions (5-7 sentences each, naming a specific
    # exit AND a concrete hazard-specific caution) need noticeably more
    # budget than the old one-liner version -- sized generously per room,
    # well above what ~10-30 rooms would need, since a truncated response
    # silently drops rooms off the end.
    max_tokens = min(16000, max(4000, len(rooms) * 320))

    try:
        parsed, raw_text, truncated = call_gpt4o_json(
            prompt, max_tokens=max_tokens, label="gpt_evacuation_routes_text"
        )
    except GPTResponseError as e:
        logger.error("GPT text evacuation route generation failed: %s", e)
        return jsonify({'error': str(e)}), 500

    routes = parsed.get('routes') if isinstance(parsed, dict) else None
    if not isinstance(routes, list):
        return jsonify({'error': 'GPT returned an unexpected response shape.'}), 500

    # Match back to the original room list by (room_name, floor_level) so
    # the response always carries the SAME risk_label/hazard_index the rest
    # of the UI already trusts -- GPT only supplies 'instruction', nothing
    # about risk classification gets taken from its output.
    lookup = {(r.get('room_name'), r.get('floor_level')): r for r in rooms}
    out = []
    seen = set()
    for entry in routes:
        if not isinstance(entry, dict):
            continue
        key = (entry.get('room_name'), entry.get('floor_level'))
        room = lookup.get(key)
        if room is None:
            # fall back to name-only match in case GPT dropped/altered floor_level
            candidates = [r for r in rooms if r.get('room_name') == entry.get('room_name')]
            room = candidates[0] if len(candidates) == 1 else None
        if room is None:
            continue
        seen.add((room.get('room_name'), room.get('floor_level')))
        out.append({
            'room_name': room.get('room_name'),
            'floor_level': room.get('floor_level'),
            'room_type': room.get('room_type'),
            'risk_color': room.get('risk_color'),
            'risk_label': room.get('risk_label'),
            'instruction': str(entry.get('instruction', '')).strip(),
        })

    # Any room GPT skipped entirely still gets a row, so the guide always
    # covers every room that was analyzed -- same spirit as the old
    # adjacency-based version never silently dropping a room. Prefers that
    # room's own nearest_exit_used value (the same ground truth used in the
    # prompt above); only falls back to the first labeled exit/stairwell for
    # that room's floor, then a generic phrase, if nearest_exit_used isn't
    # available.
    fallback_by_floor = {
        floor: entries[0]['label'] + " (" + entries[0]['direction'] + ")"
        for floor, entries in labeled_markers.items()
        if entries
    }
    for room in rooms:
        key = (room.get('room_name'), room.get('floor_level'))
        if key in seen:
            continue
        target = room.get('nearest_exit_used') or fallback_by_floor.get(room.get('floor_level'))
        caution = _fallback_hazard_caution(room.get('room_type'))
        avoid_clause = ", avoiding any High Risk rooms along the way" if has_high_risk_rooms else ""
        if target:
            instruction = (
                f"Leave the room and head toward {target}{avoid_clause}. "
                f"{caution} Move calmly but quickly, and if this floor is "
                f"above ground level, take the stairs -- never the elevator "
                f"-- during a fire evacuation."
            )
        else:
            instruction = (
                f"Leave the room and head toward the nearest visible exit or "
                f"exterior door{avoid_clause}. {caution} Move calmly but "
                f"quickly, and if this floor is above ground level, take "
                f"the stairs -- never the elevator -- during a fire "
                f"evacuation."
            )
        out.append({
            'room_name': room.get('room_name'),
            'floor_level': room.get('floor_level'),
            'room_type': room.get('room_type'),
            'risk_color': room.get('risk_color'),
            'risk_label': room.get('risk_label'),
            'instruction': instruction,
        })

    return jsonify({
        'success': True,
        'routes': out,
        'truncated': truncated,
    })


@app.route('/save-analysis', methods=['POST'])
def save_analysis():
    """Saves a completed analysis snapshot so it shows up in the history
    panel, and its recommendations show up in the reports panel."""
    data = request.json or {}
    analysis = data.get('analysis')
    if not analysis:
        return jsonify({'error': 'No analysis data provided'}), 400

    ai_recommendations = data.get('ai_recommendations', [])
    floorplan_images = data.get('floorplan_images', [])

    rooms = analysis.get('rooms', [])
    high_risk_count = len([r for r in rooms if r.get('risk_color') == 'red'])

    conn = get_db()
    cur = conn.execute(
        '''INSERT INTO saved_analyses
           (created_at, building_name, building_risk_index, building_risk_label,
            room_count, high_risk_count, cluster_count,
            analysis_json, ai_recommendations_json, floorplan_images_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            datetime.now().isoformat(),
            analysis.get('building_name', 'Unnamed Building'),
            analysis.get('building_risk_index'),
            analysis.get('building_risk_label'),
            len(rooms),
            high_risk_count,
            len(analysis.get('clusters', [])),
            json.dumps(analysis),
            json.dumps(ai_recommendations),
            json.dumps(floorplan_images),
        )
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()

    return jsonify({'success': True, 'id': new_id})


@app.route('/history', methods=['GET'])
def get_history():
    """Lightweight list of saved analyses for the history sidebar panel."""
    conn = get_db()
    rows = conn.execute(
        '''SELECT id, created_at, building_name, building_risk_index,
                  building_risk_label, room_count, high_risk_count, cluster_count
           FROM saved_analyses ORDER BY id DESC'''
    ).fetchall()
    conn.close()
    return jsonify({'success': True, 'history': [dict(r) for r in rows]})


@app.route('/history/<int:analysis_id>', methods=['GET'])
def get_history_item(analysis_id):
    """Full snapshot for re-loading a past analysis back into the results view."""
    conn = get_db()
    row = conn.execute(
        'SELECT * FROM saved_analyses WHERE id = ?', (analysis_id,)
    ).fetchone()
    conn.close()
    if not row:
        return jsonify({'error': 'Not found'}), 404

    item = dict(row)
    item['analysis'] = json.loads(item.pop('analysis_json'))
    item['ai_recommendations'] = json.loads(item.pop('ai_recommendations_json') or '[]')
    item['floorplan_images'] = json.loads(item.pop('floorplan_images_json') or '[]')
    return jsonify({'success': True, 'item': item})


@app.route('/history/<int:analysis_id>', methods=['DELETE'])
def delete_history_item(analysis_id):
    conn = get_db()
    conn.execute('DELETE FROM saved_analyses WHERE id = ?', (analysis_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/reports', methods=['GET'])
def get_reports():
    """Flattens rule-based + AI recommendations from every saved analysis
    into one feed for the reports sidebar panel."""
    conn = get_db()
    rows = conn.execute(
        '''SELECT id, created_at, building_name, analysis_json, ai_recommendations_json
           FROM saved_analyses ORDER BY id DESC'''
    ).fetchall()
    conn.close()

    reports = []
    for row in rows:
        analysis = json.loads(row['analysis_json'])
        ai_recs = json.loads(row['ai_recommendations_json'] or '[]')
        rule_recs = analysis.get('recommendations', [])
        for r in rule_recs:
            reports.append({
                'source_analysis_id': row['id'],
                'created_at': row['created_at'],
                'building_name': row['building_name'],
                'origin': 'rule-based',
                'priority': r.get('priority'),
                'icon': r.get('icon'),
                'room_ref': r.get('room_ref'),
                'pd_section': r.get('pd_section'),
                'message': r.get('message'),
                'action': r.get('action'),
            })
        for r in ai_recs:
            reports.append({
                'source_analysis_id': row['id'],
                'created_at': row['created_at'],
                'building_name': row['building_name'],
                'origin': 'ai',
                'priority': r.get('priority'),
                'icon': r.get('icon'),
                'room_ref': r.get('room_ref'),
                'pd_section': r.get('pd_section'),
                'message': r.get('message'),
                'action': r.get('action'),
            })

    return jsonify({'success': True, 'reports': reports})


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)