"""
wallgrid.py
============
Builds the occupancy grid from explicit wall/door LINE SEGMENTS instead of
pixel-thickness filtering (see pathfinding.py's image_to_binary_walls, which
this replaces as the recommended primary approach).

Why this is more robust:
- Distinguishing a wall from furniture/dimension-lines by pixel thickness
  alone fails whenever the drawing style uses thin lines for interior
  partitions (very common) -- proven by the demo.py run against the real
  uploaded floor plan, where wall coverage collapsed to 3.7% of the image.
- GPT-4o, by contrast, is already good at semantic recognition: "where are
  the walls and doors" is squarely a vision-language task, not a pixel-
  statistics task. Asking it for (x1,y1)-(x2,y2) wall segments and door
  locations (in percentage coordinates) is a natural extension of the
  exits/stairwells fields already added in the SEEFLOOR prompt patch.
- Python then does ONLY deterministic geometry: rasterize the segments onto
  a grid, punch door-width gaps at door locations, run Dijkstra. No image
  thresholding, no erosion/dilation tuning per drawing style.
"""

import heapq
import math
import os

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont


@dataclass
class WallGridResult:
    grid: np.ndarray            # 2D bool, True = blocked
    cell_size_m: float
    building_width_m: float
    building_depth_m: float
    # (x, y) grid cells that auto-repair (see _repair_direct_crossing) has
    # opened because they were the most plausible location of a door GPT-4o's
    # wall trace missed. Populated lazily as shortest_path_distance() runs;
    # useful for debug logging / a future "here's what we guessed" overlay.
    auto_repaired_cells: list = field(default_factory=list)
    # (x, y) grid cells for every door build_grid_from_segments() actually
    # punched a gap at (GPT's trace + any Roboflow-added doors, already
    # merged by the time this is built -- see app.py's merge_wall_door_sources).
    # This is what lets shortest_path_distance() prefer "widen a real,
    # already-known door" over "cut through whichever wall cells are
    # cheapest" when a room comes up unreachable/detoured -- see
    # _repair_via_nearest_door().
    door_cells: list = field(default_factory=list)
    # Echoed back from build_grid_from_segments()'s parameters so repair
    # logic can widen a door by the same real-world amount the initial
    # punch used, instead of a disconnected hardcoded guess.
    wall_thickness_m: float = 0.15
    door_width_m: float = 0.9
    # (door_x, door_y, start, goal) tuples recording every time
    # _repair_via_nearest_door() successfully routed a start/goal pair
    # through a known door instead of falling back to a blind wall
    # crossing. Separate from auto_repaired_cells (blind-crossing-only)
    # so debug overlays/logs can tell the two repair strategies apart.
    door_repairs: list = field(default_factory=list)


def _pct_to_grid_xy(x_pct, y_pct, grid_w, grid_h):
    x_pct = max(0.0, min(100.0, float(x_pct)))
    y_pct = max(0.0, min(100.0, float(y_pct)))
    x = int(round((x_pct / 100.0) * (grid_w - 1)))
    y = int(round((y_pct / 100.0) * (grid_h - 1)))
    return max(0, min(grid_w - 1, x)), max(0, min(grid_h - 1, y))


def _draw_thick_line(grid, x0, y0, x1, y1, thickness_cells):
    """Rasterizes a line segment onto the grid using Bresenham, with the
    given thickness (in grid cells) so a single-pixel-wide line still
    reliably blocks a multi-cell-wide grid."""
    dx, dy = x1 - x0, y1 - y0
    steps = max(abs(dx), abs(dy), 1)
    r = max(1, thickness_cells // 2)
    for i in range(steps + 1):
        t = i / steps
        cx = int(round(x0 + dx * t))
        cy = int(round(y0 + dy * t))
        for oy in range(-r, r + 1):
            for ox in range(-r, r + 1):
                x, y = cx + ox, cy + oy
                if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                    grid[y, x] = True


def build_grid_from_segments(walls, doors, building_width_m, building_depth_m,
                              cell_size_m=0.1, wall_thickness_m=0.15, door_width_m=0.9,
                              stair_footprints=None, entry_width_m=0.9):
    """
    walls: list of {"x1_pct":.., "y1_pct":.., "x2_pct":.., "y2_pct":..}
    doors: list of {"x_pct":.., "y_pct":.., "orientation": "horizontal"|"vertical"}
           -- orientation tells us whether to clear a horizontal or vertical
              gap in the wall at that point.
    stair_footprints: optional list of
           {"x_min_pct":.., "y_min_pct":.., "x_max_pct":.., "y_max_pct":..,
            "entry_x_pct":.., "entry_y_pct":..}
           -- the rectangular floor-space a staircase run occupies. Blocked
           in full except for an entry_width_m-wide gap centered on
           (entry_x_pct, entry_y_pct), the point where the stair meets
           walkable floor on this level. Without this, a route can cut
           straight across the tread hatching as if it were open floor,
           since nothing else in `walls` marks a staircase run as blocked
           (a single stairwell point is a routing TARGET, not an obstacle).

    This is exactly the schema GPT-4o should output alongside exits/
    stairwells (see INTEGRATION.md).
    """
    grid_w = max(1, round(building_width_m / cell_size_m))
    grid_h = max(1, round(building_depth_m / cell_size_m))
    grid = np.zeros((grid_h, grid_w), dtype=bool)

    thickness_cells = max(1, round(wall_thickness_m / cell_size_m))
    for w in walls:
        x0, y0 = _pct_to_grid_xy(w["x1_pct"], w["y1_pct"], grid_w, grid_h)
        x1, y1 = _pct_to_grid_xy(w["x2_pct"], w["y2_pct"], grid_w, grid_h)
        _draw_thick_line(grid, x0, y0, x1, y1, thickness_cells)

    # Punch door gaps: clear a door-width-wide strip centered on the door
    # point, in the direction the door interrupts (this is what makes the
    # doorway actually walkable instead of a solid wall cell).
    #
    # Punched in BOTH orientations (a "+" shape), not just the one guessed
    # from the door schema's "orientation" field. That field is a guess --
    # GPT-4o's own best read of the drawing, or (for Roboflow-sourced doors,
    # see roboflow_cv.py's _box_to_door_point) just the detection box's
    # aspect ratio -- and a wrong guess used to mean the gap opened along
    # the wall's own axis instead of across it, so the wall stayed
    # effectively solid even though a "door" was recorded right there. This
    # is a large part of why a route could still cross a wall despite a
    # door marker sitting next to it: the gap existed, just facing the
    # wrong way. Clearing both directions costs a few extra cells of open
    # floor right at the doorway and makes the gap actually work regardless
    # of which way the guess was wrong.
    door_half_cells = max(1, round((door_width_m / 2) / cell_size_m))
    door_cells = []
    for d in doors:
        dx, dy = _pct_to_grid_xy(d["x_pct"], d["y_pct"], grid_w, grid_h)
        door_cells.append((dx, dy))
        for x in range(dx - door_half_cells, dx + door_half_cells + 1):
            for y in range(dy - thickness_cells, dy + thickness_cells + 1):
                if 0 <= x < grid_w and 0 <= y < grid_h:
                    grid[y, x] = False
        for y in range(dy - door_half_cells, dy + door_half_cells + 1):
            for x in range(dx - thickness_cells, dx + thickness_cells + 1):
                if 0 <= x < grid_w and 0 <= y < grid_h:
                    grid[y, x] = False

    # Block staircase footprints (see docstring) -- fill the full rectangle,
    # then punch a gap at the entry point the same way a door gap is
    # punched, so the pathfinder can still reach the stairwell target but
    # can't shortcut across the treads.
    for sf in (stair_footprints or []):
        try:
            x0, y0 = _pct_to_grid_xy(sf["x_min_pct"], sf["y_min_pct"], grid_w, grid_h)
            x1, y1 = _pct_to_grid_xy(sf["x_max_pct"], sf["y_max_pct"], grid_w, grid_h)
        except (KeyError, TypeError, ValueError):
            continue
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        grid[y0:y1 + 1, x0:x1 + 1] = True

        entry_x_pct, entry_y_pct = sf.get("entry_x_pct"), sf.get("entry_y_pct")
        if entry_x_pct is None or entry_y_pct is None:
            continue
        ex, ey = _pct_to_grid_xy(entry_x_pct, entry_y_pct, grid_w, grid_h)
        entry_half_cells = max(1, round((entry_width_m / 2) / cell_size_m))
        gx0, gx1 = max(0, ex - entry_half_cells), min(grid_w - 1, ex + entry_half_cells)
        gy0, gy1 = max(0, ey - entry_half_cells), min(grid_h - 1, ey + entry_half_cells)
        grid[gy0:gy1 + 1, gx0:gx1 + 1] = False

    return WallGridResult(grid=grid, cell_size_m=cell_size_m,
                           building_width_m=building_width_m, building_depth_m=building_depth_m,
                           door_cells=door_cells, wall_thickness_m=wall_thickness_m,
                           door_width_m=door_width_m)


def build_graph(result: WallGridResult):
    """4-directional (N/S/E/W) only -- deliberately no diagonal edges.

    Diagonal movement is what produced the "spaghetti" routes cutting
    through furniture/walls at an angle: Dijkstra will happily take a
    diagonal shortcut across open floor space even when that line doesn't
    correspond to any real walking path a person (or the floor plan's own
    corridors) would take. Restricting to cardinal moves forces every
    route into right-angle turns that hug the actual walkable corridor
    shape, which is also what makes two rooms sharing the same corridor
    segment converge onto *identical* cell coordinates instead of two
    near-parallel diagonal lines -- fixing the duplicate/crisscrossing
    lines near shared corridors as a side effect, not a separate fix.
    """
    grid = result.grid
    cs = result.cell_size_m
    h, w = grid.shape
    G = nx.Graph()
    for y in range(h):
        for x in range(w):
            if not grid[y, x]:
                G.add_node((x, y))
    for y in range(h):
        for x in range(w):
            if grid[y, x]:
                continue
            for dxi, dyi, cost in [(1, 0, cs), (0, 1, cs)]:
                nx_, ny_ = x + dxi, y + dyi
                if 0 <= nx_ < w and 0 <= ny_ < h and not grid[ny_, nx_]:
                    G.add_edge((x, y), (nx_, ny_), weight=cost)
    return G


def nearest_free_cell(cell, result: WallGridResult, max_radius=15):
    grid = result.grid
    h, w = grid.shape
    cx, cy = cell
    # If the requested cell is already free, return it.
    if 0 <= cx < w and 0 <= cy < h and not grid[cy, cx]:
        return (cx, cy)

    # Expand outward in Manhattan rings until a free cell is found.
    for r in range(1, max_radius + 1):
        for dx in range(-r, r + 1):
            dy = r - abs(dx)
            for sdy in (dy, -dy) if dy != 0 else (0,):
                x, y = cx + dx, cy + sdy
                if 0 <= x < w and 0 <= y < h and not grid[y, x]:
                    return (x, y)

    # No free cell found within search radius.
    return None


def _wall_crossing_shortcut(grid, start, goal, cell_size_m, wall_penalty_factor):
    """Finds the cheapest path from start to goal when wall cells are
    allowed to be crossed at `wall_penalty_factor`x the cost of a normal
    step. This is deliberately expensive (so Dijkstra still prefers walking
    around through a real doorway whenever that's actually cheaper) but
    finite, so when a room is genuinely sealed off, the path it's forced to
    take reveals exactly which wall cell(s) are standing in for a door that
    GPT-4o's trace missed.

    Returns the list of (x, y) wall cells the cheapest such path crosses
    (empty if start and goal are already connected with zero wall crossings,
    None if goal is unreachable even allowing wall crossings -- e.g. start
    or goal cell is outside the grid bounds)."""
    h, w = grid.shape
    if not (0 <= start[0] < w and 0 <= start[1] < h and 0 <= goal[0] < w and 0 <= goal[1] < h):
        return None

    cs = cell_size_m
    wall_cost = cs * wall_penalty_factor

    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    visited = set()
    # 4-directional only, matching build_graph -- keeps the repaired
    # crossing orthogonal instead of reintroducing a diagonal cut.
    neighbors = [(1, 0, cs), (-1, 0, cs), (0, 1, cs), (0, -1, cs)]

    while pq:
        d, cell = heapq.heappop(pq)
        if cell in visited:
            continue
        visited.add(cell)
        if cell == goal:
            path = [cell]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            path.reverse()
            return [c for c in path if grid[c[1], c[0]]]
        x, y = cell
        for dxi, dyi, base_cost in neighbors:
            nx_, ny_ = x + dxi, y + dyi
            if not (0 <= nx_ < w and 0 <= ny_ < h):
                continue
            ncell = (nx_, ny_)
            if ncell in visited:
                continue
            step_cost = wall_cost if grid[ny_, nx_] else base_cost
            nd = d + step_cost
            if nd < dist.get(ncell, float("inf")):
                dist[ncell] = nd
                prev[ncell] = cell
                heapq.heappush(pq, (nd, ncell))

    return None  # goal genuinely unreachable, even crossing walls (shouldn't happen on a bounded grid)


def _perp_distance_to_segment(point, a, b):
    """Shortest distance from `point` to the line SEGMENT a-b (not the
    infinite line) -- clamps the projection to [0, 1] so a door far past
    either endpoint doesn't read as "on the way"."""
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def _nearest_wall_cell(grid, cell, max_radius):
    """Like nearest_free_cell() but inverted: finds the nearest BLOCKED
    (wall) cell to `cell`, expanding outward ring by ring. Used to snap a
    door marker onto the actual wall line it's meant to interrupt when the
    marker's own coordinate lands a little off that line (a few pixels off
    in the source image is enough for the door's own punch to miss the
    wall entirely -- see _repair_via_nearest_door). Returns None if no
    blocked cell is found within max_radius."""
    h, w = grid.shape
    cx, cy = cell
    if 0 <= cx < w and 0 <= cy < h and grid[cy, cx]:
        return (cx, cy)
    for r in range(1, max_radius + 1):
        for dx in range(-r, r + 1):
            dy = r - abs(dx)
            for sdy in (dy, -dy) if dy != 0 else (0,):
                x, y = cx + dx, cy + sdy
                if 0 <= x < w and 0 <= y < h and grid[y, x]:
                    return (x, y)
    return None


def _punch_door_gap(grid, cx, cy, door_half_cells, thickness_cells):
    """Clears a '+'-shaped gap (both orientations, see build_grid_from_segments)
    centered on (cx, cy). Shared by the initial build and by
    _repair_via_nearest_door so both punch a door the same way."""
    h, w = grid.shape
    for x in range(cx - door_half_cells, cx + door_half_cells + 1):
        for y in range(cy - thickness_cells, cy + thickness_cells + 1):
            if 0 <= x < w and 0 <= y < h:
                grid[y, x] = False
    for y in range(cy - door_half_cells, cy + door_half_cells + 1):
        for x in range(cx - thickness_cells, cx + thickness_cells + 1):
            if 0 <= x < w and 0 <= y < h:
                grid[y, x] = False


def _repair_via_nearest_door(result: WallGridResult, start, goal,
                              max_candidates=4, search_radius_cells=None,
                              snap_radius_m=1.2):
    """STRICT door-first repair, tried before _repair_direct_crossing below.

    This is the direct answer to "the route should have to go through the
    door near it, not just through whichever wall is cheapest": rather than
    letting Dijkstra cut through arbitrary wall cells wherever that happens
    to be least-cost, this looks specifically at the doors GPT-4o/Roboflow
    already identified (result.door_cells -- see build_grid_from_segments)
    and, for any of them that plausibly sit on the way between `start` and
    `goal`, SNAPS the marker onto the nearest actual wall cell (see
    _nearest_wall_cell) and re-punches the door gap there before re-trying
    the real (zero-wall-crossing-cost) pathfind through it.

    The snap step matters: a door marker recorded a little off the traced
    wall line (a few pixels in the source image, or a slightly-off
    Roboflow box center) means the gap punched exactly AT the marker's own
    coordinate can miss the wall it was meant to open -- the door "exists"
    in the data but never actually breaks the wall. Snapping to the
    nearest wall cell within `snap_radius_m` fixes that without needing to
    guess a large, imprecise widening around the marker's raw coordinate.

    A door only counts as "near" this route if it's within
    `search_radius_cells` of the straight line between start and goal AND
    not implausibly far beyond either endpoint -- this is what stops it
    from grabbing some unrelated door on the other side of the building.
    Candidates are tried nearest-to-the-route first; the first one that
    actually connects start to goal is kept (grid mutation is permanent,
    like _repair_direct_crossing's) and recorded in result.door_repairs.

    Returns (dist, path) if a known door produced a connection, or None if
    no known door was near enough / none of the nearby ones actually
    connect (in which case the caller should fall back to
    _repair_direct_crossing as a last resort, not as the first guess).
    """
    if not result.door_cells:
        return None

    grid = result.grid
    h, w = grid.shape
    cs = result.cell_size_m
    if search_radius_cells is None:
        search_radius_cells = max(20, round(3.0 / cs))  # ~3m default "on the way" window

    candidates = []
    for (dx, dy) in result.door_cells:
        on_route_dist = _perp_distance_to_segment((dx, dy), start, goal)
        d_start = math.hypot(dx - start[0], dy - start[1])
        d_goal = math.hypot(dx - goal[0], dy - goal[1])
        if on_route_dist <= search_radius_cells and min(d_start, d_goal) <= search_radius_cells * 2.5:
            candidates.append((on_route_dist, dx, dy))
    candidates.sort(key=lambda c: c[0])

    door_half_cells = max(1, round((result.door_width_m / 2) / cs))
    thickness_cells = max(1, round(result.wall_thickness_m / cs))
    snap_radius_cells = max(3, round(snap_radius_m / cs))

    for _, dx, dy in candidates[:max_candidates]:
        wall_cell = _nearest_wall_cell(grid, (dx, dy), snap_radius_cells)
        if wall_cell is None:
            # Nothing blocked anywhere near this door marker -- it isn't
            # actually standing in for a missed wall gap, so it can't help.
            continue
        wx, wy = wall_cell

        x0, x1 = max(0, wx - door_half_cells), min(w - 1, wx + door_half_cells)
        y0, y1 = max(0, wy - door_half_cells), min(h - 1, wy + door_half_cells)
        patch_before = grid[y0:y1 + 1, x0:x1 + 1].copy()

        _punch_door_gap(grid, wx, wy, door_half_cells, thickness_cells)

        G = build_graph(result)
        if start in G and goal in G:
            try:
                dist = round(nx.dijkstra_path_length(G, start, goal, weight="weight"), 2)
                path = nx.dijkstra_path(G, start, goal, weight="weight")
                result.door_repairs.append({
                    "door_marker_cell": (dx, dy), "snapped_wall_cell": (wx, wy),
                    "start": start, "goal": goal,
                })
                return dist, path
            except nx.NetworkXNoPath:
                pass

        # Didn't connect -- revert this candidate's punch before trying the next.
        grid[y0:y1 + 1, x0:x1 + 1] = patch_before

    return None


def _repair_direct_crossing(result: WallGridResult, start, goal,
                             max_wall_cells, wall_penalty_factor):
    """Auto-repairs the most common CV failure mode: a door GPT-4o's wall
    trace missed or misplaced, which otherwise strands a room as
    'unreachable' or forces Dijkstra into a bogus detour around it (see
    verify_room_distance's docstring in app.py). This is the automatic
    equivalent of the frontend's manual door markers -- instead of requiring
    the user to click the missing opening, it locates it itself.

    Only opens the crossing if it's small (<= max_wall_cells, roughly a
    single wall's thickness -- not a whole extra room). A large crossing
    means the real issue is something else (badly wrong geometry, a
    genuinely unreachable room), and guessing there would create a fake
    shortcut worse than honestly falling back to GPT's own estimate.

    Mutates result.grid in place and records what it opened in
    result.auto_repaired_cells. Returns True if a repair was made."""
    wall_cells = _wall_crossing_shortcut(
        result.grid, start, goal, result.cell_size_m, wall_penalty_factor
    )
    if wall_cells and 0 < len(wall_cells) <= max_wall_cells:
        for (wx, wy) in wall_cells:
            result.grid[wy, wx] = False
        result.auto_repaired_cells.extend(wall_cells)
        return True
    return False


def _grid_xy_to_pct(x, y, grid_w, grid_h):
    """Inverse of _pct_to_grid_xy -- turns a grid cell back into the same
    0-100 percentage coordinate space the rest of the app uses (room
    centroids, markers, etc.), so a Dijkstra path can be handed to Gemini
    as plain x_pct/y_pct points instead of grid indices."""
    x_pct = (x / max(1, grid_w - 1)) * 100.0
    y_pct = (y / max(1, grid_h - 1)) * 100.0
    return round(x_pct, 2), round(y_pct, 2)


def path_to_exact_pct_points(path, result: WallGridResult):
    """Like path_to_pct_waypoints(), but LOSSLESS: only collapses cells
    that are exactly collinear (i.e. sit on the same straight run), never
    drops a real turn, and never caps the point count. Use this for
    anything that will actually be DRAWN as the route line and needs the
    "can't cross a wall" guarantee to be literally true (see
    render_deterministic_route_overlay below) -- path_to_pct_waypoints()'s
    RDP simplification is fine for handing Gemini a rough shape prior (a
    slightly-off hint just means a slightly different generated line), but
    it is NOT fine for a line you're drawing with certainty, because a
    straight chord between two "kept" RDP corners can visually cut through
    a wall that the real cell-by-cell path detoured around through a door
    -- especially on a compact floor plan where rooms/doorways sit close
    enough together that the detour's own deviation falls under tolerance.

    Returns [] if path is empty/None.
    """
    if not path:
        return []
    grid_h, grid_w = result.grid.shape
    pct_points = [_grid_xy_to_pct(x, y, grid_w, grid_h) for (x, y) in path]

    simplified = [pct_points[0]]
    for i in range(1, len(pct_points) - 1):
        (x0, y0), (x1, y1), (x2, y2) = simplified[-1], pct_points[i], pct_points[i + 1]
        # Collinear iff the cross product of (p1-p0) and (p2-p0) is ~0 --
        # exact for the axis-aligned steps a 4-directional grid path takes
        # (every straight run is either constant-x or constant-y), so this
        # never silently drops a real corner the way a distance tolerance could.
        cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if abs(cross) > 1e-9:
            simplified.append((x1, y1))
    simplified.append(pct_points[-1])

    return [{"x_pct": x, "y_pct": y} for (x, y) in simplified]


def _rdp_simplify(points, tolerance):
    """Ramer-Douglas-Peucker simplification on a list of (x, y) points.
    Collapses a long, cell-by-cell Dijkstra path down to just its corners
    (the points where it actually changes direction), which is what you
    want when handing a path to something that will redraw it, not trace
    every grid cell. `tolerance` is in the same units as the points
    (percent, here). Keeps first/last points always."""
    if len(points) < 3:
        return list(points)

    def _perp_dist(pt, a, b):
        (x, y), (ax, ay), (bx, by) = pt, a, b
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return ((x - ax) ** 2 + (y - ay) ** 2) ** 0.5
        t = ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)
        px, py = ax + t * dx, ay + t * dy
        return ((x - px) ** 2 + (y - py) ** 2) ** 0.5

    def _rdp(pts):
        if len(pts) < 3:
            return pts
        a, b = pts[0], pts[-1]
        max_dist, idx = -1.0, -1
        for i in range(1, len(pts) - 1):
            d = _perp_dist(pts[i], a, b)
            if d > max_dist:
                max_dist, idx = d, i
        if max_dist > tolerance:
            left = _rdp(pts[:idx + 1])
            right = _rdp(pts[idx:])
            return left[:-1] + right
        return [a, b]

    return _rdp(list(points))


def path_to_pct_waypoints(path, result: WallGridResult, max_points=6, tolerance_pct=1.0):
    """Converts a raw Dijkstra grid-cell path (as returned by
    shortest_path_distance) into a short list of {"x_pct", "y_pct"}
    waypoints -- the "corners" of the path, in the same percentage
    coordinate space as room centroids/markers/exits.

    This is deliberately NOT the full cell-by-cell path (that can be
    hundreds of points for a long route): it's simplified down to the
    handful of points where the path actually turns, since the intent is
    to hand Gemini a rough "shape" to use as a prior when drawing the
    route line -- not force it to trace an exact polyline. If tightening
    tolerance_pct doesn't get under max_points, points are evenly
    subsampled as a fallback so the hint block never blows up in size.

    Returns [] if path is empty/None.
    """
    if not path:
        return []
    grid_h, grid_w = result.grid.shape
    pct_points = [_grid_xy_to_pct(x, y, grid_w, grid_h) for (x, y) in path]

    simplified = pct_points
    tol = tolerance_pct
    for _ in range(6):
        simplified = _rdp_simplify(pct_points, tol)
        if len(simplified) <= max_points:
            break
        tol *= 1.8

    if len(simplified) > max_points:
        step = (len(simplified) - 1) / (max_points - 1)
        simplified = [simplified[round(i * step)] for i in range(max_points)]

    return [{"x_pct": x, "y_pct": y} for (x, y) in simplified]


def shortest_path_distance(result: WallGridResult, start_pct, goal_pct,
                            auto_repair=True, auto_repair_max_wall_cells=6,
                            auto_repair_wall_penalty=40, auto_repair_detour_ratio=2.5):
    if start_pct is None or goal_pct is None:
        return None, None
    try:
        start_x_pct, start_y_pct = start_pct
        goal_x_pct, goal_y_pct = goal_pct
    except (TypeError, ValueError):
        return None, None

    grid_h, grid_w = result.grid.shape
    start_cell = _pct_to_grid_xy(start_x_pct, start_y_pct, grid_w, grid_h)
    goal_cell = _pct_to_grid_xy(goal_x_pct, goal_y_pct, grid_w, grid_h)
    start = nearest_free_cell(start_cell, result)
    goal = nearest_free_cell(goal_cell, result)
    if start is None or goal is None:
        return None, None

    def _run():
        G = build_graph(result)
        if start not in G or goal not in G:
            return None, None
        try:
            d = nx.dijkstra_path_length(G, start, goal, weight="weight")
            p = nx.dijkstra_path(G, start, goal, weight="weight")
            return round(d, 2), p
        except nx.NetworkXNoPath:
            return None, None

    dist, path = _run()

    if auto_repair:
        # Trigger repair either when there's no path at all (a room stranded
        # by a missing door), or when the path that WAS found is far longer
        # than the straight-line distance would justify (Dijkstra taking a
        # long way around a door that should be there but isn't drawn).
        straight_cells = ((goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2) ** 0.5
        straight_m = straight_cells * result.cell_size_m
        needs_repair = dist is None or (
            straight_m > 0 and dist > straight_m * auto_repair_detour_ratio + result.cell_size_m * 10
        )
        if needs_repair:
            # Try a known, real door near this route FIRST. Only if no
            # door GPT-4o/Roboflow actually identified is near enough (or
            # none of the nearby ones connect) do we fall back to guessing
            # a crossing through whichever wall cells are cheapest -- see
            # _repair_via_nearest_door()'s docstring.
            door_repair = _repair_via_nearest_door(result, start, goal)
            if door_repair is not None:
                new_dist, new_path = door_repair
                if new_dist is not None and (dist is None or new_dist < dist):
                    dist, path = new_dist, new_path
                return dist, path

            repaired = _repair_direct_crossing(
                result, start, goal, auto_repair_max_wall_cells, auto_repair_wall_penalty
            )
            if repaired:
                new_dist, new_path = _run()
                if new_dist is not None and (dist is None or new_dist < dist):
                    dist, path = new_dist, new_path

    return dist, path


# ─────────────────────────────────────────────
# DETERMINISTIC ROUTE-LINE RENDERING
# ─────────────────────────────────────────────
# Draws evacuation route lines directly onto the floor plan image from the
# same route_waypoints_pct this module already computes for every room whose
# distance_source == 'graph_verified' (see verify_room_distance in app.py).
# No generative image model is involved -- a line drawn here is, by
# construction, a sequence of cells that were free (not blocked) in the
# wallgrid, so it cannot cross a wall the way an image-generation model's
# unaided guess can. This exists to replace/augment the Gemini-drawn
# evacuation diagram for floors/rooms where a fully trustworthy route is
# already available, so the drawing is only ever as good (or as uncertain)
# as the underlying Dijkstra result -- never worse, and never inconsistent
# between two runs on the same input.

_ROUTE_COLOR = (39, 174, 96, 255)        # green edges/route lines
_NODE_COLOR = (41, 128, 185, 255)        # blue circle fallback if the start icon file is missing
_EXIT_CIRCLE_COLOR = (20, 20, 20, 255)   # black circle at each computed exit endpoint (was a red X)
_EXIT_LABEL_COLOR = (255, 255, 255, 255)  # white "E1"/"E2"/... text on the exit circle
_EXTINGUISHER_COLOR = (211, 47, 47, 255)  # red fire-extinguisher marker icon
_EXTINGUISHER_OUTLINE = (0, 0, 0, 255)
_LEGEND_BG = (255, 255, 255, 235)
_LEGEND_BORDER = (200, 200, 200, 255)
_LEGEND_TEXT = (30, 30, 30, 255)
_LEGEND_WIDTH_PX = 230

# Cosmetic-only room-start icon (a pin-style marker) pasted in place of the
# plain blue dot -- purely visual, see _paste_start_icon(). If this file is
# ever missing/unreadable the code falls back to the original blue circle
# instead of failing the whole render.
_ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "icons")
_START_ICON_PATH = os.path.join(_ICON_DIR, "start_marker.png")

# Circulation space, not an occupiable room -- these come through in
# rooms_for_floor because build_spatial_graph()/the hazard-scoring pass
# needs them as adjacency nodes (see ROOM_TYPE_WEIGHTS in app.py), but a
# person doesn't "start" an evacuation from a hallway or a stairwell the
# way they start from a bedroom or kitchen. Drawing a node for every one
# of these was why earlier renders showed a cluster of blue dots at every
# corridor junction/landing instead of one dot per actual room. They are
# only skipped for the NODE drawn here -- the risk/color-coded map in
# app.py builds from the unfiltered rooms_analyzed list and is untouched
# by this constant.
DEFAULT_NON_ROOM_NODE_TYPES = frozenset({"corridor", "stairwell", "lobby"})


def _pct_xy_to_px(x_pct, y_pct, img_w, img_h):
    return (x_pct / 100.0) * img_w, (y_pct / 100.0) * img_h


def _draw_arrowhead(draw, p_from, p_to, size=10, fill=_ROUTE_COLOR):
    ang = math.atan2(p_to[1] - p_from[1], p_to[0] - p_from[0])
    left = (p_to[0] - size * math.cos(ang - math.pi / 6), p_to[1] - size * math.sin(ang - math.pi / 6))
    right = (p_to[0] - size * math.cos(ang + math.pi / 6), p_to[1] - size * math.sin(ang + math.pi / 6))
    draw.polygon([p_to, left, right], fill=fill)


def _draw_x_marker(draw, center, size, fill, width):
    """Draws a bold X (two crossing diagonal strokes) centered on `center`.
    No longer used by render_deterministic_route_overlay (exit endpoints
    are now drawn with _draw_exit_marker's numbered black circle instead)
    -- kept as-is so nothing that imports it directly breaks."""
    cx, cy = center
    draw.line([(cx - size, cy - size), (cx + size, cy + size)], fill=fill, width=width)
    draw.line([(cx - size, cy + size), (cx + size, cy - size)], fill=fill, width=width)


def _load_font(size):
    """Best-effort TrueType font loader for the exit-circle labels and the
    legend text. Tries a couple of common system font paths first (crisper
    text), then falls back across Pillow's default-font API (the `size`
    kwarg on ImageFont.load_default only exists on newer Pillow), so a
    missing font/older Pillow never breaks the render -- worst case the
    text is just the small built-in bitmap font."""
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _paste_start_icon(base_rgba, center, icon_path=_START_ICON_PATH, diameter=34):
    """Pastes the room-start icon (static/icons/start_marker.png) centered
    on `center`, scaled to `diameter` px, directly onto `base_rgba` (an
    RGBA Image, composited in place). Purely cosmetic -- replaces the old
    plain blue dot with the same icon used elsewhere for "room/start".
    Returns True if it pasted the icon, False if the icon file couldn't be
    read (caller should fall back to the old blue-circle marker so a
    missing asset never breaks the render)."""
    try:
        icon = Image.open(icon_path).convert("RGBA")
    except Exception:
        return False
    icon = icon.resize((diameter, diameter), Image.LANCZOS)
    cx, cy = center
    base_rgba.alpha_composite(icon, (int(round(cx - diameter / 2)), int(round(cy - diameter / 2))))
    return True


def _draw_exit_marker(draw, center, index, size, fill=_EXIT_CIRCLE_COLOR, text_fill=_EXIT_LABEL_COLOR):
    """Draws a filled black circle at `center` labeled 'E{index}' -- the
    numbered exit-endpoint marker (replaces the old plain red X) so each
    distinct evacuation endpoint reads as which exit it is, in the order
    it was first reached (matching the legend's 'E1, E2, ...' entry)."""
    cx, cy = center
    draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=fill)
    label = f"E{index}"
    font = _load_font(max(11, int(size * 1.05)))
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((cx - tw / 2 - bbox[0], cy - th / 2 - bbox[1]), label, fill=text_fill, font=font)


def _draw_extinguisher_icon(draw, center, size, fill=_EXTINGUISHER_COLOR, outline=_EXTINGUISHER_OUTLINE):
    """Draws a small stylized fire-extinguisher pictogram (cylinder body +
    cap + trigger/hose) centered on `center`. Purely a visual marker for
    user-placed fire_extinguisher markers passed in via
    `extinguisher_markers` -- doesn't read or affect any computed route,
    distance, or recommendation data."""
    cx, cy = center
    body_w, body_h = size * 1.1, size * 2.0
    body = [cx - body_w / 2, cy - body_h / 2 + size * 0.3, cx + body_w / 2, cy + body_h / 2 + size * 0.3]
    draw.rounded_rectangle(body, radius=size * 0.3, fill=fill, outline=outline, width=max(1, int(size * 0.12)))
    cap_w = body_w * 0.7
    cap_bottom = cy - body_h / 2 + size * 0.3
    cap = [cx - cap_w / 2, cap_bottom - size * 0.5, cx + cap_w / 2, cap_bottom]
    draw.rounded_rectangle(cap, radius=size * 0.15, fill=outline)
    handle_y = cap_bottom - size * 0.5
    draw.line([(cx, handle_y), (cx + size * 0.9, handle_y - size * 0.55)], fill=outline, width=max(1, int(size * 0.18)))


def _wrap_text_to_width(text, font, max_width_px, draw):
    """Greedy word-wrap of `text` into a list of lines, each <= max_width_px
    when rendered in `font` (measured via `draw.textlength`). Pure text
    layout helper -- no drawing side effects."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width_px or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_legend(base_img, entries, panel_width=_LEGEND_WIDTH_PX, footer_note=None):
    """Returns a new RGB image = `base_img` with a legend panel appended on
    the right, listing each (icon_fn, label) in `entries`. Purely cosmetic
    compositing on top of the already-finished route render -- runs after
    every route/marker pixel is already drawn, so it never touches the
    computed route/marker data itself, only adds a reference key beside
    the image. Each icon_fn is called as icon_fn(panel_rgba, panel_draw,
    icon_center_x, icon_center_y).

    `footer_note`: optional string rendered word-wrapped beneath the
    icon/label rows, in smaller italic-weight-unavailable body text, e.g. a
    disclaimer about how to read a symbol in a specific case (see the
    stairwell-exit note in render_deterministic_route_overlay). Purely
    cosmetic -- if present the panel grows taller to fit it; if the base
    image is taller than the rows+note need, the panel still matches the
    image height as before."""
    w, h = base_img.size
    note_font = _load_font(13)
    note_lines = []
    note_block_h = 0
    if footer_note:
        # Wrap against a throwaway draw context first since we need the
        # wrapped line count to know the panel height before the real
        # panel image (which is sized off `h`) even exists.
        _measure_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        note_lines = _wrap_text_to_width(footer_note, note_font, panel_width - 32, _measure_draw)
        note_block_h = 14 + len(note_lines) * 18 + 12  # top padding + lines + bottom padding

    rows_bottom = 56 + len(entries) * 44
    panel_h = max(h, rows_bottom + note_block_h)

    panel = Image.new("RGBA", (panel_width, panel_h), _LEGEND_BG)
    pdraw = ImageDraw.Draw(panel)
    pdraw.line([(0, 0), (0, panel_h)], fill=_LEGEND_BORDER, width=2)

    title_font = _load_font(18)
    label_font = _load_font(15)
    pdraw.text((16, 18), "Legend", fill=_LEGEND_TEXT, font=title_font)

    y = 56
    row_h = 44
    icon_cx = 16 + 14
    for icon_fn, label in entries:
        icon_cy = y + row_h / 2 - 6
        icon_fn(panel, pdraw, icon_cx, icon_cy)
        pdraw.text((16 + 36, y + row_h / 2 - 14), label, fill=_LEGEND_TEXT, font=label_font)
        y += row_h

    if note_lines:
        # Thin divider line separating the symbol key from the disclaimer.
        pdraw.line([(16, y + 4), (panel_width - 16, y + 4)], fill=_LEGEND_BORDER, width=1)
        ny = y + 14
        for line in note_lines:
            pdraw.text((16, ny), line, fill=_LEGEND_TEXT, font=note_font)
            ny += 18

    new_h = max(h, panel_h)
    combined = Image.new("RGBA", (w + panel_width, new_h), (255, 255, 255, 255))
    combined.alpha_composite(base_img.convert("RGBA"), (0, 0))
    combined.alpha_composite(panel, (w, 0))
    return combined.convert("RGB")


def render_deterministic_route_overlay(image_path, rooms_for_floor, markers=None,
                                        line_width=4, exit_dedup_pct_radius=2.5, out_path=None,
                                        exclude_node_room_types=DEFAULT_NON_ROOM_NODE_TYPES,
                                        extinguisher_markers=None, show_legend=True,
                                        start_icon_diameter=34, floor_label=None):
    """Renders every room's already-computed route_waypoints_pct (plus its
    centroid as the starting point) as a green polyline directly on top of
    the floor plan image, the start-icon image (static/icons/start_marker.png,
    falls back to a plain blue dot if that file is missing) at each room's
    own start point (one per actual room -- never one per turn, and never
    one for a corridor/stairwell/lobby entry that only exists for graph
    adjacency -- see exclude_node_room_types), and a numbered black circle
    ("E1", "E2", ...) at each distinct exit endpoint the drawn routes
    actually terminate at, numbered in the order it's first reached while
    walking `rooms_for_floor`. When `show_legend` is True (default), a
    legend panel explaining these symbols is appended to the right of the
    image. Returns (PNG bytes, list of room_names drawn with a verified
    route, list of room_names skipped because no graph_verified route
    exists for them) -- unchanged from before; none of this is new output,
    only the pixels look different.

    None of the above touches the underlying computation: which rooms
    count as "drawn"/"skipped", which route each room gets, and where each
    exit endpoint lands are all still decided purely by
    route_path_exact_pct/route_waypoints_pct + distance_source exactly as
    before -- this function only changed how those same points get drawn.

    `exclude_node_room_types`: room_type values that should NOT get a
    node/route drawn even if they have a graph_verified route -- these are
    circulation nodes (hallway/stairwell/lobby), not rooms someone would
    evacuate "from". Pass an empty set/frozenset to draw every room_type
    (old behavior). Rooms with a missing/unrecognized room_type are still
    drawn, so this only ever removes types you explicitly know about.

    Drop-in alternative to call_gemini_generate_evacuation_routes(): same
    idea (a picture of where each room should walk to get out), but the
    line geometry comes straight from the Dijkstra path over the wallgrid
    instead of being redrawn/guessed by an image model -- identical every
    time you run it on the same analysis, and it can never cross a wall
    cell that was actually marked blocked.

    `rooms_for_floor`: list of room dicts as returned by /analyze, expected
    to have centroid_x_pct/centroid_y_pct and (when available)
    route_path_exact_pct (preferred -- see path_to_exact_pct_points; this
    is the lossless version and what actually gets drawn) or
    route_waypoints_pct (older/fallback -- the RDP-simplified hint built
    for Gemini, only used here if the exact points aren't present, e.g.
    data computed before this field existed) + distance_source ==
    'graph_verified'.
    `markers`: accepted for backward compatibility but intentionally NOT
    drawn -- raw user-placed exit/stair/elevator/door markers are input
    hints for the analysis step, not part of this output. The exit circles
    shown here are derived only from where the verified routes actually
    end up, so what's drawn always matches what was actually computed.
    `extinguisher_markers`: optional list of {"x_pct", "y_pct"} dicts for
    user-placed fire_extinguisher markers -- purely a drawing input (see
    _draw_extinguisher_icon), never read by any routing/distance logic.
    Pass None/empty to draw none (and the legend won't mention them).

    `floor_label`: optional floor identifier for THIS image (e.g. 1, "2",
    "2nd Floor"). Purely cosmetic -- when it's provided and doesn't parse
    as floor 1, the legend gets an extra disclaimer line explaining that
    an exit marker here can be the point where a route reaches the
    stairwell node, not a street-level exit: the room's route was computed
    to the nearest stairwell because that's the nearest way out of THIS
    floor, so after going down the stairs a person should continue to
    that stairwell's linked exit on floor 1. Pass None (default) to skip
    the note, e.g. when the caller doesn't track floor numbers or is
    rendering floor 1 itself, where every exit marker already is a real
    exit.
    """
    import io

    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size

    drawn, skipped = [], []
    exit_endpoints_px = []  # (px, py) of each route's final point, deduped below

    for room in (rooms_for_floor or []):
        name = room.get("room_name", "Unknown Room")

        # Circulation node (corridor/stairwell/lobby) -- not an occupiable
        # room, so it doesn't get a start-point node of its own even
        # though it has a valid graph_verified route. Neither "drawn" nor
        # "skipped": it was never a candidate to draw in the first place.
        room_type = (room.get("room_type") or "").lower()
        if room_type in exclude_node_room_types:
            continue

        cx, cy = room.get("centroid_x_pct"), room.get("centroid_y_pct")
        # Prefer the lossless exact path (see path_to_exact_pct_points) --
        # only fall back to the RDP-simplified Gemini-hint waypoints for
        # older room data that predates the exact-points field, since those
        # simplified corners are the reason routes could visually cut
        # through a wall despite the underlying path being valid (see this
        # function's + path_to_exact_pct_points' docstrings).
        waypoints = room.get("route_path_exact_pct") or room.get("route_waypoints_pct")
        verified = room.get("distance_source") == "graph_verified"

        if not verified or not waypoints or cx is None or cy is None:
            skipped.append(name)
            continue

        pct_points = [(cx, cy)] + [(wp["x_pct"], wp["y_pct"]) for wp in waypoints]
        px_points = [_pct_xy_to_px(x, y, w, h) for (x, y) in pct_points]

        draw.line(px_points, fill=_ROUTE_COLOR, width=line_width, joint="curve")

        # Room-start marker: room's own start point only, not every turn.
        # Uses the start-icon image; falls back to the old blue dot if the
        # icon file can't be read so a missing asset never breaks a render.
        sx, sy = px_points[0]
        if not _paste_start_icon(overlay, (sx, sy), diameter=start_icon_diameter):
            r = line_width + 2
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=_NODE_COLOR)

        if len(px_points) >= 2:
            _draw_arrowhead(draw, px_points[-2], px_points[-1], size=line_width * 3)

        exit_endpoints_px.append(px_points[-1])
        drawn.append(name)

    # Dedup exit endpoints so overlapping/nearby routes (e.g. two rooms
    # funneling through the same doorway) draw one marker, not a stack of
    # them -- unchanged dedup logic, just what gets drawn per unique exit
    # changed (numbered black circle instead of a red X).
    dedup_radius_px = (exit_dedup_pct_radius / 100.0) * max(w, h)
    unique_exits = []
    for (ex, ey) in exit_endpoints_px:
        if not any(((ex - ux) ** 2 + (ey - uy) ** 2) ** 0.5 <= dedup_radius_px
                   for (ux, uy) in unique_exits):
            unique_exits.append((ex, ey))

    exit_size = line_width * 4    # bigger than the start icons on purpose
    for i, (ex, ey) in enumerate(unique_exits, start=1):
        _draw_exit_marker(draw, (ex, ey), i, size=exit_size)

    # User-placed fire-extinguisher markers, if any -- drawing only, no
    # effect on routes/exits/dedup computed above.
    extinguisher_markers = extinguisher_markers or []
    ext_size = max(9, line_width * 3)
    for m in extinguisher_markers:
        ex, ey = _pct_xy_to_px(m["x_pct"], m["y_pct"], w, h)
        _draw_extinguisher_icon(draw, (ex, ey), size=ext_size)

    composited = Image.alpha_composite(img, overlay).convert("RGBA")

    if show_legend:
        legend_entries = [
            (lambda pimg, pd, cx, cy: pd.line([(cx - 12, cy), (cx + 12, cy)], fill=_ROUTE_COLOR, width=4),
             "Evacuation route"),
            (lambda pimg, pd, cx, cy: _paste_start_icon(pimg, (cx, cy), diameter=22) or
             pd.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=_NODE_COLOR),
             "Room start point"),
            (lambda pimg, pd, cx, cy: _draw_exit_marker(pd, (cx, cy), 1, size=11),
             "Exit (E1, E2, ...)"),
        ]
        if extinguisher_markers:
            legend_entries.append(
                (lambda pimg, pd, cx, cy: _draw_extinguisher_icon(pd, (cx, cy), size=9),
                 "Fire extinguisher")
            )

        # Upper-floor disclaimer: an "exit" marker on this floor can be a
        # stairwell node a route was sent to (nearest way out of THIS
        # floor), not a ground-level exit -- see floor_label docstring
        # above. Only shown when floor_label is given and isn't floor 1;
        # unknown/unparseable floor_label is treated as "could be an upper
        # floor" so the disclaimer errs toward showing rather than hiding.
        footer_note = None
        if floor_label is not None:
            floor_str = str(floor_label).strip().lower()
            is_floor_one = floor_str in ("1", "1st", "1st floor", "floor 1", "ground", "ground floor")
            if not is_floor_one:
                footer_note = (
                    "Note: if an exit marker here links to the stairs, it means the "
                    "stairs are this room's nearest way out of this floor -- after "
                    "going down, proceed to that stairwell's linked exit on Floor 1."
                )

        final_img = _draw_legend(composited, legend_entries, footer_note=footer_note)
    else:
        final_img = composited.convert("RGB")

    buf = io.BytesIO()
    final_img.save(buf, format="PNG")

    if out_path:
        final_img.save(out_path, format="PNG")

    return buf.getvalue(), drawn, skipped