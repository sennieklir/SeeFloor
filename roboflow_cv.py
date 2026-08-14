"""
roboflow_cv.py
==============
Optional second, independent source of wall/door geometry for the CV
wallgrid (see wallgrid.py / verify_room_distance in app.py), using a
private Roboflow Workflow instead of relying solely on GPT-4o's own
wall/door tracing.

Roboflow Workflow:
    workspace: PRIV
    workflow:
        seefloor-vseefloor-2-rfdetr-small-t1-logic

Declared Workflow input:
    image

This is INTENTIONALLY additive/defensive, not a replacement for GPT-4o:
- GPT-4o's traced walls/doors remain the primary source.
- Roboflow detections are filtered before being trusted.
- Windows are ignored entirely.
- Any Roboflow failure must NEVER break the GPT-4o analysis pipeline.

Roboflow geometry is only ever ADDED to GPT's trace, never used to
silently remove or move something GPT drew. However, when Roboflow
strongly detects a door/opening sitting on top of a wall GPT traced
as solid, that disagreement is surfaced as a "conflict" (see
merge_wall_door_sources) so the caller/UI can decide what to do with
it, instead of it being invisibly absorbed as a duplicate.

The Roboflow Workflow is called directly through its HTTP API using
requests, so this file does not require inference_sdk.
"""

import logging
import os
import base64
import requests

logger = logging.getLogger(__name__)


# ============================================================
# ROBoflow configuration
# ============================================================

# Real key lives in .env (see .env.example) -- never hardcode it here.
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")

# Roboflow Hosted Inference server.
ROBOFLOW_API_URL = "https://serverless.roboflow.com"

# Your private Roboflow workspace slug.
ROBOFLOW_WORKSPACE = "cenia-claire-l-belaguas"

# Your Workflow ID / slug from the Deploy My API page.
ROBOFLOW_WORKFLOW_ID = "seefloor-vseefloor-2-rfdetr-small-t1-logic"

# The Workflow declares one image input named exactly "image".
ROBOFLOW_IMAGE_INPUT_NAME = "image"


# ============================================================
# Detection filtering
# ============================================================

ROBOFLOW_WALL_CONF_MIN = 0.35
ROBOFLOW_DOOR_CONF_MIN = 0.45

# Noise thresholds, in pixels.
#
# Walls:
# A real wall's short axis can naturally be thin, so we filter
# using the LONG axis instead.
ROBOFLOW_MIN_WALL_LENGTH_PX = 35

# Doors are approximately square, so filter using their SHORT axis.
ROBOFLOW_MIN_DOOR_SIDE_PX = 20

# Dedup tolerance in percent of image dimensions.
ROBOFLOW_DEDUP_TOL_PCT = 3.0

# --------------------------------------------------------------
# Conflict detection (Roboflow disagreeing with GPT, not just
# adding to it).
# --------------------------------------------------------------

# A door has to clear this confidence bar (higher than the base
# ROBOFLOW_DOOR_CONF_MIN filter above) before it's trusted enough
# to flag a conflict against a GPT wall. Low-confidence doors are
# still dropped/kept per the normal filtering, they just don't get
# to accuse GPT of being wrong.
ROBOFLOW_DOOR_CONFLICT_CONF_MIN = 0.60

# How close (in percent of image dimensions) a Roboflow door's
# center has to be to a GPT wall segment (measured as perpendicular
# distance to the segment, clamped to its endpoints) to count as
# "sitting on" that wall.
ROBOFLOW_CONFLICT_DIST_TOL_PCT = 2.5


# ============================================================
# Exceptions
# ============================================================

class RoboflowResponseError(Exception):
    """Raised when the Roboflow Workflow API call fails."""


# ============================================================
# Roboflow Workflow call
# ============================================================

def call_roboflow_wall_door_detection(filepath, max_retries=1):
    """
    Runs the private Roboflow Workflow on one floor-plan image.

    The Workflow endpoint is:

        POST
        https://serverless.roboflow.com/infer/workflows/
            {workspace}/{workflow_id}

    The Workflow input is:

        image

    Returns:
        A list of raw Roboflow prediction dictionaries.

    Raises:
        RoboflowResponseError on authentication, network, HTTP,
        or unexpected response failures.

    IMPORTANT:
        This function does NOT let Roboflow failures break the
        rest of the application. The caller should catch the
        exception and fall back to GPT-only geometry.
    """

    if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY == "YOUR_ROBOFLOW_API_KEY_HERE":
        raise RoboflowResponseError(
            "Please replace ROBOFLOW_API_KEY with your actual Roboflow API key."
        )

    # --------------------------------------------------------
    # Read image and convert to base64
    # --------------------------------------------------------

    try:
        with open(filepath, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        raise RoboflowResponseError(
            f"Could not read floor-plan image '{filepath}': {e}"
        ) from e

    # --------------------------------------------------------
    # Private Workflow endpoint
    # --------------------------------------------------------

    url = (
        f"{ROBOFLOW_API_URL.rstrip('/')}"
        f"/infer/workflows/"
        f"{ROBOFLOW_WORKSPACE}/"
        f"{ROBOFLOW_WORKFLOW_ID}"
    )

    headers = {
        "Content-Type": "application/json",
    }

    # Roboflow Workflow HTTP API expects Workflow inputs
    # under the "inputs" object.
    #
    # The image input is declared as a WorkflowImage, so its
    # external representation is:
    #
    #     {
    #         "type": "base64",
    #         "value": "<base64 image>"
    #     }
    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {
            ROBOFLOW_IMAGE_INPUT_NAME: {
                "type": "base64",
                "value": img_b64,
            }
        },
    }

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            logger.info(
                "Calling Roboflow Workflow '%s' "
                "(attempt %d/%d)",
                ROBOFLOW_WORKFLOW_ID,
                attempt + 1,
                max_retries + 1,
            )

            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=60,
            )

            resp.raise_for_status()

            result = resp.json()

            logger.debug(
                "Raw Roboflow Workflow response: %s",
                result,
            )

            predictions = _extract_predictions_from_workflow_result(result)

            if predictions is None:
                raise RoboflowResponseError(
                    "Roboflow Workflow returned successfully, "
                    "but no usable predictions were found in the response."
                )

            return predictions

        except RoboflowResponseError:
            raise

        except Exception as e:
            last_error = e

            logger.warning(
                "Roboflow Workflow API error on attempt %d/%d: %s",
                attempt + 1,
                max_retries + 1,
                e,
            )

    raise RoboflowResponseError(
        f"Roboflow Workflow failed after "
        f"{max_retries + 1} attempts: {last_error}"
    )


# ============================================================
# Workflow response parsing
# ============================================================

def _extract_predictions_from_workflow_result(result):
    """
    Converts Roboflow Workflow output into the raw prediction list
    expected by the existing conversion/filtering code.

    Roboflow Workflows return a list of dictionaries for image
    batches. The exact output key depends on the Workflow Output
    block, so this parser supports several common forms.

    Supported examples include:

        [
            {
                "predictions": {
                    ...
                }
            }
        ]

    or:

        [
            {
                "result": {
                    ...
                }
            }
        ]

    or a direct prediction dictionary/list.

    Returns:
        list of prediction dictionaries, or None if no recognizable
        prediction data was found.
    """

    if result is None:
        return None

    # --------------------------------------------------------
    # Case 1: Workflow response is a list.
    #
    # One image normally produces one dictionary in the list.
    # --------------------------------------------------------

    if isinstance(result, list):
        if not result:
            return []

        # For this application we submit one image at a time.
        first = result[0]

        if isinstance(first, dict):
            return _extract_predictions_from_dict(first)

        # Some response forms may already return a list of
        # prediction dictionaries.
        if all(isinstance(item, dict) for item in result):
            if any(
                "class" in item or "confidence" in item
                for item in result
            ):
                return result

        return None

    # --------------------------------------------------------
    # Case 2: Direct dictionary response.
    # --------------------------------------------------------

    if isinstance(result, dict):
        return _extract_predictions_from_dict(result)

    return None


def _extract_predictions_from_dict(data):
    """
    Attempts to locate object-detection predictions inside one
    Workflow output dictionary.

    Handles common Roboflow Workflow output names:
        predictions
        result
        detections

    Also handles nested dictionaries where the prediction list
    is under one of those fields.
    """

    if not isinstance(data, dict):
        return None

    # Most likely output names first.
    for key in ("predictions", "result", "detections"):
        if key not in data:
            continue

        value = data[key]

        extracted = _normalise_prediction_container(value)

        if extracted is not None:
            return extracted

    # --------------------------------------------------------
    # Sometimes the Workflow Output may use a custom name.
    #
    # Search one level deeper for a prediction-like container.
    # --------------------------------------------------------

    for value in data.values():

        if isinstance(value, dict):
            extracted = _normalise_prediction_container(value)

            if extracted is not None:
                return extracted

        elif isinstance(value, list):
            extracted = _normalise_prediction_container(value)

            if extracted is not None:
                return extracted

    return None


def _normalise_prediction_container(value):
    """
    Turns a Roboflow prediction container into a list of
    prediction dictionaries.

    Standard object-detection Workflow serialization is generally
    based on Roboflow's inference prediction format.
    """

    if value is None:
        return None

    # Already a list of prediction dictionaries.
    if isinstance(value, list):

        if not value:
            return []

        # Direct list of predictions.
        if all(isinstance(item, dict) for item in value):
            if any(
                "class" in item
                or "confidence" in item
                or "x" in item
                or "width" in item
                for item in value
            ):
                return value

        # Sometimes a wrapper list contains one prediction object.
        flattened = []

        for item in value:
            if isinstance(item, dict):

                nested = _normalise_prediction_container(item)

                if nested is not None:
                    flattened.extend(nested)

        if flattened:
            return flattened

        return None

    # Dictionary containing a standard prediction list.
    if isinstance(value, dict):

        if "predictions" in value:
            return _normalise_prediction_container(
                value["predictions"]
            )

        # A single prediction.
        if (
            "class" in value
            and (
                "confidence" in value
                or "x" in value
                or "width" in value
            )
        ):
            return [value]

        # Some serialized detection responses can contain
        # prediction data under nested fields.
        for key in ("detections", "result"):
            if key in value:
                nested = _normalise_prediction_container(
                    value[key]
                )

                if nested is not None:
                    return nested

    return None


# ============================================================
# Noise filtering
# ============================================================

def _is_noise_wall(pred):
    """
    Drops dimension-label-text false positives shaped like walls.

    Real walls are long even when thin.
    """

    w = pred.get("width", 0) or 0
    h = pred.get("height", 0) or 0

    return max(w, h) < ROBOFLOW_MIN_WALL_LENGTH_PX


def _is_noise_door(pred):
    """
    Drops sliver-shaped false door boxes.

    Real doors are roughly square, so an extremely thin detection
    is treated as noise.
    """

    w = pred.get("width", 0) or 0
    h = pred.get("height", 0) or 0

    return min(w, h) < ROBOFLOW_MIN_DOOR_SIDE_PX


# ============================================================
# Convert wall bounding box -> wall segment
# ============================================================

def _box_to_wall_segment(pred, img_w, img_h):
    """
    Reduces one axis-aligned wall detection to a single line
    segment along its long axis, in percentage coordinates.

    Output schema:

        {
            "x1_pct": ...,
            "y1_pct": ...,
            "x2_pct": ...,
            "y2_pct": ...
        }
    """

    points = pred.get("points") or []

    xs = [
        p["x"]
        for p in points
        if isinstance(p, dict) and "x" in p
    ]

    ys = [
        p["y"]
        for p in points
        if isinstance(p, dict) and "y" in p
    ]

    # --------------------------------------------------------
    # Prefer polygon/segmentation points when available.
    # --------------------------------------------------------

    if not xs or not ys:

        # Fall back to bounding box.
        cx = pred.get("x")
        cy = pred.get("y")

        w = pred.get("width", 0) or 0
        h = pred.get("height", 0) or 0

        if cx is None or cy is None:
            return None

        xs = [
            cx - w / 2,
            cx + w / 2,
        ]

        ys = [
            cy - h / 2,
            cy + h / 2,
        ]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    width = x_max - x_min
    height = y_max - y_min

    # --------------------------------------------------------
    # Horizontal wall
    # --------------------------------------------------------

    if width >= height:

        mid_y = (y_min + y_max) / 2

        x1 = x_min
        y1 = mid_y
        x2 = x_max
        y2 = mid_y

    # --------------------------------------------------------
    # Vertical wall
    # --------------------------------------------------------

    else:

        mid_x = (x_min + x_max) / 2

        x1 = mid_x
        y1 = y_min
        x2 = mid_x
        y2 = y_max

    if img_w <= 0 or img_h <= 0:
        return None

    return {
        "x1_pct": round(x1 / img_w * 100, 2),
        "y1_pct": round(y1 / img_h * 100, 2),
        "x2_pct": round(x2 / img_w * 100, 2),
        "y2_pct": round(y2 / img_h * 100, 2),
    }


# ============================================================
# Convert door bounding box -> door point
# ============================================================

def _box_to_door_point(pred, img_w, img_h):
    """
    Converts one door detection into the app's:

        {
            "x_pct": ...,
            "y_pct": ...,
            "orientation": ...,
            "confidence": ...
        }

    Orientation is estimated from the bounding-box aspect ratio.
    `confidence` is carried through (rather than dropped) so that
    downstream conflict detection can decide whether a given door
    detection is trustworthy enough to flag a disagreement with a
    GPT-traced wall.
    """

    cx = pred.get("x")
    cy = pred.get("y")

    w = pred.get("width", 0) or 0
    h = pred.get("height", 0) or 0

    if (
        cx is None
        or cy is None
        or img_w <= 0
        or img_h <= 0
    ):
        return None

    orientation = (
        "horizontal"
        if w >= h
        else "vertical"
    )

    return {
        "x_pct": round(cx / img_w * 100, 2),
        "y_pct": round(cy / img_h * 100, 2),
        "orientation": orientation,
        "confidence": round(float(pred.get("confidence", 0) or 0), 4),
    }


# ============================================================
# Convert Roboflow predictions -> wallgrid input
# ============================================================

def convert_predictions_to_wallgrid_input(
    predictions,
    img_w,
    img_h,
):
    """
    Filters raw Roboflow predictions and converts them into the
    percentage-based schema expected by wallgrid.py.

    Returns:

        {
            "walls": [...],
            "doors": [...]
        }
    """

    walls = []
    doors = []

    for pred in predictions or []:

        if not isinstance(pred, dict):
            continue

        cls = (
            pred.get("class")
            or ""
        ).lower()

        conf = (
            pred.get("confidence", 0)
            or 0
        )

        # ----------------------------------------------------
        # Wall
        # ----------------------------------------------------

        if cls == "wall":

            if (
                conf < ROBOFLOW_WALL_CONF_MIN
                or _is_noise_wall(pred)
            ):
                continue

            seg = _box_to_wall_segment(
                pred,
                img_w,
                img_h,
            )

            if seg is not None:
                walls.append(seg)

        # ----------------------------------------------------
        # Door
        # ----------------------------------------------------

        elif cls == "door":

            if (
                conf < ROBOFLOW_DOOR_CONF_MIN
                or _is_noise_door(pred)
            ):
                continue

            pt = _box_to_door_point(
                pred,
                img_w,
                img_h,
            )

            if pt is not None:
                doors.append(pt)

        # ----------------------------------------------------
        # Windows intentionally ignored.
        # ----------------------------------------------------

    return {
        "walls": walls,
        "doors": doors,
    }


# ============================================================
# Deduplication helpers
# ============================================================

def _pct_points_close(
    a,
    b,
    tol_pct=ROBOFLOW_DEDUP_TOL_PCT,
):
    return (
        abs(a["x_pct"] - b["x_pct"]) < tol_pct
        and
        abs(a["y_pct"] - b["y_pct"]) < tol_pct
    )


def _segments_overlap(
    a,
    b,
    tol_pct=ROBOFLOW_DEDUP_TOL_PCT,
):
    """
    Rough dedup check.

    Compares both endpoint pairings because the segment start/end
    order is not guaranteed to match between GPT and Roboflow.
    """

    def close(
        p1x,
        p1y,
        p2x,
        p2y,
    ):
        return (
            abs(p1x - p2x) < tol_pct
            and
            abs(p1y - p2y) < tol_pct
        )

    same_order = (
        close(
            a["x1_pct"],
            a["y1_pct"],
            b["x1_pct"],
            b["y1_pct"],
        )
        and
        close(
            a["x2_pct"],
            a["y2_pct"],
            b["x2_pct"],
            b["y2_pct"],
        )
    )

    swapped = (
        close(
            a["x1_pct"],
            a["y1_pct"],
            b["x2_pct"],
            b["y2_pct"],
        )
        and
        close(
            a["x2_pct"],
            a["y2_pct"],
            b["x1_pct"],
            b["y1_pct"],
        )
    )

    return same_order or swapped


# ============================================================
# Conflict detection helpers
# (Roboflow disagreeing with GPT, not just adding to it)
# ============================================================

def _point_to_segment_distance_pct(px, py, seg):
    """
    Perpendicular distance (in percent units) from point (px, py)
    to the wall segment `seg`, clamped to the segment's endpoints
    (i.e. standard point-to-segment distance, not point-to-line).
    """

    x1, y1 = seg["x1_pct"], seg["y1_pct"]
    x2, y2 = seg["x2_pct"], seg["y2_pct"]

    dx = x2 - x1
    dy = y2 - y1

    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq == 0:
        # Degenerate segment (a point) - just measure to that point.
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

    # Project point onto the line, clamped to [0, 1] along the segment.
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5


def _find_door_wall_conflicts(
    rf_doors,
    gpt_walls,
    conf_min=ROBOFLOW_DOOR_CONFLICT_CONF_MIN,
    dist_tol_pct=ROBOFLOW_CONFLICT_DIST_TOL_PCT,
):
    """
    Flags Roboflow doors that sit on top of a GPT-traced wall.

    This is the disagreement case that plain additive merging
    can't catch: if GPT drew a solid wall exactly where Roboflow
    sees a door/opening, the door is nowhere near GPT's own (empty)
    door list, so it never overlaps with anything and would
    otherwise just get silently appended as a "new" door - masking
    the fact that GPT's wall trace is likely wrong at that spot.

    Only doors at or above `conf_min` confidence are considered,
    since this is used to cast doubt on GPT's trace and a weak
    detection shouldn't get that authority.

    Returns:
        list of conflict dicts:

            {
                "type": "door_vs_wall",
                "door": <rf door dict>,
                "wall": <gpt wall dict>,
                "distance_pct": <float>,
            }
    """

    conflicts = []

    for door in rf_doors or []:

        if door.get("confidence", 0) < conf_min:
            continue

        for wall in gpt_walls or []:

            dist = _point_to_segment_distance_pct(
                door["x_pct"],
                door["y_pct"],
                wall,
            )

            if dist < dist_tol_pct:
                conflicts.append({
                    "type": "door_vs_wall",
                    "door": door,
                    "wall": wall,
                    "distance_pct": round(dist, 2),
                })

    return conflicts


# ============================================================
# Merge GPT + Roboflow geometry
# ============================================================

def merge_wall_door_sources(
    gpt_walls,
    gpt_doors,
    rf_walls,
    rf_doors,
):
    """
    Unions GPT-4o's traced walls/doors with Roboflow's filtered
    detections.

    GPT geometry is always preserved - Roboflow is never allowed to
    silently delete or move something GPT drew.

    Roboflow geometry is only added when it does not already
    overlap something GPT found.

    Separately, this also checks for outright DISAGREEMENT: a
    high-confidence Roboflow door sitting on top of a wall GPT
    traced as solid. That case doesn't get merged in as a new door
    (GPT's wall stays, since we don't auto-edit GPT's trace) - it's
    returned as a `conflicts` list so the caller can log it, surface
    it in the UI for manual review, etc. A conflict logged here is a
    signal your GPT trace may be wrong at that spot, not just "extra
    geometry Roboflow happened to notice".

    Returns:

        (
            merged_walls,
            merged_doors,
            added_wall_count,
            added_door_count,
            conflicts,
        )
    """

    gpt_walls = gpt_walls or []
    gpt_doors = gpt_doors or []

    # --------------------------------------------------------
    # Walls
    # --------------------------------------------------------

    merged_walls = list(gpt_walls)
    added_walls = 0

    for rw in rf_walls or []:

        if not any(
            _segments_overlap(rw, gw)
            for gw in gpt_walls
        ):
            merged_walls.append(rw)
            added_walls += 1

    # --------------------------------------------------------
    # Doors
    # --------------------------------------------------------

    merged_doors = list(gpt_doors)
    added_doors = 0

    for rd in rf_doors or []:

        if not any(
            _pct_points_close(rd, gd)
            for gd in gpt_doors
        ):
            merged_doors.append(rd)
            added_doors += 1

    # --------------------------------------------------------
    # Conflicts: Roboflow door vs. GPT solid wall.
    #
    # These are found against the ORIGINAL gpt_walls (not
    # merged_walls) since we only care about genuine GPT trace
    # decisions, not Roboflow walls we just added ourselves.
    # --------------------------------------------------------

    conflicts = _find_door_wall_conflicts(rf_doors, gpt_walls)

    if conflicts:
        for c in conflicts:
            logger.warning(
                "Roboflow/GPT disagreement: Roboflow detected a door "
                "(confidence=%.2f) at (%.1f%%, %.1f%%) which sits "
                "%.2f%% away from a GPT-traced wall spanning "
                "(%.1f%%, %.1f%%) -> (%.1f%%, %.1f%%). GPT's wall was "
                "kept as-is; flagging for manual review.",
                c["door"]["confidence"],
                c["door"]["x_pct"],
                c["door"]["y_pct"],
                c["distance_pct"],
                c["wall"]["x1_pct"],
                c["wall"]["y1_pct"],
                c["wall"]["x2_pct"],
                c["wall"]["y2_pct"],
            )

    return (
        merged_walls,
        merged_doors,
        added_walls,
        added_doors,
        conflicts,
    )