import unittest

import numpy as np

from app import verify_room_distance
from wallgrid import WallGridResult, shortest_path_distance


class DistanceAlgorithmTests(unittest.TestCase):
    def test_ground_floor_uses_navigable_path(self):
        grid = WallGridResult(
            grid=np.zeros((20, 20), dtype=bool),
            cell_size_m=1.0,
            building_width_m=20.0,
            building_depth_m=20.0,
        )
        room = {
            "distance_to_exit": 99.0,
            "centroid_x_pct": 20.0,
            "centroid_y_pct": 20.0,
        }
        exits = [{"name": "main exit", "x_pct": 80.0, "y_pct": 80.0}]

        verified = verify_room_distance(
            room,
            1,
            grid,
            exits,
            [],
            None,
            False,
            20.0,
            20.0,
            20.0,
            20.0,
            sanity_max_distance=100.0,
        )

        self.assertEqual(verified["distance_source"], "graph_verified")
        self.assertGreater(verified["distance_to_exit"], 0)

    def test_upper_floor_prefers_elevator_transition_when_shorter(self):
        floor_grid = WallGridResult(
            grid=np.zeros((30, 30), dtype=bool),
            cell_size_m=1.0,
            building_width_m=30.0,
            building_depth_m=30.0,
        )
        ground_grid = WallGridResult(
            grid=np.zeros((30, 30), dtype=bool),
            cell_size_m=1.0,
            building_width_m=30.0,
            building_depth_m=30.0,
        )
        room = {
            "distance_to_exit": 99.0,
            "centroid_x_pct": 10.0,
            "centroid_y_pct": 10.0,
        }
        exits = [{"name": "ground exit", "x_pct": 90.0, "y_pct": 90.0}]
        stairwells = [{"name": "main stair", "x_pct": 90.0, "y_pct": 90.0}]
        elevators = [{"name": "main elevator", "x_pct": 10.0, "y_pct": 10.0}]

        verified = verify_room_distance(
            room,
            2,
            floor_grid,
            exits,
            stairwells,
            ground_grid,
            False,
            30.0,
            30.0,
            30.0,
            30.0,
            sanity_max_distance=100.0,
            elevators_list=elevators,
        )

        self.assertIn("elevator", verified["cv_target_used"].lower())
        self.assertEqual(verified["distance_source"], "graph_verified")

    def test_upper_floor_uses_ground_floor_exit_targets_for_landing_leg(self):
        floor_grid = WallGridResult(
            grid=np.zeros((20, 20), dtype=bool),
            cell_size_m=1.0,
            building_width_m=20.0,
            building_depth_m=20.0,
        )
        ground_grid = WallGridResult(
            grid=np.zeros((20, 20), dtype=bool),
            cell_size_m=1.0,
            building_width_m=20.0,
            building_depth_m=20.0,
        )
        room = {
            "distance_to_exit": 99.0,
            "centroid_x_pct": 10.0,
            "centroid_y_pct": 10.0,
        }
        current_floor_exits = []
        ground_floor_exits = [{"name": "ground exit", "x_pct": 80.0, "y_pct": 80.0}]
        stairwells = [{"name": "main stair", "x_pct": 50.0, "y_pct": 50.0}]

        verified = verify_room_distance(
            room,
            2,
            floor_grid,
            current_floor_exits,
            stairwells,
            ground_grid,
            False,
            20.0,
            20.0,
            20.0,
            20.0,
            sanity_max_distance=100.0,
            ground_floor_exits_list=ground_floor_exits,
        )

        self.assertEqual(verified["distance_source"], "graph_verified")
        self.assertIn("ground exit", verified["cv_target_used"])


class AutoRepairTests(unittest.TestCase):
    """No markers involved anywhere in this class -- these grids simulate
    GPT-4o's wall trace missing a door entirely, which is exactly the
    situation that used to force a fall-back to gpt_estimate_only."""

    def test_missing_door_is_self_healed(self):
        # Two rooms split by a solid dividing wall with NO door gap drawn.
        grid = np.zeros((10, 11), dtype=bool)
        grid[:, 5] = True
        result = WallGridResult(grid=grid, cell_size_m=0.5, building_width_m=5.5, building_depth_m=5.0)

        dist, path = shortest_path_distance(result, (10.0, 50.0), (90.0, 50.0))

        self.assertIsNotNone(dist)
        self.assertGreater(dist, 0)
        # Only a door-sized gap should have been opened, not the whole wall.
        self.assertLessEqual(len(result.auto_repaired_cells), 6)
        self.assertGreater(int(grid[:, 5].sum()), 0)

    def test_verify_room_distance_stays_graph_verified_without_markers(self):
        grid = np.zeros((10, 11), dtype=bool)
        grid[:, 5] = True
        floor_grid = WallGridResult(grid=grid, cell_size_m=0.5, building_width_m=5.5, building_depth_m=5.0)
        room = {
            "distance_to_exit": 99.0,
            "centroid_x_pct": 10.0,
            "centroid_y_pct": 50.0,
        }
        exits = [{"name": "main exit", "x_pct": 90.0, "y_pct": 50.0}]

        verified = verify_room_distance(
            room, 1, floor_grid, exits, [], None, False, 5.5, 5.0, 5.5, 5.0,
            sanity_max_distance=20.0,
        )

        self.assertEqual(verified["distance_source"], "graph_verified")

    def test_large_gap_is_not_repaired(self):
        # A wall interrupted by something too wide to plausibly be a door
        # (e.g. a whole missing room boundary) should NOT be auto-opened --
        # that's a different, bigger problem than a missed door.
        grid = np.zeros((30, 31), dtype=bool)
        grid[:, 13:18] = True  # 5-cell-thick barrier, wider than any real wall
        result = WallGridResult(grid=grid, cell_size_m=0.5, building_width_m=15.5, building_depth_m=15.0)

        dist, path = shortest_path_distance(
            result, (10.0, 50.0), (90.0, 50.0), auto_repair_max_wall_cells=2
        )

        # With the cap set unrealistically low, repair should refuse and the
        # room should genuinely come back unreachable rather than faking a
        # doorway through an implausibly large opening.
        self.assertIsNone(dist)
        self.assertEqual(result.auto_repaired_cells, [])


if __name__ == "__main__":
    unittest.main()
