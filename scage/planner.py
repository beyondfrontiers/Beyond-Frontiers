"""OMPL RRT* on the flight space's validity rule.

Built on FrontierNet's OccupancyGrid3DPathPlanner: the OMPL setup, the
straight-line state space and the path densification (interpolate_path) are
its code, unchanged. What differs:

- every state and every edge (checked every motion_check_step) must pass
  FlightSpace.valid: known free, under the roof, clear of obstacles;
- only exact solutions are accepted. When RRT* runs out of time without
  reaching the goal, OMPL returns an *approximate* path that stops short of it,
  and FrontierNet's solve() reported success. interpolate_path() then replaces
  the last waypoint with the goal, so the final segment - from wherever the tree
  got to, to the goal - is never checked. In the first runs that segment ran
  through unknown space; the robot's per-step re-check caught it and replanned,
  got another approximate path, and so on (210 of 217 rejected steps in s2);
- an invalid start is moved to the nearest valid point of the flight space,
  not just to the nearest free voxel that is clear of obstacles;
- the path is densified at trans_step / rot_step per step (FrontierNet: 0.3 m,
  0.3 rad), so each exploration step covers more ground.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from scipy.spatial import KDTree

from scage.geometry import (compute_alignment_transforms, interpolate_waypoints,
                            pose2posquat, pose_difference, posquat2pose)


class FlightPlanner:
    MAX_TRANS_STEP: float = 0.30  # meters per interp step
    MAX_ROT_STEP: float = 0.30  # radians per interp step

    def __init__(self, params, space):
        self.sp = ob.RealVectorStateSpace(3)
        self.set_bounds(params.get("bounds", [-5, 5, -5, 5, -5, 5]))
        self._occ_kdt: Optional[KDTree] = None
        self._min_dist2occ: float = float(params.get("min_dist2occ", 0.1))

        # Start/goal
        self.start_pos: Optional[np.ndarray] = None
        self.goal_pos: Optional[np.ndarray] = None
        self.start_quat: Optional[np.ndarray] = None
        self.goal_quat: Optional[np.ndarray] = None

        # OMPL setup
        self.ss = og.SimpleSetup(self.sp)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.sp.setup()
        self.ss.getSpaceInformation().setStateValidityCheckingResolution(0.01)
        self.solution: List[Dict[str, np.ndarray]] = []

        self.space = space
        sc = params.get("scage", {})
        self.MAX_TRANS_STEP = float(sc.get("trans_step", self.MAX_TRANS_STEP))
        self.MAX_ROT_STEP = float(sc.get("rot_step", self.MAX_ROT_STEP))
        self.stats = dict(exact=0, approximate=0, failed=0)

    def set_bounds(self, bounds_input: Sequence[float]) -> None:
        """
        bounds_input: [low_x, high_x, low_y, high_y, low_z, high_z]
        """
        assert (
            len(bounds_input) == 6
        ), "Bounds must be [low_x, high_x, low_y, high_y, low_z, high_z]"
        lx, hx, ly, hy, lz, hz = map(float, bounds_input)
        assert (
            lx < hx and ly < hy and lz < hz
        ), "Bounds must have low < high for each axis"

        bounds = ob.RealVectorBounds(3)
        bounds.setLow(0, lx)
        bounds.setHigh(0, hx)
        bounds.setLow(1, ly)
        bounds.setHigh(1, hy)
        bounds.setLow(2, lz)
        bounds.setHigh(2, hz)
        self.sp.setBounds(bounds)

    def update_space(self, occ_vx=None):
        """Obstacles only; free space is the flight space's lattice lookup."""
        if occ_vx is not None:
            occ_vx = np.asarray(occ_vx, dtype=float)
            assert occ_vx.ndim == 2 and occ_vx.shape[1] == 3, "occ_vx should be (M,3)"
            self._occ_kdt = KDTree(occ_vx)

    def isfree(self, point):
        return bool(self.space.flyable(np.asarray(point, float)[None])[0])

    def isoccupied(self, point):
        return self._occ_kdt is not None and \
            self._occ_kdt.query(np.asarray(point, float))[0] <= self._min_dist2occ

    def is_state_valid(self, state):
        return self.space.valid((state[0], state[1], state[2]))

    def set_start_goal(self, start, goal):
        """Pose fields only, no OMPL problem - for paths not planned by OMPL."""
        start = pose2posquat(start) if isinstance(start, np.ndarray) else start
        goal = pose2posquat(goal) if isinstance(goal, np.ndarray) else goal
        self.start_pos = np.asarray(start["pos"], float).copy()
        self.goal_pos = np.asarray(goal["pos"], float).copy()
        self.start_quat = np.asarray(start["quat"], float).copy()
        self.goal_quat = np.asarray(goal["quat"], float).copy()

    def update_start_goal(self, start, goal):
        self.set_start_goal(start, goal)
        if not self.space.valid(self.start_pos):
            q = self.space.nearest_valid(self.start_pos)
            if q is None:
                return False
            self.start_pos = q
        if not self.space.valid(self.goal_pos):
            return False
        s, g = ob.State(self.sp), ob.State(self.sp)
        for st, p in ((s, self.start_pos), (g, self.goal_pos)):
            st()[0], st()[1], st()[2] = map(float, p)
        self.ss.clear()
        self.ss.setStartAndGoalStates(s, g, 0.1)
        return True

    def solve(self, time_limit=5.0, method="rrtstar"):
        si = self.ss.getSpaceInformation()
        planner = {"rrtstar": og.RRTstar, "rrtconnect": og.RRTConnect,
                   "rrt": og.RRT}[method](si)
        self.ss.setPlanner(planner)
        status = self.ss.solve(float(time_limit))
        if status.getStatus() == ob.PlannerStatus.EXACT_SOLUTION:
            self.ss.simplifySolution()
            self.stats["exact"] += 1
            return True
        key = "approximate" if status.getStatus() == ob.PlannerStatus.APPROXIMATE_SOLUTION \
            else "failed"
        self.stats[key] += 1
        return False

    def interpolate_path(
        self,
        num_interp_points: int = 1,
        external_waypoints: Optional[List[Sequence[float]]] = None,
    ) -> Optional[List[Dict[str, np.ndarray]]]:
        """
        Build a dense, orientation-aware path. If `external_waypoints` is given, uses those.
        Otherwise uses the OMPL solution path. Orientation is:
          - start quat at first,
          - goal quat at last,
          - for intermediates: face the next waypoint.
        """
        try:
            if external_waypoints is None:
                path = self.ss.getSolutionPath()
                if path is None or path.getStateCount() == 0:
                    print("No OMPL path available.")
                    return None
                waypoints = [[s[0], s[1], s[2]] for s in path.getStates()]
            else:
                waypoints = [
                    np.asarray(w, dtype=float).tolist() for w in external_waypoints
                ]
                if len(waypoints) < 2:
                    print("Need at least two waypoints for interpolation.")
                    return None

            # Ensure endpoints match start/goal positions
            waypoints[0] = np.asarray(self.start_pos, dtype=float).tolist()
            waypoints[-1] = np.asarray(self.goal_pos, dtype=float).tolist()

            # Build coarse (pos, quat) sequence with
            coarse: List[Dict[str, np.ndarray]] = []
            coarse.append(
                {
                    "pos": np.asarray(self.start_pos, dtype=float),
                    "quat": np.asarray(self.start_quat, dtype=float),
                }
            )

            for i in range(1, len(waypoints) - 1):
                pos = np.asarray(waypoints[i], dtype=float)
                prev_quat = coarse[-1]["quat"]

                if i == len(waypoints) - 2:
                    # second last: reuse previous orientation
                    coarse.append({"pos": pos, "quat": prev_quat})
                    continue

                next_pos = np.asarray(waypoints[i + 1], dtype=float)
                v = next_pos - pos
                n = np.linalg.norm(v)
                if n > 1e-8:
                    v /= n
                    # Compute alignment transform for a single origin
                    t_mat = compute_alignment_transforms(
                        origins=[pos],
                        align_vec=v,
                        align_axis=[0, 0, 1],
                        appr_vec=[0, 0, -1],  # CV camera convention
                        appr_axis=[0, 1, 0],
                    )[0]
                    quat = pose2posquat(t_mat)["quat"]
                    coarse.append({"pos": pos, "quat": quat})
                else:
                    coarse.append({"pos": pos, "quat": prev_quat})

            # Append goal
            coarse.append(
                {
                    "pos": np.asarray(self.goal_pos, dtype=float),
                    "quat": np.asarray(self.goal_quat, dtype=float),
                }
            )

        except Exception as e:
            print(f"No solution found: {e}")
            return None

        # Densify via segment-wise interpolation
        dense: List[Dict[str, np.ndarray]] = []
        for i in range(len(coarse) - 1):
            a, b = coarse[i], coarse[i + 1]

            # Spacing policy based on translation and rotation magnitudes
            tdiff, rdiff = pose_difference(
                posquat2pose(a).reshape(1, 4, 4),
                posquat2pose(b).reshape(1, 4, 4),
            )
            # Extract scalar (pose_difference returns (N,M))
            td = float(np.asarray(tdiff)[0, 0])
            rd = float(np.asarray(rdiff)[0, 0])

            n_trans = int(np.ceil(td / self.MAX_TRANS_STEP))
            n_rot = int(np.ceil(rd / self.MAX_ROT_STEP))
            n = max(1, n_trans, n_rot, int(num_interp_points))

            seg = interpolate_waypoints(a, b, num_interp_points=n)
            if i == 0:
                dense.extend(seg)
            else:
                # Avoid duplicating previous endpoint
                dense.extend(seg[1:])

        self.solution = dense
        return self.solution

    def get_solution_path(self, return_type: str = "dict"):
        """
        Return the dense path.
        return_type: "dict" -> [{'pos':..., 'quat':...}, ...]
                     "mat"  -> [4x4, 4x4, ...]
        """
        assert return_type in {"dict", "mat"}, "return_type must be 'dict' or 'mat'"
        if return_type == "dict":
            return self.solution
        mats = [posquat2pose(sol) for sol in self.solution]
        return mats
