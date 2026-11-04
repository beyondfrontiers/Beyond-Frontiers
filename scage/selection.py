"""Which views to visit, and in what order: paper Eq. 4, with one view
allowed to cover several clusters.

Each Delta-Net cluster gets its own target (scage.viewpoints). But a camera at
one of them often sees other clusters too, and the targets of neighbouring
clusters often sit nearly on top of each other. So:

1. Coverage. For every candidate view, the anomalous points it would resolve
   are raycast: the point itself and a probe probe_dist along its Delta-Net
   bearing (into the unobserved space it points at) are both inside the image,
   within see_range and not behind an obstacle (AnomalyManager.visible). Seeing
   a rim without seeing past it does not resolve it. The view's anomaly mass S
   is the summed score of those points times the 0.05 m cell area: anomalous
   surface, in m^2, weighted by how anomalous.

2. Clubbing. Views within group_trans and group_angle of each other form one
   group (greedy, around the highest g * S). A group also gets its mean pose:
   position averaged in R^3, orientation by the Karcher mean on SO(3), levelled
   (scage.so3). The group is represented by whichever of its members, or that
   mean, scores the best g * S over the union of what its members see. The mean
   only counts if it passes every target safety check itself: an average of
   two safe poses need not be safe.

3. Tour. Greedy weighted set cover, on Eq. 4:
       u = g * S_left / max(d, eps)
   S_left counts only the anomalous points no earlier stop sees. d is the travel
   distance (FlightSpace) from the previous stop, or from the robot for the first
   stop. This repeats up to tour_len stops. A view the robot cannot reach is dropped.
"""

import numpy as np

from scage import so3
from scage.geometry import pose_difference


def select_views(m, check, robot_pos, A_xyz, A_s, A_probe, sc, field=None):
    """m: AnomalyManager holding this Delta-Net run's cluster targets.
    check(P): None if a camera at P passes every target safety check.
    A_xyz, A_s, A_probe: the anomalous points, their scores, and a point a
    step along each one's Delta-Net bearing.
    Leaves the chosen views as m's targets, with utility set, and the tour in
    m.tour. -> stats dict."""
    see, tol = float(sc["see_range"]), float(sc.get("vis_tol", 0.2))
    area = float(sc["cloud_voxel"]) ** 2
    min_mass, eps, tour_len = float(sc["min_mass"]), m.eps_dist, int(sc["tour_len"])
    A_w = np.asarray(A_s, float) * area
    st = dict(candidates=0, groups=0, mean_tried=0, mean_valid=0, kinds={},
              unreachable=0, tour=[])
    m.tour = []

    m.gain_adjustment()
    m.filter_frontiers()
    fts = m.valid_frontiers
    st["candidates"] = len(fts)
    if not fts or not len(A_xyz):
        m.remove_frontiers([ft.id for ft in fts])
        return st

    # 1. coverage
    P = np.stack([ft.pose6d for ft in fts])
    V = m.visible(P, A_xyz, see, tol, A_probe)
    g = np.array([ft.u_gain for ft in fts])
    value = g * (V @ A_w)

    # 2. clubbing
    dt, dr = pose_difference(P, P)
    close = (dt < m.group_trans) & (dr < m.group_angle)
    taken = np.zeros(len(fts), bool)
    groups = []
    for i in np.argsort(-value):
        if not taken[i]:
            mem = np.flatnonzero(close[i] & ~taken)
            taken[mem] = True
            groups.append(mem[np.argsort(-value[mem])])
    st["groups"] = len(groups)

    means = {}
    for k, mem in enumerate(groups):
        if len(mem) > 1:
            T = so3.mean_pose(P[mem], np.maximum(value[mem], 1e-9))
            st["mean_tried"] += 1
            if check(T[:3, 3]) is None and abs(T[2, 2]) <= m.filter_max_vd_z + 1e-9:
                means[k] = T
    st["mean_valid"] = len(means)
    if means:
        Tm = np.stack(list(means.values()))
        gm = m.history_gain(Tm, m.expected_gain(Tm))
        Vm = m.visible(Tm, A_xyz, see, tol, A_probe)
        means = {k: (Tm[i], gm[i], Vm[i]) for i, k in enumerate(means)}

    reps = []
    for k, mem in enumerate(groups):
        union = V[mem].any(0)
        kind = "cluster" if len(mem) == 1 else "member"
        opts = [(P[i], g[i], V[i], kind, fts[i]) for i in mem]
        if k in means:
            opts.append(means[k] + ("karcher", None))
        best = opts[int(np.argmax([o[1] * ((o[2] & union) @ A_w) for o in opts]))]
        reps.append(dict(pose=best[0], g=float(best[1]), vis=best[2], kind=best[3],
                         members=len(mem), ft=best[4]))

    keep = {r["ft"].id for r in reps if r["ft"] is not None}
    m.remove_frontiers([ft.id for ft in fts if ft.id not in keep])
    for r in reps:
        if r["ft"] is None:
            r["ft"] = m.add_view(r["pose"])
            r["ft"].gain = r["ft"].u_gain = r["g"]
        vis, ft = r["vis"], r["ft"]
        w = A_w[vis]
        st["kinds"][r["kind"]] = st["kinds"].get(r["kind"], 0) + 1
        ft.features.update(
            mass=float(w.sum()), n_vis=int(vis.sum()), members=r["members"], kind=r["kind"],
            score=float(A_s[vis].mean()) if vis.any() else 0.0, tour_rank=None,
            anchor=(A_xyz[vis] * w[:, None]).sum(0) / w.sum() if w.sum() > 0
            else np.asarray(ft.pos3d) + np.asarray(ft.view_direction))

    # 3. travel distance and the tour
    pos = np.stack([np.asarray(r["ft"].pos3d, float) for r in reps])
    if field is None:
        field = m.space.field(robot_pos)
    if field is None:                     # the robot cannot join the graph at all
        st["straight_line"] = True
        d = np.linalg.norm(pos - np.asarray(robot_pos, float), axis=1)
    else:
        d = m.space.travel(field, pos)
    reach = np.isfinite(d)
    st["unreachable"] = int((~reach).sum())
    m.remove_frontiers([reps[i]["ft"].id for i in np.flatnonzero(~reach)])
    reps, pos, d = [r for r, ok in zip(reps, reach) if ok], pos[reach], d[reach]
    if not reps:
        return st
    Vr = np.stack([r["vis"] for r in reps])
    gr = np.array([r["g"] for r in reps])
    u0 = gr * (Vr @ A_w) / np.maximum(d, eps)
    for r, u, di in zip(reps, u0, d):
        r["ft"].utility = float(u)
        r["ft"].features["travel"] = float(di)

    resid, left, tour = A_w.copy(), set(range(len(reps))), []
    for rank in range(tour_len):
        S_left = Vr @ resid
        u = gr * S_left / np.maximum(d, eps)
        cand = [i for i in left if S_left[i] >= min_mass and np.isfinite(d[i])]
        if not cand:
            break
        i = max(cand, key=lambda j: u[j])
        tour.append(i)
        left.discard(i)
        resid[Vr[i]] = 0.0
        ft = reps[i]["ft"]
        ft.features["tour_rank"] = rank
        st["tour"].append(dict(id=ft.id, pos=pos[i].round(3).tolist(), g=round(gr[i], 2),
                               mass=round(float(S_left[i]), 4), d=round(float(d[i]), 2),
                               kind=reps[i]["kind"], members=reps[i]["members"]))
        if rank + 1 < tour_len:
            f = m.space.field(pos[i])
            d = m.space.travel(f, pos) if f is not None else np.full(len(pos), np.inf)
    m.tour = [reps[i]["ft"].id for i in tour]
    return st
