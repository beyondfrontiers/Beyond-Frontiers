# Delta-Net — boundary scoring, inference only

Self-contained. Give it a point cloud, get a per-point score saying how much
each point looks like the **outside the learnt distribution**, plus the direction the unobserved space lies in.

```
delta_net.py                      architecture, inference path only
infer.py                          preprocessing -> model -> outputs
ptv3/                             the PTv3 backbone, vendored from Pointcept
../model_weights/delta_net.pth    trained weights, 6,983,171 params
example/                          one capture with its label, to check the install
```

This folder lives inside the SCAGE project (`FrontierNet/`, see the README
there), which uses it through `scage/anomaly.py`. Apart from the weights in
`../model_weights/`, nothing outside this folder is needed to run the model on
its own. `ptv3/` is 56 KB of Pointcept's PTv3
copied in, so there is no second repository to clone — see `ptv3/__init__.py`
for exactly what was changed (three removals, none of them on a code path this
model reaches). `example/` is only there so you can verify a fresh install
reproduces the numbers below; delete it if you don't want it.

## Install

Everything is a normal pip package. Versions this was built and tested against:

```
torch 2.5.1+cu124    spconv (cu124)      torch_scatter 2.1.2+pt25cu124
timm 0.9.2           addict 2.4.0        numpy 2.2.6
scipy 1.15.3         open3d 0.19.0       flash-attn 2.8.3.post1  (optional)
```

`flash-attn` is optional. Without it the backbone falls back to plain softmax
attention, which shares the same weights and computes the same thing —
measured on the bundled example, per-point scores differ by 5e-3 on average
and precision/recall by under 0.005. A GPU with ~3 GB free.

## Run it

From the project root (`FrontierNet/`):

```bash
python deltanet/infer.py                                    # the bundled example
python deltanet/infer.py --cloud my_scan.ply --out out/     # any cloud, no label needed
```

On the bundled example you should see, give or take the run-to-run spread:

```
coverage 99.7% of points, 2.1 patches each
score mean 0.122  p90 0.301  max 0.960    above 0.25: 12.1% of points
agreement between overlapping patches: 0.009
against the stored label: MAE 0.031  corr +0.946  precision 0.926  recall 0.823
```

## What you must do to the data

All six steps are in `infer.py`; this is what they are and why.

**1. Z-up, metres.** The model was trained in a single gravity-aligned frame
and is **not** rotation invariant about x or y — floors were horizontal and
walls vertical in everything it saw. Yaw about z is fine, that was augmented
over the full circle. Convert a live cloud into this frame rather than
canonicalising each chunk.

**2. Voxelise at 0.05 m.** One point per 5 cm cell. The score is built from
local neighbourhood statistics, so a denser cloud shifts them and a sparser
one reads as holes.

**3. Cut patches.** The model runs on patches, never a whole scene: the
nearest neighbours of a seed point, at most 4096, discarded past a 2.0 m
radius. Seeds come from farthest-point sampling so patches tile the cloud.
`--overlap` controls how many times over (default 3.0, about 2 patches per
point).

**4. Recentre and rescale each patch.** Subtract the seed *snapped to the
voxel grid* — `round(seed/0.05)*0.05`, not the raw seed, so the integer grid
stays consistent between patches — then divide by **2.0**. Both `coord` and
`feat` are these normalised coordinates. There is no colour input; RGB is
carried through for visualisation only.

**5. Deduplicate grid cells.** `grid_coord` must hold unique non-negative
integer cells. Duplicates make spconv produce garbage silently rather than
raise.

**6. Run in fp32.** spconv's bf16/fp16 path fails here with `can't find
suitable algorithm for 0`. Training used bf16; inference cannot.

Point counts vary per patch by design — PTv3 is offset-based, so a batch is a
concatenation plus a cumulative-sum `offset`, not a padded tensor.

## What comes out

One denoising displacement vector per point, `delta`, learned with a Chamfer-denoising objective:

| | |
|---|---|
| `‖delta‖` | raw correction magnitude, used as an anomaly response during exploration; not constrained to [0, 1] |
| `delta/‖delta‖` | predicted correction direction for nonzero displacements |

The score is **not** rescaled by the 2.0 — only the input coordinates were,
and the target was always in score units.

Points covered by several patches take the mean, and the standard deviation
across patches is reported as a self-consistency check (0.009 on the example
against a mean score of 0.12 — the patch split is not driving the answer).

`infer.py` writes `_1_input`, `_2_score` (blue interior → red edge),
`_3_flagged` (binary at `--threshold`), `_4_label` when the input has one, and
a `_pred.npz` holding `points, rgb, score, direction, coverage, spread`.

## How well it works

Held-out validation, whole trajectories, 16 captures across both corpora:

| | MP3D | HM3D |
|---|---|---|
| MAE | 0.0334 | 0.0315 |
| MAE on edge points | 0.0638 | 0.0629 |
| correlation with label | +0.946 | +0.943 |
| precision @ 0.30 | 0.916 | 0.905 |
| recall @ 0.30 | 0.811 | 0.803 |

The two corpora score the same, so the model is not fitting one
reconstruction pipeline's quirks.

**Use 0.25, not 0.30, as the operating threshold.** The model shaves the peaks
slightly — mean predicted 0.128 against a true 0.133, p99 0.579 against 0.610
— which is ordinary regression shrinkage. Precision barely moves between 0.20
and 0.30 (0.906 → 0.905) while recall falls, so the lower threshold is free
recall. F1 peaks at 0.864.

## Reproducibility

Output is **not** bit-identical between runs: the backbone shuffles its
serialization order each forward, and the sparse-convolution reductions use
atomics. On the bundled example two runs of the same weights differ by at most
0.06 per point, 0.002 on average, with precision and recall moving by under
0.003. Patch-overlap averaging already smooths most of this. 

