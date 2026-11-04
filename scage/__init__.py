"""SCAGE - scene-anomaly guided exploration with Delta-Net.

This folder is the project root and goes on the path, so `scage` imports from
anywhere (tests/, tools/). So does `deltanet/`, the Delta-Net model
(`delta_net.py`, `infer.py`, `ptv3/`), which imports its own files as
top-level modules. The names do not collide - ptv3 only uses relative imports.
"""

import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELTANET = os.path.join(ROOT, "deltanet")
CONFIG = os.path.join(ROOT, "config", "scage_hm3d.yaml")
WEIGHT = os.path.join(ROOT, "model_weights", "delta_net.pth")
MESH = os.path.join(ROOT, "examples", "mv2HUxq3B53.glb")

for _p in (DELTANET, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_config(path=CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)
