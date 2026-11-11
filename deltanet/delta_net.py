"""Delta-Net, inference only: per-point denoising displacements.

The model is trained with a Chamfer-denoising objective. Displacement
magnitudes provide anomaly responses during exploration.

Identical architecture and weights to the trained model (6,983,171 params);
the training-only branches and the other two task heads have been stripped so
there is nothing here that does not run at inference time.

The one deviation from stock PTv3 is the information bottleneck: the decoder
keeps the hierarchical upsampling (it needs the encoder's point layout to know
where to scatter features) but drops the *feature* skip connections, so
everything the decoder knows about the geometry travels through the
512-channel latent. `NoSkipUnpooling` is what removes them, and a checkpoint
trained with it will not load correctly into stock PTv3.
"""

import torch.nn as nn

from ptv3.modules import PointModule, PointSequential
from ptv3.point_transformer_v3m1_base import PointTransformerV3

# `ptv3/` is Pointcept's PTv3 backbone, vendored here so this folder needs no
# checkout of that repo - see ptv3/__init__.py. It is the *base* variant on
# purpose: Pointcept's package __init__ re-exports a later variant under the
# same name, and that one asserts head_dim % 3 == 0, which these channel
# widths do not satisfy.

try:                      # only needed for the fast attention kernel
    import flash_attn  # noqa: F401
    _FLASH = True
except ImportError:
    _FLASH = False


BACKBONE = dict(
    in_channels=3,
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 1, 2, 0),
    enc_channels=(32, 32, 64, 192, 512),
    enc_num_head=(2, 2, 4, 12, 32),
    enc_patch_size=(48, 48, 128, 512, 512),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(32, 32, 64, 192),
    dec_num_head=(2, 2, 4, 12),
    dec_patch_size=(48, 48, 128, 512),
    mlp_ratio=4,
    qkv_bias=True,
    drop_path=0.0,              # 0.1 in training, irrelevant under .eval()
    shuffle_orders=True,
    pre_norm=True,
    enable_rpe=False,
    # Flash attention if it is installed, plain softmax attention otherwise.
    # The two share the same qkv and proj weights and compute the same thing,
    # so the checkpoint loads either way. Measured on the bundled example the
    # per-point scores differ by 5e-3 on average and precision/recall move by
    # less than 0.005, which is inside this model's own run-to-run spread. The
    # upcast flags must be False under flash, which forbids them.
    enable_flash=_FLASH,
    upcast_attention=not _FLASH,
    upcast_softmax=not _FLASH,
)


class NoSkipUnpooling(PointModule):
    """SerializedUnpooling with the encoder->decoder feature skip removed."""

    def __init__(self, proj: PointSequential):
        super().__init__()
        self.proj = proj

    @classmethod
    def from_serialized(cls, module):
        return cls(module.proj)

    def forward(self, point):
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent.feat = point.feat[inverse]
        if "sparse_conv_feat" in parent.keys():
            parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)
        return parent


class PTv3NoSkip(PointTransformerV3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.enc_mode:
            for _, dec in self.dec.named_children():
                dec.up = NoSkipUnpooling.from_serialized(dec.up)


class DeltaNet(nn.Module):
    """Per-point denoising displacement predictor.

    Output `delta` is a 3D correction vector for each input point. Its
    magnitude provides a raw anomaly response during exploration; its
    normalized vector, when nonzero, is the predicted correction direction.

    The head is applied to decoder features and its output is multiplied by
    `delta_scale` (default 1.0).
    """

    def __init__(self, backbone=None, head_hidden=64, delta_scale=1.0):
        super().__init__()
        self.backbone = PTv3NoSkip(**(backbone or BACKBONE))
        self.head = nn.Sequential(
            nn.Linear((backbone or BACKBONE)["dec_channels"][0], head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 3),
        )
        self.delta_scale = delta_scale

    def forward(self, input_dict):
        point = self.backbone(input_dict)
        delta = self.head(point.feat) * self.delta_scale
        return dict(delta=delta, offset=point.offset)
