# SPDX-License-Identifier: GPL-3.0-or-later
"""
ComfyUI-SECADNET - Sketch-Extrude CAD Reconstruction from Voxels

Originally from ComfyUI-CADabra: https://github.com/PozzettiAndrea/ComfyUI-CADabra

Paper: "SECAD-Net: Self-Supervised CAD Reconstruction by Learning Sketch-Extrude Operations"
Project: https://github.com/BunnySoCrazy/SECAD-Net
"""

import sys

if 'pytest' not in sys.modules:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
