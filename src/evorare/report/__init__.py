"""Reporting: env-stamped JSON and a dependency-free hand-written SVG."""

from __future__ import annotations

from .json_report import diagnosis_to_dict, env_stamp, write_json
from .svg_report import diagnosis_to_svg, write_svg

__all__ = [
    "diagnosis_to_dict",
    "env_stamp",
    "write_json",
    "diagnosis_to_svg",
    "write_svg",
]
