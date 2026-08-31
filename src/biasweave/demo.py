"""Transparent analytic demonstration evaluator.

The equations are smooth, deterministic teaching surrogates. They are not a
transistor model and must not be used to predict fabricated silicon.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from biasweave.model import Scalar


def _number(point: Mapping[str, Scalar], name: str) -> float:
    value = point[name]
    if isinstance(value, str):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def evaluate(point: Mapping[str, Scalar]) -> dict[str, float]:
    """Evaluate a synthetic two-stage OTA sizing point in SI units."""
    input_width = _number(point, "input_width_m")
    input_length = _number(point, "input_length_m")
    load_width = _number(point, "load_width_m")
    load_length = _number(point, "load_length_m")
    bias_current = _number(point, "bias_current_a")
    compensation = _number(point, "compensation_cap_f")
    fingers = _number(point, "fingers")
    mirror_width = _number(point, "mirror_width_m")
    flavor = point["device_flavor"]
    if flavor not in ("standard", "low_leakage"):
        raise ValueError("device_flavor must be standard or low_leakage")
    speed_factor = 1.0 if flavor == "standard" else 0.88
    power_factor = 1.0 if flavor == "standard" else 0.91

    input_ratio = input_width / input_length
    load_ratio = load_width / load_length
    gm1 = speed_factor * 7.5e-4 * math.sqrt(max(input_ratio * bias_current / 1e-4, 1e-24))
    gm2 = speed_factor * 5.2e-4 * math.sqrt(max(load_ratio * bias_current / 8e-5, 1e-24))
    ro1 = 2.4e5 * (input_length / 1e-6) / (1.0 + bias_current / 1.2e-4)
    ro2 = 2.0e5 * (load_length / 1e-6) / (1.0 + bias_current / 1.5e-4)
    gain = max(gm1 * ro1 * gm2 * ro2, 1.0)
    gain_db = 20.0 * math.log10(gain)
    ugbw_hz = gm1 / (2.0 * math.pi * compensation)
    second_pole = 2.8 * gm2 / (2.0 * math.pi * (compensation + 0.18e-12 * fingers))
    phase_margin_deg = 90.0 - math.degrees(math.atan(ug_bw_ratio(ugbw_hz, second_pole)))
    slew_v_per_s = bias_current / compensation
    power_w = power_factor * 1.8 * bias_current * (2.25 + 0.025 * fingers)
    area_m2 = (
        input_width * input_length * fingers
        + load_width * load_length * fingers
        + mirror_width * input_length
        + compensation / 1.4e-3
    )
    return {
        "gain_db": gain_db,
        "ugbw_hz": ugbw_hz,
        "phase_margin_deg": phase_margin_deg,
        "slew_v_per_s": slew_v_per_s,
        "power_w": power_w,
        "area_m2": area_m2,
    }


def ug_bw_ratio(unity_gain_hz: float, second_pole_hz: float) -> float:
    """Keep the phase-margin relationship explicit and independently testable."""
    return unity_gain_hz / max(second_pole_hz, 1e-30)
