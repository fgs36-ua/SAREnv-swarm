"""
Tests para los experimentos E8 (hard-mask de celdas observadas) y E9
(estadística de probabilidad acumulada por agente), introducidos por
[docs/20_plan_accion_tutor.md](../docs/20_plan_accion_tutor.md).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*shapely.*")

sys.path.insert(0, str(Path(__file__).parent.parent))

from sarenv.swarm.config import SwarmConfig, DroneConfig
from sarenv.swarm.metrics import SwarmMetrics, _gini_coefficient
from sarenv.swarm.simulator import SwarmSimulator


SHAPE = (40, 40)
BOUNDS = (0.0, 0.0, 4000.0, 4000.0)


def _make_fake_dataset(seed: int = 7):
    rng = np.random.default_rng(seed)
    heatmap = rng.random(SHAPE, dtype=np.float32)
    heatmap /= heatmap.sum()

    class _Fake:
        def __init__(self):
            self.heatmap = heatmap
            self.bounds = BOUNDS
            self.size = "test"
            self.center_point = (0.0, 0.0)
            self.radius_km = 2.0
            self.features = gpd.GeoDataFrame(geometry=[])
            self.environment_climate = "temperate"
            self.environment_type = "flat"

    return _Fake()


def _run(*, ever_explored_penalty: float, steps: int = 200, seed: int = 0):
    drone_cfg = DroneConfig(altitude=80.0, fov_deg=45.0, budget=200_000.0)
    drone_cfg.ever_explored_penalty = ever_explored_penalty
    config = SwarmConfig(
        num_drones=3, num_dogs=0,
        budget_per_agent=200_000.0,
        max_steps=steps,
        max_hops=1,
        drone_config=drone_cfg,
    )
    sim = SwarmSimulator.from_dataset_item(_make_fake_dataset(), config, seed=seed)
    sim.run(max_steps=steps)
    return sim


# -- Gini helper ------------------------------------------------------


def test_gini_helper_equal_distribution_is_zero():
    assert _gini_coefficient([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=1e-9)


def test_gini_helper_concentrated_distribution_close_to_one():
    g = _gini_coefficient([0.0, 0.0, 0.0, 10.0])
    # Con n=4 valores y todo en uno solo, Gini = 1 - 1/n = 0.75
    assert g == pytest.approx(0.75, abs=1e-6)


def test_gini_helper_handles_empty_and_zero():
    assert _gini_coefficient([]) == 0.0
    assert _gini_coefficient([0.0, 0.0, 0.0]) == 0.0


# -- E9: prob acumulada por agente -----------------------------------


def test_e9_cumulative_probability_is_tracked_per_agent():
    sim = _run(ever_explored_penalty=0.0, steps=120)
    sweeps = {a.id: a.cumulative_probability_swept for a in sim.agents}
    assert len(sweeps) == 3
    # Todos los agentes deben haber barrido masa > 0 (mapa no nulo)
    for value in sweeps.values():
        assert value > 0.0
    # Cota superior: nadie puede haber barrido más prob que la suma del
    # mapa entero (cells_ever_explored ⊆ grid).
    total_map = float(sim.env.probability_map.sum())
    for value in sweeps.values():
        assert value <= total_map + 1e-6


def test_e9_report_exposes_gini_and_total():
    sim = _run(ever_explored_penalty=0.0, steps=120)
    report = SwarmMetrics(sim).full_report()
    assert "agent_probability_gini" in report
    assert "total_probability_swept" in report
    assert "mean_probability_swept" in report
    assert "per_agent_probability_swept" in report
    assert 0.0 <= report["agent_probability_gini"] <= 1.0


# -- E8: hard-mask sobre celdas observadas ---------------------------


def test_e8_default_is_disabled():
    drone_cfg = DroneConfig()
    assert drone_cfg.ever_explored_penalty == 0.0


def test_e8_hard_mask_reduces_revisits():
    """Con penalty=1.0 el solapamiento debe ser <= que sin penalty."""
    sim_off = _run(ever_explored_penalty=0.0, steps=300, seed=1)
    sim_on = _run(ever_explored_penalty=1.0, steps=300, seed=1)

    overlap_off = SwarmMetrics(sim_off).coverage_summary()["overlap_ratio"]
    overlap_on = SwarmMetrics(sim_on).coverage_summary()["overlap_ratio"]

    # Igual o menos solapamiento con hard-mask; tolerancia pequeña por
    # estocasticidad del Lévy/frontier.
    assert overlap_on <= overlap_off + 0.02
