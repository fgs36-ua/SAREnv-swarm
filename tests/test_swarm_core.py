"""
Unit tests for the swarm simulator core.

Tests validate:
1. Config creation and defaults
2. Environment coordinate conversions
3. LocalKnowledgeMap observation and evaporation
4. BaseSwarmAgent movement and path generation
5. CommunicationProtocol gossip exchanges
6. SwarmSimulator end-to-end run producing evaluable paths
7. SwarmMetrics compatibility with PathEvaluator
"""
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*shapely.*")

sys.path.insert(0, str(Path(__file__).parent.parent))

from sarenv.swarm.config import SwarmConfig, DroneConfig, RobotDogConfig, AgentConfig
from sarenv.swarm.environment import SwarmEnvironment
from sarenv.swarm.knowledge import LocalKnowledgeMap, MapUpdate
from sarenv.swarm.agents import BaseSwarmAgent, DroneAgent, RobotDogAgent
from sarenv.swarm.communication import CommunicationProtocol
from sarenv.swarm.simulator import SwarmSimulator
from sarenv.swarm.metrics import SwarmMetrics


# ── Fixtures ──────────────────────────────────────────────────────────

def make_fake_dataset_item(
    shape: tuple[int, int] = (50, 50),
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 5000.0, 5000.0),
    seed: int = 42,
):
    """
    Create a lightweight fake SARDatasetItem-like object (no disk I/O).
    """
    rng = np.random.default_rng(seed)
    heatmap = rng.random(shape, dtype=np.float32)
    # Normalise so sum ≈ 1
    heatmap /= heatmap.sum()

    class _FakeDatasetItem:
        def __init__(self):
            self.heatmap = heatmap
            self.bounds = bounds
            self.size = "test"
            self.center_point = (0.0, 0.0)
            self.radius_km = 2.5
            self.features = gpd.GeoDataFrame(geometry=[])
            self.environment_climate = "temperate"
            self.environment_type = "flat"

    return _FakeDatasetItem()


@pytest.fixture
def fake_dataset():
    return make_fake_dataset_item()


@pytest.fixture
def swarm_env(fake_dataset):
    return SwarmEnvironment(fake_dataset)


@pytest.fixture
def default_config():
    return SwarmConfig(
        num_drones=3,
        num_dogs=0,
        budget_per_agent=50_000.0,
        max_steps=200,
        evaporation_rate=0.01,
        max_hops=999,  # effectively global for Phase 1
    )


# ── Config tests ──────────────────────────────────────────────────────

class TestConfig:

    def test_drone_detection_radius(self):
        cfg = DroneConfig(altitude=80.0, fov_deg=45.0)
        expected = 80.0 * np.tan(np.radians(22.5))
        assert abs(cfg.detection_radius - expected) < 1e-6

    def test_robot_dog_detection_radius(self):
        cfg = RobotDogConfig(sensor_range=20.0)
        assert cfg.detection_radius == 20.0

    def test_swarm_total_agents(self):
        cfg = SwarmConfig(num_drones=3, num_dogs=2)
        assert cfg.total_agents == 5


# ── Environment tests ─────────────────────────────────────────────────

class TestEnvironment:

    def test_world_grid_roundtrip(self, swarm_env):
        """world→grid→world should return approximately the original point."""
        env = swarm_env
        x, y = 2500.0, 2500.0  # centre of [0, 5000]
        r, c = env.world_to_grid(x, y)
        x2, y2 = env.grid_to_world(r, c)
        assert abs(x2 - x) < env.grid.dx
        assert abs(y2 - y) < env.grid.dy

    def test_visible_cells_nonempty(self, swarm_env):
        env = swarm_env
        r, c = env.grid.rows // 2, env.grid.cols // 2
        vis = env.get_visible_cells(r, c, detection_radius=200.0)
        assert len(vis) > 0
        # All returned cells should be in bounds
        for vr, vc in vis:
            assert env.in_bounds(vr, vc)

    def test_reachable_neighbors_8connected(self, swarm_env):
        env = swarm_env
        # Interior cell should have exactly 8 neighbours
        r, c = env.grid.rows // 2, env.grid.cols // 2
        nb = env.get_reachable_neighbors(r, c)
        assert len(nb) == 8

    def test_corner_has_fewer_neighbors(self, swarm_env):
        nb = swarm_env.get_reachable_neighbors(0, 0)
        assert len(nb) == 3  # only (0,1), (1,0), (1,1)


# ── Knowledge map tests ──────────────────────────────────────────────

class TestKnowledge:

    def test_initial_exploration_is_zero(self):
        prob = np.ones((10, 10), dtype=np.float32)
        km = LocalKnowledgeMap(prob)
        assert km.exploration_map.sum() == 0.0
        assert km.alert_map.sum() == 0.0

    def test_record_observation_updates_exploration(self):
        prob = np.ones((10, 10), dtype=np.float32)
        km = LocalKnowledgeMap(prob)
        cells = {(2, 3), (2, 4), (3, 3)}
        km.record_observation(cells, "drone_0", timestep=0)
        for r, c in cells:
            assert km.exploration_map[r, c] == 1.0

    def test_record_alert(self):
        prob = np.ones((10, 10), dtype=np.float32)
        km = LocalKnowledgeMap(prob)
        km.record_alert((5, 5), intensity=0.9, agent_id="drone_0", timestep=1)
        assert km.alert_map[5, 5] == 0.9

    def test_evaporate(self):
        prob = np.ones((10, 10), dtype=np.float32)
        km = LocalKnowledgeMap(prob)
        km.exploration_map.fill(1.0)
        km.evaporate(exploration_rate=0.1, alert_rate=0.05)
        assert np.allclose(km.exploration_map, 0.9)

    def test_merge_updates(self):
        prob = np.ones((10, 10), dtype=np.float32)
        km_a = LocalKnowledgeMap(prob)
        km_b = LocalKnowledgeMap(prob)

        # Agent A observes
        km_a.record_observation({(3, 3)}, "a", timestep=5)
        updates = km_a.get_updates_since(0)
        assert len(updates) == 1

        # Agent B merges
        km_b.merge_updates(updates, max_hops=5)
        assert km_b.exploration_map[3, 3] == 1.0

    def test_max_hops_filters_old_updates(self):
        prob = np.ones((10, 10), dtype=np.float32)
        km = LocalKnowledgeMap(prob)
        # Simulate an update that has already travelled 5 hops
        update = MapUpdate(
            cell=(1, 1), layer="exploration", value=1.0,
            timestamp=10, origin_agent="far_away", hops=5,
        )
        km.merge_updates([update], max_hops=5)
        # Should be rejected (hops >= max_hops)
        assert km.exploration_map[1, 1] == 0.0


# ── Agent tests ───────────────────────────────────────────────────────

class TestAgent:

    def test_agent_starts_active(self, swarm_env):
        cfg = DroneConfig(budget=10_000.0)
        km = LocalKnowledgeMap(swarm_env.probability_map)
        a = DroneAgent("drone_0", cfg, swarm_env, km, (25, 25))
        assert a.active

    def test_agent_moves_and_records_path(self, swarm_env):
        cfg = DroneConfig(budget=100_000.0)
        km = LocalKnowledgeMap(swarm_env.probability_map)
        a = DroneAgent("drone_0", cfg, swarm_env, km, (25, 25))

        # Make 10 moves
        for _ in range(10):
            target = a.decide()
            a.execute_move(target)

        assert len(a.path) == 11  # start + 10 moves
        assert a.budget_remaining < 100_000.0

    def test_agent_deactivates_on_budget_exhaust(self, swarm_env):
        cfg = DroneConfig(budget=50.0)  # tiny budget
        km = LocalKnowledgeMap(swarm_env.probability_map)
        a = DroneAgent("drone_0", cfg, swarm_env, km, (25, 25))

        for _ in range(1000):
            if not a.active:
                break
            target = a.decide()
            a.execute_move(target)

        assert not a.active

    def test_path_linestring(self, swarm_env):
        cfg = DroneConfig(budget=50_000.0)
        km = LocalKnowledgeMap(swarm_env.probability_map)
        a = DroneAgent("drone_0", cfg, swarm_env, km, (25, 25))
        for _ in range(5):
            target = a.decide()
            a.execute_move(target)
        ls = a.get_path_linestring()
        assert isinstance(ls, LineString)
        assert ls.length > 0


# ── Communication tests ──────────────────────────────────────────────

class TestCommunication:

    def test_exchange_propagates_info(self, swarm_env):
        cfg = DroneConfig(budget=50_000.0, comm_range=9999.0)
        km_a = LocalKnowledgeMap(swarm_env.probability_map)
        km_b = LocalKnowledgeMap(swarm_env.probability_map)

        agent_a = DroneAgent("a", cfg, swarm_env, km_a, (10, 10))
        agent_b = DroneAgent("b", cfg, swarm_env, km_b, (20, 20))

        # A observes some cells
        km_a.record_observation({(5, 5), (5, 6)}, "a", timestep=1)

        proto = CommunicationProtocol(max_hops=999, bandwidth_limit=999)
        proto.exchange(agent_a, agent_b, current_timestep=2)

        # B should now know about those cells
        assert km_b.exploration_map[5, 5] == 1.0
        assert km_b.exploration_map[5, 6] == 1.0


# ── Simulator tests ──────────────────────────────────────────────────

class TestSimulator:

    def test_simulator_runs(self, swarm_env, default_config):
        sim = SwarmSimulator(swarm_env, default_config, seed=42)
        history = sim.run(max_steps=50)
        assert len(history) > 0
        assert sim.timestep > 0

    def test_paths_are_linestrings(self, swarm_env, default_config):
        sim = SwarmSimulator(swarm_env, default_config, seed=42)
        sim.run(max_steps=50)
        paths = sim.get_paths()
        assert len(paths) == default_config.num_drones
        for p in paths:
            assert isinstance(p, LineString)

    def test_agents_explore_different_areas(self, swarm_env, default_config):
        """With repulsion, agents should not all follow the same path."""
        sim = SwarmSimulator(swarm_env, default_config, seed=42)
        sim.run(max_steps=100)
        # Check that paths are not identical
        paths = sim.get_paths()
        coords = [list(p.coords) for p in paths if not p.is_empty]
        if len(coords) >= 2:
            assert coords[0] != coords[1], "Agents should explore different areas"

    def test_from_dataset_item(self, fake_dataset):
        cfg = SwarmConfig(num_drones=2, num_dogs=0, budget_per_agent=20_000, max_steps=30)
        sim = SwarmSimulator.from_dataset_item(fake_dataset, cfg, seed=7)
        sim.run()
        assert sim.timestep > 0

    def test_all_agents_eventually_stop(self, swarm_env):
        """With a small budget, all agents should run out and deactivate."""
        cfg = SwarmConfig(
            num_drones=2, num_dogs=0,
            budget_per_agent=500.0,  # very small
            max_steps=5000,
        )
        sim = SwarmSimulator(swarm_env, cfg, seed=1)
        sim.run()
        assert all(not a.active for a in sim.agents)


# ── Metrics tests ─────────────────────────────────────────────────────

class TestSwarmMetrics:

    def test_coverage_summary(self, swarm_env, default_config):
        sim = SwarmSimulator(swarm_env, default_config, seed=42)
        sim.run(max_steps=100)
        sm = SwarmMetrics(sim)
        summary = sm.coverage_summary()
        assert summary["explored_cells"] > 0
        assert 0 <= summary["coverage_ratio"] <= 1
        assert summary["total_timesteps"] > 0

    def test_evaluate_with_path_evaluator(self, swarm_env, default_config):
        sim = SwarmSimulator(swarm_env, default_config, seed=42)
        sim.run(max_steps=50)
        sm = SwarmMetrics(sim)
        result = sm.evaluate_with_path_evaluator()
        assert "total_likelihood_score" in result
        assert "area_covered" in result
        assert result["total_likelihood_score"] >= 0
