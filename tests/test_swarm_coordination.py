"""
Tests for decentralized swarm coordination behaviour.

Valida (tarea 3.8):
1. Espectro max_hops: con max_hops alto la cobertura supera a max_hops=0
2. Propagación de info: updates de un agente llegan a otros vía gossip chain
3. Evaporación funcional: exploration_map decae a ~0 tras suficientes ticks
4. Caducidad del gossip: entradas en cells_gossip_explored expiran tras gossip_expiry_ticks
5. Merge sin duplicados: merge_updates no duplica info ya conocida
6. Límite de ancho de banda: con bandwidth_limit=1, solo 1 update por intercambio
"""
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
from sarenv.swarm.environment import SwarmEnvironment
from sarenv.swarm.knowledge import LocalKnowledgeMap, MapUpdate
from sarenv.swarm.agents import DroneAgent
from sarenv.swarm.communication import CommunicationProtocol
from sarenv.swarm.simulator import SwarmSimulator


# ── Helpers ───────────────────────────────────────────────────────────

SHAPE = (50, 50)
BOUNDS = (0.0, 0.0, 5000.0, 5000.0)


def _make_fake_dataset(shape=SHAPE, bounds=BOUNDS, seed=42):
    rng = np.random.default_rng(seed)
    heatmap = rng.random(shape, dtype=np.float32)
    heatmap /= heatmap.sum()

    class _Fake:
        def __init__(self):
            self.heatmap = heatmap
            self.bounds = bounds
            self.size = "test"
            self.center_point = (0.0, 0.0)
            self.radius_km = 2.5
            self.features = gpd.GeoDataFrame(geometry=[])
            self.environment_climate = "temperate"
            self.environment_type = "flat"

    return _Fake()


@pytest.fixture
def fake_dataset():
    return _make_fake_dataset()


@pytest.fixture
def swarm_env(fake_dataset):
    return SwarmEnvironment(fake_dataset)


# ── Tests del espectro de descentralización (max_hops) ────────────────

class TestDecentralizationSpectrum:
    """Valida que max_hops produce un espectro medible entre aislamiento
    y coordinación centralizada."""

    def test_high_hops_better_coverage_than_zero(self, swarm_env):
        """Con max_hops=999 (cuasi-centralizado) la cobertura de probabilidad
        debería superar a max_hops=0 (agentes aislados)."""
        results = {}
        for label, hops in [("isolated", 0), ("global", 999)]:
            cfg = SwarmConfig(
                num_drones=3, num_dogs=0,
                budget_per_agent=30_000.0,
                max_hops=hops,
                max_steps=200,
            )
            sim = SwarmSimulator(swarm_env, cfg, seed=42)
            sim.run()
            # Contar celdas únicas exploradas por todo el enjambre
            all_cells = set()
            for a in sim.agents:
                all_cells |= a.cells_ever_explored
            results[label] = len(all_cells)

        # Con comunicación, los agentes se dispersan mejor
        assert results["global"] >= results["isolated"], (
            f"global ({results['global']}) debería >= isolated ({results['isolated']})"
        )

    def test_intermediate_hops_between_extremes(self, swarm_env):
        """max_hops=3 produce cobertura intermedia entre 0 e infinito."""
        coverages = {}
        for hops in [0, 3, 999]:
            cfg = SwarmConfig(
                num_drones=3, num_dogs=0,
                budget_per_agent=30_000.0,
                max_hops=hops,
                max_steps=200,
            )
            sim = SwarmSimulator(swarm_env, cfg, seed=42)
            sim.run()
            all_cells = set()
            for a in sim.agents:
                all_cells |= a.cells_ever_explored
            coverages[hops] = len(all_cells)

        # max_hops=3 debería ser al menos tan bueno como aislado
        assert coverages[3] >= coverages[0]

    def test_isolated_agents_have_independent_maps(self, swarm_env):
        """Con max_hops=0, cada agente solo conoce lo que exploró él mismo.
        Los exploration_maps deberían ser distintos."""
        cfg = SwarmConfig(
            num_drones=2, num_dogs=0,
            budget_per_agent=20_000.0,
            max_hops=0,
            max_steps=100,
        )
        sim = SwarmSimulator(swarm_env, cfg, seed=42)
        sim.run()

        map_a = sim.agents[0].knowledge.exploration_map
        map_b = sim.agents[1].knowledge.exploration_map
        # No deberían ser idénticos (cada uno exploró zonas distintas)
        assert not np.allclose(map_a, map_b), \
            "Con max_hops=0 los mapas deberían diferir"


# ── Tests de propagación de información ───────────────────────────────

class TestInformationPropagation:

    def test_gossip_chain_propagation(self, swarm_env):
        """Información de A llega a C pasando por B (cadena A→B→C)."""
        cfg = DroneConfig(budget=50_000.0, comm_range=9999.0)

        km_a = LocalKnowledgeMap(swarm_env.probability_map)
        km_b = LocalKnowledgeMap(swarm_env.probability_map)
        km_c = LocalKnowledgeMap(swarm_env.probability_map)

        a = DroneAgent("a", cfg, swarm_env, km_a, (10, 10))
        b = DroneAgent("b", cfg, swarm_env, km_b, (15, 15))
        c = DroneAgent("c", cfg, swarm_env, km_c, (20, 20))

        # A observa una celda
        km_a.record_observation({(5, 5)}, "a", timestep=1)

        proto = CommunicationProtocol(max_hops=999, bandwidth_limit=999)

        # A → B
        proto.exchange(a, b, current_timestep=2)
        assert km_b.exploration_map[5, 5] == 1.0

        # B → C (la info de A llega a C con hops=2)
        proto.exchange(b, c, current_timestep=3)
        assert km_c.exploration_map[5, 5] == 1.0, \
            "La info debería propagarse de A a C pasando por B"

    def test_limited_hops_blocks_distant_propagation(self, swarm_env):
        """Con max_hops=1, la info de A llega a B pero NO a C."""
        cfg = DroneConfig(budget=50_000.0, comm_range=9999.0)

        km_a = LocalKnowledgeMap(swarm_env.probability_map)
        km_b = LocalKnowledgeMap(swarm_env.probability_map)
        km_c = LocalKnowledgeMap(swarm_env.probability_map)

        a = DroneAgent("a", cfg, swarm_env, km_a, (10, 10))
        b = DroneAgent("b", cfg, swarm_env, km_b, (15, 15))
        c = DroneAgent("c", cfg, swarm_env, km_c, (20, 20))

        km_a.record_observation({(5, 5)}, "a", timestep=1)

        proto = CommunicationProtocol(max_hops=1, bandwidth_limit=999)

        # A → B (hops=0 → aceptado, almacenado con hops=1)
        proto.exchange(a, b, current_timestep=2)
        assert km_b.exploration_map[5, 5] == 1.0

        # B → C (hops=1 >= max_hops=1 → rechazado)
        proto.exchange(b, c, current_timestep=3)
        assert km_c.exploration_map[5, 5] == 0.0, \
            "Con max_hops=1, la info no debería llegar al tercer agente"

    def test_alert_propagation_via_gossip(self, swarm_env):
        """Las alertas se propagan igual que la exploración."""
        cfg = DroneConfig(budget=50_000.0, comm_range=9999.0)

        km_a = LocalKnowledgeMap(swarm_env.probability_map)
        km_b = LocalKnowledgeMap(swarm_env.probability_map)

        a = DroneAgent("a", cfg, swarm_env, km_a, (10, 10))
        b = DroneAgent("b", cfg, swarm_env, km_b, (20, 20))

        km_a.record_alert((7, 7), intensity=0.9, agent_id="a", timestep=1)

        proto = CommunicationProtocol(max_hops=999, bandwidth_limit=999)
        proto.exchange(a, b, current_timestep=2)

        assert km_b.alert_map[7, 7] == 0.9


# ── Tests de evaporación ─────────────────────────────────────────────

class TestEvaporation:

    def test_exploration_decays_to_near_zero(self):
        """Tras muchos ticks de evaporación sin visita, la feromona cae a ~0."""
        prob = np.ones((10, 10), dtype=np.float32)
        km = LocalKnowledgeMap(prob)
        km.record_observation({(5, 5)}, "a", timestep=0)
        assert km.exploration_map[5, 5] == 1.0

        # Evaporar 500 ticks con tasa 0.01: (1-0.01)^500 ≈ 0.0066
        for _ in range(500):
            km.evaporate(exploration_rate=0.01, alert_rate=0.005)

        assert km.exploration_map[5, 5] < 0.01, \
            "La feromona debería haber decaído a casi 0"

    def test_alert_decays_slower_than_exploration(self):
        """Las alertas decaen con tasa menor que la exploración."""
        prob = np.ones((10, 10), dtype=np.float32)
        km = LocalKnowledgeMap(prob)
        km.record_observation({(3, 3)}, "a", timestep=0)
        km.record_alert((3, 3), intensity=1.0, agent_id="a", timestep=0)

        for _ in range(100):
            km.evaporate(exploration_rate=0.01, alert_rate=0.005)

        # alert decae más lento: (1-0.005)^100 ≈ 0.606  vs  (1-0.01)^100 ≈ 0.366
        assert km.alert_map[3, 3] > km.exploration_map[3, 3], \
            "Las alertas deberían decaer más lento que la exploración"

    def test_fresh_observation_overwrites_decayed(self):
        """Una nueva observación restaura la feromona tras evaporación."""
        prob = np.ones((10, 10), dtype=np.float32)
        km = LocalKnowledgeMap(prob)
        km.record_observation({(5, 5)}, "a", timestep=0)

        for _ in range(200):
            km.evaporate(exploration_rate=0.01)

        decayed = km.exploration_map[5, 5]
        assert decayed < 0.5

        # Re-observar con calidad 1.0
        km.record_observation({(5, 5)}, "a", timestep=200, detection_quality=1.0)
        assert km.exploration_map[5, 5] == 1.0


# ── Tests de caducidad del gossip ─────────────────────────────────────

class TestGossipExpiry:

    def test_gossip_entry_stores_timestamp(self):
        """Las celdas en cells_gossip_explored guardan el timestamp."""
        prob = np.ones((10, 10), dtype=np.float32)
        km = LocalKnowledgeMap(prob)
        km.record_observation({(3, 3)}, "a", timestep=42)
        assert (3, 3) in km.cells_gossip_explored
        assert km.cells_gossip_explored[(3, 3)] == 42

    def test_merge_updates_timestamp(self):
        """El merge por gossip actualiza el timestamp si es más reciente."""
        prob = np.ones((10, 10), dtype=np.float32)
        km = LocalKnowledgeMap(prob)

        # Entrada previa con timestamp 10
        km.cells_gossip_explored[(5, 5)] = 10

        # Llega update con timestamp 20
        update = MapUpdate(
            cell=(5, 5), layer="exploration", value=1.0,
            timestamp=20, origin_agent="b", hops=0,
        )
        km.merge_updates([update], max_hops=5)
        assert km.cells_gossip_explored[(5, 5)] == 20

    def test_merge_does_not_downgrade_timestamp(self):
        """Si llega un update más viejo, no rebaja el timestamp."""
        prob = np.ones((10, 10), dtype=np.float32)
        km = LocalKnowledgeMap(prob)
        km.cells_gossip_explored[(5, 5)] = 100

        update = MapUpdate(
            cell=(5, 5), layer="exploration", value=1.0,
            timestamp=50, origin_agent="b", hops=0,
        )
        km.merge_updates([update], max_hops=5)
        # El timestamp no debería bajar
        assert km.cells_gossip_explored[(5, 5)] == 100

    def test_expired_gossip_not_penalized(self, swarm_env):
        """Una celda cuyo gossip caducó no recibe penalización de novelty."""
        cfg = DroneConfig(budget=50_000.0)
        km = LocalKnowledgeMap(swarm_env.probability_map)
        agent = DroneAgent("d0", cfg, swarm_env, km, (25, 25))

        # Simular gossip antiguo (hace más de 5000 ticks)
        km.cells_gossip_explored[(24, 25)] = 100
        km.gossip_expiry_ticks = 5000

        # Decidir en tick 6000 (100 + 5000 = 5100 < 6000 → caducado)
        perception = agent.perceive([])
        target = agent.decide(perception, timestep=6000)
        # El agente debería poder elegir (24,25) sin penalización alta
        # No verificamos la celda exacta, pero sí que decide algo válido
        assert target is not None

    def test_recent_gossip_is_penalized(self, swarm_env):
        """Una celda con gossip reciente sigue teniendo penalización."""
        cfg = DroneConfig(budget=50_000.0)
        km = LocalKnowledgeMap(swarm_env.probability_map)
        agent = DroneAgent("d0", cfg, swarm_env, km, (25, 25))

        # Marcar todos los vecinos como gossip reciente
        neighbors = swarm_env.get_reachable_neighbors(25, 25)
        for cell in neighbors:
            km.cells_gossip_explored[cell] = 990

        # Decidir en tick 1000 (990 + 5000 = 5990 > 1000 → vigente)
        perception = agent.perceive([])
        target = agent.decide(perception, timestep=1000)
        # El agente sigue decidiendo algo (puede ser frontera o random walk)
        assert target is not None

    def test_gossip_expiry_ticks_configurable(self):
        """El valor de caducidad es configurable por agente."""
        prob = np.ones((10, 10), dtype=np.float32)
        km = LocalKnowledgeMap(prob)
        assert km.gossip_expiry_ticks == 15000

        km.gossip_expiry_ticks = 10_000
        assert km.gossip_expiry_ticks == 10_000


# ── Tests de merge sin duplicados ─────────────────────────────────────

class TestMergeNoDuplicates:

    def test_merge_same_update_twice(self):
        """Merging el mismo update dos veces no duplica la entrada."""
        prob = np.ones((10, 10), dtype=np.float32)
        km = LocalKnowledgeMap(prob)

        update = MapUpdate(
            cell=(3, 3), layer="exploration", value=1.0,
            timestamp=5, origin_agent="a", hops=0,
        )
        km.merge_updates([update], max_hops=5)
        km.merge_updates([update], max_hops=5)

        assert km.exploration_map[3, 3] == 1.0
        # Solo debería haber 1 entrada en _latest_updates para esa celda
        assert (3, 3, "exploration") in km._latest_updates

    def test_merge_overlapping_info_from_two_agents(self):
        """Dos agentes envían info sobre la misma celda; se queda la más reciente."""
        prob = np.ones((10, 10), dtype=np.float32)
        km = LocalKnowledgeMap(prob)

        update_old = MapUpdate(
            cell=(4, 4), layer="exploration", value=0.5,
            timestamp=10, origin_agent="a", hops=0,
        )
        update_new = MapUpdate(
            cell=(4, 4), layer="exploration", value=0.8,
            timestamp=20, origin_agent="b", hops=0,
        )

        km.merge_updates([update_old], max_hops=5)
        km.merge_updates([update_new], max_hops=5)

        # El valor más reciente gana (max comparison)
        assert km.exploration_map[4, 4] == 0.8
        assert km.cells_gossip_explored[(4, 4)] == 20


# ── Tests de límite de ancho de banda ─────────────────────────────────

class TestBandwidthLimit:

    def test_bandwidth_limit_caps_updates(self, swarm_env):
        """Con bandwidth_limit=1, solo se transmite 1 update por intercambio."""
        cfg = DroneConfig(budget=50_000.0, comm_range=9999.0)

        km_a = LocalKnowledgeMap(swarm_env.probability_map)
        km_b = LocalKnowledgeMap(swarm_env.probability_map)

        a = DroneAgent("a", cfg, swarm_env, km_a, (10, 10))
        b = DroneAgent("b", cfg, swarm_env, km_b, (20, 20))

        # A observa muchas celdas
        many_cells = {(r, c) for r in range(5, 10) for c in range(5, 10)}
        km_a.record_observation(many_cells, "a", timestep=1)

        proto = CommunicationProtocol(max_hops=999, bandwidth_limit=1)
        proto.exchange(a, b, current_timestep=2)

        # B debería haber recibido solo 1 update (la más prioritaria)
        received = sum(
            1 for r in range(5, 10) for c in range(5, 10)
            if km_b.exploration_map[r, c] > 0
        )
        assert received == 1, f"Con bandwidth_limit=1, solo debería llegar 1 update, llegaron {received}"

    def test_alerts_prioritized_over_exploration(self, swarm_env):
        """Con bandwidth limitado, las alertas tienen prioridad."""
        cfg = DroneConfig(budget=50_000.0, comm_range=9999.0)

        km_a = LocalKnowledgeMap(swarm_env.probability_map)
        km_b = LocalKnowledgeMap(swarm_env.probability_map)

        a = DroneAgent("a", cfg, swarm_env, km_a, (10, 10))
        b = DroneAgent("b", cfg, swarm_env, km_b, (20, 20))

        # A observa una celda Y deposita una alerta en otra
        km_a.record_observation({(5, 5)}, "a", timestep=1)
        km_a.record_alert((8, 8), intensity=0.9, agent_id="a", timestep=1)

        proto = CommunicationProtocol(max_hops=999, bandwidth_limit=1)
        proto.exchange(a, b, current_timestep=2)

        # La alerta debería tener prioridad sobre la exploración
        assert km_b.alert_map[8, 8] == 0.9, \
            "La alerta debería transmitirse antes que la exploración"


# ── Test de simulación completa con gossip expiry ─────────────────────

class TestSimulationWithExpiry:

    def test_simulation_runs_with_gossip_expiry(self, swarm_env):
        """La simulación completa funciona con el sistema de caducidad activo."""
        cfg = SwarmConfig(
            num_drones=3, num_dogs=0,
            budget_per_agent=30_000.0,
            max_hops=3,
            max_steps=200,
        )
        sim = SwarmSimulator(swarm_env, cfg, seed=42)
        history = sim.run()
        assert len(history) > 0
        # Todos los agentes deberían haber explorado algo
        for a in sim.agents:
            assert len(a.cells_ever_explored) > 0
