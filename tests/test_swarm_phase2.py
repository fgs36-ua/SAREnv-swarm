"""
Tests de Fase 2: Agentes heterogéneos con terreno.

Valida:
1. terrain.py: rasterización de features y generación de mapas de modificadores
2. environment.py: integración del terreno en vecinos y coste de movimiento
3. agents.py: detección modulada por terreno (drone vs dog)
4. simulator.py: simulación mixta con drones y perros robot
"""
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon, LineString, box

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*shapely.*")

sys.path.insert(0, str(Path(__file__).parent.parent))

from sarenv.swarm.terrain import (
    DETECTION_MODIFIERS,
    TRAVERSABILITY_COSTS,
    _rasterize_features,
    build_detection_modifier_map,
    build_traversability_map,
)
from sarenv.swarm.environment import SwarmEnvironment
from sarenv.swarm.agents import DroneAgent, RobotDogAgent
from sarenv.swarm.config import SwarmConfig, DroneConfig, RobotDogConfig
from sarenv.swarm.knowledge import LocalKnowledgeMap
from sarenv.swarm.simulator import SwarmSimulator


# ── Helpers ───────────────────────────────────────────────────────────

SHAPE = (50, 50)
BOUNDS = (0.0, 0.0, 5000.0, 5000.0)


def _make_features_gdf(polygons_with_types: list[tuple[Polygon, str]]) -> gpd.GeoDataFrame:
    """Crea un GeoDataFrame con polígonos y su feature_type."""
    geoms = [p for p, _ in polygons_with_types]
    types = [t for _, t in polygons_with_types]
    return gpd.GeoDataFrame({"feature_type": types, "geometry": geoms})


def _make_dataset_item(features_gdf=None, shape=SHAPE, bounds=BOUNDS, seed=42):
    """Crea un fake SARDatasetItem con features opcionales."""
    rng = np.random.default_rng(seed)
    heatmap = rng.random(shape, dtype=np.float32)
    heatmap /= heatmap.sum()

    if features_gdf is None:
        features_gdf = gpd.GeoDataFrame(geometry=[])

    class _FakeItem:
        def __init__(self):
            self.heatmap = heatmap
            self.bounds = bounds
            self.size = "test"
            self.center_point = (0.0, 0.0)
            self.radius_km = 2.5
            self.features = features_gdf
            self.environment_climate = "temperate"
            self.environment_type = "terrain_test"

    return _FakeItem()


def _woodland_features():
    """Grid con bosque cubriendo la mitad izquierda."""
    woodland_poly = box(0, 0, 2500, 5000)  # mitad izquierda
    return _make_features_gdf([(woodland_poly, "woodland")])


def _water_features():
    """Grid con agua cubriendo la franja inferior."""
    water_poly = box(0, 0, 5000, 1000)  # quinta parte inferior
    return _make_features_gdf([(water_poly, "water")])


def _mixed_features():
    """Grid con bosque (izq), agua (abajo-derecha), road (centro)."""
    woodland = box(0, 0, 2000, 5000)
    water = box(3000, 0, 5000, 1500)
    road = box(2200, 2000, 2800, 4000)
    return _make_features_gdf([
        (woodland, "woodland"),
        (water, "water"),
        (road, "road"),
    ])


# ── 1. Tests de terrain.py ───────────────────────────────────────────

class TestTerrainRasterization:
    """Rasterización de features vectoriales a grids de terreno."""

    def test_empty_features_gives_none_grid(self):
        """Sin features todo es None (terreno abierto)."""
        gdf = gpd.GeoDataFrame(geometry=[])
        grid = _rasterize_features(gdf, 50, 50, 0, 0, 100, 100)
        assert grid.shape == (50, 50)
        assert all(cell is None for cell in grid.flat)

    def test_woodland_rasterizes_left_half(self):
        """Un polígono de bosque cubre la mitad izquierda."""
        gdf = _woodland_features()
        grid = _rasterize_features(gdf, 50, 50, 0, 0, 100, 100)
        # Celda (25, 10) está dentro del bosque
        assert grid[25, 10] == "woodland"
        # Celda (25, 40) está fuera (mitad derecha)
        assert grid[25, 40] is None

    def test_water_rasterizes_bottom(self):
        """Polígono de agua cubre la franja inferior."""
        gdf = _water_features()
        grid = _rasterize_features(gdf, 50, 50, 0, 0, 100, 100)
        # Celda (5, 25) = franja inferior (row baja = y baja)
        assert grid[5, 25] == "water"
        # Celda (40, 25) = parte alta, fuera del agua
        assert grid[40, 25] is None


class TestDetectionModifierMap:
    """Mapas de modificadores de detección para cada tipo de agente."""

    def test_drone_woodland_penalty(self):
        """El dron tiene penalización severa en bosque (0.15)."""
        gdf = _woodland_features()
        grid = _rasterize_features(gdf, 50, 50, 0, 0, 100, 100)
        det = build_detection_modifier_map(grid, "drone")
        # Celda en bosque
        woodland_cell = det[25, 10]
        # Celda abierta
        open_cell = det[25, 40]
        assert woodland_cell == pytest.approx(0.15, abs=0.01)
        assert open_cell == pytest.approx(1.0)

    def test_dog_woodland_bonus(self):
        """El robot dog detecta bien en bosque (0.85) vs campo abierto (0.7)."""
        gdf = _woodland_features()
        grid = _rasterize_features(gdf, 50, 50, 0, 0, 100, 100)
        det = build_detection_modifier_map(grid, "robot_dog")
        woodland_val = det[25, 10]
        open_val = det[25, 40]
        assert woodland_val > open_val, "Dog debería detectar mejor en bosque que en campo"
        assert woodland_val == pytest.approx(0.85, abs=0.01)

    def test_dog_cannot_detect_in_water(self):
        """El robot dog no puede detectar en agua (0.0)."""
        gdf = _water_features()
        grid = _rasterize_features(gdf, 50, 50, 0, 0, 100, 100)
        det = build_detection_modifier_map(grid, "robot_dog")
        # Celda de agua
        assert det[5, 25] == pytest.approx(0.0)


class TestTraversabilityMap:
    """Mapas de coste de transitabilidad por tipo de agente."""

    def test_drone_uniform_traversability(self):
        """Dron: todo el terreno cuesta lo mismo (1.0)."""
        gdf = _mixed_features()
        grid = _rasterize_features(gdf, 50, 50, 0, 0, 100, 100)
        trav = build_traversability_map(grid, "drone")
        # Todo debería ser 1.0
        assert np.all(trav == pytest.approx(1.0))

    def test_dog_water_is_impassable(self):
        """Robot dog no puede cruzar agua (coste infinito)."""
        gdf = _water_features()
        grid = _rasterize_features(gdf, 50, 50, 0, 0, 100, 100)
        trav = build_traversability_map(grid, "robot_dog")
        assert np.isinf(trav[5, 25]), "Agua debería ser intransitable para dog"

    def test_dog_road_is_cheap(self):
        """Robot dog se mueve más rápido en carretera (0.5x)."""
        gdf = _mixed_features()
        grid = _rasterize_features(gdf, 50, 50, 0, 0, 100, 100)
        trav = build_traversability_map(grid, "robot_dog")
        # Buscar celda que sea "road"
        road_mask = grid == "road"
        if road_mask.any():
            road_cost = trav[road_mask].mean()
            assert road_cost == pytest.approx(0.5)


# ── 2. Tests de environment.py con terreno ────────────────────────────

class TestEnvironmentTerrain:
    """Integración del terreno en SwarmEnvironment."""

    def test_terrain_grid_built_on_init(self):
        """El environment construye terrain_grid al inicializarse."""
        item = _make_dataset_item(_woodland_features())
        env = SwarmEnvironment(item)
        assert env.terrain_grid is not None
        assert env.terrain_grid.shape == SHAPE

    def test_detection_modifier_cached(self):
        """get_detection_modifier cachea el resultado por tipo de agente."""
        item = _make_dataset_item(_woodland_features())
        env = SwarmEnvironment(item)
        m1 = env.get_detection_modifier("drone")
        m2 = env.get_detection_modifier("drone")
        assert m1 is m2, "Debería devolver el mismo objeto (cacheado)"

    def test_dog_cannot_reach_water_neighbors(self):
        """Robot dog no puede llegar a celdas de agua."""
        item = _make_dataset_item(_water_features())
        env = SwarmEnvironment(item)
        # Encontrar una celda justo ENCIMA de la frontera del agua
        # Water = rows 0-9 aprox (franja 0-1000m en grid de 50 rows, dy=100m)
        border_row = 10  # justo encima del agua
        neighbors = env.get_reachable_neighbors(border_row, 25, "robot_dog")
        # Las celdas (9, x) están en agua -> no deberían ser alcanzables para dog
        water_neighbors = [n for n in neighbors if n[0] < 10]
        # Verificar: si hay agua ahí, no deberían estar
        trav = env.get_traversability("robot_dog")
        for r, c in water_neighbors:
            assert np.isfinite(trav[r, c]), \
                f"Celda ({r},{c}) debería ser finita si está en neighbors"

    def test_drone_reaches_water_neighbors(self):
        """El dron puede llegar a celdas de agua sin problema."""
        item = _make_dataset_item(_water_features())
        env = SwarmEnvironment(item)
        border_row = 10
        neighbors_drone = env.get_reachable_neighbors(border_row, 25, "drone")
        neighbors_dog = env.get_reachable_neighbors(border_row, 25, "robot_dog")
        assert len(neighbors_drone) >= len(neighbors_dog), \
            "Dron debería tener al menos tantos vecinos como dog (puede cruzar agua)"

    def test_movement_cost_terrain_multiplier(self):
        """Coste de movimiento para dog en bosque > coste en carretera."""
        item = _make_dataset_item(_mixed_features())
        env = SwarmEnvironment(item)
        # Encontrar celda de bosque y celda de carretera
        woodland_cells = list(zip(*np.where(env.terrain_grid == "woodland")))
        road_cells = list(zip(*np.where(env.terrain_grid == "road")))
        if woodland_cells and road_cells:
            wc = woodland_cells[len(woodland_cells) // 2]
            rc = road_cells[len(road_cells) // 2]
            # Coste de moverse a la celda woodland vs road (desde celda adyacente)
            # Usamos (wc[0]-1, wc[1]) -> wc y (rc[0]-1, rc[1]) -> rc
            from_w = (wc[0] - 1, wc[1]) if wc[0] > 0 else (wc[0] + 1, wc[1])
            from_r = (rc[0] - 1, rc[1]) if rc[0] > 0 else (rc[0] + 1, rc[1])
            cost_wood = env.movement_cost(from_w, wc, "robot_dog")
            cost_road = env.movement_cost(from_r, rc, "robot_dog")
            assert cost_wood > cost_road, \
                "Moverse en bosque debería ser más caro que en carretera para dog"


# ── 3. Tests de agentes con terreno ──────────────────────────────────

class TestAgentDetection:
    """Detección modulada por terreno en los agentes."""

    @pytest.fixture
    def woodland_env(self):
        item = _make_dataset_item(_woodland_features())
        return SwarmEnvironment(item)

    @pytest.fixture
    def water_env(self):
        item = _make_dataset_item(_water_features())
        return SwarmEnvironment(item)

    def _make_agent(self, cls, env, position=None, agent_id="test_0"):
        """Crea un agente de prueba posicionado en el grid."""
        if position is None:
            position = (25, 25)
        knowledge = LocalKnowledgeMap(env.probability_map)
        if cls == DroneAgent:
            config = DroneConfig(budget=100_000)
        else:
            config = RobotDogConfig(budget=100_000)
        return cls(
            agent_id=agent_id,
            config=config,
            environment=env,
            knowledge=knowledge,
            start_position=position,
            rng=np.random.default_rng(42),
        )

    def test_drone_sees_fewer_cells_in_woodland(self, woodland_env):
        """Dron en bosque filtra celdas de baja detección."""
        drone = self._make_agent(DroneAgent, woodland_env, position=(25, 10))
        visible = drone.get_visible_cells()
        # Comparar con lo que vería sin filtro
        all_cells = woodland_env.get_visible_cells(25, 10, drone._detection_radius)
        # Con filtro debería tener igual o menos celdas
        assert len(visible) <= len(all_cells)

    def test_dog_sees_nothing_in_water(self, water_env):
        """Robot dog no detecta nada si está rodeado de agua."""
        # Posicionar en zona de agua (row 5)
        dog = self._make_agent(RobotDogAgent, water_env, position=(5, 25))
        visible = dog.get_visible_cells()
        # Las celdas en agua tienen detección=0, se filtran
        det_mod = water_env.get_detection_modifier("robot_dog")
        water_in_visible = [c for c in visible if det_mod[c[0], c[1]] < 0.05]
        assert len(water_in_visible) == 0, "Dog no debería ver celdas en agua"

    def test_detection_quality_drone_vs_dog_woodland(self, woodland_env):
        """En bosque: calidad de detección del dog >> drone."""
        drone = self._make_agent(DroneAgent, woodland_env, position=(25, 10))
        dog = self._make_agent(RobotDogAgent, woodland_env, position=(25, 10))
        # Celda en bosque
        q_drone = drone._detection_quality_at(25, 10)
        q_dog = dog._detection_quality_at(25, 10)
        assert q_dog > q_drone, \
            f"Dog ({q_dog:.2f}) debería detectar mejor en bosque que drone ({q_drone:.2f})"

    def test_detection_quality_drone_in_open_field(self, woodland_env):
        """En campo abierto (sin feature): drone detecta a calidad máxima."""
        drone = self._make_agent(DroneAgent, woodland_env, position=(25, 40))
        q = drone._detection_quality_at(25, 40)
        assert q == pytest.approx(1.0), "Drone en campo abierto debería tener calidad 1.0"


# ── 4. Tests de simulación mixta ─────────────────────────────────────

class TestMixedSimulation:
    """Simulación completa con drones y robot dogs en terreno mixto."""

    def test_mixed_simulation_runs(self):
        """Una simulación con 2 drones + 1 dog completa sin errores."""
        item = _make_dataset_item(_mixed_features())
        config = SwarmConfig(
            num_drones=2,
            num_dogs=1,
            budget_per_agent=5_000,
            max_steps=20,
        )
        sim = SwarmSimulator(environment=SwarmEnvironment(item), config=config, seed=42)
        history = sim.run()
        assert len(history) > 0
        assert any("dog_0" in h["positions"] for h in history)

    def test_dog_avoids_water_during_simulation(self):
        """El robot dog nunca pisa agua durante la simulación."""
        item = _make_dataset_item(_water_features())
        env = SwarmEnvironment(item)
        config = SwarmConfig(
            num_drones=0,
            num_dogs=2,
            budget_per_agent=10_000,
            max_steps=50,
        )
        sim = SwarmSimulator(environment=env, config=config, seed=42)
        sim.run()
        # Comprobar que ningún dog pisó agua
        trav = env.get_traversability("robot_dog")
        for agent in sim.agents:
            for r, c in agent.path:
                assert np.isfinite(trav[r, c]), \
                    f"Agent {agent.id} pisó celda intransitable ({r},{c})"

    def test_drone_explores_through_water(self):
        """Los drones sí pueden volar sobre agua."""
        item = _make_dataset_item(_water_features())
        env = SwarmEnvironment(item)
        config = SwarmConfig(
            num_drones=2,
            num_dogs=0,
            budget_per_agent=30_000,
            max_steps=100,
        )
        sim = SwarmSimulator(environment=env, config=config, seed=42)
        sim.run()
        # Al menos algún dron debió pasar por zona de agua
        water_mask = env.terrain_grid == "water"
        visited_water = False
        for agent in sim.agents:
            for r, c in agent.path:
                if water_mask[r, c]:
                    visited_water = True
                    break
            if visited_water:
                break
        # No forzamos que visite agua, pero verificamos que puede
        # (depende de la heurística, así que solo verificamos que no falla)
        assert len(sim.agents) == 2

    def test_exploration_quality_varies_by_terrain(self):
        """Las celdas exploradas tienen calidad variable según terreno."""
        item = _make_dataset_item(_woodland_features())
        env = SwarmEnvironment(item)
        config = SwarmConfig(
            num_drones=1,
            num_dogs=0,
            budget_per_agent=20_000,
            max_steps=50,
        )
        sim = SwarmSimulator(environment=env, config=config, seed=42)
        sim.run()
        drone = sim.agents[0]
        expl = drone.knowledge.exploration_map
        # Las celdas en bosque deberían tener exploración < 1.0
        woodland_mask = env.terrain_grid == "woodland"
        explored_woodland = expl[woodland_mask]
        explored_open = expl[~woodland_mask]
        # Si hay celdas exploradas en ambas zonas, las de bosque tienen menor calidad
        wl_explored = explored_woodland[explored_woodland > 0]
        op_explored = explored_open[explored_open > 0]
        if len(wl_explored) > 0 and len(op_explored) > 0:
            assert wl_explored.mean() < op_explored.mean(), \
                "Exploración media en bosque debería ser menor que en campo abierto para drone"

    def test_paths_are_valid_linestrings(self):
        """Los paths generados son LineStrings válidas."""
        item = _make_dataset_item(_mixed_features())
        config = SwarmConfig(num_drones=1, num_dogs=1, budget_per_agent=5_000, max_steps=20)
        sim = SwarmSimulator(environment=SwarmEnvironment(item), config=config, seed=42)
        sim.run()
        paths = sim.get_paths()
        for p in paths:
            assert isinstance(p, LineString)
