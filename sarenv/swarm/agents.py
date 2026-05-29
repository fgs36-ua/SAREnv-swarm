# sarenv/swarm/agents.py
"""
Jerarquía de agentes del enjambre.

``BaseSwarmAgent`` implementa una heurística greedy adaptada al paradigma de
feromonas. La regla de decisión es sencilla a propósito, porque el
comportamiento coordinado emerge de la dinámica de feromonas y gossip.

    score = probability * (1 - exploration_pheromone) - repulsion

``RobotDogAgent`` añade detección y movimiento dependientes del terreno.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import AgentConfig, DroneConfig, RobotDogConfig
    from .environment import SwarmEnvironment
    from .knowledge import LocalKnowledgeMap


@dataclass
class Perception:
    """Snapshot de lo que un agente percibe en un tick."""

    visible_cells: set[tuple[int, int]]
    local_probability: dict[tuple[int, int], float]
    local_exploration: dict[tuple[int, int], float]
    local_alert: dict[tuple[int, int], float]
    neighbors: list[BaseSwarmAgent]
    budget_remaining: float
    position: tuple[int, int]
    timestep: int = 0


class BaseSwarmAgent:
    """Agente genérico del enjambre que opera sobre un grid discreto.

    Cadena de prioridades en decide():
      1. Volver a base si queda poco budget
      2. Investigar feromona de alerta
      3. Greedy: prob * novedad - repulsión
      4. Paseo aleatorio como fallback
    """

    agent_type: str = "drone"

    def __init__(
        self,
        agent_id: str,
        config: AgentConfig,
        environment: SwarmEnvironment,
        knowledge: LocalKnowledgeMap,
        start_position: tuple[int, int],
        rng: np.random.Generator | None = None,
    ) -> None:
        self.id = agent_id
        self.config = config
        self.env = environment
        self.knowledge = knowledge

        self.position: tuple[int, int] = start_position
        self.base_position: tuple[int, int] = start_position  # for return-to-base
        self.budget_remaining: float = config.budget
        self.path: list[tuple[int, int]] = [start_position]
        self.active: bool = True  # False cuando se agota budget o vuelve a base
        # Acumulador de celdas observadas a lo largo de TODA la simulación
        # (inmune a evaporación, para métricas fiables)
        self.cells_ever_explored: set[tuple[int, int]] = set()
        # E9 (docs/20): masa de probabilidad nueva barrida por este
        # agente. Se incrementa en ``record_step_observation`` con la
        # suma de ``env.probability_map[r, c]`` sobre las celdas recién
        # observadas (no contadas en ticks anteriores). Sirve para
        # cuantificar el reparto de carga entre agentes (Gini, etc.).
        self.cumulative_probability_swept: float = 0.0

        self._rng = rng or np.random.default_rng()

        # Seguimiento de transitabilidad media experimentada (para estimar
        # coste real de vuelta a base en terrenos difíciles).
        self._trav_sum: float = 0.0
        self._trav_count: int = 0

        # Frontier persistente: evita re-calcular cada tick y oscilar
        self._frontier_target: tuple[int, int] | None = None
        self._frontier_ttl: int = 0  # ticks restantes de compromiso

        # Timestamp de última visita propia a cada celda (para decaimiento
        # temporal de la penalización de novelty)
        self._visit_timestamps: dict[tuple[int, int], int] = {}
        # Constante de decaimiento temporal (en ticks): a los ~200 ticks
        # una celda visitada recupera ~63% de su novelty.
        self._novelty_decay_tau: float = 200.0

        # Anti-estancamiento: si en los últimos _stagnation_window pasos
        # no descubrimos al menos _stagnation_threshold celdas nuevas,
        # forzar búsqueda de frontera.
        self._stagnation_window: int = 50       # ventana de ticks a mirar
        self._stagnation_threshold: int = 5     # mínimo de celdas nuevas
        self._recent_new_cells: list[int] = []  # historico de celdas nuevas/tick

        # Radio de detección cacheado desde config
        self._detection_radius: float = self._compute_detection_radius()

    # -- Hooks que las subclases pueden sobreescribir --

    def _compute_detection_radius(self) -> float:
        """Radio de detección en metros (sobreescrito por subclases)."""
        return getattr(self.config, "detection_radius", 80.0)

    def get_visible_cells(self) -> set[tuple[int, int]]:
        """Cells visible from the current position."""
        return self.env.get_visible_cells(
            self.position[0], self.position[1], self._detection_radius
        )

    def record_step_observation(self, visible: set[tuple[int, int]]) -> None:
        """Actualiza el bookkeeping del agente con las celdas visibles del tick.

        Cuenta celdas nuevas para el detector de estancamiento, las acumula
        en ``cells_ever_explored`` (resistente a evaporación, usado por métricas),
        y suma la masa de probabilidad de las celdas nuevas en
        ``cumulative_probability_swept`` (E9, docs/20).
        Llamado por el simulador después de la fase de observación.
        """
        new_set = visible - self.cells_ever_explored
        self._recent_new_cells.append(len(new_set))
        if new_set:
            pmap = self.env.probability_map
            self.cumulative_probability_swept += float(
                sum(pmap[r, c] for (r, c) in new_set)
            )
        self.cells_ever_explored.update(visible)

    def _get_reachable_neighbors(self) -> list[tuple[int, int]]:
        """Grid cells the agent can move to in one tick."""
        return self.env.get_reachable_neighbors(
            self.position[0], self.position[1], self.agent_type
        )

    def _movement_cost(self, target: tuple[int, int]) -> float:
        """Cost in metres of moving from current position to *target*."""
        return self.env.movement_cost(self.position, target, self.agent_type)

    # ── perception ────────────────────────────────────────────────────

    def perceive(self, nearby_agents: list[BaseSwarmAgent]) -> Perception:
        """Build a local perception snapshot."""
        visible = self.get_visible_cells()
        return Perception(
            visible_cells=visible,
            local_probability={
                c: self.knowledge.probability_map[c[0], c[1]] for c in visible
            },
            local_exploration={
                c: self.knowledge.exploration_map[c[0], c[1]] for c in visible
            },
            local_alert={
                c: self.knowledge.alert_map[c[0], c[1]]
                for c in visible
                if self.knowledge.alert_map[c[0], c[1]] > 0
            },
            neighbors=nearby_agents,
            budget_remaining=self.budget_remaining,
            position=self.position,
        )

    # -- Decisión --

    def decide(self, perception: Perception | None = None, *, timestep: int = 0) -> tuple[int, int] | None:
        """Elige la siguiente celda. Cadena de 4 prioridades (ver docstring de clase)."""
        if not self.active:
            return None

        if perception is None:
            perception = self.perceive([])

        # Prioridad 1 -- conservar budget para volver a base
        dist_to_base = self.grid_distance(self.position, self.base_position)
        # Estimación del coste real de vuelta: distancia × transitabilidad
        # media experimentada.  Para drones (trav ≈ 1.0) es casi igual que
        # antes; para perros en terreno difícil (trav > 1) reserva mucho más.
        avg_trav = self._trav_sum / self._trav_count if self._trav_count > 0 else 1.0
        estimated_return = dist_to_base * max(avg_trav, 1.0) * self.config.return_safety_factor
        if perception.budget_remaining < estimated_return:
            return self._step_toward(self.base_position)

        # Prioridad 1b -- compromiso de frontera activo
        # Si el agente tiene un objetivo de frontera, seguir caminando
        # hacia él hasta que expire el TTL o llegue.
        if self._frontier_ttl > 0 and self._frontier_target is not None:
            self._frontier_ttl -= 1
            # Cancelar si hemos llegado (o estamos a 1 celda)
            if self.grid_distance(self.position, self._frontier_target) < self.env.grid.dx * 1.5:
                self._frontier_target = None
                self._frontier_ttl = 0
            else:
                return self._step_toward(self._frontier_target)

        # Prioridad 1c -- detector de estancamiento
        # Si en los últimos _stagnation_window ticks no hemos descubierto
        # suficientes celdas nuevas, forzar búsqueda de frontera.
        if len(self._recent_new_cells) >= self._stagnation_window:
            recent_sum = sum(self._recent_new_cells[-self._stagnation_window:])
            if recent_sum < self._stagnation_threshold:
                frontier = self._find_nearest_frontier(timestep=timestep)
                if frontier is not None:
                    dist_cells = max(
                        abs(frontier[0] - self.position[0]),
                        abs(frontier[1] - self.position[1]),
                    )
                    self._frontier_target = frontier
                    self._frontier_ttl = dist_cells + 5
                    return self._step_toward(frontier)

        # Prioridad 2 -- investigar la alerta más fuerte cercana
        if perception.local_alert:
            best_alert_cell = max(perception.local_alert, key=perception.local_alert.get)
            if perception.local_alert[best_alert_cell] > self.config.alert_threshold:
                # Cancelar frontera si hay algo más urgente
                self._frontier_target = None
                self._frontier_ttl = 0
                return self._step_toward(best_alert_cell)

        # Prioridad 3 -- exploración greedy con repulsión
        reachable = self._get_reachable_neighbors()
        if not reachable:
            return None

        best_score = -np.inf
        best_cell = None

        # Pre-calcular posiciones de vecinos para no recalcular cada vez
        neighbor_positions = [n.position for n in perception.neighbors]

        # Vector de huida del centroide de peers vistos recientemente
        # (Reynolds 1987 / Boids-separation a escala táctica). Se calcula
        # UNA VEZ por tick (no por celda) y se reutiliza dentro del loop.
        # Si dispersal_weight=0 o no hay peers activos, el término se anula.
        dispersal_weight = self.config.dispersal_weight
        escape_unit = (0.0, 0.0)
        dispersal_weight_eff = 0.0
        if dispersal_weight > 0:
            peer_positions = self.knowledge.get_active_peer_positions(
                timestep, self.config.peer_position_ttl,
            )
            if peer_positions:
                cr = sum(p[0] for p in peer_positions) / len(peer_positions)
                cc = sum(p[1] for p in peer_positions) / len(peer_positions)
                dr = self.position[0] - cr
                dc = self.position[1] - cc
                norm = float(np.hypot(dr, dc))
                if norm > 1e-9:
                    escape_unit = (dr / norm, dc / norm)
                    # Decaimiento por distancia: cerca del centroide empuja
                    # fuerte (≈ dispersal_weight), lejos se desvanece. Esto
                    # convierte la huida brusca en una tendencia suave.
                    falloff = max(self.config.dispersal_falloff, 1e-6)
                    dispersal_weight_eff = dispersal_weight / (1.0 + norm / falloff)

        for cell in reachable:
            prob = self.knowledge.probability_map[cell[0], cell[1]]
            # Estigmergia (Payton 2001 / Parunak 2002): feromona de presencia
            # ATENÚA el prior. Una zona muy pisada deja de "verse" como
            # probable → el mapa percibido tiene un agujero en el pozo
            # saturado, y el gradiente greedy local apunta hacia afuera
            # de forma natural sin necesidad de términos repulsivos extra.
            attn = self.config.pheromone_attenuation
            if attn > 0:
                pres = float(self.knowledge.presence_field[cell[0], cell[1]])
                prob = prob * np.exp(-attn * pres)
            novelty = 1.0 - self.knowledge.exploration_map[cell[0], cell[1]]

            # Penalización por celdas que NOSOTROS ya visitamos: decaimiento
            # temporal.  Recién visitada → novelty ≈ 0.05 (mínimo), pero
            # recupera con e^(-Δt/tau) conforme pasa el tiempo.
            if cell in self.cells_ever_explored:
                last_visit = self._visit_timestamps.get(cell, 0)
                dt = max(timestep - last_visit, 0)
                # Decaimiento: empieza en 0.05, sube hasta 1.0 con el tiempo
                recovery = 1.0 - np.exp(-dt / self._novelty_decay_tau)
                novelty *= max(0.05, recovery)
            # Penalización por celdas que OTROS exploraron (suave, caduca)
            elif cell in self.knowledge.cells_gossip_explored:
                ts = self.knowledge.cells_gossip_explored[cell]
                if (timestep - ts) < self.knowledge.gossip_expiry_ticks:
                    novelty *= 0.6

            # Repulsión: penalizar celdas cerca de otros agentes.
            # Curva 1/d^p con p configurable (Reynolds 1987 usa p=2 en
            # la regla de separation: la fuerza repulsiva debe crecer
            # más rápido que el inverso lineal para vencer al gradiente
            # de atracción cerca de los focos de probabilidad).
            repulsion = 0.0
            p = self.config.repulsion_exponent
            for npos in neighbor_positions:
                d = self.grid_distance(cell, npos)
                if d > 0:
                    repulsion += 1.0 / (d ** p)

            score = prob * novelty - self.config.repulsion_weight * repulsion
            # E8 (docs/20): hard-mask permanente sobre celdas ya observadas.
            # Imita el set global de celdas observadas del greedy centralizado.
            # La máscara incluye tanto celdas propias (cells_ever_explored)
            # como recibidas por gossip dentro del TTL. Con coeficiente 1.0,
            # la penalización iguala a ``prob_eff`` y deja el score ≈ -rep
            # para las celdas conocidas, replicando el efecto del hard-mask.
            eep = self.config.ever_explored_penalty
            if eep > 0:
                in_own = cell in self.cells_ever_explored
                in_gossip = False
                if not in_own:
                    ts = self.knowledge.cells_gossip_explored.get(cell)
                    if ts is not None and (timestep - ts) < self.knowledge.gossip_expiry_ticks:
                        in_gossip = True
                if in_own or in_gossip:
                    score -= eep * prob
            # Penalización estigmérgica por feromona de
            # presencia depositada en el entorno por TODOS los agentes.
            # Es repulsión regional (no sólo vecinos en comm_range) y se
            # difumina sola si los depositantes desaparecen.
            pw = self.config.presence_weight
            if pw > 0:
                score -= pw * float(self.knowledge.presence_field[cell[0], cell[1]])
            # Bonus aditivo por exploración: premia celdas no visitadas
            # con probabilidad > 0 para equilibrar explotación vs exploración.
            # Solo aplica a celdas con prob > 0 para no gastar budget en
            # zonas sin interés.
            if cell not in self.cells_ever_explored and prob > 0:
                score += self.config.exploration_weight
            # Dispersión por negociación (Boids-separation táctica): premia
            # celdas alineadas con el vector de huida del centroide de peers
            # vistos por gossip. cos_align ∈ [-1, 1] => +w en celda hacia
            # afuera, -w hacia el centroide. Ortogonal a feromonas: maneja
            # dispersión a escala TACTICA (vecinos directos en gossip), no
            # estigmérgica (cualquiera que pasó por la celda).
            if dispersal_weight_eff > 0:
                cdr = cell[0] - self.position[0]
                cdc = cell[1] - self.position[1]
                cnorm = float(np.hypot(cdr, cdc))
                if cnorm > 1e-9:
                    cos_align = (
                        (cdr / cnorm) * escape_unit[0]
                        + (cdc / cnorm) * escape_unit[1]
                    )
                    score += dispersal_weight_eff * cos_align
            # Anti-revisit corto (rompe oscilaciones A-B-A-B observadas en
            # baseline). El decaimiento exponencial de novelty con tau=200
            # satura al floor 0.05 durante los ~30 primeros ticks tras
            # visitar una celda, dejando todas las vecinas con la misma
            # novelty y permitiendo que el ruido fuerce flip-flop entre 2
            # celdas. Esta penalización LINEAL adicional, fuerte para dt=1
            # y nula para dt>=window, garantiza que el agente no desande
            # lo último a no ser que TODAS las vecinas estén también dentro
            # de la ventana (en cuyo caso elige la menos penalizada).
            arw = self.config.anti_revisit_window
            if arw > 0 and cell in self._visit_timestamps:
                dt = timestep - self._visit_timestamps[cell]
                if dt < arw:
                    score -= self.config.anti_revisit_penalty * (arw - dt) / arw
            # Jitter aleatorio para romper simetría entre agentes con
            # scoring casi idéntico (evita herding determinista)
            score += self._rng.random() * 1e-8
            if score > best_score:
                best_score = score
                best_cell = cell

        # Prioridad 3b -- buscar frontera si estamos en zona ya explorada
        # por NOSOTROS MISMOS (no por gossip, que causaría convergencia
        # de todos los drones al mismo punto frontera).
        all_own_visited = all(c in self.cells_ever_explored for c in reachable)
        if all_own_visited:
            frontier = self._find_nearest_frontier(timestep=timestep)
            if frontier is not None:
                # Comprometerse con la frontera: calcular TTL proporcional
                # a la distancia (en celdas) para llegar.
                dist_cells = max(
                    abs(frontier[0] - self.position[0]),
                    abs(frontier[1] - self.position[1]),
                )
                self._frontier_target = frontier
                self._frontier_ttl = dist_cells + 5  # margen extra
                return self._step_toward(frontier)
            # Si no hay frontera (todo explorado), intentar Lévy flight
            levy_target = self._levy_flight_target()
            if levy_target is not None:
                dist_cells = max(
                    abs(levy_target[0] - self.position[0]),
                    abs(levy_target[1] - self.position[1]),
                )
                self._frontier_target = levy_target
                self._frontier_ttl = dist_cells + 3
                return self._step_toward(levy_target)

        if best_cell is not None:
            return best_cell

        # Fallback -- paseo aleatorio (solo si no hay vecinos alcanzables)
        return self._random_walk(reachable)

    # -- Ejecución --

    def execute_move(self, target: tuple[int, int] | None, *, timestep: int = 0) -> None:
        """Mueve al agente a target consumiendo budget. Desactiva si se agota."""
        if target is None or not self.active:
            return
        cost = self._movement_cost(target)
        if cost > self.budget_remaining:
            self.active = False
            return
        # Actualizar media de transitabilidad experimentada
        trav = self.env.get_traversability(self.agent_type)
        self._trav_sum += trav[target[0], target[1]]
        self._trav_count += 1
        self.position = target
        self.budget_remaining -= cost
        self.path.append(target)
        # Registrar timestamp de visita para decaimiento temporal
        self._visit_timestamps[target] = timestep

    # -- Helpers --

    def grid_distance(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        """Distancia euclídea en metros entre dos celdas."""
        dr = (a[0] - b[0]) * self.env.grid.dy
        dc = (a[1] - b[1]) * self.env.grid.dx
        return np.sqrt(dr * dr + dc * dc)

    def _step_toward(self, target: tuple[int, int]) -> tuple[int, int]:
        """Celda adyacente que más acerca al objetivo.

        Tiebreaker anti-revisit: cuando hay vecinas equidistantes al target,
        el comportamiento por defecto de `min` produce oscilación A-B-A-B
        en frontier/return mode. Si `anti_revisit_window > 0`, la celda
        visitada hace MÁS tiempo gana (o la nunca-visitada).
        """
        neighbors = self._get_reachable_neighbors()
        if not neighbors:
            return self.position
        arw = self.config.anti_revisit_window
        if arw <= 0:
            return min(neighbors, key=lambda c: self.grid_distance(c, target))

        # Score = (distancia_al_target, antiguedad_de_visita_invertida)
        # Menor distancia primero; a igual distancia, celda visitada hace
        # más tiempo (o nunca → -inf → gana).
        def _key(c: tuple[int, int]) -> tuple[float, float]:
            d = self.grid_distance(c, target)
            last = self._visit_timestamps.get(c)
            # Sin visita: muy preferida (recencia = -infinito)
            recency = -float("inf") if last is None else float(last)
            return (d, recency)

        return min(neighbors, key=_key)

    def _random_walk(self, reachable: list[tuple[int, int]]) -> tuple[int, int]:
        """Elige un vecino alcanzable al azar."""
        idx = self._rng.integers(len(reachable))
        return reachable[idx]

    def _levy_flight_target(self) -> tuple[int, int] | None:
        """Genera un objetivo de salto largo con distribución de Lévy.

        La distancia sigue una distribución de potencia (Pareto):
        resultado entre 3 y 30 celdas con sesgo hacia distancias cortas.
        Dirección aleatoria uniforme. Devuelve la celda más cercana
        válida (dentro del grid y con prob > 0).
        """
        # Distancia con distribución de potencia: P(d) ~ d^(-1.5)
        # Pareto con alpha=1.5, mínimo 3 celdas, cap 30
        distance = min(30, int(3 * (self._rng.pareto(1.5) + 1)))
        angle = self._rng.uniform(0, 2 * np.pi)
        dr = int(round(distance * np.sin(angle)))
        dc = int(round(distance * np.cos(angle)))
        target_r = self.position[0] + dr
        target_c = self.position[1] + dc
        rows, cols = self.env.grid.rows, self.env.grid.cols
        target_r = int(np.clip(target_r, 0, rows - 1))
        target_c = int(np.clip(target_c, 0, cols - 1))
        # Solo si la celda destino tiene probabilidad > 0
        if self.knowledge.probability_map[target_r, target_c] > 0:
            return (target_r, target_c)
        return None

    def _find_nearest_frontier(self, max_scan: int = 100, *, timestep: int = 0) -> tuple[int, int] | None:
        """BFS ligero para encontrar la celda inexplorada más cercana con prob > 0.

        Expande en anillos de distancia Manhattan hasta *max_scan* celdas.
        Devuelve None si no encuentra frontera (toda la zona explorada).
        """
        r0, c0 = self.position
        rows, cols = self.env.grid.rows, self.env.grid.cols
        prob = self.knowledge.probability_map
        # Unión de celdas propias + gossip no caducado
        expiry = self.knowledge.gossip_expiry_ticks
        active_gossip = {
            c for c, ts in self.knowledge.cells_gossip_explored.items()
            if (timestep - ts) < expiry
        }
        known = self.cells_ever_explored | active_gossip

        for ring in range(1, max_scan + 1):
            best_prob = -1.0
            best_cell = None
            for dr in range(-ring, ring + 1):
                for dc in range(-ring, ring + 1):
                    if abs(dr) != ring and abs(dc) != ring:
                        continue  # solo el perímetro del anillo
                    nr, nc = r0 + dr, c0 + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if (nr, nc) not in known and prob[nr, nc] > 0:
                            if prob[nr, nc] > best_prob:
                                best_prob = prob[nr, nc]
                                best_cell = (nr, nc)
            if best_cell is not None:
                return best_cell
        return None

    def _detection_quality_at(self, row: int, col: int) -> float:
        """Calidad de detección base (sin modificar por terreno).

        Las subclases (DroneAgent, RobotDogAgent) sobreescriben este método
        para devolver la calidad ajustada al tipo de terreno.
        """
        return 1.0

    # -- Conversión a LineString (compatibilidad con PathEvaluator) --

    def get_path_linestring(self):
        """Devuelve el path como Shapely LineString en coordenadas mundo."""
        from shapely.geometry import LineString

        if len(self.path) < 2:
            return LineString()
        coords = [self.env.grid_to_world(r, c) for r, c in self.path]
        return LineString(coords)


# -- Tipos concretos de agente --


class DroneAgent(BaseSwarmAgent):
    """Dron aéreo -- ve desde arriba con penalización por dosel en zonas boscosas.

    Movimiento uniforme (vuela sobre todo el terreno sin restricciones).
    Detección modulada por terrain detection_modifier: en bosque ve peor.
    """

    agent_type = "drone"

    def _compute_detection_radius(self) -> float:
        from .config import DroneConfig
        if isinstance(self.config, DroneConfig):
            return self.config.detection_radius
        # fallback genérico
        return 80.0 * np.tan(np.radians(45.0 / 2))

    def get_visible_cells(self) -> set[tuple[int, int]]:
        """Celdas visibles con calidad de detección modulada por terreno.

        Filtra celdas cuyo modificador de detección sea < 0.05 (prácticamente
        invisible) para no desperdiciar registro de observación.
        """
        all_cells = self.env.get_visible_cells(
            self.position[0], self.position[1], self._detection_radius,
        )
        det_mod = self.env.get_detection_modifier(self.agent_type)
        return {c for c in all_cells if det_mod[c[0], c[1]] >= 0.05}

    def _detection_quality_at(self, row: int, col: int) -> float:
        """Calidad de detección [0, 1] en una celda según el terreno."""
        return float(self.env.get_detection_modifier(self.agent_type)[row, col])


class RobotDogAgent(BaseSwarmAgent):
    """Robot terrestre -- detección en suelo con bonificación en bosque.

    No puede cruzar agua (traversability infinita) y se mueve más lento
    en terrenos difíciles como roca o vegetación densa.
    Detecta mejor en bosque que el dron (sensores terrestres).
    """

    agent_type = "robot_dog"

    def _compute_detection_radius(self) -> float:
        from .config import RobotDogConfig
        if isinstance(self.config, RobotDogConfig):
            return self.config.detection_radius
        return 20.0

    def get_visible_cells(self) -> set[tuple[int, int]]:
        """Celdas visibles filtradas por calidad de detección terrestre.

        Filtra celdas donde el perro no puede detectar nada (agua=0).
        """
        all_cells = self.env.get_visible_cells(
            self.position[0], self.position[1], self._detection_radius,
        )
        det_mod = self.env.get_detection_modifier(self.agent_type)
        return {c for c in all_cells if det_mod[c[0], c[1]] >= 0.05}

    def _detection_quality_at(self, row: int, col: int) -> float:
        """Calidad de detección [0, 1] en una celda según el terreno."""
        return float(self.env.get_detection_modifier(self.agent_type)[row, col])
