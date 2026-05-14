# sarenv/swarm/simulator.py
"""
Motor tick-based del enjambre.

Orquesta el bucle completo de cada tick:
  perceive -> decide -> move -> observe -> communicate -> evaporate
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import LineString

from .agents import BaseSwarmAgent, DroneAgent, RobotDogAgent
from .communication import CommunicationProtocol
from .config import SwarmConfig
from .environment import SwarmEnvironment
from .knowledge import LocalKnowledgeMap

if TYPE_CHECKING:
    from sarenv.core.loading import SARDatasetItem


class SwarmSimulator:
    """Simulador tick-based del enjambre.

    Uso básico::

        env = SwarmEnvironment(dataset_item)
        sim = SwarmSimulator(env, config)
        history = sim.run()          # lista de snapshots por tick
        paths   = sim.get_paths()    # list[LineString] para PathEvaluator
    """

    def __init__(
        self,
        environment: SwarmEnvironment,
        config: SwarmConfig,
        seed: int | None = None,
    ) -> None:
        self.env = environment
        self.config = config
        self.timestep: int = 0
        self._rng = np.random.default_rng(seed)

        # Protocolo de comunicación
        self.comm = CommunicationProtocol(
            max_hops=config.max_hops,
            bandwidth_limit=config.bandwidth_limit,
            lookback_timesteps=config.lookback_timesteps,
        )

        # Crear agentes
        self.agents: list[BaseSwarmAgent] = self._create_agents()

    # -- Fábrica de agentes --

    def _create_agents(self) -> list[BaseSwarmAgent]:
        """Instancia drones y dogs repartidos en anillo alrededor del centro."""
        agents: list[BaseSwarmAgent] = []
        center_row, center_col = self.env.world_to_grid(
            self.env.center_x, self.env.center_y
        )

        total = self.config.total_agents
        for i in range(total):
            # Despliegue en anillo alrededor del centro.
            # Radio = min(dim) / 4 para maximizar la separación inicial,
            # escalando con √N para enjambres grandes.
            if total > 1:
                angle = 2 * np.pi * i / total
                min_dim = min(self.env.grid.rows, self.env.grid.cols)
                # Radio base: 1/4 de la dimensión menor del grid
                base_ring = max(3, min_dim // 4)
                # Escalar para enjambres grandes: √(N/4) con mínimo 1×
                ring_scale = max(1.0, np.sqrt(total / 4))
                offset = int(base_ring * ring_scale)
                # Limitar al 40 % del grid para no salir de la zona útil
                max_offset = min_dim * 2 // 5
                offset = min(offset, max_offset)
                start_r = int(np.clip(
                    center_row + int(offset * np.sin(angle)),
                    0, self.env.grid.rows - 1,
                ))
                start_c = int(np.clip(
                    center_col + int(offset * np.cos(angle)),
                    0, self.env.grid.cols - 1,
                ))
            else:
                start_r, start_c = center_row, center_col

            start_pos = (start_r, start_c)
            knowledge = LocalKnowledgeMap(self.env.probability_map)

            if i < self.config.num_drones:
                cfg = self.config.drone_config
                cfg_copy = replace(cfg, budget=self.config.budget_per_agent)
                agent = DroneAgent(
                    agent_id=f"drone_{i}",
                    config=cfg_copy,
                    environment=self.env,
                    knowledge=knowledge,
                    start_position=start_pos,
                    rng=np.random.default_rng(self._rng.integers(2**31)),
                )
            else:
                cfg = self.config.dog_config
                cfg_copy = replace(cfg, budget=self.config.budget_per_agent)
                agent = RobotDogAgent(
                    agent_id=f"dog_{i - self.config.num_drones}",
                    config=cfg_copy,
                    environment=self.env,
                    knowledge=knowledge,
                    start_position=start_pos,
                    rng=np.random.default_rng(self._rng.integers(2**31)),
                )
            agents.append(agent)
        return agents

    # -- Paso de simulación --

    def step(self) -> dict:
        """Ejecuta un tick del bucle de simulación (6 fases)."""
        active_agents = [a for a in self.agents if a.active]
        if not active_agents:
            return self._snapshot()

        # 1. PERCEPCIÓN
        perceptions: dict[str, object] = {}
        for agent in active_agents:
            nearby = [
                other for other in active_agents
                if other.id != agent.id
                and agent._grid_distance(agent.position, other.position)
                   <= agent.config.comm_range
            ]
            perceptions[agent.id] = agent.perceive(nearby)

        # 2. DECISIÓN (simultánea -- ningún agente tiene ventaja secuencial)
        decisions: dict[str, tuple[int, int] | None] = {}
        for agent in active_agents:
            decisions[agent.id] = agent.decide(perceptions[agent.id], timestep=self.timestep)

        # 3. MOVE
        for agent in active_agents:
            agent.execute_move(decisions[agent.id], timestep=self.timestep)

        # 3b. DEPÓSITO DE FEROMONA DE PRESENCIA (estigmergia swarm-local)
        # Cada agente activo deposita en SU PROPIO mapa local de presence.
        # El gossip (paso 5) se encarga de propagarlo a otros agentes en
        # rango vía merge ``np.maximum``.
        deposit = self.config.presence_deposit
        if deposit > 0:
            for agent in active_agents:
                if agent.active:
                    r, c = agent.position
                    agent.knowledge.deposit_presence(r, c, deposit)

        # 4. OBSERVACIÓN -- actualizar conocimiento local con lo que vemos
        for agent in active_agents:
            if not agent.active:
                continue
            visible = agent._get_visible_cells()
            # Contar celdas nuevas ANTES de actualizar (para anti-estancamiento)
            new_cells = len(visible - agent.cells_ever_explored)
            agent._recent_new_cells.append(new_cells)
            # Acumular celdas observadas (inmune a evaporación, para métricas)
            agent.cells_ever_explored.update(visible)
            # Registrar cada celda con su calidad de detección específica
            # según terreno (Fase 2: _detection_quality_at por agente)
            for cell in visible:
                quality = agent._detection_quality_at(cell[0], cell[1])
                agent.knowledge.record_observation(
                    {cell}, agent.id, self.timestep,
                    detection_quality=quality,
                )
                # Emitir alerta si vemos zona de alta probabilidad
                prob = self.env.probability_map[cell[0], cell[1]]
                if prob > self.config.alert_probability_threshold:
                    agent.knowledge.record_alert(
                        cell, prob * quality, agent.id, self.timestep,
                    )

        # 5. COMUNICACIÓN (gossip entre pares en rango radio)
        for a, b in self._get_pairs_in_comm_range():
            self.comm.exchange(a, b, self.timestep)

        # 6. EVAPORACIÓN
        for agent in self.agents:
            agent.knowledge.evaporate(
                self.config.evaporation_rate,
                self.config.alert_evaporation_rate,
            )

        # 6b. EVAPORACIÓN + DIFUSIÓN del campo de presencia LOCAL de cada
        # agente (estigmergia swarm pura: sin estado global del entorno).
        period = self.config.presence_diffusion_period
        diffuse_now = period > 0 and (self.timestep % period == 0)
        for agent in self.agents:
            agent.knowledge.decay_presence(
                evaporation_rate=self.config.presence_evaporation,
                diffusion_sigma=self.config.presence_diffusion_sigma,
                diffuse_now=diffuse_now,
            )

        self.timestep += 1
        return self._snapshot()

    def _get_pairs_in_comm_range(self) -> list[tuple[BaseSwarmAgent, BaseSwarmAgent]]:
        """Pares (a, b) de agentes activos dentro del rango de comunicación mutuo."""
        pairs: list[tuple[BaseSwarmAgent, BaseSwarmAgent]] = []
        active = [a for a in self.agents if a.active]
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                dist = a._grid_distance(a.position, b.position)
                comm_range = min(a.config.comm_range, b.config.comm_range)
                if dist <= comm_range:
                    pairs.append((a, b))
        return pairs

    def get_active_comm_pairs(self) -> list[tuple[BaseSwarmAgent, BaseSwarmAgent]]:
        """Pares de agentes activos en rango de comunicación mutuo (acceso público).

        Devuelve la misma lista que se usa internamente en cada step para el
        gossip exchange. Útil para logging externo y visualización.
        """
        return self._get_pairs_in_comm_range()

    def kill_agent(self, agent_id: str) -> bool:
        """Desactiva un agente (simula fallo catastrófico en campo).

        Returns True si se encontró y desactivó, False si no existe o ya
        estaba inactivo.
        """
        for agent in self.agents:
            if agent.id == agent_id and agent.active:
                agent.active = False
                return True
        return False

    # -- Ejecución completa --

    def run(self, max_steps: int | None = None) -> list[dict]:
        """Ejecuta la simulación completa hasta que no queden agentes activos
        o se alcance max_steps.
        """
        max_steps = max_steps or self.config.max_steps
        history: list[dict] = []

        for _ in range(max_steps):
            snapshot = self.step()
            history.append(snapshot)
            if not any(a.active for a in self.agents):
                break
        return history

    # -- Salida --

    def get_paths(self) -> list[LineString]:
        """LineString por agente en coords mundo (para PathEvaluator)."""
        return [agent.get_path_linestring() for agent in self.agents]

    def _snapshot(self) -> dict:
        """Captura el estado actual para análisis / animación."""
        return {
            "timestep": self.timestep,
            "positions": {
                a.id: a.position for a in self.agents
            },
            "budgets": {
                a.id: a.budget_remaining for a in self.agents
            },
            "active": {
                a.id: a.active for a in self.agents
            },
        }

    # -- Constructor de conveniencia --

    @classmethod
    def from_dataset_item(
        cls,
        dataset_item: SARDatasetItem,
        config: SwarmConfig | None = None,
        seed: int | None = None,
    ) -> SwarmSimulator:
        """Construye un simulador directamente desde un SARDatasetItem."""
        env = SwarmEnvironment(dataset_item)
        return cls(env, config or SwarmConfig(), seed=seed)
