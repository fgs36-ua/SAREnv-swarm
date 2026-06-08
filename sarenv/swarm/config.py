# sarenv/swarm/config.py
"""
Dataclasses de configuración del simulador de enjambre.

SwarmConfig controla los parámetros globales de la simulación (feromonas,
comunicación, composición del enjambre). Cada tipo de agente tiene su propia
config (DroneConfig, RobotDogConfig) con parámetros de detección y movimiento.

Los valores por defecto están pensados para un escenario SAREnv estándar
(escenarios "medium", resolución 30m/celda).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AgentConfig:
    """Configuración base compartida por todos los tipos de agente."""

    agent_type: Literal["drone", "robot_dog"] = "drone"
    budget: float = 100_000.0          # presupuesto de movimiento en metros
    comm_range: float = 2000.0         # alcance radio del gossip (metros)
    repulsion_weight: float = 0.3      # peso de la repulsión entre agentes cercanos
    repulsion_exponent: float = 1.0    # exponente p en repulsion ~ 1/d^p
    exploration_weight: float = 0.001  # bonus aditivo para celdas no visitadas
    presence_weight: float = 0.01      # peso de la feromona de presencia (estigmergia, Payton 2001)
    # Atenuación estigmérgica del prior: prob_eff = prob * exp(-attn * presence).
    # La feromona reduce la probabilidad percibida en zonas saturadas, así el
    # gradiente greedy apunta hacia afuera. 0.0 = desactivado.
    pheromone_attenuation: float = 0.0
    alert_threshold: float = 0.5       # intensidad mínima de alerta para investigar
    return_safety_factor: float = 1.2  # margen de seguridad para vuelta a base
    # Anti-revisit corto: penaliza celdas visitadas en los últimos N ticks
    # para romper oscilaciones A-B-A-B. 0 = desactivado.
    anti_revisit_window: int = 0       # nº de ticks recientes a penalizar
    anti_revisit_penalty: float = 0.0  # magnitud de la penalización (lineal en window)

    # Hard-mask sobre celdas ya observadas (propias + gossip vigente). Cada
    # celda enmascarada paga `ever_explored_penalty * prob_eff` en el scoring.
    # 0.0 = desactivado; 1.0 = hard-mask puro (solo re-visita si no hay alternativa).
    ever_explored_penalty: float = 0.0

    # Dispersión por negociación (Reynolds 1987 / Boids-separation táctica):
    # premia alejarse del centroide de los peers vistos por gossip, repartiendo
    # el espacio sin coordinador. 0 = desactivado.
    dispersal_weight: float = 0.0      # peso del término de dispersión
    peer_position_ttl: int = 50        # ticks que sigo recordando la pos de un peer
    # Decaimiento del término con la distancia: weight_eff = dispersal_weight /
    # (1 + dist/falloff). Cerca empuja fuerte, lejos se desvanece (~900 m a 30 m/celda).
    dispersal_falloff: float = 30.0


@dataclass
class DroneConfig(AgentConfig):
    """Parámetros específicos del dron aéreo."""

    agent_type: Literal["drone", "robot_dog"] = "drone"
    altitude: float = 80.0   # altitud de vuelo (metros)
    fov_deg: float = 45.0    # campo de visión de la cámara (grados)
    speed_ms: float = 15.0   # velocidad de crucero m/s (informativo)

    @property
    def detection_radius(self) -> float:
        """Radio de detección r = altitude * tan(fov/2). A 80m/FOV 45° ≈ 33m (~1 celda)."""
        import numpy as np
        return self.altitude * np.tan(np.radians(self.fov_deg / 2))


@dataclass
class RobotDogConfig(AgentConfig):
    """Parámetros del robot perro terrestre."""

    agent_type: Literal["drone", "robot_dog"] = "robot_dog"
    sensor_range: float = 20.0   # radio de detección a nivel de suelo (metros)
    max_slope: float = 30.0      # pendiente máxima transitable (grados)
    speed_ms: float = 3.0        # velocidad de crucero m/s (informativo)
    comm_range: float = 2000.0   # mismo alcance de gossip que el dron

    @property
    def detection_radius(self) -> float:
        return self.sensor_range


@dataclass
class SwarmConfig:
    """Configuración global de una ejecución del simulador de enjambre.

    Agrupa parámetros de feromonas, protocolo de comunicación y composición
    del enjambre. Los valores por defecto funcionan razonablemente con los
    escenarios medium de SAREnv (5 drones, budget 100km).
    """

    # -- Parámetros de feromonas --
    evaporation_rate: float = 0.01        # decaimiento por tick (exploración)
    alert_evaporation_rate: float = 0.005 # las alertas decaen más lento
    deposit_rate: float = 1.0             # intensidad base del depósito

    # -- Feromona de presencia estigmérgica (peso en AgentConfig.presence_weight) --
    presence_deposit: float = 1.0          # cantidad depositada por agente/tick
    presence_evaporation: float = 0.05     # decaimiento por tick (vida media ≈14)
    presence_diffusion_sigma: float = 2.0  # sigma del filtro gaussiano (celdas)
    presence_diffusion_period: int = 5     # difundir cada N ticks (0 = off)

    # -- Comunicación gossip --
    # max_hops=1 propaga solo a vecinos directos; valores mayores aumentan
    # el tráfico sin mejora medible.
    max_hops: int = 1              # profundidad máxima de retransmisión
    bandwidth_limit: int = 200     # updates máximos por intercambio
    lookback_timesteps: int = 50   # ventana temporal para compartir

    # -- Simulación --
    max_steps: int = 5_000                    # límite duro de ticks
    alert_probability_threshold: float = 0.7  # prob para emitir alerta

    # -- Composición del enjambre --
    num_drones: int = 3
    num_dogs: int = 0
    drone_config: DroneConfig = field(default_factory=DroneConfig)
    dog_config: RobotDogConfig = field(default_factory=RobotDogConfig)

    # -- Presupuesto --
    budget_per_agent: float = 100_000.0  # metros por agente

    @property
    def total_agents(self) -> int:
        return self.num_drones + self.num_dogs
