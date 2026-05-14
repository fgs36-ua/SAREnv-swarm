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
    comm_range: float = 500.0          # alcance radio para gossip (metros)
    repulsion_weight: float = 0.3      # peso de la repulsión entre agentes cercanos
    repulsion_exponent: float = 1.0    # exponente p en repulsion ~ 1/d^p
    exploration_weight: float = 0.001  # bonus aditivo para celdas no visitadas
    # Iteración 3 (docs/16): peso del campo de feromona de presencia
    # depositado en el entorno por todos los agentes.
    # Default 0.01 = óptimo del barrido iter3.b en sarenv_s1
    # (+7.1% prob_coverage vs 0.0, +5.2% eff en maigmo).
    # Estigmergia pura (Payton 2001 / Howard et al. 2002 / Parunak 2002).
    presence_weight: float = 0.01
    alert_threshold: float = 0.5       # intensidad mínima de alerta para investigar
    return_safety_factor: float = 1.2  # margen de seguridad para vuelta a base
    # Anti-revisit corto: penaliza con fuerza las celdas visitadas en los
    # últimos `anti_revisit_window` ticks para romper oscilaciones A-B-A-B
    # (cuando la novelty está saturada en su floor y los empates de score
    # los rompe el ruido). Default 0 = desactivado (compatibilidad hacia
    # atrás con experimentos comparativos previos).
    anti_revisit_window: int = 0       # nº de ticks recientes a penalizar
    anti_revisit_penalty: float = 0.0  # magnitud de la penalización (lineal en window)


@dataclass
class DroneConfig(AgentConfig):
    """Parámetros específicos del dron aéreo."""

    agent_type: Literal["drone", "robot_dog"] = "drone"
    altitude: float = 80.0   # altitud de vuelo (metros)
    fov_deg: float = 45.0    # campo de visión de la cámara (grados)
    speed_ms: float = 15.0   # velocidad de crucero m/s (informativo)

    @property
    def detection_radius(self) -> float:
        """Radio de detección calculado a partir de altitud y FOV.

        Usa trigonometría básica: r = altitude * tan(fov/2).
        A 80m con FOV 45° da ~33m de radio, que cubre aprox. 1 celda
        del grid de 30m.
        """
        import numpy as np
        return self.altitude * np.tan(np.radians(self.fov_deg / 2))


@dataclass
class RobotDogConfig(AgentConfig):
    """Parámetros del robot perro terrestre.

    En Fase 1 se comporta igual que un dron (movimiento uniforme).
    En Fase 2 se añadirá restricción de terreno con max_slope.
    """

    agent_type: Literal["drone", "robot_dog"] = "robot_dog"
    sensor_range: float = 20.0   # radio de detección a nivel de suelo (metros)
    max_slope: float = 30.0      # pendiente máxima transitable (grados)
    speed_ms: float = 3.0        # velocidad de crucero m/s (informativo)
    comm_range: float = 100.0    # radio menor que los drones (al ir por tierra)

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

    # -- Iteración 3 (docs/16): feromona de presencia estigmérgica --
    # Default seguro: deposit > 0 pero diffusion off y evaporación moderada.
    # El peso (presence_weight) está en AgentConfig y por defecto 0 → OFF.
    presence_deposit: float = 1.0          # cantidad depositada por agente/tick
    presence_evaporation: float = 0.05     # decaimiento por tick (vida media ≈14)
    presence_diffusion_sigma: float = 2.0  # sigma del filtro gaussiano (celdas)
    presence_diffusion_period: int = 5     # difundir cada N ticks (0 = off)

    # -- Comunicación gossip --
    max_hops: int = 3              # profundidad máxima de retransmisión
    bandwidth_limit: int = 200     # updates máximos por intercambio
    lookback_timesteps: int = 50   # ventana temporal para compartir

    # -- Simulación --
    max_steps: int = 5_000                    # límite duro de ticks
    alert_probability_threshold: float = 0.7  # prob para emitir alerta

    # -- Composición del enjambre --
    num_drones: int = 3
    num_dogs: int = 0  # En Fase 1 solo drones; Fase 2 añade dogs
    drone_config: DroneConfig = field(default_factory=DroneConfig)
    dog_config: RobotDogConfig = field(default_factory=RobotDogConfig)

    # -- Presupuesto --
    budget_per_agent: float = 100_000.0  # metros por agente

    @property
    def total_agents(self) -> int:
        return self.num_drones + self.num_dogs
