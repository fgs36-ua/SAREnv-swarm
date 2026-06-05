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
    # Alcance radio para gossip (metros). Valor amplio para que la
    # información de exploración se propague entre agentes vecinos.
    comm_range: float = 2000.0
    repulsion_weight: float = 0.3      # peso de la repulsión entre agentes cercanos
    repulsion_exponent: float = 1.0    # exponente p en repulsion ~ 1/d^p
    exploration_weight: float = 0.001  # bonus aditivo para celdas no visitadas
    # Peso del campo de feromona de presencia depositado en el entorno
    # por todos los agentes (estigmergia, Payton 2001 / Parunak 2002).
    presence_weight: float = 0.01
    # Atenuación estigmérgica del prior (Payton 2001 / Parunak 2002):
    # prob_eff = prob * exp(-pheromone_attenuation * presence_field).
    # En vez de competir como término separado, la feromona REDUCE la
    # probabilidad percibida → el mapa adquiere un agujero en zonas
    # saturadas y el gradiente greedy apunta naturalmente hacia afuera.
    # 0.0 = desactivado.
    pheromone_attenuation: float = 0.0
    alert_threshold: float = 0.5       # intensidad mínima de alerta para investigar
    return_safety_factor: float = 1.2  # margen de seguridad para vuelta a base
    # Anti-revisit corto: penaliza con fuerza las celdas visitadas en los
    # últimos `anti_revisit_window` ticks para romper oscilaciones A-B-A-B
    # (cuando la novelty está saturada en su floor y los empates de score
    # los rompe el ruido). 0 = desactivado.
    anti_revisit_window: int = 0       # nº de ticks recientes a penalizar
    anti_revisit_penalty: float = 0.0  # magnitud de la penalización (lineal en window)

    # Hard-mask permanente sobre celdas ya observadas.
    # Imita el comportamiento del greedy centralizado (set global de
    # celdas observadas) pero respetando la naturaleza descentralizada
    # del enjambre: la máscara se construye a partir de
    # ``cells_ever_explored`` (propias, inmunes a evaporación) y de
    # ``cells_gossip_explored`` (recibidas por gossip dentro de la
    # ventana ``gossip_expiry_ticks``).
    #
    # En el scoring, cada celda candidata que pertenezca a esa máscara
    # paga una penalización proporcional a su probabilidad efectiva:
    #     score -= ever_explored_penalty * prob_eff
    # - 0.0 → desactivado.
    # - 1.0 → hard-mask puro (re-visita solo si TODAS las vecinas están
    #   también enmascaradas; en ese caso decide el resto del score).
    ever_explored_penalty: float = 0.0

    # Dispersión por negociación (Reynolds 1987 / Boids-separation a escala
    # táctica): cuando un agente recibe gossip de un peer, registra su
    # posición. En el scoring suma un término que premia ir en dirección
    # opuesta al centroide de los peers vistos recientemente. Así se
    # reparten el espacio sin coordinador, solo con info local recibida
    # por gossip directo. 0 = desactivado.
    dispersal_weight: float = 0.0      # peso del término de dispersión
    peer_position_ttl: int = 50        # ticks que sigo recordando la pos de un peer
    # Decaimiento del término de dispersión con la distancia al centroide:
    # weight_eff = dispersal_weight / (1 + dist_to_centroid / falloff).
    # Convierte la "huida" en "tendencia": cerca del cluster empuja fuerte,
    # lejos se desvanece. Default 30 celdas (~900 m a 30 m/celda).
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
        """Radio de detección calculado a partir de altitud y FOV.

        Usa trigonometría básica: r = altitude * tan(fov/2).
        A 80m con FOV 45° da ~33m de radio, que cubre aprox. 1 celda
        del grid de 30m.
        """
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

    # -- Feromona de presencia estigmérgica --
    # El peso (presence_weight) está en AgentConfig y por defecto 0.01.
    presence_deposit: float = 1.0          # cantidad depositada por agente/tick
    presence_evaporation: float = 0.05     # decaimiento por tick (vida media ≈14)
    presence_diffusion_sigma: float = 2.0  # sigma del filtro gaussiano (celdas)
    presence_diffusion_period: int = 5     # difundir cada N ticks (0 = off)

    # -- Comunicación gossip --
    # Con max_hops=1 la información se propaga solo a vecinos directos:
    # valores mayores aumentan el tráfico de mensajes sin mejora medible.
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
