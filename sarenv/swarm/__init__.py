# sarenv/swarm/__init__.py
"""
Módulo de enjambre para SAREnv.

Simulador tick-based con agentes heterogéneos (drones + robot dogs) que se
coordinan de forma descentralizada mediante feromonas virtuales y protocolo
gossip para misiones SAR.

Estructura:
    config.py         -> Dataclasses de configuración (SwarmConfig, AgentConfig...)
    environment.py    -> Wrapper del grid sobre SARDatasetItem
    knowledge.py      -> Mapa de conocimiento local por agente (feromonas)
    agents.py         -> Jerarquía de agentes (BaseSwarmAgent, DroneAgent...)
    communication.py  -> Protocolo gossip epidémico
    simulator.py      -> Motor principal del bucle de simulación
    metrics.py        -> Adaptador a PathEvaluator + métricas propias del enjambre
    terrain.py        -> Mapas de terreno: detección y transitabilidad
    gama_network_server.py -> Servidor TCP para GAMA GUI (visualización 3D)
    export.py         -> Exportación de datos para GAMA (CSV, GeoJSON)
"""

from .config import SwarmConfig, AgentConfig, DroneConfig, RobotDogConfig
from .agents import BaseSwarmAgent, DroneAgent, RobotDogAgent
from .simulator import SwarmSimulator
from .environment import SwarmEnvironment
from .knowledge import LocalKnowledgeMap, MapUpdate
from .communication import CommunicationProtocol
from .metrics import SwarmMetrics
from .comparative import SwarmComparativeEvaluator
from .terrain import (
    DETECTION_MODIFIERS,
    TRAVERSABILITY_COSTS,
    build_detection_modifier_map,
    build_traversability_map,
)
from .export import export_scenario_for_gama

# GamaNetworkServer no tiene dependencias externas.
from .gama_network_server import GamaNetworkServer

__all__ = [
    "SwarmConfig",
    "AgentConfig",
    "DroneConfig",
    "RobotDogConfig",
    "BaseSwarmAgent",
    "DroneAgent",
    "RobotDogAgent",
    "SwarmSimulator",
    "SwarmEnvironment",
    "LocalKnowledgeMap",
    "MapUpdate",
    "CommunicationProtocol",
    "SwarmMetrics",
    "SwarmComparativeEvaluator",
    "DETECTION_MODIFIERS",
    "TRAVERSABILITY_COSTS",
    "build_detection_modifier_map",
    "build_traversability_map",
    "export_scenario_for_gama",
    "GamaNetworkServer",
]
