# sarenv/swarm/communication.py
"""
Protocolo de comunicación gossip (epidémico) entre agentes del enjambre.

En Fase 1 se usa max_hops muy alto (≈ mapa global compartido) para poder
validar contra el greedy centralizado. En Fase 3 se limitará a hops
acotados y ancho de banda real.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agents import BaseSwarmAgent


class CommunicationProtocol:
    """Intercambio epidémico (gossip) entre pares de agentes en rango radio.

    Parámetros clave:
    - max_hops: profundidad máxima de retransmisión (∞ ≈ centralizado, 1 ≈ solo info propia)
    - bandwidth_limit: máx. MapUpdates por intercambio
    - lookback_timesteps: hasta cuántos ticks atrás compartir
    """

    def __init__(
        self,
        max_hops: int = 3,
        bandwidth_limit: int = 200,
        lookback_timesteps: int = 50,
    ) -> None:
        self.max_hops = max_hops
        self.bandwidth_limit = bandwidth_limit
        self.lookback_timesteps = lookback_timesteps

    def exchange(
        self,
        agent_a: BaseSwarmAgent,
        agent_b: BaseSwarmAgent,
        current_timestep: int,
    ) -> None:
        """Intercambio bidireccional: cada agente comparte sus updates recientes.

        Las alertas tienen prioridad sobre datos de exploración.
        """
        since = max(0, current_timestep - self.lookback_timesteps)

        updates_a = agent_a.knowledge.get_updates_since(since)
        updates_b = agent_b.knowledge.get_updates_since(since)

        # Ordenar: alertas primero, luego las más recientes
        key_fn = lambda u: (u.layer == "alert", u.timestamp)
        updates_a.sort(key=key_fn, reverse=True)
        updates_b.sort(key=key_fn, reverse=True)

        # Límite de ancho de banda
        updates_a = updates_a[: self.bandwidth_limit]
        updates_b = updates_b[: self.bandwidth_limit]

        # Merge bidireccional
        agent_b.knowledge.merge_updates(updates_a, self.max_hops)
        agent_a.knowledge.merge_updates(updates_b, self.max_hops)

        # Merge bidireccional del campo de presencia LOCAL (max-merge):
        # estigmergia swarm pura sin estado global. Tras el merge ambos
        # agentes ven la misma feromona y la evaporarán por separado.
        agent_a.knowledge.merge_presence(agent_b.knowledge.presence_field)
        agent_b.knowledge.merge_presence(agent_a.knowledge.presence_field)

        # Ping directo de posición (Boids-separation táctica): cada agente
        # registra dónde estaba el otro AHORA. Sin relay a terceros: si A
        # tenía info de C vista hace 20 ticks, no se la pasa a B (la posición
        # de C ya estaría obsoleta). Esto alimenta el término de dispersión
        # del scoring (ver agents.py + docs/17_negociacion_dispersion.md).
        agent_a.knowledge.record_peer_position(
            agent_b.id, agent_b.position, current_timestep,
        )
        agent_b.knowledge.record_peer_position(
            agent_a.id, agent_a.position, current_timestep,
        )
