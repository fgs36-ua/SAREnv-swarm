# sarenv/swarm/gama_network_server.py
"""
Servidor TCP para comunicación bidireccional GAMA ↔ Python.

GAMA Platform (GUI, con visualización 3D) se conecta como cliente TCP
usando su skill ``network``. Python ejecuta toda la lógica del enjambre
y publica el estado como texto pipe-delimited cada tick.

Protocolo (líneas delimitadas por newline, campos separados por ``|``):
    Python → GAMA:
        INIT|num_drones|num_dogs|num_victims|cols|rows
        DRONE|idx|x|y|budget
        DOG|idx|x|y|budget
        VICTIM|idx|x|y
        INIT_END
        TICK|step
        AGENT|type|idx|x|y|budget|active
        FOUND|x|y
        TICK_END
        PHEROMONE
        END
    GAMA → Python:
        READY

Uso::

    server = GamaNetworkServer(port=6869)
    server.start()
    server.wait_for_gama()            # Bloquea hasta que GAMA se conecte
    server.send_init(env, agents, victim_cells)
    for ...:
        server.send_tick(snapshot, agents, env, found)
    server.send_end()
    server.stop()
"""
from __future__ import annotations

import csv
import io
import logging
import socket
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .agents import BaseSwarmAgent
    from .environment import SwarmEnvironment

logger = logging.getLogger(__name__)


class GamaNetworkServer:
    """Servidor TCP que alimenta datos al modelo GAMA con skill network.

    Parameters
    ----------
    host : str
        Dirección de escucha. ``"localhost"`` por defecto.
    port : int
        Puerto TCP. ``6869`` por defecto.
    """

    def __init__(self, host: str = "localhost", port: int = 6869) -> None:
        self.host = host
        self.port = port
        self._server_socket: socket.socket | None = None
        self._client_socket: socket.socket | None = None
        self._running = False
        self._lock = threading.Lock()
        self._gama_connected = threading.Event()
        self._accept_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Inicia el servidor TCP y espera conexión de GAMA en background."""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(1)
        self._server_socket.settimeout(1.0)  # Para poder detener limpiamente
        self._running = True

        # Aceptar conexión en thread aparte para no bloquear
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="gama-accept"
        )
        self._accept_thread.start()
        logger.info(
            "Servidor TCP escuchando en %s:%d — abre el modelo en GAMA GUI",
            self.host, self.port,
        )

    def wait_for_gama(self, timeout: float = 300.0) -> bool:
        """Bloquea hasta que GAMA se conecte (o timeout).

        Returns
        -------
        bool
            True si GAMA se conectó, False si timeout.
        """
        logger.info("Esperando conexión de GAMA (timeout=%ds)...", int(timeout))
        connected = self._gama_connected.wait(timeout=timeout)
        if connected:
            logger.info("GAMA conectado!")
        else:
            logger.error("Timeout esperando conexión de GAMA.")
        return connected

    def stop(self) -> None:
        """Detiene el servidor y cierra todas las conexiones."""
        self._running = False
        if self._client_socket:
            try:
                self._client_socket.close()
            except OSError:
                pass
            self._client_socket = None
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None
        logger.info("Servidor TCP detenido.")

    # ------------------------------------------------------------------
    # Envío de datos a GAMA
    # ------------------------------------------------------------------

    def send_init(
        self,
        environment: SwarmEnvironment,
        agents: list[BaseSwarmAgent],
        victim_cells: set[tuple[int, int]],
    ) -> None:
        """Envía los datos de inicialización a GAMA.

        GAMA usa esto para crear agentes y posicionar víctimas.
        """
        grid = environment.grid
        num_drones = sum(1 for a in agents if a.agent_type == "drone")
        num_dogs = sum(1 for a in agents if a.agent_type == "robot_dog")
        num_victims = len(victim_cells)

        # Línea cabecera — enviar cols/rows para que GAMA dimensione su mundo
        self._send_line(
            f"INIT|{num_drones}|{num_dogs}|{num_victims}"
            f"|{grid.cols}|{grid.rows}"
        )

        # Drones
        drone_idx = 0
        dog_idx = 0
        for agent in agents:
            r, c = agent.position
            x, y = c + 0.5, r + 0.5  # grid pixel coords
            budget = round(float(agent.budget_remaining), 1)
            if agent.agent_type == "drone":
                self._send_line(f"DRONE|{drone_idx}|{x:.2f}|{y:.2f}|{budget}")
                drone_idx += 1
            else:
                self._send_line(f"DOG|{dog_idx}|{x:.2f}|{y:.2f}|{budget}")
                dog_idx += 1

        # Víctimas
        for i, (r, c) in enumerate(sorted(victim_cells)):
            x, y = c + 0.5, r + 0.5  # grid pixel coords
            self._send_line(f"VICTIM|{i}|{x:.2f}|{y:.2f}")

        # Fin de init
        self._send_line("INIT_END")
        # Dar tiempo a GAMA para procesar la ráfaga de init
        time.sleep(1.0)
        logger.info("Init enviado (%d drones, %d dogs, %d victims).",
                     num_drones, num_dogs, num_victims)

    def send_tick(
        self,
        snapshot: dict,
        agents: list[BaseSwarmAgent],
        environment: SwarmEnvironment,
        found_victim_cells: set[tuple[int, int]] | None = None,
    ) -> None:
        """Envía el estado de un tick a GAMA."""
        step = snapshot.get("timestep", 0)
        self._send_line(f"TICK|{step}")

        drone_idx = 0
        dog_idx = 0
        for agent in agents:
            pos = snapshot["positions"].get(agent.id, agent.position)
            budget = snapshot["budgets"].get(agent.id, 0.0)
            active = snapshot["active"].get(agent.id, False)
            r, c = pos
            x, y = c + 0.5, r + 0.5  # grid pixel coords
            atype = agent.agent_type
            idx = drone_idx if atype == "drone" else dog_idx
            active_str = "1" if active else "0"
            self._send_line(
                f"AGENT|{atype}|{idx}|{x:.2f}|{y:.2f}|{budget:.1f}|{active_str}"
            )
            if atype == "drone":
                drone_idx += 1
            else:
                dog_idx += 1

        if found_victim_cells:
            for r, c in found_victim_cells:
                x, y = c + 0.5, r + 0.5  # grid pixel coords
                self._send_line(f"FOUND|{x:.2f}|{y:.2f}")

        self._send_line("TICK_END")

    def send_pheromone(
        self,
        agents: list[BaseSwarmAgent],
        includes_dir: Path,
        max_dim: int = 100,
    ) -> None:
        """Escribe feromona en disco y avisa a GAMA."""
        maps = [a.knowledge.exploration_map for a in agents]
        fused = np.maximum.reduce(maps) if maps else maps[0]
        fused = self._downsample_max(fused, max_dim=max_dim)

        csv_path = includes_dir / "exploration_field.csv"
        self._write_csv(fused, csv_path)

        self._send_line("PHEROMONE")

    def send_end(self) -> None:
        """Señal de fin de simulación."""
        self._send_line("END")

    # ------------------------------------------------------------------
    # Recepción de mensajes de GAMA
    # ------------------------------------------------------------------

    def _accept_loop(self) -> None:
        """Thread que acepta una conexión entrante."""
        while self._running:
            try:
                conn, addr = self._server_socket.accept()
                logger.info("Conexión aceptada desde %s:%d", addr[0], addr[1])
                self._client_socket = conn
                self._client_socket.settimeout(1.0)
                self._client_socket.setsockopt(
                    socket.IPPROTO_TCP, socket.TCP_NODELAY, 1,
                )
                self._gama_connected.set()
                return  # Solo aceptamos una conexión
            except socket.timeout:
                continue
            except OSError:
                return

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _send_line(self, line: str) -> None:
        """Envía una línea de texto con delimitador ``~`` + newline.

        GAMA raw TCP mode strips ``\n``.  When TCP coalesces multiple
        sends into one packet, GAMA delivers them as a single mailbox
        message with the ``\n`` removed.  The ``~`` sentinel lets GAMA
        split the concatenated blob back into individual commands.
        """
        if not self._client_socket:
            raise RuntimeError("GAMA no conectado.")
        data = line + "~\n"
        with self._lock:
            self._client_socket.sendall(data.encode("utf-8"))

    @staticmethod
    def _downsample_max(arr: np.ndarray, max_dim: int = 100) -> np.ndarray:
        """Max-pooling para reducir resolución."""
        h, w = arr.shape
        factor = max(1, max(h, w) // max_dim)
        if factor <= 1:
            return arr
        new_h = h // factor
        new_w = w // factor
        trimmed = arr[: new_h * factor, : new_w * factor]
        return trimmed.reshape(new_h, factor, new_w, factor).max(axis=(1, 3))

    @staticmethod
    def _write_csv(arr: np.ndarray, path: Path) -> None:
        """Escribe ndarray 2D como CSV."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        for row in arr:
            writer.writerow([f"{v:.6g}" for v in row])
        path.write_text(buf.getvalue(), encoding="utf-8")
