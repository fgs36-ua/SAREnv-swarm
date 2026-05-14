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
import os
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


class GamaDisconnected(ConnectionError):
    """Levantada cuando GAMA cierra la conexión durante un envío."""


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

    def wait_for_ready(self, timeout: float = 600.0) -> bool:
        """Bloquea hasta recibir la señal ``READY`` desde GAMA.

        GAMA envía ``READY`` cuando el usuario pulsa Play en la GUI.
        Esto permite que Python no empiece a publicar datos hasta que el
        usuario haya iniciado el experimento.

        Parameters
        ----------
        timeout : float
            Tiempo máximo de espera en segundos.

        Returns
        -------
        bool
            True si se recibió ``READY``, False si timeout o desconexión.
        """
        if not self._client_socket:
            logger.error("wait_for_ready: GAMA no conectado.")
            return False

        logger.info("Esperando señal READY de GAMA (pulsa Play en la GUI)...")
        deadline = time.monotonic() + timeout
        buf = b""
        # Usamos el timeout corto del socket (1 s) para hacer polling
        # respetando el deadline global.
        while time.monotonic() < deadline:
            try:
                chunk = self._client_socket.recv(4096)
            except socket.timeout:
                continue
            except OSError as e:
                logger.warning("Error leyendo de GAMA: %s", e)
                return False
            if not chunk:
                logger.warning("GAMA cerró la conexión antes de enviar READY.")
                return False
            buf += chunk
            # GAMA envía mensajes terminados en '~' (con o sin '\n')
            text = buf.decode("utf-8", errors="ignore")
            for token in text.replace("\n", "").replace("\r", "").split("~"):
                token = token.strip()
                if token == "READY":
                    logger.info("READY recibido de GAMA. Iniciando simulación.")
                    return True
                if token:
                    logger.debug("Mensaje GAMA→Python ignorado: %r", token)
            buf = b""
        logger.error("Timeout esperando READY de GAMA.")
        return False

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

        # Helper: micro-sleep para evitar saturar el mailbox de GAMA cuando
        # se envían cientos de líneas seguidas (víctimas en datasets grandes).
        chunk_size = 25
        chunk_pause = 0.05  # 50 ms cada 25 líneas
        sent = 0

        def _flush_chunk(force: bool = False) -> None:
            nonlocal sent
            sent += 1
            if force or sent % chunk_size == 0:
                time.sleep(chunk_pause)

        # Línea cabecera — enviar cols/rows para que GAMA dimensione su mundo
        self._send_line(
            f"INIT|{num_drones}|{num_dogs}|{num_victims}"
            f"|{grid.cols}|{grid.rows}"
        )
        # Pausa extra tras el INIT para que GAMA dimensione el mundo antes de
        # empezar a recibir agentes.
        time.sleep(0.3)

        # Drones y perros
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
            _flush_chunk()

        # Pausa antes del bloque de víctimas
        time.sleep(0.2)

        # Víctimas
        for i, (r, c) in enumerate(sorted(victim_cells)):
            x, y = c + 0.5, r + 0.5  # grid pixel coords
            self._send_line(f"VICTIM|{i}|{x:.2f}|{y:.2f}")
            _flush_chunk()

        # Asegurar que el último chunk se ha drenado antes de INIT_END
        time.sleep(chunk_pause)

        # Fin de init
        self._send_line("INIT_END")
        # Margen amplio para que GAMA termine de procesar el mailbox antes
        # de empezar a recibir TICKs.
        time.sleep(2.0)
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
        """[DEPRECADO] Sólo emite la señal ``PHEROMONE``.

        Antes escribía ``exploration_field.csv`` que GAMA nunca leía
        (el ``.gaml`` mantiene su propia ``exploration_matrix`` que se
        rellena desde los mensajes ``AGENT``). Para evitar I/O muerto
        ahora sólo manda la señal de refresco. Mantener firma por
        compatibilidad.
        """
        del agents, includes_dir, max_dim  # silenciar lint
        self._send_line("PHEROMONE")

    def send_gossip_field(
        self,
        agents: list[BaseSwarmAgent],
        timestep: int,
        max_dim: int = 80,
    ) -> None:
        """Envía a GAMA un campo 2D con la propagación gossip actual.

        Para cada celda del grid se computa cuántos agentes la conocen
        vía ``cells_gossip_explored`` (no expirado). Se downsamplea a
        ``max_dim`` × ``max_dim`` por max-pooling y se envía inline en
        un único mensaje TCP::

            GOSSIP_DATA|cols|rows|v00,v01,...;v10,v11,...;...

        Donde cada fila va separada por ``;`` y cada celda por ``,``.
        Los valores son enteros [0, num_agents] (cuántos agentes
        conocen esa celda).
        """
        if not agents:
            return
        # Cogemos la forma del primer agente (todos comparten grid).
        shape = agents[0].knowledge.exploration_map.shape
        field = np.zeros(shape, dtype=np.int16)
        for a in agents:
            expiry = a.knowledge.gossip_expiry_ticks
            for (r, c), ts in a.knowledge.cells_gossip_explored.items():
                if (timestep - ts) < expiry:
                    field[r, c] += 1
        if field.max() == 0:
            return
        ds = self._downsample_max(field.astype(np.float32), max_dim=max_dim)
        rows, cols = ds.shape
        # Empaquetar como string CSV-en-una-línea
        row_strs = [
            ",".join(f"{int(v)}" for v in row) for row in ds
        ]
        payload = ";".join(row_strs)
        self._send_line(f"GOSSIP_DATA|{cols}|{rows}|{payload}")

    def send_comm_links(self, agents: list[BaseSwarmAgent]) -> None:
        """Envía a GAMA todos los pares de agentes activos en rango de
        comunicación mutuo, para que se dibujen como líneas.

        Formato::

            LINKS|n|x1,y1,x2,y2;x3,y3,x4,y4;...

        Coordenadas en píxeles GAMA (mismo sistema que ``AGENT``).
        Si no hay enlaces se envía ``LINKS|0|``.
        """
        active = [a for a in agents if getattr(a, "active", False)]
        segs: list[str] = []
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                dist = a._grid_distance(a.position, b.position)
                cr = min(a.config.comm_range, b.config.comm_range)
                if dist <= cr:
                    ar, ac = a.position
                    br, bc = b.position
                    segs.append(
                        f"{ac + 0.5:.1f},{ar + 0.5:.1f},"
                        f"{bc + 0.5:.1f},{br + 0.5:.1f}"
                    )
        payload = ";".join(segs)
        self._send_line(f"LINKS|{len(segs)}|{payload}")

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
            try:
                self._client_socket.sendall(data.encode("utf-8"))
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError) as e:
                # GAMA cerró la conexión (Stop, ventana cerrada, crash...).
                # Marcamos el cliente como desconectado y dejamos de intentarlo.
                logger.warning("GAMA desconectado durante envío: %s", e)
                try:
                    self._client_socket.close()
                except OSError:
                    pass
                self._client_socket = None
                self._gama_connected.clear()
                raise GamaDisconnected(str(e)) from e

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
        """Escribe ndarray 2D como CSV de forma atómica.

        Usa un fichero temporal en el mismo directorio + ``os.replace``
        para evitar errores de I/O cuando GAMA tiene el CSV abierto para
        lectura. Reintenta varias veces ante OSError transitorios
        (Errno 22/13 típicos en Windows por bloqueos compartidos).
        """
        buf = io.StringIO()
        writer = csv.writer(buf)
        for row in arr:
            writer.writerow([f"{v:.6g}" for v in row])
        payload = buf.getvalue().encode("utf-8")

        tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        last_err: OSError | None = None
        for attempt in range(5):
            try:
                with open(tmp_path, "wb") as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
                return
            except OSError as e:
                last_err = e
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass
                time.sleep(0.05 * (attempt + 1))
        # Si tras 5 intentos sigue fallando, propagamos para que el
        # caller decida (send_pheromone lo captura y lo loguea).
        assert last_err is not None
        raise last_err
