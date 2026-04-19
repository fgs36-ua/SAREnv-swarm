# sarenv/swarm/export.py
"""
Utilidades de exportación de datos SAREnv para GAMA Platform.

Exporta heatmaps, features y posiciones de víctimas en formatos que
GAMA puede leer directamente (CSV, GeoJSON).

Uso::

    from sarenv.swarm.export import export_scenario_for_gama

    export_scenario_for_gama(
        dataset_item=item,
        output_dir="gama_model/includes",
        victim_cells={(10, 20), (30, 40)},
    )
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sarenv.core.loading import SARDatasetItem
    from .environment import SwarmEnvironment


def export_heatmap_csv(
    heatmap: np.ndarray,
    output_path: str | Path,
    normalize: bool = True,
) -> Path:
    """Exporta el heatmap de probabilidad como CSV para GAMA.

    Parameters
    ----------
    heatmap : np.ndarray
        Matriz 2D de probabilidad.
    output_path : str or Path
        Ruta del archivo CSV de salida.
    normalize : bool
        Si True, normaliza a [0, 1].

    Returns
    -------
    Path
        Ruta del archivo creado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = heatmap.astype(np.float32)
    if normalize:
        pmax = data.max()
        if pmax > 0:
            data = data / pmax

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow([f"{v:.6g}" for v in row])

    return output_path


def export_victims_csv(
    victim_cells: set[tuple[int, int]],
    environment: SwarmEnvironment,
    output_path: str | Path,
) -> Path:
    """Exporta posiciones de víctimas como CSV.

    Parameters
    ----------
    victim_cells : set of (row, col)
        Posiciones de víctimas en coordenadas grid.
    environment : SwarmEnvironment
        Para conversión grid→world.
    output_path : str or Path
        Ruta del archivo CSV de salida.

    Returns
    -------
    Path
        Ruta del archivo creado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "x", "y", "grid_row", "grid_col"])
        for i, (r, c) in enumerate(sorted(victim_cells)):
            x, y = environment.grid_to_world(r, c)
            writer.writerow([i, f"{x:.2f}", f"{y:.2f}", r, c])

    return output_path


def export_features_geojson(
    dataset_item: SARDatasetItem,
    output_path: str | Path,
) -> Path:
    """Exporta las features del escenario como GeoJSON.

    Parameters
    ----------
    dataset_item : SARDatasetItem
        Escenario SAREnv cargado.
    output_path : str or Path
        Ruta del archivo GeoJSON de salida.

    Returns
    -------
    Path
        Ruta del archivo creado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # features es un GeoDataFrame — exportar a GeoJSON
    gdf = dataset_item.features
    if gdf is not None and len(gdf) > 0:
        gdf.to_file(output_path, driver="GeoJSON")
    else:
        # GeoJSON vacío como fallback
        empty = {"type": "FeatureCollection", "features": []}
        output_path.write_text(json.dumps(empty))

    return output_path


def export_bounds_csv(
    environment: SwarmEnvironment,
    output_path: str | Path,
) -> Path:
    """Exporta los bounds del entorno como CSV.

    Parameters
    ----------
    environment : SwarmEnvironment
        Entorno del enjambre.
    output_path : str or Path
        Ruta del archivo CSV de salida.

    Returns
    -------
    Path
        Ruta del archivo creado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    g = environment.grid
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["param", "value"])
        writer.writerow(["minx", f"{g.minx:.2f}"])
        writer.writerow(["miny", f"{g.miny:.2f}"])
        writer.writerow(["maxx", f"{g.maxx:.2f}"])
        writer.writerow(["maxy", f"{g.maxy:.2f}"])
        writer.writerow(["rows", g.rows])
        writer.writerow(["cols", g.cols])
        writer.writerow(["dx", f"{g.dx:.4f}"])
        writer.writerow(["dy", f"{g.dy:.4f}"])

    return output_path


def export_scenario_for_gama(
    dataset_item: SARDatasetItem,
    environment: SwarmEnvironment,
    output_dir: str | Path,
    victim_cells: set[tuple[int, int]] | None = None,
) -> dict[str, Path]:
    """Exporta todos los datos de un escenario para GAMA.

    Crea los archivos necesarios en output_dir/ para que el modelo GAML
    pueda cargarlos:
      - heatmap.csv       — mapa de probabilidad normalizado
      - features.geojson  — features del terreno
      - victims.csv       — posiciones de víctimas
      - bounds.csv        — metadata del grid

    Parameters
    ----------
    dataset_item : SARDatasetItem
        Escenario cargado.
    environment : SwarmEnvironment
        Entorno (para conversiones).
    output_dir : str or Path
        Directorio de salida (normalmente gama_model/includes/).
    victim_cells : set of (row, col), optional
        Posiciones de víctimas.

    Returns
    -------
    dict[str, Path]
        Mapa nombre → ruta de los ficheros creados.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    files["heatmap"] = export_heatmap_csv(
        dataset_item.heatmap, output_dir / "heatmap.csv",
    )

    files["features"] = export_features_geojson(
        dataset_item, output_dir / "features.geojson",
    )

    files["bounds"] = export_bounds_csv(
        environment, output_dir / "bounds.csv",
    )

    if victim_cells:
        files["victims"] = export_victims_csv(
            victim_cells, environment, output_dir / "victims.csv",
        )

    return files
