# GAMA Model — SAR Realtime Visualization

Modelo GAML para visualizar en tiempo real la simulación del enjambre SAR
con GAMA Platform (GUI). Python ejecuta toda la lógica; GAMA renderiza.

## Estructura

```
gama_model/
├── models/
│   └── sar_network.gaml     ← Modelo GAML (TCP client, visualización 3D)
├── includes/                ← Datos exportados por Python (auto-generados)
│   ├── heatmap.csv          ← Mapa de probabilidad normalizado
│   ├── features.geojson     ← Features del terreno (agua, vegetación, caminos)
│   ├── victims.csv          ← Posiciones de víctimas
│   └── bounds.csv           ← Metadata del grid (bounds, resolución)
└── README.md
```

## Prerrequisitos

- **GAMA Platform** ≥ 2025-06: https://gama-platform.org/download

## Uso

### Paso 1: Ejecutar el servidor Python

```bash
python examples/02_swarm/03_gama_visualization.py --scenario 1 --num-drones 5 --num-dogs 2
```

El script carga el escenario, exporta el heatmap, e inicia un servidor TCP
en el puerto 6869. Queda esperando a que GAMA se conecte.

### Paso 2: Abrir GAMA Platform (GUI)

1. Abre GAMA Platform (la aplicación de escritorio)
2. `File → Import → Existing Projects` → selecciona la carpeta `gama_model/`
3. Abre `models/sar_network.gaml`
4. Ejecuta el experimento **`sar_gui_network`**
5. GAMA se conecta al servidor Python y empieza a recibir datos
6. Verás la visualización 3D: terreno, drones, perros, víctimas, exploración

### Opciones del servidor

| Flag | Default | Descripción |
|------|---------|-------------|
| `--scenario` | 1 | ID del escenario (1-60) |
| `--num-drones` | 5 | Número de drones |
| `--num-dogs` | 0 | Número de perros robot |
| `--num-victims` | 200 | Víctimas a generar |
| `--max-steps` | 15000 | Ticks de simulación |
| `--tick-delay-ms` | 50 | Pausa entre ticks (ms) |
| `--links-interval` | 1 | Cada cuántos ticks enviar enlaces de comunicación |
| `--server-port` | 6869 | Puerto TCP del servidor |

## Notas

- Los archivos en `includes/` se regeneran cada vez que se ejecuta el script.
- El modelo `sar_network.gaml` usa la `network` skill de GAMA para recibir datos por TCP.
- Python envía coordenadas en píxeles de grid (col, row). El mundo GAMA se ajusta al tamaño del heatmap.
