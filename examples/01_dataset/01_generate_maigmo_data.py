# examples/01_dataset/01_generate_maigmo_data.py
import os
import numpy as np
import geopandas as gpd
from sarenv import (
    CLIMATE_DRY,           # Maigmo es más seco que templado
    ENVIRONMENT_TYPE_MOUNTAINOUS, # Maigmo es montaña
    DataGenerator,
    get_logger,
)

log = get_logger()

def run_maigmo_export_example():
    """
    Generate a dataset for Sierra del Maigmó.
    """
    log.info("--- Starting Maigmó Dataset Generation ---")

    # 1. Initialize the generator.
    data_gen = DataGenerator()
    
    # 2. Define center point for Maigmó (from Google Maps)
    # Lat: 38.5027164, Lon: -0.6352669
    # Sarenv expects (lon, lat)
    maigmo_center_point = (-0.6352669, 38.5027164)
    
    output_dir = "maigmo_dataset"

    # 3. Run the export function.
    # Usamos CLIMATE_DRY y ENVIRONMENT_TYPE_MOUNTAINOUS por ser Alicante/Montaña
    data_gen.export_dataset(
        center_point=maigmo_center_point,
        output_directory=output_dir,
        environment_climate=CLIMATE_DRY,
        environment_type=ENVIRONMENT_TYPE_MOUNTAINOUS,
        meter_per_bin=30,
    )

    log.info(f"Dataset generated in {output_dir}")

if __name__ == "__main__":
    run_maigmo_export_example()
