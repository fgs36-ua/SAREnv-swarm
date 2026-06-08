# examples/01_dataset/01_generate_sar_data.py
import os

import geopandas as gpd
import numpy as np
from pathlib import Path
import shapely
from sarenv import (
    CLIMATE_DRY,
    CLIMATE_TEMPERATE,
    ENVIRONMENT_TYPE_FLAT,
    ENVIRONMENT_TYPE_MOUNTAINOUS,
    DataGenerator,
    get_logger,
)

log = get_logger()


def run_polygon_export_example():
    """
    An example function demonstrating how to use the DataGenerator
    to export features and heatmaps for a custom polygon area.
    """
    log.info("--- Starting DataGenerator Polygon Export Example ---")

    # 1. Initialize the generator.
    data_gen = DataGenerator()

    # 2. Define a simple 4-point polygon (a rectangle around Svanninge Bakker, Denmark)
    # Coordinates are in longitude, latitude (WGS84)
    polygon_coords = [
        [10.280, 55.140],  # Southwest corner
        [10.300, 55.140],  # Southeast corner
        [10.300, 55.150],  # Northeast corner
        [10.280, 55.150],  # Northwest corner
        [10.280, 55.140],  # Close the polygon (same as first point)
    ]
    
    # Create a shapely Polygon object
    custom_polygon = shapely.geometry.Polygon(polygon_coords)
    
    # Alternative: You can also use a GeoJSON-like dictionary format
    polygon_dict = {
        "type": "Polygon",
        "coordinates": [polygon_coords]
    }
    
    output_dir = "sarenv_dataset_polygon"

    # 3. Run the polygon-based export function using the shapely Polygon
    log.info("--- Exporting dataset using shapely Polygon ---")
    data_gen.export_dataset_from_polygon(
        polygon=custom_polygon,
        output_directory=output_dir,
        environment_climate=CLIMATE_TEMPERATE,
        environment_type=ENVIRONMENT_TYPE_FLAT,
        meter_per_bin=30,
    )

    # 4. Alternative example using the dictionary format
    output_dir_dict = "sarenv_dataset_polygon_dict"
    log.info("--- Exporting dataset using polygon dictionary ---")
    data_gen.export_dataset_from_polygon(
        polygon=polygon_dict,
        output_directory=output_dir_dict,
        environment_climate=CLIMATE_TEMPERATE,
        environment_type=ENVIRONMENT_TYPE_FLAT,
        meter_per_bin=30,
    )

    log.info("--- Verifying exported files ---")
    for output_directory in [output_dir, output_dir_dict]:
        try:
            # Check the files
            master_heatmap_path = Path(output_directory) / "heatmap.npy"
            master_features_path = Path(output_directory) / "features.geojson"

            if master_heatmap_path.exists():
                heatmap_matrix = np.load(master_heatmap_path)
                log.info(f"Loaded heatmap from '{output_directory}'. Shape: {heatmap_matrix.shape}")
                log.info(f"Heatmap sum (should be ~1.0): {np.sum(heatmap_matrix):.6f}")
            else:
                log.error(f"Verification failed: {master_heatmap_path} not found.")

            if master_features_path.exists():
                features_gdf = gpd.read_file(master_features_path)
                log.info(
                    f"Loaded features from '{output_directory}'. Found {len(features_gdf)} features."
                )
                if not features_gdf.empty:
                    log.info("Sample of loaded features:")
                    log.info(str(features_gdf[['feature_type', 'area_probability']].head()))
                    log.info(f"Total area probability sum: {features_gdf['area_probability'].sum():.6f}")
            else:
                log.error(f"Verification failed: {master_features_path} not found.")

        except Exception as e:
            log.error(f"An error occurred during verification of {output_directory}: {e}", exc_info=True)


def run_polygon_generation_only_example():
    """
    Example showing how to generate just the Environment object from a polygon
    without exporting files.
    """
    log.info("--- Starting Environment Generation from Polygon Example ---")
    
    # Initialize the generator
    data_gen = DataGenerator()
    
    # Define a triangle polygon (3 points + closing point)
    triangle_coords = [
        [10.285, 55.142],  # Point 1
        [10.295, 55.142],  # Point 2
        [10.290, 55.148],  # Point 3
        [10.285, 55.142],  # Close the polygon
    ]
    
    triangle_polygon = shapely.geometry.Polygon(triangle_coords)
    
    # Generate the environment
    log.info("--- Generating environment from triangle polygon ---")
    env = data_gen.generate_environment_from_polygon(
        polygon=triangle_polygon,
        meter_per_bin=20,  # Higher resolution
    )
    
    if env:
        log.info(f"Successfully generated environment with area: {env.area:.2f} m² ({env.area/1e6:.4f} km²)")
        log.info(f"Number of feature types found: {len([k for k, v in env.features.items() if v is not None and not v.empty])}")
        
        # Generate the combined heatmap
        combined_heatmap = env.get_combined_heatmap()
        if combined_heatmap is not None:
            log.info(f"Generated combined heatmap with shape: {combined_heatmap.shape}")
            log.info(f"Heatmap statistics - min: {np.min(combined_heatmap):.6f}, max: {np.max(combined_heatmap):.6f}")
        else:
            log.warning("Failed to generate combined heatmap")
    else:
        log.error("Failed to generate environment from polygon")



def run_export_example():
    """
    An example function demonstrating how to use the DataGenerator
    to export features and heatmaps for all quantiles.
    """
    log.info("--- Starting DataGenerator Export Example ---")

    # 1. Initialize the generator.
    data_gen = DataGenerator()
    
    # 2. Define a center point and an output directory for the dataset.
    initial_planning_point = 10.289470, 55.145921
    output_dir = "sarenv_dataset"

    # 3. Run the main export function.
    data_gen.export_dataset(
        center_point=initial_planning_point,
        output_directory=output_dir,
        environment_climate=CLIMATE_TEMPERATE,
        environment_type=ENVIRONMENT_TYPE_FLAT,
        meter_per_bin=30,
    )

    log.info("--- Verifying exported files ---")
    try:
        # Check the files for the 'median' quantile
        master_heatmap_path = os.path.join(output_dir, "heatmap.npy")
        master_features_path = os.path.join(output_dir, "features.geojson")

        if os.path.exists(master_heatmap_path):
            heatmap_matrix = np.load(master_heatmap_path)
            log.info(f"Loaded heatmap 'heatmap.npy'. Shape: {heatmap_matrix.shape}")
            # You could now use this matrix for analysis or as input to a model.
        else:
            log.error(f"Verification failed: {master_heatmap_path} not found.")

        if os.path.exists(master_features_path):
            features_gdf = gpd.read_file(master_features_path)
            log.info(
                f"Loaded features 'features.geojson'. Found {len(features_gdf)} features."
            )
            log.info("Sample of loaded features:")
            print(features_gdf.head())
        else:
            log.error(f"Verification failed: {master_features_path} not found.")

    except Exception as e:
        log.error(f"An error occurred during verification: {e}", exc_info=True)


if __name__ == "__main__":
    run_export_example()
