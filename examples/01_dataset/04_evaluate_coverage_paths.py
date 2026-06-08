# examples/01_dataset/04_evaluate_coverage_paths.py
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

import sarenv
from sarenv.analytics import metrics
from sarenv.utils import plot
from sarenv.analytics.evaluator import ComparativeEvaluator
log = sarenv.get_logger()

if __name__ == "__main__":
    log.info("--- Initializing the Search and Rescue Toolkit ---")
    data_dir = "maigmo_dataset"  # Path to the dataset directory

    # 1. Initialize the evaluator
    evaluator = ComparativeEvaluator(
        dataset_directory=data_dir,
        evaluation_sizes=[ "medium"],
        num_drones=3,
        num_lost_persons=100,
        budget=1000000,  # 1000 km total para cubrir el área de Maigmó
    )

    # 2. Run the evaluations
    baseline_results, time_series_data = evaluator.run_baseline_evaluations()

    # 3. Plot the results from the baseline run
    evaluator.plot_results(baseline_results)

    # 4. Plot paths on heatmaps for each algorithm and dataset
    log.info("--- Generating Path Heatmap Visualizations ---")

    # Create output directory for heatmap plots
    output_dir = Path()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through each environment and algorithm to generate plots
    for size, env_data in evaluator.environments.items():
        item = env_data["item"]
        victims_gdf = env_data["victims"]

        # Create path evaluator for this environment
        path_evaluator = metrics.PathEvaluator(
            item.heatmap,
            item.bounds,
            victims_gdf,
            evaluator.path_generator_config.fov_degrees,
            evaluator.path_generator_config.altitude_meters,
            evaluator.loader._meter_per_bin,
        )

        # Get center point in projected coordinates
        center_proj = (
            gpd.GeoDataFrame(geometry=[Point(item.center_point)], crs="EPSG:4326")
            .to_crs(env_data["crs"])
            .geometry.iloc[0]
        )

        # Plot paths for each algorithm
        for name, generator in evaluator.path_generators.items():
            log.info(f"Generating heatmap plot for {name} on '{size}' dataset...")

            # Generate paths using the same logic as in the evaluator
            generated_paths = generator(
                center_proj.x,
                center_proj.y,
                item.radius_km * 1000,
                item.heatmap,
                item.bounds,
            )

            # Define plot bounds (use heatmap bounds)
            x_min, y_min, x_max, y_max = item.bounds

            # Create output filename
            output_file = output_dir / f"{name}_{size}_heatmap.pdf"

            # Plot the heatmap with paths
            plot.plot_heatmap(
                item=item,
                generated_paths=generated_paths,
                name=name,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                output_file=output_file
            )

            log.info(f"Saved heatmap plot: {output_file}")
    log.info("--- Path Heatmap Visualization Complete ---")
