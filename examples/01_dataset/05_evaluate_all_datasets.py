# examples/01_dataset/05_evaluate_all_datasets.py
import numpy as np
import pandas as pd
from pathlib import Path
import sarenv
from sarenv.analytics.evaluator import ComparativeDatasetEvaluator, PathGenerator
from sarenv.analytics import paths
import argparse

log = sarenv.get_logger()


def create_custom_path_generator():
    """
    Example of how to create a custom path generator.
    This is a simple example that creates a straight line path.
    """

    def custom_straight_line_path(center_x, center_y, max_radius, **kwargs):
        """Custom path generator that creates a straight line."""
        num_drones = kwargs.get("num_drones", 3)
        path_point_spacing_m = kwargs.get("path_point_spacing_m", 10.0)

        # Create a simple straight line path
        num_points = int(max_radius * 2 / path_point_spacing_m)
        x_coords = np.linspace(center_x - max_radius, center_x + max_radius, num_points)
        y_coords = np.full_like(x_coords, center_y)

        from shapely.geometry import LineString

        full_path = LineString(zip(x_coords, y_coords, strict=True))

        # Split for multiple drones
        return paths.split_path_for_drones(full_path, num_drones)

    return PathGenerator(
        name="CustomStraightLine",
        func=custom_straight_line_path,
        description="Custom straight line path generator",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAR datasets with custom parameters.")
    parser.add_argument("--budget", type=int, default=300_000, help="Budget in meters (default: 100000)")
    parser.add_argument("--num_drones", type=int, default=5, help="Number of drones (default: 2)")
    args = parser.parse_args()

    log.info("--- Starting Comparative Dataset Evaluator ---")

    # Example 1: Using default path generators
    evaluator = ComparativeDatasetEvaluator(
        dataset_dirs=[f"sarenv_dataset/{i}" for i in range(1, 61)],
        budget=args.budget,
        num_drones=args.num_drones,
        evaluation_sizes=["medium",],
    )

    # # Example 2: Using custom path generators
    # custom_generators = {
    #     "CustomLine": create_custom_path_generator(),
    #     "Greedy": PathGenerator("Greedy", paths.generate_greedy_path),
    # }

    # evaluator_custom = ComparativeDatasetEvaluator(
    #     dataset_dirs=[f"sarenv_dataset/{i}" for i in range(1, 60)],
    #     budget=args.budget,
    #     num_drones=args.num_drones,
    #     evaluation_sizes=["small", "medium", "large"],  # Budget in meters
    #     custom_generators=custom_generators,
    # )

    log.info(f"Using path generators: {list(evaluator.path_generators.keys())}")

    # 2. Run the evaluations
    metrics_df, time_series_df = evaluator.evaluate(output_dir="results")

    # 3. Show summary of results
    per_dataset_results_df = evaluator.get_results_per_dataset()
    summarized_results_df = evaluator.summarize_results()
    log.info("--- Summary of Results ---")
    log.info(str(summarized_results_df))

    # 4. Generate comparative plots
    log.info("--- Generating Comparative Plots ---")
    from sarenv.utils.plot import create_individual_metric_plots
    
    # Create plots for different budget conditions
    result_files = [
        f"results/comparative_metrics_results_n{args.num_drones}_budget{args.budget}.csv",
        f"results/comparative_metrics_results_n{args.num_drones}_budget{args.budget//3}.csv"  # Lower budget for comparison
    ]
    
    # Calculate budget per drone for labels
    budget_per_drone_high = args.budget // args.num_drones
    budget_per_drone_low = (args.budget // 3) // args.num_drones
    
    budget_labels = [f'${budget_per_drone_high//1000}$km', f'${budget_per_drone_low//1000}$km']
    
    try:
        create_individual_metric_plots(
            result_files,
            environment_size='medium',
            output_dir='plots',
            budget_labels=budget_labels
        )
        log.info("Comparative plots generated successfully")
    except Exception as e:
        log.warning(f"Could not generate comparative plots: {e}")
        log.info("This is expected if you only have results for one budget condition")
    
    # Alternative: Create plots from the current results dataframe
    try:
        create_individual_metric_plots(
            metrics_df,
            environment_size='medium',
            output_dir='plots/current_run'
        )
        log.info("Current run plots generated successfully")
    except Exception as e:
        log.warning(f"Could not generate current run plots: {e}")

    log.info("--- Finished Comparative Dataset Evaluator ---")
