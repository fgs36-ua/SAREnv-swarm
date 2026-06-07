from sarenv import (
    DatasetLoader,
    get_logger,
    visualize_heatmap,
    visualize_features,
)

log = get_logger()


def run_loading_example():
    """
    An example function demonstrating how to load and visualize a single dataset.
    """
    log.info("--- Starting Single Dataset Loading and Visualization Example ---")

    dataset_dir = "maigmo_dataset"
    size_to_load = "xlarge"

    try:
        loader = DatasetLoader(dataset_directory=dataset_dir)
        log.info(f"Loading data for size: '{size_to_load}'")
        item = loader.load_environment(size_to_load)

        if item:
            # INSERT YOUR CODE HERE OR USE THE PROVIDED FUNCTIONS
            visualize_heatmap(item, plot_basemap=False, plot_inset=True)
            visualize_features(item, plot_basemap=False, plot_inset=True, num_lost_persons=300)
        else:
            log.error(f"Could not load the specified size: '{size_to_load}'")

    except FileNotFoundError:
        log.error(
            f"Error: The dataset directory '{dataset_dir}' or its master files were not found."
        )
        log.error(
            "Please run the `export_dataset()` method from the DataGenerator first."
        )
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    run_loading_example()
