import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from sarenv import (
    DatasetLoader,
    LostPersonLocationGenerator,
    get_logger,
)
from sarenv.utils.plot import DEFAULT_COLOR, FEATURE_COLOR_MAP
from shapely.geometry import Point

log = get_logger()

if __name__ == "__main__":
    log.info("--- Starting lost_person Location Generation Example ---")

    dataset_dir = "maigmo_dataset"
    size_to_load = "xlarge"
    num_locations = 100

    try:
        # 1. Load the dataset for a specific size
        log.info(f"Loading data for size: '{size_to_load}'")
        loader = DatasetLoader(dataset_directory=dataset_dir)
        dataset_item = loader.load_environment(size_to_load)

        if not dataset_item:
            log.error(f"Could not load the dataset for size '{size_to_load}'.")

        # 2. Initialize the lost_person location generator with the loaded data
        log.info("Initializing the lost_personLocationGenerator.")
        lost_person_generator = LostPersonLocationGenerator(dataset_item)

        # 3. Generate lost_person locations
        log.info(f"Generating {num_locations} lost_person locations...")
        locations = lost_person_generator.generate_locations(num_locations, 0) # 0% random samples

        if not locations:
            log.error("No lost_person locations were generated. Cannot visualize.")

    except FileNotFoundError:
        log.error(
            f"Error: The dataset directory '{dataset_dir}' or its master files were not found."
        )
        log.error(
            "Please run the `export_dataset()` method from the DataGenerator first."
        )
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)

