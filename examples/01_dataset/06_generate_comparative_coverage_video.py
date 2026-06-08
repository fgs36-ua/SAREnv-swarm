# examples/01_dataset/06_generate_comparative_coverage_video.py
"""
Simplified Comparative Coverage Video Generator

Creates a comparative coverage video showing 4 algorithms (Concentric, Pizza, Greedy, RandomWalk) 
in a 2x2 grid layout with time-series graphs showing metrics evolution.

Features:
- Efficient precomputed metrics approach
- Clean and readable code structure
- Multiple drone support with distinct colors
- Real-time metrics visualization

Usage:
    python     # Distance in meters between metric calculations (determines video granularity and performance)
    # Lower values = higher video quality but longer computation time
    # Recommended: 1000-2500m for good balance between quality and performance
    interval_distance = 2500.0  # Increased for faster testingenerate_comparative_coverage_video.py
"""

import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better performance
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import pickle

import sarenv
from sarenv.analytics import metrics
from sarenv.analytics.evaluator import ComparativeEvaluator
from sarenv.utils.plot import setup_algorithm_plot, plot_drone_paths, plot_current_drone_positions, create_time_series_graphs

log = sarenv.get_logger()


def compute_algorithm_metrics(args):
    """
    Helper function for parallel computation of algorithm metrics.
    This function is designed to work with multiprocessing.
    """
    alg_name, paths, path_evaluator_data, interval_distance = args
    
    try:
        # Reconstruct path_evaluator from serialized data
        path_evaluator = metrics.PathEvaluator(
            path_evaluator_data['heatmap'],
            path_evaluator_data['bounds'],  # This becomes 'extent' in PathEvaluator
            path_evaluator_data['victims_gdf'],
            path_evaluator_data['fov_degrees'],
            path_evaluator_data['altitude_meters'],
            path_evaluator_data['meter_per_bin']
        )
        
        log.info(f"Computing metrics for {alg_name} with {len(paths)} paths...")
        
        # Use the configured interval distance
        precomputed_data = path_evaluator.calculate_metrics_at_distance_intervals(
            paths, discount_factor=0.999, interval_distance=interval_distance
        )
        
        if not precomputed_data.get('interval_metrics'):
            log.warning(f"No precomputed metrics available for {alg_name}")
            return alg_name, None
            
        log.info(f"Successfully computed metrics for {alg_name}")
        return alg_name, precomputed_data
        
    except Exception as e:
        log.error(f"Error computing metrics for {alg_name}: {e}")
        return alg_name, None


class ComparativeCoverageVideoGenerator:
    """Generates a comparative video showing 4 algorithms side by side with time-series graphs."""
    
    def __init__(self, item, victims_gdf, path_evaluator, crs, output_dir, interval_distance=2500.0):
        self.item = item
        self.victims_gdf = victims_gdf
        self.path_evaluator = path_evaluator
        self.crs = crs
        self.output_dir = Path(output_dir)
        self.interval_distance = interval_distance  # Distance in meters between calculations
        
        # Store evaluator configuration for serialization
        self.evaluator_config = None  # Will be set later if needed
        
        # Video settings
        self.display_fps = 24  # Smooth visual display at 24fps
        self.metrics_fps = 4   # Metrics calculated at 4fps intervals
        self.dpi = 100
        self.figsize = (20, 12)  # Large figure for 2x2 + graphs
        self.n_frames = None  # Will be determined by the path lengths and interval distance
        self.metrics_frames = None  # Number of frames where metrics are actually calculated
        
        # Define colors for multiple drones
        self.drone_colors = ['blue', 'green', 'gray', 'purple', 'orange']
        
        # Algorithm names and colors for graphs
        self.algorithm_colors = {
            'RandomWalk': 'orange',
            'Greedy': 'green', 
            'Concentric': 'blue',
            'Pizza': 'red'
        }
    
    def set_evaluator_config(self, evaluator_config):
        """Set the evaluator configuration for serialization purposes."""
        self.evaluator_config = evaluator_config
    
    def create_comparative_video(self, algorithms_data):
        """Create a comparative video showing all algorithms side by side."""
        log.info("Creating comparative coverage video...")
        
        # Prepare serializable path_evaluator data for multiprocessing
        # Use the original constructor parameters
        path_evaluator_data = {
            'heatmap': self.item.heatmap,
            'bounds': self.item.bounds,  # This becomes 'extent' in PathEvaluator
            'victims_gdf': self.victims_gdf,
            'fov_degrees': self.evaluator_config['fov_degrees'] if self.evaluator_config else 45.0,
            'altitude_meters': self.evaluator_config['altitude_meters'] if self.evaluator_config else 100.0,
            'meter_per_bin': self.evaluator_config['meter_per_bin'] if self.evaluator_config else 30
        }
        
        # Prepare arguments for parallel processing
        parallel_args = []
        target_algorithms = []
        
        for alg_name, paths in algorithms_data.items():
            if alg_name in self.algorithm_colors:  # Only process target algorithms
                parallel_args.append((alg_name, paths, path_evaluator_data, self.interval_distance))
                target_algorithms.append(alg_name)
        
        if not parallel_args:
            log.error("No target algorithms found for processing")
            return
        
        # Compute metrics in parallel using all available cores
        num_workers = min(len(parallel_args), mp.cpu_count())
        log.info(f"Computing metrics for {len(parallel_args)} algorithms in parallel using {num_workers} cores out of {mp.cpu_count()} available...")
        all_animation_data = {}
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_alg = {executor.submit(compute_algorithm_metrics, args): args[0] 
                           for args in parallel_args}
            
            # Collect results as they complete
            completed_count = 0
            for future in as_completed(future_to_alg):
                alg_name = future_to_alg[future]
                completed_count += 1
                try:
                    result_alg_name, precomputed_data = future.result()
                    
                    if precomputed_data is not None:
                        log.info(f"[{completed_count}/{len(parallel_args)}] Processing animation data for {result_alg_name}...")
                        animation_data = self._process_precomputed_data(precomputed_data, result_alg_name)
                        
                        if animation_data['num_drones'] > 0:
                            all_animation_data[result_alg_name] = animation_data
                            log.info(f"✓ Prepared animation data for {result_alg_name} with {len(animation_data['positions'])} frames.")
                        else:
                            log.warning(f"✗ Skipping {result_alg_name} - no valid animation data")
                    else:
                        log.warning(f"✗ No valid metrics computed for {result_alg_name}")
                        
                except Exception as e:
                    log.error(f"✗ Error processing results for {alg_name}: {e}")
        
        parallel_time = time.time() - start_time
        log.info(f"Parallel metrics computation completed in {parallel_time:.2f} seconds using {num_workers} cores")
        
        if not all_animation_data:
            log.error("No valid animation data found for any algorithm")
            return
            
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / "comparative_coverage_video.mp4"
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(self.figsize[0] * self.dpi)
        frame_height = int(self.figsize[1] * self.dpi)
        
        video_writer = cv2.VideoWriter(str(output_file), fourcc, self.display_fps, (frame_width, frame_height))
        if not video_writer.isOpened():
            log.error(f"Failed to open video writer for {output_file}")
            return
            
        log.info(f"Video writer initialized: {output_file}")
        
        try:
            # Determine total frames based on metrics intervals - create multiple frames per interval for smoothness
            max_metrics_frames = max(len(data['positions']) for data in all_animation_data.values())
            base_frames = min(self.n_frames, max_metrics_frames) if self.n_frames else max_metrics_frames
            
            # Create multiple frames per metric interval for smoother video
            frames_per_interval = max(1, int(self.display_fps / self.metrics_fps))  # 6 frames per interval for 24fps
            total_frames = base_frames * frames_per_interval

            log.info(f"Generating {total_frames} total frames ({base_frames} metric intervals × {frames_per_interval} frames per interval)")

            # Generate each frame
            for frame_idx in range(total_frames):
                if frame_idx % (frames_per_interval * 5) == 0:  # Progress updates every 5 intervals
                    progress = 100 * frame_idx / total_frames
                    interval_idx = frame_idx // frames_per_interval
                    log.info(f"Generating frame {frame_idx + 1}/{total_frames} (interval {interval_idx})... ({progress:.1f}%)")
                    
                try:
                    frame = self._create_video_frame(frame_idx, all_animation_data, frames_per_interval)
                    if frame is not None:
                        video_writer.write(frame)
                    else:
                        log.warning(f"Failed to create frame {frame_idx}")
                except Exception as e:
                    log.error(f"Error creating frame {frame_idx}: {e}")
                    continue
                    
        except Exception as e:
            log.error(f"Error during video generation: {e}")
        finally:
            video_writer.release()
            log.info("Video writer released.")
        
        if output_file.exists():
            log.info(f"Saved comparative coverage video: {output_file}")
        else:
            log.error("Failed to save video file")
    
    def _create_video_frame(self, frame_idx, all_animation_data, frames_per_interval):
        """Create a single video frame showing progressive path building and actual drone positions."""
        try:
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            gs = fig.add_gridspec(6, 6, width_ratios=[1,1,1,1,0.8,0.8], hspace=0.25, wspace=0.25)
            
            # Algorithm subplot positions
            ax_positions = {
                'RandomWalk': (0, 0, 3, 2),
                'Greedy': (0, 2, 3, 2),
                'Concentric': (3, 0, 3, 2),
                'Pizza': (3, 2, 3, 2)
            }
            
            algorithm_axes = {}
            for alg_name, (row, col, rowspan, colspan) in ax_positions.items():
                algorithm_axes[alg_name] = fig.add_subplot(gs[row:row+rowspan, col:col+colspan])
            
            # Metrics subplots
            ax_area = fig.add_subplot(gs[0:2, 4:])
            ax_score = fig.add_subplot(gs[2:4, 4:])
            ax_victims = fig.add_subplot(gs[4:6, 4:])
            
            self._create_comparative_frame(frame_idx, all_animation_data, algorithm_axes, ax_area, ax_score, ax_victims, frames_per_interval)
            
            # Convert to video frame
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            buf = np.asarray(buf)
            buf = buf[:, :, :3]  # Remove alpha channel
            frame = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            
            plt.close(fig)
            return frame
            
        except Exception as e:
            log.error(f"Error creating video frame {frame_idx}: {e}")
            return None
        
    def _process_precomputed_data(self, precomputed_data, algorithm_name):
        """Process precomputed metrics data into animation data format."""
        start_time = time.time()
        
        try:
            # Extract data from precomputed results
            total_intervals = precomputed_data['total_intervals']
            interval_positions = precomputed_data['interval_positions']
            interval_metrics = precomputed_data['interval_metrics']
            interval_path_coordinates = precomputed_data.get('interval_path_coordinates', [])
            
            # Set n_frames to match the total intervals (metrics calculations)
            self.n_frames = total_intervals
            log.info(f"Using {total_intervals} precomputed intervals for {self.n_frames} video frames")
            
            # Map intervals to video frames - no need for fps expansion since we use actual interval data
            all_positions = []
            all_drone_positions = []
            all_metrics = []
            all_path_coordinates = []
            all_interval_distances = []
            
            for interval_idx in range(total_intervals):
                # Store the interval distance (in km)
                interval_distance_km = interval_idx * (self.interval_distance / 1000.0)
                all_interval_distances.append(interval_distance_km)
                
                # Get positions for this interval
                if interval_idx < len(interval_positions):
                    current_positions = interval_positions[interval_idx]
                    all_positions.append(current_positions[0] if current_positions else (0, 0))
                    all_drone_positions.append(current_positions)
                else:
                    # Use last available positions
                    last_positions = interval_positions[-1] if interval_positions else [(0, 0)]
                    all_positions.append(last_positions[0])
                    all_drone_positions.append(last_positions)
                
                # Get path coordinates for progressive path rendering
                if interval_path_coordinates and interval_idx < len(interval_path_coordinates[0]) if interval_path_coordinates else False:
                    frame_path_coords = []
                    for drone_idx in range(len(interval_path_coordinates)):
                        if interval_idx < len(interval_path_coordinates[drone_idx]):
                            # Get the progressive path coordinates up to this interval
                            frame_path_coords.append(interval_path_coordinates[drone_idx][interval_idx])
                        else:
                            frame_path_coords.append(interval_path_coordinates[drone_idx][-1])
                    all_path_coordinates.append(frame_path_coords)
                else:
                    # Use last available coordinates or empty
                    if all_path_coordinates:
                        all_path_coordinates.append(all_path_coordinates[-1])
                    else:
                        all_path_coordinates.append([])
                
                # Get metrics for this interval
                if interval_idx < len(interval_metrics):
                    all_metrics.append(interval_metrics[interval_idx])
                else:
                    # Use last available metrics
                    all_metrics.append(interval_metrics[-1] if interval_metrics else 
                                     {'area_covered': 0, 'likelihood_score': 0, 'victims_found_pct': 0})
            
            # Determine number of drones from precomputed data
            num_drones = len(interval_positions[0]) if interval_positions and interval_positions[0] else 0
            
            total_time = time.time() - start_time
            log.info(f"Finished processing animation data for {algorithm_name} in {total_time:.2f} seconds.")
            
            return {
                'positions': all_positions,
                'drone_positions': all_drone_positions,
                'path_coordinates': all_path_coordinates,
                'metrics': all_metrics,
                'interval_distances': all_interval_distances,
                'num_drones': num_drones
            }
            
        except Exception as e:
            log.error(f"Error processing animation data for {algorithm_name}: {e}")
            return {'positions': [], 'drone_positions': [], 'path_coordinates': [], 'metrics': [], 'interval_distances': [], 'num_drones': 0}
            
            # Determine number of drones from precomputed data
            num_drones = len(interval_positions[0]) if interval_positions and interval_positions[0] else 0
            
            total_time = time.time() - start_time
            log.info(f"Finished processing animation data for {algorithm_name} in {total_time:.2f} seconds.")
            
            return {
                'positions': all_positions,
                'drone_positions': all_drone_positions,
                'path_coordinates': all_path_coordinates,  # Natural path coordinates for rendering
                'metrics': all_metrics,
                'interval_distances': all_interval_distances,  # Store interval distances in km
                'num_drones': num_drones
            }
            
        except Exception as e:
            log.error(f"Error processing animation data for {algorithm_name}: {e}")
            return {'positions': [], 'drone_positions': [], 'path_coordinates': [], 'metrics': [], 'interval_distances': [], 'num_drones': 0}

    def _prepare_animation_data(self, paths, algorithm_name):
        """Prepare animation data using pre-computed metrics at intervals (legacy method for single-threaded usage)."""
        start_time = time.time()
        log.info(f"Preparing animation data for {algorithm_name} with {len(paths)} paths.")
        
        # Use the configured interval distance
        log.info(f"Pre-computing metrics every {self.interval_distance}m...")
        
        try:
            precomputed_data = self.path_evaluator.calculate_metrics_at_distance_intervals(
                paths, discount_factor=0.999, interval_distance=self.interval_distance
            )
            
            if not precomputed_data.get('interval_metrics'):
                log.warning(f"No precomputed metrics available for {algorithm_name}")
                return {'positions': [], 'drone_positions': [], 'path_coordinates': [], 'metrics': [], 'num_drones': 0}
            
            return self._process_precomputed_data(precomputed_data, algorithm_name)
            
        except Exception as e:
            log.error(f"Error in animation data preparation for {algorithm_name}: {e}")
            return {'positions': [], 'drone_positions': [], 'path_coordinates': [], 'metrics': [], 'interval_distances': [], 'num_drones': 0}

    def _create_comparative_frame(self, frame_idx, all_animation_data, algorithm_axes, 
                                  ax_area, ax_score, ax_victims, frames_per_interval):
        """Create a single comparative frame with progressive path building and drone positions."""
        try:
            # Calculate which metric interval this frame corresponds to
            interval_idx = frame_idx // frames_per_interval
            sub_frame_idx = frame_idx % frames_per_interval
            
            # Plot each algorithm
            for alg_name, ax in algorithm_axes.items():
                setup_algorithm_plot(ax, self.item, self.victims_gdf, self.crs, alg_name, self.algorithm_colors)
                
                if alg_name in all_animation_data:
                    animation_data = all_animation_data[alg_name]
                    
                    # Get actual drone positions for this interval
                    current_drone_positions = self._get_actual_frame_positions(animation_data, interval_idx)
                    
                    if current_drone_positions is not None:
                        # Plot cumulative paths up to current interval using existing function
                        plot_drone_paths(ax, animation_data, interval_idx, self.drone_colors)
                        plot_current_drone_positions(ax, current_drone_positions, self.drone_colors, self.path_evaluator.detection_radius)

            # Update time-series graphs for all sub-frames to avoid empty plots
            create_time_series_graphs(interval_idx, all_animation_data, ax_area, ax_score, ax_victims, 
                                    self.algorithm_colors, self.interval_distance / 1000.0)
            
        except Exception as e:
            log.warning(f"Error creating comparative frame {frame_idx}: {e}")

    def _get_actual_frame_positions(self, animation_data, frame_idx):
        """Get actual drone positions for the current frame from precomputed data."""
        try:
            drone_positions = animation_data.get('drone_positions', [])
            
            # Direct frame mapping - each frame corresponds to a metric interval
            if frame_idx < len(drone_positions):
                return drone_positions[frame_idx]
            elif drone_positions:
                # If we exceed the data, use the last available positions
                return drone_positions[-1]
            else:
                return None
            
        except Exception as e:
            log.warning(f"Error getting actual frame positions for frame {frame_idx}: {e}")
            return None



if __name__ == "__main__":
    log.info("=== Comparative Coverage Video Generation ===")
    
    # Configuration
    data_dir = "maigmo_dataset"
    output_dir = Path("coverage_videos")
    # Distance in meters between metric calculations (determines video granularity and performance)
    # Lower values = higher video quality but longer computation time
    # Recommended: 1000-2500m for good balance between quality and performance
    interval_distance = 250.0
    
    try:
        # Initialize evaluator
        evaluator = ComparativeEvaluator(
            dataset_directory=data_dir,
            evaluation_sizes=["medium"],
            num_drones=3,
            num_lost_persons=100,
            budget=1000000,
        )
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Output directory: {output_dir.absolute()}")
        
        # Load datasets
        log.info("Loading datasets...")
        evaluator.load_datasets()
        
        size = "medium"
        env_data = evaluator.environments[size]
        item = env_data["item"]
        victims_gdf = env_data["victims"]
        
        # Initialize path evaluator
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
        
        # Generate paths for target algorithms
        target_algorithms = ['Concentric', 'Pizza', 'Greedy', 'RandomWalk']
        algorithms_data = {}
        
        log.info("Generating paths for algorithms...")
        for name, generator in evaluator.path_generators.items():
            if name in target_algorithms:
                log.info(f"Generating paths for {name}...")
                try:
                    generated_paths = generator(
                        center_proj.x,
                        center_proj.y,
                        item.radius_km * 1000,
                        item.heatmap,
                        item.bounds,
                    )
                    if generated_paths:
                        algorithms_data[name] = generated_paths
                        log.info(f"Generated {len(generated_paths)} paths for {name}")
                    else:
                        log.warning(f"No paths generated for {name}")
                except Exception as e:
                    log.error(f"Error generating paths for {name}: {e}")
        
        if not algorithms_data:
            log.error("No algorithm data was generated successfully")
            exit(1)
        
        # Create video generator and configure it
        log.info("Creating video generator...")
        video_generator = ComparativeCoverageVideoGenerator(
            item, victims_gdf, path_evaluator, env_data["crs"], output_dir, 
            interval_distance=interval_distance
        )
        
        # Set evaluator configuration for serialization
        video_generator.set_evaluator_config({
            'fov_degrees': evaluator.path_generator_config.fov_degrees,
            'altitude_meters': evaluator.path_generator_config.altitude_meters,
            'meter_per_bin': evaluator.loader._meter_per_bin
        })
        
        log.info("Generating comparative coverage video...")
        video_generator.create_comparative_video(algorithms_data)
        
        log.info("=== Comparative Coverage Video Generation Complete ===")
        log.info(f"Video saved in: {output_dir.absolute()}")
        
    except Exception as e:
        log.error(f"Fatal error during video generation: {e}")
        raise
