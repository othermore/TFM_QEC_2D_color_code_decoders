import numpy as np
import argparse
from simulator import run_color_code_simulations
from plotter import plot_threshold, plot_execution_time, calculate_and_save_threshold, plot_threshold_zoom
import pandas as pd

def run_projection_evaluation(
    distances=[9, 11, 13, 15, 17, 19], 
    p_rates=None, 
    max_shots=25000000, 
    max_errors=400,
    output_file="s3_projection_results.csv",
    plot_only=False
):
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(project_root, "data", "chapter_6", output_file)
    
    """
    Executes the Projection Decoder simulation pipeline.
    
    Args:
        distances: List of distances to evaluate.
        p_rates: Array of physical error rates to evaluate.
        max_shots: Maximum shots per task.
        max_errors: Early stopping condition based on logical errors observed.
        output_file: Name of the output CSV file.
        plot_only: If True, skips simulation and only plots existing CSV data.
    """
    
    if p_rates is None:
        # We sweep from 0.02 to 20% to avoid extreme finite-size effects at very low noise
        p_rates = np.linspace(0.02, 0.20, 15)
        
    print("=====================================================")
    print("    Projection Decoder Simulation - 4.8.8 Color Code ")
    print("=====================================================")
        
    for noise_type in ['depolarizing', 'X']:
        # Adjust output file name based on noise type
        base_name, ext = os.path.splitext(output_file)
        current_output = f"{base_name}_{noise_type}{ext}" if noise_type != 'depolarizing' else output_file
        csv_path = os.path.join(project_root, "data", "chapter_6", current_output)
        
        if not plot_only:
            # Delete old CSV to force complete regeneration instead of appending
            if os.path.exists(csv_path):
                print(f"Deleting old results file {current_output} to regenerate completely...")
                os.remove(csv_path)

            print(f"\n--- Running for noise type: {noise_type.upper()} ---")
            
            csv_path = run_color_code_simulations(
                distances=distances,
                p_rates=p_rates,
                decoder_name='projection',
                output_file=current_output,
                rounds_func=lambda d: 1, # Code capacity (1 round)
                max_shots=max_shots,
                max_errors=max_errors,
                noise_type=noise_type
            )
            
            # Determine threshold to run a second pass (zoom)
            df = pd.read_csv(csv_path)
            est_threshold = calculate_and_save_threshold(csv_path, f'PROJECTION_{noise_type.upper()}', distances, df)
            
            if not np.isnan(est_threshold):
                print(f"\nRunning high-resolution zoom simulation around threshold: {est_threshold:.4f}...")
                zoom_p_rates = np.linspace(0.75 * est_threshold, 1.25 * est_threshold, 7)
                
                run_color_code_simulations(
                    distances=distances,
                    p_rates=zoom_p_rates,
                    decoder_name='projection',
                    output_file=current_output,
                    rounds_func=lambda d: 1,
                    max_shots=max_shots,
                    max_errors=max_errors * 4,
                    append=True,
                    noise_type=noise_type
                )
        else:
            if not os.path.exists(csv_path):
                print(f"Error: Could not find {csv_path} for plotting {noise_type}. Skipping...")
                continue
                
        title_suffix = "" if noise_type == 'depolarizing' else " - Pure X Noise"

        print(f"\nGenerating Threshold Plot for {noise_type}...")
        plot_threshold(csv_path, title=f"Logical noise vs physical noise - 4.8.8 Color Code (Projection{title_suffix})")
        
        print(f"Generating Execution Time Plot for {noise_type}...")
        plot_execution_time(csv_path, title=f"Decoding Complexity - Projection Decoder{title_suffix}")
        
        # Calculate threshold one last time for the zoom plot
        df = pd.read_csv(csv_path)
        est_threshold = calculate_and_save_threshold(csv_path, f'PROJECTION_{noise_type.upper()}', sorted(df['distance'].unique()), df)
        if not np.isnan(est_threshold):
            print(f"Generating Threshold Zoom Plot for {noise_type}...")
            plot_threshold_zoom(csv_path, est_threshold, title=f"Logical noise vs physical noise - 4.8.8 Color Code (Projection - Zoom{title_suffix})")
            
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Projection simulation or plot existing results.")
    parser.add_argument('--plot-only', action='store_true', help="Skip simulation and just regenerate plots from existing CSV.")
    args = parser.parse_args()

    run_projection_evaluation(
        distances=[9, 11, 13, 15, 17, 19],
        p_rates=np.linspace(0.02, 0.20, 15),
        max_shots=25000000,
        max_errors=400,
        plot_only=args.plot_only
    )
