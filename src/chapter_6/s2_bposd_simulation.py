import numpy as np
import argparse
from simulator import run_color_code_simulations
from plotter import plot_threshold, plot_execution_time

def run_bposd_evaluation(
    distances=[3, 5, 7, 9, 11], 
    p_rates=None, 
    max_shots=20000000, 
    max_errors=300,
    output_file="s2_bposd_results.csv",
    plot_only=False
):
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(project_root, "data", "chapter_6", output_file)
    
    """
    Executes the BP+OSD simulation pipeline.
    
    Args:
        distances: List of distances to evaluate.
        p_rates: Array of physical error rates to evaluate.
        max_shots: Maximum shots per task.
        max_errors: Early stopping condition based on logical errors observed.
        output_file: Name of the output CSV file.
        plot_only: If True, skips simulation and only plots existing CSV data.
    """
    
    if not plot_only:
        # Delete old CSV to force complete regeneration instead of appending
        if os.path.exists(csv_path):
            print(f"Deleting old results file {output_file} to regenerate completely...")
            os.remove(csv_path)

        if p_rates is None:
            # We sweep from 1% to 20% to ensure we cross the threshold (~5-10%)
            p_rates = np.linspace(0.01, 0.20, 15)
            
        print("=====================================================")
        print("    BP+OSD Decoder Simulation - 4.8.8 Color Code     ")
        print("=====================================================")
        
        csv_path = run_color_code_simulations(
            distances=distances,
            p_rates=p_rates,
            decoder_name='bposd',
            output_file=output_file,
            rounds_func=lambda d: 1, # Code capacity (1 round)
            # rounds_func=lambda d: d, # Phenomenological/Circuit level (d rounds)
            max_shots=max_shots,
            max_errors=max_errors
        )
    else:
        print("=====================================================")
        print("    Plotting Existing Results - 4.8.8 Color Code     ")
        print("=====================================================")
        if not os.path.exists(csv_path):
            print(f"Error: Could not find {csv_path} for plotting.")
            return None

    print("\nGenerating Threshold Plot...")
    plot_threshold(csv_path, title="Logical noise vs physical noise - 4.8.8 Color Code (BP+OSD)")
    
    print("Generating Execution Time Plot...")
    plot_execution_time(csv_path, title="Decoding Complexity - BP+OSD")
    
    return csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BP-OSD simulation or plot existing results.")
    parser.add_argument('--plot-only', action='store_true', help="Skip simulation and just regenerate plots from existing CSV.")
    args = parser.parse_args()

    run_bposd_evaluation(
        distances=[3, 5, 7, 9, 11],
        p_rates=np.linspace(0.01, 0.20, 15),
        max_shots=20000000,
        max_errors=300,
        plot_only=args.plot_only
    )
