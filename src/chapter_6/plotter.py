import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
from scipy.interpolate import interp1d

def calculate_and_save_threshold(csv_path: str, decoder_name: str, distances: list, df: pd.DataFrame) -> float:
    """
    Estimates the threshold by finding the intersection of p_L curves for different distances.
    Saves the result to data/chapter_6/thresholds_summary.csv.
    """
    intersections = []
    
    # We compare adjacent distances to find crossing points
    for i in range(len(distances) - 1):
        d1, d2 = distances[i], distances[i+1]
        df1 = df[df['distance'] == d1].sort_values('p_physical')
        df2 = df[df['distance'] == d2].sort_values('p_physical')
        
        # Only use points where we have valid probabilities > 0
        df1_valid = df1[df1['p_logical'] > 0]
        df2_valid = df2[df2['p_logical'] > 0]
        
        if len(df1_valid) > 1 and len(df2_valid) > 1:
            # Interpolate log(p_L) vs log(p)
            # f1 and f2 will be two linear functions that interpolate the logical 
            # error rate for a given physical error rate
            # We use log because that makes the functions be approximately linear
            f1 = interp1d(np.log10(df1_valid['p_physical']), np.log10(df1_valid['p_logical']), kind='linear', fill_value="extrapolate")
            f2 = interp1d(np.log10(df2_valid['p_physical']), np.log10(df2_valid['p_logical']), kind='linear', fill_value="extrapolate")
            
            # Find x where f1 - f2 = 0
            # This is the point where the two functions cross
            # We can use brentq solver for this
            p_range = np.log10(df1_valid['p_physical'])
            from scipy.optimize import brentq
            try:
                cross_log = brentq(lambda x: f1(x) - f2(x), p_range.min(), p_range.max())
                # convert back to linear scale
                intersections.append(10**cross_log)
            except ValueError:
                pass # No crossing found in this interval
                
    threshold = np.mean(intersections) if intersections else np.nan
    
    # Save to global thresholds file
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    summary_path = os.path.join(project_root, "data", "chapter_6", "thresholds_summary.csv")
    
    file_exists = os.path.isfile(summary_path)
    
    # If the file exists, we might want to update or append. 
    # For simplicity, we read existing, update the decoder's row, and write back.
    if file_exists:
        summary_df = pd.read_csv(summary_path)
        if decoder_name in summary_df['decoder'].values:
            summary_df.loc[summary_df['decoder'] == decoder_name, 'threshold'] = threshold
        else:
            new_row = pd.DataFrame({'decoder': [decoder_name], 'threshold': [threshold]})
            summary_df = pd.concat([summary_df, new_row], ignore_index=True)
    else:
        summary_df = pd.DataFrame({'decoder': [decoder_name], 'threshold': [threshold]})
        
    summary_df.to_csv(summary_path, index=False)
    
    print(f"--> Estimated Threshold for {decoder_name}: {threshold:.4f} (Saved to {summary_path})")
    return threshold


def plot_threshold(csv_path: str, title: str = None) -> None:
    """
    Reads the standardized CSV and plots Logical Error Rate (p_L) vs Physical Error Rate (p).
    """
    df = pd.read_csv(csv_path)
    
    decoder_name = df['decoder'].iloc[0].upper()
    if title is None:
        title = f'Threshold Theorem - Color Code 4.8.8 - {decoder_name}'
        
    distances = sorted(df['distance'].unique())
    
    # Calculate threshold programmatically
    est_threshold = calculate_and_save_threshold(csv_path, decoder_name, distances, df)
    
    plt.figure(figsize=(9, 6))
    
    # We use a colormap to automatically color code by distance
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(distances)))
    
    for idx, d in enumerate(distances):
        d_df = df[df['distance'] == d].sort_values('p_physical')
        
        valid_indices = []
        p_l = d_df['p_logical'].values
        for i in range(len(p_l)):
             valid_indices.append(d_df.index[i])
                
        d_df = d_df.loc[valid_indices]
        
        plt.errorbar(
            d_df['p_physical'], 
            d_df['p_logical'], 
            yerr=d_df['p_logical_err'], 
            marker='o', 
            linestyle='-', 
            linewidth=0.8,
            markersize=1.6,
            capsize=2,
            color=colors[idx],
            label=f'Distance $d={d}$'
        )

    if not np.isnan(est_threshold):
        plt.axvline(x=est_threshold, color='red', linestyle=':', alpha=0.8, label=f'Est. Threshold ($p_{{th}} \\approx {est_threshold:.4f}$)')

    # Add break-even line (p_L = p)
    min_p = df['p_physical'].min()
    max_p = df['p_physical'].max()
    plt.plot([min_p, max_p], [min_p, max_p], color='gray', linestyle='--', alpha=0.6, label='Break-even ($p_L = p$)')

    # Log scale
    plt.yscale('log')
    plt.xscale('log')
    
    import matplotlib.ticker as ticker
    ax = plt.gca()
    
    # For minor ticks on the X axis, only show round/alternate values to avoid overlapping
    def x_minor_formatter(y, _):
        valid_ticks = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]
        if any(abs(y - val) < 1e-5 for val in valid_ticks):
            return '{:g}'.format(y)
        return ''
        
    ax.xaxis.set_minor_formatter(ticker.FuncFormatter(x_minor_formatter))
    ax.tick_params(axis='x', which='minor', labelsize=8, rotation=45)
    
    # For minor ticks on the Y axis, only label 2 and 5.
    def y_minor_formatter(y, _):
        if y == 0:
            return ''
        base = y / (10 ** np.floor(np.log10(y)))
        if abs(base - 2) < 0.1 or abs(base - 5) < 0.1:
            return '{:g}'.format(y)
        return ''
        
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(y_minor_formatter))
    ax.tick_params(axis='y', which='minor', labelsize=8)
    
    plt.xlabel('Physical Error Rate ($p$)')
    plt.ylabel('Logical Error Rate ($p_L$)')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls=":", alpha=0.6)
    
    # Save the plot in the figures directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    figures_dir = os.path.join(project_root, "figures", "chapter_6")
    os.makedirs(figures_dir, exist_ok=True)
    
    filename = os.path.basename(csv_path).replace('.csv', '_threshold.png')
    save_path = os.path.join(figures_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {save_path}")
    
    plt.close()
    
    try:
        subprocess.Popen(['open', save_path])
    except Exception as e:
        print(f"Could not open image: {e}")

def plot_execution_time(csv_path: str, title: str = None) -> None:
    """
    Reads the standardized CSV and plots Average Execution Time vs Code Distance.
    """
    df = pd.read_csv(csv_path)
    
    decoder_name = df['decoder'].iloc[0].upper()
    if title is None:
        title = f'Decoding Complexity - {decoder_name}'
        
    distances = sorted(df['distance'].unique())
    avg_times = []
    
    for d in distances:
        # We average the time across all physical error rates for a given distance
        d_df = df[df['distance'] == d]
        avg_time = d_df['avg_time_sec'].mean()
        avg_times.append(avg_time)
        
    plt.figure(figsize=(8, 5))
    
    plt.plot(distances, avg_times, marker='s', linestyle='-', color='purple', linewidth=2, markersize=8)
    
    plt.xlabel('Code Distance ($d$)')
    plt.ylabel('Average Decoding Time per Shot (seconds)')
    plt.title(title)
    plt.xticks(distances)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot in the figures directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    figures_dir = os.path.join(project_root, "figures", "chapter_6")
    os.makedirs(figures_dir, exist_ok=True)
    
    filename = os.path.basename(csv_path).replace('.csv', '_time.png')
    save_path = os.path.join(figures_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {save_path}")
    
    plt.close()

    try:
        subprocess.Popen(['open', save_path])
    except Exception as e:
        print(f"Could not open image: {e}")

if __name__ == "__main__":
    pass
