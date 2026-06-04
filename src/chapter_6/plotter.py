import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
from scipy.interpolate import interp1d
import itertools

def calculate_and_save_threshold(csv_path: str, decoder_name: str, distances: list, df: pd.DataFrame) -> float:
    """
    Estimates the threshold by finding the intersection of p_L curves for different distances.
    Saves the result to data/chapter_6/thresholds_summary.csv.
    """
    intersections = []
    
    # We compare all pairs of distances (e.g., 3vs5, 3vs7, 5vs7...) to find crossing points
    for d1, d2 in itertools.combinations(distances, 2):
        df1 = df[df['distance'] == d1].sort_values('p_physical')
        df2 = df[df['distance'] == d2].sort_values('p_physical')
        
        # Only use points where we have valid probabilities > 0
        df1_valid = df1[df1['p_logical'] > 0]
        df2_valid = df2[df2['p_logical'] > 0]
        
        if len(df1_valid) > 1 and len(df2_valid) > 1:
            
            # Find the physical error rates that were simulated and are valid for BOTH distances
            shared_p = set(df1_valid['p_physical']).intersection(set(df2_valid['p_physical']))
            shared_p = sorted(list(shared_p))
            
            # We need at least two shared points to define an interval for crossing
            if len(shared_p) < 2:
                continue
                
            # Logaritmic scale applied to the shared X points
            x_nodes = np.log10(shared_p)
            
            # Logical error rates for these nodes
            pL1_vals = [df1_valid[df1_valid['p_physical'] == p]['p_logical'].iloc[0] for p in shared_p]
            pL2_vals = [df2_valid[df2_valid['p_physical'] == p]['p_logical'].iloc[0] for p in shared_p]
            
            # Getting logical error in log scale,
            # calculate the difference y = log(pL1) - log(pL2)
            y_diff = np.log10(pL1_vals) - np.log10(pL2_vals)
            
            # Iterate through each interval [x0, x1] to find crossings
            for i in range(len(x_nodes) - 1):
                x0, x1 = x_nodes[i], x_nodes[i+1]
                y0, y1 = y_diff[i], y_diff[i+1]
                
                # A crossing occurs if the difference changes sign
                # If the diff between the functions is exactly 0, we skip to avoid 
                # division by 0
                if y0 * y1 <= 0 and (y0 != 0 or y1 != 0):
                    # Filter by physical direction to avoid multiple false crossing due to noise
                    # We expect the larger code (d2) to perform better at low noise (left)
                    # and worse at high noise (right).
                    # Better means lower error: f2 < f1 -> y0 = f1 - f2 > 0
                    # Worse means higher error: f2 > f1 -> y1 = f1 - f2 < 0
                    # If the crossing is in the wrong direction we do not add it
                    if y0 > 0 and y1 <= 0:
                        # Use linear interpolation within the interval to find the exact crossing point
                        cross_log = x0 - y0 * (x1 - x0) / (y1 - y0)
                        
                        # Convert back from log scale to linear scale and store the intersection
                        intersections.append(10**cross_log)
                        
                        # We only take the first valid crossing (lowest noise). If curves 
                        # cross multiple times due to noise when they get closer
                        # to the maximum theoretical noise, we want to ignore these
                        # Crosses are more precise at lower noise levels.
                        break
                        
    if not intersections:
        threshold = np.nan
    elif len(intersections) > 2:
        # Use Kernel Density Estimation to find the continuous mode (the highest density of crossings)
        from scipy.stats import gaussian_kde
        
        # If all intersections are identical (variance is 0), gaussian_kde fails. We catch that case.
        if np.var(intersections) == 0:
            threshold = intersections[0]
        else:
            kde = gaussian_kde(intersections)
            # Evaluate the PDF over a fine grid between the min and max crossing points
            grid = np.linspace(min(intersections), max(intersections), 1000)
            # The threshold is the point with the highest density (the mode)
            threshold = grid[np.argmax(kde(grid))]
    else:
        # Fallback to mean if we have 1 or 2 crossings
        threshold = np.mean(intersections)
    
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


def plot_threshold(csv_path: str, title: str = None, distances_filter: list = None) -> None:
    """
    Reads the standardized CSV and plots Logical Error Rate (p_L) vs Physical Error Rate (p).
    """
    df = pd.read_csv(csv_path)
    
    if distances_filter is not None:
        df = df[df['distance'].isin(distances_filter)]
        
    decoder_name = df['decoder'].iloc[0].upper()
    if '_X.csv' in csv_path or '_X_zoom.csv' in csv_path:
        save_decoder_name = decoder_name + '_X'
    else:
        save_decoder_name = decoder_name + '_DEPOLARIZING'
        
    if title is None:
        title = f'Threshold Theorem - Color Code 4.8.8 - {decoder_name}'
        
    distances = sorted(df['distance'].unique())
    
    # Calculate threshold programmatically
    est_threshold = calculate_and_save_threshold(csv_path, save_decoder_name, distances, df)
    
    plt.figure(figsize=(9, 6))
    
    # We use a colormap to automatically color code by distance
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(distances)))
    
    for idx, d in enumerate(distances):
        d_df = df[df['distance'] == d].sort_values('p_physical')
        
        # Filter out points with 0 logical errors
        d_df = d_df[d_df['p_logical'] > 0]
        if d_df.empty:
            continue
        
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

def plot_threshold_zoom(csv_path: str, threshold: float, title: str = None, distances_filter: list = None) -> None:
    """
    Plots the Logical Error Rate (p_L) vs Physical Error Rate (p) in a linear scale,
    zoomed in around the given threshold [0.75 * threshold, 1.25 * threshold].
    """
    df = pd.read_csv(csv_path)
    
    zoom_csv_path = csv_path.replace('.csv', '_zoom.csv')
    if os.path.exists(zoom_csv_path):
        df_zoom = pd.read_csv(zoom_csv_path)
        df = pd.concat([df, df_zoom], ignore_index=True)
        
    if distances_filter is not None:
        df = df[df['distance'].isin(distances_filter)]
    
    if df.empty:
        print("No data available to plot zoom after filtering.")
        return
        
    decoder_name = df['decoder'].iloc[0].upper()
    if title is None:
        title = f'Threshold Zoom - Color Code 4.8.8 - {decoder_name}'
        
    distances = sorted(df['distance'].unique())
    
    plt.figure(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(distances)))
    
    x_min, x_max = 0.75 * threshold, 1.25 * threshold
    
    # Filter global limits for the Y axis based on the zoomed range
    y_vals_in_range = df[(df['p_physical'] >= x_min) & (df['p_physical'] <= x_max)]['p_logical']
    y_min, y_max = 0, y_vals_in_range.max() * 1.1 if not y_vals_in_range.empty else 1.0
    
    for idx, d in enumerate(distances):
        d_df = df[df['distance'] == d].sort_values('p_physical')
        
        # Filter out points with 0 logical errors
        d_df = d_df[d_df['p_logical'] > 0]
        
        # Only plot points roughly within the zoom window to avoid long line stretches outside
        d_df_zoom = d_df[(d_df['p_physical'] >= x_min * 0.9) & (d_df['p_physical'] <= x_max * 1.1)]
        if d_df_zoom.empty: continue
        
        plt.errorbar(
            d_df_zoom['p_physical'], 
            d_df_zoom['p_logical'], 
            yerr=d_df_zoom['p_logical_err'], 
            marker='o', 
            linestyle='-', 
            linewidth=1.2,
            markersize=4,
            capsize=3,
            color=colors[idx],
            label=f'Distance $d={d}$'
        )

    plt.axvline(x=threshold, color='red', linestyle=':', alpha=0.8, label=f'Est. Threshold ($p_{{th}} \\approx {threshold:.4f}$)')

    # Break-even line
    plt.plot([x_min, x_max], [x_min, x_max], color='gray', linestyle='--', alpha=0.6, label='Break-even ($p_L = p$)')

    plt.xscale('linear')
    plt.yscale('linear')
    
    plt.xlim(x_min, x_max)
    plt.ylim(0, y_max)
    
    plt.xlabel('Physical Error Rate ($p$)')
    plt.ylabel('Logical Error Rate ($p_L$)')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    figures_dir = os.path.join(project_root, "figures", "chapter_6")
    os.makedirs(figures_dir, exist_ok=True)
    
    filename = os.path.basename(csv_path).replace('.csv', '_threshold_zoom.png')
    save_path = os.path.join(figures_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Zoom plot saved to {save_path}")
    
    plt.close()
    
    try:
        subprocess.Popen(['open', save_path])
    except Exception as e:
        print(f"Could not open image: {e}")

def plot_execution_time(csv_path: str, title: str = None, distances_filter: list = None) -> None:
    """
    Reads the standardized CSV and plots Average Execution Time vs Code Distance.
    """
    df = pd.read_csv(csv_path)
    
    if distances_filter is not None:
        df = df[df['distance'].isin(distances_filter)]
        
    if df.empty:
        return
        
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
