import os
import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit

import config

def get_number_of_qubits(d):
    """Calculates the number of physical qubits N for the 4.8.8 color code of distance d."""
    return (d**2 + 2*d - 1) / 2

def process_thresholds():
    """Generates consolidated dataset of thresholds."""
    zoom_path = os.path.join(config.DATA_CH6_DIR, 'thresholds_summary_zoom.csv')
    if not os.path.exists(zoom_path):
        raise FileNotFoundError(f"Missing {zoom_path}. Run Chapter 6 threshold analysis first.")
        
    df_zoom = pd.read_csv(zoom_path)
    
    rows = []
    for decoder in config.DECODERS:
        name_x = f"{decoder.upper()}_X"
        name_dep = f"{decoder.upper()}_DEPOLARIZING"
        
        row_x = df_zoom[df_zoom['decoder'] == name_x]
        row_dep = df_zoom[df_zoom['decoder'] == name_dep]
        
        th_x = row_x['threshold'].iloc[0] if not row_x.empty else np.nan
        th_dep = row_dep['threshold'].iloc[0] if not row_dep.empty else np.nan
        
        rows.append({
            'decoder': decoder,
            'threshold_x': th_x,
            'threshold_dep': th_dep
        })
        
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(config.DATA_CH7_DIR, 'analysis_1_thresholds.csv'), index=False)
    print("✓ Generated analysis_1_thresholds.csv")
    return df_out

def process_mitigation():
    """Calculates the mitigation capacity Lambda = p_phys / p_logical for target distances."""
    rows = []
    for decoder, prefix in config.DECODER_PREFIXES.items():
        path = os.path.join(config.DATA_CH6_DIR, f"{prefix}_results_X.csv")
        if not os.path.exists(path): continue
            
        df = pd.read_csv(path)
        for d in config.DISTANCES_MITIGATION:
            df_d = df[(df['distance'] == d) & (df['p_logical'] > 0)]
            for _, row in df_d.iterrows():
                p_phys = row['p_physical']
                p_log = row['p_logical']
                mitigation = p_phys / p_log
                rows.append({
                    'decoder': decoder,
                    'distance': d,
                    'p_physical': p_phys,
                    'p_logical': p_log,
                    'mitigation_ratio': mitigation
                })
                
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(config.DATA_CH7_DIR, 'analysis_2_mitigation.csv'), index=False)
    print("✓ Generated analysis_2_mitigation.csv")
    return df_out

def error_scaling_model(X, c0, c1, c2, c3):
    """
    2D model to extrapolate logical error as a function of distance d and physical noise p_phys.
    log_pL = c0 + c1*d + c2*log_p + c3*d*log_p
    """
    d, log_p = X
    return c0 + c1 * d + c2 * log_p + c3 * d * log_p

def process_error_scaling():
    """Performs the 2D regression p_L(d, p_phys) to extrapolate unexplored distances."""
    df_thresholds = pd.read_csv(os.path.join(config.DATA_CH7_DIR, 'analysis_1_thresholds.csv'))
    
    rows = []
    for decoder, prefix in config.DECODER_PREFIXES.items():
        path = os.path.join(config.DATA_CH6_DIR, f"{prefix}_results_X.csv")
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        
        th_val = df_thresholds[df_thresholds['decoder'] == decoder]['threshold_x'].iloc[0]
        max_p_phys = th_val * config.SUB_THRESHOLD_RATIO
        
        # Filter valid sub-threshold points
        df_sub = df[(df['p_physical'] <= max_p_phys) & (df['p_logical'] > 0)]
        if len(df_sub) < 5:
            print(f"[!] {decoder}: Muy pocos puntos sub-umbral para un buen ajuste.")
            continue
            
        d_vals = df_sub['distance'].values
        p_vals = df_sub['p_physical'].values
        log_pL_vals = np.log10(df_sub['p_logical'].values)
        log_p_vals = np.log10(p_vals)
        
        X_data = (d_vals, log_p_vals)
        
        # Curve fitting
        try:
            popt, pcov = curve_fit(error_scaling_model, X_data, log_pL_vals)
            
            # Calculate R^2
            residuals = log_pL_vals - error_scaling_model(X_data, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((log_pL_vals - np.mean(log_pL_vals))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            rows.append({
                'decoder': decoder,
                'c0': popt[0],
                'c1': popt[1],
                'c2': popt[2],
                'c3': popt[3],
                'r_squared': r_squared
            })
            
            if r_squared < config.MIN_R2_THRESHOLD:
                print(f"[WARNING] {decoder}: R^2 bajo ({r_squared:.4f}) en extrapolación de error.")
        except Exception as e:
            print(f"[ERROR] {decoder}: curve_fit failed - {e}")
                
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(config.DATA_CH7_DIR, 'analysis_3_error_scaling.csv'), index=False)
    print("✓ Generated analysis_3_error_scaling.csv")
    return df_out

def log_time_scaling_model(N, C, A, k):
    """Logarithmic model to directly optimize the residuals of the log-log plot."""
    return np.log10(C + A * (N ** k))

def process_time_scaling():
    """Performs log-log regression for time T = C + A * N^k"""
    rows = []
    for decoder, prefix in config.DECODER_PREFIXES.items():
        path = os.path.join(config.DATA_CH6_DIR, f"{prefix}_results_X.csv")
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        
        avg_times = df.groupby('distance')['avg_time_sec'].mean().reset_index()
        avg_times['qubits'] = avg_times['distance'].apply(get_number_of_qubits)
        
        if len(avg_times) < 4: continue
        
        N_vals = avg_times['qubits'].values
        T_vals = avg_times['avg_time_sec'].values
        log_T_vals = np.log10(T_vals)
        
        # Initial guess: C=min(T)/2, A=1e-6, k=1.0
        p0 = [min(T_vals) / 2.0, 1e-6, 1.0]
        
        try:
            # Bounds: C >= 1e-9, A > 0, k >= 0.5
            popt, pcov = curve_fit(log_time_scaling_model, N_vals, log_T_vals, p0=p0, bounds=([1e-9, 1e-12, 0.5], [max(T_vals), np.inf, 5.0]))
            C_opt, A_opt, k_opt = popt
            
            # Calculate R^2
            log_T_fit = log_time_scaling_model(N_vals, *popt)
            
            ss_res = np.sum((log_T_vals - log_T_fit)**2)
            ss_tot = np.sum((log_T_vals - np.mean(log_T_vals))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            rows.append({
                'decoder': decoder,
                'constant_c': C_opt,
                'multiplier_a': A_opt,
                'exponent_k': k_opt,
                'r_squared': r_squared
            })
            
            if r_squared < config.MIN_R2_THRESHOLD:
                print(f"[WARNING] {decoder}: R^2 bajo ({r_squared:.4f}) in time extrapolation.")
        except Exception as e:
            print(f"[ERROR] {decoder}: curve_fit for time failed - {e}")
        
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(config.DATA_CH7_DIR, 'analysis_4_time_scaling.csv'), index=False)
    print("✓ Generated analysis_4_time_scaling.csv")
    return df_out

def process_extrapolations():
    """Uses the calculated coefficients to extrapolate required distances and times."""
    df_error = pd.read_csv(os.path.join(config.DATA_CH7_DIR, 'analysis_3_error_scaling.csv'))
    df_time = pd.read_csv(os.path.join(config.DATA_CH7_DIR, 'analysis_4_time_scaling.csv'))
    
    rows = []
    for p_target in config.P_LOGICAL_TARGETS:
        log_p_target = np.log10(p_target)
        for p_phys in config.P_PHYS_TARGETS:
            log_p_phys = np.log10(p_phys)
            for decoder in config.DECODERS:
                row_error = df_error[df_error['decoder'] == decoder]
                row_time = df_time[df_time['decoder'] == decoder]
                
                if row_error.empty or row_time.empty: continue
                
                c0 = row_error['c0'].iloc[0]
                c1 = row_error['c1'].iloc[0]
                c2 = row_error['c2'].iloc[0]
                c3 = row_error['c3'].iloc[0]
                
                # We want to solve: log_p_target = c0 + c1*d + c2*log_p_phys + c3*d*log_p_phys
                # log_p_target - c0 - c2*log_p_phys = d * (c1 + c3*log_p_phys)
                numerator = log_p_target - c0 - c2 * log_p_phys
                denominator = c1 + c3 * log_p_phys
                
                if denominator == 0: continue
                
                d_exact = numerator / denominator
                
                # The minimum distance is the next upper odd integer
                d_req = int(np.ceil(d_exact))
                if d_req % 2 == 0: d_req += 1
                
                # Exception: if d_req <= 0, it means p_phys is already below p_target or something failed
                if d_req < 3: d_req = 3
                
                # Extrapolate time: T = C + A * N^k
                n_req = get_number_of_qubits(d_req)
                k = row_time['exponent_k'].iloc[0]
                A = row_time['multiplier_a'].iloc[0]
                C_const = row_time['constant_c'].iloc[0]
                
                t_req = C_const + A * (n_req**k)
                
                rows.append({
                    'decoder': decoder,
                    'target_p_logical': p_target,
                    'fixed_p_physical': p_phys,
                    'required_distance': d_req,
                    'required_qubits': n_req,
                    'estimated_time_sec_per_shot': t_req
                })
                
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(config.DATA_CH7_DIR, 'analysis_5_6_extrapolations.csv'), index=False)
    print("✓ Generated analysis_5_6_extrapolations.csv")
    return df_out

def generate_latex_tables():
    """Generates LaTeX tables for the fitted mathematical models."""
    
    #Thresholds
    df_thresh = pd.read_csv(os.path.join(config.DATA_CH7_DIR, 'analysis_1_thresholds.csv'))
    
    latex_1 = "\\begin{table}[hbt!]\n\\centering\n"
    latex_1 += "\\begin{tabular}{l c c}\n\\toprule\n"
    latex_1 += "Decoder & Bit-flip Threshold ($p_{th}^X$) & Depolarizing Threshold ($p_{th}^{dep}$) \\\\\n\\midrule\n"
    for _, row in df_thresh.iterrows():
        dec_label = config.DECODER_LABELS.get(row['decoder'], row['decoder'])
        th_x = row['threshold_x']
        th_dep = row['threshold_dep']
        latex_1 += f"{dec_label} & {th_x:.4f} & {th_dep:.4f} \\\\\n"
    latex_1 += "\\bottomrule\n\\end{tabular}\n"
    latex_1 += "\\caption{Code capacity thresholds for X-type and depolarizing noise.}\n"
    latex_1 += "\\label{tab:thresholds}\n\\end{table}\n"
    
    out_1 = os.path.join(config.DATA_CH7_DIR, 'analysis_1_thresholds_table.tex')
    with open(out_1, 'w') as f: f.write(latex_1)
    print(f"✓ Generado {out_1}")
    
    #Mitigation
    df_mit = pd.read_csv(os.path.join(config.DATA_CH7_DIR, 'analysis_2_mitigation.csv'))
    
    latex_2 = "\\begin{table}[hbt!]\n\\centering\n"
    latex_2 += "\\begin{tabular}{l c c c}\n\\toprule\n"
    latex_2 += "Decoder & Distance ($d$) & $p_{phys}$ & Mitigation Ratio $\\Lambda$ \\\\\n\\midrule\n"
    for _, row in df_mit.iterrows():
        p_phys = row['p_physical']
        if not np.isclose(p_phys, 0.02, atol=1e-5):
            continue
        dec_label = config.DECODER_LABELS.get(row['decoder'], row['decoder'])
        d = int(row['distance'])
        ratio = row['mitigation_ratio']
        latex_2 += f"{dec_label} & {d} & \\num{{{p_phys:.2e}}} & \\num{{{ratio:.2e}}} \\\\\n"
    latex_2 += "\\bottomrule\n\\end{tabular}\n"
    latex_2 += "\\caption{Noise mitigation capacity ($\\Lambda = p_{phys} / p_L$) at a fixed physical error rate of $p=0.02$ (2\\%).}\n"
    latex_2 += "\\label{tab:mitigation}\n\\end{table}\n"
    
    out_2 = os.path.join(config.DATA_CH7_DIR, 'analysis_2_mitigation_table.tex')
    with open(out_2, 'w') as f: f.write(latex_2)
    print(f"✓ Generado {out_2}")

    #Error scaling
    df_error = pd.read_csv(os.path.join(config.DATA_CH7_DIR, 'analysis_3_error_scaling.csv'))
    
    latex_3 = "\\begin{table}[hbt!]\n\\centering\n"
    latex_3 += "\\begin{tabular}{l c c c c c}\n\\toprule\n"
    latex_3 += "Decoder & $c_0$ & $c_1$ & $c_2$ & $c_3$ & $R^2$ \\\\\n\\midrule\n"
    for _, row in df_error.iterrows():
        dec_label = config.DECODER_LABELS.get(row['decoder'], row['decoder'])
        c0, c1, c2, c3, r2 = row['c0'], row['c1'], row['c2'], row['c3'], row['r_squared']
        latex_3 += f"{dec_label} & {c0:.4f} & {c1:.4f} & {c2:.4f} & {c3:.4f} & {r2:.4f} \\\\\n"
    latex_3 += "\\bottomrule\n\\end{tabular}\n"
    latex_3 += "\\caption{Fitted parameters for the 2D sub-threshold error scaling model. Equation: $\\log_{10} p_L = c_0 + c_1 d + c_2 \\log_{10} p + c_3 d \\log_{10} p$}\n"
    latex_3 += "\\label{tab:error_scaling_fit}\n\\end{table}\n"
    
    out_3 = os.path.join(config.DATA_CH7_DIR, 'analysis_3_error_scaling_table.tex')
    with open(out_3, 'w') as f: f.write(latex_3)
    print(f"✓ Generado {out_3}")
    
    #Time scaling
    df_time = pd.read_csv(os.path.join(config.DATA_CH7_DIR, 'analysis_4_time_scaling.csv'))
    
    latex_4 = "\\begin{table}[hbt!]\n\\centering\n"
    latex_4 += "\\begin{tabular}{l c c c c}\n\\toprule\n"
    latex_4 += "Decoder & $C$ (s) & $A$ & $k$ & $R^2$ \\\\\n\\midrule\n"
    for _, row in df_time.iterrows():
        dec_label = config.DECODER_LABELS.get(row['decoder'], row['decoder'])
        c, a, k, r2 = row['constant_c'], row['multiplier_a'], row['exponent_k'], row['r_squared']
        latex_4 += f"{dec_label} & \\num{{{c:.2e}}} & \\num{{{a:.2e}}} & {k:.4f} & {r2:.4f} \\\\\n"
    latex_4 += "\\bottomrule\n\\end{tabular}\n"
    latex_4 += "\\caption{Fitted parameters for the empirical time complexity. Model: $T(N) = C + A \\cdot N^k$}\n"
    latex_4 += "\\label{tab:time_scaling_fit}\n\\end{table}\n"
    
    out_4 = os.path.join(config.DATA_CH7_DIR, 'analysis_4_time_scaling_table.tex')
    with open(out_4, 'w') as f: f.write(latex_4)
    print(f"✓ Generado {out_4}")

def run_all_processing():
    print("Iniciando procesamiento de datos - Capítulo 7...")
    process_thresholds()
    process_mitigation()
    process_error_scaling()
    process_time_scaling()
    process_extrapolations()
    generate_latex_tables()
    print("Procesamiento completado con éxito.")

if __name__ == '__main__':
    run_all_processing()
