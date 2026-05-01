import matplotlib.pyplot as plt
import numpy as np
from s4_mwpm_noise import sample_repetition_code_mwpm

def plot_threshold_theorem_comparison(distances: list = [3, 5, 7, 9], shots: int = 10000) -> None:
    """
    Simulates and plots a repetition code under phenomenological noise
    over a range of physical error rates to visualize the Threshold Theorem.
    (Measurement rounds = distance)
    """
    p_rates = np.linspace(0.01, 0.4, 20)
    
    plt.figure(figsize=(8, 6))
    for d in distances:
        _, l_rates = sample_repetition_code_mwpm(distance=d, shots=shots, p_rates=p_rates, noise_model=(1, 0.05, 0.05), rounds=d)
        plt.plot(p_rates, l_rates, marker='o', label=f'Distance {d}')

    plt.plot(p_rates, p_rates, linestyle='--', color='gray', label='No QEC')
    plt.title('Repetition code: Circuit-Level Noise $(1, 0.05, 0.05)$, MWPM Decoder')
    plt.xlabel('Physical Error Rate (p)')
    plt.ylabel('Logical Error Rate ($p_L$)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

def run_threshold_simulations(shots: int = 10000):
    """Executes the threshold theorem simulation and prints a text summary."""
    print("Executing Threshold Theorem simulations...")
    distances = [3, 5, 7, 9]
    p_rates = np.linspace(0.01, 0.4, 20)
    
    print("\n--- Simulation: Threshold Theorem (Phenomenological Noise) ---")
    for d in distances:
        _, l_rates = sample_repetition_code_mwpm(distance=d, shots=shots, p_rates=p_rates, noise_model=(1, 0.05, 0.05), rounds=d)
        mean_pl = np.mean(l_rates)
        print(f"Distance {d:2} | Mean pL: {mean_pl:.4f} | Max pL: {max(l_rates):.4f}")
        
    print("\nSimulations complete.")

if __name__ == "__main__":
    run_threshold_simulations(shots=10000)
