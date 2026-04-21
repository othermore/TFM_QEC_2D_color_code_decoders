import matplotlib.pyplot as plt
from s4_mwpm_noise import sample_repetition_code_mwpm
from s4_majority_vote_noise import sample_repetition_code_mv

def plot_repetition_code_comparison(distances: list = [3, 5, 7], shots: int = 10000) -> None:
    """
    Simulates and plots a repetition code over a range of physical error rates,
    and compares the performance of the majority vote decoder with the MWPM decoder.
    """
    # Plotting the results
    plt.figure(figsize=(8, 6))
    for distance in distances:
        physical_error_rates_mv, logical_error_rates_mv = sample_repetition_code_mv(distance, shots)
        physical_error_rates_mwpm, logical_error_rates_mwpm = sample_repetition_code_mwpm(distance, shots)

        plt.plot(physical_error_rates_mv, logical_error_rates_mv, marker='o', label=f'Majority Vote - Distance {distance}')
        plt.plot(physical_error_rates_mwpm, logical_error_rates_mwpm, marker='s', label=f'MWPM - Distance {distance}')

    # Reference line to show the break-even point where QEC stops helping
    plt.plot(physical_error_rates_mv, physical_error_rates_mv, linestyle='--', color='gray', label='No QEC')
    
    plt.title('Repetition Code Performance at different distances')
    plt.xlabel('Physical Error Rate (p)')
    plt.ylabel('Logical Error Rate ($p_L$)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

# Execute to input the main parameters
if __name__ == "__main__":
    plot_repetition_code_comparison(distances=[3, 5, 7], shots=10000)