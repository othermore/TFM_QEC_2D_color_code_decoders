import stim
import pymatching
import numpy as np
import matplotlib.pyplot as plt

def plot_repetition_code_performance(distance: int = 3, shots: int = 10000) -> None:
    """
    Simulates a repetition code over a range of physical error rates,
    decodes using PyMatching, and plots the logical error rate.
    """
    physical_error_rates = np.linspace(0.01, 0.15, 10)
    logical_error_rates = []

    for p in physical_error_rates:
        # Define the circuit
        circuit = stim.Circuit.generated(
            "repetition_code:memory",
            distance=distance,
            rounds=1,
            before_round_data_depolarization=p
        )
        
        # Sample the circuit to get syndromes and actual observables
        sampler = circuit.compile_detector_sampler()
        syndromes, actual_observables = sampler.sample(shots, separate_observables=True)
        
        # Decode the syndromes using PyMatching
        matcher = pymatching.Matching.from_stim_circuit(circuit)
        predicted_observables = matcher.decode_batch(syndromes)
        
        # Calculate the logical error rate as the cases where the 
        # actual observables differ from the predicted ones
        num_errors = np.sum(predicted_observables != actual_observables)
        logical_error_rates.append(num_errors / shots)

    # Plotting the results
    plt.figure(figsize=(8, 6))
    plt.plot(physical_error_rates, logical_error_rates, marker='o', label=f'Distance {distance}')
    
    # Reference line to show the break-even point where QEC stops helping
    plt.plot(physical_error_rates, physical_error_rates, linestyle='--', color='gray', label='No QEC')
    
    plt.title('Repetition Code Performance (Code-Capacity)')
    plt.xlabel('Physical Error Rate (p)')
    plt.ylabel('Logical Error Rate ($p_L$)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

# Execute to input the main parameters
plot_repetition_code_performance(distance=3, shots=10000)