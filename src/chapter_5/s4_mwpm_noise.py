import stim
import pymatching
import numpy as np

def sample_repetition_code_mwpm(distance: int = 3, shots: int = 10000, p_rates: np.ndarray = None, noise_model: tuple = (1, 1, 1), rounds: int = 1):
    """
    Simulates a repetition code over a range of physical error rates,
    using the pymatching MWPM decoder.
    """
    if p_rates is None:
        p_rates = np.linspace(0.01, 0.15, 10)
    logical_error_rates = []

    data_r, clifford_r, meas_r = noise_model

    for p in p_rates:
        # Define the circuit
        circuit = stim.Circuit.generated(
            "repetition_code:memory",
            distance=distance,
            rounds=rounds,
            before_round_data_depolarization=p * data_r,
            after_clifford_depolarization=p * clifford_r,
            before_measure_flip_probability=p * meas_r
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

    return p_rates, logical_error_rates