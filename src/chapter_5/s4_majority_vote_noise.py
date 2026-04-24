import stim
import numpy as np
from s2_majority_vote_decoder import decode_batch_majority_vote

def sample_repetition_code_mv(distance: int = 3, shots: int = 10000) -> None:
    """
    Simulates a repetition code over a range of physical error rates,
    decodes using PyMatching, and plots the logical error rate.
    """
    physical_error_rates = np.linspace(0.01, 0.15, 10)
    logical_error_rates = []

    for p in physical_error_rates:
        # Define the circuit
        #BEGIN S4_SAMPLE
        circuit = stim.Circuit.generated(
            "repetition_code:memory",
            distance=distance,
            rounds=1,
            before_round_data_depolarization=p,
            after_clifford_depolarization=p,
            before_measure_flip_probability=p
        )
        #END S4_SAMPLE
        # Sample the circuit to get syndromes and actual observables
        sampler = circuit.compile_detector_sampler()
        syndromes, actual_observables = sampler.sample(shots, separate_observables=True)
        # Decode the syndromes using our decoder
        predicted_observables = decode_batch_majority_vote(syndromes, circuit, distance)
        
        # Calculate the logical error rate as the cases where the 
        # actual observables differ from the predicted ones
        num_errors = np.sum(predicted_observables != actual_observables)
        logical_error_rates.append(num_errors / shots)
    return physical_error_rates, logical_error_rates