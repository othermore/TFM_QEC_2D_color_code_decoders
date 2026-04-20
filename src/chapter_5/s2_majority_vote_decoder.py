import numpy as np
import matplotlib.pyplot as plt

def decode_single_shot(shot_syndrome, num_observables, detectors_per_observable, distance):
    """
    Decodes a single 1D array of detectors using majority vote.
    """
    shot_predictions = []
    
    # We iterate over each observable, which corresponds to a block of detectors
    # This would allow codes encoding multiple logical qubits
    for i in range(num_observables):
        # The number of detectors in our code, for a given observable
        # is a function of the code distance. 
        # We may receive additional detectors (frontier detectors) that we can ignore.
        # For the 3-qubit repetition code, we have 2 detectors per observable.
        # We need to iterate over the blocks of detectors for each observable
        start_index = i * detectors_per_observable
        end_index = start_index + (distance -1)
        local_syndrome = shot_syndrome[start_index:end_index]
        
        # For a repetition code, decoders are based on pairs of data qubits
        # that are chained, e.g. (q0, q1) and (q1, q2).
        # We achor to the first qubit, assume that it has not flipped (set to 0). 
        qubits_state = [0] 
        current_state = 0
        
        # Let's iterate over the detectors
        for detector_value in local_syndrome:
            # If a given detector is triggered, it indicates a flip
            # of this qubit relative to the previous one.
            if detector_value == 1:
                # We flip with respect to the prior qubit state
                current_state = 1 - current_state
            # Store the state of this qubit 
            qubits_state.append(current_state) 
        
        # If, as we assumed, q0 has not flipped, then our qubit_state tells us
        # which qubtis have flipped.
        # But if q1 has flipped, then our qubit_state is the opposite 
        # of the actual flip-state of each qubit.

        # If we have more 1s than 0s, that is a very unlikely scenario 
        # (2 bits flipped)so we can infer that q1 had actually flipped
        # (and our qubits_state is the opposite of the actual flip-state).
        num_ones = sum(qubits_state)
        num_zeros = len(qubits_state) - num_ones
        
        # We now need to make our flip-prediction for the observable.
        # Stim's circuit will contain the observable measurement in the last qubit
        # of the block, because it uses the last measurement.
        if num_ones > num_zeros:
            # If we have more 1s than 0s, we assume that our flip-states are inverted,
            # so we take the opposite of the last qubit's flip-state 
            # as our prediction for the observable.
            local_prediction = 1 - qubits_state[-1]
        else:
            # If we have more 0s than 1s, we assume our flip-states are correct.
            # So the flip-state of the last qubit is the right flip-state 
            # for the observable.
            local_prediction = qubits_state[-1]
            
        shot_predictions.append(local_prediction)
        
    return shot_predictions

def decode_batch_majority_vote(syndromes, circuit, distance):
    """
    Decodes a 2D matrix of syndromes by applying the single-shot decoder to each row.
    """
    model = circuit.detector_error_model()
    num_observables = model.num_observables
    detectors_per_observable = model.num_detectors // num_observables
    
    batch_size = syndromes.shape[0]
    all_predictions = []
    
    for shot_idx in range(batch_size):
        # Get one shot's syndrome
        shot_syndrome = syndromes[shot_idx]
        
        # Decode this shot
        shot_predictions = decode_single_shot(
            shot_syndrome, 
            num_observables, 
            detectors_per_observable,
            distance
        )
        
        all_predictions.append(shot_predictions)
        
    return np.array(all_predictions)