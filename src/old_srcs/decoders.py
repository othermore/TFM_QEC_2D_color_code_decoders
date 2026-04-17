import numpy as np
import pymatching
import stim

class BaseDecoder:
    """
    Base interface for all quantum error correction decoders.
    """
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decodes a given syndrome to predict logical observable errors.

        Args:
            syndrome (np.ndarray): A 1D boolean numpy array representing the detection events 
                for a single simulation shot.

        Returns:
            np.ndarray: A 1D boolean numpy array containing the predicted logical observable flips.
        """
        raise NotImplementedError("Subclasses must implement the decode method.")
    
    @classmethod
    def name(cls) -> str:
        """
        Returns the display name of the decoder.

        Args:
            None

        Returns:
            str: The name of the decoder class.
        """
        return cls.__name__

    
class MWPMLibraryDecoder(BaseDecoder):
    """
    Wrapper for the PyMatching library using Minimum Weight Perfect Matching.
    """

    @classmethod
    def name(cls) -> str:
        """
        Returns the display name of the MWPM decoder.

        Args:
            None

        Returns:
            str: The custom name "MWPM (PyMatching Library)".
        """
        return "MWPM (PyMatching Library)"

    def __init__(self, circuit: stim.Circuit) -> None:
        """
        Initializes the MWPM decoder by constructing the matcher from the 
        circuit's detector error model.

        Args:
            circuit (stim.Circuit): The quantum circuit to be decoded.

        Returns:
            None
        """
        error_model = circuit.detector_error_model()
        self.matcher = pymatching.Matching.from_detector_error_model(error_model)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decodes the given syndrome using the configured PyMatching matcher.

        Args:
            syndrome (np.ndarray): A 1D boolean numpy array representing the detection events.

        Returns:
            np.ndarray: A 1D boolean numpy array containing the predicted logical observable flips.
        """
        return self.matcher.decode(syndrome)


class RepetitionMajorityVoteDecoder(BaseDecoder):
    """
    Custom implementation of a Majority Vote decoder for Repetition Codes.
    This decoder ignores the noise model and relies on simple "majority vote" logic.
    """

    @classmethod
    def name(cls) -> str:
        """
        Returns the display name of the Majority Vote decoder.

        Args:
            None

        Returns:
            str: The custom name "Repetition Majority Vote Decoder".
        """
        return "Repetition Majority Vote Decoder"
    
    def __init__(self, circuit: stim.Circuit) -> None:
        """
        Initializes the Majority Vote decoder by extracting code dimensions 
        from the circuit's detector error model.

        Args:
            circuit (stim.Circuit): The quantum circuit to be decoded.

        Returns:
            None
        """
        # We analyze the detector error model to find the code dimensions
        model = circuit.detector_error_model()
        self.num_detectors = model.num_detectors
        self.num_observables = model.num_observables

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decodes a repetition code by performing a majority vote.
        It assumes that the first qubit (q0) is read as the logical qubit. 
        The decoder reconstructs the relative state of all data qubits from 
        the syndrome (detectors) and predicts a logical flip if the majority 
        of the qubits are flipped relative to the first qubit (q0).

        Args:
            syndrome (np.ndarray): 1D boolean array of detection events.

        Returns:
            np.ndarray: 1D boolean array with the predicted logical flips.
        """
        # Each observable corresponds to one repetition code block.
        # So we split the syndrome into equal parts for each observable.
        detectors_per_observable = self.num_detectors // self.num_observables
        
        predictions = []
        
        for i in range(self.num_observables):
            start_index = i * detectors_per_observable
            end_index = start_index + detectors_per_observable
            local_syndrome = syndrome[start_index:end_index]
            
            # In a repetition code, detectors take pairs of qubits in order
            # (q0, q1), (q1, q2), ..., (q_{n-2}, q_{n-1}).
            # We always assume the first qubit (q0) is 0 (not flipped).
            # Iterating through the detectors we can determine if the following
            # qubit is flipped relative to the previous one. 
            # This way we can reconstruct the state of all qubits.
            qubits_state = [0] 
            
            # We iterate through the detectors. 
            # If a detector is 1, the next qubit is DIFFERENT from the current one.
            # If a detector is 0, the next qubit is the SAME as the current one.
            current_state = 0
            for detector_value in local_syndrome:
                if detector_value == 1:
                    # The next qubit changes with respect to the current state
                    current_state = 1 - current_state 
                qubits_state.append(current_state)
            
            # Now we count: are there more 1s or 0s relative to our baseline?
            # Since we defined q0 as 0, if there are more 1s, it means the majority
            # of the block flipped relative to q0. 
            # If the majority flipped, it means q0 (our logical observable) is the
            # one that is wrong, so we predict a logical flip (1).
            num_ones = sum(qubits_state)
            num_zeros = len(qubits_state) - num_ones
            
            if num_ones > num_zeros:
                local_prediction = 1 # The baseline (q0) is wrong, predict flip
            else:
                local_prediction = 0 # The baseline (q0) is right, predict no flip
                
            predictions.append(local_prediction)
            
        return np.array(predictions)


class Shor9MajorityVoteDecoder:
    """
    Custom implementation of a two-level Majority Vote decoder for the 9-qubit Shor Code.
    It counts the detector activations round by round and calculates their parity 
    to filter out measurement errors before applying the majority logic.
    """

    @classmethod
    def name(cls) -> str:
        """
        Returns the display name of the Shor 9 Majority Vote decoder.
        """
        return "Shor 9-Qubit Majority Vote"
    
    def __init__(self, circuit: stim.Circuit) -> None:
        """
        Initializes the decoder by extracting code dimensions from the circuit's 
        detector error model.

        Args:
            circuit (stim.Circuit): The quantum circuit to be decoded.
        """
        model = circuit.detector_error_model()
        self.num_detectors = model.num_detectors
        self.num_observables = model.num_observables

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decodes the Shor code by counting detector activations across all rounds,
        reconstructing the absolute state, and then predicting the exact physical 
        parity of the 9-qubit observable to match Stim's evaluation protocol.

        Args:
            syndrome (np.ndarray): 1D boolean array of detection events.

        Returns:
            np.ndarray: 1D boolean array predicting the physical parity flip.
        """
        detectors_per_observable = self.num_detectors // self.num_observables
        predictions = []
        
        DETECTORS_PER_ROUND = 8
        Z_DETECTORS_PER_ROUND = 6
        X_DETECTORS_PER_ROUND = 2
        
        NUM_INNER_BLOCKS = 3
        Z_DETECT_PER_BLOCK = 2

        for i in range(self.num_observables):
            start_index = i * detectors_per_observable
            end_index = start_index + detectors_per_observable
            local_syndrome = syndrome[start_index:end_index]
            
            z_activation_count = np.zeros(Z_DETECTORS_PER_ROUND, dtype=int)
            x_activation_count = np.zeros(X_DETECTORS_PER_ROUND, dtype=int)
            
            num_full_rounds = len(local_syndrome) // DETECTORS_PER_ROUND
            
            for round_index in range(num_full_rounds):
                start_of_round = round_index * DETECTORS_PER_ROUND
                end_of_round = start_of_round + DETECTORS_PER_ROUND
                round_data = local_syndrome[start_of_round:end_of_round]
                
                for k in range(Z_DETECTORS_PER_ROUND):
                    z_activation_count[k] += round_data[k]
                    
                for k in range(X_DETECTORS_PER_ROUND):
                    x_activation_count[k] += round_data[Z_DETECTORS_PER_ROUND + k]
                    
            boundary_detectors_count = len(local_syndrome) % DETECTORS_PER_ROUND
            start_of_boundary = num_full_rounds * DETECTORS_PER_ROUND
            boundary_data = local_syndrome[start_of_boundary:]
            
            if boundary_detectors_count == Z_DETECTORS_PER_ROUND:
                for k in range(Z_DETECTORS_PER_ROUND):
                    z_activation_count[k] += boundary_data[k]
                    
            elif boundary_detectors_count == X_DETECTORS_PER_ROUND:
                for k in range(X_DETECTORS_PER_ROUND):
                    x_activation_count[k] += boundary_data[k]
            else:
                predictions.append(0)
                continue
                
            absolute_z_syndrome = z_activation_count % 2
            absolute_x_syndrome = x_activation_count % 2
                
            # INNER CODE EVALUATION
            total_predicted_z_parity = 0
            
            for block_id in range(NUM_INNER_BLOCKS):
                start_z = block_id * Z_DETECT_PER_BLOCK
                end_z = start_z + Z_DETECT_PER_BLOCK
                block_syndrome = absolute_z_syndrome[start_z:end_z]
                
                qubits_state = [0]
                current_relative_state = 0
                
                for detector_value in block_syndrome:
                    if detector_value == 1:
                        current_relative_state = 1 - current_relative_state
                    qubits_state.append(current_relative_state)
                
                num_ones = sum(qubits_state)
                num_zeros = len(qubits_state) - num_ones
                
                block_has_error = 1 if num_ones > num_zeros else 0
                
                # Calculate physical parity of the 3 qubits based on prediction
                block_physical_parity = (num_ones % 2) ^ block_has_error
                total_predicted_z_parity ^= block_physical_parity
            
            # OUTER CODE EVALUATION
            blocks_relative_state = [0]
            current_block_state = 0
            
            for detector_value in absolute_x_syndrome:
                if detector_value == 1:
                    current_block_state = 1 - current_block_state
                blocks_relative_state.append(current_block_state)
                
            num_block_ones = sum(blocks_relative_state)
            num_block_zeros = len(blocks_relative_state) - num_block_ones
            
            outer_phase_flip = 1 if num_block_ones > num_block_zeros else 0
            
            # Calculate physical parity of the outer blocks based on prediction
            total_predicted_x_parity = (num_block_ones % 2) ^ outer_phase_flip
            
            # FINAL PREDICTION: MATCH STIM'S OBSERVABLE BEHAVIOR
            if boundary_detectors_count == Z_DETECTORS_PER_ROUND:
                predictions.append(total_predicted_z_parity)
            else:
                predictions.append(total_predicted_x_parity)
            
        return np.array(predictions)