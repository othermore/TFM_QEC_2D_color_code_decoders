import stim
import numpy as np
from decoders import MWPMLibraryDecoder
import random

def append_noisy_gate(circuit: stim.Circuit, gate: str, targets: list, noise: float) -> None:
    """
    Appends a quantum gate to the given Stim circuit and adds depolarizing noise if specified.

    Args:
        circuit (stim.Circuit): The circuit to append the gate to.
        gate (str): The name of the gate to append (e.g., "H", "CX").
        targets (list): The target qubit(s) for the gate.
        noise (float): The probability of depolarizing noise. If > 0, depolarizing noise
            is added after the gate (DEPOLARIZE1 for single-qubit gates, DEPOLARIZE2 for two-qubit gates).

    Returns:
        None
    """
    circuit.append(gate, targets)
    if noise:
        single_qubit_gates = {
            "I", "X", "Y", "Z", "H", "S", "S_DAG", 
            "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG"
        }
        two_qubit_gates = {
            "CX", "CY", "CZ", "SWAP", "ISWAP", 
            "XCX", "XCZ", "YCX", "YCZ"
        }
        
        if gate in single_qubit_gates:
            circuit.append("DEPOLARIZE1", targets, noise)
        elif gate in two_qubit_gates:
            circuit.append("DEPOLARIZE2", targets, noise)

def append_noisy_measurement(circuit: stim.Circuit, targets: list, noise: float) -> None:
    """
    Appends a measurement to the given Stim circuit and adds pre-measurement bit-flip noise if specified.

    Args:
        circuit (stim.Circuit): The circuit to append the measurement to.
        targets (list): The target qubit(s) to measure.
        noise (float): The probability of a pre-measurement bit-flip error (X_ERROR).

    Returns:
        None
    """
    if noise:
        circuit.append("X_ERROR", targets, noise)
    circuit.append("M", targets)

def build_shor_circuit(rounds: int, 
                       state: str = "0",
                       before_round_data_depolarization_noise: float = None, 
                       after_clifford_depolarization_noise: float = None, 
                       before_measure_flip_probability_noise: float = None) -> stim.Circuit:
    """
    Manually builds the Stim circuit for the 9-qubit Shor code, starting from a specific logical state.
    Includes the explicit textbook encoding.

    Args:
        rounds (int): The number of syndrome measurement rounds.
        state (str): The initial logical state to encode. Must be one of "0", "1", "+", "-", "+i", "-i".
        before_round_data_depolarization_noise (float): Depolarization noise probability on data qubits before each round.
        after_clifford_depolarization_noise (float): Depolarization noise probability applied after Clifford gates.
        before_measure_flip_probability_noise (float): Bit-flip noise probability applied right before measurements.

    Returns:
        stim.Circuit: The generated Stim circuit for the Shor code simulation.
    """
    if state in ["0", "1"]:
        basis = "Z"
    elif state in ["+", "-"]:
        basis = "X"
    elif state in ["+i", "-i"]:
        basis = "Y"
    else:
        raise ValueError("Invalid state. Must be one of '0', '1', '+', '-', '+i', '-i'")

    circuit = stim.Circuit()

    # Qubit Mapping:
    # 0 to 8   : Data Qubits
    # 9 to 14  : Z Ancillas (Measure Z parity to detect X Bit-flips)
    # 15 to 16 : X Ancillas (Measure X parity to detect Z Phase-flips)
    
    circuit.append("R", range(17))
    circuit.append("TICK") 

    # Initialize the circuit depending on the requested state
    if state == "1":
        append_noisy_gate(circuit, "X", [0], after_clifford_depolarization_noise)
    elif state == "+":
        append_noisy_gate(circuit, "H", [0], after_clifford_depolarization_noise)
    elif state == "-":
        append_noisy_gate(circuit, "X", [0], after_clifford_depolarization_noise)
        append_noisy_gate(circuit, "H", [0], after_clifford_depolarization_noise)
    elif state == "+i":
        append_noisy_gate(circuit, "H", [0], after_clifford_depolarization_noise)
        append_noisy_gate(circuit, "S", [0], after_clifford_depolarization_noise)
    elif state == "-i":
        append_noisy_gate(circuit, "X", [0], after_clifford_depolarization_noise)
        append_noisy_gate(circuit, "H", [0], after_clifford_depolarization_noise)
        append_noisy_gate(circuit, "S", [0], after_clifford_depolarization_noise)

    # Spreading the state to the block anchors
    append_noisy_gate(circuit, "CX", [0, 3, 0, 6], after_clifford_depolarization_noise)
    
    # Creating superpositions for the phase-flip code blocks
    append_noisy_gate(circuit, "H", [0, 3, 6], after_clifford_depolarization_noise)
    
    # Entangling within each block for the bit-flip code
    encoding_cx_targets = [0, 1, 0, 2, 
                        3, 4, 3, 5, 
                        6, 7, 6, 8]
    append_noisy_gate(circuit, "CX", encoding_cx_targets, after_clifford_depolarization_noise)

    circuit.append("TICK")

    # Execute each round: inject the noise and apply the stabilizers
    for r in range(rounds):
        
        # Injecting depolarizing noise
        if before_round_data_depolarization_noise:
            circuit.append("DEPOLARIZE1", range(9), before_round_data_depolarization_noise)

        # Z estabilizers: 
        #   S1: $Z_0 Z_1$, S2: $Z_1 Z_2$
        #   S3: $Z_3 Z_4$, S4: $Z_4 Z_5$
        #   S5: $Z_6 Z_7$, S6: $Z_7 Z_8$
        z_syndrome_cx_targets = [
            0,9, 1,9, 1,10, 2,10, 
            3,11, 4,11, 4,12, 5,12, 
            6,13, 7,13, 7,14, 8,14
        ]
        circuit.append("R", range(9, 15))
        append_noisy_gate(circuit, "CX", z_syndrome_cx_targets, after_clifford_depolarization_noise)
        
        # We meassure the ancillas 9 to 14
        append_noisy_measurement(circuit, range(9, 15), before_measure_flip_probability_noise)

        # Adding Z Detectors
        for i in range(6):
            if r == 0:
                circuit.append("DETECTOR", [stim.target_rec(-6 + i)])
            else: 
                circuit.append("DETECTOR", [stim.target_rec(-6 + i), stim.target_rec(-6 + i - 8)])
        circuit.append("TICK")

        # X Syndrome Measurements for Phase-Flips
        circuit.append("R", [15, 16])
        append_noisy_gate(circuit, "H", [15, 16], after_clifford_depolarization_noise)
        
        # X Stabilizers:
        #   Blocks 1 and 2: $X_0 X_1 X_2 X_3 X_4 X_5$ 
        #   Blocks 2 and 3: $X_3 X_4 X_5 X_6 X_7 X_8$
        x_syndrome_cx_targets = []
        for d in range(6):
            x_syndrome_cx_targets.extend([15, d])
        for d in range(3, 9):
            x_syndrome_cx_targets.extend([16, d])
            
        append_noisy_gate(circuit, "CX", x_syndrome_cx_targets, after_clifford_depolarization_noise)
        append_noisy_gate(circuit, "H", [15, 16], after_clifford_depolarization_noise)
        
        # We meassure the ancillas 15 and 16
        append_noisy_measurement(circuit, [15, 16], before_measure_flip_probability_noise)

        # Adding X Detectors
        for i in range(2):
            if r == 0:
                circuit.append("DETECTOR", [stim.target_rec(-2 + i)])
            else:
                circuit.append("DETECTOR", [stim.target_rec(-2 + i), stim.target_rec(-2 + i - 8)])

        circuit.append("TICK")

    # Final Measurement
    if before_round_data_depolarization_noise:
        circuit.append("DEPOLARIZE1", range(9), before_round_data_depolarization_noise)

    if basis == "Z":
        append_noisy_gate(circuit, "H", range(9), after_clifford_depolarization_noise)
    elif basis == "Y":
        append_noisy_gate(circuit, "S_DAG", range(9), after_clifford_depolarization_noise)
        append_noisy_gate(circuit, "H", range(9), after_clifford_depolarization_noise)

    append_noisy_measurement(circuit, range(9), before_measure_flip_probability_noise)

    # BOUNDARY DETECTORS
    if basis == "X":
        # $Z_0 Z_1$
        circuit.append("DETECTOR", [
            stim.target_rec(-17), 
            stim.target_rec(-9), stim.target_rec(-8)])
        # $Z_1 Z_2$
        circuit.append("DETECTOR", [
            stim.target_rec(-16), 
            stim.target_rec(-8), stim.target_rec(-7)])
        # $Z_3 Z_4$
        circuit.append("DETECTOR", [
            stim.target_rec(-15), 
            stim.target_rec(-6), stim.target_rec(-5)])
        # $Z_4 Z_5$
        circuit.append("DETECTOR", [
            stim.target_rec(-14), 
            stim.target_rec(-5), stim.target_rec(-4)])
        # $Z_6 Z_7$
        circuit.append("DETECTOR", [
            stim.target_rec(-13), 
            stim.target_rec(-3), stim.target_rec(-2)])
        # $Z_7 Z_8$
        circuit.append("DETECTOR", [
            stim.target_rec(-12), 
            stim.target_rec(-2), stim.target_rec(-1)])
    elif basis == "Z":
        #   Blocks 1 and 2: $X_0 X_1 X_2 X_3 X_4 X_5$ 
        circuit.append("DETECTOR", [
            stim.target_rec(-11), 
            stim.target_rec(-9), stim.target_rec(-8), stim.target_rec(-7), 
            stim.target_rec(-6), stim.target_rec(-5), stim.target_rec(-4)
        ])
        #   Blocks 2 and 3: $X_3 X_4 X_5 X_6 X_7 X_8$
        circuit.append("DETECTOR", [
            stim.target_rec(-10), 
            stim.target_rec(-6), stim.target_rec(-5), stim.target_rec(-4), 
            stim.target_rec(-3), stim.target_rec(-2), stim.target_rec(-1)
        ])
    # No boundary detectors are added for Y basis because transversal Y measurements
    # destroy both X and Z stabilizers.

    # Logical Observable (Only ONE is added to ensure determinism in Stim)
    if basis == "X":
        # Measure Bit-flips using Z0*Z3*Z6 (Physical parity of these qubits in Z)
        # Note: Since we measure in Z-basis at the end, these recs correspond to data qubits.
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-9), stim.target_rec(-6), stim.target_rec(-3)], 0)
    else:
        # Measure Phase-flips using X_L (Total parity in X)
        # In basis Z/Y, we apply H to all qubits, so measuring them in Z gives the X parity.
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-i) for i in range(1, 10)], 0)

    return circuit

def _run_simulation(circuit: stim.Circuit, num_shots: int) -> tuple:
    """
    Samples the circuit to generate physical syndromes and logical observables.

    Args:
        circuit (stim.Circuit): The compiled Stim circuit to simulate.
        num_shots (int): The number of independent samples/shots to draw.

    Returns:
        tuple: A tuple containing:
            - syndromes (np.ndarray): The detection events for each shot.
            - actual_observables (np.ndarray): The actual logical observable flips for each shot.
    """
    sampler = circuit.compile_detector_sampler()
    return sampler.sample(num_shots, separate_observables=True)

def _evaluate_decoder(decoder_instance, syndromes: np.ndarray, actual_observables: np.ndarray, num_shots: int) -> float:
    """
    Generic evaluation loop that works with any decoder object implementing .decode().
    It evaluates the Block Error Rate (if any observable fails, the shot fails).

    Args:
        decoder_instance: An initialized decoder object with a .decode() method.
        syndromes (np.ndarray): The matrix of syndrome events from the simulation.
        actual_observables (np.ndarray): The actual logical observable flips for each shot.
        num_shots (int): The total number of shots to evaluate.

    Returns:
        float: The logical error rate (Block Error Rate) as a percentage.
    """
    logical_errors = 0

    for i in range(num_shots):
        # The decoder predicts the array of errors for ALL logical observables
        prediction = decoder_instance.decode(syndromes[i])
        
        # We compare the entire array. 
        # np.array_equal returns True only if both arrays have the same shape 
        # and all elements match exactly.
        if not np.array_equal(prediction, actual_observables[i]):
            logical_errors += 1

    return (logical_errors / num_shots) * 100

def simulate_shor_code(
        rounds: int = None, 
        state: str = "0", 
        noise: float = 0.05, 
        num_shots: int = 10000, 
        decoder_class = MWPMLibraryDecoder,
        before_round_data_depolarization_noise: float = None, 
        after_clifford_depolarization_noise: float = None, 
        before_measure_flip_probability_noise: float = None) -> float:
    """
    Simulates the Shor code and evaluates a specified decoder class.

    Args:
        rounds (int, optional): Number of syndrome measurement rounds. Defaults to 3.
        state (str, optional): The initial state ("0", "1", "+", "-", "+i", "-i"). Defaults to "0".
        noise (float, optional): Base noise probability. Defaults to 0.05.
        num_shots (int, optional): Number of shots to simulate. Defaults to 10000.
        decoder_class (optional): Decoder class to use. Defaults to MWPMLibraryDecoder.
        before_round_data_depolarization_noise (float, optional): Specific data depolarization noise.
        after_clifford_depolarization_noise (float, optional): Specific Clifford depolarization noise.
        before_measure_flip_probability_noise (float, optional): Specific measurement flip noise.

    Returns:
        float: The logical error rate percentage for this specific configuration.
    """
    if rounds is None:
        rounds = 3

    if before_round_data_depolarization_noise is None:
        before_round_data_depolarization_noise = noise
    if after_clifford_depolarization_noise is None:
        after_clifford_depolarization_noise = noise
    if before_measure_flip_probability_noise is None:
        before_measure_flip_probability_noise = noise

    circuit = build_shor_circuit(
        rounds=rounds, 
        state=state,         
        before_round_data_depolarization_noise=before_round_data_depolarization_noise,
        after_clifford_depolarization_noise=after_clifford_depolarization_noise,
        before_measure_flip_probability_noise=before_measure_flip_probability_noise)
    
    decoder_instance = decoder_class(circuit)
    
    syndromes, actual_observables = _run_simulation(circuit, num_shots)
    return _evaluate_decoder(decoder_instance, syndromes, actual_observables, num_shots)

def run_shor_simulation() -> None:
    """
    Main entry point for running the simulation with benchmark output.
    Iterates over all 6 possible initial states to validate their stability
    in equal proportions.

    Args:
        None

    Returns:
        None
    """
    print("Starting 9-Qubit Shor Code simulation (Code Capacity Model)...\n")
    num_shots_per_state = 2000
    noise = 0.02
    
    states = ["0", "1", "+", "-", "+i", "-i"]
    
    total_shots = num_shots_per_state * len(states)
    total_errors = 0

    print(f"Physical noise (Data Depolarization): {noise * 100:.2f}%")
    print(f"Noise on Gates/Measurements: 0.00%")
    print(f"Shots per state: {num_shots_per_state}")
    print("Running simulations...\n")

    for state in states:
        logical_error_rate = simulate_shor_code(
            rounds=3,
            state=state,
            num_shots=num_shots_per_state, 
            noise=noise, 
            decoder_class=MWPMLibraryDecoder,
            before_round_data_depolarization_noise=noise,
            after_clifford_depolarization_noise=0,
            before_measure_flip_probability_noise=0
        )
        errors_for_state = (logical_error_rate / 100.0) * num_shots_per_state
        total_errors += errors_for_state
        print(f"State |{state}>: Error Rate = {logical_error_rate:.2f}%")
    
    overall_error_rate = (total_errors / total_shots) * 100

    print("\n--- BENCHMARK RESULTS ---")
    print(f"Total shots simulated: {total_shots}")
    print(f"Final Logical Error Rate (Overall): {overall_error_rate:.2f}%")
    print("\nSimulation complete.")

if __name__ == "__main__":
    run_shor_simulation()
