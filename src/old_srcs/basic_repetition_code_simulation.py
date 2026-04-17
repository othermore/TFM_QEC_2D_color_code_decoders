import stim
import numpy as np
from decoders import MWPMLibraryDecoder
import random

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

def simulate_repetition_code(distance: int = 3, 
                             rounds: int = None, 
                             state: str = "0",
                             noise: float = 0.05, 
                             num_shots: int = 10000, 
                             decoder_class = MWPMLibraryDecoder,
                             before_round_data_depolarization_noise: float = None, 
                             after_clifford_depolarization_noise: float = None, 
                             before_measure_flip_probability_noise: float = None) -> float:
    """
    Simulates a repetition code and evaluates a specified decoder class.

    Args:
        distance (int, optional): The distance of the repetition code. Defaults to 3.
        rounds (int, optional): Number of syndrome measurement rounds. Defaults to distance.
        state (str, optional): The initial logical state to encode ("0" or "1"). Defaults to "0".
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
        rounds = distance

    if state not in ["0", "1"]:
        raise ValueError("Invalid state. Must be '0' or '1' for basic repetition code.")

    if before_round_data_depolarization_noise is None:
        before_round_data_depolarization_noise = noise
    if after_clifford_depolarization_noise is None:
        after_clifford_depolarization_noise = noise
    if before_measure_flip_probability_noise is None:
        before_measure_flip_probability_noise = noise
    
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=rounds,
        distance=distance,
        before_round_data_depolarization=before_round_data_depolarization_noise,
        after_clifford_depolarization=after_clifford_depolarization_noise,
        before_measure_flip_probability=before_measure_flip_probability_noise
    )

    if state == "1":
        new_circuit = stim.Circuit()
        first_tick_passed = False
        data_qubits = [2 * i for i in range(distance)]
        
        for instr in circuit:
            new_circuit.append(instr)
            if instr.name == "TICK" and not first_tick_passed:
                # Flip data qubits to initialize in state |1>
                new_circuit.append("X", data_qubits)
                if after_clifford_depolarization_noise:
                    new_circuit.append("DEPOLARIZE1", data_qubits, after_clifford_depolarization_noise)
                first_tick_passed = True
        circuit = new_circuit

    # Instantiate the decoder provided in the arguments
    decoder_instance = decoder_class(circuit)
    
    syndromes, actual_observables = _run_simulation(circuit, num_shots)
    logical_error_rate = _evaluate_decoder(decoder_instance, syndromes, actual_observables, num_shots)
    
    return logical_error_rate

def run_repetition_code_simulation() -> None:
    """
    Main entry point for running the simulation with benchmark output.
    Iterates over the possible initial states ("0" and "1") to validate 
    their stability in equal proportions.

    Args:
        None

    Returns:
        None
    """
    print("Starting basic repetition code simulation...\n")
    num_shots_per_state = 5000
    noise = 0.05
    distance = 3
    
    states = ["0", "1"]
    
    total_shots = num_shots_per_state * len(states)
    total_errors = 0

    print(f"Physical noise injected: {noise * 100:.2f}%")
    print(f"Shots per state: {num_shots_per_state}")
    print("Running simulations...\n")

    for state in states:
        logical_error_rate = simulate_repetition_code(
            distance=distance,
            state=state,
            num_shots=num_shots_per_state, 
            noise=noise, 
            decoder_class=MWPMLibraryDecoder
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
    run_repetition_code_simulation()