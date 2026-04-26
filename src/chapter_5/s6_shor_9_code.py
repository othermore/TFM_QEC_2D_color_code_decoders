import stim
import pymatching
import numpy as np

#BEGIN S6_SNIPPET_NOISY_GATE
def append_noisy_gate(circuit: stim.Circuit, gate: str, targets: list, noise: float) -> None:
    """Appends a gate and adds depolarizing noise if noise > 0."""
    circuit.append(gate, targets)
    if noise > 0:
        if gate in ["CX", "CY", "CZ", "SWAP"]:
            circuit.append("DEPOLARIZE2", targets, noise)
        else:
            circuit.append("DEPOLARIZE1", targets, noise)
#END S6_SNIPPET_NOISY_GATE

#BEGIN S6_SNIPPET_NOISY_MEASUREMENT
def append_noisy_measurement(circuit: stim.Circuit, targets: list, noise: float) -> None:
    """Appends a measurement with pre-measurement bit-flip noise."""
    if noise > 0:
        circuit.append("X_ERROR", targets, noise)
    circuit.append("M", targets)
#END S6_SNIPPET_NOISY_MEASUREMENT

def build_shor_circuit(
    rounds: int = 3, 
    state: str = "0",
    before_round_data_depolarization: float = 0.0,
    after_clifford_depolarization: float = 0.0,
    before_measure_flip_probability: float = 0.0
) -> stim.Circuit:
    """Builds the 9-qubit Shor code circuit with state initialization and adaptive observables."""
    if state in ["0", "1"]: basis = "Z"
    elif state in ["+", "-"]: basis = "X"
    elif state in ["+i", "-i"]: basis = "Y"
    else: raise ValueError("Invalid state")

    circuit = stim.Circuit()
    circuit.append("R", range(17))
    
    # State initialization
    if state == "1": append_noisy_gate(circuit, "X", [0], after_clifford_depolarization)
    elif state == "+": append_noisy_gate(circuit, "H", [0], after_clifford_depolarization)
    elif state == "-":
        append_noisy_gate(circuit, "X", [0], after_clifford_depolarization)
        append_noisy_gate(circuit, "H", [0], after_clifford_depolarization)
    elif state == "+i":
        append_noisy_gate(circuit, "H", [0], after_clifford_depolarization)
        append_noisy_gate(circuit, "S", [0], after_clifford_depolarization)
    elif state == "-i":
        append_noisy_gate(circuit, "X", [0], after_clifford_depolarization)
        append_noisy_gate(circuit, "H", [0], after_clifford_depolarization)
        append_noisy_gate(circuit, "S", [0], after_clifford_depolarization)

    # Shor Encoding
    append_noisy_gate(circuit, "CX", [0, 3, 0, 6], after_clifford_depolarization)
    append_noisy_gate(circuit, "H", [0, 3, 6], after_clifford_depolarization)
    encoding_cx = [0, 1, 0, 2, 3, 4, 3, 5, 6, 7, 6, 8]
    append_noisy_gate(circuit, "CX", encoding_cx, after_clifford_depolarization)
    circuit.append("TICK")

    #BEGIN S6_SNIPPET_ROUND_NOISE
    # Syndrome measurement rounds
    for r in range(rounds):
        if before_round_data_depolarization > 0:
            circuit.append("DEPOLARIZE1", range(9), before_round_data_depolarization)
    #END S6_SNIPPET_ROUND_NOISE
        # Bit-flip syndromes (Z-ancillas 9-14 measuring X stabilizers which are pairs of data qubits)
        z_targets = [0,9, 1,9, 1,10, 2,10, 3,11, 4,11, 4,12, 5,12, 6,13, 7,13, 7,14, 8,14]
        circuit.append("R", range(9, 15))
        append_noisy_gate(circuit, "CX", z_targets, after_clifford_depolarization)
        #BEGIN S6_SNIPPET_MEASURE_DETECTORS
        append_noisy_measurement(circuit, range(9, 15), before_measure_flip_probability)
        for i in range(6):
            if r == 0: circuit.append("DETECTOR", [stim.target_rec(-6 + i)])
            else: circuit.append("DETECTOR", [stim.target_rec(-6 + i), stim.target_rec(-6 + i - 8)])
        #END S6_SNIPPET_MEASURE_DETECTORS
        circuit.append("TICK")

        # Phase-flip syndromes (X-ancillas 15-16 measuring Z stabilizers which are qubits 1-6 and 3-9)
        circuit.append("R", [15, 16])
        append_noisy_gate(circuit, "H", [15, 16], after_clifford_depolarization)
        x_targets = []
        for d in range(6): x_targets.extend([15, d])
        for d in range(3, 9): x_targets.extend([16, d])
        append_noisy_gate(circuit, "CX", x_targets, after_clifford_depolarization)
        append_noisy_gate(circuit, "H", [15, 16], after_clifford_depolarization)
        append_noisy_measurement(circuit, [15, 16], before_measure_flip_probability)
        for i in range(2):
            if r == 0: circuit.append("DETECTOR", [stim.target_rec(-2 + i)])
            else: circuit.append("DETECTOR", [stim.target_rec(-2 + i), stim.target_rec(-2 + i - 8)])
        circuit.append("TICK")
    '''
    # Final Measurement rotations
    if before_round_data_depolarization > 0:
        circuit.append("DEPOLARIZE1", range(9), before_round_data_depolarization)
    '''
    if basis == "Z":
        append_noisy_gate(circuit, "H", range(9), after_clifford_depolarization)
    elif basis == "Y":
        append_noisy_gate(circuit, "S_DAG", range(9), after_clifford_depolarization)
        append_noisy_gate(circuit, "H", range(9), after_clifford_depolarization)
    
    #BEGIN S6_SNIPPET_FINAL_MEASUREMENT
    append_noisy_measurement(circuit, range(9), before_measure_flip_probability)

    # Boundary detectors and observables
    if basis == "X":
        for i, targets in enumerate([(0,1), (1,2), (3,4), (4,5), (6,7), (7,8)]):
            circuit.append("DETECTOR", [stim.target_rec(-17 + i), stim.target_rec(-9 + targets[0]), stim.target_rec(-9 + targets[1])])
        # Logical X Distinguishes + and - (Measure Z0*Z3*Z6)
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-9), stim.target_rec(-6), stim.target_rec(-3)], 0)
    elif basis == "Z":
        circuit.append("DETECTOR", [stim.target_rec(-11)] + [stim.target_rec(-i) for i in range(4, 10)])
        circuit.append("DETECTOR", [stim.target_rec(-10)] + [stim.target_rec(-i) for i in range(1, 7)])
        # Logical Z Distinguishes 0 and 1 (Measure transversal X parity)
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-i) for i in range(1, 10)], 0)
    else:
        # For basis Y, we just measure transversal parity
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-i) for i in range(1, 10)], 0)
    #END S6_SNIPPET_FINAL_MEASUREMENT

    return circuit

def sample_shor_code_mwpm(shots=10000, rounds=1, states=["0"], p_rates=None, noise_ratios=(1,0,0)):
    """General purpose sampling for Shor code MWPM decoding."""
    if p_rates is None: p_rates = np.linspace(0.01, 0.12, 8)
    l_rates = []
    r_data, r_gate, r_measure = noise_ratios
    
    for p in p_rates:
        errors = 0
        shots_per_state = shots // len(states)
        for state in states:
            circuit = build_shor_circuit(
                rounds=rounds, state=state,
                before_round_data_depolarization=p*r_data,
                after_clifford_depolarization=p*r_gate,
                before_measure_flip_probability=p*r_measure
            )
            sampler = circuit.compile_detector_sampler()
            syndromes, obs = sampler.sample(shots_per_state, separate_observables=True)
            matcher = pymatching.Matching.from_stim_circuit(circuit)
            predicted = matcher.decode_batch(syndromes)
            errors += np.sum(predicted != obs)
        l_rates.append(errors / (shots_per_state * len(states)))
    return p_rates, l_rates

def run_shor_simulations(shots: int = 2000):
    """Executes the three main comparative simulations for Chapter 5 and prints a text summary."""
    print("Executing Shor 9-qubit comprehensive simulations...")
    p_rates = np.linspace(0.01, 0.12, 8)

    # 1. Noise Model Comparison
    print("\n--- Simulation: Noise Models (1 Round, State |0>) ---")
    models = {
        "L1: $E_{data}$": (1, 0, 0),
        "L2: $E_{data} + E_{meas}$": (1, 0, 1),
        "L3: $E_{data} + E_{meas} + E_{gates}$": (1, 1, 1)
    }
    for label, ratios in models.items():
        _, l_rates = sample_shor_code_mwpm(shots=shots, rounds=1, states=["0"], p_rates=p_rates, noise_ratios=ratios)
        mean_pl = np.mean(l_rates)
        print(f"{label:25} | Mean pL: {mean_pl:.4f} | Max pL: {max(l_rates):.4f}")

    # 2. Rounds Comparison
    print("\n--- Simulation: Rounds Comparison (Code Capacity, State |0>) ---")
    for r in [1, 3, 5]:
        _, l_rates = sample_shor_code_mwpm(shots=shots, rounds=r, states=["0"], p_rates=p_rates, noise_ratios=(1, 0, 0))
        mean_pl = np.mean(l_rates)
        print(f"{r} Rounds                  | Mean pL: {mean_pl:.4f} | Max pL: {max(l_rates):.4f}")

    # 3. Cardinal States Comparison
    print("\n--- Simulation: Cardinal States (Code Capacity, 1 Round) ---")
    states_list = ["0", "1", "+", "-", "+i", "-i"]
    for state in states_list:
        _, l_rates = sample_shor_code_mwpm(shots=shots, rounds=1, states=[state], p_rates=p_rates, noise_ratios=(1, 0, 0))
        mean_pl = np.mean(l_rates)
        print(f"State |{state:2}>                | Mean pL: {mean_pl:.4f}")
    
    _, mean_l_rates = sample_shor_code_mwpm(shots=shots, rounds=1, states=states_list, p_rates=p_rates, noise_ratios=(1, 0, 0))
    print(f"{'Overall Mean':25} | Mean pL: {np.mean(mean_l_rates):.4f}")
    
    print("\nSimulations complete.")

if __name__ == "__main__":
    run_shor_simulations(shots=2000)
