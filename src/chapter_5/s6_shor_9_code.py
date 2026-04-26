import stim
import pymatching
import numpy as np
import matplotlib.pyplot as plt

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
        # Bit-flip syndromes (Z-ancillas 9-14)
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

        # Phase-flip syndromes (X-ancillas 15-16)
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

    # Final Measurement and Observable logic
    if before_round_data_depolarization > 0:
        circuit.append("DEPOLARIZE1", range(9), before_round_data_depolarization)

    if basis == "Z":
        append_noisy_gate(circuit, "H", range(9), after_clifford_depolarization)
    elif basis == "Y":
        append_noisy_gate(circuit, "S_DAG", range(9), after_clifford_depolarization)
        append_noisy_gate(circuit, "H", range(9), after_clifford_depolarization)

    append_noisy_measurement(circuit, range(9), before_measure_flip_probability)

    # Boundary detectors
    if basis == "X":
        for i, targets in enumerate([(0,1), (1,2), (3,4), (4,5), (6,7), (7,8)]):
            circuit.append("DETECTOR", [stim.target_rec(-17 + i), stim.target_rec(-9 + targets[0]), stim.target_rec(-9 + targets[1])])
    else:
        circuit.append("DETECTOR", [stim.target_rec(-11)] + [stim.target_rec(-i) for i in range(4, 10)])
        circuit.append("DETECTOR", [stim.target_rec(-10)] + [stim.target_rec(-i) for i in range(1, 7)])

    # Logical Observable
    #BEGIN S6_SNIPPET_OBSERVABLES
    if basis == "X":
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-9), stim.target_rec(-6), stim.target_rec(-3)], 0)
    else:
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-i) for i in range(1, 10)], 0)
    #BEGIN S6_SNIPPET_OBSERVABLES

    return circuit

def sample_shor_code_mwpm(shots: int = 10000):
    """Simulates the Shor code over error rates and cardinal states, returning all results."""
    physical_error_rates = np.linspace(0.01, 0.12, 8)
    states = ["0", "1", "+", "-", "+i", "-i"]
    state_results = {state: [] for state in states}

    for p in physical_error_rates:
        for state in states:
            circuit = build_shor_circuit(
                rounds=3, state=state,
                before_round_data_depolarization=p,
                after_clifford_depolarization=p,
                before_measure_flip_probability=p
            )
            sampler = circuit.compile_detector_sampler()
            syndromes, actual_observables = sampler.sample(shots, separate_observables=True)
            matcher = pymatching.Matching.from_stim_circuit(circuit)
            predicted_observables = matcher.decode_batch(syndromes)
            num_errors = np.sum(predicted_observables != actual_observables)
            state_results[state].append(num_errors / shots)

    # Calculate mean logical error rate across all states
    mean_results = np.mean([state_results[s] for s in states], axis=0)
    return physical_error_rates, state_results, mean_results

def plot_shor_code_mwpm_performance(shots: int = 5000) -> None:
    """Plots performance for each cardinal state and their average."""
    p_rates, state_l_rates, mean_l_rates = sample_shor_code_mwpm(shots)
    
    plt.figure(figsize=(10, 7))
    for state, l_rates in state_l_rates.items():
        plt.plot(p_rates, l_rates, label=f'State |{state}>', alpha=0.6, linestyle=':')
    
    plt.plot(p_rates, mean_l_rates, marker='o', color='black', linewidth=2, label='Mean Performance')
    plt.plot(p_rates, p_rates, linestyle='--', color='gray', label='Physical Limit')
    
    plt.title('Shor 9-qubit Code: Cardinal States Performance (MWPM)')
    plt.xlabel('Physical Error Rate (p)')
    plt.ylabel('Logical Error Rate ($p_L$)')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, which="both", linestyle=':', alpha=0.7)
    plt.savefig('../../figures/shor_9_code_plot.png')

if __name__ == "__main__":
    plot_shor_code_mwpm_performance()
#END S6_SNIPPET
