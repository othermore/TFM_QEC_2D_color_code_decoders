import matplotlib.pyplot as plt
import numpy as np
import os
import multiprocessing

import cirq
import stimcirq
import sinter
import stim


def build_rep_code_cirq(distance: int, p: float) -> stim.Circuit:
    """
    Builds a fully functional 1D Bit-Flip Repetition Code using Cirq.
    This demonstrates injecting noise and manually placing stimcirq annotations.
    """
    #BEGIN S8_SNIPPET_CIRQ_BUILDER
    qubits = cirq.LineQubit.range(2 * distance - 1)
    circuit = cirq.Circuit()
    rounds = distance
    
    for r in range(rounds):
        # Add noise to data qubits. he NEW_THEN_INLINE prevents
        # the noise annotation to be moved to the beginning.
        circuit.append([cirq.depolarize(p).on(q) for q in qubits[0::2]], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        
        # CNOTs to compute parity -- First qubit is a data qubit. Then we alternate ancillas and qubits. Each ancilla
        # is connected to two data qubits, one on each side.
        # Odd qubits are ancillas, even qubits are data qubits.
        circuit.append([cirq.CNOT(qubits[i-1], qubits[i]) for i in range(1, 2*distance-1, 2)], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuit.append([cirq.CNOT(qubits[i+1], qubits[i]) for i in range(1, 2*distance-1, 2)], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        
        # Measure ancillas (with explicit string keys for stimcirq tracking)
        circuit.append([cirq.measure(qubits[i], key=f'm_{r}_{i}') for i in range(1, 2*distance-1, 2)], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        
        # Define Detectors using stimcirq.DetAnnotation
        det_ops = []
        for i in range(1, 2*distance-1, 2):
            key = f'm_{r}_{i}'
            if r == 0:
                det_ops.append(stimcirq.DetAnnotation(parity_keys=[key], coordinate_metadata=(i, r)))
            else:
                prev_key = f'm_{r-1}_{i}'
                det_ops.append(stimcirq.DetAnnotation(parity_keys=[key, prev_key], coordinate_metadata=(i, r)))
        circuit.append(det_ops, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
                
        # Reset ancillas for the next round
        circuit.append([cirq.reset(qubits[i]) for i in range(1, 2*distance-1, 2)], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    # Final Data measurement (of data qubits)
    circuit.append([cirq.measure(qubits[i], key=f'd_{i}') for i in range(0, 2*distance-1, 2)], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        
    # Final Detectors (comparing data parity to last ancilla measurement)
    # For each ancilla we add the ancilla measurement from prior round and the measurement of the two data qubits that it is connected to.
    final_det_ops = []
    for i in range(1, 2*distance-1, 2):
        ancilla_key = f'm_{rounds-1}_{i}'
        data_left = f'd_{i-1}'
        data_right = f'd_{i+1}'
        final_det_ops.append(stimcirq.DetAnnotation(
            parity_keys=[ancilla_key, data_left, data_right],
            coordinate_metadata=(i, rounds)
        ))
    circuit.append(final_det_ops, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        
    # Define Logical Observable
    # The observable is simply the first data qubit
    circuit.append(stimcirq.CumulativeObservableAnnotation(
        parity_keys=[f'd_0'],
        observable_index=0
    ), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    #END S8_SNIPPET_CIRQ_BUILDER
    
    # Compile to ultra-fast stim circuit
    return stimcirq.cirq_circuit_to_stim_circuit(circuit)


def sample_sinter_simulation(distances=None, physical_errors=None, num_workers=8, max_shots=1000000, max_errors=500):
    """
    Demonstrates how to run a large-scale parallel simulation using Sinter.
    """
    print("\n--- Sinter Large-Scale Simulation Demo ---")
    #BEGIN S8_SNIPPET_SINTER
    # Define the tasks we want Sinter to simulate
    tasks = []
    if distances is None:
        distances = [3, 5, 7]
    if physical_errors is None:
        physical_errors = np.linspace(0.01, 0.4, 20)
    
    for d in distances:
        for p in physical_errors:
            circuit = build_rep_code_cirq(distance=d, p=p)
            tasks.append(
                sinter.Task(
                    circuit=circuit,
                    json_metadata={'d': d, 'p': p}
                )
            )
            
    # Collect statistics using Sinter in parallel
    print("Starting Sinter parallel sampling...")
    stats = sinter.collect(
        num_workers=num_workers,
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=True
    )
    #END S8_SNIPPET_SINTER
    return stats, distances, physical_errors

def plot_sinter_simulation(stats, distances, physical_errors):
    """
    Plots the results returned by sample_sinter_simulation.
    """
    # Plot the results
    plt.figure(figsize=(8, 6))
    for d in distances:
        # Filter stats for this distance
        d_stats = [s for s in stats if s.json_metadata['d'] == d]
        # Sort by physical error rate so matplotlib draws the line continuously
        d_stats.sort(key=lambda s: s.json_metadata['p'])
        
        p_rates = [s.json_metadata['p'] for s in d_stats]
        l_rates = [s.errors / s.shots if s.shots > 0 else 0 for s in d_stats]
        plt.plot(p_rates, l_rates, marker='o', label=f'Distance {d}')
        
    plt.plot(physical_errors, physical_errors, linestyle='--', color='gray', label='No QEC')
    plt.xlabel('Physical Error Rate (p)')
    plt.ylabel('Logical Error Rate ($p_L$)')
    plt.title('Sinter Simulation: Repetition Code Performance')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

if __name__ == "__main__":
    print("Validating Sinter setup with a small text run...")
    stats, dists, errs = sample_sinter_simulation(distances=[3], physical_errors=[0.01, 0.1], max_shots=10000, num_workers=1)
    print("\nSimulation successful! Results:")
    for s in stats:
        print(s)
