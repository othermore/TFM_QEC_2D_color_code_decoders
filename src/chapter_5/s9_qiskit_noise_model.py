import matplotlib.pyplot as plt
import numpy as np
import sinter
import cirq
import stimcirq
import stim

try:
    from qiskit.providers.fake_provider import FakeManilaV2
except ImportError:
    from qiskit.providers.fake_provider import GenericBackendV2 as FakeManilaV2

#BEGIN S9_SNIPPET_EXTRACT
def get_average_error_rates(backend):
    """Extracts average error rates from a Qiskit backend."""
    target_1q = []
    target_2q = []
    target_meas = []
    
    for q in range(backend.num_qubits):
        meas_props = backend.target.get("measure", None)
        if meas_props is not None and (q,) in meas_props:
            target_meas.append(meas_props[(q,)].error)
            
        sx_props = backend.target.get("sx", None)
        if sx_props is not None and (q,) in sx_props:
            target_1q.append(sx_props[(q,)].error)

    cx_props = backend.target.get("cx", None)
    if cx_props is not None:
        for edge in cx_props:
            target_2q.append(cx_props[edge].error)
            
    p_1q = np.mean(target_1q) if target_1q else 0.001
    p_2q = np.mean(target_2q) if target_2q else 0.01
    p_meas = np.mean(target_meas) if target_meas else 0.02
    
    return p_1q, p_2q, p_meas
#END S9_SNIPPET_EXTRACT

#BEGIN S9_SNIPPET_MODEL
class IBMNoiseModel(cirq.NoiseModel):
    """Custom Cirq noise model applying extracted error rates."""
    def __init__(self, p_1q, p_2q, p_meas, scale=1.0):
        self.p_1q = min(p_1q * scale, 1.0)
        self.p_2q = min(p_2q * scale, 1.0)
        self.p_meas = min(p_meas * scale, 1.0)

    def noisy_operation(self, operation: cirq.Operation):
        if isinstance(operation, (stimcirq.DetAnnotation, stimcirq.CumulativeObservableAnnotation)):
            yield operation
            return

        if cirq.is_measurement(operation):
            yield cirq.bit_flip(p=self.p_meas).on_each(*operation.qubits)
            yield operation
        elif len(operation.qubits) == 1:
            yield operation
            yield cirq.depolarize(p=self.p_1q).on_each(*operation.qubits)
        elif len(operation.qubits) == 2:
            yield operation
            yield cirq.depolarize(p=self.p_2q).on_each(*operation.qubits)
        else:
            yield operation
#END S9_SNIPPET_MODEL

def build_clean_rep_code_cirq(distance: int) -> cirq.Circuit:
    """Builds a clean 1D Bit-Flip Repetition Code using Cirq without noise."""
    qubits = cirq.LineQubit.range(2 * distance - 1)
    circuit = cirq.Circuit()
    rounds = distance
    
    for r in range(rounds):
        circuit.append([cirq.CNOT(qubits[i-1], qubits[i]) for i in range(1, 2*distance-1, 2)], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuit.append([cirq.CNOT(qubits[i+1], qubits[i]) for i in range(1, 2*distance-1, 2)], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        
        circuit.append([cirq.measure(qubits[i], key=f'm_{r}_{i}') for i in range(1, 2*distance-1, 2)], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        
        det_ops = []
        for i in range(1, 2*distance-1, 2):
            key = f'm_{r}_{i}'
            if r == 0:
                det_ops.append(stimcirq.DetAnnotation(parity_keys=[key], coordinate_metadata=(i, r)))
            else:
                prev_key = f'm_{r-1}_{i}'
                det_ops.append(stimcirq.DetAnnotation(parity_keys=[key, prev_key], coordinate_metadata=(i, r)))
        circuit.append(det_ops, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
                
        circuit.append([cirq.reset(qubits[i]) for i in range(1, 2*distance-1, 2)], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    circuit.append([cirq.measure(qubits[i], key=f'd_{i}') for i in range(0, 2*distance-1, 2)], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        
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
        
    circuit.append(stimcirq.CumulativeObservableAnnotation(
        parity_keys=[f'd_0'],
        observable_index=0
    ), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    
    return circuit

def sample_sinter_qiskit_noise(distances=None, scales=None, num_workers=8, max_shots=1000000, max_errors=500):
    """Runs a parallel simulation scaling the base Qiskit noise model."""
    try:
        backend = FakeManilaV2()
    except Exception:
        from qiskit.providers.fake_provider import GenericBackendV2
        backend = GenericBackendV2(num_qubits=5)
        
    base_1q, base_2q, base_meas = get_average_error_rates(backend)
    
    tasks = []
    if distances is None:
        distances = [3, 5, 7]
    if scales is None:
        scales = np.linspace(0.1, 15.0, 15)
    
    for d in distances:
        clean_circuit = build_clean_rep_code_cirq(distance=d)
        for scale in scales:
            #BEGIN S9_SNIPPET_WITH_NOISE
            noise_model = IBMNoiseModel(base_1q, base_2q, base_meas, scale=scale)
            noisy_circuit = clean_circuit.with_noise(noise_model)
            stim_circuit = stimcirq.cirq_circuit_to_stim_circuit(noisy_circuit)
            #END S9_SNIPPET_WITH_NOISE
            
            tasks.append(
                sinter.Task(
                    circuit=stim_circuit,
                    json_metadata={'d': d, 'scale': scale, 'p_2q': base_2q * scale}
                )
            )
            
    stats = sinter.collect(
        num_workers=num_workers,
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=True
    )
    return stats, distances, scales

def plot_sinter_simulation(stats, distances):
    """Plots the results returned by the Sinter simulation."""
    plt.figure(figsize=(8, 6))
    for d in distances:
        d_stats = [s for s in stats if s.json_metadata['d'] == d]
        d_stats.sort(key=lambda s: s.json_metadata['p_2q'])
        
        p_rates = [s.json_metadata['p_2q'] for s in d_stats]
        l_rates = [s.errors / s.shots if s.shots > 0 else 0 for s in d_stats]
        plt.plot(p_rates, l_rates, marker='o', label=f'Distance {d}')
        
    ref_rates = [s.json_metadata['p_2q'] for s in [s for s in stats if s.json_metadata['d'] == distances[0]]]
    ref_rates.sort()
    plt.plot(ref_rates, ref_rates, linestyle='--', color='gray', label='No QEC')
    
    plt.xlabel('Physical 2Q Error Rate ($p_{2Q}$)')
    plt.ylabel('Logical Error Rate ($p_L$)')
    plt.title('Sinter Simulation: Qiskit Scaled Noise Model')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

def main():
    print("Validating Sinter setup with scaled Qiskit noise...")
    stats, dists, scales = sample_sinter_qiskit_noise(
        distances=[3], 
        scales=[0.5, 1.0], 
        max_shots=10000, 
        num_workers=1
    )
    print("\nSimulation successful! Results:")
    for s in stats:
        print(s)

if __name__ == "__main__":
    main()
