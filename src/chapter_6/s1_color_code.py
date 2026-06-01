import cirq
import stim
import stimcirq
import numpy as np
from typing import List, Tuple

class Shape:
    """Base class representing a topological polygon in the lattice."""
    def __init__(self, ox: int, oy: int, color: str):
        self.ox = ox
        self.oy = oy
        self.color = color

    def get_data_qubits(self) -> List[Tuple[int, int]]:
        """Returns the coordinates of the data qubits forming the perimeter."""
        raise NotImplementedError

    def get_ancilla_qubit(self) -> Tuple[int, int]:
        """Returns the coordinate of the central ancilla qubit."""
        raise NotImplementedError

class SquareShape(Shape):
    """Represents a 4-qubit square face."""
    def get_data_qubits(self) -> List[Tuple[int, int]]:
        dqs = [(0, 0), (0, 2), (2, 0), (2, 2)]
        return [(self.ox + dx, self.oy + dy) for dx, dy in dqs]

    def get_ancilla_qubit(self) -> Tuple[int, int]:
        return (self.ox + 1, self.oy + 1)

class OctagonShape(Shape):
    """Represents an 8-qubit octagon face."""
    def get_data_qubits(self) -> List[Tuple[int, int]]:
        dqs = [(2, 0), (4, 0), (0, 2), (6, 2), (0, 4), (6, 4), (2, 6), (4, 6)]
        return [(self.ox + dx, self.oy + dy) for dx, dy in dqs]

    def get_ancilla_qubit(self) -> Tuple[int, int]:
        return (self.ox + 3, self.oy + 3)

class TrapezoidShape(Shape):
    """Represents a boundary trapezoid face. Orientation specifies the boundary side."""
    def __init__(self, ox: int, oy: int, color: str, orientation: str):
        super().__init__(ox, oy, color)
        self.orientation = orientation

    def get_data_qubits(self) -> List[Tuple[int, int]]:
        if self.orientation == 'BOTTOM':
            dqs = [(0, 0), (6, 0), (2, 2), (4, 2)]
        elif self.orientation == 'LEFT':
            dqs = [(0, 0), (2, 0), (4, 2), (4, 4)]
        elif self.orientation == 'RIGHT':
            dqs = [(4, 0), (2, 0), (0, 2), (0, 4)]
        else:
            raise ValueError(f"Unknown orientation {self.orientation}")
        return [(self.ox + dx, self.oy + dy) for dx, dy in dqs]

    def get_ancilla_qubit(self) -> Tuple[int, int]:
        if self.orientation == 'BOTTOM':
            anc = (3, 1)
        elif self.orientation == 'LEFT':
            anc = (1, 3)
        elif self.orientation == 'RIGHT':
            anc = (3, 3)
        else:
            raise ValueError(f"Unknown orientation {self.orientation}")
        return (self.ox + anc[0], self.oy + anc[1])

class Lattice:
    """Base class for geometric lattice generation."""
    def __init__(self, distance: int):
        self.distance = distance
        self.shapes: List[Shape] = []
        self.unique_data_coords = set()
        self.faces_data: List[dict] = []
        
        self.build()
        self._compile_coordinates()

    def build(self):
        """Constructs the geometric shapes of the lattice. Implemented by subclasses."""
        raise NotImplementedError

    def _compile_coordinates(self):
        """Aggregates all shapes to extract unique data qubits and define face relationships."""
        for shape in self.shapes:
            dqs = shape.get_data_qubits()
            anc = shape.get_ancilla_qubit()
            self.unique_data_coords.update(dqs)
            self.faces_data.append({
                'anc_coord': anc,
                'data_coords': dqs,
                'color': shape.color
            })

    def plot_lattice(self, save_path=None):
        """Visualizes the color code lattice configuration."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        plt.rcParams.update({
            "font.family": "serif",
            "mathtext.fontset": "cm"
        })

        fig_size = self.distance * 2 + 2
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        # Scale fonts and markers proportionally to distance
        title_fontsize = 14 + (self.distance - 3) * 2
        coord_fontsize = 8 + (self.distance - 3) * 0.4
        ancilla_fontsize = 9 + (self.distance - 3) * 0.4
        data_markersize = 8 + (self.distance - 3) * 0.6
        anc_markersize = 8 + (self.distance - 3) * 0.6
        
        for coord in self.unique_data_coords:
            cx, cy = coord
            ax.plot(cx, cy, marker='o', markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5, markersize=data_markersize, zorder=5)
            ax.text(cx + 0.15, cy + 0.15, f"({cx},{cy})", fontsize=coord_fontsize, color='black', weight='bold', zorder=6)

        color_map = {'R': 'red', 'G': 'green', 'B': 'blue'}
        for shape in self.shapes:
            c = color_map.get(shape.color, 'gray')
            ax_c, ay_c = shape.get_ancilla_qubit()
            
            face_coords = shape.get_data_qubits()
            avg_x = sum(x for x, y in face_coords) / len(face_coords)
            avg_y = sum(y for x, y in face_coords) / len(face_coords)
            
            if isinstance(shape, TrapezoidShape) and shape.orientation in ['LEFT', 'RIGHT']:
                plot_x, plot_y = avg_x, avg_y
            else:
                plot_x, plot_y = ax_c, ay_c
            
            ax.plot(plot_x, plot_y, marker='s', color=c, markersize=anc_markersize, zorder=5)
            ax.text(plot_x, plot_y + 0.3, f"A:({ax_c},{ay_c})", fontsize=ancilla_fontsize, color=c, weight='bold', ha='center', va='bottom', zorder=6)
            
            face_coords_sorted = sorted(face_coords, key=lambda coord: np.arctan2(coord[1] - avg_y, coord[0] - avg_x))
            poly = patches.Polygon(face_coords_sorted, closed=True, facecolor=c, alpha=0.3, edgecolor=c, linewidth=2)
            ax.add_patch(poly)
            
            for cx, cy in face_coords_sorted:
                ax.plot([plot_x, cx], [plot_y, cy], color=c, alpha=0.4, linewidth=1, linestyle='--')

        ax.set_aspect('equal')
        ax.axis('off')
        plt.title(f"Color Code Lattice - Distance {self.distance}", fontsize=title_fontsize, pad=20)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Lattice saved to {save_path}")
        plt.show()

class Lattice488(Lattice):
    """
    Implements the 4.8.8 Color Code topological layout.
    """
    def build(self):
        """
        Builds the 4.8.8 lattice using a procedural level-based construction.
        The lattice is built by first placing the squares in a pyramid structure.
        Then, the gaps between these squares are filled with octagons.
        Finally, the left, right, and bottom boundaries are capped with trapezoids
        to ensure all boundary data qubits are protected by appropriate stabilizers.
        Depending on the distance, octogons and trapezoids have different colors,
        which alternates between distances (e.g. at d=3 the octogons in the first level 
        are green, but at d=5 octogons in the first level are blue and they are 
        green in the second level)
        All rules scale automatically for any valid distance.
        """
        N = (self.distance - 1) // 2
        
        for level in range(1, N + 1):
            num_squares = N - level + 1
            for idx in range(num_squares):
                ox = 4 + 4 * (level - 1) + 8 * idx
                oy = 2 + 4 * (level - 1)
                self.shapes.append(SquareShape(ox, oy, 'R'))

        for level in range(1, N):
            num_octagons = N - level
            for idx in range(num_octagons):
                ox = 6 + 4 * (level - 1) + 8 * idx
                oy = 0 + 4 * (level - 1)
                color_l1 = 'G' if self.distance % 4 == 1 else 'B'
                current_color = color_l1 if level % 2 == 1 else ('B' if color_l1 == 'G' else 'G')
                self.shapes.append(OctagonShape(ox, oy, current_color))

        start_left = 1 if self.distance % 4 == 3 else 2
        for level in range(start_left, N + 1, 2):
            sq_ox = 4 + 4 * (level - 1)
            sq_oy = 2 + 4 * (level - 1)
            self.shapes.append(TrapezoidShape(sq_ox - 4, sq_oy - 2, 'B', 'LEFT'))

        start_right = 1 if self.distance % 4 == 1 else 2
        for level in range(start_right, N + 1, 2):
            sq_ox = 4 + 4 * (level - 1) + 8 * (N - level)
            sq_oy = 2 + 4 * (level - 1)
            self.shapes.append(TrapezoidShape(sq_ox + 2, sq_oy - 2, 'G', 'RIGHT'))

        for idx in range(N):
            sq_ox = 4 + 8 * idx
            sq_oy = 2
            bottom_color = 'B' if self.distance % 4 == 1 else 'G'
            self.shapes.append(TrapezoidShape(sq_ox - 2, sq_oy - 2, bottom_color, 'BOTTOM'))

class ColorCode:
    """
    Base class for dynamically generating 2D Color Code circuits in Cirq,
    with support for noise injection and STIM compilation.
    """

    class NoiseModel(cirq.NoiseModel):
        """
        Custom Cirq Noise Model that supports independent error rates for 
        data qubits (code-capacity), measurements, and physical gates.
        """
        def __init__(self, p_data: float = 0.0, p_meas: float = 0.0, p_gate: float = 0.0, noise_type: str = 'depolarizing'):
            self.p_data = p_data
            self.p_meas = p_meas
            self.p_gate = p_gate
            self.noise_type = noise_type

        def noisy_operation(self, operation: cirq.Operation):
            if isinstance(operation.untagged, (stimcirq.DetAnnotation, stimcirq.CumulativeObservableAnnotation)):
                yield operation
                return

            if cirq.is_measurement(operation):
                if self.p_meas > 0:
                    yield cirq.bit_flip(p=self.p_meas).on_each(*operation.qubits)
                yield operation
                
            elif operation.gate == cirq.I:
                yield operation
                if self.p_data > 0:
                    if self.noise_type == 'X':
                        yield cirq.bit_flip(p=self.p_data).on_each(*operation.qubits)
                    else:
                        yield cirq.depolarize(p=self.p_data).on_each(*operation.qubits)
                    
            elif len(operation.qubits) == 1:
                yield operation
                if self.p_gate > 0:
                    if self.noise_type == 'X':
                        yield cirq.bit_flip(p=self.p_gate).on_each(*operation.qubits)
                    else:
                        yield cirq.depolarize(p=self.p_gate).on_each(*operation.qubits)
                    
            elif len(operation.qubits) == 2:
                yield operation
                if self.p_gate > 0:
                    if self.noise_type == 'X':
                        yield cirq.bit_flip(p=self.p_gate).on_each(*operation.qubits)
                    else:
                        yield cirq.depolarize(p=self.p_gate).on_each(*operation.qubits)
            else:
                yield operation

    def __init__(self, distance: int, lattice_type: str = "4.8.8"):
        if distance % 2 == 0 or distance < 3:
            raise ValueError("Distance must be an odd integer >= 3")
            
        self.distance = distance
        self.lattice_type = lattice_type
        
        if lattice_type == "4.8.8":
            self.lattice = Lattice488(distance)
        else:
            raise ValueError(f"Unknown lattice type: {lattice_type}")
            
        self.data_qubits: List[cirq.GridQubit] = []
        self.ancilla_qubits: List[cirq.GridQubit] = []
        self.faces: List[dict] = []
        self.plot_coords = {}
        
        self._map_qubits()

    def _map_qubits(self):
        """Maps geometric coordinates from the lattice into Cirq GridQubits."""
        sorted_coords = sorted(list(self.lattice.unique_data_coords), key=lambda c: (c[1], c[0]))
        coord_map = {}
        
        for i, (cx, cy) in enumerate(sorted_coords):
            new_q = cirq.GridQubit(int(cy), int(cx))
            coord_map[(cx, cy)] = new_q
            self.data_qubits.append(new_q)
            self.plot_coords[new_q] = (cx, cy)

        for i, face in enumerate(self.lattice.faces_data):
            cx, cy = face['anc_coord']
            new_a = cirq.GridQubit(int(cy), int(cx))
            self.ancilla_qubits.append(new_a)
            self.plot_coords[new_a] = (cx, cy)
            
            self.faces.append({
                'ancilla': new_a,
                'data': [coord_map[dc] for dc in face['data_coords']],
                'color': face['color']
            })

    def build_clean_circuit(self, rounds: int = 1, noise_type: str = 'depolarizing') -> cirq.Circuit:
        """
        Builds the clean Cirq circuit, including STIM annotations for decoding.
        
        Syndrome extraction is done sequentially for X stabilizers and Z stabilizers.
        Detectors are added using stimcirq.DetAnnotation to declare logical parity checks.
        Specifically:
        - X Detectors compare the X syndrome of the current round with the previous round.
          They are named 'mX_{round}_{col}' and are declared after the X measurements.
          Because data qubits are initialized in |0> (an X superposition), the first round 
          X measurements are random and just establish the baseline. Thus, X detectors 
          only start from the second round (r > 0).
        - Z Detectors compare the Z syndrome of the current round with the previous round.
          They are named 'mZ_{round}_{col}' and are declared after the Z measurements.
          Because Z data qubits are initialized in |0>, the first round Z measurements
          are deterministic and therefore form a detector without a previous round comparison.
        - Boundary Detectors close the temporal boundary at the end of the circuit by comparing
          the final Z syndrome with the parity of the destructive data qubit measurements.
          Data measurements are named 'd_{row}_{col}'.
        
        Finally, the logical observable Z_L is declared using stimcirq.CumulativeObservableAnnotation.
        It is defined transversally as the product of all data qubit measurements at the end.
        
        To prevent Z errors in the first noisy round from being hidden due to the randomness
        of |0> initialization, the circuit actually executes `rounds + 1` cycles.
        Cycle r=0 is a noiseless preparation round to establish the X syndrome baseline.
        Noise is injected from r=1 to r=rounds.
        """
        circuit = cirq.Circuit()
        
        for r in range(rounds + 1):
            if r > 0:
                circuit.append([cirq.I(q) for q in self.data_qubits], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
            
            if noise_type != 'X':
                for face in self.faces:
                    a = face['ancilla']
                    circuit.append(cirq.H(a), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
                    for dq in face['data']:
                        circuit.append(cirq.CNOT(a, dq), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
                    circuit.append(cirq.H(a), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
                    
                meas_x = [cirq.measure(f['ancilla'], key=f"mX_{r}_{f['ancilla'].row}_{f['ancilla'].col}") for f in self.faces]
                circuit.append(cirq.Moment(meas_x))
                
                # X outcomes in round 0 are random due to |0> initialization, so we only 
                # create X detectors starting from round 1 to compare against previous round.
                if r > 0:
                    det_ops = []
                    for face in self.faces:
                        ax, ay = self.plot_coords[face['ancilla']]
                        det_ops.append(stimcirq.DetAnnotation(
                            parity_keys=[f"mX_{r}_{face['ancilla'].row}_{face['ancilla'].col}", f"mX_{r-1}_{face['ancilla'].row}_{face['ancilla'].col}"],
                            coordinate_metadata=(ax, ay, r)
                        ))
                    circuit.append(cirq.Moment(det_ops))
                        
                circuit.append(cirq.Moment([cirq.reset(q) for q in self.ancilla_qubits]))
            
            for face in self.faces:
                a = face['ancilla']
                for dq in face['data']:
                    circuit.append(cirq.CNOT(dq, a), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
                    
            meas_z = [cirq.measure(f['ancilla'], key=f"mZ_{r}_{f['ancilla'].row}_{f['ancilla'].col}") for f in self.faces]
            circuit.append(cirq.Moment(meas_z))
            
            det_ops_z = []
            for face in self.faces:
                ax, ay = self.plot_coords[face['ancilla']]
                keys = [f"mZ_{r}_{face['ancilla'].row}_{face['ancilla'].col}"]
                if r > 0:
                    keys.append(f"mZ_{r-1}_{face['ancilla'].row}_{face['ancilla'].col}")
                det_ops_z.append(stimcirq.DetAnnotation(
                    parity_keys=keys,
                    coordinate_metadata=(ax, ay, r)
                ))
            circuit.append(cirq.Moment(det_ops_z))
                
            circuit.append(cirq.Moment([cirq.reset(q) for q in self.ancilla_qubits]))

        meas_d = [cirq.measure(q, key=f'd_{q.row}_{q.col}') for q in self.data_qubits]
        circuit.append(cirq.Moment(meas_d))
        
        final_dets = []
        for face in self.faces:
            ax, ay = self.plot_coords[face['ancilla']]
            keys = [f"mZ_{rounds}_{face['ancilla'].row}_{face['ancilla'].col}"]
            keys.extend([f'd_{dq.row}_{dq.col}' for dq in face['data']])
            final_dets.append(stimcirq.DetAnnotation(
                parity_keys=keys,
                coordinate_metadata=(ax, ay, rounds + 1)
            ))
        circuit.append(cirq.Moment(final_dets))
                
        obs_keys = [f'd_{q.row}_{q.col}' for q in self.data_qubits]
        circuit.append(cirq.Moment([stimcirq.CumulativeObservableAnnotation(
            parity_keys=obs_keys,
            observable_index=0
        )]))
        
        return circuit

    def get_noisy_circuit(self, rounds: int = 1, p_data: float = 0.01, p_meas: float = 0.0, p_gate: float = 0.0, noise_type: str = 'depolarizing') -> cirq.Circuit:
        """
        Builds the circuit and applies the custom NoiseModel.
        Default is Code-Capacity (1:0:0) using p_data.
        """
        clean_circ = self.build_clean_circuit(rounds, noise_type)
        noise_model = self.NoiseModel(p_data=p_data, p_meas=p_meas, p_gate=p_gate, noise_type=noise_type)
        return clean_circ.with_noise(noise_model)

    def get_stim_circuit(self, rounds: int = 1, p_data: float = 0.01, p_meas: float = 0.0, p_gate: float = 0.0, noise_type: str = 'depolarizing') -> stim.Circuit:
        """Returns the STIM compiled version of the noisy circuit, ready for Sinter/PyMatching."""
        noisy_circ = self.get_noisy_circuit(rounds, p_data, p_meas, p_gate, noise_type)
        return stimcirq.cirq_circuit_to_stim_circuit(noisy_circ)

    def draw_circuit(self, rounds: int = 1, noise_type: str = 'depolarizing'):
        """Prints the ASCII representation of the clean circuit."""
        circuit = self.build_clean_circuit(rounds, noise_type)
        print(circuit)

    def plot_lattice(self, save_path=None):
        """Delegates lattice visualization to the underlying lattice object."""
        self.lattice.plot_lattice(save_path)

if __name__ == "__main__":
    import stimcirq
    
    print("=== Color Code 4.8.8 Generation Summary ===")
    d = 3
    rounds = 1
    cc = ColorCode(distance=d, lattice_type="4.8.8")
    
    stim_circ = cc.get_stim_circuit(rounds=rounds)
    
    print(f"Distance: {d}")
    print(f"Rounds: {rounds}")
    print(f"Number of Data Qubits: {len(cc.data_qubits)}")
    print(f"Number of Ancilla Qubits: {len(cc.ancilla_qubits)}")
    print(f"Total Qubits: {len(cc.data_qubits) + len(cc.ancilla_qubits)}")
    print(f"Number of Detectors: {stim_circ.num_detectors}")
    print(f"Number of Observables: {stim_circ.num_observables}")
    
    print("\nData Qubits Summary:")
    for q in cc.data_qubits:
        print(f"Data {q}: Coordinates {cc.plot_coords[q]}")
        
    print("\nAncilla Qubits and Watched Data Qubits:")
    for face in cc.faces:
        anc = face['ancilla']
        print(f"Ancilla {anc} (Color: {face['color']}): Coordinates {cc.plot_coords[anc]}")
        print(f"  Watches Data Qubits: {[str(dq) for dq in face['data']]}")
        
    print("\nSummary complete. Use the ColorCode class in Notebooks to generate and plot circuits.")
