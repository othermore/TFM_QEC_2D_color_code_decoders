import pathlib
import sys
from typing import Dict, List, Set, Tuple

import numpy as np
import pymatching
import sinter
import stim

class RestrictionDecoder(sinter.Decoder):
    """
    Decodificador Restriction para Color Code.
    """

    def __init__(self):
        self.compiled_cache = {}
        self.detector_colors = {}

    def _compile_decoder(self, dem: stim.DetectorErrorModel, num_dets: int, cache_key: str):
        if cache_key in self.compiled_cache:
            return
            
        color_map = {0: 'R', 1: 'G', 2: 'B', 3: 'R', 4: 'G', 5: 'B'}
        self.detector_colors = {}
        for instr in dem:
            if instr.type == "detector":
                d = instr.targets_copy()[0].val
                c_idx = int(instr.args_copy()[3])
                self.detector_colors[d] = color_map[c_idx]
        
        import simulator
        dem = simulator.split_Y_errors_by_coords(dem)
        
        # Prepare color masks for the detectors
        mask_RG = np.ones(num_dets, dtype=bool)
        mask_RB = np.ones(num_dets, dtype=bool)
        
        for d_idx, color in self.detector_colors.items():
            if color == 'B':
                mask_RG[d_idx] = False
            elif color == 'G':
                mask_RB[d_idx] = False

        # Prepare a list of qubit errors in the DEM, 
        # each with a tuple of detectors and observables.
        true_qubits = []
        for instr in dem:
            if instr.type == "error":
                dets = [t.val for t in instr.targets_copy() if t.is_relative_detector_id()]
                obs = [t.val for t in instr.targets_copy() if t.is_logical_observable_id()]
                true_qubits.append((dets, obs))

        # Create a list of edges for each color            
        edges_RG = []
        edges_RB = []
        # And store, for each qubit, the index of the edge it belongs to
        true_q_to_edge_RG = []
        true_q_to_edge_RB = []
        
        for dets, _ in true_qubits:
            kept_RG = tuple(sorted([d for d in dets if self.detector_colors.get(d) != 'B']))
            if len(kept_RG) > 0 and kept_RG not in edges_RG:
                edges_RG.append(kept_RG)
            true_q_to_edge_RG.append(edges_RG.index(kept_RG) if len(kept_RG) > 0 else -1)
            
            kept_RB = tuple(sorted([d for d in dets if self.detector_colors.get(d) != 'G']))
            if len(kept_RB) > 0 and kept_RB not in edges_RB:
                edges_RB.append(kept_RB)
            true_q_to_edge_RB.append(edges_RB.index(kept_RB) if len(kept_RB) > 0 else -1)
    
        # Create a new DEM for each restricted graph
        def create_match(edges):
            lines = []
            for i, kept in enumerate(edges):
                # Set a fixed error likelyhood, which is OK for a code-capacity model
                # For more complex models, we would need to weight based on the 
                # probability of the original errors in the global DEM
                # Define a new virtual observable for this edge, so MWPM returns
                # all the flipped edges as predicted observable
                lines.append(f"error(0.01) {' '.join(f'D{d}' for d in kept)} L{i}")
            for instr in dem:
                # Add the original detectors from the DEM
                if instr.type == "detector":
                    lines.append(str(instr))
            return pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel("\n".join(lines)))
            
        match_RG = create_match(edges_RG)
        match_RB = create_match(edges_RB)
        
        # Create a lifting graph, which is an inverted graph that we will
        # use to reconstruct the logical predictions from the 
        # two restricted decoders
        #  - Edges in the restricted graph (projected physical qubits 
        #    connecting two detectors), become nodes
        #  - Physical qubits (triangular faces in the restricted graph), become edges
        #    connecting the nodes that represent the projected physical qubits

        lifting_lines = []
        num_RG_edges = len(edges_RG)
        
        for q, (dets, _) in enumerate(true_qubits):
            e_rg = true_q_to_edge_RG[q]
            e_rb = true_q_to_edge_RB[q]
            
            nodes = []
            # Define the detectors (which are edges in the restricted graphs)
            if e_rg != -1: nodes.append(f"D{e_rg}")
            if e_rb != -1: nodes.append(f"D{num_RG_edges + e_rb}")
            # Add a "virtual" physical qubit error for the two detectors
            # and assing the unique qubit index (q) as the observable
            if len(nodes) > 0:
                lifting_lines.append(f"error(0.01) {' '.join(nodes)} L{q}")
        
        # Declare all the detectors in the lifting graph
        num_Lifting_nodes = num_RG_edges + len(edges_RB)
        for i in range(num_Lifting_nodes):
            lifting_lines.append(f"detector(0, 0, 0) D{i}")
        
        lifting_dem = stim.DetectorErrorModel("\n".join(lifting_lines))
        lifting_match = pymatching.Matching.from_detector_error_model(lifting_dem)
        
        L_matrix = np.zeros((dem.num_observables, len(true_qubits)), dtype=np.uint8)
        for q, (_, obs) in enumerate(true_qubits):
            for o in obs:
                L_matrix[o, q] = 1

        self.compiled_cache[cache_key] = {
            'mask_RG': mask_RG,
            'mask_RB': mask_RB,
            'match_RG': match_RG,
            'match_RB': match_RB,
            'lifting_match': lifting_match,
            'L_matrix': L_matrix
        }

    def _execute_decode(self, compiled_data, shots, syndromes):
        num_qubits = compiled_data['L_matrix'].shape[1]
        corrections = np.zeros((shots, num_qubits), dtype=bool)
        
        mask_RG = compiled_data['mask_RG']
        mask_RB = compiled_data['mask_RB']
        match_RG = compiled_data['match_RG']
        match_RB = compiled_data['match_RB']
        lifting_match = compiled_data['lifting_match']
        L_matrix = compiled_data['L_matrix']

        for i in range(shots):
            syn = syndromes[i]
            # We deactivate from the syndrome the active detectors for the
            # color that has been restricted, using the defined masks
            syn_RG = syn & mask_RG
            syn_RB = syn & mask_RB
            
            p_RG = match_RG.decode(syn_RG)
            p_RB = match_RB.decode(syn_RB)
            
            lifting_syn = np.concatenate([p_RG, p_RB])
            # We use MWPM again to find the true physical qubits 
            # (which act as EDGES for this MWPM run, generally 
            # representing physical qubits in MWPM decoding)
            # that connect the active edges (which act as NODES/DETECTORS
            # in this run). 
            # This is equivalent to finding the minimal set of qubits
            # whose local boundaries (local meaning that their edges include the 
            # red node that connects the RG and RB edges) match the active edges 
            # found by the two restricted decoders.
            try:
                correction = lifting_match.decode(lifting_syn)
            except ValueError:
                correction = np.zeros(num_qubits, dtype=bool)
                
            corrections[i] = correction
            
        return corrections

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: pathlib.Path,
        dets_b8_in_path: pathlib.Path,
        obs_predictions_b8_out_path: pathlib.Path,
        tmp_dir: pathlib.Path,
    ) -> None:
        dem = stim.DetectorErrorModel.from_file(dem_path)
        cache_key = str(dem_path)
        
        if cache_key not in self.compiled_cache:
            self._compile_decoder(dem, num_dets, cache_key)
            
        syndromes = np.fromfile(dets_b8_in_path, dtype=np.uint8)
        bytes_per_shot = (num_dets + 7) // 8
        syndromes = syndromes.reshape(num_shots, bytes_per_shot)
        syndromes_unpacked = np.unpackbits(syndromes, axis=1, bitorder='little')[:, :num_dets]
        
        # Execute the decoding for all shots
        corrections = self._execute_decode(self.compiled_cache[cache_key], num_shots, syndromes_unpacked)
        
        # Evaluate Logical Observables - Basically, capture for each observable, the 
        # parity of the possible errors that would affect this observable
        logical_preds = (corrections @ self.compiled_cache[cache_key]['L_matrix'].T) % 2
        
        # Write output
        logical_preds_packed = np.packbits(logical_preds, axis=1, bitorder='little')
        with open(obs_predictions_b8_out_path, 'wb') as f:
            f.write(logical_preds_packed.tobytes())

    def __getstate__(self):
        state = self.__dict__.copy()
        state['compiled_cache'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
