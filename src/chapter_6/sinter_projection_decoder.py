import pathlib
import sinter
import stim
import numpy as np
import pymatching

class ProjectionDecoder(sinter.Decoder):
    """
    Strict Topological Projection Decoder (Delfosse 2014 / Stephens 2014).
    1. Projects DEM to 3 surface codes, merging parallel edges.
    2. Decodes with MWPM.
    3. Uses geometric disambiguation (cycles in the dual graph) via BFS graph coloring
       to find the true physical qubits inside the boundary.
    """
    def __init__(self):
        # Cache for detector colors
        self.color_maps = {}
        # Cache for compiled objects
        self.compiled_cache = {}

    def register_colors(self, num_detectors: int, colors: dict):
        self.color_maps[num_detectors] = colors

    def _get_edge(self, dets, c, det_colors):
        """
        Helper to project detectors onto a surface code by dropping color c.
        For a list of detectors, and a det to color mapping,
        returns a tuple of detectors that are not of the color c.
        """
        kept = tuple(sorted([d for d in dets if det_colors.get(d) != c]))
        return kept if len(kept) > 0 else None

    def _extract_projected_edges(self, dem: stim.DetectorErrorModel, det_colors: dict):
        """Extracts and merges parallel edges for the 3 surface code projections."""
        qubits_targets = []
        for instr in dem:
            if instr.type == "error":
                dets = [t.val for t in instr.targets_copy() if t.is_relative_detector_id()]
                obs = [t.val for t in instr.targets_copy() if t.is_logical_observable_id()]
                qubits_targets.append((dets, obs))
        
        edges_GB = []
        edges_BR = []
        edges_RG = []
        
        for dets, _ in qubits_targets:
            e_GB = self._get_edge(dets, 'R', det_colors)
            if e_GB is not None and e_GB not in edges_GB: edges_GB.append(e_GB)
            e_BR = self._get_edge(dets, 'G', det_colors)
            if e_BR is not None and e_BR not in edges_BR: edges_BR.append(e_BR)
            e_RG = self._get_edge(dets, 'B', det_colors)
            if e_RG is not None and e_RG not in edges_RG: edges_RG.append(e_RG)
            
        return qubits_targets, edges_GB, edges_BR, edges_RG

    def _build_dual_graph_and_bfs_tree(self, qubits_targets, edges_GB, edges_BR, edges_RG, det_colors):
        """Builds the dual graph of physical qubits and computes the BFS spanning tree matrix."""
        Q = len(qubits_targets)
        E_GB = len(edges_GB)
        E_BR = len(edges_BR)
        E_RG = len(edges_RG)
        E_total = E_GB + E_BR + E_RG
        
        # The owners of the edges are the two physical qubits that belong to this
        # edge in the projection (i.e. the two triangles that this edge divides)
        owner_GB = {i: [] for i in range(E_GB)}
        owner_BR = {i: [] for i in range(E_BR)}
        owner_RG = {i: [] for i in range(E_RG)}
        
        # For each of the three edges of a qubit, add to the owners 
        # the qubit.
        for q, (dets, _) in enumerate(qubits_targets):
            e_GB = self._get_edge(dets, 'R', det_colors)
            if e_GB is not None: owner_GB[edges_GB.index(e_GB)].append(q)
            e_BR = self._get_edge(dets, 'G', det_colors)
            if e_BR is not None: owner_BR[edges_BR.index(e_BR)].append(q)
            e_RG = self._get_edge(dets, 'B', det_colors)
            if e_RG is not None: owner_RG[edges_RG.index(e_RG)].append(q)
            
        OUTSIDE = Q
        # adj is a graph where nodes are the physical qubits (triangles in the dual graph)
        # plus a virtual node OUTSIDE (Q) for those edges that border only one qubit.
        # The edges in this graph are the virtual edges (in the projections) that border two qubits
        # Adj[i] will contain tuples of the form (neighbor, edge_index), meaning that qubit i 
        # is next to qubit neighbor across the virtual edge edge_index.
        adj = {i: [] for i in range(Q + 1)}
        
        def add_edges(owner, offset):
            for edge_idx, qs in owner.items():
                if len(qs) == 2:
                    q1, q2 = qs
                    adj[q1].append((q2, offset + edge_idx))
                    adj[q2].append((q1, offset + edge_idx))
                elif len(qs) == 1:
                    q1 = qs[0]
                    adj[q1].append((OUTSIDE, offset + edge_idx))
                    adj[OUTSIDE].append((q1, offset + edge_idx))
        
        add_edges(owner_GB, 0)
        add_edges(owner_BR, E_GB)
        add_edges(owner_RG, E_GB + E_BR)
        
        # BFS Spanning Tree matrix
        # T is a matrix where T[i, j] is 1 if edge j is in the BFS spanning tree of qubit i
        # meaning that the edge j is on the BFS-determined path from qubit i to the root (OUTSIDE)
        # T will allow determining the qubits inside a boundary by checking the parity of
        # the edges in the BFS path from each qubit to the boundary of the surface code.
        # If the parity is 1, the qubit is inside; if 0, it is outside.
        
        T = np.zeros((Q + 1, E_total), dtype=np.uint8)
        visited = set()
        components = []
        
        # We iterate on non-visited nodes as they may represent islands in the graph
        # although that will not happen for Color Codes. Each component is one of these islands
        # The first node visited will be the root (no edges need to be crossed to get to it)
        # and we will build from there.
        for start_node in range(Q + 1):
            if start_node in visited: continue
            comp_nodes = []
            queue = [start_node]
            visited.add(start_node)
            
            while queue:
                curr = queue.pop(0)
                comp_nodes.append(curr)
                for neighbor, edge_idx in adj[curr]:
                    # If this neighbor node has not been visited, we found a path to it
                    if neighbor not in visited:
                        visited.add(neighbor)
                        # The path is the path to the current node plus the edge between them
                        T[neighbor, :] = T[curr, :]
                        T[neighbor, edge_idx] ^= 1
                        queue.append(neighbor)
            components.append(comp_nodes)

        # The component that is connected with OUTSIDE is the main component
        # (in Color Codes, will be the only component) 
        # Any other component is stored in isolated_comps
        comp_with_outside = None
        isolated_comps = []
        for comp in components:
            if OUTSIDE in comp:
                comp_with_outside = comp
            else:
                isolated_comps.append(comp)
                
        return T, comp_with_outside, isolated_comps, OUTSIDE, Q

    def _create_matchings(self, dem, qubits_targets, edges_GB, edges_BR, edges_RG, det_colors):
        """Creates the PyMatching objects for the 3 surface codes."""
        def create_dem(color, edges_list):
            lines = []
            for dets, _ in qubits_targets:
                kept = self._get_edge(dets, color, det_colors)
                if kept is not None:
                    edge_idx = edges_list.index(kept)
                    # Use a nominal probability 0.01 for the merged edge to guide PyMatching
                    # This works with Code-Capacity noise models but would not be
                    # optimal for more complex noise models, in which case the weight
                    # would be calculated from the probability of error in the qubits that form this
                    # edge. That would be a Weighted Projection decoder.  
                    # IMPORTANT: we set edge IDs as logical observables
                    # to force pymatching to give us the list of errors instead of the predicted logical flips
                    lines.append(f"error(0.01) {' '.join(f'D{d}' for d in kept)} L{edge_idx}")
            for instr in dem:
                if instr.type == "detector":
                    lines.append(str(instr))
            return pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel("\n".join(lines)))

        match_GB = create_dem('R', edges_GB)
        match_BR = create_dem('G', edges_BR)
        match_RG = create_dem('B', edges_RG)
        return match_GB, match_BR, match_RG

    def _compile_decoder(self, dem: stim.DetectorErrorModel, num_dets: int, num_obs: int, cache_key: str):
        """
        Compiles and caches the geometrical dependencies and MWPM matching objects for a given DEM.
        Since Sinter executes millions of shots on the exact same circuit topology, this function 
        pre-computes all the heavy graph-theory structures exactly once per circuit to achieve 
        maximum performance during batch decoding.
        
        Args:
            dem: The `stim.DetectorErrorModel` generated from the quantum circuit. It describes 
                 how physical errors trigger changes in the measurements (syndromes).
            num_dets: The total number of detectors (stabilizer measurements) in the circuit.
                      Used to look up the previously registered color map (R/G/B).
            num_obs: The total number of logical observables.
            cache_key: A unique string identifier for this specific DEM (usually the file path). 
                       Used as the dictionary key to store and retrieve the compiled objects.
                       
        Returns:
            None. The function stores all the computed structures directly into the 
            `self.compiled_cache[cache_key]` dictionary for future reuse.
            
        Internal Behavior:
            1. Retrieves the pre-registered color mapping for the detectors.
            2. Calls `_extract_projected_edges` to find and merge parallel physical errors into 
               single mathematical edges for the 3 surface codes (GB, BR, RG).
            3. Calls `_build_dual_graph_and_bfs_tree` to build the geometrical dependencies matrix (T)
               and identify the isolated graph components used for topological disambiguation.
            4. Calls `_create_matchings` to generate the 3 `pymatching.Matching` objects.
            5. Pre-computes the logical observable matrix (L_matrix).
            6. Saves everything in the cache.
        """
        if num_dets not in self.color_maps:
            raise ValueError(f"No color map registered for num_detectors={num_dets}")
        det_colors = self.color_maps[num_dets]
        
        # extract the list of physical qubits and the list of merged edges (virtual qubits) for the 
        # three projections
        qubits_targets, edges_GB, edges_BR, edges_RG = self._extract_projected_edges(dem, det_colors)
        # From the dual_graph, build a spanning tree, a matrix that, if multiplied by the 
        # projected failing edges (pymatching detected failures in the three projections), 
        # will determine which qubits are inside the boundaries
        # comp_with_outside and isolated_comps give us the list of qubits that are
        # connected or isolated (all qubits will be connected for color codes)
        # OUTSIDE will be the number assigned to the "outside" qubit, the virtual qubit
        # connected to all the external boundaries of the dual graph. It will
        # match the total number of qubits
        T, comp_with_outside, isolated_comps, OUTSIDE, Q = self._build_dual_graph_and_bfs_tree(
            qubits_targets, edges_GB, edges_BR, edges_RG, det_colors
        )

        # From the DEM and the edges, we compute the 3 new DEM objects for each projection
        # and create a pymatching Matching object for each DEM
        match_GB, match_BR, match_RG = self._create_matchings(
            dem, qubits_targets, edges_GB, edges_BR, edges_RG, det_colors
        )
        
        E_GB = len(edges_GB)
        E_BR = len(edges_BR)
        E_RG = len(edges_RG)
        
        # This is the matrix where we will store the resulting logical observables
        L_matrix = np.zeros((num_obs, Q), dtype=np.uint8)
        for q, (_, obs) in enumerate(qubits_targets):
            for o in obs:
                L_matrix[o, q] = 1
            
        self.compiled_cache[cache_key] = {
            'match_GB': match_GB, 'match_BR': match_BR, 'match_RG': match_RG,
            'T': T, 'comp_with_outside': comp_with_outside, 'isolated_comps': isolated_comps,
            'E_GB': E_GB, 'E_BR': E_BR, 'E_RG': E_RG,
            'Q': Q, 'OUTSIDE': OUTSIDE, 'L_matrix': L_matrix, 'det_colors': det_colors
        }

    def _execute_mwpm(self, syndromes_unpacked, compiled_data, num_dets):
        """Runs MWPM batch decoding on the 3 projected surface codes."""
        det_colors = compiled_data['det_colors']
        mask_GB = np.ones(num_dets, dtype=bool)
        mask_BR = np.ones(num_dets, dtype=bool)
        mask_RG = np.ones(num_dets, dtype=bool)
        
        for d, c in det_colors.items():
            if c == 'R': mask_GB[d] = False
            elif c == 'G': mask_BR[d] = False
            elif c == 'B': mask_RG[d] = False
        
        # All the syndrome bits for detectors of the missing color are set to 0   
        syn_GB = syndromes_unpacked & mask_GB
        syn_BR = syndromes_unpacked & mask_BR
        syn_RG = syndromes_unpacked & mask_RG
        
        # Run the pymatching decoder stored in match_XX, with the corresponding
        # detected syndrome for each shot. The output will be the
        # predicted edge flips for each projection (because we defined each edge as a logical observable)
        p_GB = compiled_data['match_GB'].decode_batch(syn_GB)
        p_BR = compiled_data['match_BR'].decode_batch(syn_BR)
        p_RG = compiled_data['match_RG'].decode_batch(syn_RG)
        
        def pad(p, size):
            if p.shape[1] < size: return np.pad(p, ((0, 0), (0, size - p.shape[1])))
            return p[:, :size]
        
        # Sometimes there are missing (non-active) edges in the pymatching output, so we pad with zeros
        p_GB = pad(p_GB, compiled_data['E_GB'])
        p_BR = pad(p_BR, compiled_data['E_BR'])
        p_RG = pad(p_RG, compiled_data['E_RG'])
        # Merge all edges together to get the total logical flip matrix
        weights = np.hstack([p_GB, p_BR, p_RG]).astype(np.uint8)
        return weights

    def _execute_topological_lifting(self, weights, compiled_data, num_shots):
        """Translates the projected MWPM solutions into physical corrections via geometric graph coloring."""
        T = compiled_data['T']
        Q = compiled_data['Q']
        OUTSIDE = compiled_data['OUTSIDE']
        comp_with_outside = compiled_data['comp_with_outside']
        isolated_comps = compiled_data['isolated_comps']
        
        # Multiplying the T matrix by the edge weights tells us which qubits are inside 
        # the boundaries of the active edges
        regions = (weights @ T.T) % 2
        final_correction = np.zeros((num_shots, Q), dtype=np.uint8)
        
        if comp_with_outside is not None:
            # If the OUTSIDE qubit is in the region "1", then everything is inverted
            flip_outside = regions[:, OUTSIDE:OUTSIDE+1]
            comp_idxs = [n for n in comp_with_outside if n != OUTSIDE]
            if len(comp_idxs) > 0:
                final_correction[:, comp_idxs] = regions[:, comp_idxs] ^ flip_outside
        
        # For componets that are isolated, we can't tell wich region is the one that
        # is in error state, so we assume the smaller one is the one requiring a flip
        for comp in isolated_comps:
            c_slice = regions[:, comp]
            w1 = np.sum(c_slice, axis=1, keepdims=True)
            w0 = len(comp) - w1
            flip = (w0 < w1).astype(np.uint8)
            final_correction[:, comp] = c_slice ^ flip
            
        return final_correction

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
        
        # 1. Compilation Phase
        if cache_key not in self.compiled_cache:
            self._compile_decoder(dem, num_dets, num_obs, cache_key)
            
        compiled_data = self.compiled_cache[cache_key]
        
        # 2. Read syndromes

        # Decode the binary packed syndromes to a numpy array of shape (num_shots, num_dets)
        syndromes = np.fromfile(dets_b8_in_path, dtype=np.uint8)
        bytes_per_shot = (num_dets + 7) // 8 # Divide by 8 and round up
        syndromes = syndromes.reshape(num_shots, bytes_per_shot)
        syndromes_unpacked = np.unpackbits(syndromes, axis=1, bitorder='little')[:, :num_dets]
        
        # 3. MWPM Phase
        weights = self._execute_mwpm(syndromes_unpacked, compiled_data, num_dets)
        
        # 4. Topological Lifting Phase
        final_correction = self._execute_topological_lifting(weights, compiled_data, num_shots)
        
        # 5. Evaluate Logical Observables - Basically, capture for each observable, the 
        # parity of the possible errors that would affect this observable
        logical_preds = (final_correction @ compiled_data['L_matrix'].T) % 2
        
        # Write output
        logical_preds_packed = np.packbits(logical_preds, axis=1, bitorder='little')
        with open(obs_predictions_b8_out_path, 'wb') as f:
            f.write(logical_preds_packed.tobytes())
