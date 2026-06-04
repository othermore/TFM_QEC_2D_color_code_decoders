import pathlib
import sinter
import stim
import numpy as np
import concatmatching
import scipy.sparse as sps

try:
    from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices
except ImportError:
    detector_error_model_to_check_matrices = None

class ConcatMwpmDecoder(sinter.Decoder):
    """
    Sinter wrapper for the Concatenated MWPM Decoder by Seok-Hyung Lee.
    It takes a Stim DetectorErrorModel, extracts the parity check matrix H 
    and logical observables matrix L, and delegates decoding to ConcatMatching.
    """
    
    def __init__(self):
        # Cache the compiled concatmatching Decoder to avoid recompiling the graph for every task
        self.compiled_cache = {}
        self.color_maps = {}
        
    def _split_Y_errors_by_coords(self, dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
        """
        Takes a Stim DetectorErrorModel and returns a new one where all composite 
        Y errors are robustly split into independent X and Z error mechanisms.
        
        Why this is necessary:
        Stim's `decompose_errors=True` crashes for the 4.8.8 color code at d >= 7 because 
        bulk Y-errors are weight-6 hyperedges, and their X/Z components are weight-3. Stim 
        only decomposes into graphlike (weight <= 2) edges, so it fails and throws a ValueError.
        By manually pairing 'twin' X and Z detectors (which share the same space-time coordinates)
        we can perfectly split Y-errors into their independent X and Z components (max weight 3),
        which ConcatMatching natively supports.
        """
        # Extract coordinates directly from DEM
        coord_to_dets = {}
        for instr in dem:
            if instr.type == "detector":
                coord = tuple(instr.args_copy())
                d = instr.targets_copy()[0].val
                if coord not in coord_to_dets:
                    coord_to_dets[coord] = []
                coord_to_dets[coord].append(d)
                
        # Identify X and Z detectors. For color codes, X stabilizers are measured first, 
        # so the X detector has the smaller index of the twin pair.
        is_X = {}
        is_Z = {}
        for c, dets in coord_to_dets.items():
            if len(dets) == 2:
                is_X[min(dets)] = True
                is_Z[max(dets)] = True
                
        # Detect which graph the logical observable belongs to
        obs_in_X = False
        obs_in_Z = False
        for instr in dem:
            if instr.type == "error":
                targets = instr.targets_copy()
                has_obs = any(t.is_logical_observable_id() for t in targets)
                if has_obs:
                    has_X = any(t.val in is_X for t in targets if t.is_relative_detector_id())
                    has_Z = any(t.val in is_Z for t in targets if t.is_relative_detector_id())
                    if has_X and not has_Z:
                        obs_in_X = True
                    elif has_Z and not has_X:
                        obs_in_Z = True
                        
        new_dem = stim.DetectorErrorModel()
        for instr in dem:
            if instr.type == "error":
                p = instr.args_copy()[0]
                targets = instr.targets_copy()
                
                t_X = []
                t_Z = []
                t_U = []
                obs = []
                
                for t in targets:
                    if t.is_logical_observable_id():
                        obs.append(t)
                    elif t.is_relative_detector_id():
                        d = t.val
                        if d in is_X:
                            t_X.append(t)
                        elif d in is_Z:
                            t_Z.append(t)
                        else:
                            t_U.append(t)
                            
                # Reconstruct and split Y errors
                if len(t_X) > 0 and len(t_Z) == 0:
                    new_dem.append("error", [p], t_X + t_U + obs)
                elif len(t_Z) > 0 and len(t_X) == 0:
                    new_dem.append("error", [p], t_Z + t_U + obs)
                elif len(t_X) > 0 and len(t_Z) > 0:
                    # Y error split! Only attach the logical observable to the correct graph
                    if obs_in_X:
                        new_dem.append("error", [p], t_X + t_U + obs)
                        new_dem.append("error", [p], t_Z + t_U)
                    else:
                        new_dem.append("error", [p], t_X + t_U)
                        new_dem.append("error", [p], t_Z + t_U + obs)
                else:
                    new_dem.append("error", [p], t_U + obs)
            else:
                new_dem.append(instr)
                
        return new_dem

    def register_colors(self, num_detectors: int, colors: dict):
        self.color_maps[num_detectors] = colors

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
        
        # Load the DEM
        dem = stim.DetectorErrorModel.from_file(dem_path)
        
        # We cache the compiled decoder to save time across tasks for the same distance
        cache_key = str(dem_path)
        if cache_key not in self.compiled_cache:
            # Robustly split Y errors using coordinates
            split_dem = self._split_Y_errors_by_coords(dem)
            
            # Extract H, L, and ps from DEM using ldpc utility
            matrices = detector_error_model_to_check_matrices(
                split_dem, allow_undecomposed_hyperedges=True
            )
            H = matrices.check_matrix
            L = matrices.observables_matrix
            ps = matrices.priors
            
            # Get check colors
            det_colors = self.color_maps.get(num_dets, {})
            cmap = {'R': 0, 'G': 1, 'B': 2}
            check_colors = np.array([cmap.get(det_colors.get(i), 0) for i in range(num_dets)])
            
            # Instantiate the ConcatMatching decoder
            # ConcatMatching expects p (probabilities) as keyword argument. 
            decoder = concatmatching.Decoder(H, p=ps, check_colors=check_colors, comparison=False)
            
            self.compiled_cache[cache_key] = (decoder, L)
            
        decoder, L = self.compiled_cache[cache_key]
        
        # Read the syndromes (shots x num_detectors)
        shots = stim.read_shot_data_file(
            path=dets_b8_in_path, format="b8", num_detectors=num_dets
        )
            
        # Decode all shots
        # Since concatmatching.Decoder.decode expects a single 1D array syndrome at a time,
        # we will iterate through the shots.
        predictions = np.zeros((num_shots, num_obs), dtype=bool)
        for i in range(num_shots):
            preds_edges = decoder.decode(shots[i, :], full_output=False, verbose=False)
            # Map edge predictions to logical observables: O = (L @ e) % 2
            # Note: L is a scipy sparse matrix
            preds_logical = (L @ preds_edges) % 2
            predictions[i, :] = preds_logical
            
        # Write out the predictions
        stim.write_shot_data_file(
            data=predictions,
            path=obs_predictions_b8_out_path,
            format="b8",
            num_observables=num_obs,
        )
