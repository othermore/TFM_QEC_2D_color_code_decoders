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
        self.compiled_cache = {}
        self.detector_colors = {}

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
            color_map = {0: 'R', 1: 'G', 2: 'B', 3: 'R', 4: 'G', 5: 'B'}
            self.detector_colors = {}
            for instr in dem:
                if instr.type == "detector":
                    d = instr.targets_copy()[0].val
                    c_idx = int(instr.args_copy()[3])
                    self.detector_colors[d] = color_map[c_idx]

            # Robustly split Y errors using coordinates
            import simulator
            split_dem = simulator.split_Y_errors_by_coords(dem)
            
            # Extract H, L, and ps from DEM using ldpc utility
            matrices = detector_error_model_to_check_matrices(
                split_dem, allow_undecomposed_hyperedges=True
            )
            H = matrices.check_matrix
            L = matrices.observables_matrix
            ps = matrices.priors
            
            # Get check colors
            cmap = {'R': 0, 'G': 1, 'B': 2}
            check_colors = np.array([cmap.get(self.detector_colors.get(i), 0) for i in range(num_dets)])
            
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
