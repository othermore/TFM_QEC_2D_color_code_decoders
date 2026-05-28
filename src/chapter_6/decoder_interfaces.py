import os
from typing import Dict
import sinter

try:
    from ldpc import SinterBpOsdDecoder
except ImportError:
    SinterBpOsdDecoder = None


def get_bposd_decoder(max_iter: int = 30, osd_method: str = 'osd_cs', osd_order: int = 10) -> sinter.Decoder:
    """
    Returns a Sinter Decoder instance for BP+OSD.
    Relies on the native SinterBpOsdDecoder provided by the ldpc package.
    """
    if SinterBpOsdDecoder is None:
        raise ImportError("The 'ldpc' package is required to use the BP+OSD decoder.")
        
    return SinterBpOsdDecoder(
        max_iter=max_iter,
        bp_method='ms',
        osd_method=osd_method,
        osd_order=osd_order
    )

def get_custom_decoders() -> Dict[str, sinter.Decoder]:
    """
    Returns a dictionary of all custom Sinter decoders available in this project.
    This can be passed directly to the `custom_decoders` argument of sinter.collect().
    """
    decoders = {}
    if SinterBpOsdDecoder is not None:
        decoders['bposd'] = get_bposd_decoder()
    return decoders

