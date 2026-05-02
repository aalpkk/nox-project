from dataclasses import dataclass


@dataclass(frozen=True)
class OSCMatrixParams:
    mf_len: int = 35
    mf_smooth: int = 6
    hw_len: int = 7
    hw_sig_type: str = "SMA"
    hw_sig_len: int = 3
    hw_color_param: int = 80
    div_sens: int = 20
    rev_factor: int = 5
    meter_width: int = 3


DEFAULT_PARAMS = OSCMatrixParams()
