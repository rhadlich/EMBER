from __future__ import annotations

import numpy as np


TIMING_INDEX_TO_CAD_STEP = 0.1
TIMING_INDEX_TO_CAD_OFFSET = -360.0


def timing_indices_to_crank_angle(indices: np.ndarray) -> np.ndarray:
    """Convert combustion-timing array indices to crank angle (deg)."""
    return np.asarray(indices, dtype=np.float64) * TIMING_INDEX_TO_CAD_STEP + TIMING_INDEX_TO_CAD_OFFSET


def timing_crank_angle_to_indices(crank_angles: np.ndarray) -> np.ndarray:
    """Convert combustion-timing crank angles (deg) back to array indices."""
    return (np.asarray(crank_angles, dtype=np.float64) - TIMING_INDEX_TO_CAD_OFFSET) / TIMING_INDEX_TO_CAD_STEP


def engine_geometry(cad_step_deg: float = 0.1) -> tuple[np.ndarray, np.ndarray, float]:
    bore = 79.0 / 1000.0
    stroke = 86.0 / 1000.0
    rod_len = 160.0 / 1000.0
    delta = 0.6 / 1000.0
    crank_radius = stroke / 2.0
    compression_ratio = 17.19

    vd = np.pi * ((bore / 2.0) ** 2) * (2.0 * crank_radius)
    vc = vd / (compression_ratio - 1.0)

    cad = np.arange(-360.0, 360.0, cad_step_deg, dtype=np.float64)
    cad_rad = np.deg2rad(cad)
    area = np.pi * (bore**2) / 4.0
    volume = vc + area * (
        rod_len
        + crank_radius
        - (
            crank_radius * np.cos(cad_rad)
            + np.sqrt(rod_len**2 - (crank_radius * np.sin(cad_rad) + crank_radius * delta) ** 2)
        )
    )
    return cad, volume, vd


def _calculate_work(pressure: np.ndarray, volume: np.ndarray) -> np.ndarray:
    return np.trapz(pressure, x=volume, axis=1)


def _calculate_mprr(pressure: np.ndarray, ind_start: int, ind_end: int) -> np.ndarray:
    """Function to calculate the maximum pressure rise rate."""
    PRR = (pressure[:, 2:] - pressure[:, :-2]) / (2 * 0.1)
    return np.max(PRR[:,ind_start:ind_end],axis=1)


def _calculate_q_net(
    pressure: np.ndarray,
    volume: np.ndarray,
    soc: np.ndarray,
    eoc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pressure_pa = np.array(pressure, copy=True) * 100000.0
    d_volume = np.pad(
        (volume[2:] - volume[:-2]) / (2 * 0.1),
        pad_width=(1, 1),
        mode="constant",
        constant_values=(0, 0),
    )
    d_pressure = np.pad(
        (pressure_pa[:, 2:] - pressure_pa[:, :-2]) / (2 * 0.1),
        pad_width=((0, 0), (1, 1)),
        mode="constant",
        constant_values=(0, 0),
    )
    gamma = 1.34
    indices = np.tile(np.arange(pressure_pa.shape[1]), (pressure_pa.shape[0], 1))
    mask = (indices >= soc[:, None]) & (indices <= eoc[:, None])
    hrr = (gamma * pressure_pa * d_volume) / (gamma - 1) + (volume * d_pressure) / (gamma - 1)
    qnet = np.sum(np.where(mask, hrr, 0), axis=1) * 0.1
    valid_window = np.isfinite(soc) & np.isfinite(eoc)
    qnet = np.where(valid_window, qnet, np.nan)
    return hrr, qnet


def _calculate_mfb(pressure: np.ndarray, volume: np.ndarray, start_ind: int) -> np.ndarray:
    end_ind = int((20 + 360) / 0.1)
    volume_ref = np.min(volume)
    gamma = 1.34
    pressure_motored = pressure[:, :-1] * (volume[:-1] / volume[1:]) ** gamma
    d_pressure_comb = pressure[:, 1:] - pressure_motored
    d_pressure_comb_corrected = d_pressure_comb * (volume[1:] / volume_ref) ** gamma
    d_pressure_window = d_pressure_comb_corrected[:, start_ind:end_ind]
    cumulative = np.cumsum(d_pressure_window, axis=1)
    total = np.max(cumulative, axis=1)
    total = np.where(np.abs(total) < 1e-12, np.nan, total)
    return cumulative / total.reshape(-1, 1)


def _first_crossing(values: np.ndarray, threshold: float, start_ind: int) -> np.ndarray:
    mask = values > threshold
    idx = mask.argmax(axis=1).astype(float) + float(start_ind)
    found = mask.any(axis=1)
    idx[~found] = np.nan
    return idx


def _calculate_combustion_timings(mfb: np.ndarray, start_ind: int) -> tuple[np.ndarray, ...]:
    soc = _first_crossing(mfb, 0.005, start_ind)
    ca10 = _first_crossing(mfb, 0.1, start_ind)
    ca50 = _first_crossing(mfb, 0.5, start_ind)
    ca90 = _first_crossing(mfb, 0.9, start_ind)
    eoc = _first_crossing(mfb, 0.999, start_ind)
    return soc, ca10, ca50, ca90, eoc


def calculate_pressure_metrics(
    pressure_traces: np.ndarray,
    *,
    volume: np.ndarray,
    cad: np.ndarray | None = None,
    vd: float,
) -> dict[str, np.ndarray]:
    del cad  # Compatibility with existing call sites.
    pressure_traces = np.asarray(pressure_traces)
    if pressure_traces.ndim == 1:
        pressure_traces = pressure_traces.reshape(1, -1)
    if pressure_traces.ndim != 2:
        raise ValueError(f"pressure_traces must have shape (N, T), got {pressure_traces.shape}")

    start_ind = int((360 - 10) / 0.1)
    work = _calculate_work(pressure_traces, volume)
    imep = work / vd
    mprr = _calculate_mprr(pressure_traces, ind_start=3400, ind_end=4000)
    mfb = _calculate_mfb(pressure_traces, volume, start_ind=start_ind)
    soc_idx, ca10_idx, ca50_idx, ca90_idx, eoc_idx = _calculate_combustion_timings(mfb, start_ind=start_ind)
    hrr, qnet = _calculate_q_net(pressure_traces, volume, soc_idx, eoc_idx)

    return {
        "imep": imep,
        "mprr": mprr,
        "hrr": hrr,
        "mfb": mfb,
        "qnet": qnet,
        "soc": timing_indices_to_crank_angle(soc_idx),
        "ca10": timing_indices_to_crank_angle(ca10_idx),
        "ca50": timing_indices_to_crank_angle(ca50_idx),
        "ca90": timing_indices_to_crank_angle(ca90_idx),
        "eoc": timing_indices_to_crank_angle(eoc_idx),
    }
