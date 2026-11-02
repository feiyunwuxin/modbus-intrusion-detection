"""Parity test: H743 23-dim input vs PC sim 23-dim input on the same 16 frames.

Hypothesis: For length<16 (attack) frames, the H743 sees zeros for the body fields
(gain/reset_rate/deadband/cycle_time/rate/system_mode/control_scheme/pump/solenoid/
pressure_measurement) because the PC encodes them as 11×0x00 in the RTU frame.

But the PC sim uses the actual CSV values. This creates a drift in:
- press_mean_w (averaged over 16 frames, attack frame's 0 pulls mean down)
- All other feature channels that the attack frame distorts

This script computes BOTH views and compares them to verify the drift.
"""
import csv
import struct
import numpy as np

# H743 scaler (from scaler_export.h)
FEAT_MU_18 = np.array([
    4.0, 3.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    14135.0, 1.426832, 3.116917, 0.0, 1.0
], dtype=np.float32)

FEAT_SIGMA_18 = np.array([
    1.0, 13.0, 30.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    4849.0, 1.521081, 3.241358, 1.0, 1.0
], dtype=np.float32)

UNUSUAL_FCS = {133, 136, 137, 138, 139, 140, 171}

# CSV columns (1-indexed in CSV, 0-indexed in Python)
CSV_COLS = {
    'address': 0, 'function': 1, 'length': 2, 'setpoint': 3, 'gain': 4,
    'reset rate': 5, 'deadband': 6, 'cycle time': 7, 'rate': 8, 'system mode': 9,
    'control scheme': 10, 'pump': 11, 'solenoid': 12, 'pressure measurement': 13,
    'crc rate': 14, 'command response': 15, 'time': 16, 'binary result': 17,
    'categorized result': 18, 'specific result': 19,
}

RAW17_FEAT_COLS = [
    'setpoint', 'gain', 'reset rate', 'deadband', 'cycle time', 'rate',
    'system mode', 'control scheme', 'pump', 'solenoid', 'pressure measurement',
]


def is_unusual_fc(func):
    return int(func) in UNUSUAL_FCS


def get_csv_value(row, key):
    """Get CSV value, default 0."""
    idx = CSV_COLS[key]
    try:
        return float(row[idx])
    except (ValueError, IndexError):
        return 0.0


def build_h743_frame_18dim_view(row, prev_ts_sec, lru_dict, lru_rr_idx):
    """H743 view of 18-dim features for one frame, using RTU-encoded values.

    For length<16 frames, the body fields are 0 (per PC encoding build_raw17_frame).
    For other frames, body fields are the actual CSV values.
    """
    length = int(get_csv_value(row, 'length'))
    is_first = (prev_ts_sec is None)
    ts = get_csv_value(row, 'time')

    if is_first:
        time_diff = 0.0
    else:
        time_diff = ts - prev_ts_sec

    # LRU lookup
    slave = int(get_csv_value(row, 'address'))
    func = int(get_csv_value(row, 'function'))
    key = (slave, func)
    if key in lru_dict:
        time_since = ts - lru_dict[key]
        lru_dict[key] = ts
    else:
        time_since = 0.0
        lru_dict[key] = ts

    # 18-dim raw (KEEP_18 order)
    raw = np.zeros(18, dtype=np.float32)
    raw[0] = slave
    raw[1] = func
    raw[2] = length

    if length < 16:
        # Attack frame: body is 11×0x00 in RTU, so H743 sees zeros for these fields
        # (crc_rate is still actual, is_response is actual, etc.)
        # raw[3..12] stay 0
        pass
    else:
        # Normal frame: body is actual values
        raw[3] = get_csv_value(row, 'gain')
        raw[4] = get_csv_value(row, 'reset rate')
        raw[5] = get_csv_value(row, 'deadband')
        raw[6] = get_csv_value(row, 'cycle time')
        raw[7] = get_csv_value(row, 'rate')
        raw[8] = get_csv_value(row, 'system mode')
        raw[9] = get_csv_value(row, 'control scheme')
        raw[10] = get_csv_value(row, 'pump')
        raw[11] = get_csv_value(row, 'solenoid')
        # pressure_measurement, clip [0, 100]
        pressure = get_csv_value(row, 'pressure measurement')
        if pressure < 0: pressure = 0
        if pressure > 100: pressure = 100
        raw[12] = pressure

    # crc_rate, time_diff, time_since, is_unusual, is_response
    raw[13] = int(round(get_csv_value(row, 'crc rate')))  # uint16 truncation
    raw[14] = time_diff
    raw[15] = time_since
    raw[16] = 1.0 if is_unusual_fc(func) else 0.0
    raw[17] = 1.0 if int(get_csv_value(row, 'command response')) == 1 else 0.0

    # Normalize
    normed = (raw - FEAT_MU_18) / FEAT_SIGMA_18
    return normed, ts, lru_dict, lru_rr_idx


def build_pccsv_frame_18dim_view(row, prev_ts_sec, lru_dict):
    """PC sim view: uses actual CSV values regardless of length."""
    length = int(get_csv_value(row, 'length'))
    is_first = (prev_ts_sec is None)
    ts = get_csv_value(row, 'time')

    if is_first:
        time_diff = 0.0
    else:
        time_diff = ts - prev_ts_sec

    slave = int(get_csv_value(row, 'address'))
    func = int(get_csv_value(row, 'function'))
    key = (slave, func)
    if key in lru_dict:
        time_since = ts - lru_dict[key]
        lru_dict[key] = ts
    else:
        time_since = 0.0
        lru_dict[key] = ts

    raw = np.zeros(18, dtype=np.float32)
    raw[0] = slave
    raw[1] = func
    raw[2] = length

    # Use actual CSV values regardless of length
    raw[3] = get_csv_value(row, 'gain')
    raw[4] = get_csv_value(row, 'reset rate')
    raw[5] = get_csv_value(row, 'deadband')
    raw[6] = get_csv_value(row, 'cycle time')
    raw[7] = get_csv_value(row, 'rate')
    raw[8] = get_csv_value(row, 'system mode')
    raw[9] = get_csv_value(row, 'control scheme')
    raw[10] = get_csv_value(row, 'pump')
    raw[11] = get_csv_value(row, 'solenoid')
    pressure = get_csv_value(row, 'pressure measurement')
    if pressure < 0: pressure = 0
    if pressure > 100: pressure = 100
    raw[12] = pressure

    raw[13] = int(round(get_csv_value(row, 'crc rate')))
    raw[14] = time_diff
    raw[15] = time_since
    raw[16] = 1.0 if is_unusual_fc(func) else 0.0
    raw[17] = 1.0 if int(get_csv_value(row, 'command response')) == 1 else 0.0

    normed = (raw - FEAT_MU_18) / FEAT_SIGMA_18
    return normed, ts, lru_dict


def compute_27dim_from_18dim(frames18, length_values, func_values):
    """Compute 27-dim (16, 27) from 16 frames of 18-dim normalized features.

    Replicates H743 sliding_window_get logic.
    """
    n_t, n_c = frames18.shape
    assert n_t == 16 and n_c == 18

    # 19-dim [3] (setpoint) is always 0, so 18-dim maps directly to 19-dim [0,1,4..18]
    # 19-dim index in scaler: [0=addr, 1=func, 2=length, 3=setpoint, 4..18=misc]
    # KEEP_18[0..16] -> scaler[0,1,4..18]
    # 19-dim positions: KEEP_18[0] -> 19[0], KEEP_18[1] -> 19[1], KEEP_18[i>=2] -> 19[i+2]
    # 19-dim[3] (setpoint) is 0

    # Aggregations
    press_sum = 0.0
    crc_max = -1e30
    resp_signed = 0  # sum of is_response
    unusual_signed = 0
    seen_lens = set()

    for t in range(n_t):
        f18 = frames18[t]
        # f18[0]=slave, [1]=func, [2]=length, [3]=gain, ..., [12]=pressure, [13]=crc, [14]=td, [15]=ts, [16]=uf, [17]=resp
        press = f18[12]
        crc = f18[13]
        resp = f18[17]
        uf = f18[16]
        length_n = length_values[t]
        press_sum += press
        if crc > crc_max: crc_max = crc
        resp_signed += int(resp)
        unusual_signed += int(uf)
        seen_lens.add(length_n)

    press_mean_w = press_sum / 16.0
    balance_w = 1.0 - resp_signed / 8.0
    resp_count_w = float(resp_signed)
    unusual_w = float(unusual_signed)
    nunique_w = float(len(seen_lens))

    # crc_mean_w (calculated but dropped in KEEP_23)
    crc_sum = sum(frames18[t][13] for t in range(n_t))
    crc_mean_w = crc_sum / 16.0

    # cmd_count_w (calculated but dropped in KEEP_23)
    cmd_count_w = float(16 - unusual_signed - resp_signed)

    # Build 27-dim
    frame27 = np.zeros((16, 27), dtype=np.float32)
    for t in range(n_t):
        f18 = frames18[t]
        # KEEP_18[0..16] -> 19[0,1,4..18]
        # So 19[0] = f18[0], 19[1] = f18[1], 19[i>=2] = f18[i-2]
        # For 19-dim: [0,1,2(length),3(setpoint=0),4..18]
        frame27[t][0] = f18[0]  # address
        frame27[t][1] = f18[1]  # function
        # frame27[t][2] = length (f18[2])
        frame27[t][2] = f18[2]
        # frame27[t][3] = 0 (setpoint)
        # frame27[t][4..18] = f18[3..17]
        for i in range(3, 18):
            frame27[t][i + 1] = f18[i]  # +1 to skip setpoint slot
        # Aggregations (broadcast to all 16 frames)
        frame27[t][19] = press_mean_w
        frame27[t][20] = crc_mean_w
        frame27[t][21] = crc_max
        frame27[t][22] = cmd_count_w
        frame27[t][23] = resp_count_w
        frame27[t][24] = balance_w
        frame27[t][25] = nunique_w
        frame27[t][26] = unusual_w

    return frame27


def kEEP_23_slice_and_clip(frame27):
    """Apply KEEP_23 (23-dim model_idx -> 27-dim source idx) + clip ±10."""
    keep_idx = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26]
    out = np.zeros((16, 23), dtype=np.float32)
    for c in range(23):
        src = keep_idx[c]
        for t in range(16):
            v = frame27[t][src]
            if v > 10.0: v = 10.0
            if v < -10.0: v = -10.0
            out[t][c] = v
    return out


def load_csv_window(csv_path, start_frame, n_frames=16):
    """Load n_frames starting from start_frame (1-indexed)."""
    rows = []
    with open(csv_path, encoding='utf-8') as f:
        r = csv.reader(f)
        next(r)  # skip header
        for i, row in enumerate(r, start=1):
            if i > start_frame + n_frames - 1:
                break
            if i >= start_frame:
                rows.append(row)
    return rows


def compute_window_views(csv_path, window_idx, n_windows=918):
    """Compute H743 view and PC sim view for window W = window_idx."""
    start_frame = window_idx * 16 + 1
    rows = load_csv_window(csv_path, start_frame, 16)

    # H743 view (uses RTU-encoded zeros for length<16)
    h743_frames = []
    prev_ts = None
    lru_dict = {}
    for row in rows:
        normed, prev_ts, lru_dict, _ = build_h743_frame_18dim_view(row, prev_ts, lru_dict, 0)
        h743_frames.append(normed)
    h743_frames = np.array(h743_frames)

    # PC sim view (uses actual CSV values)
    pccsv_frames = []
    prev_ts = None
    lru_dict = {}
    for row in rows:
        normed, prev_ts, lru_dict = build_pccsv_frame_18dim_view(row, prev_ts, lru_dict)
        pccsv_frames.append(normed)
    pccsv_frames = np.array(pccsv_frames)

    # Aggregations from raw CSV
    length_values = [int(get_csv_value(r, 'length')) for r in rows]
    func_values = [int(get_csv_value(r, 'function')) for r in rows]

    h743_27 = compute_27dim_from_18dim(h743_frames, length_values, func_values)
    pccsv_27 = compute_27dim_from_18dim(pccsv_frames, length_values, func_values)

    h743_23 = kEEP_23_slice_and_clip(h743_27)
    pccsv_23 = kEEP_23_slice_and_clip(pccsv_27)

    return h743_23, pccsv_23


def main():
    # Test on FN windows from H743 test
    fn_windows = [2, 51, 136, 181, 229, 882]  # 6 representative FN windows
    # Also test on TP windows
    tp_windows = [8, 32, 46, 60, 80]

    print("=" * 80)
    print("FN WINDOWS (H743 missed attack, PC sim caught)")
    print("=" * 80)
    for W in fn_windows:
        h743_23, pccsv_23 = compute_window_views('IanArffDataset.csv', W)
        diff = h743_23 - pccsv_23
        max_diff = np.abs(diff).max()
        n_attack = sum(1 for i in range(16) if i == 0)  # placeholder

        # Check if window has attack frames (length<16)
        rows = load_csv_window('IanArffDataset.csv', W * 16 + 1, 16)
        n_atk = sum(1 for r in rows if int(get_csv_value(r, 'length')) < 16)

        print(f"\nW={W}: max |H743-PC| = {max_diff:.4f}, attack frames in window: {n_atk}/16")
        if max_diff > 0.01:
            # Find channels with biggest diff
            for c in range(23):
                ch_diff = np.abs(diff[:, c]).max()
                if ch_diff > 0.1:
                    print(f"  ch{c:2d}: max diff = {ch_diff:.3f}, "
                          f"H743[0,t]={h743_23[0,c]:.3f}, PC[0,t]={pccsv_23[0,c]:.3f}")
                    # Show per-t
                    for t in range(16):
                        if abs(h743_23[t, c] - pccsv_23[t, c]) > 0.1:
                            print(f"    t={t:2d}: H743={h743_23[t, c]:7.3f}, PC={pccsv_23[t, c]:7.3f}, "
                                  f"diff={h743_23[t, c] - pccsv_23[t, c]:7.3f}")

    print()
    print("=" * 80)
    print("TP WINDOWS (H743 caught attack)")
    print("=" * 80)
    for W in tp_windows:
        h743_23, pccsv_23 = compute_window_views('IanArffDataset.csv', W)
        diff = h743_23 - pccsv_23
        max_diff = np.abs(diff).max()

        rows = load_csv_window('IanArffDataset.csv', W * 16 + 1, 16)
        n_atk = sum(1 for r in rows if int(get_csv_value(r, 'length')) < 16)

        print(f"\nW={W}: max |H743-PC| = {max_diff:.4f}, attack frames in window: {n_atk}/16")


if __name__ == '__main__':
    main()
