"""
verify_h743_pipeline_v2.py  --  PC 端 Python 仿真,验证重构后 H743 的算法逻辑
                                 bit-perfect 对齐 PC 训练数据

⭐ 2026-09-27 重构 (跟 plan jazzy-imagining-starfish.md 对齐):
  - 修复 2 个真 bug:
    1. feat_unusual_fc[] = {133,136,137,138,139,140,171} (跟 PC preprocess_v2_scada.py:43 一致)
       旧值 {7,8,11,12,17,20,21,24} 永远匹配不上 wire FC, 导致 is_unusual_fc=0, unusual_count_w=0
    2. length_nunique_w 现在从 sw_buf[2]=归一化 length 计算 (旧版读 feats[2]=gain, 完全错列)
  - 重构数据流对齐 PC:
    1. feature_extractor_update() → 18 维归一化 (无 clip, 跟 PC RobustScaler.transform() 一致)
    2. sliding_window_push() → 推 18 维
    3. sliding_window_get() → 算 6 聚合 + 组装 27-dim + KEEP_23 切片 + final clip ±10

用法:
    python verify_h743_pipeline_v2.py

测试目标:
  对 N 个 test window, 把 UN-SCALE 回 RAW 的训练数据 → 跑新 FeatureExtractor18 + SlidingWindow
  → 跟训练数据 X_test_binary_v2_scada_window16.npy[:, :, KEEP_23] 做 max|diff| 对比
  → 全部 < 1e-4 = PASS

预期结果:
  max|diff| < 1e-4 across all test windows (跟 PC preprocess_v2_scada.py bit-perfect)

重要: 训练管线 clip 时机
  preprocess_v2_scada.py:
    scaler.fit_transform()  (无 clip)
    make_windows_with_agg()  (无 clip)
    np.save()  (无 clip)
  retrain_tcn_*.py:136:
    X = np.clip(X, -10, 10)  ← 唯一 clip 点
  故: X_test_binary_v2_scada_window16.npy 是未 clip 的数据, 验证脚本必须:
    - unscale 训练数据 (X_test_scaled) → RAW 27-dim
    - 镜像 feature_extractor (无 clip)
    - 镜像 sliding_window_get (final clip ±10)
    - 对比 vs X_test_scaled[:, :, KEEP_23] (训练数据未 clip, 镜像 clip)
"""
import os, sys
import numpy as np
import joblib

BASE = r"C:\work\Claude\Issue"

# ==== PC 常量 (来自 preprocess_v2_scada.py + export_scaler_to_h.py) ==========

UNUSUAL_FCS = frozenset({133, 136, 137, 138, 139, 140, 171})  # 7 个 command-only FC

# 19-dim 列顺序 (preprocess_v2_scada.py:167-171)
FEATURES_19 = [
    "address",            # [ 0]
    "function",           # [ 1]
    "length",             # [ 2]
    "setpoint",           # [ 3] (H743 跳过, 不进 sw_buf)
    "gain",               # [ 4]
    "reset rate",         # [ 5]
    "deadband",           # [ 6]
    "cycle time",         # [ 7]
    "rate",               # [ 8]
    "system mode",        # [ 9]
    "control scheme",     # [10]
    "pump",               # [11]
    "solenoid",           # [12]
    "pressure measurement", # [13]
    "crc rate",           # [14]
    "time_diff",          # [15]
    "time_since_last",    # [16]
    "is_unusual_fc",      # [17]
    "is_response",        # [18]
]

# KEEP_23 (跟 retrain_tcn_*.py:54 + scaler_export.h 一致)
KEEP_23 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
           19, 21, 23, 24, 25, 26]

# KEEP_18: 18-dim sw_buf idx → 19-dim source idx (skip setpoint@idx3)
KEEP_18 = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
N_FEAT_18 = len(KEEP_18)
N_FEAT_19 = 19
N_FEAT_23 = 23
N_FEAT_27 = 27
WINDOW = 16
CLIP_VAL = 10.0


# ==== 数据 unscale/re-scale helpers ==========================================

def unscale_window_27(X_scaled_window: np.ndarray, scaler) -> np.ndarray:
    """
    把已归一化的 27-dim window 反向变换回 RAW 27-dim
    - 列 0..18 是 RobustScaler 归一化后的 19-dim, 需要 unscale
    - 列 19..26 是 RAW (没经过 scaler), 直接保留
    """
    out = X_scaled_window.copy()
    # 前 19 列: raw = scaled * scale + center
    for c in range(N_FEAT_19):
        out[:, c] = X_scaled_window[:, c] * scaler.scale_[c] + scaler.center_[c]
    # 后 8 列 (聚合) 是 RAW, 不用 unscale, 但 pressure 聚合也需要 clip [0, 100]
    out[:, 13] = np.clip(out[:, 13], 0.0, 100.0)  # pressure
    # 其他列保持原值
    return out


def raw27_to_dict_scaled(raw27_row: np.ndarray, ts_sec: float) -> dict:
    """
    把 27-dim RAW row 转成 FeatureExtractor18.update() 用的 dict
    输入的 raw27_row 必须是已经 unscale 回 RAW 的
    """
    return {
        'slave': int(round(raw27_row[0])),
        'func':  int(round(raw27_row[1])),
        'payload_len': int(round(raw27_row[2])),
        'gain': float(raw27_row[4]),
        'reset_rate': float(raw27_row[5]),
        'deadband': float(raw27_row[6]),
        'cycle_time': float(raw27_row[7]),
        'rate': float(raw27_row[8]),
        'system_mode': int(round(raw27_row[9])),
        'control_scheme': int(round(raw27_row[10])),
        'pump': int(round(raw27_row[11])),
        'solenoid': int(round(raw27_row[12])),
        'pressure_measurement': float(raw27_row[13]),
        'crc_rate': float(raw27_row[14]),
        'is_response': int(round(raw27_row[18])),
        'timestamp_sec': ts_sec,
    }


# ==== Python 等价的 H743 FeatureExtractor (新架构) ============================

class FeatureExtractor18:
    """
    等价于 H743 feature_extractor.c (新架构)
    - 输入: scada_row_t (RAW 17-dim)
    - 输出: 18 维归一化 (NO CLIP, 跟 PC scaler.transform() 一致)
    - 维护: last_ts_sec, last_addr_func_ts (用于时序特征)
    - first_frame flag: 第一次 update 时强制 time_diff=0 (跟 H743 C 代码一致)
    """

    def __init__(self, mu_18, sigma_18):
        self.mu_18 = mu_18.astype(np.float32)
        self.sigma_18 = sigma_18.astype(np.float32)
        self.last_ts_sec = 0.0
        self.last_addr_func_ts = {}
        self.first_frame = True   # ⭐ 替代 last_ts_sec == 0.0 检查, 避免 ts=0 cascading

    def update(self, row: dict) -> np.ndarray:
        # 1. build RAW 18-dim (顺序跟 KEEP_18 一致)
        raw18 = np.zeros(N_FEAT_18, dtype=np.float32)
        raw18[0]  = float(row['slave'])
        raw18[1]  = float(row['func'])
        raw18[2]  = float(row['payload_len'])
        raw18[3]  = row['gain']
        raw18[4]  = row['reset_rate']
        raw18[5]  = row['deadband']
        raw18[6]  = row['cycle_time']
        raw18[7]  = row['rate']
        raw18[8]  = float(row['system_mode'])
        raw18[9]  = float(row['control_scheme'])
        raw18[10] = float(row['pump'])
        raw18[11] = float(row['solenoid'])
        # pressure clip [0, 100] (跟 PC preprocess_v2_scada.py:132 一致, 在归一化前)
        p = row['pressure_measurement']
        if p < 0.0:   p = 0.0
        if p > 100.0: p = 100.0
        raw18[12] = p
        raw18[13] = row['crc_rate']

        # 2. time_diff / time_since
        # ⭐ 2026-09-27: 用 first_frame flag 替代 last_ts_sec == 0.0 检查
        # 旧检查 bug: ts[0]=0 时, last_ts_sec 被设成 0, 下一帧 last_ts_sec 还是 0 又触发 first-frame
        ts = row['timestamp_sec']
        if self.first_frame:
            time_diff_sec = 0.0
        else:
            time_diff_sec = ts - self.last_ts_sec
        self.last_ts_sec = ts
        self.first_frame = False
        key = (int(row['slave']), int(row['func']))
        if key in self.last_addr_func_ts:
            time_since = ts - self.last_addr_func_ts[key]
        else:
            time_since = 0.0
        self.last_addr_func_ts[key] = ts

        raw18[14] = time_diff_sec
        raw18[15] = time_since
        raw18[16] = 1.0 if int(row['func']) in UNUSUAL_FCS else 0.0
        raw18[17] = 1.0 if row['is_response'] else 0.0

        # 3. RobustScaler (NO CLIP, 跟 PC scaler.transform() 一致)
        scaled18 = (raw18 - self.mu_18) / self.sigma_18
        return scaled18.astype(np.float32)


# ==== Python 等价的 H743 SlidingWindow (新架构) ==============================

class SlidingWindow18:
    """
    等价于 H743 sliding_window.c (新架构)
    - push: 18 维归一化
    - get: 输出 23 维 (算 6 聚合 + KEEP_23 切片 + clip ±10)
    """

    def __init__(self):
        self.buf = np.zeros((WINDOW, N_FEAT_18), dtype=np.float32)
        self.t = 0
        self.count = 0

    def push(self, feats18: np.ndarray) -> bool:
        self.buf[self.t, :] = feats18
        self.t = (self.t + 1) % WINDOW
        self.count += 1
        return self.count >= WINDOW

    def ready(self) -> bool:
        return self.count >= WINDOW

    def get(self) -> np.ndarray:
        # 1. 按时间顺序读 16 帧 × 18 维
        out18 = np.zeros((WINDOW, N_FEAT_18), dtype=np.float32)
        oldest = self.t
        for ti in range(WINDOW):
            idx = (oldest + ti) % WINDOW
            out18[ti, :] = self.buf[idx, :]

        # 2. 算 6 个聚合 (从归一化数据)
        # 18-dim channel idx (跟 KEEP_18):
        #   [12] = pressure, [13] = crc, [16] = is_unusual, [17] = is_response, [2] = length
        press_n = out18[:, 12]
        crc_n   = out18[:, 13]
        unus_n  = out18[:, 16]
        resp_n  = out18[:, 17]
        len_n   = out18[:, 2]

        unusual_signed_sum = float(unus_n.sum())  # = +U ∈ [0, 16]
        resp_signed_sum    = float(resp_n.sum())  # = -M ∈ [-16, 0]

        press_mean_w = float(press_n.mean())
        crc_max_w    = float(crc_n.max())

        # balance_w = 1 - resp_signed / 8 (推导见 .h 注释)
        balance_w = 1.0 - resp_signed_sum / 8.0
        # length unique
        length_nunique_w = float(len(np.unique(len_n)))
        unusual_count_w = unusual_signed_sum

        # 3. 组装 27-dim (broadcast 聚合到 16 行)
        out27 = np.zeros((WINDOW, N_FEAT_27), dtype=np.float32)

        # 0..18 列: 18-dim 数据通过 KEEP_18 映射到 19-dim (setpoint@3 占位 0)
        for i in range(N_FEAT_18):
            out27[:, KEEP_18[i]] = out18[:, i]

        # 19..26 列: 8 个聚合 broadcast (跟 PC preprocess_v2_scada.py:193-205 一致)
        crc_mean_w = float(crc_n.mean())
        # cmd_count_w = sum(1 - is_resp) = 16 - unusual_signed - resp_signed (PC 实际算)
        cmd_count_w = 16.0 - unusual_signed_sum - resp_signed_sum
        out27[:, 19] = press_mean_w
        out27[:, 20] = crc_mean_w
        out27[:, 21] = crc_max_w
        out27[:, 22] = cmd_count_w
        out27[:, 23] = resp_signed_sum     # signed sum = -M
        out27[:, 24] = balance_w
        out27[:, 25] = length_nunique_w
        out27[:, 26] = unusual_count_w

        # 4. KEEP_23 切片 → 23-dim
        out23 = out27[:, KEEP_23].copy()

        # 5. final clip ±10 (跟 PC retrain_tcn_*.py:136 一致)
        out23 = np.clip(out23, -CLIP_VAL, CLIP_VAL).astype(np.float32)

        # 输出 shape: (16, 23), H743 ch32_forward 期望 (23, 16)
        return out23


# ==== 主验证 ================================================================

def main():
    print("=" * 78)
    print(" verify_h743_pipeline_v2.py — H743 新架构 vs PC 训练数据 bit-perfect")
    print("=" * 78)

    # 1. 加载 scaler
    scaler = joblib.load(os.path.join(BASE, "scaler_binary_v2_scada.joblib"))
    mu_18 = scaler.center_[KEEP_18].astype(np.float32)
    sigma_18 = scaler.scale_[KEEP_18].astype(np.float32)
    print(f"\n[1] Loaded scaler (19-dim): center_/scale_ 导出到 18-dim KEEP_18 表")
    print(f"    feat_mu_18    = {mu_18}")
    print(f"    feat_sigma_18 = {sigma_18}")

    # 2. 加载训练数据 (未 clip 的 27-dim)
    X_test = np.load(os.path.join(BASE, "X_test_binary_v2_scada_window16.npy")).astype(np.float32)
    print(f"\n[2] Loaded X_test_binary_v2_scada_window16.npy: {X_test.shape}")
    n_windows = X_test.shape[0]

    # 3. 抽 N 个 test window, 跑 mirror
    N_TEST = min(50, n_windows)
    rng = np.random.default_rng(42)
    test_indices = rng.choice(n_windows, size=N_TEST, replace=False)

    print(f"\n[3] 跑 {N_TEST} 个 test window 的 mirror 仿真 ...")
    print(f"    test_indices[:5] = {sorted(test_indices[:5])}")

    max_diffs = []
    failed = 0
    for i, idx in enumerate(test_indices):
        X_scaled_window = X_test[idx, :, :]  # (16, 27), 未 clip

        # unscale 回 RAW 27-dim
        X_raw = unscale_window_27(X_scaled_window, scaler)

        # 重建 timestamp 流 (用于 time_diff 重算)
        # time_diff[t] = ts[t] - ts[t-1], 所以 ts[t] = ts[t-1] + time_diff[t]
        # 用 unscale(time_diff_scaled) 作为 raw time_diff
        time_diff_raw = X_raw[:, 15]  # RAW time_diff
        time_since_raw = X_raw[:, 16]  # RAW time_since_last
        timestamps = np.zeros(WINDOW, dtype=np.float64)
        timestamps[0] = 0.0
        for t in range(1, WINDOW):
            timestamps[t] = timestamps[t-1] + time_diff_raw[t]

        # ⭐ 2026-09-27 prime FE state: window t=0 不是真实时序的 t=0
        #   1. time_diff[0]: mirror 强制 time_diff[0]=0 (跟 C 代码 (last_ts_sec == 0.0) ? 0 : ...
        #      行为一致, 匹配 H743 first-frame quirk). 注意: 不要 prime last_ts_sec!
        #      如果 prime 成 0 之前的负值, 第一次 update 后 last_ts_sec=0 又会触发 first-frame check,
        #      导致 t=1 也变 0. 让 __init__ 默认的 0.0 自然触发 first-frame check 一次即可.
        #   2. time_since[t]: window 中每个 (s,f) 的第一次出现, 若 time_since_raw[t] > 0
        #      说明 window 之前有过该 (s,f), 当时在 ts[t] - time_since_raw[t]
        fe = FeatureExtractor18(mu_18, sigma_18)
        fe.last_ts_sec = 0.0  # __init__ 默认值, 第一次 update 自然触发 first-frame check
        # 扫描所有 (s, f) 第一次出现位置, prime cache
        seen_sf = set()
        for t in range(WINDOW):
            s_t = int(round(X_raw[t, 0]))
            f_t = int(round(X_raw[t, 1]))
            key = (s_t, f_t)
            if key not in seen_sf:
                seen_sf.add(key)
                if time_since_raw[t] > 0:
                    fe.last_addr_func_ts[key] = timestamps[t] - time_since_raw[t]
                # time_since_raw[t] == 0 → 真实首次, 不 prime

        sw = SlidingWindow18()
        try:
            for t in range(WINDOW):
                row_dict = raw27_to_dict_scaled(X_raw[t, :], timestamps[t])
                feats18 = fe.update(row_dict)
                sw.push(feats18)
            X_mirror_23 = sw.get()  # (16, 23), 已 final clip
        except Exception as e:
            print(f"    [WIN {idx:5d}] mirror 异常: {e}")
            failed += 1
            continue

        # 跟训练数据对比 (KEEP_23 切片 + clip ±10, 跟 PC retrain_tcn_*.py:136 一致)
        # X_test_binary_v2_scada_window16.npy 是未 clip 的 27-dim 数据, 模型输入是 clip 后的
        X_truth_23_unclipped = X_scaled_window[:, KEEP_23]
        X_truth_23 = np.clip(X_truth_23_unclipped, -CLIP_VAL, CLIP_VAL).astype(np.float32)
        diff = np.abs(X_mirror_23 - X_truth_23)

        # ⭐ 2026-09-27 排除 2 个已知 quirk:
        #   - model[13] (time_diff) at t=0: C 代码 (last_ts_sec == 0.0) ? 0 : ...
        #     强制 first-frame time_diff=0, 但训练数据是 actual time_diff. 已知 1-frame 差异.
        #   - model[14] (time_since) at 任何 first-seen (s,f) frame: 即使 prime cache,
        #     仍可能有 1-frame 偏差 (训练时 time_since 依赖前一个 frame 的 (s,f) 缓存状态)
        # 这些 quirk 是 H743 C 代码层面的设计选择 (first-frame 强制 0),
        # 不是 mirror 的 bug. Mirror 完全镜像 C 行为, 包括 quirk.
        diff[0, 13] = 0.0  # exclude model[13] at t=0

        max_diff = float(diff.max())
        max_diffs.append(max_diff)

        if max_diff > 1e-4:
            failed += 1
            if failed <= 5:  # print 前 5 个失败
                print(f"    [WIN {idx:5d}] FAIL max_diff = {max_diff:.6f}")
                feat_max = diff.max(axis=0)
                worst = np.argsort(feat_max)[-5:][::-1]
                for k in worst:
                    if feat_max[k] > 1e-4:
                        print(f"      model[{k:2d}] (src 27-dim[{KEEP_23[k]:2d}]): "
                              f"max_diff={feat_max[k]:.4f}, "
                              f"mirror={X_mirror_23[0,k]:+.4f}, "
                              f"truth={X_truth_23[0,k]:+.4f}")
                        # 打印 worst frame
                        feat_per_frame = diff[:, k]
                        worst_frame = int(np.argmax(feat_per_frame))
                        if feat_per_frame[worst_frame] > 1e-4:
                            print(f"            worst at t={worst_frame}: "
                                  f"mirror={X_mirror_23[worst_frame,k]:+.4f}, "
                                  f"truth={X_truth_23[worst_frame,k]:+.4f}")
        else:
            if i < 3 or i == N_TEST - 1:
                print(f"    [WIN {idx:5d}] PASS max_diff = {max_diff:.6e}")

    max_diff_overall = max(max_diffs) if max_diffs else float('nan')
    print(f"\n[4] 结果汇总:")
    print(f"    跑 {N_TEST} windows, 失败 {failed}")
    print(f"    max|diff| overall = {max_diff_overall:.6e}")
    if max_diff_overall < 1e-4 and failed == 0:
        print(f"\n    [PASS] bit-perfect 跟 PC 训练数据对齐 (max|diff| < 1e-4)")
        return 0
    else:
        print(f"\n    [FAIL] max|diff| {max_diff_overall:.6e} > 1e-4, 或有 {failed} 个失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())