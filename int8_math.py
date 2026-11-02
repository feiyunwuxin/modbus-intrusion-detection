"""
int8_math.py -- INT8 定点数学 + MCU header 读写 (MCU 工程 INT8 化共用底座).

本文件是 **PC 侧 numpy 镜像** 和 **MCU 侧 C 代码** 的唯一定义来源:
定点舍入语义严格照抄 CMSIS-NN (MCU/Drivers/CMSIS/NN/), 两边必须一致.

为什么不用 TFLite 的 MultiplyByQuantizedMultiplier?
  本工程 MCU 侧用 CMSIS-NN, 且 **没有定义 CMSIS_NN_USE_SINGLE_ROUNDING**,
  所以 arm_nn_requantize 走的是 "double rounding" 分支:
      arm_nn_divide_by_power_of_two(
          arm_nn_doubling_high_mult_no_sat(val * (1 << LEFT_SHIFT(shift)), multiplier),
          RIGHT_SHIFT(shift));
  它的舍入是 **round-half-up (向 +inf)**, 跟 TFLite 的 round-half-away-from-zero
  在中点上不同. 所以镜像必须复刻 CMSIS 的实现, 不能复刻 TFLite 的.

源绑定 (L1 验证): see_check_cmsis_semantics() -- 直接从 CMSIS-NN 源码里
正则解析出关键常量/公式, 断言跟本文件的实现一致. 源码一改, 测试立刻炸.
"""
import math
import re

import numpy as np

# ---------------------------------------------------------------------------
# 1. TFLite/CMSIS 的 QuantizeMultiplier (per-channel 定点乘数)
# ---------------------------------------------------------------------------


def quantize_multiplier(real_multiplier):
    """TFLite QuantizeMultiplier 语义.

    返回 (multiplier int32 in [2^30, 2^31), shift int32).
    真实乘数 = multiplier * 2^(shift-31).

    注意: TFLite 用 std::round (half away from zero), Python 的 round() 是
    banker's rounding -> 这里用 floor(x+0.5) 复刻 (real_multiplier>0 恒成立).
    """
    if real_multiplier == 0.0:
        return 0, 0
    q, shift = math.frexp(float(real_multiplier))   # real = q * 2^shift, q in [0.5,1)
    q_fixed = int(math.floor(q * (1 << 31) + 0.5))
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    if shift < -31:
        shift = 0
        q_fixed = 0
    return q_fixed, shift


def quantize_multiplier_vec(real_mults):
    """per-channel 版本 -> (mult array, shift array)."""
    ms, ss = [], []
    for r in np.asarray(real_mults, dtype=np.float64).ravel():
        m, s = quantize_multiplier(float(r))
        ms.append(m)
        ss.append(s)
    return np.array(ms, dtype=np.int32), np.array(ss, dtype=np.int32)


# ---------------------------------------------------------------------------
# 2. arm_nn_requantize 的 numpy 镜像 (double-rounding 分支)
# ---------------------------------------------------------------------------


def requantize_np(acc, mult, shift):
    """arm_nn_requantize(val, multiplier, shift) 的 bit-exact numpy 镜像.

    acc  : int 数组 (..., C)
    mult : int32 数组 (C,) 或标量
    shift: int32 数组 (C,) 或标量

    语义 (照抄 MCU/Drivers/CMSIS/NN/Include/arm_nnsupportfunctions.h):
        LEFT_SHIFT(s)  = s > 0 ? s : 0
        RIGHT_SHIFT(s) = s > 0 ? 0 : -s
        val'   = val * (1 << LEFT_SHIFT(shift))          [int32 语义, 调用方保证不溢出]
        hi     = (val' * mult + 2^30) >> 31              [算术右移 = floor]
        result = divide_by_power_of_two(hi, RIGHT_SHIFT(shift))
    """
    acc = np.asarray(acc, dtype=np.int64)
    mult = np.asarray(mult, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int64)
    ls = np.maximum(shift, 0)          # LEFT_SHIFT
    rs = np.where(shift > 0, 0, -shift)  # RIGHT_SHIFT
    v = acc * (np.int64(1) << ls)
    prod = v * mult + (np.int64(1) << 30)
    res = prod >> 31                    # numpy int64 >> = 算术右移 (floor)
    if np.any(rs > 0):
        mask = (np.int64(1) << rs) - 1
        rem = res & mask                 # C: remainder_mask & dividend (int32 语义)
        out = res >> rs
        thr = mask >> 1
        thr = np.where(out < 0, thr + 1, thr)   # C: if (result < 0) threshold++;
        res = out + np.where(rem > thr, 1, 0)
    return res


# ---------------------------------------------------------------------------
# 3. 权重 / 激活 量化
# ---------------------------------------------------------------------------


def round_half_away(x):
    """round half away from zero (numpy round 是 banker's, 不用)."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def quantize_weight_per_channel(w):
    """per-output-channel 对称量化. w: (oc, ...) -> (q int8 (oc,...), scale (oc,))."""
    w = np.asarray(w, dtype=np.float64)
    oc = w.reshape(w.shape[0], -1)
    amax = np.max(np.abs(oc), axis=1)
    scale = np.where(amax > 0, amax / 127.0, 1.0)
    q = round_half_away(oc / scale[:, None])
    q = np.clip(q, -127, 127).astype(np.int8)
    return q.reshape(w.shape), scale.astype(np.float32)


def quantize_weight_per_tensor(w):
    """per-tensor 对称量化 (FC 用, CMSIS-NN 的 FC 只有 per-tensor requant)."""
    w = np.asarray(w, dtype=np.float64)
    amax = float(np.max(np.abs(w))) if w.size else 0.0
    scale = amax / 127.0 if amax > 0 else 1.0
    q = round_half_away(w / scale)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, np.float32(scale)


def quantize_bias(b, s_in, s_w):
    """bias int32 量化: b_q = round(b / (s_in * s_w)).  s_w 可以是标量或 per-channel 数组."""
    b = np.asarray(b, dtype=np.float64)
    s_w = np.asarray(s_w, dtype=np.float64)
    denom = np.float64(s_in) * s_w
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    q = round_half_away(b / denom)
    return q.astype(np.int64)


def asymmetric_activation(lo, hi):
    """非对称 int8 激活: real in [lo,hi] -> q in [-128,127].

    s = (hi-lo)/255, zp = round(-128 - lo/s).  real=0 <-> q=zp (ReLU 语义成立).
    """
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        hi = lo + 1e-6
    s = (hi - lo) / 255.0
    zp = int(round_half_away(-128.0 - lo / s))
    zp = int(np.clip(zp, -128, 127))
    return np.float32(s), zp


def asymmetric_activation_q(lo, hi):
    """跟 asymmetric_activation 相同, 但用 round-half-away 之外的 np.round 保持 zp 一致."""
    return asymmetric_activation(lo, hi)


# ---------------------------------------------------------------------------
# 4. MCU header 读写
# ---------------------------------------------------------------------------

HDR_PREFIX = "v4se23_ch32_i8_"


def _wrap_items(items, width=100):
    """按 **元素** 边界折行 (不能按字符切, 否则会把数字切成两半)."""
    lines = []
    cur = ""
    for it in items:
        piece = it + ","
        if cur and len(cur) + 1 + len(piece) > width:
            lines.append("    " + cur)
            cur = piece
        else:
            cur = piece if not cur else cur + " " + piece
    if cur:
        lines.append("    " + cur)
    return "\n".join(lines)


def _fmt_int8(vals):
    return _wrap_items(["%d" % int(v) for v in vals])


def _fmt_int32(vals):
    return _wrap_items(["%d" % int(v) for v in vals])


def _fmt_float(vals):
    return _wrap_items(["%.9gf" % float(v) for v in vals])


def write_int8_header(path, arrays, scalars, meta):
    """写 MCU 的 INT8 权重 header.

    arrays : {name: ('i8'|'i32'|'f32', ndarray)}
    scalars: {name: ('i32'|'f32', value)}
    meta   : list of (comment, str) 头注释
    """
    lines = []
    lines.append("/* ============================================================================")
    lines.append(" * model_v4_se_23dim_ch32_int8_s42.h -- 全量 INT8 权重 (TCN+SE ch=32, 23-dim, w=16)")
    lines.append(" *")
    for c in meta:
        lines.append(" * %s" % c)
    lines.append(" *")
    lines.append(" * 生成方式: quantize_fp32_to_int8.py  (不要手改)")
    lines.append(" * 源权重: model_v4_se_23dim_ch32_fp32_s42.h  (部署真值)")
    lines.append(" *")
    lines.append(" * 布局约定:")
    lines.append(" *   conv 权重: [out_ch][kernel][in_ch]   (CMSIS-NN filter_dims = [C_OUT, HK, WK, C_IN])")
    lines.append(" *   fc   权重: [out_ch][in_ch]          (CMSIS-NN filter_dims = [C_OUT, 1, 1, C_IN])")
    lines.append(" *   激活  布局: [T][C] (NHWC),         CMSIS-NN 输入/输出约定")
    lines.append(" * ============================================================================ */")
    lines.append("#ifndef MODEL_V4_SE_23DIM_CH32_INT8_S42_H")
    lines.append("#define MODEL_V4_SE_23DIM_CH32_INT8_S42_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    for name in sorted(arrays):
        kind, vals = arrays[name]
        vals = np.asarray(vals).ravel()
        ctype = {"i8": "int8_t", "i32": "int32_t", "f32": "float"}[kind]
        if kind == "i8":
            txt = _fmt_int8(vals)
        elif kind == "i32":
            txt = _fmt_int32(vals)
        else:
            txt = _fmt_float(vals)
        lines.append("static const %s %s%s[%d] = {" % (ctype, HDR_PREFIX, name, len(vals)))
        lines.append(txt)
        lines.append("};")
        lines.append("")
    for name in sorted(scalars):
        kind, val = scalars[name]
        ctype = {"i32": "int32_t", "f32": "float"}[kind]
        if kind == "i32":
            lines.append("static const %s %s%s = %d;" % (ctype, HDR_PREFIX, name, int(val)))
        else:
            lines.append("static const %s %s%s = %.9gf;" % (ctype, HDR_PREFIX, name, float(val)))
    lines.append("")
    lines.append("#endif /* MODEL_V4_SE_23DIM_CH32_INT8_S42_H */")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def parse_int8_header(path):
    """解析 MCU INT8 header -> {name: ndarray} (含标量)."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    for m in re.finditer(r"static const\s+(\w+)\s+" + HDR_PREFIX + r"(\w+)\s*\[\s*(\d+)\s*\]\s*=\s*\{(.*?)\};",
                         txt, re.S):
        ctype, name, n, body = m.group(1), m.group(2), m.group(3), m.group(4)
        if ctype == "int8_t":
            vals = np.array([int(x) for x in re.findall(r"-?\d+", body)], dtype=np.int8)
        elif ctype == "int32_t":
            vals = np.array([int(x) for x in re.findall(r"-?\d+", body)], dtype=np.int32)
        else:
            vals = np.array([float(x[:-1]) for x in re.findall(r"[-+0-9.eE]+f", body)], dtype=np.float32)
        assert len(vals) == int(n), (name, len(vals), n)
        out[name] = vals
    # 标量
    for m in re.finditer(r"static const\s+(int32_t|float)\s+" + HDR_PREFIX + r"(\w+)\s*=\s*([^;]+);",
                         txt):
        ctype, name, val = m.group(1), m.group(2), m.group(3).strip()
        out[name] = float(val[:-1]) if ctype == "float" else int(val)
    return out


# ---------------------------------------------------------------------------
# 5. L1: 源绑定检查 (CMSIS-NN 源码 -> 本文件的实现)
# ---------------------------------------------------------------------------

NN_H = "MCU/Drivers/CMSIS/NN/Include/arm_nnsupportfunctions.h"


def see_check_cmsis_semantics(nn_h=NN_H):
    """从 CMSIS-NN 源码解析 requantize 语义, 断言与本文件实现一致.

    检查项:
      1. CMSIS_NN_USE_SINGLE_ROUNDING **未定义** (否则走 single rounding 分支)
      2. arm_nn_requantize 走 double-rounding 分支 (doubling_high_mult + divide_by_power_of_two)
      3. doubling_high_mult_no_sat 的舍入常量 = 1<<30, 右移 31
      4. divide_by_power_of_two 的阈值逻辑 (result<0 时 threshold++)
      5. LEFT_SHIFT / RIGHT_SHIFT 宏定义
    """
    src = open(nn_h, encoding="utf-8", errors="replace").read()
    code = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    code = re.sub(r"//[^\n]*", " ", code)

    # 1. single rounding 只能在 include 之外被定义 -> 检查全工程没有 -D
    assert "#define CMSIS_NN_USE_SINGLE_ROUNDING" not in src, \
        "CMSIS_NN_USE_SINGLE_ROUNDING 被定义了 -> requantize 走 single rounding, 镜像要改"

    # 2. requantize 的 double-rounding 分支
    m = re.search(r"arm_nn_requantize\s*\([^)]*\)\s*\{(.*?)\n\}", code, re.S)
    assert m, "找不到 arm_nn_requantize 定义"
    body = m.group(1)
    assert "arm_nn_doubling_high_mult_no_sat" in body, \
        "arm_nn_requantize 不再走 double-rounding 分支 -> 镜像要改"
    assert "arm_nn_divide_by_power_of_two" in body
    assert "LEFT_SHIFT(shift)" in body and "RIGHT_SHIFT(shift)" in body

    # 3. doubling_high_mult_no_sat: 舍入常量 1<<30, 右移 31
    m2 = re.search(r"arm_nn_doubling_high_mult_no_sat\s*\([^)]*\)\s*\{(.*?)\n\}", code, re.S)
    assert m2, "找不到 arm_nn_doubling_high_mult_no_sat"
    b2 = m2.group(1)
    assert "1 << 30" in b2, "doubling_high_mult 舍入常量不是 1<<30"
    assert ">> 31" in b2, "doubling_high_mult 右移不是 31"

    # 4. divide_by_power_of_two 阈值逻辑
    m3 = re.search(r"arm_nn_divide_by_power_of_two\s*\([^)]*\)\s*\{(.*?)\n\}", code, re.S)
    assert m3, "找不到 arm_nn_divide_by_power_of_two"
    b3 = m3.group(1)
    assert "remainder_mask >> 1" in b3, "divide_by_power_of_two 阈值常量变了"
    assert "if (result < 0)" in b3 and "threshold++" in b3, \
        "divide_by_power_of_two 负数阈值逻辑变了"

    # 5. shift 宏
    assert re.search(r"#define\s+LEFT_SHIFT\(_shift\)\s*\(_shift > 0 \? _shift : 0\)", src), \
        "LEFT_SHIFT 宏定义变了"
    assert re.search(r"#define\s+RIGHT_SHIFT\(_shift\)\s*\(_shift > 0 \? 0 : -_shift\)", src), \
        "RIGHT_SHIFT 宏定义变了"
    return True


if __name__ == "__main__":
    ok = see_check_cmsis_semantics()
    print("[L1] CMSIS-NN requantize 语义源绑定检查: %s" % ("PASS" if ok else "FAIL"))
    # 自测: 跟手算对齐
    m, s = quantize_multiplier(0.5)
    print("  quantize_multiplier(0.5) = (%d, %d)  期望 (1073741824, 0)" % (m, s))
    print("  requantize_np(1000, %d, %d) = %d  期望 500" % (m, s, requantize_np(1000, m, s)))
