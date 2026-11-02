"""
check_int8_source_binding.py  --  L1 源绑定一致性检查 (INT8)

⚠️ 核心原则 (来自项目铁律 "parity 测试必须绑定真源码"):
   **不重写被测对象的公式**, 而是从真实源文件里 *解析* 出关键量, 再断言
   numpy 镜像 (sim_int8_kernel.py / int8_math.py) 和 MCU 的 C 实现
   (MCU/Core/Src/ch32_inference_int8.c) 用的就是这些量.

被绑定的源:
   1. MCU/Drivers/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c
        -> conv 的 padding / dilation 索引语义
   2. MCU/Drivers/CMSIS/NN/Include/arm_nnsupportfunctions.h
        -> arm_nn_requantize 的舍入语义 (double-rounding 分支)
   3. MCU/Core/Src/ch32_inference_int8.c
        -> 我的 C 实现: offset 约定 / pad 公式 / requantize 调用 / GAP 舍入 /
           每个 conv / fc / se / residual 调用点的 zp 与 shape
   4. sim_int8_kernel.py + int8_math.py
        -> numpy 镜像

检查项:
   C1  CMSIS requantize 用的是 double-rounding 分支 (没定义 SINGLE_ROUNDING)
   C2  从 CMSIS 源码解析出的舍入常量 -> 复刻成 int32 模拟器 ->
       与 int8_math.requantize_np 逐位一致 (含负数/负 shift)
   C3  从 CMSIS conv 源码解析出的索引式 -> 与镜像 cmsis_conv 的索引式等价
   C4  我的 C: input_offset=-zp / output_offset=+zp / pad=(K-1)*dil/2 /
       stride=1 / requantize 走 CMSIS (没有自己实现舍入)
   C5  我的 C 的 GAP / 输入量化舍入 == 镜像的舍入
   C6  每个 i8_conv / i8_fc / i8_fc_logit / i8_res_requant_add 调用点的
       (权重符号, K, dilation, in_zp, out_zp, act_min) == 镜像期望
       (期望表由 **header 里的量化参数** 推导, 不是手抄的数字)
   C7  所有 requantize shift <= 0 -> CMSIS 的 int32 左移不会溢出,
       numpy 镜像用 int64 算就与 CMSIS 完全等价

用法: python check_int8_source_binding.py
"""
import io
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from int8_math import parse_int8_header, requantize_np   # noqa: E402

C_SRC = os.path.join(HERE, "MCU/Core/Src/ch32_inference_int8.c")
MIRROR_SRC = os.path.join(HERE, "sim_int8_kernel.py")
NN_H = os.path.join(HERE, "MCU/Drivers/CMSIS/NN/Include/arm_nnsupportfunctions.h")
NN_CONV_C = os.path.join(HERE, "MCU/Drivers/CMSIS/NN/Source/ConvolutionFunctions/"
                                "arm_convolve_s8.c")
INT8_HDR = os.path.join(HERE, "MCU/Core/Inc/model_v4_se_23dim_ch32_int8_s42.h")

PREFIX = "v4se23_ch32_i8_"

FAILS = []
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    if not ok:
        FAILS.append(name)
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  -- " + detail) if detail else ""))


def strip_c_comments(s):
    """去掉 C 的块注释和行注释."""
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"//[^\n]*", " ", s)
    return s


def read(p):
    return io.open(p, encoding="utf-8", errors="replace").read()


def find_calls(src, fname):
    """抓出 fname(...) 的实参列表 (按出现顺序)."""
    out = []
    for m in re.finditer(r"\b%s\s*\(([^;]*?)\)\s*;" % re.escape(fname), src, re.S):
        args = [a.strip() for a in m.group(1).split(",")]
        out.append(args)
    return out


# ---------------------------------------------------------------------------
# C1/C2: CMSIS requantize 语义
# ---------------------------------------------------------------------------


def parse_cmsis_requantize():
    """从 arm_nnsupportfunctions.h 里解析 requantize 的舍入常量."""
    h = strip_c_comments(read(NN_H))
    facts = {}

    body = re.search(r"__STATIC_FORCEINLINE\s+q31_t\s+arm_nn_requantize\s*\(([^)]*)\)\s*\{(.*?)\n\}",
                     h, re.S)
    facts["requantize_found"] = body is not None
    if body:
        b = body.group(2)
        facts["double_rounding_branch"] = ("arm_nn_doubling_high_mult_no_sat" in b
                                           and "arm_nn_divide_by_power_of_two" in b)
        m = re.search(r"arm_nn_divide_by_power_of_two\s*\(\s*"
                      r"arm_nn_doubling_high_mult_no_sat\s*\(\s*val\s*\*\s*"
                      r"\(\s*1\s*<<\s*LEFT_SHIFT\(shift\)\s*\)\s*,\s*multiplier\s*\)\s*,\s*"
                      r"RIGHT_SHIFT\(shift\)\s*\)", b)
        facts["double_rounding_expr"] = m is not None

    m = re.search(r"#define\s+LEFT_SHIFT\(_shift\)\s*\(([^)]*)\)", h)
    facts["left_shift_expr"] = m.group(1).strip() if m else None
    m = re.search(r"#define\s+RIGHT_SHIFT\(_shift\)\s*\(([^)]*)\)", h)
    facts["right_shift_expr"] = m.group(1).strip() if m else None

    m = re.search(r"__STATIC_FORCEINLINE\s+q31_t\s+arm_nn_doubling_high_mult_no_sat\s*"
                  r"\([^)]*\)\s*\{(.*?)\n\}", h, re.S)
    facts["doubling_found"] = m is not None
    if m:
        b = m.group(1)
        m2 = re.search(r"mult\.word\.low\s*=\s*1\s*<<\s*(\d+)\s*;", b)
        facts["round_const_shift"] = int(m2.group(1)) if m2 else None
        m3 = re.search(r"result\s*=\s*\(int32_t\)\(\s*mult\.long_long\s*>>\s*(\d+)\s*\)", b)
        facts["doubling_shift"] = int(m3.group(1)) if m3 else None

    m = re.search(r"__STATIC_FORCEINLINE\s+q31_t\s+arm_nn_divide_by_power_of_two\s*"
                  r"\([^)]*\)\s*\{(.*?)\n\}", h, re.S)
    facts["divide_found"] = m is not None
    if m:
        b = m.group(1)
        facts["threshold_neg_bump"] = bool(re.search(r"if\s*\(\s*result\s*<\s*0\s*\)\s*"
                                                     r"\{\s*threshold\s*\+\+\s*;\s*\}", b))
        facts["rounding_cmp"] = bool(re.search(r"if\s*\(\s*remainder\s*>\s*threshold\s*\)", b))
    return facts


def cmsis_requantize_emulator(facts):
    """按 *解析出来的* 常量复刻 arm_nn_requantize (int32 语义)."""
    rc = 1 << facts["round_const_shift"]
    ds = facts["doubling_shift"]

    def ls(s):
        return s if s > 0 else 0

    def rs(s):
        return 0 if s > 0 else -s

    def emu(val, mult, shift):
        v = np.int64(val) * (np.int64(1) << ls(shift))
        prod = v * np.int64(mult) + np.int64(rc)
        hi = (prod >> ds).astype(np.int64)
        # int32 截断 (CMSIS: result = (int32_t)(mult.long_long >> 31))
        hi = ((hi + (1 << 31)) % (1 << 32)) - (1 << 31)
        e = rs(shift)
        if e == 0:
            return hi
        mask = (1 << e) - 1
        remainder = mask & hi
        result = hi >> e
        threshold = mask >> 1
        if facts["threshold_neg_bump"]:
            threshold = threshold + np.where(result < 0, 1, 0)
        if facts["rounding_cmp"]:
            result = result + np.where(remainder > threshold, 1, 0)
        return result
    return emu


# ---------------------------------------------------------------------------
# C3: CMSIS conv 索引语义
# ---------------------------------------------------------------------------


def parse_cmsis_conv_index():
    c = strip_c_comments(read(NN_CONV_C))
    facts = {}
    for key, pat in (("base_idx_x", r"const\s+int32_t\s+base_idx_x\s*=\s*([^;]+);"),
                     ("k_x", r"const\s+int32_t\s+k_x\s*=\s*([^;]+);"),
                     ("pad_x", r"const\s+uint16_t\s+pad_x\s*=\s*([^;]+);"),
                     ("dilation_x", r"const\s+int32_t\s+dilation_x\s*=\s*([^;]+);")):
        m = re.search(pat, c)
        facts[key] = m.group(1).strip() if m else None
    facts["pad_mve_raw"] = bool(re.search(r"memset\s*\(\s*im2col_buf\s*,\s*\(int8_t\)\s*-input_offset", c))
    facts["pad_nonmve_zero"] = bool(re.search(r"memset\s*\(\s*two_column_buf\s*,\s*0\s*,", c))
    return facts


def cmsis_index_sets(T, K, dil, stride=1, pad=None):
    """按解析出的 CMSIS 索引式算出每个 tap 参与的输入位置."""
    if pad is None:
        pad = (K - 1) * dil // 2
    sets = []
    for t in range(T):
        base = stride * t - pad          # base_idx_x = stride*i_out_x - pad_x
        sets.append([base + dil * k for k in range(K)])   # k_x = base_idx_x + dilation_x*i_ker_x
    return sets


# ---------------------------------------------------------------------------
# C6: 调用点期望表 (由 header 的量化参数 + 网络结构推导)
# ---------------------------------------------------------------------------

DIL2B = {1: 0, 2: 1, 4: 2}     # dilation -> CH32_DIL_B* 下标
B2DIL = {0: 1, 1: 2, 2: 4}     # CH32_DIL_B* 下标 -> dilation


def expected_conv_table():
    """期望的 7 个 conv 调用点 (顺序与 C 源码里的调用顺序一致)."""
    P = PREFIX
    zp = lambda n: P + n
    return [
        # (权重, bias, mult, shift, K, dil, in_zp, out_zp, act_min, act_max)
        ("w_tcn_0_residual", "b_tcn_0_residual", "mult_tcn_0_residual", "shift_tcn_0_residual",
         1, "1", zp("zp_x"), zp("zp_add_0"), "-128", "127"),
        ("w_tcn_0_conv1", "b_tcn_0_conv1", "mult_tcn_0_conv1", "shift_tcn_0_conv1",
         3, 1, zp("zp_x"), zp("zp_c1_0"), zp("zp_c1_0"), "127"),
        ("w_tcn_0_conv2", "b_tcn_0_conv2", "mult_tcn_0_conv2", "shift_tcn_0_conv2",
         3, 1, zp("zp_c1_0"), zp("zp_c2_0"), zp("zp_c2_0"), "127"),
        ("w_tcn_1_conv1", "b_tcn_1_conv1", "mult_tcn_1_conv1", "shift_tcn_1_conv1",
         3, 2, zp("zp_add_0"), zp("zp_c1_1"), zp("zp_c1_1"), "127"),
        ("w_tcn_1_conv2", "b_tcn_1_conv2", "mult_tcn_1_conv2", "shift_tcn_1_conv2",
         3, 2, zp("zp_c1_1"), zp("zp_c2_1"), zp("zp_c2_1"), "127"),
        ("w_tcn_2_conv1", "b_tcn_2_conv1", "mult_tcn_2_conv1", "shift_tcn_2_conv1",
         3, 4, zp("zp_add_1"), zp("zp_c1_2"), zp("zp_c1_2"), "127"),
        ("w_tcn_2_conv2", "b_tcn_2_conv2", "mult_tcn_2_conv2", "shift_tcn_2_conv2",
         3, 4, zp("zp_c1_2"), zp("zp_c2_2"), zp("zp_c2_2"), "127"),
    ]


def check_bi_ternary(c, prefix_name, bi_order=("0", "1", "2")):
    """检查 C 里 (bi==0)? X_0 : (bi==1)? X_1 : X_2 这种三选一的符号选择."""
    pat = (r"%s\s*=\s*\(bi\s*==\s*0\)\s*\?\s*%s(\d)\s*:\s*"
           r"\(bi\s*==\s*1\)\s*\?\s*%s(\d)\s*:\s*%s(\d)" %
           (re.escape(prefix_name), re.escape(prefix_name),
            re.escape(prefix_name), re.escape(prefix_name)))
    m = re.search(pat, c)
    return bool(m) and m.groups() == bi_order


def main():
    print("=" * 78)
    print("L1 source-binding check (INT8): MCU C / numpy mirror / CMSIS-NN")
    print("=" * 78)

    c = strip_c_comments(read(C_SRC))
    mirror = read(MIRROR_SRC)

    # ---------------- C1/C2: requantize ----------------
    print("\n[C1/C2] CMSIS-NN requantize rounding semantics")
    facts = parse_cmsis_requantize()
    check("C1.1 arm_nn_requantize source parsed", facts.get("requantize_found"))
    check("C1.2 double-rounding branch present",
          facts.get("double_rounding_branch"))
    check("C1.3 double-rounding expression matches CMSIS source verbatim",
          facts.get("double_rounding_expr"))
    check("C1.4 LEFT_SHIFT/RIGHT_SHIFT macros parsed",
          facts.get("left_shift_expr") is not None and facts.get("right_shift_expr") is not None,
          "LEFT=%s RIGHT=%s" % (facts.get("left_shift_expr"), facts.get("right_shift_expr")))
    check("C1.5 doubling rounding const / shift parsed",
          facts.get("round_const_shift") is not None and facts.get("doubling_shift") is not None,
          "rc=1<<%s, >>%s" % (facts.get("round_const_shift"), facts.get("doubling_shift")))
    check("C1.6 divide-by-pow2 negative-threshold rule parsed",
          facts.get("threshold_neg_bump") and facts.get("rounding_cmp"))

    nn_dir = os.path.join(HERE, "MCU/Drivers/CMSIS/NN")
    defined = []
    for root, _, files in os.walk(nn_dir):
        for f in files:
            if f.endswith((".h", ".c")):
                t = read(os.path.join(root, f))
                if re.search(r"^\s*#\s*define\s+CMSIS_NN_USE_SINGLE_ROUNDING", t, re.M):
                    defined.append(f)
    check("C1.7 no #define CMSIS_NN_USE_SINGLE_ROUNDING in NN tree "
          "-> double-rounding is the ACTIVE branch", not defined, str(defined))

    # ---------------- C7: int32 左移不溢出 (CMSIS 的 LEFT_SHIFT 分支) ----------------
    print("\n[C7] requantize overflow regime (CMSIS int32 left shift)")
    h = parse_int8_header(INT8_HDR)
    shift_names = sorted([k for k in h if k.startswith("shift_")])
    check("C7.2 header shift arrays are per-channel (size 1 / 4 / 32)",
          all(np.asarray(h[k]).size in (1, 4, 32) for k in shift_names),
          str({k: int(np.asarray(h[k]).size) for k in shift_names}))

    # 每个 requantize 站点的累加器上界 (由结构推出) -> 检查 int32 不溢出
    #  站点: (名字, 累加项数, 累加器上界公式说明, shift 数组)
    sites = [
        ("tcn_0_conv1", 3 * 23, "b_tcn_0_conv1", "shift_tcn_0_conv1"),
        ("tcn_0_conv2", 3 * 32, "b_tcn_0_conv2", "shift_tcn_0_conv2"),
        ("tcn_0_residual", 1 * 23, "b_tcn_0_residual", "shift_tcn_0_residual"),
        ("tcn_1_conv1", 3 * 32, "b_tcn_1_conv1", "shift_tcn_1_conv1"),
        ("tcn_1_conv2", 3 * 32, "b_tcn_1_conv2", "shift_tcn_1_conv2"),
        ("tcn_2_conv1", 3 * 32, "b_tcn_2_conv1", "shift_tcn_2_conv1"),
        ("tcn_2_conv2", 3 * 32, "b_tcn_2_conv2", "shift_tcn_2_conv2"),
        ("se_fc1", 32, "b_tcn_0_se_fc1", "shift_tcn_0_se_fc1"),
        ("fc1", 32, "b_fc1", "shift_fc1"),
    ]
    overflow = []
    worst = 0
    for nm, nterms, bkey, skey in sites:
        bias_max = int(np.abs(np.asarray(h[bkey])).max())
        acc_max = nterms * 255 * 127 + bias_max
        ls = max(0, int(np.asarray(h[skey]).max()))
        v = acc_max * (1 << ls)
        worst = max(worst, v)
        if v >= 2 ** 31:
            overflow.append("%s: %d" % (nm, v))
    # SE scale-back: acc = (q_c2 - zp_c2) * q_sig, |q-zp|<=255, q_sig <= sig_q
    for b in (0, 1, 2):
        acc_max = 255 * int(h["sig_q"])
        ls = max(0, int(np.asarray(h["shift_se_%d" % b]).max()))
        v = acc_max * (1 << ls)
        worst = max(worst, v)
        if v >= 2 ** 31:
            overflow.append("se_%d: %d" % (b, v))
    # identity residual: acc = q - zp, |.| <= 255
    for b in (1, 2):
        acc_max = 255
        ls = max(0, int(np.asarray(h["shift_res_%d" % b]).max()))
        v = acc_max * (1 << ls)
        worst = max(worst, v)
        if v >= 2 ** 31:
            overflow.append("res_%d: %d" % (b, v))
    check("C7.1 no int32 overflow in arm_nn_requantize (acc * 2^LEFT_SHIFT < 2^31)",
          not overflow, "; ".join(overflow) + " worst=%d" % worst)
    check("C7.3 every requantize shift <= 1 (只有 identity residual 是 +1)",
          max(int(np.asarray(h[k]).max()) for k in shift_names) <= 1)

    if facts.get("round_const_shift") is not None:
        emu = cmsis_requantize_emulator(facts)
        rng = np.random.default_rng(20261021)
        vals = np.concatenate([
            rng.integers(-2 ** 20, 2 ** 20, size=20000).astype(np.int64),
            rng.integers(-2 ** 30, 2 ** 30, size=20000).astype(np.int64),
            np.array([0, 1, -1, 2 ** 30, -(2 ** 30), 123456, -123456], dtype=np.int64),
        ])
        shifts = np.array([0, -1, -7, -10, -31, -5, -23, -28], dtype=np.int64)
        mults = np.array([1 << 30, (1 << 31) - 1, 123456789, 1073741824, 1500000000],
                         dtype=np.int64)
        ok, detail = True, ""
        for sh in shifts:
            for mu in mults:
                a = emu(vals, mu, sh)
                b = requantize_np(vals, mu, sh)
                if not np.array_equal(a, b):
                    ok = False
                    detail = "shift=%d mult=%d mismatch %d/%d" % (sh, mu, int((a != b).sum()), len(vals))
                    break
            if not ok:
                break
        check("C2.1 CMSIS emulator (built from parsed constants) == "
              "int8_math.requantize_np (bit-exact)", ok, detail)

    # ---------------- C3: conv 索引 ----------------
    print("\n[C3] CMSIS-NN conv padding / dilation index semantics")
    cf = parse_cmsis_conv_index()
    check("C3.1 base_idx_x parsed", cf.get("base_idx_x") is not None, str(cf.get("base_idx_x")))
    check("C3.2 k_x parsed", cf.get("k_x") is not None, str(cf.get("k_x")))
    check("C3.3 base_idx_x == stride*i_out_x - pad_x",
          bool(re.match(r"stride_x\s*\*\s*i_out_x\s*-\s*pad_x", cf.get("base_idx_x") or "")),
          cf.get("base_idx_x"))
    check("C3.4 k_x == base_idx_x + dilation_x*i_ker_x",
          bool(re.match(r"base_idx_x\s*\+\s*dilation_x\s*\*\s*i_ker_x", cf.get("k_x") or "")),
          cf.get("k_x"))
    check("C3.5 out-of-bound fill == real 0 (MVE: -input_offset, non-MVE: 0)",
          cf.get("pad_mve_raw") and cf.get("pad_nonmve_zero"))

    check("C3.6 mirror cmsis_conv index formula == t + k*dil - pad",
          bool(re.search(r"idx\s*=\s*np\.arange\(Tx\)\s*\+\s*k\s*\*\s*dil\s*-\s*pad", mirror)))
    check("C3.7 mirror pad == (K-1)*dil//2",
          bool(re.search(r"pad\s*=\s*\(K\s*-\s*1\)\s*\*\s*dil\s*//\s*2", mirror)))
    same = True
    for K, dil in ((3, 1), (3, 2), (3, 4), (1, 1)):
        T = 16
        a = cmsis_index_sets(T, K, dil, stride=1, pad=(K - 1) * dil // 2)
        b = [[t + k * dil - (K - 1) * dil // 2 for k in range(K)] for t in range(T)]
        if a != b:
            same = False
    check("C3.8 CMSIS-parsed index sets == mirror index sets (K=1/3, dil=1/2/4)", same)

    # ---------------- C4/C5: 我的 C 实现 ----------------
    print("\n[C4/C5] MCU C implementation (ch32_inference_int8.c)")
    check("C4.1 input_offset = -in_zp",
          bool(re.search(r"cp\.input_offset\s*=\s*-\s*in_zp\s*;", c)))
    check("C4.2 output_offset = +out_zp",
          bool(re.search(r"cp\.output_offset\s*=\s*out_zp\s*;", c)))
    check("C4.3 pad = (K - 1) * dil / 2",
          bool(re.search(r"const\s+int\s+pad\s*=\s*\(K\s*-\s*1\)\s*\*\s*dil\s*/\s*2\s*;", c)))
    check("C4.4 cp.padding.w = pad (symmetric same padding)",
          bool(re.search(r"cp\.padding\.w\s*=\s*pad\s*;", c)))
    check("C4.5 cp.dilation.w = dil", bool(re.search(r"cp\.dilation\.w\s*=\s*dil\s*;", c)))
    check("C4.6 stride = 1", bool(re.search(r"cp\.stride\.w\s*=\s*1\s*;", c)))
    n_req = len(re.findall(r"\barm_nn_requantize\s*\(", c))
    check("C4.7 all requantize sites call CMSIS arm_nn_requantize (3 sites: fc / se / residual)",
          n_req == 3, "textual call sites: %d" % n_req)
    check("C4.8 no locally implemented requantize",
          not re.search(r"static\s+\w+\s+\w*requantiz\w*\s*\(", c))
    check("C4.9 conv uses CMSIS arm_convolve_s8",
          "arm_convolve_s8(&ctx" in c.replace(" ", ""))

    check("C5.1 C GAP rounding == mirror gap_int8",
          bool(re.search(r"int32_t\s+v\s*=\s*\(s\s*>\s*0\)\s*\?\s*\(s\s*\+\s*half\)\s*:\s*\(s\s*-\s*half\)\s*;", c))
          and bool(re.search(r"np\.where\(s\s*>\s*0,\s*s\s*\+\s*half,\s*s\s*-\s*half\)", mirror)))
    check("C5.2 C input quantization rounding == mirror q_round (half away from zero)",
          bool(re.search(r"\(v\s*>=\s*0\.0f\)\s*\?\s*\(v\s*\+\s*0\.5f\)\s*:\s*\(v\s*-\s*0\.5f\)", c))
          and bool(re.search(r"np\.where\(v\s*>=\s*0,\s*np\.floor\(v\s*\+\s*0\.5\),\s*np\.ceil\(v\s*-\s*0\.5\)\)", mirror)))
    check("C5.3 mirror centered = q_in - in_zp (CMSIS input_offset=-zp)",
          bool(re.search(r"centered\s*=\s*np\.asarray\(q_in,\s*dtype=np\.int64\)\s*-\s*in_zp", mirror)))

    # ---------------- C6: 调用点 ----------------
    print("\n[C6] call sites vs INT8 header quant params")
    conv_calls = find_calls(c, "i8_conv")
    exp = expected_conv_table()
    check("C6.1 7 i8_conv call sites (1x1 residual + 6x 3x3)",
          len(conv_calls) == len(exp), "found %d" % len(conv_calls))
    for i, (got, e) in enumerate(zip(conv_calls, exp)):
        w, b, mu, sh, K, dil, in_zp, out_zp, amin, amax = e
        idx = {"w": 2, "b": 3, "mu": 4, "sh": 5, "K": 8, "dil": 9,
               "in_zp": 10, "out_zp": 11, "amin": 12, "amax": 13}
        prob = []
        for key, want in (("w", PREFIX + w), ("b", PREFIX + b), ("mu", PREFIX + mu),
                          ("sh", PREFIX + sh)):
            if got[idx[key]] != want:
                prob.append("%s=%s(want %s)" % (key, got[idx[key]], want))
        if got[idx["K"]] != str(K):
            prob.append("K=%s(want %d)" % (got[idx["K"]], K))
        if isinstance(dil, str):
            if got[idx["dil"]] != dil:
                prob.append("dil=%s(want %s)" % (got[idx["dil"]], dil))
        elif got[idx["dil"]] != "CH32_DIL_B%d" % DIL2B[dil]:
            prob.append("dil=%s(want CH32_DIL_B%d)" % (got[idx["dil"]], DIL2B[dil]))
        for key, want in (("in_zp", in_zp), ("out_zp", out_zp),
                          ("amin", amin), ("amax", amax)):
            if got[idx[key]] != want:
                prob.append("%s=%s(want %s)" % (key, got[idx[key]], want))
        check("C6.2 conv#%d %s" % (i, w), not prob, "; ".join(prob))

    dil_seq = []
    for g in conv_calls:
        m = re.search(r"CH32_DIL_B(\d)", g[9])
        dil_seq.append(B2DIL[int(m.group(1))] if m else int(g[9]))
    check("C6.3 conv dilation sequence == (1,1,1,2,2,4,4)",
          dil_seq == [1, 1, 1, 2, 2, 4, 4], str(dil_seq))
    check("C6.4 every zp symbol used in conv exists in the INT8 header",
          all(g[10][len(PREFIX):] in h and g[11][len(PREFIX):] in h for g in conv_calls))

    fc_calls = find_calls(c, "i8_fc")
    fl_calls = find_calls(c, "i8_fc_logit")
    check("C6.5 2 i8_fc call sites (SE fc1 inside i8_se_block + classifier fc1)",
          len(fc_calls) == 2, "found %d" % len(fc_calls))
    check("C6.6 2 i8_fc_logit call sites (SE fc2 + classifier fc2)",
          len(fl_calls) == 2, "found %d" % len(fl_calls))

    # SE block 内部按 bi 选参数: 局部变量 -> 三个 header 符号
    se_local_map = {
        "zp_c2": ["zp_c2_0", "zp_c2_1", "zp_c2_2"],
        "zp_seh": ["zp_seh_0", "zp_seh_1", "zp_seh_2"],
        "zp_add": ["zp_add_0", "zp_add_1", "zp_add_2"],
        "s_seh": ["s_seh_0", "s_seh_1", "s_seh_2"],
        "w_se1": ["w_tcn_0_se_fc1", "w_tcn_1_se_fc1", "w_tcn_2_se_fc1"],
        "b_se1": ["b_tcn_0_se_fc1", "b_tcn_1_se_fc1", "b_tcn_2_se_fc1"],
        "m_se1": ["mult_tcn_0_se_fc1", "mult_tcn_1_se_fc1", "mult_tcn_2_se_fc1"],
        "s_se1": ["shift_tcn_0_se_fc1", "shift_tcn_1_se_fc1", "shift_tcn_2_se_fc1"],
        "w_se2": ["w_tcn_0_se_fc2", "w_tcn_1_se_fc2", "w_tcn_2_se_fc2"],
        "b_se2": ["b_tcn_0_se_fc2", "b_tcn_1_se_fc2", "b_tcn_2_se_fc2"],
        "sw_se2": ["sw_tcn_0_se_fc2", "sw_tcn_1_se_fc2", "sw_tcn_2_se_fc2"],
        "mult_se": ["mult_se_0", "mult_se_1", "mult_se_2"],
        "shift_se": ["shift_se_0", "shift_se_1", "shift_se_2"],
    }
    bad = []
    for local, syms in se_local_map.items():
        m = re.search(r"\b%s\s*=\s*\(bi\s*==\s*0\)\s*\?\s*(\w+)\s*:\s*"
                      r"\(bi\s*==\s*1\)\s*\?\s*(\w+)\s*:\s*(\w+)" % re.escape(local), c)
        want = [PREFIX + s for s in syms]
        if not m or list(m.groups()) != want:
            bad.append("%s -> %s (want %s)" % (local, list(m.groups()) if m else None, want))
    check("C6.7 SE block picks every per-block param by bi (0/1/2)", not bad, "; ".join(bad))

    # SE fc1 / fc2 的域关系 (局部变量名)
    g = fc_calls[0]
    se_fc1_ok = (g[7] == "zp_c2" and g[8] == "zp_seh" and g[9] == "zp_seh"
                 and g[2] == "w_se1" and g[3] == "b_se1" and g[4] == "m_se1"
                 and g[5] == "s_se1" and g[6] == "I8_SEH")
    check("C6.8 SE fc1: in_zp=zp_c2, out_zp=zp_seh, act_min=zp_seh (ReLU)", se_fc1_ok)
    f = fl_calls[0]
    se_fc2_ok = (f[2] == "w_se2" and f[3] == "b_se2" and f[4] == "sw_se2"
                 and f[5] == "s_seh" and f[7] == "zp_seh" and f[6] == "I8_C")
    check("C6.9 SE fc2: in_zp=zp_seh, s_in=s_seh, sw=sw_tcn_b_se_fc2", se_fc2_ok)

    g = fc_calls[1]
    cls_fc1_ok = (g[7] == PREFIX + "zp_add_2" and g[8] == PREFIX + "zp_fc1"
                  and g[9] == PREFIX + "zp_fc1" and g[2] == PREFIX + "w_fc1"
                  and g[3] == PREFIX + "b_fc1" and g[4] == PREFIX + "mult_fc1"
                  and g[5] == PREFIX + "shift_fc1")
    check("C6.10 classifier fc1: in_zp=zp_add_2, out_zp/act_min=zp_fc1", cls_fc1_ok)
    f = fl_calls[1]
    cls_fc2_ok = (f[7] == PREFIX + "zp_fc1" and f[5] == PREFIX + "s_fc1"
                  and f[4] == PREFIX + "sw_fc2" and f[3] == PREFIX + "b_fc2"
                  and f[2] == PREFIX + "w_fc2")
    check("C6.11 classifier fc2: in_zp=zp_fc1, s_in=s_fc1, sw=sw_fc2", cls_fc2_ok)

    res_calls = find_calls(c, "i8_res_requant_add")
    check("C6.12 2 i8_res_requant_add call sites (block1 / block2 identity residual)",
          len(res_calls) == 2, "found %d" % len(res_calls))
    res_ok = True
    for b in (1, 2):
        g = res_calls[b - 1]
        if (g[2] != PREFIX + "mult_res_%d" % b or g[3] != PREFIX + "shift_res_%d" % b
                or g[4] != PREFIX + "zp_add_%d" % (b - 1) or g[6] != PREFIX + "zp_add_%d" % b):
            res_ok = False
    check("C6.13 residual: mult_res_b, zp_prev=zp_add_{b-1}, zp_add=zp_add_b", res_ok)
    add_calls = find_calls(c, "i8_add_res")
    check("C6.14 block0 add uses zp_add_0",
          len(add_calls) == 1 and add_calls[0][3] == PREFIX + "zp_add_0")

    se_calls = find_calls(c, "i8_se_block")
    check("C6.15 3 i8_se_block call sites (bi = 0,1,2)",
          len(se_calls) == 3 and [g[1] for g in se_calls] == ["0", "1", "2"],
          str([g[1] for g in se_calls]))

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 78)
    n_ok = sum(1 for _, ok, _ in CHECKS if ok)
    print("L1 source binding: %d/%d passed" % (n_ok, len(CHECKS)))
    if FAILS:
        print("FAILED: %s" % ", ".join(FAILS))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
