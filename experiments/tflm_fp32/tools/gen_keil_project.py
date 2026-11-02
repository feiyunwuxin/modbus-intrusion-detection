#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen_keil_project.py  --  从 H743test.uvprojx 生成一个**新的**独立 TFLM Keil 工程

设计前提 (来自手写 kernel 路线踩过的坑):
  1. **不复制 app 源码**: 新工程引用同一批 ../Core/Src/*.c 文件,
     只换推理后端 (ch32_inference_tflm.c 顶替 ch32_inference.c /
     ch32_inference_int8.c), 所以 app 层行为天然一致, 数字可同口径对比.
  2. **不动既有工程**: H743test.uvprojx 的两个 target (H743test / H743test_INT8)
     一个字节都不改. 本脚本只**读**它, 只**写**H743test_TFLM.uvprojx.
  3. **.uvprojx 是 uVision 的真实配置**: 所有 include path / 宏 / 编译选项
     以文件内容为准, 不信 GUI 显示. 所以这里用 XML 精确改写, 不靠手工点.
  4. **C++ 只在 .cc 文件上打开**: libcxx 的 include path 与 -std=c++17 等
     全部走 per-file 设置, 避免 C 文件被 libcxx 的 <stdint.h> 污染
     (2026-10-15 旧路线的 libcxx 踩坑, 见 tflm_defs.h 注释).
  5. **libcxx 宏走 --include 而不是 <Define>**: <Define> 同时传给 armasm,
     __ORDER_LITTLE_ENDIAN__ 这种 token 在汇编器里是未定义符号 -> A1516E.
     所以 tflm_defs.h 里的宏全部用数值 (1234/4321/3412).

用法:
    python experiments/tflm_fp32/tools/gen_keil_project.py
    python experiments/tflm_fp32/tools/gen_keil_project.py --check-only
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MDK = os.path.join(ROOT, "MCU", "MDK-ARM")
SRC_PROJ = os.path.join(MDK, "H743test.uvprojx")
OUT_PROJ = os.path.join(MDK, "H743test_TFLM.uvprojx")

TARGET_NAME = "H743test_TFLM"
SCATTER = "H743_TFLM_arena.sct"

# --------------------------------------------------------------------------
# include path
# --------------------------------------------------------------------------
# 工程级 (所有 .c 可见): HAL/CMSIS + TFLM 头 + flatbuffers 头.
# ⚠ 故意**不加** libcxx: C 文件不能看见 libcxx 的 <stdint.h>.
PROJ_INC = ";".join([
    "../Core/Inc",
    "../Drivers/STM32H7xx_HAL_Driver/Inc",
    "../Drivers/STM32H7xx_HAL_Driver/Inc/Legacy",
    "../Drivers/CMSIS/Device/ST/STM32H7xx/Include",
    "../Drivers/CMSIS/Include",
    "../thirdParty/tflite-micro",
    "../thirdParty/flatbuffers/include",
])

# per-file (.cc) 才追加: AC6 自带的 libcxx + C 头
LIBCXX_INC = ";".join([
    r"C:\Keil_v5\ARM\ARMCLANG\include\libcxx",
    r"C:\Keil_v5\ARM\ARMCLANG\include",
])
CPP_INC = PROJ_INC + ";" + LIBCXX_INC

# 工程级 MiscControls: 只注入 libcxx 宏 fallback, 对 .c 无害 (全部 #ifndef 包裹)
PROJ_MISC = "--include ../Core/Inc/tflm_defs.h"
# per-file (.cc): 宏 + C++17 + 关异常/RTTI (TFLM 不用异常)
#
# ⚠⚠ `-x c++` 是**必须**的, 不是锦上添花:
#   本机 uVision (Keil v5.43) 对 .cc/.cpp 文件:
#     - <FileType>5</FileType> (= "C++ Source file") -> **静默跳过**, 根本不编译,
#       日志里连一行 "compiling xxx.cc" 都没有 (2026-10-26 实测, 9 种 FileType
#       + 4 种 UseCPPCompiler 组合全试过);
#     - <FileType>1</FileType> -> 会被编译, 但 uVision 传给 armclang 的语言
#       是 C -> "error: invalid argument '-std=c++17' not allowed with 'C'";
#   唯一可行的写法: FileType=1 + per-file MiscControls 里显式 `-x c++`.
#   uVision 自己的 `-x c` 排在 MiscControls **之前**, 所以我们的 `-x c++` 生效.
#   (实测: probe.cpp 带 `namespace` + `extern "C"`, 加 -x c++ 后干净编译)
CPP_MISC = ("--include ../Core/Inc/tflm_defs.h -x c++ -std=c++17 "
            "-fno-exceptions -fno-rtti")

# --------------------------------------------------------------------------
# 文件清单
# --------------------------------------------------------------------------
# app 侧: 与 H743test 相同, 但用 TFLM 后端顶替两个手写后端.
# (ch32_inference.c / ch32_inference_int8.c **不加入** —— 符号会跟
#  ch32_inference_tflm.c 撞车; 既有工程里它们原样保留, 不受影响.)
APP_CORE = [
    "main.c",
    "gpio.c",
    "dma.c",
    "tim.c",
    "usart.c",
    "stm32h7xx_it.c",
    "stm32h7xx_hal_msp.c",
    "modbus_ids.c",
    "sliding_window.c",
    "feature_extractor.c",
    "modbus_rtu.c",
    # --- TFLM 后端 ---
    "ch32_inference_tflm.c",
    "model_tflm_fp32_s42_data.c",
]

TFLM_RUNTIME = ["tflm_runtime.cpp"]

# TFLM 源文件子集 (只收 flatbuffer 里真用到的 9 种 op 所需 kernel).
# 头部 (reference/*.h) 是 header-only, 不用编译.
TFLM_CORE = [
    "micro_interpreter.cc",
    "micro_interpreter_graph.cc",
    "micro_interpreter_context.cc",
    "micro_allocator.cc",
    "micro_allocation_info.cc",
    "micro_common.cc",
    "micro_context.cc",
    "micro_log.cc",
    "micro_op_resolver.cc",
    "micro_profiler.cc",
    "micro_resource_variable.cc",
    "micro_time.cc",
    "micro_utils.cc",
    "debug_log.cc",
    "hexdump.cc",
    "flatbuffer_conversions.cc",
    "flatbuffer_utils.cc",
    "memory_helpers.cc",
]

TFLM_ARENA = [
    "arena_allocator/single_arena_buffer_allocator.cc",
    "arena_allocator/non_persistent_arena_buffer_allocator.cc",
    "arena_allocator/persistent_arena_buffer_allocator.cc",
]

TFLM_PLANNER = [
    "memory_planner/greedy_memory_planner.cc",
    "memory_planner/linear_memory_planner.cc",
    "memory_planner/non_persistent_buffer_planner_shim.cc",
]

TFLM_KERNELS_INT = [
    "kernels/internal/common.cc",
    "kernels/internal/portable_tensor_utils.cc",
    "kernels/internal/quantization_util.cc",
    "kernels/internal/runtime_shape.cc",
    "kernels/internal/tensor_ctypes.cc",
]

TFLM_KERNELS = [
    # kernel_util.cc 是所有 kernel 的公共底座 (RegisterOp / GetEvalInput /
    # GetEvalOutput / HaveSameShapes / PopulateConvolutionQuantizationParams /
    # CalculateActivationRangeQuantized / RuntimePaddingType /
    # CreateWritableTensorDimsWithCopy 都在这里) -> 少它链接不过.
    "kernels/kernel_util.cc",
    "kernels/conv.cc",
    "kernels/conv_common.cc",
    "kernels/fully_connected.cc",
    "kernels/fully_connected_common.cc",
    "kernels/reduce.cc",
    "kernels/reduce_common.cc",
    "kernels/mul.cc",
    "kernels/mul_common.cc",
    "kernels/add.cc",
    "kernels/add_common.cc",
    "kernels/logistic.cc",
    "kernels/logistic_common.cc",
    "kernels/reshape.cc",
    "kernels/reshape_common.cc",
    "kernels/transpose.cc",
    "kernels/transpose_common.cc",
    "kernels/expand_dims.cc",
    "kernels/activations.cc",
    "kernels/activations_common.cc",
]

TFLM_ROOT = "../thirdParty/tflite-micro/tensorflow/lite/micro"


# --------------------------------------------------------------------------
# XML helpers
# --------------------------------------------------------------------------
def sub(parent, tag, text=None):
    """get-or-create a child element, optionally setting text."""
    el = parent.find(tag)
    if el is None:
        el = ET.SubElement(parent, tag)
    if text is not None:
        el.text = text
    return el


def file_elem(name, ftype, path, cpp=False):
    """Build a <File> element. cpp=True -> per-file C++ options.

    ⚠ FileType 一律用 1 (= "C Source file"). 本机 uVision 对 FileType=5
      ("C++ Source file") 会**静默跳过**该文件, 详见 CPP_MISC 处的注释.
      C++ 语义靠 per-file 的 `-x c++` 打开.
    """
    f = ET.Element("File")
    ET.SubElement(f, "FileName").text = name
    ET.SubElement(f, "FileType").text = "1"
    ET.SubElement(f, "FilePath").text = path
    if not cpp:
        return f

    opt = ET.SubElement(f, "FileOption")
    cp = ET.SubElement(opt, "CommonProperty")
    for k, v in [
        ("UseCPPCompiler", "1"),
        ("RVCTCodeConst", "0"),
        ("RVCTZI", "0"),
        ("RVCTOtherData", "0"),
        ("ModuleSelection", "0"),
        ("IncludeInBuild", "1"),
        ("AlwaysBuild", "2"),
        ("GenerateAssemblyFile", "2"),
        ("AssembleAssemblyFile", "2"),
        ("PublicsOnly", "2"),
        ("StopOnExitCode", "11"),
        ("CustomArgument", ""),
        ("IncludeLibraryModules", ""),
        ("ComprImg", "1"),
    ]:
        ET.SubElement(cp, k).text = v

    fad = ET.SubElement(opt, "FileArmAds")
    cads = ET.SubElement(fad, "Cads")
    for k, v in [
        ("interw", "1"), ("Optim", "1"), ("oTime", "0"), ("SplitLS", "0"),
        ("OneElfS", "1"), ("Strict", "0"), ("EnumInt", "0"), ("PlainCh", "0"),
        ("Ropi", "0"), ("Rwpi", "0"), ("wLevel", "2"), ("uThumb", "0"),
        ("uSurpInc", "0"), ("uC99", "1"), ("uGnu", "1"), ("useXO", "0"),
        ("v6Lang", "3"), ("v6LangP", "8"),
        ("vShortEn", "1"), ("vShortWch", "1"), ("v6Lto", "0"),
        ("v6WtE", "0"), ("v6Rtti", "0"),
    ]:
        ET.SubElement(cads, k).text = v
    vc = ET.SubElement(cads, "VariousControls")
    ET.SubElement(vc, "MiscControls").text = CPP_MISC
    ET.SubElement(vc, "Define").text = ""
    ET.SubElement(vc, "Undefine").text = ""
    ET.SubElement(vc, "IncludePath").text = CPP_INC
    return f


def group_elem(name, files):
    g = ET.Element("Group")
    ET.SubElement(g, "GroupName").text = name
    fs = ET.SubElement(g, "Files")
    for f in files:
        fs.append(f)
    return g


def build():
    if not os.path.isfile(SRC_PROJ):
        raise SystemExit(f"[FATAL] source project not found: {SRC_PROJ}")

    tree = ET.parse(SRC_PROJ)
    root = tree.getroot()

    # ---------- 复制一个 target 当模板 (target 0 = H743test, FP32 手写 kernel) ----------
    targets = root.find("Targets")
    tmpl = targets.findall("Target")[0]

    # ---------- 新 root: schema/version/header 复用, Targets 清空重建 ----------
    new_root = ET.Element("Project")
    new_root.set(
        "{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation",
        "project_projx.xsd",
    )
    for tag in ("SchemaVersion", "Header"):
        el = root.find(tag)
        if el is not None:
            new_root.append(copy.deepcopy(el))

    new_targets = ET.SubElement(new_root, "Targets")
    # ⚠ 必须整块复制 Target: 除了 TargetName/TargetOption/Groups, 还有
    #   ToolsetNumber / ToolsetName / pCCUsed / uAC6 四个元素决定**用哪套工具链**.
    #   漏掉 uAC6 -> uVision 以为要用 C51 (Keil 老 C 编译器), 报
    #   "requires 'C51' Toolchain which is not installed" 然后直接崩.
    tgt = copy.deepcopy(tmpl)
    new_targets.append(tgt)
    tgt.find("TargetName").text = TARGET_NAME
    # Groups 下面重新建, 先把旧的清空
    groups = tgt.find("Groups")
    for g in list(groups):
        groups.remove(g)

    # ---------- TargetOption 改写 ----------
    topt = tgt.find("TargetOption")
    tcom = topt.find("TargetCommonOption")
    tcom.find("OutputDirectory").text = TARGET_NAME + "\\"
    tcom.find("OutputName").text = TARGET_NAME
    for tag, val in [
        ("CreateExecutable", "1"),
        ("CreateHexFile", "1"),
    ]:
        el = tcom.find(tag)
        if el is not None:
            el.text = val

    tads = topt.find("TargetArmAds")
    cads = tads.find("Cads")
    vc = cads.find("VariousControls")
    # 宏: 既有 5 个 + TFLM 后端开关
    define = vc.find("Define")
    base_defs = [
        d for d in (define.text or "").split(",")
        if d and not d.startswith("CH32_")
    ]
    base_defs.append("CH32_BACKEND_TFLM=1")
    define.text = ",".join(base_defs)
    # 工程级 MiscControls: 只注入 libcxx 宏 fallback
    vc.find("MiscControls").text = PROJ_MISC
    # 工程级 IncludePath
    vc.find("IncludePath").text = PROJ_INC
    # <Undefine> 必须为空 (否则会主动 undef 掉 __ARMCOMPILER_LIBCXX)
    if vc.find("Undefine") is not None:
        vc.find("Undefine").text = ""

    # <Aads>: Define / MiscControls 必须清空 (armasm 不认 C 宏 token)
    aads = tads.find("Aads")
    if aads is not None:
        for tag in ("Define", "MiscControls", "Undefine"):
            el = aads.find(tag)
            if el is not None:
                el.text = ""

    # 链接脚本
    ld = tads.find("LDads")
    if ld is not None:
        sf = ld.find("ScatterFile")
        if sf is not None:
            sf.text = SCATTER

    # ---------- Groups 重建 ----------
    # 旧 group 作为 file 模板来源 (FileOption 照抄, 保持 Keil 默认行为一致)
    old = {g.findtext("GroupName"): g for g in tmpl.find("Groups").findall("Group")}

    def reuse_files(gname, names, cpp=False, pathfmt="../Core/Src/{}"):
        """从旧 group 里抄 FileOption 模板, 再按需改写."""
        g = ET.Element("Group")
        ET.SubElement(g, "GroupName").text = gname
        fs = ET.SubElement(g, "Files")
        for n in names:
            path = pathfmt.format(n)
            if cpp:
                fs.append(file_elem(n, 5, path, cpp=True))
            else:
                # 抄旧工程的 FileOption (保持 Keil 默认行为一致)
                src_f = None
                og = old.get(gname)
                if og is not None and og.find("Files") is not None:
                    for f in og.find("Files").findall("File"):
                        if f.findtext("FileName") == n:
                            src_f = f
                            break
                if src_f is not None:
                    nf = copy.deepcopy(src_f)
                    nf.find("FilePath").text = path
                    fs.append(nf)
                else:
                    fs.append(file_elem(n, 1, path))
        return g

    new_groups = groups
    # 1) 启动文件 (原样)
    g_startup = old.get("Application/MDK-ARM")
    new_groups.append(copy.deepcopy(g_startup))
    # 2) app 源码 (TFLM 后端)
    new_groups.append(reuse_files("Application/User/Core", APP_CORE))
    # 3) TFLM runtime (唯一手写 C++ TU)
    new_groups.append(reuse_files("Application/User/TFLM", TFLM_RUNTIME, cpp=True))
    # 4) TFLM 源码子集
    for gname, flist in [
        ("TFLite-Micro/Core", TFLM_CORE),
        ("TFLite-Micro/Arena", TFLM_ARENA),
        ("TFLite-Micro/MemoryPlanner", TFLM_PLANNER),
        ("TFLite-Micro/Kernels", TFLM_KERNELS),
        ("TFLite-Micro/KernelsInternal", TFLM_KERNELS_INT),
    ]:
        g = ET.Element("Group")
        ET.SubElement(g, "GroupName").text = gname
        fs = ET.SubElement(g, "Files")
        for rel in flist:
            fs.append(file_elem(
                os.path.basename(rel), 5,
                TFLM_ROOT + "/" + rel, cpp=True,
            ))
        new_groups.append(g)
    # 5) 驱动 (原样): HAL + CMSIS
    for gname in ("Drivers/STM32H7xx_HAL_Driver", "Drivers/CMSIS"):
        g = old.get(gname)
        if g is not None:
            new_groups.append(copy.deepcopy(g))
    # ⚠ 不加 Drivers/CMSIS-NN: FP32 TFLM 走 reference_ops, 不需要 int8 kernel.
    #   (少 3 个 .c, 少一堆 arm_nn 头依赖, 编译更快更干净)

    # 6) RTE / LayerInfo 原样
    for tag in ("RTE", "LayerInfo"):
        el = root.find(tag)
        if el is not None:
            new_root.append(copy.deepcopy(el))

    # ---------- 落盘 ----------
    ET.indent(new_root, space="  ")
    xml = ET.tostring(new_root, encoding="unicode")
    xml = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + xml + "\n"
    # ET 会把 xsi 命名空间写成 ns0, 换回来 (uVision 不在意, 但保持文件可读/可比对)
    xml = xml.replace(
        'xmlns:ns0="http://www.w3.org/2001/XMLSchema-instance"',
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
    ).replace("ns0:noNamespaceSchemaLocation", "xsi:noNamespaceSchemaLocation")

    with open(OUT_PROJ, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(xml)

    return new_root


def verify():
    """落盘后回读校验: 文件清单 / 关键设置."""
    t = ET.parse(OUT_PROJ)
    r = t.getroot()
    tg = r.find("Targets").findall("Target")
    assert len(tg) == 1, f"expected 1 target, got {len(tg)}"
    assert tg[0].findtext("TargetName") == TARGET_NAME

    files = []
    for g in tg[0].find("Groups").findall("Group"):
        fs = g.find("Files")
        if fs is None:
            continue
        for f in fs.findall("File"):
            files.append((g.findtext("GroupName"), f.findtext("FileName"),
                          f.findtext("FilePath"), f.findtext("FileType")))

    # 硬断言 1: 不能同时出现两个手写后端
    names = {f[1] for f in files}
    assert "ch32_inference.c" not in names, "Route E backend must NOT be in TFLM project"
    assert "ch32_inference_int8.c" not in names, "INT8 backend must NOT be in TFLM project"
    assert "ch32_inference_tflm.c" in names, "TFLM backend missing"
    # 硬断言 2: 所有引用的文件真的存在
    missing = []
    for _, name, path, _ in files:
        p = os.path.normpath(os.path.join(MDK, path))
        if not os.path.isfile(p):
            missing.append(path)
    if missing:
        raise SystemExit("[FATAL] referenced files missing:\n  " + "\n  ".join(missing))
    # 硬断言 3: 每个 .cc 都是 FileType 1 且 per-file 带 `-x c++`
    #            (FileType 5 会被本机 uVision 静默跳过 -> 文件根本不进 binary)
    for gname, name, path, ftype in files:
        if path.endswith(".cc") or path.endswith(".cpp"):
            assert ftype == "1", f"{path} must be FileType 1 (FileType 5 is skipped by uVision)"
            el = None
            for g in tg[0].find("Groups").findall("Group"):
                if g.findtext("GroupName") != gname:
                    continue
                for f in g.find("Files").findall("File"):
                    if f.findtext("FilePath") == path:
                        el = f
                        break
            assert el is not None and el.find("FileOption") is not None, \
                f"{path} missing FileOption"
            misc = (el.find("FileOption").find("FileArmAds").find("Cads")
                       .find("VariousControls").findtext("MiscControls")) or ""
            assert "-x c++" in misc, f"{path} must carry '-x c++' in per-file MiscControls"
            assert "-std=c++17" in misc, f"{path} must carry '-std=c++17'"
    # 硬断言 4: 既有工程没被动过 (由调用方用 git diff 验证)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    if args.check_only:
        files = verify()
    else:
        build()
        files = verify()

    n_cpp = sum(1 for f in files if f[2].endswith(".cc") or f[2].endswith(".cpp"))
    n_c = sum(1 for f in files if f[2].endswith(".c"))
    print(f"[OK] {os.path.relpath(OUT_PROJ, ROOT)}")
    print(f"     target   : {TARGET_NAME}  (scatter {SCATTER})")
    print(f"     files    : {len(files)} total  ({n_c} .c, {n_cpp} .cc)")
    print(f"     proj inc : {PROJ_INC}")
    print(f"     proj misc: {PROJ_MISC}")
    print("     per-file : -std=c++17 -fno-exceptions -fno-rtti (only .cc)")


if __name__ == "__main__":
    main()
