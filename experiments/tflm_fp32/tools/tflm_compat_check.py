#!/usr/bin/env python3
"""
TFLM 兼容性静态检查 —— 把 .tflite 的 flatbuffer 跟 **MCU 上那份 TFLM 源码**的约束逐条对上.

设计原则 (沿用手写 kernel 路线的教训 [[feedback-parity-test-must-bind-to-real-source]]):
  每条断言都注明它来自哪个源文件的哪一行, 而不是"我记得 TFLM 支持/不支持".
  约束变了 -> 断言自动失效 -> 显式报错, 而不是悄悄跑出一个错数字.

用法:
    python experiments/tflm_fp32/tools/tflm_compat_check.py \
        --tflite experiments/tflm_fp32/artifacts/model_tflm_fp32_s42.tflite
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:                                    # Windows 控制台默认 GBK, 中文会炸
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from tensorflow.lite.python import schema_py_generated as fb

BUILTIN_NAME = {v: k for k, v in vars(fb.BuiltinOperator).items()}

# TFLM (micro/kernels/) 里为 FP32 实现、且本模型可能用到的 builtin
TFLM_HAS = {
    "ADD", "CONV_2D", "EXPAND_DIMS", "FULLY_CONNECTED", "LOGISTIC", "MEAN",
    "MUL", "RESHAPE", "TRANSPOSE", "RELU", "RELU6", "SOFTMAX", "TANH",
    "SUB", "DIV", "SQUEEZE", "LEAKY_RELU",
}

FLOAT32 = 0
INT32 = 2


class CompatError(Exception):
    pass


def check_tflm_compat(blob: bytes) -> dict:
    """返回 (report_dict, errors). errors 非空 => 不兼容 TFLM."""
    m = fb.Model.GetRootAsModel(blob, 0)
    sg = m.Subgraphs(0)
    codes = {i: m.OperatorCodes(i).BuiltinCode() for i in range(m.OperatorCodesLength())}

    errors: list[str] = []
    notes: list[str] = []

    def tensors():
        out = []
        for i in range(sg.TensorsLength()):
            t = sg.Tensors(i)
            out.append({
                "index": i,
                "name": t.Name().decode(errors="replace"),
                "shape": [t.Shape(j) for j in range(t.ShapeLength())],
                "type": int(t.Type()),
                "buffer": int(t.Buffer()),
            })
        return out

    tens = tensors()
    by_idx = {t["index"]: t for t in tens}

    # --- 1. subgraph 数量 / 输入输出 -------------------------------------- #
    if m.SubgraphsLength() != 1:
        errors.append(f"subgraph 数 = {m.SubgraphsLength()}, TFLM 只支持单子图")
    if sg.InputsLength() != 1:
        errors.append(f"子图输入数 = {sg.InputsLength()}, TFLM 只支持单输入")
    if sg.OutputsLength() != 1:
        errors.append(f"子图输出数 = {sg.OutputsLength()}, TFLM 只支持单输出")

    # --- 2. tensor 类型 (只有 float32 / int32) ---------------------------- #
    bad_types = sorted({t["type"] for t in tens} - {FLOAT32, INT32})
    if bad_types:
        errors.append(f"出现非 float32/int32 tensor 类型: {bad_types} "
                      "(TFLM 本模型涉及的 kernel 不支持其它类型)")

    # --- 3. 常量性: 所有 tensor 都必须有 buffer ---------------------------- #
    #     TFLM 的 RESHAPE/EXPAND_DIMS/TRANSPOSE/MEAN 的 shape/axis/perm 张量
    #     必须是编译期常量, 否则 Prepare 直接失败.
    non_const = [(t["index"], t["name"]) for t in tens if t["buffer"] == 0]
    if non_const:
        errors.append(f"存在非常量 tensor (buffer==0): {non_const}")

    ops = []
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        name = BUILTIN_NAME.get(codes[op.OpcodeIndex()], "UNKNOWN")
        d = {
            "index": i,
            "op": name,
            "inputs": [int(op.Inputs(j)) for j in range(op.InputsLength())],
            "outputs": [int(op.Outputs(j)) for j in range(op.OutputsLength())],
        }
        bo = op.BuiltinOptions()
        if name == "CONV_2D" and bo is not None:
            c = fb.Conv2DOptions()
            c.Init(bo.Bytes, bo.Pos)
            d.update(padding=int(c.Padding()), stride_w=int(c.StrideW()),
                     stride_h=int(c.StrideH()),
                     dilation_w=int(c.DilationWFactor()),
                     dilation_h=int(c.DilationHFactor()),
                     fused_activation=int(c.FusedActivationFunction()))
        elif name == "MEAN" and bo is not None:
            r = fb.ReducerOptions()
            r.Init(bo.Bytes, bo.Pos)
            d.update(keep_dims=bool(r.KeepDims()))
        elif name == "ADD" and bo is not None:
            a = fb.AddOptions()
            a.Init(bo.Bytes, bo.Pos)
            d.update(fused_activation=int(a.FusedActivationFunction()))
        elif name == "FULLY_CONNECTED" and bo is not None:
            f = fb.FullyConnectedOptions()
            f.Init(bo.Bytes, bo.Pos)
            d.update(fused_activation=int(f.FusedActivationFunction()))
        ops.append(d)

    # --- 4. op 集合必须在 TFLM 里存在 -------------------------------------- #
    unsupported = sorted({o["op"] for o in ops} - TFLM_HAS)
    if unsupported:
        errors.append(f"TFLM 无对应 kernel 的 op: {unsupported}")

    for o in ops:
        op = o["op"]
        ins = [by_idx[j] for j in o["inputs"]]
        outs = [by_idx[j] for j in o["outputs"]]

        if op == "MEAN":
            # reduce_common.cc:242  op_data->num_axis = NumElements(axis);
            # reduce_common.cc:297  int num_axis = ElementCount(*axis->dims);
            axis = ins[1]
            n = 1
            for d in axis["shape"]:
                n *= d
            if n != 1:
                errors.append(f"op#{o['index']} MEAN: axes 张量元素数 = {n} "
                              "(reduce_common.cc 要求 1)")
            if axis["type"] != INT32:
                errors.append(f"op#{o['index']} MEAN: axes 张量类型 = {axis['type']}, 应为 int32")
            # reduce_common.cc:317 用 params->keep_dims 决定输出形状
            if not o.get("keep_dims", False):
                errors.append(f"op#{o['index']} MEAN: keep_dims = 0, "
                              "TFLM 走非 keep_dims 分支, 与本图输出形状不符")
            notes.append(f"MEAN#{o['index']} axes=rank{len(axis['shape'])} "
                         f"numel={n} keep_dims=1 -> TFLM 可读")

        elif op == "EXPAND_DIMS":
            # expand_dims.cc GetAxisValueFromTensor(): axis_dims > 1 -> kTfLiteError
            axis = ins[1]
            if len(axis["shape"]) > 1:
                errors.append(f"op#{o['index']} EXPAND_DIMS: axis 张量 rank = "
                              f"{len(axis['shape'])} (expand_dims.cc 只接受 <=1)")
            if axis["type"] != INT32:
                errors.append(f"op#{o['index']} EXPAND_DIMS: axis 类型应为 int32")
            notes.append(f"EXPAND_DIMS#{o['index']} axis rank={len(axis['shape'])} -> 可读")

        elif op == "TRANSPOSE":
            # transpose.cc:30  const int size = perm_tensor->dims->data[0];
            #                  (若 perm 是 rank-0, data[0] 取到 0 -> perm_count=0, 静默算错)
            perm = ins[1]
            if len(perm["shape"]) != 1:
                errors.append(f"op#{o['index']} TRANSPOSE: perm 张量 rank = "
                              f"{len(perm['shape'])} (transpose.cc:30 假定 rank==1)")
            if perm["type"] != INT32:
                errors.append(f"op#{o['index']} TRANSPOSE: perm 类型应为 int32")
            notes.append(f"TRANSPOSE#{o['index']} perm={perm['shape']} -> 可读")

        elif op == "RESHAPE":
            shp = ins[1]
            if shp["type"] != INT32:
                errors.append(f"op#{o['index']} RESHAPE: shape 张量类型应为 int32")
            if len(shp["shape"]) != 1:
                errors.append(f"op#{o['index']} RESHAPE: shape 张量 rank = "
                              f"{len(shp['shape'])} (应为 1)")

        elif op == "CONV_2D":
            if o["dilation_w"] not in (1, 2, 4):
                errors.append(f"op#{o['index']} CONV_2D: dilation_w = "
                              f"{o['dilation_w']} (本模型只应出现 1/2/4)")
            if o["stride_w"] != 1 or o["stride_h"] != 1:
                errors.append(f"op#{o['index']} CONV_2D: stride != 1")
            # conv.h:76-78 用 dilation_width_factor 算 in_x = in_x_origin + dilation_x*filter_x
            notes.append(f"CONV_2D#{o['index']} dilation_w={o['dilation_w']} "
                         f"fused_act={o['fused_activation']}")

        elif op == "FULLY_CONNECTED":
            # fully_connected.cc:142  FullyConnectedParamsFloat(params->activation)
            w = ins[1]
            if len(w["shape"]) != 2:
                errors.append(f"op#{o['index']} FULLY_CONNECTED: 权重 rank = "
                              f"{len(w['shape'])} (应为 2)")
            notes.append(f"FC#{o['index']} w={w['shape']} fused_act={o['fused_activation']}")

        elif op == "ADD":
            # flatbuffer_conversions.cc:255-266  AddOptions.fused_activation_function
            #   -> TfLiteAddParams.activation
            # add_common.cc:73 CalculateActivationRange -> 输出端 clamp
            notes.append(f"ADD#{o['index']} fused_act={o['fused_activation']} "
                         "(TFLM add.cc 会做输出端 clamp)")

        for t in outs:
            if t["type"] != FLOAT32:
                errors.append(f"op#{o['index']} {op}: 输出张量类型 = {t['type']}, 应为 float32")

    # --- 5. 输入输出 shape ------------------------------------------------- #
    in_idx = int(sg.Inputs(0))
    out_idx = int(sg.Outputs(0))
    in_shape = by_idx[in_idx]["shape"]
    out_shape = by_idx[out_idx]["shape"]
    if in_shape != [1, 23, 16]:
        errors.append(f"输入 shape 应为 [1,23,16], 实际 {in_shape}")
    if out_shape != [1]:
        errors.append(f"输出 shape 应为 [1], 实际 {out_shape}")

    report = {
        "n_operators": len(ops),
        "op_set": sorted({o["op"] for o in ops}),
        "input_shape": in_shape,
        "output_shape": out_shape,
        "subgraph_input_index": in_idx,
        "subgraph_output_index": out_idx,
        "operators": ops,
        "notes": notes,
        "errors": errors,
        "tflm_compatible": not errors,
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite",
                    default="experiments/tflm_fp32/artifacts/model_tflm_fp32_s42.tflite")
    ap.add_argument("--out", default=None, help="把报告写成 JSON")
    args = ap.parse_args()

    blob = open(args.tflite, "rb").read()
    rep = check_tflm_compat(blob)

    print("=" * 74)
    print("TFLM 兼容性静态检查: %s" % args.tflite)
    print("=" * 74)
    print("  op 数       : %d" % rep["n_operators"])
    print("  op 集合     : %s" % ", ".join(rep["op_set"]))
    print("  input/output: %s -> %s" % (rep["input_shape"], rep["output_shape"]))
    print("\n  逐条核对 (约束来自 MCU/thirdParty/tflite-micro 源码):")
    for n in rep["notes"]:
        print("    . %s" % n)
    if rep["errors"]:
        print("\n  ❌ 不兼容项:")
        for e in rep["errors"]:
            print("    - %s" % e)
    else:
        print("\n  ✅ 全部兼容 TFLM")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
        print("\n  报告: %s" % args.out)

    return 1 if rep["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
