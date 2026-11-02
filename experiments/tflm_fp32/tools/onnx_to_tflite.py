#!/usr/bin/env python3
"""
ONNX -> TF -> TFLite exporter for the TFLM FP32 comparison experiment (from scratch).

设计原则 (直接来自手写 kernel 路线的教训):
  * 唯一真源 = model_v4_se_23dim_ch32_s42.onnx. 逐 node 翻译, 不手抄公式;
    翻译结果由 ONNX Runtime 对拍兜底 (parity gate).
  * 空洞卷积用 tf.nn.conv1d(dilations=) -> TFLite CONV_2D 带 dilation_w_factor,
    **不产生 SPACE_TO_BATCH_ND / BATCH_TO_SPACE_ND** (2026-10 旧 TFLM 失败路线的
    根因之一: 那批算子 TFLM reference impl 跑不对).
  * 图输入 = (N, 23, 16) channel-major, 与 MCU sw_buf[c*16+t] 逐字节一致;
    transpose 放在图内 -> MCU 端零转置 (历史 (C,T) vs (T,C) 陷阱).
  * conv 对齐语义固定为 t_in = t + k*d - pad (与修好的 Route E kernel 一致).

用法:
    python experiments/tflm_fp32/tools/onnx_to_tflite.py \
        --onnx model_v4_se_23dim_ch32_s42.onnx \
        --out  experiments/tflm_fp32/artifacts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import onnx
import tensorflow as tf
from onnx import numpy_helper
from tensorflow.lite.python import schema_py_generated as fb

# ONNX 的 3D 张量语义是 (N, C, T); TF/TFLite 的 conv 布局是 (N, T, C).
# 逐 node 翻译时按下表映射 axis.
AXIS_PERM_R3 = (0, 2, 1)  # onnx_axis -> tf_axis, 仅对 rank-3 张量成立

# 允许出现的 TFLite builtin op (TFLM 必须有对应 kernel).
ALLOWED_OPS = {
    "CONV_2D", "FULLY_CONNECTED", "MEAN", "MUL", "ADD", "LOGISTIC", "RELU",
    "RELU6", "RESHAPE", "TRANSPOSE", "EXPAND_DIMS", "SQUEEZE",
    "SOFTMAX", "TANH", "LEAKY_RELU", "SUB", "DIV",
}
# 明确禁止出现的算子 (旧 TFLM 失败弧线里出问题的那些).
FORBIDDEN_OPS = {
    "PACK", "SHAPE", "STRIDED_SLICE", "SPACE_TO_BATCH_ND", "BATCH_TO_SPACE_ND",
    "GATHER", "SLICE", "SELECT_V2", "NON_MAX_SUPPRESSION_V4",
}

BUILTIN_NAME = {v: k for k, v in vars(fb.BuiltinOperator).items()}


def zero_interleave(w: np.ndarray, dilation: int) -> np.ndarray:
    """(Cout, Cin, k) -> (Cout, Cin, (k-1)*d+1), Wd[..., j*d] = W[..., j]."""
    cout, cin, k = w.shape
    k_new = (k - 1) * dilation + 1
    wd = np.zeros((cout, cin, k_new), np.float32)
    for j in range(k):
        wd[:, :, j * dilation] = w[:, :, j]
    return wd


class OnnxToTf:
    """把 ONNX 图逐 node 翻译成等价 TF 图 (TF 布局 (N, T, C)).

    env[name] = (tf_tensor, onnx_rank)
    consts[name] = np.ndarray (axes 用到的 Constant)
    """

    def __init__(self, model: onnx.ModelProto, mode: str = "native"):
        self.model = model
        self.mode = mode
        self.init = {i.name: numpy_helper.to_array(i).astype(np.float32)
                     for i in model.graph.initializer}
        self.env: dict[str, tuple[tf.Tensor, int]] = {}
        self.consts: dict[str, np.ndarray] = {}
        self.trace: list[str] = []

    # -- helpers ---------------------------------------------------------- #
    @staticmethod
    def _tf_axis(onnx_rank: int, onnx_axis: int) -> int:
        a = onnx_axis if onnx_axis >= 0 else onnx_axis + onnx_rank
        if onnx_rank == 3:
            return AXIS_PERM_R3[a]
        if onnx_rank == 2:
            return a
        raise ValueError(f"unsupported rank {onnx_rank} axis {onnx_axis}")

    def _rank(self, name: str) -> int:
        return self.env[name][1]

    def _get(self, name: str) -> tuple[tf.Tensor, int]:
        if name in self.env:
            return self.env[name]
        if name in self.init:
            arr = self.init[name]
            self.env[name] = (tf.constant(arr), int(arr.ndim))
            return self.env[name]
        raise KeyError(f"tensor '{name}' not produced by any node")

    # -- translation ------------------------------------------------------ #
    def translate(self, inp, input_name: str):
        g = self.model.graph
        # ONNX (N,C,T) -> TF (N,T,C); transpose 留在图内 (MCU 端零转置).
        x = tf.transpose(inp, perm=[0, 2, 1])
        self.env[input_name] = (x, 3)
        self.trace.append("INPUT: transpose (N,C,T) -> (N,T,C)")

        for node in g.node:
            op, ins, out_name = node.op_type, list(node.input), node.output[0]
            self.trace.append(f"{op}: {node.name}")

            if op == "Constant":
                arr = numpy_helper.to_array(node.attribute[0].t)
                self.consts[out_name] = arr
                self.env[out_name] = (tf.constant(arr), int(arr.ndim))
                continue

            if op == "Conv":
                y, r = self._conv(ins, node)
            elif op == "Relu":
                y, r = tf.nn.relu(self._get(ins[0])[0]), self._rank(ins[0])
            elif op == "Sigmoid":
                y, r = tf.sigmoid(self._get(ins[0])[0]), self._rank(ins[0])
            elif op == "GlobalAveragePool":
                y, r = self._gap(ins)
            elif op == "Squeeze":
                y, r = self._squeeze(ins)
            elif op == "Unsqueeze":
                y, r = self._unsqueeze(ins)
            elif op == "Gemm":
                y, r = self._gemm(ins, node)
            elif op == "Mul":
                a, _ = self._get(ins[0])
                b, _ = self._get(ins[1])
                y, r = tf.multiply(a, b), self._rank(ins[0])
            elif op == "Add":
                a, _ = self._get(ins[0])
                b, _ = self._get(ins[1])
                y, r = tf.add(a, b), self._rank(ins[0])
            else:
                raise NotImplementedError(f"ONNX op '{op}' ({node.name}) not translated")

            self.env[out_name] = (y, r)

        return self.env[g.output[0].name][0]

    def _conv(self, ins, node):
        x, _ = self._get(ins[0])
        w = self.init[ins[1]]
        b = self.init[ins[2]]
        dilation = 1
        for a in node.attribute:
            if a.name == "dilations":
                dilation = int(onnx.helper.get_attribute_value(a)[0])

        if self.mode == "expand" and dilation > 1:
            # 空洞核零插值成 (k-1)*d+1 的普通核 -> 不依赖 dilation 支持.
            w = zero_interleave(w, dilation)
            dilation = 1
            self.trace[-1] += f" [expand d={dilation}]"
        else:
            self.trace[-1] += f" [native d={dilation}]"

        # ONNX W: (Cout, Cin, k) -> TF conv1d filter: (k, Cin, Cout)
        filt = np.transpose(w, (2, 1, 0)).astype(np.float32)
        y = tf.nn.conv1d(x, tf.constant(filt), stride=1, padding="SAME",
                         data_format="NWC", dilations=dilation)
        return tf.nn.bias_add(y, tf.constant(b.astype(np.float32))), 3

    def _gap(self, ins):
        x, r = self._get(ins[0])
        assert r == 3, f"GlobalAveragePool expects rank-3, got {r}"
        # ONNX 在最后一维 (T) 上池化 -> TF 布局里 T 是 axis 1.
        return tf.reduce_mean(x, axis=1, keepdims=True), 3

    def _squeeze(self, ins):
        x, r = self._get(ins[0])
        axes = np.asarray(self.consts[ins[1]]).reshape(-1)
        tf_axes = [self._tf_axis(r, int(a)) for a in axes]
        return tf.squeeze(x, axis=sorted(tf_axes)), r - len(axes)

    def _unsqueeze(self, ins):
        x, r = self._get(ins[0])
        axes = np.asarray(self.consts[ins[1]]).reshape(-1)
        res_rank = r + len(axes)
        if res_rank != 3:
            raise ValueError("only rank-3 unsqueeze supported")
        inserted = {int(a) if a >= 0 else int(a) + res_rank for a in axes}
        src_onnx_axes = [j for j in range(res_rank) if j not in inserted]

        # 1) 先把 size-1 轴插到对应的 TF 位置
        ins_tf = sorted(self._tf_axis(res_rank, j) for j in inserted)
        x2 = x
        for pos in sorted(ins_tf, reverse=True):
            x2 = tf.expand_dims(x2, axis=pos)
        # 2) 计算最终 perm: 输出 TF axis k 存 ONNX axis j (AXIS_PERM_R3[j] == k)
        non_ins_tf = [a for a in range(res_rank) if a not in ins_tf]
        src_tf_of_onnx = {j_src: non_ins_tf[p]
                          for p, j_src in enumerate(src_onnx_axes)}
        perm = []
        for k in range(res_rank):
            j = AXIS_PERM_R3.index(k)
            perm.append(ins_tf[0] if j in inserted else src_tf_of_onnx[j])
        return tf.transpose(x2, perm=perm), res_rank

    def _gemm(self, ins, node):
        x, r = self._get(ins[0])
        assert r == 2, f"Gemm expects rank-2, got {r}"
        w = self.init[ins[1]]
        b = self.init[ins[2]]
        transB = 1
        for a in node.attribute:
            if a.name == "transB":
                transB = int(onnx.helper.get_attribute_value(a))
        # ONNX Gemm: A(M,K) @ B, transB=1 时 B 以 (N,K) 存储, 需转置成 (K,N).
        wt = w.T if transB else w
        y = tf.matmul(x, tf.constant(wt.astype(np.float32)))
        return tf.add(y, tf.constant(b.astype(np.float32))), 2


# --------------------------------------------------------------------------- #
def inspect_tflite(blob: bytes) -> dict:
    m = fb.Model.GetRootAsModel(blob, 0)
    sg = m.Subgraphs(0)
    codes = {i: m.OperatorCodes(i).BuiltinCode() for i in range(m.OperatorCodesLength())}
    tensors = []
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        tensors.append({
            "index": i,
            "name": t.Name().decode(errors="replace"),
            "shape": [t.Shape(j) for j in range(t.ShapeLength())],
            "type": int(t.Type()),
            "buffer": int(t.Buffer()),
        })
    ops = []
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        name = BUILTIN_NAME.get(codes[op.OpcodeIndex()], "UNKNOWN")
        d = {
            "op": name,
            "inputs": [int(op.Inputs(j)) for j in range(op.InputsLength())],
            "outputs": [int(op.Outputs(j)) for j in range(op.OutputsLength())],
        }
        if name == "CONV_2D":
            c = fb.Conv2DOptions()
            c.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            d.update(padding=int(c.Padding()), stride_w=int(c.StrideW()),
                     stride_h=int(c.StrideH()),
                     dilation_w=int(c.DilationWFactor()),
                     dilation_h=int(c.DilationHFactor()),
                     fused_activation=int(c.FusedActivationFunction()))
        ops.append(d)
    in_idx = int(sg.Inputs(0))
    out_idx = int(sg.Outputs(0))
    in_t, out_t = sg.Tensors(in_idx), sg.Tensors(out_idx)
    return {"operators": ops, "tensors": tensors,
            "input_shape": [in_t.Shape(j) for j in range(in_t.ShapeLength())],
            "output_shape": [out_t.Shape(j) for j in range(out_t.ShapeLength())],
            "subgraph_input_index": in_idx,
            "subgraph_output_index": out_idx}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="model_v4_se_23dim_ch32_s42.onnx")
    ap.add_argument("--out", default="experiments/tflm_fp32/artifacts")
    ap.add_argument("--mode", choices=["native", "expand"], default="native",
                    help="native: 保留 dilation (依赖 TFLM 支持); expand: 零插值成 2d+1 核")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = onnx.load(args.onnx)

    tr = OnnxToTf(model, mode=args.mode)

    @tf.function(input_signature=[tf.TensorSpec([None, 23, 16], tf.float32, name="input")])
    def net(inp):
        return tr.translate(inp, model.graph.input[0].name)

    cf = net.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([cf])
    converter.optimizations = []
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    blob = converter.convert()

    out_tflite = os.path.join(args.out, "model_tflm_fp32_s42.tflite")
    with open(out_tflite, "wb") as f:
        f.write(blob)

    info = inspect_tflite(blob)
    op_names = [o["op"] for o in info["operators"]]
    conv_dils = sorted({(o["dilation_w"], o["dilation_h"])
                         for o in info["operators"] if o["op"] == "CONV_2D"})

    manifest = {
        "source_onnx": os.path.abspath(args.onnx),
        "source_onnx_sha256": hashlib.sha256(open(args.onnx, "rb").read()).hexdigest(),
        "tflite": os.path.abspath(out_tflite),
        "tflite_sha256": hashlib.sha256(blob).hexdigest(),
        "tflite_bytes": len(blob),
        "mode": args.mode,
        "op_sequence": op_names,
        "op_set": sorted(set(op_names)),
        "conv_dilation_pairs": conv_dils,
        "input_shape": info["input_shape"],
        "output_shape": info["output_shape"],
        "subgraph_input_index": info["subgraph_input_index"],
        "subgraph_output_index": info["subgraph_output_index"],
        "operators": info["operators"],
        "tensors": info["tensors"],
        "translate_trace": tr.trace,
    }
    with open(os.path.join(args.out, "model_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ---------------- 硬断言 ---------------- #
    errors = []
    bad = set(op_names) & FORBIDDEN_OPS
    if bad:
        errors.append(f"forbidden ops present: {sorted(bad)}")
    unknown = set(op_names) - ALLOWED_OPS
    if unknown:
        errors.append(f"ops outside allowed set: {sorted(unknown)}")
    if list(info["input_shape"][1:]) != [23, 16]:
        errors.append(f"input shape must be [N,23,16], got {info['input_shape']}")
    if len(info["output_shape"]) != 1:
        errors.append(f"output shape must be [N], got {info['output_shape']}")
    if args.mode == "native":
        if (2, 1) not in conv_dils or (4, 1) not in conv_dils:
            errors.append(
                f"expected dilated CONV_2D with dilation_w in {{2,4}}, got {conv_dils}; "
                "converter may have dropped dilation -> rerun with --mode expand")
    else:
        if any(d != (1, 1) for d in conv_dils):
            errors.append(f"expand mode must have no dilation, got {conv_dils}")

    print(json.dumps({
        "tflite": out_tflite,
        "bytes": len(blob),
        "op_set": manifest["op_set"],
        "op_sequence": op_names,
        "conv_dilation_pairs": conv_dils,
        "input_shape": info["input_shape"],
        "output_shape": info["output_shape"],
    }, indent=2, ensure_ascii=False))

    if errors:
        print("\nASSERTION FAILURES:")
        for e in errors:
            print("  -", e)
        return 1
    print("\nAll hard assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
