#!/usr/bin/env python3
"""
add_hybrid_fileopt.py -- 给 FP32 / INT8 target 补上 ch32_inference_hybrid.c 的
                        <IncludeInBuild>0</IncludeInBuild> 条目.

为什么
------
Keil uVision 的文件树是 **工程级** 的 (memory: project-2026-10-22-mcu-full-int8
"uVision 的文件列表是工程级的: 加 INT8 target 时必须让两个 target 列出同一棵文件树,
INT8 专属文件在 FP32 target 里标 <IncludeInBuild>0</IncludeInBuild>").

ch32_inference_hybrid.c 是在 HYBRID target 里新增的, FP32 / INT8 target 的
<Groups> 里没有它 —— 但 Keil 依然会把它编进这两个 target (空 TU, 608 B 的 .o,
链接时被 GC 掉, 对 Flash/RAM 影响 = 0). 本脚本按项目既有约定把它显式列出来
并标 IncludeInBuild=0, 让三个 target 的行为一致、构建日志干净.

做法: 文本级插入 (uvprojx 格式敏感, 不用 ElementTree 往返).
      模板 = FP32 target 里 ch32_inference_int8.c 的那个 <File> 块 (已带 IncludeInBuild=0).

用法: python add_hybrid_fileopt.py
"""
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(HERE, "MCU/MDK-ARM/H743test.uvprojx")

TARGETS = ["H743test", "H743test_INT8"]
NEW_FILE = "ch32_inference_hybrid.c"


def find_file_block(src, filename, start=0, end=None):
    """返回包含指定 FileName 的 <File>...</File> 块 (含起止 offset)."""
    end = len(src) if end is None else end
    for m in re.finditer(r"[ \t]*<File>.*?</File>\s*\n", src[start:end], re.S):
        blk = m.group(0)
        if "<FileName>%s</FileName>" % filename in blk:
            return start + m.start(), start + m.end(), blk
    return None, None, None


def main():
    src = open(PROJ, encoding="utf-8").read()

    # ---- 模板: FP32 target 里 int8 的 FileOption (已带 IncludeInBuild=0) ----
    t0 = src.index("<TargetName>%s</TargetName>" % TARGETS[0])
    t1 = src.index("<TargetName>%s</TargetName>" % TARGETS[1])
    _, _, int8_blk = find_file_block(src, "ch32_inference_int8.c", t0, t1)
    if int8_blk is None:
        print("[FAIL] 找不到 FP32 target 的 ch32_inference_int8.c 条目")
        return 1
    mo = re.search(r"[ \t]*<FileOption>.*?</FileOption>\s*\n", int8_blk, re.S)
    if mo is None:
        print("[FAIL] FP32 target 的 int8 条目里没有 <FileOption> (拿不到 IncludeInBuild=0 模板)")
        return 1
    fileopt = mo.group(0)

    out = src
    for tgt in TARGETS:
        a = out.index("<TargetName>%s</TargetName>" % tgt)
        # 该 target 的范围: 到下一个 TargetName 为止
        nxt = re.search(r"<TargetName>", out[a + 10:])
        b = a + 10 + nxt.start() if nxt else len(out)
        s, e, blk = find_file_block(out, NEW_FILE, a, b)
        if blk is not None:
            print("[skip] %s 里已有 %s 条目" % (tgt, NEW_FILE))
            continue
        s, e, blk = find_file_block(out, "ch32_inference_int8.c", a, b)
        if blk is None:
            print("[FAIL] %s 里找不到 ch32_inference_int8.c 条目" % tgt)
            return 1
        new = (blk.replace("ch32_inference_int8.c", NEW_FILE)
                  .replace("<IncludeInBuild>2</IncludeInBuild>",
                           "<IncludeInBuild>0</IncludeInBuild>"))
        if "<FileOption>" not in new:
            # int8 条目没有 FileOption (INT8 target 里它是参与编译的) -> 补一个
            new = new.replace("</File>\n", fileopt + "            </File>\n")
        else:
            new = re.sub(r"<IncludeInBuild>\d</IncludeInBuild>",
                         "<IncludeInBuild>0</IncludeInBuild>", new)
        out = out[:e] + new + out[e:]
        print("[OK] %s: 插入 %s (IncludeInBuild=0)" % (tgt, NEW_FILE))

    if out == src:
        print("[skip] 无改动")
        return 0

    shutil.copy2(PROJ, PROJ + ".bak2")
    with open(PROJ, "w", encoding="utf-8", newline="") as f:
        f.write(out)

    # ---- 自检 ----
    import xml.etree.ElementTree as ET
    t = ET.parse(PROJ)
    for tg in t.getroot().iter("Target"):
        name = tg.findtext("TargetName")
        info = []
        for f in tg.iter("File"):
            if f.findtext("FileName") == NEW_FILE:
                iib = f.findtext(".//IncludeInBuild")
                info.append(iib if iib is not None else "(默认=参与编译)")
        print("[自检] %-18s %s -> IncludeInBuild=%s" % (name, NEW_FILE, info))
    print("[备份] %s.bak2" % PROJ)
    return 0


if __name__ == "__main__":
    sys.exit(main())
