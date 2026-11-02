#!/usr/bin/env python3
"""
add_hybrid_target.py -- 在 MCU/MDK-ARM/H743test.uvprojx 里加第三个 target: H743test_HYBRID.

做法: 文本级克隆 H743test_INT8 的 <Target> 块 (Keil 的 uvprojx 格式敏感,
      用 ElementTree 往返会改缩进/属性顺序, 尽量不碰), 然后改 5 处:
        1. <TargetName>
        2. <Define>: CH32_USE_INT8=1  ->  CH32_USE_HYBRID=1
        3. <OutputDirectory>
        4. <OutputName>
        5. 文件列表里加 ../Core/Src/ch32_inference_hybrid.c

CMSIS-NN 那 3 个文件保留在文件列表里 (跟 FP32 target 一样): 链接器会 GC 掉
没被引用的代码 —— ch32_inference_int8.c 在 CH32_USE_INT8=0 时是空 TU,
所以 arm_convolve_s8 不会被链进去 (FP32 target 的 .map 里就查不到它).

用法: python add_hybrid_target.py
"""
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(HERE, "MCU/MDK-ARM/H743test.uvprojx")

NEW_NAME = "H743test_HYBRID"
NEW_FILE = "ch32_inference_hybrid.c"


def main():
    src = open(PROJ, encoding="utf-8").read()

    if "<TargetName>%s</TargetName>" % NEW_NAME in src:
        print("[skip] target %s 已存在" % NEW_NAME)
        return 0

    # ---- 定位 INT8 target 块 ----
    m = re.search(r"[ \t]*<Target>\s*\n\s*<TargetName>H743test_INT8</TargetName>", src)
    if not m:
        print("[FAIL] 找不到 H743test_INT8 target")
        return 1
    start = src.index("<Target>", m.start())
    end = src.index("</Target>", start) + len("</Target>")
    block = src[start:end]

    # ---- 改 5 处 ----
    nb = block.replace("<TargetName>H743test_INT8</TargetName>",
                       "<TargetName>%s</TargetName>" % NEW_NAME)
    nb = nb.replace(",CH32_USE_INT8=1", ",CH32_USE_HYBRID=1")
    nb = nb.replace("<OutputDirectory>H743test_INT8\\</OutputDirectory>",
                    "<OutputDirectory>%s\\</OutputDirectory>" % NEW_NAME)
    nb = nb.replace("<OutputName>H743test_INT8</OutputName>",
                    "<OutputName>%s</OutputName>" % NEW_NAME)

    # 文件列表: 在 ch32_inference_int8.c 条目后面插入 hybrid 条目
    old_entry = ("            <File>\n"
                 "              <FileName>ch32_inference_int8.c</FileName>\n"
                 "              <FileType>1</FileType>\n"
                 "              <FilePath>../Core/Src/ch32_inference_int8.c</FilePath>\n"
                 "            </File>\n")
    new_entry = old_entry + ("            <File>\n"
                             "              <FileName>%s</FileName>\n"
                             "              <FileType>1</FileType>\n"
                             "              <FilePath>../Core/Src/%s</FilePath>\n"
                             "            </File>\n" % (NEW_FILE, NEW_FILE))
    if old_entry not in nb:
        print("[FAIL] 找不到 ch32_inference_int8.c 的文件条目")
        return 1
    nb = nb.replace(old_entry, new_entry)

    # ---- 写回 (先备份) ----
    shutil.copy2(PROJ, PROJ + ".bak")
    out = src[:end] + "\n    " + nb + src[end:]
    with open(PROJ, "w", encoding="utf-8") as f:
        f.write(out)

    # ---- 自检 ----
    chk = open(PROJ, encoding="utf-8").read()
    names = re.findall(r"<TargetName>([^<]*)</TargetName>", chk)
    print("[OK] 现在有 %d 个 target: %s" % (len(names), names))
    assert NEW_NAME in names, "新 target 没写进去"
    hyb = chk[chk.index("<TargetName>%s</TargetName>" % NEW_NAME):]
    hyb = hyb[:hyb.index("</Target>")]
    assert "CH32_USE_HYBRID=1" in hyb, "Define 没改对"
    assert "CH32_USE_INT8=1" not in hyb, "Define 里还残留 CH32_USE_INT8"
    assert NEW_FILE in hyb, "新源文件没加进文件列表"
    print("[自检] %s: Define/输出名/文件列表 均正确 ✅" % NEW_NAME)
    print("[备份] %s.bak" % PROJ)
    return 0


if __name__ == "__main__":
    sys.exit(main())
