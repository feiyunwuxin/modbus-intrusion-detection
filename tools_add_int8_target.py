"""
tools_add_int8_target.py -- 给 Keil 工程加一个 INT8 target (H743test_INT8).

⚠️ uVision 的文件列表是 **工程级** 的 (所有 target 共享一棵文件树),
   每个 target 只用自己那一份 <Groups> 做同步; 单靠 "第二个 target 里多几个文件"
   不够 —— 实测 uVision 会把两个 target 的文件取并集, 于是 FP32 target 也会去编
   CMSIS-NN 的文件 (而且还没有对应的 IncludePath -> 编译失败).

   所以正确做法是:
     1. 两个 target 都列出 **同一棵** 文件树
     2. INT8 专属的文件 (ch32_inference_int8.c + CMSIS-NN 三个 .c) 在 FP32 target 里
        标 <IncludeInBuild>0</IncludeInBuild> (右键 "Include in Build" 的机制)
     3. INT8 target 的 Define 加 CH32_USE_INT8=1, IncludePath 加 CMSIS-NN/DSP

用法: python tools_add_int8_target.py
"""
import io
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(HERE, "MCU/MDK-ARM/H743test.uvprojx")

INT8_FILES = [
    ("ch32_inference_int8.c", "../Core/Src/ch32_inference_int8.c"),
]

CMSISNN_FILES = [
    ("arm_convolve_s8.c",
     "../Drivers/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c"),
    ("arm_nn_mat_mult_kernel_s8_s16.c",
     "../Drivers/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c"),
    ("arm_q7_to_q15_with_offset.c",
     "../Drivers/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c"),
]

ALL_NEW = INT8_FILES + CMSISNN_FILES


def file_block(name, path, include_in_build=None):
    """生成一个 <File> 块. include_in_build=None -> 不写 FileOption (默认参与构建)."""
    s = ("            <File>\n"
         "              <FileName>%s</FileName>\n"
         "              <FileType>1</FileType>\n"
         "              <FilePath>%s</FilePath>\n" % (name, path))
    if include_in_build is None:
        s += "            </File>\n"
        return s
    # 带 FileOption (CommonProperty + FileArmAds), 抄 gpio.c 的模板
    s += ("              <FileOption>\n"
          "                <CommonProperty>\n"
          "                  <UseCPPCompiler>2</UseCPPCompiler>\n"
          "                  <RVCTCodeConst>0</RVCTCodeConst>\n"
          "                  <RVCTZI>0</RVCTZI>\n"
          "                  <RVCTOtherData>0</RVCTOtherData>\n"
          "                  <ModuleSelection>0</ModuleSelection>\n"
          "                  <IncludeInBuild>%d</IncludeInBuild>\n"
          "                  <AlwaysBuild>2</AlwaysBuild>\n"
          "                  <GenerateAssemblyFile>2</GenerateAssemblyFile>\n"
          "                  <AssembleAssemblyFile>2</AssembleAssemblyFile>\n"
          "                  <PublicsOnly>2</PublicsOnly>\n"
          "                  <StopOnExitCode>11</StopOnExitCode>\n"
          "                  <CustomArgument></CustomArgument>\n"
          "                  <IncludeLibraryModules></IncludeLibraryModules>\n"
          "                  <ComprImg>1</ComprImg>\n"
          "                </CommonProperty>\n"
          "                <FileArmAds>\n"
          "                  <Cads>\n"
          "                    <interw>2</interw>\n"
          "                    <Optim>0</Optim>\n"
          "                    <oTime>2</oTime>\n"
          "                    <SplitLS>2</SplitLS>\n"
          "                    <OneElfS>2</OneElfS>\n"
          "                    <Strict>2</Strict>\n"
          "                    <EnumInt>2</EnumInt>\n"
          "                    <PlainCh>2</PlainCh>\n"
          "                    <Ropi>2</Ropi>\n"
          "                    <Rwpi>2</Rwpi>\n"
          "                    <wLevel>0</wLevel>\n"
          "                    <uThumb>2</uThumb>\n"
          "                    <uSurpInc>2</uSurpInc>\n"
          "                    <uC99>2</uC99>\n"
          "                    <uGnu>2</uGnu>\n"
          "                    <useXO>2</useXO>\n"
          "                    <v6Lang>0</v6Lang>\n"
          "                    <v6LangP>8</v6LangP>\n"
          "                    <vShortEn>1</vShortEn>\n"
          "                    <vShortWch>1</vShortWch>\n"
          "                    <v6Lto>0</v6Lto>\n"
          "                    <v6WtE>0</v6WtE>\n"
          "                    <v6Rtti>0</v6Rtti>\n"
          "                    <VariousControls>\n"
          "                      <MiscControls></MiscControls>\n"
          "                      <Define></Define>\n"
          "                      <Undefine></Undefine>\n"
          "                      <IncludePath></IncludePath>\n"
          "                    </VariousControls>\n"
          "                  </Cads>\n"
          "                </FileArmAds>\n"
          "              </FileOption>\n"
          "            </File>\n" % (1 if include_in_build else 0))
    return s


def main():
    src = io.open(PROJ, encoding="utf-8", newline="").read()

    if "H743test_INT8" in src:
        print("[skip] 已经存在 H743test_INT8 target")
        return

    m = re.search(r"[ ]{4}<Target>\n(.*?)\n    </Target>", src, re.S)
    if not m:
        raise SystemExit("找不到 Target 块")
    block = m.group(0)

    t = block

    # 1. 名字 / 输出
    t = t.replace("<TargetName>H743test</TargetName>",
                  "<TargetName>H743test_INT8</TargetName>")
    t = t.replace("<OutputDirectory>H743test\\</OutputDirectory>",
                  "<OutputDirectory>H743test_INT8\\</OutputDirectory>")
    t = t.replace("<OutputName>H743test</OutputName>",
                  "<OutputName>H743test_INT8</OutputName>")

    # 2. Define
    old_define = ('<Define>USE_PWR_LDO_SUPPLY,USE_HAL_DRIVER,STM32H743xx,'
                  'DEBUG_DUMP_FEATS=1,ARMCM7,'
                  'CMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE="stm32h743xx.h"</Define>')
    assert old_define in t, "Define 行没找到 (工程被改过?)"
    t = t.replace(old_define, old_define.replace("</Define>",
                                                 ",CH32_USE_INT8=1</Define>"))

    # 3. IncludePath
    old_inc = ("<IncludePath>../Core/Inc;../Drivers/STM32H7xx_HAL_Driver/Inc;"
               "../Drivers/STM32H7xx_HAL_Driver/Inc/Legacy;"
               "../Drivers/CMSIS/Device/ST/STM32H7xx/Include;../Drivers/CMSIS/Include"
               "</IncludePath>")
    assert old_inc in t, "IncludePath 行没找到 (工程被改过?)"
    t = t.replace(old_inc, old_inc.replace(
        "</IncludePath>",
        ";../Drivers/CMSIS/NN/Include;../Drivers/CMSIS/DSP/Include</IncludePath>"))

    # 4a. Core 组: INT8 target 里加 ch32_inference_int8.c (参与构建)
    anchor = ("            <File>\n"
              "              <FileName>feature_extractor.c</FileName>\n"
              "              <FileType>1</FileType>\n"
              "              <FilePath>..\\Core\\Src\\feature_extractor.c</FilePath>\n"
              "            </File>\n")
    assert anchor in t, "feature_extractor.c 条目没找到"
    t = t.replace(anchor, anchor + file_block(*INT8_FILES[0]))

    # 4b. INT8 target 里新增 CMSIS-NN 组
    grp = ["        <Group>\n",
           "          <GroupName>Drivers/CMSIS-NN</GroupName>\n",
           "          <Files>\n"]
    for n, p in CMSISNN_FILES:
        grp.append(file_block(n, p))
    grp.append("          </Files>\n")
    grp.append("        </Group>\n")
    grp = "".join(grp)
    tail_grp = "        <Group>\n          <GroupName>::CMSIS</GroupName>\n        </Group>\n"
    assert tail_grp in t, "::CMSIS 组没找到"
    t = t.replace(tail_grp, grp + tail_grp)

    # 5. 原 FP32 target: 把 INT8 专属的文件也列进去, 但 IncludeInBuild=0
    #    (uVision 的文件树是工程级的, 不列的话它会拿并集 -> FP32 也去编)
    fp32 = block
    anchor_f = ("            <File>\n"
                "              <FileName>feature_extractor.c</FileName>\n"
                "              <FileType>1</FileType>\n"
                "              <FilePath>..\\Core\\Src\\feature_extractor.c</FilePath>\n"
                "            </File>\n")
    assert anchor_f in fp32
    fp32 = fp32.replace(anchor_f, anchor_f
                         + file_block(*INT8_FILES[0], include_in_build=False))
    grp_f = ["        <Group>\n",
             "          <GroupName>Drivers/CMSIS-NN</GroupName>\n",
             "          <Files>\n"]
    for n, p in CMSISNN_FILES:
        grp_f.append(file_block(n, p, include_in_build=False))
    grp_f.append("          </Files>\n")
    grp_f.append("        </Group>\n")
    grp_f = "".join(grp_f)
    assert tail_grp in fp32
    fp32 = fp32.replace(tail_grp, grp_f + tail_grp)

    # 6. 组装: FP32 target (改过的) + INT8 target
    new_src = src.replace(block, fp32 + "\n" + t, 1)

    # 7. RTE targetInfo
    new_src = new_src.replace(
        '<targetInfo name="H743test"/>',
        '<targetInfo name="H743test"/>\n'
        '        <targetInfo name="H743test_INT8"/>')

    shutil.copyfile(PROJ, PROJ + ".bak_int8")
    io.open(PROJ, "w", encoding="utf-8", newline="").write(new_src)
    print("[ok] 已写入 %s (备份 %s.bak_int8)" % (PROJ, os.path.basename(PROJ)))


if __name__ == "__main__":
    main()
