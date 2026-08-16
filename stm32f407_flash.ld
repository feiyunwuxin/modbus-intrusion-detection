/* ============================================================================
 * stm32f407_flash.ld  —  STM32F407VGT6 链接脚本 (TCN+SE ch=32 优化版)
 * ============================================================================
 *
 * 目标: STM32F407VGT6
 *   - Flash:  1 MB   @ 0x08000000
 *   - SRAM:   128 KB (SRAM1 112 KB + SRAM2 16 KB,地址连续 0x20000000)
 *   - CCMRAM: 64 KB  @ 0x10000000 (CPU 直接访问,不走 AHB 总线,极快!)
 *
 * 优化策略:
 *   1. 权重段 (.weights) 放 Flash:23.3 KB 权重常驻,运行时无需 dequant 到 RAM
 *   2. .ccmram 段放 CCMRAM:激活缓冲 + dequant 后权重 (CPU 直访,无需等总线)
 *   3. 标准 .data/.bss 走 SRAM1
 *
 * 关键常量 (务必检查):
 *   - FLASH_SIZE    = 1024K
 *   - RAM_SIZE      = 128K  (SRAM1+SRAM2)
 *   - SRAM1_SIZE    = 112K
 *   - SRAM2_SIZE    = 16K   @ 0x2001C000
 *   - CCMRAM_SIZE   = 64K   @ 0x10000000
 *
 * 用法 (GCC):
 *   arm-none-eabi-gcc -T stm32f407_flash.ld ... -o ch32_detect.elf
 *
 * 注:本脚本基于 ST 官方 STM32F407VGTX_FLASH.ld 修改,加了 .weights 和 .ccmram 段
 * ============================================================================
 */

/* ── 内存映射 ─────────────────────────────────────────────────────── */
MEMORY
{
    FLASH (rx)       : ORIGIN = 0x08000000, LENGTH = 1024K
    SRAM1 (xrw)      : ORIGIN = 0x20000000, LENGTH = 112K
    SRAM2 (xrw)      : ORIGIN = 0x2001C000, LENGTH = 16K     /* @ end of SRAM1 */
    CCMRAM (xrw)     : ORIGIN = 0x10000000, LENGTH = 64K      /* CPU 直接访问 */
}

/* ── 最高堆栈地址 (供 _sbrk / startup 使用) ─────────────────────── */
__StackTop   = ORIGIN(SRAM1) + LENGTH(SRAM1);    /* 0x2001C000 = SRAM2 起点 */
__StackSize  = 0x1000;                             /* 4 KB 栈,够 v19_init + Modbus 解析 */
__HeapSize   = 0x2000;                             /* 8 KB 堆 (可选,本项目基本不用) */

/* ── 入口点 ───────────────────────────────────────────────────────── */
ENTRY(Reset_Handler)

/* ── 中断向量表在 Flash 起始 ─────────────────────────────────────── */
INCLUDE device/Include/cmsis_nvic_offset_stm32f4.s

/* ── 高水位标记 (用于打印内存占用) ───────────────────────────────── */
__sram1_start  = ORIGIN(SRAM1);
__sram1_end    = ORIGIN(SRAM1) + LENGTH(SRAM1);
__sram2_start  = ORIGIN(SRAM2);
__sram2_end    = ORIGIN(SRAM2) + LENGTH(SRAM2);
__ccmram_start = ORIGIN(CCMRAM);
__ccmram_end   = ORIGIN(CCMRAM) + LENGTH(CCMRAM);
__flash_start  = ORIGIN(FLASH);
__flash_end    = ORIGIN(FLASH) + LENGTH(FLASH);

/* ── Section 定义 ─────────────────────────────────────────────────── */
SECTIONS
{
    /* 1. 中断向量表 (必须最先,放 Flash 起始) */
    .isr_vector :
    {
        . = ALIGN(4);
        KEEP(*(.isr_vector))     /* startup_stm32f407xx.s 里定义 */
        . = ALIGN(4);
    } >FLASH

    /* 2. 代码段 */
    .text :
    {
        . = ALIGN(4);
        *(.text)
        *(.text*)
        *(.glue_7)
        *(.glue_7t)
        *(.eh_frame)

        KEEP(*(.init))
        KEEP(*(.fini))

        . = ALIGN(4);
        _etext = .;
    } >FLASH

    /* 3. 只读数据 (.rodata, 包括字符串常量等) */
    .rodata :
    {
        . = ALIGN(4);
        *(.rodata)
        *(.rodata*)
        . = ALIGN(4);
    } >FLASH

    /* 4. ⭐ 权重段 (来自 model_ch32_hybrid.h)
       编译时 -Wl,--section-start=.weights=0x080XXXXX 或用 >FLASH 默认放末尾
       总大小 ~23.3 KB,无需特殊位置 (Flash 1 MB 够用) */
    .weights :
    {
        . = ALIGN(4);
        /* 把所有 const float / const int8_t 数组放到 .weights 段
           (model_ch32_hybrid.h 里的 ch32_w_* / ch32_s_* / ch32_b_*) */
        KEEP(*(.weights))
        KEEP(*(.weights.*))
        . = ALIGN(4);
    } >FLASH

    /* 5. C++ 静态构造 */
    .preinit_array :
    {
        PROVIDE_HIDDEN(__preinit_array_start = .);
        KEEP(*(.preinit_array*))
        PROVIDE_HIDDEN(__preinit_array_end = .);
    } >FLASH

    .init_array :
    {
        PROVIDE_HIDDEN(__init_array_start = .);
        KEEP(*(SORT(.init_array.*)))
        KEEP(*(.init_array*))
        PROVIDE_HIDDEN(__init_array_end = .);
    } >FLASH

    .fini_array :
    {
        PROVIDE_HIDDEN(__fini_array_start = .);
        KEEP(*(SORT(.fini_array.*)))
        KEEP(*(.fini_array*))
        PROVIDE_HIDDEN(__fini_array_end = .);
    } >FLASH

    /* 6. .ARM.exidx / .ARM.extab */
    .ARM.extab :
    {
        *(.ARM.extab* .gnu.linkonce.armextab.*)
    } >FLASH

    .ARM.exidx :
    {
        __exidx_start = .;
        *(.ARM.exidx* .gnu.linkonce.armexidx.*)
        __exidx_end = .;
    } >FLASH

    /* 7. ⭐ CCMRAM 段 (激活缓冲 + dequant 后 FP32 权重)
       CCMRAM 64 KB,CPU 直访速度 = SRAM1 的 2 倍,且不占 AHB 总线
       适合放 hot path 数据 */
    .ccmram (NOLOAD) :
    {
        . = ALIGN(4);
        _sccmram = .;        /* 起始地址 (= 0x10000000) */
        /* ch32_inference.c 里的 w_b0_c1 等缓冲 (用 __attribute__((section(".ccmram"))) 标记) */
        *(.ccmram)
        *(.ccmram.*)
        . = ALIGN(4);
        _eccmram = .;        /* 结束地址 */
    } >CCMRAM

    /* 8. 标准 .data 段 (Flash → SRAM1 启动时复制) */
    _sidata = LOADADDR(.data);

    .data :
    {
        . = ALIGN(4);
        _sdata = .;
        *(.data)
        *(.data*)

        . = ALIGN(4);
        _edata = .;
    } >SRAM1 AT> FLASH

    /* 9. 标准 .bss 段 (零初始化,SRAM1) */
    .bss (NOLOAD) :
    {
        . = ALIGN(4);
        _sbss = .;
        __bss_start__ = _sbss;
        *(.bss)
        *(.bss*)
        *(COMMON)

        . = ALIGN(4);
        _ebss = .;
        __bss_end__ = _ebss;
    } >SRAM1

    /* 10. 堆 */
    .heap (NOLOAD) :
    {
        . = ALIGN(8);
        _sheap = .;
        . = . + __HeapSize;
        _eheap = .;
    } >SRAM1

    /* 11. 栈 (放在 SRAM1 末尾,即 SRAM2 起点之前)
       __StackTop 在 startup 代码里赋给 MSP */
    .stack_dummy (NOLOAD) :
    {
        . = ALIGN(8);
        . = . + __StackSize;
    } >SRAM1

    /* ── 内存使用断言 (编译时检查) ──────────────────────────────── */
    ASSERT(__StackTop >= _ebss + __HeapSize, "Region SRAM1 overflowed with stack")
}

/* ── 链接后符号 (供 startup 复制 .data 用) ─────────────────────── */
PROVIDE(_sidata = LOADADDR(.data));

/* ============================================================================
 * 内存布局总结 (ch=32 Hybrid INT8):
 *
 *   Flash (1 MB):
 *     0x08000000  ┌─ .isr_vector (~1 KB)
 *                 ├─ .text    (~6 KB 代码 + HAL)
 *                 ├─ .rodata  (~1 KB)
 *                 └─ .weights (~23.3 KB INT8 权重 + scale + bias)
 *                 ≈ 30 KB 总 Flash 占用
 *
 *   CCMRAM (64 KB):
 *     0x10000000  ├─ .ccmram  (~80 KB dequant 后 FP32 权重 + 6 KB 激活)
 *                 注: 实际只需 ~88 KB,但 CCMRAM 只有 64 KB
 *                     → 必须用 Lite 模式 (按层 dequant) 或把权重放 SRAM1
 *                 退路: 把 .ccmram 段改到 SRAM1 (0x20000000) 占 SRAM1 后部
 *
 *   SRAM1 (112 KB):
 *     0x20000000  ├─ .data  (HAL 句柄等,~4 KB)
 *                 ├─ .bss   (scada_window 等,~10 KB)
 *                 ├─ .heap  (8 KB)
 *                 └─ .stack (4 KB,栈)
 *
 *   SRAM2 (16 KB):
 *     0x2001C000  (暂未用,可作 DMA 缓冲)
 *
 * 总占用: Flash ~30 KB / SRAM1 ~26 KB / CCMRAM 80 KB (超 64KB,需要调整!)
 * ============================================================================
 */