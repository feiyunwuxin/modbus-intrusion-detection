/* ============================================================================
 * tflm_defs.h  --  AC6 (armclang) + Keil libcxx 需要的宏 fallback
 *
 * 为什么需要这个文件 (来自 2026-10-15 那条 TFLM 路线的实战):
 *   Keil v5.43 自带的 libcxx (LLVM 20 era) 在 __config:18-42 用
 *   `#if defined(__ARMCOMPILER_LIBCXX)` 包住**整个 ARM 配置 block**.
 *   这个宏 AC6 **不会自动定义**, 不定义的话:
 *     __config:165  #error _LIBCPP_HARDENING_MODE_DEFAULT is not defined
 *     availability.h:282  #error vendor availability markup
 *   然后 _LIBCPP_BEGIN_NAMESPACE_STD 永远不被定义 -> 700+ errors cascade.
 *
 *   而且 .c 文件即使 include <stdint.h> 也会踩到 (只要 libcxx 在 include path 上),
 *   因为 ARM block 被 `#ifdef __cplusplus` 包着, .c 走不到.
 *
 * 怎么用:
 *   在 .uvprojx 的 <Cads><MiscControls> 里加:
 *       --include ../Core/Inc/tflm_defs.h
 *   **不要**把这些宏塞进 <Define> 框:
 *       - <Define> 会同时传给 armasm, `__2` / `__ORDER_LITTLE_ENDIAN__` 这种
 *         token 在 armasm 里是未定义符号 -> A1516E 语法错.
 *       - 所以下面全部用**数值** (1234/4321/3412) 而不是 token.
 *   <Aads> 的 Define / MiscControls 必须**清空**, <Undefine> 也必须清空
 *   (Undefine 里如果主动 undef 了 __ARMCOMPILER_LIBCXX, ARM block 永远不会激活).
 *
 * 全部用 #ifndef 包起来: 重复注入安全, 且不覆盖编译器/ARM block 已经定过的值.
 * ============================================================================
 */
#ifndef TFLM_DEFS_H
#define TFLM_DEFS_H

/* --- 打开 libcxx 的 ARM 配置 block ------------------------------------- */
#ifndef __ARMCOMPILER_LIBCXX
#define __ARMCOMPILER_LIBCXX 1
#endif

/* --- ARM block 之外仍需显式提供的 ---------------------------------------- */
/* 用 -fno-exceptions / -fno-rtti 编译, libcxx 必须知道 */
#ifndef _LIBCPP_HAS_NO_EXCEPTIONS
#define _LIBCPP_HAS_NO_EXCEPTIONS 1
#endif
#ifndef _LIBCPP_HAS_NO_RTTI
#define _LIBCPP_HAS_NO_RTTI 1
#endif

/* compiler.h:22-32 只认 __clang__ / __apple_build_version__ / __GNUC__,
 * AC6 一个都不定义 -> 必须显式声明 "我是 clang 系的" */
#ifndef _LIBCPP_COMPILER_CLANG_BASED
#define _LIBCPP_COMPILER_CLANG_BASED 1
#endif
#ifndef _LIBCPP_CLANG_VER
#define _LIBCPP_CLANG_VER 1700
#endif

/* platform.h:47 对 __BYTE_ORDER__ 做 hard #error.
 * 用数值, 不用 __ORDER_LITTLE_ENDIAN__ token (armasm 会当未定义符号). */
#ifndef __BYTE_ORDER__
#define __BYTE_ORDER__ 1234
#endif
#ifndef __ORDER_LITTLE_ENDIAN__
#define __ORDER_LITTLE_ENDIAN__ 1234
#endif
#ifndef __ORDER_BIG_ENDIAN__
#define __ORDER_BIG_ENDIAN__ 4321
#endif
#ifndef __ORDER_PDP_ENDIAN__
#define __ORDER_PDP_ENDIAN__ 3412
#endif

/* --- ARM block 自己会定义, 但 .c 编译路径走不到, 这里兜底 --------------- */
/* (只有 .cc 真正用 libcxx 头, 这几个只是保险) */
#ifndef _LIBCPP_HAS_NO_VENDOR_AVAILABILITY_ANNOTATIONS
#define _LIBCPP_HAS_NO_VENDOR_AVAILABILITY_ANNOTATIONS 1
#endif
#ifndef _LIBCPP_HAS_NO_THREADS
#define _LIBCPP_HAS_NO_THREADS 1
#endif
#ifndef _LIBCPP_HAS_NO_MONOTONIC_CLOCK
#define _LIBCPP_HAS_NO_MONOTONIC_CLOCK 1
#endif
#ifndef _LIBCPP_ABI_VERSION
#define _LIBCPP_ABI_VERSION 2
#endif
#ifndef _LIBCPP_HAS_NO_FILESYSTEM_LIBRARY
#define _LIBCPP_HAS_NO_FILESYSTEM_LIBRARY 1
#endif
#ifndef _LIBCPP_HAS_NO_INT128
#define _LIBCPP_HAS_NO_INT128 1
#endif
#ifndef _LIBCPP_PSTL_CPU_BACKEND_SERIAL
#define _LIBCPP_PSTL_CPU_BACKEND_SERIAL 1
#endif
#ifndef _LIBCPP_HAS_NO_INCOMPLETE_PSTL
#define _LIBCPP_HAS_NO_INCOMPLETE_PSTL 1
#endif
#ifndef _LIBCPP_STATIC
#define _LIBCPP_STATIC 1
#endif
#ifndef _LIBCPP_HIDE_FROM_ABI_PER_TU_BY_DEFAULT
#define _LIBCPP_HIDE_FROM_ABI_PER_TU_BY_DEFAULT 1
#endif
#ifndef _LIBCPP_HAS_NO_C11_ALIGNED_ALLOC
#define _LIBCPP_HAS_NO_C11_ALIGNED_ALLOC 1
#endif
#ifndef _LIBCPP_HAS_NO_OFF_T_FUNCTIONS
#define _LIBCPP_HAS_NO_OFF_T_FUNCTIONS 1
#endif

#endif /* TFLM_DEFS_H */
