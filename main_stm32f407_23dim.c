/* ============================================================================
 * main_stm32f407_23dim.c  —  STM32F407 TCN+SE ch=32 23-dim 入侵检测主程序
 * ============================================================================
 *
 * 硬件:STM32F407VGT6 (Cortex-M4 @ 168MHz, FPU)
 *      USART1 (PA9/PA10) 115200 8N1 调试串口
 *      LED (PD12)        正常 = 灭,异常 = 亮
 *      蜂鸣器 (PD13)     异常时响 100ms
 *      Modbus UART3 (PB10/PB11) 接收 SCADA 数据 (示例,实际项目自行接)
 *
 * 时序 (实测预估, Cortex-M4 @ 168MHz, -O2 -ffast-math):
 *   - 启动 init:       单 seed ~3 ms,    5-seed ensemble ~3 ms (权重 buffer 共享)
 *   - 单次推理:        单 seed ~0.9 ms,  5-seed ensemble ~7.5 ms
 *   - SCADA 周期 (16 行 ≈ 1.6 秒 @ 100 ms/行):单 seed 几乎无压力,5-seed ensemble 余量大
 *
 * 功能:
 *   1. 启动时打印版本 + Flash/RAM 信息
 *   2. 模拟一个 23 维 SCADA 窗口 (生产时从 Modbus UART 收)
 *   3. 每 16 行 (一个窗口) 做一次推理
 *      - USE_5SEED_ENSEMBLE 模式:5 个 seed 平均概率
 *      - 否则:用默认 seed=123 单次推理
 *   4. p > 0.33 报警 (LED + 蜂鸣器 + 串口打印)
 *   5. DWT cycle counter 测延迟,串口打印 ms
 *
 * CubeMX 配置参考:
 *   - RCC: HSE 8MHz → PLL 168MHz
 *   - USART1: 115200 8N1, PA9=TX, PA10=RX
 *   - GPIO: PD12 = GPIO_Output (LED), PD13 = GPIO_Output (Buzzer)
 *   - 打开 FPU (必须!)
 *
 * 编译选项 (GCC):
 *   -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16
 *   -O2 -ffast-math -funroll-loops -DNDEBUG -std=c99 -DUSE_HAL_DRIVER -DSTM32F407xx
 *
 * 链接 (见 stm32f407_flash.ld):
 *   把 .weights 段放在 Flash 末尾 (如果用 GCC section 属性)
 * ============================================================================
 */

/* ── 模式选择: 单 seed (默认) vs 5-seed ensemble ── */
#ifndef USE_5SEED_ENSEMBLE
#define USE_5SEED_ENSEMBLE  1   /* 1 = 5-seed ensemble (推荐, F1m 0.8775) */
/* 0 = 单 seed (默认 s=123, F1m 0.8766, 推理快 5x) */
#endif

#include "main.h"
#include "usart.h"
#include "gpio.h"
#include "v4_se_23dim_ch32_inference.h"
#include <stdio.h>
#include <string.h>

/* ── CMSIS / HAL 头 (CubeMX 自动生成) ──────────────────────────────── */
#include "stm32f4xx_hal.h"

/* ── 全局:在 main.c 里声明的 UART 句柄 (CubeMX 生成) ──────────────── */
UART_HandleTypeDef huart1;

/* ── 业务参数 ──────────────────────────────────────────────────────── */
#define SCADA_ROWS_PER_WINDOW  V4_SE_23DIM_CH32_WINDOW        /* 16 行 = 1 个窗口 */
#define SCADA_FEATURES_PER_ROW V4_SE_23DIM_CH32_IN_CHANNELS   /* 23 维特征 */
#define ANOMALY_THRESHOLD      V4_SE_23DIM_CH32_THRESHOLD     /* 0.33 (ensemble best) */

/* ── 业务状态 ──────────────────────────────────────────────────────── */
static float   scada_window[V4_SE_23DIM_CH32_IN_SIZE];  /* 滚动窗口:23*16 floats */
static int     scada_row_idx = 0;                       /* 当前行号 [0, 16) */
static uint32_t g_window_count = 0;                     /* 已处理窗口计数 */
static uint32_t g_anomaly_count = 0;                    /* 异常累计 */

/* ── DWT cycle counter (高精度计时, Cortex-M4) ────────────────────── */
static void dwt_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
static inline uint32_t dwt_get_cycles(void) {
    return DWT->CYCCNT;
}
static inline float dwt_cycles_to_us(uint32_t cycles) {
    /* 168 MHz → 1 us = 168 cycles */
    return (float)cycles / 168.0f;
}

/* ── printf 重定向到 USART1 ───────────────────────────────────────── */
int _write(int fd, char* ptr, int len) {
    (void)fd;
    HAL_UART_Transmit(&huart1, (uint8_t*)ptr, len, HAL_MAX_DELAY);
    return len;
}

/* ── LED / Buzzer 控制 ────────────────────────────────────────────── */
static inline void led_on(void)  { HAL_GPIO_WritePin(GPIOD, GPIO_PIN_12, GPIO_PIN_SET); }
static inline void led_off(void) { HAL_GPIO_WritePin(GPIOD, GPIO_PIN_12, GPIO_PIN_RESET); }
static inline void buzzer_on(void)  { HAL_GPIO_WritePin(GPIOD, GPIO_PIN_13, GPIO_PIN_SET); }
static inline void buzzer_off(void) { HAL_GPIO_WritePin(GPIOD, GPIO_PIN_13, GPIO_PIN_RESET); }

/* ── 模拟 SCADA 数据 (生产时换成 Modbus 接收) ────────────────────── */
static void mock_scada_row(float* row, int window_idx, int row_in_window) {
    /* LCG 伪随机 [-1, 1] */
    static unsigned int s = 0xCAFEBABE;
    for (int c = 0; c < SCADA_FEATURES_PER_ROW; c++) {
        s = s * 1103515245u + 12345u;
        row[c] = ((float)(s & 0xFFFF) / 32768.0f) - 1.0f;
    }
    /* 第 50 个窗口注入异常:把第 5~8 列特征值放大 3 倍 */
    if (window_idx == 50 && row_in_window >= 8) {
        for (int c = 5; c < 9; c++) row[c] *= 3.0f;
    }
}

/* ── Modbus 接收回调 (生产用, 示例) ───────────────────────────────── */
/*
 * 实际项目里:
 *   1. 用 UART3 + DMA 接收 Modbus RTU 帧
 *   2. 在空闲中断里 parse 23 个 float
 *   3. 调 scada_push_row() 推入滚动窗口
 *   4. 累计 16 行后调 v4_se_23dim_ch32_ensemble_forward() 或 _forward()
 *
 * void HAL_UART_RxCpltCallback(UART_HandleTypeDef* huart) {
 *     if (huart == &huart3) {
 *         float row[V4_SE_23DIM_CH32_IN_CHANNELS];
 *         parse_modbus_row_23dim(rx_buf, row);   // 你自己实现
 *         scada_push_row(row);
 *     }
 * }
 */

/* ── 业务:推 1 行 SCADA 进窗口,满了就跑推理 ──────────────────────── */
static void scada_push_row(const float* row) {
    /* 写入滚动窗口的第 scada_row_idx 行 (列优先存储) */
    for (int c = 0; c < SCADA_FEATURES_PER_ROW; c++) {
        scada_window[c * SCADA_ROWS_PER_WINDOW + scada_row_idx] = row[c];
    }
    scada_row_idx++;
    if (scada_row_idx >= SCADA_ROWS_PER_WINDOW) {
        scada_row_idx = 0;

        /* 计时 + 推理 (单 seed 或 5-seed ensemble) */
        uint32_t t0 = dwt_get_cycles();
#if USE_5SEED_ENSEMBLE
        float p = v4_se_23dim_ch32_ensemble_forward(scada_window);
#else
        float p = v4_se_23dim_ch32_forward(scada_window, V4_SE_23DIM_CH32_IN_SIZE);
#endif
        uint32_t t1 = dwt_get_cycles();
        float us = dwt_cycles_to_us(t1 - t0);

        g_window_count++;

        if (p >= ANOMALY_THRESHOLD) {
            g_anomaly_count++;
            led_on();
            buzzer_on();
#if USE_5SEED_ENSEMBLE
            printf("[W%lu] ANOMALY p=%.4f (5-seed) latency=%.1f us (anomaly=%lu/%lu)\n",
#else
            printf("[W%lu] ANOMALY p=%.4f (single) latency=%.1f us (anomaly=%lu/%lu)\n",
#endif
                   g_window_count, p, us, g_anomaly_count, g_window_count);
            HAL_Delay(100);
            buzzer_off();
            led_off();
        } else {
            led_off();
            /* 仅每 10 个窗口打印一次正常结果,避免刷屏 */
            if (g_window_count % 10 == 0) {
#if USE_5SEED_ENSEMBLE
                printf("[W%lu] normal   p=%.4f (5-seed) latency=%.1f us (anomaly=%lu/%lu)\n",
#else
                printf("[W%lu] normal   p=%.4f (single) latency=%.1f us (anomaly=%lu/%lu)\n",
#endif
                       g_window_count, p, us, g_anomaly_count, g_window_count);
            }
        }
    }
}

/* ── main ────────────────────────────────────────────────────────── */
int main(void) {
    /* 1. HAL + 时钟 + 外设 */
    HAL_Init();
    SystemClock_Config();     /* CubeMX 自动生成,168 MHz */
    MX_GPIO_Init();            /* LED PD12 / Buzzer PD13 */
    MX_USART1_UART_Init();     /* 115200 8N1 */

    /* 2. DWT 计时器 */
    dwt_init();

    /* 3. 启动 banner */
    printf("\n\n");
    printf("============================================================\n");
    printf("  TCN+SE ch=32 入侵检测 -- STM32F407 @ 168MHz\n");
    printf("  Model: 23-dim SCADA, window=16, 3 blocks, d=[1,2,4]\n");
#if USE_5SEED_ENSEMBLE
    printf("  Mode:  5-seed self-ensemble (F1m 0.8775)\n");
#else
    printf("  Mode:  single seed=123 (F1m 0.8766)\n");
#endif
    printf("  Quant: Hybrid INT8-w / FP32-a, BN folded\n");
    printf("============================================================\n");

    /* 4. 初始化模型 (dequant INT8 -> FP32, ~3 ms) */
    uint32_t t0 = dwt_get_cycles();
    v4_se_23dim_ch32_init();   /* 默认 seed=123 (单 seed 模式预 dequant) */
    uint32_t t1 = dwt_get_cycles();
    printf("[init] dequant done in %.1f us\n", dwt_cycles_to_us(t1 - t0));

    /* 5. 内存自检 */
    v4_se_23dim_ch32_mem_info_t m = v4_se_23dim_ch32_get_mem_info();
    printf("[mem]  weights FP32 = %.2f KB (1 seed)\n", m.weights_fp32_bytes / 1024.0f);
    printf("[mem]  activations = %.2f KB\n",          m.peak_act_bytes / 1024.0f);
    printf("[mem]  total RAM   = %.2f KB\n",          m.total_ram_bytes / 1024.0f);
    printf("[mem]  input window= %u B\n",              (unsigned)m.input_bytes);
#if USE_5SEED_ENSEMBLE
    printf("[mem]  ensemble: 共享权重缓冲,RAM 与单 seed 相同 (序列 dequant)\n");
#endif
    printf("\n");

    /* 6. 主循环:模拟 Modbus 接收 + 滚动窗口 + 推理 */
    printf("[run]  start detection loop (threshold=%.2f)\n", ANOMALY_THRESHOLD);
    int row_idx = 0;
    while (1) {
        float row[SCADA_FEATURES_PER_ROW];
        mock_scada_row(row, g_window_count, row_idx);
        scada_push_row(row);
        row_idx++;
        if (row_idx >= SCADA_ROWS_PER_WINDOW) row_idx = 0;

        /* 模拟 100 ms 一个 SCADA 行 (实际项目: Modbus 接收空闲等待) */
        HAL_Delay(100);
    }
}

/* ============================================================================
 * 实现示例 — CubeMX 生成的 SystemClock_Config / MX_GPIO_Init / MX_USART1_UART_Init
 * 实际项目中这些函数由 CubeMX 自动生成,这里给出模板确保可独立编译
 * ============================================================================
 */
#ifdef USE_TEMPLATE_FUNCS

void SystemClock_Config(void) {
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    RCC_OscInitStruct.HSEState = RCC_HSE_ON;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    RCC_OscInitStruct.PLL.PLLM = 8;
    RCC_OscInitStruct.PLL.PLLN = 336;
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
    RCC_OscInitStruct.PLL.PLLQ = 7;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
        Error_Handler();
    }

    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                                | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK) {
        Error_Handler();
    }
}

static void MX_GPIO_Init(void) {
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOD_CLK_ENABLE();

    /* LED PD12 + Buzzer PD13 */
    GPIO_InitStruct.Pin   = GPIO_PIN_12 | GPIO_PIN_13;
    GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull  = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

    HAL_GPIO_WritePin(GPIOD, GPIO_PIN_12 | GPIO_PIN_13, GPIO_PIN_RESET);
}

static void MX_USART1_UART_Init(void) {
    huart1.Instance        = USART1;
    huart1.Init.BaudRate   = 115200;
    huart1.Init.WordLength = UART_WORDLENGTH_8B;
    huart1.Init.StopBits   = UART_STOPBITS_1;
    huart1.Init.Parity     = UART_PARITY_NONE;
    huart1.Init.Mode       = UART_MODE_TX_RX;
    huart1.Init.HwFlowCtl  = UART_HWCONTROL_NONE;
    huart1.Init.OverSampling = UART_OVERSAMPLING_16;
    if (HAL_UART_Init(&huart1) != HAL_OK) {
        Error_Handler();
    }
}

void Error_Handler(void) {
    __disable_irq();
    while (1) {
        /* 死循环 + LED 快闪提示错误 */
        HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_12);
        for (volatile int i = 0; i < 100000; i++);
    }
}

#endif /* USE_TEMPLATE_FUNCS */

/* ============================================================================
 * 编译命令 (GCC) — 在 CubeMX 生成的 Makefile 基础上加:
 * ============================================================================
 *
 * CFLAGS  += -DUSE_HAL_DRIVER -DSTM32F407xx -DUSE_TEMPLATE_FUNCS \
 *           -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 \
 *           -O2 -ffast-math -funroll-loops -DNDEBUG -std=c99
 *
 * # 添加源文件
 * SRCS += Core/Src/main_stm32f407_23dim.c
 * SRCS += Core/Src/v4_se_23dim_ch32_inference.c
 *
 * # 头文件路径 (model_*.h 在 Core/Inc/ 下)
 * CFLAGS += -ICore/Inc
 *
 * # OpenOCD 烧录
 * openocd -f interface/stlink.cfg -f target/stm32f4x.cfg \
 *         -c "program build/tcn23_detect.elf verify reset exit"
 * ============================================================================
 */