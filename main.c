/* ============================================================================
 * main.c  —  V19 Inference Demo / 集成示例
 * ============================================================================
 *
 * 用途:展示如何调用 v19_inference API,以及 3 种集成方式
 *   1. PC 上跑(用 main() 测试)            ← 你可以直接编译运行
 *   2. STM32 HAL 项目(用 MX_ 模板)         ← 替换 main,加 usart 重定向
 *   3. ESP-IDF (用 app_main)
 *
 * 编译 (PC 测试):
 *   gcc -O2 -std=c99 -o v19_demo main.c v19_inference.c -lm
 *   ./v19_demo
 *
 * 编译 (STM32, 假设 arm-none-eabi-gcc):
 *   arm-none-eabi-gcc -O2 -std=c99 -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 \
 *       -DNDEBUG -o v19.elf main.c v19_inference.c -lm
 *
 * 编译 (ESP-IDF, CMakeLists.txt 里):
 *   idf_component_register(SRCS "main.c" "v19_inference.c"
 *                          INCLUDE_DIRS "."
 *                          REQUIRES driver)
 * ============================================================================
 */
#include <stdio.h>
#include <string.h>
#include "v19_inference.h"

/* ────────────────────────────────────────────────────────────────────────
 * 演示 1:用合成数据测试 (无外部依赖)
 * ──────────────────────────────────────────────────────────────────────── */
static void demo_synthetic(void) {
    printf("\n=== Demo 1: Synthetic random input ===\n");
    v19_init();

    /* 模拟 16 步的 44 维特征窗口 */
    float x[V19_IN_SIZE];
    for (int i = 0; i < V19_IN_SIZE; i++) {
        /* 简单 LCG 伪随机 [-1, 1] */
        static unsigned int s = 0xDEADBEEF;
        s = s * 1103515245u + 12345u;
        x[i] = ((float)(s & 0xFFFF) / 32768.0f) - 1.0f;
    }

    float p = v19_forward(x, V19_IN_SIZE);
    printf("  Probability of anomaly: %.4f\n", p);
    printf("  Verdict: %s (threshold 0.5)\n", p >= 0.5f ? "ANOMALY" : "NORMAL");

    v19_benchmark(2000);
    v19_mem_info_t m = v19_get_mem_info();
    printf("  Memory: weights=%u B (%.2f KB), activations peak=%u B (%.2f KB)\n",
           (unsigned)m.weights_fp32_bytes, m.weights_fp32_bytes / 1024.0f,
           (unsigned)m.peak_act_bytes, m.peak_act_bytes / 1024.0f);
}

/* ────────────────────────────────────────────────────────────────────────
 * 演示 2:从 .bin 文件加载输入(若有的话)
 *   - 文件格式: 704 个 float32, layout = (44, 16) row-major
 *   - 你可以从 PyTorch 测试集 dump 1 个 sample 出来:
 *     >>> np_sample = Xte[0].T.astype(np.float32).tobytes()  # 704 * 4 = 2816 B
 *     >>> open('sample.bin', 'wb').write(np_sample)
 * ──────────────────────────────────────────────────────────────────────── */
#include <stdlib.h>
static int demo_load_from_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("  (skip: %s not found)\n", path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz != V19_IN_SIZE * (long)sizeof(float)) {
        printf("  (skip: %s size=%ld, expected %ld)\n",
               path, sz, (long)(V19_IN_SIZE * sizeof(float)));
        fclose(f);
        return -1;
    }
    float x[V19_IN_SIZE];
    if (fread(x, sizeof(float), V19_IN_SIZE, f) != V19_IN_SIZE) {
        fclose(f); return -1;
    }
    fclose(f);
    float p = v19_forward(x, V19_IN_SIZE);
    printf("  Loaded %s, prob=%.4f, verdict=%s\n",
           path, p, p >= 0.5f ? "ANOMALY" : "NORMAL");
    return 0;
}

/* ────────────────────────────────────────────────────────────────────────
 * 演示 3:循环实时推理 (生产用模式)
 *   - 模拟从 Modbus 串口接收数据,每收到 16 行特征做一次推理
 * ──────────────────────────────────────────────────────────────────────── */
static void demo_realtime_loop(void) {
    printf("\n=== Demo 3: Simulated real-time loop (10 windows) ===\n");
    static float window[V19_IN_SIZE];
    int n = 0;
    for (int step = 0; step < 160; step++) {  /* 10 windows × 16 steps */
        /* 模拟 1 行新数据 (44 floats) */
        static unsigned int s = 0xCAFEBABE;
        for (int c = 0; c < V19_IN_CHANNELS; c++) {
            s = s * 1103515245u + 12345u;
            window[(n * V19_IN_CHANNELS) + c] = ((float)(s & 0xFFFF) / 32768.0f) - 1.0f;
        }
        n++;
        if (n == V19_WINDOW) {
            float p = v19_forward(window, V19_IN_SIZE);
            printf("  Window %d: prob=%.4f %s\n",
                   step / V19_WINDOW + 1, p,
                   p >= 0.5f ? "ANOMALY" : "NORMAL");
            n = 0;
        }
    }
}

/* ────────────────────────────────────────────────────────────────────────
 * main
 * ──────────────────────────────────────────────────────────────────────── */
int main(void) {
    printf("TCN V19 MCU Inference Demo\n");
    printf("==========================\n");
    printf("Model: 4,237 params, ch=12, b=2, SE=8 (hybrid INT8-w + FP32-a)\n");

    demo_synthetic();
    printf("\n=== Demo 2: Load from .bin (optional) ===\n");
    demo_load_from_file("sample_input.bin");
    demo_realtime_loop();

    printf("\nDone.\n");
    return 0;
}

/* ============================================================================
 * STM32 HAL 集成模板
 * ============================================================================
 *
 * 把 int main(void) 替换成:
 *
 * #include "main.h"
 * #include "usart.h"
 * #include "v19_inference.h"
 *
 * // 在 main.h 里加:
 * // extern UART_HandleTypeDef huart1;
 * // int _write(int fd, char* ptr, int len) {
 * //     HAL_UART_Transmit(&huart1, (uint8_t*)ptr, len, HAL_MAX_DELAY);
 * //     return len;
 * // }
 *
 * int main(void) {
 *     HAL_Init();
 *     SystemClock_Config();   // 168 MHz for STM32F407
 *     MX_USART1_UART_Init();  // 115200 baud
 *     MX_GPIO_Init();
 *
 *     printf("TCN V19 init...\n");
 *     v19_init();
 *     printf("OK, weights=%u B, act peak=%u B\n", ...);
 *
 *     // 主循环:从 Modbus/UART 收数据
 *     float window[V19_IN_SIZE];
 *     int n = 0;
 *     while (1) {
 *         // ... 收 1 行 44 floats ...
 *         // (略, 实际用 DMA + 环形缓冲)
 *         if (++n == V19_WINDOW) {
 *             float p = v19_forward(window, V19_IN_SIZE);
 *             if (p >= 0.5f) HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);  // 报警 LED
 *             else           HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);
 *             n = 0;
 *         }
 *     }
 * }
 *
 * ============================================================================
 * ESP-IDF 集成模板
 * ============================================================================
 *
 * #include "v19_inference.h"
 *
 * void app_main(void) {
 *     v19_init();
 *     ESP_LOGI("V19", "Init OK");
 *
 *     float window[V19_IN_SIZE];
 *     int n = 0;
 *     while (1) {
 *         // ... 收 1 行 44 floats ...
 *         if (++n == V19_WINDOW) {
 *             float p = v19_forward(window, V19_IN_SIZE);
 *             ESP_LOGI("V19", "p=%.4f", p);
 *             n = 0;
 *         }
 *         vTaskDelay(pdMS_TO_TICKS(10));
 *     }
 * }
 *
 * ============================================================================
 */
