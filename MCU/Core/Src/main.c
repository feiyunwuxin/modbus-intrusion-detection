/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include <stdio.h>
#include "main.h"
#include "dma.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

#include "ch32_inference.h"          /* TFLite Micro wrapper (C shim) */
#include "modbus_ids.h"
#include "modbus_rtu.h"
#include "sliding_window.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* 2026-09-19 第七轮重构: H743 永远是纯 sniff 模式, 不再需要
 * IDS_SNIFF_ONLY 切换. PC 模拟整条 Modbus 总线 (主站+从站),
 * H743 只解析 PC 重放的 raw17 帧, 不主动轮询, 不发指令. */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
/* Modbus RX DMA buffer — D-Cache 行对齐 (32B) + AXI SRAM
 * (避免与 ch32_inference 的 section 冲突,放在不同 .bss 段)
 *
 * 注意: 不能加 static, 因为 stm32h7xx_it.c 通过 extern 引用它做
 *       SCB_InvalidateDCache_by_Addr() + modbus_ids_on_frame(). */
__attribute__((aligned(32), section(".dma_buffer")))
uint8_t modbus_rx_buf[256];

/* 2026-09-19: 删除 Modbus 主站轮询状态变量 (H743 永远 sniff) */

/* 2026-10-17 sw_buf dump 验证: ASCII 'D' 单字节命令接收 (修复 1B IT bug)
 *   - 2026-10-14 错误设计: HAL_UART_Receive_IT(1B) 与 256B DMA-RX (IDLE) 看似并行,
 *     实际 HAL_UART_Receive_IT 检查 RxState==READY, 而 DMA-RX 已置 BUSY_RX,
 *     导致 1B IT 静默返回 HAL_BUSY (return value 未检查), 永远不使能.
 *   - 2026-10-17 修复: 'D' 检测移到 stm32h7xx_it.c 的 HAL_UARTEx_RxEventCallback
 *     (DMA-RX IDLE 回调), 检测 1 字节 'D' → 置 cmd_rx_ready.
 *   - main.c while(1) 看到 cmd_rx_ready, DMA-TX g_last_ch32_input (1472B) 给 PC.
 *   注: cmd_rx_shadow 保留 (usart.c HAL_UART_RxCpltCallback 仍引用), 但实际不工作. */
__attribute__((aligned(32), section(".axi_sram")))
uint8_t cmd_rx_shadow[1];
volatile uint8_t cmd_rx_byte  = 0;
volatile uint8_t cmd_rx_ready = 0;

/* ⭐ 2026-10-17 Watchdog: USART1 RX 健康监控 (v4 简化)
 *   v1-v3: 周期性 5 秒无 IDLE 增长 = stalled 强制 abort+restart
 *   v4 简化: TFLM 推理移到 main loop 后 ISR 几 us 完成, USART 不会因 ISR 阻塞死锁.
 *           去掉周期性 stalled 检测 (避免误判打断 TFLM 推理).
 *   保留: ISR 内 restart 失败时置 g_rerx_need_recovery flag,
 *         main loop 检测 flag 后非阻塞 abort+restart (兜底极端情况). */
extern volatile uint32_t g_idle_count;
extern volatile uint32_t g_rerx_fail_count;
extern volatile uint8_t  g_rerx_need_recovery;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
void PeriphCommonClock_Config(void);
/* USER CODE BEGIN PFP */
 uint8_t	TextTest[]	=	"hello H743";
 //__attribute__((section(".ram_d2"))) 	
volatile	uint8_t Uart1RxBuf[1024];// __attribute__((section(".ARM.__at_0x24000000")));
 volatile	uint16_t	Uart1RxLen;
 volatile	uint8_t	Uart1Revd	=0;
 volatile	uint8_t	Uart1TxCplt =	0;
 uint16_t	Uart1TxLen;
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
	void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct;

  /* Disable the MPU */
  HAL_MPU_Disable();

  /* Configure the MPU as Strongly ordered for not defined regions */
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.BaseAddress = 0x00;
  MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
  MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER0;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.SubRegionDisable = 0x87;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;

  HAL_MPU_ConfigRegion(&MPU_InitStruct);

  /* Enable the MPU */
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
//	uint16_t	i;
	//MPU_Config();
	HAL_MPU_Disable();

  /* USER CODE END 1 */
//		  SCB_DisableDCache();  // ��ʱ�ر�
//  SCB_DisableICache();
  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* Configure the peripherals common clocks */
  PeriphCommonClock_Config();

  /* ⭐ 2026-10-15 修复: cache 启用必须放到 SystemClock_Config 之后
   * 原因: H743 HAL_Init 之后立即 SystemClock_Config 会从 FLASH_LATENCY_DEFAULT
   *       切到 FLASH_LATENCY_2 (重置 flash wait state). PLL 还没起来前开 DCache,
   *       D-Cache line fill 会撞 spurious bus stall,触发 HardFault 或使 NVIC
   *       全部不响应 (包括 TIM1 update). */
  SCB_InvalidateICache();
  SCB_EnableICache();
  SCB_InvalidateDCache();
  SCB_EnableDCache();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_USART1_UART_Init();
  MX_TIM1_Init();
  /* USER CODE BEGIN 2 */
	HAL_TIM_Base_Start_IT(&htim1); 
	__HAL_UART_CLEAR_IDLEFLAG(&huart1);
//	__HAL_UART_ENABLE_IT(&huart1,UART_IT_IDLE);
#ifdef LED
	HAL_GPIO_WritePin(GPIOB,GPIO_PIN_1,GPIO_PIN_RESET);
	HAL_GPIO_WritePin(GPIOB,GPIO_PIN_0,GPIO_PIN_RESET);
	#endif
	HAL_UART_Transmit(&huart1,TextTest,10,1000);
	Uart1Revd	=	0;
	//HAL_UART_Receive_DMA(&huart1,(uint8_t *)Uart1RxBuf,UART_BUFFER_SIZE); //
	 if (HAL_UARTEx_ReceiveToIdle_DMA(&huart1, modbus_rx_buf, sizeof(modbus_rx_buf)) != HAL_OK) {
    Error_Handler();
  }
	/* 2026-10-17: 删除 1B IT 启动 (HAL_UART_Receive_IT 静默返回 HAL_BUSY 因 RxState=BUSY_RX)
	 *   'D' 命令改在 stm32h7xx_it.c HAL_UARTEx_RxEventCallback IDLE 回调里检测. */
	printf("\r\n=== Route E Initial (Hand-rolled C kernel, no TFLM) ===\r\n");
	/* DBG BYPASS 2: 早期 bypass, 即使 ch32_init fail 也能验证 GPIO */
	for (volatile int k = 0; k < 4000000; k++);  /* 让 user 看到 startup print */
	/* ⭐ 2026-10-18 路线 E 迁移: ch32_init() 返回 void (不再是 uint8_t)
	 *   TFLM AllocateTensors 可能 arena OOM, 旧版返回 0 失败; 手写 C 路径
	 *   无此风险, 启动失败直接 HardFault 即可. */
	ch32_init();  /* 启动期一次性 INT8 -> FP32 dequant (~3 ms @ 480 MHz) */
	ch32_mem_info_t mi = ch32_get_mem_info();
	printf("\r\n=== TCN+SE ch=32 23-dim H743 Deployment (Route E) ===\r\n");
	printf("Weights (FP32): %u B (%u KB)\r\n",
	       (unsigned)mi.weights_fp32_bytes,
	       (unsigned)(mi.weights_fp32_bytes / 1024));
	printf("Peak act:       %u B (%u KB)\r\n",
	       (unsigned)mi.peak_act_bytes,
	       (unsigned)(mi.peak_act_bytes / 1024));
	printf("Input:          %u B (%u KB)\r\n",
	       (unsigned)mi.input_bytes,
	       (unsigned)(mi.input_bytes / 1024));
	printf("Total RAM:      %u B (%u KB)\r\n",
	       (unsigned)mi.total_ram_bytes,
	       (unsigned)(mi.total_ram_bytes / 1024));
	printf("Threshold:      %.2f  (T scaling = %.2f)\r\n",
	       CH32_BEST_THRESHOLD, CH32_TEMPERATURE);
				 
	 printf("H743 IDS ready\r\n");			 
	 
				 
				 
				 /* ���ܻ�׼:100 ������ */
//  uint32_t avg_cycles = ch32_benchmark(100);
//  float avg_us = (float)avg_cycles / 480.0f;  /* 480 MHz */
//  printf("--- Benchmark (100 runs) ---\r\n");
//  printf("Avg cycles: %u\r\n", (unsigned)avg_cycles);
//  printf("Avg time:   %.3f us @ 480 MHz\r\n", avg_us);
//  printf("Throughput: %.0f inferences/sec\r\n", 1e6f / avg_us);
//HAL_UARTEx_ReceiveToIdle_DMA(&huart1,Uart1RxBuf,UART_BUFFER_SIZE);
	/* ---- Modbus IDS ��ʼ�� ---- */
  modbus_ids_init();
	printf("Modbus IDS armed on USART1 (PA10 RX) @ 115200 8N1\r\n");
  printf("Alarm LED: PB0 (ON = attack, prob >= %.2f)\r\n", CH32_BEST_THRESHOLD);
  printf("Mode: PURE BUS SNIFFER, 17-dim raw payload from PC replay\r\n");
  printf("  CRC check + 23-dim TCN+SE inference\r\n");
  /* USER CODE END 2 */

  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
		/* ⭐ 2026-10-17 FIX: 移除 DBG BYPASS continue 短路, 让 'D' dump 命令处理真正运行
		 * (原 2026-10-15 bypass 让 ch32_init 之后所有 main loop 逻辑不可达, 链接器 GC
		 *  会把 ch32_init + 47KB model data 当 unreferenced 删掉, 烧录的固件只剩 LED toggle) */
//		{
//			static int phase = 0;
//			phase++;
//			if (phase & 1) {
//				HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_SET);
//			} else {
//				HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_RESET);
//			}
//		}
//		if(Uart1Revd	==	1)
//		{
//			HAL_GPIO_TogglePin(GPIOB,GPIO_PIN_1);
//		}

		/* 2026-10-17: 处理 PC 'D' 命令, DMA-TX g_last_ch32_input (1472B)
		 *   'D' 检测由 stm32h7xx_it.c HAL_UARTEx_RxEventCallback 完成 (IDLE 回调里).
		 *   这里只消费 cmd_rx_ready 标志. */
		if (cmd_rx_ready) {
			cmd_rx_ready = 0;
			if (cmd_rx_byte == 'D') {
				/* 2026-10-17 FIX: printf 必须先于 HAL_UART_Transmit_DMA, 否则 fputc 阻塞写
				 * TDR 1 byte @ 87us 会跟 DMA 1 byte @ 87us 在 UART 硬件层交错, 污染 dump.
				 * 改成 printf 完 → 启动 DMA → DMA 全程独占 UART TX (直到 TC). */
				printf("\r\n[D-CMD] rx 'D' main loop, dump 1472B starting\r\n");
				/* g_last_ch32_input 在 .axi_sram (write-back), DMA 读前必须 clean DCache */
				SCB_CleanDCache_by_Addr((uint32_t *)g_last_ch32_input,
				                        (int32_t)sizeof(g_last_ch32_input));
				/* 非阻塞 DMA-TX, 启动后 UART TX 寄存器由 DMA1 Stream1 独占写入 */
				HAL_StatusTypeDef drc = HAL_UART_Transmit_DMA(&huart1,
				                      (uint8_t *)g_last_ch32_input,
				                      (uint16_t)sizeof(g_last_ch32_input));
				if (drc != HAL_OK) {
					printf("[D-CMD] HAL_UART_Transmit_DMA failed rc=%d (DMA busy?)\r\n", (int)drc);
				}
				/* ⭐ 2026-10-17 fix: 不能再 printf "[D-CMD] DMA-TX 1472B started", 会跟 DMA 字符交错.
				 * 但 PC 端 XCOM 默认文本模式看不到 binary (1472B 中 0x00/0x0A/0x0D 被截断),
				 * 加 "DUMP-END" 标记让 XCOM 知道 binary 在哪一段.
				 * 风险: 跟 DMA 字符交错 → 实际 1472B @ 115200 = 128ms, printf 在 128ms 后才调,
				 *      DMA 早已完成. 安全. */
				uint32_t tc_tick = HAL_GetTick();
				while (huart1.gState != HAL_UART_STATE_READY) {
					if (HAL_GetTick() - tc_tick > 500u) break;  /* 500ms 超时保护 */
				}
				printf("\r\n[D-CMD] DMA-TX 1472B done\r\n");
			}
			/* 不需要 re-arm 1B IT: 'D' 由 IDLE 回调里下一个 1-byte 'D' 触发 */
		}

		/* ⭐ 2026-10-17: 消费 ISR 置的 inference pending flag.
		 *   ISR 凑齐 16 帧时 memcpy 23*16=368 floats 到 snapshot, 置 flag.
		 *   TFLM 推理 (~2ms) 在 main loop 调, 避免 ISR 内阻塞导致 USART 死锁. */
		modbus_ids_run_inference_if_pending();

		/* ⭐ 2026-10-17 USART1 RX Watchdog (v4 简化)
		 *   v1-v3: 周期性 5 秒无 IDLE 增长 = stalled 强制 abort+restart
		 *   v4 简化: TFLM 推理移到 main loop 后 ISR 几 us 完成, USART 不会因 ISR 阻塞死锁.
		 *           去掉周期性 stalled 检测 (避免误判打断 TFLM 推理).
		 *   保留: ISR 内 restart 失败时置 g_rerx_need_recovery flag,
		 *         main loop 检测 flag 后非阻塞 abort+restart (兜底极端情况). */
		if (g_rerx_need_recovery) {
			g_rerx_need_recovery = 0;
			printf("[WD] ISR reported re-arm fail, force abort+restart\r\n");
			HAL_StatusTypeDef rc_abort = HAL_UART_AbortReceive(&huart1);
			if (rc_abort != HAL_OK) {
				printf("[WD] abort FAILED rc=%d, skip restart\r\n", (int)rc_abort);
			} else {
				HAL_StatusTypeDef rc_wd = HAL_UARTEx_ReceiveToIdle_DMA(&huart1,
				                       modbus_rx_buf, sizeof(modbus_rx_buf));
				__HAL_DMA_DISABLE_IT(&hdma_usart1_rx, DMA_IT_HT);
				if (rc_wd != HAL_OK) {
					printf("[WD] recovery restart FAILED rc=%d\r\n", (int)rc_wd);
				} else {
					printf("[WD] recovery restart OK\r\n");
				}
			}
		}

  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 2;
  RCC_OscInitStruct.PLL.PLLN = 64;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief Peripherals Common Clock Configuration
  * @retval None
  */
void PeriphCommonClock_Config(void)
{
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

  /** Initializes the peripherals clock
  */
  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_TIM;
  PeriphClkInitStruct.TIMPresSelection = RCC_TIMPRES_ACTIVATED;

  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
