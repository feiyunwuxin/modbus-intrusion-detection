/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    stm32h7xx_it.c
  * @brief   Interrupt Service Routines.
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
#include "main.h"
#include "stm32h7xx_it.h"
#include <stdio.h>
/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN TD */

/* USER CODE END TD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/* External variables --------------------------------------------------------*/
extern TIM_HandleTypeDef htim1;
extern DMA_HandleTypeDef hdma_usart1_rx;
extern DMA_HandleTypeDef hdma_usart1_tx;
extern UART_HandleTypeDef huart1;

/* External variables --------------------------------------------------------*/
/* USER CODE BEGIN EV */

/* Modbus RX DMA buffer (������ main.c) */
extern uint8_t modbus_rx_buf[256];
/* Modbus IDS ����״̬�� */
extern void modbus_ids_on_frame(const uint8_t* buf, uint16_t len);
/* USER CODE END EV */
/* USER CODE BEGIN EV */

/* 2026-10-17 IDLE 路径诊断计数器 (volatile 防优化, no-static 让 main.c extern 可见) */
volatile uint32_t g_idle_count = 0;
volatile uint32_t g_rerx_fail_count = 0;
volatile uint32_t g_last_rerx_rc = 0;
volatile uint32_t g_last_size = 0;

/* ⭐ 2026-10-17: ISR 内 restart 失败时, 置 flag 给 main loop watchdog 处理.
 *   关键: ISR 内禁止调 HAL_UART_AbortReceive (阻塞等 DMA 停止会卡死 NVIC).
 *         实际表现: 13 alarm 后 USART 挂掉, log 显示 14th 帧之后再无 IDLE 回调.
 *         真因: ISR 内 abort 半完成, huart->RxState 卡在中间状态, IDLE 中断不再使能. */
volatile uint8_t  g_rerx_need_recovery = 0;

/* USER CODE END EV */

/******************************************************************************/
/*           Cortex Processor Interruption and Exception Handlers          */
/******************************************************************************/
/**
  * @brief This function handles Non maskable interrupt.
  */
void NMI_Handler(void)
{
  /* USER CODE BEGIN NonMaskableInt_IRQn 0 */

  /* USER CODE END NonMaskableInt_IRQn 0 */
  /* USER CODE BEGIN NonMaskableInt_IRQn 1 */
   while (1)
  {
  }
  /* USER CODE END NonMaskableInt_IRQn 1 */
}

/**
  * @brief This function handles Hard fault interrupt.
  */
void HardFault_Handler(void)
{
  /* USER CODE BEGIN HardFault_IRQn 0 */

  /* USER CODE END HardFault_IRQn 0 */
	
	
  __BKPT(0);    /* ⭐ 触发软件断点,Keil 会停在 BKPT,Call Stack 自动显示真正触发处 */
  while (1) {}
//  while (1)
//  {
//    /* USER CODE BEGIN W1_HardFault_IRQn 0 */
//    /* USER CODE END W1_HardFault_IRQn 0 */
//  }
}

/**
  * @brief This function handles Memory management fault.
  */
void MemManage_Handler(void)
{
  /* USER CODE BEGIN MemoryManagement_IRQn 0 */

  /* USER CODE END MemoryManagement_IRQn 0 */
  while (1)
  {
    /* USER CODE BEGIN W1_MemoryManagement_IRQn 0 */
    /* USER CODE END W1_MemoryManagement_IRQn 0 */
  }
}

/**
  * @brief This function handles Pre-fetch fault, memory access fault.
  */
void BusFault_Handler(void)
{
  /* USER CODE BEGIN BusFault_IRQn 0 */

  /* USER CODE END BusFault_IRQn 0 */
  while (1)
  {
    /* USER CODE BEGIN W1_BusFault_IRQn 0 */
    /* USER CODE END W1_BusFault_IRQn 0 */
  }
}

/**
  * @brief This function handles Undefined instruction or illegal state.
  */
void UsageFault_Handler(void)
{
  /* USER CODE BEGIN UsageFault_IRQn 0 */

  /* USER CODE END UsageFault_IRQn 0 */
  while (1)
  {
    /* USER CODE BEGIN W1_UsageFault_IRQn 0 */
    /* USER CODE END W1_UsageFault_IRQn 0 */
  }
}

/**
  * @brief This function handles System service call via SWI instruction.
  */
void SVC_Handler(void)
{
  /* USER CODE BEGIN SVCall_IRQn 0 */

  /* USER CODE END SVCall_IRQn 0 */
  /* USER CODE BEGIN SVCall_IRQn 1 */

  /* USER CODE END SVCall_IRQn 1 */
}

/**
  * @brief This function handles Debug monitor.
  */
void DebugMon_Handler(void)
{
  /* USER CODE BEGIN DebugMonitor_IRQn 0 */

  /* USER CODE END DebugMonitor_IRQn 0 */
  /* USER CODE BEGIN DebugMonitor_IRQn 1 */

  /* USER CODE END DebugMonitor_IRQn 1 */
}

/**
  * @brief This function handles Pendable request for system service.
  */
void PendSV_Handler(void)
{
  /* USER CODE BEGIN PendSV_IRQn 0 */

  /* USER CODE END PendSV_IRQn 0 */
  /* USER CODE BEGIN PendSV_IRQn 1 */

  /* USER CODE END PendSV_IRQn 1 */
}

/**
  * @brief This function handles System tick timer.
  */
void SysTick_Handler(void)
{
  /* USER CODE BEGIN SysTick_IRQn 0 */

  /* USER CODE END SysTick_IRQn 0 */
  HAL_IncTick();
  /* USER CODE BEGIN SysTick_IRQn 1 */

  /* USER CODE END SysTick_IRQn 1 */
}

/******************************************************************************/
/* STM32H7xx Peripheral Interrupt Handlers                                    */
/* Add here the Interrupt Handlers for the used peripherals.                  */
/* For the available peripheral interrupt handler names,                      */
/* please refer to the startup file (startup_stm32h7xx.s).                    */
/******************************************************************************/

/**
  * @brief This function handles DMA1 stream0 global interrupt.
  */
void DMA1_Stream0_IRQHandler(void)
{
  /* USER CODE BEGIN DMA1_Stream0_IRQn 0 */

  /* USER CODE END DMA1_Stream0_IRQn 0 */
  HAL_DMA_IRQHandler(&hdma_usart1_rx);
  /* USER CODE BEGIN DMA1_Stream0_IRQn 1 */

  /* USER CODE END DMA1_Stream0_IRQn 1 */
}

/**
  * @brief This function handles DMA1 stream1 global interrupt.
  */
void DMA1_Stream1_IRQHandler(void)
{
  /* USER CODE BEGIN DMA1_Stream1_IRQn 0 */

  /* USER CODE END DMA1_Stream1_IRQn 0 */
  HAL_DMA_IRQHandler(&hdma_usart1_tx);
  /* USER CODE BEGIN DMA1_Stream1_IRQn 1 */

  /* USER CODE END DMA1_Stream1_IRQn 1 */
}

/**
  * @brief This function handles TIM1 update interrupt.
  */
void TIM1_UP_IRQHandler(void)
{
  /* USER CODE BEGIN TIM1_UP_IRQn 0 */

  /* USER CODE END TIM1_UP_IRQn 0 */
  HAL_TIM_IRQHandler(&htim1);
  /* USER CODE BEGIN TIM1_UP_IRQn 1 */
	#ifdef LED
	HAL_GPIO_TogglePin(GPIOB,GPIO_PIN_0);
	#endif
  /* USER CODE END TIM1_UP_IRQn 1 */
}

/**
  * @brief This function handles USART1 global interrupt.
  */
//void USART1_IRQHandler(void)
//{
//  /* USER CODE BEGIN USART1_IRQn 0 */
//	uint32_t TempFlag = 0;
//  /* USER CODE END USART1_IRQn 0 */
//  HAL_UART_IRQHandler(&huart1);
//  /* USER CODE BEGIN USART1_IRQn 1 */
//	//__disable_irq();    
//	
//	TempFlag =  __HAL_UART_GET_FLAG(&huart1,UART_FLAG_IDLE);

//  if((TempFlag != RESET)){

//    __HAL_UART_CLEAR_IDLEFLAG(&huart1);
//    HAL_UART_DMAStop(&huart1);
//		Uart1RxLen =  UART_BUFFER_SIZE - __HAL_DMA_GET_COUNTER(&hdma_usart1_rx);
// 		if(Uart1RxLen)
//		{
//			Uart1Revd	=	1;
//		}
////__enable_irq();
//  }
//	TempFlag =  __HAL_UART_GET_FLAG(&huart1,UART_FLAG_IDLE);

////  if((TempFlag != RESET)){
////	
////		Uart1TxCplt	=	1;
////	}
//	HAL_UART_Receive_DMA(&huart1,(uint8_t *)Uart1RxBuf,UART_BUFFER_SIZE);
//  /* USER CODE END USART1_IRQn 1 */
//}
/**
  * @brief This function handles USART1 global interrupt.
  */
void USART1_IRQHandler(void)
{
  uint32_t TempFlag;                                        /* ⭐ 2026-10-15: 从文件作用域移进函数体内 */
  /* USER CODE BEGIN USART1_IRQn 0 */
  /* USER CODE END USART1_IRQn 0 */
  HAL_UART_IRQHandler(&huart1);
  /* USER CODE BEGIN USART1_IRQn 1 */
	TempFlag =  __HAL_UART_GET_FLAG(&huart1,UART_FLAG_IDLE);
	if((TempFlag != RESET)){

    __HAL_UART_CLEAR_IDLEFLAG(&huart1);
	}
  /* USER CODE END USART1_IRQn 1 */
}
/* USER CODE BEGIN 1 */
 /**
 * USART �����ж� + DMA ��ɻص� (HAL_UARTEx_ReceiveToIdle_DMA)
 * ÿ�յ�һ�� Modbus RTU ֡ (����г�ʱ���) ����һ��.
 *
 * ? �ؼ�:Cache һ����
 *   DMA д�� AXI SRAM (D-Cache ͨ�� RAM),CPU ��ǰ���� Invalidate.
 *   �������� D-Cache ��ľ����� (�ϴ� printf �Ĳ���).
 */
/* 2026-10-17: ASCII 'D' 单字节命令接收
 *   - 修复 2026-10-14 设计: 原计划 HAL_UART_Receive_IT(1B) 与 DMA-RX 并行,
 *     但 main.c:187 在 HAL_UARTEx_ReceiveToIdle_DMA 之后调用, RxState=BUSY_RX
 *     导致 HAL_UART_Receive_IT 静默返回 HAL_BUSY, RXNE IT 永远不使能.
 *   - 修复: 在 DMA-RX IDLE 回调里检测 1 字节 'D' 帧, 置 cmd_rx_ready.
 *     PC 端发 'D' 单字节 + 1 char time 静默 (≈87us @ 115200) → IDLE flag 触发 →
 *     HAL_UARTEx_RxEventCallback(Size=1) → 我们设 flag + re-arm DMA-RX.
 *     main.c while(1) 看到 cmd_rx_ready, DMA-TX g_last_ch32_input (1472B).
 */
extern volatile uint8_t cmd_rx_byte;
extern volatile uint8_t cmd_rx_ready;
void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t Size)
{
  if (huart->Instance == USART1) {
    g_idle_count++;
    g_last_size = Size;

    /* 1. Cache invalidate 整段 buffer (D-Cache 行 32B;用 Size 可能漏覆盖尾部,
     *    导致 modbus_rtu_parse 读 buf[len-1]/buf[len-2] 时拿到上一帧残留)
     *    干脆 invalidate 整个 256B buffer,代价 ≤8 cache line,可忽略 */
    SCB_InvalidateDCache_by_Addr((uint32_t *)modbus_rx_buf,
                                 (int32_t)sizeof(modbus_rx_buf));

    /* 2. 2026-10-17 'D' 命令检测: 1 字节 'D' = sw_buf dump 请求
     *    IDLE 在 1 char time 静默后触发, Size=1, modbus_rx_buf[0]='D'
     *    不进入 IDS 处理 (会污染 frames_total 统计) */
    if (Size == 1 && modbus_rx_buf[0] == 'D') {
      printf("[D-CMD] rx 'D' Size=1, set cmd_rx_ready\r\n");
      cmd_rx_byte  = 'D';
      cmd_rx_ready = 1;
      /* 重新启动 RX DMA + 空闲检测, 准备下一帧 modbus 或 'D' */
      HAL_StatusTypeDef rc_d = HAL_UARTEx_ReceiveToIdle_DMA(&huart1, modbus_rx_buf, sizeof(modbus_rx_buf));
      __HAL_DMA_DISABLE_IT(&hdma_usart1_rx, DMA_IT_HT);
      if (rc_d != HAL_OK) {
        g_rerx_fail_count++; g_last_rerx_rc = (uint32_t)rc_d;
        g_rerx_need_recovery = 1;   /* ⭐ ISR 内禁止 abort, 交给 main loop watchdog */
        printf("[IDLE-D] Re-arm FAILED rc=%d, flag main loop to recover\r\n", (int)rc_d);
      }
      return;  /* 跳过 IDS 处理 */
    }

#ifdef CH32_BACKEND_TFLM
    /* 3'. 2026-10-26 TFLM 工程: 1 字节 'T' = controlled-input 自检请求.
     *     跟 'D' 一样走 IDLE 回调检测 (Size=1), 只置 flag, 不在 ISR 里推理. */
    if (Size == 1 && modbus_rx_buf[0] == 'T') {
      printf("[T-CMD] rx 'T' Size=1, set cmd_rx_ready\r\n");
      cmd_rx_byte  = 'T';
      cmd_rx_ready = 1;
      HAL_StatusTypeDef rc_t = HAL_UARTEx_ReceiveToIdle_DMA(&huart1, modbus_rx_buf, sizeof(modbus_rx_buf));
      __HAL_DMA_DISABLE_IT(&hdma_usart1_rx, DMA_IT_HT);
      if (rc_t != HAL_OK) {
        g_rerx_fail_count++; g_last_rerx_rc = (uint32_t)rc_t;
        g_rerx_need_recovery = 1;
        printf("[IDLE-T] Re-arm FAILED rc=%d, flag main loop to recover\r\n", (int)rc_t);
      }
      return;  /* 跳过 IDS 处理 */
    }
#endif

    /* 3. 喂给 IDS 处理状态机 */
    modbus_ids_on_frame(modbus_rx_buf, Size);
		#ifdef LED
		HAL_GPIO_TogglePin(GPIOB,GPIO_PIN_0);           //green
		#endif
    /* 4. ⭐ 2026-10-17 修复: 重新启动 RX DMA + 空闲检测 + 返回值处理
     *    现象: IDLE 回调触发几帧后, HAL_UARTEx_ReceiveToIdle_DMA 返回 HAL_BUSY,
     *           IDLE 中断不再 enable, USART RX 永久挂掉 (绿色 LED 不再闪).
     *    原因: DMA NORMAL 模式下 HAL_IDLE handler 走 line 2399-2406 把
     *           huart->RxState 设回 READY, 但某些 race condition 下 (例如
     *           HAL_DMA_Abort 没完全结束) 第二次 ReceiveToIdle_DMA 仍返回 HAL_BUSY.
     *    ⭐⭐ 关键修正: ISR 内禁止调 HAL_UART_AbortReceive (阻塞等 DMA 停止
     *         会卡死 NVIC + 整个系统). 改为置 flag, main loop watchdog 处理.
     *    实际表现: 13 alarm 后挂掉 = ISR 内 abort 半完成, RxState 卡死. */
    HAL_StatusTypeDef rc = HAL_UARTEx_ReceiveToIdle_DMA(&huart1, modbus_rx_buf, sizeof(modbus_rx_buf));
    __HAL_DMA_DISABLE_IT(&hdma_usart1_rx, DMA_IT_HT);  /* 禁止 half-transfer 二次回调 */
    if (rc != HAL_OK) {
      g_rerx_fail_count++;
      g_last_rerx_rc = (uint32_t)rc;
      g_rerx_need_recovery = 1;       /* main loop watchdog 会处理 */
      printf("[IDLE] #%u Re-arm FAILED rc=%d Size=%u, flag main loop to recover\r\n",
             (unsigned)g_idle_count, (int)rc, (unsigned)Size);
    }
  }
}
/* USER CODE END 1 */
