# ============================================================================
# makefile  —  V19 MCU Inference 编译脚本
# ============================================================================
# 用法:
#   make pc            # PC 上编译运行 (需要 gcc, 跑 demo)
#   make stm32         # STM32 (Cortex-M4) 交叉编译
#   make esp32         # ESP32 交叉编译 (需要 ESP-IDF toolchain)
#   make clean         # 清理产物
# ============================================================================

CC          := gcc
CFLAGS_PC   := -O2 -std=c99 -ffast-math -funroll-loops -Wall -Wextra -DNDEBUG
LIBS_PC     := -lm

# STM32 交叉编译工具链 (常见路径, 按你的实际环境调整)
ARM_CC      := arm-none-eabi-gcc
CFLAGS_STM  := -O2 -std=c99 -ffast-math -funroll-loops -mcpu=cortex-m4 \
               -mfloat-abi=hard -mfpu=fpv4-sp-d16 -DNDEBUG \
               -ffunction-sections -fdata-sections
# 注意: -lm 在 newlib-nano 里通常需要加 -specs=nano.specs

# ESP32 工具链由 ESP-IDF 管理,这里只给参考命令
ESP32_CC    := xtensa-esp32-elf-gcc
CFLAGS_ESP  := -O2 -std=c99 -ffast-math -mlongcalls

SRCS        := main.c v19_inference.c
HDRS        := v19_inference.h model_v19_hybrid.h

.PHONY: pc stm32 esp32 clean

# ── PC (跑 demo) ───────────────────────────────────────────────────────
pc: v19_demo
	./v19_demo

v19_demo: $(SRCS) $(HDRS)
	$(CC) $(CFLAGS_PC) -o $@ $(SRCS) $(LIBS_PC)

# ── STM32 ──────────────────────────────────────────────────────────────
stm32: v19_stm32.elf

v19_stm32.elf: $(SRCS) $(HDRS)
	$(ARM_CC) $(CFLAGS_STM) -o $@ $(SRCS) -lm \
		-specs=nano.specs -specs=nosys.specs \
		-Wl,--gc-sections -Wl,-Map=v19_stm32.map
	@echo ""
	@echo "编译产物大小:"
	@arm-none-eabi-size v19_stm32.elf

# ── ESP32 (参考,需 ESP-IDF 环境) ───────────────────────────────────────
esp32:
	@echo "ESP32 推荐用 ESP-IDF (CMake),命令参考:"
	@echo "  idf.py create-project && cd v19_esp32"
	@echo "  cp /path/to/{main.c,v19_inference.c,v19_inference.h,model_v19_hybrid.h} main/"
	@echo "  idf.py set-target esp32"
	@echo "  idf.py build flash monitor"

# ── 清理 ───────────────────────────────────────────────────────────────
clean:
	rm -f v19_demo v19_stm32.elf v19_stm32.map
