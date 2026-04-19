# Auto-generated Makefile for STM32 Block IDE.
# Requires:
#   - arm-none-eabi-gcc (GNU Arm Embedded Toolchain)
#   - STM32CubeF4 firmware package (HAL drivers + CMSIS)
#   - st-flash (from stlink-tools), only for `make flash`
#
# Set CUBE_F4 to the root of your STM32CubeF4 install, e.g.:
#   Windows cmd:  set CUBE_F4=C:/STM32Cube/STM32Cube_FW_F4_V1.28.0
#   bash:         export CUBE_F4=$HOME/STM32Cube/Repository/STM32Cube_FW_F4_V1.28.0

PROJECT  = blockide_app
TARGET   = $(PROJECT).elf
BIN      = $(PROJECT).bin

PREFIX   = arm-none-eabi-
CC       = $(PREFIX)gcc
AS       = $(PREFIX)gcc -x assembler-with-cpp
OBJCOPY  = $(PREFIX)objcopy
SIZE     = $(PREFIX)size

ifeq ($(strip $(CUBE_F4)),)
  $(error CUBE_F4 is not set. Point it at your STM32CubeF4 install, e.g. $$HOME/STM32Cube/Repository/STM32Cube_FW_F4_V1.28.0)
endif

HAL_DIR = $(CUBE_F4)/Drivers/STM32F4xx_HAL_Driver
CMSIS   = $(CUBE_F4)/Drivers/CMSIS

ARCH    = -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -DSTM32F446xx -DUSE_HAL_DRIVER
CFLAGS  = $(ARCH) -O2 -Wall -ffunction-sections -fdata-sections \
          -I. \
          -I$(CMSIS)/Include \
          -I$(CMSIS)/Device/ST/STM32F4xx/Include \
          -I$(HAL_DIR)/Inc
ASFLAGS = $(ARCH)
LDSCRIPT = STM32F446RETx_FLASH.ld
LDFLAGS  = $(ARCH) -T$(LDSCRIPT) -Wl,--gc-sections -specs=nano.specs -specs=nosys.specs

# CubeF4 ships startup_stm32f446xx.s for gcc and system_stm32f4xx.c as a
# template. We reference them directly from the package so the generated
# project needs no extra file copies.
STARTUP = $(CMSIS)/Device/ST/STM32F4xx/Source/Templates/gcc/startup_stm32f446xx.s
SYSTEM  = $(CMSIS)/Device/ST/STM32F4xx/Source/Templates/system_stm32f4xx.c

SRCS_C = main.c \
         $(SYSTEM) \
         $(HAL_DIR)/Src/stm32f4xx_hal.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_cortex.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_rcc.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_rcc_ex.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_pwr.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_pwr_ex.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_gpio.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_dma.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_dma_ex.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_flash.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_flash_ex.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_flash_ramfunc.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_uart.c

all: $(BIN)

$(TARGET): $(SRCS_C) $(STARTUP)
	$(CC) $(CFLAGS) $(LDFLAGS) $(SRCS_C) $(STARTUP) -o $@
	$(SIZE) $@

$(BIN): $(TARGET)
	$(OBJCOPY) -O binary $< $@

flash: $(BIN)
	st-flash write $(BIN) 0x8000000

clean:
	rm -f *.elf *.bin *.hex *.o

.PHONY: all flash clean
