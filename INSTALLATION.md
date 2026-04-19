# Installation Guide — STM32 Block IDE

This guide walks you through everything you need to install before you can open the IDE, place blocks, and click "Run on Board" to flash an STM32.

There are three layers of dependencies:

1. The Python application itself (the IDE window)
2. The ARM cross-compiler (turns the generated C code into firmware)
3. The flashing tool (writes firmware to the board over USB)

Allow about 20–30 minutes for a clean install on a fresh machine.

---

## 1. Python application

You need Python 3.9 or newer. Check with:

```
python --version
```

Then install the runtime libraries used by the IDE:

```
pip install PyQt5 pyqtgraph pyserial numpy
```

PyQt5 powers the windowing and the block diagram canvas. pyqtgraph draws the live oscilloscope plots. pyserial reads samples coming back from the microcontroller over the ST-Link's USB virtual COM port. numpy is used by the host-side simulator and is also exposed inside the Python workspace tab.

Verify the install by launching the IDE from the project folder:

```
python stm32_block_ide.py
```

You should see a window titled "STM32 Block IDE 0.1.0" with a small example model already laid out (a Square Wave block driving the on-board LED). If that comes up, the GUI half of the install is done.

## 2. ARM cross-compiler (arm-none-eabi-gcc)

The IDE generates C source code, but you need a compiler that can produce ARM Cortex-M4 instructions. Install the GNU Arm Embedded Toolchain.

On Ubuntu / Debian:

```
sudo apt install gcc-arm-none-eabi binutils-arm-none-eabi libnewlib-arm-none-eabi
```

On macOS with Homebrew:

```
brew install --cask gcc-arm-embedded
```

On Windows, download the installer from arm.com (search for "GNU Arm Embedded Toolchain") and tick the box "Add path to environment variable" during install.

Verify with:

```
arm-none-eabi-gcc --version
```

You also need `make`. Linux and macOS ship with it; on Windows install MSYS2 or use the `make` that comes with the Arm toolchain on some distributions, or install `mingw32-make` and alias it.

## 3. STM32 HAL drivers (CubeF4)

The generated firmware uses ST's HAL library. Download STM32CubeF4 from st.com (the package is roughly 800 MB) and unzip it somewhere stable, for example:

```
$HOME/STM32Cube/Repository/STM32Cube_FW_F4_V1.27.1
```

The generated Makefile looks for it at that exact path. If you put it elsewhere, set the environment variable before building:

```
export CUBE_F4=/path/to/STM32Cube_FW_F4_V1.27.1
```

You don't need STM32CubeIDE itself — only the firmware package.

## 4. Flashing tool (st-flash)

To get the binary onto the board over USB you need `st-flash` from the stlink-tools project.

On Ubuntu / Debian:

```
sudo apt install stlink-tools
```

On macOS:

```
brew install stlink
```

On Windows, download the latest `stlink-w64-*.zip` release from the stlink-org/stlink GitHub page, extract it, and put the `bin` folder on your PATH.

Verify by plugging in your Nucleo board (USB cable into the ST-Link end, the one near the Ethernet-jack-looking debug header) and running:

```
st-info --probe
```

You should see one programmer found, with a serial number, and a chip ID matching the F446RE.

### Linux udev permissions

On Linux you'll get "permission denied" until you install the udev rules that come with stlink-tools. Most distribution packages do this for you automatically; if not, grab the `49-stlinkv2-1.rules` file from the stlink repo, drop it in `/etc/udev/rules.d/`, and run `sudo udevadm control --reload-rules && sudo udevadm trigger`. Then unplug and replug the board.

## 5. (Optional) Live serial scope

The Scope block can stream sample data back to the IDE over the ST-Link's virtual COM port. To use this, find out what device name the OS gave it:

- Linux: `/dev/ttyACM0` (or ACM1, ACM2…)
- macOS: `/dev/tty.usbmodemXXXX`
- Windows: `COM3` or similar — check Device Manager

In the IDE, switch to the "Scope / Serial" tab, pick the port from the dropdown, and click Connect.

---

## Quick sanity check

Once everything is installed, the fastest way to confirm the full chain works is:

```
python stm32_block_ide.py
```

then click the orange ▶ Run on Board button on the toolbar. The build log tab should show `arm-none-eabi-gcc` compiling, `arm-none-eabi-size` printing the section sizes, and finally `st-flash write …` finishing with "Flash written and verified! jolly good!". If you see all three, you're ready to write your first block diagram. Continue with `GETTING_STARTED.md`.
