# Getting Started — Blink the On-Board LED

This walkthrough takes you from a freshly installed IDE to a blinking LED on the NUCLEO-F446RE. It covers placing blocks, wiring them, setting parameters, using a workspace variable for the blink rate, and flashing the result.

If you haven't installed the prerequisites yet, do that first using `INSTALLATION.md`.

## What you'll build

A two-block diagram: a Square Wave generator drives a GPIO output that maps to pin PA5 — the green user LED on the Nucleo board. A third Scope block lets you confirm in software what waveform is being produced before you commit it to the hardware.

Total time: about five minutes.

## Step 1 — Launch the IDE and pick a board

Run the application:

```
python stm32_block_ide.py
```

The toolbar at the top has a "Board" dropdown. It should already show NUCLEO-F446RE, which is the only board currently supported. If you ever connect a different Nucleo, you'd switch it here before generating code, because the pin map and clock setup are board-specific.

The IDE opens with an example diagram pre-loaded so you have something to look at — feel free to delete those blocks (click them and press Delete) so you can build your own from scratch.

## Step 2 — Place blocks from the palette

The left side of the Block Diagram tab is the block palette. Drag three blocks onto the canvas:

- A "Square Wave" block on the left
- A "GPIO Out" block in the middle
- A "Scope" block on the right

You can reposition any block by dragging it. Zoom with the mouse wheel.

## Step 3 — Wire the blocks

Each block has small square ports on its sides — outputs on the right, inputs on the left. Click and drag from an output port to an input port to draw a wire.

Make these two connections:

- Square Wave's `y` output → GPIO Out's `u` input
- Square Wave's `y` output → Scope's `u0` input

A single output can fan out to multiple inputs. The Scope is essentially a passive observer.

## Step 4 — Configure the GPIO Out block

Click the GPIO Out block. The right panel shows its parameters. Set:

- `pin` to `PA5`. This is the green LED next to the reset button on the Nucleo.
- `threshold` to `0.5`. Anything above 0.5 will drive the pin high.

The pin name format is the chip notation (`P` + port letter + pin number). All pins from PA0 through PH15 are accepted; the IDE will only enable the GPIO clock for ports actually used.

## Step 5 — Set the blink rate using a workspace variable

This is where the IDE differs from a fixed-parameter block diagram tool. Switch to the Python Workspace tab and type:

```python
blink_hz = 2.0
```

Press Enter. You'll see the variable appear in the workspace listing at the bottom.

Now go back to the Block Diagram tab, click the Square Wave block, and change its `frequency_hz` parameter from `1.0` to `blink_hz`. The parameter accepts any Python expression — `blink_hz`, `2 * blink_hz`, `1/period_s`, even `np.pi` — and the value is resolved at code generation time. This is the same pattern Simulink uses with its base workspace, except the workspace here is a real Python interpreter.

Leave the other parameters at their defaults (amplitude 1.0, offset 0.0, duty 0.5).

## Step 6 — Try it in software first

Before you flash anything, you can verify the model by running it on the host. Click the "Simulate" button on the toolbar. The IDE switches to the Scope tab and plots the waveform that the model would produce — a 2 Hz square wave, between 0 and 1, alternating high and low.

If you tweak `blink_hz` in the workspace and click Simulate again, you'll see the rate change. You haven't touched any C code or hardware yet.

## Step 7 — Flash to the board

Plug the Nucleo into your computer with a USB cable (the connector near the Ethernet-jack-looking debug header, not the user side). The red `LD3` power LED on the board should come on.

Click the ▶ Run on Board button on the toolbar.

The IDE switches to the Build Log tab and shows three things in sequence:

1. The path of the freshly generated project, plus a log line for each `arm-none-eabi-gcc` invocation.
2. The output of `arm-none-eabi-size` — typically a few KB of code, well under 1% of the F446RE's 512 KB Flash.
3. `st-flash write …` finishing with a "verified" message.

A second or two later, the green LED next to the reset button starts blinking at 2 Hz. Done.

## Step 8 — See the live signal from the board

If you'd like to watch the waveform the MCU is actually producing (as opposed to the host simulation), switch to the Scope / Serial tab, pick the ST-Link's COM port from the dropdown, and click Connect. Because the Scope block has its `stream` parameter set to 1, the firmware sends each sample over USART2 (which is wired to the ST-Link's USB virtual COM port on every Nucleo). You should see the same square wave updating in real time.

## What to try next

Change `blink_hz` in the workspace to `0.5`, click ▶ again — the LED slows to one blink every two seconds. The whole round-trip from edit to flashed firmware takes a few seconds.

Add a GPIO In block tied to `PC13` (the user button on the F446RE) and use it to gate something visible on the scope. The button is active-low, which is the default in the GPIO In block's `active_low` parameter.

Save the model with File → Save Model so you can come back to it later. The on-disk format is plain JSON, so you can also commit it to version control alongside your firmware.

When something doesn't work, the Build Log tab is the first place to look. Most failures are either "arm-none-eabi-gcc not found" (revisit the toolchain section of `INSTALLATION.md`) or "st-flash: failed to open" (board not plugged in, or udev rules missing on Linux).
