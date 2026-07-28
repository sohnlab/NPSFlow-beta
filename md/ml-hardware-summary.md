# ML Model & Hardware Selection Summary

## Goal
Deploy a real-time pulse detection + classification system for NPS (Node Pore Sensing) data, with particle sorting capability via piezo pumps.

## Classification Task
- **Input**: Streaming NPS signal (10–50 kHz sample rate, 14-bit)
- **Output**: Per-pulse classification — 4 classes:
  - `0` = single, `1` = coincident, `2` = noise, `3` = uncertain
- **Constraint**: Classification decision must be made by the end of each pulse (real-time)

## Selected Model: Small 1D CNN
- **Architecture**: 3–4 convolutional layers, < 50K parameters
- **Input**: Fixed-length resampled pulse window (e.g., 256 or 512 samples), normalized
- **Output**: 4-class softmax
- **Quantization**: INT8 for TFLite deployment
- **Why chosen**:
  - Fast inference (~1–10 ms per pulse on ARM)
  - Learns pulse shape features directly from waveform — no manual feature engineering
  - Works well with small-to-medium training sets (hundreds to low thousands of labeled pulses)
  - Easily quantizable for edge deployment

### Alternatives Considered
| Model | Verdict |
|-------|---------|
| LSTM/GRU | Slower inference, overkill for fixed-length windows |
| TCN (Temporal Convolutional Network) | Good but more complex; consider if causal streaming needed later |
| Random Forest on hand-crafted features | Fast but loses shape information |

## Selected Hardware: Red Pitaya STEMlab 125-14 Gen 2

### Specs
- **SoC**: Xilinx Zynq 7020
  - Dual-core ARM Cortex-A9 @ 866 MHz
  - FPGA: 85K logic cells, 220 DSP slices
- **RAM**: 512 MB DDR3
- **ADC**: 2 channels, 14-bit, 125 MSPS
- **DAC**: 2 channels, 14-bit, 125 MSPS
- **I/O**: I2C, SPI, 16 digital GPIO on extension connector (E2)

### Pipeline Architecture (Single Board)
```
Signal In → ADC (14-bit 125 MSPS)
         → FPGA: threshold-based pulse detection (~us latency)
         → ARM: TFLite INT8 CNN classification (~1-10 ms)
         → FPGA: trigger DAC/GPIO for sorting
         → DAC → piezo driver amplifier → piezo pump
```

Total decision latency: ~1–10 ms from end of pulse to actuation trigger.
Typical microfluidic particle transit time (sensing to sorting junction): 10–100 ms — latency is acceptable.

### DAC Output Note
Red Pitaya DAC outputs +/-1V. Piezo pumps typically need 10–200V. An external high-voltage amplifier stage is required between DAC and piezo.

## ADC Expansion: AD7606C-16
- **Why needed**: Red Pitaya has only 2 ADC channels; project needs at least 7 extra
- **Specs**: 16-bit, 8 channels, simultaneous sampling, up to 1 MSPS/ch
- **Interface**: SPI (connects to Red Pitaya E2 extension connector)
- **Input range**: Bipolar +/-10V / +/-5V / +/-2.5V (configurable)
- **Cost**: ~$15–20/chip, eval boards available
- **Key advantage**: Simultaneous sampling — no phase skew between channels

## Alternatives Ruled Out

### ESP32 / Arduino
- **Inference**: Feasible on ESP32-S3 (~1–10 ms/pulse with TFLite Micro)
- **Deal-breaker**: ADC quality — ESP32 ADC is 12-bit ~1 MSPS but noisy (~9-bit effective). Arduino Uno ADC is 10-bit ~10 kSPS. Neither matches Red Pitaya's 14-bit 125 MSPS.
- **Verdict**: No benefit over Red Pitaya's built-in ARM cores; adds complexity without improving signal quality.

### Digilent Arty S7 (Spartan-7 FPGA)
- **FPGA**: XC7S50 — 52K logic cells, 120 DSP slices (fewer than Zynq 7020)
- **No ARM cores**: No Linux, no TFLite. ML inference requires soft-core (MicroBlaze) or pure HLS — major development effort.
- **No built-in ADC/DAC**: Need external modules for everything Red Pitaya includes.
- **Verdict**: Rebuilds everything Red Pitaya already provides, with less capability.

### ADS1115 (I2C ADC)
- 16-bit, 4 channels, but only 860 SPS max — far too slow for NPS signals.
- Could work for slow auxiliary sensors, but not needed in this project.

## Training Data Strategy
- **Source**: NPSflow Detection Review block outputs (classified pulse regions + raw signal)
- **Format**: CSV — one row per pulse, columns: `label, sample_0, sample_1, ..., sample_N`
- **Sidecar metadata**: `_meta.json` with source file, sample rate, window size, normalization method, export date
- **Workflow**: Export multiple CSVs from different experiments → concatenate → train 1D CNN → quantize to INT8 → deploy `.tflite` to Red Pitaya ARM
