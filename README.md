# Python Wavetable Synthesis Library

A simple but powerful Python library for additive synthesis and phase modulation (which is commonly called frequency modulation or FM), with support for exporting both audio and wavetables. Compatible with Xfer Serum, Ableton Wavetable and other wavetable synthesizers.

## Installation

### Clone and Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/patdemichele/python-wavetable-synthesis.git
cd python-wavetable-synthesis

# Activate the included virtual environment
source wavetable_env/bin/activate  # On macOS/Linux
# OR
wavetable_env\Scripts\activate     # On Windows

# Install dependencies (if needed)
pip install -r requirements.txt

# Run examples. See generated/ for the output wav files
python example.py
```

The repository includes a pre-configured virtual environment with all dependencies installed. Simply clone and activate to get started immediately.

## Quick Start -- Static Wavetable

The library is based on additive synthesis -- creating waves by adding harmonics. Here's an example of a static wavetable with 3 distinct harmonics.

```python
from wavetable import *

# Create basic wave
fundamental = H(1)                    # 1st harmonic (fundamental)
second_harmonic = H(2, 0.5)          # 2nd harmonic at half amplitude
# 5th harmonic at 1/10 amplitude (Note the * operator scales amplitude)
fifth_harmonic = 0.1 * H(5)
my_wave = fundamental + second_harmonic + fifth_harmonic  # Additive synthesis

# Export as wavetable for Serum (automatically normalized)
Render(my_wave, "wavetable", "my_wavetable.wav")

# Generate playable audio (automatically normalized)
Render(my_wave, "audio", "audio.wav", note="A4", duration=2.0)
```

## Core Features

### WaveSegment
The core data structure is the `WaveSegment`, which represents either a
- **Static Wave** (length=0): Single waveform with harmonics
- **Morphing Segment** (length>0): An evolving waveform over time

A static wave is constructed as the sum of frequencies, and the morphing segments are built up from static waves or other morphing segments. Segments can be scaled by amplitude or length. A set of segments can be concatenated (over time). We can perform addition and phase modulation between segments of the same length. The `Render` function exports segments to wavetables, audio files, or arrays for analysis.

### Function Spec

#### Core Additive Synthesis
- `H(n: int, amplitude: float = 1.0) -> WaveSegment` - Create nth harmonic
- `Segment(start: WaveSegment, end: WaveSegment, length: float) -> WaveSegment` - Create morphing segment
- `wave1 + wave2` - Add two WaveSegments (combines harmonics or morphs compatible segments)
- `scalar * wave` or `wave * scalar` - Scale WaveSegment amplitude by scalar value
- `Cat(*segments: WaveSegment) -> WaveSegment` - Concatenate segments
- `PM(carrier: WaveSegment, modulator: WaveSegment, amount: float) -> WaveSegment` - Phase modulation

#### Utilities
- `N(wave: WaveSegment) -> WaveSegment` - If non-zero, normalize to maximum absolute amplitude of 1.0 (automatic in Render)
- `SetLength(wave: WaveSegment, length: float) -> WaveSegment` - Set segment length
- `Clip(wave: WaveSegment, min_val: float = -1.0, max_val: float = 1.0) -> WaveSegment` - Clip amplitude to specified range
- `Center(wave: WaveSegment) -> WaveSegment` - Remove DC offset by centering wave at average value
- `DC(offset: float) -> WaveSegment` - Create DC (constant) offset wave (-1.0 to 1.0)

#### Rendering
- `Render(wave: WaveSegment, output_type: str = "array", **kwargs) -> np.ndarray` - Unified rendering (wavetable/audio/array)
- `Visualize(wave: WaveSegment, frame_index: int = 0, domain: str = "time") -> tuple` - Plot time/frequency domain
- `AnalyzeFrame(wave: WaveSegment, frame_index: int = 0) -> dict` - Detailed frame analysis
- `print(wave_segment)` - Display wave structure as tree

**Important**: Length parameters are abstract units until rendered. A `Segment(start, end, 8.0)` with length 8.0 represents relative timing - the actual duration depends on the total wavetable frames or audio duration when exported.

## Examples

### Basic Harmonics
```python
# Individual harmonics
fundamental = H(1)        # F0
second = H(2)            # F0 * 2
third = H(3, 0.5)        # F0 * 3 at half amplitude

# DC offset (constant value)
dc_offset = DC(0.1)       # Constant +0.1 offset
wave_with_offset = H(1) + DC(0.2)  # Sine wave with DC offset

# Remove DC offset
centered_wave = Center(wave_with_offset)  # Removes the DC(0.2) component

# Amplitude clipping
loud_wave = 2.0 * H(1)    # Wave with amplitude > 1.0
clipped_wave = Clip(loud_wave, -0.8, 0.8)  # Clip to ±0.8 range

# Additive synthesis - using sum() with proper start value
sawtooth = sum((H(i, 1.0/i) for i in range(1, 16)), H(0, 0.0))
square_approx = sum((H(i, 1.0/i) for i in range(1, 16, 2)), H(0, 0.0))  # Odd harmonics
```

### Phase Modulation
```python
# Basic PM
pm_basic = PM(H(1), H(2), 0.5)

# Complex PM
carrier = H(1) + 0.3*H(2)
modulator = H(3)
pm_complex = PM(carrier, modulator, 0.8)

# Nested PM
inner = PM(H(1), H(2), 0.3)
outer = PM(inner, H(4), 0.6)
```

### Morphing and Segments
```python
# Morph from fundamental to 5th harmonic over 8 abstract units
morph = Segment(H(1), H(5), 8.0)

# Complex morphing
rich_wave = H(1) + 0.5*H(2) + 0.3*H(3)
simple_wave = H(1)
evolution = Segment(rich_wave, simple_wave, 16.0)
```

### ADSR Envelope Example
```python
# Attack-Decay-Sustain-Release envelope using morphing
# Note: Length parameters are abstract until wavetable rendering

# Attack: silence to full volume (2 units)
attack = Segment(Zero(), H(1), 2.0)

# Decay: full volume to sustain level (1 unit)
decay = Segment(H(1), 0.5*H(1), 1.0)

# Sustain: hold sustain level (4 units)
sustain = SetLength(0.5*H(1), 4.0)

# Release: sustain to silence (3 units)
release = Segment(0.5*H(1), Zero(), 3.0)

# Combine into full ADSR envelope
adsr_envelope = Cat(attack, decay, sustain, release)

# Export as wavetable - the abstract lengths become actual frame timing
Render(adsr_envelope, "wavetable", "adsr_wavetable.wav")

# Or as audio with specific duration - abstract lengths scale to fit
Render(adsr_envelope, "audio", "adsr_audio.wav", note="C4", duration=3.0)
```

### Concatenation
```python
# Different segments with abstract lengths
seg1 = SetLength(H(1), 4.0)
seg2 = SetLength(H(2), 4.0)
seg3 = SetLength(PM(H(1), H(3), 0.5), 4.0)

# Join them
complete = Cat(seg1, seg2, seg3)
```

## Enhanced Mathematical Operations

```python
# Addition (combines harmonics)
combined = H(1) + H(2) + 0.5*H(3)

# Scalar multiplication (works with all segment types)
scaled_wave = 0.5 * H(1)
scaled_segment = 0.7 * Segment(H(1), H(3), 8.0)
scaled_pm = 0.8 * PM(H(1), H(2), 0.5)

# Addition of segments (auto-handles different lengths)
static_wave = H(2) + 0.3*H(4)
morph_wave = Segment(H(1), H(3), 10.0)
combined = static_wave + morph_wave  # Static expands to match morph length

# Same-length segment addition
morph1 = Segment(H(1), H(2), 8.0)
morph2 = Segment(H(3), H(4), 8.0)
combined_morph = morph1 + morph2

# Automatic normalization (no need for explicit N() calls)
# Render() automatically normalizes to prevent clipping
```

## Unified Render Function

The `Render()` function is the recommended way to output wavetables, audio, or arrays for analysis:

```python
# Array mode (for debugging and analysis)
array_data = Render(wave_segment)  # Returns numpy array
print(f"Shape: {array_data.shape}, RMS: {np.sqrt(np.mean(array_data**2))}")

# Wavetable export
wavetable_array = Render(wave_segment, "wavetable", "output.wav")
# Custom format for Ableton Live
Render(wave_segment, "wavetable", "ableton.wav", samples_per_frame=ABLETON_SAMPLES_PER_FRAME)

# Audio export
audio_array = Render(wave_segment, "audio", "audio.wav", note="A4", duration=2.0)
# Using frequency instead of note
Render(wave_segment, "audio", "audio.wav", frequency=440, duration=1.5)

# All modes return numpy arrays for further analysis
print(f"Audio length: {len(audio_array)} samples")
print(f"Wavetable shape: {wavetable_array.shape}")  # [frames, samples_per_frame]
```

### Render Parameters

- **output_type**: `"array"` (default), `"wavetable"`, or `"audio"`
- **filename**: Output file (required for wavetable/audio modes)
- **frames**: Wavetable frame count (default 256)
- **samples_per_frame**: `SERUM_SAMPLES_PER_FRAME` (2048) or `ABLETON_SAMPLES_PER_FRAME` (1024)
- **frame_index**: Extract specific frame (0 to frames-1). If None, returns all frames
- **frequency/note**: Audio frequency (use either one)
- **duration**: Audio duration in seconds
- **normalize**: Auto-normalize output (default True)

## Complete Rendering Examples

### Render() Function - The Universal Interface

The `Render()` function provides four powerful modes for working with wavetables:

```python
# Create an interesting test wave
carrier = Segment(H(1) + 0.3*H(2), H(1) + 0.5*H(4), 8.0)
test_wave = PM(carrier, H(3) + 0.2*H(5), 0.6)

# 1. Array Mode (default) - Perfect for debugging
array_data = Render(test_wave)
print(f"Shape: {array_data.shape}")           # (8, 2048)
print(f"Frame 0 RMS: {np.sqrt(np.mean(array_data[0]**2)):.3f}")  # 0.626

# 2. Wavetable Mode - Export for synthesizers
wavetable_data = Render(test_wave, "wavetable", "my_wavetable.wav")
print(f"Wavetable shape: {wavetable_data.shape}")  # (256, 2048)
print(f"File size: {os.path.getsize('my_wavetable.wav')} bytes")  # 1048620

# 3. Audio Mode - Create playable audio
audio_data = Render(test_wave, "audio", "my_audio.wav", note="A4", duration=2.0)
print(f"Audio samples: {audio_data.shape}")    # (88200,)
print(f"Duration: {len(audio_data)/44100:.1f}s")  # 2.0s

# 4. Frame Selection - Extract specific moments
frame_128 = Render(test_wave, frame_index=128)
print(f"Single frame: {frame_128.shape}")     # (2048,)

# Export specific frame as wavetable
Render(test_wave, "wavetable", "frame_128.wav", frame_index=128)
```

### AnalyzeFrame() - Detailed Wave Analysis

Get comprehensive metrics for any frame:

```python
# Analyze evolution across frames
for frame_idx in [0, 64, 128, 192, 255]:
    analysis = AnalyzeFrame(test_wave, frame_index=frame_idx)
    print(f"Frame {frame_idx:3d}: RMS={analysis['rms']:.3f}, THD={analysis['thd']:.3f}")

# Output:
# Frame   0: RMS=0.626, THD=0.508
# Frame  64: RMS=0.591, THD=0.522
# Frame 128: RMS=0.565, THD=0.575
# Frame 192: RMS=0.550, THD=0.671
# Frame 255: RMS=0.545, THD=0.812

# Detailed analysis of a specific frame
detailed = AnalyzeFrame(test_wave, frame_index=128)
print(f"RMS: {detailed['rms']:.4f}")                    # 0.5654
print(f"Peak: {detailed['peak']:.4f}")                  # 0.8130
print(f"DC Offset: {detailed['dc_offset']:.6f}")        # 0.000000
print(f"THD: {detailed['thd']:.4f}")                    # 0.5748
print(f"Fundamental: {detailed['fundamental_freq']:.4f}") # 0.6932

# Harmonic content analysis
for harmonic, value in list(detailed['harmonics'].items())[:5]:
    print(f"{harmonic}: {value:.4f}")
# Output:
# DC: 0.0000
# H1: 0.6932  (fundamental)
# H2: 0.2383  (second harmonic)
# H3: 0.0008  (weak third)
# H4: 0.3153  (strong fourth)
```

### Visualize() - Time & Frequency Domain Plots

Generate publication-quality plots (requires `pip install matplotlib`):

```python
# Time domain visualization
fig_time, axes = Visualize(test_wave, frame_index=128, domain="time")

# Frequency domain visualization
fig_freq, axes = Visualize(test_wave, frame_index=128, domain="frequency")

# Combined time and frequency analysis
fig_both, axes = Visualize(test_wave, frame_index=128, domain="both", show=False)
fig_both.savefig("wave_analysis.png", dpi=150, bbox_inches="tight")
```

![Wavetable Frame Analysis](generated/visualization_combined.png)

*Example visualization showing both time domain waveform and frequency domain harmonics for a PM-modulated morphing segment*

### print() - Wave Structure Inspection

Understand how complex waves are built:

```python
# Simple additive wave
simple = H(1) + 0.5*H(3) + DC(0.1)
print(simple)
# Output: (DC(0.100) + H(1) + 0.500*H(3))

# Complex nested structure
print(test_wave)
# Output:
# PM(amount=0.60,
#   carrier: Segment(length=8.0,
#     start: (H(1) + 0.300*H(2))
#     end:   (H(1) + 0.500*H(4))
#   )
#   modulator: (H(3) + 0.200*H(5))
# )

# Concatenated segments with tree structure
seg1 = SetLength(H(1), 2.0)
seg2 = Segment(H(2), H(4), 3.0)
seg3 = PM(H(1), H(3), 0.5)
concat = Cat(seg1, seg2, seg3)
print(concat)
# Output:
# Cat(
#   ├─ Segment(length=2.0, start: H(1), end: H(1))
#   ├─ Segment(length=3.0, start: H(2), end: H(4))
#   └─ PM(amount=0.50, carrier: H(1), modulator: H(3))
# )
```

## Configuration

The library uses configurable constants for easy customization:

```python
# In wavetable.py - synthesizer-specific constants
SERUM_SAMPLES_PER_FRAME = 2048     # Xfer Serum wavetable frame size
ABLETON_SAMPLES_PER_FRAME = 1024   # Ableton Live Wavetable device frame size
SAMPLES_PER_FRAME = SERUM_SAMPLES_PER_FRAME  # Default to Serum format

DEFAULT_SAMPLE_RATE = 44100   # CD quality sample rate
DEFAULT_FRAME_COUNT = 256     # Standard number of frames in wavetable
```

## Working with the Library

### Project Structure
```
python-wavetable-synthesis/
├── wavetable.py          # Core synthesis library
├── example.py            # Comprehensive examples and tests
├── requirements.txt      # Python dependencies
├── wavetable_env/        # Pre-configured virtual environment
├── generated/            # Output directory for generated files
└── README.md            # This file
```

### Development Workflow

1. **Clone and Setup** (one-time):
   ```bash
   git clone https://github.com/patdemichele/python-wavetable-synthesis.git
   cd python-wavetable-synthesis
   source wavetable_env/bin/activate
   ```

2. **Daily Usage**:
   ```bash
   # Option 1: Activate environment (recommended)
   source wavetable_env/bin/activate
   python3 example.py
   python3 your_script.py

   # Option 2: Use direct path (if activation doesn't work)
   wavetable_env/bin/python3 example.py
   wavetable_env/bin/python3 your_script.py
   ```

3. **Creating Your Own Scripts**:
   Create new Python files in the project directory:
   ```python
   # your_synthesis.py
   from wavetable import *

   # Your synthesis code here
   wave = H(1) + 0.5*H(3)
   Render(wave, "wavetable", "generated/my_wavetable.wav")
   ```

### Output Files

All generated files are saved to the `generated/` directory:
- **Wavetable files** (`.wav`): Load directly into Xfer Serum or other wavetable synthesizers
- **Audio files** (`*_audio.wav`): Playable audio demonstrations of the waveforms
- The `generated/` directory is automatically created and is ignored by git

## Tips

1. **Length is abstract**: Segment lengths are relative until rendering - they scale to fit the output duration
2. **Automatic normalization**: `Render()` automatically normalizes to prevent clipping
3. **Experiment with PM amounts**: Try values from 0.1 to 2.0
4. **Layer harmonics creatively**: Combine odd/even harmonics for different timbres
5. **Use morphing for evolution**: Create evolving textures with `Segment()`
6. **Mathematical operations**: All segment types support `+` and `*` operations

## Advanced Techniques

```python
# Evolving PM amount
start_pm = PM(H(1), H(2), 0.1)
end_pm = PM(H(1), H(2), 1.5)
pm_evolution = Segment(start_pm, end_pm, 32.0)

# Complex additive with evolving harmonics
start_additive = H(1) + 0.8*H(2)
end_additive = H(1) + 0.2*H(3) + 0.6*H(5)
additive_morph = Segment(start_additive, end_additive, 16.0)

# Multi-stage evolution with varying segment lengths
stage1 = SetLength(H(1), 8.0)
stage2 = Segment(H(1), H(1) + 0.5*H(2), 8.0)
stage3 = Segment(H(1) + 0.5*H(2), PM(H(1), H(3), 1.0), 16.0)
evolution = Cat(stage1, stage2, stage3)

# Complex amplitude modulation using segment addition and scaling
base = H(1)
modulation = 0.3 * Segment(H(2), H(4), 12.0)
amplitude_modulated = base + modulation
```

## System Requirements

- **Python 3.6+**
- **NumPy** (automatically installed via requirements.txt)
- **Git** (for cloning the repository)

The repository includes a pre-configured virtual environment, so no additional setup is required beyond cloning and activating.

## Synthesizer Compatibility

The library supports different wavetable formats for maximum compatibility:

### Supported Synthesizers

| Synthesizer | Samples per Frame | Constant | Notes |
|-------------|-------------------|----------|-------|
| **Xfer Serum** | 2048 | `SERUM_SAMPLES_PER_FRAME` | 256 frames standard |
| **Ableton Live Wavetable** | 1024 | `ABLETON_SAMPLES_PER_FRAME` | Frame count flexible |
| **Custom/Other** | Any | Custom value | User-defined |

### Format Examples

```python
from wavetable import *

# Create a test wave
wave = H(1) + 0.3*H(2) + 0.1*H(5)

# Export for Serum (default format)
Render(wave, "wavetable", "serum_wavetable.wav")

# Export for Ableton Live (1024 samples per frame)
Render(wave, "wavetable", "ableton_wavetable.wav", samples_per_frame=ABLETON_SAMPLES_PER_FRAME)

# Export for some other synthesizer (64 frames, 512 samples each)
Render(wave, "wavetable", "other_synth.wav", frames=64, samples_per_frame=512)
```

**File Size Comparison**: Serum format files are exactly 2x larger than Ableton format due to the sample count difference (2048 vs 1024 samples per frame).

### Using Wavetables in Xfer Serum and Serum 2

To properly use generated wavetables in Serum:

1. **Drag and drop** the `.wav` file into an oscillator (for Serum 2, ensure it's in Wavetable mode). 
2. When several options appear, make sure to drag the file into the **"Constant Framesize [Pitch Average]"** box. This ensures Serum treats each frame as a separate wavetable position
3. The wavetable position can then be modulated or automated

**⚠️ Warning**: Because Serum auto-detects the pitch, this may cause the framesize to be interpreted as something other than 2048, leading to misaligned wavetables. If this happens, you may need to experiment with the other wavetable import mechanisms or modify your wavetable to have a more prominent first harmonic (`H(1)`).

### Using Wavetables in Ableton

Ableton's Wavetable synth uses a constant framesize of 1024, which makes this utility perfect, as long as you remember to export with the correct framesize.
