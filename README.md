# Python Wavetable Synthesis Library

A simple but powerful Python library for additive synthesis and phase modulation (which is common called frequency modulation or FM), with support for exporting both audio and wavetables. Compatible with Xfer Serum, Ableton Wavetable and other wavetable synthesizers.

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

## Quick Start

Here's an extra simple example of the usage. For a more robust series of examples, see `example.py`.

```python
from wavetable import *

# Create basic waves
fundamental = H(1)                    # 1st harmonic (fundamental)
second_harmonic = H(2, 0.5)          # 2nd harmonic at half amplitude
complex_wave = H(1) + 0.3*H(3) + 0.1*H(5)  # Additive synthesis

# Export as wavetable for Serum (automatically normalized)
Wavetable(complex_wave, "my_wavetable.wav")

# Generate playable audio (automatically normalized)
Play(complex_wave, "audio.wav", note="A4", duration=2.0)
```

## Core Concepts

### WaveSegment
The fundamental building block representing either:
- **Static Wave** (length=0): Single waveform with harmonics
- **Morphing Segment**: Transition between two waves over time

**Important**: Length parameters are abstract units until rendered. A `Segment(start, end, 8.0)` with length 8.0 represents relative timing - the actual duration depends on the total wavetable frames or audio duration when exported.

### Key Functions

- `H(n, amplitude=1.0)` - Create nth harmonic
- `PM(carrier, modulator, amount)` - Phase modulation
- `Segment(start, end, length)` - Create morphing segment (length is abstract)
- `Cat(*segments)` - Concatenate segments
- `N(wave)` - Normalize to prevent clipping (automatic in Wavetable/Play)
- `Wavetable(wave, filename)` - Export wavetable (auto-normalizes)
- `Play(wave, filename, note/frequency, duration)` - Export audio (auto-normalizes)

## Examples

### Basic Harmonics
```python
# Individual harmonics
fundamental = H(1)        # F0
second = H(2)            # F0 * 2
third = H(3, 0.5)        # F0 * 3 at half amplitude

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
attack = Segment(H(0, 0.0), H(1), 2.0)

# Decay: full volume to sustain level (1 unit)
decay = Segment(H(1), 0.5*H(1), 1.0)

# Sustain: hold sustain level (4 units)
sustain = SetLength(0.5*H(1), 4.0)

# Release: sustain to silence (3 units)
release = Segment(0.5*H(1), H(0, 0.0), 3.0)

# Combine into full ADSR envelope
adsr_envelope = Cat(attack, decay, sustain, release)

# Export as wavetable - the abstract lengths become actual frame timing
Wavetable(adsr_envelope, "adsr_wavetable.wav")

# Or as audio with specific duration - abstract lengths scale to fit
Play(adsr_envelope, "adsr_audio.wav", note="C4", duration=3.0)
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
# Wavetable() and Play() automatically normalize to prevent clipping
```

## Export Options

### Wavetable Export
```python
# Basic export (256 frames, 2048 samples each - Serum format)
Wavetable(wave_segment, "output.wav")

# Export for specific synthesizers
Wavetable(wave_segment, "serum_wavetable.wav", samples_per_frame=SERUM_SAMPLES_PER_FRAME)
Wavetable(wave_segment, "ableton_wavetable.wav", samples_per_frame=ABLETON_SAMPLES_PER_FRAME)

# Custom frame count with specific format
Wavetable(wave_segment, "custom.wav", frames=128, samples_per_frame=ABLETON_SAMPLES_PER_FRAME)
```

### Audio Export
```python
# Using frequency
Play(wave, "audio.wav", frequency=440, duration=2.0)

# Using musical notes
Play(wave, "audio.wav", note="A4", duration=2.0)
Play(wave, "audio.wav", note="C#5", duration=1.5)
Play(wave, "audio.wav", note="Bb3", duration=3.0)
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
   python example.py
   python your_script.py

   # Option 2: Use direct path (if activation doesn't work)
   wavetable_env/bin/python example.py
   wavetable_env/bin/python your_script.py
   ```

3. **Creating Your Own Scripts**:
   Create new Python files in the project directory:
   ```python
   # your_synthesis.py
   from wavetable import *

   # Your synthesis code here
   wave = H(1) + 0.5*H(3)
   Wavetable(wave, "generated/my_wavetable.wav")
   ```

### Output Files

All generated files are saved to the `generated/` directory:
- **Wavetable files** (`.wav`): Load directly into Xfer Serum or other wavetable synthesizers
- **Audio files** (`*_audio.wav`): Playable audio demonstrations of the waveforms
- The `generated/` directory is automatically created and is ignored by git

## Tips

1. **Length is abstract**: Segment lengths are relative until rendering - they scale to fit the output duration
2. **Automatic normalization**: `Wavetable()` and `Play()` automatically normalize to prevent clipping
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
Wavetable(wave, "serum_wavetable.wav")

# Export for Ableton Live (1024 samples per frame)
Wavetable(wave, "ableton_wavetable.wav", samples_per_frame=ABLETON_SAMPLES_PER_FRAME)

# Export for some other synthesizer (64 frames, 512 samples each)
Wavetable(wave, "other_synth.wav", frames=64, samples_per_frame=512)
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
