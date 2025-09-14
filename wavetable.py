"""
Wavetable synthesis library for generating wavetables and audio.

This library provides a flexible Python API for creating complex wavetables
using phase modulation, morphing, concatenation, and other synthesis techniques.
Compatible with Xfer Serum and other wavetable synthesizers.

Key Concept - Abstract Length Parameters:
Length parameters in segments are abstract units that represent relative timing.
They only become concrete durations when rendered to wavetables or audio:
- In wavetables: lengths scale to fit the total frame count
- In audio: lengths scale to fit the specified duration
- Example: Segment(wave1, wave2, 8.0) takes 8/total_length of the output

This abstraction allows the same segment definitions to work for both
wavetable synthesis (256 frames) and audio generation (any duration).
"""

import numpy as np
import struct
import math
from typing import Union, List, Optional
import warnings

# Configuration constants for different synthesizers
SERUM_SAMPLES_PER_FRAME = 2048    # Xfer Serum wavetable frame size
ABLETON_SAMPLES_PER_FRAME = 1024  # Ableton Live Wavetable device frame size
SAMPLES_PER_FRAME = SERUM_SAMPLES_PER_FRAME  # Default to Serum format

DEFAULT_SAMPLE_RATE = 44100   # CD quality sample rate
DEFAULT_FRAME_COUNT = 256     # Standard number of frames in wavetable


class WaveSegment:
    """
    Represents a wave segment that can morph between two waveforms over time.

    A WaveSegment is the core building block of the library. It represents
    either a static wave (length=0) or a morphing segment between two waves.
    """

    def __init__(self, start_wave=None, end_wave=None, length: float = 0.0):
        """
        Create a new WaveSegment.

        Args:
            start_wave: Starting waveform (WaveSegment or None)
            end_wave: Ending waveform (WaveSegment or None)
            length: Abstract length units (0.0 for static waves, >0 for morphing)
        """
        self.start_wave = start_wave
        self.end_wave = end_wave if end_wave is not None else start_wave
        self.length = length

        # For static waves, store harmonics directly
        self.harmonics = {}  # freq_multiple -> amplitude
        self.max_amplitude = 0.0

        if start_wave is None and end_wave is None:
            # Empty wave
            self.harmonics = {}
            self.max_amplitude = 0.0

    @classmethod
    def from_harmonics(cls, harmonics_dict: dict) -> 'WaveSegment':
        """Create a WaveSegment from a dictionary of harmonics."""
        wave = cls()
        wave.harmonics = harmonics_dict.copy()
        wave.max_amplitude = sum(abs(amp) for amp in harmonics_dict.values())
        return wave

    def generate_samples(self, sample_count: int = SAMPLES_PER_FRAME) -> np.ndarray:
        """Generate audio samples for this wave segment."""
        if not self.harmonics:
            return np.zeros(sample_count)

        samples = np.zeros(sample_count)
        t = np.linspace(0, 2 * np.pi, sample_count, endpoint=False)

        for freq_multiple, amplitude in self.harmonics.items():
            samples += amplitude * np.sin(freq_multiple * t)

        return samples

    def generate_frames(self, frame_count: int = DEFAULT_FRAME_COUNT, samples_per_frame: int = SAMPLES_PER_FRAME) -> List[np.ndarray]:
        """
        Generate a sequence of frames for wavetable synthesis.

        Args:
            frame_count: Number of frames to generate
            samples_per_frame: Number of samples per frame (for different synthesizers)

        Returns:
            List of numpy arrays, each containing samples_per_frame samples
        """
        # Handle concatenated segments
        if hasattr(self, '_segments'):
            return self._generate_concatenated_frames(frame_count, samples_per_frame)

        if self.length == 0.0 or self.start_wave is self.end_wave:
            # Static wave - all frames are identical
            frame = self.generate_samples(samples_per_frame)
            return [frame.copy() for _ in range(frame_count)]

        # Morphing segment
        frames = []
        for i in range(frame_count):
            t = i / max(frame_count - 1, 1)  # 0.0 to 1.0

            # Generate interpolated wave
            interpolated = self._interpolate_waves(self.start_wave, self.end_wave, t)
            frames.append(interpolated.generate_samples(samples_per_frame))

        return frames

    def _generate_concatenated_frames(self, frame_count: int, samples_per_frame: int = SAMPLES_PER_FRAME) -> List[np.ndarray]:
        """Generate frames for concatenated segments."""
        if not hasattr(self, '_segments') or not self._segments:
            return [np.zeros(samples_per_frame) for _ in range(frame_count)]

        total_length = sum(seg.length for seg in self._segments)
        if total_length == 0:
            return [np.zeros(samples_per_frame) for _ in range(frame_count)]

        frames = []
        for i in range(frame_count):
            # Determine which segment we're in based on frame position
            t = i / max(frame_count - 1, 1)  # 0.0 to 1.0
            position = t * total_length

            current_pos = 0
            for segment in self._segments:
                if position <= current_pos + segment.length:
                    # We're in this segment
                    segment_t = (position - current_pos) / max(segment.length, 1e-10)
                    segment_frames = segment.generate_frames(2, samples_per_frame)
                    if segment_t <= 0:
                        frames.append(segment_frames[0])
                    elif segment_t >= 1:
                        frames.append(segment_frames[-1])
                    else:
                        # Interpolate between first and last frame of segment
                        frame1 = segment_frames[0]
                        frame2 = segment_frames[-1]
                        interpolated = frame1 * (1 - segment_t) + frame2 * segment_t
                        frames.append(interpolated)
                    break
                current_pos += segment.length
            else:
                # Fallback - use last segment
                frames.append(self._segments[-1].generate_frames(1, samples_per_frame)[0])

        return frames

    def _interpolate_waves(self, wave1: 'WaveSegment', wave2: 'WaveSegment', t: float) -> 'WaveSegment':
        """Interpolate between two waves."""
        result_harmonics = {}

        # Get all harmonics from both waves
        all_freqs = set(wave1.harmonics.keys()) | set(wave2.harmonics.keys())

        for freq in all_freqs:
            amp1 = wave1.harmonics.get(freq, 0.0)
            amp2 = wave2.harmonics.get(freq, 0.0)
            result_harmonics[freq] = amp1 * (1 - t) + amp2 * t

        return WaveSegment.from_harmonics(result_harmonics)

    def __add__(self, other: 'WaveSegment') -> 'WaveSegment':
        """Add two wave segments."""
        # Handle static waves
        if self.length == 0 and other.length == 0:
            result_harmonics = self.harmonics.copy()
            for freq, amp in other.harmonics.items():
                result_harmonics[freq] = result_harmonics.get(freq, 0.0) + amp
            return WaveSegment.from_harmonics(result_harmonics)

        # Handle one static, one morphing - convert static to morphing of same length
        if self.length == 0 and other.length != 0:
            return Segment(self, self, other.length) + other
        if other.length == 0 and self.length != 0:
            return self + Segment(other, other, self.length)

        # Handle both morphing - must be same length
        if self.length != other.length:
            raise ValueError(f"Cannot add segments of different lengths: {self.length} vs {other.length}")

        # Create new morphing segment by adding start and end waves
        start_sum = self.start_wave + other.start_wave
        end_sum = self.end_wave + other.end_wave
        return Segment(start_sum, end_sum, self.length)

    def __mul__(self, scalar: Union[float, int]) -> 'WaveSegment':
        """Multiply wave segment by scalar."""
        if self.length == 0:
            # Static wave - multiply harmonics
            result_harmonics = {freq: amp * scalar for freq, amp in self.harmonics.items()}
            return WaveSegment.from_harmonics(result_harmonics)
        elif hasattr(self, '_segments'):
            # Concatenated segment - multiply each sub-segment
            multiplied_segments = [seg * scalar for seg in self._segments]
            return Cat(*multiplied_segments)
        else:
            # Morphing segment - multiply start and end waves
            multiplied_start = self.start_wave * scalar
            multiplied_end = self.end_wave * scalar
            return Segment(multiplied_start, multiplied_end, self.length)

    def __rmul__(self, scalar: Union[float, int]) -> 'WaveSegment':
        """Right multiply (scalar * wave)."""
        return self.__mul__(scalar)


class PMWave(WaveSegment):
    """Phase modulation wave segment."""

    def __init__(self, carrier: WaveSegment, modulator: WaveSegment, amount: float):
        super().__init__()
        self.carrier = carrier
        self.modulator = modulator
        self.amount = amount

        # Estimate max amplitude (PM can increase amplitude)
        self.max_amplitude = carrier.max_amplitude * (1 + abs(amount))

    def generate_samples(self, sample_count: int = SAMPLES_PER_FRAME) -> np.ndarray:
        """Generate phase-modulated samples."""
        if abs(self.amount) < 1e-10:
            return self.carrier.generate_samples(sample_count)

        # Generate modulator samples
        modulator_samples = self.modulator.generate_samples(sample_count)

        # Generate phase-modulated samples
        result = np.zeros(sample_count)
        t = np.linspace(0, 2 * np.pi, sample_count, endpoint=False)

        for freq_multiple, amplitude in self.carrier.harmonics.items():
            base_phase = freq_multiple * t
            modulated_phase = base_phase + self.amount * modulator_samples
            result += amplitude * np.sin(modulated_phase)

        return result

    def __mul__(self, scalar: Union[float, int]) -> 'PMWave':
        """Multiply PMWave by scalar - multiplies carrier amplitude."""
        multiplied_carrier = self.carrier * scalar
        result = PMWave(multiplied_carrier, self.modulator, self.amount)
        result.max_amplitude = self.max_amplitude * abs(scalar)
        return result

    def __rmul__(self, scalar: Union[float, int]) -> 'PMWave':
        """Right multiply (scalar * PMWave)."""
        return self.__mul__(scalar)


# Utility functions for creating common waveforms

def H(harmonic: int, amplitude: float = 1.0) -> WaveSegment:
    """Create a harmonic wave."""
    return WaveSegment.from_harmonics({harmonic: amplitude})


def Segment(start_wave: WaveSegment, end_wave: WaveSegment, length: float = 1.0) -> WaveSegment:
    """
    Create a morphing segment between two waves.

    Args:
        start_wave: Starting waveform
        end_wave: Ending waveform
        length: Abstract length units (scales relative to other segments)
    """
    segment = WaveSegment(start_wave, end_wave, length)
    # Calculate max amplitude for morphing segment
    segment.max_amplitude = max(start_wave.max_amplitude, end_wave.max_amplitude)
    return segment


def Cat(*segments: WaveSegment) -> WaveSegment:
    """Concatenate multiple wave segments."""
    if len(segments) == 0:
        return WaveSegment()
    if len(segments) == 1:
        return segments[0]

    # For now, implement as a simple wrapper
    # In a full implementation, this would create a ConcatenatedSegment class
    total_length = sum(seg.length for seg in segments)

    # Create a segment that represents the concatenation
    result = WaveSegment()
    result.length = total_length
    result._segments = list(segments)  # Store for frame generation
    result.max_amplitude = max(seg.max_amplitude for seg in segments)

    return result


def PM(carrier: WaveSegment, modulator: WaveSegment, amount: float) -> PMWave:
    """Create a phase-modulated wave."""
    return PMWave(carrier, modulator, amount)


def N(wave_segment: WaveSegment) -> WaveSegment:
    """Normalize a wave segment to prevent clipping."""
    if wave_segment.max_amplitude == 0:
        return wave_segment

    scale_factor = 1.0 / wave_segment.max_amplitude

    if hasattr(wave_segment, 'carrier'):  # PMWave
        # Create normalized PM wave by normalizing the carrier recursively
        normalized_carrier = N(wave_segment.carrier) if wave_segment.carrier.max_amplitude > 0 else wave_segment.carrier
        # Scale the PM amount to maintain relative modulation intensity
        adjusted_amount = wave_segment.amount * (wave_segment.carrier.max_amplitude / wave_segment.max_amplitude)
        result = PMWave(normalized_carrier, wave_segment.modulator, adjusted_amount)
        result.max_amplitude = 1.0  # Set to normalized amplitude
        return result
    elif hasattr(wave_segment, '_segments'):  # Concatenated segment
        # Normalize each sub-segment
        normalized_segments = [N(seg) for seg in wave_segment._segments]
        return Cat(*normalized_segments)
    elif wave_segment.length == 0:  # Static wave
        return wave_segment * scale_factor
    else:  # Morphing segment
        if wave_segment.start_wave is None or wave_segment.end_wave is None:
            return wave_segment
        normalized_start = N(wave_segment.start_wave)
        normalized_end = N(wave_segment.end_wave)
        return Segment(normalized_start, normalized_end, wave_segment.length)


def SetLength(wave_segment: WaveSegment, new_length: float) -> WaveSegment:
    """Set the length of a wave segment."""
    if wave_segment.length == 0:
        # Convert static wave to segment
        return Segment(wave_segment, wave_segment, new_length)

    result = WaveSegment(wave_segment.start_wave, wave_segment.end_wave, new_length)
    result.max_amplitude = wave_segment.max_amplitude
    if hasattr(wave_segment, '_segments'):
        result._segments = wave_segment._segments
    return result


# Audio generation and export functions

def Wavetable(wave_segment: WaveSegment, filename: str, frames: int = DEFAULT_FRAME_COUNT,
              samples_per_frame: int = SAMPLES_PER_FRAME):
    """
    Export a wave segment as a wavetable WAV file.

    The wave segment is automatically normalized to prevent clipping.

    Args:
        wave_segment: The wave segment to export
        filename: Output filename (.wav)
        frames: Number of frames in the wavetable (default DEFAULT_FRAME_COUNT)
        samples_per_frame: Samples per frame (SERUM_SAMPLES_PER_FRAME=2048 or ABLETON_SAMPLES_PER_FRAME=1024)
    """
    # Always normalize to prevent clipping
    normalized_segment = N(wave_segment)
    frame_data = normalized_segment.generate_frames(frames, samples_per_frame)

    # Convert to flat array
    wavetable_data = np.concatenate(frame_data).astype(np.float32)

    # Export as WAV
    _export_wavetable_wav(wavetable_data, filename, samples_per_frame)


def Play(wave_segment: WaveSegment, filename: str, frequency: float = None,
         note: str = None, duration: float = 1.0):
    """
    Export a wave segment as playable audio.

    The wave segment is automatically normalized to prevent clipping.

    Args:
        wave_segment: The wave segment to export
        filename: Output filename (.wav)
        frequency: Frequency in Hz (use either this or note)
        note: MIDI note (e.g., 'A4', 'C#5')
        duration: Duration in seconds
    """
    if frequency is None and note is None:
        raise ValueError("Must specify either frequency or note")
    if frequency is not None and note is not None:
        raise ValueError("Cannot specify both frequency and note")

    if note is not None:
        frequency = _note_to_frequency(note)

    # Always normalize to prevent clipping
    normalized_segment = N(wave_segment)
    frame_data = normalized_segment.generate_frames(DEFAULT_FRAME_COUNT)

    # Generate audio at specified frequency and duration
    sample_rate = DEFAULT_SAMPLE_RATE
    total_samples = int(duration * sample_rate)
    samples_per_cycle = int(sample_rate / frequency)

    audio_data = np.zeros(total_samples)

    for i in range(total_samples):
        # Calculate which frame to use based on time progression
        time_progress = i / total_samples
        frame_index = min(int(time_progress * (len(frame_data) - 1)), len(frame_data) - 1)

        # Calculate position within current cycle
        cycle_position = i % samples_per_cycle
        wave_position = cycle_position / samples_per_cycle

        # Sample from the wavetable frame
        sample_index = int(wave_position * SAMPLES_PER_FRAME)
        sample_index = min(sample_index, SAMPLES_PER_FRAME - 1)

        audio_data[i] = frame_data[frame_index][sample_index]

    # Export as audio WAV
    _export_audio_wav(audio_data, filename, sample_rate)


# Helper functions

def _note_to_frequency(note: str) -> float:
    """Convert MIDI note to frequency."""
    if not note:
        raise ValueError("Empty note string")

    # Parse note name
    note_letter = note[0].upper()
    if note_letter not in 'CDEFGAB':
        raise ValueError(f"Invalid note letter: {note_letter}")

    # Parse sharps/flats and octave
    i = 1
    semitone_offset = 0
    if i < len(note) and note[i] in '#b':
        semitone_offset = 1 if note[i] == '#' else -1
        i += 1

    if i >= len(note) or not note[i].isdigit():
        raise ValueError(f"Invalid note format: {note}")

    octave = int(note[i:])

    # Convert to semitones from C
    semitones = {
        'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11
    }[note_letter] + semitone_offset

    # Calculate MIDI note number (C4 = 60)
    midi_note = (octave + 1) * 12 + semitones

    # Convert to frequency (A4 = 440 Hz)
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def _export_wavetable_wav(wavetable_data: np.ndarray, filename: str, samples_per_frame: int = SAMPLES_PER_FRAME):
    """Export wavetable data as WAV file."""
    if len(wavetable_data) % samples_per_frame != 0:
        raise ValueError(f"Wavetable data length must be multiple of {samples_per_frame}")

    # Convert to 16-bit PCM
    samples_16bit = np.clip(wavetable_data * 32767, -32768, 32767).astype(np.int16)

    # Write WAV file
    with open(filename, 'wb') as f:
        # WAV header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(samples_16bit) * 2))  # File size
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # Subchunk1Size
        f.write(struct.pack('<H', 1))   # AudioFormat (PCM)
        f.write(struct.pack('<H', 1))   # NumChannels (mono)
        f.write(struct.pack('<I', DEFAULT_SAMPLE_RATE))  # SampleRate
        f.write(struct.pack('<I', DEFAULT_SAMPLE_RATE * 2))  # ByteRate
        f.write(struct.pack('<H', 2))   # BlockAlign
        f.write(struct.pack('<H', 16))  # BitsPerSample
        f.write(b'data')
        f.write(struct.pack('<I', len(samples_16bit) * 2))  # Data size

        # Audio data
        f.write(samples_16bit.tobytes())


def _export_audio_wav(audio_data: np.ndarray, filename: str, sample_rate: int = DEFAULT_SAMPLE_RATE):
    """Export audio data as standard WAV file."""
    # Convert to 16-bit PCM
    samples_16bit = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)

    # Write WAV file
    with open(filename, 'wb') as f:
        # WAV header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(samples_16bit) * 2))  # File size
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # Subchunk1Size
        f.write(struct.pack('<H', 1))   # AudioFormat (PCM)
        f.write(struct.pack('<H', 1))   # NumChannels (mono)
        f.write(struct.pack('<I', sample_rate))  # SampleRate
        f.write(struct.pack('<I', sample_rate * 2))  # ByteRate
        f.write(struct.pack('<H', 2))   # BlockAlign
        f.write(struct.pack('<H', 16))  # BitsPerSample
        f.write(b'data')
        f.write(struct.pack('<I', len(samples_16bit) * 2))  # Data size

        # Audio data
        f.write(samples_16bit.tobytes())


# Convenience aliases for common operations
F = lambda: H(1)  # Fundamental frequency
F2 = lambda: H(2)  # Second harmonic
F3 = lambda: H(3)  # Third harmonic
F4 = lambda: H(4)  # Fourth harmonic
F5 = lambda: H(5)  # Fifth harmonic
Zero = lambda: 0 * H(1)