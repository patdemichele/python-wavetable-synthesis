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
        # Morphing segments should use generate_frames, not generate_samples
        if self.length > 0:
            # This shouldn't be called directly for morphing segments
            # Return zeros to indicate this is not the right approach
            return np.zeros(sample_count)

        # Handle static waves with harmonics
        if not self.harmonics:
            return np.zeros(sample_count)

        samples = np.zeros(sample_count)
        t = np.linspace(0, 2 * np.pi, sample_count, endpoint=False)

        for freq_multiple, amplitude in self.harmonics.items():
            if freq_multiple == 0:
                # DC component - constant offset
                samples += amplitude
            else:
                # AC component - sine wave
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

    def __str__(self) -> str:
        """Human-readable representation showing the wave structure."""
        return self._format_tree()

    def __repr__(self) -> str:
        """Developer representation showing the wave structure."""
        return self._format_tree()

    def _format_tree(self, indent: int = 0, prefix: str = "") -> str:
        """Format the wave segment as a tree structure."""
        spaces = "  " * indent

        # Handle concatenated segments
        if hasattr(self, '_segments') and self._segments:
            result = f"{spaces}{prefix}Cat(\n"
            for i, segment in enumerate(self._segments):
                is_last = (i == len(self._segments) - 1)
                segment_prefix = "└─ " if is_last else "├─ "
                result += segment._format_tree(indent + 1, segment_prefix)
                if not is_last:
                    result += "\n"
            result += f"\n{spaces})"
            return result

        # Handle morphing segments
        if self.length > 0 and self.start_wave is not None and self.end_wave is not None:
            result = f"{spaces}{prefix}Segment(length={self.length:.1f},\n"
            result += self.start_wave._format_tree(indent + 1, "start: ")
            result += "\n"
            result += self.end_wave._format_tree(indent + 1, "end:   ")
            result += f"\n{spaces})"
            return result

        # Handle static harmonic waves
        if self.harmonics:
            harmonics_list = []
            for freq, amp in sorted(self.harmonics.items()):
                if freq == 0:
                    harmonics_list.append(f"DC({amp:.3f})")
                else:
                    if amp == 1.0:
                        harmonics_list.append(f"H({freq})")
                    else:
                        harmonics_list.append(f"{amp:.3f}*H({freq})")

            if len(harmonics_list) == 1:
                return f"{spaces}{prefix}{harmonics_list[0]}"
            else:
                result = f"{spaces}{prefix}("
                result += " + ".join(harmonics_list)
                result += ")"
                return result

        # Empty wave
        return f"{spaces}{prefix}Zero"


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

        # Check if carrier has harmonics (static wave) or is a complex wave (morphing, etc.)
        if hasattr(self.carrier, 'harmonics') and self.carrier.harmonics:
            # Static harmonic wave - use harmonic-based PM
            result = np.zeros(sample_count)
            t = np.linspace(0, 2 * np.pi, sample_count, endpoint=False)

            for freq_multiple, amplitude in self.carrier.harmonics.items():
                if freq_multiple == 0:
                    # DC component - not affected by phase modulation
                    result += amplitude
                else:
                    # AC component - apply phase modulation
                    base_phase = freq_multiple * t
                    modulated_phase = base_phase + self.amount * modulator_samples
                    result += amplitude * np.sin(modulated_phase)

            return result
        else:
            # Complex wave (morphing, concatenated, etc.) - use sample-based PM
            # For complex waves, we approximate PM by using the carrier as a base oscillator
            # and applying the modulator as phase offset

            t = np.linspace(0, 2 * np.pi, sample_count, endpoint=False)

            # Use the carrier's amplitude envelope
            carrier_samples = self.carrier.generate_samples(sample_count)

            # Extract amplitude envelope (RMS of carrier)
            carrier_rms = np.sqrt(np.mean(carrier_samples**2))
            if carrier_rms < 1e-10:
                return np.zeros(sample_count)

            # Use carrier's spectral content by applying FM to a fundamental
            # This preserves the carrier's characteristics while adding modulation
            base_oscillator = np.sin(t)
            modulated_oscillator = np.sin(t + self.amount * modulator_samples)

            # Blend the base carrier with the modulated version
            # This maintains the carrier's character while adding FM effects
            result = 0.7 * carrier_samples + 0.3 * carrier_rms * modulated_oscillator

            return result

    def generate_frames(self, frame_count: int = DEFAULT_FRAME_COUNT, samples_per_frame: int = SAMPLES_PER_FRAME):
        """Generate frames with proper morphing support for PM."""
        # If both carrier and modulator are static (length == 0), use the standard approach
        if (getattr(self.carrier, 'length', 0) == 0 and
            getattr(self.modulator, 'length', 0) == 0):
            # Both static - all frames identical
            frame = self.generate_samples(samples_per_frame)
            return [frame.copy() for _ in range(frame_count)]

        # At least one is morphing - generate frame-by-frame with proper time evolution
        frames = []
        for i in range(frame_count):
            t = i / max(frame_count - 1, 1)  # 0.0 to 1.0

            # Get carrier at this time position
            if getattr(self.carrier, 'length', 0) > 0:
                # Morphing carrier - interpolate
                carrier_at_t = self.carrier._interpolate_waves(
                    self.carrier.start_wave, self.carrier.end_wave, t)
            else:
                # Static carrier
                carrier_at_t = self.carrier

            # Get modulator at this time position
            if getattr(self.modulator, 'length', 0) > 0:
                # Morphing modulator - interpolate
                modulator_at_t = self.modulator._interpolate_waves(
                    self.modulator.start_wave, self.modulator.end_wave, t)
            else:
                # Static modulator
                modulator_at_t = self.modulator

            # Create PM at this time position
            pm_at_t = PMWave(carrier_at_t, modulator_at_t, self.amount)
            frames.append(pm_at_t.generate_samples(samples_per_frame))

        return frames

    def __mul__(self, scalar: Union[float, int]) -> 'PMWave':
        """Multiply PMWave by scalar - multiplies carrier amplitude."""
        multiplied_carrier = self.carrier * scalar
        result = PMWave(multiplied_carrier, self.modulator, self.amount)
        result.max_amplitude = self.max_amplitude * abs(scalar)
        return result

    def __rmul__(self, scalar: Union[float, int]) -> 'PMWave':
        """Right multiply (scalar * PMWave)."""
        return self.__mul__(scalar)

    def _format_tree(self, indent: int = 0, prefix: str = "") -> str:
        """Format the PMWave as a tree structure."""
        spaces = "  " * indent
        result = f"{spaces}{prefix}PM(amount={self.amount:.2f},\n"
        result += self.carrier._format_tree(indent + 1, "carrier: ")
        result += "\n"
        result += self.modulator._format_tree(indent + 1, "modulator: ")
        result += f"\n{spaces})"
        return result


# Utility functions for creating common waveforms

def H(harmonic: int, amplitude: float = 1.0) -> WaveSegment:
    """Create a harmonic wave."""
    return WaveSegment.from_harmonics({harmonic: amplitude})


def DC(offset: float) -> WaveSegment:
    """
    Create a DC (constant) offset wave.

    Args:
        offset: DC offset value (-1.0 to 1.0)

    Returns:
        WaveSegment with constant DC offset
    """
    if not (-1.0 <= offset <= 1.0):
        raise ValueError(f"DC offset must be between -1.0 and 1.0, got {offset}")

    # DC is represented as the 0th harmonic (constant term)
    return WaveSegment.from_harmonics({0: offset})


def Center(wave: WaveSegment) -> WaveSegment:
    """
    Remove DC offset by centering the wave at its average value.

    Args:
        wave: Input wave segment to center

    Returns:
        New WaveSegment with DC offset removed
    """
    # For static waves with harmonics, simply remove the 0th harmonic (DC component)
    if hasattr(wave, 'harmonics') and wave.harmonics is not None:
        centered_harmonics = {h: amp for h, amp in wave.harmonics.items() if h != 0}
        if not centered_harmonics:
            # If only DC was present, return a zero wave
            return WaveSegment.from_harmonics({1: 0.0})
        return WaveSegment.from_harmonics(centered_harmonics)

    # For complex waves (PM, morphing, concatenated), we need to generate samples
    # and calculate the actual DC offset, then subtract it
    samples = wave.generate_frames(1, SAMPLES_PER_FRAME)[0]
    dc_offset = np.mean(samples)

    # Create a new wave that subtracts this DC offset
    return wave + DC(-dc_offset)


def Clip(wave: WaveSegment, min_val: float = -1.0, max_val: float = 1.0) -> WaveSegment:
    """
    Clip a wave segment to specified amplitude range.

    Args:
        wave: Input wave segment to clip
        min_val: Minimum amplitude value (default -1.0)
        max_val: Maximum amplitude value (default 1.0)

    Returns:
        New WaveSegment with amplitude clipped to [min_val, max_val]
    """
    if min_val >= max_val:
        raise ValueError(f"min_val ({min_val}) must be less than max_val ({max_val})")

    class ClippedWave(WaveSegment):
        def __init__(self, source_wave, min_val, max_val):
            super().__init__()
            self.source_wave = source_wave
            self.min_val = min_val
            self.max_val = max_val
            self.length = source_wave.length
            self.max_amplitude = max_val  # Clipped max amplitude

        def generate_frames(self, frame_count: int, samples_per_frame: int = SAMPLES_PER_FRAME):
            source_frames = self.source_wave.generate_frames(frame_count, samples_per_frame)
            clipped_frames = []
            for frame in source_frames:
                clipped_frame = np.clip(frame, self.min_val, self.max_val)
                clipped_frames.append(clipped_frame)
            return clipped_frames

    return ClippedWave(wave, min_val, max_val)


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

def Wavetable(wave_segment: WaveSegment, filename: str = None, frames: int = DEFAULT_FRAME_COUNT,
              samples_per_frame: int = SAMPLES_PER_FRAME):
    """
    Export a wave segment as a wavetable WAV file.

    ⚠️  DEPRECATED: Use Render() instead for better functionality:
        Render(wave, "wavetable", "file.wav", frames=256, samples_per_frame=2048)

    The wave segment is automatically normalized to prevent clipping.

    Args:
        wave_segment: The wave segment to export
        filename: Output filename (.wav)
        frames: Number of frames in the wavetable (default DEFAULT_FRAME_COUNT)
        samples_per_frame: Samples per frame (SERUM_SAMPLES_PER_FRAME=2048 or ABLETON_SAMPLES_PER_FRAME=1024)
    """
    # Better error messages for common mistakes
    if filename is None:
        raise ValueError(
            "Wavetable() requires a filename parameter.\n"
            "Usage: Wavetable(wave, 'output.wav')\n"
            "Or better yet, use: Render(wave, 'wavetable', 'output.wav')"
        )

    # Check if user is trying to use Render() syntax
    if filename == "wavetable":
        raise ValueError(
            "It looks like you're trying to use Render() syntax with Wavetable().\n"
            "Use: Render(wave, 'wavetable', 'filename.wav')\n"
            "Or: Wavetable(wave, 'filename.wav')"
        )
    # Always normalize to prevent clipping
    normalized_segment = N(wave_segment)
    frame_data = normalized_segment.generate_frames(frames, samples_per_frame)

    # Convert to flat array
    wavetable_data = np.concatenate(frame_data).astype(np.float32)

    # Export as WAV
    _export_wavetable_wav(wavetable_data, filename, samples_per_frame)


def Play(wave_segment: WaveSegment, filename: str = None, frequency: float = None,
         note: str = None, duration: float = 1.0):
    """
    Export a wave segment as playable audio.

    ⚠️  DEPRECATED: Use Render() instead for better functionality:
        Render(wave, "audio", "file.wav", note="A4", duration=1.0)

    The wave segment is automatically normalized to prevent clipping.

    Args:
        wave_segment: The wave segment to export
        filename: Output filename (.wav)
        frequency: Frequency in Hz (use either this or note)
        note: MIDI note (e.g., 'A4', 'C#5')
        duration: Duration in seconds
    """
    # Better error messages for common mistakes
    if filename is None:
        raise ValueError(
            "Play() requires a filename parameter.\n"
            "Usage: Play(wave, 'output.wav', note='A4')\n"
            "Or better yet, use: Render(wave, 'audio', 'output.wav', note='A4')"
        )

    # Check if user is trying to use Render() syntax
    if filename == "audio":
        raise ValueError(
            "It looks like you're trying to use Render() syntax with Play().\n"
            "Use: Render(wave, 'audio', 'filename.wav', note='A4')\n"
            "Or: Play(wave, 'filename.wav', note='A4')"
        )

    if frequency is None and note is None:
        raise ValueError(
            "Must specify either frequency or note parameter.\n"
            "Examples:\n"
            "  Play(wave, 'output.wav', frequency=440)\n"
            "  Play(wave, 'output.wav', note='A4')\n"
            "Or use: Render(wave, 'audio', 'output.wav', note='A4')"
        )
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


def Render(wave_segment: WaveSegment,
           output_type: str = "array",
           filename: str = None,
           # Wavetable parameters
           frames: int = DEFAULT_FRAME_COUNT,
           samples_per_frame: int = SAMPLES_PER_FRAME,
           # Frame selection
           frame_index: int = None,
           # Audio parameters
           frequency: float = None,
           note: str = None,
           duration: float = 1.0,
           sample_rate: int = DEFAULT_SAMPLE_RATE,
           # Processing options
           normalize: bool = True):
    """
    Unified rendering function that can output to wavetable, audio, or numpy array.

    Args:
        wave_segment: The wave segment to render
        output_type: "wavetable", "audio", or "array"
        filename: Output filename (.wav) - required for wavetable/audio output

        # Wavetable mode parameters:
        frames: Number of frames in wavetable (default 256)
        samples_per_frame: Samples per frame (2048 for Serum, 1024 for Ableton)

        # Frame selection:
        frame_index: Specific frame to extract (0 to frames-1). If None, returns all frames.

        # Audio mode parameters:
        frequency: Frequency in Hz (use either this or note)
        note: MIDI note (e.g., 'A4', 'C#5')
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

        # Processing:
        normalize: Whether to normalize output (default True)

    Returns:
        numpy.ndarray: Raw audio data (for all modes)
        - Wavetable (frame_index=None): 2D array [frames, samples_per_frame]
        - Wavetable (frame_index=N): 1D array [samples_per_frame] for frame N
        - Audio: 1D array [total_samples]
        - Array: 1D or 2D array depending on wave_segment type and frame_index
    """
    import numpy as np

    # Apply normalization if requested
    segment = N(wave_segment) if normalize else wave_segment

    if output_type == "wavetable":
        # Generate wavetable frames
        frame_data = segment.generate_frames(frames, samples_per_frame)
        audio_array = np.array(frame_data)  # 2D: [frames, samples_per_frame]

        # Handle frame selection
        if frame_index is not None:
            if not (0 <= frame_index < frames):
                raise ValueError(f"frame_index must be between 0 and {frames-1}, got {frame_index}")
            selected_frame = audio_array[frame_index]  # 1D: [samples_per_frame]

            if filename:
                # Export single frame as wavetable (duplicate frame to fill wavetable)
                single_frame_data = [selected_frame] * frames
                flat_data = np.concatenate(single_frame_data)
                _export_wavetable_wav(flat_data, filename, samples_per_frame)

            return selected_frame
        else:
            # Return all frames
            if filename:
                # Flatten frames for export
                flat_data = np.concatenate(frame_data)
                _export_wavetable_wav(flat_data, filename, samples_per_frame)

            return audio_array

    elif output_type == "audio":
        # Validate audio parameters
        if frequency is None and note is None:
            raise ValueError("Audio mode requires either frequency or note")
        if filename is None:
            raise ValueError("Audio mode requires filename")

        # Convert note to frequency if needed
        if note is not None:
            frequency = _note_to_frequency(note)

        # Generate audio samples
        total_samples = int(duration * sample_rate)
        fundamental_samples_per_cycle = sample_rate / frequency

        # For morphing segments, render across time
        if hasattr(segment, 'length') and segment.length > 0:
            # Check if it's a concatenated segment
            if hasattr(segment, '_segments'):
                # Concatenated segment - use generate_frames approach
                frame_count = max(1, int(duration * 10))  # 10 frames per second
                frames = segment.generate_frames(frame_count, int(total_samples / frame_count))
                audio_array = np.concatenate(frames)[:total_samples]
            else:
                # Morphing/evolving segment - render across duration
                samples_list = []
                time_frames = max(1, int(duration * 10))  # 10 frames per second for smooth evolution

                for i in range(time_frames):
                    t = i / max(time_frames - 1, 1)
                    frame_samples = int(total_samples / time_frames)

                    if hasattr(segment, 'start_wave') and hasattr(segment, 'end_wave'):
                        # Morphing segment
                        interpolated = segment._interpolate_waves(segment.start_wave, segment.end_wave, t)
                        frame_audio = interpolated.generate_samples(frame_samples)
                    else:
                        # Other time-varying segments
                        frame_audio = segment.generate_samples(frame_samples)

                    samples_list.append(frame_audio)

                audio_array = np.concatenate(samples_list)[:total_samples]
        else:
            # Static segment - generate single cycle and tile
            cycle_samples = max(1, int(fundamental_samples_per_cycle))
            single_cycle = segment.generate_samples(cycle_samples)

            # Tile the cycle to fill duration
            cycles_needed = int(np.ceil(total_samples / cycle_samples))
            tiled_audio = np.tile(single_cycle, cycles_needed)
            audio_array = tiled_audio[:total_samples]

        # Export to file
        _export_audio_wav(audio_array, filename, sample_rate)

        return audio_array

    elif output_type == "array":
        # Return raw array for debugging/analysis
        if isinstance(segment, PMWave) or (hasattr(segment, 'length') and segment.length > 0):
            # PMWave or morphing segment - return frames showing evolution
            analysis_frames = frames if frame_index is not None else 8  # Use full frame count if selecting specific frame
            frame_data = segment.generate_frames(analysis_frames, samples_per_frame)
            frame_array = np.array(frame_data)  # 2D array

            if frame_index is not None:
                if not (0 <= frame_index < analysis_frames):
                    raise ValueError(f"frame_index must be between 0 and {analysis_frames-1}, got {frame_index}")
                return frame_array[frame_index]  # 1D array for specific frame
            else:
                return frame_array  # 2D array for all frames
        else:
            # Static segment - return single cycle (frame_index ignored for static waves)
            return segment.generate_samples(samples_per_frame)  # 1D array

    else:
        raise ValueError(f"output_type must be 'wavetable', 'audio', or 'array', got '{output_type}'")


def Visualize(wave_segment: WaveSegment,
             frame_index: int = 0,
             domain: str = "time",
             frames: int = DEFAULT_FRAME_COUNT,
             samples_per_frame: int = SAMPLES_PER_FRAME,
             show: bool = True):
    """
    Visualize a specific frame in time or frequency domain.

    Args:
        wave_segment: The wave segment to visualize
        frame_index: Which frame to visualize (default 0)
        domain: "time", "frequency", or "both"
        frames: Total frame count for morphing segments
        samples_per_frame: Samples per frame
        show: Whether to display the plot immediately

    Returns:
        tuple: (figure, axes) matplotlib objects for further customization
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")

    import numpy as np

    # Get the specific frame data
    frame_data = Render(wave_segment, "array", frame_index=frame_index,
                       frames=frames, samples_per_frame=samples_per_frame)

    if domain == "both":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        axes = (ax1, ax2)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        axes = (ax,)

    if domain in ["time", "both"]:
        ax_time = axes[0] if domain == "both" else axes[0]

        # Time domain plot
        time_axis = np.linspace(0, 1, len(frame_data))  # Normalized time (0 to 1 cycle)
        ax_time.plot(time_axis, frame_data, 'b-', linewidth=1.5)
        ax_time.set_xlabel('Normalized Time (0-1 cycle)')
        ax_time.set_ylabel('Amplitude')
        ax_time.set_title(f'Time Domain - Frame {frame_index}')
        ax_time.grid(True, alpha=0.3)

        # Add amplitude statistics
        rms = np.sqrt(np.mean(frame_data**2))
        peak = np.max(np.abs(frame_data))
        ax_time.text(0.02, 0.98, f'RMS: {rms:.3f}\\nPeak: {peak:.3f}',
                    transform=ax_time.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    if domain in ["frequency", "both"]:
        ax_freq = axes[1] if domain == "both" else axes[0]

        # Frequency domain plot (FFT)
        fft_data = np.fft.fft(frame_data)
        fft_magnitude = np.abs(fft_data[:len(fft_data)//2])  # Only positive frequencies
        freq_axis = np.arange(len(fft_magnitude))  # Harmonic numbers

        # Only plot significant harmonics (above threshold)
        threshold = np.max(fft_magnitude) * 0.01  # 1% of peak
        significant_indices = np.where(fft_magnitude > threshold)[0]

        if len(significant_indices) > 0:
            ax_freq.stem(freq_axis[significant_indices], fft_magnitude[significant_indices],
                        basefmt='k-', linefmt='r-', markerfmt='ro')
            ax_freq.set_xlabel('Harmonic Number')
            ax_freq.set_ylabel('Magnitude')
            ax_freq.set_title(f'Frequency Domain - Frame {frame_index}')
            ax_freq.grid(True, alpha=0.3)

            # Limit x-axis to show only relevant harmonics
            max_harmonic = min(50, np.max(significant_indices) + 5)
            ax_freq.set_xlim(0, max_harmonic)

            # Add harmonic info
            fundamental = fft_magnitude[1] if len(fft_magnitude) > 1 else 0
            total_energy = np.sum(fft_magnitude**2)
            ax_freq.text(0.98, 0.98, f'Fundamental: {fundamental:.3f}\\nTotal Energy: {total_energy:.1f}',
                        transform=ax_freq.transAxes, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            ax_freq.text(0.5, 0.5, 'No significant frequency content',
                        transform=ax_freq.transAxes, ha='center', va='center')

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes


def AnalyzeFrame(wave_segment: WaveSegment,
                frame_index: int = 0,
                frames: int = DEFAULT_FRAME_COUNT,
                samples_per_frame: int = SAMPLES_PER_FRAME):
    """
    Detailed numerical analysis of a specific frame.

    Args:
        wave_segment: The wave segment to analyze
        frame_index: Which frame to analyze (default 0)
        frames: Total frame count for morphing segments
        samples_per_frame: Samples per frame

    Returns:
        dict: Analysis results with keys: rms, peak, dc_offset, thd, harmonics
    """
    import numpy as np

    # Get the specific frame data
    frame_data = Render(wave_segment, "array", frame_index=frame_index,
                       frames=frames, samples_per_frame=samples_per_frame)

    # Time domain analysis
    rms = np.sqrt(np.mean(frame_data**2))
    peak = np.max(np.abs(frame_data))
    dc_offset = np.mean(frame_data)

    # Frequency domain analysis
    fft_data = np.fft.fft(frame_data)
    fft_magnitude = np.abs(fft_data[:len(fft_data)//2])

    # Extract harmonics (first 10 harmonics)
    harmonics = {}
    for i in range(min(10, len(fft_magnitude))):
        if i == 0:
            harmonics[f'DC'] = fft_magnitude[i] / len(frame_data)  # Normalize DC
        else:
            harmonics[f'H{i}'] = fft_magnitude[i] / (len(frame_data) / 2)  # Normalize harmonics

    # Calculate Total Harmonic Distortion (THD)
    if len(fft_magnitude) > 1:
        fundamental = fft_magnitude[1]
        harmonics_sum = np.sum(fft_magnitude[2:min(10, len(fft_magnitude))]**2)
        thd = np.sqrt(harmonics_sum) / fundamental if fundamental > 0 else 0
    else:
        thd = 0

    return {
        'rms': rms,
        'peak': peak,
        'dc_offset': dc_offset,
        'thd': thd,
        'harmonics': harmonics,
        'fundamental_freq': fft_magnitude[1] / (len(frame_data) / 2) if len(fft_magnitude) > 1 else 0,
        'total_energy': np.sum(fft_magnitude**2)
    }


# Convenience aliases for common operations
F = lambda: H(1)  # Fundamental frequency
F2 = lambda: H(2)  # Second harmonic
F3 = lambda: H(3)  # Third harmonic
F4 = lambda: H(4)  # Fourth harmonic
F5 = lambda: H(5)  # Fifth harmonic
Zero = lambda: 0 * H(1)