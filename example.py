#!/usr/bin/env python3
"""
Comprehensive examples for the wavetable synthesis library.

This demonstrates all key features of the Python wavetable library including
harmonic generation, phase modulation, morphing, concatenation, and export.
Combines functionality from previous demo.py and test_simple.py files.
"""

from wavetable import *
import os

def basic_functionality_tests():
    """Test basic functionality with simple examples."""
    print("=== Basic Functionality Tests ===")

    # Ensure generated directory exists
    os.makedirs('generated', exist_ok=True)

    # Test 1: Basic harmonic
    print("1. Creating fundamental harmonic")
    fund = H(1)
    Wavetable(fund, "generated/test_fundamental.wav")
    print("   ✓ Generated test_fundamental.wav")

    # Test 2: Phase modulation
    print("2. Creating phase modulation")
    pm_basic = PM(H(1), H(2), 0.5)
    Wavetable(pm_basic, "generated/test_pm.wav")
    Play(pm_basic, "generated/test_pm_audio.wav", frequency=440, duration=1.0)
    print("   ✓ Generated test_pm.wav and test_pm_audio.wav")

    # Test 3: Additive synthesis
    print("3. Creating additive synthesis")
    additive = H(1) + 0.5*H(2) + 0.3*H(3)
    Wavetable(additive, "generated/test_additive.wav")
    Play(additive, "generated/test_additive_audio.wav", note="A4", duration=1.5)
    print("   ✓ Generated test_additive.wav and test_additive_audio.wav")

    # Test 4: Morphing
    print("4. Creating morphing segment")
    morph = Segment(H(1), H(5), 8.0)
    Wavetable(morph, "generated/test_morph.wav")
    Play(morph, "generated/test_morph_audio.wav", note="C4", duration=2.0)
    print("   ✓ Generated test_morph.wav and test_morph_audio.wav")

    print("   Basic tests completed!\n")

def original_wt_equivalents():
    """Demonstrate Python equivalents of the original .wt files."""
    print("=== Original .wt File Equivalents ===")

    # Original: f
    print("1. Basic fundamental (equivalent to 'f')")
    fundamental = H(1)
    Wavetable(fundamental, "generated/python_fundamental.wav")
    Play(fundamental, "generated/python_fundamental_play.wav", note="A4", duration=1.5)
    print("   ✓ Generated fundamental wavetable and audio")

    # Original: pm(f, 2f, 0.0)
    print("2. PM with zero modulation (equivalent to 'pm(f, 2f, 0.0)')")
    pm_zero = PM(H(1), H(2), 0.0)
    Wavetable(pm_zero, "generated/python_pm_zero.wav")
    Play(pm_zero, "generated/python_pm_zero_play.wav", frequency=440, duration=1.0)
    print("   ✓ Generated zero PM wavetable and audio")

    # Original: pm(f, 2f, 0.5)
    print("3. Basic PM (equivalent to 'pm(f, 2f, 0.5)')")
    pm_basic = PM(H(1), H(2), 0.5)
    Wavetable(pm_basic, "generated/python_pm_basic.wav")
    Play(pm_basic, "generated/python_pm_basic_play.wav", note="C5", duration=2.0)
    print("   ✓ Generated basic PM wavetable and audio")

    # Original: segment(f, 5f, 8)
    print("4. Morphing segment (equivalent to 'segment(f, 5f, 8)')")
    morph_seg = Segment(H(1), H(5), 8.0)
    Wavetable(morph_seg, "generated/python_morph.wav")
    Play(morph_seg, "generated/python_morph_play.wav", frequency=220, duration=3.0)
    print("   ✓ Generated morphing wavetable and audio")

    # Complex PM example
    print("5. Complex PM (equivalent to test_pm_mixed.wt)")
    carrier_seg = Segment(H(1), H(5), 8.0)
    modulator_wave = H(3)
    pm_mixed = PM(carrier_seg, modulator_wave, 0.8)
    Wavetable(pm_mixed, "generated/python_pm_mixed.wav")
    Play(pm_mixed, "generated/python_pm_mixed_play.wav", note="G4", duration=4.0)
    print("   ✓ Generated complex PM wavetable and audio")

    print("   Original equivalents completed!\n")

def additive_synthesis_examples():
    """Demonstrate additive synthesis by combining harmonics."""
    print("=== Additive Synthesis ===")

    # Create a complex wave with multiple harmonics
    complex_wave = H(1, 1.0) + H(3, 0.3) + H(5, 0.1)
    Wavetable(complex_wave, "generated/additive_complex.wav")
    Play(complex_wave, "generated/additive_complex_audio.wav", note="C4", duration=2.0)
    print("Created additive synthesis wavetable and audio")

    # Create sawtooth-like wave
    sawtooth = sum((H(i, 1.0/i) for i in range(1, 16)), H(0, 0.0))
    Wavetable(sawtooth, "generated/sawtooth_approx.wav")
    print("Created sawtooth approximation")

    # Square wave approximation (odd harmonics only)
    square = sum((H(i, 1.0/i) for i in range(1, 16, 2)), H(0, 0.0))
    Wavetable(square, "generated/square_approx.wav")
    print("Created square wave approximation")

    print("   Additive synthesis completed!\n")

def morphing_examples():
    """Demonstrate morphing between different waveforms."""
    print("=== Morphing Examples ===")

    # Morph from fundamental to 5th harmonic
    morph_segment = Segment(H(1), H(5), length=8.0)
    Wavetable(morph_segment, "generated/morph_1_to_5.wav")
    Play(morph_segment, "generated/morph_1_to_5_audio.wav", frequency=220, duration=3.0)
    print("Created morphing segment from fundamental to 5th harmonic")

    # Complex morph: rich harmonic to simple sine
    rich_wave = H(1) + 0.5 * H(2) + 0.3 * H(3) + 0.2 * H(4)
    simple_wave = H(1)
    complex_morph = Segment(rich_wave, simple_wave, length=16.0)
    Wavetable(complex_morph, "generated/complex_morph.wav")
    print("Created complex morphing wavetable")

    print("   Morphing examples completed!\n")

def phase_modulation_examples():
    """Demonstrate phase modulation synthesis."""
    print("=== Phase Modulation ===")

    # Basic PM: fundamental modulated by second harmonic
    pm_basic = PM(H(1), H(2), 0.5)
    Wavetable(pm_basic, "generated/pm_basic.wav")
    Play(pm_basic, "generated/pm_basic_audio.wav", note="G3", duration=2.5)
    print("Created basic phase modulation")

    # Complex PM with different amounts
    for amount in [0.0, 0.3, 0.8, 1.5]:
        pm_wave = PM(H(1), H(3), amount)
        Wavetable(pm_wave, f"generated/pm_amount_{amount}.wav")
    print("Created PM wavetables with varying amounts")

    # PM with complex carrier and modulator
    carrier = H(1) + 0.3 * H(2)
    modulator = H(3) + 0.2 * H(5)
    pm_complex = PM(carrier, modulator, 0.8)
    Wavetable(pm_complex, "generated/pm_complex.wav")
    print("Created complex PM with multiple harmonics")

    print("   Phase modulation completed!\n")

def concatenation_examples():
    """Demonstrate concatenation of wave segments."""
    print("=== Concatenation ===")

    # Create different segments
    seg1 = SetLength(H(1), 4.0)
    seg2 = SetLength(H(2), 4.0)
    seg3 = SetLength(H(3), 4.0)

    # Concatenate them
    concatenated = Cat(seg1, seg2, seg3)
    Wavetable(concatenated, "generated/concatenated.wav")
    Play(concatenated, "generated/concatenated_audio.wav", frequency=330, duration=4.0)
    print("Created concatenated wavetable")

    # More complex concatenation
    additive = H(1) + 0.5 * H(2) + 0.3 * H(3) + 0.2 * H(4) + 0.1 * H(5)
    Wavetable(additive, "generated/python_additive.wav")
    Play(additive, "generated/python_additive_play.wav", frequency=330, duration=2.0)

    seg_complex1 = SetLength(H(1), 4.0)
    seg_complex2 = SetLength(H(2) + 0.3 * H(3), 4.0)
    seg_complex3 = SetLength(PM(H(1), H(2), 0.5), 4.0)
    concatenated_complex = Cat(seg_complex1, seg_complex2, seg_complex3)
    Wavetable(concatenated_complex, "generated/python_concatenated.wav")
    Play(concatenated_complex, "generated/python_concatenated_play.wav", note="D4", duration=3.0)
    print("Created complex concatenated wavetable")

    print("   Concatenation completed!\n")

def advanced_examples():
    """More complex synthesis examples."""
    print("=== Advanced Examples ===")

    # Create a complex evolving wavetable
    start_wave = H(1) + 0.5 * H(2)
    end_wave = PM(H(1), H(3), 1.2)
    evolution = Segment(start_wave, end_wave, length=32.0)
    Wavetable(evolution, "generated/evolution.wav")
    print("Created evolving wavetable (additive to PM)")

    # Nested PM: carrier is itself PM'd
    inner_pm = PM(H(1), H(2), 0.3)
    outer_pm = PM(inner_pm, H(4), 0.6)
    Wavetable(outer_pm, "generated/nested_pm.wav")
    Play(outer_pm, "generated/nested_pm_audio.wav", note="E4", duration=3.0)
    print("Created nested phase modulation")

    # Multiple morphing segments
    morph1 = Segment(H(1), H(2), 8.0)
    morph2 = Segment(H(3), H(4), 8.0)
    morph3 = Segment(H(5), H(1), 8.0)
    multi_morph = Cat(morph1, morph2, morph3)
    Wavetable(multi_morph, "generated/multi_morph.wav")
    print("Created multi-segment morphing wavetable")

    print("   Advanced examples completed!\n")

def test_new_operations():
    """Test the new scalar multiplication and addition operations."""
    print("=== New Operations Tests ===")

    # Test scalar multiplication with segments
    print("1. Testing scalar multiplication with segments")
    base_wave = H(1) + 0.5 * H(2)
    morph_seg = Segment(base_wave, H(3), 8.0)
    scaled_morph = 0.5 * morph_seg  # This should now work
    Wavetable(scaled_morph, "generated/scaled_morph.wav")
    print("   ✓ Scaled morphing segment")

    # Test PMWave scaling
    pm_wave = PM(H(1), H(2), 0.8)
    scaled_pm = 0.7 * pm_wave  # This should now work
    Wavetable(scaled_pm, "generated/scaled_pm.wav")
    print("   ✓ Scaled PM wave")

    # Test addition of segments with different lengths
    print("2. Testing segment addition")
    static_wave = H(2) + 0.3 * H(4)
    morph_wave = Segment(H(1), H(3), 10.0)
    added_segments = static_wave + morph_wave  # Static should be expanded to morph length
    Wavetable(added_segments, "generated/added_segments.wav")
    print("   ✓ Added static + morphing segment")

    # Test addition of same-length segments
    morph1 = Segment(H(1), H(2), 8.0)
    morph2 = Segment(H(3), H(4), 8.0)
    combined_morph = morph1 + morph2
    Wavetable(combined_morph, "generated/combined_morph.wav")
    print("   ✓ Added two morphing segments of same length")

    print("   New operations tests completed!\n")

def interactive_examples():
    """Interactive examples for testing different parameters."""
    print("=== Interactive Examples ===")

    # Different PM amounts
    print("Testing different PM amounts:")
    for amount in [0.0, 0.2, 0.5, 1.0, 2.0]:
        pm_wave = PM(H(1), H(3), amount)
        Play(pm_wave, f"generated/pm_test_{amount}.wav", frequency=440, duration=0.8)
        print(f"  ✓ PM amount {amount}")

    # Different notes
    print("\nTesting different musical notes:")
    test_wave = H(1) + 0.3 * H(2) + 0.1 * H(4)
    notes = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
    for note in notes:
        Play(test_wave, f"generated/note_{note}.wav", note=note, duration=0.5)
        print(f"  ✓ Note {note}")

    print("   Interactive examples completed!\n")

def synthesizer_compatibility_examples():
    """Demonstrate wavetable export for different synthesizers."""
    print("=== Synthesizer Compatibility ===")

    # Create a test wave for comparison
    test_wave = H(1) + 0.3*H(2) + 0.1*H(5) + 0.05*H(7)

    # Serum format (2048 samples per frame) - default
    print("1. Exporting for Xfer Serum (2048 samples per frame)")
    Wavetable(test_wave, "generated/serum_format.wav")  # Default is Serum
    Wavetable(test_wave, "generated/serum_explicit.wav", samples_per_frame=SERUM_SAMPLES_PER_FRAME)
    print("   ✓ Generated Serum-compatible wavetables")

    # Ableton Wavetable format (1024 samples per frame)
    print("2. Exporting for Ableton Live Wavetable (1024 samples per frame)")
    Wavetable(test_wave, "generated/ableton_format.wav", samples_per_frame=ABLETON_SAMPLES_PER_FRAME)
    print("   ✓ Generated Ableton-compatible wavetable")

    # Compare file sizes
    import os
    serum_size = os.path.getsize("generated/serum_format.wav")
    ableton_size = os.path.getsize("generated/ableton_format.wav")
    print(f"   File size comparison:")
    print(f"   - Serum format: {serum_size:,} bytes")
    print(f"   - Ableton format: {ableton_size:,} bytes")
    print(f"   - Ratio: {serum_size/ableton_size:.1f}x larger")

    # More complex example with morphing
    print("3. Complex morphing for different synthesizers")
    morph_wave = Segment(H(1) + 0.2*H(3), PM(H(1), H(2), 0.8), 16.0)

    Wavetable(morph_wave, "generated/complex_serum.wav", samples_per_frame=SERUM_SAMPLES_PER_FRAME)
    Wavetable(morph_wave, "generated/complex_ableton.wav", samples_per_frame=ABLETON_SAMPLES_PER_FRAME)
    print("   ✓ Generated complex morphing wavetables for both formats")

    print("   Synthesizer compatibility examples completed!\n")

def adsr_envelope_example():
    """Demonstrate ADSR envelope using morphing segments."""
    print("=== ADSR Envelope Example ===")

    # Attack-Decay-Sustain-Release envelope using morphing
    # Note: Length parameters are abstract until wavetable rendering

    # Attack: silence to full volume (2 abstract units)
    attack = Segment(H(0, 0.0), H(1), 2.0)

    # Decay: full volume to sustain level (1 abstract unit)
    decay = Segment(H(1), 0.5*H(1), 1.0)

    # Sustain: hold sustain level (4 abstract units)
    sustain = SetLength(0.5*H(1), 4.0)

    # Release: sustain to silence (3 abstract units)
    release = Segment(0.5*H(1), H(0, 0.0), 3.0)

    # Combine into full ADSR envelope
    adsr_envelope = Cat(attack, decay, sustain, release)

    # Export as wavetable - the abstract lengths become actual frame timing
    Wavetable(adsr_envelope, "generated/adsr_wavetable.wav")
    print("   ✓ Generated ADSR envelope wavetable")

    # Export as audio with specific duration - abstract lengths scale to fit
    Play(adsr_envelope, "generated/adsr_audio.wav", note="C4", duration=3.0)
    print("   ✓ Generated ADSR envelope audio (3 second duration)")

    # Create a more complex ADSR with harmonic content
    rich_attack = Segment(H(0, 0.0), H(1) + 0.3*H(2), 2.0)
    rich_decay = Segment(H(1) + 0.3*H(2), 0.6*(H(1) + 0.2*H(3)), 1.0)
    rich_sustain = SetLength(0.6*(H(1) + 0.2*H(3)), 4.0)
    rich_release = Segment(0.6*(H(1) + 0.2*H(3)), H(0, 0.0), 3.0)

    rich_adsr = Cat(rich_attack, rich_decay, rich_sustain, rich_release)
    Wavetable(rich_adsr, "generated/rich_adsr_wavetable.wav")
    Play(rich_adsr, "generated/rich_adsr_audio.wav", note="A3", duration=4.0)
    print("   ✓ Generated rich harmonic ADSR envelope")

    print("   ADSR envelope examples completed!\n")

def main():
    """Run all examples."""
    try:
        print("=== Wavetable Synthesis Library Examples ===\n")

        basic_functionality_tests()
        original_wt_equivalents()
        additive_synthesis_examples()
        morphing_examples()
        phase_modulation_examples()
        concatenation_examples()
        advanced_examples()
        test_new_operations()
        synthesizer_compatibility_examples()
        adsr_envelope_example()
        interactive_examples()

        print("=== Summary ===")
        print("All examples completed successfully!")
        print("Generated wavetable files in generated/ directory (.wav) can be loaded into Xfer Serum")
        print("Generated audio files (*_audio.wav) can be played directly")
        print("\nThe library now supports:")
        print("  • Scalar multiplication of all segment types")
        print("  • Addition of segments (same length or static + morphing)")
        print("  • Automatic normalization in Wavetable() and Play() functions")
        print("  • Comprehensive wavetable synthesis capabilities")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()