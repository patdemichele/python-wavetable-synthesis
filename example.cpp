#include "wavetable_generator.h"
#include <iostream>

int main() {
    try {
        // Generate basic waveforms
        auto sine = WavetableGenerator::generateSineWave();
        auto square = WavetableGenerator::generateSquareWave();
        auto triangle = WavetableGenerator::generateTriangleWave();
        auto saw = WavetableGenerator::generateSawWave();

        std::cout << "Generated basic waveforms (2048 samples each)" << std::endl;

        // Export single waves as 256-frame wavetables
        std::vector<std::vector<float>> single_sine(256, sine);
        std::vector<std::vector<float>> single_square(256, square);
        std::vector<std::vector<float>> single_triangle(256, triangle);
        std::vector<std::vector<float>> single_saw(256, saw);

        WavetableGenerator::exportWAV(WavetableGenerator::createWavetable(single_sine), "sine_wave.wav");
        WavetableGenerator::exportWAV(WavetableGenerator::createWavetable(single_square), "square_wave.wav");
        WavetableGenerator::exportWAV(WavetableGenerator::createWavetable(single_triangle), "triangle_wave.wav");
        WavetableGenerator::exportWAV(WavetableGenerator::createWavetable(single_saw), "saw_wave.wav");

        std::cout << "Exported single waveforms (256 frames each)" << std::endl;

        // Create morphing wavetable from sine to square (256 steps)
        auto morph_sine_square = WavetableGenerator::createMorphingWavetable(sine, square, 256);
        WavetableGenerator::exportWAV(morph_sine_square, "sine_to_square_morph.wav");

        std::cout << "Created sine to square morph wavetable (256 waves)" << std::endl;

        // Create morphing wavetable from triangle to saw (256 steps)
        auto morph_tri_saw = WavetableGenerator::createMorphingWavetable(triangle, saw, 256);
        WavetableGenerator::exportWAV(morph_tri_saw, "triangle_to_saw_morph.wav");

        std::cout << "Created triangle to saw morph wavetable (256 waves)" << std::endl;

        // Create custom wave using lambda
        auto custom_wave = WavetableGenerator::generateCustomWave([](float phase) {
            return 0.5f * std::sin(phase) + 0.3f * std::sin(3.0f * phase) + 0.2f * std::sin(5.0f * phase);
        });
        std::vector<std::vector<float>> single_custom(256, custom_wave);
        WavetableGenerator::exportWAV(WavetableGenerator::createWavetable(single_custom), "custom_harmonic_wave.wav");

        std::cout << "Created custom harmonic wave (256 frames)" << std::endl;

        // Create complex wavetable with multiple different waves, then pad to 256
        std::vector<std::vector<float>> mixed_waves;

        // Add 51 copies of each base wave (5 * 51 = 255) plus one more sine (total 256)
        for (int i = 0; i < 51; ++i) {
            mixed_waves.push_back(sine);
            mixed_waves.push_back(square);
            mixed_waves.push_back(triangle);
            mixed_waves.push_back(saw);
            mixed_waves.push_back(custom_wave);
        }
        mixed_waves.push_back(sine); // 256th frame

        auto mixed_wavetable = WavetableGenerator::createWavetable(mixed_waves);
        WavetableGenerator::exportWAV(mixed_wavetable, "mixed_wavetable.wav");

        std::cout << "Created mixed wavetable with 256 frames" << std::endl;

        // Test morphing between different amounts
        auto morph_25 = WavetableGenerator::morphWaves(sine, square, 0.25f);
        auto morph_50 = WavetableGenerator::morphWaves(sine, square, 0.5f);
        auto morph_75 = WavetableGenerator::morphWaves(sine, square, 0.75f);

        std::vector<std::vector<float>> single_morph_25(256, morph_25);
        std::vector<std::vector<float>> single_morph_50(256, morph_50);
        std::vector<std::vector<float>> single_morph_75(256, morph_75);

        WavetableGenerator::exportWAV(WavetableGenerator::createWavetable(single_morph_25), "sine_square_25_morph.wav");
        WavetableGenerator::exportWAV(WavetableGenerator::createWavetable(single_morph_50), "sine_square_50_morph.wav");
        WavetableGenerator::exportWAV(WavetableGenerator::createWavetable(single_morph_75), "sine_square_75_morph.wav");

        std::cout << "Created individual morph examples (25%, 50%, 75%) with 256 frames each" << std::endl;

        std::cout << "\nAll wavetables generated successfully!" << std::endl;
        std::cout << "Files can be loaded into Xfer Serum as wavetables." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}