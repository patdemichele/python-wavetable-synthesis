#pragma once

#include <vector>
#include <string>
#include <functional>
#include <cmath>

class WavetableGenerator {
public:
    static const int SAMPLES_PER_WAVE = 2048;
    static const int MAX_WAVES = 256;

    using WaveFunction = std::function<float(float)>;

    static std::vector<float> generateSineWave();
    static std::vector<float> generateSquareWave();
    static std::vector<float> generateTriangleWave();
    static std::vector<float> generateSawWave();

    static std::vector<float> generateCustomWave(WaveFunction func);

    static std::vector<float> morphWaves(const std::vector<float>& wave1,
                                       const std::vector<float>& wave2,
                                       float morph_amount);

    static std::vector<float> createWavetable(const std::vector<std::vector<float>>& waves);

    static std::vector<float> createMorphingWavetable(const std::vector<float>& start_wave,
                                                    const std::vector<float>& end_wave,
                                                    int num_steps);

    static bool exportWAV(const std::vector<float>& wavetable,
                         const std::string& filename,
                         int sample_rate = 44100);

private:
    static void writeWAVHeader(std::vector<uint8_t>& buffer,
                              int sample_rate,
                              int num_samples);
    static void writeInt32(std::vector<uint8_t>& buffer, uint32_t value);
    static void writeInt16(std::vector<uint8_t>& buffer, uint16_t value);
};