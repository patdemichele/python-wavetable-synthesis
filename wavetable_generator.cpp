#include "wavetable_generator.h"
#include <fstream>
#include <algorithm>
#include <cmath>

std::vector<float> WavetableGenerator::generateSineWave() {
    std::vector<float> wave(SAMPLES_PER_WAVE);
    for (int i = 0; i < SAMPLES_PER_WAVE; ++i) {
        float phase = 2.0f * M_PI * i / SAMPLES_PER_WAVE;
        wave[i] = std::sin(phase);
    }
    return wave;
}

std::vector<float> WavetableGenerator::generateSquareWave() {
    std::vector<float> wave(SAMPLES_PER_WAVE);
    for (int i = 0; i < SAMPLES_PER_WAVE; ++i) {
        wave[i] = (i < SAMPLES_PER_WAVE / 2) ? 1.0f : -1.0f;
    }
    return wave;
}

std::vector<float> WavetableGenerator::generateTriangleWave() {
    std::vector<float> wave(SAMPLES_PER_WAVE);
    for (int i = 0; i < SAMPLES_PER_WAVE; ++i) {
        if (i < SAMPLES_PER_WAVE / 2) {
            wave[i] = 4.0f * i / SAMPLES_PER_WAVE - 1.0f;
        } else {
            wave[i] = 3.0f - 4.0f * i / SAMPLES_PER_WAVE;
        }
    }
    return wave;
}

std::vector<float> WavetableGenerator::generateSawWave() {
    std::vector<float> wave(SAMPLES_PER_WAVE);
    for (int i = 0; i < SAMPLES_PER_WAVE; ++i) {
        wave[i] = 2.0f * i / SAMPLES_PER_WAVE - 1.0f;
    }
    return wave;
}

std::vector<float> WavetableGenerator::generateCustomWave(WaveFunction func) {
    std::vector<float> wave(SAMPLES_PER_WAVE);
    for (int i = 0; i < SAMPLES_PER_WAVE; ++i) {
        float phase = 2.0f * M_PI * i / SAMPLES_PER_WAVE;
        wave[i] = func(phase);
    }
    return wave;
}

std::vector<float> WavetableGenerator::morphWaves(const std::vector<float>& wave1,
                                                const std::vector<float>& wave2,
                                                float morph_amount) {
    if (wave1.size() != wave2.size() || wave1.size() != SAMPLES_PER_WAVE) {
        throw std::invalid_argument("Wave sizes must match and be 2048 samples");
    }

    morph_amount = std::clamp(morph_amount, 0.0f, 1.0f);
    std::vector<float> result(SAMPLES_PER_WAVE);

    for (int i = 0; i < SAMPLES_PER_WAVE; ++i) {
        result[i] = wave1[i] * (1.0f - morph_amount) + wave2[i] * morph_amount;
    }

    return result;
}

std::vector<float> WavetableGenerator::createWavetable(const std::vector<std::vector<float>>& waves) {
    if (waves.empty() || waves.size() > MAX_WAVES) {
        throw std::invalid_argument("Number of waves must be between 1 and 256");
    }

    std::vector<float> wavetable;
    wavetable.reserve(waves.size() * SAMPLES_PER_WAVE);

    for (const auto& wave : waves) {
        if (wave.size() != SAMPLES_PER_WAVE) {
            throw std::invalid_argument("Each wave must be exactly 2048 samples");
        }
        wavetable.insert(wavetable.end(), wave.begin(), wave.end());
    }

    return wavetable;
}

std::vector<float> WavetableGenerator::createMorphingWavetable(const std::vector<float>& start_wave,
                                                             const std::vector<float>& end_wave,
                                                             int num_steps) {
    if (num_steps < 1 || num_steps > MAX_WAVES) {
        throw std::invalid_argument("Number of steps must be between 1 and 256");
    }

    std::vector<std::vector<float>> waves;
    waves.reserve(num_steps);

    for (int i = 0; i < num_steps; ++i) {
        float morph_amount = static_cast<float>(i) / (num_steps - 1);
        waves.push_back(morphWaves(start_wave, end_wave, morph_amount));
    }

    return createWavetable(waves);
}

bool WavetableGenerator::exportWAV(const std::vector<float>& wavetable,
                                  const std::string& filename,
                                  int sample_rate) {
    if (wavetable.size() % SAMPLES_PER_WAVE != 0) {
        return false;
    }

    std::vector<uint8_t> buffer;
    writeWAVHeader(buffer, sample_rate, wavetable.size());

    for (float sample : wavetable) {
        int16_t pcm_sample = static_cast<int16_t>(std::clamp(sample, -1.0f, 1.0f) * 32767.0f);
        writeInt16(buffer, pcm_sample);
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }

    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    return file.good();
}

void WavetableGenerator::writeWAVHeader(std::vector<uint8_t>& buffer,
                                       int sample_rate,
                                       int num_samples) {
    int data_size = num_samples * 2;
    int file_size = 36 + data_size;

    buffer.insert(buffer.end(), {'R', 'I', 'F', 'F'});
    writeInt32(buffer, file_size);
    buffer.insert(buffer.end(), {'W', 'A', 'V', 'E'});
    buffer.insert(buffer.end(), {'f', 'm', 't', ' '});
    writeInt32(buffer, 16);
    writeInt16(buffer, 1);
    writeInt16(buffer, 1);
    writeInt32(buffer, sample_rate);
    writeInt32(buffer, sample_rate * 2);
    writeInt16(buffer, 2);
    writeInt16(buffer, 16);
    buffer.insert(buffer.end(), {'d', 'a', 't', 'a'});
    writeInt32(buffer, data_size);
}

void WavetableGenerator::writeInt32(std::vector<uint8_t>& buffer, uint32_t value) {
    buffer.push_back(value & 0xFF);
    buffer.push_back((value >> 8) & 0xFF);
    buffer.push_back((value >> 16) & 0xFF);
    buffer.push_back((value >> 24) & 0xFF);
}

void WavetableGenerator::writeInt16(std::vector<uint8_t>& buffer, uint16_t value) {
    buffer.push_back(value & 0xFF);
    buffer.push_back((value >> 8) & 0xFF);
}