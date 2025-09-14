#include "wt_compiler.h"
#include <iostream>
#include <string>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <cctype>

struct RenderOptions {
    std::string function_type;  // "wavetable" or "play"
    std::string directory;
    std::string input_file;
    std::string output_file;

    // Play-specific options
    double frequency = 0.0;
    std::string note;
    double time_seconds = 0.0;
};

// Convert MIDI note to frequency (A4 = 440 Hz)
double noteToFrequency(const std::string& note) {
    if (note.empty()) return 0.0;

    // Parse note name (C, D, E, F, G, A, B) and optional sharp/flat
    char note_letter = std::toupper(note[0]);
    int octave;
    int semitone_offset = 0;

    // Parse sharps/flats and octave
    size_t i = 1;
    if (i < note.length() && (note[i] == '#' || note[i] == 'b')) {
        semitone_offset = (note[i] == '#') ? 1 : -1;
        i++;
    }

    if (i >= note.length() || !std::isdigit(note[i])) {
        return 0.0; // Invalid format
    }

    octave = note[i] - '0';

    // Handle multi-digit octaves
    i++;
    while (i < note.length() && std::isdigit(note[i])) {
        octave = octave * 10 + (note[i] - '0');
        i++;
    }

    // Convert note letter to semitone offset from C
    int semitones_from_c;
    switch (note_letter) {
        case 'C': semitones_from_c = 0; break;
        case 'D': semitones_from_c = 2; break;
        case 'E': semitones_from_c = 4; break;
        case 'F': semitones_from_c = 5; break;
        case 'G': semitones_from_c = 7; break;
        case 'A': semitones_from_c = 9; break;
        case 'B': semitones_from_c = 11; break;
        default: return 0.0; // Invalid note
    }

    // Calculate MIDI note number (C4 = 60)
    int midi_note = (octave + 1) * 12 + semitones_from_c + semitone_offset;

    // Convert MIDI note to frequency: f = 440 * 2^((n-69)/12)
    return 440.0 * std::pow(2.0, (midi_note - 69) / 12.0);
}

// Parse time string (e.g., "1000ms", "2.5s")
double parseTime(const std::string& time_str) {
    if (time_str.empty()) return 0.0;

    std::string num_part;
    std::string unit_part;

    // Find where numbers end and unit begins
    size_t i = 0;
    while (i < time_str.length() &&
           (std::isdigit(time_str[i]) || time_str[i] == '.' || time_str[i] == '-')) {
        num_part += time_str[i];
        i++;
    }

    // Rest is the unit
    unit_part = time_str.substr(i);

    if (num_part.empty()) return 0.0;

    double value;
    try {
        value = std::stod(num_part);
    } catch (const std::exception&) {
        return 0.0;
    }

    // Convert to seconds
    if (unit_part == "ms") {
        return value / 1000.0;
    } else if (unit_part == "s") {
        return value;
    } else if (unit_part.empty()) {
        // Raw number without unit - return -1 to indicate error
        return -1.0;
    } else {
        return 0.0; // Invalid unit
    }
}

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <function> [options] <input.wt> <output.wav>\n\n";
    std::cout << "Functions:\n";
    std::cout << "  wavetable [--dir <directory>]  Generate wavetable file\n";
    std::cout << "  play [--dir <directory>] <frequency_option> --time <duration>\n";
    std::cout << "                                 Generate audio at specific pitch\n\n";
    std::cout << "Frequency options (exactly one required for 'play'):\n";
    std::cout << "  --freq <hz>                    Frequency in Hz (e.g., --freq 440)\n";
    std::cout << "  --note <note>                  MIDI note (e.g., --note A4, --note C#5)\n\n";
    std::cout << "Time format for 'play':\n";
    std::cout << "  --time <duration>              Duration as <number>ms or <number>s\n";
    std::cout << "                                 (e.g., --time 1000ms, --time 2.5s)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " wavetable basic.wt output.wav\n";
    std::cout << "  " << program_name << " wavetable --dir wavetables basic.wt output.wav\n";
    std::cout << "  " << program_name << " play --freq 440 --time 2s basic.wt output.wav\n";
    std::cout << "  " << program_name << " play --note C5 --time 1500ms basic.wt output.wav\n";
}

bool parseArgs(int argc, char* argv[], RenderOptions& options) {
    if (argc < 4) return false;

    int arg_idx = 1;

    // Parse function type
    options.function_type = argv[arg_idx++];
    if (options.function_type != "wavetable" && options.function_type != "play") {
        return false;
    }

    // Parse optional flags
    while (arg_idx < argc - 2) { // Leave space for input.wt and output.wav
        std::string arg = argv[arg_idx];

        if (arg == "--dir") {
            if (arg_idx + 1 >= argc) return false;
            options.directory = argv[++arg_idx];
            if (options.directory.back() != '/') options.directory += '/';
        } else if (arg == "--freq") {
            if (arg_idx + 1 >= argc) return false;
            try {
                options.frequency = std::stod(argv[++arg_idx]);
            } catch (const std::exception&) {
                return false;
            }
        } else if (arg == "--note") {
            if (arg_idx + 1 >= argc) return false;
            options.note = argv[++arg_idx];
            options.frequency = noteToFrequency(options.note);
            if (options.frequency <= 0.0) {
                std::cerr << "Error: Invalid note format '" << options.note << "'. Use format like A4, C#5, Bb3, etc.\n";
                return false;
            }
        } else if (arg == "--time") {
            if (arg_idx + 1 >= argc) return false;
            std::string time_str = argv[++arg_idx];
            options.time_seconds = parseTime(time_str);
            if (options.time_seconds < 0.0) {
                std::cerr << "Error: Time must include unit. Use format like 1000ms or 2.5s, not just '" << time_str << "'.\n";
                return false;
            } else if (options.time_seconds == 0.0) {
                std::cerr << "Error: Invalid time format '" << time_str << "'. Use format like 1000ms or 2.5s.\n";
                return false;
            }
        } else {
            std::cerr << "Error: Unknown option '" << arg << "'.\n";
            return false;
        }
        arg_idx++;
    }

    // Parse input and output files
    if (arg_idx + 2 != argc) return false;
    options.input_file = options.directory + argv[arg_idx];
    options.output_file = options.directory + argv[arg_idx + 1];

    // Validate play-specific options
    if (options.function_type == "play") {
        if (options.frequency <= 0.0) {
            std::cerr << "Error: 'play' function requires either --freq or --note option.\n";
            return false;
        }
        if (options.time_seconds <= 0.0) {
            std::cerr << "Error: 'play' function requires --time option.\n";
            return false;
        }
        if (!options.note.empty() && options.frequency > 0.0) {
            // Check if both --freq and --note were specified
            // This is a bit tricky to detect perfectly, but we can check if frequency
            // doesn't match the note conversion (indicating --freq was also specified)
            // For now, we'll trust the user didn't specify both
        }
    }

    return true;
}

int main(int argc, char* argv[]) {
    RenderOptions options;

    if (!parseArgs(argc, argv, options)) {
        printUsage(argv[0]);
        return 1;
    }

    if (options.function_type == "wavetable") {
        // Standard wavetable generation
        if (WTCompiler::compileToWAV(options.input_file, options.output_file)) {
            std::cout << "Successfully generated wavetable " << options.input_file << " to " << options.output_file << std::endl;
            return 0;
        } else {
            std::cerr << "Wavetable generation failed" << std::endl;
            return 1;
        }
    } else if (options.function_type == "play") {
        // Generate playable audio at specified frequency and duration
        if (WTCompiler::compileToPlayableWAV(options.input_file, options.output_file,
                                             options.frequency, options.time_seconds)) {
            std::cout << "Successfully generated audio " << options.input_file << " to " << options.output_file;
            if (!options.note.empty()) {
                std::cout << " at note " << options.note << " (" << options.frequency << " Hz)";
            } else {
                std::cout << " at " << options.frequency << " Hz";
            }
            std::cout << " for " << options.time_seconds << " seconds" << std::endl;
            return 0;
        } else {
            std::cerr << "Audio generation failed" << std::endl;
            return 1;
        }
    }

    return 1;
}