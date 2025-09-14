#include "wt_compiler.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::string directory = "";
    std::string input_file;
    std::string output_file;

    if (argc == 3) {
        // Standard usage: ./compile input.wt output.wav
        input_file = argv[1];
        output_file = argv[2];
    } else if (argc == 5 && std::string(argv[1]) == "--dir") {
        // Directory usage: ./compile --dir wavetables input.wt output.wav
        directory = argv[2];
        if (directory.back() != '/') directory += '/';
        input_file = directory + argv[3];
        output_file = directory + argv[4];
    } else {
        std::cerr << "Usage: " << argv[0] << " [--dir <directory>] <input.wt> <output.wav>" << std::endl;
        return 1;
    }

    if (WTCompiler::compileToWAV(input_file, output_file)) {
        std::cout << "Successfully compiled " << input_file << " to " << output_file << std::endl;
        return 0;
    } else {
        std::cerr << "Compilation failed" << std::endl;
        return 1;
    }
}