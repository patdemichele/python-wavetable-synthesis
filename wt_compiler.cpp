#include "wt_compiler.h"
#include "wavetable_generator.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cmath>

// Lexer Implementation
Lexer::Lexer(const std::string& source) : source(source), position(0), line(1), column(1) {}

char Lexer::peek(int offset) {
    size_t pos = position + offset;
    if (pos >= source.length()) return '\0';
    return source[pos];
}

char Lexer::advance() {
    if (position >= source.length()) return '\0';
    char ch = source[position++];
    if (ch == '\n') {
        line++;
        column = 1;
    } else {
        column++;
    }
    return ch;
}

void Lexer::skipWhitespace() {
    while (position < source.length() && std::isspace(peek()) && peek() != '\n') {
        advance();
    }
}

Token Lexer::readNumber() {
    int start_col = column;
    std::string number;

    while (position < source.length() && (std::isdigit(peek()) || peek() == '.')) {
        number += advance();
    }

    return {TokenType::NUMBER, number, line, start_col};
}

Token Lexer::readIdentifier() {
    int start_col = column;
    std::string identifier;

    while (position < source.length() && (std::isalnum(peek()) || peek() == '_')) {
        identifier += advance();
    }

    TokenType type = TokenType::IDENTIFIER;
    if (identifier == "let") type = TokenType::LET;
    else if (identifier == "segment") type = TokenType::SEGMENT;
    else if (identifier == "cat") type = TokenType::CAT;
    else if (identifier == "n") type = TokenType::NORMALIZE;
    else if (identifier == "setlen") type = TokenType::SETLEN;
    else if (identifier == "fm") type = TokenType::FM;

    return {type, identifier, line, start_col};
}

Token Lexer::readComment() {
    int start_col = column;
    std::string comment;

    while (position < source.length() && peek() != '\n') {
        comment += advance();
    }

    return {TokenType::COMMENT, comment, line, start_col};
}

std::vector<Token> Lexer::tokenize() {
    std::vector<Token> tokens;

    while (position < source.length()) {
        skipWhitespace();

        if (position >= source.length()) break;

        char ch = peek();
        int start_col = column;

        switch (ch) {
            case '\n':
                advance();
                tokens.push_back({TokenType::NEWLINE, "\n", line - 1, start_col});
                break;
            case '#':
                tokens.push_back(readComment());
                break;
            case '=':
                advance();
                tokens.push_back({TokenType::EQUALS, "=", line, start_col});
                break;
            case '+':
                advance();
                tokens.push_back({TokenType::PLUS, "+", line, start_col});
                break;
            case '-':
                advance();
                tokens.push_back({TokenType::MINUS, "-", line, start_col});
                break;
            case '*':
                advance();
                tokens.push_back({TokenType::MULTIPLY, "*", line, start_col});
                break;
            case '(':
                advance();
                tokens.push_back({TokenType::LPAREN, "(", line, start_col});
                break;
            case ')':
                advance();
                tokens.push_back({TokenType::RPAREN, ")", line, start_col});
                break;
            case ',':
                advance();
                tokens.push_back({TokenType::COMMA, ",", line, start_col});
                break;
            default:
                if (std::isdigit(ch)) {
                    tokens.push_back(readNumber());
                } else if (std::isalpha(ch) || ch == '_') {
                    // Check if it's a standalone 'f' after reading the full identifier
                    auto token = readIdentifier();
                    if (token.value == "f") {
                        token.type = TokenType::FUNDAMENTAL;
                    }
                    tokens.push_back(token);
                } else {
                    advance(); // Skip unknown characters
                }
                break;
        }
    }

    tokens.push_back({TokenType::EOF_TOKEN, "", line, column});
    return tokens;
}

// Parser Implementation
Parser::Parser(const std::vector<Token>& tokens) : tokens(tokens), position(0) {}

Token& Parser::peek(int offset) {
    size_t pos = position + offset;
    if (pos >= tokens.size()) {
        static Token eof_token = {TokenType::EOF_TOKEN, "", 0, 0};
        return eof_token;
    }
    return tokens[pos];
}

Token& Parser::advance() {
    if (position < tokens.size()) position++;
    return peek(-1);
}

bool Parser::match(TokenType type) {
    if (check(type)) {
        advance();
        return true;
    }
    return false;
}

bool Parser::check(TokenType type) {
    return peek().type == type;
}

std::shared_ptr<Program> Parser::parse() {
    return program();
}

std::shared_ptr<Program> Parser::program() {
    auto prog = std::make_shared<Program>();

    // Skip initial newlines and comments
    while (match(TokenType::NEWLINE) || match(TokenType::COMMENT)) {}

    // Parse assignments
    while (check(TokenType::LET)) {
        prog->assignments.push_back(assignment());
        while (match(TokenType::NEWLINE) || match(TokenType::COMMENT)) {}
    }

    // Parse final expression
    if (!check(TokenType::EOF_TOKEN)) {
        prog->final_expression = expression();
    }

    return prog;
}

std::shared_ptr<Assignment> Parser::assignment() {
    if (!match(TokenType::LET)) {
        throw std::runtime_error("Expected 'let' keyword");
    }

    if (!check(TokenType::IDENTIFIER)) {
        throw std::runtime_error("Expected identifier after 'let'");
    }

    std::string var_name = advance().value;

    if (!match(TokenType::EQUALS)) {
        throw std::runtime_error("Expected '=' after variable name");
    }

    auto value = expression();

    return std::make_shared<Assignment>(var_name, value);
}

std::shared_ptr<Expression> Parser::expression() {
    return additive();
}

std::shared_ptr<Expression> Parser::additive() {
    auto expr = multiplicative();

    while (check(TokenType::PLUS) || check(TokenType::MINUS)) {
        TokenType op = advance().type;
        auto right = multiplicative();
        expr = std::make_shared<BinaryOp>(expr, op, right);
    }

    return expr;
}

std::shared_ptr<Expression> Parser::multiplicative() {
    auto expr = unary();

    while (check(TokenType::MULTIPLY)) {
        TokenType op = advance().type;
        auto right = unary();
        expr = std::make_shared<BinaryOp>(expr, op, right);
    }

    return expr;
}

std::shared_ptr<Expression> Parser::unary() {
    if (check(TokenType::MINUS)) {
        TokenType op = advance().type;
        auto operand = unary();
        return std::make_shared<UnaryOp>(op, operand);
    }

    return primary();
}

std::shared_ptr<Expression> Parser::primary() {
    // Handle number followed by 'f' (e.g., "3f")
    if (check(TokenType::NUMBER)) {
        double num = std::stod(peek().value);
        if (peek(1).type == TokenType::FUNDAMENTAL) {
            advance(); // consume number
            advance(); // consume 'f'
            return std::make_shared<FundamentalLiteral>(static_cast<int>(num));
        } else {
            advance(); // consume number
            return std::make_shared<NumberLiteral>(num);
        }
    }

    if (check(TokenType::FUNDAMENTAL)) {
        advance(); // consume 'f'
        return std::make_shared<FundamentalLiteral>(1);
    }

    if (check(TokenType::IDENTIFIER) || check(TokenType::SEGMENT) || check(TokenType::CAT) ||
        check(TokenType::NORMALIZE) || check(TokenType::SETLEN) || check(TokenType::FM)) {
        std::string name = advance().value;

        if (check(TokenType::LPAREN)) {
            return functionCall(name);
        }

        return std::make_shared<Variable>(name);
    }

    if (match(TokenType::LPAREN)) {
        auto expr = expression();
        if (!match(TokenType::RPAREN)) {
            throw std::runtime_error("Expected ')' after expression");
        }
        return expr;
    }

    throw std::runtime_error("Unexpected token in expression");
}

std::shared_ptr<Expression> Parser::functionCall(const std::string& name) {
    if (!match(TokenType::LPAREN)) {
        throw std::runtime_error("Expected '(' after function name");
    }

    std::vector<std::shared_ptr<Expression>> args;

    if (!check(TokenType::RPAREN)) {
        args.push_back(expression());

        while (match(TokenType::COMMA)) {
            args.push_back(expression());
        }
    }

    if (!match(TokenType::RPAREN)) {
        throw std::runtime_error("Expected ')' after function arguments");
    }

    return std::make_shared<FunctionCall>(name, args);
}

// Wave Implementation
std::shared_ptr<Wave> Wave::multiply(double scalar) const {
    auto result = std::make_shared<Wave>();
    result->max_amplitude = std::abs(scalar) * max_amplitude;
    for (const auto& h : harmonics) {
        result->harmonics.push_back(Harmonic(h.frequency_multiple, h.amplitude * scalar));
    }
    return result;
}

std::shared_ptr<Wave> Wave::add(const Wave& other) const {
    auto result = std::make_shared<Wave>();
    result->max_amplitude = max_amplitude + other.max_amplitude;

    // Copy all harmonics from this wave
    for (const auto& h : harmonics) {
        result->harmonics.push_back(h);
    }

    // Add harmonics from other wave
    for (const auto& other_h : other.harmonics) {
        bool found = false;
        for (auto& h : result->harmonics) {
            if (h.frequency_multiple == other_h.frequency_multiple) {
                h.amplitude += other_h.amplitude;
                found = true;
                break;
            }
        }
        if (!found) {
            result->harmonics.push_back(other_h);
        }
    }

    return result;
}

std::vector<float> Wave::generateSamples(int sample_count) const {
    std::vector<float> samples(sample_count, 0.0f);

    for (int i = 0; i < sample_count; ++i) {
        float sample = 0.0f;
        for (const auto& h : harmonics) {
            float phase = 2.0f * M_PI * h.frequency_multiple * i / sample_count;
            sample += h.amplitude * std::sin(phase);
        }
        samples[i] = sample;
    }

    return samples;
}

std::shared_ptr<Wave> Wave::normalize() const {
    if (max_amplitude == 0.0) {
        // Return a copy of the zero wave
        return std::make_shared<Wave>(*this);
    }

    // Normalize by dividing by max amplitude
    return multiply(1.0 / max_amplitude);
}

// Segment Implementation
std::shared_ptr<Segment> Segment::concatenate(const Segment& other) const {
    return std::make_shared<ConcatenatedSegment>(
        std::make_shared<Segment>(*this),
        std::make_shared<Segment>(other)
    );
}

std::shared_ptr<Segment> Segment::normalize() const {
    if (max_amplitude == 0.0) {
        return std::make_shared<Segment>(start_wave, end_wave, length);
    }

    double scale_factor = 1.0 / max_amplitude;
    auto normalized_start = start_wave->multiply(scale_factor);
    auto normalized_end = end_wave->multiply(scale_factor);
    return std::make_shared<Segment>(normalized_start, normalized_end, length);
}

std::shared_ptr<Segment> Segment::setLength(double new_length) const {
    return std::make_shared<Segment>(start_wave, end_wave, new_length);
}

std::vector<std::vector<float>> Segment::generateFrames(int frame_count) const {
    std::vector<std::vector<float>> frames;
    frames.reserve(frame_count);

    for (int i = 0; i < frame_count; ++i) {
        float t = (frame_count <= 1) ? 0.0f : static_cast<float>(i) / (frame_count - 1);

        // Create interpolated wave
        Wave interpolated_wave;

        // Combine harmonics from start and end waves
        std::map<int, std::pair<float, float>> harmonic_pairs;

        // Initialize all harmonics to 0.0 amplitude
        for (const auto& h : start_wave->harmonics) {
            harmonic_pairs[h.frequency_multiple] = {h.amplitude, 0.0f};
        }

        for (const auto& h : end_wave->harmonics) {
            if (harmonic_pairs.find(h.frequency_multiple) != harmonic_pairs.end()) {
                harmonic_pairs[h.frequency_multiple].second = h.amplitude;
            } else {
                harmonic_pairs[h.frequency_multiple] = {0.0f, h.amplitude};
            }
        }

        for (const auto& [freq, amplitudes] : harmonic_pairs) {
            float start_amp = amplitudes.first;
            float end_amp = amplitudes.second;
            float interpolated_amp = start_amp * (1.0f - t) + end_amp * t;

            if (std::abs(interpolated_amp) > 1e-6) {  // Only add significant harmonics
                interpolated_wave.harmonics.push_back(Wave::Harmonic(freq, interpolated_amp));
            }
        }

        frames.push_back(interpolated_wave.generateSamples());
    }

    return frames;
}

// Evaluator Implementation
Value Evaluator::evaluate(std::shared_ptr<Expression> expr) {
    if (auto num_lit = std::dynamic_pointer_cast<NumberLiteral>(expr)) {
        return num_lit->value;
    }

    if (auto fund_lit = std::dynamic_pointer_cast<FundamentalLiteral>(expr)) {
        return std::make_shared<Wave>(fund_lit->harmonic);
    }

    if (auto var = std::dynamic_pointer_cast<Variable>(expr)) {
        auto it = variables.find(var->name);
        if (it == variables.end()) {
            throw std::runtime_error("Undefined variable: " + var->name);
        }
        return it->second;
    }

    if (auto bin_op = std::dynamic_pointer_cast<BinaryOp>(expr)) {
        auto left_val = evaluate(bin_op->left);
        auto right_val = evaluate(bin_op->right);

        if (bin_op->op == TokenType::MULTIPLY) {
            // Handle scalar * wave
            if (std::holds_alternative<double>(left_val) &&
                std::holds_alternative<std::shared_ptr<Wave>>(right_val)) {
                auto scalar = std::get<double>(left_val);
                auto wave = std::get<std::shared_ptr<Wave>>(right_val);
                return wave->multiply(scalar);
            }
            // Handle wave * scalar
            if (std::holds_alternative<std::shared_ptr<Wave>>(left_val) &&
                std::holds_alternative<double>(right_val)) {
                auto wave = std::get<std::shared_ptr<Wave>>(left_val);
                auto scalar = std::get<double>(right_val);
                return wave->multiply(scalar);
            }
        }

        if (bin_op->op == TokenType::PLUS) {
            // Handle wave + wave
            if (std::holds_alternative<std::shared_ptr<Wave>>(left_val) &&
                std::holds_alternative<std::shared_ptr<Wave>>(right_val)) {
                auto wave1 = std::get<std::shared_ptr<Wave>>(left_val);
                auto wave2 = std::get<std::shared_ptr<Wave>>(right_val);
                return wave1->add(*wave2);
            }
        }
    }

    if (auto unary_op = std::dynamic_pointer_cast<UnaryOp>(expr)) {
        auto operand_val = evaluate(unary_op->operand);

        if (unary_op->op == TokenType::MINUS) {
            if (std::holds_alternative<double>(operand_val)) {
                return -std::get<double>(operand_val);
            }
            if (std::holds_alternative<std::shared_ptr<Wave>>(operand_val)) {
                auto wave = std::get<std::shared_ptr<Wave>>(operand_val);
                return wave->multiply(-1.0);
            }
        }
    }

    if (auto func_call = std::dynamic_pointer_cast<FunctionCall>(expr)) {
        if (func_call->name == "segment") {
            if (func_call->args.size() != 3) {
                throw std::runtime_error("segment() requires exactly 3 arguments");
            }

            auto wave1_val = evaluate(func_call->args[0]);
            auto wave2_val = evaluate(func_call->args[1]);
            auto length_val = evaluate(func_call->args[2]);

            if (!std::holds_alternative<std::shared_ptr<Wave>>(wave1_val) ||
                !std::holds_alternative<std::shared_ptr<Wave>>(wave2_val) ||
                !std::holds_alternative<double>(length_val)) {
                throw std::runtime_error("Invalid arguments to segment()");
            }

            auto wave1 = std::get<std::shared_ptr<Wave>>(wave1_val);
            auto wave2 = std::get<std::shared_ptr<Wave>>(wave2_val);
            auto length = std::get<double>(length_val);

            return std::make_shared<Segment>(wave1, wave2, length);
        }

        if (func_call->name == "cat") {
            if (func_call->args.size() != 2) {
                throw std::runtime_error("cat() requires exactly 2 arguments");
            }

            auto seg1_val = evaluate(func_call->args[0]);
            auto seg2_val = evaluate(func_call->args[1]);

            if (!std::holds_alternative<std::shared_ptr<Segment>>(seg1_val) ||
                !std::holds_alternative<std::shared_ptr<Segment>>(seg2_val)) {
                throw std::runtime_error("cat() arguments must be segments");
            }

            auto seg1 = std::get<std::shared_ptr<Segment>>(seg1_val);
            auto seg2 = std::get<std::shared_ptr<Segment>>(seg2_val);

            return seg1->concatenate(*seg2);
        }
    }

    throw std::runtime_error("Cannot evaluate expression");
}

Value Evaluator::evaluate(std::shared_ptr<Program> program) {
    // Evaluate all assignments
    for (auto& assignment : program->assignments) {
        evaluateAssignment(assignment);
    }

    // Evaluate final expression
    if (program->final_expression) {
        auto result = evaluate(program->final_expression);

        // Convert Wave to Segment if needed
        if (std::holds_alternative<std::shared_ptr<Wave>>(result)) {
            auto wave = std::get<std::shared_ptr<Wave>>(result);
            return std::make_shared<Segment>(wave, wave, 1.0);
        }

        return result;
    }

    throw std::runtime_error("No final expression found");
}

Value Evaluator::evaluateAssignment(std::shared_ptr<Assignment> assignment) {
    auto value = evaluate(assignment->value);
    variables[assignment->variable] = value;
    return value;
}

// Compiler Implementation
std::string WTCompiler::readFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

bool WTCompiler::compileToWAV(const std::string& wt_file, const std::string& output_wav) {
    try {
        // Read source file
        std::string source = readFile(wt_file);

        // Tokenize
        Lexer lexer(source);
        auto tokens = lexer.tokenize();

        // Parse
        Parser parser(tokens);
        auto program = parser.parse();

        // Evaluate
        Evaluator evaluator;
        auto result = evaluator.evaluate(program);

        // Convert to wavetable
        if (!std::holds_alternative<std::shared_ptr<Segment>>(result)) {
            throw std::runtime_error("Final expression must be a segment");
        }

        auto segment = std::get<std::shared_ptr<Segment>>(result);

        // Always normalize the final segment to prevent clipping
        auto normalized_segment = segment->normalize();
        auto frames = normalized_segment->generateFrames(256);

        // Convert to flat array for WAV export
        std::vector<float> wavetable_data;
        wavetable_data.reserve(256 * 2048);

        for (const auto& frame : frames) {
            wavetable_data.insert(wavetable_data.end(), frame.begin(), frame.end());
        }

        // Export as WAV
        return WavetableGenerator::exportWAV(wavetable_data, output_wav);

    } catch (const std::exception& e) {
        std::cerr << "Compilation error: " << e.what() << std::endl;
        return false;
    }
}

// ConcatenatedSegment Implementation
std::shared_ptr<Segment> ConcatenatedSegment::normalize() const {
    if (max_amplitude == 0.0) {
        return std::make_shared<ConcatenatedSegment>(first_segment, second_segment);
    }

    // Normalize each segment individually
    auto normalized_first = first_segment->normalize();
    auto normalized_second = second_segment->normalize();

    return std::make_shared<ConcatenatedSegment>(normalized_first, normalized_second);
}

std::vector<std::vector<float>> ConcatenatedSegment::generateFrames(int frame_count) const {
    // Calculate how many frames each segment should get based on their relative lengths
    double total_length = first_segment->length + second_segment->length;
    int first_frames = static_cast<int>(frame_count * first_segment->length / total_length);
    int second_frames = frame_count - first_frames;

    // Ensure each segment gets at least 1 frame
    if (first_frames == 0 && frame_count > 0) {
        first_frames = 1;
        second_frames = frame_count - 1;
    }
    if (second_frames == 0 && frame_count > 0) {
        second_frames = 1;
        first_frames = frame_count - 1;
    }

    // Generate frames for each segment
    auto first_segment_frames = first_segment->generateFrames(first_frames);
    auto second_segment_frames = second_segment->generateFrames(second_frames);

    // Concatenate the results
    std::vector<std::vector<float>> result;
    result.reserve(frame_count);

    result.insert(result.end(), first_segment_frames.begin(), first_segment_frames.end());
    result.insert(result.end(), second_segment_frames.begin(), second_segment_frames.end());

    return result;
}