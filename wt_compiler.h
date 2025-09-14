#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <variant>

// Token types for lexer
enum class TokenType {
    IDENTIFIER,
    NUMBER,
    FUNDAMENTAL,  // 'f'
    LET,
    EQUALS,
    PLUS,
    MINUS,
    MULTIPLY,
    LPAREN,
    RPAREN,
    COMMA,
    SEGMENT,
    CAT,
    NORMALIZE,  // 'n'
    SETLEN,
    PM,
    NEWLINE,
    EOF_TOKEN,
    COMMENT
};

struct Token {
    TokenType type;
    std::string value;
    int line;
    int column;
};

// Forward declarations
class Wave;
class Segment;

// AST Node types
class ASTNode {
public:
    virtual ~ASTNode() = default;
};

class Expression : public ASTNode {
public:
    virtual ~Expression() = default;
};

// Value types that expressions can evaluate to
using Value = std::variant<double, std::shared_ptr<Wave>, std::shared_ptr<Segment>>;

// Specific AST node types
class NumberLiteral : public Expression {
public:
    double value;
    NumberLiteral(double v) : value(v) {}
};

class FundamentalLiteral : public Expression {
public:
    int harmonic;
    FundamentalLiteral(int h = 1) : harmonic(h) {}
};

class Variable : public Expression {
public:
    std::string name;
    Variable(const std::string& n) : name(n) {}
};

class BinaryOp : public Expression {
public:
    std::shared_ptr<Expression> left;
    std::shared_ptr<Expression> right;
    TokenType op;

    BinaryOp(std::shared_ptr<Expression> l, TokenType o, std::shared_ptr<Expression> r)
        : left(l), op(o), right(r) {}
};

class UnaryOp : public Expression {
public:
    std::shared_ptr<Expression> operand;
    TokenType op;

    UnaryOp(TokenType o, std::shared_ptr<Expression> operand)
        : op(o), operand(operand) {}
};

class FunctionCall : public Expression {
public:
    std::string name;
    std::vector<std::shared_ptr<Expression>> args;

    FunctionCall(const std::string& n, std::vector<std::shared_ptr<Expression>> a)
        : name(n), args(a) {}
};

class Assignment : public ASTNode {
public:
    std::string variable;
    std::shared_ptr<Expression> value;

    Assignment(const std::string& var, std::shared_ptr<Expression> val)
        : variable(var), value(val) {}
};

class Program : public ASTNode {
public:
    std::vector<std::shared_ptr<Assignment>> assignments;
    std::shared_ptr<Expression> final_expression;
};

// Runtime types
class Wave {
public:
    struct Harmonic {
        int frequency_multiple;  // 1 for fundamental, 2 for second harmonic, etc.
        double amplitude;

        Harmonic(int freq, double amp) : frequency_multiple(freq), amplitude(amp) {}
    };

    std::vector<Harmonic> harmonics;
    double max_amplitude;  // Theoretical maximum amplitude

    Wave() : max_amplitude(0.0) {}
    Wave(int harmonic_num, double amplitude = 1.0) : max_amplitude(std::abs(amplitude)) {
        harmonics.push_back(Harmonic(harmonic_num, amplitude));
    }

    virtual ~Wave() = default;

    virtual std::shared_ptr<Wave> multiply(double scalar) const;
    std::shared_ptr<Wave> add(const Wave& other) const;
    std::shared_ptr<Wave> normalize() const;
    std::shared_ptr<Wave> phaseModulate(const Wave& modulator, double amount) const;
    virtual std::vector<float> generateSamples(int sample_count = 2048) const;
    double getMaxAmplitude() const { return max_amplitude; }
};

class PMWave : public Wave {
public:
    Wave carrier;
    Wave modulator;
    double pm_amount;

    PMWave(const Wave& carr, const Wave& mod, double amount)
        : Wave(), carrier(carr), modulator(mod), pm_amount(amount) {
        // Max amplitude is carrier's max amplitude (PM doesn't change overall amplitude much)
        max_amplitude = carrier.getMaxAmplitude();
    }

    std::vector<float> generateSamples(int sample_count = 2048) const override;
    std::shared_ptr<Wave> multiply(double scalar) const override;
};

class ConcatenatedSegment;

class Segment {
public:
    std::shared_ptr<Wave> start_wave;
    std::shared_ptr<Wave> end_wave;
    double length;
    double max_amplitude;  // Maximum amplitude across the entire segment

    Segment(std::shared_ptr<Wave> start, std::shared_ptr<Wave> end, double len)
        : start_wave(start), end_wave(end), length(len) {
        // Max amplitude is the maximum of start and end wave max amplitudes
        max_amplitude = std::max(start_wave->getMaxAmplitude(), end_wave->getMaxAmplitude());
    }

    virtual ~Segment() = default;

    std::shared_ptr<Segment> concatenate(const Segment& other) const;
    virtual std::shared_ptr<Segment> normalize() const;
    std::shared_ptr<Segment> setLength(double new_length) const;
    virtual std::shared_ptr<Segment> phaseModulate(const Segment& modulator, double amount) const;
    virtual std::vector<std::vector<float>> generateFrames(int frame_count = 256) const;
    double getMaxAmplitude() const { return max_amplitude; }
};

class ConcatenatedSegment : public Segment {
public:
    std::shared_ptr<Segment> first_segment;
    std::shared_ptr<Segment> second_segment;

    ConcatenatedSegment(std::shared_ptr<Segment> first, std::shared_ptr<Segment> second)
        : Segment(first->start_wave, second->end_wave, first->length + second->length),
          first_segment(first), second_segment(second) {
        max_amplitude = std::max(first->getMaxAmplitude(), second->getMaxAmplitude());
    }

    std::shared_ptr<Segment> normalize() const override;
    std::vector<std::vector<float>> generateFrames(int frame_count = 256) const override;
};

// Lexer
class Lexer {
public:
    Lexer(const std::string& source);
    std::vector<Token> tokenize();

private:
    std::string source;
    size_t position;
    int line;
    int column;

    char peek(int offset = 0);
    char advance();
    void skipWhitespace();
    Token readNumber();
    Token readIdentifier();
    Token readComment();
};

// Parser
class Parser {
public:
    Parser(const std::vector<Token>& tokens);
    std::shared_ptr<Program> parse();

private:
    std::vector<Token> tokens;
    size_t position;

    Token& peek(int offset = 0);
    Token& advance();
    bool match(TokenType type);
    bool check(TokenType type);

    std::shared_ptr<Program> program();
    std::shared_ptr<Assignment> assignment();
    std::shared_ptr<Expression> expression();
    std::shared_ptr<Expression> additive();
    std::shared_ptr<Expression> multiplicative();
    std::shared_ptr<Expression> unary();
    std::shared_ptr<Expression> primary();
    std::shared_ptr<Expression> functionCall(const std::string& name);
};

// Evaluator
class Evaluator {
public:
    Value evaluate(std::shared_ptr<Expression> expr);
    Value evaluate(std::shared_ptr<Program> program);

private:
    std::map<std::string, Value> variables;

    Value evaluateAssignment(std::shared_ptr<Assignment> assignment);
};

// Compiler
class WTCompiler {
public:
    static bool compileToWAV(const std::string& wt_file, const std::string& output_wav);
    static bool compileToPlayableWAV(const std::string& wt_file, const std::string& output_wav,
                                     double frequency, double duration_seconds);

private:
    static std::string readFile(const std::string& filename);
    static bool exportAudioWAV(const std::vector<float>& audio_data,
                               const std::string& filename,
                               int sample_rate);
};