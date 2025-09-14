CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# Targets
WAVETABLE_TARGET = wavetable_example
COMPILER_TARGET = render

# Sources
WAVETABLE_SOURCES = wavetable_generator.cpp example.cpp
COMPILER_SOURCES = wavetable_generator.cpp wt_compiler.cpp render.cpp

# Objects
WAVETABLE_OBJECTS = $(WAVETABLE_SOURCES:.cpp=.o)
COMPILER_OBJECTS = $(COMPILER_SOURCES:.cpp=.o)

all: $(WAVETABLE_TARGET) $(COMPILER_TARGET)

$(WAVETABLE_TARGET): $(WAVETABLE_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(WAVETABLE_TARGET) $(WAVETABLE_OBJECTS)

$(COMPILER_TARGET): $(COMPILER_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(COMPILER_TARGET) $(COMPILER_OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(WAVETABLE_TARGET) $(COMPILER_TARGET) wavetables/*.wav wavetables/*.asd wavetables/tests/*.wav wavetables/tests/*.asd

.PHONY: clean all