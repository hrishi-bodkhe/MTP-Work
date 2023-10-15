# Makefile for compiling the project

# Compiler settings
CXX = nvcc

# Source files
SOURCES = preprocessing.cpp Task4.cu

# Header files
HEADERS = preprocessing.h

# Output executable
OUTPUT = output

all: $(OUTPUT)

$(OUTPUT): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(OUTPUT)

clean:
	rm -f $(OUTPUT)