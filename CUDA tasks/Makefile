# Makefile for compiling the project

# Compiler settings
CXX = nvcc
ARCH = sm_35
RDC = true
CXXFLAGS = -arch=$(ARCH) -rdc=$(RDC)

# Source files
SOURCES = preprocessing.cpp SSSPEdge.cu

# Header files
HEADERS = preprocessing.h

# Output executable
OUTPUT = output

all: $(OUTPUT)

$(OUTPUT): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(OUTPUT)

clean:
	rm -f $(OUTPUT)