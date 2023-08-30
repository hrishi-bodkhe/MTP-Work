# Makefile for compiling the project

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -g

# Source files
SOURCES = bfs.cpp dfs.cpp graph.cpp apsp.cpp connectedComp.cpp scc.cpp sssp.cpp weaklycc.cpp mst.cpp mis.cpp pagerank.cpp triangleCount.cpp graphColor.cpp kcore.cpp betweennessCentrality.cpp

# Header files
HEADERS = preprocessing.h bfs.h dfs.h apsp.h connectedComp.h scc.h sssp.h weaklycc.h mst.h mis.h pagerank.h triangleCount.h graphColor.h kcore.h betweennessCentrality.h

# Output executable
OUTPUT = output

all: $(OUTPUT)

$(OUTPUT): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(OUTPUT)

clean:
	rm -f $(OUTPUT)
