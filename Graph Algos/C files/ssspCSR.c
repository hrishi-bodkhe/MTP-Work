#include<stdio.h>
#include<stdlib.h>
#include<limits.h>

// Node structure for adjacency list
typedef struct Node{
	int vertex;
	int wt;
	struct Node* next;
}Node;


// Graph structure with an array of adjacency lists
typedef struct Graph{
	int numVertices;
	Node** adjList;
}Graph;

// Function to create a new node
Node* createNode(int v, int weighted, int wt){
	Node* node = (Node*)malloc(sizeof(Node));
	node->vertex = v;
	node->next = NULL;

	if(weighted == 1) node->wt = wt;
	else wt = 1;

	return node;
}

// Function to create a graph with a given number of vertices
Graph* createGraph(int vertices){
	Graph* graph = (Graph*)malloc(sizeof(Graph));
	graph->numVertices = vertices;

	graph->adjList = (Node**)malloc(vertices * sizeof(Node*));

	for(int i = 0; i < vertices; ++i)
		graph->adjList[i] = NULL;

	return graph;
}

// Function to add an edge to an undirected graph
void addEdge(Graph* graph, int u, int v, int weighted, int directed, int wt){

	Node* node = createNode(v, weighted, wt);
	
	Node* temp = graph->adjList[u];
	if(!temp) graph->adjList[u] = node;
	else{
		while(temp->next) temp = temp->next;
		temp->next = node;
	}

	if(directed == 1) return;

	node = createNode(u, weighted, wt);
	
	temp = graph->adjList[v];
	if(!temp) graph->adjList[v] = node;
	else{
		while(temp->next) temp = temp->next;
		temp->next = node;
	}
}

// Function to print the adjacency list representation of a graph
void printGraph(Graph* graph, int weighted){
	if(!graph || !graph->adjList) return;
	
	for(int u = 0; u < graph->numVertices; ++u){
		Node* node = graph->adjList[u];

		printf("For %d: ", u);
		while(node){
			printf("(%d", node->vertex);
			if(weighted == 1) printf(", %d", node->wt);
			printf(")-> ");
			node = node->next;
		}
		printf("\n");
	}
}

void freeGraph(Graph* graph){
	if(!graph || !graph->adjList) return;

	for(int i = 0; i < graph->numVertices; ++i){
		Node* node = graph->adjList[i];

		while(node){
			Node* prev = node;
			node = node->next;
			prev->next = NULL;
			free(prev);
		}
	}

	free(graph->adjList);
	free(graph);
}

void constructAdjList(Graph* graph, int edges, FILE* file, int weighted, int directed){
	if(weighted == 1){
		int src, dest, wt;
		for(int i = 0; i < edges; ++i) {
			fscanf(file, "%d %d %d", &src, &dest, &wt);
			addEdge(graph, src, dest, weighted, directed, wt);
		}
	}else{
		int src, dest;
		for(int i = 0; i < edges; ++i) {
			fscanf(file, "%d %d", &src, &dest);
			addEdge(graph, src, dest, weighted, directed, -1);
		}
	}
}


void constructCSR(Graph* graph, int edges, int weighted, int directed, int* index, int* headVertex, int* weight){
	if(!graph || !graph->adjList) return;
	int n = graph->numVertices;
	int edgeCount = 0;
	
	if(weighted == 1){
		for(int i = 0; i < n; ++i){
			int src = i;
			index[src] = edgeCount;
			
			Node* node = graph->adjList[src];
			
			if(!node) {
                headVertex[edgeCount] = -1;
                weight[edgeCount] = -1;
                ++edgeCount;

            }

			while(node){
				int dest = node->vertex;
				int wt = node->wt;

				headVertex[edgeCount] = dest;
				weight[edgeCount] = wt;
				++edgeCount;

				node = node->next;
			}
		}
	}else{
		for(int i = 0; i < n; ++i){
			int src = i;
			index[src] = edgeCount;
			
			Node* node = graph->adjList[src];
			
			if(!node) {
                headVertex[edgeCount] = -1;
                ++edgeCount;
            }

			while(node){
				int dest = node->vertex;

				headVertex[edgeCount] = dest;
				++edgeCount;

				node = node->next;
			}
		}
	}

	index[n] = edgeCount;
	
}

void printCSR(int vertices, int edges, int* index, int* headVertex, int* weight, int weighted){
	printf("Index Array: ");
	for(int i = 0; i < vertices + 1; ++i) printf("%d ", index[i]);
	printf("\n");
	
	printf("Head Vertex Array: ");
	for(int i = 0; i < edges; ++i) printf("%d ", headVertex[i]);
	printf("\n");

	if(weighted == 0) return;

	printf("Weight Array: ");
	for(int i = 0; i < edges; ++i) printf("%d ", weight[i]);
	printf("\n");
}

void sssp(int vertices, int* index, int* headVertex, int* weight, int src){
	int n = vertices;
	int dist[n], parent[n];

	for(int i = 0; i < n; ++i){
		dist[i] = INT_MAX;
		parent[i] = -1;
	}

	dist[src] = 0;

	int changed;

	while(1){
		changed = 0;
		for(int u = 0; u < n; ++u){
			int startIdx = index[u];
            int endIdx = index[u + 1];

			for(int i = startIdx; i < endIdx; ++i){
				int v = headVertex[i];
				int wt = weight[i];

                if(v == -1) break;

				if(dist[v] > dist[u] + wt){
					dist[v] = dist[u] + wt;
					parent[v] = u;
					changed = 1;
				}
			}
		}

		if(changed == 0) break;
	}

	for(int i = 0; i < n; ++i)
		printf("%d ", dist[i]);
	printf("\n");
	for(int i = 0; i < n; ++i)
		printf("%d ", parent[i]);
}

int main(){
	FILE *file = fopen("input.txt", "r");
	if(file == NULL){
		printf("Could not open file\n");
		return 0;
	}

	int vertices, edges, directed, weighted;
	fscanf(file, "%d %d %d %d", &vertices, &edges, &directed, &weighted);
    int n = vertices;
	Graph* graph = createGraph(n);
	
	constructAdjList(graph, edges, file, weighted, directed);
	printGraph(graph, weighted);

	int* index = (int*)malloc((vertices + 1) * sizeof(int)); 
	int* headVertex = NULL;
	int* weight = NULL;

	if(directed == 1) {
		headVertex = (int*)malloc(edges * sizeof(int));
		if(weighted == 1) weight = (int*)malloc(edges * sizeof(int));
	}
	else{
		headVertex = (int*)malloc(2 * edges * sizeof(int));
		if(weighted == 1) weight = (int*)malloc(2 * edges * sizeof(int));
	}

	constructCSR(graph, edges, weighted, directed, index, headVertex, weight);

	if(directed == 0) edges *= 2;
	printCSR(vertices, edges, index, headVertex, weight, weighted);

    sssp(vertices, index, headVertex, weight, 0);

	freeGraph(graph);
	free(index);
	free(headVertex);
	free(weight);

	return 0;
}