#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

// Node structure for adjacency list
typedef struct Node
{
	int vertex;
	struct Node *next;
} Node;

// Graph structure with an array of adjacency lists
typedef struct Graph
{
	int numVertices;
	Node **adjList;
} Graph;

// Function to create a new node
Node *createNode(int v)
{
	Node *node = (Node *)malloc(sizeof(Node));
	node->vertex = v;
	node->next = NULL;

	return node;
}

// Function to create a graph with a given number of vertices
Graph *createGraph(int vertices)
{
	Graph *graph = (Graph *)malloc(sizeof(Graph));
	graph->numVertices = vertices;

	graph->adjList = (Node **)malloc(vertices * sizeof(Node *));

	for (int i = 0; i < vertices; ++i)
		graph->adjList[i] = NULL;

	return graph;
}

// Function to add an edge to an undirected graph
void addEdge(Graph *graph, int u, int v)
{
	Node *node = createNode(v);
	node->next = graph->adjList[u];
	graph->adjList[u] = node;

	node = createNode(u);
	node->next = graph->adjList[v];
	graph->adjList[v] = node;
}

// Function to print the adjacency list representation of a graph
void printGraph(Graph *graph)
{
	for (int u = 0; u < graph->numVertices; ++u)
	{
		Node *node = graph->adjList[u];

		printf("For %d: ", u);
		while (node)
		{
			printf("%d=> ", node->vertex);
			node = node->next;
		}
		printf("\n");
	}
}

void traverse(int u, Graph *graph, int color[], int *clock, int starttime[], int fintime[])
{
	*clock = *clock + 1;
	starttime[u] = *clock;
	color[u] = 1;

	Node *node = graph->adjList[u];

	while (node)
	{
		int v = node->vertex;
		if (color[v] == 0)
			traverse(v, graph, color, clock, starttime, fintime);

		node = node->next;
	}

	color[u] = 2;
	*clock = *clock + 1;
	fintime[u] = *clock;
}

void dfs(Graph *graph, int n, int color[], int starttime[], int fintime[])
{
	// 0 - white
	// 1 - grey
	// 2 - black

	int clock = 0;

	for (int u = 0; u < n; ++u)
	{
		color[u] = 0;
		starttime[u] = -1;
		fintime[u] = -1;
	}

	for (int u = 0; u < n; ++u)
	{
		if (color[u] == 0)
			traverse(u, graph, color, &clock, starttime, fintime);
	}
}

int main()
{
	int n = 6;
	Graph *graph = createGraph(n);

	addEdge(graph, 0, 3);
	addEdge(graph, 1, 3);
	addEdge(graph, 1, 4);
	addEdge(graph, 2, 5);
	addEdge(graph, 3, 4);
	addEdge(graph, 4, 5);

	int color[n], starttime[n], fintime[n];
	dfs(graph, n, color, starttime, fintime);

	for (int i = 0; i < n; ++i)
		printf("%d ", starttime[i]);

	printf("\n");

	for (int i = 0; i < n; ++i)
		printf("%d ", fintime[i]);

	return 0;
}