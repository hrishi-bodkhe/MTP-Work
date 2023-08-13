#include <stdio.h>
#include <limits.h>
#include <stdlib.h>

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
	if (!graph || !graph->adjList)
		return;

	for (int u = 0; u < graph->numVertices; ++u)
	{
		Node *node = graph->adjList[u];

		printf("For %d: ", u);
		while (node)
		{
			printf("%d->", node->vertex);
			node = node->next;
		}
		printf("\n");
	}
}

void freeGraph(Graph *graph)
{
	if (!graph || !graph->adjList)
		return;

	for (int i = 0; i < graph->numVertices; ++i)
	{
		Node *node = graph->adjList[i];

		while (node)
		{
			Node *prev = node;
			node = node->next;
			prev->next = NULL;
			free(prev);
		}
	}

	free(graph->adjList);
	free(graph);
}

void traverse(int u, Graph *graph, int *compNum, int color[], int component[])
{
	color[u] = 1;
	component[u] = *compNum;

	Node *node = graph->adjList[u];
	while (node)
	{
		int v = node->vertex;
		if (color[v] == 0)
		{
			traverse(v, graph, compNum, color, component);
		}

		node = node->next;
	}

	color[u] = 2;
}

void dfs(Graph *graph)
{
	int clock = 0;
	int count = 0;

	int n = graph->numVertices;
	int color[n], component[n];
	for (int i = 0; i < n; ++i)
	{
		color[i] = 0;
		component[i] = -1;
	}
	for (int u = 0; u < n; ++u)
	{
		if (color[u] == 0)
		{
			traverse(u, graph, &count, color, component);
			++count;
		}
	}

	for (int i = 0; i < n; ++i)
	{
		printf("%d is in component: %d\n", i, component[i]);
	}
}

int main()
{
	int n = 12;
	Graph *graph = createGraph(n);

	addEdge(graph, 0, 6);
	addEdge(graph, 6, 7);
	addEdge(graph, 1, 7);
	addEdge(graph, 2, 8);
	addEdge(graph, 3, 9);
	addEdge(graph, 3, 8);
	addEdge(graph, 4, 5);
	addEdge(graph, 4, 10);
	addEdge(graph, 5, 11);
	addEdge(graph, 10, 11);

	dfs(graph);

	printGraph(graph);

	return 0;
}