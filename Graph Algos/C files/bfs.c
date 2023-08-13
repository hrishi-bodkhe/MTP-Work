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

	int parent[n];
	int dist[n];

	for (int i = 0; i < n; ++i)
	{
		dist[i] = INT_MAX;
		parent[i] = -1;
	}

	dist[0] = 0;

	int changed;
	while (1)
	{
		changed = 0;

		for (int u = 0; u < n; ++u)
		{
			Node *node = graph->adjList[u];
			while (node)
			{
				int v = node->vertex;
				if (dist[u] != INT_MAX && dist[v] > dist[u] + 1)
				{
					dist[v] = dist[u] + 1;
					parent[v] = u;
					changed = 1;
				}

				node = node->next;
			}
		}

		if (changed == 0)
			break;
	}

	printGraph(graph);
	for (int i = 0; i < n; ++i)
	{
		printf("%d ", dist[i]);
	}

	printf("\n");

	for (int i = 0; i < n; ++i)
	{
		printf("%d ", parent[i]);
	}

	return 0;
}