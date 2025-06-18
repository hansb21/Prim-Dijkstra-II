import networkx as nx
import mst 
import matplotlib.pyplot as plt
import heapq

def prim(G, source):
    visited = {source}
    mst = nx.Graph()
    mst.add_node(source)
    heap = [] 
    
    for v in G.neighbors(source):
        weight = G[v][source].get('weight', 1)
        heapq.heappush(heap, (weight, source, v))

    while (len(heap) != 0 and visited != set(G.nodes())):
        
        (weight, u, v) = heapq.heappop(heap)

        if v in visited: continue

        mst.add_edge(u, v, weight=weight)
        visited.add(v)

        for w in G.neighbors(v):
            if w not in visited:
                edge_weight = G[v][w].get('weight', 1)
                heapq.heappush(heap, (edge_weight, v, w))

    return mst
        

def dijkstra(G, source):
    prev = {v: None for v in G.nodes()}
    dist = {v: float('inf') for v in G.nodes()}
    dist[source] = 0
    visited = set()
    heap = []

    heapq.heappush(heap, (0, source))

    while ((len(heap)) != 0):
        (_, u) = heapq.heappop(heap)

        if u in visited: continue 

        visited.add(u)

        for v in G.neighbors(u):
            new_dist = dist[u] + G[u][v].get('weight', 1)

            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, ((new_dist, v)))
    
    return dist, prev

def build_dijkstra_tree(prev, dist, source):
    G = nx.Graph()
    G.add_node(source)
    for v in prev:
        if prev[v] is not None:
            print(v)
            u = prev[v]
            edge_weight = dist[v] - dist[u]
            G.add_edge(u, v, weight=edge_weight)
    
    return G

def pfd(mst, alpha, source):
    visited = {source}
    mst = nx.Graph()
    mst.add_node(source)
    heap = [] 
    
    for v in G.neighbors(source):
        weight = G[v][source].get('weight', 1)
        heapq.heappush(heap, (weight, source, v))

    while (len(heap) != 0 and visited != set(G.nodes())):
        
        (weight, u, v) = heapq.heappop(heap)

        if v in visited: continue

        mst.add_edge(u, v, weight=weight)
        visited.add(v)

        for w in G.neighbors(v):
            if w not in visited:
                edge_weight = G[v][w].get('weight', 1)
                heapq.heappush(heap, (edge_weight, v, w))

    return mst


def pd_ii(mst, alpha):
    pass


def create_test_graph():
    G = nx.Graph()
    
    nodes = ['A', 'B', 'C', 'D', 'E', 'F']
    G.add_nodes_from(nodes)
    
    edges = [
        ('A', 'B', 4),
        ('A', 'C', 2),
        ('B', 'C', 1),
        ('B', 'D', 5),
        ('C', 'D', 8),
        ('C', 'E', 10),
        ('D', 'E', 2),
        ('D', 'F', 6),
        ('E', 'F', 3)
    ]
    
    for u, v, weight in edges:
        G.add_edge(u, v, weight=weight)
    
    return G



G = create_test_graph()
mst = prim(G, source='A')
print(mst)
(dist, prev) = dijkstra(mst, source='A')
spt = build_dijkstra_tree(prev, dist, source='A')
print(prev, dist)
nx.draw(G, with_labels=True, node_color='lightgreen')
nx.draw(mst, with_labels=True, node_color='lightblue')
nx.draw(spt, with_labels=True, node_color='red')
plt.show()
