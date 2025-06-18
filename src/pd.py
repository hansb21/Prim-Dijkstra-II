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
    dist = {}
    prev = {}
    for v in G.nodes():
        pass

def pd(mst, alpha):
    pass


def pd_ii(mst, alpha):
    pass


G = nx.Graph()
G.add_edges_from([
    (1, 2, {'weight': 2}),
    (3, 2, {'weight': 4}),
    (4, 2, {'weight': 1}),
    (1, 3, {'weight': 3}),
    (2, 3, {'weight': 1}),
    (3, 4, {'weight': 4})
])

mst = prim(G, source=1)
print(mst)
nx.draw(G, with_labels=True, node_color='lightgreen')
nx.draw(mst, with_labels=True, node_color='lightblue')
plt.show()
