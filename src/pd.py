import networkx as nx
import mst 
import matplotlib.pyplot as plt
import heapq
from itertools import combinations
from collections import defaultdict, deque

def prim(G, source):
    print(G.nodes(data=True))
    visited = {source}
    mst = nx.Graph()
    mst.add_node(source)
    mst.nodes[source].update(G.nodes[source])
    heap = [] 
    
    for v in G.neighbors(source):
        weight = G[v][source].get('weight', 1)
        heapq.heappush(heap, (weight, source, v))

    while (len(heap) != 0 and visited != set(G.nodes())):
        
        (weight, u, v ) = heapq.heappop(heap)

        if v in visited: continue
        if v not in mst.nodes():
            mst.add_node(v)
            if 'pos' in G.nodes[v]:
                mst.nodes[v]['pos'] = G.nodes[v]['pos']
           
            mst.nodes[v].update(G.nodes[v])
        edge_data = G[u][v].copy()
        mst.add_edge(u, v, **edge_data)
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
    
    tree = nx.Graph()
    tree.add_node(source)
    tree.nodes[source].update(G.nodes[source])

    heapq.heappush(heap, (0, source))

    while ((len(heap)) != 0):
        (current_dist, u) = heapq.heappop(heap)

        if u in visited: continue 

        visited.add(u)
        
        if prev[u] is not None:
            if u not in tree.nodes():
                tree.add_node(u)
                tree.nodes[u].update(G.nodes[u])
            
            edge_data = G[prev[u]][u].copy()
            tree.add_edge(prev[u], u, **edge_data)


        for v in G.neighbors(u):
            new_dist = dist[u] + G[u][v].get('weight', 1)

            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, ((new_dist, v)))
    return tree, dist, prev


def pd(G, alpha, source):
    prev = {v: None for v in G.nodes()}
    dist = {v: float('inf') for v in G.nodes()}
    dist[source] = 0
    visited = set()
    heap = []
    
    tree = nx.Graph()
    tree.add_node(source)
    tree.nodes[source].update(G.nodes[source])

    heapq.heappush(heap, (0, source))

    while ((len(heap)) != 0):
        (current_dist, u) = heapq.heappop(heap)

        if u in visited: continue 

        visited.add(u)
        
        if prev[u] is not None:
            if u not in tree.nodes():
                tree.add_node(u)
                tree.nodes[u].update(G.nodes[u])
            
            edge_data = G[prev[u]][u].copy()
            tree.add_edge(prev[u], u, **edge_data)


        for v in G.neighbors(u):
            new_dist = alpha * dist[u] + G[u][v].get('weight', 1)

            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, ((new_dist, v)))
    
    return tree, dist, prev

def candidateEdge(spanning_tree, e, D):
    u, v = e
    candidates = set()
    
    reachable_nodes = set()
    
    for node in spanning_tree.nodes():
        try:
            dist = nx.shortest_path_length(spanning_tree, u, node)
            if dist <= D:
                reachable_nodes.add(node)
        except nx.NetworkXNoPath:
            pass
    
    for node1 in reachable_nodes:
        for node2 in reachable_nodes:
            if node1 >= node2:  continue
                
            candidate_edge = (node1, node2)
            
            if candidate_edge == e or (node2, node1) == e: continue
                
            if spanning_tree.has_edge(node1, node2): continue
            
            candidates.add(candidate_edge)
    
    return candidates


def get_pl_from_source(G, source):
    pl = {}

    for node in G.nodes():
        if node == source:
            pl[node] = 0

        else:
            try:
                length = nx.shortest_path_length(G, source, node, weight='weight')
                pl[node] = length
            except nx.NetworkXNoPath:
                pl[node] = float('inf')

    return pl 

def get_md_from_source(G, source):

    if source not in G.nodes(): return {}
    
    source_pos = G.nodes[source].get('pos')
    if source_pos == None: return {}

    x, y = source_pos

    md = {}

    for node in G.nodes():
        if node == source:
            md[node] = 0
        else:
            pos = G.nodes[node].get('pos')
            if pos != None:
                n_x, n_y = pos
                md[node] = abs(x - n_x) + abs(y - n_y)
            else:
                md[node] = float('inf')

    return md

def detour_cost(G, source):
    pl = get_pl_from_source(G, source)
    md = get_md_from_source(G, source)
    
    sum_pl = sum(pl[node] for node in G.nodes() if node != source) 
    sum_md = sum(md[node] for node in G.nodes() if node != source) 
    
    cost = sum_pl - sum_md 

    return cost


def calculate_manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def flip_cost(G, G_flipped, edge, edge_, alpha, source):
    detour_edge = detour_cost(G, source)
    detour_edge_ = detour_cost(G_flipped, source)

    len_rem = G[edge[0]][edge[1]].get('weight', 0)
    len_ins = G_flipped[edge_[0]][edge_[1]].get('weight', 0)

    return alpha * (detour_edge_ - detour_edge) + (1 - alpha) * (len_ins - len_rem)


def create_flipped_tree(original_tree, old_edge, new_edge):
    flipped_tree = original_tree.copy()
    
    if flipped_tree.has_edge(old_edge[0], old_edge[1]):
        flipped_tree.remove_edge(old_edge[0], old_edge[1])
    else:
        return None
    
    components = list(nx.connected_components(flipped_tree))
    if len(components) != 2:
        return None
    
    comp1, comp2 = components
    
    if not ((new_edge[0] in comp1 and new_edge[1] in comp2) or 
            (new_edge[0] in comp2 and new_edge[1] in comp1)):
        return None
    
    pos1 = original_tree.nodes[new_edge[0]]['pos']
    pos2 = original_tree.nodes[new_edge[1]]['pos']
    weight = calculate_manhattan_distance(pos1, pos2)
    
    flipped_tree.add_edge(new_edge[0], new_edge[1], weight=weight)
    
    if nx.is_connected(flipped_tree) and nx.is_tree(flipped_tree):
        return flipped_tree
    else:
        return None
  


def pd_ii(G, alpha, source, D=1):
    tout = G.copy()
    # No início da função
  #  print(f"Árvore inicial - Conectada: {nx.is_connected(tout)}, É árvore: {nx.is_tree(tout)}")
 #   print(f"Nós: {len(tout.nodes())}, Arestas: {len(tout.edges())}")
    best_detour_cost = -1
    while True:

        best_detour_cost = 0
        for edge in tout.edges():
            candidates = candidateEdge(tout, edge, D)
            for edge_ in candidates:
                temp_tout = tout.copy()
                temp_tout.remove_edge(*edge)
                if not (nx.is_connected(temp_tout)):
                    components = list(nx.connected_components(temp_tout))
                    comp1, comp2 = components[0], components[1]
    
                    if (edge_[0] in comp1 and edge_[1] in comp2) or (edge_[0] in comp2 and edge_[1] in comp1):
                        u_pos = tout.nodes[edge_[0]]['pos']
                        v_pos = tout.nodes[edge_[1]]['pos']
                        md = abs(u_pos[0] - v_pos[0]) + abs(u_pos[1] - v_pos[1])
                      #  print(edge_)
                      #  print(f"x pos {u_pos}")
                      #  print(f"y pos {v_pos}")
                      #  print(f"md: {md}")
                        temp_tout.add_edge(edge_[0], edge_[1], weight=md)
                        if (nx.is_connected(temp_tout) and nx.is_tree(temp_tout)):
            #                print("novo grafo é arvore conectada")
                            atual_detour_cost = flip_cost(tout, temp_tout, edge, edge_, alpha=alpha, source=source)
              #              print(f" detour cost = {atual_detour_cost}")
                            
                            if atual_detour_cost < best_detour_cost:
                                best_detour_cost = atual_detour_cost
                                best_flipped_tree = temp_tout.copy()
                    else:
                        continue

        if best_detour_cost < 0:
            if nx.is_connected(best_flipped_tree) and nx.is_tree(best_flipped_tree):
                tout = best_flipped_tree.copy()
        else:
            break

    print("finalizando PD-2.....")
    return tout

