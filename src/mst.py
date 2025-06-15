import networkx as nx
import matplotlib.pyplot as plt

def build_mst_per_net(design):
    G = nx.Graph()
    components = design['components']
    
    for comp_id, comp_data in components.items():
        G.add_node(comp_id, pos=(comp_data['x'], comp_data['y']))
    
    for _, net_data in design['nets'].items():
        net_components = net_data['components']
        if len(net_components) < 2:
            continue 


        subgraph = nx.Graph()
        for comp in net_components:
            x, y = components[comp]['x'], components[comp]['y']
            subgraph.add_node(comp, pos=(x, y))
        
        for i in range(len(net_components)):
            for j in range(i+1, len(net_components)):
                u, v = net_components[i], net_components[j]
                x1, y1 = components[u]['x'], components[u]['y']
                x2, y2 = components[v]['x'], components[v]['y']
                weight = abs(x1 - x2) + abs(y1 - y2)  
                subgraph.add_edge(u, v, weight=weight)
        
        net_mst = nx.minimum_spanning_tree(subgraph, algorithm='prim')
        for u, v, data in net_mst.edges(data=True):
            G.add_edge(u, v, weight=data['weight'])
    
    return G

def draw_manhattan_mst(G):
    plt.figure(figsize=(12, 8))
    pos = nx.get_node_attributes(G, 'pos')
    ax = plt.gca()
    
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y1], color='red', linewidth=1.5)  
        ax.plot([x2, x2], [y1, y2], color='red', linewidth=1.5)  
    
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("MST with Manhattan Routing")
    plt.tight_layout()
    plt.show()

