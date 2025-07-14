import networkx as nx
import matplotlib.pyplot as plt
import pd

def build_mst_per_net(design):
    G = nx.Graph()
    components = design['components']
    
    for comp_id, comp_data in components.items():
        G.add_node(comp_id, pos=(comp_data['x'], comp_data['y']))
    
    for net_id, net_data in design['nets'].items():
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
                subgraph.add_edge(u, v, weight=weight, pos=[(x1, y1), (x2, y2)])
        nodes = list(subgraph.nodes())
      #  net_mst = nx.minimum_spanning_tree(subgraph, algorithm='prim')
        net_mst = pd.prim(subgraph, source=nodes[0])
        for u, v, data in net_mst.edges(data=True):
            G.add_edge(u, v, weight=data['weight'], pos=data['pos'], net_id=net_id)
        
    
    return G

def draw_manhattan_mst_on_axis(G, ax):
    ax.clear()
    
    # Pega as posições dos NÓS (não das arestas)
    node_pos = nx.get_node_attributes(G, 'pos')
    
    if not node_pos:
        ax.text(0.5, 0.5, "Erro: Nós sem posição", transform=ax.transAxes)
        return
    
    # Cores para as arestas
    edge_colors = plt.cm.tab10(range(len(G.edges())))
    
    # Desenha o roteamento Manhattan para cada aresta
    for i, (u, v) in enumerate(G.edges()):
        if u in node_pos and v in node_pos:
            x1, y1 = node_pos[u]
            x2, y2 = node_pos[v]
            color = edge_colors[i % 10]  # Cicla as cores se há muitas arestas
            
            # Desenha L-shape (Manhattan routing)
            ax.plot([x1, x2], [y1, y1], color=color, linewidth=2)  # Horizontal
            ax.plot([x2, x2], [y1, y2], color=color, linewidth=2)  # Vertical
        else:
            print(f"AVISO: Aresta {u}-{v} sem posições dos nós")
    
    # Desenha os nós
    nodes_to_draw = [n for n in G.nodes() if n in node_pos]
    
    if nodes_to_draw:
        nx.draw_networkx_nodes(G, node_pos, nodelist=nodes_to_draw, 
                              node_size=150, node_color='lightblue', ax=ax)
        nx.draw_networkx_labels(G, node_pos, labels={n: n for n in nodes_to_draw}, 
                               font_size=10, ax=ax)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_title("MST with Manhattan Routing")
    ax.set_aspect('equal')



def create_complete_graph(design):
    G = nx.Graph()
    components = design['components']
    
    for comp_id, comp_data in components.items():
        G.add_node(comp_id, pos=(comp_data['x'], comp_data['y']))
    
    comp_list = list(components.keys())
    for i in range(len(comp_list)):
        for j in range(i+1, len(comp_list)):
            u, v = comp_list[i], comp_list[j]
            x1, y1 = components[u]['x'], components[u]['y']
            x2, y2 = components[v]['x'], components[v]['y']
            weight = abs(x1 - x2) + abs(y1 - y2)
            G.add_edge(u, v, weight=weight, pos=[(x1, y1), (x2, y2)])
    
    return G
