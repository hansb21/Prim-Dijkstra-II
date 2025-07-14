import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Dict, Tuple, Optional, Any

def unified_pd_analysis(pd_tree, pd2_tree, source, alpha=0.5, 
                       design=None, verbose=True, plot=True):
    start_time = time.time()
    
    if verbose:
        print("=" * 80)
        print("ANÁLISE UNIFICADA: PD vs PD-II")
        print("=" * 80)
        print(f"Parâmetro α = {alpha}")
        print(f"Source = {source}")
        print(f"Nós analisados = {len(pd_tree.nodes())}")
        print()
    
    def calculate_comprehensive_metrics(tree, tree_name):
        metrics = {}
        
        # Validação básica
        if not nx.is_connected(tree):
            print(f" {tree_name} não está conectada!")
        if not nx.is_tree(tree):
            print(f" {tree_name} não é uma árvore!")
        
        # 1. Wirelength
        metrics['wirelength'] = sum(tree[u][v].get('weight', 0) for u, v in tree.edges())
        
        # 2. Path lengths
        path_lengths = {}
        for node in tree.nodes():
            if node == source:
                path_lengths[node] = 0
            else:
                try:
                    pl = nx.shortest_path_length(tree, source, node, weight='weight')
                    path_lengths[node] = pl
                except nx.NetworkXNoPath:
                    path_lengths[node] = float('inf')
        
        # 3. Manhattan distances
        source_pos = tree.nodes[source]['pos']
        manhattan_distances = {}
        for node in tree.nodes():
            if node == source:
                manhattan_distances[node] = 0
            else:
                node_pos = tree.nodes[node]['pos']
                md = abs(source_pos[0] - node_pos[0]) + abs(source_pos[1] - node_pos[1])
                manhattan_distances[node] = md
        
        # 4. Métricas derivadas
        sink_nodes = [n for n in tree.nodes() if n != source]
        valid_pls = [path_lengths[node] for node in sink_nodes if path_lengths[node] != float('inf')]
        
        if valid_pls:
            metrics['total_path_length'] = sum(valid_pls)
            metrics['avg_path_length'] = sum(valid_pls) / len(valid_pls)
            metrics['max_path_length'] = max(valid_pls)
            metrics['min_path_length'] = min(valid_pls)
        else:
            metrics['total_path_length'] = 0
            metrics['avg_path_length'] = 0
            metrics['max_path_length'] = 0
            metrics['min_path_length'] = 0
        
        # 5. Detour costs
        detours = []
        detour_per_node = {}
        for node in sink_nodes:
            if path_lengths[node] != float('inf'):
                detour = path_lengths[node] - manhattan_distances[node]
                detours.append(detour)
                detour_per_node[node] = detour
        
        if detours:
            metrics['total_detour_cost'] = sum(detours)
            metrics['avg_detour_cost'] = sum(detours) / len(detours)
            metrics['max_detour'] = max(detours)
            metrics['min_detour'] = min(detours)
        else:
            metrics['total_detour_cost'] = 0
            metrics['avg_detour_cost'] = 0
            metrics['max_detour'] = 0
            metrics['min_detour'] = 0
        
        # 6. Dados detalhados
        metrics['path_lengths'] = path_lengths
        metrics['manhattan_distances'] = manhattan_distances
        metrics['detour_per_node'] = detour_per_node
        metrics['num_nodes'] = len(tree.nodes())
        metrics['num_edges'] = len(tree.edges())
        
        return metrics
    
    # Calcula métricas para ambas as árvores
    pd_metrics = calculate_comprehensive_metrics(pd_tree, "PD")
    pd2_metrics = calculate_comprehensive_metrics(pd2_tree, "PD-II")
    
    
    improvements = {}
    comparison_metrics = [
        'wirelength', 'total_path_length', 'avg_path_length', 'max_path_length',
        'total_detour_cost', 'avg_detour_cost', 'max_detour'
    ]
    
    for metric in comparison_metrics:
        pd_val = pd_metrics[metric]
        pd2_val = pd2_metrics[metric]
        
        if pd_val != 0:
            improvement = ((pd_val - pd2_val) / pd_val) * 100
        else:
            improvement = 0 if pd2_val == 0 else -100
        
        improvements[metric] = improvement
    
    pd_objective = alpha * pd_metrics['total_detour_cost'] + (1 - alpha) * pd_metrics['wirelength']
    pd2_objective = alpha * pd2_metrics['total_detour_cost'] + (1 - alpha) * pd2_metrics['wirelength']
    
    objective_improvement = ((pd_objective - pd2_objective) / pd_objective) * 100 if pd_objective != 0 else 0
    
    cost_analysis = {}
    
    # Complexidade estimada (baseada no número de operações)
    pd_complexity_estimate = len(pd_tree.nodes()) ** 2  # O(n²) para PD
    pd2_complexity_estimate = len(pd2_tree.nodes()) ** 3  # O(n³) para PD-II no pior caso
    
    cost_analysis['estimated_pd_ops'] = pd_complexity_estimate
    cost_analysis['estimated_pd2_ops'] = pd2_complexity_estimate
    cost_analysis['complexity_ratio'] = pd2_complexity_estimate / pd_complexity_estimate if pd_complexity_estimate > 0 else 0
    
    def classify_result():
        wl_imp = improvements['wirelength']
        detour_imp = improvements['total_detour_cost']
        obj_imp = objective_improvement
        
        if obj_imp > 10:
            return "EXCELENTE", "Grande melhoria na função objetivo"
        elif obj_imp > 5:
            return "MUITO BOM", "Boa melhoria na função objetivo"
        elif obj_imp > 0:
            return "BOM", "Melhoria positiva na função objetivo"
        elif obj_imp > -5:
            return "NEUTRO", "Resultado similar ao PD original"
        else:
            return "PIOR", "PD-II não conseguiu melhorar"
    
    classification, classification_desc = classify_result()
    
    if verbose:
        print("MÉTRICAS PRINCIPAIS:")
        print("-" * 60)
        
        metric_display = [
            ('wirelength', 'Wirelength (WL)', '📏'),
            ('total_detour_cost', 'Total Detour Cost', '🔄'),
            ('avg_path_length', 'Avg Path Length', '📊'),
            ('max_path_length', 'Max Path Length', '📈'),
            ('avg_detour_cost', 'Avg Detour Cost', '⚡')
        ]
        
        for metric_key, metric_name, emoji in metric_display:
            pd_val = pd_metrics[metric_key]
            pd2_val = pd2_metrics[metric_key]
            imp = improvements[metric_key]
            
            
            print(f"{emoji} {metric_name:20s}: PD={pd_val:8.2f} → PD-II={pd2_val:8.2f} ")
        
        print()
        print("🎯 FUNÇÃO OBJETIVO DO PAPER:")
        print("-" * 40)
        print(f"PD:    α×{pd_metrics['total_detour_cost']:.1f} + (1-α)×{pd_metrics['wirelength']:.1f} = {pd_objective:.2f}")
        print(f"PD-II: α×{pd2_metrics['total_detour_cost']:.1f} + (1-α)×{pd2_metrics['wirelength']:.1f} = {pd2_objective:.2f}")
        print(f"Melhoria: {objective_improvement:+.2f}%")
        print()
        
        print("🏅 CLASSIFICAÇÃO FINAL:")
        print(f"{classification} - {classification_desc}")
        print()
        
        print("💰 ANÁLISE DE CUSTOS:")
        print("-" * 30)
        print(f"Operações estimadas PD:    {cost_analysis['estimated_pd_ops']:,}")
        print(f"Operações estimadas PD-II: {cost_analysis['estimated_pd2_ops']:,}")
        print(f"Fator de complexidade:     {cost_analysis['complexity_ratio']:.1f}x")
        
        if objective_improvement > 0:
            efficiency = objective_improvement / cost_analysis['complexity_ratio']
            print(f"Eficiência (melhoria/custo): {efficiency:.2f}%")
        print()
    
    
    if verbose:
        print("🔍 ANÁLISE DETALHADA POR NÓ (Top 10 maiores melhorias):")
        print("-" * 75)
        print(f"{'Nó':4s} | {'PD PL':8s} | {'PD-II PL':8s} | {'Melhoria':9s} | {'Detour PD':10s} | {'Detour PD-II':10s}")
        print("-" * 75)
        
        node_improvements = []
        for node in pd_metrics['path_lengths']:
            if node != source and pd_metrics['path_lengths'][node] != float('inf'):
                pd_pl = pd_metrics['path_lengths'][node]
                pd2_pl = pd2_metrics['path_lengths'][node]
                improvement = ((pd_pl - pd2_pl) / pd_pl) * 100 if pd_pl != 0 else 0
                node_improvements.append((node, improvement, pd_pl, pd2_pl))
        
        # Ordena por melhoria e mostra top 10
        node_improvements.sort(key=lambda x: x[1], reverse=True)
        for node, improvement, pd_pl, pd2_pl in node_improvements[:10]:
            pd_detour = pd_metrics['detour_per_node'].get(node, 0)
            pd2_detour = pd2_metrics['detour_per_node'].get(node, 0)
            
            print(f"{node:4s} | {pd_pl:8.2f} | {pd2_pl:8.2f} | {improvement:+8.1f}% | "
                  f"{pd_detour:10.2f} | {pd2_detour:10.2f}")
        print()
    
    
    figure = None
    if plot:
        figure = create_comprehensive_plots(pd_metrics, pd2_metrics, improvements, 
                                          objective_improvement, alpha)
    
    
    analysis_time = time.time() - start_time
    
    results = {
        'pd_metrics': pd_metrics,
        'pd2_metrics': pd2_metrics,
        'improvements': improvements,
        'objective_improvement': objective_improvement,
        'pd_objective': pd_objective,
        'pd2_objective': pd2_objective,
        'classification': classification,
        'classification_desc': classification_desc,
        'cost_analysis': cost_analysis,
        'alpha': alpha,
        'source': source,
        'analysis_time': analysis_time,
        'figure': figure,
        'summary': {
            'is_better': objective_improvement > 0,
            'wl_change': improvements['wirelength'],
            'detour_change': improvements['total_detour_cost'],
            'main_tradeoff': f"WL {improvements['wirelength']:+.1f}% vs Detour {improvements['total_detour_cost']:+.1f}%"
        }
    }
    
    if verbose:
        print(f"⏱️ Tempo de análise: {analysis_time:.3f}s")
        print("=" * 80)
    
    return results


def create_comprehensive_plots(pd_metrics, pd2_metrics, improvements, 
                             objective_improvement, alpha):
    
    fig = plt.figure(figsize=(16, 12))
    
    # Layout: 2x3 subplots
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Métricas principais (barras)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['wirelength', 'total_detour_cost', 'avg_path_length', 'max_path_length']
    metric_names = ['Wirelength', 'Total Detour', 'Avg PL', 'Max PL']
    
    pd_values = [pd_metrics[m] for m in metrics]
    pd2_values = [pd2_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, pd_values, width, label='PD', alpha=0.8, color='lightcoral')
    ax1.bar(x + width/2, pd2_values, width, label='PD-II', alpha=0.8, color='lightblue')
    
    ax1.set_title('Comparação de Métricas Principais')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Melhorias percentuais
    ax2 = fig.add_subplot(gs[0, 1])
    improvement_values = [improvements[m] for m in metrics]
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in improvement_values]
    
    bars = ax2.bar(range(len(metrics)), improvement_values, color=colors, alpha=0.7)
    ax2.set_title('Melhorias Percentuais (PD-II vs PD)')
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels(metric_names, rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Melhoria (%)')
    ax2.grid(True, alpha=0.3)
    
    # Adiciona valores nas barras
    for bar, val in zip(bars, improvement_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # Plot 3: Função objetivo
    ax3 = fig.add_subplot(gs[1, 0])
    obj_data = [pd_metrics['total_detour_cost'], pd2_metrics['total_detour_cost'],
                pd_metrics['wirelength'], pd2_metrics['wirelength']]
    obj_labels = ['PD Detour', 'PD-II Detour', 'PD WL', 'PD-II WL']
    obj_colors = ['lightcoral', 'lightblue', 'coral', 'skyblue']
    
    ax3.bar(obj_labels, obj_data, color=obj_colors, alpha=0.8)
    ax3.set_title(f'Componentes da Função Objetivo (α={alpha})')
    ax3.set_ylabel('Valor')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distribuição de detours
    ax4 = fig.add_subplot(gs[1, 1])
    pd_detours = [d for d in pd_metrics['detour_per_node'].values() if d != float('inf')]
    pd2_detours = [d for d in pd2_metrics['detour_per_node'].values() if d != float('inf')]
    
    if pd_detours and pd2_detours:
        bins = max(10, min(len(pd_detours)//2, 20))
        ax4.hist([pd_detours, pd2_detours], bins=bins, alpha=0.7, 
                label=['PD', 'PD-II'], color=['lightcoral', 'lightblue'])
        ax4.set_title('Distribuição de Detour Costs')
        ax4.set_xlabel('Detour Cost')
        ax4.set_ylabel('Número de Nós')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Path lengths comparação nó a nó
    ax5 = fig.add_subplot(gs[2, :])
    nodes = sorted([n for n in pd_metrics['path_lengths'].keys() 
                   if n != pd_metrics.get('source', '') and pd_metrics['path_lengths'][n] != float('inf')])
    
    if nodes:
        pd_pls = [pd_metrics['path_lengths'][n] for n in nodes]
        pd2_pls = [pd2_metrics['path_lengths'][n] for n in nodes]
        
        x_pos = range(len(nodes))
        ax5.plot(x_pos, pd_pls, 'o-', label='PD', alpha=0.8, color='lightcoral', markersize=4)
        ax5.plot(x_pos, pd2_pls, 's-', label='PD-II', alpha=0.8, color='lightblue', markersize=4)
        ax5.set_title('Path Lengths por Nó')
        ax5.set_xlabel('Nós (ordenados)')
        ax5.set_ylabel('Path Length')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Mostra apenas alguns labels de nós para não ficar bagunçado
        step = max(1, len(nodes) // 10)
        ax5.set_xticks(x_pos[::step])
        ax5.set_xticklabels([nodes[i] for i in range(0, len(nodes), step)], rotation=45)
    
    plt.suptitle(f'Análise Comparativa Completa: PD vs PD-II\n'
                f'Melhoria na Função Objetivo: {objective_improvement:+.2f}%', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return fig


# Função de conveniência para análise rápida
def quick_pd_comparison(pd_tree, pd2_tree, source, alpha=0.5):
    """Análise rápida sem outputs verbosos nem gráficos"""
    results = unified_pd_analysis(pd_tree, pd2_tree, source, alpha, 
                                verbose=False, plot=False)
    
    print(f"📊 Resumo Rápido (α={alpha}):")
    print(f"   WL: {results['improvements']['wirelength']:+.1f}%")
    print(f"   Detour: {results['improvements']['total_detour_cost']:+.1f}%") 
    print(f"   Objetivo: {results['objective_improvement']:+.1f}%")
    print(f"   Status: {results['classification']}")
    
    return results

def calculate_tree_metrics(G, source):
    metrics = {}
    
    # 1. Wirelength (WL) - soma dos pesos das arestas
    total_wl = sum(G[u][v].get('weight', 0) for u, v in G.edges())
    metrics['wirelength'] = total_wl
    
    # 2. Path lengths de cada nó até a source
    path_lengths = {}
    for node in G.nodes():
        if node == source:
            path_lengths[node] = 0
        else:
            try:
                pl = nx.shortest_path_length(G, source, node, weight='weight')
                path_lengths[node] = pl
            except nx.NetworkXNoPath:
                path_lengths[node] = float('inf')
    
    # 3. Manhattan distances de cada nó até a source
    source_pos = G.nodes[source]['pos']
    manhattan_distances = {}
    for node in G.nodes():
        if node == source:
            manhattan_distances[node] = 0
        else:
            node_pos = G.nodes[node]['pos']
            md = abs(source_pos[0] - node_pos[0]) + abs(source_pos[1] - node_pos[1])
            manhattan_distances[node] = md
    
    # 4. Métricas derivadas
    sink_nodes = [n for n in G.nodes() if n != source]
    
    # Total path length
    total_pl = sum(path_lengths[node] for node in sink_nodes if path_lengths[node] != float('inf'))
    metrics['total_path_length'] = total_pl
    
    # Máximo path length (radius)
    max_pl = max(path_lengths[node] for node in sink_nodes if path_lengths[node] != float('inf'))
    metrics['max_path_length'] = max_pl
    
    # Average path length
    valid_pls = [path_lengths[node] for node in sink_nodes if path_lengths[node] != float('inf')]
    avg_pl = sum(valid_pls) / len(valid_pls) if valid_pls else 0
    metrics['avg_path_length'] = avg_pl
    
    # Total detour cost
    total_detour = sum(path_lengths[node] - manhattan_distances[node] 
                      for node in sink_nodes 
                      if path_lengths[node] != float('inf'))
    metrics['total_detour_cost'] = total_detour
    
    # Average detour cost
    valid_detours = [path_lengths[node] - manhattan_distances[node] 
                    for node in sink_nodes 
                    if path_lengths[node] != float('inf')]
    avg_detour = sum(valid_detours) / len(valid_detours) if valid_detours else 0
    metrics['avg_detour_cost'] = avg_detour
    
    # Maximum detour
    max_detour = max(valid_detours) if valid_detours else 0
    metrics['max_detour'] = max_detour
    
    # Salva os dados detalhados para análise
    metrics['path_lengths'] = path_lengths
    metrics['manhattan_distances'] = manhattan_distances
    metrics['detour_per_node'] = {node: path_lengths[node] - manhattan_distances[node] 
                                 for node in sink_nodes}
    
    return metrics


def compare_pd_algorithms(pd_tree, pd2_tree, source, alpha):
    print("=" * 60)
    print("ANÁLISE COMPARATIVA: PD vs PD-II")
    print("=" * 60)
    
    # Calcula métricas para ambas as árvores
    pd_metrics = calculate_tree_metrics(pd_tree, source)
    pd2_metrics = calculate_tree_metrics(pd2_tree, source)
    
    print(f"Parâmetro α = {alpha}")
    print(f"Nós: {len(pd_tree.nodes())}, Source: {source}")
    print()
    
    # Comparação das métricas principais
    print("MÉTRICAS PRINCIPAIS:")
    print("-" * 40)
    
    metrics_to_compare = [
        ('wirelength', 'Wirelength (WL)', 'menor é melhor'),
        ('total_path_length', 'Total Path Length', 'menor é melhor'),
        ('avg_path_length', 'Avg Path Length', 'menor é melhor'),
        ('max_path_length', 'Max Path Length', 'menor é melhor'),
        ('total_detour_cost', 'Total Detour Cost', 'menor é melhor'),
        ('avg_detour_cost', 'Avg Detour Cost', 'menor é melhor'),
        ('max_detour', 'Max Detour', 'menor é melhor')
    ]
    
    improvements = {}
    
    for metric_key, metric_name, direction in metrics_to_compare:
        pd_value = pd_metrics[metric_key]
        pd2_value = pd2_metrics[metric_key]
        
        # Calcula melhoria percentual
        if pd_value != 0:
            improvement_pct = ((pd_value - pd2_value) / pd_value) * 100
        else:
            improvement_pct = 0
        
        improvements[metric_key] = improvement_pct
        
        # Determina se melhorou
        if improvement_pct > 0:
            status = "MELHOR"
        elif improvement_pct < 0:
            status = "PIOR"
        else:
            status = "IGUAL"
        
        print(f"{metric_name:20s}: PD={pd_value:8.2f}, PD-II={pd2_value:8.2f}, "
              f"Δ={improvement_pct:+6.2f}% {status}")
    
    print()
    
    # Análise do trade-off
    print("ANÁLISE DO TRADE-OFF:")
    print("-" * 40)
    
    wl_improvement = improvements['wirelength']
    detour_improvement = improvements['total_detour_cost']
    pl_improvement = improvements['avg_path_length']
    
    if wl_improvement > 0 and detour_improvement > 0:
        print(" EXCELENTE: Melhoria em WL E detour cost!")
    elif wl_improvement > 0 and pl_improvement > 0:
        print(" EXCELENTE: Melhoria em WL E path length!")
    elif abs(wl_improvement) < 5 and detour_improvement > 10:
        print(" BOM: Grande melhoria em detour com pequeno impacto em WL")
    elif abs(detour_improvement) < 5 and wl_improvement > 10:
        print(" BOM: Grande melhoria em WL com pequeno impacto em detour")
    elif wl_improvement > 0 or detour_improvement > 0:
        print(" MODERADO: Melhoria em uma métrica")
    else:
        print(" RUIM: Sem melhorias significativas")
    
    print()
    
    # Cálculo da função objetivo do paper
    print("FUNÇÃO OBJETIVO DO PAPER:")
    print("-" * 40)
    
    pd_objective = alpha * pd_metrics['total_detour_cost'] + (1 - alpha) * pd_metrics['wirelength']
    pd2_objective = alpha * pd2_metrics['total_detour_cost'] + (1 - alpha) * pd2_metrics['wirelength']
    
    objective_improvement = ((pd_objective - pd2_objective) / pd_objective) * 100 if pd_objective != 0 else 0
    
    print(f"PD objetivo   = {alpha:.1f} × {pd_metrics['total_detour_cost']:.1f} + {1-alpha:.1f} × {pd_metrics['wirelength']:.1f} = {pd_objective:.2f}")
    print(f"PD-II objetivo= {alpha:.1f} × {pd2_metrics['total_detour_cost']:.1f} + {1-alpha:.1f} × {pd2_metrics['wirelength']:.1f} = {pd2_objective:.2f}")
    print(f"Melhoria na função objetivo: {objective_improvement:+.2f}%")
    
    if objective_improvement > 0:
        print("PD-II é superior segundo a função objetivo do paper!")
    else:
        print(" PD-II não melhorou a função objetivo")
    
    print()
    
    return pd_metrics, pd2_metrics, improvements


def plot_comparison_charts(pd_metrics, pd2_metrics, improvements):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Gráfico 1: Métricas principais
    metrics = ['wirelength', 'total_detour_cost', 'avg_path_length', 'max_path_length']
    metric_names = ['Wirelength', 'Total Detour', 'Avg Path Length', 'Max Path Length']
    
    pd_values = [pd_metrics[m] for m in metrics]
    pd2_values = [pd2_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, pd_values, width, label='PD', alpha=0.8, color='lightcoral')
    ax1.bar(x + width/2, pd2_values, width, label='PD-II', alpha=0.8, color='lightblue')
    
    ax1.set_xlabel('Métricas')
    ax1.set_ylabel('Valores')
    ax1.set_title('Comparação de Métricas: PD vs PD-II')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Melhorias percentuais
    improvement_values = [improvements[m] for m in metrics]
    colors = ['green' if x > 0 else 'red' for x in improvement_values]
    
    ax2.bar(range(len(metrics)), improvement_values, color=colors, alpha=0.7)
    ax2.set_xlabel('Métricas')
    ax2.set_ylabel('Melhoria (%)')
    ax2.set_title('Melhorias Percentuais do PD-II')
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels(metric_names, rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Distribuição de detours por nó
    pd_detours = list(pd_metrics['detour_per_node'].values())
    pd2_detours = list(pd2_metrics['detour_per_node'].values())
    
    ax3.hist([pd_detours, pd2_detours], bins=10, alpha=0.7, 
             label=['PD', 'PD-II'], color=['lightcoral', 'lightblue'])
    ax3.set_xlabel('Detour Cost')
    ax3.set_ylabel('Número de Nós')
    ax3.set_title('Distribuição de Detour Costs')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Path lengths por nó
    nodes = sorted(pd_metrics['path_lengths'].keys())
    nodes = [n for n in nodes if pd_metrics['path_lengths'][n] != float('inf')]
    
    pd_pls = [pd_metrics['path_lengths'][n] for n in nodes]
    pd2_pls = [pd2_metrics['path_lengths'][n] for n in nodes]
    
    ax4.plot(pd_pls, 'o-', label='PD', alpha=0.7, color='lightcoral')
    ax4.plot(pd2_pls, 's-', label='PD-II', alpha=0.7, color='lightblue')
    ax4.set_xlabel('Nós (ordenados)')
    ax4.set_ylabel('Path Length')
    ax4.set_title('Path Lengths por Nó')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def detailed_node_analysis(pd_metrics, pd2_metrics, source):
    print("\nANÁLISE DETALHADA POR NÓ:")
    print("-" * 70)
    print(f"{'Nó':4s} | {'PD PL':8s} | {'PD-II PL':10s} | {'Melhoria':9s} | {'Detour PD':10s} | {'Detour PD-II':12s}")
    print("-" * 70)
    
    nodes = sorted([n for n in pd_metrics['path_lengths'].keys() if n != source])
    
    for node in nodes:
        pd_pl = pd_metrics['path_lengths'][node]
        pd2_pl = pd2_metrics['path_lengths'][node]
        
        if pd_pl != float('inf') and pd2_pl != float('inf'):
            improvement = ((pd_pl - pd2_pl) / pd_pl) * 100 if pd_pl != 0 else 0
            pd_detour = pd_metrics['detour_per_node'][node]
            pd2_detour = pd2_metrics['detour_per_node'][node]
            
            print(f"{node:4s} | {pd_pl:8.2f} | {pd2_pl:10.2f} | {improvement:+8.1f}% | "
                  f"{pd_detour:10.2f} | {pd2_detour:12.2f}")


def full_comparison_analysis(pd_tree, pd2_tree, source, alpha=0.5):
    pd_metrics, pd2_metrics, improvements = compare_pd_algorithms(pd_tree, pd2_tree, source, alpha)
    
    # Análise detalhada por nó
    detailed_node_analysis(pd_metrics, pd2_metrics, source)
    
    # Gráficos comparativos
    fig = plot_comparison_charts(pd_metrics, pd2_metrics, improvements)
    
    return pd_metrics, pd2_metrics, improvements, fig


def quick_summary(pd_tree, pd2_tree, source, alpha=0.5):
    
    pd_metrics = calculate_tree_metrics(pd_tree, source)
    pd2_metrics = calculate_tree_metrics(pd2_tree, source)
    
    # Calcula melhorias principais
    wl_improvement = ((pd_metrics['wirelength'] - pd2_metrics['wirelength']) / pd_metrics['wirelength']) * 100
    detour_improvement = ((pd_metrics['total_detour_cost'] - pd2_metrics['total_detour_cost']) / pd_metrics['total_detour_cost']) * 100
    
    # Função objetivo
    pd_objective = alpha * pd_metrics['total_detour_cost'] + (1 - alpha) * pd_metrics['wirelength']
    pd2_objective = alpha * pd2_metrics['total_detour_cost'] + (1 - alpha) * pd2_metrics['wirelength']
    objective_improvement = ((pd_objective - pd2_objective) / pd_objective) * 100
    
    summary = {
        'wirelength_improvement': wl_improvement,
        'detour_improvement': detour_improvement,
        'objective_improvement': objective_improvement,
        'pd_objective': pd_objective,
        'pd2_objective': pd2_objective,
        'is_better': objective_improvement > 0
    }
    
    return summary
