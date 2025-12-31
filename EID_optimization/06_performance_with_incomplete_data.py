import numpy as np
import networkx as nx
import pandas as pd
import random
import gc
import os
import math
from joblib import Parallel, delayed
from collections import defaultdict
import EoN
import warnings
warnings.filterwarnings('ignore')

# ========== Network Removal Functions ==========
def remove_x_percent_edges_strong_random(G, X, seed=None, max_try=100000):
    if seed is not None:
        random.seed(seed)
    G = G.copy()
    
    total_edges = G.number_of_edges()
    target_remove = math.floor(total_edges * X / 100)
    
    removed = 0
    tries = 0
    all_edges = list(G.edges())
    
    while removed < target_remove and tries < max_try:
        e = random.choice(all_edges)
        if not G.has_edge(*e):
            continue
        
        G.remove_edge(*e)
        if nx.is_connected(G):
            removed += 1
        else:
            G.add_edge(*e)
        tries += 1
    
    if removed < target_remove:
        return None
    return G

def remove_x_percent_nodes_strong_random(G, X, seed=None, max_try=100000):
    if seed is not None:
        random.seed(seed)
    G = G.copy()
    
    total_nodes = G.number_of_nodes()
    target_remove = math.floor(total_nodes * X / 100)
    
    removed = 0
    tries = 0
    all_nodes = list(G.nodes())
    
    while removed < target_remove and tries < max_try:
        v = random.choice(all_nodes)
        if not G.has_node(v):
            continue
        
        neighbors = list(G.neighbors(v))
        G.remove_node(v)
        
        if nx.is_connected(G):
            removed += 1
        else:
            G.add_node(v)
            for u in neighbors:
                G.add_edge(v, u)
        tries += 1
    
    if removed < target_remove:
        return None
    return G

def generate_omission_networks(G_original, omission_pct, omission_type, n_instances=10):
    networks = []
    
    for i in range(n_instances):
        seed = random.randint(0, 1000000)
        
        if omission_type == 'edges':
            G_omission = remove_x_percent_edges_strong_random(G_original, omission_pct, seed=seed)
        else:  # nodes
            G_omission = remove_x_percent_nodes_strong_random(G_original, omission_pct, seed=seed)
        
        if G_omission is None:
            print(f"Unable to generate {omission_type} {omission_pct}% '{i+1} instance")
            return None
        
        networks.append(G_omission)
    
    return networks

# ========== Generating emergence probability Functions ==========
def unique_ranks(data):
    indexed_data = [(value, index) for index, value in enumerate(data)]
    sorted_data = sorted(indexed_data, key=lambda x: x[0])
    ranks = [0] * len(data)
    for rank, (_, original_index) in enumerate(sorted_data):
        ranks[original_index] = rank
    return ranks

def probability_generate(G, alpha, beta, corr, s):
    probabilities = list(np.random.beta(alpha, beta, s))
    importance = [t[1] for t in nx.algorithms.centrality.degree_centrality(G).items()]
    nodes = list(G.nodes)
    mean = [0, 0]
    cov = [[1, corr], [corr, 1]]
    samples = np.random.multivariate_normal(mean, cov, s)
    x_samples = list(samples[:, 0])
    y_samples = list(samples[:, 1])
    rank_importance = unique_ranks(importance)
    probaility_ranked = []
    for i in (rank_importance):
        xt = sorted(x_samples)[i]
        index_x = x_samples.index(xt)
        yt = y_samples[index_x]
        index_y = sorted(y_samples).index(yt)
        probaility_ranked.append(sorted(probabilities)[index_y])
    return probaility_ranked, nodes

def select_node_outbreak(p, n):
    return random.choices(n, weights=p, k=1)[0]

def simulate_one_spread(G, probabilities, node_list, tau=0.5, gamma=1.0):
    selected_node = select_node_outbreak(probabilities, node_list)
    sim = EoN.Gillespie_SIR(G, tau=tau, gamma=gamma, initial_infecteds=selected_node, return_full_data=True)
    result = {
        "t": list(sim.t()),
        "cumulative": sim.I() + sim.R(),
        "node_history": {node: sim.node_history(node) for node in G.nodes()}
    }
    return result

def evaluate_sim_result(sim_result, monitor_nodes, total_nodes):
    """Evaluate with monitor nodes and return percentage"""
    t_list = sim_result["t"]
    cumulative = sim_result["cumulative"]
    histories = sim_result["node_history"]
    detection_gains = []
    
    for s in monitor_nodes:
        if s not in histories:
            continue
        history = histories[s]
        if len(history[0]) == 3:
            infect_time = history[0][1]
            gain = cumulative[-1] - cumulative[t_list.index(infect_time)]
        elif len(history[0]) == 2:
            gain = cumulative[-1] - cumulative[0]
        else:
            gain = 0
        detection_gains.append(gain)
    
    max_gain = max(detection_gains) if detection_gains else 0
    return (max_gain / total_nodes) * 100

# Run batch simulations on the network
def run_batch_simulations(G, probabilities, node_list, n_batches=10, batch_size=100):
    results = []
    for batch_idx in range(n_batches):
        batch = Parallel(n_jobs=-1)(
            delayed(simulate_one_spread)(G, probabilities, node_list) for _ in range(batch_size)
        )
        results.append(batch)
        if (batch_idx + 1) % 5 == 0:
            gc.collect()
    return results

# Evaluate strategy performance on simulation results
def evaluate_strategy(sim_results, node_seq, total_nodes):
    batch_means = []
    for batch in sim_results:
        values = [evaluate_sim_result(sim, node_seq, total_nodes) for sim in batch]
        batch_means.append(np.mean(values))
    mean_val = np.mean(batch_means)
    return mean_val

# ========== Strategy Selection Functions ==========
def greedy_max_influence(G, node_list, probabilities, rounds, simulations=1000):
    rounds = min(rounds, len(node_list))
    best = []
    available_nodes = set(node_list)
    for step in range(rounds):
        candidates = [best + [x] for x in available_nodes]
        mean_gains = np.zeros(len(candidates))
        for sim_idx in range(simulations):
            selected_node = select_node_outbreak(probabilities, node_list)
            sim = EoN.Gillespie_SIR(G, tau=0.5, gamma=1., initial_infecteds=selected_node, return_full_data=True)
            b, c = list(sim.t()), sim.I() + sim.R()
            for i, candidate in enumerate(candidates):
                g_max = 0
                for s in candidate:
                    m1 = sim.node_history(s)
                    if len(m1[0]) == 3:
                        g1 = c[-1] - c[b.index(m1[0][1])]
                    elif len(m1[0]) == 2:
                        g1 = c[-1] - c[0]
                    else:
                        g1 = 0
                    g_max = max(g_max, g1)
                mean_gains[i] += g_max
        mean_gains /= simulations
        best_idx = np.argmax(mean_gains)
        best = candidates[best_idx]
        available_nodes.remove(best[-1])
    return best

def get_global_strategy(G, num_monitors):
    num_monitors = min(num_monitors, G.number_of_nodes())
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
    return sorted_nodes[:num_monitors]

def get_modular_strategy(G, num_monitors):
    num_monitors = min(num_monitors, G.number_of_nodes())
    communities = nx.community.greedy_modularity_communities(G)
    modular_nodes = []
    community_nodes_by_degree = []
    for community in communities:
        community_degrees = {node: G.degree(node) for node in community}
        sorted_community = sorted(community, key=lambda x: community_degrees[x], reverse=True)
        community_nodes_by_degree.append(sorted_community)
    selection_round = 0
    while len(modular_nodes) < num_monitors:
        added_in_round = False
        for community_nodes in community_nodes_by_degree:
            if len(modular_nodes) >= num_monitors:
                break
            if selection_round < len(community_nodes):
                candidate = community_nodes[selection_round]
                if candidate not in modular_nodes:
                    modular_nodes.append(candidate)
                    added_in_round = True
        if not added_in_round:
            break
        selection_round += 1
    if len(modular_nodes) < num_monitors:
        degrees = dict(G.degree())
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
        remaining_nodes = [n for n in sorted_nodes if n not in modular_nodes]
        needed = num_monitors - len(modular_nodes)
        modular_nodes.extend(remaining_nodes[:needed])
    return modular_nodes[:num_monitors]

def get_random_strategy(G, num_monitors):
    num_monitors = min(num_monitors, G.number_of_nodes())
    nodes = list(G.nodes())
    return random.sample(nodes, num_monitors)

# ========== Main Evaluation Function ==========
def evaluate_network_with_omissions(original_gml_path, 
                                   omission_percentages=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                                   omission_types=['edges', 'nodes'],
                                   num_sentinels_list=[3, 6, 9],
                                   n_omission_instances=10,
                                   n_strategy_repetitions=10):
    results_list = []
    
    # Load original network
    network_name = os.path.splitext(os.path.basename(original_gml_path))[0]
    
    G_original = nx.read_gml(original_gml_path)
    if not nx.is_connected(G_original):
        largest_cc = max(nx.connected_components(G_original), key=len)
        G_original = G_original.subgraph(largest_cc).copy()
    
    nodes_original = list(G_original.nodes())
    probabilities_original, _ = probability_generate(G_original, 0.1, 5, -0.7, len(nodes_original))
    node_to_prob = dict(zip(nodes_original, probabilities_original))
    
    print(f"Original network: {len(nodes_original)} nodes, {G_original.number_of_edges()} edges")
    # Pre-run simulations on original network
    all_sim_sets = []
    for rep in range(n_strategy_repetitions):
        print(f"  Simulation set {rep+1}/{n_strategy_repetitions}...", end='\r')
        sim_results = run_batch_simulations(G_original, probabilities_original, nodes_original, 
                                           n_batches=10, batch_size=100)
        all_sim_sets.append(sim_results)
        gc.collect()
    
    # Process each removal type
    for omission_type in omission_types:
        print(f"\n{'-'*80}")
        print(f"Omission type: {omission_type.upper()}")
        print(f"{'-'*80}")
        
        # Track if we've reached structural limits
        limit_reached = False
        
        # Process each removal percentage
        for omission_pct in omission_percentages:
            if limit_reached:
                print(f"  Skipping {omission_pct}% (structural limit reached)")
                continue
            
            print(f"\n  Omission: {omission_pct}%")
            
            # Generate multiple removal network instances
            print(f"    Generating {n_omission_instances} network instances...", end='')
            omission_networks = generate_omission_networks(
                G_original, omission_pct, omission_type, n_omission_instances
            )
            
            if omission_networks is None:
                limit_reached = True
                print(f" FAILED - structural limit reached at {omission_pct}%")
                continue
            
            # Process each omission network instance
            for instance_idx, G_omission in enumerate(omission_networks):
                print(f"    Instance {instance_idx+1}/{n_omission_instances}: "
                      f"{G_omission.number_of_nodes()} nodes, {G_omission.number_of_edges()} edges")
                
                nodes_omission = list(G_omission.nodes())
                probabilities_omission = [node_to_prob[node] for node in nodes_omission if node in node_to_prob]
                
                # Process each sentinel count
                for num_sentinels in num_sentinels_list:
                    if len(nodes_omission) < num_sentinels:
                        continue
                    
                    # Define strategies
                    strategy_functions = {
                        'Greedy': ...
                        'RFSM': ...
                        'Global': ...
                        'Modular': ...
                        'Random': ...
                        'GA': ...
                    }
                    
                    # Evaluate each strategy
                    for strategy_name, strategy_func in strategy_functions.items():
                        all_performances = []
                        all_valid_counts = []
                        
                        for rep in range(n_strategy_repetitions):
                            selected_nodes = strategy_func()
                            valid_monitors = [n for n in selected_nodes if n in nodes_original]
                            all_valid_counts.append(len(valid_monitors))

                            sim_results_for_rep = all_sim_sets[rep]
                            performance = evaluate_strategy(
                                sim_results_for_rep, selected_nodes, len(nodes_original)
                            )
                            all_performances.append(performance)

                        mean_performance = np.mean(all_performances)
                        std_performance  = np.std(all_performances, ddof=1)
                        mean_valid       = np.mean(all_valid_counts)

                        
                        results_list.append({
                            'network_name': network_name,
                            'omission_type': omission_type,
                            'omission_pct': omission_pct,
                            'num_sentinels': num_sentinels,
                            'strategy': strategy_name,
                            'surveillance_performance_mean': mean_performance,
                            'surveillance_performance_std': std_performance,
                            'valid_monitors': mean_valid,
                            'total_monitors': num_sentinels,
                            'original_network_size': len(nodes_original),
                            'omission_network_nodes': len(nodes_omission),
                            'omission_network_edges': G_omission.number_of_edges()
                        })
                
                gc.collect()
    
    # Clean up
    del all_sim_sets
    gc.collect()
    
    return results_list

    