import numpy as np
import numpy.random as rdm
import networkx as nx
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
import warnings
import EoN
import random
import time
import multiprocessing
from scipy import stats

warnings.filterwarnings('ignore')

# ========== Network Generation Functions ==========
def heterogeneous_dist(ID_list, mean_degree, heterogeneity):
    degree = {}
    for ID in ID_list:
        degree[ID] = mean_degree

    while np.std([degree[ID] for ID in ID_list]) < heterogeneity:
        source_list = [ID for ID in ID_list if degree[ID] > 1]
        if not source_list:
            break
        random_node = source_list[int(rdm.random() * len(source_list))]

        r = rdm.random() * len(ID_list) * mean_degree

        k = 0
        target_node = ID_list[k]
        slide = degree[target_node]
        while slide < r and k < len(ID_list) - 1:
            k = k + 1
            target_node = ID_list[k]
            slide = slide + degree[target_node]

        degree[random_node] -= 1
        degree[target_node] += 1

    return degree

def connect_stubs(stubs, edge_list):
    stubs = [(s[0], s[1]) for s in rdm.permutation(stubs)]

    if len(stubs) < 2:
        no_more_edges = True
    else:
        no_more_edges = False

    while no_more_edges == False:
        if not stubs:
            break
        source = stubs.pop(0)

        found = False
        stubs_to_check = len(stubs)

        while found == False and no_more_edges == False:
            if not stubs:
                no_more_edges = True
                break
            target = stubs.pop(0)

            new_edge = sorted([source, target])
            if new_edge in edge_list or source == target:
                stubs.append(target)
            else:
                edge_list.append(new_edge)
                found = True

            stubs_to_check = stubs_to_check - 1
            if stubs_to_check < 2:
                no_more_edges = True

    return edge_list

def modular_config_model(module_size, number_of_modules, p, heterogeneity, mean_degree):
    ID_list = []
    for m in range(number_of_modules):
        for n in range(module_size):
            node_ID = (m, n)
            ID_list.append(node_ID)

    degree = heterogeneous_dist(ID_list, mean_degree, heterogeneity)

    edge_list = []
    inter_stubs = []

    for m in range(number_of_modules):
        intra_stubs = []
        for n in range(module_size):
            for i in range(degree[(m, n)]):
                r = rdm.random()
                if r < p:
                    intra_stubs.append((m, n))
                else:
                    inter_stubs.append((m, n))

        new_edges = connect_stubs(intra_stubs, edge_list)
        edge_list = edge_list + new_edges

    new_edges = connect_stubs(inter_stubs, edge_list)
    edge_list = edge_list + new_edges

    return ID_list, edge_list

def network_generator(module_size, number_of_modules, p, heterogeneity, mean_degree):
    ID = []
    edge = []
    ID2, edge2 = modular_config_model(module_size, number_of_modules, p, heterogeneity, mean_degree)
    for i in ID2:
        ID.append(str(i))
    for i in edge2:
        x = []
        for j in i:
            j = tuple(map(int, j))
            x.append(str(j))
        edge.append(x)
    return ID, edge

# ========== Emergence probability generation functions ==========
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

# ========== Characteristics functions ==========
def weighted_closeness_centrality(G, probabilities):
    nodes_list = list(G.nodes())
    prob_dict = {nodes_list[i]: probabilities[i] for i in range(len(nodes_list))}
    weighted_centrality = {}
    
    for node in G.nodes():
        distances = nx.single_source_shortest_path_length(G, node)
        weighted_sum = 0
        total_weight = 0
        
        for target, distance in distances.items():
            if target != node:
                weight = prob_dict[target]
                weighted_sum += weight / (distance + 1)
                total_weight += weight
        
        weighted_centrality[node] = weighted_sum / total_weight if total_weight > 0 else 0
    
    return weighted_centrality

# ========== Enhanced Feature Extraction Function ==========
def extract_node_features(G, probabilities, node, selected_nodes=None):
    nodes_list = list(G.nodes())
    node_idx = nodes_list.index(node)
    
    # ==================== Basic characteristics ====================
    features = {
        'degree': G.degree(node),
        'probability': probabilities[node_idx],
    }
    
    # ====================  Global emergence probability-based characteristics ====================
    features['prob_mean'] = np.mean(probabilities)
    features['prob_std'] = np.std(probabilities)
    features['prob_skewness'] = stats.skew(probabilities)
    features['prob_kurtosis'] = stats.kurtosis(probabilities)
    
    # ==================== Global network topology-based characteristics ====================
    features['num_nodes'] = G.number_of_nodes()
    features['density'] = nx.density(G)
    features['avg_clustering'] = nx.average_clustering(G)
    
    degree_sequence = [d for n, d in G.degree()]
    features['avg_degree'] = np.mean(degree_sequence)
    features['degree_variance'] = np.var(degree_sequence)
    features['degree_skewness'] = stats.skew(degree_sequence)
    features['degree_kurtosis'] = stats.kurtosis(degree_sequence)
    
    try:
        if nx.is_connected(G):
            features['avg_path_length'] = nx.average_shortest_path_length(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            features['avg_path_length'] = nx.average_shortest_path_length(subgraph)
    except:
        features['avg_path_length'] = 0
    
    # ==================== Node centrality characteristics ====================
    try:
        betweenness_centrality = nx.betweenness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        weighted_closeness = weighted_closeness_centrality(G, probabilities)
        
        features['betweenness_centrality'] = betweenness_centrality[node]
        features['eigenvector_centrality'] = eigenvector_centrality[node]
        features['prob_weighted_distance'] = weighted_closeness[node]
    except:
        features['betweenness_centrality'] = np.nan
        features['eigenvector_centrality'] = np.nan
        features['prob_weighted_distance'] = np.nan
    
    # ==================== Local clustering characteristics ====================
    features['clustering_coeff'] = nx.clustering(G, node)
    if G.degree(node) > 0:
        subgraph_nodes = [node] + list(G.neighbors(node))
        features['local_density'] = nx.density(G.subgraph(subgraph_nodes))
    else:
        features['local_density'] = 0
    
    # ==================== Neighbor statistics (standardized) ====================
    neighbors = list(G.neighbors(node))
    neighbor_probs = [probabilities[nodes_list.index(n)] for n in neighbors]
    
    features.update({
        'avg_neighbor_degree': np.mean([G.degree(n) for n in neighbors]) if neighbors else 0,
        'avg_neighbor_prob': np.mean(neighbor_probs) if neighbors else 0,
        'std_neighbor_prob': np.std(neighbor_probs) if neighbors else 0,
        'prob_weighted_degree': features['degree'] * features['probability'],
        'prob_weighted_clustering': features['clustering_coeff'] * features['probability'],
        'degree_centrality': features['degree'] / (features['num_nodes'] - 1) if features['num_nodes'] > 1 else 0,  # 新增：度中心性（标准化度数）
    })
    
    # ==================== Monitoring node-related features ====================
    if selected_nodes:
        node_neighbors = set(neighbors)
        num_selected = len(selected_nodes)
        
        synergy_sum = 0.0
        redundancy_sum = 0.0
        synergy_count = 0 
        redundancy_count = 0
        
        for sel in selected_nodes:
            try:
                dist = nx.shortest_path_length(G, node, sel)
                if dist > 2:
                    synergy_sum += 1.0 / dist
                    synergy_count += 1
                else:
                    redundancy_sum += 1.0 / dist
                    redundancy_count += 1
            except nx.NetworkXNoPath:
                pass
        
        # Standardize synergy and redundancy scores
        synergy_score = synergy_sum / synergy_count if synergy_count > 0 else 0.0
        redundancy_score = redundancy_sum / redundancy_count if redundancy_count > 0 else 0.0
        
        # Calculate coverage ratios
        node_coverage = set([node] + neighbors)
        existing_coverage = set()
        for s in selected_nodes:
            existing_coverage.update([s] + list(G.neighbors(s)))
        
        # Calculate minimum distance to selected nodes
        try:
            min_dist = min([nx.shortest_path_length(G, node, s) for s in selected_nodes], default=10)
        except:
            min_dist = 10
        
        features.update({
            # Standardized features
            'neighbor_selected_ratio': len(set(selected_nodes) & set(neighbors)) / features['degree'] if features['degree'] > 0 else 0,
            'synergy_score': synergy_score,
            'redundancy_score': redundancy_score,
            'new_coverage_ratio': len(node_coverage - existing_coverage) / len(node_coverage) if len(node_coverage) > 0 else 0,
            'overlap_coverage_ratio': len(node_coverage & existing_coverage) / len(node_coverage) if len(node_coverage) > 0 else 0,
            'min_dist_to_selected': min_dist,
        })
    else:
        features.update({
            'neighbor_selected_ratio': 0,
            'synergy_score': 0,
            'redundancy_score': 0,
            'new_coverage_ratio': 1.0,
            'overlap_coverage_ratio': 0,
            'min_dist_to_selected': 10,
        })
    
    return features

# ========== Greedy Selection ==========
def select_node_outbreak(p, n):
    return random.choices(n, weights=p, k=1)[0]

def greedy_max_influence(G, node, probabilities, rounds=None, simulations=1000):
    if rounds is None:
        rounds = len(G.nodes())
    degrees = [d for _, d in G.degree()]
    k_mean = sum(degrees) / len(degrees)
    k2_mean = sum(d**2 for d in degrees) / len(degrees)
    best = []
    available_nodes = set(node)
    node_rankings = {}
    
    for step in range(rounds):
        if not available_nodes:
            break
            
        candidates = [best + [x] for x in available_nodes]
        mean_gains = np.zeros(len(candidates))
        
        for _ in range(simulations):
            selected_node = select_node_outbreak(probabilities, node)
            sim = EoN.Gillespie_SIR(G, tau=3.0/((k2_mean-k_mean)/k_mean), gamma=1., initial_infecteds=selected_node, return_full_data=True)
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
        best = candidates[np.argmax(mean_gains)]
        
        selected_node = best[-1]
        node_rankings[selected_node] = step
        available_nodes.remove(selected_node)
    
    return node_rankings

# ========== Training data generating ==========
def generate_single_network_data(network_id):
    # Generating network parameters
    module_size = np.random.randint(15, 30)
    number_of_modules = np.random.randint(3, 8)
    p = np.random.uniform(0.2, 0.9)
    heterogeneity = np.random.uniform(1, 15)
    mean_degree = np.random.randint(3, 8)
    
    alpha = np.random.uniform(0.05, 2.0)
    beta = np.random.uniform(2, 10)
    corr = np.random.uniform(-0.9, 0.9)
    
    network_params = {
        'module_size': module_size,
        'number_of_modules': number_of_modules,
        'p': p,
        'heterogeneity': heterogeneity,
        'mean_degree': mean_degree,
        'alpha': alpha,
        'beta': beta,
        'corr': corr
    }
    
    try:
        ID, edge = network_generator(module_size, number_of_modules, p, heterogeneity, mean_degree)
        G = nx.Graph()
        G.add_nodes_from(ID)
        G.add_edges_from(edge)
        
        if G.number_of_edges() == 0:
            return None

        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        
        if G.number_of_nodes() < 20 or G.number_of_edges() < 25:
            return None
        
        probabilities, nodes = probability_generate(G, alpha, beta, corr, len(G.nodes()))
        
        if np.std(probabilities) < 0.005 or np.mean(probabilities) < 0.005:
            return None
        
        node_rankings = greedy_max_influence(G, nodes, probabilities, rounds=len(G.nodes()), simulations=1000)
        
        training_samples = []
        nodes_list = list(G.nodes())
        
        for node in nodes_list:
            if node not in node_rankings:
                continue
                
            ranking = node_rankings[node]
            selected_nodes = [n for n, r in node_rankings.items() if r < ranking]
            node_features = extract_node_features(G, probabilities, node, selected_nodes)
            
            sample = {
                'network_id': network_id,
                'node_id': node,
                'ranking': ranking,
                **network_params,
                **node_features
            }
            
            training_samples.append(sample)

        return training_samples
        
    except Exception as e:
        print(f"Network {network_id} generation failed: {e}")
        return None

def generate_training_dataset(n_networks=100, n_jobs=4, save_path='training_data.csv', batch_size=50):
    base_path = save_path.rsplit('.', 1)[0]
    extension = save_path.rsplit('.', 1)[-1]
    
    all_samples = []
    batch_count = 0
    
    for i in range(0, n_networks, batch_size):
        batch_end = min(i + batch_size, n_networks)
        print(f"Batch {batch_count + 1}: Network {i}-{batch_end-1}")

        batch_samples = Parallel(n_jobs=n_jobs)(
            delayed(generate_single_network_data)(network_id) for network_id in range(i, batch_end)
        )
        valid_batch_samples = [s for s in batch_samples if s is not None]
        all_samples.extend(valid_batch_samples)
        
        batch_df = pd.DataFrame([item for sublist in valid_batch_samples for item in sublist])
        batch_file_path = f"{base_path}_batch_{batch_count + 1}.{extension}"
        
        if len(batch_df) > 0:
            batch_df.to_csv(batch_file_path, index=False)
            print(f"Batch {batch_count + 1} have be saved: {batch_file_path}")
            print(f"Number of Sampling: {len(batch_df)} | Number of networks: {batch_df['network_id'].nunique()}")
        else:
            print(f"Batch {batch_count + 1} has no valid sample")
        
        batch_count += 1
        
        if batch_end < n_networks:
            time.sleep(5)
    
    # Saved
    if all_samples:
        df = pd.DataFrame([item for sublist in all_samples for item in sublist])
        df.to_csv(save_path, index=False)
        print(f"All data saved to {save_path}")
        print(f"Parameters: ['module_size', 'number_of_modules', 'p', 'heterogeneity', 'mean_degree', 'alpha', 'beta', 'corr']")
        print(f"Characteristics: {[col for col in df.columns if col not in ['network_id', 'node_id', 'ranking', 'module_size', 'number_of_modules', 'p', 'heterogeneity', 'mean_degree', 'alpha', 'beta', 'corr']]}")
        
        return df
    else:
        print("No data!")
        return None

if __name__ == "__main__":
    random.seed(42)
    training_data = generate_training_dataset(
        n_networks=1000,
        n_jobs=max(1, multiprocessing.cpu_count() - 2),
        save_path='D:\\data_generating\\training_data.csv',
        batch_size=50
    )