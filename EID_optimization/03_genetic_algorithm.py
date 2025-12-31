import numpy as np
import random
import networkx as nx
import EoN
from collections import defaultdict

# Emergence probability generation and assignment
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
    probability_ranked = []
    for i in rank_importance:
        xt = sorted(x_samples)[i]
        index_x = x_samples.index(xt)
        yt = y_samples[index_x]
        index_y = sorted(y_samples).index(yt)
        probability_ranked.append(sorted(probabilities)[index_y])
    return probability_ranked, nodes

# Genetic algorithm
class GeneticAlgorithmNodeSelection:
    def __init__(self, G, probabilities, node_list, l,
                 population_size=100, pcrossover=0.8, pmutation=0.05,
                 tau=0.5, gamma=1.0, num_simulations=1000):
        self.G = G
        self.probabilities = probabilities
        self.node_list = node_list
        self.l = l
        self.population_size = population_size
        self.pcrossover = pcrossover
        self.pmutation = pmutation
        self.tau = tau
        self.gamma = gamma
        self.num_simulations = num_simulations
        self.best_individual = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.generation_count = 0
        self.stability_count = 0
        self.fitness_cache = {}

    def simulate_multiple_spreads(self, sublist):

        results = []

        for _ in range(self.num_simulations):
            selected_node = random.choices(self.node_list, weights=self.probabilities, k=1)[0]
            sim = EoN.Gillespie_SIR(self.G, tau=self.tau, gamma=self.gamma,
                                    initial_infecteds=selected_node, return_full_data=True)
            time_series = list(sim.t())
            IplusR = sim.I() + sim.R()

            node_histories = {s: sim.node_history(s) for s in sublist}
            results.append((IplusR, time_series, node_histories))

        return results

    def evaluate_monitoring_objective(self, sublist):

        sim_results = self.simulate_multiple_spreads(sublist)
        g_objective = []

        for IplusR, t_series, node_histories in sim_results:
            g = []
            for s in sublist:
                m1 = node_histories[s]
                if len(m1[0]) == 3:
                    a = m1[0][1]
                    try:
                        g1 = IplusR[-1] - IplusR[t_series.index(a)]
                    except ValueError:
                        g1 = 0
                elif len(m1[0]) == 2:
                    g1 = IplusR[-1] - IplusR[0]
                elif len(m1[0]) == 1:
                    g1 = 0
                else:
                    g1 = 0
                g.append(g1)
            g_objective.append(max(g) if g else 0)

        return np.mean(g_objective)

    def initialize_population(self):

        population = []
        for _ in range(self.population_size):
            individual = random.sample(self.node_list, self.l)
            population.append(individual)
        return population

    def evaluate_fitness(self, population, verbose_debug=False):

        fitness_scores = []
        for i, individual in enumerate(population):
            fitness = self.evaluate_monitoring_objective(individual)
            fitness_scores.append(fitness)
            if verbose_debug and i < 3:
                print(f"  Individual {i}: {individual[:3]}... -> Fitness: {fitness:.4f}")
        return fitness_scores

    def selection(self, population, fitness_scores):
        min_fitness = min(fitness_scores)
        if min_fitness <= 0:
            adjusted_fitness = [f - min_fitness + 1e-6 for f in fitness_scores]
        else:
            adjusted_fitness = [f + 1e-6 for f in fitness_scores]

        selected_individuals = []
        for _ in range(self.population_size):
            selected_idx = random.choices(range(len(population)),
                                          weights=adjusted_fitness, k=1)[0]
            selected_individuals.append(population[selected_idx][:])

        return selected_individuals

    def uniform_crossover(self, parent1, parent2):
        offspring1 = []
        offspring2 = []

        for i in range(self.l):
            if random.random() < 0.5:
                offspring1.append(parent1[i])
                offspring2.append(parent2[i])
            else:
                offspring1.append(parent2[i])
                offspring2.append(parent1[i])
        offspring1 = self.fix_duplicates(offspring1)
        offspring2 = self.fix_duplicates(offspring2)

        return offspring1, offspring2

    def fix_duplicates(self, individual):
        seen = set()
        unique_individual = []
        for node in individual:
            if node not in seen:
                unique_individual.append(node)
                seen.add(node)
        while len(unique_individual) < self.l:
            available_nodes = [node for node in self.node_list if node not in seen]
            if not available_nodes:
                break
            new_node = random.choice(available_nodes)
            unique_individual.append(new_node)
            seen.add(new_node)

        return unique_individual[:self.l]

    def mutate(self, individual):

        individual = individual[:]
        if random.random() < self.pmutation:
            mutation_idx = random.randint(0, self.l - 1)
            available_nodes = [node for node in self.node_list if node not in individual]
            if available_nodes:
                individual[mutation_idx] = random.choice(available_nodes)
        return individual

    def crossover_and_mutation(self, selected_population):

        new_population = []
        random.shuffle(selected_population)

        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1] if i + 1 < len(selected_population) else selected_population[0]

            if random.random() < self.pcrossover:
                offspring1, offspring2 = self.uniform_crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1[:], parent2[:]
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)

            new_population.extend([offspring1, offspring2])

        return new_population[:self.population_size]

    def check_convergence(self, current_best_fitness, current_best_individual):

        if (current_best_fitness == self.best_fitness and
                current_best_individual is not None and
                self.best_individual is not None and
                set(current_best_individual) == set(self.best_individual)):
            self.stability_count += 1
        else:
            self.stability_count = 0
            self.best_fitness = current_best_fitness
            self.best_individual = current_best_individual[:]

        return self.stability_count >= 30

    def run(self, max_generations=100, verbose=True):
        population = self.initialize_population()
        global_best_fitness = -np.inf
        global_best_individual = None

        for generation in range(max_generations):
            self.generation_count = generation
            fitness_scores = self.evaluate_fitness(population, verbose_debug=(generation == 0 and verbose))
            max_fitness_idx = np.argmax(fitness_scores)
            current_gen_best_fitness = fitness_scores[max_fitness_idx]
            current_gen_best_individual = population[max_fitness_idx][:]
            if current_gen_best_fitness > global_best_fitness:
                global_best_fitness = current_gen_best_fitness
                global_best_individual = current_gen_best_individual[:]
                if verbose:
                    print(f"Generation {generation}: NEW BEST fitness = {global_best_fitness:.4f}")
                    print(f"Best individual: {global_best_individual}")

            self.fitness_history.append(global_best_fitness)

            if verbose and generation % 20 == 0:
                print(f"Generation {generation}: Global best fitness = {global_best_fitness:.4f}")
                print(f"Current gen best fitness = {current_gen_best_fitness:.4f}")
            if self.check_convergence(global_best_fitness, global_best_individual):
                if verbose:
                    print(f"Converged after {generation + 1} generations")
                break
            population_with_elite = population[:]
            if global_best_individual is not None:
                population_with_elite[0] = global_best_individual[:]

            selected_population = self.selection(population_with_elite, fitness_scores)
            new_population = self.crossover_and_mutation(selected_population)

            if global_best_individual is not None:
                new_population[0] = global_best_individual[:]

            population = new_population

        self.best_individual = global_best_individual
        self.best_fitness = global_best_fitness
        return self.best_individual, self.best_fitness


def load_network_and_run_ga(gml_file_path, probabilities_data, l, **kwargs):
    G = nx.read_gml(gml_file)
    connected_components = list(nx.connected_components(G))
    largest_cc = max(connected_components, key=len)
    G = G.subgraph(largest_cc)
    node_list = list(G.nodes())

    if isinstance(probabilities_data, str):
        probabilities = np.loadtxt(probabilities_data)
    else:
        probabilities = probabilities_data
    if len(probabilities) != len(node_list):
        raise ValueError(
            f"Number of probabilities ({len(probabilities)}) doesn't match number of nodes ({len(node_list)})")

    ga = GeneticAlgorithmNodeSelection(G, probabilities, node_list, l, **kwargs)
    best_subset, best_fitness = ga.run()

    return best_subset, best_fitness, ga

def run_data(gml_file_path, probabilities_data, l, **kwargs):
    try:
        best_subset, best_fitness, ga_instance = load_network_and_run_ga(
            gml_file_path, probabilities_data, l, **kwargs
        )

        print(f"Optimization completed!")
        print(f"Best subset: {best_subset}")
        print(f"Best fitness: {best_fitness:.4f}")
        print(f"Converged in {ga_instance.generation_count + 1} generations")

        return best_subset, best_fitness, ga_instance

    except Exception as e:
        print(f"Error running GA: {e}")
        print("Try running the simple test first to debug the issue.")
        return None, None, None

# Example usage:
if __name__ == "__main__":
    print("Genetic Algorithm for Network Node Selection")
    network_files = [
    'D:\\figure\\50.gml',
    'D:\\figure\\100.gml',
    'D:\\figure\\150.gml',
    'D:\\figure\\200.gml',
    'D:\\figure\\250.gml',
    'D:\\figure\\300.gml',
    'D:\\figure\\350.gml',
    'D:\\figure\\400.gml',
    'D:\\figure\\450.gml',
    'D:\\figure\\500.gml'
    ]
    for gml_file in network_files:
        print(f"\nProcessing network: {gml_file}")  
        G = nx.read_gml(gml_file)
        connected_components = list(nx.connected_components(G))
        largest_cc = max(connected_components, key=len)
        G = G.subgraph(largest_cc)

        probabilities, nodes = probability_generate(G, 0.1, 5, -0.7, len(G.nodes()))
        l = 6
        import time
        start_time = time.time()
        best_subset, best_fitness, ga_instance = run_data(
            gml_file, probabilities, l,
            num_simulations=100,
            population_size=100,
        )
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(best_subset)
        print("\nDone!")
