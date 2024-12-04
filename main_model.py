import numpy as np

import random

import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

import concurrent.futures
import time
import json

# Cell class for grid management
class Cell:
    def __init__(self):
        self.resources = 10  # Base resources per cell
        self.agents = []

    def consume_resources(self, amount):
        consumed = min(self.resources, amount)
        self.resources -= consumed
        return consumed

    def replenish_resources(self):
        self.resources = min(self.resources + 5, 10)

# Agent class for species behavior
class Agent:
    def __init__(self, species_id, energy, reproduction_rate, predation_rate, resource_consumption):
        self.species_id = species_id
        self.energy = energy
        self.reproduction_rate = reproduction_rate
        self.predation_rate = predation_rate
        self.resource_consumption = resource_consumption
        self.is_alive = True

    def act(self, grid, x, y):
        cell = grid[x][y]
        self.energy += cell.consume_resources(self.resource_consumption)

        if random.random() < self.reproduction_rate and self.energy > 10:
            self.energy -= 5
            return Agent(self.species_id, 10, self.reproduction_rate, self.predation_rate, self.resource_consumption)
        return None

    def interact(self, other):
        if self.species_id > other.species_id:
            if random.random() < self.predation_rate:
                self.energy += other.energy
                other.is_alive = False
        elif self.species_id < other.species_id:
            if random.random() < other.predation_rate:
                other.energy += self.energy
                self.is_alive = False

# Simulation class with stopping conditions and dynamic species addition
class DynamicSpeciesSimulation:
    def __init__(self, grid_size, initial_populations, steps, species_addition_interval=1000, max_population_threshold=7000):
        self.grid_size = grid_size
        self.grid = [[Cell() for _ in range(grid_size)] for _ in range(grid_size)]
        self.steps = steps
        self.species_addition_interval = species_addition_interval
        self.max_population_threshold = max_population_threshold
        self.agents = []

        # Initialize agents
        self.species_parameters = initial_populations.copy()
        for species_id, (population, reproduction_rate, predation_rate, resource_consumption) in enumerate(initial_populations):
            for _ in range(population):
                x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
                agent = Agent(species_id, 10, reproduction_rate, predation_rate, resource_consumption)
                self.agents.append((agent, x, y))
                self.grid[x][y].agents.append(agent)

    def add_species(self):
        """Add a new species dynamically."""
        new_species_id = len(self.species_parameters)
        new_species_params = (10, 0.02, 0.5, 2)  # New species parameters
        self.species_parameters.append(new_species_params)

        for _ in range(new_species_params[0]):
            x, y = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            agent = Agent(new_species_id, 10, new_species_params[1], new_species_params[2], new_species_params[3])
            self.agents.append((agent, x, y))
            self.grid[x][y].agents.append(agent)

    def step(self):
        new_agents = []
        for agent, x, y in self.agents:
            if not agent.is_alive:
                continue

            if agent in self.grid[x][y].agents:
                self.grid[x][y].agents.remove(agent)

            dx, dy = random.choice([-1, 0, 1]), random.choice([-1, 0, 1])
            nx, ny = (x + dx) % self.grid_size, (y + dy) % self.grid_size
            self.grid[nx][ny].agents.append(agent)

            offspring = agent.act(self.grid, nx, ny)
            if offspring:
                new_agents.append((offspring, nx, ny))
                self.grid[nx][ny].agents.append(offspring)

            interactions = 0
            for other in self.grid[nx][ny].agents:
                if other != agent and other.is_alive and interactions < 3:
                    agent.interact(other)
                    interactions += 1

        for row in self.grid:
            for cell in row:
                cell.replenish_resources()
                cell.agents = [agent for agent in cell.agents if agent.is_alive]

        self.agents = [(agent, x, y) for agent, x, y in self.agents if agent.is_alive] + new_agents

    def run(self):
        population_history = [[] for _ in range(len(self.species_parameters))]
        for step in range(self.steps):
            self.step()

            if step > 0 and step % self.species_addition_interval == 0:
                self.add_species()
                population_history.append([])

            counts = [0] * len(population_history)
            for agent, _, _ in self.agents:
                if agent.is_alive:
                    counts[agent.species_id] += 1

            if any(count < 1 for count in counts):
                print(f"Stopping simulation: A species went extinct at step {step}.")
                break
            if any(count > self.max_population_threshold for count in counts):
                print(f"Stopping simulation: A species exceeded {self.max_population_threshold} population at step {step}.")
                break

            for i, count in enumerate(counts):
                population_history[i].append(count)

        return population_history

# Hyperparameter optimizer with extinction-avoiding fitness metric
class HyperparameterOptimizer:
    def __init__(self, grid_size, steps, species_addition_interval, max_population_threshold, param_ranges, max_threads):
        self.grid_size = grid_size
        self.steps = steps
        self.species_addition_interval = species_addition_interval
        self.max_population_threshold = max_population_threshold
        self.param_ranges = param_ranges
        self.max_threads = max_threads

    def run_simulation(self, params):
        initial_populations = [
            (params["rabbits_population"], params["rabbits_reproduction"], 0.0, 1),
            (params["foxes_population"], params["foxes_reproduction"], params["foxes_predation"], 2),
        ]

        sim = DynamicSpeciesSimulation(self.grid_size, initial_populations, self.steps,
                                       self.species_addition_interval, self.max_population_threshold)
        population_history = sim.run()

        final_counts = [len(history) and history[-1] for history in population_history]
        if any(count < 1 for count in final_counts):  # Penalize extinction
            return float('-inf')

        fitness = -sum(abs(max(history) - min(history)) for history in population_history if len(history) > 0)
        return fitness

    '''
    def grid_search(self):
        best_params = None
        best_fitness = float('-inf')

        param_combinations = list(product(*self.param_ranges.values()))
        with tqdm(total=len(param_combinations), desc="Grid Search Progress") as pbar:
            for combination in param_combinations:
                params = {key: combination[i] for i, key in enumerate(self.param_ranges.keys())}

                
                fitness = self.run_simulation(params)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = params

                pbar.update(1)
                pbar.set_postfix({"Best Fitness": best_fitness})

        print(f"Best parameters: {best_params}")
        print(f"Best fitness: {best_fitness}")
        return best_params, best_fitness
        '''

    def grid_search(self):
        best_params = None
        best_fitness = float('-inf')

        param_combinations = list(product(*self.param_ranges.values()))
        params_all = [{key: combination[i] for i, key in enumerate(self.param_ranges.keys())} for combination in param_combinations]
        
        results = []

        with concurrent.futures.ThreadPoolExecutor(self.max_threads) as executor:
            future_to_params = {executor.submit(self.run_simulation, params): params for params in params_all}

            for future in concurrent.futures.as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    fitness = future.result()
                    results.append([params, fitness])
                    print(f"Simulation terminated for params: {params}")
                except Exception as exc:
                    print(f"Simulation with params {params} generated an exception: {exc}")

        
        results.sort(key = lambda x: x[1])
        return results[0]

# Hyperparameter optimization
param_ranges = {
    "rabbits_population": [20, 30, 40],
    "rabbits_reproduction": [0.1, 0.15, 0.2],
    "foxes_population": [5, 10],
    "foxes_reproduction": [0.03, 0.05],
    "foxes_predation": [0.3, 0.4, 0.5],
}

'''
param_ranges = {
    "rabbits_population": np.linspace(10, 50, 41, dtype = int),
    "rabbits_reproduction": np.linspace(0.1, 2, 20),
    "foxes_population": np.linspace(3, 25, 22, dtype = int),
    "foxes_reproduction": np.linspace(0.05, 1, 20),
    "foxes_predation": np.linspace(0.1, 2, 20),
}'''

optimizer = HyperparameterOptimizer(
    grid_size=20,
    steps=1000,
    species_addition_interval=500,
    max_population_threshold=7000,
    param_ranges=param_ranges,
    max_threads = 20
)

best_params, best_fitness = optimizer.grid_search()

# Run the best simulation
initial_populations = [
    (best_params["rabbits_population"], best_params["rabbits_reproduction"], 0.0, 1),
    (best_params["foxes_population"], best_params["foxes_reproduction"], best_params["foxes_predation"], 2),
]
final_sim = DynamicSpeciesSimulation(20, initial_populations, 3000, 1000, 7000)
final_population_history = final_sim.run()

# Plot results
plt.figure(figsize=(12, 6))
for i, history in enumerate(final_population_history):
    plt.plot(history, label=f"Species {i}")
plt.xlabel("Steps")
plt.ylabel("Population")
plt.title("Best Simulation with Survival-Based Fitness Metric")
plt.legend()
plt.show()
