import random
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, species, energy=10):
        self.species = species
        self.energy = energy
        self.pos = (random.randint(0, 19), random.randint(0, 19))  # Random initial position in a 20x20 grid

    def move(self, grid_size):
        # Randomly move to an adjacent cell in a 20x20 grid
        self.pos = ((self.pos[0] + random.choice([-1, 0, 1])) % grid_size,
                    (self.pos[1] + random.choice([-1, 0, 1])) % grid_size)

    def reproduce(self):
        if self.energy >= 20:  # Reproduction threshold
            self.energy -= 10
            return Agent(self.species, energy=10)
        return None

class Environment:
    def __init__(self, prey_birth_rate, predator_birth_rate, predator_death_rate, max_population, prey_energy_gain=5, predator_energy_gain=15):
        self.grid_size = 20
        self.agents = []
        self.prey_birth_rate = prey_birth_rate
        self.predator_birth_rate = predator_birth_rate
        self.predator_death_rate = predator_death_rate
        self.max_population = max_population
        self.prey_energy_gain = prey_energy_gain
        self.predator_energy_gain = predator_energy_gain

        # Initialize prey and predator populations
        for _ in range(50):  # Initial prey
            self.agents.append(Agent("prey"))
        for _ in range(10):  # Initial predators
            self.agents.append(Agent("predator"))

    def update(self):
        new_agents = []
        for agent in self.agents:
            agent.move(self.grid_size)

            # Prey behavior
            if agent.species == "prey":
                if random.random() < self.prey_birth_rate:  # Reproduction chance
                    new_agent = agent.reproduce()
                    if new_agent:
                        new_agents.append(new_agent)

            # Predator behavior
            elif agent.species == "predator":
                # Predator hunts if prey is in the same position
                for other_agent in self.agents:
                    if other_agent.species == "prey" and other_agent.pos == agent.pos:
                        agent.energy += self.prey_energy_gain  # Predator gains energy from eating prey
                        other_agent.energy = 0  # Prey is "eaten" (energy set to zero)
                        break

                # Predator reproduction or natural death
                if random.random() < self.predator_birth_rate and agent.energy >= 20:
                    new_agent = agent.reproduce()
                    if new_agent:
                        new_agents.append(new_agent)
                elif random.random() < self.predator_death_rate:
                    agent.energy = 0  # Predator dies

        # Remove agents that "died" (energy = 0) and add newly born agents
        self.agents = [agent for agent in self.agents if agent.energy > 0]
        self.agents.extend(new_agents)

        # Ensure population does not exceed max_population
        if len(self.agents) > self.max_population:
            self.agents = random.sample(self.agents, self.max_population)

    def count_species(self):
        prey_count = sum(1 for agent in self.agents if agent.species == "prey")
        predator_count = sum(1 for agent in self.agents if agent.species == "predator")
        return prey_count, predator_count

# Refined adjustments to support prey sustainability and modest predator growth
prey_birth_rate = 0.65           # Slightly higher probability of prey reproduction per time step
predator_birth_rate = 0.015      # Lower probability of predator reproduction per time step
predator_death_rate = 0.008      # Very low predator death rate for population maintenance
max_population = 200             # Maximum total agents (prey + predator)
prey_energy_gain = 3             # Moderate energy gain per prey for sustained predator energy
predator_energy_gain = 7         # Lower threshold for predator reproduction

# Reinitialize environment with these refined parameters
env = Environment(prey_birth_rate, predator_birth_rate, predator_death_rate, max_population, prey_energy_gain, predator_energy_gain)

# Run the simulation again and capture population history
prey_history, predator_history = [], []
for _ in range(300):
    env.update()
    prey_count, predator_count = env.count_species()
    prey_history.append(prey_count)
    predator_history.append(predator_count)


# Plot results
plt.plot(prey_history, label="Prey Population", color="blue")
plt.plot(predator_history, label="Predator Population", color="red")
plt.legend()
plt.title("2-Species Lotka-Volterra Model (Agent-Based)")
plt.xlabel("Time Steps")
plt.ylabel("Population")
plt.grid(True)
plt.show()
