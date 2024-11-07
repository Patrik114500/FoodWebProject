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
    def __init__(self, prey_birth_rate, predator1_birth_rate, predator1_death_rate, predator2_birth_rate, predator2_death_rate, max_population, prey_energy_gain=5, predator1_energy_gain=15, predator2_energy_gain=20):
        self.grid_size = 20
        self.agents = []
        self.prey_birth_rate = prey_birth_rate
        self.predator1_birth_rate = predator1_birth_rate
        self.predator1_death_rate = predator1_death_rate
        self.predator2_birth_rate = predator2_birth_rate
        self.predator2_death_rate = predator2_death_rate
        self.max_population = max_population
        self.prey_energy_gain = prey_energy_gain
        self.predator1_energy_gain = predator1_energy_gain
        self.predator2_energy_gain = predator2_energy_gain

        # Initialize populations for prey, Predator 1, and Predator 2
        for _ in range(50):  # Initial prey
            self.agents.append(Agent("prey"))
        for _ in range(20):  # Initial Predator 1
            self.agents.append(Agent("predator1"))
        for _ in range(10):  # Initial Predator 2
            self.agents.append(Agent("predator2"))

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

            # Predator 1 behavior
            elif agent.species == "predator1":
                # Predator 1 hunts prey if in the same position
                for other_agent in self.agents:
                    if other_agent.species == "prey" and other_agent.pos == agent.pos:
                        agent.energy += self.prey_energy_gain  # Gain energy from prey
                        other_agent.energy = 0  # Prey is "eaten"
                        break

                # Predator 1 reproduction and death
                if random.random() < self.predator1_birth_rate and agent.energy >= 20:
                    new_agent = agent.reproduce()
                    if new_agent:
                        new_agents.append(new_agent)
                elif random.random() < self.predator1_death_rate:
                    agent.energy = 0  # Predator 1 dies

            # Predator 2 behavior
            elif agent.species == "predator2":
                # Predator 2 hunts Predator 1 if in the same position
                for other_agent in self.agents:
                    if other_agent.species == "predator1" and other_agent.pos == agent.pos:
                        agent.energy += self.predator1_energy_gain  # Gain energy from Predator 1
                        other_agent.energy = 0  # Predator 1 is "eaten"
                        break

                # Predator 2 reproduction and death
                if random.random() < self.predator2_birth_rate and agent.energy >= 20:
                    new_agent = agent.reproduce()
                    if new_agent:
                        new_agents.append(new_agent)
                elif random.random() < self.predator2_death_rate:
                    agent.energy = 0  # Predator 2 dies

        # Remove dead agents and add new agents
        self.agents = [agent for agent in self.agents if agent.energy > 0]
        self.agents.extend(new_agents)

        # Limit population to max_population
        if len(self.agents) > self.max_population:
            self.agents = random.sample(self.agents, self.max_population)

    def count_species(self):
        prey_count = sum(1 for agent in self.agents if agent.species == "prey")
        predator1_count = sum(1 for agent in self.agents if agent.species == "predator1")
        predator2_count = sum(1 for agent in self.agents if agent.species == "predator2")
        return prey_count, predator1_count, predator2_count

# Simulation parameters
prey_birth_rate = 0.3             # Probability of prey reproduction per time step
predator1_birth_rate = 0.05       # Probability of Predator 1 reproduction per time step
predator1_death_rate = 0.02       # Probability of Predator 1 natural death per time step
predator2_birth_rate = 0.03       # Probability of Predator 2 reproduction per time step
predator2_death_rate = 0.01       # Probability of Predator 2 natural death per time step
max_population = 300              # Maximum total agents (prey + predators)
prey_energy_gain = 3              # Energy gained by Predator 1 from each prey
predator1_energy_gain = 6         # Energy gained by Predator 2 from each Predator 1

# Initialize environment
env = Environment(prey_birth_rate, predator1_birth_rate, predator1_death_rate, predator2_birth_rate, predator2_death_rate, max_population, prey_energy_gain, predator1_energy_gain, predator1_energy_gain)

# Run the simulation
prey_history, predator1_history, predator2_history = [], [], []
for _ in range(100):
    env.update()
    prey_count, predator1_count, predator2_count = env.count_species()
    prey_history.append(prey_count)
    predator1_history.append(predator1_count)
    predator2_history.append(predator2_count)

# Plot results
plt.plot(prey_history, label="Prey", color="blue")
plt.plot(predator1_history, label="Predator 1", color="orange")
plt.plot(predator2_history, label="Predator 2", color="red")
plt.legend()
plt.title("3-Species Lotka-Volterra Model (Agent-Based)")
plt.xlabel("Time Steps")
plt.ylabel("Population")
plt.grid(True)
#plt.savefig('3lvAgent.png')
plt.show()
