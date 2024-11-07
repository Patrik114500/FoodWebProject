import numpy as np
import matplotlib.pyplot as plt

class PuppetLV2SpeciesModel:
    def __init__(self, prey_birth_rate, prey_death_rate, predator_birth_rate, predator_death_rate, initial_counts, max_population):
        # Model parameters
        self.prey_birth_rate = prey_birth_rate
        self.prey_death_rate = prey_death_rate
        self.predator_birth_rate = predator_birth_rate
        self.predator_death_rate = predator_death_rate
        self.prey_count, self.predator_count = initial_counts
        self.max_population = max_population

        # Initialize history for plotting
        self.prey_history = [self.prey_count]
        self.predator_history = [self.predator_count]

    def update_population(self):
        # Calculate next populations based on L-V equations
        prey_next = self.prey_count + self.prey_count * (self.prey_birth_rate - self.prey_death_rate * self.predator_count)
        predator_next = self.predator_count + self.predator_count * (self.predator_birth_rate * self.prey_count - self.predator_death_rate)

        # Apply upper and lower limits to prevent negative populations and excessive growth
        self.prey_count = max(1, min(prey_next, self.max_population))
        self.predator_count = max(1, min(predator_next, self.max_population))

        # Update history
        self.prey_history.append(self.prey_count)
        self.predator_history.append(self.predator_count)

    def run_simulation(self, steps):
        for _ in range(steps):
            self.update_population()

# Parameters for the puppet model
prey_birth_rate = 0.05       # Reduced natural growth rate of prey
prey_death_rate = 0.01       # Rate at which prey are eaten by predators
predator_birth_rate = 0.005  # Growth rate of predators based on prey availability
predator_death_rate = 0.2    # Increased natural death rate of predators
initial_counts = [50, 10]    # Initial populations of prey and predators
max_population = 500         # Maximum allowed population for both prey and predators

# Run the puppet model
model = PuppetLV2SpeciesModel(prey_birth_rate, prey_death_rate, predator_birth_rate, predator_death_rate, initial_counts, max_population)
model.run_simulation(500)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(model.prey_history, label="Prey Population", color="blue")
plt.plot(model.predator_history, label="Predator Population", color="red")
plt.legend()
plt.title("2-Species Lotka-Volterra Model")
plt.xlabel("Time Steps")
plt.ylabel("Population")
plt.grid(True)
#plt.savefig('2lvPuppet.png')
plt.show()
