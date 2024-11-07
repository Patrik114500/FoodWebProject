import numpy as np
import matplotlib.pyplot as plt

class PuppetLV3SpeciesModel:
    def __init__(self, prey_birth_rate, prey_death_rate, predator1_birth_rate, predator1_death_rate, predator2_birth_rate, predator2_death_rate, initial_counts, max_population):
        # Model parameters
        self.prey_birth_rate = prey_birth_rate
        self.prey_death_rate = prey_death_rate
        self.predator1_birth_rate = predator1_birth_rate
        self.predator1_death_rate = predator1_death_rate
        self.predator2_birth_rate = predator2_birth_rate
        self.predator2_death_rate = predator2_death_rate
        self.prey_count, self.predator1_count, self.predator2_count = initial_counts
        self.max_population = max_population

        # Initialize history for plotting
        self.prey_history = [self.prey_count]
        self.predator1_history = [self.predator1_count]
        self.predator2_history = [self.predator2_count]

    def update_population(self):
        # Calculate next populations based on extended L-V equations
        prey_next = self.prey_count + self.prey_count * (self.prey_birth_rate - self.prey_death_rate * self.predator1_count)
        predator1_next = self.predator1_count + self.predator1_count * (self.predator1_birth_rate * self.prey_count - self.predator1_death_rate - self.predator2_birth_rate * self.predator2_count)
        predator2_next = self.predator2_count + self.predator2_count * (self.predator2_birth_rate * self.predator1_count - self.predator2_death_rate)

        # Apply upper and lower limits to prevent negative populations and excessive growth
        self.prey_count = max(1, min(prey_next, self.max_population))
        self.predator1_count = max(1, min(predator1_next, self.max_population))
        self.predator2_count = max(1, min(predator2_next, self.max_population))

        # Update history
        self.prey_history.append(self.prey_count)
        self.predator1_history.append(self.predator1_count)
        self.predator2_history.append(self.predator2_count)

    def run_simulation(self, steps):
        for _ in range(steps):
            self.update_population()

# Parameters for the 3-species model
prey_birth_rate = 0.1           # Natural growth rate of prey
prey_death_rate = 0.01          # Rate at which prey are eaten by Predator 1
predator1_birth_rate = 0.005    # Growth rate of Predator 1 based on prey availability
predator1_death_rate = 0.2      # Natural death rate of Predator 1
predator2_birth_rate = 0.003    # Growth rate of Predator 2 based on Predator 1 availability
predator2_death_rate = 0.1      # Natural death rate of Predator 2
initial_counts = [50, 10, 5]    # Initial populations for prey, Predator 1, and Predator 2
max_population = 500            # Maximum allowed population for each species

# Run the 3-species model
model_3species = PuppetLV3SpeciesModel(prey_birth_rate, prey_death_rate, predator1_birth_rate, predator1_death_rate, predator2_birth_rate, predator2_death_rate, initial_counts, max_population)
model_3species.run_simulation(500)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(model_3species.prey_history, label="Prey Population", color="blue")
plt.plot(model_3species.predator1_history, label="Predator 1 Population", color="orange")
plt.plot(model_3species.predator2_history, label="Predator 2 Population", color="red")
plt.legend()
plt.title("3-Species Lotka-Volterra Model (Puppet-Based)")
plt.xlabel("Time Steps")
plt.ylabel("Population")
plt.grid(True)
plt.savefig('3lvPuppet.png')
plt.show()


