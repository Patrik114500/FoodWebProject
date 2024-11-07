import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

class Resource:
    def __init__(self, initial_amount, replenish_rate):
        self.amount = initial_amount
        self.replenish_rate = replenish_rate
    
    def replenish(self):
        self.amount += self.replenish_rate

class Agent:
    def __init__(self, energy, metabolism, predation_radius=1, predation_success_rate=0.7, move_distance=0.1, turn_range=45, pos=(0.5, 0.5)):
        self.energy = energy
        self.metabolism = metabolism
        self.predation_radius = predation_radius
        self.predation_success_rate = predation_success_rate
        self.move_distance = move_distance
        self.turn_range = turn_range
        self.pos = np.array(pos, dtype=float)
        self.angle = np.random.uniform(0, 360)
    
    def move(self, grid_size):
        self.angle += np.random.uniform(-self.turn_range, self.turn_range)
        dx = self.move_distance * np.cos(np.radians(self.angle))
        dy = self.move_distance * np.sin(np.radians(self.angle))
        self.pos = np.clip(self.pos + np.array([dx, dy]), 0, grid_size - 1)

    def consume(self, resource_map):
        cell_pos = tuple(np.round(self.pos).astype(int))
        cell_resource = resource_map.get(cell_pos, 0)
        if cell_resource > 0:
            self.energy += 1
            resource_map[cell_pos] -= 1

    def age_and_check_death(self):
        self.energy -= self.metabolism
        return self.energy <= 0

    def can_reproduce(self):
        return self.energy > 10

class Predator(Agent):
    def __init__(self, energy, metabolism, prey_population, predation_radius=1, predation_success_rate=0.7, move_distance=0.1, turn_range=45, pos=(0.5, 0.5)):
        super().__init__(energy, metabolism, predation_radius, predation_success_rate, move_distance, turn_range, pos)
        self.prey_population = prey_population
    
    def hunt(self):
        nearby_prey = [prey for prey in self.prey_population if np.linalg.norm(self.pos - prey.pos) <= self.predation_radius]
        if nearby_prey:
            prey = np.random.choice(nearby_prey)
            if np.random.random() < self.predation_success_rate:
                self.energy += prey.energy * 0.8
                self.prey_population.remove(prey)

    def can_reproduce(self):
        return self.energy > 15

class Simulation:
    def __init__(self, resource, prey_population, predator_population, grid_size, carrying_capacity):
        self.resource = resource
        self.prey_population = prey_population
        self.predator_population = predator_population
        self.grid_size = grid_size
        self.carrying_capacity = carrying_capacity
        self.resource_map = {(x, y): resource.amount for x in range(grid_size) for y in range(grid_size)}
        self.time_steps = 0
        self.prey_count_history = []
        self.predator_count_history = []

    def replenish_resources(self):
        for pos in self.resource_map:
            self.resource_map[pos] = min(self.resource_map[pos] + self.resource.replenish_rate, self.resource.amount)

    def step(self):
        self.replenish_resources()
        new_prey = []
        for prey in self.prey_population:
            prey.move(self.grid_size)
            prey.consume(self.resource_map)
            if prey.age_and_check_death():
                self.prey_population.remove(prey)
            elif prey.can_reproduce() and len(self.prey_population) < self.carrying_capacity:
                offspring = Agent(energy=5, metabolism=prey.metabolism, move_distance=prey.move_distance, turn_range=prey.turn_range, pos=prey.pos.copy())
                self.prey_population.append(offspring)
                prey.energy -= 5

        new_predators = []
        for predator in self.predator_population:
            predator.move(self.grid_size)
            predator.hunt()
            if predator.age_and_check_death():
                self.predator_population.remove(predator)
            elif predator.can_reproduce() and len(self.predator_population) < self.carrying_capacity:
                offspring = Predator(
                    energy=7,
                    metabolism=predator.metabolism,
                    prey_population=self.prey_population,
                    predation_radius=predator.predation_radius,
                    predation_success_rate=predator.predation_success_rate,
                    move_distance=predator.move_distance,
                    turn_range=predator.turn_range,
                    pos=predator.pos.copy()
                )
                new_predators.append(offspring)
                predator.energy -= 7
        self.predator_population.extend(new_predators)

        # Record population counts
        self.prey_count_history.append(len(self.prey_population))
        self.predator_count_history.append(len(self.predator_population))
        self.time_steps += 1

    def get_agent_positions(self):
        prey_positions = [prey.pos for prey in self.prey_population]
        predator_positions = [predator.pos for predator in self.predator_population]
        return prey_positions, predator_positions

    def check_extinction(self):
        if len(self.prey_population) == 0:
            messagebox.showinfo("Extinction Event", "All prey have gone extinct! Restarting prey.")
            self.prey_population = [Agent(energy=5, metabolism=1, pos=(np.random.uniform(0, self.grid_size), np.random.uniform(0, self.grid_size))) for _ in range(prey_var.get())]

        if len(self.predator_population) == 0:
            messagebox.showinfo("Extinction Event", "All predators have gone extinct! Restarting predators.")
            self.predator_population = [Predator(energy=15, metabolism=0.8, prey_population=self.prey_population, pos=(np.random.uniform(0, self.grid_size), np.random.uniform(0, self.grid_size))) for _ in range(predator_var.get())]

# Setup tkinter UI
root = tk.Tk()
root.title("Food Web Simulation Controls")

# Now, initialize the IntVar variables after the Tk root is created
prey_var = tk.IntVar(value=500)
predator_var = tk.IntVar(value=50)

# Control frame setup
control_frame = ttk.Frame(root)
control_frame.grid(row=0, column=0, padx=10, pady=10)

# Controls for setting initial populations
ttk.Label(control_frame, text="Initial Prey Population:").grid(row=0, column=0, sticky="w")
ttk.Entry(control_frame, textvariable=prey_var, width=5).grid(row=0, column=1)

ttk.Label(control_frame, text="Initial Predator Population:").grid(row=1, column=0, sticky="w")
ttk.Entry(control_frame, textvariable=predator_var, width=5).grid(row=1, column=1)

# Control sliders and buttons
speed_var = tk.DoubleVar(value=300)
ttk.Label(control_frame, text="Simulation Speed (ms):").grid(row=2, column=0, sticky="w")
speed_slider = ttk.Scale(control_frame, from_=50, to=1000, variable=speed_var, orient="horizontal")
speed_slider.grid(row=2, column=1)

# Initialize parameters and create simulation instance
grid_size = 30
resource = Resource(initial_amount=5, replenish_rate=5)

# Functions for animation control
def start_simulation():
    global anim
    anim.event_source.start()

def pause_simulation():
    anim.event_source

# Functions for animation control
def start_simulation():
    global anim
    anim.event_source.start()

def pause_simulation():
    anim.event_source.stop()

def restart_simulation():
    global simulation
    # Create new populations based on user-specified values
    prey_population = [Agent(energy=5, metabolism=1, pos=(np.random.uniform(0, grid_size), np.random.uniform(0, grid_size))) for _ in range(prey_var.get())]
    predator_population = [Predator(energy=15, metabolism=0.8, prey_population=prey_population, pos=(np.random.uniform(0, grid_size), np.random.uniform(0, grid_size))) for _ in range(predator_var.get())]
    # Reset the simulation instance
    simulation = Simulation(resource, prey_population, predator_population, grid_size=grid_size, carrying_capacity=2000)
    anim.event_source.stop()
    anim.event_source.start()

# Control buttons for start, pause, and restart
ttk.Button(control_frame, text="Start", command=start_simulation).grid(row=3, column=0)
ttk.Button(control_frame, text="Pause", command=pause_simulation).grid(row=3, column=1)
ttk.Button(control_frame, text="Restart", command=restart_simulation).grid(row=3, column=2)

# Main simulation visualization setup
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
ax1.set_xlim(0, grid_size)
ax1.set_ylim(0, grid_size)
ax1.grid(True)
# Scatters for prey, predators, and grass visualization
prey_scatter = ax1.scatter([], [], color="blue", s=5, label="Prey", edgecolor="black")
predator_scatter = ax1.scatter([], [], color="red", s=5, label="Predator", edgecolor="black")
ax1.legend(loc="upper right")
ax1.set_title("Simulation Grid")

# Dynamic population plot setup
ax2.set_xlim(0, 200)
ax2.set_ylim(0, max(prey_var.get(), predator_var.get()) * 1.5)
ax2.set_xlabel("Time")
ax2.set_ylabel("Population Size")
prey_line, = ax2.plot([], [], color="blue", label="Prey Population")
predator_line, = ax2.plot([], [], color="red", label="Predator Population")
ax2.legend()
ax2.set_title("Population Dynamics")

# Update function for animation
def update(frame):
    simulation.step()
    simulation.check_extinction()

    # Update positions for prey and predators
    prey_positions, predator_positions = simulation.get_agent_positions()
    prey_scatter.set_offsets(prey_positions if prey_positions else [])
    predator_scatter.set_offsets(predator_positions if predator_positions else [])

    # Update population dynamics plot
    prey_line.set_data(range(simulation.time_steps), simulation.prey_count_history)
    predator_line.set_data(range(simulation.time_steps), simulation.predator_count_history)

    # Adjust axis limits dynamically
    ax2.set_xlim(0, max(200, simulation.time_steps))
    ax2.set_ylim(0, max(max(simulation.prey_count_history, default=10), max(simulation.predator_count_history, default=10)) * 1.2)

# Function to initialize and start the animation
def animate():
    global simulation, anim
    # Initialize populations based on user input
    prey_population = [Agent(energy=5, metabolism=1, pos=(np.random.uniform(0, grid_size), np.random.uniform(0, grid_size))) for _ in range(prey_var.get())]
    predator_population = [Predator(energy=15, metabolism=0.8, prey_population=prey_population, pos=(np.random.uniform(0, grid_size), np.random.uniform(0, grid_size))) for _ in range(predator_var.get())]
    # Create simulation instance
    simulation = Simulation(resource, prey_population, predator_population, grid_size=grid_size, carrying_capacity=2000)
    # Start animation
    anim = FuncAnimation(fig, update, frames=200, interval=speed_var.get(), repeat=False)
    plt.show()

# Button to run simulation with updated parameters
ttk.Button(control_frame, text="Run Simulation", command=animate).grid(row=4, column=0, columnspan=3, pady=10)

# Start the tkinter main loop
root.mainloop()
