# description of the model
# We have 2 species: rabbits and foxes. Each animal has an energy level. If the energy level is 0, the animal dies.
# Each individual of the same species has the same energy level.
# They don't have any mating season, and they don't even distinguish their own sex.
# The grass is always green, and the rabbits can always find food.
# The foxes can eat rabbits, and they can also die naturally.
# The rabbits can escape from foxes, but not always successfully.
# Each animal can move one step at a time, and each step consumes 1 energy.
# Each step of each animal is purely random, and they don't have any strategy.
# The rabbits can reproduce, and the foxes can also reproduce.
# The newborn animals are randomly placed on the grid, and they don't have any strategy.
# The animals can interact with each other only if they are in the same place, exactly the same (x, y) coordinates.
# The animals can't interact with the same species.
# That's all.


import numpy as np
import matplotlib.pyplot as plt
from progressbar import progressbar as pr_bar
import warnings
from tqdm import tqdm

from itertools import product

import concurrent.futures
import threading

import time
import json

warnings.filterwarnings("ignore")

RABBIT = 0  # species identifiers
FOX = 1  # species identifiers

UP = 0  # direction identifiers
DOWN = 1  # direction identifiers
LEFT = 2  # direction identifiers
RIGHT = 3  # direction identifiers
STAY = 4  # direction identifiers

def escape(weaker):
    weaker.energy -= 1
    weaker.move(np.random.randint(0, 5))  # move randomly

class Animal(object):
    # This class tracks the animal's position, energy, species (rabbit/fox) and state (live/dead).
    def __init__(self, x0, y0, init_energy, species,
                 Grid_x_size, Grid_y_size,
                 reborn_chance, rabbit_being_eaten_chance, fox_being_dead_chance):
        
        self.x = x0
        self.y = y0
        self.energy = init_energy
        self.species = species
        self.isDead = False

        self.Grid_x_size = Grid_x_size
        self.Grid_y_size = Grid_y_size

        self.reborn_chance = reborn_chance

        self.rabbit_being_eaten_chance = rabbit_being_eaten_chance
        self.fox_being_dead_chance = fox_being_dead_chance

    def interact(self, other):
        # this method is used to interact with another animal:
        # - If they're from the same species, ignore each other.
        # - Fox eats rabbit, but Rabbit escapes from fox sometimes.
        # - During interaction, both animals lose energy.
        # - When predation is successful, the predator gains energy.
        if self.species == RABBIT and other.species == FOX:
            if np.random.rand() <= self.rabbit_being_eaten_chance:
                self.die()
                other.energy += 2
            else:
                escape(self)
                self.energy -= 3
                other.energy -= 2

        elif self.species == FOX and other.species == RABBIT:
            if np.random.rand() <= self.rabbit_being_eaten_chance:
                other.die()
                self.energy += 2
            else:
                escape(other)
                other.energy -= 3
                self.energy -= 2

    def fox_dead(self):
        # this method is used to judge whether a fox is dead naturally.
        if self.species == FOX:
            if np.random.rand() <= self.fox_being_dead_chance:
                self.die()

    def die(self):
        # this method is used to judge whether an animal is dead.
        self.isDead = True

    def move(self, direction_param):
        # this method is used to move a step on the grid. Each step consumes 1 energy; if no energy left, die.
        self.energy -= 1
        if direction_param == LEFT:
            self.x += 1 if self.x > 0 else -1  # "bounce back"
        if direction_param == RIGHT:
            self.x -= 1 if self.x < self.Grid_x_size - 1 else -1
        if direction_param == UP:
            self.y += 1 if self.y < self.Grid_y_size - 1 else -1
        if direction_param == DOWN:
            self.y -= 1 if self.y > 0 else -1
        if direction_param == STAY:
            pass

        if self.energy <= 0:
            self.die()

def run_simulation(params, record = True):
    thread_id = threading.get_ident()
    print(f"############################ Task is running on thread ID: {thread_id} with {params} ############################")
    
    steps = params['steps']
    
    rabbits = params['rabbits']  # initial number of rabbits
    foxes = params['foxes'] # initial number of foxes

    Grid_x_size = params['Grid_x_size']  # size of the grid
    Grid_y_size = params['Grid_y_size']  # size of the grid

    rabbit_energy_level = params['rabbit_energy_level'] # initial energy level of a rabbit
    fox_energy_level = params['fox_energy_level']# initial energy level of a fox

    rabbit_newborn_chance = params['rabbit_newborn_chance'] # chance of a rabbit being born in a grid
    fox_newborn_chance = params['fox_newborn_chance']  # chance of a fox being born in a grid

    rabbit_being_eaten_chance = params['rabbit_being_eaten_chance'] # chance of a rabbit being eaten by a fox
    fox_being_dead_chance = params['fox_being_dead_chance'] # chance of a fox dying

    animals = []
    # initialize the grid
    x_coords = np.arange(Grid_x_size)
    y_coords = np.arange(Grid_y_size)
    coords = np.transpose([np.tile(x_coords, len(y_coords)), np.repeat(y_coords, len(x_coords))])
    random_coords = np.random.permutation(coords)
    # randomly place animals on the grid
    rabbit_coords = random_coords[:rabbits]
    fox_coords = random_coords[rabbits:(rabbits + foxes)]
    # initialize animals
    for (x, y) in rabbit_coords:
        animals.append(Animal(x0=x, y0=y, init_energy=rabbit_energy_level, species=RABBIT,
                              Grid_x_size = Grid_x_size, Grid_y_size = Grid_y_size, reborn_chance = rabbit_newborn_chance,
                              rabbit_being_eaten_chance = rabbit_being_eaten_chance, fox_being_dead_chance = fox_being_dead_chance))
    for (x, y) in fox_coords:
        animals.append(Animal(x0=x, y0=y, init_energy=fox_energy_level, species=FOX,
                              Grid_x_size = Grid_x_size, Grid_y_size = Grid_y_size, reborn_chance = fox_newborn_chance,
                              rabbit_being_eaten_chance = rabbit_being_eaten_chance, fox_being_dead_chance = fox_being_dead_chance))
        
    # population of rabbits and foxes
    rabbit_nums, fox_nums = [rabbits], [foxes]

    #for i in tqdm(range(steps)):
    for i in range(steps):

        # randomly move each animal: UP, DOWN, LEFT, RIGHT, STAY
        directions = np.random.randint(0, 5, size=len(animals))
        for animal, direction in zip(animals, directions):
            animal.move(direction)

        # fox dead naturally
        for animal in animals:
            animal.fox_dead()

        # reproduce new rabbits
        # generate rabbit_newborn_chance * rabbits new rabbits in different places
        # check if there is already a rabbit in a place, if so, delete the newborn rabbit in order to avoid overlapping
        # put the remaining newborn rabbits on the grid
        num_new_born_rabbits = int(np.random.poisson(rabbit_newborn_chance * rabbits))
        new_rabbit_coords = np.random.permutation(coords)[:num_new_born_rabbits]
        for (x, y) in new_rabbit_coords:
            if any(animal.x == x and animal.y == y for animal in animals):
                new_rabbit_coords = np.delete(new_rabbit_coords, np.where((new_rabbit_coords == (x, y)).all(axis=1)),axis=0)
        for (x, y) in new_rabbit_coords:
            animals.append(Animal(x0=x, y0=y, init_energy=rabbit_energy_level, species=RABBIT,
                              Grid_x_size = Grid_x_size, Grid_y_size = Grid_y_size, reborn_chance = rabbit_newborn_chance,
                              rabbit_being_eaten_chance = rabbit_being_eaten_chance, fox_being_dead_chance = fox_being_dead_chance))

        # do the same thing for foxes
        num_new_born_foxes = np.random.poisson(fox_newborn_chance * foxes * rabbits)
        new_fox_coords = np.random.permutation(coords)[:num_new_born_foxes]
        for (x, y) in new_fox_coords:
            if any(animal.x == x and animal.y == y for animal in animals):
                new_fox_coords = np.delete(new_fox_coords, np.where((new_fox_coords == (x, y)).all(axis=1)), axis=0)
        for (x, y) in new_fox_coords:
            animals.append(Animal(x0=x, y0=y, init_energy=fox_energy_level, species=FOX,
                                  Grid_x_size = Grid_x_size, Grid_y_size = Grid_y_size, reborn_chance = fox_newborn_chance,
                                  rabbit_being_eaten_chance = rabbit_being_eaten_chance, fox_being_dead_chance = fox_being_dead_chance))

        # interaction between animals
        # only if two animals are in the same place, they can interact
        for j, animal1 in enumerate(animals):
            for animal2 in animals[j:]:
                if animal1.x == animal2.x and animal1.y == animal2.y:
                    animal1.interact(animal2)

        # clean up corpses
        dead_indexes = []
        for j, animal in enumerate(animals):
            if animal.isDead:
                dead_indexes.append(j)
        animals = list(np.delete(animals, dead_indexes))

        # count animals
        fox_num, rab_num = 0, 0
        for animal in animals:
            if animal.species == RABBIT:
                rab_num += 1
            elif animal.species == FOX:
                fox_num += 1
        rabbit_nums.append(rab_num)
        fox_nums.append(fox_num)
        if rab_num == 0 or fox_num == 0:
            break

    ###TO DO: what is fittnes
    with open("D:/0xMESTERI_II_EV/1_FELEV/Computational_Intelligence/PROJECT/Results/" + "results.json", "w") as f:
        json.dump(params, f, indent=4)
        json.dump([rabbit_nums, fox_nums], f, indent=5)

    ###TO DO: what is fittnes  
    return 0, [rabbit_nums, fox_nums] 
    
def simulation_space(threads_nr, param_grid, record = True):
    param_combinations = list(product(*param_grid.values()))
    params_all = [{key: combination[i] for i, key in enumerate(param_grid.keys())} for combination in param_combinations]

    results = []

    with concurrent.futures.ThreadPoolExecutor(threads_nr) as executor:
        future_to_params = {executor.submit(run_simulation, params, record): params for params in params_all}

        for future in concurrent.futures.as_completed(future_to_params):
            params = future_to_params[future]
 
            try:
                fitness, records = future.result()
                results.append([params, fitness, records])
                print(f"Simulation terminated for params: {params}")
            except Exception as exc:
                print(f"Simulation with params {params} generated an exception: {exc}")

    #results.sort(key = lambda x: x[1])
    return results[0]


param_grid = {
    "steps" : [1000],
    "Grid_x_size" : [50],
    "Grid_y_size" : [50],
    
    "rabbits": [15, 30, 50],
    "rabbit_energy_level": [10],
    "rabbit_newborn_chance": [0.5],
    "rabbit_being_eaten_chance": [0.95],

    "foxes": [5, 10, 15],
    "fox_energy_level": [30],
    "fox_newborn_chance": [0.1],
    "fox_being_dead_chance": [0.08],
    }

results = simulation_space(9, param_grid)

rabbit_nums = results[2][0]
fox_nums = results[2][1]

# plot population vs time
plt.figure(figsize=(20, 6))
plt.grid(True)
plt.plot(rabbit_nums, 'b-', label="rabbits", )
plt.plot(fox_nums, 'r-', label="foxes")
plt.xlabel('t')
plt.ylabel('population')
plt.suptitle("Population VS Time for params")
plt.legend()
plt.savefig("D:/0xMESTERI_II_EV/1_FELEV/Computational_Intelligence/PROJECT/Results/" + "Naive_2_species_simulation.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

with open("D:/0xMESTERI_II_EV/1_FELEV/Computational_Intelligence/PROJECT/Results/" + "best_parameter_set.json", "w") as f:
        json.dump(results[0], f, indent=4)
