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
WOLF = 2
BEAR = 3

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
                 reborn_chance,
                 rabbit_being_eaten_chance, fox_being_eaten_chance, wolf_being_eaten_chance,
                 predator_being_dead_chance):
        
        self.x = x0
        self.y = y0
        self.energy = init_energy
        self.species = species
        self.isDead = False

        self.Grid_x_size = Grid_x_size
        self.Grid_y_size = Grid_y_size

        ###
        self.reborn_chance = reborn_chance
        ###
        
        self.rabbit_being_eaten_chance = rabbit_being_eaten_chance
        self.fox_being_eaten_chance = fox_being_eaten_chance
        
        self.predator_being_dead_chance = predator_being_dead_chance

    def interact(self, other):
        # this method is used to interact with another animal:
        # - If they're from the same species, ignore each other.
        # - Fox eats rabbit, but Rabbit escapes from fox sometimes.
        # - During interaction, both animals lose energy.
        # - When predation is successful, the predator gains energy.

        """ RABBIT - FOX / FOX - RABBIT interaction """
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
                self.energy -= 2 # FOX - WOLF / WOLF - FOX interaction 
        elif self.species == FOX and other.species == WOLF:
            if np.random.rand() <= self.fox_being_eaten_chance:
                self.die()
                other.energy += 2
            else:
                escape(self)
                self.energy -= 3
                other.energy -= 2
        elif self.species == WOLF and other.species == FOX:
            if np.random.rand() <= self.fox_being_eaten_chance:
                other.die()
                self.energy += 2
            else:
                escape(other)
                other.energy -= 3
                self.energy -= 2 # WOLF - BEAR / BEAR - WOLF interaction 
        elif self.species == WOLF and other.species == BEAR:
            if np.random.rand() <= self.fox_being_eaten_chance:
                self.die()
                other.energy += 2
            else:
                escape(self)
                self.energy -= 3
                other.energy -= 2
        elif self.species == BEAR and other.species == WOLF:
            if np.random.rand() <= self.fox_being_eaten_chance:
                other.die()
                self.energy += 2
            else:
                escape(other)
                other.energy -= 3
                self.energy -= 2


    def predator_dead(self):
        # this method is used to judge whether a fox is dead naturally.
        if self.energy <= 0:
            self.die()
            return
            
        if self.species != RABBIT:
            if np.random.rand() <= self.predator_being_dead_chance * 100 / self.energy:
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
    
    steps_1 = params['steps_1']
    steps_2 = params['steps_2']
    steps_3 = params['steps_3']

    Grid_x_size = params['Grid_x_size']  # size of the grid
    Grid_y_size = params['Grid_y_size']  # size of the grid
    
    rabbits = params['rabbits']  # initial number of rabbits
    rabbit_energy_level = params['rabbit_energy_level'] # initial energy level of a rabbit
    rabbit_newborn_chance = params['rabbit_newborn_chance'] # chance of a rabbit being born in a grid
    rabbit_being_eaten_chance = params['rabbit_being_eaten_chance'] # chance of a rabbit being eaten by a fox
    
    foxes = params['foxes'] # initial number of foxes    
    fox_energy_level = params['fox_energy_level']# initial energy level of a fox
    fox_newborn_chance = params['fox_newborn_chance']  # chance of a fox being born in a grid
    fox_being_eaten_chance = params['fox_being_eaten_chance'] # chance of a fox being eaten by a wolf
    fox_being_dead_chance = params['fox_being_dead_chance'] # chance of a fox dying
    
    wolfs = params['wolfs'] # initial number of wolfs    
    wolf_energy_level = params['wolf_energy_level']# initial energy level of a wolf
    wolf_newborn_chance = params['wolf_newborn_chance']  # chance of a wolf being born in a grid
    wolf_being_eaten_chance = params['wolf_being_eaten_chance'] # chance of a wolf being eaten by a bear
    wolf_being_dead_chance = params['wolf_being_dead_chance'] # chance of a wolf dying

    bears = params['bears'] # initial number of bears    
    bear_energy_level = params['bear_energy_level']# initial energy level of a bear
    bear_newborn_chance = params['bear_newborn_chance']  # chance of a bear being born in a grid
    bear_being_dead_chance = params['bear_being_dead_chance'] # chance of a bear dying
    
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
                              rabbit_being_eaten_chance = rabbit_being_eaten_chance,
                              fox_being_eaten_chance = fox_being_eaten_chance,
                              wolf_being_eaten_chance = wolf_being_eaten_chance,
                              predator_being_dead_chance = 0))
    for (x, y) in fox_coords:
        animals.append(Animal(x0=x, y0=y, init_energy=fox_energy_level, species=FOX,
                              Grid_x_size = Grid_x_size, Grid_y_size = Grid_y_size, reborn_chance = fox_newborn_chance,
                              rabbit_being_eaten_chance = rabbit_being_eaten_chance,
                              fox_being_eaten_chance = fox_being_eaten_chance,
                              wolf_being_eaten_chance = wolf_being_eaten_chance,
                              predator_being_dead_chance = fox_being_dead_chance))
        
    # population of rabbits and foxes
    rabbit_nums, fox_nums, wolf_nums, bear_nums = [rabbits], [foxes], [wolfs], [bears]

    #for i in tqdm(range(steps)):
    for i in range(steps_1):

        # randomly move each animal: UP, DOWN, LEFT, RIGHT, STAY
        directions = np.random.randint(0, 5, size=len(animals))
        for animal, direction in zip(animals, directions):
            animal.move(direction)

        # fox dead naturally
        for animal in animals:
            animal.predator_dead()

        # reproduce new rabbits
        # generate rabbit_newborn_chance * rabbits new rabbits in different places
        # check if there is already a rabbit in a place, if so, delete the newborn rabbit in order to avoid overlapping
        # put the remaining newborn rabbits on the grid
        
        num_new_born_rabbits = np.random.poisson(rabbit_newborn_chance * rabbits)
        new_rabbit_coords = np.random.permutation(coords)[:num_new_born_rabbits]
        for (x, y) in new_rabbit_coords:
            if any(animal.x == x and animal.y == y for animal in animals):
                new_rabbit_coords = np.delete(new_rabbit_coords, np.where((new_rabbit_coords == (x, y)).all(axis=1)),axis=0)
        for (x, y) in new_rabbit_coords:
            animals.append(Animal(x0=x, y0=y, init_energy=rabbit_energy_level, species=RABBIT,
                                  Grid_x_size = Grid_x_size, Grid_y_size = Grid_y_size, reborn_chance = rabbit_newborn_chance,
                                  rabbit_being_eaten_chance = rabbit_being_eaten_chance,
                                  fox_being_eaten_chance = fox_being_eaten_chance,
                                  wolf_being_eaten_chance = wolf_being_eaten_chance,
                                  predator_being_dead_chance = 0))

        # do the same thing for foxes
        num_new_born_foxes = np.random.poisson(fox_newborn_chance * foxes * rabbits)
        new_fox_coords = np.random.permutation(coords)[:num_new_born_foxes]
        
        for (x, y) in new_fox_coords:
            if any(animal.x == x and animal.y == y for animal in animals):
                new_fox_coords = np.delete(new_fox_coords, np.where((new_fox_coords == (x, y)).all(axis=1)), axis=0)
        for (x, y) in new_fox_coords:
            animals.append(Animal(x0=x, y0=y, init_energy=fox_energy_level, species=FOX,
                                  Grid_x_size = Grid_x_size, Grid_y_size = Grid_y_size, reborn_chance = fox_newborn_chance,
                                  rabbit_being_eaten_chance = rabbit_being_eaten_chance,
                                  fox_being_eaten_chance = fox_being_eaten_chance,
                                  wolf_being_eaten_chance = wolf_being_eaten_chance,
                                  predator_being_dead_chance = fox_being_dead_chance))

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
        
    ###########################################################################################################################################################################
    random_coords = np.random.permutation(coords)
    wolf_coords = random_coords[:wolfs]
    for (x, y) in wolf_coords:
        animals.append(Animal(x0=x, y0=y, init_energy = wolf_energy_level, species=WOLF,
                              Grid_x_size = Grid_x_size, Grid_y_size = Grid_y_size, reborn_chance = wolf_newborn_chance,
                              rabbit_being_eaten_chance = rabbit_being_eaten_chance,
                              fox_being_eaten_chance = fox_being_eaten_chance,
                              wolf_being_eaten_chance = wolf_being_eaten_chance,
                              predator_being_dead_chance = wolf_being_dead_chance))

    for i in range(steps_2):
        directions = np.random.randint(0, 5, size=len(animals))
        for animal, direction in zip(animals, directions):
            animal.move(direction)

        # fox dead naturally
        for animal in animals:
            animal.predator_dead()
        
        num_new_born_rabbits = np.random.poisson(rabbit_newborn_chance * rabbits)
        new_rabbit_coords = np.random.permutation(coords)[:num_new_born_rabbits]
        for (x, y) in new_rabbit_coords:
            if any(animal.x == x and animal.y == y for animal in animals):
                new_rabbit_coords = np.delete(new_rabbit_coords, np.where((new_rabbit_coords == (x, y)).all(axis=1)),axis=0)
        for (x, y) in new_rabbit_coords:
            animals.append(Animal(x0=x, y0=y, init_energy=rabbit_energy_level, species=RABBIT,
                                  Grid_x_size = Grid_x_size, Grid_y_size = Grid_y_size, reborn_chance = rabbit_newborn_chance,
                                  rabbit_being_eaten_chance = rabbit_being_eaten_chance,
                                  fox_being_eaten_chance = fox_being_eaten_chance,
                                  wolf_being_eaten_chance = wolf_being_eaten_chance,
                                  predator_being_dead_chance = 0))

        
        num_new_born_foxes = np.random.poisson(fox_newborn_chance * foxes * rabbits)
        new_fox_coords = np.random.permutation(coords)[:num_new_born_foxes]
        for (x, y) in new_fox_coords:
            if any(animal.x == x and animal.y == y for animal in animals):
                new_fox_coords = np.delete(new_fox_coords, np.where((new_fox_coords == (x, y)).all(axis=1)), axis=0)
        for (x, y) in new_fox_coords:
            animals.append(Animal(x0=x, y0=y, init_energy=fox_energy_level, species=FOX,
                                  Grid_x_size = Grid_x_size, Grid_y_size = Grid_y_size, reborn_chance = fox_newborn_chance,
                                  rabbit_being_eaten_chance = rabbit_being_eaten_chance,
                                  fox_being_eaten_chance = fox_being_eaten_chance,
                                  wolf_being_eaten_chance = wolf_being_eaten_chance,
                                  predator_being_dead_chance = fox_being_dead_chance))
            

        num_new_born_wolfs = np.random.poisson(wolf_newborn_chance * foxes * wolfs)
        new_wolf_coords = np.random.permutation(coords)[:num_new_born_wolfs]
        for (x, y) in new_wolf_coords:
            if any(animal.x == x and animal.y == y for animal in animals):
                new_wolf_coords = np.delete(new_wolf_coords, np.where((new_wolf_coords == (x, y)).all(axis=1)), axis=0)
        for (x, y) in new_wolf_coords:
            animals.append(Animal(x0=x, y0=y, init_energy = wolf_energy_level, species=WOLF,
                                  Grid_x_size = Grid_x_size, Grid_y_size = Grid_y_size, reborn_chance = wolf_newborn_chance,
                                  rabbit_being_eaten_chance = rabbit_being_eaten_chance,
                                  fox_being_eaten_chance = fox_being_eaten_chance,
                                  wolf_being_eaten_chance = wolf_being_eaten_chance,
                                  predator_being_dead_chance = wolf_being_dead_chance))

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
        fox_num, rab_num, wolf_num = 0, 0, 0
        for animal in animals:
            if animal.species == RABBIT:
                rab_num += 1
            elif animal.species == FOX:
                fox_num += 1
            elif animal.species == WOLF:
                wolf_num += 1
                
        rabbit_nums.append(rab_num)
        fox_nums.append(fox_num)
        wolf_nums.append(wolf_num)
        
        if rab_num == 0 or fox_num == 0 or wolf_num == 0:
            break
        
    ###########################################################################################################################################################################
    random_coords = np.random.permutation(coords)
    bear_coords = random_coords[:bears]
    for (x, y) in bear_coords:
        animals.append(Animal(x0=x, y0=y, init_energy = bear_energy_level, species=BEAR,
                              Grid_x_size = Grid_x_size, Grid_y_size = Grid_y_size, reborn_chance = bear_newborn_chance,
                              rabbit_being_eaten_chance = rabbit_being_eaten_chance,
                              fox_being_eaten_chance = fox_being_eaten_chance,
                              wolf_being_eaten_chance = wolf_being_eaten_chance,
                              predator_being_dead_chance = bear_being_dead_chance))

    for i in range(steps_3):
        directions = np.random.randint(0, 5, size=len(animals))
        for animal, direction in zip(animals, directions):
            animal.move(direction)

        # fox dead naturally
        for animal in animals:
            animal.predator_dead()
        
        num_new_born_rabbits = np.random.poisson(rabbit_newborn_chance * rabbits)
        new_rabbit_coords = np.random.permutation(coords)[:num_new_born_rabbits]
        for (x, y) in new_rabbit_coords:
            if any(animal.x == x and animal.y == y for animal in animals):
                new_rabbit_coords = np.delete(new_rabbit_coords, np.where((new_rabbit_coords == (x, y)).all(axis=1)),axis=0)
        for (x, y) in new_rabbit_coords:
            animals.append(Animal(x0=x, y0=y, init_energy=rabbit_energy_level, species=RABBIT,
                                  Grid_x_size = Grid_x_size, Grid_y_size = Grid_y_size, reborn_chance = rabbit_newborn_chance,
                                  rabbit_being_eaten_chance = rabbit_being_eaten_chance,
                                  fox_being_eaten_chance = fox_being_eaten_chance,
                                  wolf_being_eaten_chance = wolf_being_eaten_chance,
                                 predator_being_dead_chance = 0))

        
        num_new_born_foxes = np.random.poisson(fox_newborn_chance * foxes * rabbits)
        new_fox_coords = np.random.permutation(coords)[:num_new_born_foxes]
        for (x, y) in new_fox_coords:
            if any(animal.x == x and animal.y == y for animal in animals):
                new_fox_coords = np.delete(new_fox_coords, np.where((new_fox_coords == (x, y)).all(axis=1)), axis=0)
        for (x, y) in new_fox_coords:
            animals.append(Animal(x0=x, y0=y, init_energy=fox_energy_level, species=FOX,
                                  Grid_x_size = Grid_x_size, Grid_y_size = Grid_y_size, reborn_chance = fox_newborn_chance,
                                  rabbit_being_eaten_chance = rabbit_being_eaten_chance,
                                  fox_being_eaten_chance = fox_being_eaten_chance,
                                  wolf_being_eaten_chance = wolf_being_eaten_chance,
                                  predator_being_dead_chance = fox_being_dead_chance))
            

        num_new_born_wolfs = np.random.poisson(wolf_newborn_chance * foxes * wolfs)
        new_wolf_coords = np.random.permutation(coords)[:num_new_born_wolfs]
        for (x, y) in new_wolf_coords:
            if any(animal.x == x and animal.y == y for animal in animals):
                new_wolf_coords = np.delete(new_wolf_coords, np.where((new_wolf_coords == (x, y)).all(axis=1)), axis=0)
        for (x, y) in new_wolf_coords:
            animals.append(Animal(x0=x, y0=y, init_energy = wolf_energy_level, species=WOLF,
                                  Grid_x_size = Grid_x_size, Grid_y_size = Grid_y_size, reborn_chance = wolf_newborn_chance,
                                  rabbit_being_eaten_chance = rabbit_being_eaten_chance,
                                  fox_being_eaten_chance = fox_being_eaten_chance,
                                  wolf_being_eaten_chance = wolf_being_eaten_chance,
                                  predator_being_dead_chance = wolf_being_dead_chance))
            

        num_new_born_bears = np.random.poisson(wolf_newborn_chance * wolfs * bears)
        new_bear_coords = np.random.permutation(coords)[:num_new_born_wolfs]
        for (x, y) in new_bear_coords:
            if any(animal.x == x and animal.y == y for animal in animals):
                new_bear_coords = np.delete(new_bear_coords, np.where((new_bear_coords == (x, y)).all(axis=1)), axis=0)
        for (x, y) in new_wolf_coords:
            animals.append(Animal(x0=x, y0=y, init_energy = bear_energy_level, species=BEAR,
                                  Grid_x_size = Grid_x_size, Grid_y_size = Grid_y_size, reborn_chance = wolf_newborn_chance,
                                  rabbit_being_eaten_chance = rabbit_being_eaten_chance,
                                  fox_being_eaten_chance = fox_being_eaten_chance,
                                  wolf_being_eaten_chance = wolf_being_eaten_chance,
                                  predator_being_dead_chance = bear_being_dead_chance))

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
        fox_num, rab_num, wolf_num, bear_num = 0, 0, 0, 0
        for animal in animals:
            if animal.species == RABBIT:
                rab_num += 1
            elif animal.species == FOX:
                fox_num += 1
            elif animal.species == WOLF:
                wolf_num += 1
            elif animal.species == BEAR:
                bear_num += 1
                
        rabbit_nums.append(rab_num)
        fox_nums.append(fox_num)
        wolf_nums.append(wolf_num)
        bear_nums.append(bear_num)
        
        if rab_num == 0 or fox_num == 0 or wolf_num == 0:
            break
        
    ###TO DO: what is fittnes
    with open("D:/0xMESTERI_II_EV/1_FELEV/Computational_Intelligence/PROJECT/Results/" + "results.json", "w") as f:
        json.dump(params, f, indent=4)
        json.dump([rabbit_nums, fox_nums, wolf_nums, bear_nums], f, indent=5)

    ###TO DO: what is fittnes
    fitness = len(bear_nums)
    if len(bear_nums) >= steps_3:
        fitness += bear_nums[-1]
    
    return fitness, [rabbit_nums, fox_nums, wolf_nums, bear_nums]
    
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

    results.sort(reverse=True, key = lambda x: x[1])
    return results[0]

"""
param_grid = {
    "steps" : [1000],
    "steps_2" : [500],
    "Grid_x_size" : [50],
    "Grid_y_size" : [50],
    
    "rabbits": [15, 30, 50],
    "rabbit_energy_level": [10],
    "rabbit_newborn_chance": [0.5],
    "rabbit_being_eaten_chance": [0.95],

    "foxes": [5, 10, 15],
    "fox_energy_level": [30],
    "fox_newborn_chance": [0.1],
    "fox_being_eaten_chance" : [0.05],
    "fox_being_dead_chance": [0.08],

    "wolfs": [5, 10, 15],
    "wolf_energy_level": [30],
    "wolf_newborn_chance": [0.1],
    "wolf_being_eaten_chance" : [0.05],
    "wolf_being_dead_chance" : [0.04],

    "bears": [5],
    "bear_energy_level": [30],
    "bear_newborn_chance": [0.1],
    "bear_being_dead_chance" : [0.08],
    }
"""

param_grid = {
    "steps_1" : [500],
    "steps_2" : [500],
    "steps_3" : [1000],
    "Grid_x_size" : [50],
    "Grid_y_size" : [50],
    
    "rabbits": [15],
    "rabbit_energy_level": [10],
    "rabbit_newborn_chance": [0.5],
    "rabbit_being_eaten_chance": [0.95],

    "foxes": [5],
    "fox_energy_level": [30],
    "fox_newborn_chance": [0.1],
    "fox_being_eaten_chance" : [0.05],
    "fox_being_dead_chance": [0.08],

    "wolfs": [5, 10, 15],
    "wolf_energy_level": [10, 20, 30],
    "wolf_newborn_chance": [0.05, 0.1, 0.15],
    "wolf_being_eaten_chance" : [0.05],
    "wolf_being_dead_chance" : [0.08],

    "bears": [5, 10, 15],
    "bear_energy_level": [10, 20, 30],
    "bear_newborn_chance": [0.05, 0.1, 0.15],
    "bear_being_dead_chance" : [0.08],
    }

results = simulation_space(100, param_grid)

fitness = results[1]
rabbit_nums = results[2][0]
fox_nums = results[2][1]
wolf_nums = results[2][2]
bear_nums = results[2][3]

# plot population vs time
plt.figure(figsize=(20, 6))
plt.grid(True)
plt.xlabel('t')
plt.ylabel('population')
plt.suptitle("Population VS Time for params, run length: " + str(len(wolf_nums) + 499) + ', fitness score: ' + str(fitness))

plt.plot(rabbit_nums, 'black', label="rabbits")
plt.plot(fox_nums, 'orange', label="foxes")
plt.plot(range(500, 500 + len(wolf_nums)), wolf_nums, 'gray', label = 'wolfs')
if len(wolf_nums) >= 500:
    plt.plot(range(1000, 1000 + len(bear_nums)), bear_nums, 'brown', label = 'bears')

plt.legend()
#plt.savefig("D:/0xMESTERI_II_EV/1_FELEV/Computational_Intelligence/PROJECT/Results/" + "Naive_4_species_simulation.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig("Naive_4_species_simulation.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

with open("D:/0xMESTERI_II_EV/1_FELEV/Computational_Intelligence/PROJECT/Results/" + "best_parameter_set_4s.json", "w") as f:
        json.dump(results[0], f, indent=4)
