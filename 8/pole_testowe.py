import random

# Define the parameters of the problem
items = [(3, 266), (13, 442), (10, 671), (9, 526), (7, 388), (1, 245), (8, 210), (8, 145), (2, 126), (9, 322)]
max_weight = 35
population_size = 8
elite_size = int(population_size * 0.25)
mutation_prob = 0.05

# Define the fitness function
def fitness(chromosome):
    weight=0
    for i in range(len(chromosome)):
        if chromosome[i] == 1:
            weight+=items[i][0]

    if weight > max_weight:
        return 0
    value = 0
    for i in range(len(chromosome)):
        if chromosome[i] == 1:
            value+=items[i][1]
    return value

# Generate the initial population
population = []
for i in range(population_size):
    chromosome = [random.randint(0, 1) for j in range(len(items))]
    population.append(chromosome)

# Implement the genetic algorithm
for generation in range(100):
    # Evaluate the fitness of each chromosome
    population_fitness = [fitness(chromosome) for chromosome in population]
    
    # Select the elite chromosomes
    elite_indexes = sorted(range(len(population_fitness)), key=lambda i: population_fitness[i], reverse=True)[:elite_size]
    elite = [population[i] for i in elite_indexes]
    
    # Select the parents for crossover using roulette wheel selection
    parents = []
    population_fitness_sum = sum(population_fitness)
    for _ in range(population_size - elite_size):
        rand = random.uniform(0, population_fitness_sum)
        for i, chromosome in enumerate(population):
            rand -= population_fitness[i]
            if rand <= 0:
                parents.append(chromosome)
                break
    
    # Perform crossover to generate the new population
    population = elite
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i+1]
        crossover_point = random.randint(1, len(items)-1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        population.append(child1)
        population.append(child2)
    
    # Perform mutation on the new population
    for i in range(len(population)):
        if random.uniform(0, 1) <= mutation_prob:
            mutation_point = random.randint(0, len(items)-1)
            population[i][mutation_point] = 1 - population[i][mutation_point]

# Find the best solution
best_chromosome = max(population, key=fitness)
best_weight = sum([items[i][0] for i in range(len(best_chromosome)) if best_chromosome[i] == 1])
best_value = fitness(best_chromosome)
# best_chromosome

print("Optimal solution:")
print("Chromosome: ", best_chromosome)
print("Weight: ", best_weight)
print("Value: ", best_value)

