import random

# 1 wartość to waga a druga to wartość
items = [(3, 266), (13, 442), (10, 671), (9, 526), (7, 388), (1, 245), (8, 210), (8, 145), (2, 126), (9, 322)]
max_weight = 35
population_size = 8
elite_size = int(population_size * 0.25)
mutation_prob = 0.05

# Generate the initial population
population = []
for i in range(population_size):
    chromosome = [random.randint(0, 1) for j in range(len(items))]
    population.append(chromosome)
    
def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.uniform(0, 1) < mutation_prob:
            chromosome[i] = 1 - chromosome[i]
    # if random.uniform(0, 1) < mutation_prob:
    #     index = random.randint(0,7)
    #     chromosome[index] = 1 - chromosome[index]

# Define the fitness function
def fitness_function(chromosome):
    weight=0
    for i in range(len(chromosome)):
        if chromosome[i] == 1:
            weight+=items[i][0]

    if weight > max_weight:
        return 0
    value = 0
    #obliczamy sume elementów plecaka
    for i in range(len(chromosome)):
        if chromosome[i] == 1:
            value+=items[i][1]
    return value

def genetic_algorithm(population):
    generations = 0
    while(generations<1000):
        # population_fitness2 = [fitness_function(chromosome) for chromosome in population]
        population_fitness = []
        #tu jest okej
        for i in range(len(population)):
            population_fitness.append([fitness_function(population[i]),i])

        # print(population_fitness)
        #wykorzystamy ze w pythonie z automatu sortuje po pierwszym elemencie
        population_fitness.sort(reverse=True)
        # test 
        
        elite = []
        for i in range(elite_size):
            for j in range(population_size):
                if fitness_function(population[j]) == population_fitness[i][0]:
                    elite.append(population[j])
                    break
        # print(elite)

        temp_population_size = population_size - len(elite)
        temp_population = []
        for i in range(temp_population_size):
            temp_population.append(select_by_roulette(population))
        

        #crossover
        population = elite
        for i in range(0, temp_population_size-1, 2):
            parent1, parent2 = temp_population[i], temp_population[i+1]
            crossover_point = 4
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            population.append(child1)
            population.append(child2)
        generations+=1

        for i in range(population_size):
            mutate(population[i])

        best_chromosome = max(population, key=fitness_function)
        best_weight = sum([items[i][0] for i in range(len(best_chromosome)) if best_chromosome[i] == 1])
        best_value = fitness_function(best_chromosome)

        if best_value>=2222:
            return best_chromosome,best_weight,best_value,generations

        # print(best_chromosome)
        # print(best_weight)
        # print(best_value)


    return best_chromosome,best_weight,best_value,generations


def select_by_roulette(population):
    #jak tak zrobimy to nie musimy tego robić na wartościach 0-1 wydaje mi się nieco prostsze wtedy
    total_fitness=0
    for chromosome in population:
        total_fitness+=fitness_function(chromosome)

    pick = random.uniform(0, total_fitness)
    current = 0
    for chromosome in population:
        #dodawanie bo ma sie "koło" zamykać 
        current += fitness_function(chromosome)
        if current > pick:
            return chromosome


def main():
    # Tworzymy początkową populację losowych chromosomów
    chromosome,weight,value,generations=genetic_algorithm(population)
    print("How many generations: " + str(generations))
    print("Final chromosome: " + str(chromosome))
    print("Final weight: " + str(weight))
    print("Final value: " + str(value))

if __name__ == "__main__":
    main()
