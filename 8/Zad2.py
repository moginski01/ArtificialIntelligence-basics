import random

def mutate(chromosome):
    if random.uniform(0, 1) < 0.1:
        index = random.randint(0,7)
        chromosome[index] = 1 - chromosome[index]


def binary_to_decimal(binary_list):
    decimal_list = []
    #map służy do tego by zastosować funkcje z 1 argumentu na każdym elemencie listy 2 argumentu
    #jak zrobimy str(binary_list) to po prostu cała lista będzie stringiem a chcemy dostać liste stringow
    decimal_list.append(int("".join(map(str,binary_list[0:4])), 2))
    decimal_list.append(int("".join(map(str,binary_list[4:8])), 2))
    return decimal_list

# Algorytm genetyczny
def genetic_algorithm(population):
    i=0
    while(True):
        new_population = []
        for j in range(5):#5 bo potrzebujemy połowe ponieważ bierzemy dwóch rodziców
            parent1 = select_by_roulette(population)
            parent2 = select_by_roulette(population)
            parent_list = [parent1, parent2]
            # child1, child2 = crossover(parent1, parent2)
            children = [parent1[:4] + parent2[4:],parent2[:4] + parent1[4:]]


            
            which_parent = random.randint(0,1)
            which_child = random.randint(0,1)

            mutate(children[which_child])
            mutate(parent_list[which_parent])

            new_population.append(children[which_child])
            new_population.append(parent_list[which_parent])
        population = new_population
        i += 1
        best_chromosome = max(population, key=fitness_function)
        temp = fitness_function(best_chromosome)
        if fitness_function(best_chromosome) == 300:
            return i, population

def fitness_function(chromosome):
    a = int("".join(map(str, chromosome[0:4])), 2)
    b = int("".join(map(str, chromosome[4:8])), 2)
    #po prostu przekształcone równanie 0 = cała reszta, im dalej od 0 tym gozej
    # if a==0:
    #     return 0
    res_val = 300-abs(33 - 2*(a**2) - b)
    # if res_val > 300:
    #     temp = res_val-300
    #     res_val=-(2*temp)
    
    return res_val

# Selekcja metodą ruletki
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
    population = [[] for i in range(10)]

    for i in range(10):
        for j in range(8):
            population[i].append(random.randint(0,1))

    # Wyświetlamy populację
    print("Wylosowana populacja:")
    for i in range(10):
        print(i, end=" ")

        for j in range(8):
            print(population[i][j], end="")
        print()

    # Wywołujemy algorytm genetyczny
    iterations, final_population = genetic_algorithm(population)
    print("Ilość iteracji: " + str(iterations))
    # print(final_population)
    correct_chromosome = max(final_population, key=fitness_function)
    a,b = binary_to_decimal(correct_chromosome)
    print("a= ",a)
    print("b= ",b)
    print(correct_chromosome)

if __name__ == "__main__":
    main()

