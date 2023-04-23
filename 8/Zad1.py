import random

# Tworzymy początkową populację losowych chromosomów
population = [[] for i in range(10)]

for i in range(10):
    for j in range(10):
        population[i].append(random.randint(0,1))

# Wyświetlamy populację
for i in range(10):
    print(i, end=" ")

    for j in range(10):
        print(population[i][j], end="")
    print()

def fitness_function(chromosome):
  # Liczba genów o wartości 1 jest równa poziomowi przystosowania
  return sum(chromosome)

def mutate(chromosom):
    if random.uniform(0, 1) < 0.6:
        index = random.randint(0,9)
        #odwracamy wartość genu
        chromosom[index] = 1 - chromosom[index]

def darwin_evolution(population):
    i=0
    while(True):
        # Sortujemy populację według ilosci jedynek (od najlepszych do najgorszych)
        population.sort(key=fitness_function, reverse=True)

        # Wybieramy dwóch najlepszych i dwóch najgorszych osobników
        best_individuals = population[0:2]
        worst_individuals = population[8:10]

        # Jeśli ilość genów o wartości 1 w najlepszym osobniku wynosi 10, zwracamy "ilość "
        if fitness_function(best_individuals[0]) == 10:
            return i,population

        # Krzyżujemy dwóch najlepszych osobników i tworzymy potomków
        # Krzyżowanie jednopunktowe
        children = [best_individuals[0][:5] + best_individuals[1][5:],best_individuals[1][:5] + best_individuals[0][5:]]
        mutate(children[0])
        mutate(children[1])

        # Zastępujemy dwóch najgorszych osobników potomkami
        population[-2:] = children
        i+=1

# Wywołujemy algorytm ewolucyjny
res = darwin_evolution(population)
print("Ilość iteracji: " + str(res[0]))
print("Wynik: " + str(res[1][0]))


