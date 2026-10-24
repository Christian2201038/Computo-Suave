import random


def fitness(c):
    # Suma 1 por cada par de reinas en la misma fila o diagonal
    conflictos = sum(
        c[i] == c[j] or abs(c[i] - c[j]) == abs(i - j)
        for i in range(8)
        for j in range(i + 1, 8)
    )
    return 28 - conflictos


def torneo(poblacion):
    # Toma 3 individuos al azar y devuelve el de mayor fitness
    return max(random.sample(poblacion, 3), key=fitness)


def cruce(p1, p2):
    if random.random() < 0.8:
        pt = random.randint(1, 7)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
    return p1[:], p2[:]


def mutacion(cromosoma):
    nuevo_cromosoma = []
    for gen in cromosoma:
        if random.random() < 0.1:
            nuevo_cromosoma.append(random.randint(0, 7))
        else:
            nuevo_cromosoma.append(gen)
    return nuevo_cromosoma


def algoritmo_genetico():
    poblacion = [[random.randint(0, 7) for _ in range(8)] for _ in range(100)]

    for gen in range(10000):
        mejor = max(poblacion, key=fitness)
        if fitness(mejor) == 28:
            return mejor, gen

        nueva_poblacion = []
        while len(nueva_poblacion) < 100:
            h1, h2 = cruce(torneo(poblacion), torneo(poblacion))
            nueva_poblacion.extend([mutacion(h1), mutacion(h2)])

        poblacion = nueva_poblacion[:100]

    return max(poblacion, key=fitness), 100000


def mostrar_tablero(c):
    print("\nVisualización del Tablero:")
    print("Columna 0 1 2 3 4 5 6 7")
    for fila in range(8):
        print(f"Fila {fila}: " + " ".join("R" if g == fila else "." for g in c))


if __name__ == "__main__":
    sol, gen = algoritmo_genetico()
    print(f"Solución encontrada en generación {gen}: {sol}")
    print(f"Fitness: {fitness(sol)} / 28")
    mostrar_tablero(sol)
