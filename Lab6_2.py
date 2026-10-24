import random

# Tiempos de ejecución asignados a cada tarea (T1 a T6)
TIEMPOS = [8, 5, 7, 3, 6, 4]


def calcular_cargas(cromosoma):
    # Calcula el tiempo total de ejecución asignado a P1, P2 y P3.
    cargas = [0, 0, 0]
    for i in range(len(cromosoma)):
        procesador = cromosoma[i]
        cargas[procesador - 1] += TIEMPOS[i]
    return cargas


def M(cromosoma):
    # Obtiene la máxima carga entre los procesadores M(x).
    return max(calcular_cargas(cromosoma))


def fitness(cromosoma):
    # Calcula la función de aptitud f(x) = 1 / (1 + M(x)).
    return 1.0 / (1.0 + M(cromosoma))


def seleccion_ruleta(poblacion, aptitudes):
    # Selección por Ruleta proporcional a la aptitud.
    suma_total = sum(aptitudes)
    ruleta = random.uniform(0, suma_total)
    acumulado = 0.0

    for i in range(len(poblacion)):
        acumulado += aptitudes[i]
        if acumulado >= ruleta:
            return poblacion[i]
    return poblacion[-1]


def cruce(padre1, padre2, p_cruce=0.8):
    # Cruce en un punto.
    if random.random() < p_cruce:
        punto = random.randint(1, len(padre1) - 1)
        hijo1 = padre1[:punto] + padre2[punto:]
        hijo2 = padre2[:punto] + padre1[punto:]
        return hijo1, hijo2
    return padre1[:], padre2[:]


def mutacion(cromosoma, p_mutacion=0.1):
    # Mutación aleatoria para valores decimales entre 1 y 3.
    nuevo_cromosoma = []
    for gen in cromosoma:
        if random.random() < p_mutacion:
            nuevo_cromosoma.append(random.randint(1, 3))
        else:
            nuevo_cromosoma.append(gen)
    return nuevo_cromosoma


def algoritmo_genetico(
    tam_poblacion=50, max_generaciones=1000, p_cruce=0.8, p_mutacion=0.1
):
    # Población inicial con asignaciones aleatorias a procesadores (1 a 3)
    poblacion = [
        [random.randint(1, 3) for _ in range(6)] for _ in range(tam_poblacion)
    ]

    for generacion in range(max_generaciones):
        aptitudes = [fitness(ind) for ind in poblacion]

        # La suma total de tiempos es 33 (8+5+7+3+6+4).
        # El balance perfecto asigna 11 a cada procesador, logrando M(x) = 11.
        mejor_ind = max(poblacion, key=fitness)
        if M(mejor_ind) == 11:
            return mejor_ind, generacion

        nueva_poblacion = []
        while len(nueva_poblacion) < tam_poblacion:
            p1 = seleccion_ruleta(poblacion, aptitudes)
            p2 = seleccion_ruleta(poblacion, aptitudes)

            h1, h2 = cruce(p1, p2, p_cruce)
            nueva_poblacion.append(mutacion(h1, p_mutacion))
            nueva_poblacion.append(mutacion(h2, p_mutacion))

        poblacion = nueva_poblacion[:tam_poblacion]

    return max(poblacion, key=fitness), max_generaciones


def mostrar_resultado(cromosoma, generacion):
    cargas = calcular_cargas(cromosoma)
    max_carga = max(cargas)
    fit_val = fitness(cromosoma)

    print(f"Cromosoma Solución: {cromosoma}")
    print(f"Generación alcanzada: {generacion}")
    print(f"Cargas por Procesador: P1={cargas[0]}, P2={cargas[1]}, P3={cargas[2]}")
    print(f"Máxima carga M(x): {max_carga}")
    print(f"Fitness f(x): {fit_val:.6f}\n")

    print("Detalle de asignación de tareas:")
    for i in range(len(cromosoma)):
        print(f"  Tarea T{i+1} (Tiempo {TIEMPOS[i]}) -> Procesador P{cromosoma[i]}")


if __name__ == "__main__":
    sol, gen = algoritmo_genetico()
    mostrar_resultado(sol, gen)
