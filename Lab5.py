import random

# ==========================================
# DATOS INICIALES DEL PROBLEMA
# ==========================================
# Lista de valores y pesos de los 5 objetos disponibles
valores   = [5, 2, 4, 5, 1]
pesos     = [1, 2, 3, 4, 5]
capacidad = 5  # Límite máximo de peso de la mochila


# ==========================================
# 1. FUNCIÓN PARA EVALUAR CADA MOCHILA
# ==========================================
def calcular_puntaje(mochila):
    peso_total = 0
    valor_total = 0
    
    # Recorremos cada objeto de la mochila (del 0 al 4)
    for i in range(len(mochila)):
        # Si el valor en la mochila es 1, llevamos ese objeto
        if mochila[i] == 1:
            peso_total = peso_total + pesos[i]
            valor_total = valor_total + valores[i]
            
    # Si la mochila pesa más de lo permitido, aplicamos un castigo (penalización)
    if peso_total > capacidad:
        exceso = peso_total - capacidad
        penalizacion = exceso * 5  # Restamos 5 puntos por cada kg sobrante
        puntaje_final = valor_total - penalizacion
        
        # Evitamos que el puntaje sea negativo
        if puntaje_final < 0:
            puntaje_final = 0
            
        return puntaje_final
    else:
        return valor_total


# ==========================================
# 2. ALGORITMO GENÉTICO PRINCIPAL
# ==========================================
def algoritmo_genetico():
    cantidad_individuos = 20
    generaciones = 30
    
    # --- PASO A: Crear mochilas iniciales al azar ---
    poblacion = []
    for i in range(cantidad_individuos):
        mochila = []
        for j in range(len(valores)):
             decision = random.randint(0, 1)  # Genera 0 o 1 de forma aleatoria
             mochila.append(decision)
        poblacion.append(mochila)

    # --- PASO B: Evolucionar la población durante varias generaciones ---
    for gen in range(generaciones):
        
        # 1. Calcular el puntaje de cada mochila
        puntajes = []
        for mochila in poblacion:
            puntos = calcular_puntaje(mochila)
            puntajes.append(puntos)
            
        # 2. Agrupar puntaje con su mochila para poder ordenarlas
        lista_ordenada = []
        for i in range(len(poblacion)):
            lista_ordenada.append((puntajes[i], poblacion[i]))
            
        # Ordenar de mayor a menor puntaje
        lista_ordenada.sort(reverse=True)
        
        # Seleccionar las 10 mejores mochilas
        mejores_mochilas = []
        for par in lista_ordenada[:10]:
            mejores_mochilas.append(par[1])
            
        # 3. Crear la nueva generación combinando a los mejores padres
        nueva_poblacion = []
        
        # Conservamos a los 10 mejores directamente
        for mejor in mejores_mochilas:
            nueva_poblacion.append(mejor)
            
        # Llenamos los 10 espacios restantes con "hijos" mezclados
        while len(nueva_poblacion) < cantidad_individuos:
            # Elegir dos padres al azar entre las mejores mochilas
            padre1 = random.choice(mejores_mochilas)
            padre2 = random.choice(mejores_mochilas)
            
            # Elegir punto de corte para el cruce
            corte = random.randint(1, len(valores) - 1)
            
            # El hijo toma la primera parte del padre1 y la segunda del padre2
            hijo = padre1[:corte] + padre2[corte:]
            
            # Mutación: un 10% de probabilidad de cambiar un objeto al azar
            if random.random() < 0.1:
                posicion = random.randint(0, len(hijo) - 1)
                if hijo[posicion] == 0:
                    hijo[posicion] = 1
                else:
                    hijo[posicion] = 0
                    
            nueva_poblacion.append(hijo)
            
        poblacion = nueva_poblacion

    # --- PASO C: Retornar la mejor mochila encontrada ---
    mejor_mochila = poblacion[0]
    mejor_puntaje = calcular_puntaje(mejor_mochila)
    return mejor_mochila, mejor_puntaje


# ==========================================
# 3. EJECUCIÓN Y MOSTRAR RESULTADOS
# ==========================================
solucion, valor_maximo = algoritmo_genetico()

# Calcular el peso total de la mejor solución
peso_final = 0
for i in range(len(solucion)):
    if solucion[i] == 1:
        peso_final = peso_final + pesos[i]

print("=== MEJOR SOLUCIÓN ENCONTRADA ===")
print("Mochila (1 = Llevar objeto, 0 = Dejar objeto):", solucion)
print("Valor acumulado:", valor_maximo)
print("Peso total utilizado:", peso_final, "kg de", capacidad, "kg máximos")
