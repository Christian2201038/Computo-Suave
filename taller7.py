# -*- coding: utf-8 -*-
import random
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, r2_score

# -------------------------------------------------------------
# Funciones de Activación
# -------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def deriv_leaky_relu(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

ACTIVACIONES = {
    'sigmoid': (sigmoid, d_sigmoid),
    'relu': (relu, d_relu),
    'leaky_relu': (leaky_relu, deriv_leaky_relu)
}

# -------------------------------------------------------------
# Función Costo MSE
# -------------------------------------------------------------
def funcion_costo(T, YH):
    return 0.5 * np.mean((T - YH) ** 2)

# -------------------------------------------------------------
# Clase Capa con Matriz Aumentada W_tilde
# -------------------------------------------------------------
class Capa:
    def __init__(self, entradas: int, neuronas: int, activacion, d_activacion, bias: bool = True):
        self.neuronas = neuronas
        self.activacion = activacion
        self.d_activacion = d_activacion
        self.bias = bias

        filas = entradas + 1 if bias else entradas
        self.pesos = np.random.uniform(-0.5, 0.5, (filas, neuronas))
        self.entrada_aumentada = None
        self.a = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.bias:
            col_ones = np.ones((x.shape[0], 1))
            self.entrada_aumentada = np.hstack([x, col_ones])
        else:
            self.entrada_aumentada = x

        self.a = np.dot(self.entrada_aumentada, self.pesos)
        return self.activacion(self.a)

# -------------------------------------------------------------
# Feed Forward
# -------------------------------------------------------------
def feedforward(capas: list, X: np.ndarray) -> np.ndarray:
    a = X
    for capa in capas:
        a = capa.forward(a)
    return a

# -------------------------------------------------------------
# Back Propagation
# -------------------------------------------------------------
def backpropagation(capas: list, T: np.ndarray, YH: np.ndarray, alpha: float):
    deltas = [None] * len(capas)
    num_capas = len(capas)

    for i in reversed(range(num_capas)):
        if i == num_capas - 1:
            de_dyh = -(T - YH)
            deltas[i] = de_dyh * capas[i].d_activacion(capas[i].a)
        else:
            pesos_sig = capas[i + 1].pesos
            if capas[i + 1].bias:
                pesos_sig = pesos_sig[:-1, :]

            deltas[i] = np.dot(deltas[i + 1], pesos_sig.T) * capas[i].d_activacion(capas[i].a)

        gradiente_W = np.dot(capas[i].entrada_aumentada.T, deltas[i])
        capas[i].pesos -= alpha * gradiente_W

# -------------------------------------------------------------
# Entrenamiento
# -------------------------------------------------------------
def Train(trainX: np.ndarray, trainT: np.ndarray, epocas: int, tol: float, alpha: float, capas: list, verbose=False):
    loss_final = 0.0
    for epoca in range(1, epocas + 1):
        YH = feedforward(capas, trainX)
        loss = funcion_costo(trainT, YH)
        loss_final = loss

        if loss <= tol:
            if verbose:
                print(f"Parada por tolerancia alcanzada en la época {epoca}. Loss: {loss:.6f}")
            break

        backpropagation(capas, trainT, YH, alpha)

        if verbose and (epoca % 500 == 0 or epoca == 1):
            print(f"Época {epoca}/{epocas} - Loss (MSE): {loss:.6f}")

    return feedforward(capas, trainX), loss_final

# -------------------------------------------------------------
# Funciones de Carga de Datasets
# -------------------------------------------------------------
def cargar_ionosphere(archivos_posibles=["ion/ionosphere.data", "ion/ionosphere.csv"]):
    archivo_encontrado = None
    for f in archivos_posibles:
        if os.path.exists(f):
            archivo_encontrado = f
            break
            
    if archivo_encontrado is None:
        raise FileNotFoundError("No se encontró 'ionosphere.data' ni 'ionosphere.csv' en el directorio.")

    print(f"Cargando dataset: {archivo_encontrado}")
    df = pd.read_csv(archivo_encontrado, header=None)
    
    # Las primeras 34 columnas son los atributos cuantitativos
    X = df.iloc[:, :-1].values.astype(float)
    # La última columna tiene 'g' (good) o 'b' (bad)
    y_str = df.iloc[:, -1].values
    y = np.where(y_str == 'g', 1, 0).reshape(-1, 1)
    
    return X, y

def cargar_german_dataset(filepath_numeric="german/german.data-numeric", filepath_data="german/german.data"):
    """
    Carga german.data-numeric o german.data (con One-Hot Encoding).
    Etiquetas mapeadas: 1 (Good) -> 0, 2 (Bad) -> 1.
    """
    if os.path.exists(filepath_numeric):
        print(f"Cargando dataset numérico: {filepath_numeric}")
        df = pd.read_csv(filepath_numeric, sep=r"\s+", header=None)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    elif os.path.exists(filepath_data):
        print(f"Cargando dataset original: {filepath_data}")
        df = pd.read_csv(filepath_data, sep=r"\s+", header=None)
        X_raw = df.iloc[:, :-1]
        y = df.iloc[:, -1].values
        X = pd.get_dummies(X_raw, drop_first=True).values.astype(float)
    else:
        raise FileNotFoundError("No se encontró 'german.data-numeric' ni 'german.data' en el directorio actual.")

    # Convertir etiquetas (1 = Good -> 0, 2 = Bad -> 1)
    y = np.where(y == 1, 0, 1).reshape(-1, 1)
    return X, y

# -------------------------------------------------------------
# Análisis Exploratorio de Datos
# -------------------------------------------------------------
def realizar_eda(nombre_dataset, X, y):
    print("\n" + "=" * 60)
    print(f" 1. ANÁLISIS EXPLORATORIO DE DATOS (EDA): {nombre_dataset}")
    print("=" * 60)
    print(f"Número de instancias (filas): {X.shape[0]}")
    print(f"Número de atributos (columnas X): {X.shape[1]}")
    
    clases, conteos = np.unique(y, return_counts=True)
    print("Distribución de la Variable Objetivo (y):")
    for c, cnt in zip(clases, conteos):
        pct = (cnt / len(y)) * 100
        print(f"  - Clase {int(c)}: {cnt} ejemplos ({pct:.2f}%)")
        
    print(f"Rango de datos X -> Mín: {X.min():.4f} | Máx: {X.max():.4f}")
    print(f"Valores nulos en X: {np.isnan(X).sum()}")

# -------------------------------------------------------------
# ALGORITMO GENÉTICO DE HIPERPARÁMETROS
# -------------------------------------------------------------
OPCIONES_HIPERPARAMETROS = {
    'neuronas': [8, 12, 16, 24, 32],
    'alpha': [0.001, 0.005, 0.01, 0.05],
    'activacion': ['relu', 'sigmoid', 'leaky_relu'],
    'epocas': [500, 1000, 1500]
}

def crear_individuo():
    return [
        random.choice(OPCIONES_HIPERPARAMETROS['neuronas']),
        random.choice(OPCIONES_HIPERPARAMETROS['alpha']),
        random.choice(OPCIONES_HIPERPARAMETROS['activacion']),
        random.choice(OPCIONES_HIPERPARAMETROS['epocas'])
    ]

def fitness_hiperparametros(cromosoma, trainX, trainT, valX, valT):
    neuronas, alpha, act_str, epocas = cromosoma
    act_func, d_act_func = ACTIVACIONES[act_str]
    
    capas = [
        Capa(entradas=trainX.shape[1], neuronas=neuronas, activacion=act_func, d_activacion=d_act_func, bias=True),
        Capa(entradas=neuronas, neuronas=1, activacion=sigmoid, d_activacion=d_sigmoid, bias=True)
    ]
    
    Train(trainX=trainX, trainT=trainT, epocas=epocas, tol=1e-5, alpha=alpha, capas=capas, verbose=False)
    
    probs_val = feedforward(capas, valX)
    preds_val = (probs_val >= 0.5).astype(int)
    return accuracy_score(valT, preds_val)

def torneo_ag(poblacion, aptitudes, k=3):
    indices = random.sample(range(len(poblacion)), k)
    best_idx = max(indices, key=lambda i: aptitudes[i])
    return poblacion[best_idx][:]

def cruce_ag(p1, p2, p_cruce=0.8):
    if random.random() < p_cruce:
        pt = random.randint(1, len(p1) - 1)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
    return p1[:], p2[:]

def mutacion_ag(cromosoma, p_mutacion=0.2):
    nuevo = cromosoma[:]
    if random.random() < p_mutacion:
        idx = random.randint(0, len(nuevo) - 1)
        keys = list(OPCIONES_HIPERPARAMETROS.keys())
        nuevo[idx] = random.choice(OPCIONES_HIPERPARAMETROS[keys[idx]])
    return nuevo

def algoritmo_genetico_hiperparametros(trainX, trainT, valX, valT, tam_poblacion=10, generaciones=5):
    print("\n--- Ejecutando Algoritmo Genético de Hiperparámetros ---")
    poblacion = [crear_individuo() for _ in range(tam_poblacion)]
    
    for gen in range(1, generaciones + 1):
        aptitudes = [fitness_hiperparametros(ind, trainX, trainT, valX, valT) for ind in poblacion]
        
        best_idx = max(range(len(poblacion)), key=lambda i: aptitudes[i])
        mejor_ind = poblacion[best_idx]
        mejor_fit = aptitudes[best_idx]
        
        print(f"Gen {gen}/{generaciones} - Mejor Accuracy (Validación): {mejor_fit:.4f} | Config: Neuronas={mejor_ind[0]}, Alpha={mejor_ind[1]}, Act={mejor_ind[2]}, Épocas={mejor_ind[3]}")
        
        nueva_poblacion = [mejor_ind[:]]
        
        while len(nueva_poblacion) < tam_poblacion:
            p1 = torneo_ag(poblacion, aptitudes)
            p2 = torneo_ag(poblacion, aptitudes)
            h1, h2 = cruce_ag(p1, p2)
            nueva_poblacion.append(mutacion_ag(h1))
            if len(nueva_poblacion) < tam_poblacion:
                nueva_poblacion.append(mutacion_ag(h2))
                
        poblacion = nueva_poblacion[:tam_poblacion]
        
    return mejor_ind, mejor_fit

# -------------------------------------------------------------
# Pipeline Principal de Evaluación
# -------------------------------------------------------------
def evaluar_dataset(nombre, fn_cargar):
    X_raw, y_raw = fn_cargar()
    
    # 1. EDA
    realizar_eda(nombre, X_raw, y_raw)
    
    # Partición: 60% Train, 20% Val, 20% Test
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
        X_train_raw, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    # Normalización Min-Max
    x_min, x_max = X_train_raw.min(axis=0), X_train_raw.max(axis=0)
    X_train = (X_train_raw - x_min) / (x_max - x_min + 1e-8)
    X_test = (X_test_raw - x_min) / (x_max - x_min + 1e-8)
    
    X_tr_s = (X_tr_sub - x_min) / (x_max - x_min + 1e-8)
    X_val_s = (X_val_sub - x_min) / (x_max - x_min + 1e-8)
    
    # 2. Red Neuronal Base Propuesta
    print("\n" + "=" * 60)
    print(f" 2. ENTRENAMIENTO MODELO BASE: {nombre}")
    print("=" * 60)
    capas_base = [
        Capa(entradas=X_train.shape[1], neuronas=12, activacion=relu, d_activacion=d_relu, bias=True),
        Capa(entradas=12, neuronas=1, activacion=sigmoid, d_activacion=d_sigmoid, bias=True)
    ]
    _, loss_base = Train(trainX=X_train, trainT=y_train, epocas=1000, tol=1e-5, alpha=0.005, capas=capas_base, verbose=True)
    
    probs_test_base = feedforward(capas_base, X_test)
    preds_test_base = (probs_test_base >= 0.5).astype(int)
    acc_base = accuracy_score(y_test, preds_test_base)
    auc_base = roc_auc_score(y_test, probs_test_base)
    
    print(f"\n[Resultado Modelo Base] Loss Final: {loss_base:.6f} | Accuracy Test: {acc_base:.4f} | ROC-AUC Test: {auc_base:.4f}")
    
    # 3. Optimización mediante Algoritmo Genético
    print("\n" + "=" * 60)
    print(f" 3. OPTIMIZACIÓN DE PARÁMETROS CON AG: {nombre}")
    print("=" * 60)
    best_config, _ = algoritmo_genetico_hiperparametros(X_tr_s, y_tr_sub, X_val_s, y_val_sub, tam_poblacion=10, generaciones=5)
    
    neuronas_opt, alpha_opt, act_opt, epocas_opt = best_config
    act_fn_opt, d_act_fn_opt = ACTIVACIONES[act_opt]
    
    capas_opt = [
        Capa(entradas=X_train.shape[1], neuronas=neuronas_opt, activacion=act_fn_opt, d_activacion=d_act_fn_opt, bias=True),
        Capa(entradas=neuronas_opt, neuronas=1, activacion=sigmoid, d_activacion=d_sigmoid, bias=True)
    ]
    
    print(f"\nEntrenando Modelo Optimizado (AG) con {epocas_opt} épocas y alpha={alpha_opt}...")
    _, loss_opt = Train(trainX=X_train, trainT=y_train, epocas=epocas_opt, tol=1e-5, alpha=alpha_opt, capas=capas_opt, verbose=False)
    
    probs_test_opt = feedforward(capas_opt, X_test)
    preds_test_opt = (probs_test_opt >= 0.5).astype(int)
    acc_opt = accuracy_score(y_test, preds_test_opt)
    auc_opt = roc_auc_score(y_test, probs_test_opt)
    
    # 4. Tabla Comparativa
    print("\n" + "=" * 60)
    print(f" 4. COMPARACIÓN DE RESULTADOS: {nombre}")
    print("=" * 60)
    df_comparacion = pd.DataFrame({
        "Parámetro / Métrica": [
            "Neuronas Capa Oculta",
            "Tasa Aprendizaje (Alpha)",
            "Función Activación",
            "Épocas",
            "Pérdida Total (MSE Train)",
            "Accuracy (Test)",
            "ROC-AUC (Test)"
        ],
        "Modelo Base": [
            12, 0.005, "relu", 1000,
            f"{loss_base:.6f}", f"{acc_base:.4f}", f"{auc_base:.4f}"
        ],
        "Modelo Optimizado (AG)": [
            neuronas_opt, alpha_opt, act_opt, epocas_opt,
            f"{loss_opt:.6f}", f"{acc_opt:.4f}", f"{auc_opt:.4f}"
        ]
    })
    print(df_comparacion.to_string(index=False))

if __name__ == "__main__":
    evaluar_dataset("GERMAN CREDIT DATASET", cargar_german_dataset)
    evaluar_dataset("IONOSPHERE DATASET", cargar_ionosphere)
