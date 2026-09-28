# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 14:39:32 2026

@author: Chris
"""

import numpy as np
import pandas as pd

    
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, precision_score, roc_auc_score

# -------------------------------------------------------------
# Funciones de Activación y sus Derivadas
# -------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1.0 - s)

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

        # Pesos aumentados: (entradas + 1, neuronas) si tiene bias
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
        # Si la capa es la última:
        if i == num_capas - 1:
            de_dyh = -(T - YH)
            deltas[i] = de_dyh * capas[i].d_activacion(capas[i].a)
        # De lo contrario:
        else:
            pesos_sig = capas[i + 1].pesos
            if capas[i + 1].bias:
                pesos_sig = pesos_sig[:-1, :]  # Quitar pesos del bias para propagar

            deltas[i] = np.dot(deltas[i + 1], pesos_sig.T) * capas[i].d_activacion(capas[i].a)

        # Gradiente y actualización SGD
        gradiente_W = np.dot(capas[i].entrada_aumentada.T, deltas[i])
        capas[i].pesos -= alpha * gradiente_W

# -------------------------------------------------------------
# Entrenamiento
# -------------------------------------------------------------
def Train(trainX: np.ndarray, trainT: np.ndarray, epocas: int, tol: float, alpha: float, capas: list):
    for epoca in range(1, epocas + 1):
        YH = feedforward(capas, trainX)
        loss = funcion_costo(trainT, YH)

        if loss <= tol:
            print(f"Parada por tolerancia alcanzada en la época {epoca}. Loss: {loss:.6f}")
            break

        backpropagation(capas, trainT, YH, alpha)

        if epoca % 500 == 0 or epoca == 1:
            print(f"Época {epoca}/{epocas} - Loss (MSE): {loss:.6f}")

    return feedforward(capas, trainX)

# -------------------------------------------------------------
# Laboratorio 4
# -------------------------------------------------------------
if __name__ == "__main__":

    # =============================================================
    # DATASET 1: energy.csv (Regresión Multi-Salida: Y1, Y2)
    # =============================================================
    print("=" * 60)
    print(" EVALUACIÓN DATASET 1: energy.csv (Regresión Multi-Salida)")
    print("=" * 60)

    df_energy = pd.read_csv("energy.csv")

    X_raw_e = df_energy[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]].values
    y_raw_e = df_energy[["Y1", "Y2"]].values

    # Separación 40% Train / 60% Test
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
        X_raw_e, y_raw_e, test_size=0.6, random_state=42
    )

    # Normalización Min-Max
    x_min_e, x_max_e = X_train_e.min(axis=0), X_train_e.max(axis=0)
    X_train_e = (X_train_e - x_min_e) / (x_max_e - x_min_e + 1e-8)
    X_test_e = (X_test_e - x_min_e) / (x_max_e - x_min_e + 1e-8)

    y_min_e, y_max_e = y_train_e.min(axis=0), y_train_e.max(axis=0)
    y_train_e_norm = (y_train_e - y_min_e) / (y_max_e - y_min_e + 1e-8)

    capas_energy = [
        Capa(
            entradas=8,
            neuronas=16,
            activacion=sigmoid,
            d_activacion=d_sigmoid,
            bias=True,
        ),
        Capa(
            entradas=16,
            neuronas=2,
            activacion=sigmoid,
            d_activacion=d_sigmoid,
            bias=True,
        ),
    ]

    print("\n--- Entrenamiento Energy ---")
    Train(
        trainX=X_train_e,
        trainT=y_train_e_norm,
        epocas=3000,
        tol=1e-5,
        alpha=0.1,
        capas=capas_energy,
    )

    # Evaluación
    pred_norm_e = feedforward(capas_energy, X_test_e)
    pred_test_e = pred_norm_e * (y_max_e - y_min_e + 1e-8) + y_min_e

    # Métricas de Regresión
    r2_y1 = r2_score(y_test_e[:, 0], pred_test_e[:, 0])
    r_y1 = np.corrcoef(y_test_e[:, 0], pred_test_e[:, 0])[0, 1]

    r2_y2 = r2_score(y_test_e[:, 1], pred_test_e[:, 1])
    r_y2 = np.corrcoef(y_test_e[:, 1], pred_test_e[:, 1])[0, 1]

    print("\n--- Resultados Test (energy.csv) ---")
    print(f"Variable Y1 -> R^2: {r2_y1:.4f} | R: {r_y1:.4f}")
    print(f"Variable Y2 -> R^2: {r2_y2:.4f} | R: {r_y2:.4f}")

    # =============================================================
    # DATASET 2: house.csv (Regresión)
    # =============================================================
    print("\n" + "=" * 60)
    print(" EVALUACIÓN DATASET 2: house.csv (Regresión)")
    print("=" * 60)

    df_house = pd.read_csv("house.csv")

    X_raw_h = df_house[["bedrooms", "bathrooms", "sqft_living","sqft_lot","floors","waterfront","view","condition", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "price_per_sqft"]].values
    y_raw_h = df_house[["price"]].values

    # Separación 40% Train / 60% Test
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X_raw_h, y_raw_h, test_size=0.6, random_state=42
    )

    # Normalización Min-Max
    x_min_h, x_max_h = X_train_h.min(axis=0), X_train_h.max(axis=0)
    X_train_h = (X_train_h - x_min_h) / (x_max_h - x_min_h + 1e-8)
    X_test_h = (X_test_h - x_min_h) / (x_max_h - x_min_h + 1e-8)

    y_min_h, y_max_h = y_train_h.min(axis=0), y_train_h.max(axis=0)
    y_train_h_norm = (y_train_h - y_min_h) / (y_max_h - y_min_h + 1e-8)

    capas_house = [
        Capa(
            entradas=X_train_h.shape[1],
            neuronas=12,
            activacion=sigmoid,
            d_activacion=d_sigmoid,
            bias=True,
        ),
        Capa(
            entradas=12,
            neuronas=1,
            activacion=sigmoid,
            d_activacion=d_sigmoid,
            bias=True,
        ),
    ]

    print("\n--- Entrenamiento House ---")
    Train(
        trainX=X_train_h,
        trainT=y_train_h_norm,
        epocas=3000,
        tol=1e-5,
        alpha=0.1,
        capas=capas_house,
    )

    # Evaluación
    pred_norm_h = feedforward(capas_house, X_test_h)
    pred_test_h = pred_norm_h * (y_max_h - y_min_h + 1e-8) + y_min_h

    # Métricas de Regresión
    r2_h = r2_score(y_test_h, pred_test_h)
    r_h = np.corrcoef(y_test_h.squeeze(), pred_test_h.squeeze())[0, 1]

    print("\n--- Resultados Test (house.csv) ---")
    print(f"Coeficiente de Determinación (R^2): {r2_h:.4f}")
    print(f"Coeficiente de Correlación (R): {r_h:.4f}")

    # =============================================================
    # DATASET 3: cancer.csv (Clasificación Binaria)
    # =============================================================
    print("\n" + "=" * 60)
    print(" EVALUACIÓN DATASET 3: cancer.csv (Clasificación)")
    print("=" * 60)

    df_cancer = pd.read_csv("cancer.csv")

    # Convertir 'M' a 1 y 'B' a 0 en la columna diagnosis
    df_cancer["diagnosis"] = df_cancer["diagnosis"].map({"M": 1, "B": 0})
    X_raw_c = df_cancer.drop(columns=["id", "diagnosis","Unnamed: 32"]).values

    y_raw_c = df_cancer[["diagnosis"]].values

    # Separación 40% Train / 60% Test
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_raw_c, y_raw_c, test_size=0.6, random_state=42
    )

    # Normalización Min-Max
    x_min_c, x_max_c = X_train_c.min(axis=0), X_train_c.max(axis=0)
    X_train_c = (X_train_c - x_min_c) / (x_max_c - x_min_c + 1e-8)
    X_test_c = (X_test_c - x_min_c) / (x_max_c - x_min_c + 1e-8)

    capas_cancer = [
        Capa(
            entradas=X_train_c.shape[1],
            neuronas=16,
            activacion=sigmoid,
            d_activacion=d_sigmoid,
            bias=True,
        ),
        Capa(
            entradas=16,
            neuronas=1,
            activacion=sigmoid,
            d_activacion=d_sigmoid,
            bias=True,
        ),
    ]

    print("\n--- Entrenamiento Cancer ---")
    Train(
        trainX=X_train_c,
        trainT=y_train_c,
        epocas=3000,
        tol=1e-5,
        alpha=0.01,
        capas=capas_cancer,
    )

    # Predicción
    probs_test_c = feedforward(capas_cancer, X_test_c)
    preds_bin_c = (probs_test_c >= 0.5).astype(int)

    # Métricas de Clasificación
    acc_c = accuracy_score(y_test_c, preds_bin_c)
    prec_c = precision_score(y_test_c, preds_bin_c)
    auc_c = roc_auc_score(y_test_c, probs_test_c)

    print("\n--- Resultados Test (cancer.csv) ---")
    print(f"Exactitud (Accuracy): {acc_c:.4f}")
    print(f"Precisión (Precision): {prec_c:.4f}")
    print(f"Área Bajo la Curva (ROC-AUC): {auc_c:.4f}")
