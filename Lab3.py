import numpy as np

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

        # Gradiente y actualización SGD (Diapositiva 7)
        gradiente_W = np.dot(capas[i].entrada_aumentada.T, deltas[i])
        capas[i].pesos -= alpha * gradiente_W

# -------------------------------------------------------------
# Pseudocódigo Entrenamiento
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
# Laboratorio 3
# -------------------------------------------------------------
if __name__ == "__main__":
    X = np.ones((3, 2)) * 5.0
    y = np.zeros((3, 2))

    # Construcción de la Red Multicapa Profunda
    capas = [
        Capa(entradas=2, neuronas=4, activacion=sigmoid, d_activacion=d_sigmoid, bias=True),
        Capa(entradas=4, neuronas=2, activacion=sigmoid, d_activacion=d_sigmoid, bias=True)
    ]

    print("--- Pasada hacia adelante inicial ---")
    YH_inicial = feedforward(capas, X)
    print("Predicción YH Inicial:\n", YH_inicial)
    print("Loss Inicial:", funcion_costo(y, YH_inicial))

    print("\n--- Iniciando Entrenamiento ---")
    YH_final = Train(trainX=X, trainT=y, epocas=3000, tol=1e-5, alpha=0.1, capas=capas)

    print("\n--- Resultado Final ---")
    print("Predicción Final YH:\n", YH_final)
    print("Loss Final:", funcion_costo(y, YH_final))
