import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    shifted_x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

class Layer:
    def __init__(self, entradas: int, neuronas: int, activacion: str = "relu", con_bias: bool = True, weights: np.ndarray = None):
        funciones = {
            "sigmoid": sigmoid,
            "softmax": softmax,
            "relu": relu,
            "leaky_relu": leaky_relu
        }

        self.neuronas = neuronas
        self.con_bias = con_bias
        
        if weights is not None:
            self.W = weights
        else:
            self.W = np.random.randn(entradas, neuronas)
            
        self.b = np.zeros((1, neuronas)) if con_bias else None
        self.activacion = funciones.get(activacion.lower(), relu)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.dot(x, self.W)
        if self.con_bias and self.b is not None:
            x += self.b
        return self.activacion(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


if __name__ == "__main__":
    x = np.array([[1, 2, 3],
                  [4, 5, 6]], dtype=np.float32)

    w = np.array([[0, 0, 0.1, 0.2],
                  [0.3, -1, 0.4, 0.5],
                  [1, 0.6, 0.7, 0.8]], dtype=np.float32)

    capa = Layer(entradas=3, neuronas=4, activacion="relu", con_bias=False, weights=w)

    salida = capa(x)
    print("Salida:\n", salida)
