# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------
# 1. Funciones de Membresía (Conjuntos Difusos Tipo 1)
# -------------------------------------------------------------
def mf_triangular(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Función de membresía Triangular con parámetros a <= b <= c."""
    x = np.asarray(x, dtype=float)
    izq = np.where(b == a, np.where(x >= a, 1.0, 0.0), (x - a) / np.where(b == a, 1.0, b - a))
    der = np.where(c == b, np.where(x <= c, 1.0, 0.0), (c - x) / np.where(c == b, 1.0, c - b))
    return np.maximum(0.0, np.minimum(izq, der))


def mf_trapezoidal(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Función de membresía Trapezoidal con parámetros a <= b <= c <= d."""
    x = np.asarray(x, dtype=float)
    izq = np.where(b == a, np.where(x >= a, 1.0, 0.0), (x - a) / np.where(b == a, 1.0, b - a))
    der = np.where(d == c, np.where(x <= d, 1.0, 0.0), (d - x) / np.where(d == c, 1.0, d - c))
    return np.maximum(0.0, np.minimum(np.minimum(izq, 1.0), der))


def mf_gaussiana(x: np.ndarray, c: float, sigma: float) -> np.ndarray:
    """Función de membresía Gaussiana con centro c y desviación estándar sigma."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * ((x - c) / sigma) ** 2)


def mf_campana(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Función de membresía Campana Generalizada (ancho a, pendiente b, centro c)."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.abs((x - c) / a) ** (2.0 * b))


def mf_sigmoide(x: np.ndarray, a: float, c: float) -> np.ndarray:
    """Función de membresía Sigmoide (pendiente a, punto de inflexión c)."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(a * (x - c), -500, 500)))


# -------------------------------------------------------------
# 2. Clase Variable Lingüística
# -------------------------------------------------------------
class VariableLinguistica:
    def __init__(self, nombre: str, universo: np.ndarray, unidad: str = ""):
        self.nombre = nombre
        self.universo = np.asarray(universo, dtype=float)
        self.unidad = unidad
        self.terminos = {}  # {nombre_termino: (funcion_mf, parametros, etiqueta_mf)}

    def agregar_termino(self, nombre_termino: str, funcion_mf, params: tuple, etiqueta_mf: str):
        self.terminos[nombre_termino] = (funcion_mf, params, etiqueta_mf)

    def evaluar_termino(self, nombre_termino: str, x: np.ndarray) -> np.ndarray:
        fn, params, _ = self.terminos[nombre_termino]
        return fn(x, *params)

    def fusificar(self, valor_crisp: float) -> dict:
        grados = {}
        for termino, (fn, params, _) in self.terminos.items():
            grados[termino] = float(fn(np.array([valor_crisp]), *params)[0])
        return grados

    def graficar(self, ax, valor_prueba: float = None):
        for termino, (fn, params, etiqueta_mf) in self.terminos.items():
            mu = fn(self.universo, *params)
            ax.plot(self.universo, mu, linewidth=2, label=f"{termino} ({etiqueta_mf})")

        if valor_prueba is not None:
            ax.axvline(
                valor_prueba,
                color="black",
                linestyle="--",
                alpha=0.7,
                label=f"Entrada x = {valor_prueba} {self.unidad}",
            )

        ax.set_title(f"Variable Lingüística: {self.nombre}", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Universo de Discurso [{self.unidad}]" if self.unidad else "Universo de Discurso")
        ax.set_ylabel("Grado de Membresía μ(x)")
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=8.5)


# -------------------------------------------------------------
# 3. Operadores de Conjuntos Difusos Tipo 1 (Una función c/u)
# -------------------------------------------------------------
def union_difusa(mu_A: np.ndarray, mu_B: np.ndarray, s_norma: str = "max") -> np.ndarray:
    """
    Operador de Unión para conjuntos difusos Tipo 1 (A ∪ B).
    Soporta S-Norma estándar 'max' y 'suma_probabilistica'.
    """
    mu_A = np.asarray(mu_A, dtype=float)
    mu_B = np.asarray(mu_B, dtype=float)
    if s_norma == "max":
        return np.maximum(mu_A, mu_B)
    elif s_norma == "suma_probabilistica":
        return mu_A + mu_B - (mu_A * mu_B)
    else:
        raise ValueError("S-Norma no soportada. Use 'max' o 'suma_probabilistica'.")


def interseccion_difusa(mu_A: np.ndarray, mu_B: np.ndarray, t_norma: str = "min") -> np.ndarray:
    """
    Operador de Intersección para conjuntos difusos Tipo 1 (A ∩ B).
    Soporta T-Norma estándar 'min' y 'producto'.
    """
    mu_A = np.asarray(mu_A, dtype=float)
    mu_B = np.asarray(mu_B, dtype=float)
    if t_norma == "min":
        return np.minimum(mu_A, mu_B)
    elif t_norma == "producto":
        return mu_A * mu_B
    else:
        raise ValueError("T-Norma no soportada. Use 'min' o 'producto'.")


def complemento_difuso(mu_A: np.ndarray) -> np.ndarray:
    """
    Operador de Complemento estándar para conjuntos difusos Tipo 1 (¬A = 1 - μ_A).
    """
    mu_A = np.asarray(mu_A, dtype=float)
    return 1.0 - mu_A


# -------------------------------------------------------------
# 4. Composición de Relaciones Difusas
# -------------------------------------------------------------
def composicion_difusa(R: np.ndarray, S: np.ndarray, t_norma: str = "min", verbose: bool = False) -> np.ndarray:
    """
    Calcula la composición relacional difusa T = R ∘ S:
        μ_{R ∘ S}(x_i, z_k) = max_j [ T_norma( μ_R(x_i, y_j), μ_S(y_j, z_k) ) ]
    """
    R = np.asarray(R, dtype=float)
    S = np.asarray(S, dtype=float)

    if R.ndim != 2 or S.ndim != 2:
        raise ValueError("Las relaciones R y S deben ser matrices bidimensionales.")
    if R.shape[1] != S.shape[0]:
        raise ValueError(f"Dimensiones incompatibles para composición: R{R.shape} y S{S.shape}.")

    filas, _ = R.shape
    columnas = S.shape[1]
    T = np.zeros((filas, columnas), dtype=float)

    for i in range(filas):
        for k in range(columnas):
            if t_norma == "min":
                pares = np.minimum(R[i, :], S[:, k])
                simbolo = "min"
            elif t_norma == "producto":
                pares = R[i, :] * S[:, k]
                simbolo = "prod"
            else:
                raise ValueError("T-Norma no soportada. Use 'min' o 'producto'.")

            T[i, k] = np.max(pares)

            if verbose:
                detalle = ", ".join(
                    [f"{simbolo}({R[i, j]:.2f}, {S[j, k]:.2f})={pares[j]:.4f}" for j in range(R.shape[1])]
                )
                print(f"  T[{i},{k}] = max({detalle}) = {T[i, k]:.4f}")

    return T


# -------------------------------------------------------------
# Pipeline Principal de Ejecución
# -------------------------------------------------------------
def ejecutar_actividad():
    # =========================================================
    # EJERCICIO 1: 3 Variables Lingüísticas (≥ 3 términos c/u)
    # =========================================================
    print("=" * 65)
    print(" 1. VARIABLES LINGÜÍSTICAS Y FUNCIONES DE MEMBRESÍA")
    print("=" * 65)

    # Variable 1: Temperatura (4 términos: Trapezoidal + Triangular + Gaussiana)
    temp = VariableLinguistica("Temperatura Ambiente", np.linspace(0, 50, 500), "°C")
    temp.agregar_termino("Muy Fría", mf_trapezoidal, (0, 0, 8, 15), "Trapezoidal")
    temp.agregar_termino("Fría", mf_triangular, (10, 18, 25), "Triangular")
    temp.agregar_termino("Templada", mf_gaussiana, (28, 4.0), "Gaussiana")
    temp.agregar_termino("Caliente", mf_trapezoidal, (32, 38, 50, 50), "Trapezoidal")

    # Variable 2: Humedad Relativa (3 términos: Sigmoide + Campana Generalizada)
    humedad = VariableLinguistica("Humedad Relativa", np.linspace(0, 100, 500), "%")
    humedad.agregar_termino("Seca", mf_sigmoide, (-0.25, 30), "Sigmoide Inv.")
    humedad.agregar_termino("Confortable", mf_campana, (15, 2.5, 50), "Campana Gen.")
    humedad.agregar_termino("Húmeda", mf_sigmoide, (0.25, 70), "Sigmoide")

    # Variable 3: Potencia del Ventilador (4 términos: Trapezoidal + Gaussiana + Triangular)
    potencia = VariableLinguistica("Potencia del Ventilador", np.linspace(0, 100, 500), "%")
    potencia.agregar_termino("Mínima", mf_trapezoidal, (0, 0, 15, 30), "Trapezoidal")
    potencia.agregar_termino("Baja", mf_gaussiana, (38, 9.0), "Gaussiana")
    potencia.agregar_termino("Media", mf_triangular, (45, 65, 82), "Triangular")
    potencia.agregar_termino("Alta", mf_trapezoidal, (72, 85, 100, 100), "Trapezoidal")

    variables_prueba = [(temp, 22.0), (humedad, 55.0), (potencia, 60.0)]
    for var, x_test in variables_prueba:
        grados = var.fusificar(x_test)
        print(f"\nFusificación de '{var.nombre}' para x = {x_test} {var.unidad}:")
        for term, mu in grados.items():
            print(f"  - μ_{term}({x_test}) = {mu:.4f}")

    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 11))
    for ax, (var, x_test) in zip(axes1, variables_prueba):
        var.graficar(ax, valor_prueba=x_test)
    plt.tight_layout()
    plt.savefig("ej1_variables_linguisticas.png", dpi=150)

    # =========================================================
    # EJERCICIO 2: Operadores Unión, Intersección y Complemento
    # =========================================================
    print("\n" + "=" * 65)
    print(" 2. OPERADORES DIFUSOS TIPO 1 (UNIÓN, INTERSECCIÓN, COMPLEMENTO)")
    print("=" * 65)

    A_disc = np.array([0.2, 0.5, 0.8, 1.0, 0.4])
    B_disc = np.array([0.6, 0.3, 0.8, 0.2, 0.9])

    union_AB = union_difusa(A_disc, B_disc)
    inter_AB = interseccion_difusa(A_disc, B_disc)
    comp_A = complemento_difuso(A_disc)
    comp_B = complemento_difuso(B_disc)

    df_ops = pd.DataFrame({
        "A": A_disc,
        "B": B_disc,
        "A ∪ B (Max)": union_AB,
        "A ∩ B (Min)": inter_AB,
        "¬A (1 - A)": comp_A,
        "¬(A ∪ B)": complemento_difuso(union_AB),
        "¬A ∩ ¬B": interseccion_difusa(comp_A, comp_B),
    })
    print("\nEvaluación sobre conjuntos difusos discretos (y verificación Ley de De Morgan):")
    print(df_ops.to_string(index=False))

    # Visualización continua sobre conjuntos 'Fría' y 'Templada'
    x_temp = temp.universo
    mu_fria = temp.evaluar_termino("Fría", x_temp)
    mu_templada = temp.evaluar_termino("Templada", x_temp)

    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 7))
    axes2[0, 0].plot(x_temp, mu_fria, label="A: Fría", linewidth=2)
    axes2[0, 0].plot(x_temp, mu_templada, label="B: Templada", linewidth=2)
    axes2[0, 0].set_title("Conjuntos Originales A y B")
    axes2[0, 0].legend()
    axes2[0, 0].grid(True, linestyle=":")

    axes2[0, 1].plot(x_temp, mu_fria, "--", alpha=0.5, label="A: Fría")
    axes2[0, 1].plot(x_temp, mu_templada, "--", alpha=0.5, label="B: Templada")
    axes2[0, 1].plot(x_temp, union_difusa(mu_fria, mu_templada), color="green", linewidth=2.5, label="A ∪ B (Max)")
    axes2[0, 1].set_title("Unión Difusa (A ∪ B)")
    axes2[0, 1].legend()
    axes2[0, 1].grid(True, linestyle=":")

    axes2[1, 0].plot(x_temp, mu_fria, "--", alpha=0.5, label="A: Fría")
    axes2[1, 0].plot(x_temp, mu_templada, "--", alpha=0.5, label="B: Templada")
    axes2[1, 0].plot(x_temp, interseccion_difusa(mu_fria, mu_templada), color="red", linewidth=2.5, label="A ∩ B (Min)")
    axes2[1, 0].set_title("Intersección Difusa (A ∩ B)")
    axes2[1, 0].legend()
    axes2[1, 0].grid(True, linestyle=":")

    axes2[1, 1].plot(x_temp, mu_fria, "--", alpha=0.5, label="A: Fría")
    axes2[1, 1].plot(x_temp, complemento_difuso(mu_fria), color="purple", linewidth=2.5, label="¬A (Complemento)")
    axes2[1, 1].set_title("Complemento Difuso (¬A = 1 - μ_A)")
    axes2[1, 1].legend()
    axes2[1, 1].grid(True, linestyle=":")

    plt.tight_layout()
    plt.savefig("ej2_operadores_difusos.png", dpi=150)

    # =========================================================
    # EJERCICIO 3: Composición R ∘ S (Comprobación Identidad)
    # =========================================================
    print("\n" + "=" * 65)
    print(" 3. COMPOSICIÓN DE RELACIONES DIFUSAS (VERIFICACIÓN)")
    print("=" * 65)

    R_ej3 = np.array([[1, 0],
                      [0, 1]], dtype=float)
    S_ej3 = np.array([[1, 0, 0],
                      [1, 0, 1]], dtype=float)

    print("Relación R (2x2):\n", R_ej3)
    print("Relación S (2x3):\n", S_ej3)
    print("\nCálculo paso a paso (Composición Max-Min):")
    T_ej3 = composicion_difusa(R_ej3, S_ej3, t_norma="min", verbose=True)
    print("\nResultado R ∘ S:\n", T_ej3)
    assert np.array_equal(T_ej3, S_ej3), "Error en la composición identidad."
    print("Verificación exitosa: Al ser R la relación identidad, R ∘ S = S.")

    # =========================================================
    # EJERCICIO 4: Composición con T-Norma = Producto
    # =========================================================
    print("\n" + "=" * 65)
    print(" 4. COMPOSICIÓN CON T-NORMA = PRODUCTO (MAX-PRODUCTO)")
    print("=" * 65)

    # Ejemplo clásico de composición relacional (Ross, Cap. 3).
    # Nota: Sustituye R_pres y S_pres si tu diapositiva usa otros valores numéricos.
    R_pres = np.array([[0.7, 0.5],
                       [0.8, 0.4]], dtype=float)
    S_pres = np.array([[0.9, 0.6, 0.2],
                       [0.1, 0.7, 0.5]], dtype=float)

    print("Relación R:\n", R_pres)
    print("Relación S:\n", S_pres)
    print("\nCálculo paso a paso con T-Norma = Producto (Max-Producto):")
    T_prod = composicion_difusa(R_pres, S_pres, t_norma="producto", verbose=True)
    T_min = composicion_difusa(R_pres, S_pres, t_norma="min", verbose=False)

    print("\nResultado Composición Max-Producto (T-Norma = Producto):\n", np.round(T_prod, 4))
    print("\nComparativa vs Composición Max-Min (T-Norma = Mínimo):\n", np.round(T_min, 4))

    plt.show()


if __name__ == "__main__":
    ejecutar_actividad()
