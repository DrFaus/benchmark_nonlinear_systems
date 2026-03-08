import pandas as pd
import numpy as np
from math import exp, sqrt
import matplotlib.pyplot as plt

f_objetivo_critico = 1.0
factor_01 = 0 # CSTR: 0 moderadamente no lineal, 1 altamente no lineal
factor_02 = 1 # FIS Type: 0 Mamdani, 1 Tsukamoto
factor_03 = 1 # Parametrización: 0 Discreto, 1 continuo
factor_04 = 1 # 0 incremental-form, 1 level-form


if factor_01 == 0:
    # df = pd.read_csv("https://raw.githubusercontent.com/DrFaus/benchmark_nonlinear_systems/refs/heads/main/datos_casos/caso_01_semilla_147.csv")
    df = pd.read_csv("https://raw.githubusercontent.com/DrFaus/benchmark_nonlinear_systems/refs/heads/main/datos_casos/caso_01_semilla_150.csv")
else:
    df = pd.read_csv("https://raw.githubusercontent.com/DrFaus/benchmark_nonlinear_systems/refs/heads/main/datos_casos/caso_02_semilla_103.csv")

df = df.loc[:999,:]
x1_prom = float(df.y1.mean())
x2_prom = float(df.u.mean())

y_prom = factor_04 * float(df.y2.mean())

numero_datos_entrenamiento = df.shape[0]

list_x1 = list(df.y1)
list_x2 = list(df.u)
list_respuesta = list(df.y2)

prefijo = "https://raw.githubusercontent.com/DrFaus/benchmark_nonlinear_systems/refs/heads/main/datos_experimentos/"
sufijo = f"experimento_{factor_01}_{factor_02}_{factor_03}_{factor_04}.csv"

soluciones = pd.read_csv(prefijo+sufijo)

lineal = lambda x0, x1, y: y * (x1 - x0) + x0


def trapezoidal(params, w):
    a, b, c, d =  tuple(params)
    return max([min([(w - a) / (b - a), min([1.0, (d - w) / (d - c)])]), 0.0])

def triangular(params, w):
    a, b, c = tuple(params)
    return max([min([(w - a) / (b - a), (c - w) / (c - b)]), 0.0])

def base_de_reglas(reglas):
    meta_params = [[-10, -9, 1, 2], [1, 2, 2, 3], [2, 3, 4, 10]]
    base = np.zeros(27).reshape((3, 3, 3))

    fila = 0
    columna = 0

    for idx in range(9):
        base[fila, columna, 0] = trapezoidal(meta_params[0], reglas[idx])
        base[fila, columna, 1] = trapezoidal(meta_params[1], reglas[idx])
        base[fila, columna, 2] = trapezoidal(meta_params[2], reglas[idx])

        fila += 1

        if fila > 2:
            fila = 0
            columna += 1
    return base

def fis_tsukamoto(base, params, entrada_01, entrada_02):
    antecedente_01 = []
    antecedente_01.append(trapezoidal(params[0:4], entrada_01))
    antecedente_01.append(triangular(params[4:7], entrada_01))
    antecedente_01.append(trapezoidal(params[7:11], entrada_01))

    antecedente_02 = []
    antecedente_02.append(trapezoidal(params[11:15], entrada_02))
    antecedente_02.append(triangular(params[15:18], entrada_02))
    antecedente_02.append(trapezoidal(params[18:22], entrada_02))

    numerador, denominador = 0.0, 0.0

    for i in range(3):
        for j in range(3):
            implicacion_larsen = antecedente_01[i] * antecedente_02[j]
            denominador += implicacion_larsen 
            for k in range(3):
                a = params[22 + 2 * k]
                b = params[23 + 2 * k]
                lineal_k = lineal(a, b, implicacion_larsen)
                numerador += implicacion_larsen * lineal_k * base[i, j, k]
    
    if abs(denominador) < 1e-10:
        denominador = 1e-10
    return numerador / denominador


def x_bar_trapecio(x0, x1, altura_izquierda, altura_derecha):
    h_i = altura_izquierda 
    h_d = altura_derecha 

    b = x1 - x0 
    
    if (h_i+h_d) == 0 or (b == 0):
        x_bar = 0.0
    else:
        x_bar = x0 + b * (h_i + 2.0 * h_d) / (3.0 * (h_i + h_d)) 

    return x_bar

def area_trapecio(x0, x1, altura_izquierda, altura_derecha):
    h_i = altura_izquierda 
    h_d = altura_derecha 
    b = x1 - x0 

    return (h_i + h_d) * b / 2.0


def zona_conflicto(w1, w2, h1, h2):
    areas = 4
    wp = 0.5 * (w1 + w2)

    if (h1 + h2) < 1e-7:
        x_bar = 0.0
        area_total = 0.0
        return x_bar, area_total
    
    if (h1 >= 0.5 and h2 >= 0.5):
        w1x = w2 + h1 * (w1 - w2)
        w2x = w1 + h2 * (w2 - w1)
    elif (h1 >= 0.5 and h2 < 0.5):
        w1x = w2 + h1 * (w1 - w2)
        w2x = w2 + h2 * (w1 - w2)
    elif (h1 < 0.5 and h2 >= 0.5):
        w1x = w1 + h1 * (w2 - w1)
        w2x = w1 + h2 * (w2 - w1)
    elif (h1 < 0.5 and h2 < 0.5 and h1 >= h2):
        w1x = w2 + h1 * (w1 - w2)
        w2x = w2 + h2 * (w1 - w2)
        areas = 3
    else:
        w1x = w1 + h1 * (w2 - w1)
        w2x = w1 + h2 * (w2 - w1)
        areas = 3

    if (areas == 3):
        area_1 = area_trapecio(w1,  w1x, h1, h1)
        area_2 = area_trapecio(w1x, w2x, h1, h2)
        area_3 = area_trapecio(w2x, w2,  h2, h2)
        area_4 = 0.0

        x_bar_1 = x_bar_trapecio(w1,  w1x, h1, h1)
        x_bar_2 = x_bar_trapecio(w1x, w2x, h1, h2)
        x_bar_3 = x_bar_trapecio(w2x, w2,  h2, h2)
        x_bar_4 = 0.0
    else:
        area_1 = area_trapecio(w1,  w1x, h1,  h1)
        area_2 = area_trapecio(w1x, wp,  h1,  0.5)
        area_3 = area_trapecio(wp,  w2x, 0.5, h2)
        area_4 = area_trapecio(w2x, w2,  h2,  h2)

        x_bar_1 = x_bar_trapecio(w1,  w1x, h1,  h1)
        x_bar_2 = x_bar_trapecio(w1x, wp,  h1,  0.5)
        x_bar_3 = x_bar_trapecio(wp,  w2x, 0.5, h2)
        x_bar_4 = x_bar_trapecio(w2x, w2,  h2,  h2)

    area_total = area_1 + area_2 + area_3 + area_4
    denom = area_total

    if (abs(denom) < 1e-14):
        x_bar = 0.0
        area_total = 0.0
        return x_bar, area_total

    x_bar = (area_1*x_bar_1 + area_2*x_bar_2 + area_3*x_bar_3 + area_4*x_bar_4) / denom
    return x_bar, area_total

def centroide_total(params, fuzs):
    a0, w1, w2, w3, b0 = tuple(params)
    h1, h2, h3 = tuple(fuzs)

    # Left rectangle: [a0, w1] at height h1
    areas, ponderados = 0.0, 0.0 
    area = (w1 - a0) * h1
    ponderado = area * 0.5 * (w1 + a0)
    areas += area
    ponderados += ponderado

    # Conflict zone between (w1,w2) with heights (h1,h2)
    c, a_conf = zona_conflicto(w1, w2, h1, h2)
    ponderados += c * a_conf
    areas = areas + a_conf

    # Conflict zone between (w2,w3) with heights (h2,h3)
    c, a_conf = zona_conflicto(w2, w3, h2, h3)
    ponderados += c * a_conf
    areas += a_conf

    # Right rectangle: [w3, b0] at height h3
    area = (b0 - w3) * h3
    ponderado = area * 0.5 * (w3 + b0)
    areas += area
    ponderados += ponderado

    # Protect against division by zero
    if (abs(areas) < 1e-14):
        centroide = 0.0
    else:
        centroide = ponderados / areas
    return centroide

def fis_mamdani(base, params, entrada_01, entrada_02):
    antecedente_01 = []
    antecedente_01.append(trapezoidal(params[0:4], entrada_01))
    antecedente_01.append(triangular(params[4:7], entrada_01))
    antecedente_01.append(trapezoidal(params[7:11], entrada_01))

    antecedente_02 = []
    antecedente_02.append(trapezoidal(params[11:15], entrada_02))
    antecedente_02.append(triangular(params[15:18], entrada_02))
    antecedente_02.append(trapezoidal(params[18:22], entrada_02))

    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    tau = 8.0

    for i in range(3):
        for j in range(3):
            alpha = antecedente_01[i] * antecedente_02[j]
            s1 += base[i, j, 0] * alpha
            s2 += base[i, j, 1] * alpha
            s3 += base[i, j, 2] * alpha
    
    den = exp(tau * s1) + exp(tau * s2) + exp(tau * s3)

    fuzs = []
    fuzs.append(exp(tau * s1) / den)
    fuzs.append(exp(tau * s2) / den)
    fuzs.append(exp(tau * s3) / den)

    puntos = [params[23], params[27], params[28], params[29], params[32]]

    salida = centroide_total(puntos, fuzs)

    return salida


def tsukamoto_entrenamiento(x):
    RMSE_crit = 1.0
    SSE = 0.0

    if factor_03 == 0:
        base = base_de_reglas([round(i) for i in x[0:9]])
    else:
        base = base_de_reglas(x[0:9])
    
    antecedente01 = [x1_prom - x[9], x1_prom, x1_prom + x[10]]
    antecedente02 = [x2_prom - x[11], x2_prom, x2_prom + x[12]]
    params = []
    params.append(-1000000.0)
    params.append(-100000.0)
    params.append(antecedente01[0])
    params.append(antecedente01[1])

    params.append(antecedente01[0])
    params.append(antecedente01[1])
    params.append(antecedente01[2])

    params.append(antecedente01[1])
    params.append(antecedente01[2])
    params.append(100000.0)
    params.append(1000000.0)


    params.append(-1000000.0)
    params.append(-100000.0)
    params.append(antecedente02[0])
    params.append(antecedente02[1])

    params.append(antecedente02[0])
    params.append(antecedente02[1])
    params.append(antecedente02[2])

    params.append(antecedente02[1])
    params.append(antecedente02[2])
    params.append(100000.0)
    params.append(1000000.0)

    params.append(y_prom - x[13])
    params.append(y_prom - x[14])
    params.append(y_prom)
    params.append(y_prom)
    params.append(y_prom + x[15])
    params.append(y_prom + x[16])

    prediccion_0 = list_x1[0]
    for i in range(len(list_x1)):
        prediccion_1 = prediccion_0 * (float(factor_04 == 0)) + fis_tsukamoto(base, params, prediccion_0, list_x2[i])
        SSE += (list_respuesta[i] - prediccion_1)**2
        RMSE = sqrt(SSE / (i+1))
        if (RMSE > RMSE_crit):
            RMSE = sqrt(SSE + (float(numero_datos_entrenamiento) - i - 1) * RMSE_crit ** 2) / sqrt(float(numero_datos_entrenamiento))
            break
        prediccion_0 = prediccion_1
        lista_entrenamiento.append(prediccion_1)
    
    return RMSE

def mamdani_entrenamiento(x):
    RMSE_crit = 1.0
    SSE = 0.0

    if factor_03 == 0:
        base = base_de_reglas([round(i) for i in x[0:9]])
    else:
        base = base_de_reglas(x[0:9])
    
    antecedente01 = [x1_prom - x[9], x1_prom, x1_prom + x[10]]
    antecedente02 = [x2_prom - x[11], x2_prom, x2_prom + x[12]]
    params = []
    params.append(-1000000.0)
    params.append(-100000.0)
    params.append(antecedente01[0])
    params.append(antecedente01[1])

    params.append(antecedente01[0])
    params.append(antecedente01[1])
    params.append(antecedente01[2])

    params.append(antecedente01[1])
    params.append(antecedente01[2])
    params.append(100000.0)
    params.append(1000000.0)


    params.append(-1000000.0)
    params.append(-100000.0)
    params.append(antecedente02[0])
    params.append(antecedente02[1])

    params.append(antecedente02[0])
    params.append(antecedente02[1])
    params.append(antecedente02[2])

    params.append(antecedente02[1])
    params.append(antecedente02[2])
    params.append(100000.0)
    params.append(1000000.0)

    params.append(y_prom - x[13] - x[14] - 0.00001)
    params.append(y_prom - x[13] - x[14])
    params.append("espureo")
    params.append(y_prom - x[13])
    params.append(y_prom)

    params.append(y_prom - x[13])
    params.append(y_prom)
    params.append(y_prom + x[15])

    params.append(y_prom)
    params.append(y_prom + x[15])
    params.append(y_prom + x[15] + x[16])
    params.append(y_prom + x[15] + x[16] + 0.00001)

    prediccion_0 = list_x1[0]
    for i in range(len(list_x1)):
        prediccion_1 = prediccion_0 * (float(factor_04 == 0)) + fis_mamdani(base, params, prediccion_0, list_x2[i])
        SSE += (list_respuesta[i] - prediccion_1)**2
        RMSE = sqrt(SSE / (i+1))
        if (RMSE > RMSE_crit):
            RMSE = sqrt(SSE + (float(numero_datos_entrenamiento) - i - 1) * RMSE_crit ** 2) / sqrt(float(numero_datos_entrenamiento))
            break
        prediccion_0 = prediccion_1
        lista_entrenamiento.append(prediccion_1)
    
    return RMSE

def grafico_comparacion(xp, yp, xs, ys, titulo, plot_label_01, plot_label_02, x_label, y_label, save_as):
  # --- Configuración general del gráfico ---
  plt.figure(
    figsize=(16, 4),   # tamaño de la figura (ancho, alto) en pulgadas
    dpi=120           # resolución del gráfico
  )

  plt.plot(
    xp, yp,
    color='blue',   # color de la línea
    linewidth=1.0,        # grosor de la línea
    linestyle='-',        # estilo de línea
    marker='o',           # marcador en cada punto
    markersize=0,         # tamaño del marcador
    markerfacecolor='white',
    markeredgecolor='black',
    label=plot_label_01
)

  plt.plot(
    xs, ys,
    color='red',   # color de la línea
    linewidth=1.0,        # grosor de la línea
    linestyle='--',        # estilo de línea
    marker='o',           # marcador en cada punto
    markersize=0,         # tamaño del marcador
    markerfacecolor='white',
    markeredgecolor='black',
    label=plot_label_02
)

  # --- Título ---
  plt.title(
      titulo,
      fontsize=14,
      fontweight='bold'
  )

  # --- Etiquetas de los ejes ---
  plt.xlabel(
      x_label,
      fontsize=12
  )

  plt.ylabel(
      y_label,
      fontsize=12
  )

  # --- Fuente de los ticks ---
  plt.xticks(fontsize=10)
  plt.yticks(fontsize=10)

  # --- Márgenes ---
  plt.margins(x=0.05, y=0.05)  # espacio extra alrededor de los datos
  plt.gca().spines['right'].set_visible(False) # derecha
  plt.gca().spines['top'].set_visible(False)   # superior

  # Para eliminar márgenes completamente, usar:
  # plt.margins(0)

  # --- Cuadrícula (opcional, pero didáctica) ---
  plt.grid(
      visible=True,
      linestyle='--',
      linewidth=0.7,
      alpha=0.1,
      color="red"
  )

  # --- Leyenda ---
  plt.legend(
      fontsize=10,
      loc='best',
      frameon=False
  )


  # --- Guardar gráfico ---
  plt.savefig(
      save_as,
      bbox_inches='tight'
      )

y_pred = []

for i in range(30):
    lista_entrenamiento = []
    x = list(soluciones.iloc[i, 0:17])
    if factor_02 == 0:
        y_pred.append(mamdani_entrenamiento(x))
    else:
        y_pred.append(tsukamoto_entrenamiento(x))

    try:
        grafico_comparacion(
        df["t"], df["y1"],
        df["t"], lista_entrenamiento,
        "Sistema 01",
        "Real",
        "FIS",
        "Time [min]",
        "$C_A$ [mol/l]",
        f"./{factor_01}{factor_02}{factor_03}{factor_04}/{i}"
        )
    except:
        pass

soluciones["y_pred"] = y_pred
soluciones.to_csv(sufijo, index=False)



