import csv 
import pandas as pd 
import matplotlib.pyplot as plt 
import math     
import sys    
import numpy as np
import scipy.stats as stats
from scipy.stats import studentized_range
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tabulate import tabulate
from scipy.stats import linregress


nivelSignificancia = 0.95
Alfa = 1 - nivelSignificancia


def ValoresAlCuadrado(archivo_csv):
    df = pd.read_csv(archivo_csv)
    
    df_combinado = pd.DataFrame()
    
    for col in df.columns:
        df_combinado[f'{col}'] = df[col]
        df_combinado[f'{col}²'] = (df[col] ** 2).round(2)
    
    return df_combinado

def ContarTotales(archivo_csv):
    df = pd.read_csv(archivo_csv)

    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    totales = df.sum().round(2)

    OxidoNT = totales['Óxido nitroso']
    HumedadT = totales['Humedad(x1)']
    TemperaturaT = totales['Temperatura(x2)']
    PresionT = totales['Presión(x3)']

    return OxidoNT, HumedadT, TemperaturaT, PresionT

def ContarTotalesAlCuadrado(archivo_csv):
    df = pd.read_csv(archivo_csv)

    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    df_cuadrado = df ** 2

    totales_cuadrados = df_cuadrado.sum().round(2)

    OxidoNT_cuadrado = totales_cuadrados['Óxido nitroso'] 
    HumedadT_cuadrado = totales_cuadrados['Humedad(x1)'] 
    TemperaturaT_cuadrado = totales_cuadrados['Temperatura(x2)']
    PresionT_cuadrado = totales_cuadrados['Presión(x3)']
    
    return OxidoNT_cuadrado, HumedadT_cuadrado, TemperaturaT_cuadrado, PresionT_cuadrado

def CalcularDatos(archivo_csv):
    
    df = pd.read_csv(archivo_csv)
    
    totalN_Oxido = df['Óxido nitroso'].count()
    totalN_Humedad = df['Humedad(x1)'].count()
    totalN_Temperatura = df['Temperatura(x2)'].count()
    totalN_Presion = df['Presión(x3)'].count()
    
    return totalN_Oxido, totalN_Humedad, totalN_Temperatura, totalN_Presion

def CalcularMedias(archivo_csv):
    
    df  = pd.read_csv(archivo_csv)
    
    media_Oxido = round(df['Óxido nitroso'].mean(), 4)
    media_Humedad = round(df['Humedad(x1)'].mean(), 4)
    media_Temperatura = round(df['Temperatura(x2)'].mean(), 4)
    media_Presion = round(df['Presión(x3)'].mean(), 4)
    
    return  media_Oxido, media_Humedad, media_Temperatura, media_Presion

archivo_csv = r'C:\Users\User\Documents\Riki\Estadistica III\Datos.csv'
df = pd.read_csv(archivo_csv)
OxidoNT, HumedadT, TemperaturaT, PresionT = ContarTotales(archivo_csv)
n = 30
OxidoNT_cuadrado, HumedadT_cuadrado, TemperaturaT_cuadrado, PresionT_cuadrado = ContarTotalesAlCuadrado(archivo_csv)
media_Oxido, media_Humedad, media_Temperatura, media_Presion = CalcularMedias(archivo_csv)
totalN_Oxido, totalN_Humedad, totalN_Temperatura, totalN_Presion = CalcularDatos(archivo_csv)
Σxt = (OxidoNT + HumedadT + TemperaturaT + PresionT).round(4)
Σxt2 =  (OxidoNT_cuadrado + HumedadT_cuadrado + TemperaturaT_cuadrado + PresionT_cuadrado).round(4)
Σxt_2 = ((OxidoNT**2).round(4)+(HumedadT**2).round(4)+(TemperaturaT**2).round(4)+(PresionT**2).round(4)).round(4)
nt = (totalN_Oxido + totalN_Humedad + totalN_Temperatura + totalN_Presion).round(4)
Σxt_2N = (Σxt_2/n) 
C = (Σxt*Σxt/nt).round(4)
SCT = (Σxt2-C).round(4)
SCTR = (Σxt_2N-C).round(4)
SCE = (SCT - SCTR).round(4)
t = 4
NmenosT = nt-4
gl_t = t-1
totalTabla2 = gl_t+NmenosT
TotalSC = (SCTR +SCE).round(4)
MCTR = (SCTR/gl_t).round(4)
MCE = (SCE/NmenosT).round(4)
F = (MCTR/MCE).round(4)

print(SCTR)
print(SCE)

df_resultado = ValoresAlCuadrado(archivo_csv)

print(df_resultado.to_string(index=False))
print("\nΣxt:   ", "", OxidoNT, "                    ", HumedadT, "                       ", TemperaturaT, "                       ", PresionT) 
print("Σxt²:  ", "                ", OxidoNT_cuadrado, "                 ", HumedadT_cuadrado, "                         ", TemperaturaT_cuadrado, "                 ", PresionT_cuadrado) 
print("(Σxt)²: ", (OxidoNT**2).round(4), "            ", (HumedadT**2).round(4), "                   ", (TemperaturaT**2).round(4), "                  ", (PresionT**2).round(4)) 
print("nt: ", "      ", totalN_Oxido, "                         ",totalN_Humedad, "                           ",totalN_Temperatura, "                           ",totalN_Presion)
print("x̅: ","   ", media_Oxido, "                    ",media_Humedad, "                      ",media_Temperatura, "                      ",media_Presion)
print("\nFactor de Correcion: ", C)
print("Suma Total de Cuadrados: ", SCT)
print("Suma Cuadrados del Tratamiento: ", SCTR)
print("Suma Cuadrados del Error: ", SCE, "\n")

print("-------------------------------------------------------------------------------")
print("|Fuente de Variacion","  |  ","SC","       |  ","gl","|          ","MC","   |     ","F(RV)    |")
print("|----------------------|-------------|------|-----------------|---------------|")
print("|Tratamiento","          |", SCTR ," |   ",gl_t,"|   ", MCTR,"  |  ", F,"    |")
print("|----------------------|-------------|------|-----------------|---------------|")
print("|Error","                |", SCE," | ",NmenosT,"|      ",MCE,"  |               |")
print("|----------------------|-------------|------|-----------------|---------------|")
print("|Total","                |",TotalSC,"| ",totalTabla2,"|                 |               |")
print("|-----------------------------------------------------------------------------|")


Ftab = stats.f.ppf(1 - Alfa, t, NmenosT)
print(f"F tabular (valor crítico): {round(Ftab, 4)}")

if F > Ftab:
    decision = "Rechazar H₀ (Existe diferencia significativa)"
else:
    decision = "Aceptar H₀ (No hay diferencia significativa)"

print(f"Decisión: {decision}")

"""q = stats.t.ppf(1 - Alfa / n, NmenosT)  
DHS = q * math.sqrt(MCE / n)"""

ni = nt / t
q = studentized_range.ppf(1-Alfa, gl_t + 1, NmenosT)
DHS = q * (MCE / ni) ** 0.5

print("El valor de DHS es: ", round(DHS, 4))

medias = {
    "Óxido Nitroso": media_Oxido,
    "Humedad": media_Humedad,
    "Temperatura": media_Temperatura,
    "Presión": media_Presion
}

pares = [
    ("Óxido Nitroso", "Humedad"),
    ("Óxido Nitroso", "Temperatura"),
    ("Óxido Nitroso", "Presión"),
    ("Humedad", "Temperatura"),
    ("Humedad", "Presión"),
    ("Temperatura", "Presión")
]

tabla = []
for g1, g2 in pares:
    meandiff = medias[g1] - medias[g2]
    independencia = "Independiente" if meandiff > DHS or meandiff > -DHS else "Dependiente"
    tabla.append([g1, g2, f"{meandiff:.4f}", f"{DHS:.4f}", independencia])

print("\nPrueba de Tukey\n")
headers = ["Grupo 1", "Grupo 2", "Diferencia", "DHS", "Independencia"]
print(tabulate(tabla, headers=headers, tablefmt="grid"))

def generar_tabla_correlacion(df, var_x, var_y):
        
    df_resultado = pd.DataFrame({
        f"x1 ({var_x})": df[var_x].round(4),
        f"y1 ({var_y})": df[var_y].round(4),
        f"x1²": (df[var_x] ** 2).round(4),
        f"y1²": (df[var_y] ** 2).round(4),
        f"x1.y1": (df[var_x] * df[var_y]).round(4)
    })
    
    suma_columnas = {
        f"x1 ({var_x})": df[var_x].sum().round(4),
        f"y1 ({var_y})": df[var_y].sum().round(4),
        f"x1²": (df[var_x] ** 2).sum().round(4),
        f"y1²": (df[var_y] ** 2).sum().round(4),
        f"x1.y1": (df[var_x] * df[var_y]).sum().round(4)
    }

    df_resultado.loc["Σ"] = suma_columnas

    return df_resultado

def generar_tabla_regresion(df):
        
    df_resultadoR = pd.DataFrame({
        "y (Óxido nitroso)": df["Óxido nitroso"].round(4),
        "x1 (Humedad)": df["Humedad(x1)"].round(4),
        "x2 (Temperatura)": df["Temperatura(x2)"].round(4),
        "x3 (Presión)": df["Presión(x3)"].round(4),
        "x1²": (df["Humedad(x1)"] ** 2).round(4),
        "x1 * x2": (df["Humedad(x1)"] * df["Temperatura(x2)"]).round(4),
        "x1 * x3": (df["Humedad(x1)"] * df["Presión(x3)"]).round(4),
        "x1 * y": (df["Humedad(x1)"] * df["Óxido nitroso"]).round(4),
        "x2²": (df["Temperatura(x2)"] ** 2).round(4),
        "x2 * x3": (df["Temperatura(x2)"] * df["Presión(x3)"]).round(4),
        "x2 * y": (df["Temperatura(x2)"] * df["Óxido nitroso"]).round(4),
        "x3²": (df["Presión(x3)"] ** 2).round(4),
        "x3 * y": (df["Presión(x3)"] * df["Óxido nitroso"]).round(4)
    })
    
    suma_columnasR = {
        "y (Óxido nitroso)": df["Óxido nitroso"].sum(),
        "x1 (Humedad)": df["Humedad(x1)"].sum(),
        "x2 (Temperatura)": df["Temperatura(x2)"].sum(),
        "x3 (Presión)": df["Presión(x3)"].sum(),
        "x1²": (df["Humedad(x1)"] ** 2).sum(),
        "x1 * x2": (df["Humedad(x1)"] * df["Temperatura(x2)"]).sum(),
        "x1 * x3": (df["Humedad(x1)"] * df["Presión(x3)"]).sum(),
        "x1 * y": (df["Humedad(x1)"] * df["Óxido nitroso"]).sum(),
        "x2²": (df["Temperatura(x2)"] ** 2).sum(),
        "x2 * x3": (df["Temperatura(x2)"] * df["Presión(x3)"]).sum(),
        "x2 * y": (df["Temperatura(x2)"] * df["Óxido nitroso"]).sum(),
        "x3²": (df["Presión(x3)"] ** 2).sum(),
        "x3 * y": (df["Presión(x3)"] * df["Óxido nitroso"]).sum()
    }

    df_resultadoR.loc["Σxt"] = suma_columnasR

    return df_resultadoR

df_resultadoR = generar_tabla_regresion(df)

tabla_humedad_presion = generar_tabla_correlacion(df, "Humedad(x1)", "Presión(x3)")
tabla_temperatura_presion = generar_tabla_correlacion(df, "Temperatura(x2)", "Presión(x3)")

print("\n")
print("Tabla Humedad vs Presión:")
print(tabulate(tabla_humedad_presion, headers="keys", tablefmt="grid"))
print("\n")

print("Tabla Temperatura vs Presión:")
print(tabulate(tabla_temperatura_presion, headers="keys", tablefmt="grid"))
print("\n")

correlacion_humedad_presion = df["Humedad(x1)"].corr(df["Presión(x3)"])
correlacion_temperatura_presion = df["Temperatura(x2)"].corr(df["Presión(x3)"])


print(f"Correlación (Humedad vs Presión): {round(correlacion_humedad_presion, 4)}")
print(f"Correlación (Temperatura vs Presión): {round(correlacion_temperatura_presion, 4)}")
print("\n")

slope_hum, intercept_hum, _, _, _ = linregress(df["Humedad(x1)"], df["Presión(x3)"])

slope_temp, intercept_temp, _, _, _ = linregress(df["Temperatura(x2)"], df["Presión(x3)"])


print(f"Ecuación de la recta para Humedad vs Presión: y = {round(slope_hum, 4)} * x + {round(intercept_hum, 4)}")
print(f"Ecuación de la recta para Temperatura vs Presión: y = {round(slope_temp, 4)} * x + {round(intercept_temp, 4)}")

print("\n")
print(tabulate(df_resultadoR, headers="keys", tablefmt="grid"))



y = df["Óxido nitroso"].values
x1 = df["Humedad(x1)"].values
x2 = df["Temperatura(x2)"].values
x3 = df["Presión(x3)"].values

# Tabla de Regresión Múltiple
dfmultiple = pd.DataFrame({
    "Óxido Nitroso(y)": y,
    "Humedad(x1)": x1,
    "Temperatura(x2)": x2,
    "Presión(x3)": x3,
    "y^2": np.square(y),
    "x1^2": np.square(x1),
    "x2^2": np.square(x2),
    "x3^2": np.square(x3),
    "y*x1": np.multiply(y, x1),
    "y*x2": np.multiply(y, x2),
    "y*x3": np.multiply(y, x3),
    "x1*x2": np.multiply(x1, x2),
    "x2*x3": np.multiply(x2, x3),
    "x1*x3": np.multiply(x1, x3)
})

sumatorias = dfmultiple.sum()
dfmultiple.loc["-------------"] = ["-" * 10] * dfmultiple.shape[1]
dfmultiple.loc["Σ"] = sumatorias

print("\nTabla de Contingencia con Datos Calculados:")


# Función para resolver el sistema de ecuaciones usando el método de Gauss-Jordan
def gauss_jordan(A, B):
    AB = np.hstack([A, B.reshape(-1, 1)])  # Matriz ampliada [A|B]
    n = len(B)
    
    for i in range(n):
        # Hacer el pivote 1
        AB[i] = AB[i] / AB[i, i]
        
        for j in range(n):
            if i != j:
                AB[j] = AB[j] - AB[j, i] * AB[i]
    
    return AB  # Retornar la matriz ampliada resuelta

# Función para calcular la regresión
def calcular_regresion(dfmultiple):
    try:
        # Acceder a la fila de sumatorias correctamente
        sumatorias = dfmultiple.loc["Σ"]  
        n = len(y)  # Número de observaciones
        
        # Construir la matriz A y el vector B
        A = np.array([
            [n, sumatorias["Temperatura(x2)"], sumatorias["Presión(x3)"]],
            [sumatorias["Temperatura(x2)"], sumatorias["x2^2"], sumatorias["x2*x3"]],
            [sumatorias["Presión(x3)"], sumatorias["x2*x3"], sumatorias["x3^2"]]
        ])
        
        # Verificar si la matriz A es invertible
        det_A = np.linalg.det(A)
        if np.isclose(det_A, 0):
            raise ValueError("La matriz A no es invertible (det(A) = 0). El sistema no tiene solución única.")
        
        # Vector B
        B = np.array([
            sumatorias["Humedad(x1)"],
            sumatorias["x1*x2"],
            sumatorias["x1*x3"]
        ])
                                     
        
        # Mostrar la matriz ampliada [A|B]
        print("Matriz ampliada [A|B]:")
        print(tabulate(np.hstack([A, B.reshape(-1, 1)]), headers=["B0", "B1", "B2", "B"], tablefmt='grid', floatfmt='.4f'))
        
        
        # Resolver el sistema usando Gauss-Jordan
        matriz_resuelta = gauss_jordan(A, B)
        
        # Mostrar la matriz resultante
        print("\nMatriz resultante [A|B]:")
        print(tabulate(matriz_resuelta, headers=["B0", "B1", "B2", "B"], tablefmt='grid', floatfmt='.4f'))
        
        # Extraer los resultados
        resultados = matriz_resuelta[:, -1]
        
        # Mostrar resultados
        print("\nResultados:")
        print(f"B0 = {resultados[0]:.4f}")
        print(f"B1 = {resultados[1]:.4f}")
        print(f"B2 = {resultados[2]:.4f}")
        
        return resultados
    except Exception as e:
        print(f"Error al calcular la regresión: {e}")
        return None

# Calcular los coeficientes de regresión
coeficientes = calcular_regresion(dfmultiple)



def graficar_regresion(df, var_x, var_y, slope, intercept, title):
    plt.figure(figsize=(8, 6))
    
    plt.scatter(df[var_x], df[var_y], color='blue', label='Datos', alpha=0.7)

    plt.plot(df[var_x], slope * df[var_x] + intercept, color='red', label=f'Recta de ajuste: y = {round(slope, 4)} * x + {round(intercept, 4)}')
    
    plt.title(title)
    plt.xlabel(var_x)
    plt.ylabel(var_y)
    
    plt.legend()

    plt.grid(True)
    plt.show()

graficar_regresion(df, "Humedad(x1)", "Presión(x3)", slope_hum, intercept_hum, "Humedad vs Presión")

graficar_regresion(df, "Temperatura(x2)", "Presión(x3)", slope_temp, intercept_temp, "Temperatura vs Presión")

x = np.linspace(0, Ftab + 5, 1000)
y = stats.f.pdf(x, t, NmenosT)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Distribución F", color="royalblue", lw=2)

plt.fill_between(x, y, where=(x >= Ftab), color='darkred', alpha=0.4, label="Región de rechazo", zorder=1)

plt.axvline(Ftab, color='red', linestyle='dashed', label=f"Ftab = {round(Ftab, 4)}", lw=2, zorder=2)

plt.axvline(F, color='green', linestyle='dashed', label=f"Fcal = {round(F, 4)}", lw=2, zorder=2)

plt.fill_between(x, y, where=(x < Ftab), color='lightgreen', alpha=0.3, label="Región de aceptación", zorder=0)

plt.xlabel("Valores de F", fontsize=12)
plt.ylabel("Densidad de probabilidad", fontsize=12)
plt.title("Distribución F de Fisher con Región de Rechazo y Aceptación", fontsize=14)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
