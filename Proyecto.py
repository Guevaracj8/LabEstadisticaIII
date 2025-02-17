import csv 
import pandas as pd 
import matplotlib.pyplot as plt 
import math     
import sys    
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


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

archivo_csv = r'C:\Users\User\Documents\Ricardo\Proyecto Estadistica III\Datos.csv'
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
t = 4-1
NmenosT = nt-4
totalTabla2 = t+NmenosT
TotalSC = (SCTR +SCE).round(4)
MCTR = (SCTR/t).round(4)
MCE = (SCE/NmenosT).round(4)
F = (MCTR/MCE).round(4)

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
print("|Tratamiento","          |", SCTR ," |   ",t,"|   ", MCTR,"  |  ", F,"    |")
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

t_value = stats.t.ppf(1 - Alfa / 2, 1)  

HSD = t_value * math.sqrt(MCE / n)
print(f"El valor aproximado de q de Tukey es: {HSD:.2f}")

print("\n")
print("---------------------------------------------------------")
print("|","           |  ","Y1   | "," X1","    |    X2","   |     X3    |")
print("|------------|--------|----------|----------|-----------|")
print("|Y1:",media_Oxido," |","//////","|",(media_Oxido-media_Humedad).round(4),"|", (media_Oxido-media_Temperatura).round(4),"|", (media_Oxido-media_Presion).round(4)," |")
print("|------------|--------|----------|----------|-----------|")
print("|X1:",media_Humedad,"|","//////","|","//////","  |",(media_Humedad-media_Temperatura).round(4)," |", (media_Humedad-media_Presion).round(4),"   |")
print("|------------|--------|----------|----------|-----------|")
print("|X2:",media_Temperatura,"|","//////","|","//////","  |","//////","  |",(media_Temperatura-media_Presion).round(4),"   |")
print("|------------|--------|----------|----------|-----------|")
print("|X3:",media_Presion,"|","//////","|","//////","  |","//////","  |","//////","   |")
print("|-------------------------------------------------------|")



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



