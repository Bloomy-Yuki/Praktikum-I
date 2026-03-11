#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
In diesem Beispiel lesen wir mit der Praktikumsbibliothek Messdaten
ein, die mit dem Cassy-System bei der Messung der Dampfdruckkurve von
Wasser aufgezeichnet wurden. Während des Abkühlvorgangs wurden dabei
der Druck der Gasphase und die Temperatur des Wassers gemessen.

Die Clausius-Clapeyron Gleichung liefert den Zusammenhang

log(p/p0) = -Lambda/R * (1/T - 1/T0)

mit der allgemeinen Gaskonstante R und der molaren
Verdampfungsenthalpie Lambda.

Nach entsprechender Transformation der Messdaten erhalten wir einen
linearen Zusammenhang. Durch eine lineare Regression lässt sich die
Steigung bestimmen und daraus der Wert von Lambda.

Achtung: Hier wird ein (zu) großer Bereich für die lineare Regression
verwendet und es fehlt der Residuenplot am Ende. Für die tatsächliche
Durchführung im Praktikum muss diese grobe Analyse noch verfeinert
werden.

"""

from praktikum import cassy
from praktikum import analyse
import numpy as np
import scipy.constants
import matplotlib.pyplot as plt
from uncertainties import ufloat

# Gut lesbare und ausreichend große Beschriftung der Achsen, nicht zu
# dünne Linien. Dabei muss man beachten, dass die hier angegebene
# Schriftgröße ein Absolutwert ist. Deshalb muss die Schriftgröße
# ('font.size') entsprechend angepasst werden, wenn man das Format der
# Abbildung ('figure.figsize') ändert.
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 24.0
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.0

# Rauschmessung
data1 = cassy.CassyDaten('lab/Thermo_Rauschmessung.lab')
p = data1.messung(1).datenreihe('p_A1').werte
T = data1.messung(1).datenreihe('T_B11').werte

plt.figure()

# Histogramm der Druckwerte aus der Rauschmessung
plt.subplot(2,1,1)
plt.title('Histogramm der Druckwerte')
plt.hist(p, bins=100, range=(1000.,1020.), color='green')
plt.xlabel('p / mbar')
plt.xlim(1000., 1020.)

# Berechnung des Rauschens des Drucksensors (Standardabweichung)
p_mean = np.mean(p)
p_stdabw = np.std(p, ddof=1)
print(f'p_mean = {p_mean:.2f} mbar, p_stdabw = {p_stdabw:.2f} mbar')

# Histogramm der Temperaturwerte aus der Rauschmessung
plt.subplot(2,1,2)
plt.title('Histogramm der Temperaturwerte')
T1 = np.min(T) - 0.2
T2 = np.max(T) + 0.2
plt.hist(T, bins=100, range=(T1,T2), color='blue')
plt.xlabel('T / K')
plt.xlim(T1, T2)

plt.tight_layout()


# Berechnung des Rauschens des Temperatursensors
T_mean = np.mean(T)
T_stdabw = np.std(T, ddof=1)
print(f'T_mean = {T_mean:.2f} K, T_stdabw = {T_stdabw:.2f} K')

# lese Daten der Hauptmessung
data2 = cassy.CassyDaten('lab/Thermo_Hauptmessung.lab')
p = data2.messung(1).datenreihe('p_A1').werte
T = data2.messung(1).datenreihe('T_B11').werte

# Ziel: Plot log(p) vs 1/T (-> Clausius-Clapeyron Gleichung)
p0 = 1013.  # mbar
logP = np.log(p / p0)
Tinv = 1.0 / T

# Fehlerfortpflanzung
sigma_logP = p_stdabw / p
sigma_Tinv = T_stdabw / T**2

# Untermenge für lineare Regression
T2, p2 = analyse.untermenge_daten(T, p, 1./0.00284, 1./0.00272)
logP2 = np.log(p2/p0)
Tinv2 = 1.0 / T2
sigma_logP2 = p_stdabw / p2
sigma_Tinv2 = T_stdabw / T2**2

# Lineare Regression an log(p) gegen 1/T liefert Steigung a = -Lambda/R
a, ea, b, eb, chiq, corr = analyse.lineare_regression_xy(Tinv2, logP2, sigma_Tinv2, sigma_logP2)
dof = len(Tinv2) - 2
ua = ufloat(a, ea)
ub = ufloat(b, eb)
Lambda = -ua*scipy.constants.R * 1.e-3  # J/mol -> kJ/mol
print(f'Lin.Reg.: a={ua}, b={ub}, chi2/dof={chiq:.2f}/{dof}, corr={corr:.5f}')
print(f'Lambda = ({Lambda}) kJ/mol')

# Plot von Messdaten und angepasster Kurve
plt.figure()
# Wir zeichnen ausnahmsweise wegen der hohen Punktdichte keine Marker
# (fmt='o'):
plt.errorbar(Tinv, logP, xerr=sigma_Tinv, yerr=sigma_logP, fmt='.')
plt.plot(Tinv2, a*Tinv2 + b, '-', color='red', linewidth=3)
plt.xlabel(r'$T^{-1} / \mathrm{K}^{-1}$')
plt.ylabel(r'$\log(p/p_0)$')
plt.grid()
plt.tight_layout()

plt.show()
