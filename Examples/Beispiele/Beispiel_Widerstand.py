#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wir lesen mit der Praktikumsbibliothek Messdaten ein, die mit dem
Cassy-System bei einer einfachen Messung von Spannung U gegen Strom I
an einem Ohmschen Widerstand aufgezeichnet wurden. Durch eine lineare
Regression bestimmen wir die Steigung, also den Widerstand R.

(Dieses Beispiel ist auch auf der Homepage der Praktikumsbibliothek
gezeigt.)

"""

from praktikum import analyse
from praktikum import cassy
import numpy as np
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

# Cassy-Datei, die wir analysieren wollen:
inputfile = 'labx/widerstand.labx'
#inputfile = 'txt/widerstand.txt'
data = cassy.CassyDaten(inputfile)

# Es gibt nur eine einzige Messung in der Datei.
U = data.messung(1).datenreihe('U_B1').werte
I = data.messung(1).datenreihe('I_A1').werte

# Messbereich -> Digitalisierungsfehler (Cassy-ADC hat 12 bits => 4096 mögliche Werte)
sigmaU = 20.0 / 4096. / np.sqrt(12.) * np.ones_like(U)
sigmaI = 0.2 / 4096. / np.sqrt(12.) * np.ones_like(I)

# Erstelle eine schön große Abbildung mit zwei Achsenpaaren (oben für
# die Messdaten samt angepasster Gerade, unten für den
# Residuenplot. Die x-Achse teilen sich beide Plots. Als
# Höhenverhältnis verwenden wir 5:2.
fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2]})

# Grafische Darstellung der Rohdaten
ax[0].errorbar(I, U, xerr=sigmaI, yerr=sigmaU, color='red', fmt='o')
ax[0].set_xlabel('$I$ / A')
ax[0].set_ylabel('$U$ / V')

# Lineare Regression
R, eR, b, eb, chiq, corr = analyse.lineare_regression_xy(I, U, sigmaI, sigmaU)
uR = ufloat(R, eR)
ub = ufloat(b, eb)
dof = len(I) - 2
print(f'R = ({uR}) Ohm, b = ({ub}) V, chi2/dof = {chiq:.1f} / {dof},  corr = {corr:g}')
ax[0].plot(I, R*I+b, color='green')

# Für den Residuenplot werden die Beiträge von Ordinate und Abszisse
# (gewichtet mit der Steigung) quadratisch addiert.
sigmaRes = np.sqrt((R*sigmaI)**2 + sigmaU**2)

# Zunächst plotten wir eine gestrichelte Nulllinie, dann den eigentlichen Residuenplot:
ax[1].axhline(y=0., color='black', linestyle='--')
ax[1].errorbar(I, U - (R*I + b), yerr=sigmaRes, color='red', fmt='o')
ax[1].set_xlabel('$I$ / A')
ax[1].set_ylabel('$(U-(RI+b))$ / V')

# Wir sorgen dafür, dass die y-Achse beim Residuenplot symmetrisch um die Nulllinie ist:
ymax = max([abs(x) for x in ax[1].get_ylim()])
ax[1].set_ylim(-ymax, ymax)

# Finales Layout:
plt.tight_layout()
fig.subplots_adjust(hspace=0.0)

plt.show()
