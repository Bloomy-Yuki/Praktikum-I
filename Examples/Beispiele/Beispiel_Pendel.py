#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
In diesem Beispiel werden mit Cassy aufgezeichnete Messdaten einer
Pendelschwingung eingelesen, nämlich die vom Winkelsensor ausgegebene
Spannung als Funktion der Zeit. Nach der Bestimmung der Nulllage wird
eine (grobe) exponentielle Einhüllende für die Daten gefunden und die
Frequenz der Schwingung wird mittels einer schnellen
Fourier-Transformation (FFT) ermittelt. Die Daten und Ergebnisse
werden grafisch dargestellt.

Achtung: Wie man dem Plot entnehmen kann, liefert die Routine für die
exponentielle Einhüllende nur einen sehr groben Anhaltspunkt. Außerdem
ist die Methode der Frequenzbestimmung mittels FFT nicht übermäßig
exakt, und ohne Berücksichtigung der genauen Form des
Fourier-Spektrums liefert der verwendete Peakfinder nicht den genauen
Wert des Maximums. (Man zeichne eine gedämpfte Schwingung mit den
ermittelten Parametern in die Messdaten ein.)

Im Praktikum wird daher eine andere Methode für die Auswertung
verwendet, aber nichtsdestotrotz kann man an diesem Beispiel
studieren, wie man mit der Praktikumsbibliothek Cassy-Messdaten
einlesen und dann eine schnelle Auswertung an diesen vornehmen
kann.

"""

from praktikum import cassy
from praktikum import analyse
import numpy as np
import matplotlib.pyplot as plt

# Gut lesbare und ausreichend große Beschriftung der Achsen, nicht zu
# dünne Linien. Dabei muss man beachten, dass die hier angegebene
# Schriftgröße ein Absolutwert ist. Deshalb muss die Schriftgröße
# ('font.size') entsprechend angepasst werden, wenn man das Format der
# Abbildung ('figure.figsize') ändert.
plt.rcParams['figure.figsize'] = (22, 9)
plt.rcParams['font.size'] = 24.0
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.0

data = cassy.CassyDaten('lab/Pendel.lab')
timeValues = data.messung(1).datenreihe('t').werte
voltage = data.messung(1).datenreihe('U_A1').werte
voltageError = 0. * voltage + 0.01
offset = analyse.gewichtetes_mittel(voltage, voltageError)[0]
voltage = voltage - offset

plt.figure(1, layout='tight')
plt.title('Pendel')

plt.subplot(2,1,1)
plt.plot(timeValues, voltage, '.')
plt.grid()
plt.xlabel('Zeit / s')
plt.ylabel('Spannung / V')
einhuellende = analyse.exp_einhuellende(timeValues, voltage, voltageError)
plt.plot(timeValues, +einhuellende[0] * np.exp(-einhuellende[2] * timeValues))
plt.plot(timeValues, -einhuellende[0] * np.exp(-einhuellende[2] * timeValues))

plt.subplot(2,1,2)
fourier = analyse.fourier_fft(timeValues, voltage)
frequency = fourier[0]
amplitude = fourier[1]
plt.plot(frequency, amplitude, 'o-')
plt.grid()
plt.xlabel('Frequenz / Hz')
plt.ylabel('Amplitude')

maximumIndex = amplitude.argmax()
plt.xlim(frequency[max(0, maximumIndex-10)], frequency[min(maximumIndex+10, len(frequency))])
peak = analyse.peakfinder_schwerpunkt(frequency, amplitude)
plt.axvline(peak)

L = 0.667
g = ((2 * np.pi * peak)**2) * L

print(f'g = {g:.2f} m/s^2')

plt.show()
