#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
In diesem Beispiel werden Messdaten ausgewertet, die bei der
Aufladung eines Kondensators C über einen Ohmschen Widerstand R
gewonnen wurden. Gemessen wurden die an Kondensator und Widerstand
abfallenden Spannungen U_C und U_R als Funktion der Zeit während der
Aufladung. Der Spannungsverlauf am Widerstand wird während des
Ladevorgangs durch eine Exponentialfunktion, U_R=U_0*exp(-t/tau),
beschrieben. Durch Logarithmierung der Spannung erhält man eine
Gerade, aus deren Steigung man die Zeitkonstante tau=RC des
Ladevorgangs bestimmen kann.

"""

from praktikum import analyse
from praktikum import cassy
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat

# Gut lesbare und ausreichend große Beschriftung der Achsen, nicht zu dünne Linien:
plt.rcParams['font.size'] = 24.0
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.0

# Mit Cassy aufgezeichnete Daten:
inputfile = 'labx/kondensator.labx'
#inputfile = 'txt/kondensator.txt'
data = cassy.CassyDaten(inputfile)
data.info()

# Bereich für logarithmischen Fit (in ms)
tmin = 0.101
tmax = 4.001

# Beginn der "Rauschmessung" für die Offset-Korrektur (in ms)
toffset = 8.0

plt.figure(figsize=(22,12), layout='tight')

N = data.anzahl_messungen()
for m in range(1, N+1):

    t = data.messung(m).datenreihe('t').werte
    data.messung(m).datenreihe('t').info()

    # Im labx-Format scheinen die Zeitwerte in s gespeichert zu sein, obwohl als Einheit "ms"
    # angegeben ist. Die Praktikumsbibliothek versucht diesen Fehler seit der Version 2.3.3
    # automatisch zu korrigieren, gleichwohl sollte man die Zeitachse immer sorgfältig
    # überprüfen.
    # if os.path.splitext(inputfile)[1] == '.labx':
    #     t *= 1000.0 # s -> ms, sollte ab Version 2.3.3 nicht mehr notwendig sein.

    UR = data.messung(m).datenreihe('U_A1').werte
    UC = data.messung(m).datenreihe('U_B1').werte

    plt.subplot(3,N,m)
    # Grafische Darstellung der Rohdaten
    plt.plot(t, UR, color='red', label='$U_R$')
    plt.plot(t, UC, color='green', label='$U_C$')
    plt.xlabel('$t$ / ms')
    plt.ylabel('$U$ / V')
    plt.xlim(left=0.)
    plt.ylim(0., 10.5)
    plt.legend(loc='right')
    plt.minorticks_on()

    plt.subplot(3,N,m+N)
    UR0 = UR[0]

    # Extrahiere Daten für Fit
    t_fit, UR_fit = analyse.untermenge_daten(t, UR, tmin, tmax)

    # Offsetkorrektur
    _, Uend = analyse.untermenge_daten(t, UR, toffset, t[-1])
    Uoffset = Uend.mean()
    print(f'Uoffset = {Uoffset:.2g} V')

    # Wir logarithmieren die über dem Ohmschen Widerstand im Kreis abfallende Spannung, die
    # dem Ladestrom proportional ist.
    logUR_fit = np.log((UR_fit - Uoffset) / UR0)

    # Messbereich -> Digitalisierungsfehler (Cassy-ADC hat 12 bits => 4096 mögliche Werte)
    sigmaU = 20.0 / 4096. / np.sqrt(12.)
    sigmaLogUR_fit = sigmaU / UR_fit

    plt.errorbar(t_fit, logUR_fit, yerr=sigmaLogUR_fit, fmt='.')
    plt.xlabel('$t$ / ms')
    plt.ylabel(r'$\log\,U_R/U_0$')

    # Lineare Regression zur Bestimmung der Zeitkonstanten
    a,ea,b,eb,chiq,corr = analyse.lineare_regression(t_fit, logUR_fit, sigmaLogUR_fit)
    tau = -1.0 / a
    sigma_tau = abs(tau * ea/a)
    utau = ufloat(tau, sigma_tau)
    ub = ufloat(b, eb)
    print(f'tau = ({utau}) ms   b = ({ub})  chi2/dof = {chiq:.2f} / {len(t_fit)-2}')
    plt.plot(t_fit, a*t_fit + b, color='red')
    plt.minorticks_on()

    plt.subplot(3,N,m+2*N)
    # Residuenplot
    resUR = logUR_fit - (a*t_fit + b)
    eresUR = sigmaLogUR_fit
    plt.errorbar(t_fit, resUR, yerr=eresUR, fmt='.')
    plt.xlabel('$t$ / ms')
    plt.ylabel(r'$\log\,U_R/U_0 - (t/\tau + b)$')
    plt.minorticks_on()
    plt.grid()

plt.show()
