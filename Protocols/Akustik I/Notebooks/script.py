from praktikum import analyse
from praktikum import cassy
import matplotlib.pyplot as plt


inputfile = 'StabA.labx'
data = cassy.CassyDaten(inputfile)
#---
t = data.messung(1).datenreihe('t').werte
A = data.messung(1).datenreihe('U_A1').werte
#---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  
#---
ax1.plot(t[0:150], A[0:150], color='Green', label='Messwerte')
ax1.set_xlabel('t / s')
ax1.set_ylabel('A / V')
ax1.set_title('Stab A: Zeitverlauf bis 150 Messwerte')
ax1.legend(loc='upper right')
ax1.grid(color='gray', linestyle='--')
#---
F, A_m = analyse.fourier_fft(t, A)
ax2.plot(F, A_m, color='blue', label='Spektrum')
ax2.set_xlabel('Frequenz / Hz')
ax2.set_ylabel('Amplitude / V')
ax2.set_title('Fourier-Analyse')
ax2.legend(loc='upper right')
ax2.grid(color='gray', linestyle='--')
#---
plt.show()
fig.savefig('StabA.pdf')



inputfile = 'StabB.labx'
data = cassy.CassyDaten(inputfile)
#---
t = data.messung(1).datenreihe('t').werte
A = data.messung(1).datenreihe('U_A1').werte
#---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  
#---
ax1.plot(t[0:150], A[0:150], color='Green', label='Messwerte')
ax1.set_xlabel('t / s')
ax1.set_ylabel('A / V')
ax1.set_title('Stab B: Zeitverlauf bis 150 Messwerte')
ax1.legend(loc='upper right')
ax1.grid(color='gray', linestyle='--')
#---
F, A_m = analyse.fourier_fft(t, A)
ax2.plot(F, A_m, color='blue', label='Spektrum')
ax2.set_xlabel('Frequenz / Hz')
ax2.set_ylabel('Amplitude / V')
ax2.set_title('Fourier-Analyse')
ax2.legend(loc='upper right')
ax2.grid(color='gray', linestyle='--')
#---
plt.show()
fig.savefig('StabB.pdf')



inputfile = 'StabC.labx'
data = cassy.CassyDaten(inputfile)
#---
t = data.messung(1).datenreihe('t').werte
A = data.messung(1).datenreihe('U_A1').werte
#---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  
#---
ax1.plot(t[0:150], A[0:150], color='Green', label='Messwerte')
ax1.set_xlabel('t / s')
ax1.set_ylabel('A / V')
ax1.set_title('Stab C: Zeitverlauf bis 150 Messwerte')
ax1.legend(loc='upper right')
ax1.grid(color='gray', linestyle='--')
#---
F, A_m = analyse.fourier_fft(t, A)
ax2.plot(F, A_m, color='blue', label='Spektrum')
ax2.set_xlabel('Frequenz / Hz')
ax2.set_ylabel('Amplitude / V')
ax2.set_title('Fourier-Analyse')
ax2.legend(loc='upper right')
ax2.grid(color='gray', linestyle='--')
#---
plt.show()
fig.savefig('StabC.pdf')



inputfile = 'StabD.labx'
data = cassy.CassyDaten(inputfile)
#---
t = data.messung(1).datenreihe('t').werte
A = data.messung(1).datenreihe('U_A1').werte
#---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  
#---
ax1.plot(t[200:400], A[200:400], color='Green', label='Messwerte')
ax1.set_xlabel('t / s')
ax1.set_ylabel('A / V')
ax1.set_title('Stab D: Zeitverlauf von 200. bis 400. Messwerte')
ax1.legend(loc='upper right')
ax1.grid(color='gray', linestyle='--')
#---
F, A_m = analyse.fourier_fft(t[200:1000], A[200:1000])
ax2.plot(F, A_m, color='blue', label='Spektrum')
ax2.set_xlabel('Frequenz / Hz')
ax2.set_ylabel('Amplitude / V')
ax2.set_title('Fourier-Analyse von 200. bis 1000. Messwerte')
ax2.legend(loc='upper right')
ax2.grid(color='gray', linestyle='--')
#---
plt.show()
fig.savefig('StabD.pdf')