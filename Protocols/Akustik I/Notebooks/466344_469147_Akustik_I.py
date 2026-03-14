from praktikum import analyse
from praktikum import cassy
import matplotlib.pyplot as plt
import numpy as np


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
#---finding the peaks

def peak(in_text, a, b, c = 200, d = 1000):
    inputfile = 'Stab' + in_text +'.labx'
    data = cassy.CassyDaten(inputfile)
    #---
    t = data.messung(1).datenreihe('t').werte
    A = data.messung(1).datenreihe('U_A1').werte
    #---
    F, A_m = analyse.fourier_fft(t[c:d], A[c:d])
    #---
    filtered = (b > F) & (F > a)
    F_filtered = F[filtered]
    A_m_filtered = A_m[filtered]
    print(analyse.peakfinder_schwerpunkt(F_filtered, A_m_filtered))
    return analyse.peakfinder_schwerpunkt(F_filtered, A_m_filtered)

print(peak('A', 1900, 2100))
print(peak('A2', 1900, 2100))
print("")
print(peak('B',1000, 1500))
print(peak('B2',1000, 1500))
print("")
print(peak('C',1500, 2000))
print(peak('C2',1500, 2000))
print("")
print(peak('D',1400, 1600))
print(peak('D2',1400, 1600))
#----
#---measurements and their error
A = [407.5, 129, 13.435, 1966.23]
a = [0.11 , 0.04, 0.0048, 0.56]

B = [1233.5, 129, 13.455, 1323.14]
b = [0.11 , 0.04, 0.0088, 0.56]

C = [1325.8, 149.75, 13.83, 1726.26]
c = [0.11 , 0.29, 0.1, 0.54]

D = [1302.3, 129, 13.46, 1512.5]
d = [0.11 , 0.04, 0.002, 12.5]
#---
#---functions to calculate E and rho and standard error with message function
def E(A):
	return (16/np.pi) * ((A[3])**2) * (A[1]/100) * (A[0]/1000) / ((A[2]/1000)**2)/1E9

def rho(A):
	return (A[0]/1000) /(A[1]/100 * (A[2]/1000)**2 * np.pi/4)

def fort(A, m, l, d, f):
	return np.sqrt( (m/A[0])**2 + (l/A[1])**2 + (2 * d/A[2])**2 + (2 * f/A[3])**2)

def err(A,a):
	print("Elasticity: ", round(E(A),2) , " +/- ",round(E(A)*fort(A, a[0],a[1],a[2],a[3]),2), " GPa") 
	print("Density: ", round(rho(A),2) , " +/- ",round(rho(A)*fort(A, a[0],a[1],a[2],0),2), " kg/m^3")
	print("")
	return None
#---
#---
err(A,a)#Stab A
err(B,b)#Stab B
err(C,c)#Stab C
err(D,c)#Stab D
