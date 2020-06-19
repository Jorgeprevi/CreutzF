import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

pasos=100000                         # número de pasos montecarlo
N=100                            # número de cortes temporales 
nr=10                           # número de pasos rechazados entre cada medida    
medidas=int(pasos/nr+1)
pterm=50                         # número de pasos de termalización
x=np.zeros(N, dtype=float)       # vector de trayectoria
xpre=np.zeros(N, dtype=float)
xm2=np.zeros(N, dtype=float) 
xm4=np.zeros(N, dtype=float)
num=np.zeros(N, dtype=float)
den=np.zeros(N, dtype=float)
dtau=1.0
nbar=1
lam=1
m=0.5
mu=-2
a=0.5
h=2.0*np.sqrt(a)
f=1.0
muestra=np.zeros(0, dtype=float)
y=np.linspace(0,N,N)

def pbc(num, L):
    '''
    Función que implementa condiciones de contorno periódicas
    '''
    if num>L:
        return num-L-1 
    elif num<0:
        return L+num+1
    else:
        return num

def accion(x):
    '''
    Función que devuelve la acción a partir del vector x, m y omega
    '''
    S=0.0
    for i in range(N):
        #S+=0.5*m*(x[pbc(i+1,N-1)]-x[i])**2.0/a + 0.5*mu**2.0*x[i]**2.0*a                   # HO
        S+=0.5*m*(x[pbc(i+1,N-1)]-x[i])/a + lam*(x[i]**2.0-f**2.0)**2.0*a                   # HA

    return S

def acepto(x0, x1):
    '''
    Función booleana que decide si se acepta un paso a partir del factor de Boltzmann
    '''
    S0=accion(x0)
    S1=accion(x1)
    if random.uniform(0,1)<(min(1.0,np.exp(S0-S1))):
        return True 
    else: 
        return False

def energ(x_2, x_4):
    '''
    Función que devuelve la energía a partir de las medias
    '''
    #e=mu**2.0*x_2                          #HO
    e=-mu**2.0*x_2+3.0*lam*x_4                 #HA
    return e

si=0
no=0
# se inicia el proceso de termalización
print("\niniciando proceso de termalización")
for i in tqdm(range(pterm)):
    for j in range(N):
        for k in range(nbar):
            x[j]+=random.uniform(-h,h)
            if acepto(xpre, x)==True:
                si+=1
                xpre[j]=x[j]
            else:
                no+=1
                x[j]=xpre[j]
perc=100*float(si)/float(si+no)
print("Porcentaje de aceptación para h=%s:  %s" % (h, perc))

# se inicia el proceso principal de medidas
count=0
print("\niniciando proceso de medida")
for i in tqdm(range(pasos)): 
    for j in range(N):
        for k in range(nbar):
            x[j]+=random.uniform(-h,h)
            if acepto(xpre, x)==True:
                xpre[j]=x[j]
            else:
                x[j]=xpre[j]

    if i%(nr)==0:
        # plt.xlabel("x")
        # plt.ylabel(r"$\tau$")
        # plt.plot(x, y)
        # plt.gca().invert_yaxis()
        # plt.grid()
        # plt.show()
        for j in range(1, N-1):
            num[j]+=x[0]*x[j]
            den[j]+=x[0]*x[j+1]

        xm2+=x**2.0
        xm4+=x**4.0
        muestra=np.append(muestra, x)
        count+=1

# calculo energía
xm2/=float(count)
xm4/=float(count)
xp2=np.sum(xm2)/float(N)
xp4=np.sum(xm4)/float(N)
energia=energ(xp2,xp4)

num/=float(count)
den/=float(count)
energia1=np.zeros(N, dtype=float)
energia1=-(1.0/(a*float(dtau)))*np.log(num/den)+energia

# plt.plot(y, num)
# plt.plot(y, den)
# plt.grid()
# plt.show()
# plt.plot(y, energia1)
# plt.grid()
# plt.show()
# print("Energía1: %s" % (energia1))
# print(num)
# print(den)
# print(num/den)
# calculo error
error0=0.0
error1=0.0
emed=0.0
for i in range(N):
    error0+=(energ(xm2[i], xm4[i])-energia)**2.0
error0/=float(N)
error0=np.sqrt(error0)
for i in range(10):
    emed+=energia1
emed/=10
# print(emed)
# ploteo resultados
pesos=np.ones_like(muestra)/float(len(muestra))
print("\nDatos de entrada: T=%s, a=%s, tamaño=%s, pasos=%s, pasos term=%s, pasos rechaz=%s" %(a*N, a, N, pasos, pterm, nbar))
print("Pasos MC: %s" % (count))
print("Energía: %s ± %s" % (energia, error0))
print("<x^2>: %s <x^4>: %s" % (xp2, xp4))
print("\n")
xlm=3
plt.xlim(-xlm,xlm)
plt.hist(muestra, weights=(pesos*100.0/(2*xlm)), bins=100)
plt.xlabel("x")
plt.ylabel(r"$|\Psi_{0}|^{2}$")
plt.show()
plt.plot()