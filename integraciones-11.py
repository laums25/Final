import numpy  as np
import random as rnd
import matplotlib.pylab as plt
import math

#Función exponencial de decaimiento
def f(x):
    return np.exp(-x) 

#Calculo del error a partir del valor teórico para la función exponencial evaluada en los límites 1 y 0
valorexacto=1-np.exp(-1)
Valor=[]
Valor.append(valorexacto)
Valores=Valor*10
Valores=np.asarray(Valores)
error=abs(listatotal-Valores)/Valores
print (error)
errorn=abs(listasi-Valores)/Valores
print (errorn)


#Función para integrar por Simpson
#Apoyada en el libro de Landau, Paez, Bordeianu
#Apoyada en: https://stackoverflow.com/questions/5326112/how-to-round-each-item-in-a-list-of-floats-to-2-decimal-places
n=0
def Simpson(i,h):
    if(i==1 or i==n):
        w=h/3.0
    elif (i%2==0):
        w=(4*h)/3.0
    elif (i%2!=0):
        w=(2*h)/3.0
    return w

def Sim(n):
    A=0
    B=1
    h=(B-A)/(n-1)
    suma2=0

    for i in range(1, n+1):
        t=A+(i-1)*h
        p=Simpson(i,h)
        suma2=suma2+p* f(t)
    return suma2

n2=[3,41,81,161,641,901, 1281,3001,3501,4501]
lista2=[]
for i in range(len(n2)):
    r2=Sim(n2[i])
    lista2.append(r2)
    listanueva2= [ '%.15f' % elem for elem in lista2 ]
    listatotal2=[float(j) for j in listanueva2]
    listasiete2=[ '%.7f' % elem for elem in lista2 ]
    listasi2=[float(p) for p in listasiete2]
    
print (listatotal2)
print(listasi2)

#Calculo del error a partir del valor teórico para la función exponencial evaluada en los límites 1 y 0
valorexacto=1-np.exp(-1)
Valor=[]
Valor.append(valorexacto)
Valores=Valor*10
Valores=np.asarray(Valores)
error=abs(listatotal-Valores)/Valores
print (error)
errorn=abs(listasi-Valores)/Valores
print (errorn)



#Función para integrar por Simpson
#Apoyada en el libro de Landau, Paez, Bordeianu
#Apoyada en: https://stackoverflow.com/questions/5326112/how-to-round-each-item-in-a-list-of-floats-to-2-decimal-places
n=0
def Simpson(i,h):
    if(i==1 or i==n):
        w=h/3.0
    elif (i%2==0):
        w=(4*h)/3.0
    elif (i%2!=0):
        w=(2*h)/3.0
    return w

def Sim(n):
    A=0
    B=1
    h=(B-A)/(n-1)
    suma2=0

    for i in range(1, n+1):
        t=A+(i-1)*h
        p=Simpson(i,h)
        suma2=suma2+p* f(t)
    return suma2

n2=[3,41,81,161,641,901, 1281,3001,3501,4501]
lista2=[]
for i in range(len(n2)):
    r2=Sim(n2[i])
    lista2.append(r2)
    listanueva2= [ '%.15f' % elem for elem in lista2 ]
    listatotal2=[float(j) for j in listanueva2]
    listasiete2=[ '%.7f' % elem for elem in lista2 ]
    listasi2=[float(p) for p in listasiete2]
    
print (listatotal2)
print(listasi2)


#Errores para 15 y 7 decimales 
error2=abs(listatotal2-Valores)/Valores
print (error2)
errorn2=abs(listasi2-Valores)/Valores
print (errorn2)




#Apoyada en: https://stackoverflow.com/questions/33457880/different-intervals-for-gauss-legendre-quadrature-in-numpy/33462230#33462230
#Apoyada en: https://austingwalters.com/gaussian-quadrature/
#Apoyada en: https://stackoverflow.com/questions/5326112/how-to-round-each-item-in-a-list-of-floats-to-2-decimal-places
def Gaussian(n):
    a = 0.0
    b = 1
    x, w = np.polynomial.legendre.leggauss(n)
    t = 0.5*x*(b - a) + (0.5*(a+b))
    gauss = sum(w * f(t))* 0.5*(b - a)
    return gauss

lista3=[]
for i in range(len(n1)):
    r3=Gaussian(n1[i])
    lista3.append(r3)
    listanueva3= [ '%.15f' % elem for elem in lista3 ]
    listatotal3= [float(o) for o in listanueva3]
    listasiete3=['%.7f' % elem for elem in lista3]
    listasi3=[float(t) for t in listasiete3]
print (listatotal3)
print(listasi3)


#Calculo de errores con 15 y 7 decimales
error3=abs(listatotal3-Valores)/Valores
print (error3)
errorn3=abs(listasi3-Valores)/Valores
print (errorn3)

#Gráfica de errores para cada integral versus n
plt.figure(1, figsize=(12,4.5))
plt.subplot(1,2,1)
plt.plot(n1,error, label='Trapecio')
plt.plot(n2,error2, label='Simpson')
plt.plot(n1,error3, label='Gaussian')
plt.loglog()
plt.xlabel('N')
plt.ylabel('Error')
plt.legend()
plt.title('Errores de integración vs N para 15 decimales')

plt.subplot(1,2,2)
plt.plot(n1,errorn, label='Trapecio')
plt.plot(n2,errorn2, label='Simpson')
plt.plot(n1,errorn3, label='Gaussian')
plt.loglog()
plt.xlabel('N')
plt.ylabel('Error')
plt.legend()
plt.title('Errores de integración vs N para 7 decimales')
plt.savefig('logError.png')
plt.show()