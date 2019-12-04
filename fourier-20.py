import numpy  as np
import random as rnd
import matplotlib.pylab as plt

#modificado de: https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
def fourier(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    parte = np.exp(-2j * np.pi *k*n/ N)
    resultado= np.dot(parte, x)
    return resultado



#Modificado de: https://pybonacci.org/2012/09/29/transformada-de-fourier-discreta-en-python-con-scipy/
#https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.stem.html
#señal
n=60
T=2*np.pi
h=T/n
t=np.linspace(0, (n-1)*h, n)
y=1/(1-(0.9*np.sin(t)))
plt.figure(1, figsize=(14,4) )
plt.subplot(1,3,1)
valor=fourier(y)
long=len(valor)
l=np.linspace(0,long-1,long)
normalizar=np.abs(valor/long)

A=normalizar**2

plt.semilogy(l, A)
plt.xlabel('k')
plt.ylabel('s(w)^2')


#Autocorrelation
#Fuente: https://www.iteramos.com/pregunta/45406/como-puedo-utilizar-numpyse-correlacionan-para-hacer-de-autocorrelacion
def autocorr(x):
    prom=x-np.mean(x)
    cuadra=prom**2
    norma = np.sum(cuadra)
    result = np.correlate(prom, prom, 'same')/norma

    return result

autocorre=autocorr(y)

plt.subplot(1,3,2)
plt.plot(t,autocorre)
plt.xlabel('tau(s)')
plt.ylabel('A(tau)')


plt.subplot(1,3,3)
valor2=fourier(autocorre)
long2=len(valor2)
l2=np.linspace(0,long2-1,long2)
normalizar2=np.abs(valor2/long2)
plt.scatter(l2, normalizar2)
plt.stem(l2, normalizar2, use_line_collection=True)
plt.xlabel('k')
plt.ylabel('|s(w)^2/N|')

plt.savefig('potencias.png')


#Ruido
alpha=4
r=np.ones(n)
for i in range(n):
    r[i]*=np.random.random()
yrui=y+(alpha*((2*r)-1))

plt.figure(1, figsize=(14,4) )
plt.subplot(1,3,1)
plt.plot(t,yrui)
plt.xlabel('t')
plt.ylabel('y ruido')

plt.subplot(1,3,2)
valor3=fourier(yrui)
long3=len(valor3)
l3=np.linspace(0,long3-1,long3)
normalizar3=np.abs(valor3/long3)
plt.scatter(l3, normalizar3)
plt.stem(l3, normalizar3, use_line_collection=True)
plt.xlabel('k')
plt.ylabel('|s(w)^2/N| ruido')

A3=normalizar3**2
plt.subplot(1,3,3)
plt.semilogy(l3, A3)
plt.stem(l3,A3, use_line_collection=True )

plt.savefig('ruido.png')



#Autocorrelación ruido
autocorre2=autocorr(yrui)

plt.figure(1, figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(l3,autocorre2)

plt.subplot(1,2,2)
valor4=fourier(autocorre2)
long4=len(valor4)
l4=np.linspace(0,long4-1,long4)
normalizar4=np.abs(valor4/long4)
plt.scatter(l4, normalizar4)
plt.stem(l4, normalizar4, use_line_collection=True)

plt.savefig('correlacion.png')


plt.figure(1, figsize=(10,4))
plt.subplot(1,2,1)
plt.semilogy(l3, A3)
plt.stem(l3,A3, use_line_collection=True )

plt.subplot(1,2,2)
plt.scatter(l4, normalizar4)
plt.stem(l4, normalizar4, use_line_collection=True)

plt.savefig('comparación.png')



#Ruido
#Ruido
alpha2=7.2
yrui2=y+(alpha2*((2*r)-1))

valor5=fourier(yrui2)
long5=len(valor5)
l5=np.linspace(0,long5-1,long5)
normalizar5=np.abs(valor5/long5)

plt.figure(1, figsize=(10,4))
plt.subplot(1,2,1)
A5=normalizar5**2
plt.semilogy(l5, A5)
plt.stem(l5,A5, use_line_collection=True )
plt.title('alpha=7.2')


autocorre5=autocorr(yrui2)

plt.subplot(1,2,2)
valor6=fourier(autocorre5)
long6=len(valor6)
l6=np.linspace(0,long6-1,long6)
normalizar6=np.abs(valor6/long6)
plt.scatter(l6, normalizar6)
plt.stem(l6, normalizar6, use_line_collection=True)
plt.title('alpha=7.2')

plt.savefig('alpha.png')