import numpy as np
import matplotlib.pyplot as plt

# (10 puntos) Cálculo de pi usando el Algoritmo 1.1 del 
# libro de Krauth (Imprimir el valor de pi).  

def direct_pi(N):
    N_hits = 0 
    for i in range(N):
        x = 2.0*(np.random.random()-0.5)
        y = 2.0*(np.random.random()-0.5)
        if(x**2 + y**2 < 1):
            N_hits += 1
    return 4*N_hits/N

print('direct_pi(10000)',direct_pi(10000))

# (10 puntos) El cálculo de pi usando el Algoritmo 1.2 del 
# libro de Krauth (Imprimir el valor de pi).

def markov_pi(N):
    N_hits = 0
    x = 1.0
    y = 1.0
    delta = 0.1
    for i in range(N):
        delta_x = delta * 2.0 * (np.random.random()-0.5)
        delta_y = delta * 2.0 * (np.random.random()-0.5)
        if(abs(x+delta_x)<1 and abs(y+delta_y)<1):
            x += delta_x
            y += delta_y
        if (x**2 + y**2 < 1):
            N_hits += 1
    return 4*N_hits/N

print('markov_pi(10000):', markov_pi(10000))

# (20 puntos)cálculo de una cadena de Markov de dos posiciones 
# usando el Algoritmo 1.8 de Krauth (Imprimir una secuencia de 20 valores).

def markov_two_site(k):
    def proba(m):
        p_0 = 0.7
        if(m==0): a = p_0
        if(m==1): a = 1 - p_0
        return a

    if(k==0): l=1
    if(k==1): l=0
    
    gamma = proba(l)/proba(k)
    r = np.random.random()
    if r < gamma:
        k = l
    return k

print('markov_two_site:')
k = 0
k_list = []
for i in range(20):
    k = markov_two_site(k)
    k_list.append(k)
print(k_list)

# (20 puntos) El cálculo de números que siguen una distribución gaussiana 
# usando el Algoritmo 1.18 de Krauth (Graficar el histograma de los valores 
# y comparar con la función que describe una gaussiana).

def gauss(sigma, n_points):
    phi = np.random.random(n_points) * 2.0 * np.pi
    gamma = -np.log(np.random.random(n_points))
    r = sigma * np.sqrt(2.0 * gamma)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y

def gaussian(x, sigma):
    return np.exp(-x**2/(2.0*sigma**2))/np.sqrt(2.0*np.pi*sigma**2)

plt.figure()
n_points = 10000
sigma = 1.0
x, y = gauss(sigma ,n_points)

x_model = np.linspace(x.min(), x.max(), n_points)
y_model = gaussian(x_model, sigma)

_ = plt.hist(x, bins=30, density=True, label='Box-Mueller')
plt.plot(x_model, y_model, label='Modelo')

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("gaussian.png")

# (20 puntos) cálculo de números que siguen una distribución gaussiana usando Metrópolis-Hastings 

def gauss_metropolis(sigma, N=100000, delta=1.0):
    lista = [np.random.random()]

    for i in range(1,N):
        propuesta  = lista[i-1] + (np.random.random()-0.5)*delta
        r = min(1, gaussian(propuesta, sigma)/gaussian(lista[i-1], sigma))
        alpha = np.random.random()
        if(alpha<r):
            lista.append(propuesta)
        else:
            lista.append(lista[i-1])
    return np.array(lista)


plt.figure()
n_points = 10000
sigma = 1.0
x = gauss_metropolis(sigma, N=n_points)

x_model = np.linspace(x.min(), x.max(), n_points)
y_model = gaussian(x_model, sigma)

_ = plt.hist(x, bins=30, density=True, label='Metropolis-Hastings')
plt.plot(x_model, y_model, label='Modelo')

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("gaussian_metropolis.png")
