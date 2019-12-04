import numpy as np
import matplotlib.pyplot as plt
import random

def walk(n_steps=1000):
    """Hace una marcha aleatoria"""
    delta_x = 2.0*np.random.random(n_steps) - 1.0
    delta_y = 2.0*np.random.random(n_steps) - 1.0
    L = np.sqrt(delta_x**2 + delta_y**2)
    delta_x = delta_x/L
    delta_y = delta_y/L
    x = np.cumsum(delta_x)
    y = np.cumsum(delta_y)
    R2 = x[-1]**2 + y[-1]**2 
    return {'x':x, 'y':y, 'delta_x':delta_x, 'delta_y':delta_y, 'R2':R2}

def ensemble(n_steps=1000):
    """Crea un conjunto de marchas aleatorias"""
    K = 31
    e = {}
    for k in range(K):
        e[k] = walk(n_steps=n_steps)
    return e

def average_ensemble(E):
    """Calcula el promedio de R2 sobre un ensemble de marchas"""
    r2 = np.zeros(len(E))
    for k in E.keys():
        r2[k] = E[k]['R2']
    return np.mean(r2)

# Crea un ensemble de marchas aleatorias 2D
E = ensemble(n_steps=1000)

# Hace la grafica de las marchas
plt.figure()
for k in E.keys():
    plt.plot(E[k]['x'], E[k]['y'], color='black', alpha=0.4)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ensemble de {:d} marchas aleatorias'.format(len(E)))
plt.savefig('ensemble.png')

# (30 puntos)
# Ensembles para diferentes valores de N
N_list = [10, 100, 500, 1000, 5000, 10000, 20000, 100000]
R2 = np.zeros(len(N_list))
for i in range(len(N_list)):
    E = ensemble(n_steps=N_list[i])
    R2[i] = average_ensemble(E)

# Hace la grafica R2 en funcion de N
plt.figure()
plt.plot(N_list, R2, label='Experimento')
plt.plot(N_list, N_list, label='Teoria')
plt.loglog()
plt.legend()
plt.title('Marchas Aleatorias en 2D')
plt.xlabel("N_steps")
plt.ylabel("<$R^2$>")
plt.savefig('comparacion_N_R2.png')

# (40 puntos)
# decaimiento radiactivo
def decay(N_init=1000, l=0.01):
    t = 0
    N_values = [N_init]
    T_values = [t]
    N = N_init
    while N > 0:
        r = np.random.random(N)
        deltaN = np.count_nonzero(r<l)
        t += 1
        N = N - deltaN
        if N > 0:
            N_values.append(N)
            T_values.append(t)
    return {'N':N_values, 'T':T_values}

data = {}
N_init = [10, 100, 1000, 10000, 100000]

for i,N in enumerate(N_init):
    data[i] = decay(N_init=N)

plt.figure()
for k in data.keys():
    plt.plot(data[k]['T'], np.log10(data[k]['N']), label='$N_i$ = {}'.format(N_init[k]))
plt.legend()
plt.xlabel('t')
plt.ylabel('$\log_{10} N$')
plt.xlim([0,1200])
plt.ylim([0,5])
plt.savefig("decay.png")
