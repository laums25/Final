import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg

data = pd.read_csv('USArrests.csv')

#https://es.switch-case.com/54769911
#https://www.aprendemachinelearning.com/comprende-principal-component-analysis/
datos=np.array((data['Murder'], data['Assault'], data['UrbanPop'], data['Rape']))
new=datos.T

#Normalización
new -= np.mean(new, 0)
new /= np.std(new, 0)

#Covarianza
covarianza = np.cov(new.T)
print(covarianza)

#Autovalores y autovectores
autova, autove = linalg.eig(covarianza)

#orden
orden = np.argsort(autova)[::-1]
autove = autove[:,orden]
autova = autova[orden]

#Datos de los autovectores
autove = autove[:, :2]


#Producto punto
produ=np.dot(autove[:,0], new.T).T
produ2=np.dot(autove[:,1], new.T).T

x=autove[:,0]
y=autove[:,1]

#plt.scatter(produ,produ2)

lista=['Murder','Assault', 'UrbanPop', 'Rape']
lista2=['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
       'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'lowa', 'Kansas', 'Kentucky', 'Lousiana', 'Maine', 'Maryland', 'Massachusetts',
       'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
       'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsyvalnia', 'Rhode Island', 'South Carolina',
       'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wiscosin', 'Wyoming']

#print(len(lista2))
#https://jbhender.github.io/Stats506/F17/Projects/G18.html
plt.figure(figsize=(15,9))

for i in range(0,4):
    plt.arrow(0, 0, 1.5*x[i], -1.5*y[i], color='r', width=0.0005, head_width=0.05)
    plt.text(1.5*x[i]+0.1, -1.5*y[i]+0.1, lista[i], color='r')

for k in range(0,50):
    plt.scatter(produ[k], -produ2[k],s=0.05)
    plt.text(produ[k], -produ2[k], lista2[k], color='b',fontsize=8)
    
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
    
plt.savefig('arrestos.png')




data2 = pd.read_csv('Cars93.csv')

datos2=np.array((data2['Horsepower'], data2['Length'], data2['Width'], data2['Fuel.tank.capacity']))
new2=datos2.T


#Normalización
new2 -= np.mean(new2, 0)
new2 /= np.std(new2, 0)

#Covarianza
covarianza2 = np.cov(new2.T)

#Autovalores y autovectores
autova2, autove2 = linalg.eig(covarianza2)

#orden
orden2 = np.argsort(autova2)[::-1]
autove2 = autove2[:,orden2]
autova2 = autova2[orden2]

#Datos de los autovectores
autove2 = autove2[:, :2]


#Producto punto
produ3=np.dot(autove2[:,0], new2.T).T
produ4=np.dot(autove2[:,1], new2.T).T

x2=autove2[:,0]
y2=autove2[:,1]

#plt.scatter(produ,produ2)
lista3=['Horsepower','Length', 'Width', 'Fuel.tank.capacity']


plt.figure(figsize=(15,9))



for m in range(0,4):
    plt.arrow(0, 0, 1.5*x2[m], -1.5*y2[m], color='r', width=0.0005, head_width=0.05)
    plt.text(1.5*x2[m]+0.1, -1.5*y2[m]+0.1, lista3[m], color='r', fontsize=12)
    
for s in range(0,93):
    lista4=data2['Make'][s]
    plt.scatter(produ3[s], -produ4[s], s=0.05)
    plt.text(produ3[s], -produ4[s], lista4, color='b',fontsize=7)  
    
plt.xlim(-5,5)  
plt.ylim(-2,2.5)
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
    
plt.savefig('cars.png')