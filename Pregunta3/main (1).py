import numpy as np
import random
n_nodos = 5
n_poblacion= 10
mutacion = 0.3
# Generating a list of coordenades representing each city
pesos=  [
    [0, 7, 9, 8, 20],
    [7, 0, 10, 4, 11],
    [9, 10, 0, 15, 5],
    [8, 4, 15, 0, 17],
    [20, 11, 5, 17, 0]
]
names_nodos = np.array(['A', 'B', 'C', 'D', 'E'])
nodos_dic = { x:y for x,y in zip(names_nodos,pesos)}
def distancia_nodos(a,b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
def distancia_nodos2(nodo_a, nodo_b, nod_dict):
    return distancia_nodos(nod_dict[nodo_a], nod_dict[nodo_b])
def genesis(n_list, n_poblacion):
    poblacion_set = []
    for i in range(n_poblacion):
        sol_i = n_list[np.random.choice(list(range(n_nodos)), n_nodos, replace=False)]
        poblacion_set.append(sol_i)
    return np.array(poblacion_set)
poblacion_set = genesis(names_nodos, n_poblacion)
poblacion_set
def fit_(n_list, nodos_dic):
    total = 0
    for i in range(n_nodos-1):
        a = n_list[i]
        b = n_list[i+1]
        total += distancia_nodos2(a,b, nodos_dic)
    return total
def fit_2(poblacion_set, nodos_dic):
    fitnes_list = np.zeros(n_poblacion)
    for i in  range(n_poblacion):
        fitnes_list[i] = fit_(poblacion_set[i], nodos_dic)
    return fitnes_list
fitnes_list = fit_2(poblacion_set,nodos_dic)
fitnes_list
def progenitor(poblacion_set,fitnes_list):
    total_fit = fitnes_list.sum()
    prob_list = fitnes_list/total_fit
    progenitor_list_a = np.random.choice(list(range(len(poblacion_set))), len(poblacion_set),p=prob_list, replace=True)
    progenitor_list_b = np.random.choice(list(range(len(poblacion_set))), len(poblacion_set),p=prob_list, replace=True)  
    progenitor_list_a = poblacion_set[progenitor_list_a]
    progenitor_list_b = poblacion_set[progenitor_list_b]   
    return np.array([progenitor_list_a,progenitor_list_b])
progenitor_list = progenitor(poblacion_set,fitnes_list)
progenitor_list[0][2]
def progenitor2(prog_a, prog_b):
    offspring = prog_a[0:5]
    for nodo in prog_b:
        if not nodo in offspring:
            offspring = np.concatenate((offspring,[nodo]))
    return offspring  
def poblacion_(progenitor_list):
    new_poblacion_set = []
    for i in range(progenitor_list.shape[1]):
        prog_a, prog_b = progenitor_list[0][i], progenitor_list[1][i]
        offspring = progenitor2(prog_a, prog_b)
        new_poblacion_set.append(offspring)  
    return new_poblacion_set
new_poblacion_set = poblacion_(progenitor_list)
new_poblacion_set[0]
def mutacion_offspring(offspring):
    for q in range(int(n_nodos*mutacion)):
        a = np.random.randint(0,n_nodos)
        b = np.random.randint(0,n_nodos)
        offspring[a], offspring[b] = offspring[b], offspring[a]
    return offspring   
def mutacion_poblacion(new_poblacion_set):
    mutacion_pop = []
    for offspring in new_poblacion_set:
        mutacion_pop.append(mutacion_offspring(offspring))
    return mutacion_pop
mutacion_pop = mutacion_poblacion(new_poblacion_set)
mutacion_pop[0]
mejor_solucion = [-1,np.inf,np.array([])]
for i in range(10000):
    if i%100==0: print(i, fitnes_list.min(), fitnes_list.mean())
    fitnes_list = fit_2(mutacion_pop,nodos_dic)
    if fitnes_list.min() < mejor_solucion[1]:
        mejor_solucion[0] = i
        mejor_solucion[1] = fitnes_list.min()
        mejor_solucion[2] = np.array(mutacion_pop)[fitnes_list.min() == fitnes_list]
    progenitor_list = progenitor(poblacion_set,fitnes_list)
    new_poblacion_set = poblacion_(progenitor_list) 
    mutacion_pop = mutacion_poblacion(new_poblacion_set)
print(mejor_solucion)


