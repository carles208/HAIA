# -*- coding: utf-8 -*-
import time
from Agents.RandomAgent import RandomAgent as ra
from Agents.AdrianHerasAgent import AdrianHerasAgent as aha
from Agents.AlexPastorAgent import AlexPastorAgent as apa
from Agents.AlexPelochoJaimeAgent import AlexPelochoJaimeAgent as apja
from Agents.CarlesZaidaAgent import CarlesZaidaAgent as cza
from Agents.CrabisaAgent import CrabisaAgent as ca
from Agents.EdoAgent import EdoAgent as ea
from Agents.PabloAleixAlexAgent import PabloAleixAlexAgent as paaa
from Agents.SigmaAgent import SigmaAgent as sa
from Agents.TristanAgent import TristanAgent as ta
from Managers.GameDirector import GameDirector
from numpy import random
from enum import Enum
import numpy as np

class crossoverTypes(Enum):
    FIRST_TO_LAST = 1
    ELITISM = 2

class mutationTypes(Enum):
    RANDOM = 1
    VARIATION = 2
    ZEROS = 3

class oponentSelection(Enum):
    SECUENTIAL = 1
    RANDOM = 2
    SAME = 3

class scorePuntuation(Enum):
    FIRST_ONLY = 1
    FIRST_SECOND = 2
    SCALATED = 3

POPULATION_SIZE = 20
NUM_AGENTS = 10
ITERATIONS = 50
MUTATE_PER = 0.1
DISCARD_RATIO = 0.5
REPRODUCTIVE_RATE = 2
MUTATE_TYPE = mutationTypes.VARIATION
MAX_VARIATION = 0.25
CROSSOVER_ORDER_TYPE = crossoverTypes.FIRST_TO_LAST

''' 
||---- Función fitness ----||
    Opciones:
    -- Número de partidas a jugar
    -- Orden de juego en las paridas:
        --- Siempre primero (ACTUAL)
        --- Orden secuencial
        --- Siempre último
    -- Posición en la que se finaliza:
        --- Posición 1ª (ACTUAL)
        --- Puntuación escalable por posición
        --- Posición 2ª
    -- Adversarios:
        --- Siempre los mismos
        --- Aleatorios
        --- Secuenciales
''' 
NUM_PLAYS = 1
OPONENT_SELECTION_TYPE = oponentSelection.SECUENTIAL
SCORE_PUNTUATION_TYPE = scorePuntuation.FIRST_ONLY
AGENTS = [ra, aha, apa, apja, cza, ca, ea, paaa, sa, ta]

class Genome:
    prob = []

    def __init__(self, arr):
        self.prob = arr
    
    def __str__(self):
        return f"|| GENOME || \n --prob: {self.prob} \n"
    
    def __repr__(self):
        # Este método define la representación que se usa al imprimir una lista de objetos
        return f"Genome({self.prob})"

def circular_slice(arr, i, size):
    return np.concatenate((arr, arr))[i:i+size]

def testText(num:int):
    print(f"||{'-'*4} Test {num} {'-'*4}||")

def normaliceArr(arr):
    maxi = sum(arr)
    if maxi == 0:  # Evitar división por cero
        return arr
    return [x / maxi for x in arr]  # Normalizar cada elemento

def randomPopulation(numPopulation, numAgents):
    return  np.array([randomIndividual(numAgents) for _ in range(numPopulation)])

def randomIndividual(numAgents:int):
    indiv = random.rand(numAgents)
    indiv = normaliceArr(indiv)
    return Genome(indiv)

def mutate(son):
    mutation = None
    agents = []
    if(MUTATE_TYPE == mutationTypes.ZEROS):
        # Creando los agentes mutados
        for i in range(len(son.prob)):
           agents += [random.choice([0,son.prob[i]])] 
        
        agents = normaliceArr(agents)

        mutation = Genome(agents)
    elif(MUTATE_TYPE == mutationTypes.RANDOM):
        # Creando los agentes mutados
        for i in range(len(son.prob)):
           agents += [random.choice([random.random(),son.prob[i]])] 
        
        agents = normaliceArr(agents)
        
        mutation = Genome(agents)
    elif(MUTATE_TYPE == mutationTypes.VARIATION):
        # Creando los agentes mutados
        for i in range(len(son.prob)):
           agents += [random.choice([random.uniform(0,MAX_VARIATION),son.prob[i]])] 
        
        agents = normaliceArr(agents)
        
        mutation = Genome(agents)
    
    return mutation

def reproduce(genome1, genome2, number_to_reproduce):
    sons = []

    for son in range(number_to_reproduce):
        agents = []
        for i in range(NUM_AGENTS):
            agents += [random.choice([genome1.prob[i], genome2.prob[i]])]

            son = Genome(agents)

            son = mutate(son) if random.random() >= MUTATE_PER else son

        sons.append(son)
    return sons

from concurrent.futures import ThreadPoolExecutor
import threading

# Reescribiendo la función fitnessF con multithreading
def fitnessF(population, num_threads=4):
    print(population)
    print(len(population))
    start = time.time()

    fitness_lock = threading.Lock()
    fitness = []
    min_fitness = float('inf')

    def compute_fitness_for_indiv(indiv):
        acc_indiv_scores = []
        for i in range(NUM_AGENTS):
            acc_score = 0.0
            player = indiv.prob[i]
            player_tag = AGENTS[i]

            for j in range(NUM_PLAYS):  # Inicio partidas
                oponents = circular_slice(indiv.prob, i+1, 3)
                oponents_tag = circular_slice(AGENTS, i+1, 3)

                users_tag = [player_tag] + list(oponents_tag)
                users = [player] + list(oponents)

                try:
                    gameDirector = GameDirector(agents=users_tag, max_rounds=60, store_trace=False)
                    game_trace = gameDirector.game_start(print_outcome=False)

                    last_round = max(game_trace["game"].keys(), key=lambda r: int(r.split("_")[-1]))
                    last_turn = max(game_trace["game"][last_round].keys(), key=lambda t: int(t.split("_")[-1].lstrip("P")))
                    victory_points = game_trace["game"][last_round][last_turn]["end_turn"]["victory_points"]
                    winner = max(victory_points, key=lambda player: int(victory_points[player]))

                    if SCORE_PUNTUATION_TYPE == scorePuntuation.FIRST_ONLY:
                        if users_tag[int(winner.lstrip("J"))] == player_tag:
                            acc_score += 1

                except Exception as e:
                    print(f"Error: {e}")

            acc_indiv_scores += [acc_score]

        acc_indiv_scores = normaliceArr(acc_indiv_scores)

        dif_l = np.array(indiv.prob) - np.array(acc_indiv_scores)
        difn_l = np.abs(dif_l)
        score_indiv = sum(difn_l) * 100

        nonlocal min_fitness
        with fitness_lock:
            if score_indiv < min_fitness:
                min_fitness = score_indiv
            fitness.append((indiv, score_indiv))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(compute_fitness_for_indiv, population)

    print(time.time()-start)
    return fitness, min_fitness

# Fitness sin threading
def fitnessFST(population):
    print(population)
    print(len(population))
    start = time.time()
    min_fitness = float('inf')
    fitness = []
    for indiv in population:
        acc_indiv_scores = []
        for i in range(NUM_AGENTS):
            #print("-- ",indiv.prob)
            acc_score = 0.0
            player = indiv.prob[i]
            player_tag = AGENTS[i]
            #print(player_tag)
            for j in range(NUM_PLAYS): # Inicio partidas
                ''' 1.Selección de contrincantes '''
                oponents = []
                if(OPONENT_SELECTION_TYPE == oponentSelection.SECUENTIAL):
                    oponents = circular_slice(indiv.prob, i+1, 3)
                    oponents_tag = circular_slice(AGENTS, i+1, 3)
                
                users_tag = [player_tag] + (list)(oponents_tag)
                users = [player] + (list)(oponents)
                #print(users_tag)
                
                ''' 2.Ejecución de la partida y extracción del ganador'''
                try:
                    # Inicialización de partida:
                    gameDirector = GameDirector(agents=users_tag, max_rounds=60, store_trace=False)
                    game_trace = gameDirector.game_start(print_outcome=False)

                    # Recuperación del ganador:
                    last_round = max(game_trace["game"].keys(), key=lambda r: int(r.split("_")[-1]))
                    last_turn = max(game_trace["game"][last_round].keys(), key=lambda t: int(t.split("_")[-1].lstrip("P")))
                    victory_points = game_trace["game"][last_round][last_turn]["end_turn"]["victory_points"]
                    winner = max(victory_points, key=lambda player: int(victory_points[player]))

                    # Debugging
                    # print(player_tag)
                    # print(users_tag[int(winner.lstrip("J"))])
                    # print(users[int(winner.lstrip("J"))])

                    # print(users_tag[int(winner.lstrip("J"))] == player_tag)

                    ''' 3.Comprobación de ganador y asignación de puntos '''
                    if(SCORE_PUNTUATION_TYPE == scorePuntuation.FIRST_ONLY):
                        if(users_tag[int(winner.lstrip("J"))] == player_tag):
                            acc_score += 1

                except Exception as e:
                    print(f"Error: {e}")

            acc_indiv_scores += [acc_score]

        ''' Devolver fitness como la normal de la diferencia de probabilidades junto a su array individuo correspondiente '''
        acc_indiv_scores = normaliceArr(acc_indiv_scores)

        dif_l = np.array(indiv.prob) - np.array(acc_indiv_scores)
        difn_l = np.abs(dif_l)
        score_indiv = sum(difn_l) * 100 # Escalada del fitness para mayor diferenciación

        if(score_indiv < min_fitness):
            min_fitness = score_indiv

        fitness_indiv = (indiv,score_indiv)

        fitness.append(fitness_indiv)
    print(time.time()-start)
    return fitness, min_fitness

def selectSurvivors(population):
    united = sorted(population,key= lambda x:x[1])
    del united[(int)(POPULATION_SIZE*DISCARD_RATIO):]
    survivors_zip = united # Descartamos la mitad inferior del array 
    survivors, fit = zip(*survivors_zip) if survivors_zip else ([], [])
    return (list)(survivors)

def crossover(survivors):
    cross_num = int((1 - DISCARD_RATIO) * POPULATION_SIZE)
    new_population = survivors.copy()
    sons_created = 0

    if CROSSOVER_ORDER_TYPE == crossoverTypes.FIRST_TO_LAST:
        pairs = len(survivors) // 2
        for i in range(pairs):
            if sons_created >= cross_num:
                break  # Parar si ya tienes los hijos necesarios

            parent1 = survivors[i]
            parent2 = survivors[-(i + 1)]

            remaining_sons = cross_num - sons_created
            sons_to_create = min(REPRODUCTIVE_RATE, remaining_sons)

            sons = reproduce(parent1, parent2, sons_to_create)
            new_population += sons
            sons_created += sons_to_create

    return new_population


def geneticAlgorithm():
    print(f"||{"-"*10} INICIO ALGORITMO GENÉTICO {"-"*10}||")

    # Inicializamos la población
    population = randomPopulation(POPULATION_SIZE, NUM_AGENTS)

    actual_min_fitness = float("inf")
    stop_signals = 10

    # Algoritmo genético
    for i in range(1, ITERATIONS + 1):
        print(f"----> Iteration: {i} <----")
        # 1. Cálculo del fitness -> lista de tuplas [((individuo),score),....]
        fitness, min_fitness = fitnessF(population) # Se podría poner fitness y max_fitness_value
        survivors = selectSurvivors(fitness)
        population = crossover(survivors)

        # Condición de parada aquí
        if(min_fitness > actual_min_fitness):
            stop_signals -= 1
            if(stop_signals == 0):
                print(f"Condición de parada activada con fitness mínimo de {actual_min_fitness}")
                exit(0)
        else:
            stop_signals = 10
            actual_min_fitness = min_fitness

        best_indiv, best_fitness = min(fitness, key=lambda x: x[1])
        print("Mejor individuo:", best_indiv, "con fitness:", actual_min_fitness)

def testFunctions():
    #Test de population
    testText(1)
    population = randomPopulation(POPULATION_SIZE, NUM_AGENTS)
    print(f"Initial population:\n{population}\n")

    # Test de Crossover
    testText(2)
    initial = randomIndividual(NUM_AGENTS)
    initial2 = randomIndividual(NUM_AGENTS)
    print(f"Init: {initial}\nInit2: {initial2}\nCossover:{reproduce(initial,initial2)}\n")

    # Test de mutación
    testText(3)
    print(f"Initial: {initial}\nMutation: {mutate(initial)}\nSuma de probabilidades de la mutación: {sum(mutate(initial).prob)}")

    # Test de fitness
    testText(4)
    print(fitnessF(population))


#testFunctions()
geneticAlgorithm()