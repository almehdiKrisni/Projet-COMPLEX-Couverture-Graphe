


#                                   PROJET COMPLEX 2021 GROUPE 3
#                                       COUVERTURE DE GRAPHE
#                                   Almehdi KRISNI , Alessia LOI






#######################################################################################################
# LIBRAIRIES PYTHON
#######################################################################################################

import copy
import random
import networkx as nx
import matplotlib.pyplot as plt
from numpy import empty, save
import time
import datetime



#######################################################################################################
# OUTILS
#######################################################################################################

# Méthodes permettant de convertir un tuple (V,E) ou un graphe G en un graphe de la librairie nxgraph
def convertGraph(G) :
    newG = nx.Graph()

    newG.add_nodes_from(list(G.keys()))
    for v1 in G.keys() :
        for v2 in G.keys() :
            if (v2, v1) not in newG.edges and v2 in G[v1]:
                newG.add_edge(v1, v2)

    return newG

#------------------------------------------------------------------------------------------------------

# Méthode permettant d'obtenir une liste d'arêtes à partir d'un graphe G (utile pour la partie 3)
def areteGraphe(G) :
    E = []
    for s1 in G.keys() :
        for s2 in G[s1] :
            if (s2,s1) not in E :
                E.append((s1,s2))
    return E

#------------------------------------------------------------------------------------------------------

# Méthode permettant d'acquérir un graphe G (modelisation : dictionnaire) depuis un fichier texte
def acquisitionGraphe(nomFichier):
    G = {}
    phase = 0
    with open(nomFichier, 'r') as fichier:
        for ligne in fichier:
            if ligne.startswith('Nombre de sommets') or ligne.startswith('Nombre d aretes'):
                phase = 0
                continue
            if ligne.startswith('Sommets'):
                phase = 1
                continue
            if ligne.startswith('Aretes'):
                phase = 2
                continue
            
            if phase == 1:
                G[ligne.strip()] = []
            if phase == 2:
                e = ligne.strip().split()
                if len(e) == 2:
                    (s1, s2) = e
                    G[s1].append(s2)
                    G[s2].append(s1)
                else :
                    print("Format de fichier invalide : chaque arete doit avoir 2 sommets")

    return G   

#------------------------------------------------------------------------------------------------------

# Méthode permettant d'afficher à l'écran un graphe non orienté et, éventuellement, un titre
def showGraphe(G, titre = ""):

    plt.title(titre)
    nx.draw(G, with_labels=True, node_size=1500, node_color="skyblue", pos=nx.circular_layout(G))

    plt.show()   

#------------------------------------------------------------------------------------------------------

# Méthode permettant d'afficher un graphique de comparaison des performances (temps de calcul) des algorithmes algo_couplage et algo_glouton
def plotPerformancesCouplage(p, nbIterations, secondesMaxAutorises, verbose = False, save = False):
    """ p : la probabilité qu'une arete entre 2 sommets soit crée, p E ]0,1[
    """
    resAlgoCouplage = []

    # nMax : taille jusqu'à laquelle l'algorithme tourne rapidement, i.e temps G(nMax,p) < secondesMaxAutorises
    
    # Calcul de la taille nMax pour l'algorithme algoCouplage(G)
    nMaxACouplage = 0
    t = 0
    while t < secondesMaxAutorises:
        nMaxACouplage += 1
        
        # Méthode permettant de générer des graphes aléatoires
        G = randomGraphe(nMaxACouplage, p)

        t1 = time.time()
        algoCouplage(G)
        t2 = time.time()
        t = t2-t1

    if verbose :
        print("nMaxACouplage = ", nMaxACouplage, "\n")

    print("hello 1")

    yCouplage = []  # axe des ordonnées : liste des temps de calcul moyen, pour l'algorithme algoCouplage(G)
    
    # Pour chaque 1/10 de nMax (1/10 nMaxACouplage, 1/10 nMaxAGlouton)
    for i in range(1, 11):

        tabTempsCouplage = []
        moyTempsCouplage = 0

        # Pour chacune des nbIterations démandées
        for ite in range(nbIterations):

            # Méthode permettant de générer des graphes aléatoires
            G = randomGraphe((int)(nMaxACouplage * i/10), p)

            # Execution et recueil statistiques algoCouplage(G)
            t1 = time.time()
            res = algoCouplage(G)
            t2 = time.time()
            t = t2-t1
            tabTempsCouplage.append(t) # temps de calcul de l'algorithme pour l'itération courante
            resAlgoCouplage.append(len(res)) # qualité des solutions pour l'itération courante

            if verbose : 
                print("x = ", i, "/10 nMax, iteration n.", ite+1, ":", "\n\t\ttabTempsCouplage =", tabTempsCouplage, "\n")

        # Calcul et stockage du temps d'execution moyen de chaque algorithme par rapport aux 'nbIterations' éxecutions
        moyTempsCouplage = sum(tabTempsCouplage)/len(tabTempsCouplage)

        yCouplage.append(moyTempsCouplage)

        if verbose : 
            print("\nx = ", i, "/10 nMax : moyTempsCouplage =", moyTempsCouplage)
            print("----------------------------------------------------------------------------------------------\n")

    print("hello 2")


    # Construction et affichage du tracé
    plt.rc('xtick', labelsize=5)    # fontsize of the tick labels

    x = ["1/10 nMAX", "2/10 nMAX", "3/10 nMAX", "4/10 nMAX", "5/10 nMAX", "6/10 nMAX", "7/10 nMAX", "8/10 nMAX", "9/10 nMAX", "nMAX"]
    plt.figure()
    plt.suptitle("Performances de l'algorithme de couplage classique", color = 'red')
    plt.title("Analyse des temps de calcul en fonction de nMax. nMax algo_couplage = " + str(nMaxACouplage))
    plt.xlabel("n")
    plt.ylabel("t(n)")
    plt.plot(x, yCouplage)
    plt.legend()

    # Sauvegarde du tracé
    if save != None:
        plt.savefig("TestResults/algo_couplage_" + (str)(datetime.datetime.today()) + ".jpeg", transparent = True)

    plt.show()



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot pour les performances de l'algorithme glouton
def plotPerformancesGlouton(p, nbIterations, secondesMaxAutorises, verbose = False, save = False):
    """ p : la probabilité qu'une arete entre 2 sommets soit crée, p E ]0,1[
    """
    resAlgoGlouton = []

    # nMax : taille jusqu'à laquelle l'algorithme tourne rapidement, i.e temps G(nMax,p) < secondesMaxAutorises

    # Calcul de la taille nMax pour l'algorithme algoGluton(G)
    nMaxAGlouton = 0
    t = 0
    while t < secondesMaxAutorises:
        nMaxAGlouton += 1
        print(nMaxAGlouton)

        # Méthode permettant de générer des graphes aléatoires

        G = randomGraphe(nMaxAGlouton, p)

        t1 = time.time()
        algoGlouton(G)
        t2 = time.time()
        t = t2-t1
        

    if verbose :
        print("nMaxAGlouton = ", nMaxAGlouton, "\n")

    yGlouton = []   # axe des ordonnées : liste des temps de calcul moyen, pour l'algorithme algoGluton(G)  
    
    # Pour chaque 1/10 de nMax (1/10 nMaxACouplage, 1/10 nMaxAGlouton)
    for i in range(1, 11):

        tabTempsGlouton = []
        moyTempsGlouton = 0

        # Pour chacune des nbIterations démandées
        for ite in range(nbIterations):

            # Méthode permettant de générer des graphes aléatoires
            G = randomGraphe((int)(nMaxAGlouton * i/10), p)

            # Execution et recueil statistiques algoGluton(G)
            t1 = time.time()
            res = algoGlouton(G)
            t2 = time.time()
            t = t2-t1
            tabTempsGlouton.append(t) # temps de calcul de l'algorithme pour l'itération courante
            resAlgoGlouton.append(len(res)) # qualité des solutions pour l'itération courante

            if verbose : 
                print("x = ", i, "/10 nMax, iteration n.", ite+1, ":", "\n\t\ttabTempsGlouton = ", tabTempsGlouton)

        # Calcul et stockage du temps d'execution moyen de chaque algorithme par rapport aux 'nbIterations' éxecutions
        moyTempsGlouton = sum(tabTempsGlouton)/len(tabTempsGlouton)

        yGlouton.append(moyTempsGlouton)

        if verbose : 
            print("\nx = ", i, "/10 nMax :\tmoyTempsGlouton = ", moyTempsGlouton)
            print("----------------------------------------------------------------------------------------------\n")


    # Construction et affichage du tracé
    plt.rc('xtick', labelsize=5)    # fontsize of the tick labels

    x = ["1/10 nMAX", "2/10 nMAX", "3/10 nMAX", "4/10 nMAX", "5/10 nMAX", "6/10 nMAX", "7/10 nMAX", "8/10 nMAX", "9/10 nMAX", "nMAX"]
    plt.figure()
    plt.suptitle("Performances de l'algorithme de couplage glouton", color = 'red')
    plt.title("Analyse des temps de calcul en fonction de nMax. nMax algo_glouton =" + str(nMaxAGlouton))
    plt.xlabel("n")
    plt.ylabel("t(n)")
    plt.plot(x, yGlouton)
    plt.legend()

    # Sauvegarde du tracé
    if save != None:
        plt.savefig("TestResults/algo_glouton_" + (str)(datetime.datetime.today()) + ".jpeg", transparent = True)

    plt.show()




#######################################################################################################
# METHODES PARTIE 2
#######################################################################################################

# Méthode permet de supprimer un sommet d'un graphe G et d'obtenir le graphe G' résultant de la suppression du sommet v
def suppSommet(initG, v) :
    if v not in initG.keys() :
        print("Le sommet", v, "n'est pas dans le graphe G. Le graphe G' est équivalent à G.\n")
        return initG

    # On effectue une copie de G
    G = copy.deepcopy(initG)

    # On retire le sommet v
    del G[v]

    # On retire les aretes liées au sommet v en créant une nouvelle liste de jointures
    for s in G.keys() :
        l = []
        for e in G[s] :
            if (e != v) :
                l.append(e)
        G[s] = l
            
    # On retourne G'
    return G

#------------------------------------------------------------------------------------------------------

# Méthode permettant de supprimer plusieurs sommets à la fois d'un graphe G et d'obtenir le graphe G' résultant de la suppression des sommets
def multSuppSommet(G, ensv) :
    modifG = copy.deepcopy(G)

    for v in ensv :
        modifG = suppSommet(modifG, v)

    return modifG

#------------------------------------------------------------------------------------------------------

# Méthode renvoyant un tableau (dictionnaire) contenant les degres de chaque sommet du graphe G
def degresSommet(G) :

    # Création d'un tableau (dictionnaire) contenant les degres de chaque sommet du graphe G
    tab = dict()
    for v in G.keys() :
        tab[v] = len(list(G[v]))

    return tab

#------------------------------------------------------------------------------------------------------

# Méthode permettant de retourner l'indice du sommet ayant le degres maximal dans le graphe G
def sommetDegresMax(G) :

    """ a) create a list of the dict's keys and values; 
        b) return the key with the max value"""  
    deg = degresSommet(G)
    degres = list(deg.values())
    v = list(deg.keys())
    return v[degres.index(max(degres))]

#------------------------------------------------------------------------------------------------------

# Méthode permettant de générer des graphes aléatoires
def randomGraphe(n, p) :
    """ n : nombre de sommets, n > 0
        p : la probabilité qu'une arete entre 2 sommet soit créée, p E ]0,1[
    """
    if n < 1 :
        print("Il faut que n soit supérieur ou égal à 1 (n = nombre de sommets).\n")
        return ([],[])

    # Création du graphe
    G = dict()

    # Liste des sommets
    for i in range(n) :
        G[i] = []
    
    # Liste des aretesS
    for v1 in G.keys() :
        for v2 in G.keys() :
            if v1 != v2 :
                if random.uniform(0,1) < p :
                    if (v2 not in G[v1]) and (v1 not in G[v2]) :
                        G[v1].append(v2)
                        G[v2].append(v1)
    
    # On organise les listes de sommets adjacents pour faciliter la lecture
    for s in G :
        G[s].sort()

    return G





#######################################################################################################
# METHODES PARTIE 3
#######################################################################################################

# Couplage = ensemble d'arêtes n'ayant pas d'extrémité en commun

# Méthode representant l'algorithme de couplage sur le graphe G
def algoCouplage(G) :
    C = list() # Ensemble représentant le couplage

    # Début de l'algorithme
    for s1 in list(G.keys()) :
        for s2 in list(G[s1]) :
            if (s1 not in C) and (s2 not in C) :
                C.append(s1)
                C.append(s2)
                break

    return C

#------------------------------------------------------------------------------------------------------

# Méthode représentant l'algorithme glouton de couplage sur le graphe G
def algoGlouton(G, visual=False) :
    C = [] # Ensemble représentant le couplage
    copyG = copy.deepcopy(G) # On réalise une copie de G afin de ne pas modifier l'original
    E = areteGraphe(copyG) # Liste des arêtes du graphe G

    # Début de l'algorithme
    while E != [] :
        v = sommetDegresMax(copyG) # Sommet au degrès maximal

        if (visual) :
            print(v)
            showGraphe(convertGraph(copyG))

        suppSommet(copyG, v) # On supprime ce sommet du graphe
        C.append(v) # On ajoute le sommet à la couverture
        E = [e for e in E if v not in e] # On supprime les arêtes couvertes par le sommet

    return C



#######################################################################################################
# METHODES PARTIE 4
#######################################################################################################

# Fonction réalisant le branchement
def branchement(G) :
    optiC = None # optiC = ensemble de sommets représentant la solution optimate (on cherche à minimiser la taille de la couverture)

    areteInitiale = areteGraphe(G)[0] # On récupère la première arete du graphe

    # Un état est de la forme [ Couverture C actuelle, Dictionnaire de graphe G ]
    statesToStudy = [] # Pile des états du branchement à étudier
    statesToStudy.append([[areteInitiale[0]], suppSommet(G, areteInitiale[0])])
    statesToStudy.append([[areteInitiale[1]], suppSommet(G, areteInitiale[1])])

    # Début de l'algorithme de branchement
    while (statesToStudy != []) :

        # On récupère la tete de la file et on la supprime de statesToStudy
        state = statesToStudy.pop(0)

        # Cas où G est un graphe sans aretes
        if (areteGraphe(state[1]) == []) :
            if (optiC == None) or (len(state[0]) < len(optiC)) :
                optiC = state[0]

        # Cas où G n'est pas un graphe sans aretes
        else :
            # On récupère une arete aléatoire
            areteEtude = areteGraphe(state[1])[0] # On récupère la première arete du graphe
            leftNode = areteEtude[0]
            rightNode = areteEtude[1]

            # On ajoute deux feuilles à la liste (on priorise le fils de gauche, soit le premier élément de l'arete étudiée)
            statesToStudy.insert(0, [state[0] + [rightNode], suppSommet(state[1], rightNode)])
            statesToStudy.insert(0, [state[0] + [leftNode], suppSommet(state[1], leftNode)])
        
    # On retourne C
    return optiC


    



#######################################################################################################
# TESTS
#######################################################################################################

# Instanciation d'un graphe G (modelisation : dictionnaire)
# G = {0 : [1, 2, 3], 1 : [0, 2], 2 : [0, 1], 3 : [0]}
# showGraphe(convertGraph(G))

# Instanciation d'un graphe G (modelisation : librairie graphe networkx)
# V = [0, 1, 2, 3]
# E = [(0,1), (0,2), (0,3), (1,2)]

# G = nx.Graph()
# G.add_nodes_from(V) # sommets
# G.add_edges_from(E) # aretes
# showGraphe(G)

#------------------------------------------------------------------------------------------------------

# Test méthode suppSommet
# print("Graphe G\n", G, "\n")
# newG = suppSommet(G, 0)
# print("Graphe G'\n", newG, "\n")

#------------------------------------------------------------------------------------------------------

# Test méthode multSuppSommet
# newG = multSuppSommet(G, [0, 1])
# print("Graphe G'\n", newG, "\n")

#------------------------------------------------------------------------------------------------------

# Tests des méthodes degresSommet et sommetDegresSommet
# print(degresSommet(G))
# print(sommetDegresMax(G))

#------------------------------------------------------------------------------------------------------

# Tests sur la génération aléatoire de graphe
# randG = randomGraphe(8, 0.1)
# print("Graphe G\n", randG, "\n")
# showGraphe(convertGraph(randG))

#------------------------------------------------------------------------------------------------------

# Tests sur l'algorithme de couplage
# G = randomGraphe(8, 0.2)
# print(algoCouplage(G))
# print(areteGraphe(G))
# showGraphe(convertGraph(G))

#------------------------------------------------------------------------------------------------------

# Tests sur l'algorithme de couplage glouton
# G = randomGraphe(20, 0.5)
# print(algoGlouton(G))
# showGraphe(convertGraph(G))

#------------------------------------------------------------------------------------------------------

# Tests de comparaison d'efficacité des 2 algorithmes

#------------------------------------------------------------------------------------------------------

# Test méthode acquisitionGraphe depuis un fichier texte
# G3 = acquisitionGraphe("exempleinstance.txt")
# print("G = ", G3, "\n")
# showGraphe(convertGraph(G3))

#------------------------------------------------------------------------------------------------------

# Test méthode plotPerformances(p, nbIterations, secondesMaxAutorises, verbose = False, save = False)
# plotPerformancesGlouton(0.3, 15, 0.01, verbose=True, save = False)
# plotPerformancesCouplage(0.3, 15, 0.01, verbose=True, save = False)

#------------------------------------------------------------------------------------------------------

# Test sur la méthode de branchement
print(branchement(acquisitionGraphe("exempleinstance.txt")))
showGraphe(convertGraph(acquisitionGraphe("exempleinstance.txt")))