


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
from numpy import empty



#######################################################################################################
# OUTILS
#######################################################################################################

# Méthode permettant d'afficher à l'écran un graphe non orienté et, éventuellement, un titre
def showGraphe(G, titre = ""):

    plt.title(titre)
    nx.draw(G, with_labels=True, node_size=1500, node_color="skyblue", pos=nx.circular_layout(G))

    plt.show()   

    

#######################################################################################################
# METHODES PARTIE 2
#######################################################################################################

# Méthode permet de supprimer un sommet d'un graphe G et d'obtenir le graphe G' résultant de la suppression du sommet v
def suppSommet(G, v) :
    if v not in G.keys() :
        print("Le sommet", v, "n'est pas dans le graphe G. Le graphe G' est équivalent à G.\n")
        return G

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

# Méthode permettant de générer des graphes aléatoires (avec n sommets et n > 0, p E ]0,1[ la probabilité qu'une arete entre 2 sommet soit créée)
def randomGraphe(n, p) :
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

#------------------------------------------------------------------------------------------------------

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
def algoGlouton(G) :
    C = [] # Ensemble représentant le couplage
    copyG = copy.deepcopy(G) # On réalise une copie de G afin de ne pas modifier l'original
    E = areteGraphe(copyG) # Liste des arêtes du graphe G

    # Début de l'algorithme
    while E != [] :
        v = sommetDegresMax(copyG) # Sommet au degrès maximal

        # Affichage complémentaire (permet de suivre l'évolution de l'algorithme avec les suppressions)
        # print(v)
        # showGraphe(convertGraph(copyG))

        suppSommet(copyG, v) # On supprime ce sommet du graphe
        C.append(v) # On ajoute le sommet à la couverture
        E = [e for e in E if v not in e] # On supprime les arêtes couvertes par le sommet

    return C

#######################################################################################################
# TESTS
#######################################################################################################

# Instanciation d'un graphe G
# G = {0 : [1, 2, 3], 1 : [0, 2], 2 : [0, 1], 3 : [0]}
# showGraphe(convertGraph(G))


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
G = randomGraphe(20, 0.5)
print(algoGlouton(G))
# showGraphe(convertGraph(G))

#------------------------------------------------------------------------------------------------------

# Tests de comparaison d'efficacité des 2 algorithmes