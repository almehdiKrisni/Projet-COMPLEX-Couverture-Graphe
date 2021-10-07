import copy
import random

# Méthode permet de supprimer un sommet d'un graphe G et d'obtenir le graphe G' résultant de la suppression du sommet v
def suppSommet(G, v) :
    if v not in G[0] :
        print("Le sommet", v, "n'est pas dans le graphe G. Le graphe G' est équivalent à G.\n")
        return G

    # On retire le sommet v
    Vprime = [i for i in G[0] if i != v]

    # On retire les aretes liées au sommet v
    Eprime = []
    for e in G[1] :
        if v not in e :
            Eprime.append(e)
            

    # On retourne G'
    return (Vprime, Eprime)

# Test méthode suppSommet

V = [12, 7, 3, 4, 0, 1, 9, 10]
E = [(7,12), (12,4), (12,0), (0,1), (0,9), (7,3), (7,10), (0,3)]
G = (V,E)

# print("Graphe G\n", G[0], "\n", G[1], "\n")
# newG = suppSommet(G, 99)
# print("Graphe G'\n", newG[0], "\n", newG[1])

# Méthode permettant de supprimer plusieurs sommets à la fois d'un graphe G et d'obtenir le graphe G' résultant de la suppression des sommets
def multSuppSommet(G, ensv) :
    modifG = copy.deepcopy(G)

    for v in ensv :
        modifG = suppSommet(modifG, v)

    return modifG

# Test méthode multSuppSommet

newG = multSuppSommet(G, [12, 9])
# print("Graphe G'\n", newG[0], "\n", newG[1])

# Méthode renvoyant un tableau (dictionnaire) contenant les degres de chaque sommet du graphe G
def degresSommet(G) :

    # Création d'un tableau (dictionnaire) contenant les degres de chaque sommet du graphe G
    tab = dict()
    for v in G[0] :
        val = 0
        for e in G[1] :
            if v in e :
                val += 1
        tab[v] = val

    return tab

# Méthode permettant de retourner l'indice du sommet ayant le degres maximal dans le graphe G
def sommetDegresMax(G) :

    """ a) create a list of the dict's keys and values; 
        b) return the key with the max value"""  
    deg = degresSommet(G)
    degres = list(deg.values())
    v = list(deg.keys())
    return v[degres.index(max(degres))]

# Tests des méthodes degresSommet et sommetDegresSommet
print(degresSommet(G))
print(sommetDegresMax(G))

# Méthode permettant de générer des graphes aléatoires (avec n sommets et n > 0, p E ]0,1[ la probabilité qu'une arete entre 2 sommet soit créée)
def randomGraphe(n, p) :
    if n < 1 :
        print("Il faut que n soit supérieur ou égal à 1 (n = nombre de sommets).\n")
        return ([],[])

    # Liste des sommets
    V = [i for i in range(n)]
    
    # Liste des aretes
    E = []
    for v in V :
        for v2 in V :
            if v != v2 :
                if random.uniform(0,1) < p :
                    if (v2,v) not in E :
                        E.append((v,v2))
    
    return (V,E)

# Tests sur la génération aléatoire de graphe
randG = randomGraphe(8, 0.3)
print("Graphe G\n", randG[0], "\n", randG[1], "\n")