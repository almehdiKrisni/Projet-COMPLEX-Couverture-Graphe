


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
import time
import datetime
import math



#######################################################################################################
# OUTILS
#######################################################################################################


# Méthode permettant d'obtenir une liste d'arêtes à partir d'un graphe G (utile pour la partie 3)
def aretesGrapheToList(G) :
    """ G : un dictionnaire representant un graphe { sommet s : sommets adjacents à s}
    """
    E = []
    for s1 in G.keys() :
        for s2 in G[s1] :
            if (s2,s1) not in E :
                E.append((s1,s2))
    return E

#------------------------------------------------------------------------------------------------------

# Méthode permettant d'acquérir un graphe G (modelisation : dictionnaire) depuis un fichier texte
def acquisitionGraphe(nomFichier):
    """ nomFichier : chaine de caractéres representant un fichier texte d'extension .txt
    """
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
                    print("Format de fichier invalide : chaque arete doit etre constituée de exactement 2 sommets")

    return G   

#------------------------------------------------------------------------------------------------------

# Méthode permettant d'afficher à l'écran un graphe non orienté et, éventuellement, un titre
def showGraphe(G, titre = "G"):
    """ G : un dictionnaire representant un graphe { sommet s : sommets adjacents à s}
        titre : titre du graphe à afficher, 'G' par defaut
    """
    newG = nx.Graph()
    newG.add_nodes_from(list(G.keys()))
    for v1 in G.keys() :
        for v2 in G.keys() :
            if (v2, v1) not in newG.edges and v2 in G[v1]:
                newG.add_edge(v1, v2)

    plt.title(titre)
    nx.draw(newG, with_labels=True, node_size=1500, node_color="skyblue", pos=nx.circular_layout(G))

    plt.show()   

#------------------------------------------------------------------------------------------------------

# Méthode permettant d'afficher un graphique de comparaison des performances ("temps de calcul" et "qualité des Solutions") de l'algorithme choisi
def plotPerformances(p, nbIterations, secondesMaxAutorises, mode, verbose = False, save = False):
    """ p : la probabilité qu'une arete entre 2 sommets soit crée, p E ]0,1[
        nbIterations : nombre d'éxecutions de l'algorithme, dans le but d'en déduir une performance moyenne
        secondesMaxAutorises : temps maximum autorisé pour l'éxecution de l'algorithme
        nbNoeuds : nombre de nodes allant etre créées au maximum dans le graphe
        mode : valeur déterminant l'algorithme allant etre utilisé
        verbose : "True" pour afficher le détail des itérations
        save : "True" pour enregistrer le tracé en format jpg
    """
    # Calcul de la taille nMaxAGlouton pour l'algorithme (G)
    # nMax : taille jusqu'à laquelle l'algorithme tourne rapidement, i.e temps G(nMax,p) < secondesMaxAutorises
    nMax = 0
    t = 0
    while t < secondesMaxAutorises :
        nMax += 1
        
        # Méthode permettant de générer des graphes aléatoires
        G = randomGraphe(nMax, p)

        t1 = time.time()

        # Selection du mode (algorithme allant etre utilisé)
        if (mode == 1) :
            res = algoCouplage(G)
        elif (mode == 2) :
            res = algoGlouton(G)
        elif (mode == 3) :
            res = branchement(G)
        elif (mode == 4) :
            res = branchementBornesCouplage(G)
        elif (mode == 5) :
            res = branchementOptimiseCouplage(G)
        elif (mode == 6) :
            res = branchementOptimiseCouplage_uDegreMax(G)
        else :
            print("Aucun mode ne correspond à la valeur passée en paramètre. Veuillez choisir une autre valeur de mode.")
            return

        t2 = time.time()
        t = t2-t1

    if verbose :
        print("nMax = ", nMax, "\n")

    y1 = []  # axe des ordonnées : liste des temps de calcul moyen, pour l'algorithme sélectionné(G)
    y2 = []  # axe des ordonnées : liste des tailles des couplages (nombre de sommets) moyen, pour l'algorithme sélectionné(G)
    y3 = []  # axe des ordonnées : liste du nombre de noeuds générés pour l'algorithme de branchement (G)
    x = []   # axe des abscisses : liste de "nombre de sommets" {1/10 nbIterations, 2/10 nbIterations, ... , nbIterations}
    
    # Pour chaque 1/10 de nMax
    for i in range(1, 11) :

        tabTemps = []
        moyTemps = 0
        resAlgo = []
        moyQualiteSolutions = 0
        tabNoeudsGeneneres = []
        moyNbNoeudsGeneres = 0
        nbNoeuds = 0
        

        # Pour chacune des nbIterations démandées en paramètre
        for ite in range(nbIterations):

            # Méthode permettant de générer des graphes aléatoires
            G = randomGraphe(int(nMax * (i / 10)), p)

            # Execution et recueil statistiques de l'algorithme (G)
            t1 = time.time()

            # Variable res et noeud permettant de stocker le résultat de l'algorithme et le nombre de noeuds générés
            
            # Selection du mode (algorithme allant etre utilisé)
            if (mode == 1) :
                res = algoCouplage(G)
            elif (mode == 2) :
                res = algoGlouton(G)
            elif (mode == 3) :
                res, nbNoeuds = branchement(G)
            elif (mode == 4) :
                res, nbNoeuds = branchementBornesCouplage(G)
            elif (mode == 5) :
                res, nbNoeuds = branchementOptimiseCouplage(G)
            elif (mode == 6) :
                res, nbNoeuds = branchementOptimiseCouplage_uDegreMax(G)
            else :
                print("Aucun mode ne correspond à la valeur passée en paramètre. Veuillez choisir une autre valeur de mode.")
                return

            t2 = time.time()
            t = t2-t1

            tabTemps.append(t) # temps de calcul de l'algorithme pour l'itération courante
            resAlgo.append(len(res)) # qualité des solutions pour l'itération courante
            if (mode > 2) : # Dans le cas ou on utilise un algorithme de branchement
                tabNoeudsGeneneres.append(nbNoeuds)

            if verbose : 
                print("x = ", i, "/10 nMax, iteration n.", ite+1, ":", "\n\t\ttabTemps =", tabTemps, "\n\t\tresAlgo =", resAlgo, "\n")

        # Calcul et stockage du temps d'execution moyen et de la qualité des solutions moyenne par rapport aux 'nbIterations' éxecutions
        moyTemps = sum(tabTemps)/len(tabTemps)
        moyQualiteSolutions = int(sum(resAlgo)/len(resAlgo))
        if (mode > 2) :
            moyNbNoeudsGeneres = int(sum(tabNoeudsGeneneres)/len(tabNoeudsGeneneres))
        

        y1.append(moyTemps)
        y2.append(moyQualiteSolutions)
        if (mode > 2) :
            y3.append(moyNbNoeudsGeneres)
        x.append(int(nMax * (i / 10)))

        if verbose : 
            print("\nx = ", i, "/10 nMax (" + str(int(nbIterations * i/10)) + ") : moyTemps =", moyTemps, "moyQualiteSolutions =", moyQualiteSolutions)
            print("----------------------------------------------------------------------------------------------\n")

    # Selection du nom de l'algorithme
    if (mode == 1) :
        nomAlgo = "algo_Couplage"
    elif (mode == 2) :
        nomAlgo = "algo_Glouton"
    elif (mode == 3) :
        nomAlgo = "branchement"
    elif (mode == 4) :
        nomAlgo = "branchement_Bornes_Couplage"
    elif (mode == 5) :
        nomAlgo = "branchement_Optimise_Couplage"
    elif (mode == 6) :
        nomAlgo = "branchement_Optimise_Couplage_uDegreMax"
    else :
        print("Aucun mode ne correspond à la valeur passée en paramètre. Veuillez choisir une autre valeur de mode.")
        return

    # Affichage graphique
    plt.figure(figsize = (10, 10))
    plt.suptitle("Performances de l'algorithme " + nomAlgo + " avec nMax =" + str(nMax) + " nodes dans le graphe et p = " + str(p) + "\n", color = 'black', size = 10)
    plt.rc('xtick', labelsize=10)    # fontsize of the tick labels

    # Construction et affichage du tracé "temps de calcul"
    plt.subplot(3, 1, 1)
    plt.title("Analyse du temps de calcul en fonction du nombre de sommets n")
    plt.xlabel("n") # nombre de sommets du graphe G
    plt.ylabel("t(n)") # temps de calcul en fonction du nombre de sommets du graphe G
    plt.plot(x, y1, color = 'blue')

    # Construction et affichage du tracé "qualité des solutions"
    plt.subplot(3, 1, 2)
    plt.title("Analyse de la qualité des solutions en fonction du nombre de sommets n")
    plt.xlabel("n") # nombre de sommets du graphe G
    plt.ylabel("q(n)") # qualité des solutions (taille du couplage) en fonction du nombre de sommets du graphe G
    plt.plot(x, y2, color = 'green')

    if (mode > 2) : # Construction et affichage du tracé "nombre de noeuds générés"
        plt.subplot(3, 1, 3)
        plt.title("Nombre de noeuds générés dans l'algorithme de branchement en fonction du nombre de sommets n")
        plt.xlabel("n") # nombre de sommets du graphe G
        plt.ylabel("c(n)") # nombre de noeuds crées durant le branchement en fonction du nombre de sommets du graphe G
        plt.plot(x, y3, color = 'red')

    # Sauvegarde du tracé
    if (save) :
        plt.savefig("TestResults/" + nomAlgo + "_p=" + str(p) + "_" + str(datetime.date.today()) + str(datetime.datetime.now().strftime("_%H_%M_%S")) + ".jpeg", transparent = True)

    plt.show()

#------------------------------------------------------------------------------------------------------

# Méthode permettant d'afficher le rapport d'approximation de algoCouplage et algoGlouton
def plotRapportApproximation(nMax, p, nbIterations, mode, verbose = False, save = False):
    """ nMax : nombre de noeuds maximale pour le graphe
        p : la probabilité qu'une arete entre 2 sommets soit crée, p E ]0,1[
        nbIterations : nombre d'éxecutions de l'algorithme, dans le but d'en déduir une performance moyenne
        mode : valeur déterminant l'algorithme allant etre utilisé, 1 = algoCouplage ; 2 = algoGlouton
        verbose : "True" pour afficher le détail des itérations
        save : "True" pour enregistrer le tracé en format jpg
    """
    y = []   # axe des ordonnées : rapport d'approximation des algorithmes couplage et glouton
    x = []   # axe des abscisses : liste de "nombre de sommets" {1/10 nbIterations, 2/10 nbIterations, ... , nbIterations}
    
    # Pour chaque 1/10 de nMax
    for i in range(1, 11) :

        tabRappApprox = []
        
        # Pour chacune des nbIterations démandées en paramètre
        for ite in range(nbIterations):
            r = -1
            res = -1

            # Méthode permettant de générer des graphes aléatoires
            G = randomGraphe(int(nMax * (i / 10)), p)

            # Calcul du rapport d'approximation r
            # mode : 1 = algoCouplage ; 2 = algoGlouton
            if (mode == 1) :
                res = len(algoCouplage(G))
            elif (mode == 2) :
                res = len(algoGlouton(G))
            else :
                print("Aucun mode ne correspond à la valeur passée en paramètre. Veuillez choisir une autre valeur de mode.")
                return

            opt = len(branchement(G))

            if opt != 0 :
                r = res/opt
                tabRappApprox.append(r)

            if verbose : 
                print("x = ", i, "/10 nMax, iteration n.", ite+1, ":", "\n\t\tRapport d'approximation :", r, "\n")

        if len(tabRappApprox) != 0 :
            moyR = sum(tabRappApprox)/len(tabRappApprox)
        else :
            moyR = -1
        
        y.append(moyR)
        x.append(int(nMax * (i / 10)))

        if verbose : 
            print("\nx = ", i, "/10 nMax (" + str(int(nbIterations * i/10)) + ")\n\t\tRapport d'approximation :", r, "\n")
            print("----------------------------------------------------------------------------------------------\n")


    # Affichage graphique
    plt.figure(figsize = (10, 10))
    plt.title("Rapport d'approximation des algorithmes algoCouplage et algoGlouton en f(n) avec nMax =" + str(nMax) + " nodes dans le graphe et p = " + str(p) + "\n", color = 'black', size = 15)
    plt.rc('xtick', labelsize=10)    # fontsize of the tick labels

    # Construction et affichage du tracé
    plt.xlabel("n") # nombre de sommets du graphe G
    plt.ylabel("r") # rapport d'approximation
    plt.axis([0, nMax, 0, r+1])
    plt.plot(x, y, color = 'blue')

    # Sauvegarde du tracé
    if (save) :
        plt.savefig("TestResults/rapportApproximation_p=" + str(p) + "_" + str(datetime.date.today()) + str(datetime.datetime.now().strftime("_%H_%M_%S")) + ".jpeg", transparent = True)

    plt.show()





#######################################################################################################
# METHODES PARTIE 2
#######################################################################################################

# Méthode permet de supprimer un sommet d'un graphe G et d'obtenir le graphe G' résultant de la suppression du sommet v
def suppSommet(initG, v) :
    """ initG : un dictionnaire representant un graphe { sommet s : sommets adjacents à s}
        v : un sommet à supprimer
    """
    if v not in initG.keys() :
        print("\nLe sommet", v, "n'est pas dans le graphe G : le graphe G'=G\{v} est équivalent à G.\n")
        return initG

    # On réalise une copie de initG pour ne pas le modifier
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

    # On retire de G' les clés contenant des listes vides
    cleanG = []
    for k in G.keys() :
        if G[k] == []:
            cleanG.append(k)

    for s in cleanG:
        del G[s]
        
    # On retourne G'
    return G

#------------------------------------------------------------------------------------------------------

# Méthode permettant de supprimer plusieurs sommets à la fois d'un graphe G et d'obtenir le graphe G' résultant de la suppression des sommets
def multSuppSommet(G, ensv) :
    """ G : un dictionnaire representant un graphe { sommet s : sommets adjacents à s}
        ensv : liste de sommets à supprimer
    """
    modifG = copy.deepcopy(G)

    for v in ensv :
        modifG = suppSommet(modifG, v)

    return modifG

#------------------------------------------------------------------------------------------------------

# Méthode renvoyant un tableau (dictionnaire) contenant les degres de chaque sommet du graphe G
def degresSommet(G) :
    """ G : un dictionnaire representant un graphe { sommet s : sommets adjacents à s}
    """
    # Création d'un dictionnaire 'tab' contenant les degres de chaque sommet du graphe G
    tab = dict()
    for v in G.keys() :
        tab[v] = len(list(G[v]))

    return tab

#------------------------------------------------------------------------------------------------------

# Méthode permettant de retourner le sommet ayant le degre maximal dans le graphe G
def sommetDegresMax(G) :
    """ G : un dictionnaire representant un graphe { sommet s : sommets adjacents à s}
    """
    deg = degresSommet(G) # deg est un dictionnaire { nbSommet : sommetsAdjacents }
    degres = list(deg.values())
    v = list(deg.keys())
    return v[degres.index(max(degres))]

#------------------------------------------------------------------------------------------------------

# Méthode permettant de retourner le degre du sommet ayant le degre maximum dans le graphe G
def valeurDegresMax(G) :
    """ G : un dictionnaire representant un graphe { sommet s : sommets adjacents à s}
    """
    deg = degresSommet(G) # deg est un dictionnaire { nbSommet : sommetsAdjacents }
    degres = list(deg.values())
    return max(degres)

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

    # Creation de la liste des sommets
    for i in range(n) :
        G[i] = []
    
    # Creation de la liste des aretes, ajoutées au graphe G suivant une probabilité p de présence
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
    """ G : un dictionnaire representant un graphe { sommet s : sommets adjacents à s}
    """
    C = list()  # C = liste de sommets représentant le couplage

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
    """ G : un dictionnaire representant un graphe { sommet s : sommets adjacents à s}
    """
    C = []  # C = liste de sommets représentant le couplage

    # On réalise une copie de G afin de ne pas modifier l'original
    copyG = copy.deepcopy(G)
    E = aretesGrapheToList(copyG) # Liste des arêtes du graphe G

    while E != [] :

        # On determine le sommet de degrès maximal et on le supprime du graphe
        v = sommetDegresMax(copyG)
        copyG = suppSommet(copyG, v)

        # On ajoute le sommet à la couverture
        C.append(v)

        # On supprime les arêtes couvertes par le sommet v
        E = [e for e in E if v not in e]

    return C



#######################################################################################################
# METHODES PARTIE 4
#######################################################################################################

# Méthode réalisant le branchement simple
def branchement(G, randomSelection=False, verbose=False) :
    """ G : un dictionnaire representant un graphe { sommet s : sommets adjacents à s}
        randomSelection : "True" si on veux dépiler les états de la pile en ordre aléatoire
        verbose : "True" pour afficher le détail des itérations
    """
    nbNoeudsGeneres = 1 # nombre de noeuds générés
    optiC = None # optiC = ensemble de sommets représentant la solution optimale (on cherche à minimiser la taille de la couverture)

    if G == {} :
        print("Le graphe est vide, C = {}")
        return -1

    # optiC = ensemble de sommets représentant la solution optimale (on cherche à minimiser la taille de la couverture)
    optiC = algoCouplage(G)

    # On vérifie qu'il existe des aretes dans le graphe. Si non, on retourne l'ensemble des sommets du graphe comme solution
    if (aretesGrapheToList(G) == []) :
        return [s for s in G.keys()], nbNoeudsGeneres
    
    #  On récupère la première arete du graphe
    areteInitiale = aretesGrapheToList(G)[0]

    # Un état est de la forme [ Couverture C actuelle, Dictionnaire de graphe G ]
    statesToStudy = list() # Pile des états du branchement à étudier

    if verbose:
        print("\nInitialisation algorithme 'branchement' (noeud racine) :")
        print("\t\tSolution optimale de la racine :", optiC)
        print("\t\tArete à traiter :", areteInitiale)
        print("\n")


    # Création des informations du noeud de droite
    newGraphe = copy.deepcopy(G)
    newGraphe = suppSommet(newGraphe, areteInitiale[1])
    statesToStudy.insert(0, [[areteInitiale[1]], newGraphe])
    nbNoeudsGeneres += 1

    if verbose :
        print("Ajout du noeud n.", nbNoeudsGeneres, "dans la branche droite (branchement v =", areteInitiale[1] ,") :")
        print("\t\tGraphe G\{" + str(areteInitiale[1]) +"} :", newGraphe)
        print("\t\tCouverture pour G\{" + str(areteInitiale[1]) +"} :", areteInitiale[1])
        print("\n")


    # Création des informations du noeud de gauche
    newGraphe = copy.deepcopy(G)
    newGraphe = suppSommet(newGraphe, areteInitiale[0])
    statesToStudy.insert(0, [[areteInitiale[0]], newGraphe])
    nbNoeudsGeneres += 1

    if verbose :
        print("Ajout du noeud n.", nbNoeudsGeneres, "dans la branche gauche (branchement u =", areteInitiale[0] ,") :")
        print("\t\tGraphe G\{" + str(areteInitiale[0]) +"} :", newGraphe)
        print("\t\tCouverture pour G\{" + str(areteInitiale[0]) +"} :", areteInitiale[0])
        print("\n")


    # Début de l'algorithme de branchement
    while (len(statesToStudy) != 0) :

        # Choix de la méthode de sélection d'état
        if (randomSelection) :
            # On récupère un état aléatoire de la liste d'états à étudier
            state = statesToStudy.pop(random.randint(0, len(statesToStudy) - 1))
        else :
            # On récupère la tete de la pile et on la supprime de statesToStudy
            state = statesToStudy.pop(0)

        # Cas où G (state[1]) est un graphe sans aretes
        if (aretesGrapheToList(state[1]) == []) :
            if (optiC == None) or (len(state[0]) < len(optiC)) :
                optiC = state[0]
                
            if verbose :    
                print("Meilleure couverture optimale :", optiC)


        # Cas où G (state[1]) n'est pas un graphe sans aretes
        else :
            # On récupère la première arete du graphe
            areteEtude = aretesGrapheToList(state[1])[0]

            # Calcul des informations du noeud de droite
            newGraphe = copy.deepcopy(state[1])
            newGraphe = suppSommet(newGraphe, areteEtude[1])
            statesToStudy.insert(0, [state[0] + [areteEtude[1]], newGraphe])
            nbNoeudsGeneres += 1

            if verbose :
                print("Ajout du noeud n.", nbNoeudsGeneres, "dans la branche droite (branchement v =", areteEtude[1] ,") :")
                print("\t\tGraphe G\{" + str(areteEtude[1]) +"} :", newGraphe)
                print("\t\tCouverture pour G\{" + str(areteEtude[1]) +"} :", state[0])
                print("\n")


            # Calcul des informations du noeud de gauche
            newGraphe = copy.deepcopy(state[1])
            newGraphe = suppSommet(newGraphe, areteEtude[0])

            statesToStudy.insert(0, [state[0] + [areteEtude[0]], newGraphe])
            nbNoeudsGeneres += 1

            if verbose :
                print("Ajout du noeud n.", nbNoeudsGeneres, "dans la branche gauche (branchement u =", areteEtude[0] ,") :")
                print("\t\tGraphe G\{" + str(areteEtude[0]) +"} :", newGraphe)
                print("\t\tCouverture pour G\{" + str(areteEtude[0]) +"} :", state[0])
                print("\n")

    if verbose :
        print("Nombre de noeuds générés avec la méthode 'branchement' :", nbNoeudsGeneres)
        print("Couverture optimale retournée par la méthode 'branchement' :", optiC)

    # On retourne la meilleure couverture trouvée et le nombre de noeuds générés
    return optiC, nbNoeudsGeneres

#------------------------------------------------------------------------------------------------------

# Méthode permettant de calculer le max parmi les bornes b1, b2 et b3
def calculBorneInf(G, verbose=False) :
    """ G : un dictionnaire representant un graphe { sommet s : sommets adjacents à s}
        verbose : "True" pour afficher le détail des itérations
    """
    l = []

    # Calcul de M, M est un couplage de G
    M = algoCouplage(G)

    # Calcul de n, m et c
    n = len(list(G.keys())) # n = nombre de sommets
    m = len(aretesGrapheToList(G)) # m = nombre d'aretes
    c = len(branchement(G, randomSelection=False)) # c = cardinalite de la couverture minimale de G donnée par l'algo branchement

    # Calcul de b1 = partie entière supérieure de m / delta (m = nombre aretes de G, delta = degrès maximum sommets du graphe)
    b1 = 0
    b1 = math.ceil(m / valeurDegresMax(G))
    l.append(b1)

    # Calcul de b2 = cardinalité de M (M un couplage de G)
    b2 = 0
    b2 = (len(M) / 2)
    l.append(b2)
    
    # Calcul de b3
    b3 = 0
    b3 = (2*n-1-(math.sqrt( ((2*n-1)**2)-8*m) ))/2
    l.append(b3)
    
    # Valeur maximale entre b1, b2, b3
    maxB = max(l)

    if (verbose) :
        print("CalculBorneInf : b1 =", b1, "; b2 =", b2, "; b3 =", b3, "; |C| =", c,"\t------> |C| >= max{b1,b2,b3}\t<===>\t", c, ">=", maxB, "\n")

    # On retourne la valeur maximale entre b1, b2, b3
    return maxB

#------------------------------------------------------------------------------------------------------

# Fonction réalisant le branchement avec bornes, qui associe à chaque noeud
# le calcul d'une borne inférieure et d'un borne superieure (Exercice 4.2.2)
# CONDITION POUR ELAGUER : (BORNE SUP < BORNE INF) ou (BORNE INF > TAILLE OPTI_C)
# CONDITIONS DE REUSSITE : (BORNE SUP >= BORNE INF) ou (BORNE INF <= TAILLE OPTI_C)
def branchementBornesCouplage(G, verbose=False) :
    """ G : un dictionnaire representant un graphe { sommet s : sommets adjacents à s}
        verbose : "True" pour afficher le détail des itérations
    """
    nbNoeudsGeneres = 1 # nombre de noeuds générés

    if G == {} :
        print("Le graphe est vide, C = {}")
        return -1

    # On vérifie qu'il existe des aretes dans le graphe. Si non, on retourne l'ensemble des sommets du graphe comme solution
    if (aretesGrapheToList(G) == []) :
        print("exception", [s for s in G.keys()], nbNoeudsGeneres)
        return [s for s in G.keys()], nbNoeudsGeneres

    # On calcule la borne inférieure et la borne supérieure pour la racine
    rootBorneInf = calculBorneInf(G)
    rootBorneSup = len(algoCouplage(G))

    # Dans le cas où les deux bornes sont égales, on retourne immédiatement la solution
    if (rootBorneInf >= rootBorneSup) :
        return algoCouplage(G), nbNoeudsGeneres

    # optiC = ensemble de sommets représentant la solution optimale (on cherche à minimiser la taille de la couverture)
    optiC = algoCouplage(G)

    #  On récupère la première arete du graphe
    areteInitiale = aretesGrapheToList(G)[0]

    # Un état est de la forme [ Couverture C actuelle, Dictionnaire de graphe G , Borne Inf , Borne Sup]
    statesToStudy = list() # Pile des états du branchement à étudier

    if verbose:
        print("\nInitialisation algorithme 'branchementBornesCouplage' (noeud racine) :")
        print("\t\tSolution optimale de la racine :", optiC)
        print("\t\tArete à traiter :", areteInitiale)
        print("\n")


    # Création des informations du noeud de droite
    newGraphe = copy.deepcopy(G)
    newGraphe = suppSommet(newGraphe, areteInitiale[1])
    
    if not (newGraphe == {}):
        newBorneInf = calculBorneInf(newGraphe) + 1
        newBorneSup = len(algoCouplage(newGraphe))
    else :
        newBorneInf = None
        newBorneSup = None

    if (newBorneInf != None and not(newBorneSup < newBorneInf or newBorneInf > len(optiC))) :
        statesToStudy.insert(0, [[areteInitiale[1]], newGraphe, newBorneInf, newBorneSup])
        nbNoeudsGeneres += 1

    if verbose :
        print("Ajout du noeud n.", nbNoeudsGeneres, "dans la branche droite (branchement v =", areteInitiale[1] ,") :")
        print("\t\tGraphe G\{" + str(areteInitiale[1]) +"} :", newGraphe)
        print("\t\tCouverture pour G\{" + str(areteInitiale[1]) +"} :", areteInitiale[1])
        print("\t\tBorne inferieure =", newBorneInf)
        print("\t\tBorne superieure =", newBorneSup)
        print("\n")


    # Création des informations du noeud de gauche
    newGraphe = copy.deepcopy(G)
    newGraphe = suppSommet(newGraphe, areteInitiale[0])

    if not (newGraphe == {}):
        newBorneInf = calculBorneInf(newGraphe) + 1
        newBorneSup = len(algoCouplage(newGraphe))
    else :
        newBorneInf = None
        newBorneSup = None

    if (newBorneInf != None and not(newBorneSup < newBorneInf or newBorneInf > len(optiC))) :
        statesToStudy.insert(0, [[areteInitiale[0]], newGraphe, newBorneInf, newBorneSup])
        nbNoeudsGeneres += 1

    if verbose :
        print("Ajout du noeud n.", nbNoeudsGeneres, "dans la branche gauche (branchement u =", areteInitiale[0] ,") :")
        print("\t\tGraphe G\{" + str(areteInitiale[0]) +"} :", newGraphe)
        print("\t\tCouverture pour G\{" + str(areteInitiale[0]) +"} :", areteInitiale[0])
        print("\t\tBorne inferieure =", newBorneInf)
        print("\t\tBorne superieure =", newBorneSup)
        print("\n")


    # Début de l'algorithme de branchement
    while (len(statesToStudy) != 0) :

        # On récupère la tete de la pile et on la supprime de statesToStudy
        state = statesToStudy.pop(0)

        # Cas où G (state[1]) est un graphe sans aretes
        if (aretesGrapheToList(state[1]) == []) :
            if (optiC == None) or (len(state[0]) < len(optiC)) :
                optiC = state[0]

            if verbose :    
                print("Meilleure couverture optimale :", optiC)


        # Cas où G (state[1]) n'est pas un graphe sans aretes
        else :

            # On récupère la première arete du branchement
            areteEtude = aretesGrapheToList(state[1])[0] # On récupère la première arete du graphe

            # Calcul des informations du noeud de droite
            newGraphe = copy.deepcopy(state[1])
            newGraphe = suppSommet(newGraphe, areteEtude[1])
 
            if not (newGraphe == {}):
                newBorneInf = calculBorneInf(newGraphe)
                newBorneSup = len(algoCouplage(newGraphe))
            else :
                newBorneInf = None
                newBorneSup = None

            if (newBorneInf != None and not(newBorneSup < newBorneInf or newBorneInf > len(optiC))) :
                statesToStudy.insert(0, [state[0] + [areteEtude[1]], newGraphe, newBorneInf, newBorneSup])
                nbNoeudsGeneres += 1

                if verbose :
                    print("Ajout du noeud n.", nbNoeudsGeneres, "dans la branche droite (branchement v =", areteEtude[1] ,") :")
                    print("\t\tGraphe G\{" + str(areteEtude[1]) +"} :", newGraphe)
                    print("\t\tCouverture pour G\{" + str(areteEtude[1]) +"} :", state[0])
                    print("\t\tBorne inferieure =", newBorneInf)
                    print("\t\tBorne superieure =", newBorneSup)
                    print("\n")


            # Calcul des informations du noeud de gauche
            newGraphe = copy.deepcopy(state[1])
            newGraphe = suppSommet(newGraphe, areteEtude[0])

            if not (newGraphe == {}):
                newBorneInf = calculBorneInf(newGraphe)
                newBorneSup = len(algoCouplage(newGraphe))
            else :
                newBorneInf = None
                newBorneSup = None

            if (newBorneInf != None and not(newBorneSup < newBorneInf or newBorneInf > len(optiC))) :
                statesToStudy.insert(0, [state[0] + [areteEtude[0]], newGraphe, newBorneInf, newBorneSup])
                nbNoeudsGeneres += 1

                if verbose :
                    print("Ajout du noeud n.", nbNoeudsGeneres, "dans la branche gauche (branchement u =", areteEtude[0] ,") :")
                    print("\t\tGraphe G\{" + str(areteEtude[0]) +"} :", newGraphe)
                    print("\t\tCouverture pour G\{" + str(areteEtude[0]) +"} :", state[0])
                    print("\t\tBorne inferieure =", newBorneInf)
                    print("\t\tBorne superieure =", newBorneSup)
                    print("\n")

    if verbose :
        print("Nombre de noeuds générés avec la méthode 'branchementBornesCouplage' :", nbNoeudsGeneres)
        print("Couverture optimale retournée par la méthode 'branchementBornesCouplage' :", optiC)

    # On retourne la meilleure couverture trouvée et le nombre de noeuds générés
    return optiC, nbNoeudsGeneres

#------------------------------------------------------------------------------------------------------

# Fonction réalisant le branchement avec bornes optimisé (Exercice 4.3.1)
# CONDITION POUR ELAGUER : (BORNE SUP < BORNE INF) ou (BORNE INF > TAILLE OPTI_C)
# CONDITIONS DE REUSSITE : (BORNE SUP >= BORNE INF) ou (BORNE INF <= TAILLE OPTI_C)
def branchementOptimiseCouplage(G, verbose=False) :
    """ G : un dictionnaire representant un graphe { sommet s : sommets adjacents à s}
        verbose : "True" pour afficher le détail des itérations
    """
    nbNoeudsGeneres = 1 # nombre de noeuds générés

    if G == {} :
        print("Le graphe est vide, C = {}")
        return -1

    # On vérifie qu'il existe des aretes dans le graphe. Si non, on retourne l'ensemble des sommets du graphe comme solution
    if (aretesGrapheToList(G) == []) :
        return [s for s in G.keys()], nbNoeudsGeneres

    # On calcule la borne inférieure et la borne supérieure pour la racine
    rootBorneInf = calculBorneInf(G)
    rootBorneSup = len(algoCouplage(G))

    # Dans le cas où les deux bornes sont égales, on retourne immédiatement la solution
    if (rootBorneInf >= rootBorneSup) :
        return algoCouplage(G), nbNoeudsGeneres

    # optiC = ensemble de sommets représentant la solution optimale (on cherche à minimiser la taille de la couverture)
    optiC = algoCouplage(G)

    #  On récupère la première arete du graphe
    areteInitiale = aretesGrapheToList(G)[0]

    # Un état est de la forme [ Couverture C actuelle, Dictionnaire de graphe G , Borne Inf , Borne Sup]
    statesToStudy = list() # Pile des états du branchement à étudier

    if verbose:
        print("\nInitialisation algorithme 'branchementOptimiseCouplage' (noeud racine) :")
        print("\t\tSolution optimale de la racine :", optiC)
        print("\t\tArete à traiter :", areteInitiale)
        print("\n")


    # Création des informations du noeud de droite
    newGraphe = copy.deepcopy(G)
    newGraphe = suppSommet(newGraphe, areteInitiale[1])

    if not newGraphe == {}:
        voisinsU = newGraphe[areteInitiale[0]]
        for s in voisinsU:
            if s in newGraphe:
                newGraphe = suppSommet(newGraphe, s)
    else :
        voisinsU = None

    if not (newGraphe == {}):
        newBorneInf = calculBorneInf(newGraphe) + 1
        newBorneSup = len(algoCouplage(newGraphe))
    else :
        newBorneInf = None
        newBorneSup = None

    if (newBorneInf != None and not(newBorneSup < newBorneInf or newBorneInf > len(optiC))) :
        if voisinsU != None:
            statesToStudy.insert(0, [[[areteInitiale[1]] + voisinsU], newGraphe, newBorneInf, newBorneSup])
        else:
            statesToStudy.insert(0, [[areteInitiale[1]], newGraphe, newBorneInf, newBorneSup])
        nbNoeudsGeneres += 1 

    if verbose :
        print("Ajout du noeud n.", nbNoeudsGeneres, "dans la branche droite (branchement v =", areteInitiale[1] ,") :")
        print("\t\tGraphe G\{" + str(areteInitiale[1]) +"} :", newGraphe)
        print("\t\tCouverture pour G\{" + str(areteInitiale[1]) +"} :", areteInitiale[1])
        print("\t\tBorne inferieure =", newBorneInf)
        print("\t\tBorne superieure =", newBorneSup)
        print("\n")


    # Création des informations du noeud de gauche
    newGraphe = copy.deepcopy(G)
    newGraphe = suppSommet(newGraphe, areteInitiale[0])

    if not (newGraphe == {}):
        newBorneInf = calculBorneInf(newGraphe) + 1
        newBorneSup = len(algoCouplage(newGraphe))
    else :
        newBorneInf = None
        newBorneSup = None

    if (newBorneInf != None and not(newBorneSup < newBorneInf or newBorneInf > len(optiC))) :
        statesToStudy.insert(0, [[areteInitiale[0]], newGraphe, newBorneInf, newBorneSup])
        nbNoeudsGeneres += 1

    if verbose :
        print("Ajout du noeud n.", nbNoeudsGeneres, "dans la branche gauche (branchement u =", areteInitiale[0] ,") :")
        print("\t\tGraphe G\{" + str(areteInitiale[0]) +"} :", newGraphe)
        print("\t\tCouverture pour G\{" + str(areteInitiale[0]) +"} :", areteInitiale[0])
        print("\t\tBorne inferieure =", newBorneInf)
        print("\t\tBorne superieure =", newBorneSup)
        print("\n")


    # Début de l'algorithme de branchement
    while (len(statesToStudy) != 0) :

        # On récupère la tete de la pile et on la supprime de statesToStudy
        state = statesToStudy.pop(0)

        # Cas où G (state[1]) est un graphe sans aretes
        if (aretesGrapheToList(state[1]) == []) :
            if (optiC == None) or (len(state[0]) < len(optiC)) :
                optiC = state[0]

            if verbose :    
                print("Meilleure couverture optimale :", optiC)


        # Cas où G (state[1]) n'est pas un graphe sans aretes
        else :

            # On récupère la première arete du branchement
            areteEtude = aretesGrapheToList(state[1])[0] # On récupère la première arete du graphe

            # Calcul des informations du noeud de droite
            newGraphe = copy.deepcopy(state[1])
            newGraphe = suppSommet(newGraphe, areteEtude[1])

            if not newGraphe == {}:
                voisinsU = newGraphe[areteEtude[0]]
                for s in voisinsU:
                    if s in newGraphe:
                        newGraphe = suppSommet(newGraphe, s)
            else :
                voisinsU = None

            if not (newGraphe == {}):
                newBorneInf = calculBorneInf(newGraphe)
                newBorneSup = len(algoCouplage(newGraphe))
            else :
                newBorneInf = None
                newBorneSup = None

            if (newBorneInf != None and not(newBorneSup < newBorneInf or newBorneInf > len(optiC))) :
                if voisinsU != None:
                    statesToStudy.insert(0, [[state[0] + [areteEtude[1]] + voisinsU], newGraphe, newBorneInf, newBorneSup])
                else:
                    statesToStudy.insert(0, [[state[0] + [areteEtude[1]]], newGraphe, newBorneInf, newBorneSup])
                nbNoeudsGeneres += 1    

                if verbose :
                    print("Ajout du noeud n.", nbNoeudsGeneres, "dans la branche droite (branchement v =", areteEtude[1] ,") :")
                    print("\t\tGraphe G\{" + str(areteEtude[1]) +"} :", newGraphe)
                    print("\t\tCouverture pour G\{" + str(areteEtude[1]) +"} :", state[0])
                    print("\t\tBorne inferieure =", newBorneInf)
                    print("\t\tBorne superieure =", newBorneSup)
                    print("\n")


            # Calcul des informations du noeud de gauche
            newGraphe = copy.deepcopy(state[1])
            newGraphe = suppSommet(newGraphe, areteEtude[0])

            if not (newGraphe == {}):
                newBorneInf = calculBorneInf(newGraphe)
                newBorneSup = len(algoCouplage(newGraphe))
            else :
                newBorneInf = None
                newBorneSup = None

            if (newBorneInf != None and not(newBorneSup < newBorneInf or newBorneInf > len(optiC))) :
                statesToStudy.insert(0, [state[0] + [areteEtude[0]], newGraphe, newBorneInf, newBorneSup])
                nbNoeudsGeneres += 1

                if verbose :
                    print("Ajout du noeud n.", nbNoeudsGeneres, "dans la branche gauche (branchement u =", areteEtude[0] ,") :")
                    print("\t\tGraphe G\{" + str(areteEtude[0]) +"} :", newGraphe)
                    print("\t\tCouverture pour G\{" + str(areteEtude[0]) +"} :", state[0])
                    print("\t\tBorne inferieure =", newBorneInf)
                    print("\t\tBorne superieure =", newBorneSup)
                    print("\n")

    if verbose :
        print("Nombre de noeuds générés avec la méthode 'branchementOptimiseCouplage' :", nbNoeudsGeneres)
        print("Couverture optimale retournée par la méthode 'branchementOptimiseCouplage' :", optiC)

    # On retourne la meilleure couverture trouvée et le nombre de noeuds générés
    return optiC, nbNoeudsGeneres

#------------------------------------------------------------------------------------------------------

# Fonction réalisant le branchement avec bornes optimisé, sans traiter u dans la branche droite (Exercice 4.3.2)
# CONDITION POUR ELAGUER : (BORNE SUP < BORNE INF) ou (BORNE INF > TAILLE OPTI_C)
# CONDITIONS DE REUSSITE : (BORNE SUP >= BORNE INF) ou (BORNE INF <= TAILLE OPTI_C)
def branchementOptimiseCouplage_uDegreMax(G, verbose=False) :
    """ G : un dictionnaire representant un graphe { sommet s : sommets adjacents à s}
        verbose : "True" pour afficher le détail des itérations
    """
    nbNoeudsGeneres = 1 # nombre de noeuds générés

    if G == {} :
        print("Le graphe est vide, C = {}")
        return -1

    # On vérifie qu'il existe des aretes dans le graphe. Si non, on retourne l'ensemble des sommets du graphe comme solution
    if (aretesGrapheToList(G) == []) :
        return [s for s in G.keys()], nbNoeudsGeneres

    # On calcule la borne inférieure et la borne supérieure pour la racine
    rootBorneInf = calculBorneInf(G)
    rootBorneSup = len(algoCouplage(G))

    # optiC = ensemble de sommets représentant la solution optimale (on cherche à minimiser la taille de la couverture)
    optiC = algoCouplage(G)

    # Dans le cas où les deux bornes sont égales, on retourne immédiatement la solution
    if (rootBorneInf >= rootBorneSup) :
        if verbose :
            print("Couverture optimale retournée par la méthode 'branchementOptimiseCouplage_uDegreMax' :", optiC)
        return optiC, nbNoeudsGeneres

    #  On récupère la premiere arete du graphe
    uDegreMax = sommetDegresMax(G)
    Gprime = {}
    Gprime[uDegreMax] = G[uDegreMax]
    areteInitiale = aretesGrapheToList(Gprime)[0]

    # Un état est de la forme [ Couverture C actuelle, Dictionnaire de graphe G , Borne Inf , Borne Sup]
    statesToStudy = list() # Pile des états du branchement à étudier

    if verbose:
        print("\nInitialisation algorithme 'branchementOptimiseCouplage_uDegreMax' (noeud racine) :")
        print("\t\tSolution optimale de la racine :", optiC)
        print("\t\tArete à traiter :", areteInitiale)
        print("\n")

    # Création des informations du noeud de droite
    newGraphe = copy.deepcopy(G)
    newGraphe = suppSommet(newGraphe, areteInitiale[1])

    if not newGraphe == {}:
        voisinsU = newGraphe[areteInitiale[0]]
        for s in voisinsU:
            if s in newGraphe:
                newGraphe = suppSommet(newGraphe, s)
    else :
        voisinsU = None

    if not (newGraphe == {}):
        newBorneInf = calculBorneInf(newGraphe) + 1
        newBorneSup = len(algoCouplage(newGraphe))
    else :
        newBorneInf = None
        newBorneSup = None

    if (newBorneInf != None and not(newBorneSup < newBorneInf or newBorneInf > len(optiC))) :
        if voisinsU != None:
            statesToStudy.insert(0, [[[areteInitiale[1]] + voisinsU], newGraphe, newBorneInf, newBorneSup])
        else:
            statesToStudy.insert(0, [[areteInitiale[1]], newGraphe, newBorneInf, newBorneSup])
        nbNoeudsGeneres += 1

        if verbose :
            print("Ajout du noeud n.", nbNoeudsGeneres, "dans la branche droite (branchement v =", areteInitiale[1] ,") :")
            print("\t\t[Debug] Liste des voisins du sommet u", voisinsU)
            print("\t\tGraphe G\{" + str(areteInitiale[1]) +"} :", newGraphe)
            print("\t\tCouverture pour G\{" + str(areteInitiale[1]) +"} :", areteInitiale[1])
            print("\t\tBorne inferieure =", newBorneInf)
            print("\t\tBorne superieure =", newBorneSup)
            print("\n")


    # Création des informations du noeud de gauche
    newGraphe = copy.deepcopy(G)
    newGraphe = suppSommet(newGraphe, areteInitiale[0])

    if not (newGraphe == {}):
        newBorneInf = calculBorneInf(newGraphe) + 1
        newBorneSup = len(algoCouplage(newGraphe))
    else :
        newBorneInf = None
        newBorneSup = None

    if (newBorneInf != None and not(newBorneSup < newBorneInf or newBorneInf > len(optiC))) :
        statesToStudy.insert(0, [[areteInitiale[0]], newGraphe, newBorneInf, newBorneSup])
        nbNoeudsGeneres += 1
        
        if verbose :
            print("Ajout du noeud n.", nbNoeudsGeneres, "dans la branche gauche (branchement u =", areteInitiale[0] ,") :")
            print("\t\tGraphe G\{" + str(areteInitiale[0]) +"} :", newGraphe)
            print("\t\tCouverture pour G\{" + str(areteInitiale[0]) +"} :", areteInitiale[0])
            print("\t\tBorne inferieure =", newBorneInf)
            print("\t\tBorne superieure =", newBorneSup)
            print("\n")


    # Début de l'algorithme de branchement
    while (len(statesToStudy) != 0) :

        # On récupère la tete de la pile et on la supprime de statesToStudy
        state = statesToStudy.pop(0)

        # Cas où G (state[1]) est un graphe sans aretes
        if (aretesGrapheToList(state[1]) == []) :
            if (optiC == None) or (len(state[0]) < len(optiC)) :
                optiC = state[0]
            
            if verbose :    
                print("Meilleure couverture optimale :", optiC)


        # Cas où G (state[1]) n'est pas un graphe sans aretes
        else :

            # On récupère la première arete du branchement
            uDegreMax = sommetDegresMax(state[1])
            Gprime = {}
            Gprime[uDegreMax] = (state[1])[uDegreMax]
            areteEtude = aretesGrapheToList(Gprime)[0] # On récupère la  arete du graphe avec max d

            # Calcul des informations du noeud de droite
            newGraphe = copy.deepcopy(state[1])
            newGraphe = suppSommet(newGraphe, areteEtude[1])

            if not newGraphe == {}:
                voisinsU = newGraphe[areteEtude[0]]
                for s in voisinsU:
                    if s in newGraphe:
                        newGraphe = suppSommet(newGraphe, s)
            else :
                voisinsU = None
 
            if not (newGraphe == {}):
                newBorneInf = calculBorneInf(newGraphe) + 1
                newBorneSup = len(algoCouplage(newGraphe))
            else :
                newBorneInf = None
                newBorneSup = None

            if (newBorneInf != None and not(newBorneSup < newBorneInf or newBorneInf > len(optiC))) :
                if voisinsU != None:
                    statesToStudy.insert(0, [[state[0] + [areteEtude[1]] + voisinsU], newGraphe, newBorneInf, newBorneSup])
                else:
                    statesToStudy.insert(0, [[state[0] + [areteEtude[1]]], newGraphe, newBorneInf, newBorneSup])
                nbNoeudsGeneres += 1

                if verbose :
                    print("Ajout du noeud n.", nbNoeudsGeneres, "dans la branche droite (branchement v =", areteEtude[1] ,") :")
                    print("\t\tGraphe G\{" + str(areteEtude[1]) +"} :", newGraphe)
                    print("\t\tCouverture pour G\{" + str(areteEtude[1]) +"} :", state[0])
                    print("\t\tBorne inferieure =", newBorneInf)
                    print("\t\tBorne superieure =", newBorneSup)
                    print("\n")

            # Calcul des informations du noeud de gauche
            newGraphe = copy.deepcopy(state[1])
            newGraphe = suppSommet(newGraphe, areteEtude[0])

            if not (newGraphe == {}):
                newBorneInf = calculBorneInf(newGraphe)
                newBorneSup = len(algoCouplage(newGraphe))
            else :
                newBorneInf = None
                newBorneSup = None

            if (newBorneInf != None and not(newBorneSup < newBorneInf or newBorneInf > len(optiC))) :
                statesToStudy.insert(0, [state[0] + [areteEtude[0]], newGraphe, newBorneInf, newBorneSup])
                nbNoeudsGeneres += 1

                if verbose :
                    print("Ajout du noeud n.", nbNoeudsGeneres, "dans la branche gauche (branchement u =", areteEtude[0] ,") :")
                    print("\t\tGraphe G\{" + str(areteEtude[0]) +"} :", newGraphe)
                    print("\t\tCouverture pour G\{" + str(areteEtude[0]) +"} :", state[0])
                    print("\t\tBorne inferieure =", newBorneInf)
                    print("\t\tBorne superieure =", newBorneSup)
                    print("\n")

    if verbose :
        print("Nombre de noeuds générés avec la méthode 'branchementOptimiseCouplage_uDegreMax' :", nbNoeudsGeneres)
        print("Couverture optimale retournée par la méthode 'branchementOptimiseCouplage_uDegreMax' :", optiC)

    # On retourne la meilleure couverture trouvée et le nombre de noeuds générés
    return optiC, nbNoeudsGeneres





#######################################################################################################
# EVALUATIONS
#######################################################################################################

# Dans cette partie, on s'occupe de l'évaluation des différents algorithmes
# Ces résultats seront présentés dans le rapport dans les parties désignées

# Méthode d'évaluation permettant de réaliser les tests
def evaluationAlgorithm(n, p, a) :
    """ n : nombre de sommets, n > 0
        p : la probabilité qu'une arete entre 2 sommet soit créée, p E ]0,1[
        a : valeur qui permet de choisir l'algorithme à utiliser :
                1 = Test de branchement (4.1)
                2 = Test de branchementBornesCouplage (4.2.2)
    """
    # a = 1 - Test de branchement (4.1)
    if (a == 1) :
        print("EVALUATION - Algorithme : branchement (4.1).\nDébut de l'évaluation des performances pour :\nn =", n, "\tp =", p)
        testGraphe = randomGraphe(n,p)
        startTime = time.time()
        solution = branchement(testGraphe, verbose=True)
        endTime = time.time()
        execTime = endTime - startTime
        print("Temps d'exécution =", execTime, "secondes.\n")

    # a = 2 - Test de branchementBornesCouplage (4.2.2)
    elif (a == 2) :
        print("EVALUATION - Algorithme : branchementBornesCouplage (4.2.2).\nDébut de l'évaluation des performances pour :\nn =", n, "\tp =", p)
        testGraphe = randomGraphe(n,p)
        startTime = time.time()
        solution = branchementBornesCouplage(testGraphe, verbose=True)
        endTime = time.time()
        execTime = endTime - startTime
        print("Temps d'exécution =", execTime, "secondes.\n")

    # La valeur de a ne correspond à aucun algorithme
    else :
        print("EVALUATION - Aucun algorithme correspondant.\nVeuillez choisir une valeur de a différente.")





#######################################################################################################
# TESTS
#######################################################################################################

# Instanciation d'un graphe G (modelisation : dictionnaire)
# G = {0 : [1, 2, 3], 1 : [0, 2], 2 : [0, 1], 3 : [0]}
# showGraphe(G)

# Instanciation d'un graphe G (modelisation : librairie graphe networkx)
# V = [0, 1, 2, 3]
# E = [(0,1), (0,2), (0,3), (1,2)]

# showGraphe(G)
# print("\n----------------------------------------------------------------------------------------\n")

#------------------------------------------------------------------------------------------------------

# Test méthode suppSommet
# print("Graphe G\n", G, "\n")
# newG = suppSommet(G, 0)
# print("Graphe G'\n", newG, "\n")
# print("\n----------------------------------------------------------------------------------------\n")

#------------------------------------------------------------------------------------------------------

# Test méthode multSuppSommet
# newG = multSuppSommet(G, [0, 1])
# print("Graphe G'\n", newG, "\n")
# print("\n----------------------------------------------------------------------------------------\n")

#------------------------------------------------------------------------------------------------------

# Tests des méthodes degresSommet et sommetDegresSommet
# print(degresSommet(G))
# print(sommetDegresMax(G))
# print("\n----------------------------------------------------------------------------------------\n")

#------------------------------------------------------------------------------------------------------

# Tests sur la génération aléatoire de graphe
# G = randomGraphe(8, 0.1)
# print("Graphe G = ", G, "\n")
# showGraphe(G)
# print("\n----------------------------------------------------------------------------------------\n")

#------------------------------------------------------------------------------------------------------

# Tests sur l'algorithme de couplage
# G = randomGraphe(8, 0.2)
# print(algoCouplage(G))
# print(aretesGrapheToList(G))
# showGraphe(G)
# print("\n----------------------------------------------------------------------------------------\n")

#------------------------------------------------------------------------------------------------------

# Tests sur l'algorithme de couplage glouton
# G = randomGraphe(20, 0.5)
# print(algoGlouton(G))
# showGraphe(G)
# print("\n----------------------------------------------------------------------------------------\n")

#------------------------------------------------------------------------------------------------------

# Test méthode acquisitionGraphe depuis un fichier texte
# G = acquisitionGraphe("exempleinstance.txt")
# print("G = ", G, "\n")
# showGraphe(G)
# print("\n----------------------------------------------------------------------------------------\n")

#------------------------------------------------------------------------------------------------------

# Test méthodes plotPerformances sur Couplage et Glouton
# plotPerformances(0.3, 15, 0.01, 1, verbose=True, save=True)
# plotPerformances(0.3, 15, 0.01, 2, verbose=True, save=True)
# print("\n----------------------------------------------------------------------------------------\n")

#------------------------------------------------------------------------------------------------------

# Test méthode plotPerformances sur l'algorithme de branchement simple
# plotPerformances(0.2, 15, 2.5, 3, verbose=True, save=True)
# plotPerformances(0.5, 15, 2.5, 3, verbose=True, save=True)
# plotPerformances(0.9, 15, 2.5, 3, verbose=True, save=True)
# plotPerformances(0.22, 15, 2.5, 3, verbose=True, save=True)
# print("\n----------------------------------------------------------------------------------------\n")

#------------------------------------------------------------------------------------------------------

# Test méthode plotPerformances sur les algorithmes de branchement
# plotPerformances(0.3, 15, 2.5, 4, verbose=True, save=True)
# plotPerformances(0.5, 15, 2.5, 4, verbose=True, save=True)
# plotPerformances(0.7, 15, 2.5, 4, verbose=True, save=True)
# print("\n----------------------------------------------------------------------------------------\n")

#------------------------------------------------------------------------------------------------------

# Test méthode plotPerformances sur l
# plotPerformances(0.3, 15, 2.5, 5, verbose=True, save=True)
# plotPerformances(0.5, 15, 2.5, 5, verbose=True, save=True)
# plotPerformances(0.7, 15, 2.5, 5, verbose=True, save=True)
# print("\n----------------------------------------------------------------------------------------\n")

#------------------------------------------------------------------------------------------------------

# Test sur la méthode de branchement simple
# G = acquisitionGraphe("exempleinstance.txt")
# print(branchement(G, verbose = True))
# showGraphe(G)
# print("\n----------------------------------------------------------------------------------------\n")

#------------------------------------------------------------------------------------------------------
# Test sur la méthode de branchement branchementBornesCouplage
# G = acquisitionGraphe("exempleinstance.txt")
# print(branchementBornesCouplage(G, verbose = True))
# showGraphe(G)
# print("\n----------------------------------------------------------------------------------------\n")

#------------------------------------------------------------------------------------------------------

# Test sur la méthode de branchement branchementOptimiseCouplage
# G = acquisitionGraphe("exempleinstance.txt")
# print(branchementOptimiseCouplage(G, verbose = True))
# showGraphe(G)
# print("\n----------------------------------------------------------------------------------------\n")

#------------------------------------------------------------------------------------------------------

# Test sur la méthode de branchement branchementOptimiseCouplage_uDegreMax
# G = acquisitionGraphe("exempleinstance.txt")
# print(branchementOptimiseCouplage_uDegreMax(G, verbose = True))
# showGraphe(G)
# print("\n----------------------------------------------------------------------------------------\n")

#------------------------------------------------------------------------------------------------------

# Evalutation de branchement (question 4.1)
# n = 20 # Il est recommandé de choisir une valeur de n divisible par d pour faciliter les calculs
# d = 20 # Facteur de division de la valeur de n (plus d est elevé, plus le nombre de tests est élevé)
# for i in range(d) :
#     numberOfNodes = (int)(n * ((i + 1) / d))
#     evaluationAlgorithm(numberOfNodes, 0.2, 1)
# print("\n----------------------------------------------------------------------------------------\n")

#------------------------------------------------------------------------------------------------------

# Evalutation du rapport d'approximation (question 4.4.1)
plotRapportApproximation(100, 0.5, 10, 1, verbose = False, save = False)
# print("\n----------------------------------------------------------------------------------------\n")