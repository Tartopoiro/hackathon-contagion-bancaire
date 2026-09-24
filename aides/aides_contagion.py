"""
Aides Python du hackathon « Réseaux bancaires et contagion » (Université de Bordeaux).

Ce module ne contient NI simulateur NI construction de graphe : ces deux fonctions
sont à écrire par votre équipe. Les aides reçoivent VOS fonctions en argument et
se chargent des calculs répétitifs : table des scénarios, références médianes,
budget d'intervention, tirages aléatoires reproductibles, comparaison de stratégies.

Conventions (section 2 du sujet)
--------------------------------
* Le graphe est un ``nx.DiGraph`` : un sommet par banque, avec l'attribut
  ``"capital"`` ; une arête ``i -> j`` d'attribut ``"montant"`` signifie
  « la banque i doit ``montant`` à la banque j ». Si i fait défaut, c'est j qui perd.
* Les deux fonctions de votre équipe :

  - ``construire_graphe(banques_r, expositions_r) -> nx.DiGraph`` pour UN réseau,
    banques isolées comprises ;
  - ``simuler(G, origine, lam) -> (defauts, nouveaux_par_etape)`` où ``defauts``
    est un ensemble (``set``) qui contient l'origine et ``nouveaux_par_etape``
    la liste des nombres de nouveaux défauts à chaque étape (sans l'origine,
    sans zéro final). Exemple A-B-C : ``({"A", "B", "C"}, [1, 1])``.

Importer les aides depuis un notebook
-------------------------------------
>>> import sys
>>> sys.path.append("aides")      # dossier qui contient ce fichier (à adapter)
>>> import aides_contagion as ac
>>> ac.depasse_capital(0.5 * 10, 5.0)
False

Contenu
-------
LAMBDAS, TOL_REL                                 constantes communes à toutes les équipes
depasse_capital(perte, capital)                  « perte > capital », robuste aux arrondis
proportion_secondaire(defauts, n)                y = (|D| - 1) / (N - 1)
generer_scenarios(...)                           table réseau x origine x lambda
references_medianes(apprentissage)               médiane globale et médianes par lambda
predire_references(refs, table)                  prédictions des deux références
capitaux_renforces(G, choisies)                  copie de G renforcée avec le budget B
tirer_banques_aleatoires(ids, graine, numero)    tirage reproductible de k banques
charger_tirages(chemin)                          lecture de tirages_aleatoires.csv
comparer_interventions(G, simuler, strategies)   évaluation de stratégies de renforcement
resumer_interventions(resultats)                 synthèse par stratégie (moyennes, gains)

Compatibilité : Python >= 3.10 ; testé avec pandas 2.1 et 3.0, NumPy 1.26 et 2.5,
NetworkX 3.2 et 3.7.
"""

from __future__ import annotations

import copy
import numbers
import re
import sys
import time
import warnings

import networkx as nx
import numpy as np
import pandas as pd

__all__ = [
    "LAMBDAS",
    "TOL_REL",
    "depasse_capital",
    "proportion_secondaire",
    "generer_scenarios",
    "references_medianes",
    "predire_references",
    "capitaux_renforces",
    "tirer_banques_aleatoires",
    "charger_tirages",
    "comparer_interventions",
    "resumer_interventions",
]

# ---------------------------------------------------------------------------
# Constantes communes
# ---------------------------------------------------------------------------

#: Fractions de perte utilisées par toutes les équipes.
LAMBDAS = (0.3, 0.6, 0.9)

#: Tolérance relative de ``depasse_capital`` (voir SPECIFICATION, section 5).
TOL_REL = 1e-9

#: Colonnes de la table produite par ``generer_scenarios``, dans cet ordre.
COLONNES_SCENARIOS = (
    "reseau_id",
    "usage",
    "origine",
    "lambda",
    "n_defauts",
    "y",
    "n_etapes",
    "nouveaux_par_etape",
    "defauts_finaux",
)

#: Colonnes de la table produite par ``comparer_interventions``, dans cet ordre.
COLONNES_INTERVENTIONS = (
    "strategie",
    "repetition",
    "banques_renforcees",
    "origine",
    "lambda",
    "n_defauts",
    "y",
)

_DECIMALES_LAMBDA = 10  # arrondi des clés lambda (0.1 + 0.2 et 0.3 donnent la même clé)


class ResultatSimulationInvalide(ValueError):
    """Le résultat renvoyé par ``simuler`` ne respecte pas le contrat du sujet."""


class GrapheInvalide(ValueError):
    """Le graphe renvoyé par ``construire_graphe`` ne respecte pas le contrat du sujet."""


# ---------------------------------------------------------------------------
# Règle de défaut et mesure principale
# ---------------------------------------------------------------------------

def depasse_capital(perte, capital, tol_rel=TOL_REL):
    """Indique si une perte dépasse STRICTEMENT le capital, sans se laisser piéger par les arrondis.

    Une banque fait défaut lorsque sa perte dépasse strictement son capital ; une perte
    exactement égale au capital ne déclenche pas de défaut. En nombres flottants, un
    calcul comme ``0.3 * 10.3`` donne ``3.0900000000000003`` au lieu de ``3.09`` : la
    comparaison naïve ``perte > capital`` conclurait à tort à un défaut. On compare donc
    avec une petite marge relative :

        perte > capital + tol_rel * max(1, |capital|)

    Paramètres
    ----------
    perte, capital : float (ou tableaux NumPy de même forme)
    tol_rel : float, tolérance relative (1e-9 par défaut, commune à toutes les équipes)

    Renvoie
    -------
    bool (ou tableau de booléens si l'on passe des tableaux)

    Exemples
    --------
    >>> depasse_capital(0.5 * 10, 4.0)       # 5 > 4 : défaut
    True
    >>> depasse_capital(0.5 * 10, 5.0)       # perte égale au capital : pas de défaut
    False
    >>> 0.3 * 10.3 > 3.09                     # piège des flottants
    True
    >>> depasse_capital(0.3 * 10.3, 3.09)
    False
    """
    if np.ndim(perte) == 0 and np.ndim(capital) == 0:
        return bool(perte > capital + tol_rel * max(1.0, abs(capital)))
    perte = np.asarray(perte, dtype=float)
    capital = np.asarray(capital, dtype=float)
    return perte > capital + tol_rel * np.maximum(1.0, np.abs(capital))


def proportion_secondaire(defauts, n):
    """Proportion de défauts secondaires y = (|D_final| - 1) / (N - 1).

    Paramètres
    ----------
    defauts : ensemble des banques en défaut (origine comprise), ou directement
        leur nombre |D_final| (entier)
    n : nombre N de banques du réseau, banques isolées comprises

    Exemples
    --------
    >>> proportion_secondaire({"A", "B", "C"}, 3)    # exemple A-B-C : tout le réseau tombe
    1.0
    >>> proportion_secondaire({"B07"}, 60)           # le choc reste limité à l'origine
    0.0
    >>> proportion_secondaire(16, 60)                # 15 défauts secondaires sur 59 possibles
    0.2542372881355932
    """
    if isinstance(defauts, numbers.Integral) and not isinstance(defauts, bool):
        nb = int(defauts)
    else:
        nb = len(defauts)
    n = int(n)
    if n < 2:
        raise ValueError(f"N = {n} : il faut au moins deux banques pour définir la proportion y.")
    if not 1 <= nb <= n:
        raise ValueError(
            f"{nb} défaut(s) pour N = {n} banques : le nombre de défauts doit être compris entre "
            "1 (l'origine, dont le défaut est imposé) et N."
        )
    return (nb - 1) / (n - 1)


# ---------------------------------------------------------------------------
# Contrôles internes (messages d'erreur pédagogiques)
# ---------------------------------------------------------------------------

def _decrire(objet, longueur=80):
    """Description courte d'un objet pour les messages d'erreur."""
    texte = repr(objet)
    if len(texte) > longueur:
        texte = texte[: longueur - 3] + "..."
    return f"{type(objet).__name__} {texte}"


def _apercu(elements, nb=5):
    """Quelques éléments d'une collection, pour les messages d'erreur."""
    elements = list(elements)
    texte = ", ".join(repr(e) for e in elements[:nb])
    return texte + (", ..." if len(elements) > nb else "")


def _verifier_colonnes(table, colonnes, nom):
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"« {nom} » doit être un DataFrame pandas ; reçu : {_decrire(table)}.")
    manquantes = [c for c in colonnes if c not in table.columns]
    if manquantes:
        raise ValueError(
            f"Colonne(s) absente(s) de « {nom} » : {', '.join(manquantes)}. "
            f"Colonnes présentes : {', '.join(map(str, table.columns))}."
        )


def _normaliser_lambdas(lambdas):
    """Tuple trié de fractions de perte distinctes, toutes dans [0, 1]."""
    if np.ndim(lambdas) == 0:
        lambdas = [lambdas]
    valeurs = []
    for lam in lambdas:
        try:
            lam = float(lam)
        except (TypeError, ValueError):
            raise TypeError(
                f"Fraction de perte invalide : {_decrire(lam)} (un nombre entre 0 et 1 est attendu)."
            ) from None
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"Fraction de perte λ = {lam} hors de [0, 1].")
        valeurs.append(lam)
    if not valeurs:
        raise ValueError("La liste des fractions de perte (lambdas) est vide.")
    if len({round(v, _DECIMALES_LAMBDA) for v in valeurs}) != len(valeurs):
        raise ValueError(f"Fractions de perte en double dans {valeurs}.")
    return tuple(sorted(valeurs))


def _cle_lambda(lam):
    """Clé de dictionnaire robuste aux flottants : 0.30000000000000004 -> 0.3."""
    return round(float(lam), _DECIMALES_LAMBDA)


def _controler_graphe(G, ids_attendus=None, reseau_id=None):
    """Vérifie que G respecte le contrat (orienté, attributs, sommets attendus)."""
    ou = f" du réseau {reseau_id}" if reseau_id is not None else ""
    if G is None:
        raise GrapheInvalide(
            f"construire_graphe a renvoyé None{ou} : avez-vous oublié « return G » à la fin de la fonction ?"
        )
    if not isinstance(G, nx.Graph):
        raise GrapheInvalide(f"Un graphe NetworkX (nx.DiGraph) est attendu{ou} ; reçu : {_decrire(G)}.")
    if not G.is_directed():
        raise GrapheInvalide(
            f"Le graphe{ou} n'est pas orienté : utilisez nx.DiGraph() et non nx.Graph(). "
            "Le sens compte : i -> j signifie « i doit à j »."
        )
    if G.is_multigraph():
        raise GrapheInvalide(
            f"Le graphe{ou} est un multigraphe : utilisez nx.DiGraph() (une seule arête par paire orientée)."
        )
    if ids_attendus is not None:
        attendus = set(ids_attendus)
        n_attendu = len(attendus)
        n_obtenu = G.number_of_nodes()
        absents = sorted(attendus - set(G.nodes), key=str)
        en_trop = sorted(set(G.nodes) - attendus, key=str)
        if n_obtenu < n_attendu or absents:
            raise GrapheInvalide(
                f"Le graphe{ou} a {n_obtenu} sommets alors que « banques » compte {n_attendu} banques : "
                f"banques isolées oubliées ? Absentes du graphe : {_apercu(absents)}. "
                "Ajoutez d'abord tous les sommets depuis banques_r (G.add_node(banque_id, capital=...)), "
                "puis les arêtes : une banque sans dette ni créance compte quand même dans N."
            )
        if en_trop:
            raise GrapheInvalide(
                f"Le graphe{ou} contient des sommets absents de « banques » : {_apercu(en_trop)}. "
                "Les identifiants des expositions correspondent-ils à ceux des banques "
                "(mêmes colonnes, pas d'espace parasite) ?"
            )
    sans_capital = [n for n, d in G.nodes(data=True) if "capital" not in d]
    if sans_capital:
        raise GrapheInvalide(
            f"Sommet(s) sans attribut 'capital'{ou} : {_apercu(sans_capital)}. "
            "Utilisez G.add_node(banque_id, capital=float(capital))."
        )
    sans_montant = [(i, j) for i, j, d in G.edges(data=True) if "montant" not in d]
    if sans_montant:
        raise GrapheInvalide(
            f"Arête(s) sans attribut 'montant'{ou} : {_apercu(sans_montant)}. "
            "Utilisez G.add_edge(debiteur_id, creancier_id, montant=float(montant))."
        )


def _empreinte_graphe(G):
    """Capitaux et montants de G : permet de détecter un simulateur qui modifie le graphe."""
    return (
        tuple((n, d.get("capital")) for n, d in G.nodes(data=True)),
        tuple((i, j, d.get("montant")) for i, j, d in G.edges(data=True)),
    )


def _valider_resultat_simulation(resultat, G, origine, lam):
    """Contrôle le couple renvoyé par ``simuler`` et renvoie (set, list[int]).

    Lève ``ResultatSimulationInvalide`` avec un message qui explique comment corriger
    (sans rappeler l'appel : c'est l'appelant qui ajoute le contexte).
    """
    if not isinstance(resultat, tuple) or len(resultat) != 2:
        raise ResultatSimulationInvalide(
            "simuler doit renvoyer un couple (defauts, nouveaux_par_etape), par exemple "
            f"« return defauts, nouveaux_par_etape » ; reçu : {_decrire(resultat)}."
        )
    defauts, etapes = resultat
    if not isinstance(defauts, (set, frozenset)):
        raise ResultatSimulationInvalide(
            "le premier élément renvoyé doit être un ensemble (set) de banques, par exemple "
            f"{{'A', 'B'}} ; reçu : {_decrire(defauts)}. Convertissez avec set(...)."
        )
    if not isinstance(etapes, list):
        raise ResultatSimulationInvalide(
            "le second élément renvoyé doit être une liste d'entiers (nouveaux défauts par étape), "
            f"par exemple [1, 1] ; reçu : {_decrire(etapes)}."
        )
    for nb in etapes:
        if isinstance(nb, bool) or not isinstance(nb, numbers.Integral):
            raise ResultatSimulationInvalide(
                "nouveaux_par_etape doit contenir des NOMBRES de défauts (entiers), par exemple "
                f"[4, 1, 6] ; reçu : {_decrire(etapes)}. Ajoutez len(nouveaux), et non l'ensemble lui-même."
            )
        if nb <= 0:
            raise ResultatSimulationInvalide(
                f"nouveaux_par_etape = {etapes} contient {nb}. La liste ne compte que les étapes avec au "
                "moins un nouveau défaut : n'ajoutez pas l'étape finale sans nouveau défaut (pas de zéro final)."
            )
    if origine not in defauts:
        raise ResultatSimulationInvalide(
            f"l'origine {origine!r} ne fait pas partie des défauts renvoyés. Son défaut est imposé par le "
            "choc initial, quel que soit son capital : commencez avec defauts = {origine}."
        )
    inconnues = [b for b in defauts if b not in G]
    if inconnues:
        raise ResultatSimulationInvalide(f"banque(s) en défaut absente(s) du graphe : {_apercu(inconnues)}.")
    if sum(etapes) != len(defauts) - 1:
        raise ResultatSimulationInvalide(
            f"la somme des nouveaux défauts par étape ({sum(etapes)}) devrait valoir |defauts| - 1 = "
            f"{len(defauts) - 1} (l'origine n'est pas comptée dans nouveaux_par_etape)."
        )
    return set(defauts), [int(nb) for nb in etapes]


_MESSAGE_GRAPHE_MODIFIE = (
    "a modifié le graphe G (capital, montant ou arêtes). Les appels suivants seraient faux : ne "
    "modifiez jamais G dans simuler, travaillez sur des variables locales (ensembles, dictionnaires)."
)


def _signaler_contexte(erreur, message):
    """Ajoute le contexte (réseau, origine, λ) à une exception levée par le code de l'équipe."""
    if hasattr(erreur, "add_note"):          # Python >= 3.11
        erreur.add_note(message)
    else:
        print(message, file=sys.stderr)


def _simuler_et_valider(simuler, G, origine, lam, empreinte=None, contexte=""):
    """Appelle simuler, valide le résultat et vérifie que G n'a pas été modifié.

    ``contexte`` (par exemple « Réseau R03, ») est ajouté au début des messages d'erreur.
    """
    appel = f"{contexte}simuler(G, {origine!r}, {lam:g})"
    try:
        resultat = simuler(G, origine, lam)
    except Exception as erreur:
        _signaler_contexte(erreur, f"Erreur levée par votre fonction pendant {appel}.")
        raise
    try:
        defauts, etapes = _valider_resultat_simulation(resultat, G, origine, lam)
    except ResultatSimulationInvalide as erreur:
        raise ResultatSimulationInvalide(f"{appel} : {erreur}") from None
    if empreinte is not None and _empreinte_graphe(G) != empreinte:
        raise ResultatSimulationInvalide(f"{appel} {_MESSAGE_GRAPHE_MODIFIE}")
    return defauts, etapes


def _en_liste_de_banques(selection, nom="choisies"):
    """Convertit une sélection de banques en liste (None -> liste vide)."""
    if selection is None:
        return []
    if isinstance(selection, str):
        raise TypeError(
            f"« {nom} » doit être une LISTE de banques, par exemple [{selection!r}] "
            f"(et non la chaîne {selection!r})."
        )
    if isinstance(selection, (pd.Series, pd.Index, np.ndarray)):
        return list(selection.tolist())
    if isinstance(selection, (set, frozenset)):
        return sorted(selection, key=str)
    try:
        return list(selection)
    except TypeError:
        raise TypeError(f"« {nom} » doit être une liste de banques ; reçu : {_decrire(selection)}.") from None


class _Progression:
    """Barre de progression textuelle minimale (sans dépendance)."""

    def __init__(self, total, actif=True, largeur=28, unite="origines"):
        self.total = max(int(total), 1)
        self.actif = bool(actif)
        self.largeur = largeur
        self.unite = unite
        self.fait = 0
        self.debut = time.perf_counter()
        self._dernier = -1.0

    def avancer(self, n=1, etiquette=""):
        self.fait += n
        if not self.actif:
            return
        maintenant = time.perf_counter()
        if self.fait < self.total and maintenant - self._dernier < 0.2:
            return
        self._dernier = maintenant
        part = min(self.fait / self.total, 1.0)
        plein = int(round(part * self.largeur))
        ecoule = maintenant - self.debut
        reste = ecoule * (1 - part) / part if part > 0 else 0.0
        sys.stdout.write(
            f"\r[{'#' * plein}{'-' * (self.largeur - plein)}] {100 * part:5.1f} %  "
            f"{self.fait}/{self.total} {self.unite}  {etiquette:<8} "
            f"{ecoule:5.0f} s écoulées, ~{reste:.0f} s restantes   "
        )
        sys.stdout.flush()

    def terminer(self, message=""):
        if self.actif:
            sys.stdout.write("\n" + (message + "\n" if message else ""))
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# Table des scénarios
# ---------------------------------------------------------------------------

def generer_scenarios(banques, expositions, construire_graphe, simuler, lambdas=LAMBDAS,
                      repartition=None, reseaux=None, progression=True):
    """Simule tous les scénarios (réseau, banque d'origine, λ) avec VOS fonctions.

    Pour chaque réseau, le graphe est construit une fois avec ``construire_graphe`` ;
    chaque banque (identifiants triés) sert ensuite d'origine pour chaque valeur de λ.
    Avec 12 réseaux de 60 banques et 3 valeurs de λ : 2 160 scénarios.

    Paramètres
    ----------
    banques, expositions : DataFrames lus depuis banques.csv et expositions.csv
        (tous les réseaux ; la fonction sélectionne elle-même les lignes de chaque réseau)
    construire_graphe : votre fonction ``construire_graphe(banques_r, expositions_r)``
    simuler : votre fonction ``simuler(G, origine, lam)``
    lambdas : fractions de perte (par défaut LAMBDAS = (0.3, 0.6, 0.9))
    repartition : DataFrame de repartition.csv (colonnes reseau_id, usage), facultatif ;
        sans lui, la colonne ``usage`` reste vide
    reseaux : identifiant ou liste d'identifiants de réseaux à traiter (par défaut : tous)
    progression : afficher une barre de progression (False pour la désactiver)

    Renvoie
    -------
    DataFrame trié par reseau_id, origine, lambda, avec les colonnes :

    ================== ==========================================================
    reseau_id          identifiant du réseau
    usage              apprentissage / validation / test (vide sans repartition)
    origine            banque dont le défaut est imposé
    lambda             fraction de perte λ
    n_defauts          |D_final|, origine comprise
    y                  (n_defauts - 1) / (N - 1), N = nombre de sommets du graphe
    n_etapes           nombre d'étapes avec au moins un nouveau défaut
    nouveaux_par_etape texte "4;1;6" (chaîne vide s'il n'y a aucune propagation)
    defauts_finaux     texte "B01;B07;B12" (identifiants triés)
    ================== ==========================================================

    Contrôles effectués (erreur explicite en cas de problème) : graphe orienté,
    présence de toutes les banques du réseau (banques isolées comprises), attributs
    ``capital`` et ``montant``, format du résultat de ``simuler`` (couple (set, list),
    origine incluse, pas de zéro dans la liste), graphe non modifié par ``simuler``.

    Exemple d'utilisation
    ---------------------
        banques = pd.read_csv("donnees/banques.csv")
        expositions = pd.read_csv("donnees/expositions.csv")
        repartition = pd.read_csv("donnees/repartition.csv")
        table = generer_scenarios(banques, expositions, construire_graphe, simuler,
                                  repartition=repartition)
        table.to_csv("scenarios.csv", index=False)
        essai = generer_scenarios(banques, expositions, construire_graphe, simuler,
                                  reseaux=["R01"], progression=False)   # un seul réseau

    Remarque : pour relire le fichier CSV sans que pandas transforme « 4;1;6 » ou la
    chaîne vide (aucune propagation) en nombre ou en valeur manquante, écrivez
    ``pd.read_csv("scenarios.csv", dtype={"nouveaux_par_etape": str}, keep_default_na=False)``.
    """
    _verifier_colonnes(banques, ["reseau_id", "banque_id", "capital"], "banques")
    _verifier_colonnes(expositions, ["reseau_id", "debiteur_id", "creancier_id", "montant"], "expositions")
    if not callable(construire_graphe) or not callable(simuler):
        raise TypeError(
            "Passez vos FONCTIONS (sans parenthèses) : "
            "generer_scenarios(banques, expositions, construire_graphe, simuler, ...)."
        )
    lambdas = _normaliser_lambdas(lambdas)

    disponibles = sorted(banques["reseau_id"].unique(), key=str)
    if reseaux is None:
        ids_reseaux = disponibles
    else:
        ids_reseaux = [reseaux] if isinstance(reseaux, str) else list(reseaux)
        inconnus = [r for r in ids_reseaux if r not in set(disponibles)]
        if inconnus:
            raise ValueError(f"Réseau(x) absent(s) de « banques » : {_apercu(inconnus)}. "
                             f"Réseaux disponibles : {_apercu(disponibles, 12)}.")
        ids_reseaux = sorted(dict.fromkeys(ids_reseaux), key=str)

    usages = {}
    if repartition is not None:
        _verifier_colonnes(repartition, ["reseau_id", "usage"], "repartition")
        doublons = repartition["reseau_id"][repartition["reseau_id"].duplicated()].tolist()
        if doublons:
            raise ValueError(f"Réseau(x) présent(s) plusieurs fois dans « repartition » : {_apercu(doublons)}.")
        usages = dict(zip(repartition["reseau_id"], repartition["usage"]))
        absents = [r for r in ids_reseaux if r not in usages]
        if absents:
            raise ValueError(f"Réseau(x) absent(s) de « repartition » : {_apercu(absents)}. "
                             "Passez repartition=None pour ignorer la répartition.")

    groupes_banques = {r: t for r, t in banques.groupby("reseau_id", sort=False)}
    groupes_expos = {r: t for r, t in expositions.groupby("reseau_id", sort=False)}
    total = sum(len(groupes_banques[r]) for r in ids_reseaux)
    barre = _Progression(total, actif=progression, unite="origines")
    lignes = []
    try:
        for r in ids_reseaux:
            banques_r = groupes_banques[r].reset_index(drop=True)
            if r in groupes_expos:
                expositions_r = groupes_expos[r].reset_index(drop=True)
            else:  # réseau sans aucune dette : table vide mais colonnes présentes
                expositions_r = expositions.iloc[0:0].reset_index(drop=True)
            ids = banques_r["banque_id"].tolist()
            if len(set(ids)) != len(ids):
                doublons = sorted({b for b in ids if ids.count(b) > 1}, key=str)
                raise ValueError(f"Banque(s) en double dans le réseau {r} : {_apercu(doublons)}.")
            try:
                G = construire_graphe(banques_r, expositions_r)
            except Exception as erreur:
                _signaler_contexte(erreur, f"Erreur levée par votre fonction construire_graphe (réseau {r}).")
                raise
            _controler_graphe(G, ids, r)
            n = G.number_of_nodes()
            empreinte = _empreinte_graphe(G)
            for origine in sorted(ids, key=str):
                for lam in lambdas:
                    defauts, etapes = _simuler_et_valider(simuler, G, origine, lam, empreinte, f"Réseau {r}, ")
                    lignes.append((
                        r,
                        usages.get(r),
                        origine,
                        lam,
                        len(defauts),
                        proportion_secondaire(defauts, n),
                        len(etapes),
                        ";".join(str(nb) for nb in etapes),
                        ";".join(sorted((str(b) for b in defauts))),
                    ))
                barre.avancer(1, str(r))
    finally:
        barre.terminer()
    table = pd.DataFrame(lignes, columns=list(COLONNES_SCENARIOS))
    table = table.sort_values(["reseau_id", "origine", "lambda"], kind="mergesort").reset_index(drop=True)
    if progression:
        print(f"{len(table)} scénarios simulés ({len(ids_reseaux)} réseau(x), {len(lambdas)} valeur(s) de λ) "
              f"en {time.perf_counter() - barre.debut:.1f} s.")
    return table


# ---------------------------------------------------------------------------
# Références médianes
# ---------------------------------------------------------------------------

def references_medianes(apprentissage):
    """Calcule les deux références du protocole à partir des scénarios d'APPRENTISSAGE.

    * référence globale : médiane de y sur tous les scénarios d'apprentissage ;
    * référence par λ : une médiane distincte pour chaque valeur de λ.

    La seconde isole ce qu'apportent les variables du réseau au-delà de l'intensité
    du choc : un modèle utile doit faire mieux qu'elle.

    Paramètres
    ----------
    apprentissage : DataFrame des scénarios d'apprentissage (colonnes ``lambda`` et ``y``)

    Renvoie
    -------
    dict ``{"globale": float, "par_lambda": {λ: float}}`` ; les clés λ sont arrondies
    à 10 décimales (0.30000000000000004 devient 0.3).

    Exemple d'utilisation
    ---------------------
        app = table[table["usage"] == "apprentissage"]
        refs = references_medianes(app)
        refs["globale"], refs["par_lambda"][0.6]
    """
    _verifier_colonnes(apprentissage, ["lambda", "y"], "apprentissage")
    if len(apprentissage) == 0:
        raise ValueError("La table d'apprentissage est vide : vérifiez le filtre sur la colonne usage.")
    if "usage" in apprentissage.columns:
        autres = sorted({str(u) for u in apprentissage["usage"].dropna().unique()} - {"apprentissage"})
        if autres:
            warnings.warn(
                f"La table passée à references_medianes contient des lignes {autres} : les références "
                "doivent être calculées sur l'APPRENTISSAGE seulement, sinon l'évaluation est biaisée. "
                "Filtrez d'abord : table[table['usage'] == 'apprentissage'].",
                UserWarning,
                stacklevel=2,
            )
    y = pd.to_numeric(apprentissage["y"], errors="coerce")
    if y.isna().any():
        raise ValueError("La colonne y contient des valeurs manquantes ou non numériques.")
    cles = apprentissage["lambda"].map(_cle_lambda)
    par_lambda = {float(lam): float(med) for lam, med in y.groupby(cles).median().items()}
    return {"globale": float(y.median()), "par_lambda": dict(sorted(par_lambda.items()))}


def predire_references(refs, table):
    """Prédictions des deux références pour chaque ligne de ``table``.

    Paramètres
    ----------
    refs : dictionnaire renvoyé par ``references_medianes``
    table : DataFrame avec une colonne ``lambda`` (validation, test, ...)

    Renvoie
    -------
    DataFrame de même index que ``table``, colonnes ``pred_globale`` et ``pred_par_lambda``.

    Exemple d'utilisation
    ---------------------
        val = table[table["usage"] == "validation"]
        preds = predire_references(refs, val)
        (preds["pred_par_lambda"] - val["y"]).abs().mean()     # MAE de la référence par λ
        val = val.join(preds)                                  # ou pd.concat([val, preds], axis=1)
    """
    if not isinstance(refs, dict) or "globale" not in refs or "par_lambda" not in refs:
        raise TypeError("« refs » doit être le dictionnaire renvoyé par references_medianes(apprentissage).")
    _verifier_colonnes(table, ["lambda"], "table")
    par_lambda = {_cle_lambda(lam): float(val) for lam, val in refs["par_lambda"].items()}
    cles = table["lambda"].map(_cle_lambda)
    absentes = sorted(set(cles.unique()) - set(par_lambda))
    if absentes:
        raise ValueError(
            f"Valeur(s) de λ sans référence : {absentes}. Les références par λ ont été calculées pour "
            f"{sorted(par_lambda)} : utilisez les mêmes valeurs de λ en apprentissage et en évaluation."
        )
    return pd.DataFrame(
        {
            "pred_globale": np.full(len(table), float(refs["globale"])),
            "pred_par_lambda": cles.map(par_lambda).astype(float).to_numpy(),
        },
        index=table.index,
    )


# ---------------------------------------------------------------------------
# Interventions : budget, tirages aléatoires, comparaison
# ---------------------------------------------------------------------------

def capitaux_renforces(G, choisies, part_budget=0.10):
    """Copie de G dans laquelle les banques choisies reçoivent le budget d'intervention.

    Le budget vaut B = part_budget x (somme des capitaux de G). Il est réparti à parts
    égales : chaque banque choisie reçoit B / len(choisies). Les dettes ne changent pas.
    G n'est pas modifié (copie profonde) : passez toujours le graphe INITIAL, sinon B
    serait calculé sur des capitaux déjà renforcés.

    Paramètres
    ----------
    G : graphe initial (attribut ``capital`` sur chaque sommet)
    choisies : liste de banques distinctes à renforcer, par exemple ["B07", "B12"]
    part_budget : part de la somme des capitaux consacrée à l'intervention (0,10 dans le sujet)

    Renvoie
    -------
    nx.DiGraph : copie de G avec les capitaux renforcés

    Exemple d'utilisation
    ---------------------
        H = capitaux_renforces(G, ["B03", "B17", "B21", "B40", "B55"])   # k = 5 : B/5 chacune
        H.nodes["B03"]["capital"] - G.nodes["B03"]["capital"]           # = 0.10 * somme / 5
        simuler(H, "B01", 0.6)          # même règle, mêmes dettes, capitaux renforcés
    """
    choisies = _en_liste_de_banques(choisies, "choisies")
    if not choisies:
        raise ValueError(
            "La liste des banques à renforcer est vide. Pour « sans intervention », "
            "simulez directement sur G (ou passez None dans comparer_interventions)."
        )
    doublons = sorted({b for b in choisies if choisies.count(b) > 1}, key=str)
    if doublons:
        raise ValueError(
            f"Banque(s) choisie(s) plusieurs fois : {_apercu(doublons)}. Les banques renforcées "
            "doivent être distinctes (tirage sans remise)."
        )
    inconnues = [b for b in choisies if b not in G]
    if inconnues:
        exemples = _apercu(sorted(G.nodes, key=str), 3)
        raise ValueError(
            f"Banque(s) absente(s) du graphe : {_apercu(inconnues)}. "
            f"Les identifiants du graphe ressemblent à : {exemples}."
        )
    try:
        part_budget = float(part_budget)
    except (TypeError, ValueError):
        raise TypeError(
            f"part_budget doit être un nombre, par exemple 0.10 ; reçu : {_decrire(part_budget)}."
        ) from None
    if part_budget < 0:
        raise ValueError(f"part_budget = {part_budget} : la part du budget doit être positive ou nulle.")
    sans_capital = [n for n, d in G.nodes(data=True) if "capital" not in d]
    if sans_capital:
        raise ValueError(f"Sommet(s) sans attribut 'capital' : {_apercu(sans_capital)}.")
    total = sum(float(G.nodes[n]["capital"]) for n in G.nodes)
    budget = part_budget * total
    par_banque = budget / len(choisies)
    H = copy.deepcopy(G)
    for b in choisies:
        H.nodes[b]["capital"] = float(H.nodes[b]["capital"]) + par_banque
    return H


def _numero_reseau(numero_reseau):
    """7 -> 7 ; "R07" -> 7."""
    if isinstance(numero_reseau, numbers.Integral) and not isinstance(numero_reseau, bool):
        numero = int(numero_reseau)
    elif isinstance(numero_reseau, str):
        trouve = re.search(r"(\d+)\s*$", numero_reseau)
        if not trouve:
            raise ValueError(
                f"Impossible de lire un numéro de réseau dans {numero_reseau!r} : "
                "passez un entier (R07 -> 7)."
            )
        numero = int(trouve.group(1))
    else:
        raise TypeError(f"numero_reseau doit être un entier (R07 -> 7) ; reçu : {_decrire(numero_reseau)}.")
    if numero < 0:
        raise ValueError(f"numero_reseau = {numero} : un entier positif est attendu.")
    return numero


def tirer_banques_aleatoires(ids, graine, numero_reseau, k=5):
    """Tire k banques distinctes, sans remise, de façon reproductible.

    Le tirage est entièrement déterminé par la graine et le numéro du réseau :

        ids_tries = sorted(ids)
        rng = np.random.default_rng([graine, numero_reseau])
        positions = rng.choice(len(ids_tries), size=k, replace=False)

    Les tirages officiels (graines 0 à 9) sont fournis dans tirages_aleatoires.csv :
    utilisez de préférence ce fichier (voir ``charger_tirages``), commun à toutes les équipes.

    Paramètres
    ----------
    ids : identifiants des banques du réseau (liste, Series, ensemble...) ; ils sont triés
    graine : entier positif (0 à 9 pour les tirages officiels)
    numero_reseau : entier tiré de l'identifiant (R07 -> 7) ; la chaîne "R07" est acceptée
    k : nombre de banques tirées (5 dans le sujet)

    Renvoie
    -------
    list : les k identifiants, dans l'ordre du tirage

    Exemple d'utilisation
    ---------------------
        ids = banques.loc[banques["reseau_id"] == "R07", "banque_id"]
        tirer_banques_aleatoires(ids, graine=0, numero_reseau=7)
        [tirer_banques_aleatoires(ids, g, 7) for g in range(10)]      # dix répétitions
    """
    ids_tries = sorted(ids)
    if len(set(ids_tries)) != len(ids_tries):
        doublons = sorted({b for b in ids_tries if ids_tries.count(b) > 1}, key=str)
        raise ValueError(f"Identifiant(s) en double : {_apercu(doublons)}.")
    if isinstance(graine, bool) or not isinstance(graine, numbers.Integral) or graine < 0:
        raise ValueError(f"graine = {graine!r} : un entier positif ou nul est attendu (0 à 9).")
    k = int(k)
    if not 1 <= k <= len(ids_tries):
        raise ValueError(f"Impossible de tirer k = {k} banques distinctes parmi {len(ids_tries)}.")
    rng = np.random.default_rng([int(graine), _numero_reseau(numero_reseau)])
    positions = rng.choice(len(ids_tries), size=k, replace=False)
    return [ids_tries[int(p)] for p in positions]


def charger_tirages(chemin):
    """Lit tirages_aleatoires.csv et renvoie les sélections aléatoires officielles.

    Paramètres
    ----------
    chemin : chemin du fichier (colonnes reseau_id, graine, ordre, banque_id),
        ou DataFrame déjà lu

    Renvoie
    -------
    dict ``{(reseau_id, graine): [banque_id, ...]}`` : les banques dans l'ordre 1, 2, ..., k

    Exemple d'utilisation
    ---------------------
        tirages = charger_tirages("donnees/tirages_aleatoires.csv")
        tirages[("R07", 0)]
        aleatoires_R07 = [tirages[("R07", g)] for g in range(10)]      # pour comparer_interventions
    """
    table = chemin if isinstance(chemin, pd.DataFrame) else pd.read_csv(chemin)
    _verifier_colonnes(table, ["reseau_id", "graine", "ordre", "banque_id"], "tirages")
    if table[["reseau_id", "graine", "ordre", "banque_id"]].isna().any().any():
        raise ValueError("Le fichier des tirages contient des valeurs manquantes.")
    table = table.sort_values(["reseau_id", "graine", "ordre"], kind="mergesort")
    tirages = {}
    for (r, graine), groupe in table.groupby(["reseau_id", "graine"], sort=True):
        ordres = [int(o) for o in groupe["ordre"]]
        ids = [str(b) for b in groupe["banque_id"]]
        if ordres != list(range(1, len(ordres) + 1)):
            raise ValueError(f"Tirage ({r}, graine {graine}) : ordres {ordres} au lieu de 1, 2, ..., k.")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Tirage ({r}, graine {graine}) : banque tirée deux fois ({ids}).")
        tirages[(str(r), int(graine))] = ids
    return tirages


def _normaliser_strategie(nom, selection):
    """Une stratégie -> liste de sélections (une par répétition ; [] = sans intervention)."""
    if selection is None:
        return [[]]
    if isinstance(selection, str):
        raise TypeError(
            f"Stratégie « {nom} » : une liste de banques est attendue, par exemple [{selection!r}] "
            f"(et non la chaîne {selection!r})."
        )
    if isinstance(selection, (pd.Series, pd.Index, np.ndarray)):
        selection = selection.tolist()
    elements = list(selection)
    if not elements:
        return [[]]
    composees = [isinstance(e, (list, tuple, set, frozenset, np.ndarray, pd.Series)) or e is None
                 for e in elements]
    if all(composees):          # liste de listes : plusieurs répétitions
        return [_en_liste_de_banques(e, f"{nom}[{rep}]") for rep, e in enumerate(elements)]
    if not any(composees):      # liste de banques : une seule sélection
        return [elements]
    raise TypeError(
        f"Stratégie « {nom} » : mélange de banques et de listes. Donnez soit une liste de banques "
        "(stratégie déterministe), soit une liste de listes (répétitions, par exemple les 10 tirages)."
    )


def comparer_interventions(G, simuler, strategies, lambdas=LAMBDAS, part_budget=0.10):
    """Évalue des stratégies de renforcement sur TOUS les scénarios d'un réseau.

    Chaque sélection de banques reçoit le budget (voir ``capitaux_renforces``), puis la
    cascade est simulée pour chaque origine (toutes les banques du graphe) et chaque λ.
    Le défaut de l'origine reste imposé, même si elle a été renforcée : le renforcement
    n'agit que sur la contagion secondaire.

    Paramètres
    ----------
    G : graphe INITIAL d'un réseau
    simuler : votre fonction ``simuler(G, origine, lam)``
    strategies : dictionnaire nom -> sélection, où la sélection est
        * une liste de banques : stratégie déterministe (une seule répétition) ;
        * une liste de listes : plusieurs répétitions (par exemple les 10 tirages aléatoires) ;
        * None ou [] : sans intervention (capitaux initiaux).
    lambdas : fractions de perte (par défaut LAMBDAS)
    part_budget : part de la somme des capitaux consacrée à l'intervention (0,10)

    Renvoie
    -------
    DataFrame au format long, une ligne par (strategie, repetition, origine, lambda) :
    strategie, repetition (0, 1, ... dans l'ordre des sélections fournies),
    banques_renforcees (texte "B03;B17"), origine, lambda, n_defauts, y.

    Exemple d'utilisation
    ---------------------
        tirages = charger_tirages("donnees/tirages_aleatoires.csv")
        strategies = {
            "sans": None,
            "aleatoire": [tirages[("R07", g)] for g in range(10)],
            "ciblee": ["B03", "B17", "B21", "B40", "B55"],       # votre règle
        }
        resultats = comparer_interventions(G, simuler, strategies)
        resultats.insert(0, "reseau_id", "R07")     # utile pour empiler plusieurs réseaux
        resumer_interventions(resultats)
    """
    if not isinstance(strategies, dict) or not strategies:
        raise TypeError(
            "« strategies » doit être un dictionnaire non vide nom -> sélection, par exemple "
            "{'sans': None, 'ciblee': ['B03', 'B17', 'B21', 'B40', 'B55']}."
        )
    _controler_graphe(G)
    lambdas = _normaliser_lambdas(lambdas)
    origines = sorted(G.nodes, key=str)
    n = G.number_of_nodes()
    lignes = []
    for nom, selection in strategies.items():
        for rep, choisies in enumerate(_normaliser_strategie(nom, selection)):
            try:
                H = capitaux_renforces(G, choisies, part_budget) if choisies else G
            except (TypeError, ValueError) as erreur:
                raise type(erreur)(f"Stratégie « {nom} », répétition {rep} : {erreur}") from erreur
            texte = ";".join(str(b) for b in choisies)
            empreinte = _empreinte_graphe(H)
            for origine in origines:
                for lam in lambdas:
                    defauts, _ = _simuler_et_valider(simuler, H, origine, lam, empreinte,
                                                      f"Stratégie « {nom} » (répétition {rep}), ")
                    lignes.append((nom, rep, texte, origine, lam, len(defauts),
                                   proportion_secondaire(defauts, n)))
    return pd.DataFrame(lignes, columns=list(COLONNES_INTERVENTIONS))


def resumer_interventions(resultats):
    """Tableau de synthèse des stratégies évaluées par ``comparer_interventions``.

    Une ligne par stratégie (et par réseau si ``resultats`` contient une colonne
    ``reseau_id`` : le gain est alors calculé réseau par réseau). Colonnes :

    ====================== =======================================================
    n_repetitions          nombre de sélections évaluées (10 pour l'aléatoire)
    n_scenarios            nombre de scénarios (origine, λ) par répétition
    y_moyen                moyenne de y sur tous les scénarios (et répétitions)
    y_moyen_ecart_type     écart-type des moyennes par répétition (NaN si une seule)
    y_moyen_min/_max       plus petite et plus grande moyenne par répétition
    y_max                  plus grande cascade observée (toutes répétitions)
    y_moyen_lambda_0.3 ... moyenne de y pour chaque valeur de λ
    gain_pp                100 x (y_moyen de « sans » - y_moyen) : gain en points de
                           pourcentage ; NaN si aucune stratégie ne s'appelle « sans »
    ====================== =======================================================

    Exemple d'utilisation
    ---------------------
        resume = resumer_interventions(resultats)
        resume[["strategie", "y_moyen", "y_max", "gain_pp"]]
    """
    _verifier_colonnes(resultats, ["strategie", "repetition", "origine", "lambda", "y"], "resultats")
    if len(resultats) == 0:
        raise ValueError("La table des résultats est vide.")
    table = resultats.copy()
    table["_lambda"] = table["lambda"].map(_cle_lambda)
    lambdas = sorted(table["_lambda"].unique())
    colonnes_lambda = [f"y_moyen_lambda_{lam:g}" for lam in lambdas]
    par_reseau = "reseau_id" in table.columns
    groupes = [(r, t) for r, t in table.groupby("reseau_id", sort=True)] if par_reseau else [(None, table)]
    lignes = []
    for r, sous_table in groupes:
        lignes_groupe = []
        for nom in dict.fromkeys(sous_table["strategie"]):          # ordre d'apparition
            s = sous_table[sous_table["strategie"] == nom]
            moyennes_rep = s.groupby("repetition", sort=True)["y"].mean()
            ligne = {} if r is None else {"reseau_id": r}
            ligne.update({
                "strategie": nom,
                "n_repetitions": int(len(moyennes_rep)),
                "n_scenarios": int(len(s[["origine", "_lambda"]].drop_duplicates())),
                "y_moyen": float(s["y"].mean()),
                "y_moyen_ecart_type": float(moyennes_rep.std(ddof=1)) if len(moyennes_rep) > 1 else float("nan"),
                "y_moyen_min": float(moyennes_rep.min()),
                "y_moyen_max": float(moyennes_rep.max()),
                "y_max": float(s["y"].max()),
            })
            moyennes_lambda = s.groupby("_lambda")["y"].mean()
            for lam, colonne in zip(lambdas, colonnes_lambda):
                ligne[colonne] = float(moyennes_lambda[lam]) if lam in moyennes_lambda.index else float("nan")
            lignes_groupe.append(ligne)
        reference = [lg["y_moyen"] for lg in lignes_groupe if lg["strategie"] == "sans"]
        for ligne in lignes_groupe:
            ligne["gain_pp"] = 100.0 * (reference[0] - ligne["y_moyen"]) if reference else float("nan")
        lignes.extend(lignes_groupe)
    colonnes = (["reseau_id"] if par_reseau else []) + [
        "strategie", "n_repetitions", "n_scenarios", "y_moyen", "y_moyen_ecart_type",
        "y_moyen_min", "y_moyen_max", "y_max"] + colonnes_lambda + ["gain_pp"]
    return pd.DataFrame(lignes, columns=colonnes)
