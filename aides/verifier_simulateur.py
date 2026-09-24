"""
Vérification de votre simulateur de cascade : les 9 tests de la section 3 du sujet.

Dans un notebook (le dossier « aides » doit être accessible) ::

    import sys
    sys.path.append("aides")                 # chemin du dossier qui contient ce fichier
    from verifier_simulateur import verifier
    verifier(simuler)                        # teste simuler(G, origine, lam)
    verifier(simuler, construire_graphe)     # teste aussi construire_graphe (sens, banques isolées)

En ligne de commande, depuis le dossier qui contient mon_module.py ::

    python aides/verifier_simulateur.py mon_module

mon_module.py doit définir simuler(G, origine, lam) et, si possible,
construire_graphe(banques_r, expositions_r). Évitez d'y lancer des calculs longs
au chargement : protégez-les par « if __name__ == "__main__": ».

Les tests construisent de petits réseaux directement avec NetworkX (attribut
« capital » sur les sommets, « montant » sur les arêtes ; i -> j = « i doit à j »).
Ce fichier ne contient AUCUN simulateur : les résultats attendus sont écrits en dur
ou vérifiés par des propriétés. Pour chaque test en échec, vous obtenez ce qui était
attendu, ce que votre fonction a renvoyé et l'erreur la plus probable.

Les affichages (print) de votre simulateur sont masqués pendant les tests, et un
appel qui dure plus de quelques secondes est considéré comme une boucle infinie.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import os
import sys
import textwrap
import threading
import traceback
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

try:
    from aides_contagion import (
        LAMBDAS,
        _MESSAGE_GRAPHE_MODIFIE,
        GrapheInvalide,
        ResultatSimulationInvalide,
        _controler_graphe,
        _empreinte_graphe,
        _valider_resultat_simulation,
        depasse_capital,
    )
except ImportError as _erreur:  # pragma: no cover - message pour l'utilisateur
    raise ImportError(
        "verifier_simulateur.py a besoin de aides_contagion.py : gardez les deux fichiers dans le "
        "même dossier et ajoutez ce dossier au chemin (sys.path.append('aides'))."
    ) from _erreur

__all__ = ["verifier", "executer_tests", "ResultatTest"]

NB_TESTS = 9
DELAI_PAR_APPEL = 5.0          # secondes : au-delà, on suppose une boucle infinie
MESSAGES_AFFICHES = 3          # cas en échec détaillés par test (les autres sont comptés)
_LARGEUR = 100                 # largeur des lignes du rapport
_NOM_FIL = "verifier_simulateur"
_FICHIERS_KIT = {"verifier_simulateur.py", "aides_contagion.py", "threading.py"}


# ---------------------------------------------------------------------------
# Résultats et affichage
# ---------------------------------------------------------------------------

@dataclass
class ResultatTest:
    """Résultat d'un test : numéro, titre, succès et explications des cas en échec.

    Chaque message est un texte ; ses lignes suivantes (« → ... ») donnent l'erreur probable.
    """

    numero: int
    titre: str
    reussi: bool = True
    messages: list = field(default_factory=list)

    def echec(self, message):
        self.reussi = False
        if message not in self.messages:
            self.messages.append(message)


_REMPLACEMENTS = {
    "✅": "[OK]", "❌": "[ECHEC]", "→": "->", "λ": "lambda", "×": "x", "≤": "<=", "≥": ">=",
    "…": "...", "•": "-", "Σ": "somme", "≠": "!=", "\u2013": "-", "’": "'", "«": '"', "»": '"', "\u00a0": " ",
}


def _adapter(texte, encodage):
    """Remplace les seuls caractères que la console ne sait pas écrire (✅ -> [OK], é -> e...)."""
    if encodage is None:                     # flux texte Python (StringIO...) : tout Unicode passe
        return texte
    try:
        texte.encode(encodage)
        return texte
    except (UnicodeEncodeError, LookupError):
        pass
    morceaux = []
    for caractere in texte:
        try:
            caractere.encode(encodage)
            morceaux.append(caractere)
        except (UnicodeEncodeError, LookupError):
            remplacement = _REMPLACEMENTS.get(caractere)
            if remplacement is None:
                remplacement = unicodedata.normalize("NFKD", caractere).encode("ascii", "ignore").decode() or "?"
            morceaux.append(remplacement)
    return "".join(morceaux)


class _Ecrivain:
    """Écrit le rapport en s'adaptant à l'encodage de la console (Windows, redirections...)."""

    def __init__(self, actif=True):
        self.actif = actif
        self.encodage = getattr(sys.stdout, "encoding", None)

    def __call__(self, texte=""):
        if self.actif:
            print(_adapter(texte, self.encodage))


def _mettre_en_forme(message):
    """« • contexte ... » puis « → erreur probable », coupés à la largeur du rapport."""
    lignes = []
    for k, morceau in enumerate(message.split("\n")):
        morceau = morceau.strip()
        if k == 0:
            debut, suite = "   • ", "     "
        else:
            morceau = morceau[1:].strip() if morceau.startswith("→") else morceau
            debut, suite = "     → ", "       "
        lignes.append(textwrap.fill(morceau, width=_LARGEUR, initial_indent=debut, subsequent_indent=suite,
                                    break_long_words=False, break_on_hyphens=False))
    return "\n".join(lignes)


class _FiltreSortie:
    """Remplace sys.stdout pendant les tests : masque les print venant du simulateur testé."""

    def __init__(self, sortie):
        self._sortie = sortie
        self.caracteres_masques = 0

    def write(self, texte):
        if threading.current_thread().name.startswith(_NOM_FIL):
            self.caracteres_masques += len(texte)
            return len(texte)
        return self._sortie.write(texte)

    def flush(self):
        return self._sortie.flush()

    def __getattr__(self, nom):
        return getattr(self._sortie, nom)


# ---------------------------------------------------------------------------
# Appel protégé de la fonction testée
# ---------------------------------------------------------------------------

class _ErreurAppel(Exception):
    """L'appel a échoué (exception, délai dépassé, résultat invalide, graphe modifié)."""


class _Bloque(Exception):
    """Un appel précédent ne s'est pas terminé : on n'appelle plus la fonction."""


class _DelaiDepasse(Exception):
    """L'appel a dépassé le délai ; ``interrompu`` indique si le calcul a pu être arrêté."""

    def __init__(self, interrompu):
        super().__init__(interrompu)
        self.interrompu = interrompu


class _Interruption(BaseException):
    """Levée dans un fil bloqué pour l'arrêter (hérite de BaseException : ne pas l'attraper)."""


@dataclass
class _Contexte:
    simuler: object
    construire_graphe: object = None
    delai: float = DELAI_PAR_APPEL
    bloque: bool = False


def _arreter_fil(fil, attente=2.0):
    """Tente d'arrêter un fil bloqué dans une boucle Python (CPython) ; renvoie True si c'est fait."""
    try:
        import ctypes

        lancer = ctypes.pythonapi.PyThreadState_SetAsyncExc
        nb = lancer(ctypes.c_ulong(fil.ident), ctypes.py_object(_Interruption))
        if nb != 1:
            if nb > 1:                       # ne devrait jamais arriver : on annule
                lancer(ctypes.c_ulong(fil.ident), None)
            return False
    except Exception:  # noqa: BLE001 - autre interpréteur que CPython, etc.
        return False
    fil.join(attente)
    return not fil.is_alive()


def _executer_avec_delai(fonction, args, delai):
    """Exécute fonction(*args) dans un fil séparé ; lève _DelaiDepasse au-delà de « delai » secondes."""
    resultat = {}

    def cible():
        try:
            resultat["valeur"] = fonction(*args)
        except BaseException as erreur:  # transmise au fil principal
            resultat["erreur"] = erreur

    fil = threading.Thread(target=cible, name=f"{_NOM_FIL}-{id(resultat)}", daemon=True)
    fil.start()
    fil.join(delai)
    if fil.is_alive():
        raise _DelaiDepasse(_arreter_fil(fil))
    if "erreur" in resultat:
        raise resultat["erreur"]
    return resultat["valeur"]


def _lieu_dans_votre_code(erreur):
    """« fichier mon_module.py, ligne 12 : perte = ... » pour la dernière ligne du code de l'équipe."""
    lieu = None
    for cadre in traceback.extract_tb(erreur.__traceback__):
        nom = os.path.basename(cadre.filename)
        if nom in _FICHIERS_KIT or "site-packages" in cadre.filename or "dist-packages" in cadre.filename:
            continue
        lieu = f"fichier {nom}, ligne {cadre.lineno}" + (f" : {cadre.line.strip()}" if cadre.line else "")
    return lieu


def _expliquer_exception(erreur, fonction="simuler"):
    texte = f"{fonction} a levé l'exception {type(erreur).__name__} : {erreur}"
    lieu = _lieu_dans_votre_code(erreur)
    if lieu:
        texte += f" ({lieu})"
    conseil = None
    if isinstance(erreur, KeyError) and erreur.args and erreur.args[0] in ("montant", "capital"):
        conseil = ("Les attributs s'appellent 'capital' sur les sommets (G.nodes[j]['capital']) et "
                   "'montant' sur les arêtes (G[i][j]['montant']).")
    elif isinstance(erreur, KeyError):
        conseil = ("Clé introuvable : lisez les banques et les dettes dans G (G.nodes, G.predecessors(j), "
                   "G[i][j]) plutôt que dans une table ou une variable extérieure à la fonction.")
    elif isinstance(erreur, TypeError) and "argument" in str(erreur):
        conseil = ("Signatures attendues : simuler(G, origine, lam) et "
                   "construire_graphe(banques_r, expositions_r), arguments dans cet ordre.")
    elif isinstance(erreur, nx.NetworkXError):
        conseil = "Erreur NetworkX : n'utilisez que des banques présentes dans G."
    elif isinstance(erreur, RecursionError):
        conseil = "Récursion trop profonde : préférez une boucle while sur les étapes."
    return texte + "." + (f"\n→ {conseil}" if conseil else "")


def _apercu_resultat(brut, longueur=100):
    texte = repr(brut)
    return texte if len(texte) <= longueur else texte[: longueur - 3] + "..."


def _appeler(ctx, G, origine, lam):
    """Appelle simuler(G, origine, lam) et renvoie (defauts, nouveaux_par_etape) validés."""
    if ctx.bloque:
        raise _Bloque()
    empreinte = _empreinte_graphe(G)
    try:
        brut = _executer_avec_delai(ctx.simuler, (G, origine, lam), ctx.delai)
    except _DelaiDepasse as delai:
        ctx.bloque = True
        suite = ("Le calcul a été interrompu." if delai.interrompu else
                 "Dans un notebook, redémarrez le noyau pour arrêter le calcul resté en arrière-plan.")
        raise _ErreurAppel(
            f"simuler ne s'est pas terminé en {ctx.delai:g} s : boucle infinie probable."
            "\n→ Arrêtez-vous dès qu'une étape n'apporte aucun nouveau défaut, et excluez des « nouveaux » "
            f"les banques déjà en défaut. {suite}"
        ) from None
    except Exception as erreur:  # noqa: BLE001 - toute erreur de l'équipe est expliquée
        raise _ErreurAppel(_expliquer_exception(erreur)) from None
    try:
        defauts, etapes = _valider_resultat_simulation(brut, G, origine, lam)
    except ResultatSimulationInvalide as erreur:
        raise _ErreurAppel(f"résultat refusé, obtenu {_apercu_resultat(brut)}.\n→ {erreur}") from None
    if _empreinte_graphe(G) != empreinte:
        raise _ErreurAppel(f"simuler {_MESSAGE_GRAPHE_MODIFIE}")
    return defauts, etapes


def _construire(ctx, banques_r, expositions_r):
    """Appelle construire_graphe avec les mêmes protections que simuler."""
    try:
        return _executer_avec_delai(ctx.construire_graphe, (banques_r, expositions_r), ctx.delai)
    except _DelaiDepasse:
        ctx.bloque = True
        raise _ErreurAppel(f"construire_graphe ne s'est pas terminé en {ctx.delai:g} s.") from None
    except Exception as erreur:  # noqa: BLE001
        raise _ErreurAppel(_expliquer_exception(erreur, "construire_graphe")) from None


# ---------------------------------------------------------------------------
# Petits outils pour écrire les tests
# ---------------------------------------------------------------------------

def _graphe(capitaux, dettes):
    """Graphe de test : capitaux = {banque: capital}, dettes = [(debiteur, creancier, montant)]."""
    G = nx.DiGraph()
    for banque, capital in capitaux.items():
        G.add_node(banque, capital=float(capital))
    for debiteur, creancier, montant in dettes:
        G.add_edge(debiteur, creancier, montant=float(montant))
    return G


def _ens(banques, maximum=8):
    """{A, B, C} ; les grands ensembles sont abrégés."""
    noms = sorted(map(str, banques))
    if len(noms) > maximum:
        return "{" + ", ".join(noms[:maximum - 2]) + f", ... ({len(noms)} banques)" + "}"
    return "{" + ", ".join(noms) + "}"


def _res(defauts, etapes):
    """« {A, B, C} et [1, 1] » ; la liste reste sur une seule ligne (espaces insécables)."""
    return f"{_ens(defauts)} et " + str(list(etapes)).replace(", ", ",\u00a0")


def _perte(G, j, defauts, lam):
    """L_j(D) = λ × somme des montants dus à j par les banques de D."""
    return lam * sum(G[i][j]["montant"] for i in sorted(G.predecessors(j), key=str) if i in defauts)


def _nb(x):
    """Nombre au format français, sans zéros inutiles : 3.5 -> « 3,5 »."""
    return f"{x:.6g}".replace(".", ",")


def _conseil_generique(defauts, etapes, attendu_defauts, attendu_etapes):
    if defauts == attendu_defauts:
        return ("L'ensemble final est juste, mais pas le nombre de nouveaux défauts par étape : les défauts "
                "d'une étape s'ajoutent tous ensemble (voir les tests 1 et 4).")
    manquants, en_trop = attendu_defauts - defauts, defauts - attendu_defauts
    morceaux = []
    if manquants:
        morceaux.append(f"défaut(s) manquant(s) : {_ens(manquants)}")
    if en_trop:
        morceaux.append(f"défaut(s) en trop : {_ens(en_trop)}")
    texte = " ; ".join(morceaux)
    return texte[0].upper() + texte[1:] + "."


def _essai(ctx, resultat, G, origine, lam, attendu_defauts, attendu_etapes, contexte, conseil_fn=None):
    """Compare un appel au résultat attendu ; ajoute l'explication au test si l'appel diffère.

    conseil_fn(defauts, etapes) renvoie l'erreur la plus probable (ou None : conseil générique).
    Renvoie True si l'appel est conforme.
    """
    attendu_defauts, attendu_etapes = set(attendu_defauts), list(attendu_etapes)
    try:
        defauts, etapes = _appeler(ctx, G, origine, lam)
    except _ErreurAppel as erreur:
        resultat.echec(f"{contexte} : {erreur}")
        return False
    if defauts == attendu_defauts and etapes == attendu_etapes:
        return True
    conseil = conseil_fn(defauts, etapes) if conseil_fn else None
    if not conseil:
        conseil = _conseil_generique(defauts, etapes, attendu_defauts, attendu_etapes)
    resultat.echec(f"{contexte} : attendu {_res(attendu_defauts, attendu_etapes)} ; "
                   f"obtenu {_res(defauts, etapes)}.\n→ {conseil}")
    return False


# ---------------------------------------------------------------------------
# Les 9 tests (chaque fonction remplit le ResultatTest r)
# ---------------------------------------------------------------------------

_CONSEIL_SENS = ("Avez-vous inversé débiteur et créancier ? Une arête i -> j signifie « i doit à j » : "
                 "la banque j perd de l'argent quand ses DÉBITEURS font défaut, c'est-à-dire ses "
                 "prédécesseurs G.predecessors(j), et non ses successeurs.")
_CONSEIL_SIMULTANE = ("Les défauts d'une étape se calculent à partir des seuls défauts connus au début de "
                      "l'étape, puis sont ajoutés tous ensemble. Ne modifiez pas l'ensemble des défauts "
                      "pendant le parcours des banques : rangez les nouveaux dans un second ensemble.")
_CONSEIL_CUMUL = ("Les pertes sont-elles cumulées d'une étape à l'autre ? L_j(D_t) est la perte TOTALE due "
                  "aux banques déjà défaillantes : recalculez-la à chaque étape à partir de l'ensemble des "
                  "défauts, sans l'ajouter aux pertes de l'étape précédente.")
_CONSEIL_ORIGINE = ("Le défaut de l'origine est imposé par le choc, quel que soit son capital : commencez "
                    "avec defauts = {origine} et ne testez jamais le capital de l'origine.")
_CONSEIL_ARRET = ("La cascade s'arrête trop tôt : répétez les étapes jusqu'à ce qu'aucun nouveau défaut "
                  "n'apparaisse (point fixe).")


def _test_1_exemple_abc(ctx, r):
    G = _graphe({"A": 5, "B": 4, "C": 3}, [("A", "B", 10), ("B", "C", 8)])

    def conseil(defauts, etapes):
        if "A" not in defauts:
            return _CONSEIL_ORIGINE
        if defauts == {"A"}:
            return "B devrait perdre 0,5 × 10 = 5 > 4. " + _CONSEIL_SENS
        if defauts == {"A", "B"}:
            return "Une fois B en défaut, C perd 0,5 × 8 = 4 > 3. " + _CONSEIL_ARRET
        if defauts == {"A", "B", "C"} and etapes == [2]:
            return "B et C sont comptés à la même étape alors que C ne peut tomber qu'après B. " + _CONSEIL_SIMULTANE
        if defauts == {"A", "B", "C"}:
            return ("L'ensemble final est juste mais nouveaux_par_etape doit valoir [1, 1] : un nombre par "
                    "étape, sans l'origine et sans zéro final.")
        return "Vérifiez la règle « perte > capital » et le sens des arêtes (i -> j : i doit à j)."

    _essai(ctx, r, G, "A", 0.5, {"A", "B", "C"}, [1, 1], "Origine A, λ = 0,5", conseil)


def _test_2_aucun_defaut_secondaire(ctx, r):
    # a) λ = 0 : aucune dette n'est perdue.
    G = _graphe({"A": 5, "B": 4, "C": 3}, [("A", "B", 10), ("B", "C", 8)])
    for origine in ("A", "B", "C"):
        _essai(ctx, r, G, origine, 0.0, {origine}, [], f"λ = 0, origine {origine}",
               lambda d, e: "Avec λ = 0, aucune perte : la perte doit valoir lam × (somme des montants dus).")
    # b) réseau sans aucune arête, capitaux minuscules.
    G = _graphe({"P": 0.01, "Q": 0.01, "R": 0.01, "S": 0.01}, [])
    for origine in ("P", "S"):
        _essai(ctx, r, G, origine, 0.9, {origine}, [], f"Réseau sans arête, origine {origine}, λ = 0,9",
               lambda d, e: "Sans dette, personne ne perd : la perte d'une banque sans débiteur vaut 0.")
    # c) capital de chaque banque > somme de ses créances : aucune perte ne peut l'atteindre.
    dettes = [("A", "B", 6), ("A", "C", 3), ("B", "C", 7), ("C", "D", 9), ("D", "A", 4), ("B", "D", 2),
              ("E", "A", 5), ("D", "E", 8)]
    capitaux = {b: 1.0 for b in "ABCDE"}
    for _, j, montant in dettes:
        capitaux[j] += montant                                   # c_j = créances + 1
    G = _graphe(capitaux, dettes)
    for origine in "ABCDE":
        for lam in LAMBDAS:
            _essai(ctx, r, G, origine, lam, {origine}, [], f"Capitaux élevés, origine {origine}, λ = {_nb(lam)}",
                   lambda d, e: ("Chaque capital dépasse la somme des créances de la banque : aucune perte ne "
                                 "peut l'atteindre. La perte de j ne compte que les montants dus à j par ses "
                                 "débiteurs en défaut, multipliés par lam."))


def _test_3_perte_egale_au_capital(ctx, r):
    # a) égalité exacte : 0,5 × 10 = 5 = c.
    G = _graphe({"S": 1, "J": 5}, [("S", "J", 10)])
    exact_ok = _essai(
        ctx, r, G, "S", 0.5, {"S"}, [], "S doit 10 à J, c_J = 5, λ = 0,5 (perte = 5 = capital)",
        lambda d, e: ("Une perte exactement égale au capital ne déclenche pas de défaut : l'inégalité doit "
                      "être STRICTE (perte > capital, pas >=). Utilisez depasse_capital(perte, capital)."),
    )
    # b) cas du sujet : 0,3 × 10 = 3 (ce produit est exact en binaire : il ne piège personne).
    G = _graphe({"S": 1, "J": 3}, [("S", "J", 10)])
    _essai(ctx, r, G, "S", 0.3, {"S"}, [], "S doit 10 à J, c_J = 3, λ = 0,3 (perte = 3 = capital)",
           lambda d, e: "Une perte égale au capital ne déclenche pas de défaut : l'inégalité doit être STRICTE.")
    # c) vrais pièges flottants : le produit calculé dépasse le capital d'un cheveu.
    for lam, montant, capital in [(0.3, 10.30, 3.09), (0.6, 5.15, 3.09), (0.9, 2.10, 1.89)]:
        G = _graphe({"S": 1, "J": capital}, [("S", "J", montant)])

        def conseil(d, e, calcule=repr(lam * montant), capital=capital):
            if not exact_ok:
                return "Même cause que ci-dessus : l'inégalité doit être STRICTE."
            return (f"En nombres flottants, ce produit vaut {calcule} au lieu de {capital} : la comparaison "
                    "naïve « perte > capital » conclut à tort à un défaut. Utilisez "
                    "depasse_capital(perte, capital), qui tolère ces erreurs d'arrondi.")

        _essai(ctx, r, G, "S", lam, {"S"}, [],
               f"S doit {_nb(montant)} à J, c_J = {_nb(capital)}, λ = {_nb(lam)} "
               f"(perte exacte = {_nb(capital)} = capital)", conseil)


def _test_4_defauts_simultanes(ctx, r):
    # Ordre des sommets S, X, Y, Z : une mise à jour « au fil de l'eau » ferait tomber Z trop tôt.
    G = _graphe({"S": 1, "X": 4, "Y": 4, "Z": 4},
                [("S", "X", 10), ("S", "Y", 10), ("X", "Z", 5), ("Y", "Z", 5)])

    def conseil(defauts, etapes):
        tous = {"S", "X", "Y", "Z"}
        if "S" not in defauts:
            return _CONSEIL_ORIGINE
        if defauts == tous and etapes == [3]:
            return "Z tombe à la même étape que X et Y. " + _CONSEIL_SIMULTANE
        if defauts == tous and len(etapes) == 3:
            return "X et Y doivent tomber à la même étape : ajoutez tous les nouveaux défauts d'une étape ensemble."
        if defauts == {"S", "X", "Y"}:
            return ("Z doit perdre 0,5 × (5 + 5) = 5 > 4 : la perte d'une banque est la SOMME des montants "
                    "dus par TOUS ses débiteurs en défaut. " + _CONSEIL_ARRET)
        if not {"X", "Y"} <= defauts:
            return "X et Y perdent chacun 0,5 × 10 = 5 > 4 dès la première étape. " + _CONSEIL_SENS
        return None

    _essai(ctx, r, G, "S", 0.5, {"S", "X", "Y", "Z"}, [2, 1],
           "S doit 10 à X et à Y, qui doivent chacun 5 à Z ; c = 4, λ = 0,5", conseil)


def _reseaux_aleatoires(graine, nombre):
    """Petits réseaux aléatoires reproductibles (graine NumPy fixe), montants au centime."""
    rng = np.random.default_rng(graine)
    reseaux = []
    for _ in range(nombre):
        n = int(rng.integers(8, 21))
        ids = [f"B{k:02d}" for k in range(1, n + 1)]
        densite = rng.uniform(0.12, 0.30)
        dettes = []
        for i in ids:
            for j in ids:
                if i != j and rng.random() < densite:
                    dettes.append((i, j, round(float(rng.uniform(1.0, 20.0)), 2)))
        creances = {b: 0.0 for b in ids}
        for _, j, montant in dettes:
            creances[j] += montant
        fragilite = rng.uniform(0.1, 0.5)
        capitaux = {b: round(max(0.01, fragilite * creances[b] * rng.uniform(0.5, 1.5) + rng.uniform(0.1, 1.0)), 2)
                    for b in ids}
        reseaux.append(_graphe(capitaux, dettes))
    return reseaux


def _test_5_monotonie(ctx, r):
    rng = np.random.default_rng(5)
    for numero, G in enumerate(_reseaux_aleatoires(2025, 10), start=1):
        tous = sorted(G.nodes)
        origines = [tous[int(k)] for k in rng.choice(len(tous), size=min(6, len(tous)), replace=False)]
        H1 = copy.deepcopy(G)                                  # la moitié des banques renforcées
        for b in H1.nodes:
            if rng.random() < 0.5:
                H1.nodes[b]["capital"] = round(H1.nodes[b]["capital"] + float(rng.uniform(0.5, 5.0)), 2)
        H2 = copy.deepcopy(G)                                  # toutes les banques renforcées
        for b in H2.nodes:
            H2.nodes[b]["capital"] = round(1.5 * H2.nodes[b]["capital"], 2)
        for origine in origines:
            for lam in LAMBDAS:
                contexte = f"Réseau aléatoire n° {numero}, origine {origine}, λ = {_nb(lam)}"
                try:
                    defauts, _ = _appeler(ctx, G, origine, lam)
                    for description, H in (("la moitié des capitaux augmentés", H1),
                                            ("tous les capitaux multipliés par 1,5", H2)):
                        en_trop = _appeler(ctx, H, origine, lam)[0] - defauts
                        if en_trop:
                            r.echec(
                                f"{contexte}, {description} : {_ens(en_trop)} fait défaut avec les capitaux "
                                "renforcés mais pas avec les capitaux initiaux.\n→ La règle doit comparer la "
                                "perte de chaque banque à SON capital dans le graphe reçu "
                                "(G.nodes[j]['capital']). Utilisez-vous une variable globale ou une table "
                                "extérieure ? Des pertes cumulées d'une étape à l'autre peuvent aussi "
                                "produire cet effet (voir le test 7)."
                            )
                except _ErreurAppel as erreur:
                    r.echec(f"{contexte} : {erreur}")
                if len(r.messages) >= MESSAGES_AFFICHES:
                    return


def _test_6_sens_des_aretes(ctx, r):
    # c_A = 2 : si le sens était inversé, A « perdrait » 0,5 × 10 = 5 > 2 quand B tombe.
    G = _graphe({"A": 2, "B": 4}, [("A", "B", 10)])

    def conseil_b(defauts, etapes):
        if "A" in defauts:
            return "A est le débiteur : c'est B qui perd si A tombe, jamais l'inverse. " + _CONSEIL_SENS
        return None

    _essai(ctx, r, G, "B", 0.5, {"B"}, [], "Origine B (le créancier), λ = 0,5 : A ne perd rien", conseil_b)
    _essai(ctx, r, G, "A", 0.5, {"A", "B"}, [1], "Origine A (le débiteur), λ = 0,5 : B perd 5 > 4",
           lambda d, e: _CONSEIL_SENS if d == {"A"} else None)
    if ctx.construire_graphe is None:
        return
    banques_r = pd.DataFrame({"reseau_id": ["T6", "T6"], "banque_id": ["A", "B"], "capital": [2.0, 4.0]})
    expositions_r = pd.DataFrame({"reseau_id": ["T6"], "debiteur_id": ["A"], "creancier_id": ["B"],
                                  "montant": [10.0]})
    try:
        H = _construire(ctx, banques_r, expositions_r)
        _controler_graphe(H, ["A", "B"], "de test")
        if H.has_edge("B", "A") and not H.has_edge("A", "B"):
            r.echec("construire_graphe : l'arête va de B vers A alors que A doit 10 à B.\n→ Créez l'arête "
                    "du débiteur vers le créancier : G.add_edge(debiteur_id, creancier_id, montant=...).")
        elif not H.has_edge("A", "B") or H.number_of_edges() != 1:
            r.echec(f"construire_graphe : une seule arête A -> B est attendue ; obtenu {list(H.edges)}.")
        elif abs(H["A"]["B"]["montant"] - 10.0) > 1e-12:
            r.echec(f"construire_graphe : montant de A -> B = {H['A']['B']['montant']!r} au lieu de 10.")
    except (_ErreurAppel, GrapheInvalide) as erreur:
        r.echec(f"construire_graphe sur le réseau « A doit 10 à B » : {erreur}")


def _test_7_pas_de_double_comptage(ctx, r):
    # Étape 1 : J perd 0,5 × 6 = 3 <= 4 ; K perd 5 > 1 et tombe.
    # Étape 2 : J perd AU TOTAL 0,5 × (6 + 1) = 3,5 <= 4 : toujours pas de défaut.
    # En cumulant, on compterait 3 + 3,5 = 6,5 > 4 : faux défaut de J.
    G = _graphe({"S": 1, "K": 1, "J": 4}, [("S", "J", 6), ("S", "K", 10), ("K", "J", 1)])

    def conseil(defauts, etapes):
        if "S" not in defauts:
            return _CONSEIL_ORIGINE
        if "J" in defauts:
            return ("J perd 3 à l'étape 1, puis 3,5 au total à l'étape 2 : jamais plus que son capital 4. "
                    "S'il tombe, c'est que 3 + 3,5 = 6,5 a été compté. " + _CONSEIL_CUMUL)
        if "K" not in defauts:
            return "K perd 0,5 × 10 = 5 > 1 à la première étape. " + _CONSEIL_SENS
        return None

    _essai(ctx, r, G, "S", 0.5, {"S", "K"}, [1],
           "S doit 6 à J et 10 à K ; K doit 1 à J ; c_J = 4, c_K = 1, λ = 0,5", conseil)


def _test_8_origine_imposee_et_isolees(ctx, r):
    # A a un capital énorme mais son défaut est imposé ; D est isolée (aucune dette, aucune créance).
    G = _graphe({"A": 1e9, "B": 4, "C": 3, "D": 0.01}, [("A", "B", 10), ("B", "C", 8)])

    def conseil(defauts, etapes):
        if "A" not in defauts:
            return _CONSEIL_ORIGINE
        if "D" in defauts:
            return "D est isolée : sa perte vaut 0 et ne peut pas dépasser son capital, même petit."
        if defauts == {"A"}:
            return "B devrait perdre λ × 10 > 4 (le capital de l'origine ne compte pas). " + _CONSEIL_SENS
        if defauts == {"A", "B"}:
            return _CONSEIL_ARRET
        return None

    for lam in (0.5, 0.9):
        _essai(ctx, r, G, "A", lam, {"A", "B", "C"}, [1, 1],
               f"Origine A avec un capital de 10^9, λ = {_nb(lam)}", conseil)
    _essai(ctx, r, G, "D", 0.9, {"D"}, [], "Origine D (banque isolée), λ = 0,9",
           lambda d, e: _CONSEIL_ORIGINE if "D" not in d else "Une banque isolée ne doit rien : son défaut ne propage rien.")
    if ctx.construire_graphe is None:
        return
    banques_r = pd.DataFrame({"reseau_id": ["T8"] * 4, "banque_id": ["A", "B", "C", "D"],
                              "capital": [5.0, 4.0, 3.0, 2.0]})
    expositions_r = pd.DataFrame({"reseau_id": ["T8"] * 2, "debiteur_id": ["A", "B"],
                                  "creancier_id": ["B", "C"], "montant": [10.0, 8.0]})
    try:
        H = _construire(ctx, banques_r, expositions_r)
        if isinstance(H, nx.Graph) and "D" not in H:
            r.echec(
                "construire_graphe : la banque isolée D (ni dette ni créance) a disparu du graphe.\n→ Construire "
                "le graphe à partir des seules dettes (add_edge, nx.from_pandas_edgelist) oublie les banques "
                "isolées. Ajoutez d'abord tous les sommets depuis banques_r (G.add_node(banque_id, "
                "capital=...)) : N compte les banques isolées (N = 4 ici, et non 3)."
            )
            return
        _controler_graphe(H, ["A", "B", "C", "D"], "de test")
        capitaux = {b: H.nodes[b]["capital"] for b in "ABCD"}
        if any(abs(capitaux[b] - c) > 1e-12 for b, c in zip("ABCD", (5.0, 4.0, 3.0, 2.0))):
            r.echec(f"construire_graphe : capitaux {capitaux} au lieu de A = 5, B = 4, C = 3, D = 2.")
            return
        _essai(ctx, r, H, "A", 0.5, {"A", "B", "C"}, [1, 1],
               "Graphe produit par construire_graphe (A-B-C et D isolée), origine A, λ = 0,5",
               lambda d, e: "Vérifiez le sens des arêtes créées par construire_graphe (voir le test 6).")
    except (_ErreurAppel, GrapheInvalide) as erreur:
        r.echec(f"construire_graphe sur A-B-C et D isolée : {erreur}")


def _test_9_irreversibilite_et_arret(ctx, r):
    for numero, G in enumerate(_reseaux_aleatoires(2026, 12), start=1):
        n = G.number_of_nodes()
        for origine in sorted(G.nodes):
            atteignables = nx.descendants(G, origine) | {origine}
            for lam in LAMBDAS:
                contexte = f"Réseau aléatoire n° {numero} ({n} banques), origine {origine}, λ = {_nb(lam)}"
                try:
                    defauts, etapes = _appeler(ctx, G, origine, lam)
                except _ErreurAppel as erreur:
                    r.echec(f"{contexte} : {erreur}")
                    if len(r.messages) >= MESSAGES_AFFICHES:
                        return
                    continue
                problemes = []
                if len(etapes) > n - 1:
                    problemes.append(f"{len(etapes)} étapes pour {n} banques (au plus N - 1 = {n - 1}).")
                for j in sorted(set(G.nodes) - defauts):
                    perte = _perte(G, j, defauts, lam)
                    if depasse_capital(perte, G.nodes[j]["capital"]):
                        problemes.append(
                            f"Arrêt trop tôt : avec les défauts finaux, {j} perd {_nb(perte)} > capital "
                            f"{_nb(G.nodes[j]['capital'])} mais n'est pas en défaut. Continuez jusqu'au point "
                            "fixe (plus aucun nouveau défaut) ; un défaut est irréversible.")
                        break
                for j in sorted(defauts - {origine}):
                    perte = _perte(G, j, defauts, lam)
                    if not depasse_capital(perte, G.nodes[j]["capital"]):
                        problemes.append(
                            f"Défaut incohérent : {j} est en défaut alors que sa perte totale due à ses "
                            f"débiteurs en défaut vaut {_nb(perte)} <= capital {_nb(G.nodes[j]['capital'])}. "
                            "Pertes comptées plusieurs fois (test 7) ? Sens des arêtes (test 6) ?")
                        break
                hors_d_atteinte = defauts - atteignables
                if hors_d_atteinte:
                    problemes.append(f"{_ens(hors_d_atteinte)} en défaut sans chaîne de dettes depuis l'origine "
                                     "(sens des arêtes ?).")
                premiere = sum(1 for j in G.successors(origine)
                               if depasse_capital(lam * G[origine][j]["montant"], G.nodes[j]["capital"]))
                obtenue = etapes[0] if etapes else 0
                if obtenue != premiere:
                    problemes.append(
                        f"Première étape : {obtenue} nouveau(x) défaut(s) au lieu de {premiere}. À la première "
                        "étape, seuls les créanciers directs de l'origine (ses successeurs) dont la perte "
                        "dépasse le capital peuvent tomber. Défauts ajoutés un par un pendant l'étape "
                        "(test 4) ? Sens des arêtes (test 6) ?")
                if problemes:
                    r.echec(f"{contexte} : obtenu {_res(defauts, etapes)}.\n→ " + "\n→ ".join(problemes))
                    if len(r.messages) >= MESSAGES_AFFICHES:
                        return


_TESTS = (
    ("Exemple A-B-C (λ = 0,5, origine A)", _test_1_exemple_abc),
    ("Aucun défaut secondaire (λ = 0, réseau sans arête, capitaux élevés)", _test_2_aucun_defaut_secondaire),
    ("Perte égale au capital (inégalité stricte, piège des flottants)", _test_3_perte_egale_au_capital),
    ("Défauts simultanés (S -> X, Y -> Z)", _test_4_defauts_simultanes),
    ("Monotonie : plus de capital ne crée jamais plus de défauts", _test_5_monotonie),
    ("Sens des arêtes (A doit à B)", _test_6_sens_des_aretes),
    ("Pas de double comptage des pertes", _test_7_pas_de_double_comptage),
    ("Origine imposée et banques isolées", _test_8_origine_imposee_et_isolees),
    ("Irréversibilité et arrêt (propriétés sur des réseaux aléatoires)", _test_9_irreversibilite_et_arret),
)


# ---------------------------------------------------------------------------
# Points d'entrée
# ---------------------------------------------------------------------------

def _executer(simuler, construire_graphe=None, delai=DELAI_PAR_APPEL):
    """Exécute les tests ; renvoie (résultats, nombre de caractères affichés par simuler et masqués)."""
    if not callable(simuler):
        raise TypeError("Passez votre FONCTION simuler (sans parenthèses) : verifier(simuler).")
    if construire_graphe is not None and not callable(construire_graphe):
        raise TypeError("construire_graphe doit être une fonction (ou None).")
    ctx = _Contexte(simuler=simuler, construire_graphe=construire_graphe, delai=delai)
    filtre = _FiltreSortie(sys.stdout)
    sys.stdout = filtre
    resultats = []
    try:
        for numero, (titre, test) in enumerate(_TESTS, start=1):
            resultat = ResultatTest(numero, titre)
            try:
                test(ctx, resultat)
            except _Bloque:
                if resultat.messages:
                    resultat.echec("Test interrompu : les appels suivants n'ont pas été lancés.")
                else:
                    resultat.echec("Test non exécuté : un appel précédent de simuler ne s'est pas terminé.")
            resultats.append(resultat)
    finally:
        fils_bloques = [f for f in threading.enumerate() if f.name.startswith(_NOM_FIL) and f.is_alive()]
        if not fils_bloques:          # sinon on garde le filtre pour masquer les print du calcul bloqué
            sys.stdout = filtre._sortie
    return resultats, filtre.caracteres_masques


def executer_tests(simuler, construire_graphe=None, delai=DELAI_PAR_APPEL):
    """Exécute les 9 tests sans rien afficher et renvoie la liste des ``ResultatTest``.

    Chaque ``ResultatTest`` a les attributs ``numero``, ``titre``, ``reussi`` et
    ``messages`` (explications des cas en échec). Une exception, un résultat invalide ou
    une boucle infinie dans ``simuler`` fait échouer le test concerné sans arrêter les
    autres ; après une boucle infinie, les tests suivants ne sont plus lancés.
    """
    return _executer(simuler, construire_graphe, delai)[0]


def verifier(simuler, construire_graphe=None, afficher=True):
    """Lance les 9 tests du sujet sur votre simulateur et affiche un rapport.

    Paramètres
    ----------
    simuler : votre fonction ``simuler(G, origine, lam) -> (defauts, nouveaux_par_etape)``
    construire_graphe : votre fonction ``construire_graphe(banques_r, expositions_r)``,
        facultative ; si elle est fournie, les tests 6 et 8 vérifient aussi le sens des
        arêtes et la conservation des banques isolées
    afficher : False pour ne rien afficher (seul le booléen est renvoyé)

    Renvoie
    -------
    bool : True si les 9 tests sont réussis

    Exemple d'utilisation
    ---------------------
        from verifier_simulateur import verifier
        verifier(simuler, construire_graphe)
    """
    ecrire = _Ecrivain(afficher)
    nom = getattr(simuler, "__name__", "simuler")
    module = getattr(simuler, "__module__", None)
    provenance = f" (module {module})" if module and module != "__main__" else ""
    ecrire(f"Vérification de {nom}{provenance}" + (" et de construire_graphe" if construire_graphe else ""))
    ecrire("-" * _LARGEUR)
    resultats, caracteres_masques = _executer(simuler, construire_graphe)
    for res in resultats:
        ecrire(f"{'✅' if res.reussi else '❌'} Test {res.numero} - {res.titre}")
        for message in res.messages[:MESSAGES_AFFICHES]:
            ecrire(_mettre_en_forme(message))
        autres = len(res.messages) - MESSAGES_AFFICHES
        if autres > 0:
            ecrire(f"   • ... et {autres} autre(s) cas en échec dans ce test.")
    nb_reussis = sum(res.reussi for res in resultats)
    ecrire("-" * _LARGEUR)
    if caracteres_masques:
        ecrire("(Les affichages print de votre simulateur ont été masqués pendant les tests.)")
    if construire_graphe is None:
        ecrire("(construire_graphe non fourni : le sens des arêtes créées et la conservation des banques "
               "isolées n'ont pas été testés.)")
    ecrire(f"Bilan : {nb_reussis}/{NB_TESTS} tests réussis.")
    if nb_reussis < NB_TESTS:
        ecrire("Corrigez d'abord le premier test en échec : les suivants en dépendent souvent.")
    return nb_reussis == NB_TESTS


def _importer_module(nom):
    """Importe « mon_module » (dossier courant ou chemin Python) ou « chemin/mon_module.py »."""
    if nom.endswith(".ipynb"):
        raise SystemExit("Un notebook ne peut pas être importé : copiez simuler (et construire_graphe) dans un "
                         "fichier .py, ou appelez verifier(simuler) directement dans le notebook.")
    chemin = Path(nom)
    if nom.endswith(".py") or chemin.parent != Path("."):
        chemin = chemin if chemin.suffix == ".py" else chemin.with_suffix(".py")
        if not chemin.exists():
            raise SystemExit(f"Fichier introuvable : {chemin}")
        dossier = str(chemin.resolve().parent)
        if dossier not in sys.path:
            sys.path.insert(0, dossier)
        spec = importlib.util.spec_from_file_location(chemin.stem, chemin)
        module = importlib.util.module_from_spec(spec)
        sys.modules[chemin.stem] = module
        spec.loader.exec_module(module)
        return module
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    try:
        return importlib.import_module(nom)
    except ModuleNotFoundError as erreur:
        if erreur.name != nom:
            raise
        raise SystemExit(f"Module « {nom} » introuvable. Lancez la commande depuis le dossier qui contient "
                         f"{nom}.py, ou donnez le chemin du fichier : python verifier_simulateur.py "
                         f"chemin/vers/{nom}.py") from None


def main(argv=None):
    """Interface en ligne de commande : python verifier_simulateur.py mon_module."""
    parser = argparse.ArgumentParser(
        description="Vérifie simuler (et construire_graphe s'il existe) définis dans un module Python.")
    parser.add_argument("module", help="nom du module (mon_module) ou chemin du fichier (mon_module.py)")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")      # jamais d'erreur d'encodage à l'affichage
    except (AttributeError, ValueError):
        pass
    try:
        module = _importer_module(args.module)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - erreur dans le code de l'équipe au chargement
        traceback.print_exc()
        print(f"\nLe chargement de « {args.module} » a échoué (voir ci-dessus) : corrigez cette erreur, "
              "ou protégez les calculs du module par « if __name__ == '__main__': ».")
        return 2
    simuler = getattr(module, "simuler", None)
    if not callable(simuler):
        print(f"« {args.module} » ne définit pas de fonction simuler(G, origine, lam).")
        return 2
    construire_graphe = getattr(module, "construire_graphe", None)
    if not callable(construire_graphe):
        construire_graphe = None
    return 0 if verifier(simuler, construire_graphe) else 1


if __name__ == "__main__":
    sys.exit(main())
