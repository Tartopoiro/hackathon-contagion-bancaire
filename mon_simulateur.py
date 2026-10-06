"""Votre simulateur de cascade : c'est le fichier que votre équipe complète.

Écrivez les deux fonctions ci-dessous, puis vérifiez-les depuis le dossier du dépôt :

    python aides/verifier_simulateur.py mon_simulateur.py

Le vérificateur lance les 9 tests du sujet et explique chaque échec.
Ne changez pas le nom des fonctions ni leurs arguments : les aides les appellent ainsi.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import networkx as nx


# Rend le dossier aides/ importable, que ce fichier soit lancé ou importé depuis un notebook.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "aides"))
from aides_contagion import depasse_capital  # noqa: E402


def construire_graphe(banques_r, expositions_r):
    """Construit le graphe orienté pondéré d'UN réseau.

    banques_r : lignes de banques.csv pour ce réseau (colonnes banque_id, capital).
    expositions_r : lignes de expositions.csv pour ce réseau (colonnes debiteur_id, creancier_id, montant).

    Renvoie un nx.DiGraph avec :
      * un sommet par banque, avec l'attribut "capital", y compris les banques isolées ;
      * une arête debiteur -> creancier par dette, avec l'attribut "montant"
        (la banque i doit le montant à la banque j : si i fait défaut, j perd).
    """
    G = nx.DiGraph()
    for banque in banques_r.itertuples(index=False):
      G.add_node(banque.banque_id, capital=banque.capital)
    for exposition in expositions_r.itertuples(index=False):
      G.add_edge(exposition.debiteur_id, exposition.creancier_id, montant=exposition.montant)
    return G


def simuler(G, origine, lam):
    """Simule la cascade à seuil du sujet (section 3) à partir de la banque d'origine.

    G : graphe renvoyé par construire_graphe (ne le modifiez pas).
    origine : identifiant de la banque en défaut au départ (son défaut est imposé).
    lam : fraction de perte, entre 0 et 1.

    Renvoie le couple (defauts, nouveaux_par_etape) :
      * defauts : ensemble (set) des banques en défaut à la fin, origine comprise ;
      * nouveaux_par_etape : liste du nombre de nouveaux défauts aux étapes 1, 2, 3...,
        sans l'origine et sans zéro final ([] si rien ne se propage).

    Pour comparer une perte au capital, utilisez depasse_capital(perte, capital) :
    une perte exactement égale au capital ne provoque pas de défaut.
    """
    defauts = {origine}
    nouveaux_par_etape = []

    while True:
      pertes = {banque: 0.0 for banque in G.nodes if banque not in defauts}
      for debiteur in defauts:
        for creancier in G.successors(debiteur):
          if creancier not in defauts:
            pertes[creancier] += lam * G[debiteur][creancier]["montant"]

      nouveaux = {
        banque
        for banque, perte in pertes.items()
        if depasse_capital(perte, G.nodes[banque]["capital"])
      }
      if not nouveaux:
        break
      defauts.update(nouveaux)
      nouveaux_par_etape.append(len(nouveaux))

    return defauts, nouveaux_par_etape

def data_loading():
    """Load the data from CSV files."""
    import pandas as pd
    from pathlib import Path

    ROOT = Path(__file__).parent
    banques = pd.read_csv(ROOT / "donnees" / "banques.csv")
    expositions = pd.read_csv(ROOT / "donnees" / "expositions.csv")
    return banques, expositions



if __name__ == "__main__":
    banques, expositions = data_loading()

        