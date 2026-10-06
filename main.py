"""Pipeline de chargement, construction et visualisation d'un réseau bancaire."""

from pathlib import Path
import argparse

import pandas as pd

from mon_simulateur import construire_graphe
from IHM_graphe import lancer_interface


def charger_donnees(dossier):
	"""Charge les tables CSV du dossier de données."""
	dossier = Path(dossier)
	banques = pd.read_csv(dossier / "banques.csv")
	expositions = pd.read_csv(dossier / "expositions.csv")
	return banques, expositions


def construire_graphe_reseau(banques, expositions, reseau_id):
	"""Filtre un réseau puis construit son graphe orienté pondéré."""
	banques_r = banques[banques["reseau_id"] == reseau_id]
	expositions_r = expositions[expositions["reseau_id"] == reseau_id]

	if banques_r.empty:
		raise ValueError(f"Réseau introuvable : {reseau_id}")
	return construire_graphe(banques_r, expositions_r)


def main():
	parser = argparse.ArgumentParser(description="Visualise un réseau de contagion bancaire.")
	parser.add_argument(
		"--reseau",
		default=None,
		help="identifiant du réseau (par défaut : le premier réseau disponible)",
	)
	parser.add_argument(
		"--donnees",
		type=Path,
		default=Path(__file__).parent / "donnees",
		help="dossier contenant banques.csv et expositions.csv",
	)
	args = parser.parse_args()

	banques, expositions = charger_donnees(args.donnees)
	reseau_id = args.reseau or banques["reseau_id"].iloc[0]
	graphe = construire_graphe_reseau(banques, expositions, reseau_id)
	print(f"Réseau {reseau_id} : {graphe.number_of_nodes()} banques, "
		  f"{graphe.number_of_edges()} expositions")
	lancer_interface(graphe)


if __name__ == "__main__":
	main()
