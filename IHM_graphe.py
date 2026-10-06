"""Interface graphique de visualisation et de simulation d'un reseau bancaire."""

from __future__ import annotations

import networkx as nx
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
	QApplication,
	QComboBox,
	QFormLayout,
	QHBoxLayout,
	QLabel,
	QMainWindow,
	QPushButton,
	QVBoxLayout,
	QWidget,
)

from mon_simulateur import simuler


class FenetreGraphe(QMainWindow):
	"""Fenetre plein ecran pour explorer un graphe et lancer une cascade."""

	def __init__(self, graphe: nx.DiGraph):
		super().__init__()
		self.graphe = graphe
		self.defauts = set()
		self.positions = self._calculer_positions()

		self.setWindowTitle("Contagion bancaire")
		self._construire_interface()
		self._dessiner_graphe()

	def _calculer_positions(self):
		"""Ecarte les noeuds avec le layout a ressort de NetworkX."""
		nombre_noeuds = max(self.graphe.number_of_nodes(), 1)
		return nx.spring_layout(
			self.graphe,
			seed=42,
			k=max(1.2, 4.0 / nombre_noeuds**0.5),
			iterations=250,
			scale=1.0,
		)

	def _construire_interface(self):
		conteneur = QWidget()
		conteneur.setStyleSheet(
			"""
			QWidget { background: #101820; color: #e8f0f2; }
			QLabel { color: #b9cbd0; }
			QComboBox, QPushButton {
				background: #1d3038;
				border: 1px solid #3b5962;
				border-radius: 4px;
				padding: 8px 12px;
				color: #f2f7f8;
			}
			QComboBox:hover, QPushButton:hover { border-color: #54c2b5; }
			QPushButton { background: #247f76; font-weight: bold; }
			QPushButton:pressed { background: #1c625c; }
			"""
		)
		principal = QVBoxLayout(conteneur)
		principal.setContentsMargins(18, 14, 18, 14)
		principal.setSpacing(10)

		titre = QLabel("RESEAU DE CONTAGION BANCAIRE")
		titre.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
		principal.addWidget(titre)

		barre = QHBoxLayout()
		formulaire = QFormLayout()
		self.origine = QComboBox()
		self.origine.addItems(sorted((str(noeud) for noeud in self.graphe.nodes)))
		self.lambda_ = QComboBox()
		self.lambda_.addItems(["0.3", "0.6", "0.9"])
		formulaire.addRow("Banque origine", self.origine)
		formulaire.addRow("Fraction de perte lambda", self.lambda_)
		barre.addLayout(formulaire)

		self.bouton_simuler = QPushButton("Lancer la simulation")
		self.bouton_simuler.setCursor(Qt.CursorShape.PointingHandCursor)
		self.bouton_simuler.clicked.connect(self._lancer_simulation)
		barre.addWidget(self.bouton_simuler, 0, Qt.AlignmentFlag.AlignBottom)
		barre.addStretch()
		principal.addLayout(barre)

		self.canvas = FigureCanvas(Figure(facecolor="#101820"))
		principal.addWidget(self.canvas, 1)

		self.resultat = QLabel("Selectionnez une origine puis lancez la simulation.")
		self.resultat.setWordWrap(True)
		self.resultat.setStyleSheet("color: #dcebed; padding: 4px 0;")
		principal.addWidget(self.resultat)
		self.setCentralWidget(conteneur)

	def _dessiner_graphe(self):
		self.canvas.figure.clear()
		axes = self.canvas.figure.add_subplot(111)
		axes.set_facecolor("#101820")
		axes.axis("off")

		couleurs = []
		for noeud in self.graphe.nodes:
			if noeud in self.defauts:
				couleurs.append("#f08a72")
			elif noeud == self.origine.currentText():
				couleurs.append("#f5c451")
			else:
				couleurs.append("#55b8ad")

		nx.draw_networkx_edges(
			self.graphe,
			self.positions,
			ax=axes,
			edge_color="#6e858d",
			width=0.8,
			alpha=0.55,
			arrows=True,
			arrowsize=10,
			connectionstyle="arc3,rad=0.04",
		)
		nx.draw_networkx_nodes(
			self.graphe,
			self.positions,
			ax=axes,
			node_color=couleurs,
			node_size=650,
			edgecolors="#e8f0f2",
			linewidths=0.8,
		)
		nx.draw_networkx_labels(
			self.graphe,
			self.positions,
			ax=axes,
			font_size=7,
			font_weight="bold",
			font_color="#102027",
		)
		self.canvas.figure.tight_layout(pad=0.5)
		self.canvas.draw_idle()

	def _lancer_simulation(self):
		origine = self.origine.currentText()
		noeud_origine = next(noeud for noeud in self.graphe if str(noeud) == origine)
		lam = float(self.lambda_.currentText())
		self.defauts, etapes = simuler(self.graphe, noeud_origine, lam)
		propagation = ", ".join(str(nombre) for nombre in etapes) or "aucune"
		self.resultat.setText(
			f"Origine : {origine} | lambda = {lam:.1f} | "
			f"Defauts finaux : {len(self.defauts)} / {self.graphe.number_of_nodes()} | "
			f"Nouveaux defauts par etape : {propagation}"
		)
		self._dessiner_graphe()


def lancer_interface(graphe: nx.DiGraph):
	"""Lance l'interface graphique en plein ecran."""
	application = QApplication.instance() or QApplication([])
	fenetre = FenetreGraphe(graphe)
	fenetre.showFullScreen()
	fenetre.raise_()
	fenetre.activateWindow()
	return application.exec()
