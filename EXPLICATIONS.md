# Modélisation, Prévention et Gestion du Risque Systémique en Réseaux Interbancaires

---

## 1. Contexte et Problématique Métier

Les crises financières globales ont démontré la vulnérabilité des réseaux interbancaires : la défaillance d'un seul établissement peut se propager par un effet domino à l'ensemble du système. Ce phénomène, appelé **contagion systémique**, découle des prêts et expositions croisées entre banques.

Lorsqu'une banque initiale fait défaut :
1. Ses créanciers subissent une perte directe proportionnelle à leur exposition ($\lambda$).
2. Si cette perte dépasse le **capital de réserve** d'une banque créancière, celle-ci fait à son tour faillite.
3. Le choc se propage de manière récursive dans le réseau.

### Le Problème Réglé
Ce projet répond à deux questions fondamentales pour un **régulateur financier ou un État** disposant d'un budget d'intervention limité ($X$) :
1. **Comment prédire l'ampleur d'une cascade d'effondrement** à partir de la seule structure du réseau et des caractéristiques d'une banque ?
2. **Comment allouer de manière optimale des fonds publics** (sous forme de fonds propres ou de garanties) pour minimiser le nombre total de banqueroutes, soit en réaction à un choc identifié, soit à titre préventif sous incertitude ?

---

## 2. Architecture Stratégique & Répartition Fonctionnelle

Chaque module du projet répond à une phase précise de la gestion du risque systémique :

```
                                 ┌───────────────────────────┐
                                 │   BANQUES & EXPOSITIONS   │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │    mon_simulateur.py      │
                                 │  (Moteur de Contagion)    │
                                 └─────────────┬─────────────┘
                                               │
          ┌────────────────────────────────────┼────────────────────────────────────┐
          ▼                                    ▼                                    ▼
┌──────────────────┐                 ┌──────────────────┐                 ┌──────────────────┐
│   sauvetage.py   │                 │  subvention.py   │                 │scenarii_generator│
│   (Problème P1)  │                 │ (Problèmes P2/P3)│                 │ + ML_for_sceanrii│
└─────────┬────────┘                 └─────────┬────────┘                 └─────────┬────────┘
          │                                    │                                    │
          ▼                                    ▼                                    ▼
  Gestion de crise                  Prévention sous incertitude          Prédiction rapide
  (Ciblage direct)                  (Politique globale)                  (Machine Learning)
```

### Détail des Composants

| Module | Rôle Fonctionnel | Problème Métier Résolu |
| :--- | :--- | :--- |
| **`mon_simulateur.py`** | Moteur d'évaluation du risque de base | Modélise le réseau sous forme de graphe orienté et simule la propagation exacte de la cascade de défauts selon le modèle à seuil. |
| **`sauvetage.py`** | Intervention ciblée d'urgence (**P1**) | **Sauvetage post-crise** : Sachant *quelle* banque s'effondre, comment répartir le budget $X$ pour couper les chaînes de contagion et sauver le plus de banques possible ? |
| **`subvention.py`** | Politique de prévention globale (**P2 / P3**) | **Subvention préventive** : *Avant* qu'une crise ne survienne, quelle allocation de capital renforce au mieux la résilience globale du réseau face à n'importe quel choc ? |
| **`scenarii_generator.py`** | Générateur d'observables & scénarios | Extrait la topologie des réseaux (degré, levier, connectivité, portées) et simule des milliers de crises pour constituer un jeu de données d'apprentissage. |
| **`ML_for_sceanrii.py`** | Modèle prédictif & Analyse causale | Prédit instantanément la sévérité d'une crise sans exécuter de simulation lourde, et identifie les indicateurs structurels les plus critiques. |

---

## 3. Justification des Choix de Modélisation & d'Optimisation

### 3.1. Filtrage Stratégique : Banques "Isolées" vs "Contagieuses"
* **Le Constat** : Toutes les faillites ne se valent pas. La défaillance d'une banque périphérique n'entraîne parfois aucun effet domino (cascade = 1).
* **Le Choix Métier** : Dans les modules d'optimisation (`sauvetage.py` et `subvention.py`), nous **excluons systématiquement les banques non-contagieuses** des calculs de rentabilité d'injection de capital. Injecter du budget pour sauver une banque qui n'a aucun impact systémique constitue un mauvais usage des deniers publics. Nous concentrons l'effort là où l'effet de levier sur la contagion est maximal.

### 3.2. P1 : Sauvetage Reconstructif (Post-choc)
* **Objectif** : Minimiser la taille totale de la cascade connaissant la banque d'origine.
* **Approche** : Utilisation d'un algorithme d'**Évolution Différentielle** (`scipy.optimize.differential_evolution`). 
* **Pourquoi ce choix ?** Le problème de propagation sur un réseau avec seuil est hautement non-linéaire, non-convexe et discontinu (effet de palier : un euro de plus sauve une banque, un euro de moins la laisse couler). Les algorithmes basés sur le gradient échouent ; l'optimisation stochastique globale est indispensable.

### 3.3. P2 vs P3 : Arbitrage entre Espérance et Robustesse
En prévention (avant la crise), l'État ne sait pas quelle banque va défaillir. Deux philosophies s'affrontent :

* **P2 (L'approche moyenne)** : Minimise le taux moyen de faillites sur l'ensemble des origines contagieuses possibles.
  $$\min \mathbb{E}[\text{défauts}]$$
* **P3 (L'approche robuste / Pire-cas)** : Intègre la variance des scénarios avec un paramètre de pénalisation $\alpha = 2.0$.
  $$\min \left( \mathbb{E}[\text{défauts}] + \alpha \cdot \sigma(\text{défauts}) \right)$$
* **Pourquoi ce choix ?** L'approche P2 peut accepter de laisser s'effondrer le système dans un scénario extrême si cela améliore la moyenne globale. L'approche P3 force l'allocation à sur-protéger les banques "pivots" afin d'éviter les catastrophes systémiques majeures, quitte à sacrifier un peu d'efficacité moyenne.

---

## 4. Stratégie de Machine Learning & Sélection de Variables

Plutôt que d'exécuter des simulations de cascades complexes (coûteuses en temps de calcul dans un contexte de décision rapide), nous entraînons des modèles de régression (**Arbres de Décision** et **Forêts Aléatoires**) pour estimer directement le **taux de banqueroute final**.

### 4.1. Validation Croisée par Réseau (Leave-One-Group-Out)
* **Problème** : Si un modèle apprend sur des banques d'un réseau $A$ et est évalué sur d'autres banques du même réseau $A$, il y a fuite d'information.
* **Choix Méthodologique** : Utilisation stricte de la validation croisée **Grouped (LOGO)** par `reseau_id`. Les modèles sont évalués sur leur capacité à **généraliser la prédiction à un réseau financier entièrement inconnu**.

### 4.2. Algorithme de Sélection avec *Lookahead* (Anticipation)
Les descripteurs de graphes (connectivité, leviers, ratios de capital) souffrent de forte **multicolinéarité**. Une sélection pas-à-pas classique (*Forward Selection* ou *Backward Elimination*) peut s'arrêter prématurément dès qu'une variable seule n'apporte pas d'amélioration.

```
Étape actuelle ────► [Ajout variable X] ───► Pas d'amélioration (Stagnation)
                         │
                         ▼ (Lookahead : Test du cran suivant)
                     [Ajout variable X + Y] ───► Forte baisse de l'erreur ! (Conservation de X et Y)
```

* **Le Choix de l'Algorithme** : Nous avons implémenté une tolérance d'un cran (*Lookahead*). Si l'ajout d'une variable seule stagne, l'algorithme teste temporairement les combinaisons à deux variables. Cela permet de franchir les plateaux causés par la redondance des indicateurs et de retenir le sous-ensemble de variables le plus sobre et explicatif.

---

## 5. Synthèse des Résultats & Retombées Métier

1. **Seuil d'efficacité du budget public** : Les visualisations (heatmaps et courbes de sensibilité) démontrent l'existence de **paliers budgétaires**. Au-delà d'un certain pourcentage du capital du réseau injecté ($X$), le gain marginal devient nul. Le système est stabilisé.
2. **Priorisation des indicateurs** : Les analyses de sélection de variables révèlent que les caractéristiques individuelles de la banque (portée maximale atteinte, degré sortant, pression relative sur le capital médian) prédisent le risque systémique avec une précision équivalente aux simulations lourdes.
3. **Comportement des politiques P2 vs P3** : L'allocation P3 tend à lisser la vulnérabilité globale du réseau, évitant la formation de "maillons faibles" systémiques, ce qui est la stratégie recommandée pour la régulation prudentielle (Bâle III / Bâle IV).