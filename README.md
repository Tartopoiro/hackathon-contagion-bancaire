# Hackathon Data Science & Risque : réseaux bancaires et contagion

Université de Bordeaux, du 25 septembre au 23 octobre 2026.

Ce dépôt contient tout ce dont votre équipe a besoin : le sujet, le cours, les données et les aides Python. Lisez cette page en entier avant de commencer (5 minutes).

## 1. Télécharger le dépôt

Choisissez **une** des trois méthodes :

| Méthode | Comment faire |
|---|---|
| Sans rien installer | Bouton vert **Code**, puis **Download ZIP**, puis décompressez le dossier. |
| Avec Git | `git clone https://github.com/marioveruetephd/hackathon-contagion-bancaire.git` ; plus tard, `git pull` récupère les mises à jour. |
| Google Colab | Dans une cellule : `!git clone https://github.com/marioveruetephd/hackathon-contagion-bancaire.git`, puis `%cd hackathon-contagion-bancaire` |

Des fichiers pourront être ajoutés pendant le hackathon : pensez à mettre votre copie à jour avant chaque séance.

## 2. Ce que contient le dépôt

| Dossier ou fichier | Contenu |
|---|---|
| `sujet/` | le sujet du hackathon (PDF) |
| `cours/` | le cours du 25 septembre (PDF) : notions, règle de la cascade, protocole |
| `donnees/banques.csv` | une ligne par banque : `reseau_id`, `banque_id`, `capital` |
| `donnees/expositions.csv` | une ligne par dette : `reseau_id`, `debiteur_id`, `creancier_id`, `montant` |
| `donnees/repartition.csv` | l'usage de chaque réseau : apprentissage, validation ou test |
| `donnees/tirages_aleatoires.csv` | les 10 tirages aléatoires officiels pour l'intervention |
| `donnees/dictionnaire_variables.md` | la définition précise de chaque colonne |
| `donnees/exemples/` | les deux petits exemples du cours : A-B-C (3 banques) et RELAIS (12 banques) |
| `donnees/brutes/` | des copies volontairement abîmées, pour vous entraîner aux contrôles de données |
| `aides/` | les aides Python fournies (à utiliser, sans les modifier) |
| `mon_simulateur.py` | **le fichier que votre équipe complète** |
| `requirements.txt` | les bibliothèques Python nécessaires |

Les données contiennent 12 réseaux de 60 banques (720 banques et 2 134 dettes). Tous les montants sont dans une même unité monétaire fictive.

## 3. Installer les bibliothèques

Python 3.10 ou plus récent. Dans un terminal, depuis le dossier du dépôt :

```bash
pip install -r requirements.txt
```

Sur Google Colab, ces bibliothèques sont déjà installées.

## 4. La règle à ne jamais oublier

Une ligne de `expositions.csv` avec `debiteur_id = i`, `creancier_id = j` et `montant = e` signifie : **la banque i doit e à la banque j**. Dans le graphe, c'est la flèche **i → j**. Si i fait défaut, c'est j qui perd de l'argent : les pertes suivent le sens des flèches.

## 5. Par où commencer

1. Lisez le sujet (`sujet/`) et gardez le cours (`cours/`) sous la main.
2. Chargez les données :

   ```python
   import pandas as pd
   banques = pd.read_csv("donnees/banques.csv")
   expositions = pd.read_csv("donnees/expositions.csv")
   repartition = pd.read_csv("donnees/repartition.csv")
   print(len(banques), len(expositions))   # 720 2134
   ```

3. Écrivez vos contrôles de données, puis testez-les sur les fichiers de `donnees/brutes/`.
4. Construisez le graphe d'un réseau d'apprentissage avec NetworkX et faites une première figure.
5. Complétez `mon_simulateur.py`, puis vérifiez-le (section 6).

## 6. Vérifier votre simulateur

Dans un terminal, depuis le dossier du dépôt :

```bash
python aides/verifier_simulateur.py mon_simulateur.py
```

Ou dans un notebook placé à la racine du dépôt :

```python
import sys
sys.path.append("aides")
from verifier_simulateur import verifier
from mon_simulateur import construire_graphe, simuler
verifier(simuler, construire_graphe)
```

Les 9 tests reprennent les vérifications du sujet : l'exemple A-B-C, l'absence de contagion quand elle est impossible, la perte égale au capital, les défauts simultanés, la monotonie, le sens des arêtes, l'absence de double comptage, le défaut imposé de l'origine avec les banques isolées, et l'arrêt de la cascade. Chaque échec est accompagné d'une piste de correction.

## 7. Les aides fournies (`aides/aides_contagion.py`)

| Fonction | À quoi elle sert |
|---|---|
| `depasse_capital(perte, capital)` | tester « perte > capital » sans erreur d'arrondi |
| `generer_scenarios(...)` | calculer les 2 160 scénarios avec **vos** fonctions |
| `references_medianes(...)`, `predire_references(...)` | les deux références du sujet (médianes) |
| `capitaux_renforces(G, choisies)` | appliquer le budget d'intervention (10 % du capital, réparti entre les banques choisies) |
| `charger_tirages(...)` | lire les 10 tirages aléatoires officiels |
| `comparer_interventions(...)`, `resumer_interventions(...)` | comparer des stratégies sur les mêmes scénarios |

Pour la documentation détaillée d'une aide, avec des exemples : `help(generer_scenarios)`.

## 8. Règles importantes

- Tous les scénarios d'un réseau restent dans le même ensemble : utilisez `repartition.csv`, jamais un partage aléatoire des lignes.
- Les variables explicatives se calculent sur le réseau **avant** le choc. Les identifiants et tout résultat de la cascade (par exemple le nombre de défauts à la première étape) sont interdits.
- Réglez vos choix sur la validation. Le test ne sert qu'une fois, à la fin.
- Interventions : le budget vaut 10 % du capital total initial, réparti à parts égales entre 5 banques choisies une seule fois, avant les chocs. Le défaut initial reste toujours imposé.

## 9. Calendrier

| Date | Ce qui doit être prêt |
|---|---|
| 25 septembre | données chargées, sens des liens expliqué, première figure |
| 7 octobre | simulateur validé (9 tests sur 9), table des scénarios lancée |
| 9 octobre | références et modèle évalués sur la validation, premier essai d'intervention |
| 23 octobre | projet et présentation remis en fin de matinée (8 minutes de présentation, 4 minutes d'échange) |

Le rendu final comprend un notebook exécutable du début à la fin, un court fichier de présentation (auteurs, instructions, dépendances, graines, hypothèses), les figures demandées et cinq à six diapositives. Le détail est dans le sujet.

Le simulateur est un mécanisme pédagogique simplifié : les résultats ne constituent pas des prévisions sur des banques réelles.
