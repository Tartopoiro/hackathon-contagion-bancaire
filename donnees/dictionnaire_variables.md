# Dictionnaire des variables

Tous les fichiers sont au format CSV : séparateur « , », encodage UTF-8, point décimal, une ligne d'en-tête. Les montants sont exprimés dans une même unité monétaire fictive, au centime près.

## Convention essentielle : le sens d'une dette

Chaque banque est un sommet du graphe. Une ligne `debiteur_id = i`, `creancier_id = j`, `montant = e_ij` de `expositions.csv` correspond à l'**arête orientée i → j** : **la banque i doit e_ij à la banque j**. Si i fait défaut, c'est j qui perd de l'argent. Les pertes suivent donc le sens des flèches.

## `banques.csv` : une ligne par banque et par réseau

| Colonne | Type | Description | Contraintes |
|---|---|---|---|
| `reseau_id` | texte | identifiant du réseau (`R01` … `R12`) | 12 réseaux indépendants |
| `banque_id` | texte | identifiant de la banque dans son réseau (`B01` … `B60`) | unique dans un réseau ; les mêmes identifiants se retrouvent d'un réseau à l'autre, mais désignent des banques différentes |
| `capital` | réel | capital c_j : montant de pertes que la banque peut absorber | strictement positif |

Chaque réseau compte 60 banques. Quelques banques n'ont ni dette ni créance : ce sont des **banques isolées**. Il faut les garder dans le graphe, car elles comptent dans N.

## `expositions.csv` : une ligne par dette

| Colonne | Type | Description | Contraintes |
|---|---|---|---|
| `reseau_id` | texte | réseau de la dette | existe dans `banques.csv` |
| `debiteur_id` | texte | banque i qui **doit** le montant | existe dans le même réseau |
| `creancier_id` | texte | banque j à qui le montant est **dû** | existe dans le même réseau ; différente du débiteur |
| `montant` | réel | e_ij | strictement positif |

Il n'y a jamais deux lignes pour la même paire orientée (i, j). En revanche, i peut devoir à j et j devoir à i : ce sont deux arêtes distinctes.

## `repartition.csv` : une ligne par réseau

| Colonne | Type | Description |
|---|---|---|
| `reseau_id` | texte | réseau |
| `usage` | texte | `apprentissage` (8 réseaux), `validation` (2) ou `test` (2) |

Tous les scénarios d'un réseau restent dans le même ensemble.

## `tirages_aleatoires.csv` : les 10 sélections aléatoires de l'intervention

| Colonne | Type | Description |
|---|---|---|
| `reseau_id` | texte | réseau |
| `graine` | entier | 0 à 9 |
| `ordre` | entier | 1 à 5 (ordre du tirage) |
| `banque_id` | texte | banque tirée au sort, sans remise |

Règle de tirage, reproduite par `aides_contagion.tirer_banques_aleatoires` : `rng = numpy.random.default_rng([graine, numero_reseau])`, puis `rng.choice(60, size=5, replace=False)` appliqué aux identifiants triés (`numero_reseau` vaut 7 pour `R07`).

## `exemples/` : exemples pédagogiques (exclus de l'apprentissage, de la validation et du test)

* `abc_banques.csv`, `abc_expositions.csv` : réseau `ABC`. A doit 10 à B, B doit 8 à C ; c_A = 5, c_B = 4, c_C = 3. Avec λ = 0,5 et l'origine A, la cascade touche les trois banques, avec 1 nouveau défaut à chaque étape.
* `relais_banques.csv`, `relais_expositions.csv` : réseau `RELAIS` à 12 banques (bloc Nord N1…N5, relais R, bloc Sud S1…S6). Il montre comment renforcer une banque intermédiaire peut arrêter une cascade.

## `brutes/` : fichiers d'entraînement aux contrôles

`banques_brutes.csv` et `expositions_brutes.csv` reprennent la base en y introduisant volontairement des anomalies : doublons, valeurs manquantes, montants ou capitaux non positifs, dettes d'une banque envers elle-même, banques inconnues, valeurs non numériques. Ils servent uniquement à écrire et tester vos contrôles. **Travaillez ensuite sur les fichiers propres.**

## Grandeurs à construire (rappel du sujet)

| Grandeur | Définition |
|---|---|
| degré entrant d_in(j) | nombre de débiteurs de j (arêtes qui arrivent en j) |
| degré sortant d_out(j) | nombre de créanciers de j (arêtes qui partent de j) |
| poids entrant w_in(j) = Σ_i e_ij | créances détenues par j |
| poids sortant w_out(j) = Σ_k e_jk | dettes de j envers les autres banques |
| fragilité v_j = w_in(j) / c_j | créances rapportées au capital |
| y = (\|D_final\| − 1)/(N − 1) | proportion de défauts secondaires d'un scénario (N = 60) |
