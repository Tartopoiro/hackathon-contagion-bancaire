import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from mon_simulateur import construire_graphe, simuler, data_loading

def filtrer_origines_contagieuses(G_base, lam=0.5):
    """
    Exclut les banques isolées (cascade == 1) pour ne conserver
    que les origines déclenchant de la contagion (cascade > 1).
    """
    nodes_contagieux = []
    for orig in G_base.nodes():
        defauts = simuler(G_base, orig, lam)[0]
        if len(defauts) > 1:
            nodes_contagieux.append(orig)
    return nodes_contagieux

def optimiser_p2(G_base, origines, budget_X, lam=0.5, seed=42):
    """
    P2 : Minimise le taux moyen de défauts (%) sur les origines contagieuses.
    """
    nodelist = list(G_base.nodes())
    N = len(nodelist)

    if not origines:
        return {}, 0.0, 0.0

    def fonction_objectif(poids):
        somme = np.sum(poids)
        allocations = budget_X * (poids / somme) if somme > 0 else np.zeros(N)
        
        G_temp = G_base.copy()
        for i, noeud in enumerate(nodelist):
            G_temp.nodes[noeud]["capital"] += allocations[i]
            
        defauts_totaux = [len(simuler(G_temp, orig, lam)[0]) for orig in origines]
        return np.mean(defauts_totaux)

    bornes = [(0, 1) for _ in range(N)]
    resultat = differential_evolution(
        fonction_objectif, bounds=bornes, popsize=10, maxiter=30, tol=0.05, seed=seed
    )
    
    meilleure_allocation = budget_X * (resultat.x / np.sum(resultat.x))
    
    G_final = G_base.copy()
    for i, noeud in enumerate(nodelist):
        G_final.nodes[noeud]["capital"] += meilleure_allocation[i]
        
    defauts_finaux = [len(simuler(G_final, orig, lam)[0]) for orig in origines]
    
    # Conversion en pourcentage du réseau N
    moyenne_pct = (np.mean(defauts_finaux) / N) * 100
    std_pct = (np.std(defauts_finaux) / N) * 100
    
    repartition = {nodelist[i]: round(meilleure_allocation[i], 2) for i in range(N)}
    return repartition, moyenne_pct, std_pct

def optimiser_p3(G_base, origines, budget_X, lam=0.5, alpha=2.0, seed=42):
    """
    P3 : Minimise (Moyenne + alpha * Std) en % du réseau sur les origines contagieuses.
    """
    nodelist = list(G_base.nodes())
    N = len(nodelist)

    if not origines:
        return {}, 0.0, 0.0

    def fonction_objectif(poids):
        somme = np.sum(poids)
        allocations = budget_X * (poids / somme) if somme > 0 else np.zeros(N)
        
        G_temp = G_base.copy()
        for i, noeud in enumerate(nodelist):
            G_temp.nodes[noeud]["capital"] += allocations[i]
            
        defauts_totaux = [len(simuler(G_temp, orig, lam)[0]) for orig in origines]
        return np.mean(defauts_totaux) + alpha * np.std(defauts_totaux)

    bornes = [(0, 1) for _ in range(N)]
    resultat = differential_evolution(
        fonction_objectif, bounds=bornes, popsize=10, maxiter=30, tol=0.05, seed=seed
    )
    
    meilleure_allocation = budget_X * (resultat.x / np.sum(resultat.x))
    
    G_final = G_base.copy()
    for i, noeud in enumerate(nodelist):
        G_final.nodes[noeud]["capital"] += meilleure_allocation[i]
        
    defauts_finaux = [len(simuler(G_final, orig, lam)[0]) for orig in origines]
    
    moyenne_pct = (np.mean(defauts_finaux) / N) * 100
    std_pct = (np.std(defauts_finaux) / N) * 100
    
    repartition = {nodelist[i]: round(meilleure_allocation[i], 2) for i in range(N)}
    return repartition, moyenne_pct, std_pct

def _worker_p2_p3(args):
    """Worker multiprocessing pour un palier de budget."""
    r_idx, G_base, origines, X, lam, j_budget, seed = args
    _, moy_p2_pct, std_p2_pct = optimiser_p2(G_base, origines, X, lam, seed=seed)
    _, moy_p3_pct, std_p3_pct = optimiser_p3(G_base, origines, X, lam, alpha=2.0, seed=seed)
    return r_idx, j_budget, moy_p2_pct, std_p2_pct, moy_p3_pct, std_p3_pct

def generer_plots_12_reseaux_pct(banques_df, expositions_df, etapes=8, ratio_capital_max=0.5, lam=0.5):
    """
    Génère la grille 3x4 comparant P2 et P3 en pourcentage de défauts du réseau.
    """
    reseau_ids = banques_df["reseau_id"].unique()[:12]
    
    donnees_reseaux = []
    taches = []

    print("Filtrage des banques isolées et préparation des 12 réseaux...")
    for r_idx, r_id in enumerate(reseau_ids):
        banques_r = banques_df[banques_df["reseau_id"] == r_id]
        expos_r = expositions_df[expositions_df["reseau_id"] == r_id]
        G_r = construire_graphe(banques_r, expos_r)
        
        origines_contagieuses = filtrer_origines_contagieuses(G_r, lam=lam)
        capital_total = banques_r["capital"].sum()
        budget_max_r = capital_total * ratio_capital_max
        budgets_X = np.linspace(0, budget_max_r, etapes)
        
        donnees_reseaux.append({
            "id": r_id,
            "N": len(G_r.nodes()),
            "nb_contagieuses": len(origines_contagieuses),
            "capital_total": capital_total,
            "budgets_X": budgets_X,
            "moy_p2": np.zeros(etapes),
            "std_p2": np.zeros(etapes),
            "moy_p3": np.zeros(etapes),
            "std_p3": np.zeros(etapes)
        })
        
        if origines_contagieuses:
            for j, X in enumerate(budgets_X):
                seed_task = 42 + r_idx * 100 + j
                taches.append((r_idx, G_r, origines_contagieuses, X, lam, j, seed_task))

    if taches:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(_worker_p2_p3, t): t for t in taches}
            for future in tqdm(as_completed(futures), total=len(taches), desc="Calcul P2/P3 en % (12 réseaux)", unit="étape"):
                r_idx, j, moy_p2_pct, std_p2_pct, moy_p3_pct, std_p3_pct = future.result()
                donnees_reseaux[r_idx]["moy_p2"][j] = moy_p2_pct
                donnees_reseaux[r_idx]["std_p2"][j] = std_p2_pct
                donnees_reseaux[r_idx]["moy_p3"][j] = moy_p3_pct
                donnees_reseaux[r_idx]["std_p3"][j] = std_p3_pct

    # Tracé de la grille 3x4
    fig, axes = plt.subplots(3, 4, figsize=(22, 13))
    axes = axes.flatten()

    for r_idx, data in enumerate(donnees_reseaux):
        ax = axes[r_idx]
        
        if data["nb_contagieuses"] == 0:
            ax.text(0.5, 0.5, "Aucune origine contagieuse\n(Taux de défaut irréductible)", ha='center', va='center', fontsize=10)
            ax.set_title(f"Réseau {data['id']}", fontsize=10, fontweight='bold')
            continue

        X_pct = np.linspace(0, ratio_capital_max * 100, etapes)
        moy_p2 = data["moy_p2"]
        std_p2 = data["std_p2"]
        moy_p3 = data["moy_p3"]
        std_p3 = data["std_p3"]

        # P2 (Bleu)
        ax.plot(X_pct, moy_p2, label="P2 : Moyenne (%)", marker='o', color='#1f77b4', linewidth=1.8, markersize=4)
        ax.fill_between(X_pct, 
                        np.maximum(0, moy_p2 - std_p2), 
                        np.minimum(100, moy_p2 + std_p2), 
                        color='#1f77b4', alpha=0.15, label="P2 : ±1 Std (%)")

        # P3 (Vert)
        ax.plot(X_pct, moy_p3, label="P3 : Robuste (%)", marker='s', color='#2ca02c', linewidth=1.8, markersize=4)
        ax.fill_between(X_pct, 
                        np.maximum(0, moy_p3 - std_p3), 
                        np.minimum(100, moy_p3 + std_p3), 
                        color='#2ca02c', alpha=0.2, label="P3 : ±1 Std (%)")

        ax.set_title(f"Réseau {data['id']} (N={data['N']}, Contagieuses={data['nb_contagieuses']})", fontsize=10, fontweight='bold')
        ax.set_xlabel("Budget (% cap. réseau)", fontsize=8)
        ax.set_ylabel("Taux de défaut (% du réseau)", fontsize=8)
        ax.set_ylim(-2, 102)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=7, loc='upper right')

    plt.suptitle("Taux de défaut du réseau (%) — Comparaison P2 vs P3 (Origines contagieuses)", fontsize=15, y=0.99)
    plt.tight_layout()
    
    nom_fichier = "plot_p2_p3_12_reseaux_pct.png"
    plt.savefig(nom_fichier, dpi=300)
    print(f"\nGraphique sauvegardé sous '{nom_fichier}'.")
    plt.show()

if __name__ == "__main__":
    banques_df, expositions_df = data_loading()
    generer_plots_12_reseaux_pct(banques_df, expositions_df, etapes=8, ratio_capital_max=0.5, lam=0.5)