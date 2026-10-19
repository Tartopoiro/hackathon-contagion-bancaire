import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from mon_simulateur import construire_graphe, simuler, data_loading

def optimiser_p1(G_base, origine_connue, budget_X, lam=0.5, seed=42):
    """Résout P1 pour une origine donnée de manière déterministe."""
    nodelist = list(G_base.nodes())
    N = len(nodelist)

    def fonction_objectif(poids):
        somme = np.sum(poids)
        allocations = budget_X * (poids / somme) if somme > 0 else np.zeros(N)
        
        G_temp = G_base.copy()
        for i, noeud in enumerate(nodelist):
            G_temp.nodes[noeud]["capital"] += allocations[i]
            
        defauts, _ = simuler(G_temp, origine_connue, lam)
        return len(defauts)

    bornes = [(0, 1) for _ in range(N)]
    resultat = differential_evolution(
        fonction_objectif, bounds=bornes, popsize=8, maxiter=20, tol=0.05, seed=seed
    )
    return fonction_objectif(resultat.x)

def _worker_task(args):
    """Worker multiprocessing pour traiter un triplet (réseau, origine, budget)."""
    reseau_id, G_base, origine, X, lam, r_idx, i_bank, j_budget = args
    if X == 0:
        defauts = len(simuler(G_base, origine, lam)[0])
    else:
        defauts = optimiser_p1(G_base, origine, X, lam)
    return r_idx, i_bank, j_budget, defauts

def generer_12_heatmaps_cascade_pct(banques_df, expositions_df, etapes=8, ratio_capital_max=0.5, lam=0.5):
    """
    Génère les 12 heatmaps exprimées en % de la cascade évitée, 
    en filtrant les banques qui n'impactent qu'elles-mêmes.
    """
    reseau_ids = banques_df["reseau_id"].unique()[:12]
    
    donnees_reseaux = []
    taches = []

    print("Préparation et filtrage des banques contagieuses (défauts de base > 1)...")
    for r_idx, r_id in enumerate(reseau_ids):
        banques_r = banques_df[banques_df["reseau_id"] == r_id]
        expos_r = expositions_df[expositions_df["reseau_id"] == r_id]
        G_r = construire_graphe(banques_r, expos_r)
        
        # Identification et filtrage des banques qui propagent la contagion
        nodes_contagieux = []
        defauts_base_list = []
        
        for orig in G_r.nodes():
            d_base = len(simuler(G_r, orig, lam)[0])
            if d_base > 1:  # Exclut les banques qui ne font couler qu'elles-mêmes
                nodes_contagieux.append(orig)
                defauts_base_list.append(d_base)
        
        N_filtre = len(nodes_contagieux)
        capital_total = banques_r["capital"].sum()
        budget_max_r = capital_total * ratio_capital_max
        budgets_X = np.linspace(0, budget_max_r, etapes)
        
        if N_filtre > 0:
            Z_sans = np.zeros((N_filtre, etapes))
            for i, orig in enumerate(nodes_contagieux):
                d_base = defauts_base_list[i]
                Z_sans[i, :] = d_base
                for j, X in enumerate(budgets_X):
                    taches.append((r_id, G_r, orig, X, lam, r_idx, i, j))
        else:
            Z_sans = np.zeros((0, etapes))

        donnees_reseaux.append({
            "id": r_id,
            "nodes": nodes_contagieux,
            "budgets_X": budgets_X,
            "capital_total": capital_total,
            "Z_sans": Z_sans,
            "Z_avec": np.zeros((N_filtre, etapes))
        })

    # Exécution parallèle
    if taches:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(_worker_task, t): t for t in taches}
            for future in tqdm(as_completed(futures), total=len(taches), desc="Calcul P1 (% Cascade Évitée)", unit="task"):
                r_idx, i, j, defauts = future.result()
                donnees_reseaux[r_idx]["Z_avec"][i, j] = defauts

    # Tracé de la grille 3x4
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for r_idx, data in enumerate(donnees_reseaux):
        ax = axes[r_idx]
        N_filtre = len(data["nodes"])
        
        if N_filtre == 0:
            ax.text(0.5, 0.5, "Aucune contagion\n(défauts isolés uniquement)", ha='center', va='center', fontsize=10)
            ax.set_title(f"Réseau {data['id']}", fontsize=10, fontweight='bold')
            continue

        # Pourcentage de la cascade évitée par rapport aux défaillantes de base
        Z_gain_pct = ((data["Z_sans"] - data["Z_avec"]) / data["Z_sans"]) * 100
        
        X_pct = np.linspace(0, ratio_capital_max * 100, etapes)
        X_grid, Y_grid = np.meshgrid(X_pct, range(N_filtre))
        
        c = ax.pcolormesh(X_grid, Y_grid, Z_gain_pct, cmap='YlGn', shading='auto', vmin=0, vmax=100)
        fig.colorbar(c, ax=ax, label="% cascade évitée")
        
        ax.set_title(f"Réseau {data['id']} (Cap: {data['capital_total']:,.0f}€)", fontsize=10, fontweight='bold')
        ax.set_xlabel("Budget (% cap. réseau)", fontsize=8)
        ax.set_ylabel("Banque contagieuse", fontsize=8)
        ax.set_yticks(range(N_filtre))
        ax.set_yticklabels([str(n) for n in data["nodes"]], fontsize=7)

    plt.suptitle("Pourcentage de la cascade évitée par l'État (P1) — Banques contagieuses uniquement", fontsize=15, y=0.99)
    plt.tight_layout()
    
    nom_fichier = "heatmaps_12_reseaux_cascade_pct.png"
    plt.savefig(nom_fichier, dpi=300)
    print(f"\nGraphique sauvegardé sous '{nom_fichier}'.")
    plt.show()

if __name__ == "__main__":
    banques_df, expositions_df = data_loading()
    generer_12_heatmaps_cascade_pct(banques_df, expositions_df, etapes=8, ratio_capital_max=0.5, lam=0.5)