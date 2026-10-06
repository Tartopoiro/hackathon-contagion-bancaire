import pandas as pd
import numpy as np
import networkx as nx
from mon_simulateur import construire_graphe, simuler, data_loading


def portee(M, o):
    """Nombre de banques atteignables depuis o en suivant les arêtes de M (booléen)."""
    atteint = np.zeros(M.shape[0], bool)
    atteint[o] = True
    while True:
        nouveau = M[atteint].any(0) & ~atteint
        if not nouveau.any():
            return atteint.sum() - 1
        atteint |= nouveau


def create_scenarii(banques, expositions, filename="scenarii.pkl"):
    """Crée tous les scénarios en extrayant directement tous les indicateurs clés (avec features enrichies incluses systématiquement)."""
    reseau_ids = banques["reseau_id"].unique()
    scenarii_list = []
    
    for reseau_id in reseau_ids:
        reseau_banques = banques[banques["reseau_id"] == reseau_id]
        reseau_size = len(reseau_banques)
        reseau_expositions = expositions[expositions["reseau_id"] == reseau_id]
        
        # Construction du graphe et alignement des nœuds
        graphe = construire_graphe(reseau_banques, reseau_expositions)
        nodelist = list(graphe.nodes())
        
        A = nx.to_numpy_array(graphe, nodelist=nodelist, weight='montant', dtype=np.float32)
        reseau_banques_indexe = reseau_banques.set_index("banque_id")
        cap = reseau_banques_indexe.loc[nodelist, "capital"].to_numpy(dtype=np.float32)
        
        liens = A > 0
        expo_in = A.sum(0)  # Ce que chaque banque a prêté (perte maximale subie)
        
        # --- Indicateurs globaux au niveau du réseau ---
        net_liens_mean = float(liens.mean())
        net_degre_moyen = float(liens.sum(1).mean() / reseau_size)
        net_expo_mean = float(np.log1p(A[liens].mean()) if liens.any() else 0.0)
        net_levier = float(np.log1p(A.sum() / cap.sum()) if cap.sum() > 0 else 0.0)
        net_cap_mean = float(np.log1p(cap).mean())
        net_cap_std_ratio = float(cap.std() / cap.mean() if cap.mean() > 0 else 0.0)
        
        lambdas = [0.3, 0.6, 0.9]
        
        for reseau_banque in reseau_banques.itertuples(index=False):
            o = nodelist.index(reseau_banque.banque_id)
            
            for lam in lambdas:
                # --- Indicateurs dépendants de lambda (réseau) ---
                net_fragiles_sans_cascade = float(np.mean(lam * expo_in > cap))
                
                # --- Indicateurs au niveau de l'origine ---
                orig_cap_log = float(np.log1p(cap[o]))
                orig_degre_sortant = float(liens[o].sum() / reseau_size)
                orig_degre_entrant = float(liens[:, o].sum() / reseau_size)
                orig_pression_med = float(np.log1p(lam * A[o].sum() / np.median(cap)) if np.median(cap) > 0 else 0.0)
                orig_portee_max = float(portee(liens, o) / reseau_size)
                
                # --- Indicateurs enrichis (systématiques) ---
                fortes = lam * A > cap[None, :]
                orig_aretes_critiques = float(fortes[o].sum() / reseau_size)
                orig_portee_critique = float(portee(fortes, o) / reseau_size)
                
                # Simulation de la cascade
                graphe_sim = graphe.copy()
                _, simulation_result = simuler(graphe_sim, reseau_banque.banque_id, lam)
                
                banqueroute_number = sum(simulation_result) + 1
                banqueroute_rate = banqueroute_number / reseau_size
                
                # Constitution de la ligne de données claire et tabulaire
                scenario_data = {
                    "reseau_id": str(reseau_id),
                    "origine": str(reseau_banque.banque_id),
                    "lambda": float(lam),
                    # Features réseau
                    "net_liens_mean": net_liens_mean,
                    "net_degre_moyen": net_degre_moyen,
                    "net_expo_mean": net_expo_mean,
                    "net_levier": net_levier,
                    "net_cap_mean": net_cap_mean,
                    "net_cap_std_ratio": net_cap_std_ratio,
                    "net_fragiles_sans_cascade": net_fragiles_sans_cascade,
                    # Features origine
                    "orig_cap_log": orig_cap_log,
                    "orig_degre_sortant": orig_degre_sortant,
                    "orig_degre_entrant": orig_degre_entrant,
                    "orig_pression_med": orig_pression_med,
                    "orig_portee_max": orig_portee_max,
                    # Features enrichies
                    "orig_aretes_critiques": orig_aretes_critiques,
                    "orig_portee_critique": orig_portee_critique,
                    # Cible
                    "y_banqueroute_rate": float(banqueroute_rate)
                }
                
                scenarii_list.append(scenario_data)
                
    # Sauvegarde sous forme de DataFrame propre et lisible
    df_scenarii = pd.DataFrame(scenarii_list)
    df_scenarii.to_pickle(filename)
    
    print(f"Dataset généré avec succès : {len(df_scenarii)} scénarios enregistrés dans '{filename}'.")
    return filename

if __name__ == "__main__":
    banques, expositions = data_loading()
    create_scenarii(banques, expositions)