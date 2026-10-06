import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error

# Ignore les avertissements superflus
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def charger_donnees(filename="scenarii.pkl"):
    """Charge le dataset contenant directement les indicateurs clés."""
    df = pd.read_pickle(filename)
    
    cols_exclues = ["reseau_id", "origine", "lambda", "y_banqueroute_rate"]
    feature_cols = [c for c in df.columns if c not in cols_exclues]
    
    X = df[feature_cols].values
    y = df["y_banqueroute_rate"].values
    groups = df["reseau_id"].values
    
    return df, X, y, groups, feature_cols


def evaluer_indices_features(feature_indices, X, y, groups, model_type="tree", random_state=42):
    """Évalue un sous-ensemble spécifique de features via Leave-One-Group-Out (100% séquentiel)."""
    if not feature_indices:
        return float("inf"), [], []
        
    X_sub = X[:, list(feature_indices)]
    logo = LeaveOneGroupOut()
    mses = []
    group_names = []
    
    for train_idx, test_idx in logo.split(X_sub, y, groups):
        X_tr, X_te = X_sub[train_idx], X_sub[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        current_group = groups[test_idx][0]
        
        if model_type == "tree":
            model = DecisionTreeRegressor(random_state=random_state, max_depth=8)
        elif model_type == "forest":
            model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=1)
        else:
            raise ValueError("Modèle inconnu")
            
        model.fit(X_tr, y_tr)
        y_pred = np.clip(model.predict(X_te), 0, 1)
        
        mses.append(mean_squared_error(y_te, y_pred))
        group_names.append(current_group)
        
    return np.mean(mses), mses, group_names


def forward_selection_with_lookahead(X, y, groups, feature_names, model_type="tree"):
    """
    Forward Selection avec tolérance d'un cran en cas de stagnation (multicolinéarité).
    Si une étape n'améliore pas le score absolu, on regarde si l'étape d'après le dépasse.
    """
    remaining = list(range(X.shape[1]))
    selected = []
    best_overall_mse = float("inf")
    
    print(f"\n--- Forward Selection (avec lookahead) [{model_type.upper()}] ---")
    
    while remaining:
        best_step_mse = float("inf")
        best_feature = None
        
        # Test de toutes les features restantes
        for f in remaining:
            trial = selected + [f]
            mse, _, _ = evaluer_indices_features(trial, X, y, groups, model_type)
            if mse < best_step_mse:
                best_step_mse = mse
                best_feature = f
                
        # Cas 1 : Amélioration directe
        if best_step_mse < best_overall_mse:
            selected.append(best_feature)
            remaining.remove(best_feature)
            best_overall_mse = best_step_mse
            print(f" [+] Ajout direct de '{feature_names[best_feature]}' -> MSE = {best_overall_mse:.5f}")
        
        # Cas 2 : Stagnation/Dégradation (Test du cran d'après pour contrer la multicolinéarité)
        else:
            print(f" [?] Stagnation détectée (meilleure MSE étape = {best_step_mse:.5f} vs globale = {best_overall_mse:.5f}). Test du cran suivant...")
            
            # On simule l'ajout de cette meilleure feature intermédiaire
            temp_selected = selected + [best_feature]
            temp_remaining = [r for r in remaining if r != best_feature]
            
            lookahead_amelioration = False
            if temp_remaining:
                best_next_mse = float("inf")
                best_next_feature = None
                
                for f2 in temp_remaining:
                    trial2 = temp_selected + [f2]
                    mse2, _, _ = evaluer_indices_features(trial2, X, y, groups, model_type)
                    if mse2 < best_next_mse:
                        best_next_mse = mse2
                        best_next_feature = f2
                
                # Si le cran d'après parvient à battre le record global historique
                if best_next_mse < best_overall_mse:
                    print(f" [!] Lookahead réussi ! Ajout de '{feature_names[best_feature]}' puis '{feature_names[best_next_feature]}' -> Nouvelle MSE = {best_next_mse:.5f}")
                    selected.append(best_feature)
                    selected.append(best_next_feature)
                    remaining.remove(best_feature)
                    remaining.remove(best_next_feature)
                    best_overall_mse = best_next_mse
                    lookahead_amelioration = True
            
            # Si le lookahead n'a pas non plus battu le record, on stoppe réellement
            if not lookahead_amelioration:
                print(" [*] Fin de la sélection Forward (pas d'amélioration même au cran suivant).")
                break
                
    return selected, best_overall_mse


def backward_elimination_with_lookahead(X, y, groups, feature_names, model_type="tree"):
    """
    Backward Elimination avec tolérance d'un cran en cas de stagnation.
    """
    selected = list(range(X.shape[1]))
    best_overall_mse, _, _ = evaluer_indices_features(selected, X, y, groups, model_type)
    
    print(f"\n--- Backward Elimination (avec lookahead) [{model_type.upper()}] ---")
    print(f" Départ avec toutes les features -> MSE = {best_overall_mse:.5f}")
    
    while len(selected) > 1:
        best_step_mse = float("inf")
        feature_to_remove = None
        
        for f in selected:
            trial = [x for x in selected if x != f]
            mse, _, _ = evaluer_indices_features(trial, X, y, groups, model_type)
            if mse < best_step_mse:
                best_step_mse = mse
                feature_to_remove = f
                
        # Cas 1 : Amélioration directe en retirant la variable
        if best_step_mse <= best_overall_mse:
            selected.remove(feature_to_remove)
            best_overall_mse = best_step_mse
            print(f" [-] Retrait direct de '{feature_names[feature_to_remove]}' -> MSE = {best_overall_mse:.5f}")
        
        # Cas 2 : Stagnation -> Test d'un cran supplémentaire de retrait
        else:
            print(f" [?] Stagnation au retrait (MSE étape = {best_step_mse:.5f}). Test du retrait d'une seconde variable...")
            
            temp_selected = [x for x in selected if x != feature_to_remove]
            lookahead_amelioration = False
            
            if len(temp_selected) > 1:
                best_next_mse = float("inf")
                feature_to_remove_2 = None
                
                for f2 in temp_selected:
                    trial2 = [x for x in temp_selected if x != f2]
                    mse2, _, _ = evaluer_indices_features(trial2, X, y, groups, model_type)
                    if mse2 < best_next_mse:
                        best_next_mse = mse2
                        feature_to_remove_2 = f2
                
                if best_next_mse <= best_overall_mse:
                    print(f" [!] Lookahead réussi ! Retrait de '{feature_names[feature_to_remove]}' et de '{feature_names[feature_to_remove_2]}' -> Nouvelle MSE = {best_next_mse:.5f}")
                    selected.remove(feature_to_remove)
                    selected.remove(feature_to_remove_2)
                    best_overall_mse = best_next_mse
                    lookahead_amelioration = True
            
            if not lookahead_amelioration:
                print(" [*] Fin de la sélection Backward.")
                break
                
    return selected, best_overall_mse


def tracer_resultats_complets(group_names, mses_tree, mses_rf, features_tree, features_rf, methode_nom=""):
    """Trace les 2 subplots : MSE par réseau (LOO) et Moyenne/Écart-type global."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(group_names))
    width = 0.35
    
    # --- Plot 1 : LOO pli par pli ---
    ax1.bar(x - width/2, mses_tree, width, label='Arbre de Décision', color='skyblue', edgecolor='black')
    ax1.bar(x + width/2, mses_rf, width, label='Forêt Aléatoire', color='salmon', edgecolor='black')
    ax1.set_xlabel("Réseau laissé de côté (LOO)", fontsize=11)
    ax1.set_ylabel("MSE", fontsize=11)
    ax1.set_title(f"MSE par réseau ({methode_nom})", fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(group_names, rotation=45)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # --- Plot 2 : Moyenne et Écart-type global ---
    means = [np.mean(mses_tree), np.mean(mses_rf)]
    stds = [np.std(mses_tree), np.std(mses_rf)]
    models = ['Arbre de Décision', 'Forêt Aléatoire']
    colors = ['skyblue', 'salmon']
    
    feat_str_tree = "\n".join(f"- {f}" for f in features_tree)
    feat_str_rf = "\n".join(f"- {f}" for f in features_rf)
    
    bars = ax2.bar(models, means, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_ylabel("MSE Moyenne", fontsize=11)
    ax2.set_title("Performance Globale (Moyenne ± Écart-type)", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    legend_labels = [
        f"Arbre (Features):\n{feat_str_tree}",
        f"Forêt (Features):\n{feat_str_rf}"
    ]
    ax2.legend(bars, legend_labels, loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f"comparaison_performances_{methode_nom.lower().replace(' ', '_')}.png")
    plt.show()


def executer_strategie(X, y, groups, feature_names, methode="forward"):
    """Exécute la sélection (Forward ou Backward) avec lookahead."""
    if methode == "forward":
        idx_tree, mse_tree = forward_selection_with_lookahead(X, y, groups, feature_names, "tree")
        idx_rf, mse_rf = forward_selection_with_lookahead(X, y, groups, feature_names, "forest")
        nom_methode = "Forward Selection (Lookahead)"
    else:
        idx_tree, mse_tree = backward_elimination_with_lookahead(X, y, groups, feature_names, "tree")
        idx_rf, mse_rf = backward_elimination_with_lookahead(X, y, groups, feature_names, "forest")
        nom_methode = "Backward Elimination (Lookahead)"
        
    features_tree = [feature_names[i] for i in idx_tree]
    features_rf = [feature_names[i] for i in idx_rf]
    
    _, folds_tree, names_tree = evaluer_indices_features(idx_tree, X, y, groups, "tree")
    _, folds_rf, _ = evaluer_indices_features(idx_rf, X, y, groups, "forest")
    
    print(f"\n[{nom_methode}] Résultats finaux :")
    print(f" - Arbre de Décision (MSE = {mse_tree:.5f}) | Features : {features_tree}")
    print(f" - Forêt Aléatoire  (MSE = {mse_rf:.5f}) | Features : {features_rf}")
    
    tracer_resultats_complets(names_tree, folds_tree, folds_rf, features_tree, features_rf, nom_methode)


def main():
    df, X, y, groups, feature_names = charger_donnees("scenarii.pkl")
    print(f"Dataset chargé : {X.shape[0]} scénarios, {X.shape[1]} features, {len(np.unique(groups))} réseaux.")
    
    # 1. Forward Selection avec lookahead
    executer_strategie(X, y, groups, feature_names, methode="forward")
    
    # 2. Backward Elimination avec lookahead
    executer_strategie(X, y, groups, feature_names, methode="backward")


if __name__ == "__main__":
    main()