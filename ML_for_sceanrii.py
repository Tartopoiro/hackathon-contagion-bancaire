import pickle
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeRegressor

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


def evaluer_indices_features(
    feature_indices, X, y, groups, model_type="tree", random_state=42
):
  """Évalue un sous-ensemble spécifique de features via Leave-One-Group-Out.

  Calcule la MSE et la MAE à chaque pli.
  """
  if not feature_indices:
    return float("inf"), float("inf"), [], [], []

  X_sub = X[:, list(feature_indices)]
  logo = LeaveOneGroupOut()
  mses, maes = [], []
  group_names = []

  for train_idx, test_idx in logo.split(X_sub, y, groups):
    X_tr, X_te = X_sub[train_idx], X_sub[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    current_group = groups[test_idx][0]

    if model_type == "tree":
      model = DecisionTreeRegressor(random_state=random_state, max_depth=8)
    elif model_type == "forest":
      model = RandomForestRegressor(
          n_estimators=100, random_state=random_state, n_jobs=1
      )
    else:
      raise ValueError("Modèle inconnu")

    model.fit(X_tr, y_tr)
    y_pred = np.clip(model.predict(X_te), 0, 1)

    mses.append(mean_squared_error(y_te, y_pred))
    maes.append(mean_absolute_error(y_te, y_pred))
    group_names.append(current_group)

  return np.mean(mses), np.mean(maes), mses, maes, group_names


def forward_selection_with_lookahead(
    X, y, groups, feature_names, model_type="tree"
):
  """Forward Selection avec tolérance d'un cran en cas de stagnation (multicolinéarité)."""
  remaining = list(range(X.shape[1]))
  selected = []
  best_overall_mse = float("inf")

  print(f"\n--- Forward Selection (avec lookahead) [{model_type.upper()}] ---")

  while remaining:
    best_step_mse = float("inf")
    best_feature = None

    for f in remaining:
      trial = selected + [f]
      mse, _, _, _, _ = evaluer_indices_features(
          trial, X, y, groups, model_type
      )
      if mse < best_step_mse:
        best_step_mse = mse
        best_feature = f

    if best_step_mse < best_overall_mse:
      selected.append(best_feature)
      remaining.remove(best_feature)
      best_overall_mse = best_step_mse
      print(
          f" [+] Ajout direct de '{feature_names[best_feature]}' -> MSE ="
          f" {best_overall_mse:.5f}"
      )
    else:
      print(
          f" [?] Stagnation détectée (MSE étape = {best_step_mse:.5f} vs globale"
          f" = {best_overall_mse:.5f}). Test du cran suivant..."
      )

      temp_selected = selected + [best_feature]
      temp_remaining = [r for r in remaining if r != best_feature]

      lookahead_amelioration = False
      if temp_remaining:
        best_next_mse = float("inf")
        best_next_feature = None

        for f2 in temp_remaining:
          trial2 = temp_selected + [f2]
          mse2, _, _, _, _ = evaluer_indices_features(
              trial2, X, y, groups, model_type
          )
          if mse2 < best_next_mse:
            best_next_mse = mse2
            best_next_feature = f2

        if best_next_mse < best_overall_mse:
          print(
              f" [!] Lookahead réussi ! Ajout de '{feature_names[best_feature]}'"
              f" puis '{feature_names[best_next_feature]}' -> Nouvelle MSE ="
              f" {best_next_mse:.5f}"
          )
          selected.append(best_feature)
          selected.append(best_next_feature)
          remaining.remove(best_feature)
          remaining.remove(best_next_feature)
          best_overall_mse = best_next_mse
          lookahead_amelioration = True

      if not lookahead_amelioration:
        print(" [*] Fin de la sélection Forward.")
        break

  return selected, best_overall_mse


def backward_elimination_with_lookahead(
    X, y, groups, feature_names, model_type="tree"
):
  """Backward Elimination avec tolérance d'un cran en cas de stagnation."""
  selected = list(range(X.shape[1]))
  best_overall_mse, _, _, _, _ = evaluer_indices_features(
      selected, X, y, groups, model_type
  )

  print(
      f"\n--- Backward Elimination (avec lookahead) [{model_type.upper()}] ---"
  )
  print(f" Départ avec toutes les features -> MSE = {best_overall_mse:.5f}")

  while len(selected) > 1:
    best_step_mse = float("inf")
    feature_to_remove = None

    for f in selected:
      trial = [x for x in selected if x != f]
      mse, _, _, _, _ = evaluer_indices_features(
          trial, X, y, groups, model_type
      )
      if mse < best_step_mse:
        best_step_mse = mse
        feature_to_remove = f

    if best_step_mse <= best_overall_mse:
      selected.remove(feature_to_remove)
      best_overall_mse = best_step_mse
      print(
          f" [-] Retrait direct de '{feature_names[feature_to_remove]}' -> MSE"
          f" = {best_overall_mse:.5f}"
      )
    else:
      print(
          f" [?] Stagnation au retrait (MSE étape = {best_step_mse:.5f}). Test"
          " du retrait d'une seconde variable..."
      )

      temp_selected = [x for x in selected if x != feature_to_remove]
      lookahead_amelioration = False

      if len(temp_selected) > 1:
        best_next_mse = float("inf")
        feature_to_remove_2 = None

        for f2 in temp_selected:
          trial2 = [x for x in temp_selected if x != f2]
          mse2, _, _, _, _ = evaluer_indices_features(
              trial2, X, y, groups, model_type
          )
          if mse2 < best_next_mse:
            best_next_mse = mse2
            feature_to_remove_2 = f2

        if best_next_mse <= best_overall_mse:
          print(
              f" [!] Lookahead réussi ! Retrait de"
              f" '{feature_names[feature_to_remove]}' et de"
              f" '{feature_names[feature_to_remove_2]}' -> Nouvelle MSE ="
              f" {best_next_mse:.5f}"
          )
          selected.remove(feature_to_remove)
          selected.remove(feature_to_remove_2)
          best_overall_mse = best_next_mse
          lookahead_amelioration = True

      if not lookahead_amelioration:
        print(" [*] Fin de la sélection Backward.")
        break

  return selected, best_overall_mse


def comparer_et_choisir_meilleur(
    X, y, groups, feature_names, model_type="tree"
):
  """Exécute Forward et Backward puis ne conserve que le meilleur résultat (selon la MSE)."""
  idx_fw, mse_fw = forward_selection_with_lookahead(
      X, y, groups, feature_names, model_type
  )
  idx_bw, mse_bw = backward_elimination_with_lookahead(
      X, y, groups, feature_names, model_type
  )

  if mse_fw <= mse_bw:
    best_idx, best_strat = idx_fw, "Forward"
  else:
    best_idx, best_strat = idx_bw, "Backward"

  best_features = [feature_names[i] for i in best_idx]
  mean_mse, mean_mae, mses, maes, names = evaluer_indices_features(
      best_idx, X, y, groups, model_type
  )

  print(
      f"\n>>> Stratégie retenue pour [{model_type.upper()}] : {best_strat} <<<"
  )
  print(f"    MSE : {mean_mse:.5f} | MAE : {mean_mae:.5f}")
  print(f"    Features sélectionnées : {best_features}")

  return {
      "idx": best_idx,
      "features": best_features,
      "strat": best_strat,
      "mse": mean_mse,
      "mae": mean_mae,
      "mses": mses,
      "maes": maes,
      "names": names,
  }


def tracer_meilleurs_resultats(
    group_names,
    res_tree,
    res_rf,
    features_tree,
    features_rf,
    strat_tree,
    strat_rf,
):
  """Trace un graphique unique 2x2 contenant la MSE et la MAE par réseau ainsi que les moyennes globales."""
  fig, axes = plt.subplots(2, 2, figsize=(16, 10))

  x = np.arange(len(group_names))
  width = 0.35

  mses_tree, maes_tree = res_tree["mses"], res_tree["maes"]
  mses_rf, maes_rf = res_rf["mses"], res_rf["maes"]

  # --- 1. MSE par réseau ---
  axes[0, 0].bar(
      x - width / 2,
      mses_tree,
      width,
      label=f"Arbre ({strat_tree})",
      color="skyblue",
      edgecolor="black",
  )
  axes[0, 0].bar(
      x + width / 2,
      mses_rf,
      width,
      label=f"Forêt ({strat_rf})",
      color="salmon",
      edgecolor="black",
  )
  axes[0, 0].set_ylabel("MSE", fontsize=11)
  axes[0, 0].set_title("MSE par réseau (LOO)", fontsize=12, fontweight="bold")
  axes[0, 0].set_xticks(x)
  axes[0, 0].set_xticklabels(group_names, rotation=45)
  axes[0, 0].legend(loc="upper right")
  axes[0, 0].grid(axis="y", linestyle="--", alpha=0.7)

  # --- 2. MAE par réseau ---
  axes[0, 1].bar(
      x - width / 2,
      maes_tree,
      width,
      label=f"Arbre ({strat_tree})",
      color="skyblue",
      edgecolor="black",
  )
  axes[0, 1].bar(
      x + width / 2,
      maes_rf,
      width,
      label=f"Forêt ({strat_rf})",
      color="salmon",
      edgecolor="black",
  )
  axes[0, 1].set_ylabel("MAE", fontsize=11)
  axes[0, 1].set_title("MAE par réseau (LOO)", fontsize=12, fontweight="bold")
  axes[0, 1].set_xticks(x)
  axes[0, 1].set_xticklabels(group_names, rotation=45)
  axes[0, 1].legend(loc="upper right")
  axes[0, 1].grid(axis="y", linestyle="--", alpha=0.7)

  # --- 3. MSE Globale (Moyenne ± std) ---
  means_mse = [np.mean(mses_tree), np.mean(mses_rf)]
  stds_mse = [np.std(mses_tree), np.std(mses_rf)]
  models = [f"Arbre ({strat_tree})", f"Forêt ({strat_rf})"]
  colors = ["skyblue", "salmon"]

  feat_str_tree = "\n".join(f"- {f}" for f in features_tree)
  feat_str_rf = "\n".join(f"- {f}" for f in features_rf)

  bars_mse = axes[1, 0].bar(
      models,
      means_mse,
      yerr=stds_mse,
      capsize=5,
      color=colors,
      edgecolor="black",
      alpha=0.85,
  )
  axes[1, 0].set_ylabel("MSE Moyenne", fontsize=11)
  axes[1, 0].set_title(
      "MSE Globale (Moyenne ± Écart-type)", fontsize=12, fontweight="bold"
  )
  axes[1, 0].grid(axis="y", linestyle="--", alpha=0.7)

  legend_labels_mse = [
      f"Arbre ({strat_tree}):\n{feat_str_tree}",
      f"Forêt ({strat_rf}):\n{feat_str_rf}",
  ]
  axes[1, 0].legend(
      bars_mse, legend_labels_mse, loc="upper right", fontsize=8, framealpha=0.9
  )

  # --- 4. MAE Globale (Moyenne ± std) ---
  means_mae = [np.mean(maes_tree), np.mean(maes_rf)]
  stds_mae = [np.std(maes_tree), np.std(maes_rf)]

  bars_mae = axes[1, 1].bar(
      models,
      means_mae,
      yerr=stds_mae,
      capsize=5,
      color=colors,
      edgecolor="black",
      alpha=0.85,
  )
  axes[1, 1].set_ylabel("MAE Moyenne", fontsize=11)
  axes[1, 1].set_title(
      "MAE Globale (Moyenne ± Écart-type)", fontsize=12, fontweight="bold"
  )
  axes[1, 1].grid(axis="y", linestyle="--", alpha=0.7)
  axes[1, 1].legend(
      bars_mae, legend_labels_mse, loc="upper right", fontsize=8, framealpha=0.9
  )

  plt.suptitle(
      "Comparaison des Meilleurs Sous-ensembles (MSE & MAE)",
      fontsize=14,
      fontweight="bold",
  )
  plt.tight_layout()
  plt.savefig("meilleurs_resultats_complets.png")
  plt.show()


def main():
  df, X, y, groups, feature_names = charger_donnees("scenarii.pkl")
  print(
      f"Dataset chargé : {X.shape[0]} scénarios, {X.shape[1]} features,"
      f" {len(np.unique(groups))} réseaux."
  )

  # Recherche de la meilleure stratégie pour l'Arbre de Décision
  res_tree = comparer_et_choisir_meilleur(
      X, y, groups, feature_names, model_type="tree"
  )

  # Recherche de la meilleure stratégie pour la Forêt Aléatoire
  res_rf = comparer_et_choisir_meilleur(
      X, y, groups, feature_names, model_type="forest"
  )

  # Tracé unique comparatif
  tracer_meilleurs_resultats(
      res_tree["names"],
      res_tree,
      res_rf,
      res_tree["features"],
      res_rf["features"],
      res_tree["strat"],
      res_rf["strat"],
  )


if __name__ == "__main__":
  main()