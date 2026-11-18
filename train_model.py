"""
train_model.py
--------------
Run in the terminal:  python train_model.py
Prints the model comparison table and saves charts into 'outputs/' (use them in your report).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ml_pipeline as ml

os.makedirs("outputs", exist_ok=True)
df = ml.load_data()
print("Dataset shape:", df.shape)
print(df[ml.TARGET].describe().round(2), "\n")

fitted, results, best, (_, X_test, _, y_test) = ml.train_and_evaluate(df)

print(f"{'Model':20}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'CV R2':>8}")
for n, r in results.items():
    print(f"{n:20}{r['mae']:8.3f}{r['rmse']:8.3f}{r['r2']:8.3f}{r['cv_r2']:8.3f}")
print("\nBest model:", best)

# Chart 1: actual vs predicted
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_test, results[best]["pred"], alpha=0.5, color="#4C78A8")
lim = [0, max(y_test.max(), results[best]["pred"].max()) + 0.5]
ax.plot(lim, lim, "r--", label="Perfect prediction")
ax.set_xlabel("Actual yield (t/ha)"); ax.set_ylabel("Predicted yield (t/ha)")
ax.set_title(f"Actual vs Predicted - {best}"); ax.legend()
fig.tight_layout(); fig.savefig("outputs/actual_vs_predicted.png", dpi=150)

# Chart 2: R2 comparison
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.bar(results.keys(), [r["r2"] for r in results.values()], color="#59A14F")
ax.set_ylim(0, 1); ax.set_ylabel("R2 score"); ax.set_title("Model Comparison (higher is better)")
plt.setp(ax.get_xticklabels(), rotation=15)
fig.tight_layout(); fig.savefig("outputs/model_comparison.png", dpi=150)

# Chart 3: feature importance
imp = ml.feature_importance(fitted["Random Forest"])
fig, ax = plt.subplots(figsize=(6, 5))
imp.sort_values().plot.barh(ax=ax, color="#F58518")
ax.set_title("Feature Importance (Random Forest)")
fig.tight_layout(); fig.savefig("outputs/feature_importance.png", dpi=150)
print("Charts saved in outputs/")
