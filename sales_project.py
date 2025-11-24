
"""
Sales Forecasting & Clustering - Final Project
Prepared by: Subalakshmi P (Batch 202)

Instructions:
- Place your dataset as 'train.csv' in the same folder as this script (or change file_path).
- Run: python sales_project.py
- The script produces visualizations, trains an XGBoost regressor with TimeSeriesSplit CV,
  performs KMeans clustering, shows metrics (RMSE, MAE, R2), and saves charts to the project folder.

Requirements:
- pandas, numpy, matplotlib, scikit-learn, xgboost, python-pptx (optional for PPT work)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score
from sklearn.cluster import KMeans
import xgboost as xgb

# ---------- Parameters ----------
file_path = 'train.csv'   # change if needed
date_col = 'Order Date'
sales_col = 'Sales'
max_lag = 7
test_fraction = 0.20
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# ---------- Load & parse ----------
df_raw = pd.read_csv(file_path)
print("Initial rows:", len(df_raw))
df_raw[date_col] = pd.to_datetime(df_raw[date_col], dayfirst=True, errors='coerce', infer_datetime_format=True)
df_raw = df_raw.dropna(subset=[date_col])
daily = df_raw.groupby(date_col, as_index=True)[sales_col].sum().sort_index().rename('Sales')
idx = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
daily = daily.reindex(idx).interpolate(method='time')
daily.index.name = 'Date'

# ---------- Visualization 1: Sales Trend ----------
plt.figure(figsize=(12,5))
plt.plot(daily.index, daily.values, label='Daily Sales')
plt.title('Daily Sales Trend')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sales_trend.png"))
plt.close()

# ---------- Feature Engineering ----------
df = daily.to_frame()
for lag in range(1, max_lag + 1):
    df[f'lag_{lag}'] = df['Sales'].shift(lag)
df['rolling_7_mean'] = df['Sales'].shift(1).rolling(window=7, min_periods=1).mean()
df['rolling_30_mean'] = df['Sales'].shift(1).rolling(window=30, min_periods=1).mean()
df['dayofweek'] = df.index.dayofweek
df['day'] = df.index.day
df['month'] = df.index.month
df['weekofyear'] = df.index.isocalendar().week.astype(int)
df = df.dropna().copy()

# ---------- Visualization 2: Weekday boxplot ----------
plt.figure(figsize=(8,5))
df_box = df[['Sales', 'dayofweek']].copy()
plt.boxplot([df_box[df_box['dayofweek']==i]['Sales'] for i in range(7)], labels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
plt.title('Sales distribution by Weekday')
plt.ylabel('Sales')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "weekday_boxplot.png"))
plt.close()

# ---------- Prepare features ----------
feature_cols = [c for c in df.columns if c != 'Sales']
X = df[feature_cols]
y = df['Sales']
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

# ---------- Unsupervised: KMeans ----------
cluster_cols = [f'lag_{i}' for i in range(1, max_lag+1)] + ['rolling_7_mean', 'rolling_30_mean']
X_cluster = X_scaled[cluster_cols]
best_k = None
best_sil = -1
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster)
    sil = silhouette_score(X_cluster, labels)
    print(f'k={k}, silhouette={sil:.4f}')
    if sil > best_sil:
        best_sil = sil
        best_k = k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_cluster)
df['cluster'] = kmeans.labels_

# ---------- Visualization 3: Cluster scatter ----------
plt.figure(figsize=(8,6))
for c in sorted(df['cluster'].unique()):
    sel = df[df['cluster']==c]
    plt.scatter(sel['lag_1'], sel['lag_7'], label=f'Cluster {c}', alpha=0.6, s=20)
plt.xlabel('lag_1 (Sales)')
plt.ylabel('lag_7 (Sales)')
plt.title(f'KMeans clusters (k={best_k}) on lag features')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cluster_scatter.png"))
plt.close()

# ---------- Train/Test split ----------
n = len(df)
split_idx = int(n * (1 - test_fraction))
X_train = X_scaled.iloc[:split_idx]
X_test  = X_scaled.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# ---------- XGBoost with TimeSeriesSplit + GridSearchCV ----------
tscv = TimeSeriesSplit(n_splits=5)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0)
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8]
}
gsearch = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=tscv, verbose=1, n_jobs=-1)
gsearch.fit(X_train, y_train)
best_model = gsearch.best_estimator_
print("Best params:", gsearch.best_params_)

# ---------- Evaluate ----------
preds = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE : {mae:.4f}")
print(f"Test R2  : {r2:.4f}")

# Save metrics and predictions
results = pd.DataFrame({'actual': y_test, 'predicted': preds})
results.to_csv(os.path.join(output_dir, "predictions.csv"))
with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write(f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}\nBest params: {gsearch.best_params_}\nSilhouette: {best_sil}\nClusters: {sorted(df['cluster'].unique())}\n")

# ---------- Feature importance ----------
fi = best_model.feature_importances_
fi_series = pd.Series(fi, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
fi_series.head(15).plot(kind='bar')
plt.title('Feature importances (Top 15)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.close()

print("All outputs saved to:", output_dir)
