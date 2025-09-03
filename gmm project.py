import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Example: Load your data (replace with your actual data source)
# data = pd.read_csv('your_data.csv')  # Uncomment and adjust if using CSV
# For demonstration, here's a random dataset (replace this with your own):
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.normal(loc=0, scale=1, size=150),
    'feature2': np.random.normal(loc=5, scale=2, size=150),
    'feature3': np.random.normal(loc=10, scale=3, size=150)
})

# Standardize features for GMM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Fit GMM with 3 clusters
gmm = GaussianMixture(n_components=3, random_state=42)
clusters = gmm.fit_predict(X_scaled)
data['cluster'] = clusters

# Show feature means and ranges for each cluster
result = {}
for c in range(3):
    cluster_data = data[data['cluster'] == c]
    means = cluster_data.mean(numeric_only=True)
    ranges = cluster_data.max(numeric_only=True) - cluster_data.min(numeric_only=True)
    result[f'Cluster {c}'] = {
        'Means': means.to_dict(),
        'Ranges': ranges.to_dict()
    }

# Print summarized results
for cluster, stats in result.items():
    print(f"\n{cluster}:")
    print("Means:")
    for feat, mean in stats['Means'].items():
        if feat != 'cluster':
            print(f"  {feat}: {mean:.3f}")
    print("Ranges:")
    for feat, rng in stats['Ranges'].items():
        if feat != 'cluster':
            print(f"  {feat}: {rng:.3f}")

# Optional: Check which features differentiate groups (feature importance)
# For GMM, use cluster means and ranges as indicative metrics
