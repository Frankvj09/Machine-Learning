import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Dataset generation (reproducible) ────────────────────────────────────────
def generate_dataset(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    s1_age    = rng.normal(24, 4, 400).clip(18, 35)
    s1_income = rng.normal(2200, 500, 400).clip(1000, 4000)
    s1_spend  = rng.normal(800, 200, 400).clip(300, 1800)
    s2_age    = rng.normal(40, 5, 400).clip(33, 52)
    s2_income = rng.normal(5500, 700, 400).clip(3500, 8000)
    s2_spend  = rng.normal(2200, 400, 400).clip(1200, 4000)
    s3_age    = rng.normal(57, 4, 400).clip(50, 68)
    s3_income = rng.normal(8500, 800, 400).clip(6000, 12000)
    s3_spend  = rng.normal(4000, 600, 400).clip(2500, 6500)

    ages    = np.concatenate([s1_age, s2_age, s3_age]).round(1)
    incomes = np.concatenate([s1_income, s2_income, s3_income]).round(0)
    spends  = np.concatenate([s1_spend, s2_spend, s3_spend]).round(0)

    idx = rng.permutation(n)
    df = pd.DataFrame({
        'customer_id':   [f'C{i+1:04d}' for i in range(n)],
        'age':           ages[idx],
        'annual_income': incomes[idx],
        'monthly_spend': spends[idx],
    })
    return df


def run_clustering():
    df = generate_dataset()
    X = df[['age', 'annual_income', 'monthly_spend']].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled) + 1

    # Sort clusters by income for consistent labelling
    cluster_means = df.groupby('cluster')['annual_income'].mean().sort_values()
    label_map = {old: new for new, old in enumerate(cluster_means.index, 1)}
    df['cluster'] = df['cluster'].map(label_map)

    labels = {1: 'Young / Low Income', 2: 'Middle-Aged / Mid Income', 3: 'Senior / High Income'}
    df['segment'] = df['cluster'].map(labels)

    centroids_scaled = kmeans.cluster_centers_
    cents_reorder = np.zeros_like(centroids_scaled)
    for old, new in label_map.items():
        cents_reorder[new - 1] = centroids_scaled[old - 1]
    centroids_orig = scaler.inverse_transform(cents_reorder)

    return df, centroids_orig


def generate_cluster_plot(df, centroids):
    colors  = {1: '#E74C3C', 2: '#3498DB', 3: '#2ECC71'}
    markers = {1: 'o', 2: 's', 3: '^'}
    labels  = {1: 'Young / Low Income', 2: 'Middle-Aged / Mid Income', 3: 'Senior / High Income'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#F8F9FA')

    for k in [1, 2, 3]:
        sub = df[df['cluster'] == k]
        axes[0].scatter(sub['age'], sub['annual_income'],
                        c=colors[k], marker=markers[k], alpha=0.6, s=35, label=labels[k])
    axes[0].scatter(centroids[:, 0], centroids[:, 1],
                    c='black', marker='X', s=250, zorder=5, label='Centroids')
    for i, c in enumerate(centroids):
        axes[0].annotate(f'C{i+1}', xy=(c[0], c[1]),
                         xytext=(c[0]+0.8, c[1]+300), fontsize=9, fontweight='bold',
                         arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    axes[0].set_xlabel('Age (years)', fontsize=12)
    axes[0].set_ylabel('Annual Income ($)', fontsize=12)
    axes[0].set_title('Age vs Annual Income', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9, loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_facecolor('white')

    for k in [1, 2, 3]:
        sub = df[df['cluster'] == k]
        axes[1].scatter(sub['annual_income'], sub['monthly_spend'],
                        c=colors[k], marker=markers[k], alpha=0.6, s=35, label=labels[k])
    axes[1].scatter(centroids[:, 1], centroids[:, 2],
                    c='black', marker='X', s=250, zorder=5, label='Centroids')
    axes[1].set_xlabel('Annual Income ($)', fontsize=12)
    axes[1].set_ylabel('Monthly Spend ($)', fontsize=12)
    axes[1].set_title('Income vs Monthly Spend', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9, loc='upper left')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_facecolor('white')

    fig.suptitle('K-Means Customer Segmentation (K=3)', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#F8F9FA')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_b64


def get_cluster_data():
    """Main function called from app.py — returns everything the template needs."""
    df, centroids = run_clustering()

    summary = (df.groupby(['cluster', 'segment'])
                 .agg(count=('customer_id', 'count'),
                      avg_age=('age', 'mean'),
                      avg_income=('annual_income', 'mean'),
                      avg_spend=('monthly_spend', 'mean'))
                 .reset_index()
                 .to_dict('records'))

    cent_labels = {1: 'Young / Low Income', 2: 'Middle-Aged / Mid Income', 3: 'Senior / High Income'}
    cent_list = [
        {'cluster': i+1, 'label': cent_labels[i+1],
         'age': round(c[0], 2), 'income': round(c[1], 2), 'spend': round(c[2], 2)}
        for i, c in enumerate(centroids)
    ]

    sample = (df.groupby('cluster').head(8)
                .sort_values(['cluster', 'customer_id'])
               [['customer_id', 'age', 'annual_income', 'monthly_spend', 'cluster', 'segment']]
                .to_dict('records'))

    plot_b64 = generate_cluster_plot(df, centroids)
    total    = len(df)

    return summary, cent_list, sample, plot_b64, total

#--------------------------------------------------------------------------