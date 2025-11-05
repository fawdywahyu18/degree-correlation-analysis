import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load networks
def load_network(folder_name):
    edges = pd.read_csv(f"{folder_name}/edges.csv")
    edges.columns = edges.columns.str.strip().str.replace('# ', '')
    G = nx.from_pandas_edgelist(edges, source='source', target='target')
    return G

# Load both networks
football = load_network("football-tsevans")
yeast = load_network("interactome-yeast")

networks = {
    'NCAA Football': football,
    'Yeast Interactome': yeast
}

# Compute k_nn(k)
def compute_knn(G, bins=None):
    knn = nx.average_neighbor_degree(G)
    degrees = np.array(list(dict(G.degree()).values()))
    knn_values = np.array([knn[n] for n in G.nodes()])
    
    if bins is None:
        k_unique = np.unique(degrees)
        knn_avg = [knn_values[degrees == k].mean() for k in k_unique]
        return k_unique, np.array(knn_avg)
    else:
        k_binned = []
        knn_binned = []
        for i in range(len(bins)-1):
            mask = (degrees >= bins[i]) & (degrees < bins[i+1])
            if mask.sum() > 0:
                k_binned.append(degrees[mask].mean())
                knn_binned.append(knn_values[mask].mean())
        return np.array(k_binned), np.array(knn_binned)

# Degree-preserving randomization
def randomize_network(G, n_swaps, allow_multilinks=False):
    G_rand = G.copy()
    n_tries = 0
    n_success = 0
    max_tries = n_swaps * 10
    
    while n_success < n_swaps and n_tries < max_tries:
        edges = list(G_rand.edges())
        if len(edges) < 2:
            break
            
        e1, e2 = np.random.choice(len(edges), 2, replace=False)
        u1, v1 = edges[e1]
        u2, v2 = edges[e2]
        
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            n_tries += 1
            continue
        
        if not allow_multilinks:
            if G_rand.has_edge(u1, u2) or G_rand.has_edge(v1, v2):
                n_tries += 1
                continue
        
        G_rand.remove_edge(u1, v1)
        G_rand.remove_edge(u2, v2)
        G_rand.add_edge(u1, u2)
        G_rand.add_edge(v1, v2)
        
        n_success += 1
        n_tries += 1
    
    return G_rand

# Compute correlation coefficient r
def compute_r(G):
    degrees = dict(G.degree())
    r_num = 0
    r_den1 = 0
    r_den2 = 0
    m = G.number_of_edges()
    
    for u, v in G.edges():
        r_num += degrees[u] * degrees[v]
        r_den1 += degrees[u] + degrees[v]
        r_den2 += degrees[u]**2 + degrees[v]**2
    
    r_num = r_num / m
    r_den1 = (r_den1 / (2*m))**2
    r_den2 = r_den2 / (2*m)
    
    if r_den2 - r_den1 == 0:
        return 0
    
    return (r_num - r_den1) / (r_den2 - r_den1)

# Fit power law for mu
def fit_powerlaw(k, knn, k_min=None, k_max=None):
    if k_min is None:
        k_min = k.min()
    if k_max is None:
        k_max = k.max()
    
    mask = (k >= k_min) & (k <= k_max)
    k_fit = k[mask]
    knn_fit = knn[mask]
    
    if len(k_fit) < 2:
        return np.nan, np.nan
    
    log_k = np.log(k_fit)
    log_knn = np.log(knn_fit)
    
    coeffs = np.polyfit(log_k, log_knn, 1)
    mu = coeffs[0]
    a = np.exp(coeffs[1])
    
    return mu, a

# Randomization parameters
n_randomizations = 10
n_swaps_factor = 3

results = []

for name, G in networks.items():
    n_swaps = G.number_of_edges() * n_swaps_factor
    
    # Original network
    k_orig, knn_orig = compute_knn(G)
    r_orig = compute_r(G)
    mu_orig, _ = fit_powerlaw(k_orig, knn_orig, k_min=5)
    
    # R-S randomization (simple links)
    knn_rs_all = []
    r_rs_all = []
    for i in range(n_randomizations):
        G_rs = randomize_network(G, n_swaps, allow_multilinks=False)
        k_rs, knn_rs = compute_knn(G_rs)
        knn_rs_all.append(np.interp(k_orig, k_rs, knn_rs))
        r_rs_all.append(compute_r(G_rs))
    
    knn_rs_mean = np.mean(knn_rs_all, axis=0)
    r_rs_mean = np.mean(r_rs_all)
    mu_rs, _ = fit_powerlaw(k_orig, knn_rs_mean, k_min=5)
    
    # R-M randomization (allow multilinks)
    knn_rm_all = []
    r_rm_all = []
    for i in range(n_randomizations):
        G_rm = randomize_network(G, n_swaps, allow_multilinks=True)
        k_rm, knn_rm = compute_knn(G_rm)
        knn_rm_all.append(np.interp(k_orig, k_rm, knn_rm))
        r_rm_all.append(compute_r(G_rm))
    
    knn_rm_mean = np.mean(knn_rm_all, axis=0)
    r_rm_mean = np.mean(r_rm_all)
    mu_rm, _ = fit_powerlaw(k_orig, knn_rm_mean, k_min=5)
    
    results.append({
        'Network': name,
        'k': k_orig,
        'knn_orig': knn_orig,
        'knn_rs': knn_rs_mean,
        'knn_rm': knn_rm_mean,
        'r_orig': r_orig,
        'r_rs': r_rs_mean,
        'r_rm': r_rm_mean,
        'mu_orig': mu_orig,
        'mu_rs': mu_rs,
        'mu_rm': mu_rm
    })

# Summary table
summary = []
for res in results:
    summary.append({
        'Network': res['Network'],
        'r (original)': f"{res['r_orig']:.3f}",
        'r (R-S)': f"{res['r_rs']:.3f}",
        'r (R-M)': f"{res['r_rm']:.3f}",
        'μ (original)': f"{res['mu_orig']:.3f}",
        'μ (R-S)': f"{res['mu_rs']:.3f}",
        'μ (R-M)': f"{res['mu_rm']:.3f}"
    })

df_summary = pd.DataFrame(summary)
print("\nDegree Correlation Coefficients")
print(df_summary.to_string(index=False))

# Plot k_nn(k) comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, res in enumerate(results):
    ax = axes[idx]
    k = res['k']
    
    ax.plot(k, res['knn_orig'], 'o-', label='Original', 
            markersize=5, linewidth=2, color='darkblue')
    ax.plot(k, res['knn_rs'], 's--', label='R-S (simple)', 
            markersize=4, linewidth=1.5, color='orange', alpha=0.8)
    ax.plot(k, res['knn_rm'], '^:', label='R-M (multilinks)', 
            markersize=4, linewidth=1.5, color='green', alpha=0.8)
    
    ax.set_xlabel('Degree k', fontsize=12)
    ax.set_ylabel('$k_{nn}(k)$', fontsize=12)
    ax.set_title(res['Network'], fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('degree_correlation_comparison.png', bbox_inches='tight', dpi=300)
plt.show()

# Save results to CSV
for res in results:
    df_out = pd.DataFrame({
        'k': res['k'],
        'knn_original': res['knn_orig'],
        'knn_rs': res['knn_rs'],
        'knn_rm': res['knn_rm']
    })
    filename = f"knn_{res['Network'].lower().replace(' ', '_')}.csv"
    df_out.to_csv(filename, index=False)