import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline

# Load networks
def load_network(folder_name):
    edges = pd.read_csv(f"{folder_name}/edges.csv")
    # Clean column names (remove # and spaces)
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

# Network properties
props = []
for name, G in networks.items():
    degrees = [d for n, d in G.degree()]
    props.append({
        'Network': name,
        'N': G.number_of_nodes(),
        'L': G.number_of_edges(),
        '<k>': np.mean(degrees),
        'k_max': np.max(degrees),
        '<k^2>': np.mean([d**2 for d in degrees]),
        'Density': nx.density(G)
    })

df_props = pd.DataFrame(props)
print("Network Properties")
print(df_props.to_string(index=False))
print()

# Degree distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for idx, (name, G) in enumerate(networks.items()):
    degrees = [d for n, d in G.degree()]
    
    bins = range(min(degrees), max(degrees) + 2)
    
    axes[idx].hist(degrees, bins=bins, 
                   color='steelblue', alpha=0.7, edgecolor='black')
    
    axes[idx].set_xlabel('Degree k', fontsize=12)
    axes[idx].set_ylabel('Count', fontsize=12)
    axes[idx].set_title(name, fontsize=13, fontweight='bold')
    axes[idx].set_xlim(left=0)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('degree_distribution.png', bbox_inches='tight', dpi=300)
plt.show()

# Log-log degree distribution to check power-law
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (name, G) in enumerate(networks.items()):
    degrees = [d for n, d in G.degree()]
    degree_count = pd.Series(degrees).value_counts().sort_index()
    
    k = degree_count.index.values
    pk = degree_count.values
    
    # Log-log scatter plot
    axes[idx].loglog(k, pk, 'o', markersize=8, alpha=0.7, color='steelblue', label='Data')
    
    # Fit power-law line for k >= k_min
    k_min = 2
    mask = k >= k_min
    if mask.sum() > 2:
        log_k = np.log10(k[mask])
        log_pk = np.log10(pk[mask])
        coeffs = np.polyfit(log_k, log_pk, 1)
        gamma = -coeffs[0]
        
        # Plot fitted line
        k_fit = np.logspace(np.log10(k_min), np.log10(k.max()), 50)
        pk_fit = 10**coeffs[1] * k_fit**coeffs[0]
        axes[idx].loglog(k_fit, pk_fit, '--', linewidth=2, color='red', 
                        label=f'Power-law fit: γ ≈ {gamma:.2f}')
    
    axes[idx].set_xlabel('Degree k', fontsize=12)
    axes[idx].set_ylabel('P(k)', fontsize=12)
    axes[idx].set_title(name, fontsize=13, fontweight='bold')
    axes[idx].grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
    axes[idx].legend(fontsize=10)

plt.tight_layout()
plt.savefig('degree_distribution_loglog.png', bbox_inches='tight', dpi=300)
plt.show()

# Network visualization (sample for large networks)
for name, G in networks.items():
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if G.number_of_nodes() > 300:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:150]
        G_vis = G.subgraph([n for n, d in top_nodes])
    else:
        G_vis = G
    
    degrees = dict(G_vis.degree())
    pos = nx.kamada_kawai_layout(G_vis)
    
    node_sizes = [50 + degrees[n] * 15 for n in G_vis.nodes()]
    
    nx.draw_networkx_nodes(G_vis, pos, node_size=node_sizes,
                          node_color=node_sizes, cmap='viridis',
                          alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G_vis, pos, alpha=0.2, width=0.5, ax=ax)
    
    ax.set_title(name, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.savefig(f'network_{name.lower().replace(" ", "_")}.png', 
                bbox_inches='tight', dpi=300)
    plt.show()

# Calculate structural cutoff
print("\nStructural Cutoff Analysis")
for name, G in networks.items():
    N = G.number_of_nodes()
    avg_k = np.mean([d for n, d in G.degree()])
    k_s = np.sqrt(avg_k * N)
    k_max = np.max([d for n, d in G.degree()])
    print(f"{name}: k_s = {k_s:.1f}, k_max = {k_max}")