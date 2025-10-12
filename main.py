import os
import glob
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Task 1-3: Real vs Random Network Comparison with Assortativity
def plot_real_vs_random_with_assortativity(data_dir="data", out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    gml_files = glob.glob(os.path.join(data_dir, "*.gml"))
    if not gml_files:
        print("No .gml files found.")
        return

    for path in gml_files:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"Processing {name}...")

        G = nx.read_gml(path)

        # Standardize to a simple undirected graph
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            G = nx.Graph(G)
        if nx.is_directed(G):
            G = G.to_undirected()

        # Largest connected component only
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

        # --- Original network metrics ---
        deg = dict(G.degree())
        knn = nx.average_neighbor_degree(G)
        k_vals = np.array(list(deg.values()))
        knn_vals = np.array([knn[n] for n in G.nodes()])
        r_real = nx.degree_assortativity_coefficient(G)

        # --- Randomized reference ---
        print("  -> randomizing network...")
        R = nx.algorithms.smallworld.random_reference(G, connectivity=False, seed=42)
        r_deg = dict(R.degree())
        r_knn = nx.average_neighbor_degree(R)
        rk_vals = np.array(list(r_deg.values()))
        rknn_vals = np.array([r_knn[n] for n in R.nodes()])
        r_rand = nx.degree_assortativity_coefficient(R)

        # --- Plot ---
        plt.figure(figsize=(8, 6))
        plt.scatter(k_vals, knn_vals, s=10, alpha=0.5,
                    label=f"Original (r = {r_real:.3f})")
        plt.scatter(rk_vals, rknn_vals, s=10, alpha=0.5,
                    label=f"Randomized (r = {r_rand:.3f})")

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("k (degree)")
        plt.ylabel("knn(k) (average neighbor degree)")
        plt.title(f"Average Neighbor Degree vs. Degree ({name})")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{name}_real_vs_random_assort.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"  -> saved {out_path}")
        
 #Task 4: Degree Distribution Plotting       
def plot_degree_distribution(data_dir="data", out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    gml_files = glob.glob(os.path.join(data_dir, "*.gml"))
    if not gml_files:
        print("No .gml files found.")
        return

    for path in gml_files:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"Processing {name}...")

        G = nx.read_gml(path)
        if nx.is_directed(G):
            G = G.to_undirected()
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

        degrees = np.array([d for _, d in G.degree()])
        degrees = degrees[degrees > 0]  # exclude zeros to avoid log(0)

        # Logarithmic binning
        kmin, kmax = degrees.min(), degrees.max()
        bins = np.logspace(np.log10(kmin), np.log10(kmax), num=30)

        plt.figure(figsize=(8, 6))
        plt.hist(
            degrees,
            bins=bins,
            density=True,
            alpha=0.7,
            edgecolor="black",
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("k (degree)")
        plt.ylabel("p(k) (probability density)")
        plt.title(f"Degree Distribution p(k) ({name})")
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{name}_degree_distribution.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"  → saved {out_path}")

if __name__ == "__main__":
    plot_degree_distribution()
