import os
import glob
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def scatter_plot_all(data_dir="data", out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)

    gml_files = glob.glob(os.path.join(data_dir, "*.gml"))
    if not gml_files:
        print("Keine .gml-Dateien gefunden.")
        return

    for path in gml_files:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"Bearbeite {name}...")

        G = nx.read_gml(path)
        if nx.is_directed(G):
            G = G.to_undirected()

        # Falls Graph nicht zusammenhängend ist: größte Komponente nehmen
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

        deg = dict(G.degree())
        knn = nx.average_neighbor_degree(G)

        k_values = np.array(list(deg.values()))
        knn_values = np.array([knn[n] for n in G.nodes()])

        plt.figure(figsize=(8, 6))
        plt.scatter(k_values, knn_values, s=10, alpha=0.6)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("k (degree)")
        plt.ylabel("knn(k) (average neighbor degree)")
        plt.title(f"Average Neighbor Degree vs. Degree ({name})")
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{name}_knn_vs_k.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"→ gespeichert als {out_path}")

if __name__ == "__main__":
    scatter_plot_all()
