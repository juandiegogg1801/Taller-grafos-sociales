# ------------------------------
# Streamlit App: Recomendación de amigos con GNNs
# ------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import tempfile
import os
from io import StringIO

# PyTorch / PyG imports
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn.models import Node2Vec
from sklearn.metrics import roc_auc_score, average_precision_score
from pyvis.network import Network
import streamlit.components.v1 as components
import community as community_louvain

# ---------------- Helpers ----------------
def load_edges_from_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]
    posibles_nombres = [
        ("source", "target"),
        ("src", "dst"),
        ("user1", "user2"),
        ("node1", "node2"),
        ("from", "to"),
        ("id1", "id2"),
    ]
    source_col, target_col = None, None
    for s, t in posibles_nombres:
        if s in df.columns and t in df.columns:
            source_col, target_col = s, t
            break
    if not source_col:
        raise ValueError(f"No se encontraron columnas válidas. Columnas: {df.columns}")
    edges = df[[source_col, target_col]].astype(int).values
    return edges, df


def build_pyg_data_from_edges(edges, undirected=True):
    if undirected:
        edges = np.vstack([edges, edges[:, ::-1]])
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    n_nodes = edges.max() + 1
    # Usar embeddings pequeños para acelerar
    x = torch.randn((n_nodes, 32), dtype=torch.float)
    return Data(x=x, edge_index=edge_index)


# ---------------- Models ----------------
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)


class GraphSAGE(GCN):
    def __init__(self, in_channels, hidden=64):
        super().__init__(in_channels, hidden)
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)


class GAT(GCN):
    def __init__(self, in_channels, hidden=64, heads=4):
        super().__init__(in_channels, hidden)
        self.conv1 = GATConv(in_channels, hidden // heads, heads=heads)
        self.conv2 = GATConv(hidden, hidden // heads, heads=heads)


# ---------------- Metrics ----------------
def hits_at_k(y_true, y_score, k=10):
    order = np.argsort(-y_score)
    top_k = y_true[order][:k]
    return np.mean(top_k)


def precision_recall_at_k(y_true, y_score, k=10):
    order = np.argsort(-y_score)
    top_k = y_true[order][:k]
    precision = np.mean(top_k)
    recall = np.sum(top_k) / np.sum(y_true) if np.sum(y_true) > 0 else 0
    return precision, recall


def mean_reciprocal_rank(y_true, y_score):
    order = np.argsort(-y_score)
    for rank, idx in enumerate(order, 1):
        if y_true[idx] == 1:
            return 1.0 / rank
    return 0.0


def compute_link_prediction_metrics(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    hits10 = hits_at_k(y_true, y_score, k=10)
    prec10, rec10 = precision_recall_at_k(y_true, y_score, k=10)
    mrr = mean_reciprocal_rank(y_true, y_score)
    return {
        "AUC": auc,
        "AP": ap,
        "Hits@10": hits10,
        "Precision@10": prec10,
        "Recall@10": rec10,
        "MRR": mrr,
    }


# ---------------- Training ----------------
def train_and_evaluate(model, data, epochs=15, lr=0.01, device="cpu"):
    device = torch.device(device)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    edge_label_index = data.edge_label_index
    edge_label = data.edge_label.float()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        logits = model.decode(z, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, edge_label)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        scores = model.decode(z, edge_label_index).cpu().numpy()
        y_true = edge_label.cpu().numpy()
        metrics = compute_link_prediction_metrics(y_true, scores)
    return metrics


# ---------------- Visualization ----------------
def draw_pyvis_graph(edges, selected_node=None, max_edges=500):
    import random
    G = nx.Graph()
    if len(edges) > max_edges:
        sampled = edges[np.random.choice(len(edges), max_edges, replace=False)]
    else:
        sampled = edges
    G.add_edges_from(sampled.tolist())

    # Comunidades Louvain
    partition = community_louvain.best_partition(G)
    net = Network(height="600px", width="100%", notebook=False, bgcolor="#222", font_color="white")

    for node in G.nodes():
        color = f"hsl({(partition[node]*40)%360},70%,50%)"
        size = 15 if node == selected_node else 8
        net.add_node(node, color=color, size=size, title=f"Usuario {node}")
    for u, v in G.edges():
        net.add_edge(u, v, color="gray", opacity=0.5)

    return net


def tensor_to_numpy(tensor):
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy()


# ---------------- Streamlit App ----------------
def main():
    st.set_page_config(layout="wide", page_title="Recomendador de amigos - GNN")
    st.title("🧠 Recomendación de amigos con Redes Neuronales de Grafos")

    st.sidebar.header("1️⃣ Cargar dataset")
    uploaded = st.sidebar.file_uploader("Sube CSV (source,target)", type=["csv"])

    if not uploaded:
        st.info("Sube un dataset CSV con columnas source/target o src/dst.")
        return

    try:
        edges, df = load_edges_from_csv(uploaded)
    except Exception as e:
        st.error(str(e))
        return

    st.write("### Vista previa del dataset")
    st.dataframe(df.head())

    data = build_pyg_data_from_edges(edges)
    st.write(f"📊 Nodos: {data.num_nodes} — Aristas: {data.num_edges}")

    transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=True)
    train_data, val_data, test_data = transform(data)

    st.sidebar.header("2️⃣ Configuración del modelo")
    model_choice = st.sidebar.selectbox("Modelo", ["GCN", "GraphSAGE", "GAT", "Node2Vec"])
    epochs = st.sidebar.slider("Épocas", 1, 100, 15)
    lr = st.sidebar.number_input("Learning Rate", value=0.01)

    if st.sidebar.button("🚀 Entrenar modelo"):
        with st.spinner("Entrenando y evaluando..."):
            if model_choice == "GCN":
                model = GCN(in_channels=data.num_node_features)
                metrics = train_and_evaluate(model, train_data, epochs=epochs, lr=lr)
            elif model_choice == "GraphSAGE":
                model = GraphSAGE(in_channels=data.num_node_features)
                metrics = train_and_evaluate(model, train_data, epochs=epochs, lr=lr)
            elif model_choice == "GAT":
                model = GAT(in_channels=data.num_node_features)
                metrics = train_and_evaluate(model, train_data, epochs=epochs, lr=lr)
            else:
                node2vec = Node2Vec(data.edge_index, embedding_dim=64, walk_length=10, context_size=5)
                optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=lr)
                for _ in range(epochs):
                    optimizer.zero_grad()
                    loss = node2vec.loss()
                    loss.backward()
                    optimizer.step()
                metrics = {"AUC": "-", "AP": "-", "Hits@10": "-", "Precision@10": "-", "Recall@10": "-", "MRR": "-"}

        st.subheader("📈 Métricas de evaluación")
        st.dataframe(pd.DataFrame(metrics, index=["Valor"]).T)

    st.subheader("🌐 Visualización del grafo")
    net = draw_pyvis_graph(edges)
    tmp_path = os.path.join(tempfile.gettempdir(), "graph.html")
    net.save_graph(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as f:
        components.html(f.read(), height=600)

    st.subheader("👥 Recomendaciones para un usuario")
    selected = st.number_input("ID del usuario", 0, data.num_nodes - 1, 0)
    if st.button("Generar recomendaciones"):
        model = GCN(in_channels=data.num_node_features)
        z = model.encode(data.x, data.edge_index)
        scores = tensor_to_numpy(z[selected] @ z.T)
        G = nx.Graph()
        G.add_edges_from(edges.tolist())
        neighbors = set(G.neighbors(selected))
        candidates = [(i, s) for i, s in enumerate(scores) if i != selected and i not in neighbors]
        recs = sorted(candidates, key=lambda x: -x[1])[:10]
        st.table(pd.DataFrame(recs, columns=["Usuario", "Puntaje"]))


if __name__ == "__main__":
    main()
