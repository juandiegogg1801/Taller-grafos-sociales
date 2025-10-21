# Streamlit app: Recomendación de amigos con redes neuronales de grafos
# Archivo: app_recomendacion_grafos.py

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

# Visualization
from pyvis.network import Network
import streamlit.components.v1 as components


# ------------------ Helpers ------------------

def load_edges_from_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]
    posibles_nombres = [
        ("source", "target"),
        ("src", "dst"),
        ("user1", "user2"),
        ("node1", "node2"),
        ("from", "to"),
        ("id1", "id2")
    ]
    source_col, target_col = None, None
    for s, t in posibles_nombres:
        if s in df.columns and t in df.columns:
            source_col, target_col = s, t
            break
    if not source_col or not target_col:
        raise ValueError(
            f"No se encontraron columnas válidas (ejemplo: source/target o src/dst). "
            f"Columnas detectadas: {list(df.columns)}"
        )
    edges = df[[source_col, target_col]].astype(int).values
    return edges, df


def build_pyg_data_from_edges(edges, node_features=None, undirected=True):
    if undirected:
        edges2 = np.vstack([edges, edges[:, ::-1]])
    else:
        edges2 = edges
    edge_index = torch.tensor(edges2.T, dtype=torch.long)
    n_nodes = edges.max() + 1
    if node_features is None:
        x = torch.eye(n_nodes, dtype=torch.float)
    else:
        x = torch.tensor(node_features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)


# ---------------- Models ----------------
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


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

def compute_link_prediction_metrics(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return {"AUC": auc, "AP": ap}


# ---------------- Evaluation loop ----------------

# ---------------- Evaluation loop ----------------

def train_and_evaluate(model, data, epochs=30, lr=0.01, device="cpu"):
    device = torch.device(device)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # El split de RandomLinkSplit devuelve atributos específicos
    # data.edge_label_index = pares de nodos (positivos y negativos)
    # data.edge_label = 1 o 0 (etiqueta real)
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

    # Evaluación
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        scores = model.decode(z, edge_label_index).cpu().numpy()
        y_true = edge_label.cpu().numpy()
        metrics = compute_link_prediction_metrics(y_true, scores)
    return metrics


# ---------------- Visualization ----------------

def draw_pyvis_graph(edges, selected_node=None):
    G = nx.Graph()
    G.add_edges_from(edges.tolist())
    net = Network(height="600px", width="100%")
    net.from_nx(G)
    if selected_node is not None:
        for n in net.nodes:
            if int(n["id"]) == int(selected_node):
                n["color"] = "red"
                n["size"] = 25
    return net


# ---------------- Streamlit App ----------------

def main():
    st.set_page_config(layout="wide", page_title="Recomendador de amigos - GNN")
    st.title("Taller: Recomendación de amigos con GNNs")

    st.sidebar.header("1) Cargar dataset")
    uploaded = st.sidebar.file_uploader("Sube CSV (ej: src,dst)", type=["csv"])

    if uploaded is None:
        st.info("Sube un dataset CSV con columnas source/target o src/dst.")
        return

    try:
        edges, raw_df = load_edges_from_csv(uploaded)
    except Exception as e:
        st.error(str(e))
        return

    st.write("### Vista previa del dataset")
    st.dataframe(raw_df.head())

    data = build_pyg_data_from_edges(edges)
    st.write(f"Nodos: {data.num_nodes} — Aristas: {data.num_edges}")

    # Split
    transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=True)
    train_data, val_data, test_data = transform(data)

    # Modelo
    model_choice = st.sidebar.selectbox("Modelo", ["GCN", "GraphSAGE", "GAT", "Node2Vec"])
    epochs = st.sidebar.slider("Epochs", 1, 100, 30)
    lr = st.sidebar.number_input("Learning Rate", value=0.01)

    if model_choice == "GCN":
        model = GCN(in_channels=data.num_nodes)
    elif model_choice == "GraphSAGE":
        model = GraphSAGE(in_channels=data.num_nodes)
    elif model_choice == "GAT":
        model = GAT(in_channels=data.num_nodes)
    else:
        node2vec = Node2Vec(data.edge_index, embedding_dim=64, walk_length=10, context_size=5)
        optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = node2vec.loss()
            loss.backward()
            optimizer.step()
        model = node2vec

    with st.spinner("Entrenando y evaluando el modelo..."):
        metrics = train_and_evaluate(model, train_data, epochs=epochs, lr=lr, device="cpu")

    st.subheader("Métricas del modelo")
    st.table(pd.DataFrame([metrics]).T.rename(columns={0: "Valor"}))

    # Grafo interactivo
    st.subheader("Visualización interactiva del grafo")
    net = draw_pyvis_graph(edges)
    tmp_path = os.path.join(tempfile.gettempdir(), "graph.html")
    net.save_graph(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as f:
        components.html(f.read(), height=600)

    # Recomendaciones
    st.subheader("Recomendaciones para un usuario")
    selected_user = st.number_input("ID del usuario", min_value=0, max_value=data.num_nodes - 1, value=0)
    if st.button("Generar recomendaciones"):
        with torch.no_grad():
            z = model.encode(data.x, data.edge_index)
            scores = (z[selected_user] @ z.T).cpu().numpy()
            G = nx.Graph()
            G.add_edges_from(edges.tolist())
            neighbors = set(G.neighbors(selected_user))
            candidates = [(i, s) for i, s in enumerate(scores) if i != selected_user and i not in neighbors]
            recs = sorted(candidates, key=lambda x: -x[1])[:10]
            st.table(pd.DataFrame(recs, columns=["Usuario", "Puntaje"]))

    st.markdown("---")
    st.markdown("**Requerimientos:** streamlit, pandas, numpy, networkx, pyvis, torch, torch-geometric, scikit-learn, python-louvain")


if __name__ == "__main__":
    main()
