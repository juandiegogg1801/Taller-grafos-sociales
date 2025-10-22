import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx
from pyvis.network import Network
import numpy as np
import random
import os

st.set_page_config(page_title="Recomendador GNN", layout="wide")
st.title("🤝 Sistema de Recomendación basado en Grafos (GNN)")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.success(f"Usando dispositivo: {device.upper()}")

# -----------------------
# Sidebar - Dataset
# -----------------------
st.sidebar.header("📂 Selección de dataset")

dataset_option = st.sidebar.selectbox(
    "Selecciona un dataset:",
    [
        "SNAP simulado (demo)",
        "Sintético aleatorio",
        "Cargar CSV (src,dst)",
        "Cora (torch_geometric)"
    ]
)

uploaded_file = None
if dataset_option == "Cargar CSV (src,dst)":
    uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV con columnas src y dst", type=['csv'])

# -----------------------
# Función para cargar dataset
# -----------------------
@st.cache_data
def load_dataset(option, uploaded_file=None):
    if option == "Cargar CSV (src,dst)" and uploaded_file:
        df = pd.read_csv(uploaded_file)
        G = nx.from_pandas_edgelist(df, 'src', 'dst')
        return df, G, None
    elif option == "Sintético aleatorio":
        df = pd.DataFrame({
            'src': np.random.randint(0, 100, 300),
            'dst': np.random.randint(0, 100, 300)
        })
        G = nx.from_pandas_edgelist(df, 'src', 'dst')
        return df, G, None
    elif option == "Cora (torch_geometric)":
        try:
            from torch_geometric.datasets import Planetoid
            dataset = Planetoid(root='data/Cora', name='Cora')
            data = dataset[0]
            df = pd.DataFrame(data.edge_index.t().cpu().numpy(), columns=['src', 'dst'])
            G = nx.from_pandas_edgelist(df, 'src', 'dst')
            return df, G, data
        except Exception as e:
            st.error(f"No se pudo cargar Cora: {e}")
            return None, None, None
    else:  # SNAP simulado
        df = pd.DataFrame({
            'src': np.random.randint(0, 200, 1000),
            'dst': np.random.randint(0, 200, 1000)
        })
        G = nx.from_pandas_edgelist(df, 'src', 'dst')
        return df, G, None

df, G, pyg_data = load_dataset(dataset_option, uploaded_file)

if G is None:
    st.stop()

# -----------------------
# Mostrar información del grafo
# -----------------------
st.subheader("📈 Información del grafo")
st.write(f"Nodos: {G.number_of_nodes()}, Aristas: {G.number_of_edges()}, Densidad: {nx.density(G):.4f}")

# -----------------------
# Visualización con PyVis
# -----------------------
st.subheader("🌐 Visualización del grafo")
num_nodes = st.slider("Nodos a visualizar:", 50, min(300, G.number_of_nodes()))
sub_nodes = list(G.nodes())[:num_nodes]
subG = G.subgraph(sub_nodes)
nt = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
nt.from_nx(subG)
out_html = "graph_temp.html"
nt.save_graph(out_html)
st.components.v1.html(open(out_html, 'r', encoding='utf-8').read(), height=500)

# -----------------------
# Preparar datos para PyG
# -----------------------
if pyg_data is None:
    node_map = {n: i for i, n in enumerate(G.nodes())}
    edges = torch.tensor([[node_map[u], node_map[v]] for u, v in G.edges()], dtype=torch.long).t()
    data = Data(edge_index=edges)
    data.x = torch.ones((data.num_nodes, 1))
else:
    data = pyg_data
    if not hasattr(data, "x") or data.x is None:
        data.x = torch.ones((data.num_nodes, 1))

# -----------------------
# Split de entrenamiento y prueba
# -----------------------
def split_edges(data, test_ratio=0.2):
    edges = data.edge_index.t().cpu().numpy()
    np.random.shuffle(edges)
    num_test = int(len(edges) * test_ratio)
    test_edges = edges[:num_test]
    train_edges = edges[num_test:]
    data_train = Data(edge_index=torch.tensor(train_edges).t(), x=data.x)
    data_train.test_edges = torch.tensor(test_edges)
    return data_train

data = split_edges(data)

# -----------------------
# Selección de modelo
# -----------------------
st.sidebar.header("🧠 Modelo GNN")
model_option = st.sidebar.selectbox("Selecciona el modelo:", ["GCN", "GraphSAGE", "GAT"])
epochs = st.sidebar.slider("Épocas de entrenamiento", 10, 200, 50)
embedding_dim = st.sidebar.slider("Dimensión de embedding", 16, 256, 64)

# -----------------------
# Entrenamiento del modelo
# -----------------------
def train_and_evaluate(model, data, epochs=50):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x, edge_index = data.x.to(device), data.edge_index.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model(x, edge_index)
        neg_edge_index = negative_sampling(edge_index=edge_index, num_nodes=x.size(0))
        pos_out = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        neg_out = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
        out = torch.cat([pos_out, neg_out])
        y = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).to(device)
        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        optimizer.step()

    # Evaluación
    model.eval()
    with torch.no_grad():
        z = model(x, edge_index)
        test_edges = data.test_edges.to(device)
        pos_scores = (z[test_edges[:, 0]] * z[test_edges[:, 1]]).sum(dim=1).cpu()
        neg_edge_index = negative_sampling(edge_index=edge_index, num_nodes=z.size(0),
                                           num_neg_samples=test_edges.size(0)).to(device)
        neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1).cpu()
        y_true = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])
        y_scores = torch.cat([pos_scores, neg_scores])
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
    return z, auc, ap

# -----------------------
# Crear el modelo
# -----------------------
if model_option == "GCN":
    model = GCNConv(data.x.shape[1], embedding_dim)
elif model_option == "GraphSAGE":
    model = SAGEConv(data.x.shape[1], embedding_dim)
else:
    model = GATConv(data.x.shape[1], embedding_dim, heads=2)

if st.button("🚀 Entrenar modelo"):
    with st.spinner(f"Entrenando modelo {model_option}..."):
        embeddings, auc, ap = train_and_evaluate(model, data, epochs)
        st.success(f"Entrenamiento finalizado ✅  |  AUC: {auc:.4f}  |  AP: {ap:.4f}")
        st.session_state.embeddings = embeddings
        st.session_state.model = model

# -----------------------
# Recomendaciones
# -----------------------
st.subheader("🎯 Recomendaciones de amistad")
if "embeddings" in st.session_state:
    user_id = st.number_input("Selecciona ID de usuario:", min_value=0, max_value=data.num_nodes-1, value=0)
    top_k = st.slider("Número de sugerencias", 1, 20, 5)

    z = st.session_state.embeddings
    scores = torch.matmul(z, z[user_id])
    scores[user_id] = -1e9  # evitar recomendarse a sí mismo
    top_k_idx = torch.topk(scores, top_k).indices.tolist()
    st.write(f"👥 Usuarios recomendados para {user_id}: {top_k_idx}")
else:
    st.info("Primero entrena un modelo para generar recomendaciones.")
