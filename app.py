# app_grafos_recomendacion_avanzado.py

import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, Node2Vec
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import random

st.set_page_config(page_title="Recomendador de Amigos Avanzado", layout="wide")
st.title("💡 Recomendador de Amigos con Link Prediction")

# -----------------------------------
# Sidebar: Cargar dataset
# -----------------------------------
st.sidebar.header("📂 Cargar dataset")
dataset_option = st.sidebar.selectbox(
    "Selecciona un dataset predefinido o carga uno propio",
    ["SNAP: Facebook Social Circles", "Sintético", "Cargar CSV"]
)

if dataset_option == "Cargar CSV":
    uploaded_file = st.sidebar.file_uploader("Sube tu CSV (src, dst)", type=["csv"])
    if uploaded_file:
        df_edges = pd.read_csv(uploaded_file)
else:
    st.sidebar.write(f"Dataset seleccionado: {dataset_option}")
    # Para ejemplo, generamos un grafo sintético
    df_edges = pd.DataFrame({
        'src': np.random.randint(0, 100, 500),
        'dst': np.random.randint(0, 100, 500)
    })

# -----------------------------------
# Construir grafo y resumen
# -----------------------------------
G = nx.from_pandas_edgelist(df_edges, 'src', 'dst')
st.subheader("📊 Resumen del grafo")
st.write(f"Nodos: {G.number_of_nodes()}, Aristas: {G.number_of_edges()}, Densidad: {nx.density(G):.4f}")

# Comunidades con Louvain
try:
    import community as community_louvain
    partition = community_louvain.best_partition(G)
    st.write(f"Número de comunidades detectadas: {len(set(partition.values()))}")
except ImportError:
    st.write("Instala `python-louvain` para detección de comunidades.")

# -----------------------------------
# Visualización parcial
# -----------------------------------
st.subheader("🕸️ Visualización parcial del grafo")
max_nodes = st.slider("Número de nodos a mostrar", 100, 1000, 500)
sub_nodes = list(G.nodes())[:max_nodes]
subG = G.subgraph(sub_nodes)

nt = Network(height="500px", width="100%", notebook=False)
nt.from_nx(subG)
nt.show_buttons(filter_=['physics'])
nt.save_graph("grafo_parcial.html")
st.components.v1.html(open("grafo_parcial.html", 'r').read(), height=500)

if st.button("Ver grafo completo"):
    nt_full = Network(height="700px", width="100%", notebook=False)
    nt_full.from_nx(G)
    nt_full.show_buttons(filter_=['physics'])
    nt_full.save_graph("grafo_completo.html")
    st.markdown(f"[Abrir grafo completo](grafo_completo.html)")

# -----------------------------------
# Preparar datos para PyG
# -----------------------------------
node_mapping = {n: i for i, n in enumerate(G.nodes())}
edges = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges()], dtype=torch.long).t()
data = Data(edge_index=edges)

# Atributos de nodo dummy si no existen
if not hasattr(data, 'x') or data.x is None:
    data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)

# -----------------------------------
# Separar train/test edges
# -----------------------------------
def train_test_split_edges(data, test_ratio=0.2):
    edges = data.edge_index.t().tolist()
    random.shuffle(edges)
    num_test = int(len(edges) * test_ratio)
    test_edges = edges[:num_test]
    train_edges = edges[num_test:]
    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t()
    data_train = Data(edge_index=train_edge_index, x=data.x)
    data_train.test_edges = torch.tensor(test_edges, dtype=torch.long)
    return data_train

data = train_test_split_edges(data)

# -----------------------------------
# Modelo GNN
# -----------------------------------
st.subheader("🧠 Selección de modelo de grafo")
model_option = st.selectbox("Modelo GNN", ["GCN", "GraphSAGE", "GAT", "Node2Vec"])
device = "cuda" if torch.cuda.is_available() else "cpu"

num_features = data.x.shape[1]
embedding_dim = 64

if model_option == "GCN":
    model = GCNConv(num_features, embedding_dim).to(device)
elif model_option == "GraphSAGE":
    model = SAGEConv(num_features, embedding_dim).to(device)
elif model_option == "GAT":
    model = GATConv(num_features, embedding_dim, heads=2).to(device)
elif model_option == "Node2Vec":
    model = Node2Vec(data.edge_index, embedding_dim=embedding_dim, walk_length=20,
                     context_size=10, walks_per_node=10).to(device)

# -----------------------------------
# Entrenamiento para link prediction
# -----------------------------------
def train_link_prediction(model, data, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    test_edges = data.test_edges.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if isinstance(model, Node2Vec):
            loss = model.loss()
        else:
            z = model(x, edge_index)
            # Sample negative edges
            neg_edge_index = negative_sampling(edge_index=edge_index, num_nodes=x.size(0), num_neg_samples=edge_index.size(1))
            # Concatenate positive y=1 and negative y=0
            pos_out = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
            neg_out = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
            out = torch.cat([pos_out, neg_out])
            y = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).to(device)
            loss = torch.nn.BCEWithLogitsLoss()(out, y)
        loss.backward()
        optimizer.step()
    return model

st.subheader("⚡ Entrenamiento")
if st.button("Entrenar modelo"):
    with st.spinner("Entrenando modelo para link prediction..."):
        model = train_link_prediction(model, data)
    st.success("✅ Entrenamiento completado")

# -----------------------------------
# Embeddings y recomendaciones
# -----------------------------------
st.subheader("🔍 Recomendaciones de amistad")
user_id = st.number_input("Ingresa ID de usuario para sugerencias", min_value=0, max_value=data.num_nodes-1, value=0)

def recommend_friends(model, data, user_id, top_k=5):
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    if isinstance(model, Node2Vec):
        z = model().detach()
    else:
        z = model(x, edge_index).detach()
    user_emb = z[user_id]
    scores = torch.matmul(z, user_emb)
    # Filtrar usuarios ya conectados
    neighbors = set(edge_index[1][edge_index[0]==user_id].cpu().numpy())
    neighbors.add(user_id)
    candidates = [(i, s.item()) for i, s in enumerate(scores) if i not in neighbors]
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    return [c[0] for c in candidates[:top_k]]

if st.button("Generar sugerencias"):
    top_users = recommend_friends(model, data, user_id)
    st.write(f"Sugerencias para usuario {user_id}: {top_users}")

# -----------------------------------
# 📈 Métricas avanzadas de evaluación (AUC, AP, Hits@K, MRR, Recall@K, Precision@K)
# -----------------------------------
st.subheader("📈 Métricas de evaluación (Link Prediction avanzadas)")

import pandas as pd
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_model(model, data, ks=[1,3,5,10], device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    test_edges = data.test_edges.to(device)

    # Obtener embeddings
    if isinstance(model, Node2Vec):
        z = model().detach()
    else:
        model.eval()
        with torch.no_grad():
            z = model(x, edge_index).detach()

    num_nodes = z.size(0)
    num_test = test_edges.size(0)

    # --- Métricas globales (AUC, AP) ---
    pos_scores = (z[test_edges[:,0]] * z[test_edges[:,1]]).sum(dim=1).cpu()

    neg_edge_index = negative_sampling(
        edge_index=edge_index, num_nodes=num_nodes, num_neg_samples=num_test
    ).to(device)
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1).cpu()

    y_true = torch.cat([
        torch.ones(pos_scores.size(0)),
        torch.zeros(neg_scores.size(0))
    ])
    y_scores = torch.cat([pos_scores, neg_scores])

    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    # --- Métricas por consulta (Hits@K, Recall@K, Precision@K, MRR) ---
    train_adj = {i: set() for i in range(num_nodes)}
    for u, v in edge_index.cpu().T.numpy():
        train_adj[u].add(v)
        train_adj[v].add(u)

    ks_sorted = sorted(ks)
    hits_counts = {k: 0 for k in ks_sorted}
    rr_total = 0.0

    for i in range(num_test):
        u, v = test_edges[i]
        u, v = int(u.item()), int(v.item())
        pos_score = (z[u] * z[v]).sum().item()

        scores = torch.matmul(z, z[u]).cpu()
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        for n in train_adj[u]:
            mask[n] = True
        mask[u] = True
        scores[mask] = float("-inf")

        rank = int((scores >= pos_score).sum().item())
        if scores[v] == float("-inf"):
            continue

        rr_total += 1.0 / max(rank, 1)
        for k in ks_sorted:
            if rank <= k:
                hits_counts[k] += 1

    num_eval = num_test
    hits_at_k = {k: hits_counts[k] / num_eval for k in ks_sorted}
    recall_at_k = {k: hits_counts[k] / num_eval for k in ks_sorted}
    precision_at_k = {k: hits_counts[k] / (k * num_eval) for k in ks_sorted}
    mrr = rr_total / num_eval

    # --- Organizar en DataFrame ---
    metrics_table = pd.DataFrame({
        "Métrica": ["AUC", "AP", "MRR"] + [f"Hits@{k}" for k in ks_sorted] + [f"Recall@{k}" for k in ks_sorted] + [f"Precision@{k}" for k in ks_sorted],
        "Valor": [auc, ap, mrr] + [hits_at_k[k] for k in ks_sorted] + [recall_at_k[k] for k in ks_sorted] + [precision_at_k[k] for k in ks_sorted]
    })
    metrics_table["Valor"] = metrics_table["Valor"].map(lambda x: f"{x:.4f}")

    return metrics_table

# -----------------------------------
# Botón para calcular métricas
# -----------------------------------
if st.button("Calcular métricas avanzadas"):
    with st.spinner("Calculando métricas..."):
        metrics_df = evaluate_model(model, data)
    st.success("✅ Métricas calculadas correctamente")
    st.dataframe(metrics_df, use_container_width=True)

