import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import random
import plotly.graph_objects as go

st.set_page_config(page_title="Recomendador de Amigos Final", layout="wide")
st.title("💡 Recomendador de Amigos con Link Prediction - Final")

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Sidebar: Dataset
# -----------------------------
st.sidebar.header("📂 Dataset")
dataset_option = st.sidebar.selectbox("Selecciona dataset", ["SNAP simulado", "Sintético", "Cargar CSV"])
uploaded_file = None
if dataset_option == "Cargar CSV":
    uploaded_file = st.sidebar.file_uploader("Sube tu CSV (src,dst)", type=['csv'])

@st.cache_data
def load_dataset(option, uploaded_file=None):
    if option == "Cargar CSV" and uploaded_file:
        df = pd.read_csv(uploaded_file)
    elif option == "Sintético":
        df = pd.DataFrame({'src': np.random.randint(0,100,500),
                           'dst': np.random.randint(0,100,500)})
    else:  # SNAP simulado
        df = pd.DataFrame({'src': np.random.randint(0,200,1000),
                           'dst': np.random.randint(0,200,1000)})
    G = nx.from_pandas_edgelist(df, 'src', 'dst')
    return df, G

df_edges, G = load_dataset(dataset_option, uploaded_file)

# -----------------------------
# Resumen del grafo
# -----------------------------
st.subheader("📊 Resumen del grafo")
st.write(f"Nodos: {G.number_of_nodes()}, Aristas: {G.number_of_edges()}, Densidad: {nx.density(G):.4f}")

try:
    import community as community_louvain
    partition = community_louvain.best_partition(G)
    st.write(f"Número de comunidades detectadas: {len(set(partition.values()))}")
except ImportError:
    st.write("Instala `python-louvain` para detección de comunidades.")

# -----------------------------
# Visualización parcial
# -----------------------------
st.subheader("🕸️ Grafo parcial")
max_nodes = st.slider("Nodos a mostrar", 100, 1000, 500)
sub_nodes = list(G.nodes())[:max_nodes]
subG = G.subgraph(sub_nodes)
nt = Network(height="500px", width="100%", notebook=False)
nt.from_nx(subG)
nt.show_buttons(filter_=['physics'])
nt.save_graph("grafo_parcial.html")
try:
    with open("grafo_parcial.html", "r") as f:
        html_content = f.read()
    if len(html_content.strip()) < 100:
        st.warning("El archivo HTML del grafo está vacío o incompleto. No se puede mostrar el grafo.")
    else:
        st.components.v1.html(html_content, height=500)
except Exception as e:
    st.warning(f"No se pudo mostrar el grafo: {e}")

# El resto de la app debe mostrarse aunque falle la visualización

# -----------------------------
# Preparar datos PyG
# -----------------------------
node_mapping = {n:i for i,n in enumerate(G.nodes())}
edges = torch.tensor([[node_mapping[u], node_mapping[v]] for u,v in G.edges()], dtype=torch.long).t()
data = Data(edge_index=edges)
if not hasattr(data, 'x') or data.x is None:
    data.x = torch.ones((data.num_nodes,1),dtype=torch.float)

def train_test_split_edges(data, test_ratio=0.2):
    edges_list = data.edge_index.t().tolist()
    random.shuffle(edges_list)
    num_test = int(len(edges_list)*test_ratio)
    test_edges = edges_list[:num_test]
    train_edges = edges_list[num_test:]
    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t()
    data_train = Data(edge_index=train_edge_index, x=data.x)
    data_train.test_edges = torch.tensor(test_edges, dtype=torch.long)
    return data_train

data = train_test_split_edges(data)

# -----------------------------
# Selección modelo
# -----------------------------
st.subheader("🧠 Modelo GNN")
model_option = st.selectbox("Selecciona modelo", ["GCN","GraphSAGE","GAT"])
embedding_dim = 64
num_features = data.x.shape[1]

def get_model(option, num_features, embedding_dim, edge_index=None):
    if option=="GCN":
        return GCNConv(num_features, embedding_dim)
    elif option=="GraphSAGE":
        return SAGEConv(num_features, embedding_dim)
    elif option=="GAT":
        return GATConv(num_features, embedding_dim, heads=2)

model = get_model(model_option, num_features, embedding_dim, edge_index=data.edge_index)

# -----------------------------
# Session state
# -----------------------------
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "metrics_df" not in st.session_state:
    st.session_state.metrics_df = None

# -----------------------------
# Entrenamiento y métricas
# -----------------------------
def train_and_evaluate(_model, data, epochs=50, ks=[1,3,5,10]):
    model = _model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    test_edges = data.test_edges.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model(x, edge_index)
        neg_edge_index = negative_sampling(edge_index=edge_index, num_nodes=x.size(0), num_neg_samples=edge_index.size(1))
        pos_out = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        neg_out = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
        out = torch.cat([pos_out, neg_out])
        y = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).to(device)
        loss = torch.nn.BCEWithLogitsLoss()(out, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model(x, edge_index).detach()

    # --- Métricas ---
    num_nodes = z.size(0)
    num_test = test_edges.size(0)
    pos_scores = (z[test_edges[:,0]] * z[test_edges[:,1]]).sum(dim=1).cpu()
    neg_edge_index = negative_sampling(edge_index=edge_index, num_nodes=num_nodes, num_neg_samples=num_test).to(device)
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1).cpu()
    y_true = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])
    y_scores = torch.cat([pos_scores, neg_scores])
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    train_adj = {i:set() for i in range(num_nodes)}
    for u,v in edge_index.cpu().T.numpy():
        train_adj[u].add(v)
        train_adj[v].add(u)

    hits_counts = {k:0 for k in ks}
    rr_total = 0.0
    for i in range(num_test):
        u,v = test_edges[i]
        u,v = int(u.item()), int(v.item())
        pos_score = (z[u]*z[v]).sum().item()
        scores = torch.matmul(z,z[u]).cpu()
        mask = torch.zeros(num_nodes,dtype=torch.bool)
        for n in train_adj[u]: mask[n]=True
        mask[u]=True
        scores[mask]=float("-inf")
        rank = int((scores>=pos_score).sum().item())
        if scores[v]==float("-inf"): continue
        rr_total += 1.0/max(rank,1)
        for k in ks:
            if rank<=k: hits_counts[k]+=1

    num_eval = num_test
    hits_at_k = {k:hits_counts[k]/num_eval for k in ks}
    recall_at_k = {k:hits_counts[k]/num_eval for k in ks}
    precision_at_k = {k:hits_counts[k]/(k*num_eval) for k in ks}
    mrr = rr_total/num_eval

    metrics_table = pd.DataFrame({
        "Métrica": ["AUC","AP","MRR"] + [f"Hits@{k}" for k in ks] + [f"Recall@{k}" for k in ks] + [f"Precision@{k}" for k in ks],
        "Valor": [auc,ap,mrr] + [hits_at_k[k] for k in ks] + [recall_at_k[k] for k in ks] + [precision_at_k[k] for k in ks]
    })
    metrics_table["Valor"] = metrics_table["Valor"].map(lambda x:f"{x:.4f}")
    return z, metrics_table

if st.button("Entrenar y calcular métricas"):
    with st.spinner("⚡ Entrenando y evaluando..."):
        embeddings, metrics_df = train_and_evaluate(model, data)
        st.session_state.embeddings = embeddings
        st.session_state.metrics_df = metrics_df
    st.success("✅ Entrenamiento y métricas completadas")

# -----------------------------
# Mostrar métricas si existen
# -----------------------------
if st.session_state.metrics_df is not None:
    st.subheader("📈 Métricas avanzadas")
    st.dataframe(st.session_state.metrics_df, use_container_width=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=st.session_state.metrics_df["Métrica"],
        y=st.session_state.metrics_df["Valor"].astype(float),
        text=st.session_state.metrics_df["Valor"],
        textposition='outside',
        marker_color='blue'
    ))
    fig.update_layout(yaxis=dict(range=[0,1], title="Valor"), title="Métricas Link Prediction", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Recomendacione
# -----------------------------
user_id = st.number_input("ID de usuario", min_value=0, max_value=data.num_nodes-1, value=0)
top_k = st.number_input("Top-K sugerencias", min_value=1, max_value=20, value=5)

def recommend_detailed(z, data, user_id, top_k=5):
    edge_index = data.edge_index
    user_emb = z[user_id]
    scores = torch.matmul(z,user_emb)
    neighbors = set(edge_index[1][edge_index[0]==user_id].cpu().numpy())
    neighbors.add(user_id)
    candidates = [(i,s.item()) for i,s in enumerate(scores) if i not in neighbors]
    candidates_sorted = sorted(candidates, key=lambda x:x[1], reverse=True)
    top_users = [c[0] for c in candidates_sorted[:top_k]]
    all_candidates_df = pd.DataFrame(candidates_sorted, columns=["Usuario","Score"])
    return len(candidates_sorted), top_users, all_candidates_df

if st.session_state.embeddings is not None:
    total_candidates, top_users, candidates_df = recommend_detailed(st.session_state.embeddings, data, user_id, top_k)
    st.subheader(f"🔍 Recomendaciones para usuario {user_id}")
    st.write(f"Total de candidatos disponibles: {total_candidates}")
    st.write(f"Top-{top_k} sugerencias: {top_users}")
    st.write("📋 Lista completa de candidatos ordenados por relevancia:")
    st.dataframe(candidates_df, use_container_width=True)
