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
import matplotlib.pyplot as plt

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
st.subheader("🧠 Selección de modelo GNN")
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

if st.button("Entrenar modelo"):
    with st.spinner("⚡ Entrenando y evaluando..."):
        embeddings, metrics_df = train_and_evaluate(model, data)
        st.session_state.embeddings = embeddings
        st.session_state.metrics_df = metrics_df
    st.success("✅ Entrenamiento completado")

# -----------------------------
# Mostrar métricas si existen
# -----------------------------
if st.session_state.metrics_df is not None:
    st.subheader("📈 Resultados de métricas")
    st.dataframe(st.session_state.metrics_df, use_container_width=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=st.session_state.metrics_df["Métrica"],
        y=st.session_state.metrics_df["Valor"].astype(float),
        text=st.session_state.metrics_df["Valor"],
        textposition='outside',
        marker_color='blue'
    ))
    fig.update_layout(yaxis=dict(range=[0,1], title="Valor"), title="Gráfico del resultado de las métricas", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Recomendaciones
# -----------------------------
st.subheader("🤝 Recomendaciones de amigos")
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

# -----------------------------
# Visualización parcial sincronizada con recomendaciones (Plotly interactivo)
# -----------------------------
st.subheader("🕸️ Visualización del grafo con recomendaciones")
max_nodes = st.slider("Nodos a mostrar", 100, 1000, 500, key="slider_grafo_usuario")
sub_nodes = list(G.nodes())[:max_nodes]
subG = G.subgraph(sub_nodes)

# Definir top_users y recommended_nodes correctamente
if st.session_state.embeddings is not None:
    _, top_users, _ = recommend_detailed(st.session_state.embeddings, data, user_id, top_k)
else:
    top_users = []
recommended_nodes = [n for n in subG.nodes if n in top_users and n != user_id]

highlight_node = user_id if user_id in subG.nodes else None
highlight_edges = [(highlight_node, v) for v in subG.neighbors(highlight_node)] if highlight_node is not None else []

# Aristas recomendadas (usuario → nodos recomendados)
recommended_edges = []
if st.session_state.embeddings is not None and highlight_node is not None:
    for rec in recommended_nodes:
        # Mostrar arista recomendada aunque ya sea vecino
        recommended_edges.append((highlight_node, rec))

# Calcular posiciones
import numpy as np
pos = nx.spring_layout(subG, seed=42)
x_coords = np.array([pos[n][0] for n in subG.nodes])
y_coords = np.array([pos[n][1] for n in subG.nodes])

# Colores y tamaños
node_colors = []
node_sizes = []
for n in subG.nodes:
    if n == highlight_node:
        node_colors.append('red')
        node_sizes.append(18)
    elif n in recommended_nodes:
        node_colors.append('limegreen')
        node_sizes.append(14)
    else:
        node_colors.append('#1f78b4')
        node_sizes.append(8)

node_text = [f"ID: {n}" for n in subG.nodes]

# Edges normales
edge_x = []
edge_y = []
for u, v in subG.edges:
    # No filtrar aristas recomendadas ni actuales
    if highlight_node is not None and (u == highlight_node or v == highlight_node):
        continue
    if (highlight_node, v) in recommended_edges or (highlight_node, u) in recommended_edges:
        continue
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

# Edges actuales (rojo)
actual_edge_x = []
actual_edge_y = []
for u, v in highlight_edges:
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    actual_edge_x += [x0, x1, None]
    actual_edge_y += [y0, y1, None]

# Edges recomendados (verde)
rec_edge_x = []
rec_edge_y = []
for u, v in recommended_edges:
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    rec_edge_x += [x0, x1, None]
    rec_edge_y += [y0, y1, None]

import plotly.graph_objects as go
# Mostrar información básica arriba del gráfico
num_actual_edges = len(highlight_edges)
num_candidate_edges = len(recommended_edges)
if highlight_node is not None:
    st.markdown(f"<div style='font-size:18px;'><b>ID seleccionado:</b> {highlight_node} &nbsp; <b>Aristas actuales:</b> {num_actual_edges} &nbsp; <b>Aristas recomendadas:</b> {num_candidate_edges}</div>", unsafe_allow_html=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='#888', width=1), hoverinfo='none', showlegend=False))
if actual_edge_x:
    fig.add_trace(go.Scatter(x=actual_edge_x, y=actual_edge_y, mode='lines', line=dict(color='red', width=2), hoverinfo='none', showlegend=False))
if rec_edge_x:
    fig.add_trace(go.Scatter(x=rec_edge_x, y=rec_edge_y, mode='lines', line=dict(color='limegreen', width=2, dash='dash'), hoverinfo='none', showlegend=False))
fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='markers', marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color='white')), text=node_text, hoverinfo='text', showlegend=False))
fig.update_layout(title=f"Grafo parcial (Usuario seleccionado: {user_id})", height=600, margin=dict(b=20,l=5,r=5,t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), plot_bgcolor='rgba(240,240,240,0.9)')
st.plotly_chart(fig, use_container_width=True)

# Mostrar información debajo del gráfico
num_actual_edges = len(highlight_edges)
num_candidate_edges = len(recommended_edges)
if highlight_node is not None:
    st.info(f"ID seleccionado: {highlight_node} | Aristas actuales: {num_actual_edges} | Aristas candidatas: {num_candidate_edges}")

# -----------------------------
# Comparación automática de modelos
# -----------------------------
def compare_models(data, epochs=50, ks=[1,3,5,10]):
    modelos = ["GCN", "GraphSAGE", "GAT"]
    resultados = []
    for modelo in modelos:
        m = get_model(modelo, num_features, embedding_dim, edge_index=data.edge_index)
        _, metrics_df = train_and_evaluate(m, data, epochs=epochs, ks=ks)
        for idx, row in metrics_df.iterrows():
            resultados.append({
                "Modelo": modelo,
                "Métrica": row["Métrica"],
                "Valor": float(row["Valor"])
            })
    return pd.DataFrame(resultados)

if st.button("Comparar modelos automáticamente"):
    with st.spinner("Entrenando y comparando modelos..."):
        df_comparacion = compare_models(data)
    st.subheader("📊 Comparación de métricas entre modelos")
    st.dataframe(df_comparacion.pivot(index="Modelo", columns="Métrica", values="Valor"), use_container_width=True)
    # Gráfica comparativa
    fig = go.Figure()
    for metrica in df_comparacion["Métrica"].unique():
        fig.add_trace(go.Bar(
            x=df_comparacion["Modelo"].unique(),
            y=[df_comparacion[(df_comparacion["Modelo"]==modelo) & (df_comparacion["Métrica"]==metrica)]["Valor"].values[0] for modelo in df_comparacion["Modelo"].unique()],
            name=metrica
        ))
    fig.update_layout(barmode='group', title="Comparación de métricas por modelo", yaxis=dict(title="Valor"), xaxis=dict(title="Modelo"), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)