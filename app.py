# app_recomendacion_grafos.py
# Recomendador de amigos con GNNs (PyVis interactivo)
# Mejorado: guarda estado, recomendaciones correctas, visualización controlada

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import tempfile
import os
import random

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, Node2Vec
from sklearn.metrics import roc_auc_score, average_precision_score

from pyvis.network import Network
import streamlit.components.v1 as components

# Optional community detection
try:
    import community as community_louvain
except Exception:
    community_louvain = None

st.set_page_config(layout="wide", page_title="Recomendador GNN (mejorado)")
st.title("🤝 Recomendador de amigos — GNN (mejorado)")

# -------------- Helpers --------------

def detect_columns(df):
    cols = [c.lower().strip() for c in df.columns]
    posibles = [
        ("source", "target"),
        ("src", "dst"),
        ("user1", "user2"),
        ("node1", "node2"),
        ("from", "to"),
        ("id1", "id2"),
    ]
    for s, t in posibles:
        if s in cols and t in cols:
            # return original-case names by mapping
            orig = {c.lower().strip(): c for c in df.columns}
            return orig[s], orig[t]
    # fallback: take first two columns
    return df.columns[0], df.columns[1]

def load_edges_from_csv(uploaded):
    if isinstance(uploaded, str):
        df = pd.read_csv(uploaded)
    else:
        uploaded.seek(0)
        df = pd.read_csv(uploaded)
    s_col, t_col = detect_columns(df)
    edges = df[[s_col, t_col]].astype(int).values
    return edges, df

def remap_ids(edges):
    unique = np.unique(edges.flatten())
    id_map = {old: new for new, old in enumerate(unique)}
    inv_map = {v: k for k, v in id_map.items()}
    remapped = np.vectorize(lambda x: id_map[x])(edges)
    return remapped, id_map, inv_map

def build_pyg_data_from_edges(edges, undirected=True, feat_dim=32):
    if undirected:
        edges2 = np.vstack([edges, edges[:, ::-1]])
    else:
        edges2 = edges
    edge_index = torch.tensor(edges2.T, dtype=torch.long)
    n_nodes = int(edges.max()) + 1
    # use random low-d embeddings to speed up (instead of giant identity)
    x = torch.randn((n_nodes, feat_dim), dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

def tensor_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.requires_grad:
            tensor = tensor.detach()
        return tensor.cpu().numpy()
    return np.array(tensor)

# -------------- Models --------------

class GCNModel(torch.nn.Module):
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

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)
    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)

class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden=64, heads=2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden // heads, heads=heads)
        self.conv2 = GATConv(hidden, hidden // heads, heads=heads)
    def encode(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)
    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)

# -------------- Training/Eval functions --------------

def compute_link_prediction_metrics(y_true, y_score, k_list=[1,5,10]):
    # robust handling
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = float("nan")
    try:
        ap = float(average_precision_score(y_true, y_score))
    except Exception:
        ap = float("nan")
    results = {"AUC": auc, "AP": ap}
    order = np.argsort(-y_score)
    sorted_y = np.array(y_true)[order]
    total_pos = max(1, int(np.sum(y_true)))
    for k in k_list:
        topk = sorted_y[:k]
        hits = int(np.sum(topk))
        results[f"Hits@{k}"] = hits
        results[f"Precision@{k}"] = float(hits / k)
        results[f"Recall@{k}"] = float(hits / total_pos)
    # MRR
    ranks = np.where(sorted_y == 1)[0]
    results["MRR"] = float(np.mean(1.0/(ranks+1))) if len(ranks)>0 else 0.0
    return results

def train_gnn_and_get_embeddings(model, data, epochs=10, lr=0.01, device="cpu"):
    device = torch.device(device)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # RandomLinkSplit already applied outside; data must have edge_label_index & edge_label
    edge_label_index = data.edge_label_index
    edge_label = data.edge_label.float().to(device)

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
        embeddings = z.detach().cpu().numpy()
    return metrics, embeddings, model

def train_node2vec_and_get_embeddings(data, epochs=10, lr=0.01, embed_dim=64, device="cpu"):
    device = torch.device(device)
    node2vec = Node2Vec(data.edge_index, embedding_dim=embed_dim, walk_length=20, context_size=10, walks_per_node=5, num_negative_samples=1, sparse=True).to(device)
    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=lr)
    node2vec.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = node2vec.loss()
        loss.backward()
        optimizer.step()
    node2vec.eval()
    with torch.no_grad():
        embeddings = node2vec.embedding.weight.detach().cpu().numpy()
    # for metrics, create a split and compute simple dot scores
    data_split = RandomLinkSplit(is_undirected=True, add_negative_train_samples=True)(data)[2]  # test part
    pos_idx = data_split.edge_label_index[:, data_split.edge_label==1]
    neg_idx = data_split.edge_label_index[:, data_split.edge_label==0]
    pos_scores = (embeddings[pos_idx[0]] * embeddings[pos_idx[1]]).sum(axis=1)
    neg_scores = (embeddings[neg_idx[0]] * embeddings[neg_idx[1]]).sum(axis=1)
    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_score = np.concatenate([pos_scores, neg_scores])
    metrics = compute_link_prediction_metrics(y_true, y_score)
    return metrics, embeddings, node2vec

# -------------- PyVis helpers --------------

def pyvis_from_nx(G, highlight_nodes=None, highlight_center=None, bg_color="#222", font_color="white"):
    net = Network(height="650px", width="100%", notebook=False, bgcolor=bg_color, font_color=font_color)
    # prefer deterministic layout with physics enabled
    net.barnes_hut()
    # add nodes
    for n in G.nodes():
        title = f"Node {n}"
        color = None
        size = 10
        if highlight_nodes and n in highlight_nodes:
            color = "orange"
            size = 22
        if highlight_center is not None and n == highlight_center:
            color = "red"
            size = 28
        net.add_node(n, label=str(n), title=title, color=color, size=size)
    for u, v in G.edges():
        net.add_edge(u, v)
    # options
    net.set_options("""
    var options = {
      "nodes": { "font": {"size": 12} },
      "physics": { "barnesHut": {"gravitationalConstant": -20000, "springLength": 100} }
    }
    """)
    return net

def display_pyvis(net):
    tmpfile = os.path.join(tempfile.gettempdir(), f"pyvis_{random.randint(0,10**9)}.html")
    net.save_graph(tmpfile)
    with open(tmpfile, "r", encoding="utf-8") as f:
        html = f.read()
    components.html(html, height=650, scrolling=True)

# -------------- App UI --------------

# session state init
if "trained" not in st.session_state:
    st.session_state["trained"] = False
if "model" not in st.session_state:
    st.session_state["model"] = None
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = None
if "inv_map" not in st.session_state:
    st.session_state["inv_map"] = None
if "id_map" not in st.session_state:
    st.session_state["id_map"] = None
if "edges_remapped" not in st.session_state:
    st.session_state["edges_remapped"] = None

st.sidebar.header("1) Dataset")
uploaded = st.sidebar.file_uploader("Sube CSV (src,dst o source,target)", type=["csv"])
# also allow local files in ./datasets/
local_opts = {"SNAP example": "./datasets/snap_full_like.csv", "OGB example": "./datasets/ogb_medium_like.csv"}
sel_local = st.sidebar.selectbox("O usar dataset local (ejemplos)", ["-- ninguno --"] + list(local_opts.keys()))
if sel_local != "-- ninguno --":
    path = local_opts[sel_local]
    if os.path.exists(path):
        uploaded = path

if not uploaded:
    st.info("Sube o selecciona un dataset para continuar.")
    st.stop()

# Load edges and remap ids
try:
    edges_orig, df = load_edges_from_csv(uploaded)
except Exception as e:
    st.error(f"Error leyendo CSV: {e}")
    st.stop()

# remap IDs to contiguous 0..N-1
edges_remapped, id_map, inv_map = remap_ids(edges_orig)
n_nodes = int(edges_remapped.max()) + 1
st.session_state["id_map"] = id_map
st.session_state["inv_map"] = inv_map
st.session_state["edges_remapped"] = edges_remapped

st.write("### Vista previa dataset")
st.dataframe(df.head())
st.write(f"Nodos (remapped): {n_nodes} — Aristas: {edges_remapped.shape[0]}")

# Visualization controls
st.sidebar.header("2) Visualización")
MAX_NODES_FULL = st.sidebar.number_input("Máx nodos para mostrar grafo completo (seguro)", min_value=200, max_value=5000, value=1000)
sample_size = st.sidebar.number_input("Muestra PyVis (aristas)", min_value=100, max_value=5000, value=500, step=100)

# Show full graph if small enough, otherwise sample but show button to attempt full
st.subheader("🌐 Visualización del grafo")
G_full = nx.Graph()
G_full.add_edges_from(edges_remapped.tolist())

if G_full.number_of_nodes() <= MAX_NODES_FULL:
    st.write(f"Mostrando grafo completo ({G_full.number_of_nodes()} nodos).")
    net = pyvis_from_nx(G_full)
    display_pyvis(net)
else:
    st.warning(f"Grafo grande ({G_full.number_of_nodes()} nodos). Mostrando una muestra de {sample_size} aristas para rendimiento.")
    # sample edges (keeping node ids in remapped space)
    if edges_remapped.shape[0] > sample_size:
        idx = np.random.choice(edges_remapped.shape[0], sample_size, replace=False)
        sampled = edges_remapped[idx]
    else:
        sampled = edges_remapped
    G_sample = nx.Graph()
    G_sample.add_edges_from(sampled.tolist())
    net = pyvis_from_nx(G_sample)
    display_pyvis(net)
    if st.button("Mostrar grafo completo (puede tardar / bloquear)"):
        net_full = pyvis_from_nx(G_full)
        display_pyvis(net_full)

# Community filter panel
st.sidebar.header("3) Filtrado / Selección")
if community_louvain is not None:
    try:
        partition_full = community_louvain.best_partition(G_full)
        ncom = len(set(partition_full.values()))
        sel = st.sidebar.selectbox("Filtrar por comunidad", ["Todas"] + [f"Comunidad {i}" for i in range(ncom)])
        if sel != "Todas":
            cid = int(sel.split()[1])
            nodes_in_com = [n for n, c in partition_full.items() if c == cid]
            subG = G_full.subgraph(nodes_in_com).copy()
            st.write(f"Subgrafo comunidad {cid} — nodos: {subG.number_of_nodes()}, aristas: {subG.number_of_edges()}")
            netc = pyvis_from_nx(subG)
            display_pyvis(netc)
    except Exception as e:
        st.info("Error calculando comunidades: " + str(e))
else:
    st.info("Para filtrar por comunidad instala python-louvain (pip install python-louvain).")

# Model selection
st.sidebar.header("4) Modelo y entrenamiento")
model_choice = st.sidebar.selectbox("Modelo", ["GCN", "GraphSAGE", "GAT", "Node2Vec"])
epochs = st.sidebar.slider("Epochs", 1, 100, 15)
lr = st.sidebar.number_input("Learning rate", value=0.01, format="%.5f")
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Dispositivo: {device}")

# Train button
if st.sidebar.button("🚀 Entrenar modelo"):
    st.info("Entrenando... (se guardarán modelo y embeddings en memoria de sesión)")
    pyg_data = build_pyg_data_from_edges(edges_remapped)
    # split using RandomLinkSplit
    transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=True)
    train_data, val_data, test_data = transform(pyg_data)
    # use train_data for training but it already contains edge_label_index & edge_label
    if model_choice == "GCN":
        model = GCNModel(in_channels=train_data.num_node_features)
        metrics, embeddings, model_trained = train_gnn_and_get_embeddings(model, train_data, epochs=epochs, lr=lr, device=device)
    elif model_choice == "GraphSAGE":
        model = GraphSAGEModel(in_channels=train_data.num_node_features)
        metrics, embeddings, model_trained = train_gnn_and_get_embeddings(model, train_data, epochs=epochs, lr=lr, device=device)
    elif model_choice == "GAT":
        model = GATModel(in_channels=train_data.num_node_features)
        metrics, embeddings, model_trained = train_gnn_and_get_embeddings(model, train_data, epochs=epochs, lr=lr, device=device)
    else:
        metrics, embeddings, model_trained = train_node2vec_and_get_embeddings(pyg_data, epochs=epochs, lr=lr, embed_dim=64, device=device)

    # Save to session state
    st.session_state["trained"] = True
    st.session_state["model"] = model_trained
    st.session_state["embeddings"] = embeddings
    st.session_state["metrics"] = metrics

    st.success("Entrenamiento completado")
    st.subheader("📈 Métricas")
    st.table(pd.DataFrame(metrics, index=["valor"]).T)

# Show current training status / metrics
if st.session_state.get("trained", False):
    st.info("Modelo entrenado en esta sesión — puedes generar recomendaciones.")
    st.write("Últimas métricas:")
    st.table(pd.DataFrame(st.session_state["metrics"], index=["valor"]).T)

# Recommendations UI
st.subheader("🤝 Recomendaciones de amistad")
top_k = st.number_input("Top K recomendaciones", min_value=1, max_value=100, value=10)
selected_original = st.selectbox("Selecciona usuario (ID original)", list(st.session_state["inv_map"].values())[:1000])
# Map original -> remapped
selected_remapped = None
for orig, rem in st.session_state["id_map"].items():
    if orig == selected_original:
        selected_remapped = rem
        break
if selected_remapped is None:
    # fallback: if user chose index rather than original id
    try:
        selected_remapped = int(selected_original)
    except:
        selected_remapped = 0

if st.button("Generar recomendaciones y mostrar aristas actuales"):
    if not st.session_state.get("trained", False):
        st.error("Entrena un modelo primero (botón en la barra lateral).")
    else:
        embeddings = st.session_state["embeddings"]  # numpy array (N, dim)
        if isinstance(embeddings, torch.Tensor):
            embeddings = tensor_to_numpy(embeddings)
        if selected_remapped >= embeddings.shape[0]:
            st.error("El ID seleccionado excede el tamaño de los embeddings (remapped).")
        else:
            # compute scores using dot product
            z = embeddings
            # normalize optional
            #scores = (z[selected_remapped] * z).sum(axis=1)
            scores = z @ z[selected_remapped]  # (N,)
            # mask existing neighbors
            Gmask = nx.Graph()
            Gmask.add_edges_from(edges_remapped.tolist())
            neighbors = set(Gmask.neighbors(selected_remapped)) if selected_remapped in Gmask else set()
            # build candidates excluding neighbors and self
            candidates = [(i, float(scores[i])) for i in range(len(scores)) if i != selected_remapped and i not in neighbors]
            # sort and take top_k
            candidates_sorted = sorted(candidates, key=lambda x: -x[1])[:top_k]
            # map back to original IDs
            recs_display = [(st.session_state["inv_map"][i], sc) for i, sc in candidates_sorted]
            st.subheader(f"Aristas actuales del usuario {st.session_state['inv_map'][selected_remapped]}")
            # show current neighbors (original ids)
            neighs_orig = [st.session_state["inv_map"][n] for n in neighbors] if neighbors else []
            st.write(f"Vecinos actuales (count={len(neighs_orig)}):")
            st.write(neighs_orig)

            st.subheader("Recomendaciones (Top K)")
            st.table(pd.DataFrame(recs_display, columns=["user_original_id", "score"]))

            # build a pyvis view highlighting selected node and its neighbors + recommendations
            highlight_nodes = set([i for i, _ in candidates_sorted])
            net = pyvis_from_nx(G_full, highlight_nodes=highlight_nodes, highlight_center=selected_remapped)
            display_pyvis(net)

st.markdown("---")
st.markdown("Notas: los IDs se remapearon internamente para asegurar contigüidad. 'user_original_id' muestra el id original del CSV. Para grafos muy grandes la visualización puede tardar o bloquear; usa la muestra o ajusta el parámetro de muestra.")
