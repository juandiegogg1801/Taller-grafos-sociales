import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from pyvis.network import Network
from pyvis.network import Network
from pyvis.network import Network
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, ndcg_score, average_precision_score
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
st.set_page_config(page_title="Recomendador de Amigos Final", layout="wide")
st.title("💡 Recomendador de Amigos con Link Prediction - Final")

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Sidebar: Dataset
# -----------------------------
st.sidebar.header("📂 Dataset")
dataset_option = st.sidebar.selectbox("Selecciona dataset", ["SNAP simulado", "Sintético", "Cargar CSV"])
dataset_option = st.sidebar.selectbox("Selecciona dataset", ["SNAP simulado", "Sintético", "Cargar CSV"])
if dataset_option == "Cargar CSV":
    uploaded_file = st.sidebar.file_uploader("Sube tu CSV (src,dst)", type=['csv'])

@st.cache_data
def load_dataset(option, uploaded_file=None):
    if option == "Cargar CSV" and uploaded_file:
        # Leer el archivo facebook_combined.txt completo
        # Si no existe, mostrar instrucción para descargarlo
        import os
        snap_path = "facebook_combined.txt"
        if not os.path.exists(snap_path):
            st.error("El archivo facebook_combined.txt real no está presente. Descárgalo desde https://snap.stanford.edu/data/facebook_combined.txt y colócalo en la raíz del proyecto.")
            st.stop()
        edge_list = []
        with open(snap_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    edge_list.append((int(parts[0]), int(parts[1])))
        df = pd.DataFrame(edge_list, columns=['src', 'dst'])
        with open("facebook_combined.txt", "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    edge_list.append((int(parts[0]), int(parts[1])))
        df = pd.DataFrame(edge_list, columns=['src', 'dst'])
        with open("facebook_combined.txt", "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    edge_list.append((int(parts[0]), int(parts[1])))
        df = pd.DataFrame(edge_list, columns=['src', 'dst'])
        df = pd.DataFrame({'src': np.random.randint(0,200,1000),
                           'dst': np.random.randint(0,200,1000)})
    G = nx.from_pandas_edgelist(df, 'src', 'dst')
                           'dst': np.random.randint(0,200,1000)})
    G = nx.from_pandas_edgelist(df, 'src', 'dst')

df_edges, G = load_dataset(dataset_option, uploaded_file)

# -----------------------------
# Resumen del grafo
# -----------------------------
st.subheader("📊 Resumen del grafo")
st.write(f"Nodos: {G.number_of_nodes()}, Aristas: {G.number_of_edges()}, Densidad: {nx.density(G):.4f}")

# Visualización del grafo original (sin recomendaciones)
st.subheader("🕸️ Grafo original")
max_nodes_original = st.slider("Nodos a mostrar (grafo original)", 100, 1000, 500, key="slider_grafo_original")
sub_nodes_original = list(G.nodes())[:max_nodes_original]
subG_original = G.subgraph(sub_nodes_original)

import numpy as np
pos_original = nx.spring_layout(subG_original, seed=42)
x_coords_original = np.array([pos_original[n][0] for n in subG_original.nodes])
y_coords_original = np.array([pos_original[n][1] for n in subG_original.nodes])


    num_test = int(len(edges_list)*test_ratio)
    test_edges = edges_list[:num_test]
    num_test = int(len(edges_list)*test_ratio)
    test_edges = edges_list[:num_test]
    train_edges = edges_list[num_test:]
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1).cpu()
    y_true = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])
    y_scores = torch.cat([pos_scores, neg_scores])
        for n in train_adj[u]: mask[n]=True
    ap = average_precision_score(y_true, y_scores)
    # --- Métricas ---
        if scores[v]==float("-inf"): continue
    train_adj = {i:set() for i in range(num_nodes)}
def train_and_evaluate(_model, data, epochs=50, ks=[1,3,5,10]):
    y_true = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])
    y_scores = torch.cat([pos_scores, neg_scores])
        u,v = int(u.item()), int(v.item())
    ap = average_precision_score(y_true, y_scores)
    # --- Métricas ---
        for n in train_adj[u]: mask[n]=True
    train_adj = {i:set() for i in range(num_nodes)}
    pos_scores = (z[test_edges[:,0]] * z[test_edges[:,1]]).sum(dim=1).cpu()
    # --- Métricas ---
        train_adj[u].add(v)
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1).cpu()
    pos_scores = (z[test_edges[:,0]] * z[test_edges[:,1]]).sum(dim=1).cpu()
    y_scores = torch.cat([pos_scores, neg_scores])
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1).cpu()
        ks = [1, 3, 5, 10]
    y_scores = torch.cat([pos_scores, neg_scores])
        train_adj[v].add(u)
    ap = average_precision_score(y_true, y_scores)
    for i in range(num_test):
    train_adj = {i:set() for i in range(num_nodes)}
    for u,v in edge_index.cpu().T.numpy():
        train_adj[u].add(v)
        train_adj[v].add(u)
    # --- Nuevas métricas ---
    from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score
    from sklearn.metrics import ndcg_score, average_precision_score
    import numpy as np

        scores[mask]=float("-inf")
        ks = [1, 3, 5, 10]
    pos_scores = (z[test_edges[:,0]] * z[test_edges[:,1]]).sum(dim=1).cpu()
    for i in range(num_test):
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1).cpu()
    y_true = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])
    y_scores = torch.cat([pos_scores, neg_scores])

    # RMSE y MAE
    rmse = mean_squared_error(y_true, y_scores, squared=False)
    mae = mean_absolute_error(y_true, y_scores)
        scores = torch.matmul(z,z[u]).cpu()
    pos_scores = (z[test_edges[:,0]] * z[test_edges[:,1]]).sum(dim=1).cpu()
    y_pred = (y_scores >= 0.5).int()
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1).cpu()
    y_true = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])
    y_scores = torch.cat([pos_scores, neg_scores])

    # RMSE y MAE
    rmse = mean_squared_error(y_true, y_scores, squared=False)
    mae = mean_absolute_error(y_true, y_scores)
    # NDCG (usando sklearn, requiere arrays 2D)
    # Precision y Recall (binario, umbral 0.5)
    y_pred = (y_scores >= 0.5).int()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    # HR (Hit Rate): proporción de positivos correctamente identificados
    pos_scores = (z[test_edges[:,0]] * z[test_edges[:,1]]).sum(dim=1).cpu()
    ndcg = ndcg_score([y_true.numpy()], [y_scores.numpy()])
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1).cpu()
    y_true = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])
    y_scores = torch.cat([pos_scores, neg_scores])

    # RMSE y MAE
    rmse = np.sqrt(mean_squared_error(y_true, y_scores))
    mae = mean_absolute_error(y_true, y_scores)

    # Precision y Recall (binario, umbral 0.5)
    y_pred = (y_scores >= 0.5).int()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # NDCG (usando sklearn, requiere arrays 2D)
    ndcg = ndcg_score([y_true.numpy()], [y_scores.numpy()])

    # MAP (media de average_precision_score por usuario)
    map_score = average_precision_score(y_true, y_scores)

    # HR (Hit Rate): proporción de positivos correctamente identificados
    hr = (y_pred[y_true==1].sum().item()) / (y_true==1).sum().item()


        "Métrica": ["RMSE", "MAE", "Precision", "Recall", "NDCG", "MAP", "HR"],
        "Valor": [rmse, mae, precision, recall, ndcg, map_score, hr]
