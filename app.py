import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx
from community import community_louvain
import plotly.graph_objects as go

# -------------------------------
# 🔧 MODELOS GNN
# -------------------------------
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, out_channels)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=4, concat=True)
        self.conv2 = GATConv(hidden * 4, out_channels, heads=1, concat=False)

    def encode(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


# -------------------------------
# 📊 MÉTRICAS DE ENLACES
# -------------------------------
def compute_metrics(y_true, y_score, k=10):
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    order = np.argsort(-y_score)
    hits_k = np.mean(y_true[order][:k])
    precision_k = np.mean(y_true[order][:k])
    recall_k = np.sum(y_true[order][:k]) / np.sum(y_true) if np.sum(y_true) > 0 else 0
    mrr = 0
    for rank, idx in enumerate(order, 1):
        if y_true[idx] == 1:
            mrr = 1.0 / rank
            break
    return {
        "AUC": round(auc, 4),
        "AP": round(ap, 4),
        "Hits@10": round(hits_k, 4),
        "Precision@10": round(precision_k, 4),
        "Recall@10": round(recall_k, 4),
        "MRR": round(mrr, 4),
    }


# -------------------------------
# 🔄 ENTRENAMIENTO
# -------------------------------
def train_and_evaluate(model, data, epochs=50, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        src, dst = data.edge_index
        scores = (z[src] * z[dst]).sum(dim=1)
        loss = -torch.mean(torch.log(torch.sigmoid(scores)))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        src, dst = data.edge_index
        y_true = np.ones(len(src))
        y_score = torch.sigmoid((z[src] * z[dst]).sum(dim=1)).cpu().numpy()
        metrics = compute_metrics(y_true, y_score)
    return metrics, z.detach()


# -------------------------------
# 🌐 VISUALIZACIÓN INTERACTIVA
# -------------------------------
def plotly_graph(G, highlight_nodes=None, highlight_center=None, title="Grafo de usuarios"):
    pos = nx.spring_layout(G, seed=42, k=0.3)
    x_edges, y_edges = [], []
    for e in G.edges():
        x_edges += [pos[e[0]][0], pos[e[1]][0], None]
        y_edges += [pos[e[0]][1], pos[e[1]][1], None]

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_color, node_size = [], []

    for n in G.nodes():
        if highlight_center is not None and n == highlight_center:
            node_color.append("red")
            node_size.append(25)
        elif highlight_nodes and n in highlight_nodes:
            node_color.append("orange")
            node_size.append(15)
        else:
            node_color.append("skyblue")
            node_size.append(8)

    edge_trace = go.Scatter(
        x=x_edges, y=y_edges, line=dict(width=0.5, color="#888"),
        hoverinfo="none", mode="lines"
    )
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(size=node_size, color=node_color, line_width=1),
        text=[f"Nodo {n}" for n in G.nodes()],
        hoverinfo="text"
    )
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title=title, showlegend=False, hovermode="closest",
                                     margin=dict(b=0, l=0, r=0, t=40),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     height=650))
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------
# 🚀 INTERFAZ STREAMLIT
# -------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Recomendador de Amigos con GNNs")
    st.title("🧠 Sistema de Recomendación de Amigos con Redes Neuronales de Grafos")

    uploaded_file = st.file_uploader("📂 Carga un dataset CSV (source, target)", type=["csv"])
    model_name = st.selectbox("🧩 Selecciona un modelo", ["GCN", "GraphSAGE", "GAT"])
    epochs = st.slider("⏱️ Épocas", 10, 200, 50)
    lr = st.number_input("💡 Learning Rate", 0.0001, 0.1, 0.01, step=0.001)

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "source" not in df.columns or "target" not in df.columns:
            st.error("El CSV debe tener columnas: source, target")
            return

        G = nx.from_pandas_edgelist(df, source="source", target="target")
        st.success(f"Grafo cargado con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")

        communities = community_louvain.best_partition(G)
        nx.set_node_attributes(G, communities, "community")

        community_options = ["Todas"] + sorted(list(set(communities.values())))
        selected_community = st.selectbox("🧩 Filtrar por comunidad", community_options)

        if selected_community != "Todas":
            nodes_in_comm = [n for n, c in communities.items() if c == selected_community]
            G_sub = G.subgraph(nodes_in_comm)
            st.info(f"Comunidad seleccionada: {selected_community} — Nodos: {len(G_sub)}")
            plotly_graph(G_sub, title=f"Comunidad {selected_community}")
        else:
            plotly_graph(G, title="Grafo completo")

        num_nodes = G.number_of_nodes()
        x = torch.eye(num_nodes)
        edges = np.array(list(G.edges())).T
        edge_index = torch.tensor(edges, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)

        if st.button("🚀 Entrenar modelo"):
            if model_name == "GCN":
                model = GCN(num_nodes, 64, 32)
            elif model_name == "GraphSAGE":
                model = GraphSAGE(num_nodes, 64, 32)
            else:
                model = GAT(num_nodes, 64, 32)

            metrics, z = train_and_evaluate(model, data, epochs=epochs, lr=lr)
            st.subheader("📈 Métricas de evaluación")
            st.dataframe(pd.DataFrame(metrics, index=["Valor"]).T)

            st.subheader("🎯 Recomendaciones de amistad")
            selected_user = st.selectbox("Selecciona un usuario", list(G.nodes()))
            if st.button("Generar recomendaciones"):
                scores = (z[selected_user] @ z.T).detach().numpy()
                neighbors = list(G.neighbors(selected_user))
                candidates = [(i, s) for i, s in enumerate(scores)
                              if i != selected_user and i not in neighbors]
                recs = sorted(candidates, key=lambda x: -x[1])[:10]
                st.dataframe(pd.DataFrame(recs, columns=["Usuario", "Puntaje"]))
                plotly_graph(G, highlight_nodes=[r[0] for r in recs],
                             highlight_center=selected_user,
                             title=f"Recomendaciones para el usuario {selected_user}")


if __name__ == "__main__":
    main()
