# ===============================
# 📦 Importaciones necesarias
# ===============================
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import train_test_split_edges, negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

# ===============================
# 📂 Cargar dataset SNAP Facebook
# ===============================
# Tu archivo: dataset_snap_facebook.csv
data_path = "dataset_snap_facebook.csv"
df = pd.read_csv(data_path)

# Crear el grafo no dirigido
G = nx.from_pandas_edgelist(df, source='source', target='target')
print(f"Grafo cargado: nodos={G.number_of_nodes()}, aristas={G.number_of_edges()}")

# ===============================
# 🔧 Preparar datos para PyG
# ===============================
from torch_geometric.data import Data

edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
data = Data(edge_index=edge_index)
data.num_nodes = G.number_of_nodes()

# Inicialización de embeddings (aleatorios en lugar de todos 1)
data.x = torch.randn((data.num_nodes, 16))

# Dividir en entrenamiento y prueba
data = train_test_split_edges(data)
print(f"Train edges: {data.train_pos_edge_index.size(1)}, Test edges: {data.test_pos_edge_index.size(1)}")

# ===============================
# 🧠 Modelos GCN, GAT, GraphSAGE
# ===============================
class GNNModel(torch.nn.Module):
    def __init__(self, conv_layer, in_feats, hidden, out_feats, dropout=0.3):
        super().__init__()
        self.conv1 = conv_layer(in_feats, hidden)
        self.conv2 = conv_layer(hidden, out_feats)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def get_model(option, num_features, embedding_dim):
    if option == "GCN":
        return GNNModel(GCNConv, num_features, 32, embedding_dim)
    elif option == "GraphSAGE":
        return GNNModel(SAGEConv, num_features, 32, embedding_dim)
    elif option == "GAT":
        return GNNModel(lambda in_f, out_f: GATConv(in_f, out_f, heads=2, dropout=0.3, concat=False),
                        num_features, 32, embedding_dim)

# ===============================
# ⚙️ Entrenamiento y evaluación
# ===============================
def get_link_preds(emb, edge_index):
    src, dst = edge_index
    return (emb[src] * emb[dst]).sum(dim=-1)

def evaluate(emb, pos_edge_index, neg_edge_index):
    preds = torch.cat([get_link_preds(emb, pos_edge_index), get_link_preds(emb, neg_edge_index)])
    labels = torch.cat([torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))])
    auc = roc_auc_score(labels.cpu(), preds.detach().cpu())
    ap = average_precision_score(labels.cpu(), preds.detach().cpu())
    return auc, ap

def get_ranking_metrics(emb, test_pos_edge_index, k_values=[1,3,5,10]):
    src, dst = test_pos_edge_index
    scores = torch.matmul(emb, emb.t())  # Similaridad total
    metrics = {f"Hits@{k}": 0.0 for k in k_values}
    metrics.update({f"Recall@{k}": 0.0 for k in k_values})
    metrics.update({f"Precision@{k}": 0.0 for k in k_values})

    mrr = 0.0
    for i in range(len(src)):
        s, t = src[i].item(), dst[i].item()
        rank = torch.argsort(scores[s], descending=True)
        position = (rank == t).nonzero(as_tuple=True)[0].item() + 1
        mrr += 1.0 / position
        for k in k_values:
            if position <= k:
                metrics[f"Hits@{k}"] += 1
                metrics[f"Recall@{k}"] += 1
                metrics[f"Precision@{k}"] += 1 / k
    n = len(src)
    for k in k_values:
        metrics[f"Hits@{k}"] /= n
        metrics[f"Recall@{k}"] /= n
        metrics[f"Precision@{k}"] /= n
    metrics["MRR"] = mrr / n
    return metrics

# ===============================
# 🔁 Entrenamiento principal
# ===============================
def train_model(model_name, data, epochs=150, embedding_dim=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, data.x.size(1), embedding_dim).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in tqdm(range(epochs), desc=f"Entrenando {model_name}"):
        model.train()
        optimizer.zero_grad()
        emb = model(data.x, data.train_pos_edge_index)
        pos_pred = get_link_preds(emb, data.train_pos_edge_index)
        neg_edge_index = negative_sampling(data.train_pos_edge_index, num_nodes=data.num_nodes)
        neg_pred = get_link_preds(emb, neg_edge_index)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_pred, neg_pred]),
            torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))]).to(device)
        )
        loss.backward()
        optimizer.step()

    # Evaluación
    model.eval()
    with torch.no_grad():
        emb = model(data.x, data.train_pos_edge_index)
        neg_test_edges = negative_sampling(data.test_pos_edge_index, num_nodes=data.num_nodes)
        auc, ap = evaluate(emb, data.test_pos_edge_index, neg_test_edges)
        ranking_metrics = get_ranking_metrics(emb, data.test_pos_edge_index)
    return {"Modelo": model_name, "AUC": auc, "AP": ap, **ranking_metrics}

# ===============================
# 📊 Comparar todos los modelos
# ===============================
results = []
for model_name in ["GCN", "GraphSAGE", "GAT"]:
    res = train_model(model_name, data)
    results.append(res)

df_results = pd.DataFrame(results)
print("\n=== 📈 Resultados finales ===")
print(df_results)
