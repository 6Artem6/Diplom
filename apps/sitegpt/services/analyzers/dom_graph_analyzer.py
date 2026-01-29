import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class DOMGraphAnalyzer(nn.Module):
    def __init__(self, in_channels=128, hidden_channels=64, out_channels=32):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x  # матрица [N узлов × 32 признака]

    @staticmethod
    def build_dom_graph(dom_tree):
        nodes = []
        edges = []

        for idx, element in enumerate(dom_tree):
            nodes.append(DOMGraphAnalyzer.extract_features(element))
            # parent-child связь
            if element.parent is not None:
                edges.append([idx, element.parent.index])
            # соседство
            if element.previous_sibling is not None:
                edges.append([idx, element.previous_sibling.index])

        data = Data(
            x=torch.tensor(nodes, dtype=torch.float),
            edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        )
        return data

    @staticmethod
    def extract_features(element):
        # простая фича: one-hot тег + длина текста + число атрибутов
        tag_encoding = hash(element.tag) % 64
        return [tag_encoding, len(element.attrs), len(element.text)]
