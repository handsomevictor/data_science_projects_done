import argparse
from os import path
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from dgl import batch
from dgl.data.ppi import LegacyPPIDataset
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader


PROJECT_DIR = path.dirname(path.abspath(__file__))
MODEL_STATE_FILE = path.join(PROJECT_DIR, "model_state.pth")


class BasicGraphModel(nn.Module):

    def __init__(self, g, n_layers, input_size, hidden_size, output_size, nonlinearity):
        super().__init__()

        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_size, hidden_size, activation=nonlinearity))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(hidden_size, hidden_size, activation=nonlinearity))
        self.layers.append(GraphConv(hidden_size, output_size))

    def forward(self, inputs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = layer(self.g, outputs)

        return outputs


# --------------- my models --------------------


class ConvGraphModel(nn.Module):

    def __init__(self, g, layer_sizes, input_size, output_size, nonlinearity):
        super().__init__()

        n_layers = len(layer_sizes)
        assert n_layers > 0

        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_size, layer_sizes[0], activation=nonlinearity))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(layer_sizes[i], layer_sizes[i+1], activation=nonlinearity))
        self.layers.append(GraphConv(layer_sizes[-1], output_size, 1))

    def forward(self, inputs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = layer(self.g, outputs)

        return outputs


class AttentionGraphModel(nn.Module):

    def __init__(self, g, layer_sizes, input_size, output_size, nonlinearity, num_heads=4):
        super().__init__()

        n_layers = len(layer_sizes)
        assert n_layers > 0

        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(input_size, layer_sizes[0], num_heads, activation=nonlinearity))
        for i in range(n_layers - 1):
            self.layers.append(GATConv(layer_sizes[i] * num_heads, layer_sizes[i+1], num_heads, activation=nonlinearity))
        self.layers.append(GATConv(layer_sizes[-1] * num_heads, output_size, 1))

    def forward(self, inputs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = layer(self.g, outputs)
            outputs = outputs.view(-1, outputs.size(1) * outputs.size(2))

        return outputs


class GAT(nn.Module):

    def __init__(self, g, input_size, output_size, nonlinearity,
                 hidden_size=256, n_layers=2, num_heads=4, skip_connections=True):
        super().__init__()

        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(input_size, hidden_size, num_heads, activation=nonlinearity))
        for i in range(n_layers - 1):
            self.layers.append(GATConv(hidden_size * num_heads, hidden_size, num_heads, activation=nonlinearity))
        self.layers.append(GATConv(hidden_size * num_heads, output_size, 6))

        self.skip_connections = skip_connections
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = layer(self.g, outputs)
            if i < len(self.layers)-1:
                outputs = outputs.view(-1, outputs.size(1) * outputs.size(2))  # Flatten the last two dimensions
                if self.skip_connections and i > 0:
                    outputs = outputs + inputs  # Skip connection across the intermediate attention(s) layer(s)
            inputs = outputs
        outputs = outputs.mean(1)  # Average over the 6 output heads
        # outputs = self.sigmoid(outputs)  # Apply a sigmoid

        return outputs


# --------------- my models (end) --------------------


def main(args):

    # load dataset and create dataloader
    train_dataset, test_dataset = LegacyPPIDataset(mode="train"), LegacyPPIDataset(mode="test")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    n_features, n_classes = train_dataset.features.shape[1], train_dataset.labels.shape[1]

    # create the model, loss function and optimizer
    device = torch.device("cpu" if args.gpu < 0 else "cuda:" + str(args.gpu))
    print("Device:", device)

    model = GAT(g=train_dataset.graph, input_size=n_features,
                output_size=n_classes, nonlinearity=F.elu,
                n_layers=3, num_heads=4, skip_connections=True).to(device)

    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # train
    if args.mode == "train":
        train(model, loss_fcn, device, optimizer, train_dataloader, test_dataset)
        torch.save(model.state_dict(), MODEL_STATE_FILE)

    # import model from file
    model.load_state_dict(torch.load(MODEL_STATE_FILE))

    # test the model
    _test(model, loss_fcn, device, test_dataloader)

    return model


def train(model, loss_fcn, device, optimizer, train_dataloader, test_dataset):

    f1_score_list = []
    epoch_list = []

    for epoch in range(args.epochs):
        model.train()
        losses = []
        start_time = time.time()
        for batch, data in enumerate(train_dataloader):
            subgraph, features, labels = data
            subgraph = subgraph.to(device)
            features = features.to(device)
            labels = labels.to(device)
            model.g = subgraph
            for layer in model.layers:
                layer.g = subgraph
            logits = model(features.float())
            loss = loss_fcn(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_data = np.array(losses).mean()
        print("Epoch {:05d} | Loss: {:.4f} | Time elapsed: {:.2f}s.".format(
            epoch + 1, loss_data, time.time() - start_time))

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            scores = []
            for batch, test_data in enumerate(test_dataset):
                subgraph, features, labels = test_data
                subgraph = subgraph.clone().to(device)
                features = features.clone().detach().to(device)
                labels = labels.clone().detach().to(device)
                score, _ = evaluate(features.float(), model, subgraph, labels.float(), loss_fcn)
                scores.append(score)
            f1_score_list.append(np.array(scores).mean())
            epoch_list.append(epoch)
            print("F1-Score: {:.4f} ".format(np.array(scores).mean()))

    plot_f1_score(epoch_list, f1_score_list)


def _test(model, loss_fcn, device, test_dataloader):
    test_scores = []
    for batch, test_data in enumerate(test_dataloader):
        subgraph, features, labels = test_data
        subgraph = subgraph.to(device)
        features = features.to(device)
        labels = labels.to(device)
        test_scores.append(evaluate(features, model, subgraph, labels.float(), loss_fcn)[0])
    mean_scores = np.array(test_scores).mean()
    print("F1-Score (test): {:.4f}".format(np.array(test_scores).mean()))
    return mean_scores


def evaluate(features, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.layers:
            layer.g = subgraph
        output = model(features.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), predict, average="micro")
        return score, loss_data.item()


def collate_fn(sample):
    # concatenate graph, features and labels w.r.t batch size
    graphs, features, labels = map(list, zip(*sample))
    graph = batch(graphs)
    features = torch.from_numpy(np.concatenate(features))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, features, labels


def plot_f1_score(epoch_list, f1_score_list) :

    plt.plot(epoch_list, f1_score_list)
    plt.title("Evolution of f1 score w.r.t epochs")
    plt.xlabel("Epochs")
    plt.ylabel("F1-score")
    plt.savefig(path.join(PROJECT_DIR, "f1_score.png"))
    # plt.show()


if __name__ == "__main__":

    # PARSER TO ADD OPTIONS
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["train", "test"], default="train")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    # READ MAIN
    main(args)
