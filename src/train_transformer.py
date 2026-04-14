import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import yaml
import mlflow
import mlflow.pytorch
import os
from math import sqrt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from sklearn.metrics import accuracy_score, f1_score

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def scaled_dot_product(query, key, value):
    dim_k = key.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

class AttentionHead(nn.Module):
    def __init__(self, embd_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embd_dim, head_dim)
        self.k = nn.Linear(embd_dim, head_dim)
        self.v = nn.Linear(embd_dim, head_dim)

    def forward(self, hidden_state):
        return scaled_dot_product(
            self.q(hidden_state),
            self.k(hidden_state),
            self.v(hidden_state)
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embd_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embd_dim // num_heads
        self.heads = nn.ModuleList([
            AttentionHead(embd_dim, head_dim) for _ in range(num_heads)
        ])
        self.output_layer = nn.Linear(embd_dim, embd_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        return self.output_layer(x)

class FeedForwardLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.layer_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        return self.dropout(self.layer_2(self.gelu(self.layer_1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForwardLayer(config)

    def forward(self, x):
        hidden_state = self.layer_norm_1(x)
        x = x + self.attention(hidden_state)
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        embeddings = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        return self.dropout(self.layer_norm(embeddings))

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.encoder(x)[:, 0, :]
        return self.classifier(self.dropout(x))

class IMDBDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], torch.tensor(self.labels[idx])

def tokenize_dataset(df, tokenizer, max_length):
    all_input_ids = []
    all_labels = []
    for _, row in df.iterrows():
        inputs = tokenizer(
            row["review"], return_tensors="pt",
            add_special_tokens=True, max_length=max_length,
            padding="max_length", truncation=True
        )
        all_input_ids.append(inputs.input_ids.squeeze(0))
        all_labels.append(row["sentiment"])
    return all_input_ids, all_labels

def train_transformer():
    params = load_params()
    processed_path = params["data"]["processed_path"]
    t_params = params["transformer"]
    experiment_name = params["mlflow"]["experiment_name"]

    mlflow.set_experiment(experiment_name)

    train_df = pd.read_csv(os.path.join(processed_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(processed_path, "test.csv"))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = AutoConfig.from_pretrained("bert-base-uncased")
    config.hidden_size = t_params["hidden_size"]
    config.num_attention_heads = t_params["num_attention_heads"]
    config.num_hidden_layers = t_params["num_hidden_layers"]
    config.intermediate_size = t_params["intermediate_size"]
    config.hidden_dropout_prob = t_params["hidden_dropout_prob"]
    config.num_labels = t_params["num_labels"]

    print("Tokenizing dataset...")
    train_ids, train_labels = tokenize_dataset(train_df, tokenizer, t_params["max_length"])
    test_ids, test_labels = tokenize_dataset(test_df, tokenizer, t_params["max_length"])

    train_loader = DataLoader(IMDBDataset(train_ids, train_labels), batch_size=t_params["batch_size"], shuffle=True)
    test_loader = DataLoader(IMDBDataset(test_ids, test_labels), batch_size=t_params["batch_size"], shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TransformerForSequenceClassification(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t_params["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    with mlflow.start_run(run_name="CustomTransformer"):
        mlflow.log_params(t_params)

        for epoch in range(t_params["epochs"]):
            model.train()
            total_loss = 0
            for input_ids, labels in train_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(input_ids), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for input_ids, labels in test_loader:
                    input_ids = input_ids.to(device)
                    preds = torch.argmax(model(input_ids), dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="weighted")

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            mlflow.log_metric("f1_score", f1, step=epoch)

            print(f"Epoch {epoch+1}/{t_params['epochs']} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/transformer.pt")
        mlflow.pytorch.log_model(model, "transformer_model")
        print("Transformer training complete!")

if __name__ == "__main__":
    train_transformer()