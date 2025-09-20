import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
PATIENCE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Carregando dataset")
df = pd.read_csv("dataset/WELFake_Dataset.csv")
df.drop_duplicates(subset=["title", "text", "label"], inplace=True)

# Tem valores faltando :(
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

df["conteudo"] = df["title"] + " " + df["text"]

x_treino, x_teste, y_treino, y_teste = train_test_split(df["conteudo"], df["label"], test_size=0.2, random_state=1234567890, stratify=df["label"])

print(f"Qtd treino: {len(x_treino)}")
print(f"Qtd teste: {len(x_teste)}")

print("Montando TFIDF")
# O ideal seria usar umas 25k features, mas minha GPUzinha de 4GB não aguenta :'(
tfidf = TfidfVectorizer(max_df=0.7, max_features=5000, ngram_range=(1, 2), sublinear_tf=True)

x_treino_tfidf = tfidf.fit_transform(x_treino)
x_teste_tfidf = tfidf.transform(x_teste)

print("Carregando bert")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
bert_model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").to(DEVICE)
bert_model.eval()

def pegar_embeddings_bert(texts, batch_size=32):
    # Obs: função feita com ajuda de IA
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size].tolist()
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(DEVICE)
            outputs = bert_model(**enc)
            # Pega o CLS token
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
    return np.vstack(embeddings)

print("Gerando embeddings dos textos, vai demorar um pouco, pode ir pegar um café ou um pão com manteiga...")
x_treino_bert = pegar_embeddings_bert(x_treino)
x_teste_bert = pegar_embeddings_bert(x_teste)

print("Concatenando TFIDF e embeddigns")
x_treino_combinado = np.hstack([x_treino_tfidf.toarray(), x_treino_bert])
x_teste_combinado = np.hstack([x_teste_tfidf.toarray(), x_teste_bert])

class CombinedDataset(Dataset):
    def __init__(self, x, y):
        self.X = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, indice):
        return self.X[indice], self.y[indice]

dataset_treino = CombinedDataset(x_treino_combinado, y_treino)
dataset_teste = CombinedDataset(x_teste_combinado, y_teste)

loader_treino = DataLoader(dataset_treino, batch_size=BATCH_SIZE, shuffle=True)
loader_teste = DataLoader(dataset_teste, batch_size=BATCH_SIZE)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x)

modelo = MLP(x_treino_combinado.shape[1]).to(DEVICE)
criterio = nn.BCEWithLogitsLoss()
otimizador = optim.Adam(modelo.parameters(), lr=LR)

# Treino...
print("Começando o treino")

melhor_perda_teste = float("inf")
contador_paciencia = 0

# Listas para registrar histórico
historico = {
    "epoca": [],
    "perda_treino": [],
    "acuracia_treino": [],
    "f1_treino": [],
    "perda_teste": [],
    "acuracia_teste": [],
    "f1_teste": []
}

for epoch in range(1, EPOCHS + 1):
    modelo.train()
    perda, y_true, y_pred = 0.0, [], []
    for xb, yb in loader_treino:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)

        otimizador.zero_grad()
        outputs = modelo(xb)
        loss = criterio(outputs, yb)
        loss.backward()
        otimizador.step()

        perda += loss.item() * xb.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).int().cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(yb.cpu().numpy())

    perda_treino = perda / len(loader_treino.dataset)
    acuracia_treino = accuracy_score(y_true, y_pred)
    f1_treino = f1_score(y_true, y_pred)

    modelo.eval()
    perda_corrente_teste, y_true, y_pred = 0.0, [], []
    with torch.no_grad():
        for xb, yb in loader_teste:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
            outputs = modelo(xb)
            loss = criterio(outputs, yb)
            perda_corrente_teste += loss.item() * xb.size(0)

            preds = (torch.sigmoid(outputs) >= 0.5).int().cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(yb.cpu().numpy())

    perda_teste = perda_corrente_teste / len(loader_teste.dataset)
    acuracia_teste = accuracy_score(y_true, y_pred)
    f1_teste = f1_score(y_true, y_pred)

    # Salvar no histórico
    historico["epoca"].append(epoch)
    historico["perda_treino"].append(perda_treino)
    historico["acuracia_treino"].append(acuracia_treino)
    historico["f1_treino"].append(f1_treino)
    historico["perda_teste"].append(perda_teste)
    historico["acuracia_teste"].append(acuracia_teste)
    historico["f1_teste"].append(f1_teste)

    print(f"Epoca {epoch}/{EPOCHS} - "
          f"Perda treino: {perda_treino:.4f} | Acurácia treino: {acuracia_treino:.4f} | F1 Treino: {f1_treino:.4f} || "
          f"Perda teste: {perda_teste:.4f} | Acurácia teste: {acuracia_teste:.4f} | F1 Teste: {f1_teste:.4f}")

    if perda_teste < melhor_perda_teste:
        melhor_perda_teste = perda_teste
        contador_paciencia = 0
        torch.save(modelo.state_dict(), "best_model.pt")
    else:
        contador_paciencia += 1
        if contador_paciencia >= PATIENCE:
            print("Acabou a paciencia, parando o treinamento (p/ evitar overfitting)")
            break

# ========================================
# Plotar evolução das métricas
# ========================================
plt.figure(figsize=(15, 4))

# Perdas
plt.subplot(1, 3, 1)
plt.plot(historico["epoca"], historico["perda_treino"], label="Treino")
plt.plot(historico["epoca"], historico["perda_teste"], label="Teste")
plt.xlabel("Época")
plt.ylabel("Perda")
plt.title("Evolução da perda")
plt.legend()

# Acurácia
plt.subplot(1, 3, 2)
plt.plot(historico["epoca"], historico["acuracia_treino"], label="Treino")
plt.plot(historico["epoca"], historico["acuracia_teste"], label="Teste")
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.title("Evolução da acurácia")
plt.legend()

# F1
plt.subplot(1, 3, 3)
plt.plot(historico["epoca"], historico["f1_treino"], label="Treino")
plt.plot(historico["epoca"], historico["f1_teste"], label="Teste")
plt.xlabel("Época")
plt.ylabel("F1-Score")
plt.title("Evolução do F1-Score")
plt.legend()

plt.tight_layout()
plt.show()