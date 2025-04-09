from dataset import create_datasets_from_samples
from model import SimpleSNN
from spike_encoding import rate_code_zscore_sigmoid
from utils import auto_label_multiclass

import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
from spikingjelly.activation_based import functional
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# ▶️ 데이터 준비
samples = auto_label_multiclass("./data", window_size=20, stride=5)
datasets = create_datasets_from_samples(samples, rate_code_zscore_sigmoid, T=20)
full_dataset = ConcatDataset(datasets)

# ▶️ Train/Validation 분할
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ▶️ 모델 및 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleSNN(num_classes=3).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ▶️ 학습 루프
for epoch in range(20):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        functional.reset_net(model)  # SNN state reset

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    # ▶️ Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            functional.reset_net(model)
            out = model(x)
            val_correct += (out.argmax(1) == y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total

    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # 🔍 Confusion Matrix + Classification Report (마지막 Epoch에만 출력하고 싶다면 조건 추가 가능)
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            functional.reset_net(model)
            out = model(x)
            pred = out.argmax(1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    print("\n📊 Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=["BG", "Up", "Down"], digits=4))

    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BG", "Up", "Down"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix (Epoch {epoch+1})")
    plt.show()

