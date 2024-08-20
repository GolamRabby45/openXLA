import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import transformers
from transformers import AdamW

import warnings
warnings.filterwarnings("ignore")

# Configuration
class config:
    MAX_LEN = 224
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 8
    EPOCHS = 1
    MODEL_PATH = "model.bin"
    TRAINING_FILE = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv'
    TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Dataset Class
class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, text, target):
        self.text = text
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = str(self.text[idx])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.target[idx], dtype=torch.long)
        }

# Model Class
class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.bert_drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(768, 1)

    def forward(self, ids, mask):
        _, o2 = self.bert(ids, mask)
        bo = self.bert_drop(o2)
        out = self.fc1(bo)
        return out

# Loss Function
def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

# Evaluation Function
def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

# Training Function
def train_fn(data_loader, model, optimizer, device, scheduler, epoch, num_steps):
    model.train()
    for bi, d in enumerate(data_loader):
        ids = d["ids"]
        mask = d["mask"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask)
        loss = loss_fn(outputs, targets)
        loss.backward()

        # Standard PyTorch optimizer step
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if (bi + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/1], bi[{bi + 1}/{num_steps}], Loss: {loss.item():.4f}')

# Main Run Function
def run():
    train = pd.read_csv(config.TRAINING_FILE).fillna("none").sample(n=4000)
    train_df, valid_df, train_tar, valid_tar = train_test_split(
        train.comment_text, train.toxic, stratify=train.toxic.values, random_state=42, test_size=0.2, shuffle=True
    )

    train_dataset = BERTDataset(text=train_df.values, target=train_tar.values)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4
    )

    valid_dataset = BERTDataset(text=valid_df.values, target=valid_tar.values)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    lr = 3e-5
    num_train_steps = int(len(train_dataset) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=lr)

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        train_fn(train_data_loader, model, optimizer, device, scheduler=None, epoch=epoch, num_steps=num_train_steps)
        outputs, targets = eval_fn(valid_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.roc_auc_score(targets, outputs)
        print(f"AUC_SCORE = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy

# Running the Training and Evaluation Process
if __name__ == "__main__":
    run()
