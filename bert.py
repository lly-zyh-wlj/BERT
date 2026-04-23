import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 加这行！

import torch
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

# ====================== 1. 加载并划分数据集 ======================
# 只选两个类别：无神论 / 基督教
categories = ['alt.atheism', 'soc.religion.christian']

# 加载数据，去掉邮件头、引用等无关内容
dataset = fetch_20newsgroups(
    subset='all',
    categories=categories,
    remove=('headers', 'footers', 'quotes'),
    random_state=42
)

texts = dataset.data
labels = dataset.target

# 划分：训练70%，验证15%，测试15%
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

# ====================== 2. BERT 分词器 ======================
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 128  # 句子最大长度

# 自定义数据集类
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # BERT 预处理：转成模型能看懂的编号
        inputs = tokenizer(
            text,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 构建 DataLoader
batch_size = 8
train_loader = DataLoader(NewsDataset(train_texts, train_labels), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(NewsDataset(val_texts, val_labels), batch_size=batch_size)
test_loader  = DataLoader(NewsDataset(test_texts, test_labels), batch_size=batch_size)

# ====================== 3. 构建 BERT 分类模型 ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # 二分类
)
model.to(device)

# 优化器
optimizer = AdamW(model.parameters(), lr=2e-5)

# ====================== 4. 训练 ======================
epochs = 2
for epoch in range(epochs):
    model.train()
    total_loss = 0

    # 训练批次
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 前向传播
        outputs = model(input_ids, attention_mask=mask, labels=labels)
        loss = outputs.loss

        # 反向传播更新参数
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 每个 epoch 结束后在验证集测试
    model.eval()
    preds = []
    truths = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().numpy())
            truths.extend(labels.cpu().numpy())

    val_acc = accuracy_score(truths, preds)
    print(f"Epoch {epoch+1} | 训练损失: {total_loss:.4f} | 验证准确率: {val_acc:.4f}")

# ====================== 5. 在测试集上最终评估 ======================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=mask)
        pred = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 输出结果
print("\n===== BERT 测试集结果 =====")
print(f"准确率: {accuracy_score(all_labels, all_preds):.4f}")
print(classification_report(all_labels, all_preds, target_names=categories))