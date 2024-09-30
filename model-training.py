# model-training.py

import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from empathy_model import EmpathyClassifier  # 从empathy_model.py中导入EmpathyClassifier
import torch.nn as nn
import torch.optim as optim

# Step 1: 数据转换部分
# ----------------------------
# 加载CSV文件

file_path = r'E:\Study\UoA\705\Epitome\pythonProject\labeled_csv.csv'
df = pd.read_csv(file_path)

# 初始化空列表，用于存储字典格式的数据
data = []

# 定义用于匹配序号（1, 2, 3, 4, 5）的正则表达式模式
split_pattern = r'\d\.\s?'  # 匹配数字+点+空格，如 '1. ', '2. '

# 遍历每一行
for index, row in df.iterrows():
    # 获取seeker的文本
    seeker_text = row.iloc[0]

    # 使用正则表达式分割response
    responses = re.split(split_pattern, row.iloc[3])

    # 如果没有序号，则直接使用整个response
    if len(responses) == 1:
        responses = [row.iloc[3].strip()]  # 没有序号的response，使用原文本
    else:
        # 移除第一个空字符串（如果有）和第一条回复
        responses = [r.strip() for r in responses if r.strip()][1:]  # 移除第一个元素

    label_1 = row.iloc[-2]  # 倒数第二列是label1
    label_2 = row.iloc[-1]  # 最后一列是label2
    average_label = (label_1 + label_2) / 2  # 计算平均值

    # 为每个回复生成一个字典条目
    for response in responses:
        entry = {
            "seeker": seeker_text,
            "response": response,  # 每个分割后的回复
            "label": average_label
        }
        data.append(entry)

# Step 2: 模型训练部分
# ----------------------------

# 加载分词器和RoBERTa模型
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


# 定义自定义数据集类
class EmpathyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取seeker和response
        seeker = self.data[idx]['seeker']
        response = self.data[idx]['response']
        label = self.data[idx]['label']  # 假设有标签，如果没有标签需要自行设定或生成

        # 对文本进行分词和编码，并添加 padding 和 truncation 参数
        seeker_encoding = tokenizer(seeker, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
        response_encoding = tokenizer(response, return_tensors='pt', padding='max_length', max_length=128, truncation=True)

        return seeker_encoding, response_encoding, label

# 定义collate_fn函数来处理不同长度的文本
def collate_fn(batch):
    seeker_inputs = {key: torch.cat([item[0][key] for item in batch], dim=0) for key in batch[0][0].keys()}
    response_inputs = {key: torch.cat([item[1][key] for item in batch], dim=0) for key in batch[0][1].keys()}
    labels = torch.tensor([item[2] for item in batch])
    return seeker_inputs, response_inputs, labels

# 初始化数据集和DataLoader
dataset = EmpathyDataset(data)  # data是转换后的字典列表
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# 初始化模型、损失函数和优化器
model = EmpathyClassifier()
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 训练模型
def train_model(model, dataloader, criterion, optimizer, num_epochs=5):
    model.train()  # 设置为训练模式

    for epoch in range(num_epochs):
        running_loss = 0.0

        for seeker_input, response_input, labels in dataloader:
            # 将输入和标签移动到相同的设备上（如果使用GPU）
            labels = labels.long()

            # 前向传播
            outputs = model(seeker_input, response_input)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')
        # 在所有epoch完成后，保存模型
    torch.save(model.state_dict(), 'empathy_classifier_model.pth')
    print("模型已保存为 'empathy_classifier_model.pth'")
# 开始训练
train_model(model, dataloader, criterion, optimizer, num_epochs=5)
