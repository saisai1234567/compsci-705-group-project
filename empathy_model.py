# empathy_model.py
import torch
import torch.nn as nn
from transformers import RobertaModel

class EmpathyClassifier(nn.Module):
    def __init__(self):
        super(EmpathyClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.fc = nn.Linear(768 * 2, 3)  # 768是RoBERTa隐藏层大小，3是分类类别（无、弱、强）

    def forward(self, seeker_input, response_input):
        seeker_output = self.roberta(**seeker_input).last_hidden_state[:, 0, :]  # 提问者的输出
        response_output = self.roberta(**response_input).last_hidden_state[:, 0, :]  # 回答者的输出

        # 将seeker和response拼接在一起
        combined_output = torch.cat((seeker_output, response_output), dim=-1)
        logits = self.fc(combined_output)  # 分类输出

        return logits
