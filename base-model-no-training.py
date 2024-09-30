from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# 加载预训练的RoBERTa模型和对应的Tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)  # num_labels根据任务类别修改

# 定义对话的 seeker 和 response
seeker_text = "Just heard a song they play at my work all the time and now I'm depressed because it reminded me I work tomorrow."
response_text = "It's understandable that hearing a song associated with work can evoke feelings of dread or sadness, especially if you're not looking forward to going into work the next day. If you find that your job is causing you distress, it may be helpful to talk to your manager about your concerns and see if there are any accommodations that can be made to make your work environment more enjoyable. It's also important to prioritize self-care and make time for activities that bring you joy outside of work. If you continue to feel down or are concerned about your mental health, please consider reaching out to a trusted person in your life or a mental health professional for support."

# 使用特殊标记 [SEP] 来分隔对话中的 seeker 和 response
combined_text = f"{seeker_text} [SEP] {response_text}"

# 对文本进行tokenization
inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True)

# 模型预测
outputs = model(**inputs)
logits = outputs.logits

# 获取分类结果
predicted_class = torch.argmax(logits, dim=1)

# 输出分类结果
print(f"Predicted class (empathy level): {predicted_class.item()}")
