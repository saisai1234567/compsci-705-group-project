from flask import Flask, render_template, request
import torch
from transformers import RobertaTokenizer
from empathy_model import EmpathyClassifier  # 从empathy_model.py导入模型

# 创建 Flask 应用
app = Flask(__name__)

# 加载分词器和模型
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = EmpathyClassifier()
model.load_state_dict(torch.load('empathy_classifier_model.pth'))  # 加载训练好的模型
model.eval()  # 设置为评估模式

# 首页路由
@app.route('/')
def index():
    return render_template('index.html')  # 渲染 HTML 表单

# 处理表单提交，进行同理心预测
@app.route('/predict', methods=['POST'])
def predict():
    # 从表单中获取seeker_text和response_text
    seeker_text = request.form['seeker_text']
    response_text = request.form['response_text']

    # 对新的seeker和response进行编码
    seeker_encoding = tokenizer(seeker_text, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
    response_encoding = tokenizer(response_text, return_tensors='pt', padding='max_length', max_length=128, truncation=True)

    # 关闭梯度计算，加速推理过程
    with torch.no_grad():
        # 使用模型进行前向传播，得到分类结果
        outputs = model(seeker_encoding, response_encoding)
        probabilities = torch.softmax(outputs, dim=1)  # 获取类别概率
        predicted_class = torch.argmax(probabilities, dim=1)  # 获取最大概率对应的类别

    # 定义同理心类别名称
    empathy_labels = {0: '无同理心', 1: '弱同理心', 2: '强同理心'}

    # 返回预测结果到前端
    return render_template('result.html',
                           seeker_text=seeker_text,
                           response_text=response_text,
                           predicted_class=empathy_labels[predicted_class.item()],
                           probabilities=probabilities)

# 运行应用
if __name__ == '__main__':
    app.run(debug=True)
