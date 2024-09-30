import pandas as pd

# 使用pandas读取pkl文件
data = pd.read_pickle('EMO_KNOW_v1.pkl')

# 提取第二列的内容，假设第二列的列名为 'emotion'，请根据实际列名替换
emotion_column = data.iloc[:, 1]  # 使用iloc提取第二列

# 去除重复的感情标签
unique_emotions = emotion_column.drop_duplicates()

# 计算感情标签的数量
num_emotions = len(unique_emotions)

# 输出不重复感情标签的数量和标签内容
print(f"There are {num_emotions} types of emo labels :")
print(unique_emotions.tolist())  # 转换为列表形式输出

