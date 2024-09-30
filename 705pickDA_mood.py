import pandas as pd

# 使用pandas读取pkl文件
data = pd.read_pickle('EMO_KNOW_v1.pkl')

# 提取第二列的内容，假设第二列的列名为 'emotion'，请根据实际列名替换
emotion_column = data.iloc[:, 1]  # 使用iloc提取第二列

# 筛选标签为 'depressed' 和 'anxious' 的数据
filtered_data = data[(emotion_column == 'depressed') | (emotion_column == 'anxious')]

# 统计符合条件的数据条数
num_filtered_data = len(filtered_data)

# 输出筛选后的数据条数和数据内容
print(f"一共有 {num_filtered_data} 条数据符合条件，它们是：")
print(filtered_data)

# 将筛选后的数据导出为Excel文件
filtered_data.to_excel('filtered_emotions.xlsx', index=False)

# 将筛选后的数据导出为pkl文件
filtered_data.to_pickle('filtered_emotions.pkl')

print("筛选后的数据已保存为 filtered_emotions.xlsx")
