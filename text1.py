import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Masking

#将数据看作一个6*n维矩阵，n为5-12之间的自然数

# 读取Excel文件
file_path = r"C:\Users\何家乐\AppData\Roaming\JetBrains\PyCharmCE2024.2\scratches\data2.csv"
df = pd.read_csv(file_path)

# 参数设置
num_features = 6  # 每组数据的特征维数（6维向量）
label_column = 6  # 标签列（假设是第七列，Python中从0开始）
num_rows = df.shape[0]  # 数据的行数

# 初始化数据存储
data_groups = []
labels = []

# 遍历数据集，按每组数据的结构读取特征和标签
for i in range(0, num_rows, num_features):
    # 提取6*n的矩阵
    matrix = df.iloc[i:i + num_features, :num_features].values

    # 提取标签，假设第一行第七列是标签
    label = df.iloc[i, label_column]

    # 检查标签是否为空或无效，填充为 "0"（假目标）
    if pd.isna(label) or label not in ['0', '1']:
        label = '0'

    # 保存数据和标签
    data_groups.append(matrix)
    labels.append(label)

# 将每组数据填充为相同长度（max_length）
max_length = max([matrix.shape[0] for matrix in data_groups])

# 使用numpy对每组矩阵进行填充
padded_data = np.array(
    [np.pad(matrix, ((0, max_length - matrix.shape[0]), (0, 0)), mode='constant') for matrix in data_groups])

# 转换标签为numpy数组并确保是浮点数
labels = np.array(labels).astype(float)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(padded_data, labels, test_size=0.2, random_state=42)


# 3. 构建Transformer模型
def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)

    # Masking层忽略填充值
    masked_inputs = Masking(mask_value=0.0)(inputs)

    # 多头自注意力
    attention_output = MultiHeadAttention(num_heads=8, key_dim=input_shape[-1])(masked_inputs, masked_inputs)

    # 残差连接和Layer Norm
    attention_output = LayerNormalization()(attention_output + masked_inputs)

    # 前馈神经网络
    feed_forward = Dense(128, activation='relu')(attention_output)

    # 池化层将序列数据转换为单一输出
    pooled_output = Dense(1, activation='sigmoid')(feed_forward[:, 0, :])  # 使用第一个时间步的输出

    model = Model(inputs=inputs, outputs=pooled_output)
    return model


# 4. 创建模型
input_shape = (max_length, num_features)
model = build_transformer_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 6. 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'测试集损失: {loss}, 准确率: {accuracy}')