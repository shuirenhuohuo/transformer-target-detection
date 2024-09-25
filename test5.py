import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 自定义 Dataset 类
class UAVDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 2. 数据处理流程
def process_data(file_path, num_features=6, label_column=6):
    df = pd.read_csv(file_path)
    num_rows = df.shape[0]

    data_groups = []
    labels = []

    for i in range(0, num_rows, num_features):
        matrix = df.iloc[i:i + num_features, :num_features].values

        if np.isnan(matrix).any() or np.isinf(matrix).any():
            continue

        label = df.iloc[i, label_column]

        if pd.isna(label) or label not in ['0', '1']:
            label = '0'

        data_groups.append(matrix)
        labels.append(label)

    max_length = max([matrix.shape[0] for matrix in data_groups])

    padded_data = np.array(
        [np.pad(matrix, ((0, max_length - matrix.shape[0]), (0, 0)), mode='constant', constant_values=0) for matrix in
         data_groups]
    )

    scaler = StandardScaler()
    padded_data_reshaped = padded_data.reshape(-1, num_features)
    padded_data_reshaped = scaler.fit_transform(padded_data_reshaped)
    padded_data = padded_data_reshaped.reshape(-1, max_length, num_features)

    labels = np.array(labels).astype(float)

    return padded_data, labels

# 3. Transformer 模型定义
class TransformerModel(nn.Module):
    def __init__(self, num_features, num_heads, num_layers, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=num_features, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x[:, 0, :])  # 使用第一个时间步的输出
        return self.sigmoid(x)

# 4. 训练过程
def train_model(model, train_loader, val_loader, epochs, device):
    criterion = nn.BCELoss()  # 使用二元交叉熵作为损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    total_training_loss = 0.0
    total_validation_loss = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        processed_data = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            processed_data += inputs.size(0)  # 累加处理的数据数量

        avg_training_loss = running_loss / len(train_loader)
        total_training_loss += avg_training_loss
        print(f'Epoch {epoch + 1}/{epochs} - Processed {processed_data} samples, '
              f'Training Loss: {avg_training_loss:.4f}')

        # 验证模型
        model.eval()
        val_loss = 0.0
        processed_val_data = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                processed_val_data += inputs.size(0)  # 累加验证集处理的数据数量

        avg_val_loss = val_loss / len(val_loader)
        total_validation_loss += avg_val_loss
        print(f'Epoch {epoch + 1}/{epochs} - Processed {processed_val_data} validation samples, '
              f'Validation Loss: {avg_val_loss:.4f}')

    # 输出十组 epoch 处理后的总总结
    print(f'\nTraining completed over {epochs} epochs.')
    print(f'Average Training Loss: {total_training_loss / epochs:.4f}')
    print(f'Average Validation Loss: {total_validation_loss / epochs:.4f}')

# 5. 主程序
def main():
    file_path = r"C:\Users\何家乐\AppData\Roaming\JetBrains\PyCharmCE2024.2\scratches\data2.csv"

    # 处理数据
    data, labels = process_data(file_path)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # 创建 Dataset 和 DataLoader
    train_dataset = UAVDataset(X_train, y_train)
    test_dataset = UAVDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 定义模型超参数
    num_features = X_train.shape[2]  # 确定特征维度
    num_heads = 2  # 确保 num_heads 可以整除 num_features
    num_layers = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    model = TransformerModel(num_features, num_heads, num_layers)

    # 训练模型
    train_model(model, train_loader, val_loader, epochs=10, device=device)

# 6. 执行主程序
if __name__ == "__main__":
    main()
