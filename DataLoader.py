import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
import os
import csv
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn


"""
/brief: padding the data to a fixed length.

/author: CHEN Yi-xuan

/date: 2024-09-24

/param: data: the data to be padded
        fixed_length: the length to be padded to, should be greater than the length of the data
"""
def padding_data_to_fixed_length(data, fixed_length: int):
    # check if the fixed length is greater than the length of the data
    if fixed_length < len(data):
        raise ValueError("The fixed length should be greater than the length of the data.")
    # get the dimension of each data point
    dim = len(data[0])
    # padding the data to the fixed length
    padding = [[0.0 for _ in range(dim)] for _ in range(fixed_length - len(data))]
    data.extend(padding)

    return data


"""
/brief: extract event_2 data from a CSV file and process it into a numpy array.

/author: CHEN Yi-xuan

/date: 2024-09-24

/param: file_path: str, the path of the UTF-8 CSV file

/input: the raw format is as follows:
    azimuth angle, slant range, relative height, radial velocity, record time, RCS, label  # start of track k
     azimuth angle, slant range, relative height, radial velocity, record time, RCS,
        ...
     azimuth angle, slant range, relative height, radial velocity, record time, RCS,       # end of track k
    azimuth angle, slant range, relative height, radial velocity, record time, RCS, label  # start of track k+1
        ...
    only the first row has the label, the rest of the rows have an empty label.
    Because radar points in a track are continuous and have the same label.
    
/output: the processed data is as follows:
    [N, [[L, 6], 1, 1]] i.e. [N, [track, label, L]]
    N is the number of tracks;
    L is the length of a track, i.e. number of radar points in a track;
    (optional: padding L to a fixed length, e.g. 15; and remove the length of the track;)
    6 is the number of features of a radar point;
    first 1 is the label of the track, it is either 0 or 1, 0 for non-drone, 1 for drone.
    second 1 is the length of the track, it is the same as L.
"""
def extract_event_2_data_from_csv(raw_data_path: str, fixed_length: int = -1) -> np.ndarray:
    if -1 < fixed_length < 11:
        raise ValueError("The fixed length should be -1(no padding) or greater than 10(11 is the max_len_of_track).")
    data_track_list = [] # N * ([L, 6] + [1] + [1]), record the data of N tracks
    track = [] # [N, 6], record the data of a track
    last_label = -1 # keep the label of the last track
    cnt = 0 # count the number of radar points in last track
    num_of_tracks = 0 # count the number of tracks

    with open(raw_data_path, 'r', encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header

        for line in reader:
            if line[6].strip():  # Check if the 7th column (label) is not empty
                if last_label != -1:
                    # Add previous track to the data_track_list
                    if fixed_length == -1:
                        data_track_list.append([track, last_label, cnt])
                    else:
                        # Padding the track to fixed length
                        track = padding_data_to_fixed_length(track, fixed_length)
                        data_track_list.append([track, last_label])
                    track = []
                cnt = 1
                last_label = int(line[6])
                track.append([float(x) for x in line[:6]]) # add the first radar point to the track
                num_of_tracks += 1
            else:
                track.append([float(x) for x in line[:6]]) # Take only the first 6 columns as radar point of the track
                cnt += 1 # count the number of radar points in a track

        # Add the last track to the data_track_list
        if fixed_length == -1:
            data_track_list.append([track, last_label, cnt])
        else:
            if cnt < fixed_length:
                track = padding_data_to_fixed_length(track, fixed_length)
            data_track_list.append([track, last_label])

    # calculate the statistics of the length of tracks
    print(f"track total nums: {num_of_tracks}")
    if fixed_length == -1:
        track_length = [x[2] for x in data_track_list]
        print(f"track min points nums: {min(track_length)}")
        print(f"track max points nums: {max(track_length)}")
        print(f"track mean points nums: {np.mean(track_length)}")

    # process data to npy format and print its shape info
    data_track_list = np.array(data_track_list, dtype=object)
    print(f"Processed data shape: {len(data_track_list)}")

    # save the processed data to a npy file under the same directory as the raw csv data
    np.save(raw_data_path.replace('.csv', '.npy'), data_track_list)

    return data_track_list


"""
/brief: custom scaler class to conduct normalization on the non-zero values in the data
        but keep the zero values as 0

/author: CHEN Yi-xuan

/date: 2024-09-24

/param: feature_range: the range of the normalized data
"""
class KeepZeroMinMaxScaler:
    def __init__(self, feature_range=(0.1, 1.1)):
        self.feature_range = feature_range

    def fit_transform(self, X):
        original_shape = X.shape
        X_2d = X.reshape(-1, X.shape[-1])

        non_zero_mask = (X_2d != 0)
        X_scaled = torch.zeros_like(X_2d, dtype=torch.float32)

        for col in range(X_2d.shape[1]):
            col_data = X_2d[:, col]
            if torch.any(non_zero_mask[:, col]):
                col_non_zero = col_data[non_zero_mask[:, col]]
                min_val = col_non_zero.min()
                max_val = col_non_zero.max()
                if min_val != max_val:
                    scaled = (col_non_zero - min_val) / (max_val - min_val)
                    scaled = scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
                    X_scaled[non_zero_mask[:, col], col] = scaled

        return X_scaled.reshape(original_shape)


"""
/brief: custom dataset class for radar drone tracking

/author: CHEN Yi-xuan

/date: 2024-09-24
"""
class DroneRadarDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        # normalize the features
        self.Scaler = KeepZeroMinMaxScaler()
        self.features = self.Scaler.fit_transform(self.features)
        # convert to tensor and use float32 type to align with the model weight and bias type
        self.features = torch.tensor(self.features, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

"""
/brief: load the data and split it into train, validation, and test datasets
        then create data loaders for future training
        the data contains nan values, so we need to handle it before training in this function

/author: CHEN Yi-xuan

/date: 2024-09-24

/param: data_path: the path of the npy data file
"""
def load_data(data_path):
    # Load the data from object array
    data = np.load(data_path, allow_pickle=True)

    # Separate features and labels
    X = data[:, 0]  # Shape: (58613, 15, 6)
    y = data[:, 1]  # Shape: (58613,)

    # deal with the nan values in the data, convert nan to 0
    for i in range(len(X)):
        X[i] = np.nan_to_num(X[i])

    # Convert to PyTorch tensors
    X = torch.FloatTensor(X.tolist())
    y = torch.LongTensor(y.tolist())

    # Split the data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # Create custom datasets
    train_dataset = DroneRadarDataset(X_train, y_train)
    val_dataset = DroneRadarDataset(X_val, y_val)
    test_dataset = DroneRadarDataset(X_test, y_test)

    # Create data loaders
    Batch_Size = 32
    train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False)

    return train_loader, val_loader, test_loader
if __name__ == "__main__":
    csv_file_path = r'D:\project\pythonProject1\data2.csv'  # 指定要处理的 CSV 文件路径
    fixed_length = 15  # 设置数据填充的固定长度
    result = extract_event_2_data_from_csv(csv_file_path, fixed_length)
    print(f"Data has been processed and saved to {result}")


