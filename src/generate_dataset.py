import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew
from sklearn.utils import shuffle


def load_train_val_test(path, features="SPD", transform=True):
    """Load training, validation and test sequences

    Parameters:
        path: Source directory containing train and validation dataset
        features: Feature to be dropped
        transform: Transform dataset using MinMaxScaler

    Returns:
        train, validation and test sequences, length of sequence, number of features and scaler
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    path = [os.path.join(path, f) for f in os.listdir(path)]
    train_path = [f for f in path if "train" in f][0]
    val_path = [f for f in path if "val" in f][0]
    test_path = [f for f in path if "test" in f][0]
    
    train_df = pd.read_csv(train_path)[:5000]
    val_df = pd.read_csv(val_path)[:5000]
    test_df = pd.read_csv(test_path)[5000:]
    test_df = shuffle(test_df)

    train_df = train_df.drop(features, axis="columns").astype(np.float32)
    val_df = val_df.drop(features, axis="columns").astype(np.float32)
    test_df = test_df.drop(features, axis="columns").astype(np.float32)

    if transform:
        train_df = scaler.fit_transform(train_df.values)
        val_df = scaler.fit_transform(val_df.values)
        test_df = scaler.fit_transform(test_df.values)
    else:
        train_df = train_df.values
        val_df = val_df.values
        test_df = test_df.values

    train_seq, seq_len, n_features = create_dataset(train_df)
    val_seq, _, _ = create_dataset(val_df)
    test_seq, _, _ = create_dataset(test_df)

    return train_seq, val_seq, test_seq, seq_len, n_features, scaler

def load_labels():
    labels = pd.read_csv("training_logs/labels.csv")
    return labels

def create_dataset(sequences):
    """Create dataset to be passed to LSTM autoencoder"""
    dataset = [torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape

    return dataset, seq_len, n_features


def prelim_plot(path):
    df = pd.read_csv(path)

    angle = df["ANG"]
    torque = df["TRQ"]

    TRQ_mean = np.mean(torque)
    TRQ_std = np.std(torque)
    criterion = TRQ_mean + 2 * TRQ_std

    ANG_mean = np.mean(angle)
    ANG_std = np.std(angle)
    skew_t = skew(torque)
    skew_a = skew(angle)
    criterion1 = ANG_mean + 2 * ANG_std
    criterion2 = ANG_mean - 2 * ANG_std

    plt.style.use("ggplot")
    plt.hist(torque.ravel(), bins=50, color="green")
    plt.vlines(criterion, 0, 30000, color="red")
    plt.text(2400, 50000, f'$\mu={TRQ_mean:.2f}, \sigma={TRQ_std:.2f}, skew={skew_t:.2f}$')
    plt.text(2700, 25000, f'$\mu + 2*\sigma = {criterion:.2f}$')
    plt.xlabel("TRQ")
    plt.savefig("../figures/torque_pdf.png", bbox_inches="tight", dpi=600)
    plt.close()

    plt.hist(angle.ravel(), bins=50, color="orange")
    plt.vlines(criterion1, 0, 6000, color="green")
    plt.vlines(criterion2, 0, 6000, color="green")
    plt.text(-70, 7000, f"$\mu={ANG_mean:.2f}, \sigma={ANG_std:.2f}, skew={skew_a:.2f}$")
    plt.text(130, 3000, f"$\mu + 2*\sigma = {criterion1:.2f}$")
    plt.text(-140, 3000, f"$\mu - 2*\sigma = {criterion2:.2f}$")
    plt.xlabel("ANG")
    plt.savefig("../figures/angle_pdf.png", bbox_inches="tight", dpi=600)
    plt.close()


if __name__ == "__main__":
    train_sequence, val_sequence, test_sequence, target_len, num_features, scaler = \
        load_train_val_test(path="../data/20220823", features="SPD")

    print(f"Number of features: {num_features}")
    print(f"test seq input size: {test_sequence[0].shape}")
    print(f"Val seq input size: {val_sequence[0].shape}")

    prelim_plot(path="../data/20220823/20220823.csv")
