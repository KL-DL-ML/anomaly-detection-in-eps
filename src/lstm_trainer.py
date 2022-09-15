from cProfile import label
import os
import random
from statistics import mean
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from time import time
from tqdm.auto import trange
from collections import defaultdict
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

from generate_dataset import load_train_val_test

RANDOM_SEED = 11
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, num_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2 * embedding_dim

        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.embedding_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        x, hidden_states = self.lstm1(x)
        x, hidden_states = self.lstm2(x)

        return hidden_states[0].reshape((-1, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, output_dim=2, num_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = 2 * input_dim
        self.output_dim = output_dim

        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.lstm2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=output_dim)

    def forward(self, x):
        x = x.repeat((self.seq_len, 1))
        x = x.reshape((1, self.seq_len, self.input_dim))

        x, hidden_states = self.lstm1(x)
        x, hidden_states = self.lstm2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))

        return self.linear(x)


class LAE(nn.Module):
    """LSTM AutoEncoder for anomaly detection"""

    def __init__(self, seq_len, n_features, embedding_dim=64, num_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(self.seq_len, self.n_features, self.embedding_dim, num_layers=num_layers)
        self.decoder = Decoder(self.seq_len, self.embedding_dim, self.n_features, num_layers=num_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class Trainer(nn.Module):
    """Training class for LSTM autoencoder and make predictions"""

    def __init__(self, log_dir, model_name):
        super().__init__()
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.L1Loss(reduction="mean").to(self.device)
        self.history = defaultdict(list)

    def fit(self, model, train_dataset, val_dataset, num_epochs, learning_rate=1e-03):
        """Train model

        Parameters:
            model (nn.Module): LSTM autoencoder
            train_dataset (Tensor): train dataset
            val_dataset (Tensor): validation dataset
            num_epochs (int):      number of epochs
            learning_rate (float): learning rate

        Returns:
            losses:  array of loss function for each epoch
        """
        device = self.device
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model.to(device)
        criterion = self.criterion

        print(f"Training LSTM AutoEncoder for {num_epochs} epochs...\n")
        for epoch in range(num_epochs):
            model.train()
            train_losses = []

            for i, seq_input in enumerate(train_dataset):
                optimizer.zero_grad()
                seq_input = seq_input.to(device)
                seq_pred = model(seq_input)
                loss = criterion(seq_pred, seq_input)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_losses = []
            model.eval()
            with torch.no_grad():
                for seq_input in val_dataset:
                    seq_input = seq_input.to(device)
                    seq_pred = model(seq_input)
                    loss = criterion(seq_pred, seq_input)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            self.history["train loss"].append(train_loss)
            self.history["val loss"].append(val_loss)

            print(f"Epoch: {epoch + 1}  train loss: {train_loss:.5f}  Val loss: {val_loss:.5f}")

        torch.save(model.state_dict(), f"{self.log_dir}/{self.model_name}.pth")
        np.savez(
            f"{self.log_dir}/losses", train_loss=self.history["train loss"],
            val_loss=self.history["val loss"]
        )

        return self.history

    def plot_loss_history(self, history, filename):
        save_dir = "figures"
        os.makedirs(save_dir, exist_ok=True)

        ax = plt.figure().gca()
        ax.plot(history["train loss"], color="green", label="train")
        ax.plot(history["val loss"], color="orange", label="val")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{filename}.png", dpi=500)
        plt.close()

    def predict(self, model, model_path, dataset):
        criterion = self.criterion
        model.load_state_dict(torch.load(model_path, map_location=self.device), strict=True)
        model.to(self.device)

        predictions, losses = [], []
        with torch.no_grad():
            model.eval()
            for seq_input in dataset:
                seq_input = seq_input.to(self.device)
                seq_pred = model(seq_input)
                loss = criterion(seq_pred, seq_input)
                seq_pred = seq_pred.cpu().numpy().flatten()
                predictions.append(seq_pred)
                losses.append(loss.item())

        losses = np.array(losses).reshape(-1, 1)
        predictions = np.array(predictions).reshape(-1, 1)

        return predictions, losses

    def plot_prediction(self, dataset, model, model_path, scaler, show_plot=True):

        predictions, losses = self.predict(model=model, model_path=model_path, dataset=dataset)

        # Load training loss and fix threshold
        loss_data = np.load("training_logs/losses.npz", allow_pickle=True)
        reconstruction_error = np.asarray(loss_data["train_loss"])
        threshold = np.mean(reconstruction_error) + np.std(reconstruction_error)
        # np.max(reconstruction_error)

        # convert dataset to ndarray and unscale the values
        true_values = np.asarray([s.numpy() for s in dataset]).reshape((-1, 1))
        true_values = scaler.inverse_transform(true_values)
        predictions = scaler.inverse_transform(predictions)

        # compute metrics
        mae = metrics.mean_absolute_error(true_values, predictions)
        R2 = metrics.r2_score(true_values, predictions)
        correct = sum(l <= threshold for l in losses)
        anomaly = [l >= threshold for l in losses]
        anomaly = np.asarray(anomaly, dtype=np.int32).reshape(-1, 1)

        # plot reconstruction error, test error and anomaly
        sns.histplot(reconstruction_error, bins=50, kde=True, stat="probability", legend=False)
        plt.xlabel("Reconstruction Error")
        plt.savefig(f"figures/reconstruction_error.png", bbox_inches="tight", dpi=500)
        plt.close()

        sns.histplot(losses, bins=50, kde=True, stat="probability", legend=False)
        plt.xlabel("Reconstruction Error")
        plt.savefig(f"figures/test_error.png", bbox_inches="tight", dpi=500)
        plt.close()

        sns.histplot(anomaly, bins=50, kde=True, stat="probability", legend=False)
        plt.xlabel("Reconstruction Error")
        plt.savefig(f"figures/anomaly.png", bbox_inches="tight", dpi=500)
        plt.close()

        if show_plot:
            plt.show()

        with open("training_logs/result.txt", "w") as f:
            f.write(f"Threshold: {threshold:.3f}\n")
            f.write(f"Accuracy: {correct[0] / len(dataset) * 100: .2f}\n")
            f.write(f"Mean Absolute Error: {mae:.4f}\n")
            f.write(f"R2 score: {R2:.4f}")

        # data = {"ANG": list(true_values[:, 0]), "TRQ": list(true_values[:, 1]),
        #         "ANG_pred": list(predictions[:, 0]), "TRQ pred": list(predictions[:, 1]),
        #         "MAE": list(losses.flatten())}

        data = {"TRQ": list(true_values[:, 0]), "TRQ pred": list(predictions[:, 0]), "MAE": list(losses.flatten())}

        df = pd.DataFrame.from_dict(data, orient="columns")
        df["Anomaly"] = df["MAE"] >= threshold
        df.to_csv("training_logs/anomaly.csv", index=False)

        return

if __name__ == "__main__":
    # Load train, val and test datasets
    train_ds, val_ds, test_ds, target_len, num_features, scaler = load_train_val_test(path="data/20220823", transform=True)
    model = LAE(seq_len=target_len, n_features=num_features, embedding_dim=128, num_layers=1)

    # instantiate trainer and train
    start = time()
    trainer = Trainer(log_dir="training_logs", model_name="steering")
    hist = trainer.fit(model, train_ds, val_ds, num_epochs=50, learning_rate=0.0009)
    print('Training time: ' + "{:10.4f}".format(time() - start) + ' s')
    trainer.plot_loss_history(history=hist, filename="loss")

    forecast, _losses = trainer.predict(model, model_path="training_logs/steering.pth", dataset=test_ds)

    trainer.plot_prediction(dataset=test_ds, model=model, model_path="training_logs/steering.pth",
                            scaler=scaler, show_plot=True)




# COLOR CODE
# Soil Color = F4B183
# Egg Shell Color = FFF2CC
# Light Green = E2F0D9