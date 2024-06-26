"""
The following script solves project 4 of the Introduction to Machine Learning 2023 course at ETH. The task is to predict the 
LUMO-HOMO energy gap of a molecule based on features extracted from its SMILE string using the rdkit package. This energy gap 
is relevant for the manufacturing of photovoltaic materials. Only 100 datapoints with HOMO-LUMO-gap labels are available. 
However, also a large dataset with LUMO energy labels is provided. The goal is to apply transfer learning from the LUMO problem 
to the HOMO-LUMO gap problem by training embeddings and an autoencoder.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from matplotlib import pyplot as plt
import time
torch.manual_seed(482)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None
    
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("./task4/pretrain_features.csv", index_col="Id").drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("./task4/pretrain_labels.csv", index_col="Id").to_numpy().squeeze(-1)
    x_train = pd.read_csv("./task4/train_features.csv", index_col="Id").drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("./task4/train_labels.csv", index_col="Id").to_numpy().squeeze(-1)
    x_test = pd.read_csv("./task4/test_features.csv", index_col="Id").drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test


class LumoNet(nn.Module):
    """
    Model class used to predict LUMO-energies. Later in the pipeline the embeddings learned from this problem will be used to solve
    the LUMO-HOMO gap prediction problem.
    """
    def __init__(self, dim):
        """
        input:  dim: integer, dimension of the embeddings to be used for the LUMO-HOMO gap prediction.
        """
        super().__init__()
        self.fc1 = nn.Linear(1000, 700)
        self.fc2 = nn.Linear(700, 350)
        self.fc3 = nn.Linear(350, dim)
        self.fc4 = nn.Linear(dim, 1)


    def forward(self, x):
        """
        The forward pass of the model.

        input:  x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.squeeze(x)


class AE(nn.Module):
    """
    Model class used to train an autoencoder on the pretraining data.
    """
    def __init__(self, dim):
        """
        input:  dim: integer, dimension to which the dimensionality of the features is to be reduced
        """
        super().__init__()
         
        self.encoder = nn.Sequential(
            nn.Linear(1000, 650),
            nn.ReLU(),
            nn.Linear(650, 350),
            nn.ReLU(),
            nn.Linear(350, 150),
            nn.ReLU(),
            nn.Linear(150, dim),)
         
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(dim, 150),
            torch.nn.ReLU(),
            torch.nn.Linear(150, 350),
            torch.nn.ReLU(),
            torch.nn.Linear(350, 650),
            torch.nn.ReLU(),
            torch.nn.Linear(650, 1000),)
 
    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

def train_nn(model, X, Y, epochs, batch_size, lr, vali=False, plot=True):
    """
    Trains a torch neural network and returns the trained model. Also prints and plots the learning curve if specified.

    input:  model: neural network instance of a torch.nn.Module child class
            X: np.array, input features
            Y: np.array, input labels
            epochs: int, number of epochs with which the network should be trained
            batch_size: int, batch size to be used during optimization
            lr: float, learning rate to be used by the optimizer
            vali: bool, states whether to perform a training/validation split
            plot: bool, states whether to plot the learning curve
    
    output: model, trained neural network instance of a torch.nn.Module child class 
    """
    
    print(f"\nStart training {type(model).__name__} bs{batch_size} lr{lr} e{epochs} V{int(vali)}")

    if vali:
        X_tr, X_val, Y_tr, Y_val = train_test_split(X, Y, train_size=0.8, random_state=42, shuffle=True)
    else:
        X_tr = X
        Y_tr = Y

    X_tr, Y_tr = torch.tensor(X_tr, dtype=torch.float), torch.tensor(Y_tr, dtype=torch.float)

    if vali:
        X_val, Y_val = torch.tensor(X_val, dtype=torch.float), torch.tensor(Y_val, dtype=torch.float)

    model.train()
    model.to(device)
  
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

    vali_losses = []
    train_losses = []
    
    n = np.shape(X_tr)[0]
    num_batches = int(np.ceil(n/batch_size))
    
    for epoch in range(epochs):
        print(f'--------EPOCH {epoch+1}--------')
        tic = time.perf_counter()

        for l in range(num_batches):
            start = batch_size * l
            end = batch_size * (l + 1)
            X_batch = X_tr[start:end, :]
            Y_batch = Y_tr[start:end]

            optimizer.zero_grad()
            Y_pred = model.forward(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()

        # after each epoch calculate the training loss and if specified the validation loss
        with torch.no_grad():
            model.eval()

            Y_pred = model.forward(X_tr)
            train_loss = criterion(Y_pred, Y_tr)
            train_losses.append(train_loss)
            print(f"train RMSE       {np.sqrt(train_loss):.4f}")

            if vali:
                Y_pred = model.forward(X_val)
                vali_loss = criterion(Y_pred, Y_val)
                vali_losses.append(vali_loss)
                print(f"vali RMSE        {np.sqrt(vali_loss):.4f}")

            toc = time.perf_counter()
            print(f"elapsed minutes: {(toc-tic)/60:.1f}")

        model.train()

    # plot the learning curve
    if plot:
        plt.clf()

        axes = plt.plot([i+1 for i in range(epochs)], [np.sqrt(item.item()) for item in train_losses], 'b-', label="RMSE train")
        if vali: 
            axes = plt.plot([i+1 for i in range(epochs)], [np.sqrt(item.item()) for item in vali_losses], 'g-', label="RMSE vali")

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(loc="upper right")
        plt.grid()
        plt.ylim(0, 0.2)
        plt.savefig(f"./task4/{type(model).__name__}_bs{batch_size}_lr{lr}_e{epochs}_V{int(vali)}.png")

    model.eval()
    return model


def make_nn_feature_trafo(model, layer=None):
    """
    Derives a feature transformation based on an input neural network. If the model is an autoencoder then the encoder will be 
    extracted. Otherwise the embeddings learned by the layer with index "layer" are extracted.

    input:  model: neural network instance of a torch.nn.Module child class
            layer: int, indicates the index of the layer from which the features should be extracted if model is not an autoencoder
        
    output: trafo, a function mapping numpy arrays to the feature transformed numpy array
    """

    if type(model).__name__ == "AE":
        model = model.encoder
    else:
        assert layer is not None
        model = nn.Sequential(*list(model.children())[:layer])

    def trafo(x):
        x = torch.tensor(x, dtype=torch.float)
        with torch.no_grad():
            model.eval()
            return model.forward(x).detach().numpy()

    return trafo


if __name__ == '__main__':
    # Load data
    X_pretr, Y_pretr, X_tr, Y_tr, X_te = load_data()

    # dimension of the embeddings created by the lumo predictor
    dim_lumo = 100
    epochs_lumo = 2
    lr_lumo = 5e-5
    batch_size = 32
    
    # dimension to which the autoencoder reduces the original features
    dim_ae = 100
    epochs_ae = 10
    lr_ae = 3e-4

    # train lumo predictor and autoencoder on lumo pretraining data
    lumo_net = train_nn(LumoNet(dim=dim_lumo), X_pretr, Y_pretr, epochs_lumo, batch_size, lr_lumo, vali=False)
    ae = train_nn(AE(dim=dim_ae), np.vstack((X_pretr, X_te)), np.vstack((X_pretr, X_te)), epochs_ae, batch_size, lr_ae, vali=False)

    # extract feature transformations from the nns
    lumo_emb = make_nn_feature_trafo(lumo_net, layer=-1)
    encoder = make_nn_feature_trafo(ae)

    # create lumo features for both training and test data
    X_tr_lumo = lumo_emb(X_tr)
    X_te_lumo = lumo_emb(X_te.to_numpy())
    scaler1 = StandardScaler()
    X_te_lumo = scaler1.fit_transform(X_te_lumo)
    X_tr_lumo = scaler1.transform(X_tr_lumo)

    # create autoencoder features for both training and test data
    X_tr_ae = encoder(X_tr)
    X_te_ae = encoder(X_te.to_numpy())
    scaler2 = StandardScaler()
    X_te_ae = scaler2.fit_transform(X_te_ae)
    X_tr_ae = scaler2.transform(X_tr_ae)

    # create combined lumo and autoencoder features for both training and test data
    X_tr_comb = np.hstack((X_tr_lumo, X_tr_ae))
    X_te_comb = np.hstack((X_te_lumo, X_te_ae))

    # cross validate and fit kernelized gaussian process on training data
    kernel = RationalQuadratic(length_scale=1.0, alpha=1.5, length_scale_bounds=(1e-08, 100000.0))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, n_restarts_optimizer=3, random_state=5)
    score = -np.mean(cross_val_score(gpr, X_tr_comb, Y_tr, cv=20, scoring="neg_root_mean_squared_error"))
    print(f"\n CV-RMSE: {score}\n")

    gpr.fit(X_tr_comb, Y_tr)
    Y_pred = gpr.predict(X_te_comb)

    # create results
    assert Y_pred.shape == (X_te.shape[0],)
    Y_pred = pd.DataFrame({"y": Y_pred}, index=X_te.index)
    Y_pred.to_csv(f"./task4/results.csv", index_label="Id")
    print("Predictions saved, all done!")
    
    