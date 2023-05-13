# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from matplotlib import pyplot as plt
import time

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
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.fc1 = nn.Linear(1000, 700)
        self.fc2 = nn.Linear(700, 350)
        self.fc3 = nn.Linear(350, 100)
        self.fc4 = nn.Linear(100, 1)


    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """

        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.squeeze(x)


class AE(nn.Module):
    def __init__(self):
        super().__init__()
         
        self.encoder = nn.Sequential(
            nn.Linear(1000, 650),
            nn.ReLU(),
            nn.Linear(650, 350),
            nn.ReLU(),
            nn.Linear(350, 150),
            nn.ReLU(),
            nn.Linear(150, 60),)
         
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(60, 150),
            torch.nn.ReLU(),
            torch.nn.Linear(150, 350),
            torch.nn.ReLU(),
            torch.nn.Linear(350, 650),
            torch.nn.ReLU(),
            torch.nn.Linear(650, 1000),)
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

def train_nn(model, X, Y, epochs, batch_size, lr, vali=False):
    
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
        print(f'--------EPOCH {epoch}--------')
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

        with torch.no_grad():
            model.eval()

            Y_pred = model.forward(X_tr)
            train_loss = criterion(Y_pred, Y_tr)
            train_losses.append(train_loss)
            if vali:
                Y_pred = model.forward(X_val)
                vali_loss = criterion(Y_pred, Y_val)
                vali_losses.append(vali_loss)

            toc = time.perf_counter()
            print(f"train RMSE       {np.sqrt(train_loss):.4f}")
            if vali: print(f"vali RMSE        {np.sqrt(vali_loss):.4f}")
            print(f"elapsed minutes: {(toc-tic)/60:.1f}")

        model.train()

    plt.clf()
    axes = plt.plot([i for i in range(epochs)], [np.sqrt(item.item()) for item in train_losses], 'b-', label="RMSE train")
    if vali: axes = plt.plot([i for i in range(epochs)], [np.sqrt(item.item()) for item in vali_losses], 'g-', label="RMSE vali")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.grid()
    plt.ylim(0, 0.2)
    plt.savefig(f"./task4/{type(model).__name__}_bs{batch_size}_lr{lr}_e{epochs}_V{int(vali)}.png")

    model.eval()
    return model


def make_nn_feature_func(model, AE):

    if AE:
        model = model.encoder
    else:
        model = nn.Sequential(*list(model.children())[:-1])

    def trafo(x):
        x = torch.tensor(x, dtype=torch.float)
        with torch.no_grad():
            model.eval()
            return model.forward(x).detach().numpy()

    return trafo


def make_pretraining_class(feature_extractors):
    """
    The wrapper function which makes pretraining API compatible with sklearn pipeline
    
    input: feature_extractors: dict, a dictionary of feature extractors

    output: PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode

        def fit(self, X=None, y=None):
            return self

        def transform(self, X):
            assert self.feature_extractor is not None
            X_new = feature_extractors[self.feature_extractor](X)
            return X_new
        
    return PretrainedFeatures

if __name__ == '__main__':

    torch.manual_seed(1)
    alphas = np.linspace(0,100,2000)

    # Load data
    X_pretrain, Y_pretrain, X_train, Y_train, X_test = load_data()

    lumo_net = train_nn(LumoNet(), X_pretrain, Y_pretrain, 2, 32, 5e-5, vali=True)
    ae = train_nn(AE(), X_pretrain, X_pretrain, 2, 32, 1e-4, vali=True)

    lumo_func = make_nn_feature_func(lumo_net, AE=False)
    encoder_func = make_nn_feature_func(ae, AE=True)

    PretrainedFeatureClass = make_pretraining_class({"lumo": lumo_func, "ae": encoder_func})

    lumo_emb = PretrainedFeatureClass(feature_extractor="lumo")
    encoder = PretrainedFeatureClass(feature_extractor="ae")

    X_tr_lumo = lumo_emb.transform(X_train)
    X_te_lumo = lumo_emb.transform(X_test.to_numpy())
    
    scaler1 = StandardScaler()
    X_te_lumo_s = scaler1.fit_transform(X_te_lumo)
    X_tr_lumo_s = scaler1.transform(X_tr_lumo)

    X_tr_ae = encoder.transform(X_train)
    X_te_ae = encoder.transform(X_test.to_numpy())

    X_tr_comb = np.hstack((X_tr_lumo, X_tr_ae))
    X_te_comb = np.hstack((X_te_lumo, X_te_ae))

    scaler2 = StandardScaler()
    X_te_comb_s = scaler2.fit_transform(X_te_comb)
    X_tr_comb_s = scaler2.transform(X_tr_comb)

    RMSEs = []

    print("\ntraining ridge regression wo AE\n")
    for l in alphas:
        ridge_model = Ridge(alpha=l)
        cv_scores = cross_val_score(ridge_model, X_tr_lumo_s, Y_train, cv=20, scoring="neg_root_mean_squared_error")
        RMSE = -np.mean(cv_scores)
        RMSEs.append(RMSE)

    j = RMSEs.index(min(RMSEs))
    print(f"\nbest RMSE and alpha  {RMSEs[j]}, {alphas[j]}")

    RMSEs = []

    print("\ntraining ridge regression w AE\n")
    for l in alphas:
        ridge_model = Ridge(alpha=l)
        cv_scores = cross_val_score(ridge_model, X_tr_comb_s, Y_train, cv=20, scoring="neg_root_mean_squared_error")
        RMSE = -np.mean(cv_scores)
        RMSEs.append(RMSE)

    j = RMSEs.index(min(RMSEs))
    print(f"\nbest RMSE and alpha  {RMSEs[j]}, {alphas[j]}")
    
    model = Ridge(alpha=alphas[j])

    model = model.fit(X_tr_comb_s, Y_train)

    Y_pred = model.predict(X_te_comb_s)

    assert Y_pred.shape == (X_test.shape[0],)
    Y_pred = pd.DataFrame({"y": Y_pred}, index=X_test.index)
    Y_pred.to_csv("./task4/results.csv", index_label="Id")
    print("Predictions saved, all done!")
    