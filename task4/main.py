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
epochs = 12
learning_rate = 1e-4
num_neurons1 = 700
num_neurons2 = 350
#num_neurons3 = 250
#num_neurons4 = 200
do1 = 0
do2 = 0
batch_size = 32


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

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.fc1 = nn.Linear(1000, num_neurons1)
        #self.do1 = nn.Dropout(do1)
        self.fc2 = nn.Linear(num_neurons1, num_neurons2)
        #self.do2 = nn.Dropout(do2)
        #self.fc3 = nn.Linear(num_neurons2, num_neurons3)
        #self.do3 = nn.Dropout(do2)
        #self.fc4 = nn.Linear(num_neurons3, num_neurons4)
        self.fc5 = nn.Linear(num_neurons2, 100)
        #self.do4 = nn.Dropout(do2)
        self.fc6 = nn.Linear(100, 1)


    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # TODO: Implement the forward pass of the model, in accordance with the architecture 
        # defined in the constructor.

        x = F.relu(self.fc1(x))  
        #x = self.do1(x)
        x = F.relu(self.fc2(x))
        #x = self.do2(x)
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return torch.squeeze(x)
    
def make_feature_extractor(x, y):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
            
    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    # Pretraining data loading
    #x = x[:100, :]
    #y = y[:100]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, train_size=0.8, random_state=42, shuffle=True)

    n = np.shape(x_tr)[0]

    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

    # model declaration
    model = Net()
    model.train()
    model.to(device)
  
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    vali_losses = []
    train_losses = []
    num_batches = int(np.ceil(n/batch_size))

    for epoch in range(epochs):
        print(f'--------EPOCH {epoch}--------')
        tic = time.perf_counter()

        for l in range(num_batches):
            start = batch_size * l
            end = batch_size * (l + 1)
            x_batch = x_tr[start:end, :]
            y_batch = y_tr[start:end]

            optimizer.zero_grad()
            y_pred = model.forward(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()


        with torch.no_grad():
            model.eval()

            y_pred = model.forward(x_tr)
            train_loss = criterion(y_pred, y_tr)
            train_losses.append(train_loss)

            y_pred = model.forward(x_val)
            vali_loss = criterion(y_pred, y_val)
            vali_losses.append(vali_loss)

            toc = time.perf_counter()
            print(f"train RMSE       {np.sqrt(train_loss):.2f}")
            print(f"vali RMSE        {np.sqrt(vali_loss):.2f}")
            print(f"elapsed minutes: {(toc-tic)/60:.1f}")

        model.train()

    axes = plt.plot([i for i in range(epochs)], [np.sqrt(item.item()) for item in train_losses], 'b-', label="RMSE train")
    axes = plt.plot([i for i in range(epochs)], [np.sqrt(item.item()) for item in vali_losses], 'g-', label="RMSE vali")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.grid()
    plt.ylim(0, 0.2)
    plt.savefig(f"./task4/lr{learning_rate}_N{num_neurons1}_{num_neurons2}_b{batch_size}_epochs{epochs}.png")

    model.eval()

    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        model.eval()
        feature_model = nn.Sequential(*list(model.children())[:-1])
        # TODO: Implement the feature extraction, a part of a pretrained model used later in the pipeline.
        x = torch.tensor(x, dtype=torch.float) 
        return feature_model.forward(x).detach().numpy()

    return make_features

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

def get_regression_model():
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.
    # TODO: wie mache ich mein eigenes sklearn model?
    # https://towardsdatascience.com/how-to-build-a-custom-estimator-for-scikit-learn-fddc0cb9e16e, muss nur predict selber implementiren
    model = Ridge(alpha=1.0)
    return model


# Main function. You don't have to change this
if __name__ == '__main__':
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")
    # Utilize pretraining data by creating feature extractor which extracts lumo energy 
    # features from available initial features
    feature_extractor =  make_feature_extractor(x_pretrain, y_pretrain)
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    
    # regression model
    regression_model = get_regression_model()

    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.

    # sehr verwirrend gemacht. schlussendlich calle ich PretrainedFearureClass.transform(X_train) für meine features
    # danach mache ich standard scaler (standard scaler darf ich nicht anwenden wenn ich mein lomo modell dann mitverwende)
    # wahrscheinlcih soll ich sklearn.pipeline.Pipeline verwenden
    # danach regression cross validatiojn mit diversen regularisierungs parameter
    # danach training auf vollem datensatz

    pretrain_feature_trafo = PretrainedFeatureClass(feature_extractor="pretrain")
    x_train = pretrain_feature_trafo.transform(x_train)
    x_test_ = pretrain_feature_trafo.transform(x_test.to_numpy())

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test_ = scaler.transform(x_test_)

    RMSEs = []

    alphas = [0.1, 1, 10, 100, 1000]

    for l in alphas:
        ridge_model = Ridge(alpha=l)
        cv_scores = cross_val_score(ridge_model, x_train, y_train, cv=10, scoring="neg_root_mean_squared_error")
        RMSE = -np.mean(cv_scores)
        RMSEs.append(RMSE)
    print(f"\nRMSEs {RMSEs}")
    j = RMSEs.index(min(RMSEs))

    model = Ridge(alpha=alphas[j])

    model = model.fit(x_train, y_train)

    y_pred = model.predict(x_test_)

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")