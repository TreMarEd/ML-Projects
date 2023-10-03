# The following script solves project 3 of the 2023 Introduction to Machine Learning course at ETH. 10'000 pictures of dishes are
# provided. Given a triplet pictures of dishes (A,B,C) the goal is to predict whether the taste of A is more similar to B than C.
# A labeled training set of triplets is provided in the task.

import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet, AlexNet_Weights
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# alexnet embedding size
embedding_size = 9216
learning_rate = 1e-4
batches_size = 32
# number of neurons of layer i
num_neurons1 = 500
num_neurons2 = 250
# boolean stating whether to perform training/vali split or to produce productive results on entire dataset
vali = False


def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    train_transforms = transforms.Compose([transforms.ToTensor(), AlexNet_Weights.IMAGENET1K_V1.transforms()])
    train_dataset = datasets.ImageFolder(root="./task 3/dataset/", transform=train_transforms)

    batch_size = 50
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    alexnet_model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).features
    alexnet_model.eval()

    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))

    l = 0
    with torch.no_grad():
        for features, labels in train_loader:
            output = alexnet_model(features)

            for j in range(batch_size):
                embeddings[l, :] = torch.flatten(output[j])
                l += 1

    np.save('./task 3/dataset/embeddings.npy', embeddings)


def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(
        root="./task 3/dataset/", transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '')
                 for s in train_dataset.samples]

    embeddings = np.load('./task 3/dataset/embeddings.npy')

    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]

    X = []
    y = []

    # use the individual embeddings to generate the features and labels for triplets
    if train:
        alternator = 0
        for t in triplets:
            emb = [file_to_embedding[a] for a in t.split()]
            if alternator % 2 == 0:
                X.append(np.hstack([emb[0], emb[1], emb[2]]))
                y.append(1)
            else:
                X.append(np.hstack([emb[0], emb[2], emb[1]]))
                y.append(0)
            alternator += 1
    else:
        for t in triplets:
            emb = [file_to_embedding[a] for a in t.split()]
            X.append(np.hstack([emb[0], emb[1], emb[2]]))
            y.append(1)

    X = np.vstack(X)
    y = np.hstack(y)

    return X, y


def create_loader_from_np(X, y=None, train=True, batch_size=batches_size, shuffle=False, num_workers=0):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels

    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), torch.from_numpy(y).type(torch.float))

    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader


class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """

    def __init__(self, dropout_rate1, dropout_rate2):
        """
        The constructor of the model.
        """
        super().__init__()
        self.do1 = nn.Dropout(dropout_rate1)
        self.fc1 = nn.Linear(embedding_size, num_neurons1)
        self.do2 = nn.Dropout(dropout_rate2)
        self.fc2 = nn.Linear(num_neurons1, num_neurons2)


    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """

        # classify each separate dish
        A = x[:, :embedding_size]
        B = x[:, embedding_size:(2*embedding_size)]
        C = x[:, (2*embedding_size):(3*embedding_size)]

        A = self.do1(A)
        B = self.do1(B)
        C = self.do1(C)
        A = F.relu(self.fc1(A))
        B = F.relu(self.fc1(B))
        C = F.relu(self.fc1(C))

        A = self.do2(A)
        B = self.do2(B)
        C = self.do2(C)
        A = self.fc2(A)
        B = self.fc2(B)
        C = self.fc2(C)

        A = F.softmax(A, dim=1)
        B = F.softmax(B, dim=1)
        C = F.softmax(C, dim=1)

        # compute cross entropy on classifier distributions to measure dissimilarities of the classifications
        B = torch.log(B)
        C = torch.log(C)
        delta_AB = -torch.sum(torch.mul(A, B), dim=1)
        delta_AC = -torch.sum(torch.mul(A, C), dim=1)

        x = delta_AC - delta_AB
        x = F.sigmoid(x)
        return x


def train_model(train_loader, model, epochs, plt_string, vali):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data

    output: model: torch.nn.Module, the trained model
    """
    print(f"\nTRAINING {plt_string}\n")

    model.train()
    model.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    vali_losses = []
    train_losses = []
    vali_accuracies = []
    train_accuracies = []

    for epoch in range(epochs):
        print(f'--------EPOCH {epoch}--------')
        tic = time.perf_counter()

        for [X_batch, y_batch] in train_loader:
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            y_pred = model.forward(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            y_pred = model.forward(X_train)
            train_loss = criterion(y_pred, y_train)
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0
            train_accuracy = torch.sum(y_pred == torch.round(y_train))/y_train.size(0)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            if vali:
                y_pred = model.forward(X_vali)
                vali_loss = criterion(y_pred, y_vali)
                y_pred[y_pred >= 0.5] = 1
                y_pred[y_pred < 0.5] = 0
                vali_accuracy = torch.sum(y_pred == torch.round(y_vali))/y_vali.size(0)
                vali_losses.append(vali_loss)
                vali_accuracies.append(vali_accuracy)

            print(f"train loss       {100*train_loss:.2f}%")
            print(f"train accuracy   {100*train_accuracy:.2f}%")

            if vali:
                print(f"vali loss        {100*vali_loss:.2f}%")
                print(f"vali accuracy    {100*vali_accuracy:.2f}%")

            toc = time.perf_counter()
            print(f"elapsed minutes: {(toc-tic)/60:.1f}")

        model.train()

    axes = plt.plot([i for i in range(epochs)], [item.item() for item in train_losses], 'b-', label="train BCE")
    axes = plt.plot([i for i in range(epochs)], [item.item() for item in train_accuracies], 'b--', label="train accuracy")

    if vali:
        axes = plt.plot([i for i in range(epochs)], [item.item() for item in vali_losses], 'g-', label="vali BCE")
        axes = plt.plot([i for i in range(epochs)], [item.item() for item in vali_accuracies], 'g--', label="vali accuracy")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig(f"./task 3/lr{learning_rate}_N{num_neurons1/1000}_{num_neurons2/1000}_b{batches_size}_do{plt_string}_epochs{epochs}_vali{vali}.png")
    plt.clf()
    model.eval()
    return model


def test_model(model, plt_name, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data

    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for [x_batch] in loader:
            x_batch = x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.hstack(predictions)
    np.savetxt(f"./task 3/results_{plt_name}.txt", predictions, fmt='%i')


if __name__ == '__main__':
    TRAIN_TRIPLETS = './task 3/train_triplets.txt'
    TEST_TRIPLETS = './task 3/test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists('./task 3/dataset/embeddings.npy') == False):
        generate_embeddings()

    # load the training data
    print("\nloading training data")
    X_train, y_train = get_data(TRAIN_TRIPLETS)

    if vali:
        print("\npreparing train vali split")
        p = 0.8
        X_train, X_vali, y_train, y_vali = train_test_split(X_train, y_train, train_size=p, random_state=34)

    print("\npreparing train loader")
    train_loader = create_loader_from_np(X_train, y_train, train=True, batch_size=batches_size)

    print("\nstart training:\n")
    X_train = torch.from_numpy(X_train).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float)

    if vali:
        X_vali = torch.from_numpy(X_vali).type(torch.float)
        y_vali = torch.from_numpy(y_vali).type(torch.float)
    
    model_25_25 = train_model(train_loader, Net(0.25, 0.25), 7, "25_25", vali=vali)
    model_20_20 = train_model(train_loader, Net(0.20, 0.20), 7, "20_20", vali=vali)

    if not vali:
        print("\nloading test data")
        X_test, _ = get_data(TEST_TRIPLETS, train=False)
    
        print("\npreparing test loader")
        test_loader = create_loader_from_np(X_test, train=False, batch_size=1000, shuffle=False)

        print("\ngenerating results.txt")
        test_model(model_25_25, "25_25_7", test_loader)
        test_model(model_20_20, "20_20_7", test_loader)
        

