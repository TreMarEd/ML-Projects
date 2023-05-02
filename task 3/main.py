# The following script solves project 3 of the 2023 Introduction to Machine Learning course at ETH. 10'000 pictures of dishes are 
# provided. Given a triplet pictures of dishes (A,B,C) the goal is to predict whether the taste of A is more similar to B than C.
# A training set of triplets is provided in the task.

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

#alexnet embedding size
embedding_size = 9216
learning_rate = 1e-4
#mini batch size
batches_size = 32
#number of neurons of layer i
num_neurons1 = 512
num_neurons2 = 256
num_neurons3 = 128
#training epochs
num_epochs = 50
num_data = 60000
#dropout rate
do = 0

#num_data = 60000

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    train_transforms = transforms.Compose([transforms.ToTensor(), AlexNet_Weights.IMAGENET1K_V1.transforms()])
    train_dataset = datasets.ImageFolder(root="./task 3/dataset/", transform=train_transforms)

    batch_size = 50
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).features
    model.eval()
    
    num_images = len(train_dataset)
    batches = int(num_images/batch_size)
    embeddings = np.zeros((num_images, embedding_size))
    l = 0
    with torch.no_grad():
        for k in range(batches):
            print("\nworking on batch ", str(k), "\n")
            i = iter(train_loader)
            features, labels = next(i)
            output = model(features)

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
        i=0
        for line in f:
            triplets.append(line)
            i+=1
            if i-1>num_data:
                break

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="./task 3/dataset/", transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]

    embeddings = np.load('./task 3/dataset/embeddings.npy')

    #scaler = MinMaxScaler()
    #scaler = StandardScaler()
    #embeddings = scaler.fit_transform(embeddings)

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]

    X = []
    y = []

    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)

        # Generating negative samples (data augmentation)
        #if train:
            #X.append(np.hstack([emb[0], emb[2], emb[1]]))
            #y.append(0)

    X = np.vstack(X)
    y = np.hstack(y)

    return X, y

def create_loader_from_np(X, y = None, train = True, batch_size=30, shuffle=True, num_workers = 0):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.float))
        
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader


class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.fc1 = nn.Linear(embedding_size, num_neurons1)
        #self.do1 = nn.Dropout(do)
        self.fc2 = nn.Linear(num_neurons1, num_neurons2)
        #self.do2 = nn.Dropout(do)
        self.fc3 = nn.Linear(num_neurons2, num_neurons3)
        #self.do3 = nn.Dropout(do)
        

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """

        #classify each separate dish
        A = x[:,:embedding_size]
        B = x[:,embedding_size:(2*embedding_size)]
        C = x[:,(2*embedding_size):(3*embedding_size)]

        A = F.relu(self.fc1(A))
        B = F.relu(self.fc1(B))
        C = F.relu(self.fc1(C))
        #A = self.do1(A)
        #B = self.do1(B)
        #C = self.do1(C)

        A = F.relu(self.fc2(A))
        B = F.relu(self.fc2(B))
        C = F.relu(self.fc2(C))
        #A = self.do2(A)
        #B = self.do2(B)
        #C = self.do2(C)  

        A = self.fc3(A)
        B = self.fc3(B)
        C = self.fc3(C)
        #A = self.do3(A)
        #B = self.do3(B)
        #C = self.do3(C)

        A = F.softmax(A, dim=1)
        B = F.softmax(B, dim=1)
        C = F.softmax(C, dim=1)

        # compute cross entropy on classifier distributions
        B = torch.log(B)
        C = torch.log(C)
        delta_AB = -torch.sum(torch.mul(A,B),dim=1)
        delta_AC = -torch.sum(torch.mul(A,C),dim=1)
        
        # use cross entropy as dissimilarity measure
        x = delta_AC - delta_AB
        x = F.sigmoid(x)
        return x
        

def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.train()
    model.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = num_epochs
    #num_batches = int(y_train.size(dim=0)/batches_size)+1

    vali_losses = []
    train_losses = []
    vali_accuracies = []
    train_accuracies = []

    for epoch in range(epochs):
        print(f'--------EPOCH {epoch}--------')
        tic = time.perf_counter()

        #for i in range(num_batches):
        for [X_batch, y_batch] in train_loader:
            #start = i * batches_size
            #end = (i + 1) * batches_size
            optimizer.zero_grad()
            #y_pred = model.forward(X_train[start:end, :])
            #loss = criterion(y_pred, y_train[start:end])
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
            
            y_pred = model.forward(X_vali)
            vali_loss = criterion(y_pred, y_vali)
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0
            vali_accuracy = torch.sum(y_pred == torch.round(y_vali))/y_vali.size(0)
            vali_losses.append(vali_loss)
            vali_accuracies.append(vali_accuracy)

            print(f"train loss       {100*train_loss:.2f}%")
            print(f"vali loss        {100*vali_loss:.2f}%")
            print(f"train accuracy   {100*train_accuracy:.2f}%")
            print(f"vali accuracy    {100*vali_accuracy:.2f}%")  

            toc = time.perf_counter()
            print(f"elapsed minutes: {(toc-tic)/60:.1f}")

        model.train()
    
    axes = plt.plot([i for i in range(epochs)], [item.item() for item in train_losses], 'b-', label="train BCE")
    axes = plt.plot([i for i in range(epochs)], [item.item() for item in vali_losses], 'g-', label="vali BCE")
    axes = plt.plot([i for i in range(epochs)], [item.item() for item in train_accuracies], 'b--', label="train accuracy")
    axes = plt.plot([i for i in range(epochs)], [item.item() for item in vali_accuracies], 'g--', label="vali accuracy")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig(f"./task 3/learning_curve_lr{learning_rate}_N{num_neurons1/1000}_{num_neurons2/1000}_{num_neurons3/1000}_b{batches_size}_partial_{int(num_data/1000)}k_do{do}.png")
    
    return model

def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data
        
    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []

    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch = x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.hstack(predictions)
    np.savetxt("./task 3/results.txt", predictions, fmt='%i')


if __name__ == '__main__':
    TRAIN_TRIPLETS = './task 3/train_triplets.txt'
    TEST_TRIPLETS = './task 3/test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists('./task 3/dataset/embeddings.npy') == False):
        generate_embeddings()

    # load the training and testing data
    print("\nloading data")
    X, y = get_data(TRAIN_TRIPLETS)
    X_test, _ = get_data(TEST_TRIPLETS, train=False)

    print("\npreparing train vali split")
    #np.random.seed(2)
    p = 0.8
    X_train, X_vali, y_train, y_vali = train_test_split(X, y, train_size=p, random_state=42)
    #mask_train = [bool(np.random.binomial(1, p)) for i in range(np.shape(X)[0])]
    #mask_vali = [not i for i in mask_train]

    #X_train = X[mask_train, :]
    #y_train = y[mask_train]

    #X_vali = X[mask_vali, :]
    #y_vali = y[mask_vali]
    
    # Create data loaders for the training and testing data
    print("\npreparing loaders")
    train_loader = create_loader_from_np(X_train, y_train, train = True, batch_size=batches_size)
    test_loader = create_loader_from_np(X_test, train = False, batch_size=1000, shuffle=False)

    print("\nstart training:\n")
    X_train=torch.from_numpy(X_train).type(torch.float)
    y_train=torch.from_numpy(y_train).type(torch.float)
    X_vali=torch.from_numpy(X_vali).type(torch.float)
    y_vali=torch.from_numpy(y_vali).type(torch.float)

    model = train_model(train_loader)
    
    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")
