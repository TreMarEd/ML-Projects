# TODO: explain the task
# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt

#TODO: general cosmetics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """

    train_transforms = transforms.Compose([transforms.ToTensor(), ResNet50_Weights.IMAGENET1K_V2.transforms()])
    train_dataset = datasets.ImageFolder(root="./task 3/dataset/", transform=train_transforms)

    # Hint: adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
    num_workers = 0
    batch_size = 50
    batches = int(10000/batch_size)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    # define pretrained model, remove last layer and activate evaluation
    #TODO: try resnet152
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) 
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    #TODO: read embedding size automatically from the pretrained model used
    embedding_size = 2048
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))

    l = 0
    with torch.no_grad():
        for k in range(batches):
            print("\nworking on batch ", str(k), "\n")
            i = iter(train_loader)
            features, labels = next(i)
            output = model(features)

            for j in range(batch_size):
                embeddings[l, :] = np.squeeze(output[j])
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
    train_dataset = datasets.ImageFolder(root="./task 3/dataset/", transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]

    embeddings = np.load('./task 3/dataset/embeddings.npy')

    #scaler = MinMaxScaler()
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

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
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)

    X = np.vstack(X)
    y = np.hstack(y)

    return X, y

# Hint: adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
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
        #self.fc1 = nn.Linear(2*2048, 100) #2048 is current embedding dimension
        
        self.fc1 = nn.Linear(3*2048, 1000)
        self.fc2 = nn.Linear(1000,1000)
        self.fc3 = nn.Linear(1000,1)
        self.dropout = nn.Dropout(0.25)
        """
        self.fc1 = nn.Linear(2*2048,1100)
        self.fc2 = nn.Linear(1100,1100)
        self.fc3 = nn.Linear(1000,1)
        """
        #self.W = nn.Parameter(torch.randn(1100,1100))
        

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        #TODO: regularization
        """
        #currently gives killed 9 error, make matrix multiplication more efficeint
        A = x[:,0:2048]
        B = x[:,2048:(2*2048)]
        C = x[:,(2*2048):(3*2048)]

        AB = torch.cat((A, B), 1)
        AC = torch.cat((A, C), 1)

        AB = F.relu(self.fc1(AB))
        AC = F.relu(self.fc1(AC))
        AB = F.relu(self.fc2(AB))
        AC = F.relu(self.fc2(AC))
        #AB = F.relu(self.fc3(AB))
        #AC = F.relu(self.fc3(AC))

    
        M = torch.transpose(self.W,0,1) - self.W
        x = torch.matmul(AC, torch.matmul(M, torch.transpose(AB,0,1)))
        x = torch.diagonal(x)

        x = F.sigmoid(x)
        
        #x = self.dropout(x)
        return torch.squeeze(x)
        """

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.droput(x)
        x = self.fc3(x)

        return torch.squeeze(F.sigmoid(x))
        

def train_model(train_loader, X_train, y_train, X_vali, y_vali):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.train()
    model.to(device)
    
    # TODO: After choosing the best model, train it on the whole training data.

    criterion = torch.nn.BCELoss()
    #TODO: stop normalizing the loss https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    epochs = 5
    vali_losses = []
    train_losses = []

    for epoch in range(epochs):
        print(f'\nepoch: {epoch}')
            
        for [X_batch, y_batch] in train_loader:
            optimizer.zero_grad()
            y_pred = model.forward(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        
        with torch.no_grad():
            model.eval()
            y_pred = model.forward(X_train)
            loss = criterion(y_pred, y_train)
            print(f"\n train loss {loss}")
            train_losses.append(loss)

            y_pred = model.forward(X_vali)
            loss = criterion(y_pred, y_vali)
            print(f"\n vali loss {loss}")
            vali_losses.append(loss)  
             
        model.train()
    
    axes = plt.plot([i for i in range(epochs)], [item.item() for item in train_losses], 'b-')
    plt.savefig("./task 3/train.png")
    plt.clf()
    axes = plt.plot([i for i in range(epochs)], [item.item() for item in vali_losses], 'b-')
    plt.savefig("./task 3/vali.png")
    #TODO: plot cosmetics
    
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

    np.random.seed(2)
    p = 0.8
    print("\npreparing train vali split")
    mask_train = [bool(np.random.binomial(1, p)) for i in range(np.shape(X)[0])]
    mask_vali = [not i for i in mask_train]


    X_train = X[mask_train, :]
    y_train = y[mask_train]

    X_vali = X[mask_vali, :]
    y_vali = y[mask_vali]


    # Create data loaders for the training and testing data
    train_loader = create_loader_from_np(X_train, y_train, train = True, batch_size=1000)
    test_loader = create_loader_from_np(X_test, train = False, batch_size=1000, shuffle=False)

    # define a model and train it
    model = train_model(train_loader, X_train=torch.from_numpy(X_train).type(torch.float), y_train=torch.from_numpy(y_train).type(torch.float),
                        X_vali=torch.from_numpy(X_vali).type(torch.float), y_vali=torch.from_numpy(y_vali).type(torch.float))
    
    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")
