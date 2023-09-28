import pandas as pd
import numpy as np
import torch
import albumentations
import argparse
import torch.nn as nn
from torchvision import models
import time
import pickle
import cv2
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

model = models.resnet18(pretrained=True)
    
model = nn.Sequential(*list(model.children())[:-1])

model.eval()

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=5, type=int,
    help='number of epochs to train the model for')
args = vars(parser.parse_args())

df = pd.read_csv('./input/data.csv')
X = df.image_path.values
y = df.target.values

(xtrain, xtest, ytrain, ytest) = (train_test_split(X, y, 
                                test_size=0.15, random_state=42))

print(f"Training on {len(xtrain)} images")

class ASLImageDataset(Dataset):
    def __init__(self, path, labels,image_size=224):
        self.X = path
        self.y = labels
        self.image_size = image_size

        # apply augmentations
        self.aug = albumentations.Compose([
            albumentations.Resize(224, 224, always_apply=True),
        ])

    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        # image = Image.open(self.X[i])
        image = cv2.imread(self.X[i])
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        label = self.y[i]

        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long)

train_data = ASLImageDataset(xtrain, ytrain)
test_data = ASLImageDataset(xtest, ytest)
np.save('test_data.npy', test_data)

# dataloaders
trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
testloader = DataLoader(test_data, batch_size=64, shuffle=False)


train_features = []
with torch.no_grad():
    for data in tqdm(trainloader, total=int(len(train_data)/trainloader.batch_size)):
        images, _ = data
        features = model(images).detach().numpy()
        train_features.append(features)
train_features = np.vstack(train_features)
train_features = torch.from_numpy(train_features)
n_samples, n_timesteps, n_features, _ = train_features.shape
train_features_2d = train_features.reshape(n_samples, n_timesteps * n_features)

print(len(train_features_2d))
print(type(train_features_2d))
np.save('train_features.npy', train_features_2d)
features_t = np.load('train_features.npy')

test_features = []
for image, label in test_data:
    image = np.expand_dims(image, axis=0)
    features = model(torch.tensor(image, dtype=torch.float)).detach().numpy()
    features = features.reshape(features.shape[0], -1)
    test_features.append(features)
test_features = np.vstack(test_features)
print(test_features)
n_samples, n_features = test_features.shape
test_features_2d = test_features.reshape(n_samples, n_features)

np.save('test_features.npy', test_features_2d)
features_t = np.load('test_features.npy')

with open('labels.pickle','rb') as f:
    labels = pickle.load(f)
    
print(len(labels))
hmm_model = GaussianHMM(n_components=25, covariance_type='diag', n_iter=100)
print(hmm_model)
start = time.time()
hmm_model.fit(train_features_2d)

# print("Epochs Started...")
# train_loss , train_accuracy = [], []
# for epoch in range(args['epochs']):
#     print(f"Epoch {epoch+1} of {args['epochs']}")
#     hmm_model.fit(train_features_2d, labels)
#     train_epoch_loss = hmm_model.score(test_features_2d)
#     train_epoch_accuracy = None
#     train_loss.append(train_epoch_loss)
#     train_accuracy.append(train_epoch_accuracy)
            
    
end = time.time()

print(f"{(end-start)/60:.3f} minutes")


# # save the model to disk
print('Saving model...')
with open('model.pickle', 'wb') as f:
    pickle.dump(hmm_model, f)


