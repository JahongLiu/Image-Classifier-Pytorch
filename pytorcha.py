# import dependencies
import time
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get data
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)
# 1,28,28 - classes 0-9

# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28 - 6) * (28 - 6), 10),
        )

    def forward(self, x):
        return self.model(x)

# Instance of the neural network, loss, optimizer
clf = ImageClassifier().to('cpu')  # Change 'cuda' to 'cpu'
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

import time

# ... (rest of the code)

# Training flow
if __name__ == "__main__":
    for epoch in range(10):  # train for 10 epochs
        start_time = time.time()
        
        for i, batch in enumerate(dataset):
            X, y = batch
            X, y = X.to('cpu'), y.to('cpu')  # Change 'cuda' to 'cpu'
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            # Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Print progress for every 100 batches
            if i % 100 == 0:
                print(f"Epoch:{epoch}, Batch:{i}, Loss:{loss.item()}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Epoch:{epoch} loss is {loss.item()}, time elapsed: {elapsed_time:.2f} seconds")

    with open('model_state.pt', 'wb') as f:
        save(clf.state_dict(), f)

    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    img = Image.open('President_Xi.jpeg')
    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')  # Change 'cuda' to 'cpu'

    print(torch.argmax(clf(img_tensor)))
