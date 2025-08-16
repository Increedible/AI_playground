import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import csv
import pickle
import copy

if torch.cuda.is_available():
    print(f"[AGENT] Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"[AGENT] CUDA version: {torch.version.cuda}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"[AGENT] ID of current CUDA device: {torch.cuda.current_device()}")

    print(f"[AGENT] Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
else:
    print("[AGENT] Cuda is not available; training will be done on CPU.")

# Dense layer
class Layer_Dense(nn.Module):
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l2=0.):
        super(Layer_Dense, self).__init__()
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.fc = nn.Linear(n_inputs, n_neurons)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        x = self.fc(x)
        return x

# ReLU activation
class Activation_ReLU(nn.Module):
    def forward(self, x):
        return F.relu(x)

# Softmax activation
class Activation_Softmax(nn.Module):
    def forward(self, x):
        return F.softmax(x, dim=1)

# Model class
class Model(nn.Module):
    def __init__(self, layers):
        super(Model, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Accuracy calculation for classification model
class Accuracy_Categorical:
    def calculate(self, predictions, y):
        _, predicted = torch.max(predictions, 1)
        return (predicted == y).float().mean().item()

# Load dataset
def read_hand_data(csv_filename):
    with open(csv_filename, mode='r') as file:
        csv_reader = csv.reader(file)
        rows = list(csv_reader)

        if not rows:
            return np.array([]), np.array([])

        num_features = len(rows[0]) - 1
        X = np.zeros((len(rows), num_features))
        y = np.zeros(len(rows), dtype=int)

        for i, row in enumerate(rows):
            y[i] = int(row[0])
            X[i] = np.array([(float(x) * 2 - 1) for x in row[1:]])

    shuffled_indices = np.random.permutation(len(X))
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    return X_shuffled, y_shuffled

# Train the model
def train(model, optimizer, criterion, X_train, y_train, X_val, y_val, epochs=10, batch_size=96, print_every=100):
    train_data = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_data = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    accuracy_metric = Accuracy_Categorical()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_accuracy = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_accuracy += accuracy_metric.calculate(outputs, batch_y)

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)

        # Validation
        model.eval()

        if epoch % print_every == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
            val_loss = 0
            val_accuracy = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item()
                    val_accuracy += accuracy_metric.calculate(outputs, batch_y)
            val_loss /= len(val_loader)
            val_accuracy /= len(val_loader)

            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Main execution
if __name__ == "__main__":
    X_train, y_train = read_hand_data("Data/face_landmarks.csv")
    X_val, y_val = read_hand_data("Data/face_landmarks2.csv")

    model = Model([
        Layer_Dense(X_train.shape[1], 64, weight_regularizer_l2=5e-4),
        Activation_ReLU(),
        nn.Dropout(0.2),
        Layer_Dense(64, 64, weight_regularizer_l2=5e-4),
        Activation_ReLU(),
        nn.Dropout(0.2),
        Layer_Dense(64, 2, weight_regularizer_l2=5e-4),
        Activation_Softmax()
    ])

    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.0006, weight_decay=5e-7)
    criterion = nn.CrossEntropyLoss()

    train(model, optimizer, criterion, X_train, y_train, X_val, y_val, epochs=10000, batch_size=64, print_every=10)

    # Save the model
    torch.save(model.state_dict(), "face.model")

    # Load the model (example)
    loaded_model = Model([
        Layer_Dense(X_train.shape[1], 64, weight_regularizer_l2=5e-4),
        Activation_ReLU(),
        nn.Dropout(0.2),
        Layer_Dense(64, 64, weight_regularizer_l2=5e-4),
        Activation_ReLU(),
        nn.Dropout(0.2),
        Layer_Dense(64, 2, weight_regularizer_l2=5e-4),
        Activation_Softmax()
    ])
    loaded_model.load_state_dict(torch.load("face.model"))
    loaded_model = loaded_model.cuda()
