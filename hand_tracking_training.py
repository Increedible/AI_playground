import csv
import numpy as np
min_training_loss = 1
required_loss = 0.0504

surpass= False

def read_hand_data(csv_filename):
    with open(csv_filename, mode='r') as file:
        csv_reader = csv.reader(file)
        # Collect all rows from the CSV
        rows = list(csv_reader)

        # If no data, return empty arrays
        if not rows:
            return np.array([]), np.array([])

        # Calculate number of features per sample (subtract 1 for the label)
        num_features = len(rows[0]) - 1

        # Initialize X and y
        X = np.zeros((len(rows), num_features))
        y = np.zeros(len(rows), dtype=int)

        # Fill X and y with data from the CSV
        for i, row in enumerate(rows):
            # First item in row is the label
            y[i] = int(row[0])
            # The rest are features
            X[i] = np.array([(float(x) * 2 - 1) for x in row[1:]])

    # Shuffling X without affecting the data within each sub-array in X
    shuffled_indices = np.random.permutation(len(X))
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    return X_shuffled, y_shuffled

# Create dataset
#X, y = spiral_data(samples=100, classes=3)
X, y = read_hand_data("Data/face_landmarks.csv")
X1, y1 = read_hand_data("Data/face_landmarks2.csv")

# Good runs:
# 2 layer 64, dropout 0.7, 5e-4 weight and bias reguliser. learningrate=0.0006, decay=5e-7, random layer weight init = 0.005. Batch size 512/128?
# 2 layer 16, dropout 0.7, 5e-5 weight and bias reguliser. learningrate=0.0006, decay=5e-7, random layer weight init = 0.005. Batch size 96
# Epoch 26100 0.0503 loss!

from main import *

model = Model()
model.add(Layer_Dense(X.shape[1], 16, weight_regularizer_l2=5e-5, bias_regularizer_l2=5e-5))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.7))
model.add(Layer_Dense(16, 2, weight_regularizer_l2=5e-5, bias_regularizer_l2=5e-5))
model.add(Activation_Softmax())

model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.0006, decay=5e-7),
    accuracy=Accuracy_Categorical()
)

model.finalize()

model.train(X, y, validation_data=(X1, y1), batch_size=64, epochs=1000, print_every=100)

model.save("face.model")

# DATA PREPARING:
# Lossless compresion, normalise dimensions, balance each catergory, greyscale.
# Do not train in order (all 0 first, then all catergory 1, etc)

# HYPER PARAMETERS (pre-training nobs):
# OPTIMISERS
# SGD: Initial Learning Rate, Decay, Momentum :: After 10001 stable.
#      Good Learning Rate: 1.0, Decay: 0.1
# Adagrad: Epsilon (make non-zero division) :: Consistant Loss reduction, takes lot longer
# RMSprop: Epsilon, Rho (cache memory decay rate) :: See Adagrad
# Adam: Initial Learning Rate, Decay, Beta_1, Beta_2 :: Best ATM, see SGD,
#       Combination SGD & RMSprop. Good Idea try Adam first, but not always
#       the best. Simple SGD or SGD + momentum can perform better than Adam.
#       Good (starting) Learning rate: 1e-3, Decay: 1e-4.
#
# PREVENT OVERFITTING
# Amount of Epochs; Learning rate too high; Amount of layers; Amount of Neurons in a layer.
# And dropout!
#
# CHECK AFTERWARDS WITH TESTING DATA:
# can run 'mini evolution' with these parameters, pick best one(s), mutate, etc.
# IMPORTANT: RUN 'mini evolution' on unseen VALIDATION DATA (can do the cross-validation)
# And FINAL CHECK after 'mini evolution' is on unseen TESTING DATA
#
# PREPROCESSING
# Neural Networks work best on data of numbers ranging 0 to 1 (NOT) or -1 to 1 (PREFERABLE)
# Any proprocessing done on training data, also do in validation, testing and prediction data.
# If you scale (devide by biggest) on dataset to get between 0 and 1, ALSO USE THAT SAME FACTOR
# on your other data! So if you want to use on new data: use that scaler. (save it along with ur
# network) For linear scaling fine to find maximum of all datasets, but for non linear scaling
# might leak info from other datasets to training datasets so then training dataset only.
# data augmentation: splice data (for example crop and rotate images from one image) to get more data
#
# REGULARISATION METHODS
# Reduce generalisation error.
# KNOBS: lambda, higher value means more significant penalty
# L1: Sum of absolute weights and biases added to loss (penalty)
# L2: Square of weights and biases added to loss (higher the weight / bias, even higher penalty)
#
# DROPOUT LAYER
# Makes sure we don't rely on few neurons too much; per neuron p probability that it does fire, else it doesnt.
# Per deep learning framework differs if they use p (probaility to keep) or q (probability to disable)


