import cv2
import mediapipe as mp
import csv
import numpy as np
from main import Layer_Dense, Activation_ReLU

datacsv = 'hand_landmarks.csv'

answer = input("Data mode or Analyse mode (d/a): ")

# Initialize MediaPipe Hands model.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

def normalize_landmarks(landmarks, width, height):
    # Extract coordinates
    coords = [(lm.x * width, lm.y * height) for lm in landmarks]
    # Get min and max coordinates
    min_x = min(coords, key=lambda item: item[0])[0]
    max_x = max(coords, key=lambda item: item[0])[0]
    min_y = min(coords, key=lambda item: item[1])[1]
    max_y = max(coords, key=lambda item: item[1])[1]
    # Normalize coordinates
    normalized = [(round((x - min_x) / (max_x - min_x), 2), round((y - min_y) / (max_y - min_y), 2))
                  for x, y in coords]
    return normalized

def write_to_csv(data):
    with open(datacsv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def delete_last_csv_entry():
    with open(datacsv, 'r+', newline='') as file:
        lines = file.readlines()
        file.seek(0)
        file.truncate()
        file.writelines(lines[:-1])

filepath = 'trained/2024-05-03_15-15-23.csv'

if answer == "d":
    # Open webcam
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # Flip the image horizontally for a later selfie-view display
            # Convert the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # Process the image and get the hand landmarks.
            results = hands.process(image)

            # Convert the image color back so it can be displayed correctly.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            key_pressed = cv2.waitKey(5)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the hand landmarks.
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Normalize the landmarks
                    normalized_landmarks = normalize_landmarks(hand_landmarks.landmark, image.shape[1], image.shape[0])

                    # Handle key press actions for saving landmarks
                    if key_pressed in range(ord('0'), ord('6')):
                        label = chr(key_pressed)
                        data_to_save = [label] + [coord for pair in normalized_landmarks for coord in pair]
                        write_to_csv(data_to_save)
                        print(f"Saved data for key '{label}':", data_to_save)

            # Show the image.
            cv2.imshow('Hand Tracking', image)

            # Handle backspace for deleting the last entry
            if key_pressed == 8:  # Backspace key
                delete_last_csv_entry()
                print("Deleted the last saved entry.")

            if key_pressed == 27:  # Escape key
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
else:

    # Open webcam
    cap = cv2.VideoCapture(0)

    dense1 = Layer_Dense(42, 16, weight_regulariser_l2=5e-4, bias_regulariser_l2=5e-4)

    activation1 = Activation_ReLU()

    dense2 = Layer_Dense(16, 6)

    # Function to read network parameters from CSV
    def load_network(filepath, *layers):
        with open(filepath, mode='r') as file:
            csv_reader = csv.reader(file)
            rows = list(csv_reader)

            layer_index = -1
            read_mode = None  # Can be 'Weights' or 'Biases'
            current_data = []
            for row in rows:
                #print(row)
                if row[0] == 'Layer':
                    # e.g., ['Layer', '0', 'Weights']
                    if layer_index != -1:
                        # Set the last batch of data read
                        if read_mode == 'Weights':
                            for i in range(len(current_data)):
                                for j in range(len(current_data[i])):
                                    layers[layer_index].weights[i][j] = current_data[i][j]
                            print(f"Layer {layer_index} weights: ")
                            print(current_data)
                        elif read_mode == 'Biases':
                            for i in range(len(current_data)):
                                for j in range(len(current_data[i])):
                                    layers[layer_index].biases[i][j] = current_data[i][j]
                            print(f"Layer {layer_index} biases: ")
                            print(current_data)
                    layer_index = int(row[1])
                    read_mode = row[2]
                    current_data = []
                else:
                    # Collecting data
                    current_data.append([float(x) for x in row])
            if read_mode == 'Weights':
                for i in range(len(current_data)):
                    for j in range(len(current_data[i])):
                        layers[layer_index].weights[i][j] = current_data[i][j]
                print(f"Layer {layer_index} weights: ")
                print(current_data)
            elif read_mode == 'Biases':
                for i in range(len(current_data)):
                    for j in range(len(current_data[i])):
                        layers[layer_index].biases[i][j] = current_data[i][j]
                print(f"Layer {layer_index} biases: ")
                print(current_data)

    load_network(filepath, dense1, dense2)

    # Function to normalize and predict the number of fingers
    def predict_fingers(hand_landmarks, image_shape):
        normalized_landmarks = normalize_landmarks(hand_landmarks, image_shape[1], image_shape[0])
        X = np.array([(float(coord) * 2 - 1) for pair in normalized_landmarks for coord in pair]).reshape(1, -1)
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        output = dense2.output
        #print(dense2.output)
        # Softmax-like normalization to make the outputs non-negative and sum to 1
        exp_output = np.exp(output - np.max(output))
        probabilities = exp_output / np.sum(exp_output)
        prediction = np.argmax(probabilities)
        certainty = probabilities[0, prediction]
        return prediction, certainty, normalized_landmarks

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # Process image and extract hand landmarks
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the hand landmarks
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Predict number of fingers and display result
                    prediction, certainty, _ = predict_fingers(hand_landmarks.landmark, image.shape)
                    text = f"Predicted: {prediction}, Certainty: {certainty*100:.2f}%"
                    cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display the image
            cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
