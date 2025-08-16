
import cv2 # import 'opencv'!
import mediapipe as mp
from pyTorchAI import *

datacsv = 'face_landmarks.csv'

answer = input("Data mode or Analyse mode (d/a): ")

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def normalize_landmarks(landmarks, width, height, ninput):
    coords = []
    if ninput == 'd':
        coords = [(lm.x * width, lm.y * height) for lm in landmarks.landmark]
    else:
        coords = [(lm.x * width, lm.y * height) for lm in landmarks]
    min_x = min(coords, key=lambda item: item[0])[0]
    max_x = max(coords, key=lambda item: item[0])[0]
    min_y = min(coords, key=lambda item: item[1])[1]
    max_y = max(coords, key=lambda item: item[1])[1]
    normalized = [(round((x - min_x) / (max_x - min_x), 2), round((y - min_y) / (max_y - min_y), 2)) for x, y in coords]
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

filepath = 'face.model'

if answer == "d":
    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            key_pressed = cv2.waitKey(5)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                    normalized_landmarks = normalize_landmarks(face_landmarks, image.shape[1], image.shape[0], answer)
                    if key_pressed in range(ord('0'), ord('6')):
                        label = chr(key_pressed)
                        data_to_save = [label] + [coord for pair in normalized_landmarks for coord in pair]
                        write_to_csv(data_to_save)
                        print(f"Saved data for key '{label}':", data_to_save[:3])
            cv2.imshow('Face Tracking', image)
            if key_pressed == 8:
                delete_last_csv_entry()
                print("Deleted the last saved entry.")
            if key_pressed == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
else:

    # Open webcam
    cap = cv2.VideoCapture(0)

    this_model = Model.load(filepath)

    # Function to normalize and predict the number of fingers
    def predict_mood(face_landmarks, image_shape):
        normalized_landmarks = normalize_landmarks(face_landmarks, image_shape[1], image_shape[0], answer)
        X = np.array([(float(coord) * 2 - 1) for pair in normalized_landmarks for coord in pair]).reshape(1, -1)
        output = this_model.predict(X)
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
            results = face_mesh.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw the hand landmarks
                    mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

                    # Predict number of fingers and display result
                    prediction, certainty, _ = predict_mood(face_landmarks.landmark, image.shape)
                    text = f"Predicted: {prediction}, Certainty: {certainty*100:.2f}%"
                    cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display the image
            cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
