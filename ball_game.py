
import cv2
import mediapipe as mp
import pygame
import sys
from pygame.locals import *
from threading import Thread
import numpy as np
from collections import deque

# Constants for the ball
BALL_SIZE = 30
BALL_COLOR = (255, 0, 0)
GRAVITY = 0.8
DAMPING = 0.45
MOVEMENT_THRESHOLD = 0.01  # Movement threshold as a percentage of screen width/height
FRICTION = 0.95
FORCE_SCALE = 13.0
HISTORY_LENGTH = 3  # Number of past positions to keep for averaging

# MediaPipe setup for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.2)

# OpenCV setup for video capture
cap = cv2.VideoCapture(0)

# Initialize Pygame
pygame.init()

# Set up the screen, fullscreen
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_width, screen_height = screen.get_size()
background_color = (0, 0, 0)

# Set up the ball
ball_pos = [screen_width // 2, screen_height // 2]
ball_vel = [0., 0.]

# Clock to manage the frame rate
clock = pygame.time.Clock()

# Shared variables for communication between threads
position_history = deque(maxlen=HISTORY_LENGTH)
calc_force = True
def camera_thread():
    global position_history, calc_force
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands.process(frame)

        if results.multi_hand_landmarks:
            calc_force = True
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                finger_pos = np.array([index_finger_tip.x * screen_width, index_finger_tip.y * screen_height])
                print(finger_pos)
                position_history.append(finger_pos)
        else:
            calc_force = False


def calculate_force():
    if not calc_force:
        return np.array([0, 0])
    global position_history
    if len(position_history) >= 2:
        # Calculate movement based on the average of the first and last positions in the history
        first_pos = np.mean(np.array([position_history[0], position_history[1]]), axis=0)
        last_pos = np.mean(np.array([position_history[-2], position_history[-1]]), axis=0)
        direction = last_pos - first_pos
        movement = np.linalg.norm(direction)

        # Only apply a fixed force if movement exceeds the threshold
        if movement > screen_width * MOVEMENT_THRESHOLD:
            unit_vector = direction / movement  # Normalizing the direction vector
            return unit_vector * FORCE_SCALE  # Apply the same force magnitude in the direction of movement
    return np.array([0, 0])


def move_ball():
    global ball_vel, ball_pos
    current_force = calculate_force()
    ball_vel += current_force
    ball_vel[1] += GRAVITY

    # Bounce and damping
    if ball_pos[0] <= BALL_SIZE or ball_pos[0] >= screen_width - BALL_SIZE:
        ball_vel[0] = -ball_vel[0] * DAMPING
    if ball_pos[1] <= BALL_SIZE or ball_pos[1] >= screen_height - BALL_SIZE:
        ball_vel[1] = -ball_vel[1] * DAMPING

    ball_pos += ball_vel
    ball_pos = np.clip(ball_pos, BALL_SIZE, np.array([screen_width, screen_height]) - BALL_SIZE)
    ball_vel *= FRICTION

# Start the camera thread
thread = Thread(target=camera_thread)
thread.start()

while True:
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            cap.release()
            cv2.destroyAllWindows()
            pygame.quit()
            sys.exit()

    # Update game state
    move_ball()

    # Draw everything
    screen.fill(background_color)
    pygame.draw.circle(screen, BALL_COLOR, ball_pos.astype(int), BALL_SIZE)
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)
