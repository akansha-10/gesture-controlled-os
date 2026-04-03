#!/usr/bin/env python

import argparse
import json
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import keras

from hand_control import Hand, HandyMouseController
from hand_control.models import __default_model__
from hand_control.utils import (
    draw_hand_landmarks,
    draw_palm_center,
    draw_control_bounds,
    write_pose,
    __window_name__,
)
from hand_control.config import __default_config__


pyautogui.FAILSAFE = False
mp_hands = mp.solutions.hands


def main():
    with open(__default_config__) as cfg:
        default_config = json.load(cfg)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.description = (
        "Control the mouse using hand gestures captured from a webcam."
    )

    parser.add_argument("--model", type=str, default=default_config["model"])
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--minimum_prediction_confidence",
        type=float,
        default=default_config["minimum_prediction_confidence"],
    )

    args = parser.parse_args()

    trained_model = keras.models.load_model(
        args.model or __default_model__
    )

    controller = HandyMouseController()

    capture = cv2.VideoCapture(0)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while capture.isOpened():
            success, image = capture.read()
            image = cv2.flip(image, 1)

            if not success:
                continue

            results = hands.process(image)

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                draw_hand_landmarks(image, landmarks)

                hand = Hand()
                vec = hand.vectorize_landmarks(landmarks)
                processed = hand.normalize(vec)

                probs = trained_model.predict(processed, verbose=0).flatten()
                confidence = np.max(probs)
                hand.pose = Hand.Pose(np.argmax(probs))

                palm_center = controller.palm_center(vec)
                controller.update(
                    hand,
                    palm_center,
                    confidence,
                    min_conf=args.minimum_prediction_confidence,
                )

                if args.show:
                    draw_palm_center(image, palm_center)
                    write_pose(image, hand.pose.name)

            if args.show:
                draw_control_bounds(image, controller)
                cv2.imshow(__window_name__, image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    capture.release()


if __name__ == "__main__":
    main()
