import time
import enum
import numpy as np
import cv2


class Hand:
    class Pose(enum.IntEnum):
        UNDEFINED = -1
        OPEN = 0
        CLOSE = 1
        INDEX_UP = 2
        PINKY_UP = 3
        THUMB_SIDE = 4
        INDEX_MIDDLE_UP = 5

    palm_landmarks = [0, 5, 9, 13, 17]
    dimension = 2

    def __init__(self, pose=None):
        if pose is None:
            self.pose = self.Pose.UNDEFINED
        elif isinstance(pose, (int, str)):
            self.pose = self.Pose(pose)
        else:
            self.pose = pose

    def vectorize_landmarks(self, landmarks):
        vec = np.zeros(self.dimension * len(landmarks))
        for i, lm in enumerate(landmarks):
            vec[self.dimension * i] = lm.x
            vec[self.dimension * i + 1] = lm.y
        return vec

    def normalize(self, vec):
        for axis in range(self.dimension):
            vec[axis::self.dimension] -= vec[axis::self.dimension].mean()
            vec[axis::self.dimension] /= vec[axis::self.dimension].std()
        return vec.reshape(1, -1)


class HandSnapshot:
    def __init__(self, hand=None):
        self.timestamp = time.time()
        self.hand = hand if hand else Hand()

    def save_image(self, image, path="hand_snapshot.jpg"):
        cv2.imwrite(path, image)

    def save_landmarks(self, landmarks, path="sample.dat"):
        raw = self.hand.vectorize_landmarks(landmarks)
        processed = self.hand.normalize(raw)

        with open(path, "w") as f:
            f.write(f"{self.hand.pose.value}\n")
            f.write(" ".join(f"{v:.6f}" for v in processed.flatten()))
