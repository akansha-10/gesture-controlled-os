import enum
import numpy as np
import pyautogui

from .hand import Hand
from .utils import clamp
from .filter import OneEuroFilter


class HandyMouseController:
    sensitivity_range = 1.5
    min_margin = 0.15
    max_margin = 0.5

    class MouseState(enum.IntEnum):
        NORMAL = 0
        SCROLL = 1
        DRAG = 2

    class Event(enum.IntEnum):
        MOVE = Hand.Pose.OPEN
        STOP = Hand.Pose.CLOSE
        LEFT_CLICK = Hand.Pose.INDEX_UP
        RIGHT_CLICK = Hand.Pose.PINKY_UP
        SCROLL = Hand.Pose.THUMB_SIDE
        DRAG = Hand.Pose.INDEX_MIDDLE_UP

    def __init__(self, sensitivity=0.5, margin=0.25):
        self._sensitivity = clamp(sensitivity, 0, 1)
        self._margin = clamp(margin, 0, 1)

        self.state = self.MouseState.NORMAL
        self.prev_pose = Hand.Pose.UNDEFINED
        self.prev_position = np.zeros(2)
        self.screen_size = np.array(pyautogui.size())

        self.filters = [OneEuroFilter(0, 0), OneEuroFilter(0, 0)]
        self.frame = 0

    def palm_center(self, landmarks):
        self.frame += 1
        center = np.zeros(2)
        for i in range(2):
            center[i] = landmarks[i::2][Hand.palm_landmarks].mean()
            center[i] = self.filters[i](self.frame, center[i])
        return center

    def to_screen(self, xy):
        margin = self.min_margin + self._margin * (self.max_margin - self.min_margin)
        scale = 1 / (1 - margin)
        xy = scale * (xy - margin / 2)
        xy = np.clip(xy, 0, 1)
        return xy * self.screen_size

    def update(self, hand, palm_center, confidence, min_conf=0.5):
        screen_xy = self.to_screen(palm_center)

        if hand.pose == self.Event.MOVE:
            delta = (screen_xy - self.prev_position) * self._sensitivity
            pyautogui.move(delta[0], delta[1], _pause=False)

        elif hand.pose != self.prev_pose and confidence > min_conf:
            if hand.pose == self.Event.LEFT_CLICK:
                pyautogui.leftClick(_pause=False)
            elif hand.pose == self.Event.RIGHT_CLICK:
                pyautogui.rightClick(_pause=False)

        self.prev_position = screen_xy
        self.prev_pose = hand.pose
