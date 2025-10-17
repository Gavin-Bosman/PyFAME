from collections import deque
from mediapipe.tasks.python.vision.face_landmarker import Blendshapes
import numpy as np

class EyeBlendshapeSmoother:
    def __init__(self, frame_window_size = 5, lower_threshold = 0.3, upper_threshold = 0.45):
        self.left_blink_scores = deque(maxlen=frame_window_size)
        self.right_blink_scores = deque(maxlen=frame_window_size)
        self.prev_left_state = True
        self.prev_right_state = True
        self.lower_score_threshold = lower_threshold
        self.upper_score_threshold = upper_threshold
    
    def update(self, blendshapes, return_scores = False):
        left_blink_score = self._get_blendshape_score(blendshapes, "eyeBlinkLeft")
        right_blink_score = self._get_blendshape_score(blendshapes, "eyeBlinkRight")

        self.left_blink_scores.append(left_blink_score)
        self.right_blink_scores.append(right_blink_score)

        smoothed_left = np.mean(self.left_blink_scores)
        smoothed_right = np.mean(self.right_blink_scores)

        self.prev_left_state = self._is_eye_open(smoothed_left, self.prev_left_state)
        self.prev_right_state = self._is_eye_open(smoothed_right, self.prev_right_state)

        if return_scores:
            return [(self.prev_left_state, self.prev_right_state), (left_blink_score, right_blink_score)]
        else:
            return self.prev_left_state, self.prev_right_state
    
    def _is_eye_open(self, score, prev_state):
        if score >= self.upper_score_threshold:
            return False
        elif score <= self.lower_score_threshold:
            return True
        else:
            return prev_state
    
    @staticmethod
    def _get_blendshape_score(blendshapes, name):
        for bs in blendshapes:
            if bs.category_name == name:
                return bs.score
        return 0.0
    
    