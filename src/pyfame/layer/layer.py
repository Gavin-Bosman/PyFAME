from abc import ABC, abstractmethod
from cv2.typing import MatLike
import mediapipe as mp

class Layer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply_layer(self, face_mesh:mp.solutions.face_mesh.FaceMesh, frame:MatLike, roi:list[list[tuple]], weight:float) -> MatLike:
        pass

    @abstractmethod
    def supports_weight(self) -> bool:
        pass