from abc import ABC, abstractmethod
from cv2.typing import MatLike

class layer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply_layer(self, frame:MatLike, weight:float, roi:list[list[tuple]]) -> MatLike:
        pass