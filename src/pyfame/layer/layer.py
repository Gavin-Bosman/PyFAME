from abc import ABC, abstractmethod
from cv2.typing import MatLike

class Layer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply_layer(self, frame:MatLike, roi:list[list[tuple]], weight:float) -> MatLike:
        pass