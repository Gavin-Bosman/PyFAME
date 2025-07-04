class PyFameError(Exception):
    """ Base exception class for all custom exceptions in PyFame."""
    pass

class FaceNotFoundError(PyFameError):
    """ Raised when the mediapipe FaceMesh cannot successfully detect a face in a given frame."""
    def __init__(self, message:str="FaceMesh failed to identify a face in the provided image."):
        self.message = message
        super().__init__(self.message)

class UnrecognizedExtensionError(PyFameError):
    """ Raised when an input file extension does not fall under .mp4, .mov, .jpg, .jpeg, .png or .bmp"""
    def __init__(self, message:str="Unrecognized file extension.", extension:str = None):
        self.message = message
        if extension is not None:
            self.extension = extension
            self.message = f"Unrecognized file extension '{extension}'."
        
        super().__init__(self.message)

class FileReadError(PyFameError):
    """ Raised when there are errors instantiating cv2.VideoCapture(), or calling cv2.imread()"""
    def __init__(self, message:str="File may be corrupt, incorrectly encoded, or have invalid r/w permissions."):
        self.message = message
        super().__init__(self.message)

class FileWriteError(PyFameError):
    """ Raised when there are errors instantiating cv2.VideoWriter(), or calling cv2.imwrite()"""
    def __init__(self, message:str="Invalid file path, or path cannot be found in your current working directory."):
        self.message = message
        super().__init__(self.message)

class ImageShapeError(PyFameError):
    """ Raised when there are mismatching image shapes provided to the moviefy function."""
    def __init__(self, message:str="Input image shapes do not match."):
        self.message = message
        super().__init__(self.message)