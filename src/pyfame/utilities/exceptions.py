class PyFameError(Exception):
    """ Base exception class for all custom exceptions in PyFame."""
    pass

class NamespaceError(PyFameError):
    """ Raised when a pyfame specific name is already in use in the current namespace."""
    def __init__(self, message:str=None, name:str=None):
        if message is None:
            if name is None:
                self.message = "Pyfame has encountered a namespace collision (between an environment variable, directory name or file name)."
            else:
                self.name = name
                self.message = f"Pyfame has encountered a namespace collision with name '{name}'."
        else:
            self.message = message
            self.name = name

class FaceNotFoundError(PyFameError):
    """ Raised when the mediapipe FaceLandmarker cannot successfully detect a face in a given frame."""
    def __init__(self, message:str="FaceLandmarker failed to identify a face in the provided image or video."):
        self.message = message
        super().__init__(self.message)

class UnrecognizedExtensionError(PyFameError):
    """ Raised when an input file extension does not fall under .mp4, .mov, .jpg, .jpeg, .png or .bmp."""
    def __init__(self, message:str=None, extension:str=None):
        self.message = message
        if extension is not None:
            self.extension = extension
            self.message = f"Unrecognized file extension '{extension}'."
        
        super().__init__(self.message)

class FileReadError(PyFameError):
    """ Raised when there are errors instantiating cv2.VideoCapture(), or calling cv2.imread()."""
    def __init__(self, message:str=None, file_path:str=None):      
        if message is None:
            if file_path is not None:
                self.message = f"File at '{file_path}' may be corrupt, incorrectly encoded, or have invalid read/write permissions."
            else:
                self.message = "File may be corrupt, incorrectly encoded, or have invalid read/write permissions."
        else:
            self.message = message
        super().__init__(self.message)

class FileWriteError(PyFameError):
    """ Raised when there are errors instantiating cv2.VideoWriter(), or calling cv2.imwrite()."""
    def __init__(self, message:str=None, file_path:str=None):
        if message is None:
            if file_path is not None:
                self.message = f"Path '{file_path}' is either an invalid path, or the path cannot be located in the current working directory."
            else:
                self.message = "Invalid file path, or path cannot be found in your current working directory."
        else:
            self.message = message
        super().__init__(self.message)

class ImageShapeError(PyFameError):
    """ Raised when there are mismatching image shapes provided to file conversion functions."""
    def __init__(self, message:str="Input image shapes do not match. Please see pyfame.standardise_image_dimensions()."):
        self.message = message
        super().__init__(self.message)