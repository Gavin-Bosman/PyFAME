from .exceptions import *
from typing import Any, Callable
import os
from pathlib import Path

def check_has_callable_attr(obj:Any, attr_name:str) -> None:
    if not (hasattr(obj, attr_name) and callable(getattr(obj,attr_name))):
        raise ValueError(f"Object must have a callable {attr_name} attribute.")

def check_type(param:Any, accepted_types:list[type], iterable:bool=False) -> None:
    if not iterable:
        if not any(isinstance(param, t) for t in accepted_types):
            accepted_names = ", ".join(t.__name__ for t in accepted_types)
            raise TypeError(f"parameter of type {type(param).__name__} is invalid, expected one of: {accepted_names}.")
    else:
        for i in param:
            if not any(isinstance(i, t) for t in accepted_types):
                accepted_names = ", ".join(t.__name__ for t in accepted_types)
                raise TypeError(f"parameter of type {type(i).__name__} is invalid, expected one of: {accepted_names}.")

def check_value(param:Any, expected_values:list[Any]=None, **kwargs) -> None:
    min = None
    max = None
    full_range = False

    if kwargs.get("min") is not None:
        min = kwargs.get("min")
    if kwargs.get("max") is not None:
        max = kwargs.get("max")
    if min and max:
        full_range = True
    
    if expected_values is not None:
        if not any(param == val for val in expected_values):
            raise ValueError(f"Unrecognized value for input parameter.")
    elif full_range:
        if param < min or param > max:
            raise ValueError(f"Parameter must fall in the range [{min},{max}].")
    elif min is not None:
        if param < min:
            raise ValueError(f"Parameter must be >{min}.")
    elif max is not None:
        if param > max:
            raise ValueError(f"Parameter must be <{max}.")

def check_valid_path(path:str) -> None:
    try:
        _ = Path(path)
    except Exception as e:
        raise ValueError(f"Invalid path format. Reason: {e}")

def check_is_dir(path:str) -> None:
    if not os.path.isdir(path):
        raise OSError("Path string must be a path to a directory.")

def check_is_file(path:str) -> None:
    if not os.path.isfile(path):
        raise OSError("Path string must be a path to a file.")

def check_valid_file_extension(extension:str, allowed_extensions:list[str] = ['.mp4', '.mov', '.jpg', '.jpeg', '.png', '.bmp']) -> None:
    if extension not in allowed_extensions:
        raise UnrecognizedExtensionError(f"Unrecognized file extension {extension}.")

def check_return_type(func:Callable, x:Any, accepted_types:list[type]) -> None:
    y = func(x)
    check_type(y, accepted_types)