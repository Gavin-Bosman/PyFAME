import numpy as np

# Defining useful timing functions
def constant(t:float, **kwargs) -> float:
    """ Always returns 1.0, regardless of input."""
    return 1.0

def sigmoid(t:float, **kwargs) -> float:
    """ Returns the value of the sigmoid function evaluated at time t. If paramater k (scaling factor) is 
    not provided in kwargs, it will be set to 1.

    """
    k = 1
    if "k" in kwargs:
        k = kwargs["k"]

    return 1/(1 + np.exp(-k * t))

def linear(t:float, **kwargs) -> float:
    """ Normalised linear timing function.

    Parameters
    ----------

    t: float
        The current time value of the video file being processed.
    
    kwargs: dict
        The linear timing function requires a start and end time, typically you will pass 0, and the video duration
        as start and end values. 
    
    Returns
    -------

    weight: float
    """

    start = 0.0
    if "start" in kwargs:
        start = kwargs["start"]
    
    # end kwarg is always passed internally by package functions
    end = kwargs["end"]

    return (t-start) / (end-start)

def gaussian(t:float, **kwargs) -> float:
    """ Normalized gaussian timing function. Evaluates the gaussian function (with default mean:0.0, sd:1.0) at
    the given timestamp t. Both the "mean" and "sigma" parameters can be passed as keyword arguments, however this may 
    affect the normalization of the outputs.

    Parameters 
    ----------

    t: float
        The current timestamp of the video file being evaluated. 

    Key-word Arguments
    ------------------

    mean: float
        The mean or center of the gaussian distribution.
    sigma: float
        The standard deviation or spread of the gaussian distribution.
    
    returns
    -------

    weight: float
        A normalised weight in the range [0,1].
    """

    mean = 0.0
    sigma = 1.0

    if "mean" in kwargs:
        mean = kwargs["mean"]
    if "sigma" in kwargs:
        sigma = kwargs["sigma"]
    
    return np.exp(-((t - mean) ** 2) / (2 * sigma ** 2))