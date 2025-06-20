import numpy as np

# Defining useful timing functions
def timing_constant(t:float, onset:float=None, offset:float=None, **kwargs) -> float:
    """ Always returns 1.0, regardless of input.
    
    Parameters 
    ----------

    t: float
        The current timestamp of the video file being evaluated. 
    """
    return 1.0

def timing_sigmoid(t:float, onset:float, offset:float, **kwargs) -> float:
    """ Returns the value of the sigmoid function evaluated at time t. If paramater k (scaling factor) is 
    not provided in kwargs, it will be set to 1.
    
    Parameters 
    ----------

    t: float
        The current timestamp of the video file being evaluated. 
    
    Keyword Arguments
    -----------------

    k: float
        The slope or growth rate parameter, controls how quickly the sigmoid function transitions
        from zero to one. 
    
    returns
    -------

    weight: float
        A normalised weight in the range [0,1].
    """

    nt = (t-onset) / (offset-onset)

    k = 1.0
    if "k" in kwargs:
        k = kwargs["k"]

    if t < onset:
        return 0.0
    elif t >= offset:
        return 1.0
    else:
        return 1/(1 + np.exp(-k * nt))

def timing_linear(t:float, onset:float, offset:float, **kwargs) -> float:
    """ Normalised linear timing function.

    Parameters
    ----------

    t: float
        The current time value of the video file being processed.
    
    Returns
    -------

    weight: float
    """

    if t < onset:
        return 0.0
    elif t >= offset:
        return 1.0
    else:
        return (t-onset) / (offset-onset)

def timing_gaussian(t:float, onset:float, offset:float, **kwargs) -> float:
    """ Normalized gaussian timing function. Evaluates the gaussian function (with default mean:0.0, sd:1.0) at
    the given timestamp t. Both the "mean" and "sigma" parameters can be passed as keyword arguments, however this may 
    affect the normalization of the outputs.

    Parameters 
    ----------

    t: float
        The current timestamp of the video file being evaluated. 

    Keyword Arguments
    -----------------

    mean: float
        The mean or center of the gaussian distribution.
    sigma: float
        The standard deviation or spread of the gaussian distribution.
    
    returns
    -------

    weight: float
        A normalised weight in the range [0,1].
    """

    nt = (t-onset) / (offset-onset)

    mean = 0.0
    sigma = 1.0

    if "mean" in kwargs:
        mean = kwargs["mean"]
    if "sigma" in kwargs:
        sigma = kwargs["sigma"]
    
    if t < onset:
        return 0.0
    if t >= offset:
        return 1.0
    else:
        return np.exp(-((nt - mean) ** 2) / (2 * sigma ** 2))