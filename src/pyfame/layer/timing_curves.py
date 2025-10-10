import numpy as np

# Defining useful timing functions
def timing_constant(time_delta:float, time_onset:float, time_offset:float, rise_duration:float, fall_duration:float, **kwargs) -> float:
    """ Always returns 1.0, regardless of input.
    
    Parameters 
    ----------

    time_delta: float
        The current timestamp (msec) of the video file being evaluated.

    time_onset: float
        The timestamp at which to transition to 1.0.
    
    time_offset: float
        The timestamp at which to transition back to 0.0. 
    """
    time_delta /= 1000

    if time_delta < time_onset:
        return 0.0
    elif time_onset <= time_delta < time_offset:
        return 1.0
    elif time_offset <= time_delta:
        return 0.0
    else:
        return 0.0

def timing_linear(time_delta:float, time_onset:float, time_offset:float, rise_duration:float, fall_duration:float) -> float:
    """ Normalised linear timing function.

    Parameters
    ----------

    time_delta: float
        The current timestamp (msec) of the video file being processed.
    
    time_onset: float
        The timestamp at which to begin linearly transitioning to 1.0.
    
    time_offset: float
        The timestamp at which to begin linearly transitioning back to 0.0.
    
    rise_duration: float
        The time duration the linear rise transition should take.
    
    fall_duration: float
        The time duration the linear fall transition should take.
    
    Returns
    -------

    weight: float
    """
    time_delta /= 1000

    def linear(t):
        return np.clip(t, 0.0, 1.0)

    if time_delta < time_onset:
        return 0.0
    elif time_onset <= time_delta < time_onset + rise_duration:
        cur_eval = (time_delta - time_onset) / rise_duration
        return linear(cur_eval)
    elif time_onset + rise_duration <= time_delta < time_offset - fall_duration:
        return 1.0
    elif time_offset - fall_duration <= time_delta < time_offset:
        cur_eval = (time_delta - time_offset) / fall_duration
        return linear(1 - cur_eval)
    else:
        return 0.0

def timing_sigmoid(time_delta:float, time_onset:float, time_offset:float, rise_duration:float, fall_duration:float, growth_rate:float = 10.0) -> float:
    """ Returns the value of the sigmoid function evaluated at time t. If paramater k (scaling factor) is 
    not provided in kwargs, it will be set to 1.
    
    Parameters 
    ----------

    time_delta: float
        The current timestamp (msec) of the video file being evaluated. 
    
    time_onset: float
        The timestamp at which to begin linearly transitioning to 1.0.
    
    time_offset: float
        The timestamp at which to begin linearly transitioning back to 0.0.
    
    rise_duration: float
        The time duration the linear rise transition should take.
    
    fall_duration: float
        The time duration the linear fall transition should take.

    growth_rate: float
        The slope or growth rate parameter, controls how quickly the sigmoid function transitions
        from zero to one. 
    
    returns
    -------

    weight: float
        A normalised weight in the range [0,1].
    """
    time_delta /= 1000

    def scaled_sigmoid(t, k):
        raw = 1 / (1 + np.exp(-k * (t-0.5)))
        min_val = 1 / (1 + np.exp(k*0.5))
        max_val = 1 / (1 + np.exp(-k*0.5))
        return (raw - min_val) / (max_val - min_val)

    if time_delta < time_onset:
        return 0.0
    elif time_onset <= time_delta < time_onset + rise_duration:
        cur_eval = (time_delta - time_onset) / rise_duration
        return scaled_sigmoid(cur_eval, growth_rate)
    elif time_onset + rise_duration <= time_delta < time_offset - fall_duration:
        return 1.0
    elif time_offset - fall_duration <= time_delta < time_offset:
        cur_eval = (time_delta - time_offset) / fall_duration
        return scaled_sigmoid(1-cur_eval, growth_rate)
    else:
        return 0.0

def timing_gaussian(time_delta:float, time_onset:float, time_offset:float, rise_duration:float, fall_duration:float, growth_rate:float = 6.0) -> float:
    """ Normalized gaussian timing function

    Parameters 
    ----------

    time_delta: float
        The current timestamp (msec) of the video file being evaluated. 
    
    time_onset: float
        The timestamp at which to begin linearly transitioning to 1.0.
    
    time_offset: float
        The timestamp at which to begin linearly transitioning back to 0.0.
    
    rise_duration: float
        The time duration the linear rise transition should take.
    
    fall_duration: float
        The time duration the linear fall transition should take.

    gaussian_mean: float
        The mean or center of the gaussian distribution.
    
    growth_rate: float
        Controls the steepness of the rise/fall transition curves.
    
    returns
    -------

    weight: float
        A normalised weight in the range [0,1].
    """

    time_delta /= 1000

    def gaussian(t, sigma):
        raw = np.exp(-((t - 0.5) ** 2) / (2 * sigma**2))
        min_val = np.exp(-((0.0 - 0.5) ** 2) / (2 * sigma**2))
        max_val = 1.0
        return (raw - min_val) / (max_val - min_val)
    
    rise_sigma = rise_duration / growth_rate
    fall_sigma = fall_duration / growth_rate
    
    if time_delta < time_onset:
        return 0.0
    elif time_onset <= time_delta < time_onset + rise_duration:
        cur_eval = ((time_delta - time_onset) / rise_duration) * 0.5
        return gaussian(cur_eval, rise_sigma)
    elif time_onset + rise_duration <= time_delta < time_offset - fall_duration:
        return 1.0
    elif time_offset - fall_duration <= time_delta < time_offset:
        cur_eval = ((time_delta - time_offset) / fall_duration) * 0.5 + 0.5
        return gaussian(1-cur_eval, fall_sigma)
    else:
        return 0.0