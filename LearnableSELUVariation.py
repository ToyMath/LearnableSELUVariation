import numpy as np

class LearnableSELUVariation:
    """
    Implements the Learnable Scaled Exponential Linear Unit Variation (LearnableSELUVariation) 
    activation function using NumPy. This activation function extends the SELU activation function 
    by introducing learnable parameters, allowing it to dynamically adjust its behavior during the 
    training of a neural network model. It is designed to maintain self-normalizing properties 
    while introducing additional flexibility through a non-linear component for negative inputs.

    The mathematical formula for the LearnableSELUVariation activation function is defined as:
    f(x) = lambda_ * x for x > 0, and
    f(x) = lambda_ * (alpha * (exp(beta * x) - 1) + gamma * sin(omega * x)) for x <= 0,
    where lambda_, alpha, beta, gamma, and omega are learnable parameters that control the shape 
    of the activation function.

    Parameters:
    - lambda_ (float): Scaling parameter λ to ensure self-normalization. Defaults to 1.0507.
    - alpha (float): Scaling parameter α for exponential growth for negative inputs. Defaults to 1.67326.
    - beta (float): Adjustment parameter β for the exponential component. Defaults to 1.0.
    - gamma (float): Amplitude γ of the sinusoidal component. Defaults to 0.1.
    - omega (float): Frequency ω of the sinusoidal component. Defaults to 2.0.

    Methods:
    - __call__(x): Applies the LearnableSELUVariation activation function to the input array x.

    Example:
    >>> activation_fn = LearnableSELUVariation()
    >>> x = np.array([-2, -1, 0, 1, 2])
    >>> y = activation_fn(x)
    >>> print(y)
    """
    def __init__(self, lambda_=1.0507, alpha=1.67326, beta=1.0, gamma=0.1, omega=2.0):
        self.lambda_ = lambda_
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega = omega

    def __call__(self, x):
        """
        Applies the LearnableSELUVariation activation function to the input array x.

        Parameters:
        - x (numpy.ndarray): Input array to which the activation function will be applied.

        Returns:
        - numpy.ndarray: Output array after applying the LearnableSELUVariation activation function.
        """
        return np.where(x > 0, 
                        self.lambda_ * x,
                        self.lambda_ * (self.alpha * (np.exp(self.beta * x) - 1) + self.gamma * np.sin(self.omega * x)))
