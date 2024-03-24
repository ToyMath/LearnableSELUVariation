# LearnableSELUVariation Activation Function

The `LearnableSELUVariation` activation function is designed to dynamically adjust its behavior to better fit specific data characteristics by incorporating learnable parameters. This activation function combines the self-normalizing properties of SELU with additional flexibility, making it potentially more effective for complex patterns.

<img width="841" alt="LearnableSELUVariation" src="https://github.com/ToyMath/LearnableSELUVariation/assets/5700430/0168b1c7-769d-49c3-a56b-30c4c8bfcbb8">

## Mathematical Formula

The LearnableSELUVariation function is defined as follows:

```math
f(x) = \lambda \cdot \left\{ \begin{array}{ll} x & \text{if } x > 0,\\ \alpha \cdot (e^{\beta \cdot x} - 1) + \gamma \cdot \sin(\omega \cdot x) & \text{if } x \leq 0. \end{array} \right.
```

Where \($\\lambda\$\), \($\\alpha\$\), \($\\beta\$\), \($\\gamma\$\), and \($\\omega\$\) are learnable parameters, adjusting the function's behavior during training.

### Code

[LearnableSELUVariation.py](https://github.com/ToyMath/LearnableSELUVariation/blob/main/LearnableSELUVariation.py)

## Installation

```bash
git clone https://github.com/ToyMath/LearnableSELUVariation.git
cd LearnableSELUVariation
```

## Usage

### TensorFlow Implementation

The TensorFlow implementation is provided above. Here's how to use it in a model:

```python
import tensorflow as tf

class LearnableSELUVariation(tf.keras.layers.Layer):
    def __init__(self):
        super(LearnableSELUVariation, self).__init__()
        self.lambda_ = self.add_weight(name='lambda', shape=(), initializer=tf.constant_initializer(1.0507), trainable=True)
        self.alpha = self.add_weight(name='alpha', shape=(), initializer=tf.constant_initializer(1.67326), trainable=True)
        self.beta = self.add_weight(name='beta', shape=(), initializer=tf.constant_initializer(1.0), trainable=True)
        self.gamma = self.add_weight(name='gamma', shape=(), initializer=tf.constant_initializer(0.1), trainable=True)
        self.omega = self.add_weight(name='omega', shape=(), initializer=tf.constant_initializer(2.0), trainable=True)

    def call(self, inputs):
        return tf.where(inputs > 0, self.lambda_ * inputs,
                        self.lambda_ * (self.alpha * (tf.exp(self.beta * inputs) - 1) + self.gamma * tf.sin(self.omega * inputs)))
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn

class LearnableSELUVariation(nn.Module):
    def __init__(self):
        super(LearnableSELUVariation, self).__init__()
        self.lambda_ = nn.Parameter(torch.tensor(1.0507))
        self.alpha = nn.Parameter(torch.tensor(1.67326))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.omega = nn.Parameter(torch.tensor(2.0))

    def forward(self, inputs):
        return torch.where(inputs > 0, self.lambda_ * inputs,
                           self.lambda_ * (self.alpha * (torch.exp(self.beta * inputs) - 1) + self.gamma * torch.sin(self.omega * inputs)))
```

### JAX Implementation

```python
import jax.numpy as jnp
from jax import random, jit

class LearnableSELUVariation:
    def __init__(self, key):
        self.lambda_ = random.normal(key, ()) * 0.1 + 1.0507
        self.alpha = random.normal(key, ()) * 0.1 + 1.67326
        self.beta = random.normal(key, ()) * 0.1 + 1.0
        self.gamma = random.normal(key, ()) * 0.1 + 0.1
        self.omega = random.normal(key, ()) * 0.1 + 2.0
        # Initialize parameters here with JAX random if they are meant to be learnable

    @jit
    def __call__(self, inputs):
        return jnp.where(inputs > 0, self.lambda_ * inputs,
                         self.lambda_ * (self.alpha * (jnp.exp(self.beta * inputs) - 1) + self.gamma * jnp.sin(self.omega * inputs)))
```

## Customization

The `LearnableSELUVariation` activation function is highly customizable through its learnable parameters. By training these parameters alongside the model's weights, `LearnableSELUVariation` can adapt its behavior to best suit the training data, potentially leading to

## Citation

If you use LearnableSELUVariation in your research, please cite the following work:

```bibtex
@misc{LearnableSELUVariation-2024,
  author = {Aakash Apoorv},
  title = {LearnableSELUVariation},
  year = {2024},
  howpublished = {\url{https://github.com/ToyMath/LearnableSELUVariation}},
}
```
