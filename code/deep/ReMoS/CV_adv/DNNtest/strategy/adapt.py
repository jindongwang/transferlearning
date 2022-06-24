from math import ceil
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
import numpy as np

from adapt.strategy.strategy import Strategy
from adapt.utils.functional import greedy_max_set

class FeatureMatrix:
  '''A list of feature vectors of neurons.

  This class defines the feature vectors of each neurons with 29 features.
  Detailed description of feature can be found in the following paper:

  Effective White-Box Testing for Deep Neural Networks with Adaptive Neuron-Selection Strategy
  http://prl.korea.ac.kr/~pronto/home/papers/issta20.pdf
  
  Vars:
    CONST_FEATURES: The number of constant features.
    VARIABLE_FEATURES: The number of vaiable features.
    TOTAL_FEATURES: The total number of features.
  '''

  # Number of features.
  CONST_FEATURES = 17
  VARIABLE_FEATURES = 12
  TOTAL_FEATURES = 29

  def __init__(self, network):
    '''Create a feature matrix from the network.

    Generate feature vectors with constant features (f0-16) and variable
    features (f17-28). Variable feature vectors will be filled with zeros.

    Args:
      network: A wrapped Keras model with `adapt.Network`.
    '''

    # Generate constant vectors.
    self.const_vectors = []

    # For f4-9.
    weights = []

    # For each layer.
    for li, l in enumerate(network.layers[:-1]):

      # For f0-3.
      layer_location = li / (len(network.layers) - 1)
      layer_location = int(layer_location * 4)

      # For f4-9.
      w = l.get_weights()
      if len(w) > 0:
        # Use only weights, not biases.
        w = w[0]
      else:
        # If layer without weights (e.g. Pooling layer).
        w = np.zeros(l.output.shape[1:])

      # For f10-16.
      layer_type = \
        10 if isinstance(l, BatchNormalization) else \
        11 if isinstance(l, MaxPooling2D) or isinstance(l, AveragePooling2D) or isinstance(l, GlobalAveragePooling2D) else \
        12 if isinstance(l, Conv2D) or isinstance(l, ZeroPadding2D) else \
        13 if isinstance(l, Dense) else \
        14 if isinstance(l, Activation) else \
        15 if isinstance(l, Add) or isinstance(l, Concatenate) or isinstance(l, Lambda) else \
        16

      # For each neuron.
      for ni in range(l.output.shape[-1]):

        # For f4-9.
        weights.append(np.mean([w[..., ni]]))

        # Create a constant feature vector.
        vec_c = np.zeros(self.CONST_FEATURES, dtype=int)

        #  f0: Ff layer is located in front 25 percent.
        #  f1: Ff layer is located in between front 25 percent and front 50 percent.
        #  f2: Ff layer is located in between back 50 percent and back 25 percent.
        #  f3: Ff layer is located in back 25 percent.
        vec_c[layer_location] = 1

        # f10: If layer is normalization layer.
        # f11: If layer is pooling layer.
        # f12: If layer is convolution layer.
        # f13: If layer is dense layer.
        # f14: If layer is activation layer.
        # f15: If layer gets inputs from multiple layers.
        # f16: Otherwise.
        vec_c[layer_type] = 1

        self.const_vectors.append(vec_c)

    self.const_vectors = np.array(self.const_vectors)

    # For f4-f9.
    argsort_weights = np.argsort(weights)
    top_10 = int(len(argsort_weights) * 0.9)
    top_20 = int(len(argsort_weights) * 0.8)
    top_30 = int(len(argsort_weights) * 0.7)
    top_40 = int(len(argsort_weights) * 0.6)
    top_50 = int(len(argsort_weights) * 0.5)
    # f4: If neuron have weight of top 10%.
    self.const_vectors[argsort_weights[top_10:], 4] = 1
    # f5: If neuron have weight between top 10% and top 20%.
    self.const_vectors[argsort_weights[top_20:top_10], 5] = 1
    # f6: If neuron have weight between top 20% and top 30%.
    self.const_vectors[argsort_weights[top_30:top_20], 6] = 1
    # f7: If neuron have weight between top 30% and top 40%.
    self.const_vectors[argsort_weights[top_40:top_50], 7] = 1
    # f8: If neuron have weight between top 40% and top 50%.
    self.const_vectors[argsort_weights[top_50:top_40], 8] = 1
    # f9: If neuron have wiehgt of bottom 50%.
    self.const_vectors[argsort_weights[:top_50], 9] = 1

    # Create variable feature vectors.
    self.variable_vectors = np.zeros((len(self.const_vectors), self.VARIABLE_FEATURES), dtype=int)

  def update(self, covered_count, objective_covered):
    '''Update variable feature vectors.

    Update feature vectors for the variable features (f17-28).
    
    Args:
      covered_count: A list of numbers of covering for each neuron.
      objective_covered: A list of coverage vectors when objective (e.g. adversarial
        input) satisfies.

    Returns:
      Self for possible call chains.
    '''

    # Create variable feature vectors.
    self.variable_vectors = np.zeros((len(self.const_vectors), self.VARIABLE_FEATURES), dtype=int)

    # f17: If neuron activated when input satisfies objective function.
    indices = np.squeeze(np.argwhere(objective_covered > 0))
    self.variable_vectors[indices, 0] = 1

    # f18: If neuron never activated.
    indices = np.squeeze(np.argwhere(covered_count < 1))
    self.variable_vectors[indices, 1] = 1

    # For f19-28.
    sorted_indices = np.setdiff1d(np.argsort(covered_count), indices, assume_unique=True)
    top_10 = int(len(sorted_indices) * 0.9)
    top_20 = int(len(sorted_indices) * 0.8)
    top_30 = int(len(sorted_indices) * 0.7)
    top_40 = int(len(sorted_indices) * 0.6)
    top_50 = int(len(sorted_indices) * 0.5)
    top_60 = int(len(sorted_indices) * 0.4)
    top_70 = int(len(sorted_indices) * 0.3)
    top_80 = int(len(sorted_indices) * 0.2)
    top_90 = int(len(sorted_indices) * 0.1)
    # f19: If neuron activated top 10%.
    self.variable_vectors[sorted_indices[top_10:], 2] = 1
    # f20: If neuron activated between top 10% and top 20%.
    self.variable_vectors[sorted_indices[top_20:top_10], 3] = 1
    # f21: If neuron activated between top 20% and top 30%.
    self.variable_vectors[sorted_indices[top_30:top_20], 4] = 1
    # f22: If neuron activated between top 30% and top 40%.
    self.variable_vectors[sorted_indices[top_40:top_30], 5] = 1
    # f23: If neuron activated between top 40% and top 50%.
    self.variable_vectors[sorted_indices[top_50:top_40], 6] = 1
    # f24: If neuron activated between top 50% and top 60%.
    self.variable_vectors[sorted_indices[top_60:top_50], 7] = 1
    # f25: If neuron activated between top 60% and top 70%.
    self.variable_vectors[sorted_indices[top_70:top_60], 8] = 1
    # f26: If neuron activated between top 70% and top 80%.
    self.variable_vectors[sorted_indices[top_80:top_70], 9] = 1
    # f27: If neuron activated between top 80% and top 90%.
    self.variable_vectors[sorted_indices[top_90:top_80], 10] = 1
    # f28: If neuron activated between top 90% and top 100%.
    self.variable_vectors[sorted_indices[:top_90], 11] = 1

    return self

  @property
  def matrix(self):
    '''A list of all feature vectors.'''

    return np.concatenate([self.const_vectors, self.variable_vectors], axis=1)

  def dot(self, vector):
    '''Calculate dot product of the feature matrix and a vector.
    
    Args:
      vector: A 29 dimensional vector.

    Returns:
      A result vector of dot product.
    '''

    return np.dot(self.matrix, vector)

class ParameterizedStrategy(Strategy):
  '''A strategy that uses a parameterized selection strategy.
  
  Parameterized neuron selection strategy is a strategy that parameterized
  neurons and scores with a selection vector. Please see the following paper
  for more details:

  Effective White-Box Testing for Deep Neural Networks with Adaptive Neuron-Selection Strategy
  http://prl.korea.ac.kr/~pronto/home/papers/issta20.pdf
  '''

  def __init__(self, network, bound=5):
    '''Create a parameterized strategy, and initialize its variables.
    
    Args:
      network: A wrapped Keras model with `adapt.Network`.
      bound: A floating point number indicates the absolute value of minimum
        and maximum bounds.

    Example:

    >>> from adapt import Network
    >>> from adapt.strategy import ParameterizedStrategy
    >>> from tensorflow.keras.applications.vgg19 import VGG19
    >>> model = VGG19()
    >>> network = Network(model)
    >>> strategy = ParameterizedStrategy(network)
    '''

    super(ParameterizedStrategy, self).__init__(network)

    # Initialize feature vectors for each neuron.
    self.matrix = FeatureMatrix(network)

    # Create variables.
    self.bound = bound
    self.label = None
    self.covered_count = None
    self.objective_covered = None

    # Create a random strategy.
    self.strategy = np.random.uniform(-self.bound, self.bound, size=FeatureMatrix.TOTAL_FEATURES)

  def select(self, k):
    '''Select k neurons with highest scores.
    
    Args:
      k: A positive integer. The number of neurons to select.

    Returns:
      A list of locations of selected neurons.
    '''

    # Calculate scores.
    scores = self.matrix.dot(self.strategy)

    # Get k highest neurons and return their location.
    indices = np.argpartition(scores, -k)[-k:]
    return [self.neurons[i] for i in indices]

  def init(self, covered, label, **kwargs):
    '''Initialize the variables of the strategy.

    This method should be called before all other methods in the class.

    Args:
      covered: A list of coverage vectors that the initial input covers.
      label: A label of the initial input classified into.
      kwargs: Not used. Present for the compatibility with the super class.

    Returns:
      Self for possible call chains.

    Raises:
      ValueError: When the size of the passed coverage vectors are not matches
        to the network setting.
    '''
    
    # Flatten coverage vectors.
    covered = np.concatenate(covered)
    if len(covered) != len(self.neurons):
      raise ValueError('The number of neurons in network does not matches to the setting.')

    # Initialize the number of covering for each neuron.
    self.covered_count = np.zeros_like(covered, dtype=int)
    self.covered_count += covered

    # Set the initial label.
    self.label = label

    # Initialize the coverage vector when objective satisfies.
    self.objective_covered = np.zeros_like(self.covered_count, dtype=bool)

    return self

  def update(self, covered, label, **kwargs):
    '''Update the variable of the strategy.

    Args:
      covered: A list of coverage vectors that a current input covers.
      label: A label of a current input classified into.
      kwargs: Not used. Present for the compatibility with the super class.

    Returns:
      Self for possible call chains.
    '''
    
    # Flatten coverage vectors.
    covered = np.concatenate(covered)

    # Update the number of covering for each neuron.
    self.covered_count += covered

    # If adversarial input found, update the coverage vector for objective satifaction.
    if self.label != label:
      self.objective_covered = np.bitwise_or(self.objective_covered, covered)

    # Update variable vectors
    self.matrix = self.matrix.update(self.covered_count, self.objective_covered)

class AdaptiveParameterizedStrategy(ParameterizedStrategy):
  '''A adaptive and parameterized neuron selection strategy.
  
  Adaptive and parameterized neuron selection strategy is a strategy that changes
  the parameterized neuron selection strategy adaptively with respect to the model,
  data, or even time. These updates are done in online; in other words, the strategies
  are updated while testing. Please see the following paper for detail:

  Effective White-Box Testing for Deep Neural Networks with Adaptive Neuron-Selection Strategy
  http://prl.korea.ac.kr/~pronto/home/papers/issta20.pdf
  '''

  def __init__(self, network, bound=5, size=100, history=300, remainder=0.5, sigma=1):
    '''Create a adaptive parameterized strategy, and initialize its variables.
    
    Args:
      network: A wrapped Keras model with `adapt.Network`.
      bound: A floating point number indicates the absolute value of minimum
        and maximum bounds.
      size: A positive integer. The number of strategies to create at once.
      history: A positive integer. The number of strategies to look for when
        the strategy learn and generate next strategies.
      remainder: A floating point number in [0, 1]. The portion of strategies
        to generate next strategies from.
      sigma: A non-negative floating point number. The standard deviation of
        normal distribution that adds to strategies for diversity.

    Raises:
      ValueError: When arguments are not in proper range.

    Example:

    >>> from adapt import Network
    >>> from adapt.strategy import AdaptiveParameterizedStrategy
    >>> from tensorflow.keras.applications.vgg19 import VGG19
    >>> model = VGG19()
    >>> network = Network(model)
    >>> strategy = AdaptiveParameterizedStrategy(network)
    '''

    super(AdaptiveParameterizedStrategy, self).__init__(network, bound)

    # Initialize variables.
    self.size = size
    self.history = history
    self.remainder = remainder
    self.sigma = sigma

    # Create initial stratagies randomly.
    self.strategies = [np.random.uniform(-self.bound, self.bound, size=FeatureMatrix.TOTAL_FEATURES) for _ in range(self.size)]
    self.strategy = self.strategies.pop(0)

    # Create a coverage vector for a strategy.
    self.strategy_covered = None

    # Storage for used strategies and their result.
    self.records = []

  def init(self, covered, label, **kwargs):
    '''Initialize the variables of the strategy.

    This method should be called before all other methods in the class.

    Args:
      covered: A list of coverage vectors that the initial input covers.
      label: A label of the initial input classified into.
      kwargs: Not used. Present for the compatibility with the super class.

    Returns:
      Self for possible call chains.
    '''

    super(AdaptiveParameterizedStrategy, self).init(covered=covered, label=label, **kwargs)

    # Initialize coverage vector for one strategy.
    self.strategy_covered = np.zeros(len(self.neurons), dtype=bool)

    return self

  def update(self, covered, label, **kwargs):
    '''Update the variable of the strategy.

    Args:
      covered: A list of coverage vectors that a current input covers.
      label: A label of a current input classified into.
      kwargs: Not used. Present for the compatibility with the super class.

    Returns:
      Self for possible call chains.
    '''

    super(AdaptiveParameterizedStrategy, self).update(covered=covered, label=label, **kwargs)

    # Flatten coverage vectors.
    covered = np.concatenate(covered)
    self.strategy_covered = np.bitwise_or(self.strategy_covered, covered)

    return self

  def next(self):
    '''Get the next parameterized strategy.
    
    Get the next parameterized strategy. If the generated parameterized strategies
    are all used, generate new parameterized strategies.

    Returns:
      Self for possible call chains.
    '''

    # Finish current strategy.
    self.records.append((self.strategy, self.strategy_covered))

    # Get the next strategy.
    if len(self.strategies) > 0:
      self.strategy = self.strategies.pop(0)
      self.strategy_covered = np.zeros_like(self.strategy_covered, dtype=bool)
      return self

    # Generate next strategies from the past records.
    records = self.records[-self.history:]

    # Find a set of strategies that maximizes the coverage.
    n = int(self.size * self.remainder)
    strategies, covereds = tuple(zip(*records))
    _, indices = greedy_max_set(covereds, n=n)

    # Find the maximum coverages for remaining part.
    n = n - len(indices)
    if n > 0:
      coverages = list(map(np.mean, covereds))
      indices = indices + list(np.argpartition(coverages, -n)[-n:])
    
    # Get strategies.
    selected = np.array(strategies)[indices]

    # Mix strategies randomly.
    n = len(selected)
    generation = ceil(1 / self.remainder)
    left = selected[np.random.permutation(n)]
    right = selected[np.random.permutation(n)]

    for l, r in zip(left, right):
      for _ in range(generation):

        # Generate new strategy.
        s = np.array([l[i] if np.random.choice([True, False]) else r[i] for i in range(FeatureMatrix.TOTAL_FEATURES)])

        # Add little distortion.
        s = s + np.random.normal(0, self.sigma, size=FeatureMatrix.TOTAL_FEATURES)

        # Clip the ranges.
        s = np.clip(s, -self.bound, self.bound)

        self.strategies.append(s)

    self.strategies = self.strategies[:self.size]

    # Get the next strategy.
    self.strategy = self.strategies.pop(0)

    return self
