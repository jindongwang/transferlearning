import numpy as np
from pdb import set_trace as st

from .strategy import Strategy

class UncoveredRandomStrategy(Strategy):
  '''A strategy that randomly selects neurons from uncovered neurons.
  
  This strategy selects neurons from a set of uncovered neurons. This strategy
  is first introduced in the following paper, but not exactly same. Please see
  the following paper for more details:

  DeepXplore: Automated Whitebox Testing of Deep Learning Systems
  https://arxiv.org/abs/1705.06640
  '''

  def __init__(self, network):
    '''Create a strategy and initialize its variables.
    
    Args:
      network: A wrapped Keras model with `adapt.Network`.

    Example:

    >>> from adapt import Network
    >>> from adapt.strategy import UncoveredRandomStrategy
    >>> from tensorflow.keras.applications.vgg19 import VGG19
    >>> model = VGG19()
    >>> network = Network(model)
    >>> strategy = UncoveredRandomStrategy(network)
    '''

    super(UncoveredRandomStrategy, self).__init__(network)

    # A variable that keeps track of the covered neurons.
    self.covered = None

  def select(self, k):
    '''Select k uncovered neurons.
    
    Select k neurons, and returns their location.

    Args:
      k: A positive integer. The number of neurons to select.

    Returns:
      A list of locations of selected neurons.
    '''

    # Find a set of uncovered neurons.
    candidates = [
        np.squeeze(np.argwhere(self.covered[id] == 0))
        for id in range(self.num_input)
    ]
    # candidates = np.squeeze(np.argwhere(self.covered == 0))
    
    selected_indices = []
    for id in range(self.num_input):
        candidates = np.squeeze(np.argwhere(self.covered[id] == 0))
        k = min(k, len(candidates))
        indices = np.random.choice(candidates, size=k, replace=False)
        selected_indices.append(indices)

    return selected_indices

  def init(self, covered, **kwargs):
    '''Initialize the variable of the strategy.

    This method should be called before all other methods in the class.

    Args:
      covered: A list of coverage vectors that the initial input covers.
      kwargs: Not used. Present for the compatibility with the super class.

    Returns:
      Self for possible call chains.

    Raises:
      ValueError: When the size of the passed coverage vectors are not matches
        to the network setting.
    '''

    # Flatten coverage vectors.
    self.covered = covered
    self.num_input, self.num_neurons = covered.shape

    return self

  def update(self, covered, **kwargs):
    '''Update the variable of the strategy.

    Args:
      covered: A list of coverage vectors that a current input covers.
      kwargs: Not used. Present for the compatibility with the super class.

    Returns:
      Self for possible call chains.
    '''


    # Update coverage vectors.
    self.covered = np.bitwise_or(self.covered, covered)

    return self
