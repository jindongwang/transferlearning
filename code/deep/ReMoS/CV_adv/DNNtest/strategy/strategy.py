from abc import ABC
from abc import abstractmethod

class Strategy(ABC):
  '''Abstract strategy class (used as an implementation base).'''

  def __init__(self, network):
    '''Create a strategy.

    *** This method could be updated, but not mandatory. ***
    Initialize the strategy by collecting all neurons in the network.

    Args:
      network: A wrapped Keras model with `adapt.Network`.
    '''


  def __call__(self, k):
    '''Python magic call method.

    This will make object callable. Just passing the argument to select method.
    '''

    return self.select(k)

  @abstractmethod
  def select(self, k):
    '''Select k neurons.

    *** This method should be implemented. ***
    Select k neurons, and returns their location.
    
    Args:
      k: A positive integer. The number of neurons to select.

    Returns:
      A list of locations of selected neurons.
      Global id
    '''

  def init(self, **kwargs):
    '''Initialize the variables of the strategy.

    *** This method could be update, but not mandatory. ***
    Initialize the variables that managed by the strategy. This should be called
    before other methods of the strategy called.

    Args:
      kwargs: A dictionary of keyword arguments. The followings are privileged
        arguments.
      covered: A list of coverage vectors that the initial input covers.
      label: A label that initial input classified into.

    Returns:
      Self for possible call chains.
    '''

    return self

  def update(self, covered, **kwargs):
    '''Update the variables of the strategy.

    *** This method could be updated, but not mandatory. ***
    Update the variables that managed by the strategy. This method is called
    everytime after a new input is created. By default, not update anything.

    Args:
      kwargs: A dictionary of keyword arguments. The followings are privileged
        arguments.
      covered: A list of coverage vectors that a current input covers.
      label: A label that a current input classified into.

    Returns:
      Self for possible call chains.
    '''

    return self

  def next(self):
    '''Move to the next strategy.

    *** This method could be updated, but not mandatory. ***
    Update the strategy itself to next strategy. This may be important for
    strategies using multiple strategies (i.e. round-robin). Be default,
    not update strategy.
    '''

    return self
