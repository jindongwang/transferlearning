import numpy as np

from .strategy import Strategy
from pdb import set_trace as st

class RandomStrategy(Strategy):
  '''A strategy that randomly selects neurons from all neurons.
  
  This strategy selects neurons from a set of all neurons in the network,
  except for the neurons that located in skippable layers.
  '''

  def init(self, covered):
    self.num_input, self.num_neurons = covered.shape
    
    
  def select(self, k, ):
    '''Seleck k neurons randomly.

    Select k neurons randomly from a set of all neurons in the network,
    except for the neurons that located in skippable layers.

    Args:
      k: A positive integer. The number of neurons to select.

    Returns:
      A list of location of the selected neurons.
    '''

    # Choose k neurons and return their location.
    
    indices = [
        np.random.choice(self.num_neurons, size=k, replace=False)
        for _ in range(self.num_input)
    ]
    # indices = np.concatenate(indices, axis=0)
    return indices
