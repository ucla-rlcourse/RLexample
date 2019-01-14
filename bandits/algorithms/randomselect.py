import random
class RandomSelect():
  def __init__(self):
    return

  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    return

  def select_arm(self):
    return random.randrange(len(self.values))
  
  def update(self, chosen_arm, reward):
    # do nothing
    return