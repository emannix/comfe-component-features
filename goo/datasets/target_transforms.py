from pdb import set_trace as pb
import numpy as np

class SimpleThreshold:
	def __init__(self, threshold, ge=True):
		self.threshold = threshold
		self.ge = ge

	def __call__(self, targets):
		targets_nan = np.isnan(targets)
		if self.ge:
			targets = targets >= self.threshold
		else:
			targets = targets < self.threshold
		targets[targets_nan] = np.NAN
		return targets

class SimpleBinary:
	def __init__(self, positive_classes):
		self.positive_classes = positive_classes

	def __call__(self, targets):
		targets_nan = np.isnan(targets)
		targets = np.array([x in self.positive_classes for x in targets]).astype(float)
		targets[targets_nan] = np.NAN
		return targets

class SimpleReCategorise:
	def __init__(self, categories):
		self.categories = categories

	def __call__(self, targets):
		new_targets = -np.ones_like(targets)
		for i in range(len(self.categories)):
			matches = np.isin(targets, self.categories[i])
			new_targets[matches] = i
			
		if -1 in new_targets:
			# targets[new_targets == -1]
			pb()
		return new_targets

class SimpleInstanceAll:
	def __init__(self):
		pass

	def __call__(self, targets):
		new_targets = np.arange(len(targets))
		return new_targets

