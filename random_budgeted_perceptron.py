import numpy as np
from sklearn import metrics

from random import randint
from base_classifier import BaseClassifier


def _calculate_loss(label, linear_combination):
    return 0 if label == (-1 if linear_combination < 0 else 1) else 1



class RBP(BaseClassifier):
    def __init__(self, gamma, budget=200):
        self.alpha = None
        self.support_vectors = None
        self.gamma = gamma
        self.budget = budget

    def get_param_dict(self):
        return {
            'gamma': self.gamma,
            'budget': self.budget
        }

    def classify(self, instance, label):
        if self.support_vectors is None:
            linear_combination = 0
        else:
            kernel = np.exp(-self.gamma * (np.linalg.norm(self.support_vectors - instance, axis=1) ** 2))
            linear_combination = np.sum(kernel * self.alpha)

        loss = _calculate_loss(label, linear_combination)
        self._calculate_update(loss, instance, label)
        return linear_combination

    def predict(self, instance):
        return 0 if self.support_vectors is None else \
            np.sum(np.exp(-self.gamma * (np.linalg.norm(self.support_vectors - instance, axis=1) ** 2)) * self.alpha)

    def predict_instance(self, instance):
        if self.support_vectors is None:
            linear_combination = 0
        else:
            kernel = metrics.pairwise.rbf_kernel(self.support_vectors, instance, gamma=self.gamma)
            linear_combination = np.sum(np.transpose(kernel) * self.alpha)

        return linear_combination

    def _calculate_update(self, loss, example, label):
        if not loss == 0:
            if self.alpha is None:
                self.alpha = np.array([[label]])
                self.support_vectors = example
            elif self.alpha.shape[1] < self.budget:
                self.alpha = np.hstack([self.alpha, [[label]]])
                self.support_vectors = np.vstack([self.support_vectors, example])
            else:
                to_replace = randint(0, self.budget - 1)
                self.alpha[0][to_replace] = label
                self.support_vectors[to_replace] = example

    def __str__(self):
        return 'RBP_gamma={}_budget={}'.format(self.gamma, self.budget)
