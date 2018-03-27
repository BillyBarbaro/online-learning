import math
import numpy as np
from base_classifier import BaseClassifier

STEP_SIZE = 0.0002


def calculate_update(instance, linear_combination, label):
    if linear_combination * label >= 1:
        return 0
    else:
        return -label * instance


class FOGD(BaseClassifier):
    def __init__(self, gamma, budget=200):
        self.gamma = gamma
        self.budget = budget
        self.sigma = 1.0 / math.sqrt(2 * gamma)
        self.w = np.zeros((budget * 2, 1))
        self.components = None

    def get_param_dict(self):
        return {
            'gamma': self.gamma,
            'budget': self.budget
        }

    def classify(self, instance, label):
        if self.components is None:
            self.components = np.random.normal(scale=self.sigma, size=(self.budget, instance.shape[1]))
        z_instance = self.z(instance)
        linear_combination = np.dot(self.w.T, z_instance)[0][0]
        update = calculate_update(z_instance, linear_combination, label)
        self.w = self.w - STEP_SIZE * update
        return linear_combination

    def z(self, instance):
        product = np.dot(self.components, np.transpose(instance))
        cos_components = np.cos(product)
        sin_components = np.sin(product)
        return np.concatenate((cos_components, sin_components))

    def predict(self, instance):
        if self.components is None:
            self.components = np.random.normal(scale=self.sigma, size=(self.budget, instance.shape[1]))
        z_instance = self.z(instance)
        return np.dot(self.w.T, z_instance)[0][0]

    def __str__(self):
        return 'FOGD_gamma={}_B={}'.format(self.gamma, self.budget)
