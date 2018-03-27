import numpy as np
from numpy.linalg import norm
from base_classifier import BaseClassifier

LOSS = {"hinge": 1, "l1": 2, "l2": 3, "logit": 4}

# Adapted from https://github.com/tund/dsgd

class DSGD(BaseClassifier):
    def __init__(self, gamma, lbd, B=100, k=20, D=200):
        self.gamma = gamma
        self.lbd = lbd
        self.max_size = B
        self.k = k
        self.D = D
        self.t = 0
        self.size_ = 0

        self.stored_map = np.zeros([self.max_size, 2 * self.D])

        self.w_ = np.zeros([self.max_size, 1])
        self.rw_ = np.zeros([2 * self.D, 1])

        # initialize mapping matrix for random features
        self.u_ = None
        self.stored_examples = None

    def get_wx(self, x, rx):
        if self.size_ == 0:
            return [0]
        else:
            xx = (self.stored_examples[:self.size_] - x)
            return np.sum(self.w_[:self.size_] * np.exp(-self.gamma * (xx * xx).sum(axis=1, keepdims=True)), axis=0) + rx.dot(self.rw_)

    def get_grad(self, x, rx, y, wx=None):
        wx = self.get_wx(x, rx)[0] if wx is None else wx[0]
        return (-y, -1) if y * wx <= 1 else (0, -1)

    def add_to_core_set(self, instance, rX, w):
        self.stored_examples[self.size_] = instance
        self.stored_map[self.size_] = rX
        self.w_[self.size_] = w
        self.size_ += 1

    def remove(self, idx):
        n = len(idx)
        mask = np.ones(self.max_size, np.bool)
        mask[idx] = False
        self.w_[:-n] = self.w_[mask]
        self.stored_map[:-n] = self.stored_map[mask]
        self.stored_examples[:-n] = self.stored_examples[mask]
        self.size_ -= n

    def maintain_budget(self):
        i = np.argsort(norm(self.w_[:self.size_], axis=1))
        self.rw_ += self.stored_map[i[:self.k]].T.dot(self.w_[i[:self.k]])
        self.remove(i[:self.k])

    def classify(self, instance, label):
        if self.u_ is None:
            self.u_ = (2 * self.gamma) * np.random.randn(instance.shape[1], self.D)
            self.stored_examples = np.zeros([self.max_size, instance.shape[1]])
        instance = instance.reshape(1, -1)
        rX = np.zeros([instance.shape[0], 2 * self.D])
        rX[:, :self.D] = np.cos(instance.dot(self.u_)) / np.sqrt(self.D)
        rX[:, self.D:] = np.sin(instance.dot(self.u_)) / np.sqrt(self.D)

        wx = self.get_wx(instance[0], rX[0])
        alpha_t, z = self.get_grad(instance[0], rX[0], label, wx=wx)  # compute \alpha_t

        self.w_ *= (1.0 * self.t) / (self.t + 1)
        self.rw_ *= (1.0 * self.t) / (self.t + 1)

        w = -alpha_t / (self.lbd * (self.t + 1))

        if self.size_ == self.max_size:
            self.maintain_budget()
        self.add_to_core_set(instance, rX[0], w)
        self.t += 1
        return wx[0]

    def predict(self, instance):
        if self.u_ is None:
            self.u_ = (2 * self.gamma) * np.random.randn(instance.shape[1], self.D)
            self.stored_examples = np.zeros([self.max_size, instance.shape[1]])
        instance = instance.reshape(1, -1)
        rX = np.zeros([instance.shape[0], 2 * self.D])
        rX[:, :self.D] = np.cos(instance.dot(self.u_)) / np.sqrt(self.D)
        rX[:, self.D:] = np.sin(instance.dot(self.u_)) / np.sqrt(self.D)
        wx = self.get_wx(instance[0], rX[0])
        return wx

    def get_param_dict(self):
        return {
            'gamma': self.gamma,
            'lbd': self.lbd,
            'B': self.max_size,
            'D': self.D,
            'k': self.k,
        }

    def __str__(self):
        return 'DSGD_gamma={}_lbd={}_B={}_D={}_k={}'.format(self.gamma, self.lbd, self.max_size, self.D, self.k)
