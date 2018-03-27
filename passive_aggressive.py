import numpy as np
import sklearn.metrics

from base_classifier import BaseClassifier


def _calculate_loss(label, linear_combination):
    diff = 1 - label * linear_combination
    return max(0, diff)


class AbstractPAClassifier(BaseClassifier):
    def __init__(self, c=0):
        self.c = c
        self.w = None

    def get_param_dict(self):
        return {
            'c': self.c,
        }

    def classify(self, instance, label):
        if self.w is None:
            linear_combination = 0
            self.w = np.zeros((instance.shape[1], 1))
        else:
            linear_combination = np.dot(instance, self.w)[0][0]
        loss = _calculate_loss(label, linear_combination)
        update = self._calculate_update(loss, instance)
        self._update_classifier(update, label, instance)
        return linear_combination

    def predict(self, instance):
        return 0 if self.w is None else np.dot(instance, self.w)[0][0]

    def _calculate_update(self, loss, example, label=None):
        raise NotImplementedError('Implement this method in a subclasses of AbstractPAClassifier')

    def _update_classifier(self, update, label, example):
        self.w = self.w + (update * label * example).T

    def __str__(self):
        raise NotImplementedError("Subclasses should implement this")


class PAClassifier(AbstractPAClassifier):
    def _calculate_update(self, loss, example, label=None):
        return loss / (np.linalg.norm(example) ** 2)

    def __str__(self):
        return 'PA'


class PAIClassifier(AbstractPAClassifier):
    def _calculate_update(self, loss, example, label=None):
        return min(self.c, loss / (np.linalg.norm(example) ** 2))

    def __str__(self):
        return 'PAI_C={}'.format(self.c)


class PAIIClassifier(AbstractPAClassifier):
    def _calculate_update(self, loss, example, label=None):
        return loss / (np.linalg.norm(example) ** 2 + (1 / (2 * self.c)))

    def __str__(self):
        return 'PAII_C={}'.format(self.c)


# TODO: Revisit if planning on using.  Left here for example.
class KernelizedPAClassifier(AbstractPAClassifier):
    def __init__(self, num_features, gamma, C):
        super(KernelizedPAClassifier, self).__init__(num_features, C)
        self.alpha = None
        self.support_vectors = None
        self.gamma = gamma

    def classify(self, instance, label):
        instance = np.array(instance)
        if label == 0:
            label = -1
        instance = np.append(instance, np.ones((1, 1))).reshape(1, -1)

        if self.support_vectors is None:
            linear_combination = 0
        else:
            kernel = sklearn.metrics.pairwise.rbf_kernel(self.support_vectors, instance, gamma=self.gamma)
            linear_combination = np.sum(np.transpose(kernel) * self.alpha)

        loss = _calculate_loss(label, linear_combination)
        self._calculate_update(loss, instance, label)
        return linear_combination

    def predict_instance(self, instance):
        instance = np.array(instance)
        instance = np.append(instance, np.ones((1, 1))).reshape(1, -1)

        if self.support_vectors is None:
            linear_combination = 0
        else:
            kernel = sklearn.metrics.pairwise.rbf_kernel(self.support_vectors, instance, gamma=self.gamma)
            linear_combination = np.sum(np.transpose(kernel) * self.alpha)

        return linear_combination

    def _calculate_update(self, loss, example, label=None):
        if not loss == 0:
            new_alpha = label * min(self.c, loss / (sklearn.metrics.pairwise.rbf_kernel(example, example)[0][0]))
            if self.alpha is None:
                self.alpha = np.array([new_alpha])
                self.support_vectors = example
            else:
                self.alpha = np.hstack([self.alpha, new_alpha])
                self.support_vectors = np.vstack([self.support_vectors, example])

    def __str__(self):
        return 'KPA_C={}_gamma={}'.format(self.c, self.gamma)
