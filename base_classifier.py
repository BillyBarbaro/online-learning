class BaseClassifier(object):
    def classify(self, instance, label):
        raise NotImplementedError("Classifiers should have a classify method with these arguments")

    def predict(self, instance):
        raise NotImplementedError(
            "Classifiers should have a predict method with these arguments that does not update the classifier if a mistake is made")

    def get_param_dict(self):
        raise NotImplementedError("Classifiers should return a dictionary mapping their parameters to this values")

    def __str__(self):
        raise NotImplementedError("Classifiers should define a string formed <name>_<param>=<val>_<param>=<val>...")