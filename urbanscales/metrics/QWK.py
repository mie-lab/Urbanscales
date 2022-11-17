import numpy as np
from sklearn.metrics import cohen_kappa_score
from smartprint import smartprint as sprint


class QWK:
    def __init__(self, actual_y, predicted_y, n_classes=10):
        """
        List of predictions; default n_classes=10
        Can be JF/CI or class labels if predicting bin number
        The metric converts the regression values into 10 bins to compute QWK
        Order doesn't matter since
        Args:
            actual_y:
            predicted_y:
        """
        bins = np.arange(0, 10, 10 / n_classes)
        actual_y, _ = np.histogram(actual_y, bins=bins)
        predicted_y, _ = np.histogram(predicted_y, bins=bins)
        sprint(actual_y)
        sprint(predicted_y)
        self.val = cohen_kappa_score(actual_y, predicted_y, weights="quadratic")


def custom_scoring_QWK(reg, x, y):
    y_hat = reg.predict(x)
    return QWK(y, y_hat).val


if __name__ == "__main__":
    a = (np.random.rand(100, 1) * 10).flatten().tolist()
    b = (np.random.rand(100, 1) * 10 + 3).flatten().tolist()
    sprint(QWK(a, b).val)
    sprint(QWK(b, a).val)

    sprint(QWK(a, a).val)
    sprint(QWK(b, b).val)

    assert QWK(b, a).val == QWK(b, a).val
    assert QWK(a, a).val == 1
