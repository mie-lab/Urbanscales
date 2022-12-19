import numpy as np
from sklearn.metrics import cohen_kappa_score
from smartprint import smartprint as sprint

class QWK:
    def __init__(self, actual_y, predicted_y, n_classes=10):
        """
        List of predictions; default n_classes=10
        Can be JF/CI or class labels if predicting bin number
        The metric converts the regression values into 10 labels to compute QWK
        Args:
            actual_y:
            predicted_y:
        """
        actual_y = np.array(np.array(actual_y).reshape(-1, 1), "int")
        predicted_y = np.array(np.array(predicted_y).reshape(-1, 1), "int")

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
