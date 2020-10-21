from sklearn import metrics
import numpy as np


class RegressionMetrics:
    def __init__(self):
        self.metrics = {
            'mea': self._mae,
            'mse': self._mse,
            'rmse': self._rmse,
            'msle': self._msle,
            'r2': self._r2,
        }

    def __call__(self, metric, y_true, y_pred):
        if metric not in self.metrics:
            raise Exception('Metric not implemented')
        else:
            return self.metrics[metric](y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _mae(y_true, y_pred):
        return metrics.mean_absolute_error(y_true, y_pred)

    @staticmethod
    def _mse(y_true, y_pred):
        return metrics.mean_squared_error(y_true, y_pred)

    def _rmse(self, y_true, y_pred):
        return np.sqrt(self._mse(y_true, y_pred))

    @staticmethod
    def _msle(y_true, y_pred):
        return metrics.mean_squared_log_error(y_true, y_pred)

    def _rmsle(self, y_true, y_pred):
        return np.sqrt(self._msle(y_true, y_pred))

    @staticmethod
    def _r2(y_true, y_pred):
        return metrics.r2_score(y_true, y_pred)


class ClassificationMetrics:
    def __init__(self):
        self.metrics = {
            'accuracy': self._accuracy,
            'f1': self._f1,
            'precision': self._precision,
            'recall': self._auc,
            'logloss': self._logloss,
        }

    def __call__(self, metric, y_true, y_pred, y_proba=None):
        if metric not in self.metrics:
            raise Exception('Metric not implemented')
        if metric == 'auc':
            if y_proba is not None:
                return self._auc(y_true=y_true, y_pred=y_proba)
            else:
                raise Exception('y_proba cannot be None for AUC')
        elif metric == 'logloss':
            if y_proba is not None:
                return self._logloss(y_true=y_true, y_pred=y_proba)
            else:
                raise Exception('y_proba connat be None for logloss')
        else:
            return self.metrics[metric](y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _auc(y_true, y_pred):
        return metrics.roc_auc_score(y_true=y_true, y_score=y_pred)

    @staticmethod
    def _accuracy(y_true, y_pred):
        return metrics.accuracy_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _f1(y_true, y_pred):
        return metrics.f1_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _recall(y_true, y_pred):
        return metrics.recall_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _precision(y_true, y_pred):
        return metrics.precision_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _logloss(y_true, y_pred):
        return metrics.log_loss(y_true=y_true, y_pred=y_pred)
