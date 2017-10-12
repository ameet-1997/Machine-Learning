from __future__ import print_function
import warnings
import numpy as np
from sklearn.utils import deprecated
from scipy import linalg
from sklearn.externals.six import string_types
from sklearn.externals.six.moves import xrange

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.covariance import ledoit_wolf, empirical_covariance, shrunk_covariance
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import StandardScaler


__all__ = ['LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis']


def _cov(X, shrinkage=None):
    shrinkage = "empirical" if shrinkage is None else shrinkage
    if isinstance(shrinkage, string_types):
        if shrinkage == 'auto':
            sc = StandardScaler()  # standardize features
            X = sc.fit_transform(X)
            s = ledoit_wolf(X)[0]
            # rescale
            s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
        elif shrinkage == 'empirical':
            s = empirical_covariance(X)
        else:
            raise ValueError('unknown shrinkage parameter')
    elif isinstance(shrinkage, float) or isinstance(shrinkage, int):
        if shrinkage < 0 or shrinkage > 1:
            raise ValueError('shrinkage parameter must be between 0 and 1')
        s = shrunk_covariance(empirical_covariance(X), shrinkage)
    else:
        raise TypeError('shrinkage must be of string or int type')
    return s


def _class_means(X, y):
    means = []
    classes = np.unique(y)
    for group in classes:
        Xg = X[y == group, :]
        means.append(Xg.mean(0))
    return np.asarray(means)


def _class_cov(X, y, priors=None, shrinkage=None):
    classes = np.unique(y)
    covs = []
    for group in classes:
        Xg = X[y == group, :]
        covs.append(np.atleast_2d(_cov(Xg, shrinkage)))
    return np.average(covs, axis=0, weights=priors)

class RegularizedDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    def __init__(self, priors=None, reg_param=0., store_covariance=False,
                 tol=1.0e-4, store_covariances=None, alpha=1.0, gamma=1.0):
        self.priors = np.asarray(priors) if priors is not None else None
        self.reg_param = 1 - gamma
        self.store_covariances = True
        self.store_covariance = True
        self.tol = tol
        self.class_covariance_ = None
        self.total_covariance_ = None
        self.alpha = alpha
        self.gamma = gamma

    @property
    @deprecated("Attribute covariances_ was deprecated in version"
                " 0.19 and will be removed in 0.21. Use "
                "covariance_ instead")
    def covariances_(self):
        return self.covariance_

    def fit(self, X, y):
    	# Check if the dimensions are okay
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Get the unique labels
        self.classes_, y = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        # Check the number of classes
        if n_classes < 2:
            raise ValueError('y has less than 2 classes')
        if self.priors is None:
            self.priors_ = np.bincount(y) / float(n_samples)
        else:
            self.priors_ = self.priors

        cov = None
        store_covariance = self.store_covariance or self.store_covariances

        # Store the covariance if flag is true
        if store_covariance:
            cov = []
        means = [] 			# Stores the class means
        scalings = [] 		# The variance in the rotated coordinate system (scaling)
        rotations = []		# Rotation of the gaussian to principal axes

        # For all the given classes
        for ind in xrange(n_classes):
        	# Subset the classes
            Xg = X[y == ind, :]
            # Find the means of the classes
            meang = Xg.mean(0)
            means.append(meang)
            if len(Xg) == 1:
                raise ValueError('y has only 1 sample in class %s, covariance '
                                 'is ill defined.' % str(self.classes_[ind]))
            # Center thr data
            Xgc = Xg - meang
            # Xgc = U * S * V.T
            U, S, Vt = np.linalg.svd(Xgc, full_matrices=False)
            rank = np.sum(S > self.tol)
            if rank < n_features:
                warnings.warn("Variables are collinear")
            S2 = (S ** 2) / (len(Xg) - 1)
            S2 = ((1 - self.reg_param) * S2) + self.reg_param
            if self.store_covariance or store_covariance:
                # cov = V * (S^2 / (n-1)) * V.T
                cov.append(np.dot(S2 * Vt.T, Vt)) 	# .T gives the transpose
            scalings.append(S2)
            rotations.append(Vt.T)

        # Get the pooled covariance matrix estimate
        self.class_covariance_ = _class_cov(X, y)

        # Store the covariance matrices
        if self.store_covariance or store_covariance:
            self.covariance_ = cov

        # Initialize total_covariance_
        self.total_covariance_ = []

        # Change the covariance matrices depending on alpha
        for ind in xrange(n_classes):
        	self.total_covariance_.append(self.alpha*cov[ind] + (1-self.alpha)*self.class_covariance_) 	# New estimate of the covariance matrix

        # Store the other attributes
        self.means_ = np.asarray(means)
        self.scalings_ = scalings
        self.rotations_ = rotations
        return self

    # def _decision_function(self, X):
    #     check_is_fitted(self, 'classes_')

    #     X = check_array(X)
    #     norm2 = []
    #     for i in range(len(self.classes_)):
    #         R = self.rotations_[i]
    #         S = self.scalings_[i]
    #         Xm = X - self.means_[i]
    #         X2 = np.dot(Xm, R * (S ** (-0.5)))
    #         norm2.append(np.sum(X2 ** 2, 1))
    #     norm2 = np.array(norm2).T   # shape = [len(X), n_classes]
    #     u = np.asarray([np.sum(np.log(s)) for s in self.scalings_])
    #     return (-0.5 * (norm2 + u) + np.log(self.priors_))

    # def decision_function(self, X):
    #     dec_func = self._decision_function(X)
    #     # handle special case of two classes
    #     if len(self.classes_) == 2:
    #         return dec_func[:, 1] - dec_func[:, 0]
    #     return dec_func

    # def predict(self, X):
    #     d = self._decision_function(X)
    #     y_pred = self.classes_.take(d.argmax(1))
    #     return y_pred

    def predict_proba(self, X):
        values = self._my_decision_function(X)
        return values
        # likelihood = np.exp(values - values.max(axis=1)[:, np.newaxis])
        # return likelihood / likelihood.sum(axis=1)[:, np.newaxis]

    def predict_log_proba(self, X):
        probas_ = self.predict_proba(X)
        return np.log(probas_)

    def _my_decision_function(self, X):
    	columns =[] 	# Stores the columns of the predictions

    	for ind in xrange(len(self.classes_)):
    		matrix1 = X - self.means_[ind]
    		matrix2 = np.transpose(matrix1)
    		matrix3 = np.diag(np.matmul(-0.5*np.matmul(matrix1, np.linalg.inv(self.total_covariance_[ind])), matrix2)) + np.log(self.priors_[ind]) - 0.5*np.log(abs(np.linalg.det(self.covariance_[ind])))
    		columns.append(matrix3)
    	dec_func = np.transpose(np.vstack(tuple(columns)))
    	return dec_func

    def predict(self, X):
        d = self._my_decision_function(X)
        y_pred = self.classes_.take(d.argmax(1))
        return y_pred
