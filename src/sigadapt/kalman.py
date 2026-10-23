"""Kalman filter for a linear regression with drifting coefficients.

State-space model:  theta_t = theta_{t-1} + eta_t,   eta_t ~ N(0, Q)
                    y_t     = x_t' theta_t + eps_t,  eps_t ~ N(0, R)

The variances (P_1, Q, R) are estimated beforehand with the iterative grid search
of the viking_kalman package (de Vilmarest, 2022).
"""

import numpy as np


class KalmanRegression:
    def __init__(self, n_features, Q=1e-5, R=1e-2, beta_init=None, P_init=None):
        """
        n_features : dimension of the state
        Q          : state noise covariance; diagonal matrix
        R          : observation noise variance
        beta_init  : initial state (default: zero)
        P_init     : initial state covariance (default: identity)
        """
        self.n_features = n_features
        self.w = np.array(beta_init, dtype=float) if beta_init is not None else np.zeros(n_features)
        self.P = np.array(P_init, dtype=float) if P_init is not None else np.eye(n_features) * 1.0
        self.Q = Q * np.eye(n_features) if np.isscalar(Q) else np.array(Q, dtype=float)
        self.R = float(R)

    def update(self, x, y):

        x = np.asarray(x).reshape(-1)

        w_pred = self.w
        P_pred = self.P + self.Q

        Px = P_pred @ x
        S = x.T @ Px + self.R
        K = Px / S

        y_pred = x @ w_pred
        err = y - y_pred

        self.w = w_pred + K * err
        self.P = P_pred - np.outer(K, Px)
        self.y_pred = y_pred

    def predict(self, X):
        return X @ self.w
