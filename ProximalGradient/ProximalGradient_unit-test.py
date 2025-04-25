import numpy as np
import unittest
from proximal import ProximalGradient, soft_thresholding, proximal_elastic_net

class TestProximalGradient(unittest.TestCase):
    """
    Core test suite for the ProximalGradient optimizer.

    These five tests cover:
    1. Unconstrained quadratic convergence
    2. Correctness of the L1 proximal operator (soft thresholding)
    3. Convergence on a small LASSO problem
    4. Benefit of acceleration on objective value
    5. Elastic net composite regularization
    """
    
    def setUp(self):
        np.random.seed(0)

    def test_quadratic_unconstrained(self):
        """
        f(x) = 0.5 * ||x - b||^2, proximal = identity -> solution x = b
        """
        n = 5
        b = np.random.randn(n)
        f = lambda x: 0.5 * np.sum((x - b) ** 2)
        grad = lambda x: x - b
        prox = lambda x, t: x

        pg = ProximalGradient(
            f_smooth=f,
            grad_smooth=grad,
            proximal_op=prox,
            step_size=0.5,
            max_iter=100,
            tol=1e-8
        )
        x_opt, iters = pg.optimize(np.zeros(n), return_iterations=True)

        # Should recover b exactly
        np.testing.assert_allclose(x_opt, b, rtol=1e-6, atol=1e-6)
        self.assertLess(iters, 100)

    def test_l1_proximal_operator(self):
        """
        soft_thresholding should shrink values by threshold
        """
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        thr = 1.0
        expected = np.array([-1.0,  0.0, 0.0, 0.0, 1.0])
        result = soft_thresholding(x, thr)
        np.testing.assert_allclose(result, expected)

    def test_lasso_convergence(self):
        """
        Small LASSO: solve with proximal gradient and check objective reduction
        """
        X = np.random.randn(30, 10)
        w_true = np.zeros(10)
        w_true[:2] = [1.0, -0.5]
        y = X.dot(w_true)

        f = lambda w: 0.5 * np.sum((X.dot(w) - y) ** 2)
        grad = lambda w: X.T.dot(X.dot(w) - y)
        alpha = 0.1
        prox_op = lambda w, t: np.sign(w) * np.maximum(np.abs(w) - alpha * t, 0)
        step = 1.0 / (np.linalg.norm(X, 2) ** 2)

        pg = ProximalGradient(
            f_smooth=f,
            grad_smooth=grad,
            proximal_op=prox_op,
            step_size=step,
            max_iter=200,
            tol=1e-6
        )
        w_est = pg.optimize(np.zeros(10))

        obj = f(w_est) + alpha * np.sum(np.abs(w_est))
        obj0 = f(np.zeros(10)) + alpha * np.sum(np.zeros(10))
        # Expect objective reduced to 10% of initial
        self.assertLess(obj, obj0 * 0.1)

    def test_acceleration(self):
        """
        Ensure accelerated version achieves no worse final objective than standard
        """
        X = np.random.randn(30, 10)
        y = X.dot(np.random.randn(10))

        f = lambda w: 0.5 * np.sum((X.dot(w) - y) ** 2)
        grad = lambda w: X.T.dot(X.dot(w) - y)
        prox = lambda w, t: w
        step = 1.0 / (np.linalg.norm(X, 2) ** 2)
        max_iter = 100

        # Standard proximal gradient
        pg_std = ProximalGradient(
            f_smooth=f,
            grad_smooth=grad,
            proximal_op=prox,
            step_size=step,
            max_iter=max_iter,
            tol=1e-12,
            accelerated=False
        )
        w_std = pg_std.optimize(np.zeros(10))

        # Accelerated (FISTA)
        pg_acc = ProximalGradient(
            f_smooth=f,
            grad_smooth=grad,
            proximal_op=prox,
            step_size=step,
            max_iter=max_iter,
            tol=1e-12,
            accelerated=True
        )
        w_acc = pg_acc.optimize(np.zeros(10))

        obj_std = f(w_std)
        obj_acc = f(w_acc)
        # Accelerated should achieve at most equal objective
        self.assertLessEqual(obj_acc, obj_std + 1e-8)

    def test_elastic_net(self):
        """
        Composite regularization: Elastic Net (L1 + L2)
        """
        X = np.random.randn(50, 20)
        w0 = np.zeros(20)
        w0[:3] = [1.0, -1.0, 2.0]
        y = X.dot(w0) + 0.01 * np.random.randn(50)

        f = lambda w: 0.5 * np.sum((X.dot(w) - y) ** 2)
        grad = lambda w: X.T.dot(X.dot(w) - y)
        alpha1, alpha2 = 0.1, 0.05
        prox_enet = lambda w, t: proximal_elastic_net(w, t, alpha1, alpha2)
        step = 1.0 / (np.linalg.norm(X, 2) ** 2)

        pg = ProximalGradient(
            f_smooth=f,
            grad_smooth=grad,
            proximal_op=prox_enet,
            step_size=step,
            max_iter=500,
            tol=1e-6
        )
        w_est = pg.optimize(np.zeros(20))

        # Check that objective improved over zero-init
        obj0 = f(np.zeros(20)) + alpha1 * np.sum(np.abs(np.zeros(20))) + alpha2 * np.sum(np.zeros(20) ** 2)
        obj1 = f(w_est) + alpha1 * np.sum(np.abs(w_est)) + alpha2 * np.sum(w_est ** 2)
        self.assertLess(obj1, obj0)

if __name__ == '__main__':
    unittest.main()
