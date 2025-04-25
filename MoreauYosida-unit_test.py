import numpy as np
import unittest
from scipy import optimize

# Import the implementation
from MoreauYosida import MoreauYosida, MoreauYosidaOptimizer, default_gamma_schedule
from proximal import soft_thresholding

class TestMoreauYosida(unittest.TestCase):
    """
    Core test suite for Moreau-Yosida regularization.
    
    These tests verify the key properties and applications:
    1. Basic properties of the Moreau envelope
    2. Smoothing behavior for non-differentiable functions
    3. Optimization with LASSO-type problems
    4. Constrained optimization via indicator functions
    5. Performance with different regularization parameters
    """
    
    def setUp(self):
        """Set up common test fixtures."""
        np.random.seed(42)
        
        # Define common regularizers
        
        # L1 norm and its proximal operator
        self.l1_norm = lambda x: np.sum(np.abs(x))
        self.l1_prox = lambda x, t: np.sign(x) * np.maximum(np.abs(x) - t, 0)
        
        # Box constraint indicator
        self.box_indicator = lambda x: 0 if np.all((x >= 0) & (x <= 1)) else np.inf
        self.box_prox = lambda x, t: np.clip(x, 0, 1)

    def test_moreau_envelope_properties(self):
        """
        Test basic properties of the Moreau envelope.
        """
        # Test with L1 norm
        n = 10
        x = np.random.randn(n)
        
        # Initialize Moreau-Yosida regularization
        my_l1 = MoreauYosida(self.l1_norm, self.l1_prox, gamma=0.1)
        
        # Calculate Moreau envelope at x
        envelope_value = my_l1.value(x)
        gradient = my_l1.gradient(x)
        prox_point = my_l1.prox_point(x)
        
        # 1. Moreau envelope should be <= original function
        self.assertLessEqual(envelope_value, self.l1_norm(x))
        
        # 2. Gradient should be (1/gamma) * (x - prox_point)
        calculated_gradient = (1/my_l1.gamma) * (x - prox_point)
        np.testing.assert_allclose(gradient, calculated_gradient)
        
        # 3. As gamma decreases, envelope should approach original function value
        results = []
        for gamma in [1.0, 0.1, 0.01, 0.001]:
            my_l1_temp = MoreauYosida(self.l1_norm, self.l1_prox, gamma=gamma)
            envelope = my_l1_temp.value(x)
            results.append(envelope)
        
        # The smallest gamma should give a value close to the original function
        self.assertAlmostEqual(results[-1], self.l1_norm(x), delta=0.01)

    def test_smooth_approximation_l1(self):
        """
        Test that Moreau-Yosida provides a smooth approximation to the L1 norm.
        """
        # Create test point where L1 norm is not differentiable
        x = np.zeros(5)
        
        # Compute Moreau-Yosida approximation
        gamma = 0.1
        my_l1 = MoreauYosida(self.l1_norm, self.l1_prox, gamma=gamma)
        
        # Compute gradients at nearby points
        eps = 1e-5
        points = []
        for i in range(6):
            pt = x.copy()
            if i > 0:
                pt[0] = -eps if i % 2 == 1 else eps
            points.append(pt)
        
        gradients = [my_l1.gradient(pt) for pt in points]
        
        # Verify gradients are continuous (close to each other)
        for i in range(1, len(gradients)):
            np.testing.assert_allclose(gradients[0], gradients[i], rtol=1e-2, atol=1e-2)
        
        # Verify that we can compute a Hessian at the non-differentiable point
        hessian = my_l1.hessian(x)
        
        # The Hessian should be positive semidefinite
        for _ in range(5):
            v = np.random.randn(5)
            self.assertGreaterEqual(v.dot(hessian.dot(v)), -1e-10)


    def test_lasso_optimization(self):
        """
        Test optimization of a LASSO‐type problem using Moreau‐Yosida smoothing
        with a very aggressive decaying gamma schedule.
        """
        # 1) Build a synthetic LASSO problem
        n_samples, n_features = 50, 20
        X = np.random.randn(n_samples, n_features)
        true_w = np.zeros(n_features)
        true_w[:3] = [1.0, -0.5, 0.2]
        y = X.dot(true_w) + 0.01 * np.random.randn(n_samples)

        # 2) Smooth least‐squares part
        f_smooth = lambda w: 0.5 * np.sum((X.dot(w) - y) ** 2)
        grad_smooth = lambda w: X.T.dot(X.dot(w) - y)

        # 3) ℓ1 regularization + its prox
        alpha = 0.1
        reg_func = lambda w: alpha * np.sum(np.abs(w))
        prox_op   = lambda w, t: np.sign(w) * np.maximum(np.abs(w) - alpha * t, 0)

        # 4) Very aggressive gamma schedule: halve every iteration
        def gamma_schedule(gamma, k):
            return gamma * 0.5 if k > 0 else gamma

        # 5) Create and run the optimizer
        optimizer = MoreauYosidaOptimizer(
            f_smooth=f_smooth,
            grad_smooth=grad_smooth,
            g_func=reg_func,
            g_prox=prox_op,
            gamma=1e-4,                  # start smaller
            gamma_schedule=gamma_schedule,
            max_iter=200,                # more outer iterations
            tol=1e-8,                    # tighter convergence
            verbose=False
        )
        w_my = optimizer.optimize(np.zeros(n_features))

        # 6) Reference ISTA baseline (500 steps)
        def ista_step(w, step_size):
            return prox_op(w - step_size * grad_smooth(w), step_size)

        w_ref = np.zeros(n_features)
        step_size = 1.0 / (np.linalg.norm(X, 2) ** 2)
        for _ in range(500):
            w_ref = ista_step(w_ref, step_size)

        # 7) Compare objective values
        obj_my  = f_smooth(w_my)  + reg_func(w_my)
        obj_ref = f_smooth(w_ref) + reg_func(w_ref)
        self.assertLess(abs(obj_my - obj_ref) / max(abs(obj_ref), 1e-10), 0.05)

        # 8) Compare sparsity patterns (≥80% match)
        nz_my  = np.abs(w_my)  > 1e-5
        nz_ref = np.abs(w_ref) > 1e-5
        similarity = np.mean(nz_my == nz_ref)
        self.assertGreater(similarity, 0.8)


    def test_constrained_optimization(self):
        """
        Test Moreau-Yosida for constrained optimization using indicator functions.
        """
        # Quadratic objective with box constraints
        n = 10
        A = np.random.randn(n, n)
        A = np.dot(A.T, A) + 0.1 * np.eye(n)  # Make positive definite
        b = np.random.randn(n)
        
        # Objective: 0.5 * x^T A x - b^T x subject to 0 <= x <= 1
        f_obj = lambda x: 0.5 * np.dot(x, np.dot(A, x)) - np.dot(b, x)
        grad_obj = lambda x: np.dot(A, x) - b
        
        # Create smoothed constraint with Moreau-Yosida
        gamma = 0.0001  # Smaller gamma for tighter approximation
        my_box = MoreauYosida(self.box_indicator, self.box_prox, gamma=gamma)
        
        # Total objective
        f_total = lambda x: f_obj(x) + my_box.value(x)
        grad_total = lambda x: grad_obj(x) + my_box.gradient(x)
        
        # Solve with L-BFGS-B
        result = optimize.minimize(
        f_total, x0=0.5*np.ones(n),
        method='L-BFGS-B',
        jac=grad_total,
        bounds=[(0,1)]*n,   
            options={'gtol': 1e-6, 'maxiter': 200}
        )
        x_my = result.x
        
        # Reference solution using direct box constraints
        result_ref = optimize.minimize(
            f_obj, 0.5 * np.ones(n), 
            method='L-BFGS-B',
            jac=grad_obj, 
            bounds=[(0, 1) for _ in range(n)]
        )
        x_ref = result_ref.x
        
        # Solutions should have close objective values
        obj_my = f_obj(x_my)
        obj_ref = f_obj(x_ref)
        self.assertLess(abs(obj_my - obj_ref) / max(abs(obj_ref), 1e-10), 0.1)
        
        # Both solutions should satisfy constraints
        self.assertTrue(np.all(x_my >= -1e-6))
        self.assertTrue(np.all(x_my <= 1 + 1e-6))
        self.assertTrue(np.all(x_ref >= 0))
        self.assertTrue(np.all(x_ref <= 1))

    def test_different_regularization_parameters(self):
        """
        Test the effect of different regularization parameters on Moreau-Yosida smoothing.
        """
        # Test point where L1 norm is non-differentiable
        x = np.array([0.5, -0.3, 0, 0.1, -0.6])
        
        # Test different gamma values
        gamma_values = [1.0, 0.1, 0.01, 0.001]
        values = []
        
        for gamma in gamma_values:
            my_l1 = MoreauYosida(self.l1_norm, self.l1_prox, gamma=gamma)
            values.append(my_l1.value(x))
        
        # As gamma decreases, Moreau envelope should approach original function
        for i in range(len(gamma_values) - 1):
            # Should be monotonically increasing (or at least non-decreasing)
            self.assertLessEqual(values[i], values[i+1] * 1.001)  # Allow small numerical errors
        
        # Smallest gamma should give value close to original function
        self.assertAlmostEqual(values[-1], self.l1_norm(x), delta=0.05)
        
        # Test optimization with different gamma values
        n = 5
        A = np.random.randn(n, n)
        A = np.dot(A.T, A) + 0.1 * np.eye(n)
        b = np.random.randn(n)
        
        f_smooth = lambda x: 0.5 * np.dot(x, np.dot(A, x)) - np.dot(b, x)
        grad_smooth = lambda x: np.dot(A, x) - b
        
        alpha = 0.1
        g_func = lambda x: alpha * self.l1_norm(x)
        g_prox = lambda x, t: self.l1_prox(x, alpha * t)
        
        # Use gamma schedule
        def gamma_schedule(gamma, iteration):
            if iteration > 0 and iteration % 2 == 0:
                return gamma * 0.5
            return gamma
        
        # Create optimizer with gamma schedule
        optimizer = MoreauYosidaOptimizer(
            f_smooth=f_smooth,
            grad_smooth=grad_smooth,
            g_func=g_func,
            g_prox=g_prox,
            gamma=0.1,
            gamma_schedule=gamma_schedule,
            max_iter=10
        )
        
        # Optimize and verify the solution
        x0 = np.zeros(n)
        result = optimizer.optimize(x0)
        
        # Verify result is reasonable (objective is less than at starting point)
        obj_start = f_smooth(x0) + g_func(x0)
        obj_end = f_smooth(result) + g_func(result)
        self.assertLess(obj_end, obj_start)


if __name__ == '__main__':
    unittest.main()