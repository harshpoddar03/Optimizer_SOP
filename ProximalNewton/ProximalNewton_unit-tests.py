import numpy as np
import unittest
from scipy import optimize
from scipy.sparse import csr_matrix, rand as sprand

# Import the implementation
# Assuming the implementation is in a file named 'ProximalNewton.py'
from ProximalNewton import ProximalNewton

class TestProximalNewton(unittest.TestCase):
    """
    Test suite for Proximal Newton method optimization algorithm.
    
    These tests verify:
    1. Quadratic convergence rate for smooth problems
    2. Correct handling of L1 regularization in logistic regression
    3. Proper functioning of different Hessian approximations
    4. Performance on non-convex problems
    5. Performance on constrained optimization
    """
    
    def setUp(self):
        """Set up common test fixtures."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Define common proximal operators
        self.proximal_l1 = lambda x, t: np.sign(x) * np.maximum(np.abs(x) - t, 0)
        self.proximal_box = lambda x, t, lower=0, upper=1: np.clip(x, lower, upper)
    
    def test_quadratic_convergence(self):
        """
        Test that Proximal Newton achieves quadratic convergence rate for smooth problems.
        
        We test this by solving a simple quadratic problem where the Newton method
        should converge very rapidly.
        """
        # Simple quadratic problem
        n = 10
        A = np.random.randn(n, n)
        A = np.dot(A.T, A) + 0.1 * np.eye(n)  # Make positive definite
        b = np.random.randn(n)
        
        # Function and derivatives
        f = lambda x: 0.5 * np.dot(x, np.dot(A, x)) - np.dot(b, x)
        grad = lambda x: np.dot(A, x) - b
        hess = lambda x: A
        
        # Exact solution is A^(-1) * b
        x_opt = np.linalg.solve(A, b)
        
        # No regularization for this test (identity proximal operator)
        identity_prox = lambda x, t: x
        
        # Create proximal Newton solver
        pn = ProximalNewton(f, grad, hess, identity_prox, max_iter=10, tol=1e-10, verbose=False)
        
        # Optimize
        x_sol = pn.optimize(np.zeros(n))
        
        # Check that solution is close to optimal
        rel_error = np.linalg.norm(x_sol - x_opt) / max(np.linalg.norm(x_opt), 1e-10)
        self.assertLess(rel_error, 1e-5)
        
        # Check objective value
        f_sol = f(x_sol)
        f_opt = f(x_opt)
        rel_obj_diff = abs(f_sol - f_opt) / max(abs(f_opt), 1e-10)
        self.assertLess(rel_obj_diff, 1e-5)

    def test_l1_logistic_regression(self):
        """
        Test Proximal Newton for L1-regularized logistic regression.
        
        This test verifies that the Proximal Newton method correctly handles
        non-smooth L1 regularization in the context of logistic regression.
        """
        # Create synthetic classification data
        n_samples, n_features = 100, 20
        X = np.random.randn(n_samples, n_features)
        true_w = np.zeros(n_features)
        true_w[:5] = np.random.randn(5)  # Sparse ground truth
        z = np.dot(X, true_w)
        p = 1.0 / (1.0 + np.exp(-z))
        y = (np.random.rand(n_samples) < p).astype(np.float64)
        
        # Logistic regression loss and derivatives
        def f_logreg(w):
            z = np.dot(X, w)
            # Stabilized computation to avoid overflow
            pos = np.maximum(0, z)
            neg = np.minimum(0, z)
            loss = np.sum(neg + np.log1p(np.exp(pos - neg))) - np.sum(y * z)
            return loss / n_samples
        
        def grad_logreg(w):
            z = np.dot(X, w)
            p = 1.0 / (1.0 + np.exp(-z))
            return np.dot(X.T, p - y) / n_samples
        
        def hess_logreg(w):
            z = np.dot(X, w)
            p = 1.0 / (1.0 + np.exp(-z))
            d = p * (1.0 - p)
            return np.dot(X.T * d, X) / n_samples
        
        # L1 regularization parameter
        alpha = 0.1
        
        # Create proximal Newton solver
        pn = ProximalNewton(f_logreg, grad_logreg, hess_logreg, 
                          lambda x, t: self.proximal_l1(x, alpha * t),
                          max_iter=50, tol=1e-5, verbose=False)
        
        # Optimize
        w_pn = pn.optimize(np.zeros(n_features))
        
        # Create a reference solution using a simpler optimizer
        # Define the L1-regularized objective
        def f_total(w):
            return f_logreg(w) + alpha * np.sum(np.abs(w))
        
        # Optimize with L-BFGS-B to create baseline
        w0 = np.zeros(n_features)
        result_reference = optimize.minimize(f_total, w0, method='L-BFGS-B', jac=lambda w: grad_logreg(w) + alpha * np.sign(w) * (np.abs(w) > 1e-10))
        w_ref = result_reference.x
        
        # Test sparsity - more important than exact objective match
        nonzero_pn = np.abs(w_pn) > 1e-4
        nonzero_ref = np.abs(w_ref) > 1e-4
        
        # Check if enough coordinates are properly zeroed out
        true_nonzero = np.abs(true_w) > 1e-10
        
        # Test if ProximalNewton finds sparse solution
        self.assertLess(np.sum(nonzero_pn), n_features/2)
        
        # Test if the important features are identified
        # At least some of the true nonzero features should be identified
        true_pos_count = np.sum(nonzero_pn & true_nonzero)
        self.assertGreater(true_pos_count, 0)
        
        # Check classification accuracy
        pred_pn = np.dot(X, w_pn) > 0
        pred_ref = np.dot(X, w_ref) > 0
        pred_true = np.dot(X, true_w) > 0
        
        acc_pn = np.mean(pred_pn == pred_true)
        acc_ref = np.mean(pred_ref == pred_true)
        
        # Our accuracy should be within 90% of the reference accuracy
        self.assertGreater(acc_pn, 0.9 * acc_ref)

    def test_hessian_approximations(self):
        """
        Test different Hessian approximation techniques in Proximal Newton.
        
        This test compares exact Hessian, BFGS and L-BFGS approximations
        on a simple problem to verify they all converge to the same solution.
        """
        # Simple non-quadratic problem
        n = 10
        
        def f(x):
            return np.sum(np.log(1 + np.exp(x))) + 0.5 * np.sum(x**2)
        
        def grad(x):
            exp_x = np.exp(x)
            return exp_x / (1 + exp_x) + x
        
        def hess(x):
            exp_x = np.exp(x)
            d = exp_x / (1 + exp_x)**2
            return np.diag(d + 1.0)  # Add 1.0 for the quadratic term
        
        # Initial point
        x0 = np.ones(n)
        
        # Identity proximal operator (no regularization)
        identity_prox = lambda x, t: x
        
        # Methods to test
        methods = ['exact', 'bfgs', 'l-bfgs']
        results = {}
        
        for method in methods:
            pn = ProximalNewton(f, grad, hess if method == 'exact' else None,
                              identity_prox, max_iter=50, tol=1e-6,
                              hessian_type=method, memory=5 if method == 'l-bfgs' else None)
            
            results[method] = pn.optimize(x0.copy())
        
        # All methods should converge to similar solutions
        for method1 in methods:
            for method2 in methods:
                if method1 != method2:
                    diff = np.linalg.norm(results[method1] - results[method2])
                    self.assertLess(diff, 1e-4)
        
        # All solutions should have small gradient norm
        for method in methods:
            grad_norm = np.linalg.norm(grad(results[method]))
            self.assertLess(grad_norm, 1e-5)

    def test_nonconvex_optimization(self):
        """
        Test Proximal Newton on a simple non-convex problem.
        
        This test verifies that the algorithm can find a local minimum
        of a non-convex function when initialized appropriately.
        """
        # Simple non-convex function with known local minima
        # f(x) = sum(x_i^4 - 16*x_i^2 + 5*x_i)
        # Local minima occur near x_i ≈ -2.9 and x_i ≈ 2.7
        
        def f(x):
            return np.sum(x**4 - 16*x**2 + 5*x)
        
        def grad(x):
            return 4*x**3 - 32*x + 5
        
        def hess(x):
            return np.diag(12*x**2 - 32)
        
        # Test for different dimensions
        for n in [1, 5, 10]:
            # Starting near the expected local minimum at x_i ≈ -2.9
            x0 = -3.0 * np.ones(n)
            
            # Use identity proximal operator (no regularization)
            identity_prox = lambda x, t: x
            
            # Create proximal Newton solver with damping to handle non-convexity
            pn = ProximalNewton(f, grad, hess, identity_prox, 
                              max_iter=50, tol=1e-6, 
                              hessian_damping=True, damping_factor=0.1,
                              verbose=False)
            
            # Optimize
            x_sol = pn.optimize(x0)
            
            # Check if all components are close to the expected local minimum
            expected_min = -2.9 * np.ones(n)
            component_diff = np.abs(x_sol - expected_min)
            self.assertTrue(np.all(component_diff < 0.1))
            
            # Check that gradient norm is small at solution
            grad_norm = np.linalg.norm(grad(x_sol))
            self.assertLess(grad_norm, 1e-5)
            
            # Also verify we can find the other local minimum
            # Starting near the expected local minimum at x_i ≈ 2.7
            x0 = 3.0 * np.ones(n)
            x_sol2 = pn.optimize(x0)
            
            # Check if all components are close to the expected local minimum
            expected_min2 = 2.7 * np.ones(n)
            component_diff2 = np.abs(x_sol2 - expected_min2)
            self.assertTrue(np.all(component_diff2 < 0.1))

    def test_box_constrained_optimization(self):
        """
        Test Proximal Newton for box-constrained optimization.
        
        This test verifies that the method correctly enforces box constraints
        through the proximal operator.
        """
        # Quadratic objective with box constraints
        n = 10
        A = np.random.randn(n, n)
        A = np.dot(A.T, A) + 0.1 * np.eye(n)
        b = np.random.randn(n)
        
        # Objective: 0.5 * x^T A x - b^T x subject to 0 <= x <= 1
        f_obj = lambda x: 0.5 * np.dot(x, np.dot(A, x)) - np.dot(b, x)
        grad_obj = lambda x: np.dot(A, x) - b
        hess_obj = lambda x: A
        
        # Box constraints: 0 <= x <= 1
        lower, upper = 0.0, 1.0
        box_prox = lambda x, t: np.clip(x, lower, upper)
        
        # Create proximal Newton solver
        pn = ProximalNewton(f_obj, grad_obj, hess_obj, box_prox,
                          max_iter=50, tol=1e-6, verbose=False)
        
        # Optimize
        x_pn = pn.optimize(0.5 * np.ones(n))
        
        # Reference solution using direct box constraints
        result_ref = optimize.minimize(f_obj, 0.5 * np.ones(n), method='L-BFGS-B',
                                     jac=grad_obj, bounds=[(lower, upper) for _ in range(n)])
        x_ref = result_ref.x
        
        # Check constraints are satisfied
        self.assertTrue(np.all(x_pn >= lower - 1e-6))
        self.assertTrue(np.all(x_pn <= upper + 1e-6))
        
        # Check objective values are similar
        obj_pn = f_obj(x_pn)
        obj_ref = f_obj(x_ref)
        self.assertLess(abs(obj_pn - obj_ref) / (abs(obj_ref) + 1e-10), 0.1)  # Within 10%


if __name__ == '__main__':
    unittest.main()