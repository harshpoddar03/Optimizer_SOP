import numpy as np
from scipy import optimize

class MoreauYosida:
    """
    Moreau-Yosida regularization for non-smooth functions.
    
    The Moreau-Yosida regularization (or Moreau envelope) of a function g
    with parameter gamma is defined as:
    
        M_gamma[g](x) = min_y { g(y) + (1/2*gamma) * ||x - y||^2 }
    
    where the minimizer y is the proximal point of x.
    
    This provides a smooth approximation to a non-smooth function g,
    with the approximation becoming tighter as gamma approaches zero.
    The gradient of the Moreau envelope is:
    
        ∇M_gamma[g](x) = (1/gamma) * (x - prox_gamma[g](x))
    
    The Moreau envelope has the same minimizers as the original function g.
    """
    def __init__(self, func, proximal_op, gamma=0.1):
        """
        Initialize Moreau-Yosida regularization.
        
        Parameters:
        -----------
        func : callable
            Original non-smooth function
            func(x) -> float
        proximal_op : callable
            Proximal operator of the function
            proximal_op(x, t) -> array
        gamma : float, optional (default=0.1)
            Regularization parameter
        """
        self.func = func
        self.proximal_op = proximal_op
        self.gamma = gamma
        self._cached_prox_point = None
        self._cached_input = None
        
    def prox_point(self, x):
        """
        Compute the proximal point (minimizer of the Moreau envelope).
        
        Parameters:
        -----------
        x : array-like
            Input point
            
        Returns:
        --------
        p : array-like
            Proximal point
        """
        x = np.asarray(x)
        
        # Check if cached result can be used
        if (self._cached_input is not None and 
            self._cached_prox_point is not None and
            np.array_equal(x, self._cached_input)):
            return self._cached_prox_point
        
        # Compute and cache the proximal point
        self._cached_input = x.copy()
        self._cached_prox_point = self.proximal_op(x, self.gamma)
        return self._cached_prox_point
        
    def value(self, x):
        """
        Compute the Moreau envelope value.
        
        Parameters:
        -----------
        x : array-like
            Input point
            
        Returns:
        --------
        val : float
            Value of the Moreau envelope
        """
        x = np.asarray(x)
        p = self.prox_point(x)
        return self.func(p) + (1/(2*self.gamma)) * np.sum((x - p)**2)
        
    def gradient(self, x):
        """
        Compute the gradient of the Moreau envelope.
        
        Parameters:
        -----------
        x : array-like
            Input point
            
        Returns:
        --------
        grad : array-like
            Gradient of the Moreau envelope
        """
        x = np.asarray(x)
        p = self.prox_point(x)
        return (1/self.gamma) * (x - p)

    def hessian(self, x):
        """
        Approximate the Hessian of the Moreau envelope.
        
        For the L1 norm, the Hessian is a diagonal matrix where
        the diagonal elements are (1/gamma) for coordinates where
        |x_i| > gamma, and 0 elsewhere.
        
        For general functions, we use a finite difference approximation.
        
        Parameters:
        -----------
        x : array-like
            Input point
            
        Returns:
        --------
        hess : array-like
            Approximate Hessian matrix
        """
        x = np.asarray(x)
        n = len(x)
        h = 1e-8  # Step size for finite differences
        
        # Simple case: diagonal approximation
        hess_diag = np.zeros(n)
        
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            
            grad_plus = self.gradient(x_plus)
            grad_minus = self.gradient(x_minus)
            
            # Central difference
            hess_diag[i] = (grad_plus[i] - grad_minus[i]) / (2 * h)
        
        # Return diagonal Hessian for simplicity
        return np.diag(np.maximum(0, hess_diag))  # Ensure positive semidefiniteness
    
    def hessian_vector_product(self, x, v):
        """
        Compute the Hessian-vector product without forming the full Hessian.
        
        This is useful for large-scale problems where forming the full
        Hessian is impractical.
        
        Parameters:
        -----------
        x : array-like
            Input point
        v : array-like
            Vector to multiply with the Hessian
            
        Returns:
        --------
        Hv : array-like
            Hessian-vector product
        """
        x = np.asarray(x)
        v = np.asarray(v)
        h = 1e-8  # Step size for finite differences
        
        # Forward difference approximation
        grad_x = self.gradient(x)
        grad_xpv = self.gradient(x + h * v)
        
        return (grad_xpv - grad_x) / h
    
    def optimize(self, x0, method='L-BFGS-B', **kwargs):
        """
        Optimize a function using the Moreau-Yosida regularization.
        
        Parameters:
        -----------
        x0 : array-like
            Initial point
        method : str, optional (default='L-BFGS-B')
            Optimization method to use
        **kwargs : dict
            Additional arguments passed to scipy.optimize.minimize
            
        Returns:
        --------
        result : OptimizeResult
            Result of the optimization
        """
        # Set a tighter tolerance for better convergence
        if 'tol' not in kwargs:
            kwargs['tol'] = 1e-8
            
        # Add some options for better convergence if they don't exist
        if 'options' not in kwargs:
            kwargs['options'] = {'maxiter': 1000, 'gtol': 1e-8}
        
        return optimize.minimize(
            fun=self.value,
            x0=x0,
            jac=self.gradient,
            method=method,
            **kwargs
        )


class MoreauYosidaOptimizer:
    """
    Optimizer using Moreau-Yosida regularization for composite optimization problems.
    
    This optimizer is designed for problems of the form:
        min_x f(x) + g(x)
    
    where f is smooth and g is non-smooth but has a computable proximal operator.
    The optimizer creates a smoothed approximation of g using Moreau-Yosida 
    regularization and then applies standard smooth optimization techniques.
    """
    def __init__(self, f_smooth, grad_smooth, g_func, g_prox, 
                 gamma=0.001, method='L-BFGS-B',  # Changed default gamma to 0.001
                 gamma_schedule=None, max_iter=100, 
                 tol=1e-6, callback=None, verbose=False):
        """
        Initialize the Moreau-Yosida optimizer.
        
        Parameters:
        -----------
        f_smooth : callable
            Smooth part of the objective function
            f_smooth(x) -> float
        grad_smooth : callable
            Gradient of the smooth part
            grad_smooth(x) -> array
        g_func : callable
            Non-smooth part of the objective function
            g_func(x) -> float
        g_prox : callable
            Proximal operator of the non-smooth part
            g_prox(x, t) -> array
        gamma : float, optional (default=0.001)
            Initial regularization parameter (changed from 0.1 to 0.001)
        method : str, optional (default='L-BFGS-B')
            Optimization method to use
        gamma_schedule : callable or None, optional (default=None)
            Function to update gamma after each iteration
            gamma_schedule(gamma, iteration) -> new_gamma
        max_iter : int, optional (default=100)
            Maximum number of iterations
        tol : float, optional (default=1e-6)
            Convergence tolerance
        callback : callable or None, optional (default=None)
            Function called after each iteration
        verbose : bool, optional (default=False)
            Whether to print progress information
        """
        self.f_smooth = f_smooth
        self.grad_smooth = grad_smooth
        self.g_func = g_func
        self.g_prox = g_prox
        self.gamma = gamma
        self.method = method
        self.gamma_schedule = gamma_schedule
        self.max_iter = max_iter
        self.tol = tol
        self.callback = callback
        self.verbose = verbose
        
    def optimize(self, x0, return_trajectory=False):
        """
        Run the optimization algorithm.
        
        Parameters:
        -----------
        x0 : array-like
            Initial point
        return_trajectory : bool, optional (default=False)
            Whether to return the trajectory of iterates
            
        Returns:
        --------
        x : array-like
            Optimal solution
        trajectory : list, optional
            List of iterates (if return_trajectory=True)
        """
        x = np.asarray(x0).copy()
        gamma = self.gamma
        
        trajectory = []
        if return_trajectory:
            trajectory.append(x.copy())
        
        for k in range(self.max_iter):
            # Create Moreau-Yosida regularization with current gamma
            my_g = MoreauYosida(self.g_func, self.g_prox, gamma=gamma)
            
            # Total objective and gradient
            def f_total(x):
                return self.f_smooth(x) + my_g.value(x)
            
            def grad_total(x):
                return self.grad_smooth(x) + my_g.gradient(x)
            
            # Optimize with current gamma
            options = {'maxiter': 200, 'gtol': self.tol * 0.1}  # Increased maxiter and tightened convergence
            
            if k > 0:
                # Warm start from previous solution
                x0_iter = x
            else:
                x0_iter = x0
                
            result = optimize.minimize(
                fun=f_total,
                x0=x0_iter,
                jac=grad_total,
                method=self.method,
                options=options
            )
            
            x_new = result.x
            
            # Check convergence
            rel_change = np.linalg.norm(x_new - x) / max(np.linalg.norm(x), 1e-8)
            if rel_change < self.tol:
                x = x_new
                if self.verbose:
                    print(f"Converged after {k+1} outer iterations.")
                break
            
            x = x_new
            
            if return_trajectory:
                trajectory.append(x.copy())
                
            if self.verbose:
                print(f"Iteration {k+1}: gamma = {gamma:.6f}, objective = {f_total(x):.6f}")
                
            if self.callback is not None:
                self.callback(x)
                
            # Update gamma if scheduled
            if self.gamma_schedule is not None:
                gamma_new = self.gamma_schedule(gamma, k)
                if gamma_new != gamma:
                    gamma = gamma_new
                    if self.verbose:
                        print(f"Updated gamma to {gamma:.6f}")
        
        if self.verbose and k == self.max_iter - 1:
            print(f"Maximum iterations ({self.max_iter}) reached.")
            
        if return_trajectory:
            return x, trajectory
        else:
            return x


def default_gamma_schedule(gamma, iteration):
    """
    Default schedule for decreasing gamma.
    
    Parameters:
    -----------
    gamma : float
        Current gamma value
    iteration : int
        Current iteration number
        
    Returns:
    --------
    new_gamma : float
        Updated gamma value
    """
    # More aggressive gamma reduction - decrease gamma by a factor of 0.5 every 2 iterations
    if iteration > 0 and iteration % 2 == 0:
        return gamma * 0.5
    return gamma