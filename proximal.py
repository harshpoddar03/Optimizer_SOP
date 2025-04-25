import numpy as np

class ProximalGradient:
    """
    Proximal Gradient Method optimizer.
    
    This class implements the proximal gradient method for optimizing composite
    functions of the form f(x) + g(x), where f is smooth and g is potentially
    non-smooth but has a computable proximal operator.
    
    The method performs iterations of the form:
        x_{k+1} = prox_{t*g}(x_k - t*∇f(x_k))
    
    where t is the step size and prox is the proximal operator.
    
    The accelerated version (FISTA) uses a momentum term to achieve
    O(1/k^2) convergence rate instead of O(1/k).
    """
    def __init__(self, f_smooth, grad_smooth, proximal_op, 
                 step_size=0.01, max_iter=1000, tol=1e-6, 
                 accelerated=False, adaptive_step_size=False,
                 regularization_param=1.0, verbose=False):
        """
        Initialize the Proximal Gradient optimizer.
        
        Parameters:
        -----------
        f_smooth : callable
            Smooth part of the objective function
            f_smooth(x) -> float
        grad_smooth : callable
            Gradient of the smooth part
            grad_smooth(x) -> array
        proximal_op : callable
            Proximal operator of the non-smooth part
            proximal_op(x, t) -> array
        step_size : float, optional (default=0.01)
            Initial step size
        max_iter : int, optional (default=1000)
            Maximum number of iterations
        tol : float, optional (default=1e-6)
            Convergence tolerance
        accelerated : bool, optional (default=False)
            Whether to use acceleration (FISTA)
        adaptive_step_size : bool, optional (default=False)
            Whether to use backtracking line search
        regularization_param : float, optional (default=1.0)
            Regularization parameter passed to proximal operator
        verbose : bool, optional (default=False)
            Whether to print progress information
        """
        self.f_smooth = f_smooth
        self.grad_smooth = grad_smooth
        self.proximal_op = proximal_op
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol
        self.accelerated = accelerated
        self.adaptive_step_size = adaptive_step_size
        self.regularization_param = regularization_param
        self.verbose = verbose
        
    def optimize(self, x0, return_iterations=False, return_trajectory=False):
        """
        Run the optimization algorithm.
        
        Parameters:
        -----------
        x0 : array-like
            Initial point
        return_iterations : bool, optional (default=False)
            Whether to return number of iterations
        return_trajectory : bool, optional (default=False)
            Whether to return the trajectory of iterates
            
        Returns:
        --------
        x : array-like
            Optimal solution
        iterations : int, optional
            Number of iterations taken (if return_iterations=True)
        trajectory : list, optional
            List of iterates (if return_trajectory=True)
        """
        x = np.asarray(x0).copy()
        if self.accelerated:
            y = x.copy()
            t_prev = 1.0
        
        # For backtracking line search
        alpha = 0.5  # Decrease factor for step size
        beta = 0.5   # Sufficient decrease parameter
        
        trajectory = []
        if return_trajectory:
            trajectory.append(x.copy())
        
        for k in range(self.max_iter):
            if self.accelerated:
                grad_y = self.grad_smooth(y)
                step = self.step_size
                
                if self.adaptive_step_size:
                    step = self._backtracking_line_search(y, grad_y, alpha, beta)
                
                x_new = self._apply_proximal(y - step * grad_y, step)
                
                # FISTA update
                t = 0.5 * (1 + np.sqrt(1 + 4 * t_prev**2))
                y = x_new + ((t_prev - 1) / t) * (x_new - x)
                
                # Store previous values
                x, t_prev = x_new, t
            else:
                grad_x = self.grad_smooth(x)
                step = self.step_size
                
                if self.adaptive_step_size:
                    step = self._backtracking_line_search(x, grad_x, alpha, beta)
                
                x_new = self._apply_proximal(x - step * grad_x, step)
                
                # Check convergence
                rel_change = np.linalg.norm(x_new - x) / max(np.linalg.norm(x), 1e-8)
                if rel_change < self.tol:
                    x = x_new
                    if self.verbose:
                        print(f"Converged after {k+1} iterations.")
                    break
                
                x = x_new
            
            if return_trajectory:
                trajectory.append(x.copy())
                
            if self.verbose and k % 10 == 0:
                print(f"Iteration {k}: objective = {self.f_smooth(x)}")
        
        if self.verbose and k == self.max_iter - 1:
            print(f"Maximum iterations ({self.max_iter}) reached.")
        
        # Prepare return values
        result = [x]
        if return_iterations:
            result.append(k + 1)
        if return_trajectory:
            result.append(trajectory)
            
        if len(result) == 1:
            return result[0]
        else:
            return tuple(result)
        
    def _apply_proximal(self, x, t):
        """
        Apply the proximal operator with step size t.
        
        Parameters:
        -----------
        x : array-like
            Input point
        t : float
            Step size
            
        Returns:
        --------
        prox_x : array-like
            Result of applying the proximal operator
        """
        return self.proximal_op(x, t * self.regularization_param)
    
    def _backtracking_line_search(self, x, grad_x, alpha=0.5, beta=0.5, max_iter=20):
        """
        Perform backtracking line search to find an appropriate step size.
        
        Parameters:
        -----------
        x : array-like
            Current point
        grad_x : array-like
            Gradient at current point
        alpha : float, optional (default=0.5)
            Reduction factor for step size
        beta : float, optional (default=0.5)
            Sufficient decrease parameter
        max_iter : int, optional (default=20)
            Maximum number of line search iterations
            
        Returns:
        --------
        step : float
            Selected step size
        """
        step = self.step_size
        f_x = self.f_smooth(x)
        
        for _ in range(max_iter):
            # Tentative new point
            x_new = self._apply_proximal(x - step * grad_x, step)
            
            # Check sufficient decrease condition
            if self.f_smooth(x_new) <= f_x - beta * np.sum(grad_x * (x - x_new)) + (1/(2*step)) * np.sum((x - x_new)**2):
                return step
            
            # Reduce step size
            step *= alpha
            
        return step


def soft_thresholding(x, threshold):
    """
    Soft thresholding operator (proximal operator for L1 norm).
    
    Parameters:
    -----------
    x : array-like
        Input array
    threshold : float
        Threshold value
        
    Returns:
    --------
    y : array-like
        Soft-thresholded array
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def group_soft_thresholding(x, groups, threshold):
    """
    Group soft thresholding operator (proximal operator for group lasso).
    
    Parameters:
    -----------
    x : array-like
        Input array
    groups : list of arrays
        List of arrays containing indices for each group
    threshold : float
        Threshold value
        
    Returns:
    --------
    y : array-like
        Group soft-thresholded array
    """
    result = x.copy()
    
    for group in groups:
        group_norm = np.linalg.norm(x[group])
        if group_norm > 0:
            scaling = max(0, 1 - threshold / group_norm)
            result[group] = scaling * x[group]
            
    return result


def proximal_box_constraint(x, t, lower=0, upper=1):
    """
    Proximal operator for box constraints.
    
    Parameters:
    -----------
    x : array-like
        Input array
    t : float
        Step size (not used for box constraints)
    lower : float or array-like, optional (default=0)
        Lower bound
    upper : float or array-like, optional (default=1)
        Upper bound
        
    Returns:
    --------
    y : array-like
        Projected array
    """
    return np.clip(x, lower, upper)


def proximal_elastic_net(x, t, alpha_l1, alpha_l2):
    """
    Proximal operator for elastic net regularization (L1 + L2).
    
    Parameters:
    -----------
    x : array-like
        Input array
    t : float
        Step size
    alpha_l1 : float
        L1 regularization parameter
    alpha_l2 : float
        L2 regularization parameter
        
    Returns:
    --------
    y : array-like
        Result of elastic net proximal operator
    """
    # First apply L2 (ridge)
    x_l2 = x / (1 + 2 * t * alpha_l2)
    # Then apply L1 (lasso)
    return soft_thresholding(x_l2, t * alpha_l1)