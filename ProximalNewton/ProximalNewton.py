import numpy as np
from scipy import sparse, optimize
from scipy.sparse import linalg as splinalg

class ProximalNewton:
    """
    Proximal Newton optimizer for composite objectives f(x)+g(x).
    Supports exact, BFGS, and L-BFGS Hessian approximations and three subproblem solvers.
    """
    def __init__(
            self,
            f_smooth,
            grad_smooth,
            hess_smooth=None,
            proximal_op=None,
            max_iter=100,
            tol=1e-6,
            hessian_type='exact',
            subproblem_solver='prox-gradient',
            subproblem_iter=200,
            subproblem_tol=1e-8,
            memory=10,
            sparse_hessian=False,
            hessian_damping=False,
            damping_factor=1e-4,
            line_search=True,
            trust_region=False,
            trust_radius=1.0,
            min_trust_radius=1e-4,
            max_trust_radius=10.0,
            eta=0.1,
            convergence_criterion='gradient',
            verbose=False
        ):
            # Problem functions
            self.f_smooth = f_smooth
            self.grad_smooth = grad_smooth
            self.hess_smooth = hess_smooth
            self.proximal_op = proximal_op if proximal_op is not None else (lambda x, t: x)
            
            # Algorithm controls
            self.max_iter = max_iter
            self.tol = tol
            self.hessian_type = hessian_type
            self.subproblem_solver = subproblem_solver
            self.subproblem_iter = subproblem_iter
            self.subproblem_tol = subproblem_tol
            self.memory = memory
            self.sparse_hessian = sparse_hessian
            self.hessian_damping = hessian_damping
            self.damping_factor = damping_factor
            self.line_search = line_search
            self.trust_region = trust_region
            self.trust_radius = trust_radius
            self.min_trust_radius = min_trust_radius
            self.max_trust_radius = max_trust_radius
            self.eta = eta
            self.convergence_criterion = convergence_criterion
            self.verbose = verbose
            
            # Quasi-Newton storage
            self.bfgs_H = None
            self.lbfgs_s = []
            self.lbfgs_y = []
            self.lbfgs_rho = []
            
            # Validate parameters
            if self.hessian_type == 'exact' and self.hess_smooth is None:
                raise ValueError("hess_smooth must be provided when hessian_type is 'exact'")


    def optimize(self, x0, return_iterates=False, return_iterations=False):
        """
        Run the optimization algorithm.
        
        Parameters:
        -----------
        x0 : array-like
            Initial point
        return_iterates : bool, optional (default=False)
            Whether to return the trajectory of iterates
        return_iterations : bool, optional (default=False)
            Whether to return the number of iterations taken
            
        Returns:
        --------
        x : array-like
            Optimal solution
        iterates : list, optional
            List of iterates (if return_iterates=True)
        iterations : int, optional
            Number of iterations taken (if return_iterations=True)
        """
        x = np.asarray(x0).copy()
        n = x.size
        
        # Initialize Hessian approximations
        if self.hessian_type == 'bfgs':
            self.bfgs_H = np.eye(n)
        if self.hessian_type == 'l-bfgs':
            self.lbfgs_s, self.lbfgs_y, self.lbfgs_rho = [], [], []
            
        trust_radius = self.trust_radius
        grad_prev = None
        grad_x = self.grad_smooth(x)
        f_x = self.f_smooth(x)
        
        iterates = [x.copy()] if return_iterates else []
        k_final = self.max_iter
        
        for k in range(self.max_iter):
            # Compute Hessian or approximation
            H = self._compute_hessian(x, grad_x, grad_prev)
            
            # Solve the subproblem to get the search direction
            d = self._solve_subproblem(x, grad_x, H, trust_radius if self.trust_region else None)
            
            # Initialize step size
            alpha = 1.0
            
            # Line search if enabled and not using trust region
            if self.line_search and not self.trust_region:
                alpha = self._backtracking_line_search(x, d, grad_x)
            # Trust region adjustments if enabled
            elif self.trust_region:
                # Compute predicted reduction
                if isinstance(H, splinalg.LinearOperator):
                    Hd = H.matvec(d)
                else:
                    Hd = H.dot(d)
                pred = -grad_x.dot(d) - 0.5 * d.dot(Hd)
                
                # Compute actual reduction
                x_trial = x + d
                f_trial = self.f_smooth(x_trial)
                actual = f_x - f_trial
                
                # Compute trust ratio
                rho = actual / max(pred, 1e-10)
                
                # Update trust radius
                if rho < 0.25:
                    trust_radius = max(0.25 * trust_radius, self.min_trust_radius)
                elif rho > 0.75 and np.linalg.norm(d) >= 0.9 * trust_radius:
                    trust_radius = min(2.0 * trust_radius, self.max_trust_radius)
                
                # Determine if step should be accepted
                if rho < self.eta:
                    alpha = 0.0
                else:
                    f_x = f_trial
            
            # Apply the step
            x_new = x + alpha * d
            
            # Compute the new gradient
            grad_new = self.grad_smooth(x_new)
            
            # Update function value if not already updated
            if not (self.trust_region and rho >= self.eta):
                f_x = self.f_smooth(x_new)
            
            # Update Quasi-Newton approximation
            if self.hessian_type in ['bfgs', 'l-bfgs'] and alpha > 0:
                s = alpha * d
                y = grad_new - grad_x
                sy = s.dot(y)
                
                # Skip update if curvature condition is not satisfied
                if sy > 1e-10:
                    self._update_hessian_approximation(s, y)
            
            # Store previous gradient for potential use
            grad_prev = grad_x
            
            # Update variables for next iteration
            x, grad_x = x_new, grad_new
            
            # Store iterate if requested
            if return_iterates:
                iterates.append(x.copy())
            
            # Print progress if verbose
            if self.verbose and k % 5 == 0:
                print(f"Iteration {k}: objective = {f_x:.6e}, grad_norm = {np.linalg.norm(grad_x):.6e}")
            
            # Check convergence
            if self._check_convergence(x, grad_x, d, alpha, f_x, k):
                k_final = k + 1
                break
        
        # Prepare return values
        if self.verbose and k == self.max_iter - 1:
            print(f"Maximum iterations ({self.max_iter}) reached.")
        
        if return_iterates and return_iterations:
            return iterates, k_final
        
        result = [x]
        if return_iterates:
            result.append(iterates)
        if return_iterations:
            result.append(k_final)
        
        return result[0] if len(result) == 1 else tuple(result)

    def _check_convergence(self, x, grad, d, alpha, f, k):
        """
        Check if convergence criteria are met.
        
        Parameters:
        -----------
        x : array-like
            Current iterate
        grad : array-like
            Current gradient
        d : array-like
            Search direction
        alpha : float
            Step size
        f : float
            Current function value
        k : int
            Iteration count
            
        Returns:
        --------
        bool
            True if converged, False otherwise
        """
        if self.convergence_criterion == 'gradient':
            grad_norm = np.linalg.norm(grad)
            return grad_norm < self.tol
        elif self.convergence_criterion == 'function':
            if k > 0 and hasattr(self, '_prev_f'):
                rel_change = abs(f - self._prev_f) / max(abs(self._prev_f), 1.0)
                self._prev_f = f
                return rel_change < self.tol
            else:
                self._prev_f = f
                return False
        else:  # 'parameter'
            step_norm = np.linalg.norm(alpha * d)
            rel_step = step_norm / max(np.linalg.norm(x), 1.0)
            return rel_step < self.tol

    def _compute_hessian(self, x, grad_x=None, grad_prev=None):
        """
        Compute or approximate the Hessian.
        
        Parameters:
        -----------
        x : array-like
            Current iterate
        grad_x : array-like, optional
            Current gradient
        grad_prev : array-like, optional
            Previous gradient
            
        Returns:
        --------
        H : array-like or LinearOperator
            Hessian or Hessian approximation
        """
        n = x.size
        
        if self.hessian_type == 'exact':
            # Use exact Hessian
            H = self.hess_smooth(x)
            
            # Convert to sparse if requested
            if self.sparse_hessian and not sparse.issparse(H):
                H = sparse.csr_matrix(H)
            
            # Apply damping if requested
            if self.hessian_damping:
                I = sparse.eye(n) if sparse.issparse(H) else np.eye(n)
                
                # For ill-conditioned problems, check eigenvalues
                if not sparse.issparse(H):
                    try:
                        min_eig = np.min(np.linalg.eigvalsh(H))
                        if min_eig < self.damping_factor:
                            H = H + (self.damping_factor - min_eig) * I
                        else:
                            H = H + self.damping_factor * I
                    except np.linalg.LinAlgError:
                        # If eigenvalue computation fails, apply standard damping
                        H = H + self.damping_factor * I
                else:
                    # For sparse matrices, just apply standard damping
                    H = H + self.damping_factor * I
            
            return H
            
        elif self.hessian_type == 'bfgs':
            # Use BFGS approximation
            if self.bfgs_H is None:
                self.bfgs_H = np.eye(n)
            
            return splinalg.LinearOperator(
                (n, n), 
                matvec=lambda v: self.bfgs_H.dot(v),
                rmatvec=lambda v: self.bfgs_H.dot(v)
            )
            
        elif self.hessian_type == 'l-bfgs':
            # Use L-BFGS approximation
            return splinalg.LinearOperator(
                (n, n), 
                matvec=self._lbfgs_hessian_vector_product,
                rmatvec=self._lbfgs_hessian_vector_product
            )
            
        else:
            raise ValueError(f"Unknown Hessian type: {self.hessian_type}")

    def _update_hessian_approximation(self, s, y):
        """
        Update the Quasi-Newton Hessian approximation.
        
        Parameters:
        -----------
        s : array-like
            Step vector (x_new - x)
        y : array-like
            Gradient difference (grad_new - grad)
        """
        sy = s.dot(y)
        
        # Skip update if curvature condition is not satisfied
        if sy <= 1e-10:
            return
        
        rho = 1.0 / sy
        
        if self.hessian_type == 'bfgs':
            # BFGS update
            n = len(s)
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            self.bfgs_H = V.T.dot(self.bfgs_H).dot(V) + rho * np.outer(s, s)
            
            # Ensure positive definiteness
            min_eig = np.min(np.linalg.eigvalsh(self.bfgs_H))
            if min_eig < 1e-6:
                self.bfgs_H += (1e-6 - min_eig) * np.eye(n)
                
        elif self.hessian_type == 'l-bfgs':
            # L-BFGS update
            self.lbfgs_s.append(s.copy())
            self.lbfgs_y.append(y.copy())
            self.lbfgs_rho.append(rho)
            
            # Limit memory
            if len(self.lbfgs_s) > self.memory:
                self.lbfgs_s.pop(0)
                self.lbfgs_y.pop(0)
                self.lbfgs_rho.pop(0)

    def _lbfgs_hessian_vector_product(self, v):
        """
        Compute the L-BFGS Hessian-vector product without forming the full Hessian.
        
        Parameters:
        -----------
        v : array-like
            Vector to multiply with the Hessian
            
        Returns:
        --------
        Hv : array-like
            Hessian-vector product
        """
        q = v.copy()
        alphas = []
        
        # First recursion
        for i in range(len(self.lbfgs_s) - 1, -1, -1):
            alpha = self.lbfgs_rho[i] * self.lbfgs_s[i].dot(q)
            alphas.append(alpha)
            q = q - alpha * self.lbfgs_y[i]
        
        # Apply initial Hessian approximation (scaling)
        if self.lbfgs_s:
            s, y = self.lbfgs_s[-1], self.lbfgs_y[-1]
            gamma = s.dot(y) / y.dot(y)
            r = gamma * q
        else:
            # If no updates yet, use identity
            r = q
        
        # Second recursion
        for i in range(len(self.lbfgs_s)):
            beta = self.lbfgs_rho[i] * self.lbfgs_y[i].dot(r)
            r = r + self.lbfgs_s[i] * (alphas[-1 - i] - beta)
        
        return r
    
    def _proximal_coordinate(self, z, i, hess_ii):
        """
        Apply the proximal operator to a single coordinate.
        
        Parameters:
        -----------
        z : float
            Coordinate value
        i : int
            Coordinate index
        hess_ii : float
            Diagonal Hessian element
            
        Returns:
        --------
        float
            Updated coordinate value
        """
        # Create a vector with z at position i and zeros elsewhere
        x = np.zeros(max(i+1, 1))
        x[i] = z
        
        # Apply the full proximal operator
        t = 1.0 / max(hess_ii, 1e-10)
        result = self.proximal_op(x, t)
        
        # Return the i-th component
        return result[i]

    def _backtracking_line_search(self, x, d, grad, alpha=0.5, beta=0.5, max_iter=20):
        """
        Perform backtracking line search to find an appropriate step size.
        
        Parameters:
        -----------
        x : array-like
            Current iterate
        d : array-like
            Search direction
        grad : array-like
            Current gradient
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
        step = 1.0
        f0 = self.f_smooth(x)
        gd = grad.dot(d)
        
        # Ensure descent direction
        if gd > -1e-10:
            if self.verbose:
                print("Warning: not a descent direction. Using steepest descent.")
            d = -grad
            gd = -grad.dot(grad)
        
        for i in range(max_iter):
            x_new = x + step * d
            f_new = self.f_smooth(x_new)
            
            # Check Armijo condition
            if f_new <= f0 + beta * step * gd:
                return step
            
            # Reduce step size
            step *= alpha
            
            if step < 1e-10:
                # Step size too small, return minimum step
                return 1e-10
        
        # If we get here, use the final step size
        return step

    def _solve_subproblem(self, x, grad, hessian, trust_radius=None):
        """
        Solve the quadratic subproblem to determine the search direction.
        
        Parameters:
        -----------
        x : array-like
            Current iterate
        grad : array-like
            Current gradient
        hessian : array-like or LinearOperator
            Hessian or Hessian approximation
        trust_radius : float or None, optional
            Trust region radius
            
        Returns:
        --------
        d : array-like
            Search direction
        """
        n = x.size
        
        # Special case: identity proximal operator
        is_identity_prox = True
        try:
            # Test on multiple points to ensure it's really the identity
            for _ in range(3):
                u = np.random.randn(n)
                if not np.allclose(self.proximal_op(u, 1.0), u, rtol=1e-10, atol=1e-10):
                    is_identity_prox = False
                    break
        except Exception:
            is_identity_prox = False
        
        if is_identity_prox:
            # For smooth problems, solve the Newton system directly
            try:
                if isinstance(hessian, splinalg.LinearOperator):
                    # Use iterative solver for LinearOperator
                    d, info = splinalg.cg(hessian, -grad, tol=1e-10)
                    if info != 0:
                        raise ValueError(f"CG solver did not converge: {info}")
                else:
                    # Use direct solver for explicit matrices
                    if sparse.issparse(hessian):
                        d = splinalg.spsolve(hessian, -grad)
                    else:
                        # Try Cholesky first for stability
                        try:
                            L = np.linalg.cholesky(hessian)
                            d = np.linalg.solve(L.T, np.linalg.solve(L, -grad))
                        except np.linalg.LinAlgError:
                            # Fall back to standard solver if Cholesky fails
                            d = np.linalg.solve(hessian, -grad)
                
                # Apply trust region constraint if needed
                if trust_radius and np.linalg.norm(d) > trust_radius:
                    d = d * (trust_radius / np.linalg.norm(d))
                
                return d
            except Exception as e:
                if self.verbose:
                    print(f"Direct solver failed: {e}. Using subproblem solver.")
        
        # For non-smooth problems or if direct solver failed,
        # use the specified subproblem solver
        if self.subproblem_solver == 'prox-gradient':
            return self._solve_prox_gradient(x, grad, hessian, trust_radius)
        elif self.subproblem_solver == 'coordinate-descent':
            return self._solve_coordinate(x, grad, hessian, trust_radius)
        elif self.subproblem_solver == 'admm':
            return self._solve_admm(x, grad, hessian, trust_radius)
        else:
            raise ValueError(f"Unknown subproblem solver: {self.subproblem_solver}")

    def _solve_prox_gradient(self, x, grad, hessian, trust_radius):
        """
        Solve the subproblem using proximal gradient method.
        
        Parameters:
        -----------
        x : array-like
            Current iterate
        grad : array-like
            Current gradient
        hessian : array-like or LinearOperator
            Hessian or Hessian approximation
        trust_radius : float or None
            Trust region radius
            
        Returns:
        --------
        d : array-like
            Search direction
        """
        n = x.size
        d = np.zeros(n)
        
        # Estimate Lipschitz constant
        L = self._estimate_lipschitz_constant(hessian, n)
        step = 1.0 / max(L, 1e-10)
        
        # Use accelerated proximal gradient (FISTA)
        z = d.copy()
        t_prev = 1.0
        
        for k in range(self.subproblem_iter):
            # Compute gradient of the quadratic model at the current point
            if isinstance(hessian, splinalg.LinearOperator):
                hessd = hessian.matvec(z)
            else:
                hessd = hessian.dot(z)
            
            model_grad = grad + hessd
            
            # Take a proximal gradient step
            d_prev = d.copy()
            d = self.proximal_op(x + z - step * model_grad, step) - x
            
            # Apply trust region constraint if needed
            if trust_radius and np.linalg.norm(d) > trust_radius:
                d = d * (trust_radius / np.linalg.norm(d))
            
            # Check convergence
            if np.linalg.norm(d - d_prev) < self.subproblem_tol * max(np.linalg.norm(d), 1.0):
                break
            
            # FISTA update
            t = 0.5 * (1 + np.sqrt(1 + 4 * t_prev**2))
            z = d + ((t_prev - 1) / t) * (d - d_prev)
            t_prev = t
            
            # Reset acceleration if objective increases
            if k > 0 and k % 10 == 0:
                if isinstance(hessian, splinalg.LinearOperator):
                    hessz = hessian.matvec(z)
                    hessd = hessian.matvec(d)
                else:
                    hessz = hessian.dot(z)
                    hessd = hessian.dot(d)
                
                obj_z = grad.dot(z) + 0.5 * z.dot(hessz)
                obj_d = grad.dot(d) + 0.5 * d.dot(hessd)
                
                if obj_z > obj_d:
                    z = d.copy()
                    t_prev = 1.0
        
        return d

    def _estimate_lipschitz_constant(self, hessian, n):
        """
        Estimate the Lipschitz constant of the gradient of the quadratic model.
        
        Parameters:
        -----------
        hessian : array-like or LinearOperator
            Hessian or Hessian approximation
        n : int
            Dimension of the problem
            
        Returns:
        --------
        L : float
            Estimated Lipschitz constant
        """
        if isinstance(hessian, splinalg.LinearOperator):
            # Power iteration to estimate largest eigenvalue
            v = np.random.randn(n)
            v = v / np.linalg.norm(v)
            
            for _ in range(5):
                v_new = hessian.matvec(v)
                norm_v = np.linalg.norm(v_new)
                if norm_v > 1e-10:
                    v = v_new / norm_v
                else:
                    break
            
            L = v.dot(hessian.matvec(v))
        else:
            # Use scipy's eigsh for sparse matrices
            if sparse.issparse(hessian):
                try:
                    L = splinalg.eigsh(hessian, k=1, which='LM', return_eigenvectors=False)[0]
                except:
                    # Fall back to power iteration if eigsh fails
                    L = 10.0  # Default value
            else:
                # Use numpy's eigvalsh for dense matrices
                try:
                    L = np.max(np.linalg.eigvalsh(hessian))
                except:
                    # Fall back to a default value if eigenvalue computation fails
                    L = 10.0
        
        # Ensure L is positive
        return max(L, 1e-3)

    def _solve_coordinate(self, x, grad, hessian, trust_radius):
        """
        Solve the subproblem using coordinate descent.
        
        Parameters:
        -----------
        x : array-like
            Current iterate
        grad : array-like
            Current gradient
        hessian : array-like or LinearOperator
            Hessian or Hessian approximation
        trust_radius : float or None
            Trust region radius
            
        Returns:
        --------
        d : array-like
            Search direction
        """
        n = x.size
        d = np.zeros(n)
        
        # Get diagonal of Hessian
        if isinstance(hessian, splinalg.LinearOperator):
            hdiag = np.array([hessian.matvec(np.eye(n)[i])[i] for i in range(n)])
        elif sparse.issparse(hessian):
            hdiag = hessian.diagonal()
        else:
            hdiag = np.diag(hessian)
        
        # Ensure positive diagonal for stability
        hdiag = np.maximum(hdiag, 1e-6)
        
        for _ in range(self.subproblem_iter):
            # Randomize coordinate order for better convergence
            coords = np.random.permutation(n)
            max_change = 0
            
            for i in coords:
                # Compute residual at the current point
                if isinstance(hessian, splinalg.LinearOperator):
                    hessd = hessian.matvec(d)
                else:
                    hessd = hessian.dot(d)
                
                res = grad + hessd
                
                # Store old value and temporarily set to zero
                old_di = d[i]
                d[i] = 0
                
                # Compute the optimal value for this coordinate
                zi = -(res[i] - hdiag[i] * old_di) / hdiag[i]
                
                # Apply proximal operator to this coordinate
                d[i] = self._proximal_coordinate(zi, i, hdiag[i])
                
                # Track maximum change for convergence check
                max_change = max(max_change, abs(d[i] - old_di))
            
            # Apply trust region constraint if needed
            if trust_radius and np.linalg.norm(d) > trust_radius:
                d = d * (trust_radius / np.linalg.norm(d))
            
            # Check convergence
            if max_change < self.subproblem_tol * max(np.linalg.norm(d), 1.0):
                break
        
        return d

    def _solve_admm(self, x, grad, hessian, trust_radius):
        """
        Solve the subproblem using ADMM (Alternating Direction Method of Multipliers).
        
        Parameters:
        -----------
        x : array-like
            Current iterate
        grad : array-like
            Current gradient
        hessian : array-like or LinearOperator
            Hessian or Hessian approximation
        trust_radius : float or None
            Trust region radius
            
        Returns:
        --------
        d : array-like
            Search direction
        """
        n = x.size
        d = np.zeros(n)  # Primal variable
        z = np.zeros(n)  # Auxiliary variable
        u = np.zeros(n)  # Scaled dual variable
        rho = 1.0        # Penalty parameter
        
        # Factorize matrix if possible for efficiency
        linear_system_solver = None
        if not isinstance(hessian, splinalg.LinearOperator) and not sparse.issparse(hessian):
            try:
                H_rho = hessian + rho * np.eye(n)
                L = np.linalg.cholesky(H_rho)
                
                def solve_with_cholesky(b):
                    return np.linalg.solve(L.T, np.linalg.solve(L, b))
                
                linear_system_solver = solve_with_cholesky
            except:
                pass
        
        for k in range(self.subproblem_iter):
            # Update primal variable d
            q = z - u
            rhs = rho * q - grad
            
            if linear_system_solver is not None:
                # Use pre-factorized solver
                d = linear_system_solver(rhs)
            elif isinstance(hessian, splinalg.LinearOperator):
                # Use conjugate gradient for LinearOperator
                def matvec(v):
                    return hessian.matvec(v) + rho * v
                
                op = splinalg.LinearOperator((n, n), matvec=matvec)
                d, _ = splinalg.cg(op, rhs, x0=d, tol=1e-10)
            elif sparse.issparse(hessian):
                # Use sparse solver
                d = splinalg.spsolve(hessian + rho * sparse.eye(n), rhs)
            else:
                # Use dense solver
                d = np.linalg.solve(hessian + rho * np.eye(n), rhs)
            
            # Update auxiliary variable z
            z_old = z.copy()
            z = self.proximal_op(x + d + u, 1.0 / rho) - x
            
            # Apply trust region constraint if needed
            if trust_radius and np.linalg.norm(z) > trust_radius:
                z = z * (trust_radius / np.linalg.norm(z))
            
            # Update dual variable u
            u = u + d - z
            
            # Check convergence (primal and dual residuals)
            primal_res = np.linalg.norm(d - z)
            dual_res = np.linalg.norm(rho * (z - z_old))
            
            if primal_res < self.subproblem_tol and dual_res < self.subproblem_tol:
                break
            
            # Adapt rho based on residual balance
            if k > 0 and k % 10 == 0:
                if primal_res > 10 * dual_res:
                    rho *= 2.0
                    u /= 2.0
                elif dual_res > 10 * primal_res:
                    rho /= 2.0
                    u *= 2.0
        
        return z