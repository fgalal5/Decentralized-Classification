import numpy as np
from sklearn.metrics import accuracy_score
import cvxpy as cp
from itertools import combinations

# --- Task 1 ---
class MyDecentralized:
    def __init__(self, K, lam=1e-3, normalize=True, solver="HIGHS", verbose=False):
        self.K = int(K)
        self.lam = float(lam)
        self.normalize = bool(normalize)
        self.solver_name = solver
        self.verbose = bool(verbose)

        # Learned parameters
        self.W = None            
        self.b = None            
        self.mu = None           
        self.sig = None          

        # Label maps
        self.class_to_index = []
        self.index_to_class = []

    # ---------------------- helpers: labels ----------------------
    def _fit_label_maps(self, y):
        classes = np.unique(y).astype(int)
        classes = np.sort(classes)[:self.K]
        self.class_to_index = {c: i for i, c in enumerate(classes)}
        self.index_to_class = {i: c for c, i in self.class_to_index.items()}

    def _encode_y(self, y):
        return np.array([self.class_to_index[int(t)] for t in y], dtype=int)

    # ---------------------- helpers: scaling ----------------------
    def _standardize(self, X, fit=False):
        if not self.normalize:
            return X
        if fit:
            self.mu = X.mean(axis=0)
            self.sig = X.std(axis=0)
            self.sig[self.sig < 1e-8] = 1e-8 
        return (X - self.mu) / self.sig

    def _solve_lp(self, Xtr, ytr):
        import cvxpy as cp

        N, d = Xtr.shape
        K = self.K

        # Decision variables
        W = cp.Variable((K, d))
        b = cp.Variable(K)
        xi = cp.Variable(N, nonneg=True)
        T = cp.Variable((K, d), nonneg=True)  # |W|
        U = cp.Variable(K, nonneg=True)       # |b|

        constraints = [
            W <= T, -W <= T,     # absolute value for W
            b <= U, -b <= U      # absolute value for b
        ]

        # Multiclass hinge margin constraints
        for i in range(N):
            yi = ytr[i]
            for k in range(K):
                if k == yi:
                    continue
                constraints.append(
                    (W[yi] - W[k]) @ Xtr[i] + (b[yi] - b[k]) >= 1 - xi[i]
                )

        # Objective: average slack + L1 norms (linearized via T, U)
        obj = (cp.sum(xi) / N) + self.lam * (cp.sum(T) + cp.sum(U))
        prob = cp.Problem(cp.Minimize(obj), constraints)

        # trying a few solvers
        solver_order = []
        pref = self.solver_name.upper() if isinstance(self.solver_name, str) else None
        if pref in ("HIGHS", "GLPK", "CBC", "ECOS", "SCS"):
            solver_order.append(pref)
        for s in ("HIGHS", "GLPK", "CBC", "ECOS", "SCS"):
            if s not in solver_order:
                solver_order.append(s)

        name_to_solver = {n: getattr(cp, n, None) for n in solver_order}
        last_err = None
        for name in solver_order:
            S = name_to_solver.get(name)
            if S is None:
                continue
            try:
                if name == "GLPK":
                    prob.solve(solver=S, verbose=self.verbose, glpk={"tm_lim": 30000})
                elif name == "HIGHS":
                    prob.solve(solver=S, verbose=self.verbose, highs_options={"time_limit": 60})
                elif name == "ECOS":
                    prob.solve(solver=S, verbose=self.verbose, abstol=1e-6, reltol=1e-6, feastol=1e-6, max_iters=20000)
                elif name == "SCS":
                    prob.solve(solver=S, verbose=self.verbose, max_iters=8000, eps=1e-3)
                else:
                    prob.solve(solver=S, verbose=self.verbose)

                if prob.status in ("optimal", "optimal_inaccurate"):
                    return W.value, b.value
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"LP solve failed. status={prob.status}, last_err={last_err}")

    def train(self, trainX, trainY):
        X = np.asarray(trainX, dtype=float)
        y = np.asarray(trainY, dtype=int)

        # 1) Map labels to indices 0..K-1
        self._fit_label_maps(y)
        y_enc = self._encode_y(y)

        # 2) Standardize with train stats only
        Xz = self._standardize(X, fit=True)

        # 3) Solve the LP and store parameters
        W_hat, b_hat = self._solve_lp(Xz, y_enc)
        self.W, self.b = W_hat, b_hat

    def predict(self, testX):
        if self.W is None or self.b is None:
            raise ValueError("Model not trained yet. Call train() first.")

        X = np.asarray(testX, dtype=float)
        Xz = self._standardize(X, fit=False)
        scores = Xz @ self.W.T + self.b   # shape [n, K]
        pred_idx = np.argmax(scores, axis=1)

        # Map compact indices back to original labels
        return np.array([self.index_to_class[i] for i in pred_idx], dtype=int)

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        return accuracy_score(testY, predY)




##########################################################################
# --- Task 2 & Task 3 ---
class MyFeatureCompression:

    def __init__(self, K, lam=1e-2, solver="HIGHS", verbose=False):
        self.K = int(K)
        self.lam = float(lam)
        self.solver = solver
        self.verbose = bool(verbose)

        # quantizer variables
        self.x_min = None
        self.x_max = None
        self.x_range = None

    def _fit_scaler(self, X_train):
        self.x_min = X_train.min(axis=0)
        self.x_max = X_train.max(axis=0)
        # avoid divide-by-zero
        self.x_range = self.x_max - self.x_min
        self.x_range[self.x_range == 0.0] = 1.0

    def _normalize(self, X):
        return (X - self.x_min) / self.x_range

    def _denormalize(self, X_norm):
        return X_norm * self.x_range + self.x_min

    def _quantize_uniform(self, X, bits):
        if bits <= 0:
            # "0-bit" signal: everything goes to the mean -> degenerate, but defined
            X_mean = np.mean(X, axis=0, keepdims=True)
            return np.repeat(X_mean, X.shape[0], axis=0)

        X_norm = self._normalize(X)
        X_norm = np.clip(X_norm, 0.0, 1.0)

        L = 2 ** bits  # number of quantization levels
        # quantize in normalized domain
        X_q_norm = np.round(X_norm * (L - 1)) / (L - 1)

        X_q = self._denormalize(X_q_norm)
        return X_q


    def train_and_evaluate_for_bits(self, trainX, trainY, valX, valY, testX, testY, bits):
        if self.verbose:
            print(f"\n[Task 2] Running with bits={bits}")

        # Fit scaler on TRAIN data
        self._fit_scaler(trainX)

        # Quantize all splits
        q_trainX = self._quantize_uniform(trainX, bits)
        q_valX   = self._quantize_uniform(valX, bits)
        q_testX  = self._quantize_uniform(testX, bits)

        # Train classifier on quantized features
        clf = MyDecentralized(K=self.K, lam=self.lam, solver=self.solver, verbose=self.verbose)
        clf.train(q_trainX, trainY)

        test_acc = clf.evaluate(q_testX, testY)

        if self.verbose:
            print(f"[Task 2] bits={bits}, test accuracy={test_acc:.4f}")

        return test_acc

    # ----------------------- Task 2: centralized ---------------------------

    def run_centralized(self, trainX, trainY, valX, valY, testX, testY, B_tot_list):

        M = trainX.shape[1]  # features per image
        result = {'B_tot': [], 'test_accuracy': []}

        for B_req in B_tot_list:
            # bits per feature (integer)
            bits = max(0, B_req // M)
            # actual bits/image we end up using
            B_actual = bits * M

            acc = self.train_and_evaluate_for_bits(
                trainX, trainY,
                valX,   valY,
                testX,  testY,
                bits=bits,
            )

            if self.verbose:
                print(f"[Task 2] Requested B_tot={B_req}, bits={bits}, "
                      f"actual B_tot={B_actual}, test_acc={acc:.4f}")

            result['B_tot'].append(B_actual)
            result['test_accuracy'].append(acc)

        return result

    # ----------------------- Task 3.1: per-sensor k -----------------------

    def _quantize_blocks_equal_bits(self, train_blocks, val_blocks, test_blocks, bits_per_pixel):
        """
        Helper: decentralized quantization when *all* sensors use the same bit-depth.
        """
        S = len(train_blocks)
        q_train_blocks = []
        q_val_blocks   = []
        q_test_blocks  = []

        for s in range(S):
            # Fit scaler on single sensor's train data only
            self._fit_scaler(train_blocks[s])
            q_train_blocks.append(self._quantize_uniform(train_blocks[s], bits_per_pixel))
            q_val_blocks.append(self._quantize_uniform(val_blocks[s], bits_per_pixel))
            q_test_blocks.append(self._quantize_uniform(test_blocks[s], bits_per_pixel))

        q_trainX = np.concatenate(q_train_blocks, axis=1)
        q_valX   = np.concatenate(q_val_blocks,   axis=1)
        q_testX  = np.concatenate(q_test_blocks,  axis=1)
        return q_trainX, q_valX, q_testX

    def run_decentralized_per_sensor(self, train_blocks, val_blocks, test_blocks, trainY, valY, testY, k_list):
        S = len(train_blocks)
        _, d_s = train_blocks[0].shape

        result = {'k': [], 'test_accuracy': [], 'b_s': []}

        for k in k_list:
            bits_per_pixel = max(0, k // d_s)

            # quantize each sensor block with the same bits_per_pixel
            q_trainX, q_valX, q_testX = self._quantize_blocks_equal_bits(
                train_blocks, val_blocks, test_blocks, bits_per_pixel
            )

            # train classifier on quantized features
            clf = MyDecentralized(K=self.K, lam=self.lam, solver=self.solver, verbose=self.verbose)
            clf.train(q_trainX, trainY)

            test_acc = clf.evaluate(q_testX, testY)

            if self.verbose:
                B_per_sensor = bits_per_pixel * d_s
                B_tot = S * B_per_sensor
                print(f"[Task 3.1] k={k}, bits/pixel={bits_per_pixel}, "
                      f"B_per_sensor={B_per_sensor}, B_tot={B_tot}, "
                      f"test_acc={test_acc:.4f}")

            result['k'].append(k)
            result['test_accuracy'].append(test_acc)
            # record per-sensor bit-depths (all equal here)
            result['b_s'].append(tuple([bits_per_pixel] * S))

        return result

    # -------------------- Task 3.2: decentralized total B -----------------

    def _quantize_blocks_general(self, train_blocks, val_blocks, test_blocks, bits_per_sensor):
        S = len(train_blocks)
        assert len(bits_per_sensor) == S

        q_train_blocks = []
        q_val_blocks   = []
        q_test_blocks  = []

        for s in range(S):
            bits = int(bits_per_sensor[s])
            # Per-sensor scaler fit on THIS sensor's train data only
            self._fit_scaler(train_blocks[s])
            q_train_blocks.append(self._quantize_uniform(train_blocks[s], bits))

            if val_blocks is not None:
                q_val_blocks.append(self._quantize_uniform(val_blocks[s], bits))
            if test_blocks is not None:
                q_test_blocks.append(self._quantize_uniform(test_blocks[s], bits))

        q_trainX = np.concatenate(q_train_blocks, axis=1)
        q_valX   = np.concatenate(q_val_blocks,   axis=1) if val_blocks is not None else None
        q_testX  = np.concatenate(q_test_blocks,  axis=1) if test_blocks is not None else None

        return q_trainX, q_valX, q_testX


    def run_decentralized_total(self, train_blocks, val_blocks, test_blocks,
                            trainY, valY, testY, B_tot_list):
        S = len(train_blocks)
        _, d_s = train_blocks[0].shape
        M = S * d_s  # total number of features (for reference)

        result = {'B_tot': [], 'test_accuracy': [], 'best_allocation': []}

        for B_req in B_tot_list:
            # ---- 1. Build candidate allocations for this B_req ----
            if B_req <= 0:
                # trivial all-zero allocation
                candidate_allocations = [tuple([0] * S)]
            else:
                b_base = max(0, B_req // M)
                leftover_bits = B_req - (S * d_s * b_base)

                # Each +1 bit/pixel to a sensor costs d_s bits/image
                max_extra_sensors = min(S, leftover_bits // d_s)

                candidate_allocations = []
                # All sensors exactly b_base bits/pixel is always a candidate
                candidate_allocations.append(tuple([b_base] * S))

                # Also consider giving +1 bit/pixel to subsets of sensors
                for t in range(1, max_extra_sensors + 1):
                    for subset in combinations(range(S), t):
                        b_vec = [b_base] * S
                        for idx in subset:
                            b_vec[idx] += 1
                        candidate_allocations.append(tuple(b_vec))

            # ---- 2. Search over allocations using TRAIN + VAL only ----
            best_val_acc = -1.0
            best_alloc = (0,0,0,0)

            for alloc in candidate_allocations:
                # Quantize train and val for this allocation.
                # IMPORTANT: we pass test_blocks=None here so we don't touch test
                q_trainX, q_valX, _ = self._quantize_blocks_general(
                    train_blocks, val_blocks, None, alloc
                )

                clf = MyDecentralized(
                    K=self.K, lam=self.lam,
                    solver=self.solver, verbose=self.verbose
                )
                clf.train(q_trainX, trainY)
                val_acc = clf.evaluate(q_valX, valY)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_alloc = alloc

            # ---- 3. Once best_alloc is chosen, evaluate on TEST ----
            # Re-quantize train and test with the chosen allocation
            q_trainX, _, q_testX = self._quantize_blocks_general(
                train_blocks, None, test_blocks, best_alloc
            )

            clf = MyDecentralized(
                K=self.K, lam=self.lam,
                solver=self.solver, verbose=self.verbose
            )
            clf.train(q_trainX, trainY)
            best_test_acc = clf.evaluate(q_testX, testY)
            best_B_actual = d_s * sum(best_alloc)

            if self.verbose:
                print(f"[Task 3.2] Requested B_tot={B_req}, "
                    f"best_alloc={best_alloc}, "
                    f"actual B_tot={best_B_actual}, "
                    f"val_acc={best_val_acc:.4f}, "
                    f"test_acc={best_test_acc:.4f}")

            result['B_tot'].append(best_B_actual)
            result['test_accuracy'].append(best_test_acc)
            result['best_allocation'].append(best_alloc)

        return result





##########################################################################
##########################################################################
# --- Task 3.3 ---
class MyTargetAllocator:
    def __init__(self, K):
        self.K = K  # number of classes
        # TODO: add any state you need

    def minimal_bits_centralized(self, feature_compressor, trainX, trainY, valX, valY, testX, testY, alpha, B_grid):
        """
        Task 3.3 (Centralized)

        Goal:
            Given a target test accuracy α (e.g., 0.7, 0.8, 0.9), find the minimal total bit budget
            B (bits/image) so that your centralized formulation achieves test accuracy ≥ α.

        Allowed approaches (your choice, consistent with the guidelines):
            • Outer-search approach: use an outer search over candidate budgets and, for each,
              solve/evaluate your centralized formulation; pick the smallest B achieving ≥ α.
              In this case, `B_grid` provides the candidate budgets you intend to try (e.g., [784, 1568, 2352, 3136]).
            • Direct optimization approach: encode the minimal-bits objective directly in an LP/ILP/MILP
              that enforces accuracy ≥ α (as you define it) and solve for B. In this case, `B_grid` may be
              ignored or used as a search scaffold/initialization if helpful.

        Args:
            feature_compressor: an object exposing your centralized pipeline (e.g., MyFeatureCompression) if you
                choose to implement the outer-search approach. For a direct optimization approach, you may ignore it.
            trainX, trainY, valX, valY, testX, testY:
                datasets (keep train/val/test strict; no test leakage in model/quantizer selection).
            alpha (float): target test accuracy in [0,1].
            B_grid (Iterable[int]): candidate total budgets (bits/image) for the outer-search approach.
                If you implement a direct minimal-bits LP/MILP instead, you may ignore this or use it as a coarse grid.

        Returns:
            int or None:
                Minimal B (bits/image) achieving ≥ α under your centralized method; or None if not achievable
                within the search/constraints you used.

        Notes:
            • This method does not prescribe a particular quantizer or classifier; it only requires that you
              respect train/val/test separation and report bits/image clearly.
            • If multiple solutions achieve α, return the smallest B according to your method.
        """
        min_B = None
        return min_B

    def minimal_bits_decentralized(self, feature_compressor, train_blocks, val_blocks, test_blocks, trainY, valY, testY, alpha, B_grid):
        """
        Task 3.3 (Decentralized)

        Goal:
            Given a target test accuracy α, find the minimal total bit budget B (bits/image) and a corresponding
            decentralized allocation (e.g., per-sensor parameters such as (b1, b2, b3, b4), if that matches your design)
            so that your decentralized formulation achieves test accuracy ≥ α.

        Allowed approaches (your choice, consistent with the guidelines):
            • Outer-search approach: use an outer search over candidate budgets and, for each budget,
              search allocations/solve your decentralized formulation on train/val and evaluate on test;
              return the smallest B achieving ≥ α and its chosen allocation.
              In this case, `B_grid` provides the candidate budgets you intend to try.
            • Direct optimization approach: encode the minimal-bits objective directly in an LP/ILP/MILP
              with decentralized constraints and accuracy ≥ α; solve for B and its allocation.
              In this case, `B_grid` may be ignored or used to warm-start/coarsely bracket solutions.

        Args:
            feature_compressor: an object exposing your decentralized pipeline (e.g., MyFeatureCompression) if you
                follow the outer-search route. For a direct LP/MILP approach, you may ignore it.
            train_blocks, val_blocks, test_blocks:
                lists of 4 arrays [N × d_s], one per sensor/quadrant; keep train/val/test strict.
            trainY, valY, testY: labels.
            alpha (float): target test accuracy in [0,1].
            B_grid (Iterable[int]): candidate total budgets (bits/image) for the outer-search approach.
                If you implement a direct minimal-bits LP/MILP instead, you may ignore this or use it as a scaffold.

        Returns:
            (int or None, tuple or None):
                (minimal B, a representation of the chosen allocation at that B) if achievable; otherwise (None, None).
                The “allocation” is whatever your decentralized design uses (e.g., (b1, b2, b3, b4) for scalar bit-depths,
                or any quantizer-specific parameterization you choose to report).

        Notes:
            • This method does not prescribe how you search or solve; it only requires that you respect
              train/val/test separation and clearly report bits/image and the corresponding allocation.
            • If multiple solutions achieve α, return the one with the smallest B according to your method.
        """
        min_B, best_alloc = None,None
        return (min_B, best_alloc)
