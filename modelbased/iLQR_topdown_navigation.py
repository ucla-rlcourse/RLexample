"""
ilqr_topdown_navigation.py

Top-down 2D navigation (unicycle model) solved by iLQR (iterative LQR):
    min_{a_0..a_{T-1}} sum_t c(s_t, a_t)
    s.t. s_{t+1} = f(s_t, a_t)

Features:
- Unicycle dynamics s=[x,y,theta], a=[v,omega]
- Goal-reaching quadratic costs
- Smooth obstacle avoidance via sampled rectangle boundary points + Gaussian repulsion
- Random obstacles + a guaranteed "blocker" that forces a curved path
"""

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Dynamics: Unicycle
# -----------------------------
def f(s, a, dt):
    x, y, th = s
    v, w = a
    return np.array([
        x + dt * v * np.cos(th),
        y + dt * v * np.sin(th),
        th + dt * w
    ], dtype=float)


def linearize_f(s, a, dt):
    x, y, th = s
    v, w = a

    A = np.eye(3)
    A[0, 2] = -dt * v * np.sin(th)
    A[1, 2] =  dt * v * np.cos(th)

    B = np.zeros((3, 2))
    B[0, 0] = dt * np.cos(th)
    B[1, 0] = dt * np.sin(th)
    B[2, 1] = dt
    return A, B


# -----------------------------
# Obstacles: rectangles -> boundary points
# Smooth repulsive cost around boundary points:
#   c_obs(p) = w_obs * sum_i exp(-||p-q_i||^2/(2 sigma^2))
# Provide analytic gradient & Hessian wrt p
# -----------------------------
def rectangle_boundary_points(xmin, xmax, ymin, ymax, n_per_edge=30):
    xs = np.linspace(xmin, xmax, n_per_edge)
    ys = np.linspace(ymin, ymax, n_per_edge)
    pts = []
    # bottom/top
    pts += [(x, ymin) for x in xs]
    pts += [(x, ymax) for x in xs]
    # left/right
    pts += [(xmin, y) for y in ys]
    pts += [(xmax, y) for y in ys]
    return np.array(pts, dtype=float)


def obs_cost_grad_hess(p, obs_pts, w_obs, sigma):
    """
    p: (2,)
    obs_pts: (N,2)
    returns: cost scalar, grad (2,), hess (2,2)
    """
    d = p[None, :] - obs_pts            # (N,2)
    r2 = np.sum(d * d, axis=1)          # (N,)
    phi = np.exp(-r2 / (2.0 * sigma * sigma))  # (N,)

    c = w_obs * np.sum(phi)

    inv_s2 = 1.0 / (sigma * sigma)
    inv_s4 = inv_s2 * inv_s2

    # grad = w * sum phi * (-(p-q)/sigma^2)
    grad = w_obs * np.sum(phi[:, None] * (-(d) * inv_s2), axis=0)

    # Hessian = w * sum phi * ( (d d^T)/sigma^4 - I/sigma^2 )
    I = np.eye(2)
    hess = np.zeros((2, 2))
    for i in range(d.shape[0]):
        di = d[i][:, None]  # (2,1)
        hess += phi[i] * (di @ di.T * inv_s4 - I * inv_s2)
    hess *= w_obs

    return c, grad, hess


# -----------------------------
# Cost object: running + terminal and derivatives for iLQR
# state s = [x, y, theta], action a = [v, omega]
# -----------------------------
class Cost:
    def __init__(self, goal_xy, Q_run, Q_term, R, obs_pts, w_obs, sigma,
                 w_theta_term=0.0, theta_goal=0.0):
        self.g = np.array(goal_xy, dtype=float)
        self.Qr = Q_run    # 2x2 on (x,y)
        self.Qt = Q_term   # 2x2 on (x,y)
        self.R = R         # 2x2 on (v,omega)
        self.obs_pts = obs_pts
        self.w_obs = float(w_obs)
        self.sigma = float(sigma)
        self.w_theta_term = float(w_theta_term)
        self.theta_goal = float(theta_goal)

    def running(self, s, a):
        p = s[:2]
        dp = p - self.g
        c_goal = dp.T @ self.Qr @ dp
        c_u = a.T @ self.R @ a
        c_obs, _, _ = obs_cost_grad_hess(p, self.obs_pts, self.w_obs, self.sigma)
        return float(c_goal + c_u + c_obs)

    def terminal(self, s):
        p = s[:2]
        dp = p - self.g
        c_goal = dp.T @ self.Qt @ dp
        dth = s[2] - self.theta_goal
        c_th = self.w_theta_term * dth * dth
        c_obs, _, _ = obs_cost_grad_hess(p, self.obs_pts, self.w_obs, self.sigma)
        return float(c_goal + c_th + c_obs)

    def running_derivatives(self, s, a):
        # l(s,a) approx: l0 + lx^T ds + lu^T da + 0.5 ds^T lxx ds + 0.5 da^T luu da + da^T lux ds
        lx = np.zeros(3)
        lu = np.zeros(2)
        lxx = np.zeros((3, 3))
        luu = np.zeros((2, 2))
        lux = np.zeros((2, 3))

        # goal on (x,y)
        p = s[:2]
        dp = p - self.g
        lx[:2] += 2.0 * (self.Qr @ dp)
        lxx[:2, :2] += 2.0 * self.Qr

        # control
        lu += 2.0 * (self.R @ a)
        luu += 2.0 * self.R

        # obstacle (depends only on x,y)
        _, g_obs, H_obs = obs_cost_grad_hess(p, self.obs_pts, self.w_obs, self.sigma)
        lx[:2] += g_obs
        lxx[:2, :2] += H_obs

        l0 = self.running(s, a)
        return l0, lx, lu, lxx, luu, lux

    def terminal_derivatives(self, s):
        Vx = np.zeros(3)
        Vxx = np.zeros((3, 3))

        # goal on (x,y)
        p = s[:2]
        dp = p - self.g
        Vx[:2] += 2.0 * (self.Qt @ dp)
        Vxx[:2, :2] += 2.0 * self.Qt

        # theta
        dth = s[2] - self.theta_goal
        Vx[2] += 2.0 * self.w_theta_term * dth
        Vxx[2, 2] += 2.0 * self.w_theta_term

        # obstacle
        _, g_obs, H_obs = obs_cost_grad_hess(p, self.obs_pts, self.w_obs, self.sigma)
        Vx[:2] += g_obs
        Vxx[:2, :2] += H_obs

        V0 = self.terminal(s)
        return V0, Vx, Vxx


# -----------------------------
# iLQR core
# -----------------------------
def rollout(s0, a_seq, dt):
    T = a_seq.shape[0]
    s_seq = np.zeros((T + 1, 3))
    s_seq[0] = s0
    for t in range(T):
        s_seq[t + 1] = f(s_seq[t], a_seq[t], dt)
    return s_seq


def total_cost(cost_obj, s_seq, a_seq):
    T = a_seq.shape[0]
    J = 0.0
    for t in range(T):
        J += cost_obj.running(s_seq[t], a_seq[t])
    J += cost_obj.terminal(s_seq[T])
    return float(J)


def ilqr(s0, a_init, dt, cost_obj, n_iter=80, mu_init=1e-6, mu_factor=10.0,
         v_bounds=(0.0, 2.0), w_bounds=(-2.0, 2.0), verbose=True):
    """
    Simple iLQR with Levenberg-Marquardt regularization on Quu
    and line-search over alpha.
    """
    T = a_init.shape[0]
    a = a_init.copy()
    s = rollout(s0, a, dt)
    J = total_cost(cost_obj, s, a)

    mu = mu_init

    for it in range(n_iter):
        # Linearize dynamics and quadratize cost along nominal trajectory
        A = np.zeros((T, 3, 3))
        B = np.zeros((T, 3, 2))

        l0 = np.zeros(T)
        lx = np.zeros((T, 3))
        lu = np.zeros((T, 2))
        lxx = np.zeros((T, 3, 3))
        luu = np.zeros((T, 2, 2))
        lux = np.zeros((T, 2, 3))

        for t in range(T):
            A[t], B[t] = linearize_f(s[t], a[t], dt)
            l0[t], lx[t], lu[t], lxx[t], luu[t], lux[t] = cost_obj.running_derivatives(s[t], a[t])

        V0, Vx, Vxx = cost_obj.terminal_derivatives(s[T])

        # Backward pass
        k = np.zeros((T, 2))
        K = np.zeros((T, 2, 3))
        diverged = False

        for t in reversed(range(T)):
            At, Bt = A[t], B[t]

            Qx  = lx[t] + At.T @ Vx
            Qu  = lu[t] + Bt.T @ Vx
            Qxx = lxx[t] + At.T @ Vxx @ At
            Quu = luu[t] + Bt.T @ Vxx @ Bt
            Qux = lux[t] + Bt.T @ Vxx @ At

            Quu_reg = Quu + mu * np.eye(2)

            try:
                k[t] = -np.linalg.solve(Quu_reg, Qu)
                K[t] = -np.linalg.solve(Quu_reg, Qux)
            except np.linalg.LinAlgError:
                diverged = True
                break

            # value function update
            Vx  = Qx + K[t].T @ Quu @ k[t] + K[t].T @ Qu + Qux.T @ k[t]
            Vxx = Qxx + K[t].T @ Quu @ K[t] + K[t].T @ Qux + Qux.T @ K[t]
            Vxx = 0.5 * (Vxx + Vxx.T)

        if diverged:
            mu *= mu_factor
            continue

        # Forward line search
        accepted = False
        alphas = [1.0, 0.5, 0.25, 0.1, 0.05]
        for alpha in alphas:
            a_new = np.zeros_like(a)
            s_new = np.zeros_like(s)
            s_new[0] = s0

            for t in range(T):
                ds = s_new[t] - s[t]
                a_new[t] = a[t] + alpha * k[t] + K[t] @ ds

                # clamp
                a_new[t, 0] = np.clip(a_new[t, 0], v_bounds[0], v_bounds[1])
                a_new[t, 1] = np.clip(a_new[t, 1], w_bounds[0], w_bounds[1])

                s_new[t + 1] = f(s_new[t], a_new[t], dt)

            J_new = total_cost(cost_obj, s_new, a_new)
            if J_new < J:
                a, s, J = a_new, s_new, J_new
                accepted = True
                mu = max(mu / mu_factor, 1e-9)
                break

        if not accepted:
            mu *= mu_factor

        if verbose and (it % 5 == 0 or it == n_iter - 1):
            print(f"iter {it:02d}  J={J:.3f}  mu={mu:.1e}  accepted={accepted}")

        # stop when update is tiny
        if accepted and np.linalg.norm(k) < 1e-3:
            if verbose:
                print("Converged (small update).")
            break

    return s, a


# -----------------------------
# Random obstacle generation (with a guaranteed blocker)
# -----------------------------
def rect_too_close_to_point(rect, p, margin=0.40):
    xmin, xmax, ymin, ymax = rect
    return (p[0] >= xmin - margin) and (p[0] <= xmax + margin) and (p[1] >= ymin - margin) and (p[1] <= ymax + margin)


def generate_random_obstacles(start_xy, goal_xy, n_obs=5, seed=2):
    rng = np.random.default_rng(seed)

    XMIN, XMAX = 0.0, 10.0
    YMIN, YMAX = 0.0, 6.0

    obstacles = []

    # Guaranteed blocker in the middle -> forces curved trajectory
    obstacles.append((4.2, 6.2, 2.0, 4.8))

    tries = 0
    while len(obstacles) < n_obs and tries < 3000:
        tries += 1
        w = rng.uniform(0.6, 1.4)
        h = rng.uniform(0.6, 1.6)
        xmin = rng.uniform(XMIN + 0.2, XMAX - w - 0.2)
        ymin = rng.uniform(YMIN + 0.2, YMAX - h - 0.2)
        rect = (xmin, xmin + w, ymin, ymin + h)

        if rect_too_close_to_point(rect, start_xy) or rect_too_close_to_point(rect, goal_xy):
            continue

        # avoid heavy overlaps (simple area overlap check)
        ok = True
        for (ax0, ax1, ay0, ay1) in obstacles:
            bx0, bx1, by0, by1 = rect
            overlap_x = (ax0 <= bx1) and (bx0 <= ax1)
            overlap_y = (ay0 <= by1) and (by0 <= ay1)
            if overlap_x and overlap_y:
                inter_x0 = max(ax0, bx0); inter_x1 = min(ax1, bx1)
                inter_y0 = max(ay0, by0); inter_y1 = min(ay1, by1)
                inter_area = max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)
                if inter_area > 0.20:
                    ok = False
                    break
        if not ok:
            continue

        obstacles.append(rect)

    return obstacles


# -----------------------------
# Plotting
# -----------------------------
def plot_scene(ax, obstacles, start, goal_xy):
    for (xmin, xmax, ymin, ymax) in obstacles:
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=True, alpha=0.25))
    ax.plot(start[0], start[1], "o", label="start")
    ax.plot(goal_xy[0], goal_xy[1], "*", markersize=14, label="goal")
    ax.set_aspect("equal", "box")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.grid(True)


# -----------------------------
# Main demo
# -----------------------------
def main():
    # Start and corner goal
    start = np.array([1.0, 1.0, 0.0], dtype=float)
    goal_xy = np.array([9.4, 5.6], dtype=float)

    # Random obstacles + blocker
    obstacles = generate_random_obstacles(start[:2], goal_xy, n_obs=5, seed=1)

    # Build obstacle points (dense boundary sampling)
    obs_pts = []
    for rect in obstacles:
        obs_pts.append(rectangle_boundary_points(*rect, n_per_edge=60))
    obs_pts = np.vstack(obs_pts)

    # Horizon
    dt = 0.1
    T = 100

    # Costs (tune these if you want more/less aggressive obstacle avoidance)
    Q_run  = np.diag([1.5, 1.5])
    Q_term = np.diag([400.0, 400.0])
    R      = np.diag([0.05, 0.03])

    cost_obj = Cost(
        goal_xy=goal_xy,
        Q_run=Q_run,
        Q_term=Q_term,
        R=R,
        obs_pts=obs_pts,
        w_obs=200.0,
        sigma=0.22,
        w_theta_term=0.0
    )

    # Initial guess: move forward, initial turning toward goal direction
    a0 = np.zeros((T, 2), dtype=float)
    a0[:, 0] = 1.2

    heading_to_goal = np.arctan2(goal_xy[1] - start[1], goal_xy[0] - start[0])
    a0[:25, 1] = np.clip((heading_to_goal - start[2]) / (25 * dt), -1.0, 1.0)
    a0[25:, 1] = 0.0

    s_opt, a_opt = ilqr(start, a0, dt, cost_obj, n_iter=80, verbose=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    plot_scene(ax, obstacles, start, goal_xy)
    ax.plot(s_opt[:, 0], s_opt[:, 1], "-", linewidth=2, label="iLQR traj")
    ax.legend()
    ax.set_title("Top-down navigation via iLQR (iterative LQR)")
    plt.show()


if __name__ == "__main__":
    main()

