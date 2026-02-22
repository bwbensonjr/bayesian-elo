"""Glickman (2025) approximate filtering algorithm for Bayesian Elo ratings.

Implements Sections 4.1-4.3 of the paper: opponent-prior approximation,
2-point Gauss-Hermite quadrature, and one-step Newton-Raphson posterior
updates. No PyMC dependency; all updates are closed-form.

Model (Eq 3.1):
    P(home win)  ~ exp(theta_home + x*(alpha0 + alpha1*avg_theta)/4)
    P(away win)  ~ exp(theta_away - x*(alpha0 + alpha1*avg_theta)/4)
    P(draw)      ~ exp(beta0 + (1 + beta1)*avg_theta)
    Normalized via softmax.

Time evolution (Eq 3.2):
    theta_{i,t+1} ~ N(theta_{i,t}, tau^2)

Rating conversion (Eq 6.1):
    R = 1500 + 173.72 * theta
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


RATING_SCALE = 173.72  # 400 / log(10), per Eq 6.1
RATING_BASELINE = 1500


# ---------------------------------------------------------------------------
# 1. System parameters
# ---------------------------------------------------------------------------

@dataclass
class SystemParams:
    """Fixed system parameters for the Glickman model.

    These are treated as known inputs (not sampled), matching the paper's
    treatment in Sections 4-5. They can be optimized via predictive
    log-likelihood (Section 5).
    """

    alpha0: float = 0.21
    alpha1: float = 0.0
    beta0: float = -0.39
    beta1: float = 0.12
    tau: float = 0.15
    sigma_init: float = 0.5
    sigma_cap: float = 1.5


# ---------------------------------------------------------------------------
# 2. Core math: softmax probabilities and derivatives (Eq 3.1)
# ---------------------------------------------------------------------------

def _softmax_probs(theta_i, theta_j, x_i, params):
    """Compute (p_win_i, p_draw, p_loss_i) per Eq 3.1.

    Parameters
    ----------
    theta_i, theta_j : float
        Player strengths.
    x_i : float
        Home indicator for player i (+1 if home, -1 if away).
    params : SystemParams

    Returns
    -------
    np.ndarray of shape (3,): [p_win_i, p_draw, p_loss_i]
    """
    avg = (theta_i + theta_j) / 2
    ha_term = x_i * (params.alpha0 + params.alpha1 * avg) / 4

    lam_w = theta_i + ha_term
    lam_d = params.beta0 + (1 + params.beta1) * avg
    lam_l = theta_j - ha_term

    logits = np.array([lam_w, lam_d, lam_l])
    logits -= logits.max()  # log-sum-exp stability
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()


def _prob_derivatives(theta_i, theta_j, x_i, params):
    """First and second derivatives of softmax probs w.r.t. theta_i.

    Uses the identities:
        a_w = 1 + x_i*alpha1/8
        a_l = -x_i*alpha1/8
        a_d = (1 + beta1)/2
        dp_k/dtheta_i = p_k * (a_k - a_bar)
        d2p_k/dtheta_i^2 = p_k * ((a_k - a_bar)^2 - Var_p(a))

    Returns
    -------
    probs : ndarray (3,)
    dp : ndarray (3,) -- first derivatives
    d2p : ndarray (3,) -- second derivatives
    """
    probs = _softmax_probs(theta_i, theta_j, x_i, params)

    a_w = 1.0 + x_i * params.alpha1 / 8.0
    a_d = (1.0 + params.beta1) / 2.0
    a_l = -x_i * params.alpha1 / 8.0
    a = np.array([a_w, a_d, a_l])

    a_bar = (probs * a).sum()
    diff = a - a_bar
    var_a = (probs * diff ** 2).sum()

    dp = probs * diff
    d2p = probs * (diff ** 2 - var_a)

    return probs, dp, d2p


# ---------------------------------------------------------------------------
# 3. 2-point Gauss-Hermite quadrature (Section 4.2, Eq 4.7)
# ---------------------------------------------------------------------------

def _u_and_derivatives(theta_i, mu_j, sigma_j, x_i, y_idx, params):
    """Integrated likelihood and derivatives via 2-point Gauss-Hermite.

    Evaluates at quadrature points mu_j +/- sigma_j with equal weights 1/2
    (Eq 4.7).

    Parameters
    ----------
    theta_i : float
        Current evaluation point for player i's strength.
    mu_j, sigma_j : float
        Opponent's prior mean and std.
    x_i : float
        Home indicator for player i.
    y_idx : int
        Outcome index in player-i perspective: 0=win, 1=draw, 2=loss.
    params : SystemParams

    Returns
    -------
    u, du, d2u : float
        Integrated likelihood value, first and second derivatives.
    """
    u = 0.0
    du = 0.0
    d2u = 0.0
    for sign in (-1.0, 1.0):
        theta_j = mu_j + sign * sigma_j
        probs, dp, d2p = _prob_derivatives(theta_i, theta_j, x_i, params)
        u += 0.5 * probs[y_idx]
        du += 0.5 * dp[y_idx]
        d2u += 0.5 * d2p[y_idx]
    return u, du, d2u


# ---------------------------------------------------------------------------
# 4. Newton-Raphson update (Section 4.3, Eq 4.10-4.11)
# ---------------------------------------------------------------------------

def _update_player(mu_i, sigma_i, games, params):
    """One-step Newton-Raphson posterior update (Eq 4.10-4.11).

    Parameters
    ----------
    mu_i, sigma_i : float
        Player's prior mean and std.
    games : list of (mu_j, sigma_j, x_i, y_idx) tuples
        Each opponent's prior stats, home indicator for player i, and
        observed outcome from player i's perspective.
    params : SystemParams

    Returns
    -------
    mu_star, sigma_star : float
        Posterior mean and std.
    """
    if not games:
        return mu_i, sigma_i

    # Gradient: prior term vanishes at theta_i = mu_i
    g = 0.0
    # Hessian: starts with -1/sigma_i^2 from prior
    h = -1.0 / sigma_i ** 2

    for mu_j, sigma_j, x_i, y_idx in games:
        u, du, d2u = _u_and_derivatives(mu_i, mu_j, sigma_j, x_i, y_idx, params)
        if u > 1e-15:
            g += du / u
            h += (u * d2u - du ** 2) / u ** 2

    # Ensure h is negative (log-posterior should be concave at the mode)
    if h >= 0:
        return mu_i, sigma_i

    mu_star = mu_i - g / h
    sigma_star = np.sqrt(-1.0 / h)

    return mu_star, sigma_star


# ---------------------------------------------------------------------------
# 5. FilterState
# ---------------------------------------------------------------------------

class FilterState:
    """Stores per-team mu/sigma arrays and history across periods.

    Attributes
    ----------
    mu, sigma : ndarray (n_teams,)
        Current mean and std for each team.
    history : list of (mu_prior, sigma_prior, mu_post, sigma_post) tuples
        One entry per period, each element is ndarray (n_teams,).
    """

    def __init__(self, n_teams, params):
        self.n_teams = n_teams
        self.params = params
        self.mu = np.zeros(n_teams)
        self.sigma = np.full(n_teams, params.sigma_init)
        self.history = []

    def advance_period(self):
        """Apply time evolution (Eq 4.3) with sigma cap (Section 6.2)."""
        self.sigma = np.sqrt(self.sigma ** 2 + self.params.tau ** 2)
        np.clip(self.sigma, 0, self.params.sigma_cap, out=self.sigma)


# ---------------------------------------------------------------------------
# 6. Main filter loop
# ---------------------------------------------------------------------------

def run_filter(data, params):
    """Run the Glickman filtering algorithm over all periods.

    For each period: collect games, update each player via Newton-Raphson
    using opponent PRIORS (Section 4.1), then advance to the next period.

    Parameters
    ----------
    data : dict
        Output from bayesian_elo.prepare_data().
    params : SystemParams

    Returns
    -------
    FilterState
    """
    n_teams = data["n_teams"]
    n_periods = data["n_periods"]
    home_idx = data["home_idx"]
    away_idx = data["away_idx"]
    period_idx = data["period_idx"]
    outcome = data["outcome"]

    state = FilterState(n_teams, params)

    for t in range(n_periods):
        # Record priors for this period
        mu_prior = state.mu.copy()
        sigma_prior = state.sigma.copy()

        # Collect games in this period
        game_indices = np.where(period_idx == t)[0]

        # Build per-player game lists using PRIORS (Section 4.1)
        player_games = [[] for _ in range(n_teams)]

        for g in game_indices:
            h = home_idx[g]
            a = away_idx[g]
            y = outcome[g]  # A=0, D=1, H=2

            # Home player: x_i=1
            # y_idx: H=2 -> 0 (win), D=1 -> 1 (draw), A=0 -> 2 (loss)
            y_home = 2 - y
            player_games[h].append(
                (mu_prior[a], sigma_prior[a], 1.0, y_home)
            )

            # Away player: x_i=-1
            # y_idx: A=0 -> 0 (win), D=1 -> 1 (draw), H=2 -> 2 (loss)
            y_away = y
            player_games[a].append(
                (mu_prior[h], sigma_prior[h], -1.0, y_away)
            )

        # Update each player independently
        for i in range(n_teams):
            if player_games[i]:
                state.mu[i], state.sigma[i] = _update_player(
                    mu_prior[i], sigma_prior[i], player_games[i], params
                )

        # Store history
        state.history.append((
            mu_prior, sigma_prior,
            state.mu.copy(), state.sigma.copy(),
        ))

        # Advance to next period (except after the last)
        if t < n_periods - 1:
            state.advance_period()

    return state


# ---------------------------------------------------------------------------
# 7. Output functions matching bayesian_elo.py API
# ---------------------------------------------------------------------------

def _gauss_hermite_nodes_weights(n):
    """Normalized Gauss-Hermite nodes and weights for integration over N(mu, sigma^2).

    Returns nodes z_k and weights w_k such that:
        E[f(theta)] ~ sum_k w_k * f(mu + sqrt(2)*sigma*z_k)
    """
    nodes, weights = np.polynomial.hermite.hermgauss(n)
    weights = weights / np.sqrt(np.pi)
    return nodes, weights


def predict_outcomes(state, data, params):
    """Compute predicted outcome probabilities.

    Uses 9-point Gauss-Hermite quadrature (3 per dimension) to integrate
    over both players' prior distributions for each game (Section 5).

    Parameters
    ----------
    state : FilterState
    data : dict
        Output from prepare_data().
    params : SystemParams

    Returns
    -------
    DataFrame with columns: p_away_win, p_draw, p_home_win, pred_outcome,
    actual_outcome
    """
    home_idx = data["home_idx"]
    away_idx = data["away_idx"]
    period_idx = data["period_idx"]
    n_matches = len(home_idx)

    # 3-point GH quadrature per dimension -> 9 points total
    gh_nodes, gh_weights = _gauss_hermite_nodes_weights(3)
    sqrt2 = np.sqrt(2.0)

    p_home_win = np.zeros(n_matches)
    p_draw = np.zeros(n_matches)
    p_away_win = np.zeros(n_matches)

    for g in range(n_matches):
        t = period_idx[g]
        h = home_idx[g]
        a = away_idx[g]

        # Use priors for the period
        mu_h, sigma_h = state.history[t][0][h], state.history[t][1][h]
        mu_a, sigma_a = state.history[t][0][a], state.history[t][1][a]

        ph = 0.0
        pd_ = 0.0
        pa = 0.0

        for zi, wi in zip(gh_nodes, gh_weights):
            theta_h = mu_h + sqrt2 * sigma_h * zi
            for zj, wj in zip(gh_nodes, gh_weights):
                theta_a = mu_a + sqrt2 * sigma_a * zj
                w = wi * wj
                # Compute from home perspective (x=1)
                probs = _softmax_probs(theta_h, theta_a, 1.0, params)
                # probs = [p_win_home, p_draw, p_loss_home(=away_win)]
                ph += w * probs[0]
                pd_ += w * probs[1]
                pa += w * probs[2]

        p_home_win[g] = ph
        p_draw[g] = pd_
        p_away_win[g] = pa

    outcome_labels = {0: "A", 1: "D", 2: "H"}
    p_all = np.stack([p_away_win, p_draw, p_home_win], axis=1)
    pred_outcome = [outcome_labels[i] for i in np.argmax(p_all, axis=1)]
    actual_outcome = [outcome_labels[o] for o in data["outcome"]]

    return pd.DataFrame({
        "p_away_win": p_away_win,
        "p_draw": p_draw,
        "p_home_win": p_home_win,
        "pred_outcome": pred_outcome,
        "actual_outcome": actual_outcome,
    })


def extract_ratings(state, data, period=-1):
    """Convert filter estimates to Elo-scale ratings.

    Parameters
    ----------
    state : FilterState
    data : dict
    period : int
        Which period to extract (-1 = last).

    Returns
    -------
    DataFrame with columns: team, rating_mean, rating_std, ci_lower, ci_upper
    """
    if period < 0:
        period = len(state.history) + period

    mu_post = state.history[period][2]
    sigma_post = state.history[period][3]

    rating_mean = RATING_BASELINE + RATING_SCALE * mu_post
    rating_std = RATING_SCALE * sigma_post
    ci_lower = rating_mean - 1.96 * rating_std
    ci_upper = rating_mean + 1.96 * rating_std

    return pd.DataFrame({
        "team": data["team_names"],
        "rating_mean": rating_mean,
        "rating_std": rating_std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }).sort_values("rating_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 8. Parameter optimization (Section 5)
# ---------------------------------------------------------------------------

def predictive_log_likelihood(data, params):
    """One-step-ahead predictive log-likelihood (Section 5).

    Runs the filter and at each game computes the log-probability of the
    actual outcome under the prior predictive distribution (integrating over
    both players' priors with 3-point Gauss-Hermite).

    Parameters
    ----------
    data : dict
    params : SystemParams

    Returns
    -------
    float
    """
    n_teams = data["n_teams"]
    n_periods = data["n_periods"]
    home_idx = data["home_idx"]
    away_idx = data["away_idx"]
    period_idx = data["period_idx"]
    outcome = data["outcome"]

    gh_nodes, gh_weights = _gauss_hermite_nodes_weights(3)
    sqrt2 = np.sqrt(2.0)

    mu = np.zeros(n_teams)
    sigma = np.full(n_teams, params.sigma_init)

    total_ll = 0.0

    for t in range(n_periods):
        mu_prior = mu.copy()
        sigma_prior = sigma.copy()

        game_indices = np.where(period_idx == t)[0]

        # Compute predictive log-likelihood for each game using priors
        for g in game_indices:
            h = home_idx[g]
            a = away_idx[g]
            y = outcome[g]

            # Integrate over both players' priors (9-point GH)
            p_pred = np.zeros(3)  # [p_away_win, p_draw, p_home_win]

            for zi, wi in zip(gh_nodes, gh_weights):
                theta_h = mu_prior[h] + sqrt2 * sigma_prior[h] * zi
                for zj, wj in zip(gh_nodes, gh_weights):
                    theta_a = mu_prior[a] + sqrt2 * sigma_prior[a] * zj
                    w = wi * wj
                    probs = _softmax_probs(theta_h, theta_a, 1.0, params)
                    # probs = [p_win_home, p_draw, p_loss_home]
                    # Map to data outcome encoding: A=0, D=1, H=2
                    p_pred[0] += w * probs[2]  # away win = home loss
                    p_pred[1] += w * probs[1]  # draw
                    p_pred[2] += w * probs[0]  # home win

            total_ll += np.log(max(p_pred[y], 1e-15))

        # Run filter update for this period
        player_games = [[] for _ in range(n_teams)]
        for g in game_indices:
            h = home_idx[g]
            a = away_idx[g]
            y = outcome[g]
            player_games[h].append(
                (mu_prior[a], sigma_prior[a], 1.0, 2 - y)
            )
            player_games[a].append(
                (mu_prior[h], sigma_prior[h], -1.0, y)
            )

        for i in range(n_teams):
            if player_games[i]:
                mu[i], sigma[i] = _update_player(
                    mu_prior[i], sigma_prior[i], player_games[i], params
                )

        if t < n_periods - 1:
            sigma = np.sqrt(sigma ** 2 + params.tau ** 2)
            np.clip(sigma, 0, params.sigma_cap, out=sigma)

    return total_ll


def optimize_params(data, initial_params=None, maxiter=200):
    """Optimize system parameters by maximizing predictive log-likelihood.

    Uses Nelder-Mead optimization (Section 5). scipy is imported lazily.

    Parameters
    ----------
    data : dict
    initial_params : SystemParams, optional
    maxiter : int

    Returns
    -------
    SystemParams
    """
    from scipy.optimize import minimize

    if initial_params is None:
        initial_params = SystemParams()

    # Pack parameters into a vector (tau and sigma_init in log space)
    x0 = np.array([
        initial_params.alpha0,
        initial_params.alpha1,
        initial_params.beta0,
        initial_params.beta1,
        np.log(initial_params.tau),
        np.log(initial_params.sigma_init),
    ])

    def objective(x):
        p = SystemParams(
            alpha0=x[0],
            alpha1=x[1],
            beta0=x[2],
            beta1=max(x[3], 1e-6),  # beta1 >= 0
            tau=np.exp(x[4]),
            sigma_init=np.exp(x[5]),
            sigma_cap=initial_params.sigma_cap,
        )
        return -predictive_log_likelihood(data, p)

    result = minimize(
        objective, x0, method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-4},
    )

    x = result.x
    optimized = SystemParams(
        alpha0=x[0],
        alpha1=x[1],
        beta0=x[2],
        beta1=max(x[3], 1e-6),
        tau=np.exp(x[4]),
        sigma_init=np.exp(x[5]),
        sigma_cap=initial_params.sigma_cap,
    )
    return optimized


# ---------------------------------------------------------------------------
# 9. GlickmanPredictor class
# ---------------------------------------------------------------------------

class GlickmanPredictor:
    """Stateful wrapper for incremental Glickman filter prediction.

    Analogous to BayesianPredictor but uses the fast filtering algorithm
    instead of MCMC.

    Parameters
    ----------
    historical_df : DataFrame
        Historical match data (season, date, home_team, away_team, result).
    params : SystemParams, optional
        System parameters. Defaults to SystemParams().
    """

    def __init__(self, historical_df, params=None):
        from bayesian_elo import prepare_data
        self.df = historical_df.copy()
        self.params = params or SystemParams()
        self.state = None
        self.data = None
        self._prepare_data = prepare_data

    def fit(self):
        """Run the filter on historical data."""
        self.data = self._prepare_data(self.df)
        self.state = run_filter(self.data, self.params)

    def predict(self, games_df):
        """Predict upcoming games using current filter state.

        Uses the last period's posterior as a prior for prediction,
        integrated via 9-point Gauss-Hermite quadrature.

        Parameters
        ----------
        games_df : DataFrame
            Games with home_team, away_team, result columns.

        Returns
        -------
        DataFrame with p_away_win, p_draw, p_home_win, pred_outcome,
        actual_outcome columns.
        """
        team_to_idx = self.data["team_to_idx"]
        n_teams = self.data["n_teams"]

        # Get the last period's posterior
        mu_post = self.state.history[-1][2]
        sigma_post = self.state.history[-1][3]

        # Handle unseen teams (assign mean rating with initial sigma)
        all_teams = set(games_df["home_team"]) | set(games_df["away_team"])
        unseen = all_teams - set(team_to_idx)
        sentinel = n_teams

        if unseen:
            mu_ext = np.append(mu_post, 0.0)
            sigma_ext = np.append(sigma_post, self.params.sigma_init)
            team_names = list(self.data["team_names"]) + sorted(unseen)
        else:
            mu_ext = mu_post
            sigma_ext = sigma_post
            team_names = self.data["team_names"]

        home_idx = np.array([
            team_to_idx.get(t, sentinel) for t in games_df["home_team"]
        ])
        away_idx = np.array([
            team_to_idx.get(t, sentinel) for t in games_df["away_team"]
        ])

        outcome_map = {"A": 0, "D": 1, "H": 2}
        outcomes = np.array([outcome_map[r] for r in games_df["result"]])

        n_games = len(home_idx)

        # Create a temporary state with a single "period" holding the
        # posteriors as priors for prediction
        temp_state = FilterState(len(mu_ext), self.params)
        temp_state.mu = mu_ext.copy()
        temp_state.sigma = sigma_ext.copy()
        temp_state.history = [(
            mu_ext.copy(), sigma_ext.copy(),
            mu_ext.copy(), sigma_ext.copy(),
        )]

        temp_data = {
            "home_idx": home_idx,
            "away_idx": away_idx,
            "period_idx": np.zeros(n_games, dtype=int),
            "outcome": outcomes,
            "n_teams": len(mu_ext),
            "n_periods": 1,
            "team_names": team_names,
        }

        return predict_outcomes(temp_state, temp_data, self.params)

    def update(self, completed_games_df):
        """Append results and rerun filter."""
        self.df = pd.concat(
            [self.df, completed_games_df], ignore_index=True
        )
        self.fit()
