import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# -------------------------
# Simulate HMM

def simulate_hmm(T=300):
    p00, p11 = 0.9, 0.8
    mu = np.array([0.0, 2.0])
    S = np.zeros(T, int)
    for t in range(1, T):
        if S[t-1] == 0:
            S[t] = 0 if random.random() < p00 else 1
        else:
            S[t] = 1 if random.random() < p11 else 0
    g = np.array([np.random.randn() + mu[S[t]] for t in range(T)])
    return S, g

def log_post(S, g):
    p00, p11 = 0.9, 0.8
    mu = np.array([0.0, 2.0])
    lp = 0.0
    T = len(S)
    for t in range(1, T):
        i, j = S[t-1], S[t]
        if i == j == 0:
            lp += np.log(p00)
        elif i == 0 and j == 1:
            lp += np.log(1-p00)
        elif i == 1 and j == 1:
            lp += np.log(p11)
        else:
            lp += np.log(1-p11)
    for t, val in enumerate(g):
        lp += norm.logpdf(val, loc=mu[S[t]], scale=1.0)
    return lp

# -------------------------
# Samplers

def gibbs(S, g):
    T = len(S)
    for t in range(T):
        # exact full‐conditional
        logp = np.array([ log_post(np.concatenate([S[:t],[s],S[t+1:]]), g) for s in [0,1] ])
        probs = np.exp(logp - logp.max())
        probs /= probs.sum()
        S[t] = np.random.choice([0,1], p=probs)
    return S

def dula(S, g, alpha):
    lp0 = log_post(S, g)
    T = len(S)
    for t in range(T):
        logp = np.array([ log_post(np.concatenate([S[:t],[s],S[t+1:]]), g) for s in [0,1] ])
        scores = np.exp((logp - lp0) / alpha)
        scores /= scores.sum()
        S[t] = np.random.choice([0,1], p=scores)
    return S

def dmala(S, g, alpha):
    T = len(S)
    for t in range(T):
        lp0 = log_post(S, g)
        # forward
        logp = np.array([ log_post(np.concatenate([S[:t],[s],S[t+1:]]), g) for s in [0,1] ])
        pf = np.exp((logp - lp0)/alpha); pf /= pf.sum()
        prop = np.random.choice([0,1], p=pf)
        # reverse
        S2 = S.copy(); S2[t] = prop
        lp1 = log_post(S2, g)
        logp2 = np.array([ log_post(np.concatenate([S2[:t],[s],S2[t+1:]]), g) for s in [0,1] ])
        pr = np.exp((logp2 - lp1)/alpha); pr /= pr.sum()
        qf, qr = pf[prop], pr[S[t]]
        ratio = np.exp(lp1 - lp0) * (qr/(qf+1e-12))
        if random.random() < min(1, ratio):
            S[t] = prop
    return S

# -------------------------
# Main: RMSE over multiple runs

os.makedirs("results", exist_ok=True)

T = 300
S0, g = simulate_hmm(T)
# True stationary P(S=1) = (1-p00)/(2 - p00 - p11)
true_pi1 = (1 - 0.9) / (2 - 0.9 - 0.8)

iterations = 100
runs = 10
alpha = 0.2  # fixed for DULA/DMALA

samplers = [
    ("Gibbs",   lambda S: gibbs(S.copy(), g)),
    ("DULA",    lambda S: dula(S.copy(), g, alpha)),
    ("DMALA",   lambda S: dmala(S.copy(), g, alpha))
]

# rmse_store[name][r, it] = error^2 for run r at iteration it
rmse_store = {name: np.zeros((runs, iterations)) for name, _ in samplers}

for name, fn in samplers:
    for r in range(runs):
        S = S0.copy()
        for it in range(iterations):
            S = fn(S)
            pi1_hat = S.mean()
            err = pi1_hat - true_pi1
            rmse_store[name][r, it] = err**2

# Compute RMSE(k) = sqrt(mean_r runs [ err^2(r,k) ]) and then log
x = np.arange(1, iterations+1)
plt.figure(figsize=(8,5))
for name in rmse_store:
    mse = rmse_store[name].mean(axis=0)
    rmse = np.sqrt(mse)
    plt.plot(x, np.log(rmse), label=name)
plt.xlabel("Iteration")
plt.ylabel("log RMSE")
plt.title("HMM: log RMSE of regime-1 frequency")
plt.legend()
plt.tight_layout()
plt.savefig("results/hmm_log_rmse_comparison.png")
plt.close()

print("Saved: results/hmm_log_rmse_comparison.png")
