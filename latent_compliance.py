import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# -------------------------
# Simulate compliance data

def simulate_compliance_data(n=100):
    pi = np.ones(4) / 4
    mu = np.array([1.0, 0.0])
    Z = np.random.binomial(1, 0.5, size=n)
    C = np.random.choice(4, size=n, p=pi)
    D = np.zeros(n, int)
    for i in range(n):
        if   C[i] == 0: D[i] = 1
        elif C[i] == 1: D[i] = 0
        elif C[i] == 2: D[i] = Z[i]
        else:            D[i] = 1 - Z[i]
    Y = mu[D] + np.random.randn(n)
    return Z, C, D, Y

def log_post(C, Z, D, Y):
    lp = len(C) * np.log(0.25)
    for i, c in enumerate(C):
        di, zi = D[i], Z[i]
        ok = ((c==0 and di==1) or (c==1 and di==0)
              or (c==2 and di==zi) or (c==3 and di==1-zi))
        lp += 0 if ok else -1e6
        lp += norm.logpdf(Y[i], loc=[1.0,0.0][di], scale=1.0)
    return lp

# Gibbs, DULA, DMALA for compliance

def gibbs_compliance(C, Z, D, Y):
    n = len(C)
    for i in range(n):
        logp = np.zeros(4)
        for c in range(4):
            C2 = C.copy(); C2[i] = c
            logp[c] = log_post(C2, Z, D, Y)
        p = np.exp(logp - logp.max()); p /= p.sum()
        C[i] = np.random.choice(4, p=p)
    return C

def dula_compliance(C, Z, D, Y, alpha):
    lp0 = log_post(C, Z, D, Y)
    n = len(C)
    for i in range(n):
        logp = np.zeros(4)
        for c in range(4):
            C2 = C.copy(); C2[i] = c
            logp[c] = log_post(C2, Z, D, Y)
        scores = np.exp((logp - lp0) / alpha)
        scores /= scores.sum()
        C[i] = np.random.choice(4, p=scores)
    return C

def dmala_compliance(C, Z, D, Y, alpha):
    n = len(C)
    for i in range(n):
        lp0 = log_post(C, Z, D, Y)
        # forward
        logp = np.zeros(4)
        for c in range(4):
            C2 = C.copy(); C2[i] = c
            logp[c] = log_post(C2, Z, D, Y)
        pf = np.exp((logp - lp0) / alpha)
        pf /= pf.sum()
        prop = np.random.choice(4, p=pf)
        # reverse
        C2 = C.copy(); C2[i] = prop
        lp1 = log_post(C2, Z, D, Y)
        logp2 = np.array([
            log_post(np.concatenate([C2[:i],[c],C2[i+1:]]), Z, D, Y)
            for c in range(4)
        ])
        pr = np.exp((logp2 - lp1) / alpha)
        pr /= pr.sum()
        qf, qr = pf[prop], pr[C[i]]
        ratio = np.exp(lp1 - lp0) * (qr/(qf + 1e-12))
        if random.random() < min(1, ratio):
            C[i] = prop
    return C

# -------------------------
# Utility for plotting

def plot_and_save(x, ys, labels, xlabel, ylabel, title, fname):
    plt.figure(figsize=(8,5))
    for y, lab in zip(ys, labels):
        plt.plot(x, y, label=lab)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig(fname)
    plt.close()

# -------------------------
# Main: RMSE over multiple runs

os.makedirs("results", exist_ok=True)

# simulate once
Z, C0, D, Y = simulate_compliance_data(n=300)

# approximate true proportion of Compliers (type==2) via long Gibbs run
C_ref = C0.copy()
for _ in range(2000):
    C_ref = gibbs_compliance(C_ref, Z, D, Y)
true_prop = np.mean(C_ref == 2)

iterations = 100
runs = 10
alpha = 0.2  # fixed for DULA/DMALA

samplers = [
    ("Gibbs",  lambda C: gibbs_compliance(C.copy(), Z, D, Y)),
    ("DULA",   lambda C: dula_compliance(C.copy(), Z, D, Y, alpha)),
    ("DMALA",  lambda C: dmala_compliance(C.copy(), Z, D, Y, alpha))
]

# store squared errors
se_store = {name: np.zeros((runs, iterations)) for name, _ in samplers}

for name, fn in samplers:
    for r in range(runs):
        C = C0.copy()
        for it in range(iterations):
            C = fn(C)
            prop_hat = np.mean(C == 2)
            err = prop_hat - true_prop
            se_store[name][r, it] = err**2

# compute RMSE and plot log RMSE
x = np.arange(1, iterations+1)
plt.figure(figsize=(8,5))
for name in se_store:
    mse = se_store[name].mean(axis=0)
    rmse = np.sqrt(mse)
    plt.plot(x, np.log(rmse), label=name)
plt.xlabel("Iteration")
plt.ylabel("log RMSE")
plt.title("Latent Compliance: log RMSE of Complier Proportion")
plt.legend()
plt.tight_layout()
plt.savefig("results/compliance_log_rmse_comparison.png")
plt.close()

print("Saved results/compliance_log_rmse_comparison.png")
