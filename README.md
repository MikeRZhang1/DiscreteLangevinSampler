# Discrete Langevin Samplers for Economic Models

This repository implements and benchmarks discrete Langevin–based MCMC samplers—DULA (Discrete Unadjusted Langevin Algorithm) and DMALA (Discrete Metropolis–Adjusted Langevin Algorithm)—on two simple discrete economic applications:

1. **Latent Compliance Types (Causal Inference)**
   - Four compliance types (Always‑Taker, Never‑Taker, Complier, Defier) are sampled conditional on observed instrument, treatment, and outcome data.
   - Samplers compare Gibbs (exact), DULA, and DMALA for posterior sampling of latent type vectors.

2. **Two‑Regime Markov‑Switching HMM (Macroeconomics)**
   - A 2‑state hidden Markov model for macro growth regimes (expansion vs. recession) with Gaussian observations.
   - Samplers compare forward‑backward Gibbs, DULA, and DMALA for the latent regime sequence.

## Key Features

- **Discrete Langevin Proposals**: Coordinate‑wise softmax proposals based on finite‑difference gradients of the log‑posterior (DLP).
- **Unadjusted vs. Metropolis‑Adjusted**: DULA uses the raw proposal; DMALA adds the MH correction to reduce bias.
- **Convergence Diagnostics**:
  - Normalized Hamming distance between successive samples.
  - Energy (log‑posterior) trajectories.
  - Step‑size (alpha) sensitivity analysis.

## Setup and Dependencies

Requires Python 3.7+ and the following packages:

```bash
pip install numpy scipy matplotlib
```

## Usage
1. Run the main scripts:
```
python latent_compliance.py
python markov_switching.py
```

Outputs:
- ```compliance_hamming.png``` and ```compliance_energy.png```: Convergence plots for compliance types.
- ```hmm_hamming.png``` and ```hmm_energy.png```: Convergence plots for the HMM.
- ```compliance_dula_alpha_compare.png```, ```compliance_dmala_alpha_compare.png```, ```hmm_dula_alpha_compare.png```, ```hmm_dmala_alpha_compare.png```: Step‑size sensitivity plots.


References
- Zhang, R., Liu, X., & Liu, Q. (2022). A Langevin-Like Sampler for Discrete Distributions. ICML.