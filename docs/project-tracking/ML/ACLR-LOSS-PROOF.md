You asked specifically for:

> **the exact gradient form and how the A³-weighted loss alters back-prop in a one-sample-per-step DPD network, with math—not intuition.**

I will:

1. restate the constrained loss
2. derive ∂L/∂ŷ
3. show how it biases gradients toward ACLR-critical regions
4. connect that bias to Volterra coefficients
5. show what changes in SGD
6. state limits formally
7. cite sources

No hand-waving.

---

# 1. Setup

Let:

* input IQ vector with memory taps:
  [
  \mathbf{x}_n = [x[n], x[n-1], \dots, x[n-M]]
  ]

* NN predistorter:
  [
  \hat{y}[n] = f_\theta(\mathbf{x}_n)
  ]

* target linearized output:
  [
  y[n]
  ]

* error:
  [
  e[n] = \hat{y}[n] - y[n]
  ]

---

# 2. ACLR-proxy loss under single-sample constraint

We derived:

[
\boxed{
L
=

\mathbb{E}\Big[
|e[n]|^2
+
\alpha |x[n]|^3 |e[n]|^2
\Big]
}
]

Define instantaneous loss:

[
\ell_n
======

(1 + \alpha |x[n]|^3),|e[n]|^2
]

---

# 3. Gradient w.r.t. NN output

Since training uses back-prop:

[
\frac{\partial \ell_n}{\partial \hat{y}[n]}
===========================================

(1 + \alpha |x[n]|^3),
\frac{\partial |e[n]|^2}{\partial \hat{y}[n]}
]

For complex IQ:

[
|e|^2 = e_I^2 + e_Q^2
]

So:

[
\boxed{
\frac{\partial \ell_n}{\partial \hat{y}}
========================================

2(1 + \alpha |x|^3),e
}
]

This is the **entire mechanism**:

> **samples with large |x| produce proportionally larger gradients.**

No spectral math inside the NN—just gradient re-weighting.

---

# 4. Why this targets ACLR mathematically

From Volterra theory:

PA cubic term:

[
y_{IMD}[n] = a_3 |x[n]|^2 x[n]
]

Residual after DPD:

[
e[n] \approx c_3 |x[n]|^2 x[n]
]

Adjacent-band power:

[
P_{adj}
\propto
\mathbb{E}|e[n]|^2
==================

|c_3|^2,\mathbb{E}|x|^6
]

(derived by Parseval + polynomial spectrum expansion)

So the parameter (c_3) is what you must suppress.

Gradient of MSE alone:

[
\nabla_\theta L_{MSE}
\propto
\mathbb{E}\left[e \frac{\partial \hat{y}}{\partial \theta}\right]
]

Dominated by **typical** amplitudes.

Gradient of weighted loss:

[
\nabla_\theta L_{A^3}
\propto
\mathbb{E}\left[ |x|^3 e
\frac{\partial \hat{y}}{\partial \theta}\right]
]

Dominated by **peak samples**, which dominate:

[
\mathbb{E}|x|^6
]

That is the statistical connection.

Sources:

* Cripps, *RF Power Amplifiers*, Ch. 6
* Schetzen, *Volterra Theory*
* Ding et al., Wiley

---

# 5. How this modifies SGD update

Standard:

[
\theta_{k+1}
============

## \theta_k

\mu \sum_n e[n]
\frac{\partial \hat{y}[n]}{\partial \theta}
]

Weighted:

[
\boxed{
\theta_{k+1}
============

## \theta_k

\mu \sum_n
(1 + \alpha |x[n]|^3)
e[n]
\frac{\partial \hat{y}[n]}{\partial \theta}
}
]

So:

* low-amplitude samples ≈ unchanged
* peaks get amplified learning rate

This is equivalent to **importance-sampling the peak-power region**, which is exactly where IMD is generated.

---

# 6. Why this suppresses adjacent bands (formally)

Adjacent power depends on (c_3):

[
P_{adj} \propto |c_3|^2
]

Gradient descent reduces (c_3) fastest when the loss weights:

[
\propto \frac{\partial}{\partial c_3}
\mathbb{E}[|x|^3|e|^2]
\sim
\mathbb{E}[|x|^6]
]

That matches the moment appearing in ACLR.

This is the mathematical alignment.

---

# 7. Fifth-order extension

If PA has strong 5th-order:

[
y = a_1 x + a_3|x|^2x + a_5|x|^4x
]

Add:

[
L = |e|^2

* \alpha_3|x|^3|e|^2
* \alpha_5|x|^5|e|^2
  ]

Same derivation.

---

# 8. What this still cannot do (theorem-level limit)

ACLR is:

[
\int_{\text{adj}} |Y(f)|^2 df
]

which depends on **correlations across time**.

Any loss:

[
\ell_n = g(x[n],e[n])
]

cannot encode frequency localization.

Therefore:

> **No purely per-sample loss can exactly optimize ACLR.**

This follows from Parseval and the fact PSD is quadratic in time correlations.

Source:

* Oppenheim & Schafer

---

# 9. Practical parameter choice

Normalize:

[
\mathbb{E}|x|^2 = 1
]

Start:

* α ≈ 1–10

Too large → training dominated by PAPR outliers.

---

# 10. Summary

**Gradient form:**

[
\nabla_{\hat{y}} \ell
=====================

2(1+\alpha|x|^3)e
]

**Why it helps ACLR:**

* cubic PA physics
* adjacent power ∝ sixth-order moment
* peaks dominate
* gradient focuses there

---

# Socratic check

Answer these to prove you understand:

1. Why does IMD3 power scale as |x|⁶ but the gradient uses |x|³?

2. What happens if α→∞?

3. Why does this weighting not enforce spectral masks?

---

# Where to verify

Search:

* Cripps spectral regrowth polynomial
* Ding behavioral DPD
* Volterra PSD derivation
* Guan Zhu DPD loss weighting

If you want next: I can derive the **batch-FFT spectral gradient** and show how ∂ACLR/∂ŷ[n] is computed explicitly.
