# Methods equations

These equations describe the implemented pipeline. Symbols must be defined again
in the manuscript because the modalities have different endpoints and evaluation
units.

## 1. Independent binary probabilities

For target (k), a model logit (z_k) is converted to a probability by

$$
p_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}}.
$$

The targets are independent binary outputs; they are not a mutually exclusive
softmax.

## 2. Masked class-weighted binary cross-entropy

The primary multi-label training objective is

$$
\mathcal{L}_{\mathrm{WBCE}}
= \frac{1}{\sum_{i,k} m_{ik}}
\sum_{i,k} m_{ik}
\left[-w_k y_{ik}\log \sigma(z_{ik})
- (1-y_{ik})\log\left(1-\sigma(z_{ik})\right)\right],
$$

where (m_{ik}\in\{0,1\}) marks an observed label and (w_k>0) is computed from
the training partition only. The implementation uses the numerically stable
`BCEWithLogits` form.

## 3. Affine logistic calibration

Chest X-ray and knee MRI targets use held-out affine logistic scaling:

$$
\tilde p_k = \sigma(a_k z_k + b_k), \qquad a_k > 0.
$$

Parameters (a_k,b_k) minimize calibration negative log-likelihood. Identity
calibration is retained when the prespecified minimum improvement is not achieved.
The CT baseline does not claim population probability calibration.

## 4. CT study aggregation

For a study with (n) slice-level `any` scores (p_1,\ldots,p_n), the
validation-selected mean pooling rule is

$$
s_{\mathrm{study}} = \frac{1}{n}\sum_{j=1}^{n} p_j.
$$

Maximum, mean, top-(k) mean, and noisy-OR were candidate rules on validation;
mean pooling was locked before final test evaluation.

## 5. Sensitivity-constrained threshold selection

On the calibration partition, the locked operating threshold is

$$
\tau^{*} = \max\left\{\tau:\operatorname{Sensitivity}_{\mathrm{cal}}(\tau)
\geq s_{0}\right\}, \qquad s_0=0.90,
$$

and the binary decision is

$$
\hat y = \mathbb{1}[\tilde p \geq \tau^{*}].
$$

## 6. Operating characteristics

$$
\operatorname{Sensitivity}=\frac{TP}{TP+FN},\qquad
\operatorname{Specificity}=\frac{TN}{TN+FP},
$$

$$
\operatorname{PPV}=\frac{TP}{TP+FP},\qquad
\operatorname{NPV}=\frac{TN}{TN+FN},
$$

$$
\operatorname{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}.
$$

ROC-AUC is threshold-independent discrimination and must not be described as
accuracy.

## 7. Calibration diagnostics

For (N) observations, the Brier score and negative log-likelihood are

$$
\operatorname{Brier}=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2,
$$

$$
\operatorname{NLL}=-\frac{1}{N}\sum_{i=1}^{N}
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
$$

For equal-width bins (B_m), expected calibration error is

$$
\operatorname{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{N}
\left|\operatorname{acc}(B_m)-\operatorname{conf}(B_m)\right|.
$$

## 8. Grouped bootstrap uncertainty

Patients are sampled with replacement for CT and chest X-ray; MRNet uses exams
because patient identifiers are unavailable. For metric (\theta), the reported
95% percentile interval is

$$
CI_{0.95}(\theta)=
\left[Q_{0.025}(\theta^{*(1)},\ldots,\theta^{*(B)}),
Q_{0.975}(\theta^{*(1)},\ldots,\theta^{*(B)})\right],
\qquad B=2000.
$$

## 9. Transparent workflow priority

The deterministic within-zone priority score is

$$
P(z,w)=b_z + \lambda w,
$$

where (b_{\mathrm{GREEN}}=0), (b_{\mathrm{YELLOW}}=10),
(b_{\mathrm{RED}}=20), (w) is waiting time in minutes, and
(\lambda=0.002). This is a workflow ordering score, not a calibrated clinical
risk. Cases are first ordered by zone and then FIFO; authorized human overrides
are retained in the audit trail.
