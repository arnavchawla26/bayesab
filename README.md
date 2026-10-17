# bayesab

A from-scratch Bayesian A/B testing toolkit: Beta-Binomial conjugate
posteriors for conversion-rate experiments, credible intervals,
probability-to-beat-control, and expected-loss based stopping rules —
with a CLI for both one-shot analysis and sequential-test simulation.

Unlike a point-estimate or model-fitting project (see
[gradient-boosted-trees][gbdt]), this one is about *quantifying
uncertainty and risk* in a decision, not fitting a model: the kind of
thing a growth or product-analytics team runs to decide when an A/B
test is done.

[gbdt]: https://github.com/arnavchawla26/gradient-boosted-trees-from-scratch

## What it does

Given observed conversions for a control and one or more treatment
variants, `bayesab`:

1. Computes each variant's exact Beta-Binomial posterior over its true
   conversion rate (conjugate update, no sampling needed for this step).
2. Reports a credible interval per variant — either the standard
   equal-tailed interval or the highest-density interval (HDI), the
   shortest interval containing the given probability mass.
3. Computes **P(variant beats control)** exactly, via numerical
   integration of the posterior densities (not simulation), and
   **P(variant is the single best of all variants)** via Monte Carlo for
   three or more arms (no simple closed form exists there).
4. Computes the **expected loss** of shipping each variant — the average
   amount you'd lose, in conversion-rate points, if you picked it and a
   different variant was actually better. This is the standard Bayesian
   A/B testing risk measure (see Chris Stucchio's "Bayesian A/B Testing
   at VWO" and Evan Miller's Bayesian-formulas writeup).
5. Applies an **expected-loss stopping rule**: stop once the expected
   loss of the apparent winner drops below a small practical threshold
   (e.g. 0.0001 = one basis point of conversion rate). Unlike a fixed
   sample size or "stop at the first significant p-value," this bounds
   the expected cost of stopping early *by construction* — see the
   benchmark below, which checks that this bound actually holds.

## Why this is a genuinely different approach from a point estimate

A point estimate ("B converts at 7.5%, A at 6.0%, so B wins") throws away
exactly the information a good decision needs: *how sure are you?* Two
tests with the same observed rates but very different sample sizes should
not be treated the same way. Everything here is built around carrying the
full posterior distribution through the pipeline instead of collapsing to
a single number early — the credible interval, the probability-to-beat,
and the expected loss are all computed from the same Beta posteriors, and
none of them requires assuming normality or a large-sample approximation
(the exact conjugate posterior is exact for any sample size, including
tiny ones).

## Tech stack

Python 3.9+, NumPy, SciPy (`scipy.special` for the Beta distribution's
special functions directly — see "Performance," below — and
`scipy.integrate.quad` for the probability/loss integrals), pytest.
No ML framework, no external stats library beyond SciPy: the Beta-Binomial
math, the credible-interval search, the probability integrals, and the
expected-loss integrals are all implemented directly in this repo.

## Install

```bash
git clone https://github.com/arnavchawla26/bayesian-ab-testing-toolkit
cd bayesian-ab-testing-toolkit
pip install -e ".[dev]"
```

## Usage

### Analyze an experiment from observed counts

```bash
bayesab analyze \
  --control 120 2000 \
  --treatment variant_b 150 2000 \
  --treatment variant_c 170 2000 \
  --seed 1
```

```
variant       successes/trials  obs. rate   post. mean  95% credible interval     P(beats control)  P(best)
----------------------------------------------------------------------------------------------------------
control       120/2000          0.0600      0.0604      [0.0504, 0.0713]          --                0.0004
variant_b     150/2000          0.0750      0.0754      [0.0643, 0.0874]          0.9705            0.1223
variant_c *   170/2000          0.0850      0.0854      [0.0736, 0.0980]          0.9989            0.8773

leader: variant_c  (runner-up: variant_b)
expected loss of choosing variant_c: 0.000519
expected loss of choosing variant_b: 0.010509
loss threshold: 0.0001
decision: continue
```

(`decision: continue` here because the default threshold, one basis
point, hasn't been cleared yet even though variant_c looks like the
likely winner — exactly the "looks good but isn't confirmed yet" case
this toolkit is built to catch.)

### Simulate a sequential test against assumed true rates

Useful for sanity-checking a stopping rule *before* running a real
experiment: "if the true rates really are 6% vs. 10%, how much traffic
would this stopping rule typically need, and does it pick correctly?"

```bash
bayesab simulate --true-control-rate 0.06 --true-treatment-rate 0.10 \
  --batch-size 300 --max-trials 100000 --seed 3
```

```
true control rate:   0.06
true treatment rate: 0.1
stopped: True
trials per arm at stop: 300
decision: stop_choose_b
chosen variant: treatment
true best variant: treatment
correct: True
batches simulated: 1
```

### As a library

```python
from bayesab import Experiment, Variant

control = Variant("control", successes=120, trials=2000)
treatment = Variant("treatment", successes=150, trials=2000)
report = Experiment(control, [treatment], loss_threshold=0.0001).analyze(seed=0)

print(report.decision)               # "stop_choose_treatment" or "continue"
print(report.variants["treatment"].prob_beats_control)
```

## Two real things caught while building this (not assumed, traced)

**1. A test's own assumption was wrong, not the code.** An early test for
the sequential simulator asserted that giving it a prior "wrong" about
the conversion rate (centered far below the true rates) would *slow
down* the expected-loss stopping rule compared to a weak/uniform prior.
Running it: the opposite happened on the test's seed. Tracing it by hand
revealed why — the same prior is applied identically to *both* arms, so a
*strong* prior (large alpha+beta, i.e. a large effective pseudo-sample
size) shrinks each arm's posterior variance for a given amount of real
data, which can shrink the estimated gap's uncertainty (and hence the
expected loss) *faster* than a weak prior, even while it biases each
arm's mean. `benchmarks/compare_prior_strength.py` measures this across
200 seeds per prior rather than trusting one anecdote:

```
prior                                 mean trials/arm   median trials/arm   accuracy
--------------------------------------------------------------------------------------
uniform Beta(1,1)                     178               200                 100.0%
weak-and-wrong Beta(1,10)             176               200                 100.0%
strong-and-wrong Beta(1,200)          158               100                 100.0%
strong-and-roughly-right Beta(15,85)  210               200                 100.0%
```

The strong-but-wrong prior really does stop *fastest*, confirming the
traced explanation rather than the original assumption. (Accuracy stayed
100% in every column here because the underlying true gap, 10% vs. 20%,
is large enough that the bias doesn't flip the decision — a genuinely
misleading prior would need a much larger pseudo-sample-size relative to
the true gap to actually cause a wrong pick.) The test in
`tests/test_simulate.py` was rewritten to assert only what's provably
true (a strong low prior measurably pulls the very first batch's
posterior mean down) plus a documented note about the corrected,
counterintuitive finding — see `test_custom_prior_pulls_early_posterior_mean_toward_itself`.

**2. A benchmark's own framing was wrong, not the library.** The first
version of `benchmarks/compare_frequentist_peeking.py` simulated
repeated "peeking" at a running two-proportion z-test under the null
(true rates equal) — the classic demonstration that stopping the instant
p < 0.05 inflates the real false-positive rate far above the nominal 5%
(confirmed: **29.3%** over 300 runs). It then counted every stop of
`bayesab`'s expected-loss rule under the same null the *same way*, as a
"false positive" — and that came out to **84.0%**, which looks damning
until you trace what a stop actually *means* in each framework. A
frequentist p < 0.05 asserts "a difference exists" — under the null,
that's a genuine error. The Bayesian rule instead asserts "the expected
cost of picking either variant is below `threshold`" — under the null,
that assertion is *true*, because there really is no meaningful
difference to lose. Stopping there is the rule correctly recognizing
further data collection has no payoff, not a mistake. The fixed
benchmark measures the actually-comparable thing: does the Bayesian
rule's own guarantee hold in practice? Over the runs where it did stop,
the *realized* loss of the decision it made:

```
mean realized loss of the chosen decision:   0.000338
max realized loss of the chosen decision:    0.000499
threshold:                                    0.000500
every stop's realized loss <= threshold: True
```

The bound holds on every single run, as it must by construction — but
checking it empirically (rather than only trusting the derivation) is
the same discipline as everything else in this project.

## Verification

- `pytest` — 87 tests: exact-vs-scipy checks for the posterior's mean,
  variance, pdf/cdf/ppf; an **independent closed-form combinatorial
  oracle** (`tests/oracle.py`, the classic integer-parameter summation
  formula for P(B > A), independently cross-checked against
  `scipy.integrate.dblquad` before being trusted as a test oracle) used
  to verify the library's own numerical-integration implementation of
  the same quantity; Monte Carlo cross-checks of both the
  probability-to-beat and expected-loss calculations; HDI vs.
  equal-tailed interval geometric properties (narrower for skewed
  posteriors, coincide for symmetric ones, correct one-sided behavior
  for monotonic densities); an aggregate accuracy check of the
  sequential stopping rule across many simulated seeds; and CLI
  subprocess tests for both commands.
- `pyflakes` clean.
- Two benchmark scripts in `benchmarks/` that measure (not assume)
  real comparative behavior, described above.

```bash
pip install -e ".[dev]"
pytest -q          # 87 passed
pyflakes src tests benchmarks
python benchmarks/compare_prior_strength.py
python benchmarks/compare_frequentist_peeking.py
```

## Performance note

`scipy.stats.beta.pdf/cdf` go through scipy's generic `rv_continuous`
argument-broadcasting machinery on every call, which dominates runtime
when called thousands of times inside a quadrature loop (profiled at
~170ms per `expected_loss` call during development). `BetaPosterior`'s
`pdf`/`cdf`/`sf`/`ppf` call `scipy.special`'s `betaln`/`betainc`/
`betaincc`/`betaincinv` ufuncs directly instead — same values (checked
throughout the test suite against `scipy.stats.beta`), about 7x faster.
The probability and expected-loss integrals also restrict their
quadrature domain to each posterior's effective support (mean ± 12
standard deviations, via `BetaPosterior.effective_support`) rather than
the full `[0, 1]` unit interval, which matters a lot once posteriors are
narrow (large sample sizes) — the full interval is mostly flat zero
density there.

## Current status

Complete, tested v1: `analyze` (control vs. N treatments) and `simulate`
(sequential-test dry run) both work end to end from the CLI and as a
library. Nothing is stubbed. Possible future extensions: a proper N-arm
joint expected-loss (the current implementation compares only the
current leader against the runner-up, a standard practical
simplification documented in `experiment.py`); non-conversion-rate
metrics (revenue-per-user via a Normal/Log-Normal model instead of
Beta-Binomial); a small Streamlit/plotting front end for the posterior
plots this CLI currently only describes in text.

## License

MIT
