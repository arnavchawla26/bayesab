# bayesab

Bayesian A/B testing for conversion rates, written from the formulas up:
Beta-Binomial conjugate posteriors, credible intervals, probability to beat
control, and an expected-loss stopping rule, with a CLI for one-shot analysis
and for simulating a sequential test. I built it in September 2026, one of four
small exercises (tsanomaly, minietl, gbdt) done to understand the algorithms
from the inside before using libraries for them. Of the four, this is the one
about quantifying uncertainty in a decision rather than fitting a model.

Python 3.9+, NumPy and SciPy (`scipy.special` for the Beta special functions,
`scipy.integrate.quad` for the probability and loss integrals). The posterior
math, the interval search, the probability integrals and the expected-loss
integrals are implemented in this repo, not imported from a stats library.

## What it does

Given observed conversions for a control and one or more treatments:

1. Computes each variant's exact Beta-Binomial posterior over its true
   conversion rate (conjugate update, no sampling).
2. Reports a credible interval per variant, either equal-tailed or the
   highest-density interval (HDI), the shortest interval holding the given
   mass.
3. Computes P(variant beats control) exactly, by numerical integration of the
   posterior densities, and P(variant is the single best) by Monte Carlo when
   there are three or more arms (no simple closed form there).
4. Computes the expected loss of shipping each variant: the average amount, in
   conversion-rate points, you would lose by picking it when a different
   variant was actually better. This is the standard risk measure in Bayesian
   A/B testing (Chris Stucchio's "Bayesian A/B Testing at VWO", Evan Miller's
   Bayesian formulas write-up).
5. Applies the stopping rule: stop once the expected loss of the apparent
   winner drops below a small threshold (default 0.0001, one basis point of
   conversion rate). Unlike a fixed sample size or stopping at the first
   significant p-value, this bounds the expected cost of stopping early by
   construction, and the second benchmark below checks that the bound holds.

The point of carrying the full posterior through, rather than collapsing to a
point estimate early, is that two tests with the same observed rates but very
different sample sizes should not be treated the same way. The interval, the
probability to beat and the expected loss all come from the same Beta
posteriors, and none of them assumes normality or a large sample; the
conjugate posterior is exact at any sample size, including tiny ones.

## Run it

```bash
git clone https://github.com/arnavchawla26/bayesab
cd bayesab
pip install -e ".[dev]"
```

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

`decision: continue` because the default threshold of one basis point has not
been cleared, even though variant_c looks like the winner. That "looks good
but is not confirmed" case is what the rule exists for.

### Simulate a sequential test against assumed true rates

A dry run for a stopping rule before the real experiment: if the true rates
really are 6% and 10%, how much traffic would the rule need, and does it pick
correctly?

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

## Two things I got wrong while building it

Both were traced, not assumed, and both turned out to be my framing rather
than the code.

**A test's assumption was wrong.** An early test for the sequential simulator
asserted that a prior centered far below the true rates would slow the
stopping rule down compared to a uniform prior. On the test's seed the
opposite happened. Tracing it by hand: the same prior goes on both arms, so a
strong prior (large alpha+beta, a large pseudo-sample) shrinks each arm's
posterior variance for a given amount of real data, which can shrink the
uncertainty on the gap, and so the expected loss, faster than a weak prior
does, even while biasing each arm's mean. `benchmarks/compare_prior_strength.py`
measures this over 200 seeds per prior instead of trusting one run:

```
prior                                 mean trials/arm   median trials/arm   accuracy
--------------------------------------------------------------------------------------
uniform Beta(1,1)                     178               200                 100.0%
weak-and-wrong Beta(1,10)             176               200                 100.0%
strong-and-wrong Beta(1,200)          158               100                 100.0%
strong-and-roughly-right Beta(15,85)  210               200                 100.0%
```

The strong-but-wrong prior really does stop fastest. Accuracy stays at 100%
in every row because the true gap, 10% vs. 20%, is large enough that the bias
never flips the decision; a prior would need a much larger pseudo-sample
relative to the gap to cause a wrong pick. The test in `tests/test_simulate.py`
now asserts only what is provably true (a strong low prior measurably pulls
the first batch's posterior mean down) and carries a note on the corrected
finding; see `test_custom_prior_pulls_early_posterior_mean_toward_itself`.

**A benchmark's framing was wrong.** The first version of
`benchmarks/compare_frequentist_peeking.py` simulated repeated peeking at a
two-proportion z-test under the null (equal true rates), the classic
demonstration that stopping the moment p < 0.05 inflates the false-positive
rate well above 5% (confirmed: 29.3% over 300 runs). It then counted every
stop of the expected-loss rule under the same null the same way, as a false
positive, and got 84.0%, which looks damning until you ask what a stop means
in each framework. A frequentist p < 0.05 asserts "a difference exists",
which under the null is an error. The Bayesian rule asserts "the expected
cost of picking either variant is below the threshold", which under the null
is true: there is no meaningful difference to lose, and stopping is the rule
correctly noticing that more data has no payoff. The fixed benchmark measures
the comparable thing, whether the rule's own guarantee holds. Over the runs
where it stopped, the realized loss of the decision it made:

```
mean realized loss of the chosen decision:   0.000338
max realized loss of the chosen decision:    0.000499
threshold:                                    0.000500
every stop's realized loss <= threshold: True
```

It holds on every run, as it must by construction; checking it empirically
rather than trusting the derivation is the same habit as the rest of this
repo.

## A note on speed

`scipy.stats.beta.pdf/cdf` go through scipy's generic `rv_continuous`
argument-broadcasting machinery on every call, which dominates runtime when
called thousands of times inside a quadrature loop (I profiled about 170ms
per `expected_loss` call during development). `BetaPosterior`'s `pdf`, `cdf`,
`sf` and `ppf` call `scipy.special`'s `betaln`, `betainc`, `betaincc` and
`betaincinv` ufuncs directly instead: same values (checked against
`scipy.stats.beta` throughout the tests), about 7x faster. The probability
and loss integrals also restrict their quadrature domain to each posterior's
effective support (mean plus or minus 12 standard deviations, via
`BetaPosterior.effective_support`) rather than all of `[0, 1]`, which matters
once posteriors are narrow; the full interval is mostly flat zero density
there.

## Tests

```bash
pip install -e ".[dev]"
pytest -q          # 87 passed
pyflakes src tests benchmarks
python benchmarks/compare_prior_strength.py
python benchmarks/compare_frequentist_peeking.py
```

87 tests: exact-vs-scipy checks for the posterior's mean, variance, pdf, cdf
and ppf; an independent closed-form combinatorial oracle (`tests/oracle.py`,
the integer-parameter summation formula for P(B > A), itself cross-checked
against `scipy.integrate.dblquad` before being trusted) used to verify the
library's numerical-integration version of the same quantity; Monte Carlo
cross-checks of both the probability-to-beat and expected-loss calculations;
HDI vs. equal-tailed interval geometry (narrower for skewed posteriors, equal
for symmetric ones, correct one-sided behaviour for monotonic densities); an
aggregate accuracy check of the stopping rule across many simulated seeds;
and CLI subprocess tests for both commands. `pyflakes` clean.

## Known issues

- `simulate` judges `correct` against whichever arm has the higher true rate,
  however small the gap. With true rates 0.06 and 0.062 at seed 3 the rule
  stops at 8,700 trials per arm, picks control, and prints `correct: False`.
  That is the rule doing its job (the expected loss of either choice is under
  one basis point), so for tiny gaps the `correct` line is misleading. With
  equal true rates the tie goes to control.
- `simulate` prints the internal names `stop_choose_a` / `stop_choose_b`,
  while `analyze` prints variant names.
- With three or more arms the stopping decision compares the leader against
  the runner-up only, a standard simplification noted in `experiment.py`; a
  full joint multi-way loss is not implemented.
- `P(best)` is Monte Carlo (200,000 draws by default), so its last digit moves
  with the seed. Everything else is quadrature.
- Only conversion rates. Revenue-per-user would need a Normal or Log-Normal
  model instead of Beta-Binomial.
- The CLI describes posteriors in text; there are no plots.

## License

MIT
