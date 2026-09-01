# How research results lie

This note is for the person who just got a Sharpe out of `qre-backtest` and is about to put it in a deck. I am going to talk to you like a senior quant talking to a junior on day five, because that is the useful tone. The engine in this repo is the example. It is not a claim that the engine is complete, and it is not a claim that you have found an edge.

If you remember one sentence: **a labeled synthetic Sharpe is not an edge, and an unlabeled real-data Sharpe is often not an edge either.**

---

## 1. Look-ahead

The most expensive bug in this industry is using information you would not have had.

Classic forms:

- Computing a z-score with a rolling window that is centered, or that uses `min_periods=1` plus a future-aware smoother.
- Standardizing a feature with the full-sample mean and variance, then “predicting” in the middle of that sample.
- Building a signal from `close[t]` and filling at `close[t]` as if you traded the print you used to compute the signal.

This engine’s contract is explicit: a feature row dated `t` may use `close[t]`, and **must** be invariant to prices after `t`. The fill is `t+1` close. `tests/test_lookahead.py` takes two panels that agree through `t` and disagree after, and asserts the features through `t` match. `tests/test_features_no_future.py` checks that a rolling mean equals a causal numpy window. `ExecutionModel` refuses `fill_delay < 1`.

If you add a feature and you cannot write that test for it, you do not have a feature. You have a leak.

Same-bar close is the leak juniors argue about. “But in production we trade the close.” No, you don’t — not the close you just used in the z-score, not with zero latency, not at the mid. If you want a close-to-close research convention, delay the fill. If you want a next-open convention, say so and still delay. What you may not do is silently earn the bar that constructed the signal.

---

## 2. Survivorship

Point-in-time means: **who was a member on that date**, not who is a member today.

If you pull “current S&P 500 constituents, daily prices back to 2005” you have built a panel of winners. The names that were acquired, went bankrupt, or were dropped for being small and sick are gone. Long-biased strategies look like genius. Mean-reversion after crashes looks safer than it was. Cross-sectional ranks are computed against a healthier-than-reality peer set.

This repo stores membership as `(symbol, start_date, delist_date)` and filters bars that violate it. The synthetic generator can delist a name mid-sample; after `delist_date` that name is not in the panel. `tests/test_survivorship.py` will fail you if you “fix” a missing name by forward-filling it past delist.

What the engine does **not** do: reconstruct a real point-in-time index, corporate actions, or a delisting return (the last trade is not always a friendly close). When you bring USER_PROVIDED data, the survivorship error becomes your problem again. If your CSV only contains names that still exist, you are back in the index-of-winners.

---

## 3. Ignored costs

A strategy that turns over 100% a day at 10 bps round-trip needs a raw edge that most textbook signals do not have.

This engine charges

```
cost = (commission_bps + slippage_bps) / 1e4 * abs(delta_weight) * nav
```

at fill time. Tests require `gross PnL >= net PnL` when turnover is positive. That is the floor, not a market-impact model.

The stronger statement, encoded in `tests/test_execution_costs.py::test_high_turnover_costs_flip_modest_gross_profit`: a 0/1 flip every bar on a 5 bp/day uptrend is profitable with costs off and a loss at 10 bps one-way. Same path, same weights. If your notebook only prints gross PnL, it would call that path an edge. It is not. The fixture is labeled SYNTHETIC. The test asserts the sign flip and does not report a Sharpe.

Slippage in one number is already a fantasy. Capacity is worse. A pairs residual that looks clean on four synthetic names at $1mm NAV is not a license to do it with 3% of ADV in a single-name ETF. If you do not model impact, do not report the Sharpe as “after costs.” You reported “after a coupon for the borrow and a wish.”

Also: bid-ask, borrow, fees that scale with name, and the fact that next-bar close is not a fill you will get in size. The bundled 1 bp + 2 bp is a placeholder so the plumbing is honest. It is not your production cost.

---

## 4. Multiple testing and spec-search

You tried lookback 5, 10, 20, 60, 120. You tried z-caps. You tried “only trade when |z| > 1.” You kept the Sharpe that looked like a story.

That Sharpe is an in-sample statistic of the **search**, not of the rule. The more knobs you turned, the more the max Sharpe is a draw from the right tail of noise. GBM with drift 0 will still give you a lookback that “worked.” That is why this repo refuses to tune the sample until Sharpe looks impressive. A noisy or negative net Sharpe on synthetic GBM is the correct demo.

If you want a number that means something:

- Freeze the spec on a training window you are willing to throw away.
- Run it once on a test window you did not use to choose the spec.
- Pre-commit the spec (git commit, written plan, whatever you will not silently edit).
- Report the family of trials, not the winner.

This engine will happily run whatever YAML you hand it. It will not save you from yourself.

---

## 5. Walk-forward still leaks if you pick the best WF config on the full history

Walk-forward / expanding / rolling splits with an embargo are in `qre.research.walk_forward`. Test dates never appear in train. That is necessary and not sufficient.

The failure mode that eats experienced people:

1. You run walk-forward for config A, B, C, … K.
2. You pick the config whose **average walk-forward Sharpe** is best.
3. You report that average as out-of-sample.

You used the test path to select the spec. The folds were out-of-sample **conditional on a frozen spec**. They are not out-of-sample conditional on “I looked at all the fold Sharpes and kept the winner.” That is multiple testing with extra steps.

Embargoes stop rolling-feature leakage (a 20-day z-score computed at the train/test boundary still contains test-adjacent bars if you are sloppy). They do not stop you from shopping configs.

If you nest a validation fold inside train and you touch the true test set once, you are doing the job. If you touch it twenty times, you are writing fiction.

---

## 6. Non-stationarity

The mapping from “this rule made money in 2012–2015” to “this rule makes money” is a prayer.

Regimes change: vol, correlation, fee schedules, who is on the other side, whether the pair is still cointegrated. An OU process in this repo is stationary by construction. Real spreads are not. A synthetic pair that mean-reverts because we coded `kappa=0.08` is not evidence that `SYN000`/`SYN001` would have been a trade, and those tickers do not exist.

When you move to USER_PROVIDED data, ask: would I have known, in real time, that the regime I am in is the one where this signal works? If the answer is “I selected the window where it worked,” see section 4.

---

## 7. Capacity and impact

Research NAV in this engine defaults to $1mm. Nothing in the simulator cares that your notional is 1% of daily volume or 40%. Turnover is a fraction of NAV, not a fraction of the market.

If a cross-sectional rank longs the names everyone else longs, you are not paying 3 bps. You are paying the crowding. If a pairs trade is the residual of two liquid index products, maybe you can scale. If it is two single names the generator invented, you cannot.

Do not report a Sharpe that assumes infinite liquidity. The engine does not. It also does not model impact for you. The honest sentence is: “after a linear cost on traded notional, at this NAV, on this panel.”

---

## 8. Train/test leakage via full-sample feature selection

Even with a clean walk-forward on **parameters**, you can leak through **features**.

Examples:

- You computed 200 factors on the whole panel, kept the 12 with the best full-sample rank IC, then walk-forwarded a model on those 12. The feature list was chosen with the test years in the room.
- You winsorized using the full-sample 1st/99th percentiles.
- You picked the pair `SYN000`/`SYN001` because, over the whole sample, they were the most cointegrated. Then you “tested” the residual rule.

The mechanical analog in this repo: `build_features` does not search. Strategies do not screen a universe of pairs. If you add a screener, the screener’s information set has to be train-only, the same way lookback choice does. `tests/test_walk_forward.py` only proves date disjointness. It does not prove you did not peek at test returns to decide what to compute.

---

## 9. Why a labeled synthetic Sharpe is not an edge

`generate_synthetic_ohlcv` stamps `DataOrigin.SYNTHETIC`. GBM is a drift-0 martingale plus noise. There is no momentum premium in that process. A TSMOM Sharpe on it is sampling error, plus any small-sample artifact of the sign function, minus costs.

OU and `ou_pair` **do** have a mean-reverting residual, because we put it there. A mean-reversion or pairs Sharpe on that panel is a test that the pipeline can pick up a toy property we inserted. It is not a market. It is not capacity-aware. It is not robust to a kappa we did not tune for your resume.

The CLI prints:

```
*** SYNTHETIC DATA — NOT A MARKET RESULT ***
```

and ends with:

```
This run used labeled synthetic prices. Do not treat these metrics as evidence of an edge.
```

`PerformanceReport.data_origin` is `SYNTHETIC`. If you delete those strings and keep the numbers, you are misrepresenting the work. If you replace the panel with real prices and keep the same research sins listed above, you are also misrepresenting the work — you just have a more expensive false positive.

The useful output of a synthetic run is: look-ahead tests pass, delist handling works, costs can flip the sign of PnL, walk-forward dates do not overlap, and the report cannot forget where the prices came from. That is the recruiting bar this repo is aimed at. Alpha lives somewhere else, and it has to survive a process at least this hostile.

---

No live performance is claimed. No live performance should be inferred. If a number in this repository looks good, start by assuming you made a mistake, then assume you spec-searched, then assume the cost model is too kind. In that order.

---

## 10. Purged k-fold and combinatorial purged CV

Walk-forward with an embargo stops a *contiguous* test window from sitting inside train. It does not stop you from using every other Tuesday as a test set while the labels that overlap those Tuesdays stay in train.

López de Prado, *Advances in Financial Machine Learning* (2018), Chapter 7: a label formed at bar `i` that uses information through `i + h` overlaps a test interval `[t0, t1]` when `i <= t1` and `i + h >= t0`. Those train observations are **purged**. Serial correlation still leaks through the first bars *after* the test window; those are the **embargo**.

This repo encodes that in `qre.research.purged_cv`:

- `purged_kfold` — contiguous groups, one held out per fold, purge + embargo applied.
- `combinatorial_purged_cv` (CPCV) — every combination of `n_test_groups` out of `n_groups` as the test set, same purge + embargo.

`tests/test_purged_cv.py` asserts train/test disjointness, that the label-horizon overlap is gone from train, and that embargo bars after test are gone from train. CPCV fold count equals `C(n_groups, n_test_groups)`.

Purged CV is still not a panacea. If you CPCV twenty configs and keep the winner, you used the test combinations to select. That is section 4 again. Report `n_trials`.

## 11. Deflated Sharpe ratio

Bailey and López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality," *Journal of Portfolio Management* (2014).

The number you want is not the raw Sharpe. It is the **probability** that the true Sharpe exceeds the Sharpe you should have expected *from the fact that you tried `n_trials` independent tests and kept the max*. That is PSR evaluated at the expected-max Sharpe under `n_trials` (DSR).

`qre.analytics.dsr` implements PSR / DSR with `statistics.NormalDist` (no scipy). The CLI prints `deflated_sharpe` next to the pipeline demo. `n_trials: 1` in `configs/sample_momentum.yaml` is the honest setting for a **pre-declared** spec. If you searched 20 YAMLs, pass 20; DSR will fall. `tests/test_dsr.py` checks that DSR falls as `n_trials` rises, and that zero-vol returns NaN.

These two papers are cited because the modules encode them. We do not cite work we did not implement.

No live performance is claimed. No live performance should be inferred.
