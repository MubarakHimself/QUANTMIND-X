---
title: MetaTrader 5 Machine Learning Blueprint (Part 5): Sequential Bootstrapping—Debiasing Labels, Improving Returns
url: https://www.mql5.com/en/articles/20059
categories: Trading, Machine Learning
relevance_score: 6
scraped_at: 2026-01-22T17:54:16.627556
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ypsjvxbnlwoqbpofhrcywlyixdpodmjv&ssn=1769093654043575347&ssn_dr=0&ssn_sr=0&fv_date=1769093654&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F20059&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MetaTrader%205%20Machine%20Learning%20Blueprint%20(Part%205)%3A%20Sequential%20Bootstrapping%E2%80%94Debiasing%20Labels%2C%20Improving%20Returns%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17690936549144537&fz_uniq=5049469471162215446&sv=2552)

MetaTrader 5 / Trading


### Introduction

This article introduces **Sequential Bootstrapping**, a principled sampling method that addresses concurrency at its source. Rather than correcting for redundancy after sampling, sequential bootstrapping actively prevents it during the sampling process itself. By dynamically adjusting draw probabilities based on temporal overlap, this method constructs bootstrap samples with maximally independent observations.

We will demonstrate how to:

- Understand the fundamental limitations of standard bootstrap in financial contexts.
- Implement the sequential bootstrap algorithm from first principles.
- Validate its effectiveness through Monte Carlo simulations.
- Integrate it into a complete financial ML pipeline.
- Evaluate performance improvements on real trading strategies.

#### Prerequisites

This article assumes knowledge of the [label concurrency problem in financial ML](https://www.mql5.com/en/articles/19850 "Machine Learning Blueprint (Part 4): The Hidden Flaw in Your Financial ML Pipeline — Label Concurrency") and [triple-barrier labeling](https://www.mql5.com/en/articles/18864 "MetaTrader 5 Machine Learning Blueprint (Part 2): Labeling Financial Data for Machine Learning") techniques which we discussed earlier in this series. A good working knowledge of Python's ML libraries is essential to fully benefit from this article.

All source code pertaining to this series can be found in my [GitHub](https://www.mql5.com/go?link=https://github.com/pnjoroge54/Machine-Learning-Blueprint "https://github.com/pnjoroge54/Machine-Learning-Blueprint").

### Why Bootstrap Works in Traditional Statistics

The bootstrap method, introduced by Bradley Efron in 1979, is one of the most powerful tools in statistical inference. Its elegance lies in its simplicity: to estimate the sampling distribution of a statistic, repeatedly resample your data with replacement and calculate the statistic on each resample.

This works beautifully when data points are **Independent and Identically Distributed (IID)**. Each observation in a medical study, agricultural experiment, or manufacturing quality control sample typically represents an independent event. Blood samples from different patients contain independent information. Crop yields from different plots reflect independent growing conditions.

#### The 2/3 Rule: A Hidden Feature, Not a Bug

Standard bootstrap has a fascinating mathematical property that most practitioners overlook. When you sample _I_ times with replacement from _I_ observations, you'll see approximately 2/3 of your original observations in each bootstrap sample.

Let's understand why this happens.

**A Simple Thought Experiment**

Imagine you have 100 balls in a bag, numbered 1 to 100. You're going to:

1. Pick a ball randomly
2. Write down its number
3. Put it back (this is the key!)
4. Repeat 100 times

**Question:** After 100 picks, how many _different_ balls did you see at least once?

Your intuition might say "I made 100 picks, so maybe all 100?" But because you replace each ball after picking it, you'll select some balls multiple times while missing others entirely.

#### The Mathematics

Consider one specific ball - say ball #42.

Each time you pick:

- Probability of getting ball #42 = 1/100
- Probability of NOT getting ball #42 = 99/100

After 100 picks:

- Probability you NEVER picked ball #42 = (99/100)100 ≈ 0.366

This means there's approximately a 63.4% probability that ball #42 was picked at least once.

This pattern holds regardless of scale:

| Number of Items | Picks | Probability Each Item Seen |
| --- | --- | --- |
| 10 | 10 | (9/10)10 ≈ 0.651 |
| 100 | 100 | (99/100)100 ≈ 0.634 |
| 1,000 | 1,000 | (999/1000)1000 ≈ 0.632 |
| 10,000 | 10,000 | (9999/10000)10000 ≈ 0.632 |

The fraction converges to **1 - e-1 ≈ 0.632** (where e ≈ 2.71828 is Euler's number).

#### Why This Number? The Natural Growth Constant

The exact value is 1 - e-1, where e appears because we're dividing our chances into smaller and smaller pieces as _I_ grows larger. This is the same mathematical pattern that governs continuous compound interest.

The Simple Rule to Remember:

When you sample WITH replacement, making as many picks as you have items, you'll see about **63% of the items** (roughly 2/3).

This means standard bootstrap naturally leaves about 37% of your data unsampled in each iteration. In traditional statistics, this is perfectly fine—it even helps with variance estimation. But in finance, it interacts disastrously with label concurrency.

### Why This Becomes Catastrophic in Finance

The 2/3 rule assumes each observation contains independent information. In financial ML with triple-barrier labeling, this assumption fails spectacularly due to temporal overlap.

Recall the blood sample analogy from Part 3: imagine someone in your laboratory spills blood from each tube into the following nine tubes to their right. Tube 10 contains blood for patient 10, but also blood from patients 1 through 9. Now you're sampling from these contaminated tubes with replacement.

The compounding disaster:

1. Standard bootstrap samples only ~63% of observations
2. Each sampled observation overlaps temporally with others
3. The effective independent information is far less than 63%
4. Models learn the same patterns multiple times within a single bootstrap sample
5. Variance estimates become unreliable, defeating the purpose of bootstrap

Consider a concrete example from our triple-barrier labels:

**Scenario:** 100 trading observations over a volatile period

- Each trade spans 4 hours on average (triple-barrier exit time)
- New trades start every 15 minutes
- This creates approximately 16 concurrent trades at any given time

Standard Bootstrap Result:

- Draws ~63 observations
- Due to overlap, effective unique information ≈ 63/16 ≈ 4 truly independent events
- You're training on the equivalent of 4 independent samples, not 63!

This is why models trained with standard bootstrap in financial contexts exhibit:

- Artificially low training error (learned the same pattern 16 times)
- High variance across bootstrap iterations (randomly got different "copies" of the same events)
- Poor out-of-sample performance (actual frequency of patterns is much lower)

### Sequential Bootstrap: The Solution

Sequential bootstrapping fundamentally reimagines the sampling process. Instead of drawing observations with equal probability, it **dynamically adjusts probabilities** to favor observations that add unique information to the current sample.

**Conceptual Foundation**

The core insight: each observation's value to a bootstrap sample depends on what's already been selected.

If the sample already contains observations covering Monday 9:00 AM to 11:00 AM, another observation from Monday 10:00 AM to 12:00 PM adds relatively little new information. An observation from Tuesday afternoon, however, is highly valuable.

Sequential bootstrap implements this intuition through three steps repeated for each draw:

1. Assess Current State: Determine which time periods are already represented in the sample
2. Calculate Uniqueness: For each remaining observation, calculate how much unique information it would add
3. Adjust Probabilities: Draw the next observation with probability proportional to its uniqueness

**Mathematical Formulation**

Let's formalize this intuition. Consider a set of labels { _y\[i\]_} _i=1,2,3_, where:

- label _y\[1\]_ is a function of return _r\[0,3\]_
- label _y\[2\]_ is a function of return _r\[2,4\]_
- label _y\[3\]_ is a function of return _r\[4,6\]_

The rows of the matrix correspond to the index of returns which were used to label our data set and columns correspond to samples. The outcomes’ overlaps are characterized by the indicator matrix below:

| Time | Obs 1 | Obs 2 | Obs 3 |
| 1 | 1 | 0 | 0 |
| 2 | 1 | 0 | 0 |
| 3 | 1 | 1 | 0 |
| 4 | 0 | 1 | 0 |
| 5 | 0 | 0 | 1 |
| 6 | 0 | 0 | 1 |

The average uniqueness _ū\[i\]_ is calculated as:

_ū\[i\] = (1/L\[i\]) × Σ(t ∈ T\[i\]) \[1 / c\[t\]\]_

Where:

- _L\[i\]_ = number of time periods observation _i_ spans
- _T\[i\]_ = set of time periods where observation _i_ is active
- _c\[t\] =_ count of observations active at time _t,_ including all observations already in the sample plus the candidate observation _i_

The draw probability for observation _i_ is then:

_P(select i) = ū\[i\] / Σ(j ∈ candidates) ū\[j\]_

This ensures:

- Observations with no overlap receive the highest probability
- Observations heavily overlapping with the sample receive the lowest probability
- Probabilities always sum to 1 (valid probability distribution)

**A Worked Numerical Example**

Let's walk through the an example using the indicator matrix above.

Step 1: First Draw

Initially, no observations have been selected, so all have equal probability:

P(Obs 1) = P(Obs 2) = P(Obs 3) = 1/3 ≈ 33.3%

Result: Observation 2 is randomly selected

Current Sample: φ¹ = {2}

Step 2: Second Draw - Calculate Uniqueness

Now we must calculate uniqueness for each observation given that Observation 2 is in our sample.

Observation 1:

- Active at times: {1, 2, 3}
- Time 1: c\[1\] = 1 (only Obs 1), uniqueness = 1/1 = 1.0
- Time 2: c\[2\] = 1 (only Obs 1), uniqueness = 1/1 = 1.0
- Time 3: c\[3\] = 2 (Obs 1 + Obs 2), uniqueness = 1/2 = 0.5
- Average uniqueness: (1.0 + 1.0 + 0.5)/3 = 2.5/3 = 5/6 ≈ 0.833

Observation 2:

- Active at times: {3, 4}
- Time 3: c\[3\] = 2 (Obs 2 + itself), uniqueness = 1/2 = 0.5
- Time 4: c\[4\] = 1 (only Obs 2), uniqueness = 1/1 = 1.0
- Average uniqueness: (0.5 + 1.0)/2 = 1.5/2 = 3/6 = 0.5

Observation 3:

- Active at times: {5, 6}
- Time 5: c\[5\] = 1 (only Obs 3), uniqueness = 1/1 = 1.0
- Time 6: c\[6\] = 1 (only Obs 3), uniqueness = 1/1 = 1.0
- Average uniqueness: (1.0 + 1.0)/2 = 2.0/2 = 6/6 = 1.0

Calculate Probabilities:

- Sum of uniqueness: 5/6 + 3/6 + 6/6 = 14/6
- P(Obs 1) = (5/6) / (14/6) = 5/14 ≈ 35.7%
- P(Obs 2) = (3/6) / (14/6) = 3/14 ≈ 21.4% ← lowest (already selected)
- P(Obs 3) = (6/6) / (14/6) = 6/14 ≈ 42.9% ← highest (no overlap)

Result: Observation 3 is selected

Current Sample: φ² = {2, 3}

Step 3: Third Draw - Calculate Uniqueness

The overlap structure hasn't changed (Obs 2 and 3 still don't overlap), so probabilities remain identical:

- P(Obs 1) = 5/14 ≈ 35.7%
- P(Obs 2) = 3/14 ≈ 21.4%
- P(Obs 3) = 6/14 ≈ 42.9%

Complete Probability Summary:

| Draw | Observation 1 | Observation 2 | Observation 3 | Selected |
| --- | --- | --- | --- | --- |
| 1 | 1/3 (33.3%) | 1/3 (33.3%) | 1/3 (33.3%) | 2 |
| 2 | 5/14 (35.7%) | 3/14 (21.4%) | 6/14 (42.9%) | 3 |
| 3 | 5/14 (35.7%) | 3/14 (21.4%) | 6/14 (42.9%) | ? |

#### Key Observations from This Example

1. Lowest probability goes to previously selected observations (Observation 2 drops from 33.3% to 21.4%)
2. Highest probability goes to observations with zero overlap (Observation 3 jumps to 42.9%)
3. Partial overlap receives intermediate weighting (Observation 1 rises slightly to 35.7%)
4. The method successfully discourages redundancy while allowing sample diversity

### Implementation

#### Core Algorithm

The implementation requires two key functions: one to calculate the indicator matrix, and one to perform the sequential sampling.

Function 1: Indicator Matrix Construction

```
def get_ind_matrix(bar_index, t1):
    """
    Build an indicator matrix showing which observations are active at each time.

    :param bar_index: (pd.Index) Complete time index (all bars)
    :param t1: (pd.Series) End time for each observation (index = start time, value = end time)
    :return: (pd.DataFrame) Indicator matrix where ind_matrix[t, i] = 1 if obs i is active at time t
    """
    ind_matrix = pd.DataFrame(0, index=bar_index, columns=range(t1.shape[0]))

    for i, (t_in, t_out) in enumerate(t1.items()):
        # Mark all times from t_in to t_out as active for observation i
        ind_matrix.loc[t_in:t_out, i] = 1.0

    return ind_matrix
```

Function 2: Average Uniqueness Calculation

```
def get_avg_uniqueness(ind_matrix):
    """
    Calculate average uniqueness for each observation.

    Average uniqueness of observation i = mean of (1/c[t]) across all times t where i is active,
    where c[t] is the number of observations active at time t.

    :param ind_matrix: (pd.DataFrame) Indicator matrix from get_ind_matrix
    :return: (pd.Series) Average uniqueness for each observation
    """
    # Count how many observations are active at each time (row sums)
    concurrency = ind_matrix.sum(axis=1)

    # Calculate uniqueness: 1/concurrency for each observation at each time
    # Replace concurrency with NaN where ind_matrix is 0 (observation not active)
    uniqueness = ind_matrix.div(concurrency, axis=0)

    # Average uniqueness across all times where observation is active
    avg_uniqueness = uniqueness[uniqueness > 0].mean(axis=0)

    return avg_uniqueness
```

Function 3: Sequential Bootstrap Sampler

```
def seq_bootstrap(ind_matrix, sample_length=None):
    """
    Generate a bootstrap sample using sequential bootstrap method.

    :param ind_matrix: (pd.DataFrame) Indicator matrix from get_ind_matrix
    :param sample_length: (int) Length of bootstrap sample. If None, uses len(ind_matrix.columns)
    :return: (list) Indices of selected observations
    """
    if sample_length is None:
        sample_length = ind_matrix.shape[1]

    phi = []  # Bootstrap sample (list of selected observation indices)

    while len(phi) < sample_length:
        # Calculate average uniqueness for each observation
        avg_u = pd.Series(dtype=float)

        for i in ind_matrix.columns:
            # Create temporary indicator matrix with current sample + candidate i
            ind_matrix_temp = ind_matrix[phi + [i]]
            avg_u.loc[i] = get_avg_uniqueness(ind_matrix_temp).iloc[-1]

        # Convert uniqueness to probabilities
        prob = avg_u / avg_u.sum()

        # Draw next observation according to probabilities
        selected = np.random.choice(ind_matrix.columns, p=prob)
        phi.append(selected)

    return phi
```

**Computational Efficiency Considerations**

The implementation above is conceptually clear but computationally expensive for large datasets. Each iteration recalculates uniqueness for all remaining observations, leading to O(n³) complexity.

For production systems, several optimizations are critical:

1. Incremental Updates: Instead of recalculating the entire indicator matrix, maintain a running concurrency count that updates when observations are added to the sample.
2. Precomputation: Calculate pairwise overlaps once at the start, then use simple lookups during sampling.
3. Parallelization: Multiple bootstrap samples can be generated independently in parallel.
4. Sparse Matrix Operations: Financial time series often have limited concurrency; use sparse matrices to exploit this structure.

**Expected Results**

Figure 1 plots the histogram of the uniqueness from standard bootstrapped samples (left) and the sequentially bootstrapped samples (right). The median of the average uniqueness for the standard method is 0.6, and the median of the average uniqueness for the sequential method is 0.7. An ANOVA test on the difference of means returns a vanishingly small probability. Statistically speaking, samples from the sequential bootstrap method have an expected uniqueness that exceeds that of the standard bootstrap method, at any reasonable confidence level. See the attached bootstrap\_mc.py if you are interested in performing a Monte Carlo simulation of your own.

![Monte Carlo experiment of standard vs. sequential bootstraps](https://c.mql5.com/2/177/seq_bootstrap_MC.png)

Figure 1: Monte Carlo experiment of standard vs. sequential bootstraps

### Optimized Implementation

This section explains why the optimized implementation (flattened indices + Numba-accelerated sampling) is superior to the standard indicator matrix approach, and why each function in the implementation was chosen. Empirical memory and complexity comparisons referenced in the article support these claims.

Step 1: _get\_active\_indices_ — sparse mapping vs dense indicator matrix

Purpose: convert event start/end times into per-sample lists of bar indices the sample covers.

Why it’s superior: it stores only non-zero entries (the exact bar indices each sample touches) instead of allocating an _n × T_ dense matrix. This reduces memory from _O(n·T)_ to _O(sum of event lengths)_, which is often near-linear in n for realistic sparsity patterns.

Practical advantage: drastically reduced memory footprint and better cache behavior when scanning sample coverage; these gains produce the compression ratios and memory reductions reported in the optimized analysis.

```
def get_active_indices(samples_info_sets, price_bars_index):
    """
    Build an indicator mapping from each sample to the bar indices it influences.

    Args:
        samples_info_sets (pd.Series):
            Triple-barrier events (t1) returned by labeling.get_events.
            Index: start times (t0) as pd.DatetimeIndex.
            Values: end times (t1) as pd.Timestamp (or NaT for open events).
        price_bars_index (pd.DatetimeIndex or array-like):
            Sorted bar timestamps (pd.DatetimeIndex or array-like). Will be converted to
            np.int64 timestamps for internal processing.

    Returns:
        dict:
            Standard Python dictionary mapping sample_id (int) to a numpy.ndarray of
            bar indices (dtype=int64). Example: {0: array([0,1,2], dtype=int64), 1: array([], dtype=int64), ...}
    """
    t0 = samples_info_sets.index
    t1 = samples_info_sets.values
    n = len(samples_info_sets)
    active_indices = {}

    # precompute searchsorted positions to restrict scanning range
    starts = np.searchsorted(price_bars_index, t0, side="left")
    ends = np.searchsorted(price_bars_index, t1, side="right")  # exclusive

    for sample_id in range(n):
        s = starts[sample_id]
        e = ends[sample_id]
        if e > s:
            active_indices[sample_id] = np.arange(s, e, dtype=int)
        else:
            active_indices[sample_id] = np.empty(0, dtype=int)

    return active_indices
```

Step 2: _pack\_active\_indices_ — contiguous flattened layout for throughput

Purpose: convert _dict/list-of-arrays_ into _flat\_indices, offsets, lengths_, and _sample\_ids_.

Why it’s superior: contiguous arrays remove Python-object overhead and enable tight, linear memory scans. Offsets provide _O(1)_ access to each sample’s slice without per-iteration list indexing. This ragged-to-flat pattern is required for efficient Numba/JIT loops.

Practical advantage: Numba can iterate over memory sequentially, which improves CPU prefetching and reduces interpreter overhead compared to repeated Python-level operations or per-sample array objects.

```
def pack_active_indices(active_indices):
    """
    Convert dict/list-of-arrays active_indices into flattened arrays and offsets.

    Args:
        active_indices (dict or list): mapping sample_id -> 1D ndarray of bar indices

    Returns:
        flat_indices (ndarray int64): concatenated bar indices for all samples
        offsets (ndarray int64): start index in flat_indices for each sample (len = n+1)
        lengths (ndarray int64): number of indices per sample (len = n)
        sample_ids (list): list of sample ids in the order used to pack data
    """
    # Preserve sample id ordering to allow mapping between chosen index and original id
    if isinstance(active_indices, dict):
        sample_ids = list(active_indices.keys())
        values = [active_indices[sid] for sid in sample_ids]
    else:
        # assume list-like ordered by sample id 0..n-1
        sample_ids = list(range(len(active_indices)))
        values = list(active_indices)

    lengths = np.array([v.size for v in values], dtype=np.int64)
    offsets = np.empty(len(values) + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(lengths)

    total = int(offsets[-1])
    if total == 0:
        flat_indices = np.empty(0, dtype=np.int64)
    else:
        flat_indices = np.empty(total, dtype=np.int64)
        pos = 0
        for v in values:
            l = v.size
            if l:
                flat_indices[pos : pos + l] = v
            pos += l

    return flat_indices, offsets, lengths, sample_ids
```

Step 3: _\_compute\_scores\_flat_ / _\_normalize\_to\_prob_— local, incremental scoring

Purpose: compute per-sample scores using the formula _score = (1 / (1 + concurrency)).mean()_ , then normalize to a probability vector.

Why it’s superior: scoring uses only the bars a sample actually covers (via its slice in _flat\_indices_) and the current concurrency counts. Per-sample cost is proportional to its event length _k_ , not the full time horizon _T_ , producing _O(n·k)_ work per full pass instead of _O(n·T)_.

Practical advantage: numeric stability through an eps floor and deterministic normalization prevents zero-sum cases and keeps the sampling distribution robust as concurrency increases.

```
@njit
def _compute_scores_flat(flat_indices, offsets, lengths, concurrency):
    """
    Compute average uniqueness for each sample using flattened indices.

    This follows de Prado's approach: for each bar in a sample, compute uniqueness as 1/(c+1),
    then average across all bars in that sample.

    Args:
        flat_indices (ndarray int64): concatenated indices
        offsets (ndarray int64): start positions (len = n+1)
        lengths (ndarray int64): counts per sample
        concurrency (ndarray int64): current concurrency counts per bar

    Returns:
        scores (ndarray float64): average uniqueness per sample
    """
    n = offsets.shape[0] - 1
    scores = np.empty(n, dtype=np.float64)

    for i in range(n):
        s = offsets[i]
        e = offsets[i + 1]
        length = lengths[i]

        if length == 0:
            # If a sample covers no bars, assign zero average uniqueness
            scores[i] = 0.0
        else:
            # Compute uniqueness = 1/(c+1) for each bar, then average
            sum_uniqueness = 0.0
            for k in range(s, e):
                bar = flat_indices[k]
                c = concurrency[bar]
                uniqueness = 1.0 / (c + 1.0)
                sum_uniqueness += uniqueness
            avg_uniqueness = sum_uniqueness / length
            scores[i] = avg_uniqueness

    return scores
```

```
@njit
def _normalize_to_prob(scores):
    """
    Normalize non-negative scores to a probability vector. If all zero, return uniform.
    """
    n = scores.shape[0]
    total = 0.0
    for i in range(n):
        total += scores[i]

    prob = np.empty(n, dtype=np.float64)
    if total == 0.0:
        # fallback to uniform distribution
        uni = 1.0 / n
        for i in range(n):
            prob[i] = uni
    else:
        for i in range(n):
            prob[i] = scores[i] / total
    return prob
```

Step 4: _\_increment\_concurrency\_flat_ — in-place updates to concurrency

Purpose: increment _concurrency\[bar\]_ for each bar covered by the chosen sample.

Why it’s superior: updates only the affected bars rather than re-scanning or re-aggregating large arrays. A dense-matrix approach might force re-computation of many rows' sums or creation of temporary structures; here updates are localized and _O(k)_.

Practical advantage: cheap incremental updates let the sampler adapt online after each draw without expensive global re-computations, making many bootstrap draws efficient.

```
@njit
def _increment_concurrency_flat(flat_indices, offsets, chosen, concurrency):
    """
    Increment concurrency for the bars covered by sample `chosen`.
    """
    s = offsets[chosen]
    e = offsets[chosen + 1]
    for k in range(s, e):
        bar = flat_indices[k]
        concurrency[bar] += 1
```

Step 5: _\_seq\_bootstrap\_loop_ and _seq\_bootstrap\_optimized_ — full Numba acceleration with reproducible RNG

Purpose: perform the full sequential-bootstrap loop inside Numba using pre-drawn uniforms from Python for reproducible random number generator (RNG) and the flattened layout for memory efficiency.

Why it’s superior:

- Eliminates Python-per-iteration overhead: scoring, CDF selection, and concurrency updates run inside a jitted function, removing costly interpreter context switches that dominate at scale.
- Preserves reproducibility: RNG is managed in Python (NumPy RandomState) and uniforms are passed into the jitted loop, combining reproducible seeding with high-speed inner loops.
- Enables _O(n·k)_ time with small constant factors, compared to the standard algorithm’s repeated indicator-matrix re-computations and potential _O(n²)_ or worse scanning patterns.

Practical advantage: the flattened layout together with full JIT produces large speed and memory improvements, enabling sequential bootstrap for tens or hundreds of thousands of samples in production workflows.

```
@njit
def _choose_index_from_cdf(prob, u):
    """
    Convert a uniform random number u in [0,1) to an index using the cumulative distribution.\
\
    This avoids calling numpy.choice inside numba and is efficient.\
    """\
    n = prob.shape[0]\
    cum = 0.0\
    for i in range(n):\
        cum += prob[i]\
        if u < cum:\
            return i\
    # numerical fallback: return last index\
    return n - 1\
```\
\
```\
@njit\
def _seq_bootstrap_loop(flat_indices, offsets, lengths, concurrency, uniforms):\
    """\
    Njitted sequential bootstrap loop.\
\
    Args:\
        flat_indices, offsets, lengths: flattened index layout\
        concurrency (ndarray int64): initial concurrency vector (will be mutated)\
        uniforms (ndarray float64): pre-drawn uniform random numbers in [0,1), length = sample_length\
\
    Returns:\
        chosen_indices (ndarray int64): sequence of chosen sample indices (positions in packed order)\
    """\
    sample_length = uniforms.shape[0]\
    chosen_indices = np.empty(sample_length, dtype=np.int64)\
\
    for it in range(sample_length):\
        # compute scores and probabilities given current concurrency\
        scores = _compute_scores_flat(flat_indices, offsets, lengths, concurrency)\
        prob = _normalize_to_prob(scores)\
\
        # map uniform to a sample index\
        u = uniforms[it]\
        idx = _choose_index_from_cdf(prob, u)\
        chosen_indices[it] = idx\
\
        # update concurrency for selected sample\
        _increment_concurrency_flat(flat_indices, offsets, idx, concurrency)\
\
    return chosen_indices\
```\
\
```\
def seq_bootstrap_optimized(active_indices, sample_length=None, random_seed=None):\
    """\
    End-to-end sequential bootstrap using flattened arrays + Numba.\
\
    Implements the sequential bootstrap as described in de Prado's "Advances in Financial\
    Machine Learning" Chapter 4: average uniqueness per sample where uniqueness per bar\
    is 1/(concurrency+1).\
\
    Args:\
        active_indices (dict or list): mapping sample id -> ndarray of bar indices\
        sample_length (int or None): requested number of draws; defaults to number of samples\
        random_seed (int, RandomState, or None): seed controlling the pre-drawn uniforms\
\
    Returns:\
        phi (list): list of chosen original sample ids (length = sample_length)\
    """\
    # Pack into contiguous arrays and keep mapping from packed index -> original sample id\
    flat_indices, offsets, lengths, sample_ids = pack_active_indices(active_indices)\
    n_samples = offsets.shape[0] - 1\
\
    if sample_length is None:\
        sample_length = n_samples\
\
    # Concurrency vector length: bars are indices into price-bar positions.\
    # When there are no bars (flat_indices empty), create an empty concurrency of length 0.\
    if flat_indices.size == 0:\
        T = 0\
    else:\
        # max bar index + 1 (bars are zero-based indices)\
        T = int(flat_indices.max()) + 1\
\
    concurrency = np.zeros(T, dtype=np.int64)\
\
    # Prepare reproducible uniforms. Accept either integer seed or RandomState.\
    if random_seed is None:\
        rng = np.random.RandomState()\
    elif isinstance(random_seed, np.random.RandomState):\
        rng = random_seed\
    else:\
        try:\
            rng = np.random.RandomState(int(random_seed))\
        except (ValueError, TypeError):\
            rng = np.random.RandomState()\
\
    # Pre-draw uniforms in Python and pass them into njit function (numba cannot accept RandomState)\
    uniforms = rng.random_sample(sample_length).astype(np.float64)\
\
    # Run njit loop (this mutates concurrency but we don't need concurrency afterwards)\
    chosen_packed = _seq_bootstrap_loop(flat_indices, offsets, lengths, concurrency, uniforms)\
\
    # Map packed indices back to original sample ids\
    phi = [sample_ids[int(i)] for i in chosen_packed.tolist()]\
\
    return phi\
```\
\
**Complexity and deployment implications**\
\
- Memory: optimized approach reduces growth from quadratic to near-linear in _n_, turning previously infeasible problem sizes into manageable ones.\
- Time: per-draw cost becomes proportional to average event length _k_ , not the full time grid _T_. Per-draw complexity is _O(n·k)_, giving total complexity of O(n²·k) for producing n samples. This is still far better than the O(n³) or worse achieved by naive indicator matrix approaches when k << n.\
- Engineering: the flattened + njit pattern supports further optimizations (use int32 for indices when safe, parallelize sampling, precompute repeated score terms) and integrates cleanly with a pipeline that flows from labeling.get\_events into the optimized sampler.\
\
**Memory Efficiency Analysis: Optimized vs Standard Indicator Matrix Implementation**\
\
#### Memory Footprint Comparison\
\
The memory consumption data reveals dramatic differences between the standard and optimized implementations:\
\
| Sample Size | Standard | Optimized | Memory Reduction | Compression Ratio |\
| --- | --- | --- | --- | --- |\
| 500 | 7.19 MB | 0.02 MB | 99.7% | 359:1 |\
| 1,000 | 23.75 MB | 0.04 MB | 99.8% | 594:1 |\
| 2,000 | 93.65 MB | 0.07 MB | 99.9% | 1,338:1 |\
| 4,000 | 412.23 MB | 0.14 MB | 99.97% | 2,944:1 |\
| 8,000 | 1,237.82 MB | 0.28 MB | 99.98% | 4,421:1 |\
\
### Mathematical Analysis of Growth Patterns\
\
#### Standard Implementation (Quadratic Growth)\
\
The standard implementation exhibits _O(n²)_ memory complexity:\
\
```\
Memory(n) ≈ 0.0000188 × n² (MB)\
```\
\
- At n=8,000: Predicted = 1,203 MB vs Actual = 1,238 MB (97% accuracy)\
- Doubling samples: 4x memory increase (consistent with quadratic growth)\
\
**Optimized Implementation (Linear Growth)**\
\
The optimized implementation shows _O(n)_ memory complexity:\
\
```\
Memory(n) ≈ 0.000035 × n (MB)\
```\
\
- At n=8,000: Predicted = 0.28 MB vs Actual = 0.28 MB (exact match)\
- Doubling samples: 2x memory increase (consistent with linear growth)\
\
### Practical Implications for Financial ML\
\
#### Scalability Limits\
\
Standard Implementation\
\
- n=50,000: ~47 GB (impractical)\
- n=100,000: ~188 GB (impossible)\
- Maximum feasible: ~15,000 samples\
\
Optimized Implementation\
\
- n=50,000: ~1.75 MB (trivial)\
- n=100,000: ~3.5 MB (easily handled)\
- Maximum feasible: Millions of samples\
\
Real-World Deployment Scenarios\
\
```\
# Typical financial dataset scenarios\
scenarios = {\
    "Intraday Trading": {\
        "samples": 50_000,      # 2 years of 5-minute bars\
        "standard_memory": "47 GB",\
        "optimized_memory": "1.75 MB",\
        "feasible": "Only with optimized"\
    },\
    "Multi-Asset Portfolio": {\
        "samples": 200_000,     # 100 instruments × 2,000 bars\
        "standard_memory": "752 GB",\
        "optimized_memory": "7 MB",\
        "feasible": "Only with optimized"\
    },\
    "Research Backtesting": {\
        "samples": 1_000_000,   # Comprehensive market analysis\
        "standard_memory": "18.8 TB",\
        "optimized_memory": "35 MB",\
        "feasible": "Only with optimized"\
    }\
}\
```\
\
### Technical Architecture Insights\
\
#### Why the Dramatic Difference?\
\
Standard Implementation ( _get\_ind\_matrix_):\
\
```\
# Creates dense n × n matrix (O(n²) memory)\
ind_matrix = np.zeros((len(bar_index), len(label_endtime)), dtype=np.int8)\
for sample_num, label_array in enumerate(tokenized_endtimes):\
    ind_mat[label_index:label_endtime+1, sample_num] = 1  # Fills entire ranges\
```\
\
Optimized Implementation ( _precompute\_active\_indices_):\
\
```\
# Stores only active indices (O(k×n) memory, where k << n)\
active_indices = {}\
for sample_id in range(n_samples):\
    mask = (price_bars_array >= t0) & (price_bars_array <= t1)\
    indices = np.where(mask)[0]  # Stores only non-zero indices\
    active_indices[sample_id] = indices\
```\
\
**Memory Efficiency Gains**\
\
The compression ratio improves with sample size because:\
\
1. Sparsity increases - Each sample affects only a small fraction of total bars\
2. Fixed overhead - Dictionary structure has minimal base memory cost\
3. Efficient storage - Integer arrays instead of full matrices\
\
### Performance Impact on Sequential Bootstrapping\
\
#### Algorithmic Complexity Comparison\
\
```\
# Standard implementation: O(n³) time, O(n²) memory\
def seq_bootstrap_standard(ind_mat):\
    # Each iteration: O(n²) operations × n iterations\
    for i in range(n_samples):\
        avg_unique = _bootstrap_loop_run(ind_mat, prev_concurrency)  # O(n²)\
\
# Optimized implementation: O(n×k) time, O(n) memory\
def seq_bootstrap_optimized(active_indices):\
    # Each iteration: O(k) operations × n iterations (where k = avg event length)\
    for i in range(n_samples):\
        prob = _seq_bootstrap_loop(flat_indices, offsets, lengths, concurrency, uniforms)  # O(k)\
```\
\
Based on the memory patterns, we can extrapolate time performance:\
\
| Operation | n=1,000 | n=8,000 | Scaling Factor |\
| --- | --- | --- | --- |\
| Memory Allocation | 23.75 MB → 0.04 MB | 1,238 MB → 0.28 MB | 4,421x better |\
| Matrix Operations | O(1M) elements | O(64M) elements | 64x slower (standard) |\
| Cache Efficiency | Poor (large matrices) | Excellent (small arrays) | Significant advantage |\
\
### Building an Ensemble: _SequentiallyBootstrappedBaggingClassifier_\
\
Now that we have an optimized sequential bootstrap sampler, we can integrate it into a complete machine learning ensemble. The _SequentiallyBootstrappedBaggingClassifier_ combines the temporal-awareness of sequential bootstrapping with the variance-reduction power of ensemble methods.\
\
#### Why Bagging Works—And Why It Needs Sequential Bootstrap\
\
Bootstrap Aggregating (Bagging) is one of the most effective ensemble methods in machine learning. The core idea is elegant:\
\
1. Generate multiple bootstrap samples from your training data\
2. Train a separate model on each sample\
3. Aggregate predictions through voting (classification) or averaging (regression)\
\
This works beautifully when samples are independent. Each bootstrap sample provides a slightly different "view" of the data, and aggregating across these views reduces variance without increasing bias.\
\
But in financial ML with overlapping labels, standard bagging fails catastrophically.\
\
Each bootstrap sample inadvertently includes many copies of the same temporal patterns due to label concurrency. The ensemble learns the same patterns multiple times, leading to:\
\
- Overconfident predictions – Models have seen the same pattern 10+ times and believe it's highly reliable\
- Underestimated variance – Different bootstrap samples aren't truly independent\
- Poor generalization – The true frequency of patterns is much lower than the training data suggests\
\
Sequential bootstrapping solves this by ensuring each bootstrap sample maximizes temporal independence, giving the ensemble genuinely diverse training sets.\
\
**Architecture Overview**\
\
The _SequentiallyBootstrappedBaggingClassifier_ extends scikit-learn's _BaggingClassifier_ with three key modifications:\
\
1. Sequential sampling – Uses _seq\_bootstrap\_optimized_ instead of uniform random sampling\
2. Temporal metadata tracking – Maintains _samples\_info\_sets_ (label start/end times) and price\_bars\_index\
3. Active indices precomputation – Builds sparse index mapping once, reuses across all estimators\
\
### Implementation Walkthrough\
\
Step 1: Initialization and Metadata\
\
The classifier requires temporal metadata that standard bagging doesn't need:\
\
```\
def __init__(\
    self,\
    samples_info_sets,    # NEW: label temporal spans\
    price_bars_index,     # NEW: price bar timestamps\
    estimator=None,\
    n_estimators=10,\
    max_samples=1.0,\
    max_features=1.0,\
    bootstrap_features=False,\
    oob_score=False,\
    warm_start=False,\
    n_jobs=None,\
    random_state=None,\
    verbose=0,\
):\
```\
\
Key parameters explained:\
\
- _samples\_info\_sets (pd.Series)_: Index contains label start times ( _t0_), values contain label end times ( _t1_). This captures the temporal span of each observation's label.\
- _price\_bars\_index (pd.DatetimeIndex)_: Timestamps of all price bars used to construct labels. Required to map temporal spans to bar indices.\
- estimator : Base classifier (defaults to _DecisionTreeClassifier_). Each ensemble member is a clone of this estimator.\
- _n\_estimators_: Number of models in the ensemble. More estimators = smoother predictions but longer training.\
- _max\_samples_: Bootstrap sample size. If float in (0,1\], it's a fraction of training size; if _int_, it's the exact count.\
- _bootstrap\_features_: Whether to also subsample features (increases diversity but may weaken individual models).\
\
Step 2: Active Indices Computation\
\
Before sampling, we precompute the sparse index mapping once:\
\
```\
def _fit(self, X, y, max_samples=None, sample_weight=None):\
    # ... validation and setup ...\
\
    # Compute active indices mapping (once, cached for all estimators)\
    if self.active_indices_ is None:\
        self.active_indices_ = get_active_indices(\
            self.samples_info_sets,\
            self.price_bars_index\
        )\
```\
\
Why precompute? Computing active indices is _O(n)_ and deterministic—it depends only on label timestamps, not on any randomness. Computing it once and reusing saves time when training many estimators.\
\
Memory efficiency: As shown in the memory analysis section, _active\_indices\__ uses _O(n·k)_ memory where k is average label length, compared to _O(n²)_ for a dense indicator matrix. For 8,000 samples, this means 0.28 MB vs 1,238 MB—a 4,421:1 compression ratio.\
\
Step 3: Custom Bootstrap Sample Generation\
\
The key innovation: replacing uniform random sampling with sequential bootstrap:\
\
```\
def _generate_bagging_indices(\
    random_state, bootstrap_features, n_features, max_features, max_samples, active_indices\
):\
    """Randomly draw feature and sample indices."""\
    # Get valid random state - this returns a RandomState object\
    random_state_obj = check_random_state(random_state)\
\
    # Draw samples using sequential bootstrap\
    if isinstance(max_samples, numbers.Integral):\
        sample_indices = seq_bootstrap(\
            active_indices, sample_length=max_samples, random_seed=random_state_obj\
        )\
    elif isinstance(max_samples, numbers.Real):\
        n_samples = int(round(max_samples * len(active_indices)))\
        sample_indices = seq_bootstrap(\
            active_indices, sample_length=n_samples, random_seed=random_state_obj\
        )\
    else:\
        sample_indices = seq_bootstrap(\
            active_indices, sample_length=None, random_seed=random_state_obj\
        )\
\
    # Draw feature indices only if bootstrap_features is True\
    if bootstrap_features:\
        if isinstance(max_features, numbers.Integral):\
            n_feat = max_features\
        elif isinstance(max_features, numbers.Real):\
            n_feat = int(round(max_features * n_features))\
        else:\
            raise ValueError("max_features must be int or float when bootstrap_features=True")\
\
        feature_indices = _generate_random_features(\
            random_state_obj, bootstrap_features, n_features, n_feat\
        )\
    else:\
        # When not bootstrapping features, return None (will be handled downstream)\
        feature_indices = None\
\
    return sample_indices, feature_indices\
```\
\
Critical insight: We use sequential bootstrap for _samples_ (temporal dimension) but standard random sampling for _features_ (if _bootstrap\_features=True_). This is correct because:\
\
- Temporal overlap occurs across observations (rows), not features (columns)\
- Feature correlation is orthogonal to temporal concurrency\
- Standard feature bagging increases diversity without temporal issues\
\
Step 4: Parallel Estimator Training\
\
Training multiple estimators is embarrassingly parallel—each can be trained independently:\
\
```\
def _parallel_build_estimators(\
    n_estimators, ensemble, X, y, active_indices, sample_weight, seeds, total_n_estimators, verbose\
):\
    """Private function used to build a batch of estimators within a job."""\
    # Retrieve settings\
    n_samples, n_features = X.shape\
    max_samples = ensemble._max_samples\
    max_features = ensemble.max_features\
    bootstrap_features = ensemble.bootstrap_features\
    support_sample_weight = has_fit_parameter(ensemble.estimator_, "sample_weight")\
\
    # Build estimators\
    estimators = []\
    estimators_samples = []\
    estimators_features = []\
\
    for i in range(n_estimators):\
        if verbose > 1:\
            print(\
                "Building estimator %d of %d for this parallel run (total %d)..."\
                % (i + 1, n_estimators, total_n_estimators)\
            )\
\
        random_state = seeds[i]\
        estimator = ensemble._make_estimator(append=False, random_state=random_state)\
\
        # Draw samples and features\
        sample_indices, feature_indices = _generate_bagging_indices(\
            random_state, bootstrap_features, n_features, max_features, max_samples, active_indices\
        )\
\
        # Draw samples, using sample weights if supported\
        if support_sample_weight and sample_weight is not None:\
            curr_sample_weight = sample_weight[sample_indices]\
        else:\
            curr_sample_weight = None\
\
        # Store None for features if no bootstrapping (memory optimization)\
        if bootstrap_features:\
            estimators_features.append(feature_indices)\
        else:\
            estimators_features.append(None)  # Don't store redundant feature arrays\
\
        estimators_samples.append(sample_indices)\
\
        # Select data\
        if bootstrap_features:\
            X_ = X[sample_indices][:, feature_indices]\
        else:\
            X_ = X[sample_indices]  # Use all features\
\
        y_ = y[sample_indices]\
\
        estimator.fit(X_, y_, sample_weight=curr_sample_weight)\
        estimators.append(estimator)\
\
    return estimators, estimators_features, estimators_samples\
```\
\
Parallel efficiency: When n\_jobs=-1 , the implementation uses all CPU cores. Training 100 estimators on 8 cores means ~12 estimators per core, processed simultaneously. This provides near-linear speedup for large ensembles.\
\
Step 5: Out-of-Bag (OOB) Scoring\
\
One of bagging's most valuable features is built-in validation through OOB samples:\
\
```\
def _set_oob_score(self, X, y):\
    """Compute out-of-bag score"""\
\
    # Safeguard: Ensure n_classes_ is set\
    if not hasattr(self, "n_classes_"):\
        self.classes_ = np.unique(y)\
        self.n_classes_ = len(self.classes_)\
\
    n_samples = y.shape[0]\
    n_classes = self.n_classes_\
\
    predictions = np.zeros((n_samples, n_classes))\
\
    for estimator, samples, features in zip(\
        self.estimators_, self._estimators_samples, self.estimators_features_\
    ):\
        # Create mask for OOB samples\
        mask = ~indices_to_mask(samples, n_samples)\
\
        if np.any(mask):\
            # Get predictions for OOB samples\
            X_oob = X[mask]\
\
            # If features is None, use all features; otherwise subset\
            if features is not None:\
                X_oob = X_oob[:, features]\
\
            predictions[mask] += estimator.predict_proba(X_oob)\
\
    # Average predictions\
    denominator = np.sum(predictions != 0, axis=1)\
    denominator[denominator == 0] = 1  # avoid division by zero\
    predictions /= denominator[:, np.newaxis]\
\
    # Compute OOB score\
    oob_decision_function = predictions\
    oob_prediction = np.argmax(predictions, axis=1)\
\
    if n_classes == 2:\
        oob_prediction = oob_prediction.astype(np.int64)\
\
    self.oob_decision_function_ = oob_decision_function\
    self.oob_prediction_ = oob_prediction\
    self.oob_score_ = accuracy_score(y, oob_prediction)\
```\
\
Why OOB matters in financial ML:\
\
- No data waste: Every sample serves double duty—training some estimators, validating others\
- Honest estimates: OOB score approximates cross-validation without the computational cost\
- Early stopping signal: Monitor OOB score during training to detect overfitting\
- Temporal safety: With sequential bootstrap, OOB samples are truly independent temporally\
\
Step 6: Class Extension\
\
We bring it all together by creating the classes _SequentiallyBootstrappedBaseBagging_, _SequentiallyBootstrappedBaggingClassifier_ and _SequentiallyBootstrappedBaggingRegressor,_ whichcan be found in _sb\_bagging.py._\
\
#### Complete Usage Example\
\
```\
import pandas as pd\
import numpy as np\
from sklearn.tree import DecisionTreeClassifier\
from sklearn.metrics import classification_report\
\
# Assume we have triple-barrier labels from Part 3\
# samples_info_sets: pd.Series with index=t0, values=t1\
# price_bars: pd.DataFrame with DatetimeIndex\
# X: feature matrix, y: labels\
\
# Initialize classifier\
clf = SequentiallyBootstrappedBaggingClassifier(\
    samples_info_sets=samples_info_sets,  # Label temporal spans\
    price_bars_index=price_bars.index,    # Bar timestamps\
    estimator=DecisionTreeClassifier(\
        max_depth=6,\
        min_samples_leaf=50\
    ),\
    n_estimators=100,                  # Large ensemble for stability\
    max_samples=0.5,                   # Use 50% of data per estimator\
    bootstrap_features=True,           # Also subsample features\
    max_features=0.7,                  # Use 70% of features per estimator\
    oob_score=True,                    # Enable OOB validation\
    n_jobs=-1,                        # Use all CPU cores\
    random_state=42,                  # Reproducibility\
    verbose=1\
)\
\
# Train ensemble\
clf.fit(X_train, y_train)\
\
# Inspect OOB performance (no test set needed!)\
print(f"OOB Score: {clf.oob_score_:.4f}")\
\
# Make predictions on test set\
y_pred = clf.predict(X_test)\
y_proba = clf.predict_proba(X_test)\
\
# Evaluate\
print(classification_report(y_test, y_pred))\
\
# Access individual estimators if needed\
print(f"Number of estimators: {len(clf.estimators_)}")\
print(f"Average sample size: {np.mean([len(s) for s in clf.estimators_samples_]):.0f}")\
```\
\
**Parameter Tuning Guidelines**\
\
_n\_estimators_(Number of Models)\
\
- Small datasets (<1,000 samples): 50-100 estimators\
- Medium datasets (1,000-10,000): 100-200 estimators\
- Large datasets (>10,000): 200-500 estimators\
- Rule of thumb: More is better until OOB score plateaus (monitor during training)\
\
_max\_samples_(Bootstrap Sample Size)\
\
- High concurrency (>10 overlaps/bar): Use smaller samples (0.3-0.5) to maximize diversity\
- Low concurrency (<5 overlaps/bar): Can use larger samples (0.6-0.8) safely\
- Trade-off: Smaller samples → more diversity but weaker individual models\
\
_bootstrap\_features_(Feature Subsampling)\
\
- Enable when: High feature count (>50), features are correlated, seeking maximum diversity\
- Disable when: Few features (<20), each feature is critical, interpretability matters\
- Recommended max\_features: 0.5-0.7 when enabled (too low weakens individual models)\
\
**Comparison: Standard vs Sequential Bootstrap Bagging**\
\
Let's see the performance difference on a real trading strategy:\
\
```\
from sklearn.ensemble import BaggingClassifier\
\
# Standard bagging (temporal leakage)\
standard_clf = BaggingClassifier(\
    estimator=DecisionTreeClassifier(max_depth=6),\
    n_estimators=100,\
    max_samples=0.5,\
    oob_score=True,\
    random_state=42\
)\
\
# Sequential bootstrap bagging (temporal awareness)\
sequential_clf = SequentiallyBootstrappedBaggingClassifier(\
    samples_info_sets=samples_info_sets,\
    price_bars_index=price_bars.index,\
    estimator=DecisionTreeClassifier(max_depth=6),\
    n_estimators=100,\
    max_samples=0.5,\
    oob_score=True,\
    random_state=42\
)\
\
# Train both\
standard_clf.fit(X_train, y_train)\
sequential_clf.fit(X_train, y_train)\
\
# Compare results\
print("Standard Bagging:")\
print(f"  OOB Score: {standard_clf.oob_score_:.4f}")\
print(f"  Test Accuracy: {standard_clf.score(X_test, y_test):.4f}")\
\
print("\nSequential Bootstrap Bagging:")\
print(f"  OOB Score: {sequential_clf.oob_score_:.4f}")\
print(f"  Test Accuracy: {sequential_clf.score(X_test, y_test):.4f}")\
```\
\
Typical results on financial data with high concurrency:\
\
| Metric | Standard Bagging | Sequential Bagging | Improvement |\
| --- | --- | --- | --- |\
| OOB-Test Gap | 0.124 | 0.013 | **-89.5%** |\
\
### Integration with Cross-Validation\
\
While OOB scoring is convenient, proper evaluation requires purged cross-validation to prevent temporal leakage:\
\
```\
from mlfinlab.cross_validation import PurgedKFold\
\
# Setup purged cross-validation\
cv = PurgedKFold(\
    n_splits=5,\
    samples_info_sets=samples_info_sets,\
    pct_embargo=0.01  # Embargo 1% of data after each fold\
)\
```\
\
**Important:** Even with sequential bootstrap sampling _within_ each estimator, you still need purged CV to evaluate the _ensemble_ properly. Sequential bootstrap handles intra-estimator overlap; purged CV handles inter-fold temporal leakage.\
\
We must implement our own cross-validation methods to accommodate our sequentially bootstrapped bagging models. The function below provides a comprehensive of analysis of what occurs in each fold and is necessary for deeper analysis of the temporal dependencies of your data.\
\
```\
def analyze_cross_val_scores(\
    classifier: ClassifierMixin,\
    X: pd.DataFrame,\
    y: pd.Series,\
    cv_gen: BaseCrossValidator,\
    sample_weight_train: np.ndarray = None,\
    sample_weight_score: np.ndarray = None,\
):\
    # pylint: disable=invalid-name\
    # pylint: disable=comparison-with-callable\
    """\
    Advances in Financial Machine Learning, Snippet 7.4, page 110.\
\
    Using the PurgedKFold Class.\
\
    Function to run a cross-validation evaluation of the classifier using sample weights and a custom CV generator.\
    Scores are computed using accuracy_score, probability_weighted_accuracy, log_loss and f1_score.\
\
    Note: This function is different to the book in that it requires the user to pass through a CV object. The book\
    will accept a None value as a default and then resort to using PurgedCV, this also meant that extra arguments had to\
    be passed to the function. To correct this we have removed the default and require the user to pass a CV object to\
    the function.\
\
    Example:\
\
    .. code-block:: python\
\
        cv_gen = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)\
        scores_array = ml_cross_val_scores_all(classifier, X, y, cv_gen, sample_weight_train=sample_train,\
                                               sample_weight_score=sample_score, scoring=accuracy_score)\
\
    :param classifier: (BaseEstimator) A scikit-learn Classifier object instance.\
    :param X: (pd.DataFrame) The dataset of records to evaluate.\
    :param y: (pd.Series) The labels corresponding to the X dataset.\
    :param cv_gen: (BaseCrossValidator) Cross Validation generator object instance.\
    :param sample_weight_train: (np.array) Sample weights used to train the model for each record in the dataset.\
    :param sample_weight_score: (np.array) Sample weights used to evaluate the model quality.\
    :return: tuple(dict, pd.DataFrame, dict) The computed scores, a data frame of mean and std. deviation, and a dict of data in each fold\
    """\
    scoring_methods = [\
        accuracy_score,\
        probability_weighted_accuracy,\
        log_loss,\
        precision_score,\
        recall_score,\
        f1_score,\
    ]\
    ret_scores = {\
        (\
            scoring.__name__.replace("_score", "")\
            .replace("probability_weighted_accuracy", "pwa")\
            .replace("log_loss", "neg_log_loss")\
        ): np.zeros(cv_gen.n_splits)\
        for scoring in scoring_methods\
    }\
\
    # If no sample_weight then broadcast a value of 1 to all samples (full weight).\
    if sample_weight_train is None:\
        sample_weight_train = np.ones((X.shape[0],))\
\
    if sample_weight_score is None:\
        sample_weight_score = np.ones((X.shape[0],))\
\
    seq_bootstrap = isinstance(classifier, SequentiallyBootstrappedBaggingClassifier)\
    if seq_bootstrap:\
        t1 = classifier.samples_info_sets.copy()\
        common_idx = t1.index.intersection(y.index)\
        X, y, t1 = X.loc[common_idx], y.loc[common_idx], t1.loc[common_idx]\
        if t1.empty:\
            raise KeyError(f"samples_info_sets not aligned with data")\
        classifier.set_params(oob_score=False)\
\
    cms = []  # To store confusion matrices\
\
    # Score model on KFolds\
    for i, (train, test) in enumerate(cv_gen.split(X=X, y=y)):\
        if seq_bootstrap:\
            classifier = clone(classifier).set_params(\
                samples_info_sets=t1.iloc[train]\
            )  # Create new instance\
        fit = classifier.fit(\
            X=X.iloc[train, :],\
            y=y.iloc[train],\
            sample_weight=sample_weight_train[train],\
        )\
        prob = fit.predict_proba(X.iloc[test, :])\
        pred = fit.predict(X.iloc[test, :])\
        params = dict(\
            y_true=y.iloc[test],\
            y_pred=pred,\
            labels=classifier.classes_,\
            sample_weight=sample_weight_score[test],\
        )\
\
        for method, scoring in zip(ret_scores.keys(), scoring_methods):\
            if scoring in (probability_weighted_accuracy, log_loss):\
                score = scoring(\
                    y.iloc[test],\
                    prob,\
                    sample_weight=sample_weight_score[test],\
                    labels=classifier.classes_,\
                )\
                if method == "neg_log_loss":\
                    score *= -1\
            else:\
                try:\
                    score = scoring(**params)\
                except:\
                    del params["labels"]\
                    score = scoring(**params)\
                    params["labels"] = classifier.classes_\
\
            ret_scores[method][i] = score\
\
        cms.append(confusion_matrix(**params).round(2))\
\
    # Mean and standard deviation of scores\
    scores_df = pd.DataFrame.from_dict(\
        {\
            scoring: {"mean": scores.mean(), "std": scores.std()}\
            for scoring, scores in ret_scores.items()\
        },\
        orient="index",\
    )\
\
    # Extract TN, TP, FP, FN for each fold\
    confusion_matrix_breakdown = []\
    for i, cm in enumerate(cms, 1):\
        if cm.shape == (2, 2):  # Binary classification\
            tn, fp, fn, tp = cm.ravel()\
            confusion_matrix_breakdown.append({"fold": i, "TN": tn, "FP": fp, "FN": fn, "TP": tp})\
        else:\
            # For multi-class, you might want different handling\
            confusion_matrix_breakdown.append({"fold": i, "confusion_matrix": cm})\
\
    return ret_scores, scores_df, confusion_matrix_breakdown\
```\
\
Below are results from a meta-labeled Bollinger Band mean-reversion strategy (see _sample\_weights.ipynb_). All scores are better and have lower variance when sequential bootstrapping is employed.\
\
Cross-Validation Results:\
\
|  | Random Forest | Standard Bagging | **Sequential Bagging** |\
| --- | --- | --- | --- |\
| accuracy | 0.509 ± 0.024 | 0.515 ± 0.024 | 0.527 ± 0.015 |\
| pwa | 0.513 ± 0.038 | 0.519 ± 0.039 | 0.544 ± 0.018 |\
| neg\_log\_loss | -0.695 ± 0.005 | -0.694 ± 0.005 | -0.692 ± 0.001 |\
| precision | 0.637 ± 0.027 | 0.643 ± 0.026 | 0.637 ± 0.026 |\
| recall | 0.476 ± 0.095 | 0.484 ± 0.098 | 0.567 ± 0.038 |\
| f1 | 0.539 ± 0.065 | 0.546 ± 0.067 | 0.599 ± 0.026 |\
\
Out-of-Sample Results:\
\
|  | Random Forest | Standard Bagging | **Sequential Bagging** |\
| --- | --- | --- | --- |\
| accuracy | 0.505780 | 0.496628 | 0.519750 |\
| pwa | 0.493505 | 0.495487 | 0.523738 |\
| neg\_log\_loss | -0.696703 | -0.696612 | -0.692669 |\
| precision | 0.650811 | 0.646396 | 0.633913 |\
| recall | 0.461303 | 0.439847 | 0.558621 |\
| f1 | 0.539910 | 0.523484 | 0.593890 |\
| oob | 0.516976 | 0.516133 | 0.522153 |\
| oob\_test\_gap | 0.011195 | 0.019505 | 0.002403 |\
\
Key observations:\
\
1. The smaller OOB-test gap for sequential bagging (0.002 vs 0.019) shows the OOB estimate is trustworthy—OOB and test performance are closely aligned, indicating no hidden temporal leakage.\
2. Higher test accuracy demonstrates better generalization to truly unseen data\
\
**Advanced: Custom OOB Metrics**\
\
The built-in _oob\_score\__ uses accuracy for classification and R² for regression. For financial applications, you often need custom metrics:\
\
```\
def compute_custom_oob_metrics(clf, X, y, sample_weight=None):\
    """\
    Compute custom OOB metrics (F1, AUC, precision/recall) for a fitted ensemble.\
\
    Args:\
        clf: Fitted SequentiallyBootstrappedBaggingClassifier\
        X: Feature matrix used in training\
        y: True labels\
        sample_weight: Optional sample weights\
\
    Returns:\
        dict: Custom OOB metric values\
    """\
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score\
\
    n_samples = y.shape[0]\
    n_classes = clf.n_classes_\
\
    # Accumulate OOB predictions\
    oob_proba = np.zeros((n_samples, n_classes))\
    oob_count = np.zeros(n_samples)\
\
    for estimator, samples, features in zip(\
        clf.estimators_,\
        clf.estimators_samples_,\
        clf.estimators_features_\
    ):\
        mask = ~indices_to_mask(samples, n_samples)\
        if np.any(mask):\
            X_oob = X[mask][:, features]\
            oob_proba[mask] += estimator.predict_proba(X_oob)\
            oob_count[mask] += 1\
\
    # Average and get predictions\
    oob_mask = oob_count > 0\
    oob_proba[oob_mask] /= oob_count[oob_mask, np.newaxis]\
    oob_pred = np.argmax(oob_proba, axis=1)\
\
    # Compute metrics on samples with OOB predictions\
    y_oob = y[oob_mask]\
    pred_oob = oob_pred[oob_mask]\
    proba_oob = oob_proba[oob_mask]\
\
    metrics = {\
        'f1': f1_score(y_oob, pred_oob, average='weighted'),\
        'precision': precision_score(y_oob, pred_oob, average='weighted'),\
        'recall': recall_score(y_oob, pred_oob, average='weighted'),\
        'coverage': oob_mask.sum() / n_samples  # Fraction with OOB predictions\
    }\
\
    # Add AUC for binary classification\
    if n_classes == 2:\
        metrics['auc'] = roc_auc_score(y_oob, proba_oob[:, 1])\
\
    return metrics\
\
# Usage\
oob_metrics = compute_custom_oob_metrics(sequential_clf, X_train, y_train)\
print("Custom OOB Metrics:")\
for metric, value in oob_metrics.items():\
    print(f"  {metric}: {value:.4f}")\
```\
\
### Production Deployment Considerations\
\
#### Memory Management\
\
Large ensembles can consume significant memory. Monitor and optimize:\
\
```\
import sys\
\
# Check ensemble memory footprint\
def estimate_ensemble_size(clf):\
    """Estimate memory usage of fitted ensemble."""\
    total_bytes = 0\
\
    # Estimators\
    for est in clf.estimators_:\
        total_bytes += sys.getsizeof(est)\
\
    # Sample indices\
    for samples in clf.estimators_samples_:\
        total_bytes += samples.nbytes\
\
    # Feature indices\
    if clf.estimators_features_ is not None:\
        for features in clf.estimators_features_:\
            total_bytes += features.nbytes\
\
    return total_bytes / (1024 ** 2)  # Convert to MB\
\
size_mb = estimate_ensemble_size(sequential_clf)\
print(f"Ensemble size: {size_mb:.2f} MB")\
```\
\
Model Serialization\
\
Save and load trained ensembles efficiently:\
\
```\
import joblib\
\
# Save entire ensemble\
joblib.dump(sequential_clf, 'sequential_bagging_model.pkl', compress=3)\
\
# Load for prediction\
loaded_clf = joblib.load('sequential_bagging_model.pkl')\
\
# Verify predictions match\
original_pred = sequential_clf.predict_proba(X_test)\
loaded_pred = loaded_clf.predict_proba(X_test)\
assert np.allclose(original_pred, loaded_pred)\
```\
\
**Common Pitfalls and Solutions**\
\
Pitfall 1: Forgetting to Pass Temporal Metadata\
\
Problem: Attempting to use the classifier without samples\_info\_sets or price\_bars\_index .\
\
Solution: Always ensure these are properly constructed from your labeling process:\
\
```\
# From triple-barrier labeling (Part 3)\
events = get_events(\
    close=close_prices,\
    t_events=trigger_times,\
    pt_sl=[1, 1],\
    target=daily_vol,\
    min_ret=0.01,\
    num_threads=4,\
    vertical_barrier_times=vertical_barriers\
)\
\
# events['t1'] contains end times - this is samples_info_sets\
samples_info_sets = events['t1']\
price_bars_index = close_prices.index\
\
# Now safe to use\
clf = SequentiallyBootstrappedBaggingClassifier(\
    samples_info_sets=samples_info_sets,\
    price_bars_index=price_bars_index,\
    # ... other params ...\
)\
```\
\
Pitfall 2: Mismatched Index Lengths\
\
Problem: _len(samples\_info\_sets) != len(X)_ causes cryptic errors.\
\
Solution: Always align your features, labels, and metadata:\
\
```\
# After computing features and labels, ensure alignment\
assert len(X) == len(y) == len(samples_info_sets), \\
    "Feature matrix, labels, and metadata must have same length"\
\
# If they don't match, use index intersection\
common_idx = X.index.intersection(y.index).intersection(samples_info_sets.index)\
X_aligned = X.loc[common_idx]\
y_aligned = y.loc[common_idx]\
samples_aligned = samples_info_sets.loc[common_idx]\
```\
\
Pitfall 3: Ignoring Warm Start Behavior\
\
Problem: Setting _warm\_start=True_ then changing _n\_estimators_ doesn't retrain existing estimators.\
\
Solution: Understand warm start only _adds_ estimators:\
\
```\
# Initial training with 50 estimators\
clf = SequentiallyBootstrappedBaggingClassifier(\
    samples_info_sets=samples_info_sets,\
    price_bars_index=price_bars_index,\
    n_estimators=50,\
    warm_start=True,\
    random_state=42\
)\
clf.fit(X_train, y_train)\
\
# Add 50 more estimators (total=100)\
clf.n_estimators = 100\
clf.fit(X_train, y_train)  # Only trains 50 new estimators\
\
print(len(clf.estimators_))  # Output: 100\
```\
\
#### Benchmarking Against Alternatives\
\
How does sequential bootstrap bagging compare to other ensemble methods on financial data?\
\
```\
from sklearn.ensemble import (\
    RandomForestClassifier,\
    GradientBoostingClassifier,\
    BaggingClassifier\
)\
from sklearn.model_selection import cross_val_score\
\
# Define models\
models = {\
    'Standard Bagging': BaggingClassifier(\
        estimator=DecisionTreeClassifier(max_depth=6),\
        n_estimators=100,\
        random_state=42\
    ),\
    'Random Forest': RandomForestClassifier(\
        n_estimators=100,\
        max_depth=6,\
        random_state=42\
    ),\
    'Sequential Bagging': SequentiallyBootstrappedBaggingClassifier(\
        samples_info_sets=samples_info_sets,\
        price_bars_index=price_bars_index,\
        estimator=DecisionTreeClassifier(max_depth=6),\
        n_estimators=100,\
        random_state=42\
    )\
}\
\
# Benchmark with purged K-Fold CV\
results = {}\
cv_gen = PurgedKFold(n_splits, t1, pct_embargo)\
for name, model in models.items():\
    raw_scores, scores_df, folds = analyze_cross_val_scores(\
	model, X, y, cv_gen,\
	sample_weights_train=w,\
	sample_weights_score=w,\
	)\
    results[name] = dict(scores=scores_df, folds=folds)\
```\
\
**Summary and Best Practices**\
\
The _SequentiallyBootstrappedBaggingClassifier_ brings the power of ensemble learning to financial time series by addressing the fundamental problem of label concurrency. Here are the key takeaways:\
\
When to Use Sequential Bootstrap Bagging:\
\
- Triple-barrier labeling or any method that creates temporally overlapping labels\
- High-frequency data where observations naturally overlap\
- Any financial ML task where temporal structure matters\
- Production systems requiring honest variance estimates\
\
When Standard Bagging is Sufficient:\
\
- Daily or lower frequency data with minimal label overlap\
- Cross-sectional predictions (predicting across assets, not time)\
- Scenarios where temporal leakage has been eliminated through other means\
\
Configuration Checklist for Production:\
\
1. ✓ Verify samples\_info\_sets and price\_bars\_index are properly aligned\
2. ✓ Enable oob\_score=True for monitoring during training\
3. ✓ Set n\_jobs=-1 to leverage all CPU cores\
4. ✓ Use random\_state for reproducibility\
5. ✓ Monitor memory usage for large ensembles\
6. ✓ Validate with purged/embargoed cross-validation\
7. ✓ Compare OOB vs test performance to detect remaining leakage\
\
Performance Optimization Tips:\
\
- Precompute active\_indices\_ once and cache it\
- Use smaller max\_samples with high concurrency\
- Enable bootstrap\_features for high-dimensional data\
- Batch predictions for low-latency applications\
- Serialize models with compression for deployment\
\
With sequential bootstrap bagging, you now have a production-ready ensemble method that respects the temporal structure of financial data while delivering the variance reduction benefits that make bagging so powerful in traditional machine learning.\
\
### Deploying Sequential Bootstrap Models to MQL5 via ONNX\
\
After training robust sequential bootstrap models in Python, the next critical step is deploying them to MetaTrader 5 for live trading. ONNX (Open Neural Network Exchange) provides the most reliable bridge between Python's rich ML ecosystem and MQL5's production environment.\
\
**Why ONNX for MQL5 Deployment**\
\
ONNX offers several compelling advantages for deploying financial ML models:\
\
- Native MetaTrader 5 support – MetaTrader 5 has built-in ONNX runtime, no external dependencies required\
- Performance – Models run as compiled C++ code, delivering microsecond-level predictions\
- Cross-platform – Same model works on Windows, Mac, and Linux MetaTrader 5 installations\
- Broad compatibility – Supports scikit-learn ensembles, including our sequential bootstrap models\
- Version control – Binary model files are easily versioned and deployed\
\
Key limitations to understand:\
\
- Ensemble metadata (OOB scores, estimator samples) is not preserved—only prediction logic\
- Models cannot be retrained in MQL5; Python remains the training environment\
- Large ensembles (200+ estimators) increase model load time and memory footprint\
- Feature calculations must be manually replicated in MQL5 with exact parity to Python\
\
#### Complete Deployment Pipeline\
\
Step 1: Export Trained Model to ONNX Format\
\
After training your sequential bootstrap classifier, convert it to ONNX:\
\
```\
import onnx\
from skl2onnx import convert_sklearn\
from skl2onnx.common.data_types import FloatTensorType\
import numpy as np\
\
# Your trained sequential bootstrap model\
clf = SequentiallyBootstrappedBaggingClassifier(\
    samples_info_sets=samples_info_sets,\
    price_bars_index=price_bars.index,\
    estimator=DecisionTreeClassifier(max_depth=6, min_samples_leaf=50),\
    n_estimators=100,\
    max_samples=0.5,\
    random_state=42\
)\
clf.fit(X_train, y_train)\
\
# Define input shape - CRITICAL: must match feature count exactly\
n_features = X_train.shape[1]\
initial_type = [('float_input', FloatTensorType([None, n_features]))]\
\
# Convert to ONNX with appropriate settings\
onnx_model = convert_sklearn(\
    clf,\
    initial_types=initial_type,\
    target_opset=12,  # MT5 supports opset 9-15\
    options={\
        'zipmap': False  # Return raw probabilities, not dictionary\
    }\
)\
\
# Save model file\
model_filename = "sequential_bagging_model.onnx"\
with open(model_filename, "wb") as f:\
    f.write(onnx_model.SerializeToString())\
\
print(f"Model exported: {len(onnx_model.SerializeToString()) / 1024:.2f} KB")\
print(f"Input features: {n_features}")\
print(f"Output classes: {len(clf.classes_)}")\
```\
\
Step 2: Verify ONNX Model Correctness\
\
Before deployment, always verify that ONNX predictions match your original model:\
\
```\
import onnxruntime as rt\
\
# Load ONNX model\
sess = rt.InferenceSession(model_filename)\
\
# Inspect model structure\
input_name = sess.get_inputs()[0].name\
output_name = sess.get_outputs()[0].name\
print(f"Input tensor name: {input_name}")\
print(f"Output tensor name: {output_name}")\
\
# Test with sample data\
X_test_sample = X_test[:5].astype(np.float32)\
\
# Original model predictions\
sklearn_pred = clf.predict_proba(X_test_sample)\
\
# ONNX model predictions\
onnx_pred = sess.run([output_name], {input_name: X_test_sample})[0]\
\
# Verify predictions match within tolerance\
print("\nVerification Results:")\
print("Scikit-learn predictions:\n", sklearn_pred[:3])\
print("\nONNX predictions:\n", onnx_pred[:3])\
print(f"\nMax absolute difference: {np.abs(sklearn_pred - onnx_pred).max():.2e}")\
\
assert np.allclose(sklearn_pred, onnx_pred, atol=1e-5), "ERROR: Predictions don't match!"\
print("✓ Model verification passed")\
```\
\
Step 3: Document Feature Engineering Pipeline\
\
The most common deployment failure is feature misalignment. Document your exact feature calculations:\
\
```\
import json\
from datetime import datetime\
\
# Document feature metadata for MQL5 implementation\
feature_metadata = {\
    'model_version': 'v1.0_seq_bagging',\
    'timestamp': datetime.now().isoformat(),\
    'n_features': n_features,\
    'n_estimators': 100,\
    'lookback_period': 20,\
    'feature_names': [\
        'bb_position',         # (close - bb_middle) / (bb_upper - bb_lower)\
        'bb_width',           # (bb_upper - bb_lower) / bb_middle\
        'return_1d',          # (close[0] - close[1]) / close[1]\
        'return_5d',          # (close[0] - close[5]) / close[5]\
        'volatility_20d',     # std(returns, 20) / close[0]\
        'volume_ratio',       # volume[0] / ma(volume, 20)\
        'rsi_14',             # RSI with 14-period lookback\
        'mean_reversion_z',  # (close - ma_20) / std_20\
    ],\
    'bb_parameters': {\
        'period': 20,\
        'std_dev': 2.0\
    }\
}\
\
with open('feature_metadata.json', 'w') as f:\
    json.dump(feature_metadata, f, indent=2)\
\
# Create test dataset for validation in MQL5\
test_data = {\
    'features': X_test[:10].tolist(),\
    'expected_predictions': clf.predict_proba(X_test[:10]).tolist(),\
    'expected_classes': clf.predict(X_test[:10]).tolist()\
}\
\
with open('test_predictions.json', 'w') as f:\
    json.dump(test_data, f, indent=2)\
```\
\
### MQL5 Implementation: Bollinger Band Mean Reversion Strategy\
\
Now we implement the complete MQL5 system, starting with precise feature engineering that matches our Python training.\
\
Feature Calculation Module\
\
Create FeatureEngine.mqh to replicate Python feature calculations exactly:\
\
```\
//+------------------------------------------------------------------+\
//| FeatureEngine.mqh                                                 |\
//| Feature calculation engine matching Python training pipeline     |\
//+------------------------------------------------------------------+\
#property strict\
\
class CFeatureEngine {\
private:\
    int m_lookback;\
    int m_bb_period;\
    double m_bb_deviation;\
    int m_rsi_period;\
\
public:\
    CFeatureEngine(int lookback=20, int bb_period=20, double bb_dev=2.0, int rsi_period=14) {\
        m_lookback = lookback;\
        m_bb_period = bb_period;\
        m_bb_deviation = bb_dev;\
        m_rsi_period = rsi_period;\
    }\
\
    // Main feature calculation - must match Python exactly\
    bool CalculateFeatures(const double &close[],\
                              const double &high[],\
                              const double &low[],\
                              const long &volume[],\
                              double &features[]) {\
\
        if(ArraySize(close) < m_lookback + 10) return false;\
\
        // Must have exactly 8 features to match Python\
        ArrayResize(features, 8);\
        int idx = 0;\
\
        // Calculate Bollinger Bands\
        double bb_upper, bb_middle, bb_lower;\
        CalculateBollingerBands(close, m_bb_period, m_bb_deviation,\
                                bb_upper, bb_middle, bb_lower);\
\
        // Feature 1: Bollinger Band Position\
        // Measures where price sits within the bands (-1 to +1)\
        double bb_range = bb_upper - bb_lower;\
        if(bb_range > 0) {\
            features[idx++] = (close[0] - bb_middle) / bb_range;\
        } else {\
            features[idx++] = 0.0;\
        }\
\
        // Feature 2: Bollinger Band Width\
        // Normalized measure of volatility\
        if(bb_middle > 0) {\
            features[idx++] = bb_range / bb_middle;\
        } else {\
            features[idx++] = 0.0;\
        }\
\
        // Feature 3: 1-day return\
        features[idx++] = SafeReturn(close[0], close[1]);\
\
        // Feature 4: 5-day return\
        features[idx++] = SafeReturn(close[0], close[5]);\
\
        // Feature 5: 20-day volatility (annualized)\
        double returns_std = CalculateReturnsStdDev(close, m_lookback);\
        features[idx++] = returns_std / close[0];\
\
        // Feature 6: Volume ratio\
        double vol_ma = CalculateVolumeMA(volume, m_lookback);\
        if(vol_ma > 0) {\
            features[idx++] = (double)volume[0] / vol_ma;\
        } else {\
            features[idx++] = 1.0;\
        }\
\
        // Feature 7: RSI\
        features[idx++] = CalculateRSI(close, m_rsi_period) / 100.0;\
\
        // Feature 8: Mean reversion Z-score\
        double ma_20 = CalculateMA(close, 20);\
        double std_20 = CalculateStdDev(close, 20);\
        if(std_20 > 0) {\
            features[idx++] = (close[0] - ma_20) / std_20;\
        } else {\
            features[idx++] = 0.0;\
        }\
\
        return true;\
    }\
\
private:\
    // Calculate Bollinger Bands using SMA and standard deviation\
    void CalculateBollingerBands(const double &close[], int period, double deviation,\
                                  double &upper, double &middle, double &lower) {\
        middle = CalculateMA(close, period);\
        double std = CalculateStdDev(close, period);\
        upper = middle + deviation * std;\
        lower = middle - deviation * std;\
    }\
\
    // Simple Moving Average\
    double CalculateMA(const double &data[], int period) {\
        double sum = 0.0;\
        for(int i = 0; i < period; i++) {\
            sum += data[i];\
        }\
        return sum / period;\
    }\
\
    // Standard Deviation\
    double CalculateStdDev(const double &data[], int period) {\
        double mean = CalculateMA(data, period);\
        double sum_sq = 0.0;\
        for(int i = 0; i < period; i++) {\
            double diff = data[i] - mean;\
            sum_sq += diff * diff;\
        }\
        return MathSqrt(sum_sq / period);\
    }\
\
    // Standard deviation of returns (not prices)\
    double CalculateReturnsStdDev(const double &close[], int period) {\
        double returns[];\
        ArrayResize(returns, period);\
\
        for(int i = 0; i < period; i++) {\
            returns[i] = SafeReturn(close[i], close[i+1]);\
        }\
\
        return CalculateStdDev(returns, period);\
    }\
\
    // RSI calculation\
    // RSI calculation\
    double CalculateRSI(const double &close[], int period) {\
        double gains = 0.0, losses = 0.0;\
\
        for(int i = 1; i <= period; i++) {\
            double change = close[i-1] - close[i];\
            if(change > 0) {\
                gains += change;\
            } else {\
                losses -= change;\
            }\
        }\
\
        double avg_gain = gains / period;\
        double avg_loss = losses / period;\
\
        if(avg_loss == 0.0) return 100.0;\
\
        double rs = avg_gain / avg_loss;\
        return 100.0 - (100.0 / (1.0 + rs));\
    }\
\
    // Volume moving average\
    double CalculateVolumeMA(const long &volume[], int period) {\
        double sum = 0.0;\
        for(int i = 0; i < period; i++) {\
            sum += (double)volume[i];\
        }\
        return sum / period;\
    }\
\
    // Safe return calculation with division by zero protection\
    double SafeReturn(double current, double previous) {\
        if(previous == 0.0 || MathAbs(previous) < 1e-10) return 0.0;\
        return (current - previous) / previous;\
    }\
};\
```\
\
Main Expert Advisor with ONNX Integration\
\
Create the EA that loads the ONNX model and executes the Bollinger Band mean reversion strategy:\
\
```\
//+------------------------------------------------------------------+\
//| SequentialBaggingEA.mq5                                          |\
//| Bollinger Band Mean Reversion with Sequential Bootstrap Model    |\
//+------------------------------------------------------------------+\
#property copyright "Your Name"\
#property version   "1.00"\
#property strict\
\
#include <Trade\Trade.mqh>\
#include "FeatureEngine.mqh"\
\
//--- Input parameters\
input group "Model Settings"\
input string   InpModelFile = "sequential_bagging_model.onnx"; // ONNX model filename\
input double   InpConfidenceThreshold = 0.60; // Minimum confidence for trade\
\
input group "Feature Parameters"\
input int      InpLookback = 20;           // Feature lookback period\
input int      InpBBPeriod = 20;           // Bollinger Bands period\
input double   InpBBDeviation = 2.0;        // Bollinger Bands deviation\
input int      InpRSIPeriod = 14;          // RSI period\
\
input group "Risk Management"\
input double   InpRiskPercent = 1.0;        // Risk per trade (%)\
input int      InpStopLoss = 200;          // Stop loss (points)\
input int      InpTakeProfit = 400;        // Take profit (points)\
input int      InpMaxTrades = 1;           // Maximum concurrent trades\
\
input group "Trading Hours"\
input bool     InpUseTradingHours = false; // Enable trading hours filter\
input int      InpStartHour = 9;           // Trading start hour\
input int      InpEndHour = 17;            // Trading end hour\
\
//--- Global variables\
long            g_model_handle = INVALID_HANDLE;\
CTrade          g_trade;\
CFeatureEngine  g_features;\
datetime        g_last_bar_time = 0;\
\
//+------------------------------------------------------------------+\
//| Expert initialization function                                   |\
//+------------------------------------------------------------------+\
int OnInit() {\
    // Initialize feature engine\
    g_features.CFeatureEngine(InpLookback, InpBBPeriod, InpBBDeviation, InpRSIPeriod);\
\
    // Load ONNX model from MQL5/Files directory\
    g_model_handle = OnnxCreateFromFile(\
        InpModelFile,\
        ONNX_DEFAULT\
    );\
\
    if(g_model_handle == INVALID_HANDLE) {\
        Print("❌ Failed to load ONNX model: ", InpModelFile);\
        Print("Ensure model is in: Terminal_Data_Folder/MQL5/Files/");\
        return INIT_FAILED;\
    }\
\
    // Verify model structure\
    long input_count, output_count;\
    OnnxGetInputCount(g_model_handle, input_count);\
    OnnxGetOutputCount(g_model_handle, output_count);\
\
    vector input_shape;\
    OnnxGetInputShape(g_model_handle, 0, input_shape);\
\
    Print("✓ Model loaded successfully");\
    Print("  Model file: ", InpModelFile);\
    Print("  Input count: ", input_count);\
    Print("  Output count: ", output_count);\
    Print("  Expected features: ", (int)input_shape[1]);\
    Print("  Confidence threshold: ", InpConfidenceThreshold);\
\
    // Set trade parameters\
    g_trade.SetExpertMagicNumber(20241102);\
    g_trade.SetDeviationInPoints(10);\
    g_trade.SetTypeFilling(ORDER_FILLING_FOK);\
\
    return INIT_SUCCEEDED;\
}\
\
//+------------------------------------------------------------------+\
//| Expert deinitialization function                                 |\
//+------------------------------------------------------------------+\
void OnDeinit(const int reason) {\
    if(g_model_handle != INVALID_HANDLE) {\
        OnnxRelease(g_model_handle);\
        Print("Model released");\
    }\
}\
\
//+------------------------------------------------------------------+\
//| Expert tick function                                             |\
//+------------------------------------------------------------------+\
void OnTick() {\
    // Check for new bar\
    datetime current_bar_time = iTime(_Symbol, _Period, 0);\
    if(current_bar_time == g_last_bar_time) return;\
    g_last_bar_time = current_bar_time;\
\
    // Trading hours filter\
    if(InpUseTradingHours) {\
        MqlDateTime dt;\
        TimeToStruct(TimeCurrent(), dt);\
        if(dt.hour < InpStartHour || dt.hour >= InpEndHour) return;\
    }\
\
    // Get market data\
    double close[], high[], low[];\
    long volume[];\
\
    ArraySetAsSeries(close, true);\
    ArraySetAsSeries(high, true);\
    ArraySetAsSeries(low, true);\
    ArraySetAsSeries(volume, true);\
\
    int required_bars = InpLookback + 10;\
    int copied = CopyClose(_Symbol, _Period, 0, required_bars, close);\
\
    if(copied < required_bars) {\
        Print("Insufficient bars: ", copied, " < ", required_bars);\
        return;\
    }\
\
    CopyHigh(_Symbol, _Period, 0, required_bars, high);\
    CopyLow(_Symbol, _Period, 0, required_bars, low);\
    CopyTickVolume(_Symbol, _Period, 0, required_bars, volume);\
\
    // Calculate features\
    double feature_array[];\
    if(!g_features.CalculateFeatures(close, high, low, volume, feature_array)) {\
        Print("Feature calculation failed");\
        return;\
    }\
\
    // Prepare input matrix for ONNX (must be float32)\
    matrix input_matrix(1, ArraySize(feature_array));\
    for(int i = 0; i < ArraySize(feature_array); i++) {\
        input_matrix[0][i] = (float)feature_array[i];\
    }\
\
    // Run model inference\
    matrix output_matrix;\
    if(!OnnxRun(g_model_handle, ONNX_NO_CONVERSION, input_matrix, output_matrix)) {\
        Print("❌ Model prediction failed!");\
        return;\
    }\
\
    // Extract probabilities\
    // Output shape: [1, 2] for binary classification\
    // Class 0 = SELL signal, Class 1 = BUY signal\
    double prob_sell = output_matrix[0][0];\
    double prob_buy = output_matrix[0][1];\
\
    // Log predictions for monitoring\
    Comment(StringFormat(\
        "Sequential Bootstrap EA\n" +\
        "Time: %s\n" +\
        "Prob SELL: %.2f%%\n" +\
        "Prob BUY: %.2f%%\n" +\
        "Threshold: %.2f%%\n" +\
        "Positions: %d/%d",\
        TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES),\
        prob_sell * 100,\
        prob_buy * 100,\
        InpConfidenceThreshold * 100,\
        PositionsTotal(),\
        InpMaxTrades\
    ));\
\
    // Trading logic: Mean reversion strategy\
    if(PositionsTotal() < InpMaxTrades) {\
\
        // BUY signal: Price at lower band, model predicts reversion up\
        if(prob_buy > InpConfidenceThreshold) {\
            ExecuteBuy(prob_buy, feature_array);\
        }\
        // SELL signal: Price at upper band, model predicts reversion down\
        else if(prob_sell > InpConfidenceThreshold) {\
            ExecuteSell(prob_sell, feature_array);\
        }\
    }\
}\
\
//+------------------------------------------------------------------+\
//| Execute BUY order                                                |\
//+------------------------------------------------------------------+\
void ExecuteBuy(double confidence, const double &features[]) {\
    double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);\
    double sl = price - InpStopLoss * _Point;\
    double tp = price + InpTakeProfit * _Point;\
\
    // Calculate position size based on risk\
    double lot = CalculateLotSize(InpRiskPercent, InpStopLoss);\
\
    // Build comment with BB position for analysis\
    string comment = StringFormat(\
        "SB|BUY|Conf:%.2f|BB:%.3f",\
        confidence,\
        features[0]  // BB position feature\
    );\
\
    if(g_trade.Buy(lot, _Symbol, price, sl, tp, comment)) {\
        Print("✓ BUY executed: Lot=", lot, " Conf=", confidence, " BB=", features[0]);\
    } else {\
        Print("❌ BUY failed: ", g_trade.ResultRetcodeDescription());\
    }\
}\
\
//+------------------------------------------------------------------+\
//| Execute SELL order                                               |\
//+------------------------------------------------------------------+\
void ExecuteSell(double confidence, const double &features[]) {\
    double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);\
    double sl = price + InpStopLoss * _Point;\
    double tp = price - InpTakeProfit * _Point;\
\
    double lot = CalculateLotSize(InpRiskPercent, InpStopLoss);\
\
    string comment = StringFormat(\
        "SB|SELL|Conf:%.2f|BB:%.3f",\
        confidence,\
        features[0]\
    );\
\
    if(g_trade.Sell(lot, _Symbol, price, sl, tp, comment)) {\
        Print("✓ SELL executed: Lot=", lot, " Conf=", confidence, " BB=", features[0]);\
    } else {\
        Print("❌ SELL failed: ", g_trade.ResultRetcodeDescription());\
    }\
}\
\
//+------------------------------------------------------------------+\
//| Calculate lot size based on risk percentage                     |\
//+------------------------------------------------------------------+\
double CalculateLotSize(double risk_percent, int sl_points) {\
    double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);\
    double risk_amount = account_balance * risk_percent / 100.0;\
\
    double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);\
    double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);\
\
    // Calculate value of stop loss in account currency\
    double point_value = tick_value / tick_size;\
    double sl_value = sl_points * _Point * point_value;\
\
    // Calculate lot size\
    double lot_size = risk_amount / sl_value;\
\
    // Normalize to broker's lot step\
    double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);\
    lot_size = MathFloor(lot_size / lot_step) * lot_step;\
\
    // Apply broker limits\
    double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);\
    double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);\
\
    return MathMax(min_lot, MathMin(max_lot, lot_size));\
}\
```\
\
### Deployment Checklist and Validation\
\
Pre-Deployment Verification\
\
Before running your EA in production, complete these critical validation steps:\
\
1\. Feature Parity Testing\
\
```\
# Python: Generate test vectors with known outputs\
import json\
\
# Create detailed test cases\
test_cases = []\
for i in range(10):\
    features = X_test[i]\
    prediction = clf.predict_proba([features])[0]\
\
    test_cases.append({\
        'test_id': i,\
        'features': features.tolist(),\
        'expected_prob_sell': float(prediction[0]),\
        'expected_prob_buy': float(prediction[1]),\
        'expected_class': int(clf.predict([features])[0]),\
        'tolerance': 1e-4\
    })\
\
with open('mql5_validation_tests.json', 'w') as f:\
    json.dump(test_cases, f, indent=2)\
\
print(f"Generated {len(test_cases)} test cases for MQL5 validation")\
```\
\
2\. Create MQL5 Validation Script\
\
```\
//+------------------------------------------------------------------+\
//| ValidationScript.mq5                                             |\
//| Validates ONNX model predictions against Python test cases       |\
//+------------------------------------------------------------------+\
#property script_show_inputs\
\
input string InpModelFile = "sequential_bagging_model.onnx";\
\
void OnStart() {\
    // Load model\
    long model = OnnxCreateFromFile(InpModelFile, ONNX_DEFAULT);\
    if(model == INVALID_HANDLE) {\
        Print("Failed to load model");\
        return;\
    }\
\
    // Test case 1: Manually input features from Python\
    double test_features[] = {\
        -0.523,  // bb_position\
        0.042,   // bb_width\
        -0.012,  // return_1d\
        -0.034,  // return_5d\
        0.018,   // volatility_20d\
        1.234,   // volume_ratio\
        0.425,   // rsi_14 (normalized)\
        -1.823   // mean_reversion_z\
    };\
\
    // Expected output from Python (copy from test file)\
    double expected_prob_sell = 0.234;\
    double expected_prob_buy = 0.766;\
\
    // Run prediction\
    matrix input(1, 8);\
    for(int i=0; i<8; i++) {\
        input[0][i] = (float)test_features[i];\
    }\
\
    matrix output;\
    OnnxRun(model, ONNX_NO_CONVERSION, input, output);\
\
    double mql5_prob_sell = output[0][0];\
    double mql5_prob_buy = output[0][1];\
\
    // Validate\
    double tolerance = 0.0001;\
    bool sell_match = MathAbs(mql5_prob_sell - expected_prob_sell) < tolerance;\
    bool buy_match = MathAbs(mql5_prob_buy - expected_prob_buy) < tolerance;\
\
    Print("========== VALIDATION RESULTS ==========");\
    Print("Expected SELL prob: ", expected_prob_sell);\
    Print("MQL5 SELL prob:     ", mql5_prob_sell);\
    Print("Difference:         ", MathAbs(mql5_prob_sell - expected_prob_sell));\
    Print("Match: ", sell_match ? "✓ PASS" : "✗ FAIL");\
    Print("");\
    Print("Expected BUY prob:  ", expected_prob_buy);\
    Print("MQL5 BUY prob:      ", mql5_prob_buy);\
    Print("Difference:         ", MathAbs(mql5_prob_buy - expected_prob_buy));\
    Print("Match: ", buy_match ? "✓ PASS" : "✗ FAIL");\
    Print("========================================");\
\
    if(sell_match && buy_match) {\
        Print("✓✓✓ VALIDATION PASSED ✓✓✓");\
    } else {\
        Print("✗✗✗ VALIDATION FAILED ✗✗✗");\
        Print("Check feature calculations!");\
    }\
\
    OnnxRelease(model);\
}\
```\
\
Common Deployment Issues and Solutions\
\
| Issue | Symptom | Solution |\
| --- | --- | --- |\
| Feature Misalignment | Predictions differ from Python by >1% | Use validation script. Check calculation order, lookback periods, and division-by-zero handling |\
| Model Load Failure | INVALID\_HANDLE on OnnxCreateFromFile | Verify file is in MQL5/Files/, check filename spelling, ensure opset compatibility (9-15) |\
| Wrong Input Shape | OnnxRun returns false | Verify feature count matches training. Use OnnxGetInputShape to check expected dimensions |\
| Slow Predictions | EA lags on each tick | Reduce n\_estimators, simplify trees (lower max\_depth), or run predictions only on new bars |\
| Index Array Error | ArraySetAsSeries warnings | Always call ArraySetAsSeries(array, true) before CopyClose/High/Low operations |\
\
Production Monitoring Dashboard\
\
Add this code to track model performance in real-time:\
\
```\
//--- Add to global variables section\
struct PredictionStats {\
    int total_predictions;\
    int buy_signals;\
    int sell_signals;\
    double avg_confidence;\
    double max_confidence;\
    double min_confidence;\
} g_stats;\
\
//--- Add to OnInit()\
void ResetStats() {\
    g_stats.total_predictions = 0;\
    g_stats.buy_signals = 0;\
    g_stats.sell_signals = 0;\
    g_stats.avg_confidence = 0.0;\
    g_stats.max_confidence = 0.0;\
    g_stats.min_confidence = 1.0;\
}\
\
//--- Add after model prediction in OnTick()\
void UpdateStats(double prob_sell, double prob_buy) {\
    g_stats.total_predictions++;\
\
    double max_prob = MathMax(prob_sell, prob_buy);\
\
    if(prob_buy > InpConfidenceThreshold) g_stats.buy_signals++;\
    if(prob_sell > InpConfidenceThreshold) g_stats.sell_signals++;\
\
    g_stats.avg_confidence = (g_stats.avg_confidence * (g_stats.total_predictions - 1) + max_prob) /\
                              g_stats.total_predictions;\
    g_stats.max_confidence = MathMax(g_stats.max_confidence, max_prob);\
    g_stats.min_confidence = MathMin(g_stats.min_confidence, max_prob);\
}\
\
//--- Enhanced Comment() display\
Comment(StringFormat(\
    "=== Sequential Bootstrap EA ===\n" +\
    "Time: %s\n\n" +\
    "Current Prediction:\n" +\
    "  SELL: %.2f%%  %s\n" +\
    "  BUY:  %.2f%%  %s\n\n" +\
    "Statistics (Session):\n" +\
    "  Predictions: %d\n" +\
    "  BUY signals: %d\n" +\
    "  SELL signals: %d\n" +\
    "  Avg confidence: %.2f%%\n" +\
    "  Range: %.2f%% - %.2f%%\n\n" +\
    "Positions: %d / %d",\
    TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES),\
    prob_sell * 100, prob_sell > InpConfidenceThreshold ? "[SIGNAL]" : "",\
    prob_buy * 100, prob_buy > InpConfidenceThreshold ? "[SIGNAL]" : "",\
    g_stats.total_predictions,\
    g_stats.buy_signals,\
    g_stats.sell_signals,\
    g_stats.avg_confidence * 100,\
    g_stats.min_confidence * 100,\
    g_stats.max_confidence * 100,\
    PositionsTotal(),\
    InpMaxTrades\
));\
```\
\
### Performance Optimization for Production\
\
Model Size Optimization\
\
For live trading, smaller models with comparable performance are preferable:\
\
```\
# Option 1: Train a smaller production model\
clf_prod = SequentiallyBootstrappedBaggingClassifier(\
    samples_info_sets=samples_info_sets,\
    price_bars_index=price_bars.index,\
    estimator=DecisionTreeClassifier(\
        max_depth=4,          # Reduced from 6\
        min_samples_leaf=100  # Increased from 50\
    ),\
    n_estimators=50,        # Reduced from 100\
    max_samples=0.5,\
    random_state=42\
)\
clf_prod.fit(X_train, y_train)\
\
# Compare performance\
print("Full model test accuracy:", clf.score(X_test, y_test))\
print("Production model test accuracy:", clf_prod.score(X_test, y_test))\
\
# Option 2: Feature selection to reduce input dimensionality\
from sklearn.feature_selection import SelectKBest, f_classif\
\
selector = SelectKBest(f_classif, k=6)  # Keep only 6 best features\
X_train_selected = selector.fit_transform(X_train, y_train)\
X_test_selected = selector.transform(X_test)\
\
# Train on reduced features\
clf_reduced = SequentiallyBootstrappedBaggingClassifier(\
    samples_info_sets=samples_info_sets,\
    price_bars_index=price_bars.index,\
    n_estimators=50,\
    random_state=42\
)\
clf_reduced.fit(X_train_selected, y_train)\
\
# Show which features were selected\
selected_features = selector.get_support(indices=True)\
print("Selected feature indices:", selected_features)\
print("Reduced model accuracy:", clf_reduced.score(X_test_selected, y_test))\
```\
\
**Alternative Deployment: REST API**\
\
If ONNX limitations become restrictive (e.g., need for complex preprocessing or frequent model updates), a REST API provides more flexibility:\
\
```\
# Python: Simple Flask API\
from flask import Flask, request, jsonify\
import joblib\
import numpy as np\
\
app = Flask(__name__)\
model = joblib.load('sequential_bagging_model.pkl')\
\
@app.route('/predict', methods=['POST'])\
def predict():\
    try:\
        features = np.array(request.json['features']).reshape(1, -1)\
        proba = model.predict_proba(features)[0]\
\
        return jsonify({\
            'success': True,\
            'probability_sell': float(proba[0]),\
            'probability_buy': float(proba[1]),\
            'model_version': 'v1.0'\
        })\
    except Exception as e:\
        return jsonify({'success': False, 'error': str(e)}), 400\
\
if __name__ == '__main__':\
    app.run(host='0.0.0.0', port=5000)\
```\
\
```\
//--- MQL5: HTTP client for REST API\
#include <JAson.mqh>  // Or your preferred JSON library\
\
bool PredictViaAPI(const double &features[], double &prob_sell, double &prob_buy) {\
    string url = "http://localhost:5000/predict";\
\
    // Build JSON request\
    string json_request = "{";\
    json_request += "\"features\":[";\
    for(int i=0; i<ArraySize(features); i++) {\
        json_request += DoubleToString(features[i], 6);\
        if(i < ArraySize(features)-1) json_request += ",";\
    }\
    json_request += "]}";\
\
    // Send HTTP POST request\
    char post_data[];\
    char result_data[];\
    string headers = "Content-Type: application/json\r\n";\
\
    StringToCharArray(json_request, post_data, 0, WHOLE_ARRAY, CP_UTF8);\
    int res = WebRequest("POST", url, headers, 5000, post_data, result_data, headers);\
\
    if(res == -1) {\
        Print("API request failed: ", GetLastError());\
        return false;\
    }\
\
    // Parse JSON response\
    string response = CharArrayToString(result_data, 0, WHOLE_ARRAY, CP_UTF8);\
    // Parse using your JSON library\
    // prob_sell = parsed_value;\
    // prob_buy = parsed_value;\
\
    return true;\
}\
```\
\
REST API Trade-offs:\
\
| Aspect | ONNX (Recommended) | REST API |\
| --- | --- | --- |\
| Latency | ~1ms | ~10-50ms |\
| Complexity | Low (self-contained) | Medium (requires server) |\
| Updates | Manual file replacement | Hot reload possible |\
| Preprocessing | Limited (must replicate in MQL5) | Full Python ecosystem |\
| Infrastructure | None needed | Web server + monitoring |\
\
**Best Practices Summary**\
\
Critical Success Factors:\
\
1. Feature parity is paramount – Use validation scripts to verify MQL5 features match Python exactly. Even small discrepancies compound across ensemble predictions\
2. Document everything – Save feature metadata, test cases, and model versions. Future you will thank present you\
3. Start conservative – Begin with smaller ensembles (50 estimators) and simpler trees (max\_depth=4-6) for faster iterations\
4. Test in stages – Validate → Paper trade → Small live position → Full deployment\
5. Monitor continuously – Track prediction confidence, signal frequency, and compare live performance to backtest expectations\
\
Model Update Workflow:\
\
```\
# Step 1: Train new model with updated data\
clf_v2 = SequentiallyBootstrappedBaggingClassifier(...)\
clf_v2.fit(X_train_updated, y_train_updated)\
\
# Step 2: Validate against held-out test set\
score_v2 = clf_v2.score(X_test, y_test)\
assert score_v2 >= previous_score * 0.95, "New model performs worse!"\
\
# Step 3: Export with version tag\
model_file = f"sequential_bagging_v2_{datetime.now().strftime('%Y%m%d')}.onnx"\
onnx_model = convert_sklearn(clf_v2, initial_types=initial_type)\
with open(model_file, "wb") as f:\
    f.write(onnx_model.SerializeToString())\
\
# Step 4: Run backtests comparing v1 vs v2\
# Step 5: Deploy to paper trading first\
# Step 6: Monitor for 1-2 weeks before live deployment\
# Step 7: Keep v1 as fallback\
```\
\
When to Retrain:\
\
- Regular schedule – Monthly or quarterly retraining with expanded dataset\
- Performance degradation – If live accuracy drops 10%+ below backtest expectations\
- Market regime change – Major shifts in volatility, correlations, or market structure\
- Feature additions – When adding new technical indicators or data sources\
\
**Troubleshooting Guide**\
\
Problem: Predictions are random (all near 0.5)\
\
Diagnosis:\
\
```\
# Check if features have variance\
print("Feature statistics:")\
print(pd.DataFrame(X_train).describe())\
\
# Check class balance\
print("Class distribution:", np.bincount(y_train))\
\
# Verify model actually learned something\
print("Training accuracy:", clf.score(X_train, y_train))\
print("Test accuracy:", clf.score(X_test, y_test))\
```\
\
Solutions:\
\
- Ensure features aren't all zeros or constants\
- Check for severe class imbalance (consider sample\_weight)\
- Verify model converged during training\
- Increase n\_estimators or tree depth if underfitting\
\
Problem: MQL5 predictions differ significantly from Python\
\
Systematic debugging approach:\
\
```\
# 1. Print raw feature values from both systems\
# Python:\
print("Python features:", X_test[0])\
\
# MQL5: Add to EA\
// Print all features before prediction\
string feat_str = "";\
for(int i=0; i<ArraySize(feature_array); i++) {\
    feat_str += StringFormat("[%d]:%.6f ", i, feature_array[i]);\
}\
Print("MQL5 features: ", feat_str);\
\
# 2. Check intermediate calculations\
# Add debug prints to FeatureEngine.mqh for BB, RSI, etc.\
Print("BB Upper:", bb_upper, " Middle:", bb_middle, " Lower:", bb_lower);\
Print("RSI:", rsi_value);\
\
# 3. Verify data alignment\
# Ensure MQL5 arrays are time-series ordered (most recent first)\
# Python typically uses oldest first\
```\
\
Problem: Model loads slowly or EA freezes\
\
Optimization strategies:\
\
```\
// 1. Load model once in OnInit, not on every tick\
// ✓ Correct:\
int OnInit() {\
    g_model_handle = OnnxCreateFromFile(InpModelFile, ONNX_DEFAULT);\
}\
\
// ✗ Wrong:\
void OnTick() {\
    long model = OnnxCreateFromFile(InpModelFile, ONNX_DEFAULT); // DON'T DO THIS!\
}\
\
// 2. Reduce model complexity\
# Python: Train lighter model\
clf_fast = SequentiallyBootstrappedBaggingClassifier(\
    n_estimators=30,  # Reduced from 100\
    max_depth=3      # Reduced from 6\
)\
\
// 3. Predict only on new bar, not every tick\
datetime current_bar = iTime(_Symbol, _Period, 0);\
if(current_bar == g_last_bar_time) return;\
g_last_bar_time = current_bar;\
```\
\
**Real-World Deployment Example**\
\
Here's a complete production deployment timeline that has worked successfully:\
\
| Week | Activity | Success Criteria |\
| --- | --- | --- |\
| 1 | Train model, export ONNX, validate predictions match Python | Max prediction difference < 0.01% |\
| 2 | Strategy tester backtest on 2+ years historical data | Sharpe > 1.5, Max DD < 15% |\
| 3 | Forward test on demo account with full position sizing | 20+ trades executed, no technical errors |\
| 4-5 | Live trading with 10% of target capital | Performance within 20% of backtest expectations |\
| 6-8 | Gradually scale to 50% then 100% of target capital | Consistent performance, no unexpected behavior |\
| 9+ | Full production with monthly performance reviews | Monthly retraining, quarterly model evaluation |\
\
### Conclusion: ONNX Deployment Checklist\
\
Before going live with your sequential bootstrap model in MQL5:\
\
Pre-Deployment (Python):\
\
- ☐ Model achieves acceptable out-of-sample performance\
- ☐ ONNX export successful (skl2onnx)\
- ☐ ONNX predictions verified against original model\
- ☐ Feature metadata documented (names, order, calculations)\
- ☐ Test cases created with known inputs/outputs\
- ☐ Model file versioned and backed up\
\
Implementation (MQL5):\
\
- ☐ FeatureEngine.mqh matches Python calculations exactly\
- ☐ Validation script passes all test cases\
- ☐ Model loads successfully in OnInit\
- ☐ Predictions execute without errors\
- ☐ Risk management parameters configured\
- ☐ Logging and monitoring implemented\
\
Testing:\
\
- ☐ Strategy tester backtest completed (2+ years)\
- ☐ Forward test on demo account (2+ weeks)\
- ☐ Performance metrics within acceptable ranges\
- ☐ Edge cases handled (zero volume, market gaps, etc.)\
\
Production:\
\
- ☐ Start with minimal capital (10%)\
- ☐ Daily monitoring for first 2 weeks\
- ☐ Weekly performance review\
- ☐ Model update schedule established\
- ☐ Fallback procedure documented\
\
With sequential bootstrap addressing temporal leakage in training and ONNX providing reliable deployment to MQL5, you now have a complete pipeline from research to production. The combination ensures your model's robust training translates into trustworthy live trading performance.\
\
### Attachments\
\
| File Name | Description |\
| bootstrap\_mc.py | Contains Monte Carlo simulation code to compare standard vs. sequential bootstrap effectiveness. Generates random time series data and runs experiments to measure uniqueness metrics for both methods. |\
| bootstrapping.py | Core implementation of sequential bootstrapping algorithms. Includes functions for creating indicator matrices, computing uniqueness scores, and performing optimized sequential sampling using Numba acceleration. |\
| misc.py | Collection of utility functions including data formatting, memory optimization, logging decorators, performance monitoring, time helpers, and file conversion utilities. |\
| multiprocess.py | Implements parallel processing utilities for efficient computation. Contains functions for job partitioning, progress reporting, and parallel execution across multiple CPU cores. |\
| sb\_bagging.py | Implements Sequentially Bootstrapped Bagging Classifier and Regressor - ensemble methods that integrate sequential bootstrap sampling with scikit-learn's bagging framework for financial ML. |\
\
**Attached files** \|\
\
\
[Download ZIP](https://www.mql5.com/en/articles/download/20059.zip "Download all attachments in the single ZIP archive")\
\
[bootstrap\_mc.py](https://www.mql5.com/en/articles/download/20059/bootstrap_mc.py "Download bootstrap_mc.py")(1.71 KB)\
\
[bootstrapping.py](https://www.mql5.com/en/articles/download/20059/bootstrapping.py "Download bootstrapping.py")(12.15 KB)\
\
[misc.py](https://www.mql5.com/en/articles/download/20059/misc.py "Download misc.py")(19.74 KB)\
\
[multiprocess.py](https://www.mql5.com/en/articles/download/20059/multiprocess.py "Download multiprocess.py")(9 KB)\
\
[sb\_bagging.py](https://www.mql5.com/en/articles/download/20059/sb_bagging.py "Download sb_bagging.py")(32.35 KB)\
\
**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.\
\
This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.\
\
#### Other articles by this author\
\
- [MetaTrader 5 Machine Learning Blueprint (Part 6): Engineering a Production-Grade Caching System](https://www.mql5.com/en/articles/20302)\
- [Machine Learning Blueprint (Part 4): The Hidden Flaw in Your Financial ML Pipeline — Label Concurrency](https://www.mql5.com/en/articles/19850)\
- [MetaTrader 5 Machine Learning Blueprint (Part 3): Trend-Scanning Labeling Method](https://www.mql5.com/en/articles/19253)\
- [MetaTrader 5 Machine Learning Blueprint (Part 2): Labeling Financial Data for Machine Learning](https://www.mql5.com/en/articles/18864)\
- [MetaTrader 5 Machine Learning Blueprint (Part 1): Data Leakage and Timestamp Fixes](https://www.mql5.com/en/articles/17520)\
\
**[Go to discussion](https://www.mql5.com/en/forum/499313)**\
\
![Circle Search Algorithm (CSA)](https://c.mql5.com/2/118/Circle_Search_Algorithm__LOGO.png)[Circle Search Algorithm (CSA)](https://www.mql5.com/en/articles/17143)\
\
The article presents a new metaheuristic optimization Circle Search Algorithm (CSA) based on the geometric properties of a circle. The algorithm uses the principle of moving points along tangents to find the optimal solution, combining the phases of global exploration and local exploitation.\
\
![Reimagining Classic Strategies (Part 17): Modelling Technical Indicators](https://c.mql5.com/2/178/20090-reimagining-classic-strategies-logo.png)[Reimagining Classic Strategies (Part 17): Modelling Technical Indicators](https://www.mql5.com/en/articles/20090)\
\
In this discussion, we focus on how we can break the glass ceiling imposed by classical machine learning techniques in finance. It appears that the greatest limitation to the value we can extract from statistical models does not lie in the models themselves — neither in the data nor in the complexity of the algorithms — but rather in the methodology we use to apply them. In other words, the true bottleneck may be how we employ the model, not the model’s intrinsic capability.\
\
![Automating Trading Strategies in MQL5 (Part 38): Hidden RSI Divergence Trading with Slope Angle Filters](https://c.mql5.com/2/179/20157-automating-trading-strategies-logo__1.png)[Automating Trading Strategies in MQL5 (Part 38): Hidden RSI Divergence Trading with Slope Angle Filters](https://www.mql5.com/en/articles/20157)\
\
In this article, we build an MQL5 EA that detects hidden RSI divergences via swing points with strength, bar ranges, tolerance, and slope angle filters for price and RSI lines. It executes buy/sell trades on validated signals with fixed lots, SL/TP in pips, and optional trailing stops for risk control.\
\
![Market Simulation (Part 05): Creating the C_Orders Class (II)](https://c.mql5.com/2/114/Simulat6o_de_mercado_Parte_01_Cross_Order_I_LOGO.png)[Market Simulation (Part 05): Creating the C\_Orders Class (II)](https://www.mql5.com/en/articles/12598)\
\
In this article, I will explain how Chart Trade, together with the Expert Advisor, will process a request to close all of the users' open positions. This may sound simple, but there are a few complications that you need to know how to manage.\
\
[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/20059&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049469471162215446)\
\
This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).\
\
![close](https://c.mql5.com/i/close.png)\
\
![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)\
\
You are missing trading opportunities:\
\
- Free trading apps\
- Over 8,000 signals for copying\
- Economic news for exploring financial markets\
\
RegistrationLog in\
\
latin characters without spaces\
\
a password will be sent to this email\
\
An error occurred\
\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)\
\
You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)\
\
If you do not have an account, please [register](https://www.mql5.com/en/auth_register)\
\
Allow the use of cookies to log in to the MQL5.com website.\
\
Please enable the necessary setting in your browser, otherwise you will not be able to log in.\
\
[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)