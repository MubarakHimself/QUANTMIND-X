---
title: Machine Learning Blueprint (Part 4): The Hidden Flaw in Your Financial ML Pipeline — Label Concurrency
url: https://www.mql5.com/en/articles/19850
categories: Trading, Trading Systems, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T17:52:40.755484
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/19850&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068749712498031926)

MetaTrader 5 / Trading


### Introduction

In [Part 2](https://www.mql5.com/en/articles/18864) of this series, we explored the triple-barrier labeling method for creating machine learning labels from financial time series data. We discussed how this approach addresses the path-dependent nature of returns and provides more realistic training labels for classification models. This article assumes familiarity with triple-barrier labeling and supervised ML methods in sci-kit learn.

However, implementing the triple-barrier method introduces a critical challenge that most machine learning practitioners overlook: **label concurrency**. When we apply barriers to financial data, the resulting labels often overlap in time. Multiple observations may be "active" simultaneously—their information sets span overlapping periods—creating temporal dependencies that violate the fundamental assumption of most machine learning algorithms: that training samples are Independent and Identically Distributed (IID).

This violation has serious consequences. Models trained on concurrent observations exhibit inflated in-sample performance because they learn the same patterns multiple times. Yet their out-of-sample performance deteriorates because the actual frequency of those patterns is much lower than the model believes. The result is overfit models that fail in live trading.

This article addresses this challenge through sample weighting—a principled approach to correcting for label concurrency. We will demonstrate how to:

- Quantify the degree of overlap between observations using concurrency metrics
- Calculate sample weights that reflect each observation's unique information content
- Implement these weights in scikit-learn classifiers to improve model generalization
- Evaluate performance improvements across multiple strategies using proper cross-validation techniques

### Sample Weights — Addressing Concurrency

**The Concurrency Problem**

Most non-financial ML researchers can assume that observations are drawn from IID processes (IID  — Independent and Identically Distributed). For example, you can obtain blood samples from a large number of patients and measure their cholesterol. Of course, various underlying common factors will shift the mean and standard deviation of the cholesterol distribution, but the samples are still independent: there is one observation per subject. Suppose you take those blood samples, and someone in your laboratory spills blood from each tube into the following nine tubes to their right. That is, tube 10 contains blood for patient 10, but also blood from patients 1 through 9. Tube 11 contains blood for patient 11, but also blood from patients 2 through 10, and so on. Now you need to determine the features predictive of high cholesterol (diet, exercise, age, etc.), without knowing for sure the cholesterol level of each patient. That is the equivalent challenge that we face in financial ML, with the additional handicap that the spillage pattern is non-deterministic and unknown.

Models trained on concurrent observations often show inflated in-sample performance (because they're learning the same patterns multiple times) but poor out-of-sample performance (because the real frequency of those patterns is much lower than the model believes).

Sample weighting provides an elegant solution. Instead of treating all observations equally, we assign weights based on how much unique information each observation contains. Observations that overlap heavily with others receive lower weights, while truly independent observations receive higher weights.

**Mathematical Foundation**

The mathematical foundation for sample weights comes from the concept of "average uniqueness." For each observation, we need to quantify how much of its information content is unique versus shared with other concurrent observations.

López de Prado's approach calculates this through a matrix of label overlap. For any two observations _i_ and _j_, we determine how much their respective "information sets" overlap in time. If observation _i_ uses information from time _t₁_ to _t₂_ in its label, and observation _j_ uses information from time _t₃_ to _t₄_, then their overlap is the intersection of these time intervals.

The process involves three steps:

1. Concurrency Count: For each bar in your data, count how many events are "active" at that time. If three trades are all open simultaneously, each bar during that period has a concurrency of 3.
2. Uniqueness: For each event, calculate the reciprocal of concurrency (1/concurrency) at each bar during the event's lifespan, then average these values. If an event spans bars with concurrency \[3, 4, 3, 2\], its average uniqueness is (1/3 + 1/4 + 1/3 + 1/2)/4 ≈ 0.354.
3. Sample Weight: This uniqueness value becomes the weight for that observation during model training.

The average uniqueness of observation _i_ is calculated as the mean of reciprocals of concurrency across all bars in its lifespan. An observation that doesn't overlap with any others has an average uniqueness of 1.0 (maximum weight), while an observation that completely overlaps with many others approaches 0.0 (minimum weight).

This creates a natural weighting scheme where:

- Independent observations receive full weight (1.0)
- Partially overlapping observations receive proportionally reduced weight (0.3-0.7)
- Heavily overlapping observations receive minimal weight (< 0.3)

The beauty of this approach is that it doesn't eliminate overlapping observations entirely—it just reduces their influence proportionally to their redundancy. This preserves information while correcting for the artificial amplification created by temporal overlap.

**Implementation: Computing Concurrency**

The implementation of sample weights requires careful consideration of what constitutes "concurrency" in our specific context. For the triple-barrier method, two observations are concurrent if their respective time periods (from entry to exit) overlap in any way.

The first function computes how many events are active at each point in time:

```
def num_concurrent_events(close_series_index, label_endtime, molecule):
    """
    Advances in Financial Machine Learning, Snippet 4.1, page 60.

    Estimating the Uniqueness of a Label

    This function uses close series prices and label endtime (when the first barrier is touched)
    to compute the number of concurrent events per bar.

    :param close_series_index: (pd.Series) Close prices index
    :param label_endtime: (pd.Series) Label endtime series (t1 for triple barrier events)
    :param molecule: (an array) A set of datetime index values for processing
    :return: (pd.Series) Number concurrent labels for each datetime index
    """
    # Find events that span the period [molecule[0], molecule[1]]
    label_endtime = label_endtime.fillna(
        close_series_index[-1]
    )  # Unclosed events still must impact other weights
    label_endtime = label_endtime[\
        label_endtime >= molecule[0]\
    ]  # Events that end at or after molecule[0]
    # Events that start at or before t1[molecule].max()
    label_endtime = label_endtime.loc[: label_endtime[molecule].max()]

    # Count events spanning a bar
    nearest_index = close_series_index.searchsorted(
        pd.DatetimeIndex([label_endtime.index[0], label_endtime.max()])
    )
    count = pd.Series(0, index=close_series_index[nearest_index[0] : nearest_index[1] + 1])
    for t_in, t_out in label_endtime.items():
        count.loc[t_in:t_out] += 1
    return count.loc[molecule[0] : label_endtime[molecule].max()]
```

What This Code Actually Does: Imagine you have three trades:

- Trade A: Opens at 10:00, closes at 10:30
- Trade B: Opens at 10:15, closes at 10:45
- Trade C: Opens at 10:50, closes at 11:00

At 10:20, both Trade A and Trade B are open, so count\[10:20\] = 2. At 10:55, only Trade C is open, so count\[10:55\] = 1. This function builds that entire timeline.

The wrapper function parallelizes this computation by making use of mp\_pandas\_obj, a multiprocessing utility (see multiprocess.py), across your dataset:

```
def get_num_conc_events(events, close, num_threads=4, verbose=True):
    num_conc_events = mp_pandas_obj(
        num_concurrent_events,
        ("molecule", events.index),
        num_threads,
        close_series_index=close.index,
        label_endtime=events["t1"],
        verbose=verbose,
    )
    return num_conc_events
```

**Computing Average Uniqueness**

Once we know the concurrency at each bar, we calculate the average uniqueness for each event:

```
def _get_average_uniqueness(label_endtime, num_conc_events, molecule):
    """
    Advances in Financial Machine Learning, Snippet 4.2, page 62.

    Estimating the Average Uniqueness of a Label

    This function uses close series prices and label endtime (when the first barrier is touched)
    to compute the number of concurrent events per bar.

    :param label_endtime: (pd.Series) Label endtime series (t1 for triple barrier events)
    :param num_conc_events: (pd.Series) Number of concurrent labels (output from num_concurrent_events function).
    :param molecule: (an array) A set of datetime index values for processing.
    :return: (pd.Series) Average uniqueness over event's lifespan.
    """
    wght = {}
    for t_in, t_out in label_endtime.loc[molecule].items():
        wght[t_in] = (1.0 / num_conc_events.loc[t_in:t_out]).mean()

    wght = pd.Series(wght)
    return wght
```

The orchestrator function brings everything together:

```
def get_av_uniqueness_from_triple_barrier(
    triple_barrier_events, close_series, num_threads, num_conc_events=None, verbose=True
):
    """
    This function is the orchestrator to derive average sample uniqueness from a dataset labeled by the triple barrier
    method.

    :param triple_barrier_events: (pd.DataFrame) Events from labeling.get_events()
    :param close_series: (pd.Series) Close prices.
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param num_conc_events: (pd.Series) Number concurrent labels for each datetime index
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.Series) Average uniqueness over event's lifespan for each index in triple_barrier_events
    """
    out = pd.DataFrame()

    # Create processing pipeline for num_conc_events
    def process_concurrent_events(ce):
        """Process concurrent events to ensure proper format and indexing."""
        ce = ce.loc[~ce.index.duplicated(keep="last")]
        ce = ce.reindex(close_series.index).fillna(0)
        if isinstance(ce, pd.Series):
            ce = ce.to_frame()
        return ce

    # Handle num_conc_events (whether provided or computed)
    if num_conc_events is None:
        num_conc_events = get_num_conc_events(
            triple_barrier_events, close_series, num_threads, verbose
        )
        processed_ce = process_concurrent_events(num_conc_events)
    else:
        # Ensure precomputed value matches expected format
        processed_ce = process_concurrent_events(num_conc_events.copy())

    # Verify index compatibility
    missing_in_close = processed_ce.index.difference(close_series.index)
    assert missing_in_close.empty, (
        f"num_conc_events contains {len(missing_in_close)} " "indices not in close_series"
    )

    out["tW"] = mp_pandas_obj(
        _get_average_uniqueness,
        ("molecule", triple_barrier_events.index),
        num_threads,
        label_endtime=triple_barrier_events["t1"],
        num_conc_events=processed_ce,
        verbose=verbose,
    )
    return out
```

**Return Attribution**

While average uniqueness accounts for temporal overlap, it treats all events equally regardless of their magnitude. The return attribution method combines uniqueness with the absolute returns generated during each event's lifespan:

```
def _apply_weight_by_return(label_endtime, num_conc_events, close_series, molecule):
    """
    Advances in Financial Machine Learning, Snippet 4.10, page 69.

    Determination of Sample Weight by Absolute Return Attribution

    Derives sample weights based on concurrency and return. Works on a set of
    datetime index values (molecule). This allows the program to parallelize the processing.

    :param label_endtime: (pd.Series) Label endtime series (t1 for triple barrier events)
    :param num_conc_events: (pd.Series) Number of concurrent labels (output from num_concurrent_events function).
    :param close_series: (pd.Series) Close prices
    :param molecule: (an array) A set of datetime index values for processing.
    :return: (pd.Series) Sample weights based on number return and concurrency for molecule
    """

    ret = np.log(close_series).diff()  # Log-returns, so that they are additive

    weights = {}
    for t_in, t_out in label_endtime.loc[molecule].items():
        # Weights depend on returns and label concurrency
        weights[t_in] = (ret.loc[t_in:t_out] / num_conc_events.loc[t_in:t_out]).sum()

    weights = pd.Series(weights)
    return weights.abs()
```

The full implementation with proper data handling:

```
def get_weights_by_return(
    triple_barrier_events,
    close_series,
    num_threads=4,
    num_conc_events=None,
    verbose=True,
):
    """
    Determination of Sample Weight by Absolute Return Attribution
    Modified to ensure compatibility with precomputed num_conc_events

    :param triple_barrier_events: (pd.DataFrame) Events from labeling.get_events()
    :param close_series: (pd.Series) Close prices
    :param num_threads: (int) Number of threads
    :param num_conc_events: (pd.Series) Precomputed concurrent events count
    :param verbose: (bool) Report progress
    :return: (pd.Series) Sample weights
    """
    # Validate input
    assert not triple_barrier_events.isnull().values.any(), "NaN values in events"
    assert not triple_barrier_events.index.isnull().any(), "NaN values in index"

    # Create processing pipeline for num_conc_events
    def process_concurrent_events(ce):
        """Process concurrent events to ensure proper format and indexing."""
        ce = ce.loc[~ce.index.duplicated(keep="last")]
        ce = ce.reindex(close_series.index).fillna(0)
        if isinstance(ce, pd.Series):
            ce = ce.to_frame()
        return ce

    # Handle num_conc_events (whether provided or computed)
    if num_conc_events is None:
        num_conc_events = mp_pandas_obj(
            num_concurrent_events,
            ("molecule", triple_barrier_events.index),
            num_threads,
            close_series_index=close_series.index,
            label_endtime=triple_barrier_events["t1"],
            verbose=verbose,
        )
        processed_ce = process_concurrent_events(num_conc_events)
    else:
        # Ensure precomputed value matches expected format
        processed_ce = process_concurrent_events(num_conc_events.copy())

        # Verify index compatibility
        missing_in_close = processed_ce.index.difference(close_series.index)
        assert missing_in_close.empty, (
            f"num_conc_events contains {len(missing_in_close)} " "indices not in close_series"
        )

    # Compute weights using processed concurrent events
    weights = mp_pandas_obj(
        _apply_weight_by_return,
        ("molecule", triple_barrier_events.index),
        num_threads,
        label_endtime=triple_barrier_events["t1"],
        num_conc_events=processed_ce,  # Use processed version
        close_series=close_series,
        verbose=verbose,
    )

    # Normalize weights
    weights *= weights.shape[0] / weights.sum()
    return weights
```

**Time-Decay Weighting**

Markets are adaptive systems, and as they evolve older examples become less relevant than newer ones. As such, we would like the sample weights computed above to be multiplied by time decay factors, giving more weight to recent observations. Note that time is not meant to be chronological. In this implementation, decay takes place according to cumulative uniqueness because a chronological decay would reduce weights too fast in the presence of redundant observations.

```
def get_weights_by_time_decay(
    triple_barrier_events,
    close_series,
    num_threads=4,
    last_weight=1,
    linear=True,
    av_uniqueness=None,
    verbose=True,
):
    """
    Advances in Financial Machine Learning, Snippet 4.11, page 70.
    Implementation of Time Decay Factors
    """
    assert (
        bool(triple_barrier_events.isnull().values.any()) is False
        and bool(triple_barrier_events.index.isnull().any()) is False
    ), "NaN values in triple_barrier_events, delete nans"

    # Get average uniqueness if not provided
    if av_uniqueness is None:
        av_uniqueness = get_av_uniqueness_from_triple_barrier(
            triple_barrier_events, close_series, num_threads, verbose=verbose
        )
    elif isinstance(av_uniqueness, pd.Series):
        av_uniqueness = av_uniqueness.to_frame()

    # Calculate cumulative time weights
    cum_time_weights = av_uniqueness["tW"].sort_index().cumsum()

    if linear:
        # Apply linear decay (your existing linear code is correct)
        if last_weight >= 0:
            slope = (1 - last_weight) / cum_time_weights.iloc[-1]
        else:
            slope = 1 / ((last_weight + 1) * cum_time_weights.iloc[-1])
        const = 1 - slope * cum_time_weights.iloc[-1]
        weights = const + slope * cum_time_weights
        weights[weights < 0] = 0
        return weights
    else:
        # Apply exponential decay
        if last_weight == 1:
            return pd.Series(1.0, index=cum_time_weights.index)

        elif cum_time_weights.iloc[-1] == 0:
            return pd.Series(1.0, index=cum_time_weights.index)

        # Calculate normalized position (0 = newest, 1 = oldest)
        elif last_weight > 0:
            # For last_weight > 0, use standard exponential decay
            normalized_position = (cum_time_weights - cum_time_weights.iloc[0]) / (
                cum_time_weights.iloc[-1] - cum_time_weights.iloc[0]
            )
            weights = last_weight**normalized_position
        elif last_weight < 0:
            # For last_weight < 0, implement cutoff (similar to linear case)
            # This is more complex for exponential - you might want to reconsider this case
            cutoff_threshold = abs(last_weight)
            normalized_position = (cum_time_weights - cum_time_weights.iloc[0]) / (
                cum_time_weights.iloc[-1] - cum_time_weights.iloc[0]
            )
            weights = (1 - cutoff_threshold)**normalized_position
            weights[weights < 0] = 0

        return weights
```

![Time-Decay Factors](https://c.mql5.com/2/175/time_decay_factors__1.png)

Figure 1. Time-Decay Factors (Linear vs. Exponential)

**Class Weights**

In addition to sample weights, it is often useful to apply class weights. Class weights are weights that correct for underrepresented labels. This is particularly critical in classification problems where the most important classes have rare occurrences (King and Zeng \[2001\]). For example, suppose that you wish to predict liquidity crisis, like the flash crash of May 6, 2010. These events are rare relative to the millions of observations that take place in between them. Unless we assign higher weights to the samples associated with those rare labels, the ML algorithm will maximize the accuracy of the most common labels, and flash crashes will be deemed to be outliers rather than rare events.

ML libraries typically implement functionality to handle class weights. For example, sklearn penalizes errors in samples of class\[ _j_\], _j=1,…,J_, with weighting class\_weight\[ _j_\] rather than 1. Accordingly, higher class weights on label _j_ will force the algorithm to achieve higher accuracy on _j_. When class weights do not add up to _J_, the effect is equivalent to changing the regularization parameter of the classifier.

In financial applications, the standard labels of a classification algorithm are {−1, 1}, where the zero (or neutral) case will be implied by a prediction with probability only slightly above 0.5 and below some neutral threshold. There is no reason for favoring accuracy of one class over the other, and as such a good default is to assign class\_weight='balanced'. This choice re-weights observations to simulate that all classes appeared with equal frequency. In the context of bagging classifiers, you may want to consider the argument class\_weight='balanced\_subsample', which means that class\_weight='balanced' will be applied to the in-bag bootstrapped samples, rather than to the entire dataset. For full details, it is helpful to read the [source code implementing class\_weight in sklearn](https://www.mql5.com/go?link=https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/class_weight.py "https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/class_weight.py").

(López de Prado, 2018, p. 71)

### Practical Implementation

**Handling Non-IID Data in Bagging**

The violation of the IID assumption in financial data renders standard bagging ineffective, as it creates bootstrap samples plagued by serial correlation. _Advances in Financial Machine Learning_ proposes three distinct methods to overcome this fundamental challenge, with sample weighting serving as the foundation for all three approaches.

Method 1: Constraining Bootstrap Sample Size

This is one of the methods we utilize in this article. It is a pragmatic and computationally efficient approach that directly addresses the symptom of oversampling.

- Core Idea: Radically reduce the size of each bootstrap sample. By drawing a smaller number of observations, we statistically decrease the probability of including multiple, highly correlated data points in the same sample.
- Implementation: In sklearn.ensemble.BaggingClassifier, this is achieved by setting the max\_samples parameter to a value significantly less than 1.0 (e.g., 0.5, 0.3, or lower). A practical heuristic is to set it to the dataset's average uniqueness: max\_samples=out\['tW'\].mean().
- Mechanism: max\_samples controls the absolute or relative number of samples to draw from X to train each base estimator. Setting max\_samples=0.3 means each classifier is trained on a random 30% of the original dataset, forcing diversity by limiting overlap.
- Pros & Cons:
  - Pro: Simple to implement, requires only one line of code change.
  - Con: A blunt instrument; it reduces redundancy but does not actively seek out unique observations. It may also discard valuable data.

Method 2: Sample Weighting for In-Bag Estimation

This method corrects for the problem at the level of the individual base estimator, rather than during the sampling phase.

- Core Idea: Use standard bootstrap sampling, but when training each base estimator, use the sample weights (the tW column) to force the model to focus on unique observations and discount redundant ones.
- Implementation: After creating a bootstrap sample via standard methods, the sample\_weight parameter for each base estimator is set to the pre-computed weights of the in-bag observations. This requires the base estimator to support sample weights (e.g., sklearn's DecisionTreeClassifier).
- Mechanism: The model's loss function is modified to penalize errors on high-weight (unique) observations more severely than errors on low-weight (redundant) ones.
- Pros & Cons:
  - Pro: Leverages the sample weights we have already computed; can be combined with other methods.
  - Con: Does not prevent correlated data from entering the bootstrap sample; it only mitigates their influence after the fact.

Method 3: Sequential Bootstrapping

This is the rigorous, targeted solution prescribed by López de Prado. It will be covered in detail in our next article.

- Core Idea: Completely replace the standard random sampling with an intelligent, sequential algorithm that actively enforces uniqueness within each bootstrap sample.
- Implementation: A custom sampling routine that draws observations one at a time. A new observation is only added to the current sample if it is sufficiently "unique" (non-overlapping) relative to all observations already in the sample. This requires a full custom implementation of the resampling logic.
- Mechanism: It directly uses the concurrency matrix or label overlap to conditionally accept or reject each candidate observation for the bootstrap sample.
- Pros & Cons:
  - Pro: Theoretically sound; actively constructs maximally diverse and de-correlated samples.
  - Con: Computationally intensive and complex to implement correctly.

**Synthesis and Our Approach**

These three methods form a hierarchy of sophistication:

- Method 1 (Sample Size Constraint) acts as a simple, preventative measure at the sampling stage.
- Method 2 (In-Bag Weighting) acts as a corrective measure during model training.
- Method 3 (Sequential Bootstrapping) is the comprehensive, preventative solution that addresses the root cause at the sampling stage.

Methods 1 and 2 are complementary and can be combined for a defense-in-depth approach: Method 1 reduces the probability of drawing overlapping observations into each bootstrap sample, while Method 2 ensures that any overlapping observations that do make it through receive appropriately reduced influence during training.

For the purposes of this article, we employ both Method 1 and Method 2 due to their complementary nature, simplicity, and demonstrable effectiveness. By setting max\_samples=out\['tW'\].mean() (Method 1) and passing sample\_weight=out\['tW'\] to the classifier's fit() method (Method 2), we create a robust pipeline that both prevents and corrects for redundancy in our bootstrap samples.

The more sophisticated Method 3, Sequential Bootstrapping, will be the focus of a dedicated follow-up article, where we will build the custom sampling class required for its implementation.

**Using Sample Weights in Model Training**

Now for the critical part: how do we actually use these weights in our machine learning pipeline? Let us take a small detour into the topic of cross-validation and why the standard applications fail in finance. This is necessary as it is the technique used to evaluate the effectiveness of our weighting methods.

**Conceptual Foundation for Financial Cross-Validation**

Standard k-fold cross-validation (CV) relies on the assumption that data points are Independent and Identically Distributed (IID). Financial time-series data violates this core assumption due to serial correlation, temporal dependencies, and structural breaks. Using standard methods risks data leakage, where information from the future inadvertently influences the training of a model on past data, leading to overfitting and unreliable performance estimates.

Figure 2 illustrates the _k_ train/test splits carried out by a k-fold CV, where _k_ = 5\. In this scheme:

1. The dataset is partitioned into _k_ subsets.
2. For _i = 1,…,k_

![K-Fold Cross-Validation](https://c.mql5.com/2/175/k_fold_cv.png)

Figure 2. Train/test splits in a 5-fold CV scheme

> - (a) The ML algorithm is trained on all subsets excluding _i_.
> - (b) The fitted ML algorithm is tested on _i_.

To address this, López de Prado introduces two key modifications to standard k-fold CV:

- Purging: This method purges from the training set any observations whose labels overlap in time with those in the test set. This prevents the model from having knowledge of future periods it is supposed to predict.
- Embargo: An additional safety measure that further removes a small fraction of data immediately following the test period from the training set, guarding against leakage through serial correlation.

![Purging overlap in the training set](https://c.mql5.com/2/175/purging.png)

Figure 3. Purging overlap in the training set

![Embargo of post-test train observations](https://c.mql5.com/2/175/embargo.png)

Figure 4. Embargo of post-test train observations

We need to purge and embargo overlapping training observations whenever we produce a train/test split, whether it is for hyper-parameter fitting, backtesting, or performance evaluation. The code below extends scikit-learn's KFold class to account for the possibility of leakages of testing information into the training set:

```
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import _BaseKFold

from ..cross_validation.scoring import probability_weighted_accuracy

class PurgedKFold(_BaseKFold):
    """
    Extend KFold class to work with labels that span intervals

    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between

    :param n_splits: (int) The number of splits. Default to 3
    :param t1: (pd.Series) The information range on which each record is constructed from
        *t1.index*: Time when the information extraction started.
        *t1.value*: Time when the information extraction ended.
    :param pct_embargo: (float) Percent that determines the embargo size.
    """

    def __init__(self, n_splits=3, t1=None, pct_embargo=0.0):
        if not isinstance(t1, pd.Series):
            raise ValueError("Label Through Dates must be a pd.Series")

        super().__init__(n_splits, shuffle=False, random_state=None)

        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        """
        The main method to call for the PurgedKFold class

        :param X: (pd.DataFrame) Samples dataset that is to be split
        :param y: (pd.Series) Sample labels series
        :param groups: (array-like), with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        :return: (tuple) [train list of sample indices, and test list of sample indices]
        """

        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and ThruDateValues must have the same index")

        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pct_embargo)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(len(X)), self.n_splits)]

        for i, j in test_starts:
            t0 = self.t1.index[i]  # start of test set
            test_indices = indices[i:j]
            max_t1_idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            if max_t1_idx < X.shape[0]:  # right train (with embargo)
                train_indices = np.concatenate((train_indices, indices[max_t1_idx + mbrg :]))
            yield train_indices, test_indices
```

### Evaluation Methodology

**Scoring Methods**

In financial machine learning, choosing the right evaluation metrics is crucial for assessing model performance. Standard metrics like accuracy can be misleading in financial contexts, particularly when dealing with imbalanced datasets or meta-labeling applications. Let's examine the key metrics used to evaluate financial ML models.

Accuracy

Accuracy measures the overall correctness of predictions by calculating the fraction of correctly classified observations:

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Where:

- TP = True Positives
- TN = True Negatives
- FP = False Positives
- FN = False Negatives

While accuracy provides a general overview of performance, it can be deceptive in financial applications where class distributions are often imbalanced.

Precision

Precision quantifies the reliability of positive predictions by measuring what fraction of predicted positives are actually correct:

```
Precision = TP / (TP + FP)
```

High precision indicates that when the model predicts a positive outcome, it's likely to be correct—a valuable characteristic in trading systems where false signals can be costly.

Recall

Recall (or sensitivity) measures how well the model identifies actual positive cases:

```
Recall = TP / (TP + FN)
```

High recall means the model captures most of the available opportunities, which is important when missing a profitable trade is more costly than occasionally taking a suboptimal position.

F1 Score

The F1 score addresses limitations of accuracy in imbalanced scenarios by combining precision and recall into a single metric:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

This metric is particularly valuable in meta-labeling applications where negative cases (label '0') often significantly outnumber positive cases (label '1'). In such situations, a naive classifier that always predicts the majority class would achieve high accuracy while failing to identify any true opportunities.

Important Consideration: The F1 score becomes undefined in certain degenerate cases:

- When all observed values are negative (no positives to recall)
- When all predicted values are negative (no positive predictions to evaluate precision)

Scikit-learn handles these edge cases by returning an F1 score of 0 and issuing an UndefinedMetricWarning.

**Understanding Degenerate Cases in Binary Classification**

The table below summarizes how different metrics behave in extreme scenarios:

| Condition | Collapse | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| All observed 1s | TN=FP=0 | =recall | 1 | \[0,1\] | \[0,1\] |
| All observed 0s | TP=FN=0 | \[0,1\] | 0 | Undefined | Undefined |
| All predicted 1s | TN=FN=0 | =precision | \[0,1\] | 1 | \[0,1\] |
| All predicted 0s | TP=FP=0 | \[0,1\] | Undefined | 0 | Undefined |

These edge cases highlight why relying solely on accuracy can be misleading and why the F1 score and log-loss provide more robust evaluation in practical financial applications.

Log-Loss

Log-loss (or cross-entropy loss) provides a more nuanced evaluation than accuracy by considering prediction confidence:

![Log-Loss Formula](https://c.mql5.com/2/177/log_loss_formula_transparent_.png)

where:

- _pn,k_ = probability for prediction _n_ of class _k_
- _Y_ = 1-of- _K_ binary indicator matrix
- _yn,k_ = 1 if observation _n_ has label _k_, 0 otherwise

In financial applications, we typically use the negative log-loss to maintain intuitive scoring (higher values are better). This metric is particularly relevant because:

1. It accounts for prediction confidence: A wrong prediction with high confidence is penalized more severely than one with low confidence
2. It aligns with PnL considerations: When combined with sample weights based on returns, it approximates the classifier's impact on profit and loss
3. It reflects position sizing: Higher confidence predictions typically translate to larger position sizes in trading strategies

Unlike accuracy, which treats all errors equally regardless of confidence, log-loss provides a more realistic assessment of a classifier's potential impact on trading performance.

Suppose that a classifier predicts two 1s, where the true labels are 1 and 0. The first prediction is a hit and the second prediction is a miss, thus accuracy is 50%. Figure 5 plots the cross-entropy loss when these predictions come from probabilities ranging \[0.5, 0.9\]. One can observe that on the right side of the figure, log loss is large due to misses with high probability, even though the accuracy is 50% in all cases.

![Log loss as a function of predicted probabilities of hit and miss](https://c.mql5.com/2/175/log_loss.png)

Figure 5. Log loss as a function of predicted probabilities of hit and miss

Probability Weighted Accuracy (PWA)

This extends traditional accuracy by weighting correct predictions by the confidence level. A correct prediction with 90% confidence contributes more than one with 51% confidence. This better reflects real trading where we'd size positions based on prediction confidence. PWA punishes bad predictions made with high confidence more severely than accuracy, but less severely than log-loss.

![Probability Weighted Accuracy Formula](https://c.mql5.com/2/177/pwa_formula_transparent.png)

where _pn_ = max{ _pn,k_} and _yn_ is an indicator function, _y_ _n_ ∈ {0, 1}, where _yn_ = 1 when the prediction was correct, and _yn_ = 0 otherwise.

This is equivalent to standard accuracy when the classifier has absolute conviction in every prediction ( _p_ _n_ = 1 for all _n_) (Prado, 2020, p.83). The baseline adjustment _pn \- 1/K_ ensures that random guessing (probability = 1/ _K_) receives zero weight.

```
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import unique_labels

def probability_weighted_accuracy(y_true, y_prob, sample_weight=None, labels=None, eps=1e-15):
    """
    Calculates the Probability-Weighted Accuracy (PWA) score.

    PWA is a confidence-weighted accuracy that penalizes high-confidence
    mistakes more severely. This version is compatible with sklearn
    conventions: it accepts a `labels` argument to fix the class order,
    applies probability clipping, and supports sample weights.

    Args:
        y_true (array-like): True class labels, shape (n_samples,).
        y_prob (array-like or DataFrame): Predicted probabilities,
            shape (n_samples, n_classes). If DataFrame, columns must be
            class labels.
        sample_weight (array-like, optional): Per-sample weights.
        labels (array-like, optional): List of all expected class labels
            (in the order corresponding to columns of y_prob).
        eps (float): Small value to clip probabilities into [eps, 1 - eps].

    Returns:
        float: PWA score between 0 and 1.
    """
    # 1) Convert inputs to numpy arrays (or reorder DataFrame)
    y_true = np.asarray(y_true)
    if isinstance(y_prob, pd.DataFrame):
        # If labels given, reorder columns; otherwise infer column order
        cols = labels if labels is not None else y_prob.columns.tolist()
        y_prob = y_prob[cols].to_numpy()
    else:
        y_prob = np.asarray(y_prob)

    # 2) Clip probabilities to avoid zeros or ones
    y_prob = np.clip(y_prob, eps, 1 - eps)

    # 3) Determine class list and validate
    if labels is not None:
        classes = np.asarray(labels)
    else:
        # Infer classes from y_true (sorted)
        classes = unique_labels(y_true)
    n_classes = classes.shape[0]

    # 4) Handle binary case where y_prob might be 1D
    if y_prob.ndim == 1:
        # Interpret as probability of class classes[1]
        y_prob = np.vstack([1 - y_prob, y_prob]).T
        n_classes = 2

    # 5) Shape checks
    if y_prob.ndim != 2 or y_prob.shape[1] != n_classes:
        raise ValueError(
            f"y_prob must be shape (n_samples, n_classes={n_classes}), " f"but got {y_prob.shape}"
        )

    if not np.all(np.isin(y_true, classes)):
        missing = set(y_true) - set(classes)
        raise ValueError(f"y_true contains labels not in `labels`: {missing}")

    # 6) Prepare sample weights
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape[0] != y_true.shape[0]:
            raise ValueError("sample_weight must have same length as y_true")

    # 7) Predicted class index and its probability
    pred_idx = np.argmax(y_prob, axis=1)
    p_n = y_prob[np.arange(len(y_true)), pred_idx]

    # 8) Correctness indicator y_n ∈ {0,1}
    #    Map y_true labels to indices in `classes`
    label_to_index = {c: i for i, c in enumerate(classes)}
    true_idx = np.vectorize(label_to_index.get)(y_true)
    y_n = (pred_idx == true_idx).astype(int)

    # 9) Confidence weights: p_n – (1/K)
    baseline = 1.0 / n_classes
    conf_w = p_n - baseline

    # 10) Compute numerator and denominator with sample weights
    numerator = np.sum(sample_weight * y_n * conf_w)
    denominator = np.sum(sample_weight * conf_w)

    # 11) Edge case: no confidence (all p_n == 1/K)
    if np.isclose(denominator, 0.0):
        return 0.5  # random-guess baseline

    # 12) Final PWA score
    return numerator / denominator
```

### Experimental Setup

**Data and Trading Strategies**

We evaluate sample weighting techniques on EUR/USD M5 bars spanning 2018-01-01 to 2022-12-31. Two distinct meta-labeling strategies were tested using a volatility target of the 20-day exponentially weighted moving standard deviation:

**Meta-Labeled Bollinger Bands Strategy**

This strategy uses Bollinger Bands to generate primary trading signals, which are then filtered through a meta-labeling model. The primary model generates signals based on price interactions with the upper and lower bands, while the meta-model predicts the probability that acting on each signal will be profitable.

**Triple-Barrier Configuration:**

- Profit Target: 1
- Stop Loss: 2
- Time Barrier: 4 hours
- Minimum Return Threshold: 0.0

**Meta-Labeled MA\_20\_50 Crossover Strategy**

This classic trend-following approach uses the crossover of 20- and 50-period moving averages as primary signals. The meta-labeling model learns to filter these signals by predicting which crossovers are likely to generate profitable trades.

**Triple-Barrier Configuration:**

- Profit Target: 0
- Stop Loss: 2
- Time Barrier: 1 days
- Minimum Return Threshold: 0.0

**Evaluation Framework**

For each strategy, we trained a Random Forest classifier with and without sample weighting, using Purged K-Fold cross-validation to prevent data leakage.

The function below calculates all the performance metrics discussed above:

```
def ml_cross_val_scores_all(
    classifier: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
    cv_gen: BaseCrossValidator,
    sample_weight_train: np.ndarray = None,
    sample_weight_score: np.ndarray = None,
):
    # pylint: disable=invalid-name
    # pylint: disable=comparison-with-callable
    """
    Advances in Financial Machine Learning, Snippet 7.4, page 110.

    Using the PurgedKFold Class.

    Function to run a cross-validation evaluation of the classifier using sample weights and a custom CV generator.
    Scores are computed using accuracy_score, probability_weighted_accuracy, log_loss and f1_score.

    Note: This function is different to the book in that it requires the user to pass through a CV object. The book
    will accept a None value as a default and then resort to using PurgedCV, this also meant that extra arguments had to
    be passed to the function. To correct this we have removed the default and require the user to pass a CV object to
    the function.

    Example:

    .. code-block:: python

        cv_gen = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)
        scores_array = ml_cross_val_scores_all(classifier, X, y, cv_gen, sample_weight_train=sample_train,
                                               sample_weight_score=sample_score, scoring=accuracy_score)

    :param classifier: (BaseEstimator) A scikit-learn Classifier object instance.
    :param X: (pd.DataFrame) The dataset of records to evaluate.
    :param y: (pd.Series) The labels corresponding to the X dataset.
    :param cv_gen: (BaseCrossValidator) Cross Validation generator object instance.
    :param sample_weight_train: (np.array) Sample weights used to train the model for each record in the dataset.
    :param sample_weight_score: (np.array) Sample weights used to evaluate the model quality.
    :return: (dict) The computed scores.
    """
    scoring_methods = [accuracy_score, probability_weighted_accuracy, log_loss, f1_score]
    ret_scores = {
        scoring.__name__ if scoring != log_loss else "neg_log_loss": []
        for scoring in scoring_methods
    }

    # If no sample_weight then broadcast a value of 1 to all samples (full weight).
    if sample_weight_train is None:
        sample_weight_train = np.ones((X.shape[0],))

    if sample_weight_score is None:
        sample_weight_score = np.ones((X.shape[0],))

    # Score model on KFolds
    for train, test in cv_gen.split(X=X, y=y):
        fit = classifier.fit(
            X=X.iloc[train, :],
            y=y.iloc[train],
            sample_weight=sample_weight_train[train],
        )
        prob = fit.predict_proba(X.iloc[test, :])
        pred = fit.predict(X.iloc[test, :])
        for method, scoring in zip(ret_scores.keys(), scoring_methods):
            if scoring in (accuracy_score, f1_score):
                score = scoring(y.iloc[test], pred, sample_weight=sample_weight_score[test])
            else:
                score = scoring(
                    y.iloc[test],
                    prob,
                    sample_weight=sample_weight_score[test],
                    labels=classifier.classes_,
                )
                if method == "neg_log_loss":
                    score *= -1
            ret_scores[method].append(score)

    for k, v in ret_scores.items():
        ret_scores[k] = np.array(v)

    return ret_scores
```

### Experimental Results

**Strategy Performance Comparison**

We evaluated two distinct strategies with and without sample weighting using 10-fold CV:

- Meta-Labeled Bollinger Bands: Traditional mean-reversion strategy
- Meta-Labeled MA\_20\_50 Crossover: Classic trend-following approach

**Bollinger Bands Strategy Performance**

| Metric | Unweighted | Uniqueness Weighting | Return Weighting |
| --- | --- | --- | --- |
| Accuracy | 0.564 ± 0.044 | 0.584 ± 0.040 | 0.693 ± 0.020 |
| PWA | 0.563 ± 0.054 | 0.593 ± 0.044 | 0.697 ± 0.019 |
| Negative Log-Loss | -0.688 ± 0.008 | -0.682 ± 0.007 | -0.631 ± 0.023 |
| Precision | 0.650 ± 0.019 | 0.658 ± 0.024 | 0.000 ± 0.000 |
| Recall | 0.616 ± 0.167 | 0.683 ± 0.145 | 0.000 ± 0.000 |
| F1 Score | 0.622 ± 0.091 | 0.663 ± 0.073 | 0.000 ± 0.000 |

**MA 20-50 Crossover Strategy Performance**

| Metric | Unweighted | Uniqueness Weighting | Return Weighting |
| --- | --- | --- | --- |
| Accuracy | 0.589 ± 0.073 | 0.634 ± 0.068 | 0.473 ± 0.011 |
| PWA | 0.672 ± 0.101 | 0.740 ± 0.080 | 0.473 ± 0.011 |
| Negative Log-Loss | -0.650 ± 0.037 | -0.625 ± 0.036 | -0.826 ± 0.018 |
| Precision | 0.298 ± 0.026 | 0.296 ± 0.029 | 0.473 ± 0.011 |
| Recall | 0.588 ± 0.125 | 0.530 ± 0.108 | 1.000 ± 0.000 |
| F1 Score | 0.388 ± 0.015 | 0.372 ± 0.018 | 0.642 ± 0.010 |

**Key Insights from the Results**

The experimental results reveal a nuanced and strategy-dependent picture of how sample weighting impacts model performance. The effectiveness of each weighting method varies significantly based on the underlying trading logic.

**Uniqueness Weighting: A Robust Default for Meta-Labeling**

The uniqueness weighting method demonstrated consistent and meaningful improvements across both strategies, establishing itself as a robust default choice.

- Bollinger Bands Strategy: Uniqueness weighting delivered a well-rounded performance boost. Accuracy improved from 56.4% to 58.4%, but more importantly, the F1 score saw a substantial 6.7% increase (from 0.622 to 0.663). This indicates a superior balance between precision and recall, meaning the model became better at filtering out false signals while capturing true opportunities. The improvement in probability-weighted accuracy further confirms that the model's confidence became better calibrated on unique, non-redundant examples.
- MA Crossover Strategy: The benefits were even more pronounced. Uniqueness weighting led to a 7.5% increase in accuracy (58.9% to 63.4%) and a remarkable 10.2% boost in probability-weighted accuracy (67.2% to 74.0%). While the F1 score saw a slight dip, the dramatic gains in confidence-weighted metrics are critical for a meta-labeling model, where the goal is to size positions based on the probability of a primary signal's success.

**Return Attribution Weighting: A Cautionary Tale**

In stark contrast to the success of uniqueness weighting, the return attribution method produced extreme and undesirable outcomes, highlighting a critical pitfall.

- Bollinger Bands Strategy: The model collapsed into a trivial classifier. Precision, recall, and F1 score all dropped to zero, while accuracy paradoxically jumped to 69.3%. This pattern is a classic sign of a model that has learned to always predict the majority class (likely \`0\`, or "do not take the trade"). It overfitted to the magnitude of past returns, completely losing its predictive power for the classification task.
- MA Crossover Strategy: A similar, though slightly different, failure mode occurred. The model achieved a perfect recall of 1.0 and a 64.2% F1 score, but with an accuracy of only 47.3%. This suggests the model learned to predict the positive class (\`1\`, or "take the trade") almost indiscriminately, capturing all true positives but also generating a massive number of false positives. This behavior is untenable for a live trading system.

The failure of return attribution underscores that concurrency and return magnitude are distinct concepts. Weighting by returns alone corrupts the learning signal, causing the model to chase past profits rather than learn generalizable patterns from unique informational events. Return magnitude  shows its value when we are predicting directionality (labels {1, -1}). Don't forget that meta-labeling aims to improve our primary model by filtering out incorrect predictions and providing an independent bet-sizing model, which makes it the wrong place to apply return-attribution. Run your own tests and see what happens.

**The Pervasiveness of the Concurrency Problem**

The significant performance gains from uniqueness weighting on both a mean-reversion (Bollinger Bands) and a trend-following (MA Crossover) strategy provide strong evidence that label concurrency is a universal challenge in financial ML. It is not an edge case but a fundamental data leakage problem that biases models regardless of the core strategy's logic. Addressing it is not optional for robust model development.

### Conclusion

This article has tackled one of the most insidious problems in financial machine learning: the violation of the IID assumption due to label concurrency. Our experimental results deliver a clear and actionable verdict: sample weighting based on temporal uniqueness is a powerful and necessary technique for building robust meta-labeling classifiers, while weighting by return attribution is a dangerous distraction.

The uniqueness weighting method consistently improved model performance by ensuring that each observation's influence during training was proportional to its unique information content. For the Bollinger Bands strategy, it enhanced the precision-recall balance (F1 score). For the MA Crossover strategy, it significantly boosted both standard and probability-weighted accuracy. In both cases, it steered the model away from learning spurious patterns from temporally redundant data.

Conversely, the dramatic failure of return attribution weighting serves as a critical warning. It demonstrates that conflating an observation's informational uniqueness with its financial return leads to pathological model behavior, either causing total predictive collapse or encouraging reckless over-trading.

For the practitioner, these findings translate into a clear mandate:

1. Always Account for Concurrency: The IID assumption is fundamentally flawed for financial time series. Ignoring label concurrency will lead to overfit models and live trading losses.
2. Implement Uniqueness Weighting: The method outlined here, which calculates the average uniqueness of each triple-barrier label, is a tractable and highly effective solution. It should be a standard component of any financial ML pipeline.
3. Avoid Naive Return Weighting: Carefully evaluate whether return-attributed sample weights are appropriate for your models. While returns are crucial for evaluating strategy performance, they can be a misleading proxy for an observation's value during training depending on what you are targeting.
4. Validate with the Right Metrics: As shown, accuracy alone can be deceptive. A combination of log-loss, F1 score, and probability-weighted accuracy is essential to properly diagnose a model's performance and calibration.

By adopting sample weighting based on temporal uniqueness, we move from training models on a distorted, redundant view of the market to training them on a dataset that reflects the true frequency and independence of informational events. This is a foundational step toward developing machine learning models that generalize beyond the backtest and succeed in the adaptive, non-IID reality of financial markets.

In the next article of this series, we will take this a step further by exploring Sequential Bootstrapping—a more advanced sampling technique that actively prevents overlapping observations from appearing in the same training set, thereby addressing the root of the concurrency problem during the data resampling stage itself.

### Attachments

| File Name | Description |
| --- | --- |
| bollinger\_features.py | Creates Bollinger Band-based features for meta-labeling models including volatility features, technical indicators, and moving average features. Also contains visualization functions for plotting Bollinger Bands with trading signals. |
| filters.py | Implements event filtering methods including the Symmetric CUSUM Filter and Z-Score Filter to identify significant market events for triple-barrier labeling. |
| fractals.py | Provides comprehensive fractal analysis tools for identifying market structure points, trend validation, and whipsaw filtering based on Williams Bill's trading concepts. |
| ma\_crossover\_feature\_engine.py | Specialized feature engineering for forex MA crossover strategies, including currency strength analysis, risk environment features, and market microstructure patterns. |
| misc.py | Contains utility functions for data optimization, formatting, logging, performance monitoring, and time conversion used throughout the ML pipeline. |
| moving\_averages.py | Computes moving average differences and crossover signals for feature generation, with optional correlation-based feature selection. |
| multiprocess.py | Provides parallel processing utilities for efficient computation across multiple CPU cores, implementing the multiprocessing patterns from AFML. |
| returns.py | Calculates various return-based features including lagged returns, rolling autocorrelations, and return distribution statistics. |
| signal\_processing.py | Converts raw strategy signals into continuous trading positions and entry timestamps, handling CUSUM filtering and signal persistence. |
| strategies.py | Defines base and concrete trading strategy classes including Bollinger Bands and Moving Average Crossover strategies for signal generation. |
| time.py | Generates time-based features including cyclical encoding, trading session flags, and forex market timing patterns for 24-hour markets. |
| trend\_scanning.py | Implements trend-scanning labeling methodology that fits OLS regressions over multiple windows to identify significant trends for classification. |
| triple\_barrier.py | Core implementation of the triple-barrier labeling method with Numba optimization for performance, including vertical barriers and meta-labeling support. |
| volatility.py | Provides various volatility estimators including daily volatility, Parkinson, Garman-Klass, and Yang-Zhang estimators for risk assessment. |
| attribution.py | Implements sample weighting methods including return attribution and time decay factors to address label concurrency issues in financial ML, using parallel processing for efficient computation. |
| concurrent.py | Handles concurrent label analysis for triple-barrier events, including counting concurrent events and calculating average label uniqueness to address overlapping labeling periods. |
| optimized\_attribution.py | Numba-optimized version of return and time decay attribution with 5-10x performance improvements, using JIT compilation and vectorized operations for faster sample weight calculations. |
| optimized\_concurrent.py | Numba-optimized version of concurrent event analysis with 5-10x performance improvements, using parallel processing and efficient memory access patterns for faster uniqueness calculations. |

### References and Further Reading

**Primary Source:**

López de Prado, M. (2018): _Advances in Financial Machine Learning_. Wiley.

**Related Papers:**

- López de Prado, M. (2015): "The Future of Empirical Finance." _The Journal of Portfolio Management_.
- López de Prado, M. (2020): _Machine Learning for Asset Managers_. Cambridge University Press.
- Rao, C., P. Pathak and V. Koltchinskii (1997): "Bootstrap by sequential resampling." Journal of Statistical Planning and Inference, Vol. 64, No. 2, pp. 257–281.
- King, G. and L. Zeng (2001): "Logistic Regression in Rare Events Data." Working paper, Harvard University. Available at [https://gking.harvard.edu/files/0s.pdf](https://www.mql5.com/go?link=https://gking.harvard.edu/files/0s.pdf "https://gking.harvard.edu/files/0s.pdf").
- Lo, A. (2017): _Adaptive Markets, 1st ed_. Princeton University Press.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19850.zip "Download all attachments in the single ZIP archive")

[multiprocess.py](https://www.mql5.com/en/articles/download/19850/multiprocess.py "Download multiprocess.py")(9 KB)

[strategies.py](https://www.mql5.com/en/articles/download/19850/strategies.py "Download strategies.py")(4.34 KB)

[fractals.py](https://www.mql5.com/en/articles/download/19850/fractals.py "Download fractals.py")(16.45 KB)

[misc.py](https://www.mql5.com/en/articles/download/19850/misc.py "Download misc.py")(19.8 KB)

[time.py](https://www.mql5.com/en/articles/download/19850/time.py "Download time.py")(8.25 KB)

[ma\_crossover\_feature\_engine.py](https://www.mql5.com/en/articles/download/19850/ma_crossover_feature_engine.py "Download ma_crossover_feature_engine.py")(18.25 KB)

[triple\_barrier.py](https://www.mql5.com/en/articles/download/19850/triple_barrier.py "Download triple_barrier.py")(18.88 KB)

[moving\_averages.py](https://www.mql5.com/en/articles/download/19850/moving_averages.py "Download moving_averages.py")(3 KB)

[trend\_scanning.py](https://www.mql5.com/en/articles/download/19850/trend_scanning.py "Download trend_scanning.py")(11.02 KB)

[filters.py](https://www.mql5.com/en/articles/download/19850/filters.py "Download filters.py")(8.39 KB)

[returns.py](https://www.mql5.com/en/articles/download/19850/returns.py "Download returns.py")(6.27 KB)

[volatility.py](https://www.mql5.com/en/articles/download/19850/volatility.py "Download volatility.py")(5.38 KB)

[attribution.py](https://www.mql5.com/en/articles/download/19850/attribution.py "Download attribution.py")(5.88 KB)

[concurrent.py](https://www.mql5.com/en/articles/download/19850/concurrent.py "Download concurrent.py")(4.96 KB)

[optimized\_attribution.py](https://www.mql5.com/en/articles/download/19850/optimized_attribution.py "Download optimized_attribution.py")(15.08 KB)

[optimized\_concurrent.py](https://www.mql5.com/en/articles/download/19850/optimized_concurrent.py "Download optimized_concurrent.py")(13.12 KB)

[bollinger\_features.py](https://www.mql5.com/en/articles/download/19850/bollinger_features.py "Download bollinger_features.py")(11.68 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MetaTrader 5 Machine Learning Blueprint (Part 6): Engineering a Production-Grade Caching System](https://www.mql5.com/en/articles/20302)
- [MetaTrader 5 Machine Learning Blueprint (Part 5): Sequential Bootstrapping—Debiasing Labels, Improving Returns](https://www.mql5.com/en/articles/20059)
- [MetaTrader 5 Machine Learning Blueprint (Part 3): Trend-Scanning Labeling Method](https://www.mql5.com/en/articles/19253)
- [MetaTrader 5 Machine Learning Blueprint (Part 2): Labeling Financial Data for Machine Learning](https://www.mql5.com/en/articles/18864)
- [MetaTrader 5 Machine Learning Blueprint (Part 1): Data Leakage and Timestamp Fixes](https://www.mql5.com/en/articles/17520)

**[Go to discussion](https://www.mql5.com/en/forum/498960)**

![Black-Scholes Greeks: Gamma and Delta](https://c.mql5.com/2/178/20054-black-scholes-greeks-gamma-logo.png)[Black-Scholes Greeks: Gamma and Delta](https://www.mql5.com/en/articles/20054)

Gamma and Delta measure how an option’s value reacts to changes in the underlying asset’s price. Delta represents the rate of change of the option’s price relative to the underlying, while Gamma measures how Delta itself changes as price moves. Together, they describe an option’s directional sensitivity and convexity—critical for dynamic hedging and volatility-based trading strategies.

![Automating Trading Strategies in MQL5 (Part 37): Regular RSI Divergence Convergence with Visual Indicators](https://c.mql5.com/2/176/20031-automating-trading-strategies-logo.png)[Automating Trading Strategies in MQL5 (Part 37): Regular RSI Divergence Convergence with Visual Indicators](https://www.mql5.com/en/articles/20031)

In this article, we build an MQL5 EA that detects regular RSI divergences using swing points with strength, bar limits, and tolerance checks. It executes trades on bullish or bearish signals with fixed lots, SL/TP in pips, and optional trailing stops. Visuals include colored lines on charts and labeled swings for better strategy insights.

![Introduction to MQL5 (Part 27): Mastering API and WebRequest Function in MQL5](https://c.mql5.com/2/178/17774-introduction-to-mql5-part-27-logo.png)[Introduction to MQL5 (Part 27): Mastering API and WebRequest Function in MQL5](https://www.mql5.com/en/articles/17774)

This article introduces how to use the WebRequest() function and APIs in MQL5 to communicate with external platforms. You’ll learn how to create a Telegram bot, obtain chat and group IDs, and send, edit, and delete messages directly from MT5, building a strong foundation for mastering API integration in your future MQL5 projects.

![Building a Smart Trade Manager in MQL5: Automate Break-Even, Trailing Stop, and Partial Close](https://c.mql5.com/2/177/19911-building-a-smart-trade-manager-logo.png)[Building a Smart Trade Manager in MQL5: Automate Break-Even, Trailing Stop, and Partial Close](https://www.mql5.com/en/articles/19911)

Learn how to build a Smart Trade Manager Expert Advisor in MQL5 that automates trade management with break-even, trailing stop, and partial close features. A practical, step-by-step guide for traders who want to save time and improve consistency through automation.

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/19850&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068749712498031926)

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).

![close](https://c.mql5.com/i/close.png)

![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)

You are missing trading opportunities:

- Free trading apps
- Over 8,000 signals for copying
- Economic news for exploring financial markets

RegistrationLog in

latin characters without spaces

a password will be sent to this email

An error occurred


- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)

You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)

If you do not have an account, please [register](https://www.mql5.com/en/auth_register)

Allow the use of cookies to log in to the MQL5.com website.

Please enable the necessary setting in your browser, otherwise you will not be able to log in.

[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)

- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)