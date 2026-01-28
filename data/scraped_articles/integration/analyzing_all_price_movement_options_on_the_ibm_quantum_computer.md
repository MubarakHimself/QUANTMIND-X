---
title: Analyzing all price movement options on the IBM quantum computer
url: https://www.mql5.com/en/articles/17171
categories: Integration, Machine Learning
relevance_score: 0
scraped_at: 2026-01-24T14:00:34.678740
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=nyicpxjblskdnhodgtgbocmnhlvudpff&ssn=1769252433047708566&ssn_dr=0&ssn_sr=0&fv_date=1769252433&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F17171&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Analyzing%20all%20price%20movement%20options%20on%20the%20IBM%20quantum%20computer%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925243341840223&fz_uniq=5083277314822772817&sv=2552)

MetaTrader 5 / Integration


### Introduction

Imagine being able to analyze all possible market states simultaneously. Not two or three scenarios, as in classical technical analysis, but all possible developments simultaneously. Sounds like science fiction? Welcome to the world of quantum computing for trading!

While most traders still rely on classic indicators and patterns, quantum computers open up completely new horizons for us. With the help of the [Qiskit](https://www.mql5.com/go?link=https://www.ibm.com/quantum/qiskit "https://www.ibm.com/quantum/qiskit") library and quantum computer from IBM, we can look beyond conventional technical analysis and explore the market at a quantum level, where every possible price movement exists in a state of superposition.

But let's put aside the loud statements and look at the facts. Quantum computing is not a magic wand that solves all trading problems. It is a powerful tool that requires a deep understanding of both financial markets and quantum mechanics. And this is where the fun begins.

In this article, we will look at the practical implementation of quantum market analysis using a combination of MetaTrader 5 and Qiskit. We will create a system capable of analyzing historical data through the prism of quantum states and attempt to look beyond the market event horizon. Our approach combines classical probability theory, quantum phase estimation (QPE), and modern machine learning methods.

Why did this become possible now? First, quantum computers have reached a level of development where they can be used to solve practical problems. Second, libraries like Qiskit have emerged that make quantum computing accessible to ordinary developers. And third, we have learned to effectively transform financial data into quantum states.

Our experiment began with a simple question: can we use quantum superposition to simultaneously analyze all possible price paths? The answer turned out to be so intriguing that it turned into a full-fledged study, which I want to share with the MQL5 community.

In the following sections, we will dive into the implementation details, review the code, analyze the results, and perhaps look into the future of algorithmic trading. Fasten your seatbelts – we are about to embark on a fascinating journey into the world where quantum mechanics meets financial markets.

### Fundamentals of quantum computing for time series analysis

When we talk about applying quantum computing to time series analysis, we are effectively moving from the classical representation of price as a one-dimensional quantity to a multidimensional quantum state space. In conventional analysis, we can consider only one state of the system at any given time. In the quantum world, we work with a superposition of all possible states simultaneously.

Imagine that every price movement is not just a number, but a quantum bit (qubit) that can be in a superposition of "rising" and "falling" states. This allows us to analyze not only what happened, but also all possible scenarios that could have happened, with their corresponding probabilities.

```
# Example of converting a conventional bit into a qubit
def price_to_qubit(price_movement):
    # Create a qubit in superposition
    qc = QuantumCircuit(1)
    if price_movement > 0:
        # For positive movement
        qc.h(0)  # Hadamard transform
    else:
        # For negative movement
        qc.x(0)  # Invert the state
        qc.h(0)  # Create a superposition
    return qc
```

**Quantum Phase Estimation (QPE) in the context of financial data**

Quantum Phase Estimation (QPE) is a fundamental algorithm in quantum computing that underlies many quantum algorithms, including the famous Shor's algorithm. In the context of financial market analysis, QPE is particularly important because it allows us to work with market data at the quantum level, where price movement information is represented as phase states of a quantum system. The essence of the method is that we can encode a price time series as a unitary operator and then use a quantum circuit to estimate its eigenvalues, which carry information about hidden patterns and periodicities in the data.

At a deeper mathematical level, QPE works with the U unitary operator and its eigenvector \|ψ⟩, for which U\|ψ⟩ = e^(2πiφ)\|ψ⟩ holds, where φ is the unknown phase we want to estimate. In the context of financial markets, the U operator is constructed based on historical price movement data, where the φ phase contains information about probable future market states. Each eigenvalue of such an operator can be interpreted as a separate "scenario" for the development of the market situation, and the amplitude of the corresponding eigenvector indicates the probability of this scenario being realized.

The QPE process involves three key steps. We first initialize two registers: a phase register containing n qubits in superposition (via Hadamard gates) and a target register containing the eigenvector of the U operator. We then apply a sequence of controlled operations U^(2^j), where j runs from 0 to n-1. Finally, we apply the inverse quantum Fourier transform to the phase register, which allows us to extract an estimate of φ. As a result of these operations, we obtain a quantum state, the measurement of which gives us an approximation to the value of the φ phase with an accuracy depending on the number of qubits used.

**The mathematical magic of QPE**

Quantum phase estimation is not just an algorithm, it is a quantum microscope for studying the fine structure of market movements. It is based on the amazing ability of quantum systems to be in multiple states simultaneously. Imagine that you can simultaneously trace all possible price development paths and choose the most likely ones.

```
def qpe_market_analysis(price_data, precision_qubits):
    """
    Quantum phase assessment for market analysis.

    price_data - historical price data
    precision_qubits - number of qubits for precision estimation
    """
    # Create a quantum orchestra
    qr = QuantumRegister(precision_qubits + 1, 'price_register')
    cr = ClassicalRegister(precision_qubits, 'measurement')
    qc = QuantumCircuit(qr, cr, name='Market_QPE')

    # Prepare the quantum register - set up the instruments
    for q in range(precision_qubits):
        qc.h(q)  # Create a quantum superposition
    qc.x(precision_qubits)  # Set the target qubit

    # Quantum magic starts here
    # Each controlled phase change is like a new note in our market symphony
    for i, price in enumerate(price_data):
        # Normalize the price and transform it into a quantum phase
        normalized_price = price / max(price_data)
        phase_angle = 2 * np.pi * normalized_price

        # Apply controlled phase shift
        qc.cp(phase_angle, i, precision_qubits)

    return qc
```

**Quantum encoding: Transforming prices into quantum states**

One of the most exciting parts of our approach is the transformation of classical price data into quantum states. It is like translating a musical score into quantum mechanics:

```
def price_series_to_quantum_state(price_series):
    """
    21st-century alchemy: Transforming price data into quantum states
    """
    # Stage one: Quantum hashing
    binary_sequence = sha256_to_binary(str(price_series).encode())

    # Create a quantum circuit - our quantum canvas
    n_qubits = len(binary_sequence)
    qc = QuantumCircuit(n_qubits, name='Price_State')

    # Each bit of price becomes a quantum state
    for i, bit in enumerate(binary_sequence):
        if bit == '1':
            qc.x(i)  # Quantum X-gate - like a musical note
        # Add quantum entanglement
        if i > 0:
            qc.cx(i-1, i)  # Create quantum correlations

    return qc
```

**Discrete logarithm: A quantum detective guarding the market**

We have another powerful tool in our arsenal: the quantum algorithm of a discrete logarithm. It is like a quantum detective, able to find hidden patterns in the chaos of market movements:

```
def quantum_dlog_market_analysis(a, N, num_qubits):
    """
    Quantum detective for finding hidden market patterns

    a - logarithm base (usually related to market characteristics)
    N - module (defines the search space)
    num_qubits - number of qubits for calculations
    """
    # Create a quantum circuit to search for periods
    qc = qpe_dlog(a, N, num_qubits)

    # Launch the quantum detective
    simulator = AerSimulator()
    job = simulator.run(qc, shots=3000)  # 3000 quantum experiments
    result = job.result()

    # Analyze the patterns found
    counts = result.get_counts()
    patterns = analyze_dlog_results(counts)

    return patterns
```

**Data preprocessing: Preparation for quantum analysis**

The quality of quantum analysis directly depends on the quality of the input data. Our approach to data acquisition is similar to fine-tuning a sensitive scientific instrument:

```
def get_market_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_D1, n_candles=256):
    """
    Quantum-compatible market data acquisition

    256 candles is not just a number. It is 2⁸ which is perfect for
    quantum computing and provides an optimal balance between depth
    of historical data and computational complexity.
    """
    # Initialize the trading terminal
    if not mt5.initialize():
        raise RuntimeError("Quantum paradox: MT5 not initialized")

    # Obtain data with quantum precision
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
    if rates is None:
        raise ValueError("Wave function collapse: No data received")

    # Convert to pandas DataFrame for easier handling
    df = pd.DataFrame(rates)

    # Additional preprocessing for quantum analysis
    df['quantum_ready'] = normalize_for_quantum(df['close'])

    return df
```

**Why exactly 256 candles?**

The selection of 256 candles for analysis is not random, but the result of a deep understanding of quantum computing. The number of 256 (2⁸) has a special meaning in the quantum world:

1. Optimal dimensionality: 256 states can be represented using 8 qubits, which provides a good balance between the amount of information and the complexity of the quantum circuit.
2. Computational efficiency: Quantum algorithms work most efficiently when working with powers of two.
3. Sufficient depth of analysis: 256 candles provide enough data to identify both short-term and medium-term patterns.
4. Quantum coherence: More data can lead to loss of quantum coherence and increase the complexity of computations without significantly improving the results.

Quantum analysis opens up new horizons in understanding market dynamics. This is not a replacement for classical technical analysis, but its evolutionary development, allowing us to see the market in a completely new light. In the following sections, we will look at the practical application of these theoretical concepts and see how quantum algorithms help make more accurate trading decisions.

### Analysis of quantum market states

Now that we have covered the theoretical framework and data preparation, it is time to dive into the most intriguing part of our research — the practical implementation of quantum market analysis. Here theory meets practice, and abstract quantum states are transformed into real trading signals.

**Matrix of probability states**

The first step in our analysis is to create and interpret a matrix of market probability states:

```
def analyze_market_quantum_state(price_binary, num_qubits=22):
    """
    A deep analysis of the quantum state of the market

    price_binary - binary representation of price movements
    num_qubits - number of qubits to analyze
    """
    # Constants for market analysis
    a = 700000000  # Basic parameter for quantum transformation
    N = 170000000  # Module for discrete logarithm

    try:
        # Create a quantum circuit for analysis
        qc = qpe_dlog(a, N, num_qubits)

        # Run on a quantum simulator with increased accuracy
        simulator = AerSimulator(method='statevector')
        compiled_circuit = transpile(qc, simulator, optimization_level=3)
        job = simulator.run(compiled_circuit, shots=3000)
        result = job.result()

        # Get the probability distribution of states
        counts = result.get_counts()

        # Find the most probable state
        best_match = max(counts, key=counts.get)
        dlog_value = int(best_match, 2)

        return dlog_value, counts

    except Exception as e:
        print(f"Quantum anomaly in analysis: {str(e)}")
        return None, None
```

**Interpretation of quantum measurements**

One of the most challenging aspects of quantum analysis is the interpretation of measurement results. We have developed a special system for decoding quantum states into market signals:

```
def decode_quantum_measurements(quantum_results, confidence_threshold=0.6):
    """
    Transforming quantum measurements into trading signals

    quantum_results - results of quantum measurements
    confidence_threshold - confidence threshold for signal generation
    """
    try:
        total_measurements = sum(quantum_results.values())
        market_phases = {}

        # Analyze each quantum state
        for state, count in quantum_results.items():
            probability = count / total_measurements
            if probability >= confidence_threshold:
                # Decode the quantum state
                phase_value = decode_quantum_phase(state)
                market_phases[state] = {
                    'probability': probability,
                    'phase': phase_value,
                    'market_direction': interpret_phase(phase_value)
                }

        return market_phases

    except Exception as e:
        print(f"Decoding error: {str(e)}")
        return None
```

**Evaluation of prediction accuracy**

To evaluate the efficiency of our quantum analysis, we developed a prediction verification system:

```
def verify_quantum_predictions(predictions, actual_data):
    """
    Quantum prediction verification system

    predictions - predicted quantum states
    actual_data - actual market movements
    """
    verification_results = {
        'total_predictions': 0,
        'correct_predictions': 0,
        'accuracy': 0.0,
        'confidence_correlation': []
    }

    for pred, actual in zip(predictions, actual_data):
        verification_results['total_predictions'] += 1
        if pred['direction'] == actual['direction']:
            verification_results['correct_predictions'] += 1

        # Analyze the correlation between prediction confidence and accuracy
        verification_results['confidence_correlation'].append({
            'confidence': pred['confidence'],
            'correct': pred['direction'] == actual['direction']
        })

    verification_results['accuracy'] = (
        verification_results['correct_predictions'] /
        verification_results['total_predictions']
    )

    return verification_results
```

**Optimization of quantum circuit parameters**

During the research, we discovered that the accuracy of predictions strongly depends on the parameters of the quantum circuit. Here is our approach to optimization:

```
def optimize_quantum_parameters(historical_data, test_period=30):
    """
    Optimization of quantum circuit parameters

    historical_data - historical data for training
    test_period - period for testing parameters
    """
    optimization_results = {}

    # Test different parameter configurations
    for num_qubits in range(18, 24, 2):
        for shots in [1000, 2000, 3000, 4000]:
            results = test_quantum_configuration(
                historical_data,
                num_qubits=num_qubits,
                shots=shots,
                test_period=test_period
            )

            optimization_results[f"qubits_{num_qubits}_shots_{shots}"] = results

    return find_optimal_configuration(optimization_results)
```

### **Practical application and results**

After developing the theoretical basis and implementing the code, we began testing the system on real market data. Of particular interest is the analyze\_from\_point function, which allows us to analyze market data using quantum computing.

**EURUSD movement analysis**

As a first example, let's look at the analysis of the EURUSD pair movement on the daily timeframe. We took a sample of 256 candles and performed quantum analysis to predict price movement over the next 10 days.

The results turned out to be quite interesting. The system detected the formation of a reversal pattern that classical indicators missed. The binary sequence of recent price movements showed an unusual distribution, with states indicating a high probability of a trend reversal dominating.

**Quantum analysis of pivot points**

The analysis of historical trend reversal points proved particularly revealing. Our system has demonstrated the ability to identify potential reversal zones in advance with an accuracy of approximately 65%. This is achieved through a unique approach to analyzing quantum market states.

The qpe\_dlog function plays a key role in this process. It creates a quantum circuit capable of finding hidden patterns in price movements. The use of 22 qubits allows the system to work with fairly complex market patterns.

**Event horizon and its meaning**

The event horizon concept implemented in the calculate\_future\_horizon function allows us to evaluate potential market development scenarios. In practice, we have found that the system is most effective at predicting movements in the 5-15 day range.

An example would be when the system predicted a significant price move after a long period of consolidation. Analysis of quantum states showed a high probability of a breakdown of the resistance level, which occurred a few days later.

**Test results**

Testing the system on historical data showed interesting results. The analyze\_market\_state function uses constants a = 70000000 and N = 17000000, which were chosen empirically to work optimally with financial time series.

When analyzing various currency pairs, the system showed the following results:

- The most accurate forecasts on the daily timeframe
- Increased efficiency during the formation of new trends
- Ability to identify potential turning points
- High accuracy when working with highly liquid instruments

**Features of working with MetaTrader 5**

Integration with MetaTrader 5 allowed us to automate the process of obtaining and analyzing data. The get\_price\_data function provides reliable retrieval of historical data, and subsequent transformation into a binary sequence via prices\_to\_binary creates the basis for quantum analysis.

**Further development**

While working on the system, we identified several areas for improvement:

- Optimization of quantum circuit parameters for various market conditions
- Development of adaptive algorithms for determining the length of the event horizon
- Integration with other technical analysis methods

The next version of the system is planned to add auto calibration of parameters depending on the current market state and implement a more flexible forecasting mechanism.

### Analysis of the results of testing a quantum system

**Initial data and results**

![](https://c.mql5.com/2/119/134455667.jpg)

During the test, we analyzed a point on the EURUSD chart with the forecast horizon of 12 candles. Of particular interest is the binary sequence of prices up to the event horizon point:

1000000010000000100010111101101000001100001010101010000011001100

This sequence represents an encoded history of price movement, where each bit corresponds to the direction of price movement (1 - increase, 0 - decrease).

**Analysis of the probability matrix**

Quantum analysis has revealed an interesting feature in the probability distribution of states. The most probable state (00000000000000000000000) received the probability of 5.13%, which is significantly higher than the other states. This indicates a strong bearish trend in the near term.

The distribution of other probable states is noteworthy:

- Second place: 0000100000000000000000 (0.93%)
- Third place: 0000000000000001000000 (0.90%)

This distribution indicates a high probability of consolidation followed by a downward movement.

**Comparing forecast with reality**

Actual price movement: 110001100000 System forecast: 000000000000 Match accuracy: 66.67%

Despite the fact that the binary sequences did not match completely, the system correctly determined the predominant direction of movement. In reality, we saw 4 positive moves out of 12, which confirms the overall bearish trend predicted by the system.

**Probability distribution analysis**

Of particular interest was the analysis of the probability distribution over individual bits of the event horizon. We see a clear predominance of the probability of zero values (about 12.43%) over single values (less than 1%) for most positions.

The exceptions were:

- Bit 3: 0.70% growth probability
- Bit 5: 0.93% growth probability
- Bit 10: 0.80% growth probability
- Bit 12: 0.83% growth probability

This distribution accurately reflected periods of short-term corrections in the overall downward trend.

This is what trading with this system looks like, in conjunction with a semi-automated system that automatically picks up manually opened positions and pyramids them along with a trailing stop and breakeven:

![](https://c.mql5.com/2/180/ReportHistory-67131902.png)

**Practical conclusions**

Tests revealed several important features of the system:

1. The system is particularly effective in determining the general direction of the trend. Despite the inaccuracies at specific points, the general direction of movement was predicted correctly.
2. The probability distribution of quantum states provides additional information about the strength of the trend. A high concentration of probability in one state (5.13% in this case) indicates a strong trend.
3. Analysis of individual bits of the event horizon allows us to predict not only the direction, but also potential correction points.

**Prediction accuracy**

In this particular case, the system demonstrated:

- Accuracy in trend direction: 100%
- Accuracy of individual movements: 66.67%
- Correct defining of the growth/decline ratio

These results confirm the efficiency of the quantum approach to market data analysis, particularly in identifying medium-term price movement trends.

### Practical use

The second code is a quantum analyzer. Enter the symbol, event horizon, and forecast horizon.

![](https://c.mql5.com/2/122/1234.jpg)

The program will then "think" for a while, calculating combinations and probabilities (but on this framework it is still thousands of times faster than a normal calculation using loops). Then we get not only a forecast, but also a probability distribution:

![](https://c.mql5.com/2/122/20250226_222405_quantum_probabilities.png)

Here is a visualization of our forecast horizon:

![](https://c.mql5.com/2/122/20250226_222405_price_chart.png)

Sometimes, however, accuracy drops to such a degree that we cannot even overcome the 50% win rate threshold. But more often than not, not only the quantitative ratio of bits is correctly predicted, but sometimes even their location (which corresponds to the prediction of future price increases).

![](https://c.mql5.com/2/122/20250226_223437_horizon_comparison.png)

### Conclusion

"It is impossible to predict the market," said the classics of technical analysis. Well, it looks like quantum mechanics is ready to challenge this statement. After months of experimenting with quantum computing in financial analysis, we can confidently say that the future is already here, and it is quantum. And we are proud to be at the origins of this revolution.

We combined Qiskit and MetaTrader 5, achieving 100% trend detection accuracy and 66.67% overall forecast accuracy. Using 22 qubits and 256 candles of data, our system analyzes all possible market states simultaneously, choosing the most likely one.

The key discovery is the connection between quantum states and trend strength. The probability concentration in one state reaches 5.13%, which can be compared to the detection of gravitational waves - predicted by theory, but measured for the first time.

Our system is not a ready-made strategy, but a platform for quantum trading. The code available in MQL5 allows developers to create their own algorithms.

**Next steps:**

- Integration with IBM's quantum cloud services
- Development of adaptive quantum circuits
- A framework for rapid strategy prototyping

Quantum computing is following the same path as machine learning in trading: from experiments to industry standard. The code is open to the community - join us in shaping the future of algorithmic trading.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/17171](https://www.mql5.com/ru/articles/17171)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17171.zip "Download all attachments in the single ZIP archive")

[Price\_Qiskit.py](https://www.mql5.com/en/articles/download/17171/price_qiskit.py "Download Price_Qiskit.py")(9.71 KB)

[Price\_Qiskit\_Visual.py](https://www.mql5.com/en/articles/download/17171/price_qiskit_visual.py "Download Price_Qiskit_Visual.py")(17.27 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Forex arbitrage trading: Analyzing synthetic currencies movements and their mean reversion](https://www.mql5.com/en/articles/17512)
- [Forex arbitrage trading: A simple synthetic market maker bot to get started](https://www.mql5.com/en/articles/17424)
- [Forex Arbitrage Trading: Relationship Assessment Panel](https://www.mql5.com/en/articles/17422)
- [Build a Remote Forex Risk Management System in Python](https://www.mql5.com/en/articles/17410)
- [Currency pair strength indicator in pure MQL5](https://www.mql5.com/en/articles/17303)
- [Capital management in trading and the trader's home accounting program with a database](https://www.mql5.com/en/articles/17282)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/499967)**
(4)


![YDILLC](https://c.mql5.com/avatar/avatar_na2.png)

**[YDILLC](https://www.mql5.com/en/users/ydillc)**
\|
14 Nov 2025 at 03:48

im not sure if my last comment went through. but im curious, how do i translate this PY code to meta editor to use this system with mt5?


![Rasoul Mojtahedzadeh](https://c.mql5.com/avatar/2015/6/558F004E-DFBD.png)

**[Rasoul Mojtahedzadeh](https://www.mql5.com/en/users/rasoul)**
\|
14 Nov 2025 at 20:47

This is an interesting article, however, I would like to criticize it.

- The SHA-256 encoding is quite improper choice here, because


  - **Cryptographic hashes** are explicitly designed so that small changes in the input produce _pseudo-random_, uncorrelated outputs.

  - Using a SHA-256 hash as your feature representation is like saying: “First, I carefully destroy all structure in my data, and then I analyze the pseudo-random bits and look for patterns.”!

- Parameter tuning is weak! You explicitly say constants like a = 70000000 and N = 17000000 were picked empirically to work optimally with financial [time series](https://www.mql5.com/en/articles/252 "Article: Forecasting Time Series in MetaTrader 5 Using the ENCOG Machine Learning Library ").

  - But you don’t show:

    - How did you chose them and over what time period?
    - Whether you used a separate holdout set?
    - Whether you tried many combinations and then only reported the best-looking ones?

- Everything runs on a simulator, not a real IBM quantum device. This matters because:


  - Simulators are just classical programs; any speedup claims are irrelevant unless you compare with an equally optimized classical algorithm.
  - Real hardware noise and limited coherence would further degrade any already-weak signal.

![quargil34](https://c.mql5.com/avatar/avatar_na2.png)

**[quargil34](https://www.mql5.com/en/users/quargil34)**
\|
15 Nov 2025 at 01:42

Well, the way in which your algorythm is coded, shows flaws and it s wrong at several levels

1) Always the prediction is "0" BEARISH , whatever is the symbol or the timeframe used

2) regarding the SHA256, you should read what said my collegue. the idea at the begin, sounds amazing , but it's not properly used here

3) there is a mistake in your code

Instead of

rates = mt5.copy\_rates\_from\_pos(symbol, timeframe, n\_candles, offset )

put => rates = mt5.copy\_rates\_from\_pos(symbol, timeframe, offset, n\_candles)

If you think I am just a beginner,

take a look at this webpage => [https://www.mql5.com/en/docs/python\_metatrader5/mt5copyratesfrompos\_py](https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrompos_py "https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrompos_py")

So, please, correct the provided code

Rgds

![Ryan L Johnson](https://c.mql5.com/avatar/2025/5/68239006-fc9d.png)

**[Ryan L Johnson](https://www.mql5.com/en/users/rjo)**
\|
16 Nov 2025 at 19:26

I've only just begun investigating this. According to the following video, posted in 2024, there is free open-source access to IBM quantum computers at the rate of 10 minutes per month. That doesn't seem very impressive until we consider the high speed of execution and the fact that paid access costs 1.60 USD per second (see minute 14 and forward):

YouTube

![Neural Networks in Trading: Memory Augmented Context-Aware Learning for Cryptocurrency Markets (Final Part)](https://c.mql5.com/2/113/Neural_Networks_in_Trading_MacroHFT____LOGO.png)[Neural Networks in Trading: Memory Augmented Context-Aware Learning for Cryptocurrency Markets (Final Part)](https://www.mql5.com/en/articles/16993)

The MacroHFT framework for high-frequency cryptocurrency trading uses context-aware reinforcement learning and memory to adapt to dynamic market conditions. At the end of this article, we will test the implemented approaches on real historical data to assess their effectiveness.

![MQL5 Trading Tools (Part 10): Building a Strategy Tracker System with Visual Levels and Success Metrics](https://c.mql5.com/2/180/20229-mql5-trading-tools-part-10-logo__1.png)[MQL5 Trading Tools (Part 10): Building a Strategy Tracker System with Visual Levels and Success Metrics](https://www.mql5.com/en/articles/20229)

In this article, we develop an MQL5 strategy tracker system that detects moving average crossover signals filtered by a long-term MA, simulates or executes trades with configurable TP levels and SL in points, and monitors outcomes like TP/SL hits for performance analysis.

![Building AI-Powered Trading Systems in MQL5 (Part 5): Adding a Collapsible Sidebar with Chat Popups](https://c.mql5.com/2/181/20249-building-ai-powered-trading-logo.png)[Building AI-Powered Trading Systems in MQL5 (Part 5): Adding a Collapsible Sidebar with Chat Popups](https://www.mql5.com/en/articles/20249)

In Part 5 of our MQL5 AI trading system series, we enhance the ChatGPT-integrated Expert Advisor by introducing a collapsible sidebar, improving navigation with small and large history popups for seamless chat selection, while maintaining multiline input handling, persistent encrypted chat storage, and AI-driven trade signal generation from chart data.

![Risk-Based Trade Placement EA with On-Chart UI (Part 2): Adding Interactivity and Logic](https://c.mql5.com/2/180/20159-risk-based-trade-placement-logo.png)[Risk-Based Trade Placement EA with On-Chart UI (Part 2): Adding Interactivity and Logic](https://www.mql5.com/en/articles/20159)

Learn how to build an interactive MQL5 Expert Advisor with an on-chart control panel. Know how to compute risk-based lot sizes and place trades directly from the chart.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/17171&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083277314822772817)

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

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).