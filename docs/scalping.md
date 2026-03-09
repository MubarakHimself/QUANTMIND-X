This is a comprehensive technical manual for the **QUANTMIND-X** project. It details 20 scalping and retail HFT strategies specifically for MetaTrader 5 (MT5), incorporating the mathematical rigor (Prop Score, Kelly, PBO) we extracted from your videos.

---

# **QUANTMIND-X: The Master Scalping & HFT Strategy Manual**

## **I. Infrastructure Guidelines (The HFT Foundation)**

Before implementing these strategies, your MT5 EA environment must meet these "QuantMind" requirements:

1. **Direct Market Access (DMA):** Ensure your broker provides real `MqlTick` data, not just bar data.
2. **Event-Driven Execution:** Use the `void OnTick()` and `void OnBookEvent()` handlers. Standard timers are too slow for these strategies.
3. **Filling Policy:** Use `ORDER_FILLING_IOC` or `ORDER_FILLING_FOK` to prevent "Slippage Decay."
4. **Data Logging:** Log every tick to a CSV for "In-Sample/Out-of-Sample" PBO testing later.

---

## **II. Microstructure & Order Flow Strategies (HFT-Retail)**

### **1. The Passive Absorption Fade**

* **Logic:** Identifies when "Aggressive" market orders hit a "Passive" limit wall.
* **MT5 Setup:** Monitor `MqlTick.last_volume` vs. price change.
* **Entry:** Total volume over the last 5 seconds > $3\sigma$ of the daily mean, but price movement is $< 10\%$ of ATR. Enter a reversal.
* **Optimization:** Adjust the "Lookback" window based on session volatility (London vs. Asia).

### **2. Tick Velocity Momentum**

* **Logic:** Trades the "Burst" of execution speed that precedes a breakout.
* **MT5 Setup:** Track the millisecond timestamp between `OnTick()` calls.
* **Entry:** If 20 ticks occur in $< 100ms$ and the price direction is consistent, enter in that direction.
* **Exit:** Trailing stop based on the last 5-tick low.

### **3. Orderbook Imbalance (OFI) Scalp**

* **Logic:** Calculating the "Pressure" of the limit order book.
* **MT5 Setup:** Use `MarketBookGet`.
* **Entry:** Calculate $OFI = (BidVol / (BidVol + AskVol))$. If $OFI > 0.8$, Buy. If $OFI < 0.2$, Sell.
* **Risk:** Highly susceptible to "Spoofing" (fake orders). Filter with real trade volume.

### **4. Delta Divergence (Aggression Shift)**

* **Logic:** Price makes a new high, but market buying volume is drying up.
* **MT5 Setup:** Calculate `Cumulative Delta` (Market Buys - Market Sells).
* **Entry:** Price $>$ 5-min High + Delta $<$ 5-min High. Short the "exhaustion."
* **Filter:** Only trade this at key structural levels (S/R).

### **5. The "Liquidity Sweep" Reversal**

* **Logic:** Fading the "Stop-Run" liquidity.
* **MT5 Setup:** Detect high-volume spikes at the breach of a 1-hour High/Low.
* **Entry:** Price breaches the level, triggers stops (volume spike), then reverses. Enter on the candle close back inside the range.
* **Target:** The 50% retracement of the sweep move.

---

## **III. Statistical & Mean Reversion Strategies**

### **6. VWAP Standard Deviation Scalp**

* **Logic:** Regression to the Volume Weighted Average Price.
* **MT5 Setup:** Custom VWAP indicator using `iVolume` and `iClose`.
* **Entry:** Price touches the $\pm 2.5$ Standard Deviation band of the VWAP.
* **Exit:** Touch of the VWAP (The "Mean").

### **7. Relative Strength Index (RSI-2) Extreme**

* **Logic:** High-frequency oscillator for choppy markets.
* **Parameters:** RSI Period 2, Levels 95/5.
* **Entry:** Buy when RSI < 5; Sell when RSI > 95.
* **Filter:** ADX must be $< 20$ (indicating a non-trending market).

### **8. Bollinger Band Width "Squeeze"**

* **Logic:** Trading the expansion of a "coiled" market.
* **Entry:** Bollinger Band Width reaches a 48-hour low. Enter when a 1-minute candle closes outside the bands.
* **Management:** Move Stop-Loss to Breakeven after 1:1 risk/reward.

### **9. Z-Score Price Divergence**

* **Logic:** Statistically identifying "Outliers."
* **Formula:** $Z = (Price - SMA) / StdDev$.
* **Entry:** $Z > 3.0$ (Short) or $Z < -3.0$ (Long).
* **Logic:** Betting on a 99.7% statistical probability of a return to the mean.

### **10. Tick-Based ATR Expansion**

* **Logic:** Riding the surge of a volatility spike.
* **MT5 Setup:** ATR(10) on the 1-minute chart.
* **Entry:** Current ATR is $2x$ the average of the last 100 bars. Buy/Sell in direction of the trend.
* **Exit:** Time-based exit (3 minutes) or fixed pip target.

---

## **IV. Pattern & Micro-Price Action**

### **11. Fair Value Gap (FVG) Scalper**

* **Logic:** Pricing inefficiency/imbalance fill.
* **Entry:** Identify a 3-bar "Gap" (e.g., Candle 1 High < Candle 3 Low). Place a limit order at the gap entry.
* **Success Factor:** High-volume FVGs are more likely to be filled and then reverse.

### **12. London Open "Gap & Go"**

* **Logic:** Capturing the initial institutional flow at 8:00 AM GMT.
* **Entry:** Buy the break of the first 1-minute candle high of the London session.
* **Management:** Very tight stop (3-5 pips).

### **13. Inside Bar "Spring"**

* **Logic:** Trading the breakout of an "Indecision" candle.
* **Entry:** M1 Inside Bar. Trade the break of the "Mother Bar."
* **Adjustment:** Require a "Micro-Trend" on the M5 chart for confirmation.

### **14. The "Three-Tap" Liquidity Pattern**

* **Logic:** Price tests a level 3 times, clearing liquidity each time.
* **Entry:** Third touch of a resistance/support level with a "Pin Bar" rejection.
* **Exit:** The "Swing Low" of the previous tap.

### **15. Pin Bar + Volume Confirmation**

* **Logic:** Price rejection backed by high volume.
* **Entry:** M1 Pin Bar + Volume $>$ 1.5x previous bar.
* **Risk:** Ensure the "Nose" of the pin bar is pointing into a supply/demand zone.

---

## **V. Cross-Asset & Correlation (Advanced QUANTMIND-X)**

### **16. BTC/ES Monday Open Gap (The "Video Strategy")**

* **Logic:** Using BTC weekend performance to trade the Monday market open.
* **Formula:** $ES_{Predicted} = -0.003884 + 0.120356 \cdot BTC_{Wknd}$.
* **Trade:** If ES opens lower than predicted, go Long. If higher, go Short.

### **17. Currency Strength Divergence (Basket Scalp)**

* **Logic:** Trading the strongest currency against the weakest.
* **Setup:** Calculate a real-time matrix of G8 currencies.
* **Entry:** Long the #1 Currency / Short the #8 Currency on an M1 pullback.

### **18. The Gold/Silver "Lag" Scalp**

* **Logic:** Highly correlated assets often move together, but one "leads."
* **Entry:** If Gold jumps 1% but Silver hasn't moved yet, buy Silver for a catch-up trade.
* **MT5 Detail:** Requires multi-symbol handle monitoring (`iCustom` or `SymbolSelect`).

---

## **VI. The "QuantMind" Logic Filter (Applying your Research)**

Every trade generated by the 18 strategies above must pass these **three "Gatekeepers"** before execution:

### **19. The Prop Score Validator**

$$Score = (Expected\_Return / Volatility) \cdot \sqrt{Drawdown / Target}$$

* **Rule:** If the `PropScore` of the strategy drops below a threshold (e.g., 0.6) during a rolling 20-trade window, the EA automatically pauses trading to avoid a "drawdown phase."

### **20. The Anti-Overfit (PBO) Check**

* **Logic:** Every week, run a Combinatorially Symmetric Cross-Validation (CSCV) on the strategy's performance.
* **Rule:** If the **Probability of Backtest Overfitting (PBO)** is $> 40\%$, reduce the lot size by 50%. The system is becoming "too lucky" and is due for a reversion.

---

## **VII. How to Implement & Adjust**

### **The "Management" SOP:**

1. **Selection:** Pick 3 strategies (e.g., #1 Absorption, #6 VWAP, #11 FVG).
2. **Calibration:** Run a 1-month backtest on MT5 to find the "Drift" and "Volatility" of the strategy.
3. **Lot Sizing:** Use the **Kelly Criterion** based on the backtest win rate ($p$) and odds ($b$):

$$f^* = (bp - q) / b$$


4. **Live Trial:** Run on a Cent/Demo account for 100 trades to verify the "Real-World" slippage and latency impact.

### **Potential Adjustments:**

* **High Volatility (News):** Switch to Category 2 (Volatility) strategies.
* **Low Volatility (Asian Session):** Switch to Category 1 (Order Flow/OFI) strategies.
* **Trending Markets:** Use Category 3 (EMA/Ichimoku) to "ride" the move.

**Final Note for QUANTMIND-X:** This modular approach allows you to swap strategies in and out like "Apps" on a phone, keeping your equity curve smooth while managing risk according to the prop firm math we established.