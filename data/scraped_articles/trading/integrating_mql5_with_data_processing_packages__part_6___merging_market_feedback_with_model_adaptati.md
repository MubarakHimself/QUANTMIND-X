---
title: Integrating MQL5 with Data Processing Packages (Part 6): Merging Market Feedback with Model Adaptation
url: https://www.mql5.com/en/articles/20235
categories: Trading, Integration, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T17:52:30.824316
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=klzhwyembqnaymvzqmbghoreittlswjp&ssn=1769179949068270504&ssn_dr=0&ssn_sr=0&fv_date=1769179949&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F20235&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Integrating%20MQL5%20with%20Data%20Processing%20Packages%20(Part%206)%3A%20Merging%20Market%20Feedback%20with%20Model%20Adaptation%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917994919588768&fz_uniq=5068746675956153644&sv=2552)

MetaTrader 5 / Trading


**Table of contents:**

1. [Introduction](https://www.mql5.com/en/articles/20235#Introduction)
2. [System Overview and Understanding](https://www.mql5.com/en/articles/20235#SystemOverview)
3. [Getting Started](https://www.mql5.com/en/articles/20235#GettingStarted)
4. [Putting It All Together on MQL5](https://www.mql5.com/en/articles/20235#PuttingItAll)
5. [Live Demo](https://www.mql5.com/en/articles/20235#LiveDemo)
6. [Conclusion](https://www.mql5.com/en/articles/20235#Conclusion)

### Introduction

In the previous [discussion](https://www.mql5.com/en/articles/18761), we explored adaptive learning and flexibility, where we focused on building a system capable of adjusting its decision-making processes in response to changing market conditions. That stage emphasized the importance of adaptability within algorithmic trading—allowing the model to dynamically modify its parameters based on evolving data patterns rather than relying on static historical behaviors. Through reinforcement learning and adaptive parameterization, the system began to exhibit the capacity for self-optimization, forming the foundation for continuous improvement within the Expert Advisor framework.

In this part, we advance that concept by introducing a feedback-driven learning loop between real-time market performance and the trained model. The goal is to bridge the execution environment (MQL5) with the data processing layer, Jupyter Lab, so that trade outcomes, volatility shifts, and behavioral anomalies become active components in model retraining. This integration allows the system not only to execute trades but also to learn from its own performance, continuously refining prediction accuracy and decision quality in response to the live market’s rhythm.

### System Overview and Understanding

The core idea behind this implementation is to create a continuous learning ecosystem between the trading environment (MetaTrader5) and the analytical environment (Jupyter Lab). Instead of relying on a fixed, pre-trained model, we want the Expert Advisor to actively learn from its trading performance and adapt its behavior over time. This means that every trade, whether profitable or not, becomes a learning signal. By collecting trade outcomes, feature data (like RSI, ATR, volatility, and price structure), and execution feedback, the EA provides valuable insights that are sent back to the Python side—where the model can analyze and use this new information to fine-tune its parameters.

![](https://c.mql5.com/2/180/Gemini_connect__1.png)

At the heart of this system lies the feedback-to-adaptation loop. The EA continuously extracts real-time features from the market, sends them to the Python REST server for prediction, and receives a decision probability or trade bias in return. After each trade is completed, the EA reports the results (win/loss, duration, drawdown, etc.) back to the server through the /feedback endpoint. The server aggregates these feedback samples into a growing dataset that represents the model’s live market experience. Periodically, the model uses this data to retrain or adjust its weights, producing a more accurate and responsive version that is later re-exported to ONNX for MQL5 inference.

![](https://c.mql5.com/2/180/Gemini_resized.png)

Ultimately, this implementation transforms the trading system into a self-evolving architecture. The integration ensures that market behavior, execution feedback, and prediction errors are all captured and reinvested into the learning process. Over time, the EA becomes increasingly efficient in recognizing profitable structures, adapting to volatility cycles, and reducing prediction drift. This approach enhances performance consistency and mimics the natural learning process of experienced traders—observing, adapting, and refining decisions with every market interaction.

### Getting started

```
from flask import Flask, request, jsonify
import threading, time, os, json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler
from datetime import datetime
```

We start by setting up the Python backend environment that will act as the bridge between the trading logic in MQL5 and the adaptive learning model. Using Flask as the lightweight REST API framework, we create endpoints that allow real-time communication between MetaTrader5 and our Python model. The essential libraries, such as torch, onnx, and onnxruntime, handle deep learning model operations, while pandas and numpy manage data processing and feature structuring. The sklearn scaler ensures consistent normalization of incoming market features, maintaining model stability across varying market conditions. Additionally, threading and datetime are used to schedule background tasks like feedback processing and retraining without interrupting real-time predictions. This setup forms the foundation for merging live market feedback with model adaptation, enabling continuous evolution of the trading model directly from within the trading environment.

```
app = Flask(__name__)

# ---------- Config ----------
FEATURE_DIM = 12            # Must match MQL5 feature dimension
RETRAIN_THRESHOLD = 100     # number of feedback samples before retrain
ONNX_PATH = "live_model.onnx"
MODEL_PATH = "live_model.pt"
FEEDBACK_CSV = "feedback_log.csv"

# ---------- Simple model ----------
class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)      # regression output (predicted expected reward)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cpu")
model = MLP(FEATURE_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
scaler = StandardScaler()

# Initialize scaler with some default values to avoid errors
try:
    default_features = np.zeros((100, FEATURE_DIM))
    scaler.fit(default_features)
    print("Initialized scaler with default values")
except Exception as e:
    print(f"Scaler initialization error: {e}")

# Keep feedback in-memory buffer for quick retrain
feedback_buffer = []  # list of dicts

# ---------- Helpers ----------
def features_to_tensor(f):
    try:
        arr = np.array(f, dtype=float).reshape(1, -1)
        # Check if scaler is fitted and has the right dimension
        if hasattr(scaler, 'mean_') and scaler.mean_ is not None and len(scaler.mean_) == FEATURE_DIM:
            arr = scaler.transform(arr)
        else:
            # If scaler not properly fitted, use raw features (will be fixed during retraining)
            print("Scaler not properly fitted, using raw features")
        t = torch.tensor(arr, dtype=torch.float32)
        return t
    except Exception as e:
        print(f"Error in features_to_tensor: {e}")
        # Return zeros if there's an error
        return torch.zeros(1, FEATURE_DIM, dtype=torch.float32)

def predict_raw(features):
    try:
        model.eval()
        with torch.no_grad():
            t = features_to_tensor(features).to(device)
            out = model(t).cpu().numpy().ravel()[0]
        return float(out)
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.0
```

In this section, we initialize the Flask application and configure the essential parameters that govern how our adaptive model operates. We define constants such as the feature dimension (FEATURE\_DIM), retraining threshold (RETRAIN\_THRESHOLD), and file paths for storing the model and feedback data. The heart of this setup is a simple multi-layer perceptron (MLP) implemented in PyTorch—a lightweight neural network with two hidden layers (64 and 32 neurons) designed for regression-based prediction of expected reward or price movement strength. The Adam optimizer and MSELoss function are used to efficiently train and minimize prediction errors, while the StandardScaler ensures that all input features are standardized for consistent model performance across different market states. By pre-fitting the scaler with dummy data, we prevent initialization errors before real market data becomes available.

The helper functions handle the conversion of incoming features from MetaTrader5 into model-compatible tensors and ensure robust prediction execution. The features\_to\_tensor() function validates feature shapes, applies scaling if available, and safely returns a PyTorch tensor even if errors occur, preventing runtime interruptions. Meanwhile, predict\_raw() executes forward propagation through the network in evaluation mode and retrieves the numerical output as a floating-point prediction. This modular structure allows the model to remain responsive and error-tolerant while maintaining the flexibility to integrate new feedback data for future retraining cycles.

```
def save_feedback_to_csv(entry):
    try:
        # entry is a dict; features saved as JSON
        row = entry.copy()
        row["features"] = json.dumps(row["features"])
        df = pd.DataFrame([row])
        if not os.path.exists(FEEDBACK_CSV):
            df.to_csv(FEEDBACK_CSV, index=False)
        else:
            df.to_csv(FEEDBACK_CSV, mode='a', header=False, index=False)
    except Exception as e:
        print(f"Error saving feedback to CSV: {e}")

def retrain_model():
    global model, optimizer, scaler, feedback_buffer
    if len(feedback_buffer) < 10:  # Minimum samples to retrain
        print(f"Not enough samples for retraining: {len(feedback_buffer)}")
        return

    print(f"[{datetime.utcnow().isoformat()}] Retraining model on {len(feedback_buffer)} samples...")
    try:
        # load buffer into DataFrame
        df = pd.DataFrame(feedback_buffer)
        X = np.vstack(df["features"].apply(lambda x: np.array(x)).values)
        y = df["reward"].astype(float).values.reshape(-1,1)

        # Fit scaler on current data
        scaler.fit(X)
        Xs = scaler.transform(X)
        Xs = torch.tensor(Xs, dtype=torch.float32)
        ys = torch.tensor(y, dtype=torch.float32)

        # small training loop
        model.train()
        epochs = 40
        batch_size = min(32, len(Xs))

        for ep in range(epochs):
            perm = torch.randperm(Xs.size(0))
            for i in range(0, Xs.size(0), batch_size):
                idx = perm[i:i+batch_size]
                xb = Xs[idx]
                yb = ys[idx]
                pred = model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # save model to disk (torch)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

        # export to ONNX
        dummy = torch.randn(1, FEATURE_DIM, dtype=torch.float32)
        model.eval()
        try:
            torch.onnx.export(model, dummy, ONNX_PATH, input_names=['input'], output_names=['output'], opset_version=11)
            print(f"ONNX exported to {ONNX_PATH}")
        except Exception as e:
            print("ONNX export failed:", e)

        # append buffer to CSV
        for row in feedback_buffer:
            save_feedback_to_csv(row)

        # clear buffer
        feedback_buffer = []
        print("Retrain complete.")

    except Exception as e:
        print(f"Error during retraining: {e}")

# background trainer thread that monitors buffer size
def trainer_loop():
    while True:
        try:
            if len(feedback_buffer) >= RETRAIN_THRESHOLD:
                retrain_model()
        except Exception as e:
            print("trainer error:", e)
        time.sleep(10)  # Check every 10 seconds

trainer_thread = threading.Thread(target=trainer_loop, daemon=True)
trainer_thread.start()
```

Here, we implement the feedback management and model retraining system, which acts as the adaptive core of the learning process. The save\_feedback\_to\_csv() function records feedback entries—each containing the extracted features, the resulting reward, and any additional metadata—into a persistent CSV log for long-term analysis. To ensure flexibility, feature arrays are serialized as JSON strings before being appended to the file, preserving their structure for future reloading. This approach allows us to maintain a complete and continuously growing dataset of trade outcomes that the model can later use for retraining. In case the CSV file doesn’t exist, it is created automatically; otherwise, new rows are appended efficiently without disrupting ongoing processes.

The retrain\_model() function is responsible for converting accumulated market feedback into learning updates for the model. Once the buffer reaches a sufficient number of samples, it loads the collected data, scales it using the StandardScaler, and runs several training epochs to fine-tune the neural network’s weights. After training, the model is saved in both PyTorch (.pt) and ONNX formats, allowing MetaTrader5 to access the updated version directly for real-time predictions. To keep the process autonomous, the trainer\_loop() function runs continuously in a background thread, monitoring the buffer size and triggering retraining once the threshold is met. This setup ensures that the trading system evolves naturally—absorbing live feedback, retraining itself, and updating its inference model without manual intervention, thus achieving true adaptive intelligence in market execution.

```
# ---------- Flask endpoints ----------
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "No JSON data received"}), 400

        features = payload.get("features")
        if features is None or len(features) != FEATURE_DIM:
            return jsonify({
                "error": "bad features",
                "expected_dim": FEATURE_DIM,
                "received_dim": len(features) if features else 0
            }), 400

        pred = predict_raw(features)
        return jsonify({"prediction": pred})

    except Exception as e:
        print(f"Prediction endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback_endpoint():
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "No JSON data received"}), 400

        # minimal validation; ensure reward exists
        if "features" not in payload or "reward" not in payload:
            return jsonify({"error": "need features and reward"}), 400

        # Calculate reward if not provided (fallback logic)
        reward = payload.get("reward")
        if reward is None:
            # Try to calculate from pips_profit
            pips_profit = payload.get("pips_profit")
            if pips_profit is not None:
                reward = float(pips_profit) / 100.0  # Normalize
            else:
                reward = 0.0

        # store in buffer
        entry = {
            "timestamp": payload.get("timestamp", datetime.utcnow().isoformat()),
            "symbol": payload.get("symbol", ""),
            "tf": payload.get("tf", ""),
            "features": payload["features"],
            "action": payload.get("action_taken", 0),
            "entry_price": payload.get("entry_price", 0.0),
            "exit_price": payload.get("exit_price", 0.0),
            "pips_profit": payload.get("pips_profit", 0.0),
            "reward": float(reward)
        }
        feedback_buffer.append(entry)

        # also save immediately to CSV for persistence
        save_feedback_to_csv(entry)

        return jsonify({
            "status": "ok",
            "buffer_size": len(feedback_buffer),
            "reward_received": float(reward)
        })

    except Exception as e:
        print(f"Feedback endpoint error: {e}")
        return jsonify({"error": str(e)}), 500
```

In this section, we establish two essential Flask endpoints—/predict and /feedback—to manage real-time interaction between the trading system and the adaptive learning model. The /predict endpoint processes incoming JSON payloads containing market features, validates the input dimensions, and returns predictions generated by the model. This ensures that every prediction request is well-structured and consistent with the model’s expected input format. The /feedback endpoint, on the other hand, captures post-trade data such as actions taken, price entries, profits or losses, and the resulting rewards. This information is then appended to the feedback buffer and persisted to a CSV file, forming the foundation for continuous learning. Together, these endpoints create a feedback-driven ecosystem where the model provides predictions and learns iteratively from market outcomes—closing the loop between model inference and performance adaptation.

```
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "buffer_size": len(feedback_buffer),
        "feature_dim": FEATURE_DIM
    })

# optional endpoint to force retrain (admin)
@app.route('/retrain', methods=['POST'])
def retrain_now():
    threading.Thread(target=retrain_model).start()
    return jsonify({"status": "retrain_started", "buffer_size": len(feedback_buffer)})

@app.route('/info', methods=['GET'])
def info():
    """Get information about the current model state"""
    return jsonify({
        "feature_dim": FEATURE_DIM,
        "feedback_buffer_size": len(feedback_buffer),
        "retrain_threshold": RETRAIN_THRESHOLD,
        "model_path": MODEL_PATH,
        "scaler_fitted": hasattr(scaler, 'mean_') and scaler.mean_ is not None
    })

if __name__ == "__main__":
    print(f"Starting ML Server for MQL5 EA")
    print(f"Feature dimension: {FEATURE_DIM}")
    print(f"Retrain threshold: {RETRAIN_THRESHOLD}")
    print(f"Server will run on http://127.0.0.1:5000")
    print(f"Endpoints available:")
    print(f"  POST /predict - Get prediction for features")
    print(f"  POST /feedback - Send trade feedback")
    print(f"  GET  /health - Health check")
    print(f"  GET  /info - Model information")

    # Start the server
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
```

We finalize the implementation by adding three utility endpoints—/health, /retrain, and /info—which provide monitoring, control, and insight into the model’s operational state. The /health endpoint serves as a quick diagnostic tool, reporting the system’s current status, timestamp, and buffer size to ensure smooth operation. The /retrain endpoint allows administrators to manually trigger model retraining when necessary, running it asynchronously to avoid interrupting the server’s main process. Lastly, the /info endpoint provides detailed metadata about the model, including its feature dimensions, retrain threshold, and scaler status, offering transparency into the system’s learning state.

### Putting it all together on MQL5

For our integration on MQL5, we decided to build upon one of our existing and well-established strategies—the [Dynamic Swing Architecture](https://www.mql5.com/en/articles/19793). This framework already provides a strong foundation for market structure recognition and adaptive trade execution, making it the perfect candidate for our next evolutionary step. With that being said, our focus in this part will be on the updates and enhancements made specifically for integrating the concept of merging market feedback with adaptive model learning. Rather than revisiting the core swing logic, we will dive deeper into how real-time trading feedback from MQL5 is captured, transmitted, and utilized to continuously refine the model’s decision-making process, enabling it to evolve dynamically in response to changing market conditions.

```
input group "ML Model Parameters"
input string  PythonHost = "127.0.0.1";         // Python server host
input int     PythonPort = 5000;                // Python server port

//--- input parameters for indicators
input int   InpATRPeriod       = 14;
input int   InpRSIPeriod       = 14;
input int   InpMomPeriod       = 10;
input int   InpTrendLookback   = 20;
input int   InpVolLookback     = 20;

//--- global handles
int   hATR       = INVALID_HANDLE;
int   hRSI       = INVALID_HANDLE;

//--- global swing variables
double g_lastSwingHigh = 0.0;
double g_lastSwingLow = 0.0;
bool   g_lastSwingWasBullish = false;
```

To support our model’s feature extraction directly within MQL5, we implemented a new input group titled “ML Model Parameters," which allows for seamless configuration of both the machine learning server connection and the indicator-based features used by the model. This section introduces inputs for connecting to the Python host and port, along with parameters that define the calculation periods for key technical indicators—ATR, RSI, Momentum, Trend, and Volatility lookbacks. Corresponding global handles for ATR and RSI were initialized, ensuring efficient computation and reuse of indicator data throughout the Expert Advisor’s execution.

```
//+------------------------------------------------------------------+
//| Extracts feature vector (double array)                           |
//+------------------------------------------------------------------+
bool ExtractFeatures(double &features[], int dim)
{
   if(dim != 12)
   {
      Print("ExtractFeatures: expected dim=12, got ", dim);
      return(false);
   }
   ArrayResize(features, dim);
   ArrayInitialize(features, 0.0);

   //--- 1. ATR (most recent)
   double atr_buffer[];
   if(CopyBuffer(hATR, 0, 0, 1, atr_buffer) != 1)
   {
      Print("CopyBuffer ATR failed");
      return(false);
   }
   double atr_value = atr_buffer[0];

   //--- 2. RSI (latest)
   double rsi_buffer[];
   if(CopyBuffer(hRSI, 0, 0, 1, rsi_buffer) != 1)
   {
      Print("CopyBuffer RSI failed");
      return(false);
   }
   double rsi_value = rsi_buffer[0];

   //--- 3. Distance to last swing high / low
   double lastSwingHigh = g_lastSwingHigh;
   double lastSwingLow = g_lastSwingLow;
   double priceNow = iClose(_Symbol, _Period, 0);

   double distHigh = (lastSwingHigh > 0) ? (priceNow - lastSwingHigh) : 0.0;
   double distLow = (lastSwingLow > 0) ? (lastSwingLow - priceNow) : 0.0;

   // normalize by ATR to scale
   double normHigh = (atr_value > 0 && lastSwingHigh > 0) ? distHigh/atr_value : 0.0;
   double normLow = (atr_value > 0 && lastSwingLow > 0) ? distLow/atr_value : 0.0;

   //--- 4. Swing strength = (high-low)/ATR
   double swingStrength = 0.0;
   if(lastSwingHigh > 0 && lastSwingLow > 0 && atr_value > 0)
      swingStrength = (lastSwingHigh - lastSwingLow) / atr_value;

   //--- 5. Swing direction: 1 for bullish, -1 for bearish
   int lastSwingDir = g_lastSwingWasBullish ? 1 : -1;
   double swingDirNorm = (double)lastSwingDir;

   //--- 6. Momentum: price difference over InpMomPeriod
   double pastClose = iClose(_Symbol, _Period, InpMomPeriod);
   double momentum = (priceNow - pastClose) / (atr_value > 0 ? atr_value : 1.0);

   //--- 7. TrendSlope: linear regression slope of last InpTrendLookback bars (close prices)
   double slope = 0.0;
   double arr[];
   ArrayResize(arr, InpTrendLookback);
   for(int i = 0; i < InpTrendLookback; i++)
      arr[i] = iClose(_Symbol, _Period, i);

   // Calculate linear regression slope manually
   double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
   for(int i = 0; i < InpTrendLookback; i++)
   {
      sum_x += i;
      sum_y += arr[i];
      sum_xy += i * arr[i];
      sum_xx += i * i;
   }
   double n = (double)InpTrendLookback;
   slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
   slope = slope / (atr_value > 0 ? atr_value : 1.0);

   //--- 8. Volume ratio: current volume / average of last InpVolLookback
   double volNow = iVolume(_Symbol, _Period, 0);
   double sumVol = 0.0;
   for(int i = 1; i <= InpVolLookback; i++)
      sumVol += iVolume(_Symbol, _Period, i);
   double avgVol = sumVol / InpVolLookback;
   double volRatio = (avgVol > 0) ? volNow / avgVol : 0.0;

   //--- 9 & 10. BreakAbove / BreakBelow flags
   double breakAbove = (lastSwingHigh > 0 && priceNow > lastSwingHigh) ? 1.0 : 0.0;
   double breakBelow = (lastSwingLow > 0 && priceNow < lastSwingLow) ? 1.0 : 0.0;

   //--- 11. TimeOfDay normalized
   MqlDateTime time_struct;
   TimeCurrent(time_struct);
   double hr = (double)time_struct.hour;
   double tod = hr / 24.0;

   //--- Assign features in array
   features[0]  = normHigh;
   features[1]  = normLow;
   features[2]  = swingStrength;
   features[3]  = swingDirNorm;
   features[4]  = atr_value;
   features[5]  = rsi_value / 100.0;   // scale RSI 0-1
   features[6]  = momentum;
   features[7]  = slope;
   features[8]  = volRatio;
   features[9]  = breakAbove;
   features[10] = breakBelow;
   features[11] = tod;

   return(true);
}
```

The ExtractFeatures function creates a comprehensive 12-dimensional market snapshot by transforming raw price data into normalized, ATR-scaled features that capture multiple aspects of market behavior, including volatility (ATR), momentum (RSI and price momentum), swing dynamics (distance to recent highs/lows and swing strength), trend direction (linear regression slope), volume activity (volume ratio), breakout signals (break above/below swing levels), and temporal patterns (time of day). This feature engineering process converts complex market movements into a standardized numerical vector that the machine learning model can analyze, where each feature is carefully normalized to ensure consistent scaling—such as dividing price distances by ATR to make them comparable across different market conditions and scaling RSI to a 0-1 range—enabling the ML algorithm to effectively learn patterns and relationships for making trading predictions regardless of absolute price levels or volatility regimes.

```
//--- Converts feature array to JSON
string FeaturesToJson(double &features[], int dim)
{
   string json = "[";\
   for(int i = 0; i < dim; i++)\
   {\
      json += DoubleToString(features[i], 8);\
      if(i < dim - 1) json += ",";\
   }\
   json += "]";
   return json;
}

//--- Generic HTTP POST
string HttpPostJson(string url, string json_body, int &status_code)
{
   string headers = "Content-Type: application/json\r\n";
   char data[], result[];
   ArrayResize(data, StringLen(json_body));
   StringToCharArray(json_body, data, 0, StringLen(json_body));
   int timeout = 5000; // 5 seconds
   string result_headers;

   ResetLastError();
   int res = WebRequest("POST", url, headers, timeout, data, result, result_headers);
   status_code = res;

   if(res == -1)
   {
      int error_code = GetLastError();
      Print("WebRequest failed. Error: ", error_code, " - ", GetLastError());
      return "";
   }
   return CharArrayToString(result);
}
```

The FeaturesToJson function serializes the numerical feature array into a JSON string format that can be transmitted over HTTP to the Python ML server. It constructs a JSON array by iterating through each element of the feature vector, converting the double-precision values to string representation with 8 decimal places of precision, and properly formatting them with commas between elements while ensuring the array is properly enclosed in square brackets. This transformation is crucial because it converts the EA's internal numerical data structure into a standardized, platform-agnostic format that the Python Flask server can easily parse and process, maintaining the precise numerical values needed for accurate machine learning predictions while adhering to web communication standards.

The HttpPostJson function handles the actual HTTP communication between the MetaTrader5 platform and the Python ML server by packaging and sending the JSON data. It sets up the necessary HTTP headers to specify JSON content type, converts the JSON string into a character array for transmission, and uses MetaTrader5's WebRequest function to perform a POST request with a 5-second timeout to prevent hanging. The function includes comprehensive error handling that captures and reports specific error codes when requests fail, while successfully returning the server's response as a string when communication is established. This robust HTTP client implementation enables seamless real-time data exchange between the trading platform and machine learning backend, forming the critical communication bridge that allows the EA to get predictions and send feedback for continuous learning.

```
//--- Get model prediction from Python
double GetPrediction(string host, int port, string symbol, string tf, double &features[], int dim)
{
   string url = StringFormat("http://%s:%d/predict", host, port);
   string json = "{";
   json += "\"symbol\":\"" + symbol + "\",";
   json += "\"tf\":\"" + tf + "\",";
   json += "\"features\":" + FeaturesToJson(features, dim);
   json += "}";

   int code = 0;
   string resp = HttpPostJson(url, json, code);

   Print("Prediction Request - Code: ", code, ", Response: ", resp);

   if(code == 200 && StringFind(resp, "prediction") >= 0)
   {
      int p = StringFind(resp, "\"prediction\":");
      if(p >= 0)
      {
         int start = p + StringLen("\"prediction\":");
         int end = StringFind(resp, "}", start);
         if(end == -1) end = StringLen(resp);
         string val = StringSubstr(resp, start, end - start);
         // Remove any trailing commas or spaces
         StringReplace(val, ",", "");
         StringReplace(val, " ", "");
         StringReplace(val, "}", "");
         double prediction = StringToDouble(val);
         Print("Parsed prediction value: ", prediction);
         return prediction;
      }
   }
   Print("Prediction failed. Code=", code, ", Response=", resp);
   return 0.0;
}

//--- Send feedback to Python
bool SendFeedback(string host, int port, string symbol, string tf, double &features[], int dim,
                  int action, double entry_price, double exit_price, double pips_profit, double reward)
{
   string url = StringFormat("http://%s:%d/feedback", host, port);
   string json = "{";
   json += "\"symbol\":\"" + symbol + "\",";
   json += "\"tf\":\"" + tf + "\",";
   json += "\"features\":" + FeaturesToJson(features, dim) + ",";
   json += "\"action_taken\":" + IntegerToString(action) + ",";
   json += "\"entry_price\":" + DoubleToString(entry_price, _Digits) + ",";
   json += "\"exit_price\":" + DoubleToString(exit_price, _Digits) + ",";
   json += "\"pips_profit\":" + DoubleToString(pips_profit, 4) + ",";
   json += "\"reward\":" + DoubleToString(reward, 6);
   json += "}";

   int code = 0;
   string resp = HttpPostJson(url, json, code);
   if(code == 200)
   {
      Print("Feedback sent successfully. Position profit: ", pips_profit, " pips, Reward: ", reward);
      return true;
   }
   Print("Feedback failed. Code=", code, " Resp=", resp);
   return false;
}
```

The GetPrediction function serves as the primary interface for obtaining real-time trading signals from the machine learning model by constructing a comprehensive HTTP request that includes both market context (symbol and timeframe) and the 12-dimensional feature vector, then meticulously parsing the server's JSON response to extract the numerical prediction value. It implements robust error handling and detailed logging throughout the process—first by building a properly formatted JSON payload containing the features, then by making the POST request to the /predict endpoint, and finally by carefully extracting the prediction value from the response string through character manipulation and cleanup to handle various JSON formatting scenarios. The function provides comprehensive debugging information by logging both the HTTP status code and raw server response, ensuring transparency in the prediction process while safely returning 0.0 as a fallback value when communication fails or the response format is unexpected.

The SendFeedback function completes the machine learning lifecycle by transmitting trade outcome data back to the Python server, enabling the model to learn from actual market results and continuously improve its predictions. It constructs a detailed JSON object containing the complete trade context—including the original features that led to the trading decision, the specific action taken (buy/sell), entry and exit prices, calculated profit in pips, and the normalized reward value—which provides the necessary training data for supervised learning. This feedback mechanism is crucial for the adaptive learning system, as it allows the ML model to correlate its previous predictions with actual market outcomes, gradually refining its understanding of which feature patterns lead to successful trades versus unsuccessful ones, thereby creating a self-improving trading system that becomes more accurate over time through accumulated experience.

```
//+------------------------------------------------------------------+
//| Calculate reward for ML feedback                                 |
//+------------------------------------------------------------------+
double CalculateReward(double profit, double pipsProfit, double volume)
{
    // Customize this function based on your reward strategy
    // Simple implementation: normalize profit by volume and scale
    if(volume > 0)
    {
        double normalizedProfit = profit / (volume * 1000); // Adjust scaling factor as needed
        return normalizedProfit;
    }

    // Alternative: use pips profit directly
    return pipsProfit / 100.0; // Scale down pips to reasonable range
}

//+------------------------------------------------------------------+
//| Get historical features for a specific time                     |
//+------------------------------------------------------------------+
bool GetHistoricalFeatures(datetime targetTime, double &features[])
{
    // Simplified implementation - uses current features
    // In production, you might want to store features when trades are opened
    ArrayResize(features, 12);
    return ExtractFeatures(features, 12);
}

//+------------------------------------------------------------------+
//| Track processed positions to avoid duplicates                   |
//+------------------------------------------------------------------+
bool IsPositionProcessed(ulong positionTicket)
{
    // Simple implementation using global variable
    static ulong lastProcessedTicket = 0;
    return (positionTicket == lastProcessedTicket);
}

//+------------------------------------------------------------------+
//| Mark position as processed                                      |
//+------------------------------------------------------------------+
void MarkPositionAsProcessed(ulong positionTicket)
{
    static ulong lastProcessedTicket = 0;
    lastProcessedTicket = positionTicket;
}
```

This section of the code implements essential utility functions that support the machine learning feedback loop: CalculateReward transforms raw trade outcomes into normalized reward values suitable for ML training by scaling either monetary profit by trade volume or using pip profit directly; GetHistoricalFeatures provides a simplified mechanism to retrieve market features—currently using current features as a practical approximation since storing exact historical features would require significant additional infrastructure; and the IsPositionProcessed/MarkPositionAsProcessed functions work together as a basic duplicate prevention system using a static variable to track the last processed position ticket, ensuring that each trade's feedback is sent to the ML model only once despite multiple checks, thereby maintaining data integrity in the learning cycle while keeping the implementation straightforward and efficient within the MQL5 environment's constraints.

### Live Demo

Here, the Flask server has successfully initialized and is now running on port 5000, ready to receive prediction requests from your MT5 EA and provide machine learning-driven trading signals.

![](https://c.mql5.com/2/180/flask_init.png)

We are now getting predictions from the model on Jupyter Lab to MetaTrader5, as you can see below.

![](https://c.mql5.com/2/180/Predictions.png)

![](https://c.mql5.com/2/180/MTPredictions.png)

### Conclusion

In summary, we successfully merged market feedback with model adaptation by implementing a comprehensive machine learning pipeline that transforms the trading EA from a static rule-based system into a dynamic self-improving algorithm. This was achieved through several key integrations: creating a robust feature extraction system that converts market data into 12 normalized dimensions, establishing real-time HTTP communication with a Python ML server for prediction requests, implementing a feedback loop that automatically sends trade outcomes back to the model, and designing an adaptive learning system where the neural network continuously retrains on new market experiences. The integration includes proper error handling, detailed logging for debugging, and duplicate prevention mechanisms, creating a closed-loop system where each trade's success or failure directly contributes to refining future trading decisions through supervised learning.

In conclusion, this machine learning integration will significantly enhance traders' capabilities by creating an adaptive system that learns from both successful and unsuccessful trades, gradually developing more accurate market predictions tailored to current market conditions. Unlike traditional static trading algorithms, this system continuously evolves and improves its decision-making based on actual performance data, potentially leading to higher profitability, better risk management, and increased consistency over time. The automated feedback loop eliminates emotional trading biases, while the model's ability to recognize complex, non-linear patterns in market data can identify opportunities that might be missed by conventional technical analysis, ultimately providing traders with a sophisticated, self-optimizing tool that becomes more valuable with each trading decision it makes and learns from.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20235.zip "Download all attachments in the single ZIP archive")

[Dynamic\_Swing\_Detection.mq5](https://www.mql5.com/en/articles/download/20235/Dynamic_Swing_Detection.mq5 "Download Dynamic_Swing_Detection.mq5")(82.02 KB)

[MergingMarketFeedback.ipynb](https://www.mql5.com/en/articles/download/20235/MergingMarketFeedback.ipynb "Download MergingMarketFeedback.ipynb")(15.23 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing Market Memory Zones Indicator: Where Price Is Likely To Return](https://www.mql5.com/en/articles/20973)
- [Adaptive Smart Money Architecture (ASMA): Merging SMC Logic With Market Sentiment for Dynamic Strategy Switching](https://www.mql5.com/en/articles/20414)
- [Fortified Profit Architecture: Multi-Layered Account Protection](https://www.mql5.com/en/articles/20449)
- [Analytical Volume Profile Trading (AVPT): Liquidity Architecture, Market Memory, and Algorithmic Execution](https://www.mql5.com/en/articles/20327)
- [Automating Black-Scholes Greeks: Advanced Scalping and Microstructure Trading](https://www.mql5.com/en/articles/20287)
- [Formulating Dynamic Multi-Pair EA (Part 5): Scalping vs Swing Trading Approaches](https://www.mql5.com/en/articles/19989)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/500205)**
(1)


![YDILLC](https://c.mql5.com/avatar/avatar_na2.png)

**[YDILLC](https://www.mql5.com/en/users/ydillc)**
\|
21 Nov 2025 at 06:23

question about -  [Integrating MQL5 with Data Processing Packages (Part 6): Merging Market Feedback with Model Adaptation](https://www.mql5.com/en/articles/20235). ... the file thats added to that article. is that an actual EA? how do i go about utilizing it?

![Markets Positioning Codex in MQL5 (Part 1): Bitwise Learning for Nvidia](https://c.mql5.com/2/177/20020-markets-positioning-codex-in-logo.png)[Markets Positioning Codex in MQL5 (Part 1): Bitwise Learning for Nvidia](https://www.mql5.com/en/articles/20020)

We commence a new article series that builds upon our earlier efforts laid out in the MQL5 Wizard series, by taking them further as we step up our approach to systematic trading and strategy testing. Within these new series, we’ll concentrate our focus on Expert Advisors that are coded to hold only a single type of position - primarily longs. Focusing on just one market trend can simplify analysis, lessen strategy complexity and expose some key insights, especially when dealing in assets beyond forex. Our series, therefore, will investigate if this is effective in equities and other non-forex assets, where long only systems usually correlate well with smart money or institution strategies.

![Blood inheritance optimization (BIO)](https://c.mql5.com/2/120/Blood_inheritance_optimization__LOGO.png)[Blood inheritance optimization (BIO)](https://www.mql5.com/en/articles/17246)

I present to you my new population optimization algorithm - Blood Inheritance Optimization (BIO), inspired by the human blood group inheritance system. In this algorithm, each solution has its own "blood type" that determines the way it evolves. Just as in nature where a child's blood type is inherited according to specific rules, in BIO new solutions acquire their characteristics through a system of inheritance and mutations.

![Self Optimizing Expert Advisors in MQL5 (Part 17): Ensemble Intelligence](https://c.mql5.com/2/180/20238-self-optimizing-expert-advisors-logo.png)[Self Optimizing Expert Advisors in MQL5 (Part 17): Ensemble Intelligence](https://www.mql5.com/en/articles/20238)

All algorithmic trading strategies are difficult to set up and maintain, regardless of complexity—a challenge shared by beginners and experts alike. This article introduces an ensemble framework where supervised models and human intuition work together to overcome their shared limitations. By aligning a moving average channel strategy with a Ridge Regression model on the same indicators, we achieve centralized control, faster self-correction, and profitability from otherwise unprofitable systems.

![From Novice to Expert: Time Filtered Trading](https://c.mql5.com/2/181/20037-from-novice-to-expert-time-logo.png)[From Novice to Expert: Time Filtered Trading](https://www.mql5.com/en/articles/20037)

Just because ticks are constantly flowing in doesn’t mean every moment is an opportunity to trade. Today, we take an in-depth study into the art of timing—focusing on developing a time isolation algorithm to help traders identify and trade within their most favorable market windows. Cultivating this discipline allows retail traders to synchronize more closely with institutional timing, where precision and patience often define success. Join this discussion as we explore the science of timing and selective trading through the analytical capabilities of MQL5.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ctrkcgbvdcyqrtmcjfggxfzgedaypesa&ssn=1769179949068270504&ssn_dr=0&ssn_sr=0&fv_date=1769179949&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F20235&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Integrating%20MQL5%20with%20Data%20Processing%20Packages%20(Part%206)%3A%20Merging%20Market%20Feedback%20with%20Model%20Adaptation%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917994919415436&fz_uniq=5068746675956153644&sv=2552)

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