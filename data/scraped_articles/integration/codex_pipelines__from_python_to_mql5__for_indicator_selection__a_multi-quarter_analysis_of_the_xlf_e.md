---
title: Codex Pipelines, from Python to MQL5, for Indicator Selection: A Multi-Quarter Analysis of the XLF ETF with Machine Learning
url: https://www.mql5.com/en/articles/20595
categories: Integration, Indicators, Expert Advisors, Machine Learning
relevance_score: 9
scraped_at: 2026-01-22T17:40:12.554775
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/20595&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049291509192304842)

MetaTrader 5 / Integration


### Introduction

In our last study on constructing a disciplined framework for ranking indicators across quarterly regimes, we used the FXI ETF as our tradable asset. Our goal was to produce a clean, modular pipeline that filters noise and places some structure in indicator selection, for cases where trading intuition is bound to fall apart. The pipeline we produced, however, was diagnostic. It informed us of which indicators mattered and when they mattered. Even though we demonstrated how the indicator pool could be used together with weighting that was proportional to the ranking scores, our final Expert Advisor did leave some holes when it came to better generalization - since arguably the used indicator weighting amounts to a single perceptron layer that could be too simplistic. We attempt to address with a multi layer perceptron.

For this article, we still use the indicator states, not the pattern classifications, but the same forward return horizons that we looked at last, and see if we can transform this information into a predictive engine. We seek to output a compact neural network that is trained on multi-quarter indicator observations. We then export this model as an ONNX for direct use within MQL5. The idea here is simple, if our pipeline already knows how to sift signals from goo, then the model that we train down stream will not have to ‘re-learn’ what we built in the last article. Instead, it would act like a specialized inference layer that gives us probability signals for the tendencies of our test asset, the XLF, without having an Expert Advisor for every unique market regime.

Our target result therefore is a production cycle spanning: data → pipeline → features → model → ONNX → Expert Advisor.

### The XLF

The XLF ETF is a sector ETF by SPDR that represents the financial sector. Unlike the FXI we tested with the last time, it has had a predominant bullish trend since inception. Except for, the GFC period, this ETF has almost always rallied, and it thus gives us a contrasting testing ground to what we had the last time, where the FXI was languishing like a forex pair, about its 40 price handle for many years. Nonetheless, it does exhibit sharp regime transitions that are arguably driven by interest-rate policy, credit cycles, some liquidity shocks, as well as sector-specific rotation.

![i1](https://c.mql5.com/2/185/i1__1.png)

Its behavior over many quarters appears to alternate between smooth surges upwards and abrupt volatility spikes. By training and evaluating models on the XLF, we make the case for our pipeline, which is that indicators should remain consistent over different market sub-environments and that feature engineering should withstand noisy transitions. The ONNX model, our end product here, needs to survive real-time execution in MQL5 without relying on some forgiving conditions in benign markets.

### Pipeline Amendments

This indicator pipeline that we coded in the last article has already done a lot of the heavy lifting by handling things like cleansing data; segmenting quarters; computing indicator values; and scoring their usefulness. What it may not be good at, arguably, is generalizing or putting together what was learnt in a limited time window and show similar performance on unseen market regimes/test windows. That is the core argument for this expansion or alternative pipeline approach.

Our objective here, therefore, is to treat the pipeline’s outputs which are normalized indicator readings and states got from patterns, as structured features for a predictive network. This network will not replace the pipeline but will inherit its discipline and build a statistical layer on top of it. Our approach also avoids the raw price to network model approach that has a tendency of overfitting when training.

Our final framework therefore becomes a multi-stage assembly line where indicator logic gives us features, the forward return calculations give us the labels/ targets for our training and the multi-layer perceptron gets to learn this relationship. ONNX simply acts as a conduit for transferring the model from python to MQL5. This flow is designed to keep all the components independent such that specific changes, whether to indicators, pattern types or time horizons, do not necessitate ‘surgery’ on the model’s code. Our framework could be viewed as follows.

[![i7](https://c.mql5.com/2/185/i7.png)](https://c.mql5.com/2/185/i7.png "https://c.mql5.com/2/185/i7.png")

Also, after training, we will do a post validation run to get a sense of how our model is likely to perform. This will be with the most recent 20 percent dataset. We stop the validation process once the test results plateau, at which point we export the model as an ONNX for MQL5.

### The Feature Engineering

The FXI article that I wrote last, built its dataset from the common indicator trio that married trend/momentum/volatility metrics. We used for these, the MACD, RSI, and Bollinger Bands indicators. In order to test out our pipeline more or ‘push its boundaries’ we will consider an alternative set of indicators for this article in the form of FrAMA, Parabolic SAR, the Alligator, and TRIX. These are selected not only for their novelty, which is important when exploring, but also for what each represents.

The FrAMA is good at spotting trend accelerations, the SAR as its name implies is good at identifying reversals, the Alligator is very useful when it comes to detecting price-MA convergence, and finally the TRIX can help with price’s sensitivity to momentum-change. As with the last article, it can be easy for one to embrace all of them with the argument that they are all specialists, however our pipeline’s primary goal is to establish which among them is more resourceful to the XLF. Bringing them together though, with their importance weights, could work well with a multi layer perceptron since they tend to extract cross-indicator relationships better than our raw rank weighted pipeline.

The pipeline’s job will remain the same, compute these indicator rank importance weights, bar-by-bar over the quarterly segments, then feed the resulting numerical states into a unified training matrix. However, unlike the setup we had in the last article with FXI, our indicators here are introducing multi-column outputs such as the Jaw/Teeth/Lips of the Alligator as well as recursive components like with FrAMA’s smoothing of the SAR flips. All these features need to be aligned bar-by-bar or in time, and treated as a proper feature source. We are not smoothing over indicators to make them uniform, and we are also on the lookout for leaking future values into earlier rows. We therefore implement our feature engineering in preparation for using our MLP as follows. The source code in loading price data from MetaTrader 5 is not repeated here, as this was covered above.

```
# python libraries
import MetaTrader5 as mt5
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import pandas as pd
import numpy as np

from datetime import datetime, time
from pandas.tseries.holiday import USFederalHolidayCalendar

from IndicatorsAll import FRAMA, Parabolic_SAR, Alligator, TRIX  # :contentReference[oaicite:1]{index=1}

# …

# same mt5 price load and resampling as last article

# …

# Apply mask
df_market = df_resampled[market_mask].copy()

# Compute the alternative indicators
df = FRAMA(df_market, period=16)
df = Parabolic_SAR(df)
df = Alligator(df)
df = TRIX(df, period=15, signal=9)

# Build feature set
feature_cols = [\
    "FRAMA",\
    "SAR",\
    "Alligator_Jaw", "Alligator_Teeth", "Alligator_Lips",\
    "TRIX", "TRIX_signal"\
]

# Drop any rows with NaNs caused by indicator warm-up periods
X = df[feature_cols].dropna()

print("Feature matrix shape:", X.shape)
```

The feature matrix we receive as our output above is unconventional, deliberately. It brings together adaptive smoothing, fractal dimension behavior, multi-line trend alignment, and triple smoothed momentum rates. With this, the training portion of our pipeline inherits a cleaner, more expressive foundation of input data, one that arguably reflects the various dynamics of the markets.

### The Label Engineering

For almost all indicators, irrespective of how exotic they may be, they cannot be of much use to traders unless they can point toward a meaningful future outcome. Label engineering therefore exists to ensure this. With it, we convert raw price movement into a clean, learnable target that the ONNX model can later try to forecast. Also, because our indicator-set is now spread widely into adaptive trend detection with FrAMA, reverse signaling with SAR, structural alignment with the Alligator, and momentum inflection with TRIX, we need to have a label that concisely rewards the model for correctly spotting future direction continuation or failure.

For consistency, with the segmentation by quarter that we introduced in the last article, we are computing forward returns over a fixed horizon. Once this is done, we then collapse them into an either bullish or bearish or neutral classification. The precise thresholds that we use is also very important, and some traders might overlook this. If it is too tight, then our model could learn a lot of noise. If it is too wide, then our model could amount to a coin tosser. In our implementation, therefore, we are using a simple ±X% band where our X has a default value of 0.2 percent. This is in essence a hyperparameter, and readers should try to adjust this, per quarter etc., to what works best. We start off by considering two main labelling approaches of regression and classification, as indicated in our code below. We do determine later on which objective function works best for our MLP.

```
horizon = 8
# Compute forward returns
df["fwd_return"] = df["close"].shift(-horizon) / df["close"] - 1.0
up_th = 0.002 # +0.20%
down_th = -0.002 # -0.20%
def classify(r):
    if r > up_th:
        return 2 # bullish -> class index 2
    elif r < down_th:
        return 0 # bearish -> class index 0
    else:
        return 1 # neutral -> class index 1
df["label_cls"] = df["fwd_return"].apply(classify)

# Regression target
df["label_reg"] = df["fwd_return"]

# Clean training rows
Y_cls = df["label_cls"].dropna()
Y_reg = df["label_reg"].dropna()
print("Y_cls head:", Y_cls.head())
print("Y_reg head:", Y_reg.head())

# Build feature matrix
X = df[feature_cols]

# Build labels
Y_cls = df["label_cls"]
Y_reg = df["label_reg"]

# Features (whatever you defined earlier)
X = df[feature_cols]

# Find rows where ALL are valid
valid_mask = X.notna().all(axis=1) & Y_cls.notna() & Y_reg.notna()
X_final = X[valid_mask]
Y_cls_final = Y_cls[valid_mask]
Y_reg_final = Y_reg[valid_mask]
print("Final shapes:", X_final.shape, Y_cls_final.shape, Y_reg_final.shape)

# Classification labels
Y_cls = df["label_cls"]

# If you still use regression labels:
Y_reg = df["label_reg"] # or df["fwd_return"]
valid_mask = X.notna().all(axis=1) & Y_cls.notna()
X_final = X[valid_mask]
Y_cls_final = Y_cls[valid_mask]
print("Unique class labels:", sorted(Y_cls_final.unique()))

print("Y_reg head:", Y_reg.head())
```

Our dual label setup is deliberate because we could find that adaptive indicators like FrAMA and TRIX are good at forecasting magnitude, whereas reversal indicators like SAR are more suited for identifying direction. By keeping both options open, we retain the freedom to test which modelling objective gives us the most stable performance.

### Normalization and Integrity Rules for Inputs

If the indicator pipeline defines what we learn from, then normalization defines how our model perceives it. The FrAMA values may hover close to the asset price, the TRIX values could be in miniscule percentage ranges, the Alligator components could also vary widely depending on the volatility level, while the SAR values, although similar to price, can easily spike or fall whenever there is a flip. A lot is going on therefore with this diverse pool of inputs. Without some form of scaling, therefore, the MLP will treat whichever feature has the largest magnitude as the loudest voice in the room. We do not want this.

The normalization strategy we adopt also needs to respect time. Indicator values are outputted in chronological order, with trade decisions later made based on these values in MQL5. This means that even though we have a lot of data during training, the scaling we apply should use only past information. For example, global dataset scaling that is commonly used in Kaggle wonderlands would accidentally leak future volatility shifts into earlier rows, giving our model unintended clairvoyance. To avert this, we use a rolling or train-split-fitted scalers. This ensures our model behaves identically, at least by design, in both back tests and live deployment. We therefore make the following additions to the pipeline to address this.

```
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# After defining feature_cols and before splitting
alligator_cols = ["Alligator_Jaw", "Alligator_Teeth", "Alligator_Lips"]
trix_cols = ["TRIX", "TRIX_signal"]
minmax_cols = ["FRAMA", "SAR"]
std_cols = alligator_cols + trix_cols

# After valid_mask and X_final = X[valid_mask]
split = int(len(X_final) * 0.8)
X_train_df = X_final.iloc[:split]
X_val_df = X_final.iloc[split:]
y_train_df = Y_cls_final.iloc[:split]
y_val_df = Y_cls_final.iloc[split:]
std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
X_train_scaled = X_train_df.copy()
X_train_scaled[minmax_cols] = minmax_scaler.fit_transform(X_train_df[minmax_cols])
X_train_scaled[std_cols] = std_scaler.fit_transform(X_train_df[std_cols])
X_val_scaled = X_val_df.copy()
X_val_scaled[minmax_cols] = minmax_scaler.transform(X_val_df[minmax_cols])
X_val_scaled[std_cols] = std_scaler.transform(X_val_df[std_cols])

# Then convert to tensors
X_train = torch.tensor(X_train_scaled.values, dtype=torch.float32)
X_val = torch.tensor(X_val_scaled.values, dtype=torch.float32)
y_train = torch.tensor(y_train_df.values, dtype=torch.long)
y_val = torch.tensor(y_val_df.values, dtype=torch.long)
```

The z-score or standardization usually works best for TRIX and Alligator components. The min-max scaling on the other hand behaves better when handling the discrete SAR and FrAMA given that these values tend to remain range bound in predefined structural regions. Whichever scaling choice is made, it needs to be serialized alongside the ONNX export so that MQL5 can receive identically processed inputs. If this does not happen, the model can hallucinate, as stated above.

### Designing the Model

Given that our input matrix of data is normalized, and we have ‘honest features’ in place, we now need to model our MLP that can exploit this data without hallucinating or overfitting. This is the stage, often, where most trading ML projects get expansive by introducing a lot of layers to the MLP, and also appending more sophisticated layer formats. Often these could be unnecessary. The whole advantage of our pipeline is that the FrAMA, SAR, Alligator, and TRIX have already performed some signal extraction. So our model is not after spotting/identifying trend or volatility or reversal dynamics. Its main task and challenge is combining these indicators for improved generalization versus the pipeline we had in the last article.

A compact MLP can therefore be perfect for this role. It would be fast, transparent, easily debuggable, and - most importantly - have stable behaviors once exported to ONNX for MQL5 use. Unlike recurrent or convolutional networks, our simple MLP will not fail to run because data is absent, on the fly hyperparameter tuning for say the horizon window or even adjusting the input indicator parameters. While these problems do not always occur in more sophisticated networks, they always present a risk that needs to be managed in advance.

Our MLP is very basic, using 3 hidden layers, where each is reasonably sized to avoid having too many parameters. We target an ONNX export size of about 3MB. This should be large enough to capture nonlinear relationships, for example when TRIX momentum flips while SAR whipsaws and say FrAMA is in compression. We also choose our activation functions for smooth gradients and a predictable ONNX export behavior where we do not have exotic layers and unsupported operations. Our python implementation of this is given below.

```
# Update model class for larger size
class XLF_MLP(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        hidden = 512
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# Example construction:
input_dim = X_train.shape[1]
model = XLF_MLP(input_dim=input_dim)
print(model)
```

Our MLP is simple. It is not after predicting the market, but rather it produces a likelihood of directional continuation or reversal, across a predefined time horizon. The goal here is to set a modest but tradable objective, if there is such a thing. Once we export this via ONNX, the recipient Expert Advisor would receive a clean probability distribution rather than raw indicators or opaque heuristics. The model thus becomes the cap of our pipeline.

Training the Model

Once we have defined the model, aligned the input features and cleaned the labels, training the MLP should be a relatively straightforward. This of course assumes we avoid some of the usual traps, such as; temporal leakage, shuffling labels, or even training the model on data it will never see in practice. Because of the steps we took earlier in producing the input tensors for the X and Y values for the model, these two data sets are synchronized so that X lags Y. This is always important because our input indicators all use internal smoothing and shift comparisons that can silently corrupt the training process in python if we do not have proper alignment.

Ultimately, the Expert Advisor that is going to use this ONNX model expects a classification output of bullish, bearish or neutral. With this in mind, the test run that we perform uses classification labels and these will be easy to integrate into MetaTrader 5. Nonetheless, we maintain the regression labelling option in our code as a secondary target for experimentation. Regression sometimes has a way of revealing subtle differences in the intensity of the forecast that classification would miss. So, if the final Expert Advisor is considering 2 different indicator pools as inputs, this difference could help break the tie.

Our training loop is geared towards stability where we do not use schedulers, five-page loss functions, or a lot of hyperparameter adjustments. Trading models are meant to thrive on predictability. We therefore use Adam, a modest learning rate, mini batching, as well as early stopping that is based on a validation loss. Our objective is not perfection per se, but being able to generalize better. We want a model robust enough to survive market regime shifts of the XLF. This is how we implement the training.

```
train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(30):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    # validation
    model.eval()
    with torch.no_grad():
        val_loss = sum(criterion(model(xv), yv) for xv, yv in val_loader)
    print(f"Epoch {epoch}: val_loss={val_loss.item():.4f}")
```

### The Validation

Training any model, especially in python where compute costs are minimal, is relatively easy. The hard part, which is actually the whole point of training in the first place, is proving that what we’ve learnt from training works across different market regimes. We are attempting to achieve this with the XLF, an ETF whose price action in many ways is shaped by interest rate policy, credit cycles, liquidity and a host of other factors. To therefore try to evaluate this MLP with this financial sector ETF, we use a quarter aware walk forward validation approach. In essence, we measure how well the model performs when forced to learn from the current regime and then predict the subsequent one. This is meant to mimic real-world deployment, which when testing in MetaTrader 5 actually happens seamlessly, however in python extra ‘guard rails’ need to be put in place to ensure this actually happens.

The input indicator data points within the matrix all contribute differently to this robustness when forecasting. The market regime quarters are all different. Some quarters will reward fractal compression, others will reward momentum flips, while others trend acceleration. Ideally, a proper validation needs to measure performance across all these different market regimes instead of assuming the average performance from testing on just a few provides sufficient guidance going forward.

We validated on only the most recent 20 percent of the test window from 2020.01.01 to 2025.12.01. Clearly not very representative of all regimes, a pragmatic remedy though would be training on a 60 to 40 split from the inception data in December 1998, such that the recent 40 percent is stretched to cover a wider variety of market regimes. Our training regiment, nonetheless, is as follows.

[![i9](https://c.mql5.com/2/185/i9.png)](https://c.mql5.com/2/185/i9.png "https://c.mql5.com/2/185/i9.png")

The process we use of a marching evaluation window can expose, potentially, where the model is failing - say in volatile sector rotations. It can also reveal where it is excelling, and this from our testing happens in slow building rate cycles where most indicators are behaving as one would expect. The end result of our test runs thus becomes not an accuracy score per se, but a profile. A measure or indication of how well the model adapts from one market regime to another. Ideally, any ONNX model deployed into MQL5 should survive this gauntlet before being considered for live use.

### Exporting via ONNX

When the tested MLP survives quarter aware validation, what follows next is converting it into a portable format that MQL5 and MetaTrader 5 Expert Advisors can utilize. The ONNX standard makes this feasible because it enforces a limited, predictable operator set and guarantees consistent behavior across platforms. Exporting our model will only be successful if it uses operations that are supported by ONNX runtime. Fortunately, our architecture is very simple, for reasons outlined above, and therefore the linear layers ReLU activations and the single output head should all present no challenges to ONNX. The export is thus deterministic and some debugging in MQL5 can be handled.

Nonetheless, as with any ONNX export, before we send off the model we need to look at and note the exact input and output tensor shapes. Also, the effort put in above in normalizing the input matrix of data needs to be carried forward into MetaTrader 5. The ONNX export only gives us the network in MetaTrader 5, it does nothing to normalize the input data. This implies it is something we will handle in MQL5 prior to making a forward pass. The export in python, though, is as follows.

```
import torch.onnx
dummy = torch.randn(1, X_train.shape[1], dtype=torch.float32)
onnx_path = "xlf_model.onnx"
torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names=["features"],
    outputimport torch.onnx
dummy = torch.randn(1, X_train.shape[1], dtype=torch.float32)
onnx_path = "xlf_model.onnx"
torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names=["features"],
    output_names=["logits"],
    dynamic_axes={"features": {0: "batch"}},
    opset_version=12
)
print("Model exported to:", onnx_path)

import onnx
import onnxruntime as ort

# Create an ONNX Runtime session
session = ort.InferenceSession(onnx_path)

# Run inference

Running inference.= session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_data = np.random.randn(1, X_train.shape[1]).astype(np.float32)# torch.randn(1, X_train.shape[1], dtype=torch.float32)
#input_data = input_data.to(device)
output  = session.run([output_name], {input_name: input_data})

for i in session.get_inputs():
    print(f"inputs: {i.name}, Shape: {i.shape}, Type: {i.type}")

for o in session.get_outputs():
    print(f"outputs: {o.name}, Shape: {o.shape}, Type: {o.type}")
_names=["logits"],
    dynamic_axes={"features": {0: "batch"}},
    opset_version=12
)
print("Model exported to:", onnx_path)
```

### Using in MQL5

Without being able to feed data to an exported ONNX, in order to make forecasts, it is useless. MQL5 comes in handy in this regard by wrapping the ONNX runtime into a clean minimal API which implies deploying this model within an Expert Advisor is less tedious than managing a python backend. The key point is to ensure that the Expert Advisor reconstructs the exact feature vector that we have been using when training in python. The alignment should be bar for bar with no missteps. It is easy to go astray here if one uses mismatched scaling for the inputs, or uses different indicator parameters.

Our implemented EA workflow is simple. We compute indicators on every incoming bar; then we apply the scaling properties to each indicator, inline with what we used when testing in python. After this, we assemble the feature vector, in the correct order used in python, invoke the ONNX session and then interpret the output vector. Essentially, we have a three-step process. Prepare features; Run inference; and finally convert logits to trading decision. Our Expert Advisor does not worry about training, or even updating the network weights while trading, we only need a stable repeatable interface. Below is an abridged version of our code that shows the functions that perform these three steps.

Preparing the features.

```
//+------------------------------------------------------------------+
//| Detecting the "weighted" direction                               |
//+------------------------------------------------------------------+
double CSignalXLF::Direction(void)
{  m_x.Fill(0.0f);
   m_y.Fill(0.0f);
   vector _z(5), _z_out(5);
   _z.Fill(0.0);
   m_frama.Refresh(-1);
   _z[0] = m_frama.Main(X());
   m_sar.Refresh(-1);
   _z[1] = m_sar.Main(X());
   m_alligator.Refresh(-1);
   _z[2] = m_alligator.Jaw(X());
   _z[3] = m_alligator.Lips(X());
   _z[4] = m_alligator.Teeth(X());
   vector _mm(2), _mm_out(2);
   m_trix.Refresh(-1);
   _mm[0] = m_trix.Main(X());
   _mm[1] = m_trix.Main(X() + 1);
   if(NormalizeZScore(_z, _z_out) && NormalizeMinMax(_mm, _mm_out))
   {  for(int i = 0; i < 5; i++)
      {  m_x[i] = float(_z_out[i]);
      }
      for(int i = 0; i < 2; i++)
      {  m_x[5 + i] = float(_mm_out[i]);
      }
      //printf(__FUNCSIG__);
      ////Print(" z out: ",  _z_out);
      ////Print(" mm out: ",  _mm_out);
      //Print(" in x: ",  m_x);
      m_y = Infer(m_x);
      return(LongCondition() - ShortCondition());
   }
   return(0.0);
}
```

Running inference.

```
//+------------------------------------------------------------------+
//| Inference Pass.                                                  |
//+------------------------------------------------------------------+
vectorf CSignalXLF::Infer(vectorf &X)
{  vectorf _y(__CLASSES);
   _y.Fill(0.0);
   //Print(" x in: ", __FUNCTION__, X);
   ResetLastError();
   if(!OnnxRun(m_handle, ONNX_NO_CONVERSION, X, _y))
   {  printf(__FUNCSIG__ + " failed to get y forecast, err: %i", GetLastError());
   }
   vectorf _yy(_y.Size());
   _yy.Fill(0.0);
   if(_y.Size() == 3)
   {  Softmax(_y, _yy);
   }
   else
   {  _yy.Copy(_y);
   }
   //printf(__FUNCSIG__);
   //Print(" y out: ",  _yy);
   return(_yy);
}
```

Interpreting and using the output.

```
//+------------------------------------------------------------------+
//| "Voting" that price will grow.                                   |
//+------------------------------------------------------------------+
int CSignalXLF::LongCondition(void)
{  if(fabs(m_y[0]) > fabs(m_y[2]))//0.5)
   {  //printf(__FUNCSIG__);
      //Print(" in y: ",  m_y);
      //return(int(100.0 * (fabs(m_y[0]) - 0.5) / 0.5));
      return(int(100.0 * fabs(m_y[0])));
   }
   return(0);
}
//+------------------------------------------------------------------+
//| "Voting" that price will fall.                                   |
//+------------------------------------------------------------------+
int CSignalXLF::ShortCondition(void)
{  if(fabs(m_y[0]) < fabs(m_y[2]))//0.5)
   {  //printf(__FUNCSIG__);
      //Print(" in y: ",  m_y);
      //return(int(100.0 * fabs(m_y[0]) / 0.5));
      return(int(100.0 * fabs(m_y[2])));
   }
   return(0);
}
```

By using the [MQL5 wizard](https://www.mql5.com/en/articles/171), whose main prerequisite is a custom signal class, such as the one we are using above, we avoid the need to cater for some important but mundane requirements of any Expert Advisor. These include having noise-risk controls on the price returned by the broker, performing spread checks, avoiding duplicate order entries when opening a position, etc. Our focus, for these testing purposes, rests solely on the ONNX interaction. Our Expert Advisor is a thin wrapper that takes in indicator values, normalizes these into features, passes them to the ONNX model for inference, processes the logits outputs, then makes a trade decision. This could be summarized by our flow diagram below.

[![i11](https://c.mql5.com/2/185/i10-1.png)](https://c.mql5.com/2/185/i10-1.png "https://c.mql5.com/2/185/i10-1.png")

When using the classification label option, the output of the ONNX model is three logits. They are meant to represent bullish, neutral or bearish probabilities. However, they are logits and not yet probabilities. When looked at in their raw form, often the values not only include negative figures, but their magnitude often exceeds one and also there tends to be little variability over many bars in these outputs if one were to run a test, meaning no clear signal can be inferred from them in their state. What is required in order to make them usable would be applying the ONNX output through the soft-max function. We do this as follows.

```
//+------------------------------------------------------------------+
//| Compute softmax probabilities from logits                        |
//| logits[] : input raw scores (e.g. from your network)             |
//| probs[]  : output probabilities (same length as logits)          |
//+------------------------------------------------------------------+
void CSignalXLF::Softmax(vectorf &Logits, vectorf &Probs)
{  int _size = int(Logits.Size());
   if(_size == 0) return;

   Probs.Init(_size);
   Probs.Fill(0.0);

   // 1) Find max logit for numerical stability
   float _max_logit = Logits.Max();

   // 2) Exponentiate shifted logits and sum
   float _sum_exp = 0.0;
   for(int i = 0; i < _size; i++)
   {  Probs[i] = float(MathExp(Logits[i] - _max_logit));
      _sum_exp += Probs[i];
   }

   // 3) Normalize to get probabilities
   if(_sum_exp <= 0.0f)
   {  // Failsafe: uniform distribution if something goes wrong
      float _p = 1.0f / _size;
      for(int i = 0; i < _size; i++)
         Probs[i] = _p;
      return;
   }

   float _inv_sum = 1.0f / _sum_exp;
   for(int i = 0; i < _size; i++)
      Probs[i] *= _inv_sum;
}
```

With this in place, a [wizard assembled Expert Advisor](https://www.mql5.com/en/articles/171) that uses this custom signal class should be ready for testing and further development. The code of this class is attached below.

### Conclusion

In this article, we started by highlighting the potential limitations of indicator ranking and have concluded with a deployable, ONNX-based trading engine, tested with the XLF ETF. We have taken a research pipeline that encompassed clean data engineering, quarterly segmentation, indicator selection and feature construction, to produce a model that runs natively within MQL5 without constant python dependencies.

A key insight here could be that machine learning becomes resourceful only if it is paired with a disciplined structure. The indicators help with data extraction; scaling/normalization preserves this data’s integrity; quarter-aware validation makes the case for possible use; and the ONNX enables frictionless deployment. Our end system is not intended to be a black box or a brute force signal factory, but rather something that is compact, with an explainable decision layer that is forged from multi-quarter behavior and capable of evolving as market regimes evolve.

| name | description |
| --- | --- |
| XLF.mq5 | Wizard assembled Expert Advisor whose header lists referenced files |
| SignalXLF.mqh | Custome Signal class file used in wizard assembly |
| xlf-cls-model.onnx | Classification based ONNX model |
| xlf-reg-model.onnx | Regression based ONNX model |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20595.zip "Download all attachments in the single ZIP archive")

[XLF.mq5](https://www.mql5.com/en/articles/download/20595/XLF.mq5 "Download XLF.mq5")(6.33 KB)

[SignalXLF.mqh](https://www.mql5.com/en/articles/download/20595/SignalXLF.mqh "Download SignalXLF.mqh")(27.15 KB)

[xlf\_cls\_model.onnx](https://www.mql5.com/en/articles/download/20595/xlf_cls_model.onnx "Download xlf_cls_model.onnx")(3101.37 KB)

[xlf\_reg\_model.onnx](https://www.mql5.com/en/articles/download/20595/xlf_reg_model.onnx "Download xlf_reg_model.onnx")(3097.36 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Codex Pipelines: From Python to MQL5 for Indicator Selection — A Multi-Quarter Analysis of the FXI ETF](https://www.mql5.com/en/articles/20550)
- [Market Positioning Codex for VGT with Kendall's Tau and Distance Correlation](https://www.mql5.com/en/articles/20271)
- [Markets Positioning Codex in MQL5 (Part 2): Bitwise Learning, with Multi-Patterns for Nvidia](https://www.mql5.com/en/articles/20045)
- [Markets Positioning Codex in MQL5 (Part 1): Bitwise Learning for Nvidia](https://www.mql5.com/en/articles/20020)
- [MQL5 Wizard Techniques you should know (Part 85): Using Patterns of Stochastic-Oscillator and the FrAMA with Beta VAE Inference Learning](https://www.mql5.com/en/articles/19948)
- [MQL5 Wizard Techniques you should know (Part 84): Using Patterns of Stochastic Oscillator and the FrAMA - Conclusion](https://www.mql5.com/en/articles/19890)

**[Go to discussion](https://www.mql5.com/en/forum/501986)**

![Building AI-Powered Trading Systems in MQL5 (Part 7): Further Modularization and Automated Trading](https://c.mql5.com/2/186/20588-building-ai-powered-trading-logo__1.png)[Building AI-Powered Trading Systems in MQL5 (Part 7): Further Modularization and Automated Trading](https://www.mql5.com/en/articles/20588)

In this article, we enhance the AI-powered trading system's modularity by separating UI components into a dedicated include file. The system now automates trade execution based on AI-generated signals, parsing JSON responses for BUY/SELL/NONE with entry/SL/TP, visualizing patterns like engulfing or divergences on charts with arrows, lines, and labels, and optional auto-signal checks on new bars.

![The View and Controller components for tables in the MQL5 MVC paradigm: Resizable elements](https://c.mql5.com/2/160/18941-komponenti-view-i-controller-logo__2.png)[The View and Controller components for tables in the MQL5 MVC paradigm: Resizable elements](https://www.mql5.com/en/articles/18941)

In the article, we will add the functionality of resizing controls by dragging edges and corners of the element with the mouse.

![From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://c.mql5.com/2/186/20587-from-novice-to-expert-automating-logo.png)[From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)

For many traders, the gap between knowing a risk rule and following it consistently is where accounts go to die. Emotional overrides, revenge trading, and simple oversight can dismantle even the best strategy. Today, we will transform the MetaTrader 5 platform into an unwavering enforcer of your trading rules by developing a Risk Enforcement Expert Advisor. Join this discussion to find out more.

![From Novice to Expert: Trading the RSI with Market Structure Awareness](https://c.mql5.com/2/185/20554-from-novice-to-expert-trading-logo__1.png)[From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)

In this article, we will explore practical techniques for trading the Relative Strength Index (RSI) oscillator with market structure. Our focus will be on channel price action patterns, how they are typically traded, and how MQL5 can be leveraged to enhance this process. By the end, you will have a rule-based, automated channel-trading system designed to capture trend continuation opportunities with greater precision and consistency.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=vyfrmctykcpnvjhxsnszzxtdwczvnybs&ssn=1769092811819128464&ssn_dr=0&ssn_sr=0&fv_date=1769092811&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F20595&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Codex%20Pipelines%2C%20from%20Python%20to%20MQL5%2C%20for%20Indicator%20Selection%3A%20A%20Multi-Quarter%20Analysis%20of%20the%20XLF%20ETF%20with%20Machine%20Learning%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909281117682464&fz_uniq=5049291509192304842&sv=2552)

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