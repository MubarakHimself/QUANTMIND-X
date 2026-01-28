---
title: Neural Networks in Trading: A Hybrid Trading Framework with Predictive Coding (Final Part)
url: https://www.mql5.com/en/articles/16713
categories: Trading Systems, Expert Advisors, Machine Learning
relevance_score: 6
scraped_at: 2026-01-23T11:33:17.238347
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/16713&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062549686688064587)

MetaTrader 5 / Trading systems


### Introduction

In the previous [article](https://www.mql5.com/en/articles/16686), we examined in detail the theoretical aspects of the hybrid trading system _[StockFormer](https://www.mql5.com/go?link=https://www.ijcai.org/proceedings/2023/0530.pdf "https://www.ijcai.org/proceedings/2023/0530.pdf")_, which combines predictive coding and reinforcement learning algorithms to forecast market trends and the dynamics of financial assets. _StockFormer_ is a hybrid framework that brings together several key technologies and approaches to address complex challenges in financial markets. Its core feature is the use of three modified _Transformer_ branches, each responsible for capturing different aspects of market dynamics. The first branch extracts hidden interdependencies between assets, while the second and third focus on short-term and long-term forecasting, enabling the system to account for both current and future market trends.

The integration of these branches is achieved through a cascade of attention mechanisms, which enhance the model’s ability to learn from multi-head blocks, improving its processing and detection of latent patterns in the data. As a result, the system can not only analyze and predict trends based on historical data but also take into account dynamic relationships between various assets. This is particularly important for developing trading strategies capable of adapting to rapidly changing market conditions.

The original visualization of the _StockFormer_ framework is provided below.

![](https://c.mql5.com/2/173/2203917964723__1.png)

In the practical section of the previous article, we implemented the algorithms of the _Diversified Multi-Head Attention_ ( _DMH-Attn_) module, which serves as the foundation for enhancing the standard attention mechanism in the _Transformer_ model. _DMH-Attn_ significantly improves the efficiency of detecting diverse patterns and interdependencies in financial time series, which is especially valuable when working with noisy and highly volatile data.

In this article, we will continue the work by focusing on the architecture of different parts of the model and the mechanisms of their interaction in creating a unified state space. Additionally, we will examine the process of training the decision-making _Agent's_ trading policy.

### Predictive Coding Models

We begin with predictive coding models. The authors of the _StockFormer_ framework proposed using three predictive models. One is designed to identify dependencies within the data describing the dynamics of the analyzed financial assets. The other two are trained to forecast the upcoming movements of the multimodal time series under study, each with a different planning horizon.

All three models are based on the _Encoder_– _Decoder Transformer_ architecture, utilizing modified _DMH-Attn_ modules. In our implementation, the _Encoder_ and _Decoder_ will be created as separate models.

#### Dependency Search Models

The architecture of the dependency search models for time series of financial assets is defined in the method _CreateRelationDescriptions_.

```
bool CreateRelationDescriptions(CArrayObj *&encoder, CArrayObj *&decoder)
  {
//---
   CLayerDescription *descr;
//---
   if(!encoder)
     {
      encoder = new CArrayObj();
      if(!encoder)
         return false;
     }
   if(!decoder)
     {
      decoder = new CArrayObj();
      if(!decoder)
         return false;
     }
```

The method's parameters include pointers to two dynamic arrays, into which we must pass the architecture descriptions of the Encoder and Decoder. Inside the method, we check the validity of the received pointers and, if necessary, create new instances of the dynamic array objects.

For the first layer of the Encoder, we use a fully connected layer of sufficient size to accept all tensor data from the raw input.

Recall that the Encoder receives historical data across the full depth of the analyzed history.

```
//--- Encoder
   encoder.Clear();
//---
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   int prev_count = descr.count = (HistoryBars * BarDescr);
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

The raw data originates from the trading terminal. As one might expect, the multimodal time series data, comprising indicators and possibly multiple financial instruments, belongs to different distributions. Therefore, we first preprocess the input data using a batch normalization layer.

```
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBatchNormOCL;
   descr.count = prev_count;
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

The _StockFormer_ authors suggest randomly masking up to 50% of the input data during training of the dependency search models. The model must reconstruct the masked data based on the remaining information. In our _Encoder_, this masking is handled by a _Dropout_ layer.

```
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronDropoutOCL;
   descr.count = prev_count;
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.probability = 0.5f;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

Following this, we add a learnable positional encoding layer.

```
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronLearnabledPE;
   descr.count = prev_count;
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

The _Encoder_ concludes with a diversified multi-head attention module consisting of three nested layers.

```
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronDMHAttention;
   descr.window = BarDescr;
   descr.window_out = 32;
   descr.count = HistoryBars;
   descr.step = 4;               //Heads
   descr.layers = 3;             //Layers
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

The input to the Decoder in the dependency search model is the same multimodal time series, with identical masking and positional encoding applied. Thus, most of the Encoder and Decoder architectures are identical. The key difference is that we replace the diversified multi-head attention module with a cross-attention module, which aligns the data streams of the Decoder and Encoder.

```
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronCrossDMHAttention;
//--- Windows
     {
      int temp[] = {BarDescr, BarDescr};
      if(ArrayCopy(descr.windows, temp) < (int)temp.Size())
         return false;
     }
   descr.window_out = 32;
//--- Units
     {
      int temp[] = {prev_count/descr.windows[0], HistoryBars};
      if(ArrayCopy(descr.units, temp) < (int)temp.Size())
         return false;
     }
   descr.step = 4;               //Heads
   descr.layers = 3;             //Layers
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!decoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

Since the Decoder's output will be compared against the original input data, we finalize the model with a reverse normalization layer.

```
//--- layer 5
   prev_count = descr.units[0] * descr.windows[0];
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronRevInDenormOCL;
   descr.count = prev_count;
   descr.layers = 1;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!decoder.Add(descr))
     {
      delete descr;
      return false;
     }
//---
   return true;
  }
```

#### Prediction Models

Both prediction models, despite having different planning horizons, share the same architecture, which is defined in the method _CreatePredictionDescriptions_. It is worth noting that the Encoder is designed to receive the same multimodal time series previously analyzed by the dependency search model. Therefore, we fully reuse the Encoder architecture, with the exception of the _Dropout_ layer, since input masking is not applied during the training of prediction models.

The Decoder of the prediction model receives as input only the feature vector of the last bar, whose values are passed through a fully connected layer.

```
//--- Decoder
   decoder.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = (BarDescr);
   descr.activation = None;
   descr.optimization = ADAM;
   if(!decoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

As in the models described earlier, this is followed by a batch normalization layer, which we use for the initial preprocessing of raw input data.

```
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBatchNormOCL;
   descr.count = prev_count;
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!decoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

In this article, we focus on training the model to analyze historical data for a single financial instrument. Given this, having only a single-bar description vector in the input data minimizes the effectiveness of positional encoding. For this reason, we omit it here. However, when analyzing multiple financial instruments, it is recommended to add positional encoding to the input data.

Next comes a three-layer diversified multi-head cross-attention module, which uses the corresponding Encoder's output as its second source of information.

```
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronCrossDMHAttention;
//--- Windows
     {
      int temp[] = {BarDescr, BarDescr};
      if(ArrayCopy(descr.windows, temp) < (int)temp.Size())
         return false;
     }
   descr.window_out = 32;
//--- Units
     {
      int temp[] = {1, HistoryBars};
      if(ArrayCopy(descr.units, temp) < (int)temp.Size())
         return false;
     }
   descr.step = 4;               //Heads
   descr.layers = 3;             //Layers
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!decoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

At the model's output, we add a fully connected projection layer without an activation function.

```
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = BarDescr;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!decoder.Add(descr))
     {
      delete descr;
      return false;
     }
//---
   return true;
  }
```

Two important points should be emphasized here. First, unlike traditional models that predict expected values of the continuation of the analyzed time series, the authors of the _StockFormer_ framework propose predicting change coefficients of the indicators. This means that the size of the output vector matches the input tensor of the Decoder, regardless of the planning horizon. Such an approach allows us to eliminate the reverse normalization layer at the Decoder's output. Moreover, in this prediction setup, reverse normalization becomes redundant. Since change coefficients and raw indicators belong to different distributions.

Second, regarding the use of a fully connected layer at the Decoder's output. As mentioned earlier, we are analyzing a multimodal time series of a single financial instrument. Therefore, we expect all unitary sequences under analysis to exhibit varying degrees of correlation. Therefore, their change coefficients must be aligned. Therefore, a fully connected layer is appropriate in this case. If, however, you plan to perform parallel analysis of multiple financial instruments, it is advisable to replace the fully connected layer with a convolutional one, enabling independent prediction of change coefficients for each asset.

This concludes our review of the predictive coding model architectures. A full description of their design can be found in the appendix.

#### Training Predictive Coding Models

In the _StockFormer_ framework, the training of predictive coding models is implemented as a dedicated stage. After reviewing the architectures of the predictive models, we now turn to constructing an Expert Advisor for their training. The EA's base methods are largely borrowed from similar programs discussed in previous articles of this series. Therefore, in this article, we will focus primarily on the direct training algorithm, organized in the _Train_ method.

First, we will do a little preparatory work. Here, we form a probability vector for selecting trajectories from the experience replay buffer, assigning higher probabilities to those with maximum profitability. In this way, we bias the training process toward profitable runs, filling it with positive examples.

```
void Train(void)
  {
//---
   vector<float> probability = GetProbTrajectories(Buffer, 0.9);
//---
   vector<float> result, target, state;
   matrix<float> predict;
   bool Stop = false;
//---
   uint ticks = GetTickCount();
```

At this stage, we also declare the necessary local variables used to store intermediate data during training. After completing the preparation, we initiate the training iteration loop. The total number of iterations defined in the EA's external parameters.

```
   for(int iter = 0; (iter < Iterations && !IsStopped() && !Stop); iter ++)
     {
      int tr = SampleTrajectory(probability);
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2 - NForecast));
      if(i <= 0)
        {
         iter --;
         continue;
        }
      if(!state.Assign(Buffer[tr].States[i].state) ||
         MathAbs(state).Sum() == 0 ||
         !bState.AssignArray(state))
        {
         iter --;
         continue;
        }
      if(!state.Assign(Buffer[tr].States[i + NForecast].state) ||
         !state.Resize((NForecast + 1)*BarDescr) ||
         MathAbs(state).Sum() == 0)
        {
         iter --;
         continue;
        }
```

Inside the loop, we sample one trajectory from the experience replay buffer along with its initial environment state. We then check for the presence of historical data in the chosen state as well as actual data over the specified planning horizon. If these checks succeed, we transfer the historical values of the required analysis depth into the appropriate data buffer and perform the forward pass of all predictive models.

```
      //--- Feed Forward
      if(!RelateEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL) ||
         !RelateDecoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CNet*)GetPointer(RelateEncoder)) ||
         !ShortEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL) ||
         !ShortDecoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CNet*)GetPointer(ShortEncoder)) ||
         !LongEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL) ||
         !LongDecoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CNet*)GetPointer(LongEncoder)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
```

It is important to note that, despite their identical architecture, each predictive model has its own Encoder. This increases the total number of trainable models and, accordingly, the computational costs of training and operation. However, it also enables each model to capture dependencies relevant to its specific task.

Another point concerns the use of the raw input tensor in the Decoder's main stream. As discussed earlier, the prediction models' Decoders accept only the last bar as input. However, in training, the historical buffer across the full analysis depth is used in all cases. To clarify, the environment state stored in the replay buffer can be represented as a matrix. In this matrix, rows correspond to bars and columns to features (prices and indicators). The first row contains data from the last bar. Thus, when passing a tensor larger than the Decoder's input size, the model simply takes the first segment matching the input layer's size. This is exactly what we need, allowing us to avoid creating additional buffers and unnecessary data copies.

After a successful forward pass, we prepare target values and perform backpropagation. For the dependency search models, the target values are the multimodal time series itself. Therefore, we can immediately run backpropagation through the _Decoder_, pass the error gradient to the _Encoder_. Based on the gradient obtained, we update the _Encoder_ parameters accordingly.

```
      //--- Relation
      if(!RelateDecoder.backProp(GetPointer(bState), (CNet *)GetPointer(RelateEncoder)) ||
         !RelateEncoder.backPropGradient((CBufferFloat*)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
```

For the prediction models, however, target values must be defined. As mentioned earlier, the targets here are change coefficients of the parameters. We assume that the planning horizon is shorter than the analysis depth of historical data. Thus, to calculate target values, we take from the replay buffer the future environment state recorded at the required horizon steps ahead. And then we transform this tensor into a matrix where each row corresponds to a bar.

```
      //--- Prediction
      if(!predict.Resize(1, state.Size()) ||
         !predict.Row(state, 0) ||
         !predict.Reshape(NForecast + 1, BarDescr)
        )
        {
         iter --;
         continue;
        }
```

Since the first rows of such a matrix represent later bars, we take one more row than the planning horizon. The last row of this truncated matrix corresponds to the current bar under analysis.

It is important to recall that the replay buffer stores unnormalized data. To bring the calculated change coefficients into a meaningful range, we normalize them by the maximum absolute values of each parameter in the matrix of future values. As a result, we obtain coefficients typically lying within the range {-2.0, 2.0}.

```
      result = MathAbs(predict).Max(0);
```

For the short-term prediction model, the target is the change coefficient of the parameter at the next bar. This is calculated as the difference between the last two rows of the prediction matrix, divided by the vector of maximum values, and then stored in the appropriate buffer.

```
      target = (predict.Row(NForecast - 1) - predict.Row(NForecast)) / result;
      if(!bShort.AssignArray(target))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
```

For the long-term prediction model, we sum the parameter change coefficients across all bars, applying a discounting factor.

```
      for(int i = 0; i < NForecast - 1; i++)
         target += (predict.Row(i) - predict.Row(i + 1)) / result *
                              MathPow(DiscFactor, NForecast - i - 1);
      if(!bLong.AssignArray(target))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
```

Once the full set of target values is defined, we update the parameters of the prediction models to minimize forecast error. Specifically, we perform backpropagation through the Decoder and Encoder of the short-term prediction model first, followed by the long-term model.

```
      //--- Short prediction
      if(!ShortDecoder.backProp(GetPointer(bShort), (CNet *)GetPointer(ShortEncoder)) ||
         !ShortEncoder.backPropGradient((CBufferFloat*)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
```

```
      //--- Long prediction
      if(!LongDecoder.backProp(GetPointer(bLong), (CNet *)GetPointer(LongEncoder)) ||
         !LongEncoder.backPropGradient((CBufferFloat*)NULL))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
```

After updating all models trained at this stage, we log progress to inform the user and then proceed to the next training iteration.

```
      //---
      if(GetTickCount() - ticks > 500)
        {
         double percent = double(iter) * 100.0 / (Iterations);
         string str = StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Relate",
                                  percent, RelateDecoder.getRecentAverageError());
         str += StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Short", percent,
                                            ShortDecoder.getRecentAverageError());
         str += StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Long", percent,
                                             LongDecoder.getRecentAverageError());
         Comment(str);
         ticks = GetTickCount();
        }
     }
```

Upon completion of all training iterations, we clear the comments field on the chart (previously used to display training updates).

```
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Relate", RelateDecoder.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Short", ShortDecoder.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Long", LongDecoder.getRecentAverageError());
   ExpertRemove();
//---
  }
```

We print the results in the journal and initiate the termination of the EA operation.

The full source code of the predictive model training EA can be found in the attachment (file: "...\\MQL5\\Experts\\StockFormer\\Study1.mq5").

Finally, it should be noted that during model training for this article, we used the same input data structure as in previous works. Importantly, predictive model training relies solely on environment states that are independent of the Agent's actions. Therefore, training can be launched using a pre-collected dataset. We now move on to the next stage of our work.

### Policy Training

While the predictive models are being trained, we turn to the next stage - training the _Agent_ behavior policy.

#### Model Architecture

We begin by preparing the architectures of the models used in this stage, as defined in the _CreateDescriptions_ method. It is important to note that in the StockFormer framework, both the _Actor_ and _Critic_ take as input the outputs of the predictive models, which are combined into a unified subspace using a cascade of attention modules. In our library, we can build models with two data sources. So, we split the attention cascade into two separate models. In the first model, we align data from two planning horizons. The authors recommend using long-term planning data from the main stream, as it is less sensitive to noise.

The architecture of the two-horizon alignment model is straightforward. Here we create two layers:

1. A fully connected input layer.
2. A diversified cross-attention module with three internal layers.

```
//--- Long to Short predict
   long_short.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   int prev_count = descr.count = (BarDescr);
   descr.activation = None;
   descr.optimization = ADAM;
   if(!long_short.Add(descr))
     {
      delete descr;
      return false;
     }
//--- Layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronCrossDMHAttention;
//--- Windows
     {
      int temp[] = {BarDescr, BarDescr};
      if(ArrayCopy(descr.windows, temp) < (int)temp.Size())
         return false;
     }
   descr.window_out = 32;
//--- Units
     {
      int temp[] = {1, 1};
      if(ArrayCopy(descr.units, temp) < (int)temp.Size())
         return false;
     }
   descr.step = 4;               //Heads
   descr.layers = 3;             //Layers
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!long_short.Add(descr))
     {
      delete descr;
      return false;
     }
```

No normalization layer is used here, as the model input is the output of previously trained prediction models, not raw data.

The results of the two-horizon alignment are then enriched with information about the current environment state, obtained from the Encoder of the dependency search model applied to the input data.

Recall that the dependency search model was trained to reconstruct masked portions of the input data. At this stage, we expect that each unitary time series has a predictive state representation formed based on the other univariate sequences. Therefore, the Encoder output is a denoised tensor of the environment state, as outliers that do not fit the model's expectations are compensated by statistical values derived from other sequences.

The architecture of the model that enriches predictions with environment state information closely mirrors the two-horizon alignment model. The only difference is that we change the sequence length of the second data source.

```
//--- Predict to Relate
   predict_relate.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = (BarDescr);
   descr.activation = None;
   descr.optimization = ADAM;
   if(!predict_relate.Add(descr))
     {
      delete descr;
      return false;
     }
//--- Layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronCrossDMHAttention;
//--- Windows
     {
      int temp[] = {BarDescr, BarDescr};
      if(ArrayCopy(descr.windows, temp) < (int)temp.Size())
         return false;
     }
   descr.window_out = 32;
//--- Units
     {
      int temp[] = {1, HistoryBars};
      if(ArrayCopy(descr.units, temp) < (int)temp.Size())
         return false;
     }
   descr.step = 4;               //Heads
   descr.layers = 3;             //Layers
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!predict_relate.Add(descr))
     {
      delete descr;
      return false;
     }
```

After constructing the attention cascade that combines the outputs of the three predictive models into a unified subspace, we proceed to build the Actor. The input to the Actor model is the output of the attention cascade.

```
//--- Actor
   actor.Clear();
//--- Input Layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = (BarDescr);
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

The predictive expectations are combined with account state information.

```
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = LatentCount;
   descr.window = prev_count;
   descr.step = AccountDescr;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

This combined information is passed through a decision-making block implemented as an _MLP_ with a stochastic output head.

```
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = LatentCount;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   descr.probability = Rho;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 2 * NActions;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.probability = Rho;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronVAEOCL;
   descr.count = NActions;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

At the model's output, trade parameters for each direction are adjusted using a convolutional layer with a sigmoid activation function.

```
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   descr.count = NActions / 3;
   descr.window = 3;
   descr.step = 3;
   descr.window_out = 3;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   descr.probability = Rho;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

The _Critic_ has a similar architecture, but instead of account state, it analyzes the _Agent_ actions. Its output does not use a stochastic head. The full architecture of all models is available in the appendix.

#### Policy Training Procedure

Once the model architectures are defined, we organize the training algorithms. The second stage involves finding the optimal _Agent_ behavior strategy to maximize returns while minimizing risk.

As before, the training method begins with preparation. We generate a probability vector for selecting trajectories from the experience replay buffer based on their performance and declaring local variables.

```
void Train(void)
  {
//---
   vector<float> probability = GetProbTrajectories(Buffer, 0.9);
//---
   vector<float> result, target, state;
   bool Stop = false;
//---
   uint ticks = GetTickCount();
```

We then enter the training loop, with the number of iterations set by the EA's external parameters.

```
   for(int iter = 0; (iter < Iterations && !IsStopped() && !Stop); iter ++)
     {
      int tr = SampleTrajectory(probability);
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2 - NForecast));
      if(i <= 0)
        {
         iter --;
         continue;
        }
      if(!state.Assign(Buffer[tr].States[i].state) ||
         MathAbs(state).Sum() == 0 ||
         !bState.AssignArray(state))
        {
         iter --;
         continue;
        }
```

Within each iteration, we sample a trajectory and its state for the current iteration. Make sure to verify that we have all necessary data.

Unlike predictive models, policy training requires additional input data. After extracting the environment state description, we collect account balance and open positions from the replay buffer at the relevant timestep.

```
      //--- Account
      bAccount.Clear();
      float PrevBalance = Buffer[tr].States[MathMax(i - 1, 0)].account[0];
      float PrevEquity = Buffer[tr].States[MathMax(i - 1, 0)].account[1];
      bAccount.Add((Buffer[tr].States[i].account[0] - PrevBalance) / PrevBalance);
      bAccount.Add(Buffer[tr].States[i].account[1] / PrevBalance);
      bAccount.Add((Buffer[tr].States[i].account[1] - PrevEquity) / PrevEquity);
      bAccount.Add(Buffer[tr].States[i].account[2]);
      bAccount.Add(Buffer[tr].States[i].account[3]);
      bAccount.Add(Buffer[tr].States[i].account[4] / PrevBalance);
      bAccount.Add(Buffer[tr].States[i].account[5] / PrevBalance);
      bAccount.Add(Buffer[tr].States[i].account[6] / PrevBalance);
      //---
      double time = (double)Buffer[tr].States[i].account[7];
      double x = time / (double)(D'2024.01.01' - D'2023.01.01');
      bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      x = time / (double)PeriodSeconds(PERIOD_MN1);
      bAccount.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
      x = time / (double)PeriodSeconds(PERIOD_W1);
      bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      x = time / (double)PeriodSeconds(PERIOD_D1);
      bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
      if(!!bAccount.GetOpenCL())
        {
         if(!bAccount.BufferWrite())
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
        }
```

A timestamp for the analyzed state is also added.

Using this information, we perform a feed-forward pass through the predictive coding models and the attention cascade to transform the predictive outputs into a unified subspace.

```
      //--- Generate Latent state
      if(!RelateEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL) ||
         !ShortEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL) ||
         !ShortDecoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CNet*)GetPointer(ShortEncoder)) ||
         !LongEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)NULL) ||
         !LongDecoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CNet*)GetPointer(LongEncoder)) ||
         !LongShort.feedForward(GetPointer(LongDecoder), -1, GetPointer(ShortDecoder), -1) ||
         !PredictRelate.feedForward(GetPointer(LongShort), -1, GetPointer(RelateEncoder), -1)
        )
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
```

Note: At this stage, the dependency search Decoder is _not executed_, as it is not used in policy training or deployment.

Next, we optimize the _Critic_ to minimize the error in evaluating the _Agent_ actions. Actual actions from the selected state are retrieved from the replay buffer and passed through the _Critic_.

```
      //--- Critic
      target.Assign(Buffer[tr].States[i].action);
      target.Clip(0, 1);
      bActions.AssignArray(target);
      if(!!bActions.GetOpenCL())
         if(!bActions.BufferWrite())
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
      Critic.TrainMode(true);
      if(!Critic.feedForward(GetPointer(PredictRelate), -1, (CBufferFloat*)GetPointer(bActions)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
```

The estimated action obtained as a result of _Critic's_ feed-forward pass values initially approximate a random distribution. However, the experience replay buffer also stores the real rewards received for the Agent's actual actions during trajectory collection. Therefore, we can train the _Critic_, minimizing the error between the predicted and actual reward.

We extract the actual reward from the experience replay buffer and execute the _Critic's_ backpropagation pass.

```
      result.Assign(Buffer[tr].States[i + 1].rewards);
      target.Assign(Buffer[tr].States[i + 2].rewards);
      result = result - target * DiscFactor;
      Result.AssignArray(result);
      if(!Critic.backProp(Result, (CBufferFloat *)GetPointer(bActions), (CBufferFloat *)GetPointer(bGradient)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
```

Next, we proceed to the actual training of the _Actor's_ behavior policy. Using the collected input data, we perform a forward pass through the Actor to generate the action tensor according to the current policy.

```
      //--- Actor Policy
      if(!Actor.feedForward(GetPointer(PredictRelate), -1, (CBufferFloat*)GetPointer(bAccount)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
```

Immediately afterward, we evaluate the generated actions using the Critic.

```
      Critic.TrainMode(false);
      if(!Critic.feedForward(GetPointer(PredictRelate), -1, (CNet*)GetPointer(Actor), -1))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
```

Note: During the optimization of the _Actor's_ policy, the _Critic's_ training mode is disabled. This allows the error gradient to be propagated to the _Actor_ without altering the _Critic's_ parameters based on irrelevant data.

Actor policy training occurs in two stages. In the first stage, we evaluate the effectiveness of actual actions recorded in the experience replay buffer. If the reward is positive, we minimize the error between the predicted and actual action tensors. This trains a profitable policy in a supervised manner.

```
      if(result.Sum() >= 0)
         if(!Actor.backProp(GetPointer(bActions), (CBufferFloat*)GetPointer(bAccount), GetPointer(bGradient)) ||
            !PredictRelate.backPropGradient(GetPointer(RelateEncoder), -1, -1, false) ||
            !LongShort.backPropGradient(GetPointer(ShortDecoder), -1, -1, false) ||
            !ShortDecoder.backPropGradient((CNet *)GetPointer(ShortEncoder), -1, -1, false) ||
            !ShortEncoder.backPropGradient((CBufferFloat*)NULL) ||
            !LongDecoder.backPropGradient((CNet *)GetPointer(LongEncoder), -1, -1, false) ||
            !LongEncoder.backPropGradient((CBufferFloat*)NULL)
           )
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
```

Importantly, during this stage, the error gradient is propagated down to the predictive models. This fine-tunes them them to support the _Actor's_ policy optimization task.

Critic-guided stage: We optimize the Actor's policy by propagating the error gradient from the _Critic_. This stage adjusts the policy regardless of the actual actions' outcomes in the environment, relying solely on the _Critic's_ evaluation of the current policy. For this, we enhance the action evaluation by 1%.

```
      Critic.getResults(Result);
      for(int c = 0; c < Result.Total(); c++)
        {
         float value = Result.At(c);
         if(value >= 0)
            Result.Update(c, value * 1.01f);
         else
            Result.Update(c, value * 0.99f);
        }
```

We then pass this adjusted reward to the _Critic_ as the target and perform backpropagation, propagating the error gradient to the _Actor_. This operation produces an error gradient at the _Actor's_ output, directing actions toward higher profitability.

```
      if(!Critic.backProp(Result, (CNet *)GetPointer(Actor), LatentLayer) ||
         !Actor.backPropGradient((CBufferFloat*)GetPointer(bAccount), GetPointer(bGradient)) ||
         !PredictRelate.backPropGradient(GetPointer(RelateEncoder), -1, -1, false) ||
         !LongShort.backPropGradient(GetPointer(ShortDecoder), -1, -1, false) ||
         !ShortDecoder.backPropGradient((CNet *)GetPointer(ShortEncoder), -1, -1, false) ||
         !ShortEncoder.backPropGradient((CBufferFloat*)NULL) ||
         !LongDecoder.backPropGradient((CNet *)GetPointer(LongEncoder), -1, -1, false) ||
         !LongEncoder.backPropGradient((CBufferFloat*)NULL)
        )
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         Stop = true;
         break;
        }
```

The resulting gradient is propagated through all relevant models, similar to the first training stage.

We then update the user on the training progress and proceed to the next iteration.

```
      //---
      if(GetTickCount() - ticks > 500)
        {
         double percent = double(iter) * 100.0 / (Iterations);
         string str = StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Actor", percent, Actor.getRecentAverageError());
         str += StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Critic", percent, Critic.getRecentAverageError());
         Comment(str);
         ticks = GetTickCount();
        }
     }
```

Upon completion of all training iterations, we clear the chart comments, log the results in the journal, and initiate program termination, just as in the first training stage.

```
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Actor", Actor.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Critic", Critic.getRecentAverageError());
   ExpertRemove();
//---
  }
```

It should be noted that algorithm adjustments affected not only the model training Expert Advisors but also the environment interaction EAs. However, the adjustments to environment interaction algorithms largely mirror the _Actor's_ feed-forward pass described above and are left for independent study. Therefore, we will not go into the detailed logic of these algorithms here. I encourage you to explore their implementations independently. The full source code for all programs used in this article is included in the attachment.

### Testing

We have completed the extensive implementation of the _StockFormer_ framework using MQL5 and have reached the final stage of our work - training the models and evaluating their performance on real historical data.

As previously mentioned, the initial stage of training the predictive models utilized a dataset collected in earlier studies. This dataset comprises _EURUSD_ historical data for the entire year of 2023, on the _H1_ timeframe. All indicator parameters were set to their default values.

During predictive model training, we use only historical data describing the environment state, which is independent of the _Agent's_ behavior. This allows us to train the models without updating the training dataset. The training process continues until errors are stabilized within a narrow range.

The second training stage - optimizing the Actor's behavior policy - is performed iteratively, with periodic updates to the training dataset to reflect the current policy.

We evaluate the performance of the trained model using the _MetaTrader 5 Strategy Tester_ on historical data from January 2024. This period immediately follows the training dataset period. The results are presented below.

![](https://c.mql5.com/2/173/2655254200611__1.png)![](https://c.mql5.com/2/173/3892498836627__1.png)

During the testing period, the model executed 15 trades, with 10 closing in profit - over 66% success rate. Quite a good result. Notably, the average profitable trade is four times larger than the average loss. This results in a clear upward trend in the balance chart.

### Conclusion

Across these two articles, we explored the _StockFormer_ framework, which offers an innovative approach to training trading strategies for financial markets. _StockFormer_ combines predictive coding with reinforcement learning, enabling the development of flexible policies that capture dynamic dependencies among multiple assets and forecast their behavior both in the short and long term.

The three-branch predictive coding structure in _StockFormer_ allows the extraction of latent representations reflecting short-term trends, long-term changes, and inter-asset relationships. Integration of these representations is achieved via a cascade of multi-head attention modules, creating a unified state space for optimizing trading decisions.

In the practical part, we implemented the key components of the framework in _MQL5_, trained the models, and tested them on real historical data. The experimental results confirm the effectiveness of the proposed approaches. Nevertheless, applying these models in live trading requires training on a larger historical dataset and comprehensive further testing.

#### References

- [StockFormer: Learning Hybrid Trading Machines with Predictive Coding](https://www.mql5.com/go?link=https://www.ijcai.org/proceedings/2023/0530.pdf "StockFormer: Learning Hybrid Trading Machines with Predictive Coding")
- [Other articles from this series](https://www.mql5.com/en/search#!keyword=Neural%20networks%20made%20easy&module=mql5_module_articles "https://www.mql5.com/en/search#!keyword=Neural%20networks%20made%20easy&module=mql5_module_articles")

#### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | Research.mq5 | Expert Advisor | Expert Advisor for collecting samples |
| 2 | ResearchRealORL.mq5 | Expert Advisor | Expert Advisor for collecting samples using the Real-ORL method |
| 3 | Study1.mq5 | Expert Advisor | Predictive Learning Expert Advisor |
| 4 | Study2.mq5 | Expert Advisor | Policy Training Expert Advisor |
| 5 | Test.mq5 | Expert Advisor | Model Testing Expert Advisor |
| 6 | Trajectory.mqh | Class library | System state and model architecture description structure |
| 7 | NeuroNet.mqh | Class library | A library of classes for creating a neural network |
| 8 | NeuroNet.cl | Code Base | OpenCL program code library |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/16713](https://www.mql5.com/ru/articles/16713)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16713.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/16713/mql5.zip "Download MQL5.zip")(2253.87 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Final Part)](https://www.mql5.com/en/articles/17241)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Chimera)](https://www.mql5.com/en/articles/17210)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model (Final Part)](https://www.mql5.com/en/articles/17157)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model](https://www.mql5.com/en/articles/17142)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Final Part)](https://www.mql5.com/en/articles/17104)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Hidformer)](https://www.mql5.com/en/articles/17069)
- [Neural Networks in Trading: Memory Augmented Context-Aware Learning for Cryptocurrency Markets (Final Part)](https://www.mql5.com/en/articles/16993)

**[Go to discussion](https://www.mql5.com/en/forum/496917)**

![MQL5 Wizard Techniques you should know (Part 81):  Using Patterns of Ichimoku and the ADX-Wilder with Beta VAE Inference Learning](https://c.mql5.com/2/173/19781-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 81): Using Patterns of Ichimoku and the ADX-Wilder with Beta VAE Inference Learning](https://www.mql5.com/en/articles/19781)

This piece follows up ‘Part-80’, where we examined the pairing of Ichimoku and the ADX under a Reinforcement Learning framework. We now shift focus to Inference Learning. Ichimoku and ADX are complimentary as already covered, however we are going to revisit the conclusions of the last article related to pipeline use. For our inference learning, we are using the Beta algorithm of a Variational Auto Encoder. We also stick with the implementation of a custom signal class designed for integration with the MQL5 Wizard.

![Automating Trading Strategies in MQL5 (Part 36): Supply and Demand Trading with Retest and Impulse Model](https://c.mql5.com/2/173/19674-automating-trading-strategies-logo__2.png)[Automating Trading Strategies in MQL5 (Part 36): Supply and Demand Trading with Retest and Impulse Model](https://www.mql5.com/en/articles/19674)

In this article, we create a supply and demand trading system in MQL5 that identifies supply and demand zones through consolidation ranges, validates them with impulsive moves, and trades retests with trend confirmation and customizable risk parameters. The system visualizes zones with dynamic labels and colors, supporting trailing stops for risk management.

![MetaTrader 5 Machine Learning Blueprint (Part 3): Trend-Scanning Labeling Method](https://c.mql5.com/2/172/19253-metatrader-5-machine-learning-logo.png)[MetaTrader 5 Machine Learning Blueprint (Part 3): Trend-Scanning Labeling Method](https://www.mql5.com/en/articles/19253)

We have built a robust feature engineering pipeline using proper tick-based bars to eliminate data leakage and solved the critical problem of labeling with meta-labeled triple-barrier signals. This installment covers the advanced labeling technique, trend-scanning, for adaptive horizons. After covering the theory, an example shows how trend-scanning labels can be used with meta-labeling to improve on the classic moving average crossover strategy.

![Developing Advanced ICT Trading Systems: Implementing Signals in the Order Blocks Indicator](https://c.mql5.com/2/100/Final_Logo.png)[Developing Advanced ICT Trading Systems: Implementing Signals in the Order Blocks Indicator](https://www.mql5.com/en/articles/16268)

In this article, you will learn how to develop an Order Blocks indicator based on order book volume (market depth) and optimize it using buffers to improve accuracy. This concludes the current stage of the project and prepares for the next phase, which will include the implementation of a risk management class and a trading bot that uses signals generated by the indicator.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/16713&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062549686688064587)

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