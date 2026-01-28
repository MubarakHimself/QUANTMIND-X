---
title: Neural networks made easy (Part 25): Practicing Transfer Learning
url: https://www.mql5.com/en/articles/11330
categories: Integration, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:12:26.478402
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/11330&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071654721002875912)

MetaTrader 5 / Integration


### Contents

- [Introduction](https://www.mql5.com/en/articles/11330#para1)
- [1\. General test preparation issues](https://www.mql5.com/en/articles/11330#para2)
- [2\. Creating an Expert Advisor for testing](https://www.mql5.com/en/articles/11330#para3)
- [3\. Creating models for testing](https://www.mql5.com/en/articles/11330#para4)
- [4\. Testing results](https://www.mql5.com/en/articles/11330#para5)
- [Conclusion](https://www.mql5.com/en/articles/11330#para6)
- [List of references](https://www.mql5.com/en/articles/11330#para7)
- [Programs used in the article](https://www.mql5.com/en/articles/11330#para8)

### Introduction

We continue to study the Transfer Learning technology. In the previous two articles, we created a tool for creating and editing neural network models. This tool will help us transfer part of the pre-trained model to a new model and supplement it with new decision layers. Potentially, this approach should assist us in quicker training of the model created in this way for solving new problems. In this article, we will evaluate the benefits of this approach in practice. We will also check the usability of our tool.

### 1\. General test preparation issues

In this article, we want to evaluate the benefits of using the Transfer Learning technology. The best way is to compare the learning process of two models for solving one problem. For this purpose, we will use one "pure" model initiated by random weights. The second model will be created using the Transfer Learning technology.

We can use the search for fractals as the problem, just like we deed when testing all previous models in supervised learning methods. But what will we use as the donor model for Transfer Learning? Let's get back to autoencoders. We used them as donors for Transfer Learning. When studying autoencoders, we created and trained two models of variational autoencoders. In the first model, the encoder was built using fully connected neural layers. In the second one, we used an encoder based on recurrent LSTM blocks. This time we can use both models as donors. So, we can also test the efficiency of these two approaches.

Thus, we have made the first fundamental decision in the preparation of the upcoming test: as donor models, we will use variational autoencoders which we trained when studying the relevant topics.

The second conceptual question is how we will test models. We must create the most equal conditions for all models. Only then we can exclude the influence of other factors and purely evaluate the influence of the model design features.

The key point here is the "design features". How do we evaluate the benefits of Transfer Learning in essentially different models? In fact, the situation is not unambiguous. Let's remember what the autoencoder learns. Its architecture is such that we expect to receive initial data at the output of the model. The encoder compresses the original data to a "bottleneck" of the latent state, and then the decoder restores the data. That is, we simply compress the original data. In such a case, models can be considered to have identical architectures if the architecture of the model after the borrowed encoder block is equal to the architecture of the reference model.

On the other hand, along with data compression, the encoder performs data preprocessing. It picks out some features and zeroes others. In this interpretation, to align the architecture of two models, we need to create an exact copy of the model but already initialized with random weights.

Since this is still ambiguous, we will test both approaches to solving the problem.

The next question concerns the testing tool. Previously, we created a separate Expert Advisor (EA) for testing each model, because each time we described and created the model in the EA's initialization block. Now the situation is different. We have created a universal tool for creating models. Using it, we can create various model architectures and save them to a file. Then we can upload the created model to any EA to have it trained or to use it.

Therefore, now we can create one EA, in which we will train all models. Thus, we will provide the most equal conditions to test the models.

Now, we have to decide on the testing environment. That is, on which data we are going to test the models. The answer is clear: to train the models, we will use the environment similar to that used to train autoencoders. Neural networks are very sensitive to the source data, and they can work correctly only with the data on which they are trained. Therefore, to use the Transfer Learning technology, we must use the source data similar to the training sample of the donor model.

Now that we have decided on all the key issues, we can move on to preparing for testing.

### 2\. Creating an Expert Advisor for testing

The preparatory work starts by creating an EA to test the models. For this purpose, let us create an EA template "check\_net.mq5". First, include libraries in the template:

- NeuroNet.mqh — our library for creating neural networks
- SymbolInfo.mqh — standard library for accessing trading symbol data
- Oscilators.mqh — standard library for working with oscillators

Also, we declare here an enumeration for convenient work with signals.

```
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "..\..\NeuroNet_DNG\NeuroNet.mqh"
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//---
enum ENUM_SIGNAL
  {
   Sell = -1,
   Undefine = 0,
   Buy = 1
  };
```

The next step is to declare the global variables of the EA. Specify here the model file, the working timeframe and the model training period. Also, we will display all the parameters of indicators used. Indicator parameters will be split into groups to make the EA menu readable.

```
//+------------------------------------------------------------------+
//|   input parameters                                               |
//+------------------------------------------------------------------+
input int                  StudyPeriod =  2;            //Study period, years
input string               FileName = "EURUSD_i_PERIOD_H1_test_rnn";
ENUM_TIMEFRAMES            TimeFrame   =  PERIOD_CURRENT;
//---
input group                "---- RSI ----"
input int                  RSIPeriod   =  14;            //Period
input ENUM_APPLIED_PRICE   RSIPrice    =  PRICE_CLOSE;   //Applied price
//---
input group                "---- CCI ----"
input int                  CCIPeriod   =  14;            //Period
input ENUM_APPLIED_PRICE   CCIPrice    =  PRICE_TYPICAL; //Applied price
//---
input group                "---- ATR ----"
input int                  ATRPeriod   =  14;            //Period
//---
input group                "---- MACD ----"
input int                  FastPeriod  =  12;            //Fast
input int                  SlowPeriod  =  26;            //Slow
input int                  SignalPeriod =  9;            //Signal
input ENUM_APPLIED_PRICE   MACDPrice   =  PRICE_CLOSE;   //Applied price
```

Next, declare the instances of the objects used. The use of dynamic objects has been avoided where possible. This will simplify the code a little by removing unnecessary operations related to the creation of objects and checking of their relevance. Object naming is consistent with object contents. This will minimize variable confusion and will improve code readability.

```
CSymbolInfo          Symb;
CNet                 Net;
CBufferFloat        *TempData;
CiRSI                RSI;
CiCCI                CCI;
CiATR                ATR;
CiMACD               MACD;
CBufferFloat         Fractals;
```

Also, declare the global variables of the EA. Now I will describe the functionality of each of them. We will see their purposes while analyzing the algorithms of the EA's functions.

```
uint                 HistoryBars =  40;            //Depth of history
MqlRates             Rates[];
float                dError;
float                dUndefine;
float                dForecast;
float                dPrevSignal;
datetime             dtStudied;
bool                 bEventStudy;
```

You can see here a variable for the amount of source data in bars, which previously was specified in the EA's external parameters. Hiding this parameter and using it as a global variable is a forced measure. Previously, we described the model architecture in the EA initialization function. So, this parameter was one of the hyperparameters of the model which the user specified at EA start. In this article, we are going to use previously created models. The analyzed history depth parameter must match correspond to the loaded model. But since the user can use a model "blindly", not knowing this parameter, we run the risk of mismatch between the specified parameter and the loaded model. To eliminate this risk, I decided to recalculate the parameter according to the size of the source data layer of the loaded model.

Let's move on to considering the algorithms of the EA functions. We will start with the EA initialization method — OnInit. In the method body, we first load the model from the file specified in the EA parameters. Two moments here differ from the same operations in previously considered EAs.

First, since we do not use dynamic pointers, we do not need to create a new instance of the model object. For the same reason, we do not need to check the pointer validity.

Second, if the model could not be read from the file, inform the user and exit the function with the INIT\_PARAMETERS\_INCORRECT result. Also, we close the EA. As mentioned above, we are creating an EA to work with several previously created models. So, there is no default model. If there is no model, there is nothing to train. So, further EA operation makes no sense. Therefore, inform the user and terminate the EA operation.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ResetLastError();
   if(!Net.Load(FileName + ".nnw", dError, dUndefine, dForecast, dtStudied, false))
     {
      printf("%s - %d -> Error of read %s prev Net %d", __FUNCTION__, __LINE__, FileName + ".nnw", GetLastError());
      return INIT_PARAMETERS_INCORRECT;
     }
```

After successfully loading the model, calculate the size of the analyzed history depth and save the resulting value in the HistoryBars variable. Also, we check the size of the results layer. It should contain 3 neurons according to the number of possible results of the model.

```
   if(!Net.GetLayerOutput(0, TempData))
      return INIT_FAILED;
   HistoryBars = TempData.Total() / 12;
   Net.getResults(TempData);
   if(TempData.Total() != 3)
      return INIT_PARAMETERS_INCORRECT;
```

If all the checks are successful, proceed to initializing objects for working with indicators.

```
   if(!Symb.Name(_Symbol))
      return INIT_FAILED;
   Symb.Refresh();

   if(!RSI.Create(Symb.Name(), TimeFrame, RSIPeriod, RSIPrice))
      return INIT_FAILED;

   if(!CCI.Create(Symb.Name(), TimeFrame, CCIPeriod, CCIPrice))
      return INIT_FAILED;

   if(!ATR.Create(Symb.Name(), TimeFrame, ATRPeriod))
      return INIT_FAILED;

   if(!MACD.Create(Symb.Name(), TimeFrame, FastPeriod, SlowPeriod, SignalPeriod, MACDPrice))
      return INIT_FAILED;
```

Remember to control the execution of all operations.

Once all objects are initialized, generate a custom event, to which we will transfer control to the model training method. Write the result of generating a custom event to the bEventStudy variable which will act as a flag for starting the model training process.

The custom event generation operation allows completing the EA initialization method. In parallel, we can analyze the model training process without waiting for the new tick. Thus, we make the beginning of the model learning process independent of market volatility.

```
   bEventStudy = EventChartCustom(ChartID(), 1, (long)MathMax(0, MathMin(iTime(Symb.Name(), PERIOD_CURRENT,
                                  (int)(100 * Net.recentAverageSmoothingFactor * (dForecast >= 70 ? 1 : 10))), dtStudied)),

                                  0, "Init");
//---
   return(INIT_SUCCEEDED);
  }
```

In the EA deinitialization method, we delete the only dynamic object used in the EA. This is because we eliminated the use of other dynamic objects.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   if(CheckPointer(TempData) != POINTER_INVALID)
      delete TempData;
  }
```

All chart events are processed in the OnChartEvent function, including our custom event. Therefore, in this function, we are waiting for the occurrence of a user event, which can be identified by its ID. Custom event IDs start with 1000. When generating a custom event, we gave it an ID of 1. So, in this function we should receive an event with the identifier 1001. When such an event occurs, we call the model training procedure — Train.

```
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   if(id == 1001)
      Train(lparam);
  }
```

Let's take a closer look at the organization of the algorithm of probably the main function of our EA - Train for model training. In the parameters, this function receives the only value which is the training period start date. We first check to make sure this date is not outside the training period specified by the user in the EA's external parameters. If the received date does not correspond to the period specified by the user, then we shift the date to the beginning of the specified training period.

```
void Train(datetime StartTrainBar = 0)
  {
   int count = 0;
//---
   MqlDateTime start_time;
   TimeCurrent(start_time);
   start_time.year -= StudyPeriod;
   if(start_time.year <= 0)
      start_time.year = 1900;
   datetime st_time = StructToTime(start_time);
   dtStudied = MathMax(StartTrainBar, st_time);
   ulong last_tick = 0;
```

Next, prepare the local variables.

```
   double prev_er = DBL_MAX;
   datetime bar_time = 0;
   bool stop = IsStopped();
```

Then load historical data. Here we load quotes alongside with indicator data. It is important to keep the indicator buffers and loaded quotes synchronous. Therefore, we first download quotes for the specified period, determine the number of loaded bars and load the same period for all indicators used.

```
   int bars = CopyRates(Symb.Name(), TimeFrame, st_time, TimeCurrent(), Rates);
   if(!RSI.BufferResize(bars) || !CCI.BufferResize(bars) || !ATR.BufferResize(bars) || !MACD.BufferResize(bars))
     {
      ExpertRemove();
      return;
     }
   if(!ArraySetAsSeries(Rates, true))
     {
      ExpertRemove();
      return;
     }
   RSI.Refresh(OBJ_ALL_PERIODS);
   CCI.Refresh(OBJ_ALL_PERIODS);
   ATR.Refresh(OBJ_ALL_PERIODS);
   MACD.Refresh(OBJ_ALL_PERIODS);
```

Once the training sample is loaded, we will take the last 300 elements from the total number of training sample elements for validation after each training epoch. After that, create a system of learning process loops. The outer loop will count the training epochs and control whether the model training process should continue. Update the flags in the loop body:

- prev\_er — model error on the previous epoch
- stop — generating an event of program termination by the user

```
   MqlDateTime sTime;
   int total = (int)(bars - MathMax(HistoryBars, 0) - 300);
   do
     {
      prev_er = dError;
      stop = IsStopped();
```

In a nested loop, iterate over the elements of the training sample and feed them in turn into the neural network. Since we are going to use recurrent models which are sensitive to the sequence of input data, we have to avoid the selection of a random next element of the sequence. Instead, we will use the historical sequence of elements.

We immediately check the sufficiency of the data from the current element to draw up the pattern. If the data is not enough, move on to the next element.

```
      for(int it = total; it > 1 && !stop; t--)
        {
         TempData.Clear();
         int i = it + 299;
         int r = i + (int)HistoryBars;
         if(r > bars)
            continue;
```

If data is enough, form a pattern to feed into the model. We also control the availability of data in indicator buffers. If the indicator values are not defined, move on to the next element.

```
         for(int b = 0; b < (int)HistoryBars; b++)
           {
            int bar_t = r - b;
            float open = (float)Rates[bar_t].open;
            TimeToStruct(Rates[bar_t].time, sTime);
            float rsi = (float)RSI.Main(bar_t);
            float cci = (float)CCI.Main(bar_t);
            float atr = (float)ATR.Main(bar_t);
            float macd = (float)MACD.Main(bar_t);
            float sign = (float)MACD.Signal(bar_t);
            if(rsi == EMPTY_VALUE || cci == EMPTY_VALUE || atr == EMPTY_VALUE || macd == EMPTY_VALUE || sign == EMPTY_VALUE)
               continue;
            //---
            if(!TempData.Add((float)Rates[bar_t].close - open) || !TempData.Add((float)Rates[bar_t].high - open) ||
               !TempData.Add((float)Rates[bar_t].low - open) || !TempData.Add((float)Rates[bar_t].tick_volume / 1000.0f) ||
               !TempData.Add(sTime.hour) || !TempData.Add(sTime.day_of_week) || !TempData.Add(sTime.mon) ||
               !TempData.Add(rsi) || !TempData.Add(cci) || !TempData.Add(atr) || !TempData.Add(macd) || !TempData.Add(sign))
               break;
           }
         if(TempData.Total() < (int)HistoryBars * 12)
            continue;
```

After a pattern is formed successfully, call the feed forward pass method of the model. Immediately request the result of the feed forward pass.

```
         Net.feedForward(TempData, 12, true);
         Net.getResults(TempData);
```

Apply the SortMax function to the model results in order to convert the obtain values to the probabilities.

```
         float sum = 0;
         for(int res = 0; res < 3; res++)
           {
            float temp = exp(TempData.At(res));
            sum += temp;
            TempData.Update(res, temp);
           }
         for(int res = 0; (res < 3 && sum > 0); res++)
            TempData.Update(res, TempData.At(res) / sum);
         //---
         switch(TempData.Maximum(0, 3))
           {
            case 1:
               dPrevSignal = (TempData[1] != TempData[2] ? TempData[1] : 0);
               break;
            case 2:
               dPrevSignal = -TempData[2];
               break;
            default:
               dPrevSignal = 0;
               break;
           }
```

After that, display information about the learning process on the chart.

```
         if((GetTickCount64() - last_tick) >= 250)
           {
            string s = StringFormat("Study -> Era %d -> %.2f -> Undefine %.2f%% foracast %.2f%%\n %d of %d -> %.2f%% \n
                                     Error %.2f\n%s -> %.2f ->> Buy %.5f - Sell %.5f - Undef %.5f", count, dError,
                                     dUndefine, dForecast, total - it - 1, total,
                                     (double)(total - it - 1.0) / (total) * 100, Net.getRecentAverageError(),
                                      EnumToString(DoubleToSignal(dPrevSignal)), dPrevSignal, TempData[1], TempData[2], TempData[0]);
            Comment(s);
            last_tick = GetTickCount64();
           }
```

The feed forward pass in the model training process is followed by backpropagation. First, we create the target values and feed them into the backpropagation method. Also, we will immediately calculate the learning process statistics.

```
         stop = IsStopped();
         if(!stop)
           {
            TempData.Clear();
            bool sell = (Rates[i - 1].high <= Rates[i].high && Rates[i + 1].high < Rates[i].high);
            bool buy = (Rates[i - 1].low >= Rates[i].low && Rates[i + 1].low > Rates[i].low);
            TempData.Add(!(buy || sell));
            TempData.Add(buy);
            TempData.Add(sell);
            Net.backProp(TempData);
            ENUM_SIGNAL signal = DoubleToSignal(dPrevSignal);
            if(signal != Undefine)
              {
               if((signal == Sell && sell) || (signal == Buy && buy))
                  dForecast += (100 - dForecast) / Net.recentAverageSmoothingFactor;
               else
                  dForecast -= dForecast / Net.recentAverageSmoothingFactor;
               dUndefine -= dUndefine / Net.recentAverageSmoothingFactor;
              }
            else
              {
               if(!(buy || sell))
                  dUndefine += (100 - dUndefine) / Net.recentAverageSmoothingFactor;
              }
           }
        }
```

This completes the nested loop over the training sample elements within one epoch of model training. After that, we will implement validation to evaluate the model behavior on data that is not included in the training sample. To do this, run a similar loop over the last 300 elements but with a feed forward pass. During validation, there is no need to execute the backpropagation pass and to update the weight matrix.

```
      count++;
      for(int i = 0; i < 300; i++)
        {
         TempData.Clear();
         int r = i + (int)HistoryBars;
         if(r > bars)
            continue;
         //---
         for(int b = 0; b < (int)HistoryBars; b++)
           {
            int bar_t = r - b;
            float open = (float)Rates[bar_t].open;
            TimeToStruct(Rates[bar_t].time, sTime);
            float rsi = (float)RSI.Main(bar_t);
            float cci = (float)CCI.Main(bar_t);
            float atr = (float)ATR.Main(bar_t);
            float macd = (float)MACD.Main(bar_t);
            float sign = (float)MACD.Signal(bar_t);
            if(rsi == EMPTY_VALUE || cci == EMPTY_VALUE || atr == EMPTY_VALUE || macd == EMPTY_VALUE || sign == EMPTY_VALUE)
               continue;
            //---
            if(!TempData.Add((float)Rates[bar_t].close - open) || !TempData.Add((float)Rates[bar_t].high - open) ||
               !TempData.Add((float)Rates[bar_t].low - open) || !TempData.Add((float)Rates[bar_t].tick_volume / 1000.0f) ||
               !TempData.Add(sTime.hour) || !TempData.Add(sTime.day_of_week) || !TempData.Add(sTime.mon) ||
               !TempData.Add(rsi) || !TempData.Add(cci) || !TempData.Add(atr) || !TempData.Add(macd) || !TempData.Add(sign))
               break;
           }
         if(TempData.Total() < (int)HistoryBars * 12)
            continue;
         Net.feedForward(TempData, 12, true);
         Net.getResults(TempData);
         //---
         float sum = 0;
         for(int res = 0; res < 3; res++)
           {
            float temp = exp(TempData.At(res));
            sum += temp;
            TempData.Update(res, temp);
           }
         for(int res = 0; (res < 3 && sum > 0); res++)
            TempData.Update(res, TempData.At(res) / sum);
         //---
         switch(TempData.Maximum(0, 3))
           {
            case 1:
               dPrevSignal = (TempData[1] != TempData[2] ? TempData[1] : 0);
               break;
            case 2:
               dPrevSignal = (TempData[1] != TempData[2] ? -TempData[2] : 0);
               break;
            default:
               dPrevSignal = 0;
               break;
           }
```

After the validation feed forward pass, output the signals of the model on the chart to enable a visual assessment of its performance.

```
         if(DoubleToSignal(dPrevSignal) == Undefine)
            DeleteObject(Rates[i].time);
         else
            DrawObject(Rates[i].time, dPrevSignal, Rates[i].high, Rates[i].low);
        }
```

At the end of each epoch, save the current state of the model. Here we will also add the current model error to the file to control the dynamics of the learning process.

```
      if(!stop)
        {
         dError = Net.getRecentAverageError();
         Net.Save(FileName + ".nnw", dError, dUndefine, dForecast, Rates[0].time, false);
         printf("Era %d -> error %.2f %% forecast %.2f", count, dError, dForecast);
         int h = FileOpen(FileName + ".csv", FILE_READ | FILE_WRITE | FILE_CSV);
         if(h != INVALID_HANDLE)
           {
            FileSeek(h, 0, SEEK_END);
            FileWrite(h, eta, count, dError, dUndefine, dForecast);
            FileFlush(h);
            FileClose(h);
           }
        }
     }
   while(!(dError < 0.01 && (prev_er - dError) < 0.01) && !stop);
```

Next, we need to evaluate the change in the model error over the last training epoch and decide whether to continue training. If we decide to continue training, then the loop iterations will be repeated for the new learning epoch.

After completing the model training process, clear the comment area on the chart and initialize the EA completion. By now, the EA has completed the model training task and there is no need to keep it further in the memory.

```
   Comment("");
   ExpertRemove();
  }
```

Auxiliary functions for displaying labels on the chart and deleting are exactly the ones that we used in previously considered EAs, so I will not repeat their algorithms here. The full code of all EA functions can be found in the attachment.

### 3\. Creating models for testing

Now that we have created the model testing tool, we need to prepare the base for testing. I.e., we need to create the models that will be trained. o programming is needed here, as we have implemented the required coding in the previous two articles. Now we will take advantage of the results and create models using our tool.

So, we run the previously created NetCreator EA. In it, open the pre-trained autoencoder model using the recurrent encoder based on LSTM blocks. Previously, we saved it in the "EURUSD\_i\_PERIOD\_H1\_rnn\_vae.nnw" file. We will only use the encoder from this model. In the left block of the pre-trained model, find the latent state layer of the variational autoencoder (VAE). In my case, it is the eighth. So, I will only copy the first seven neural layers of the donor model.

The tool provides three ways to select the required number of layers for copying. You can use buttons in the "Transfer Layers" area or use the arrow keys ↑ and ↓. Alternatively, you can simply click on the description of the last copied later in the donor model description.

Simultaneously with the change in the number of copied layers, the description of the created model in the right block of the tool also changes. I think this is convenient and informative. You can instantly see how your actions affect the architecture of the model being created.

Next, we need to supplement the new model with several neural decision-making layers for a specific learning task. I tried not to complicate this part, since the main purpose of these tests is to evaluate the effectiveness of the approaches. I have added two fully connected layers of 500 elements and a hyperbolic tangent as an activation function.

Adding new neural layers turned out to be quite a simple task. First, select the type of neural layer. A fully connected neural layer corresponds to "Dense". Specify the number of neurons in the layer, the activation function, and the parameter update method. If you select a different type of neural layer, you will fill in the appropriate fields. After specifying all the necessary data, click "ADD LAYER".

Another convenience is that if you need to add several identical neural layers, there is no need to re-enter the data. Simply click ADD LAYER once again. This is what I used. To add the second layer, I did not enter any data but simply clicked on the new layer adding button.

The results layer is also fully connected and contains three elements, in accordance with the requirements of the EA created above. Sigmoid is used as the activation function for the results layer.

Our previous neural layers were also fully connected. So, we can only change the number of neurons and the activation function. Then we add the layer to our model.

Now, save the new model to a file. To do this, press the SAVE MODEL button and specify the file name of the new model EURUSD\_i\_PERIOD\_H1\_test\_rnn.nnw. Note that you can specify the file name without the extension. The right extension will be added automatically.

The entire model creation process is visualized in the gif below.

![Using the model creation tool](https://c.mql5.com/2/48/NetCreator.gif)

The first model is ready. Now, let's move on to creating the second model. As a donor for the second model, let's load the variational autoencoder with a fully connected encoder from the EURUSD\_i\_PERIOD\_H1\_vae.nnw file. Here comes another surprise. After loading the new donor model, we did not remove the added neural layers. So, they were automatically added to the loaded model. We only need to select the number of neural layers to copy from the donor model to the new model. So, our new model is ready.

Based on the last autoencoder model, I created not one but two models. The first model is analogous to the first one. I used the encoder from the donor model and added the previously created three layers. For the second model, I took only the source data layer and the batch normalization layer from the donor model. Then I added the same three fully connected neural layers to them. The last model will serve as a guide for training the new model. I decided that the pre-trained batch normalization layer would be used for preparing raw input data. This should increase the convergence of the new model. Furthermore, we eliminate data compression. We can assume that the last model is completely filled with random weights.

As we have discussed above, there are different ways to evaluate the impact of the architecture of a pre-trained model. That is why I have created another model for testing. I used architectures of the newly created model using the autoencoder with LSTM blocks and completely replicated it in the new model. But this time, I didn't copy the encoder from the donor model. Thus, I got a completely identical model architecture, but initialized with random weights.

### 4\. Testing results

Now that we've created all the models we need for our tests, we'll move on to training them.

We trained the models using the supervised learning, keeping the previously used training parameters. The models were trained in a time interval for the last two years, using EURUSD on the H1 timeframe. Indicators were used with default parameters.

For the purity of the experiment, all models were trained simultaneously in one terminal on different charts.

I must say that the simultaneous training of several models is not desirable. This significantly reduces the learning rate of each of them. OpenCL is used in the models to parallelize the calculation process and to make use of the available resources. During parallel training of multiple models, the available resources are shared between all models. So, each of them has access to limited resources. This increases learning time. But this time this was done intentionally, to ensure similar conditions while training the models.

#### Test 1

For the first test, we used two models with pre-trained encoders and one small fully connected model with a borrowed batch normalization layer and 2 fully connected hidden layers.

The model testing results are shown in the graph below.

![Comparison of model learning dynamics](https://c.mql5.com/2/48/TransferLearning_test.png)

As you can see in the presented graph, the best performance was shown by the model with a pre-trained recurrent encoder. Its error decreased at a significantly faster rate practically from the first training epochs.

The model with a fully connected encoder also showed error reduction during the learning process, but at a slower rate.

A fully connected model with two hidden layers, initialized with random values, look like it hasn't been trained at all. According to the presented graph, it seems that the error is stuck in place.

![Fully connected model error dynamics](https://c.mql5.com/2/48/BlancModel.png)

Upon closer examination, we can notice a tendency to error reduction. Although this reduction occurs at a much slower rate. Obviously, such a model is too simple for solving such problems.

Based on this, we can conclude that the performance of the model is still greatly influenced by the processing of the initial data by a pre-trained encoder. The architecture of such an encoder has a significant impact on the operation of the entire model.

I would like to separately mention the model training rate. Of course, the simplest model showed the lowest time for passing one epoch. But the learning rate of a model with the recurrent encoder was very close to that. In my opinion, this was influenced by a number of factors.

First of all, the architecture of the recurrent model allowed the reduction of the analyzed data window by 4 times. Therefore, the number of interneuronal connections was also reduced. As a result, the cost of their processing was reduced. At the same time, the recurrent architecture implies additional resource costs for the backpropagation pass. but we disabled the backpropagation pass for pre-trained neural layers. This ultimately reduced model retraining costs.

The model with a fully connected encoder showed slower learning rates.

#### Test 2

In the second test, we decided to minimize the architectural differences between the models and train two recurrent models with the same architecture. One model uses a pre-trained recurrent encoder. The second model is fully initialized with random weights. The same parameters that we used in the first test were used to train these models.

Testing results are shown in the chart below. As you can see, the pre-trained model started with a smaller error. But soon the second model caught up and further their values were quite close. This confirms the earlier conclusion that the encoder architecture has a significant impact on the performance of the entire model.

![Comparison of learning dynamics of recurrent models](https://c.mql5.com/2/48/LSTM.png)

Pay attention to the learning rates. The pre-trained model required six times less time to pass one epoch. Of course, this is the pure time, without taking into account the autoencoder training.

### Conclusion

Based on the above work, we can conclude that the use of the Transfer Learning technology provides a number of advantages. First of all, this technology really works. Its application enables the reuse of previously trained model blocks to solve new problems. The only condition is the unity of the initial data. The use of pre-trained blocks on non-proper input data will not work.

The use of technology reduces the new model training time. However, please note that we measured pure testing time, not including autoencoder pre-training. Probably, if we add the time spent on training the autoencoder, the time will be equal. Or maybe, due to a more complex architecture of the decoder, training of the "pure" model can be even faster. Therefore, the use of Transfer Learning can be justified when one block is supposed to be used to solve various problems. Also, it can be suited when training the model as a whole is not possible for some reason. For example, the model can be very complex, and the error gradient decays during the learning process and does not reach all layers.

Also, the technology can be applicable when searching for a best suiting model, when we gradually complicate the model in search of the optimal error value.

### List of references

1. [Neural networks made easy (Part 20): Autoencoders](https://www.mql5.com/en/articles/11172)
2. [Neural networks made easy (Part 21): Variational autoencoders (VAE)](https://www.mql5.com/en/articles/11206)
3. [Neural networks made easy (Part 22): Unsupervised learning of recurrent models](https://www.mql5.com/en/articles/11245)
4. [Neural networks made easy (Part 23): Building a tool for Transfer Learning](https://www.mql5.com/en/articles/11273)
5. [Neural networks made easy (Part 24): Improving the tool for Transfer Learning](https://www.mql5.com/en/articles/11306)

### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | check\_net.mq5 | EA | EA for additional training of models |
| 2 | NetCreator.mq5 | EA | Model building tool |
| 3 | NetCreatotPanel.mqh | Class library | Class library for creating the tool |
| 4 | NeuroNet.mqh | Class library | A library of classes for creating a neural network |
| 5 | NeuroNet.cl | Code Base | OpenCL program code library |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/11330](https://www.mql5.com/ru/articles/11330)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11330.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/11330/mql5.zip "Download MQL5.zip")(78.84 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Neural Networks in Trading: Hybrid Graph Sequence Models (GSM++)](https://www.mql5.com/en/articles/17279)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Final Part)](https://www.mql5.com/en/articles/17241)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Chimera)](https://www.mql5.com/en/articles/17210)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model (Final Part)](https://www.mql5.com/en/articles/17157)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model](https://www.mql5.com/en/articles/17142)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Final Part)](https://www.mql5.com/en/articles/17104)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Hidformer)](https://www.mql5.com/en/articles/17069)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/435606)**
(2)


![UlrichG](https://c.mql5.com/avatar/avatar_na2.png)

**[UlrichG](https://www.mql5.com/en/users/ulrichg)**
\|
28 Feb 2023 at 12:24

Hi dimitry,

I really appreciate this article series very much! Thank you for that! But please help with this problem:

If I load the file "EURUSD\_PERIOD\_H1\_rnn\_vae.nn" as mentioned in this article, I get the message "Error of load model" and "The file is damaged":

![](https://c.mql5.com/3/401/3506319004563.png)

If I [trace](https://www.mql5.com/en/docs/matrix/matrix_characteristics/matrix_trace " MQL5 Documentation: function Trace"), I find the loading fail in this line in NeuroNet.mqh:

![](https://c.mql5.com/3/401/3846952970293.png)

If I load the model from part 23, named "EURUSD\_i\_PERIOD\_H1\_test\_rnn.nnw" it seems to work, but this model only has two layers. This is not the right one. Did I miss something??

![Dmitriy Gizlyk](https://c.mql5.com/avatar/2014/8/53E8CB77-1C48.png)

**[Dmitriy Gizlyk](https://www.mql5.com/en/users/dng)**
\|
28 Feb 2023 at 12:43

**UlrichG [#](https://www.mql5.com/en/forum/435606#comment_45280591):**

If I load the file "EURUSD\_PERIOD\_H1\_rnn\_vae.nn" as mentioned in this article, I get the message "Error of load model" and "The file is damaged":

If I load the model from part 23, named "EURUSD\_i\_PERIOD\_H1\_test\_rnn.nnw" it seems to work, but it only has two layers. This is not the right one. Did I miss something??

Hi,

For load file "EURUSD\_PERIOD\_H1\_rnn\_vae.nnw" you need recompilation NetCreator with new NeuroNet.mqh library. In last model we replace CBufferDouble to CBufferFloat. And add some types of layer.

You can load last version of files [hear](https://www.mql5.com/ru/articles/download/11833/mql5.zip).


![DoEasy. Controls (Part 18): Functionality for scrolling tabs in TabControl](https://c.mql5.com/2/48/MQL5-avatar-doeasy-library-2__6.png)[DoEasy. Controls (Part 18): Functionality for scrolling tabs in TabControl](https://www.mql5.com/en/articles/11454)

In this article, I will place header scrolling control buttons in TabControl WinForms object in case the header bar does not fit the size of the control. Besides, I will implement the shift of the header bar when clicking on the cropped tab header.

![Developing a trading Expert Advisor from scratch (Part 28): Towards the future (III)](https://c.mql5.com/2/48/development__4.png)[Developing a trading Expert Advisor from scratch (Part 28): Towards the future (III)](https://www.mql5.com/en/articles/10635)

There is still one task which our order system is not up to, but we will FINALLY figure it out. The MetaTrader 5 provides a system of tickets which allows creating and correcting order values. The idea is to have an Expert Advisor that would make the same ticket system faster and more efficient.

![DoEasy. Controls (Part 19): Scrolling tabs in TabControl, WinForms object events](https://c.mql5.com/2/49/MQL5-avatar-doeasy-library-2.png)[DoEasy. Controls (Part 19): Scrolling tabs in TabControl, WinForms object events](https://www.mql5.com/en/articles/11490)

In this article, I will create the functionality for scrolling tab headers in TabControl using scrolling buttons. The functionality is meant to place tab headers into a single line from either side of the control.

![DoEasy. Controls (Part 17): Cropping invisible object parts, auxiliary arrow buttons WinForms objects](https://c.mql5.com/2/48/MQL5-avatar-doeasy-library-2__5.png)[DoEasy. Controls (Part 17): Cropping invisible object parts, auxiliary arrow buttons WinForms objects](https://www.mql5.com/en/articles/11408)

In this article, I will create the functionality for hiding object sections located beyond their containers. Besides, I will create auxiliary arrow button objects to be used as part of other WinForms objects.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/11330&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071654721002875912)

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