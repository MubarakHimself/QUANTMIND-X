---
title: MQL5 Wizard Techniques you should know (14): Multi Objective Timeseries Forecasting with STF
url: https://www.mql5.com/en/articles/14552
categories: Trading Systems, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T19:17:09.084207
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/14552&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070163465408024835)

MetaTrader 5 / Tester


### Introduction

This [paper](https://www.mql5.com/go?link=https://openreview.net/pdf?id=4SiMia0kjba "https://openreview.net/pdf?id=4SiMia0kjba") on [Spatial Temporal Fusion](https://en.wikipedia.org/wiki/Spatial%E2%80%93temporal_reasoning "https://en.wikipedia.org/wiki/Spatial%E2%80%93temporal_reasoning") (STF) piqued my interest on the subject thanks to its two-sided approach to forecasting. For a refresher, the paper is inspired by solving a probability-based forecasting problem that is collaborative for both supply and demand in two-sided ride-hailing platforms, such as Uber and Didi. Collaborative supply and demand relationships are common in various two-sided markets, such as Amazon, Airbnb, and eBay where in essence the company not only serves the traditional ‘customer’ or purchaser, but also caters to suppliers of the customer.

So, two-sided forecasting in a case where supply is partly dependent on demand can be important to these companies on a frequent basis. This dual projection though, of demand and supply, was certainly a break from the conventional approach of forecasting a specific value to a timeseries or data set. The paper also introduced what it called a _causaltrans_ framework where the causal ‘collaborative’ relationship between supply and demand was captured by a matrix _G_ and all forecasts were made via transformer network and its results were noteworthy.

Taking a leaf from that, we look to forecast supply and demand for traded securities by using bearishness and bullishness as proxies for these two metrics. Strictly speaking though the typical Expert-Signal class does compute both these values as integers in the range 0-100 as can be seen in the MQL5 library files or files we have coded in these series so far. What would be new though will be the addition of a spatial matrix and a time parameter in making our forecasts (the 2 extra inputs we cite from the paper).

Spatial quantization of trade securities is subjective and so is the choice of time metric. Using security High-prices and Low-price series as our anchors for demand and supply, we use the autocorrelation values amongst these buffers as coordinates to a spatial matrix as well as the day of week index as a time indicator. This rudimentary approach that can be customized and improved, serves our purposes for this article.

The paper used transformer networks which we will not use as it is inefficient for our purposes however all forecasts will be through a custom hand coded multi-layer perceptron. With so many libraries and code samples on the subject, it would seem a waste of time to attempt to code one’s own multilayer perceptron. However the network class used is less than 300 lines long and is reasonably scalable in as far as customizing the number of layers and size of each, something which is still lacking in most of the boiler plate libraries that are available.

So, by using a singular neural network and not transformers the implementation of the paper’s _causaltrans_ framework will not be realized here. However, we still have our plate full with what to use since we will still be doing dual forecasting for demand and supply, and also using a spatial matrix and time in the process. And as always there are inherent risks in any trade system so the reader is welcomed to undertake their own diligence before taking any material shared here for further use.

### STF Illustration

STF is predominantly used in [remote-sensing](https://en.wikipedia.org/wiki/Remote_sensing "https://en.wikipedia.org/wiki/Remote_sensing") and visual centric activities where space and time metrics can tangibly be married.

_If you are not too keen at exploring STF potential outside of trading then you can skip this section and continue to MQL5 implementation._

If we look at remote-sensing for instance images captured by satellite do capture the spatial component of the data and region under examination while time would refer to when the images are taken. Such information can not only form a timeseries but also be important in making forecasts to changes in weather, plant life, or even animal habitat in the area under study all thanks to STF.

Let’s consider, blow by blow, how STF could help with deforestation and vegetation related problems. The first step in this, as is often the case with machine learning problems, is data collection. Remote sensing satellites would capture multispectral images for the study area over a period of time. Given that satellites can be multispectral can capture reflected and absorbed wavelength information not visible to the naked eye but is in wavelength bands associated to vegetation, or water bodies, etc. All this adds a richness (and complexity) to the type of data that can be modelled into a series as these images would be captured over some time.

Spatial data integration is what could follow, since each image is already time stamped, an appropriate GPS coordinate system could be used to map each image ensuring consistency of spatial information across all the images for all the time points they were captured.

Next up would be normalizing our ‘data’ to ensure it is in a format pertinent vegetation and deforestation. One of the ways this is achieved is by logging spectral signatures of each image at different time points by [indexing](https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index#:~:text=The%20normalized%20difference%20vegetation%20index,remote%20sensors%2C%20such%20as%20satellites. "https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index#:~:text=The%20normalized%20difference%20vegetation%20index,remote%20sensors%2C%20such%20as%20satellites.") so that the data is not just easier to handle by training models, but is also focused on the subject at hand.

This normalization would also involve change detection over the captured images over time such that features that are important to vegetation can be extracted or defined and these could include growth rates, seasonal variations, distribution patterns etc. Each of these features could form a dimension to the data.

Model training then follows this and the choice of model could vary but one would expect a neural network to be the prime candidate. A fair portion of the images, which are now as normalized data, would be used for this with a smaller data set reserved for testing as is typically the case.

So, the forecasting with the model would take place on successful training and testing results with the key thing to keep in mind being the interpretation since a lot of normalization has already been done the reverse would have to be carefully done for the model outputs.

In our cited [paper](https://www.mql5.com/go?link=https://dl.acm.org/doi/pdf/10.1145/3643848 "https://dl.acm.org/doi/pdf/10.1145/3643848") which was recently accepted by [ACM](https://www.mql5.com/go?link=https://dl.acm.org/ "https://dl.acm.org/"), the equations for future demand and supply are represented as follows:

> **x v (t+** **δt)** **= f x (x v (t), G v (t), δt),**
>
> **y v (t+** **δt)** **= f y (x v (t), y v (t), G v (t), δt),**

where:

- **x v** is the demand function with respect to time,
- **y v** is the supply function with respect to time,
- **G v** is also a vector or matrix for spatial information,
- **δ** **t** is the time increment to which a forecast is made,
- **t** is time.

So, in our case the functions f x and f y will be neural networks with the parameter vectors or matrices holding data that is fed to their respective input layers.

### Implementing STF with MQL5

To implement STF in MQL5 we would follow the shared equations above to set how we structure the input data to our model. From the 2 equations, it is clear input data for each would typically take a vector or matrix format and this of course presents countless possibilities and challenges. The potential inputs of each f-function in the 2 equations are similar buffers with the only difference between the 2 equations being that the supply equation is dependent not just on its previous values but also on those of demand. This follows the paper’s author’s thesis that supply is demand dependent and not vice versa.

So, the total number of buffers, given the overlap between both equations, is 4. These are all present in the supply equation and to list them they are previous demand, previous supply, spatial values, and the time increment.

The demand buffer is interpreted as a time series buffer of ‘bullish’ price points for this article. Arguably a more concise buffer could be real volume of long contracts but such information is seldom shared by brokers and even if they did it would not be an accurate representation of volume given the fractious nature of volume information in the forex markets. So, the high prices less the open prices are chosen as an alternative buffer to real long volume contracts. Other possible buffers that could fill this role could be currency specific metrics like interest rates, inflation, or even central bank money supply indices. The choice of Highs minus Open prices as a buffer is meant to measure upward volatility and since this can be understood to correlate positively with long volume contracts, it used as a next best proxy. These values can only be positive or zero with a zero-reading indicating a hang-man or a flat price bar.

The supply buffer like its predecessor counter-part above will also be approximated by taking the Open prices minus Low prices. This can also be taken as a reading on downward volatility which correlates positively with bearish volume contracts. Also like above the values for this buffer will only be positive with a zero-value indicating a Gravestone-Doji star or a flat bar. The supply equation is different from the demand in that it takes more inputs which implies that in addition its model will also be similar to the demand but different. So, there will be two instances of the forecasting model one for demand and one for supply. Since our end result is getting a single signal then this will be determined by subtracting the supply forecast from the demand forecast. Recall from above all inputs to the demand and supply model are positive or zero so in all likelihood so should the outputs therefore by subtracting supply model output from demand output we a double number whose positive value would indicate bullishness and whose negative value would mean bearishness.

The choice of time buffer is simply and index for the day of week. Testing which is covered in the next section will be done on the daily time frame so the week of day index does easily tie into that. If, however an alternative time frame was to be considered that is say smaller than the daily time frame then an index that either spans within a day or even a week could be considered. For instance, on the 8-hr time frame there are fifteen 8-hr bars in a trading week which does provide fifteen possible time indices intra week. You could pick the hour of the day or trade session of a day etc. the choices here are plentiful and preliminary testing to pick what works best for your trade system may be a better approach. Our simple function that returns week day index is as follows:

```
//+------------------------------------------------------------------+
//| Temporal (Time) Indexing function                                |
//+------------------------------------------------------------------+
int CSignalNetwork::T(datetime Time)
{  MqlDateTime _dt;
   if(TimeToStruct(Time,_dt))
   {  if(_dt.day_of_week==TUESDAY)
      {  return(1);
      }
      else if(_dt.day_of_week==WEDNESDAY)
      {  return(2);
      }
      else if(_dt.day_of_week==THURSDAY)
      {  return(3);
      }
      else if(_dt.day_of_week==FRIDAY||_dt.day_of_week==SATURDAY)
      {  return(4);
      }
   }
   return(0);
}
```

The G matrix captures our spatial data for this model and this can be tricky in defining. How do we define space in a securities’ trade environment? If we consider our reference paper for instance a cross table of metadata between supply and demand is ‘normalized’ by feeding it through what the paper refers to as Graph Attention Transformers (GAT). These operate on two layers the first layer captures complex node relationships, and the second layer aggregates neighbor information for final node predictions. The GAT readings are then part of what gets fed through the respective neural networks for forecasting demand or supply. In our case metadata for our bullish and bearish price buffers will be got from the correlation readings shared by these buffers. These correlations are captured as shown in the code below:

```
//+------------------------------------------------------------------+
//| Spatial (Space) Indexing function. Returns Matrix Determinant    |
//| This however can be customised to return all matrix values as    |
//| a vector, depending on the detail required.                      |
//+------------------------------------------------------------------+
double CSignalNetwork::G(vector &X,vector &Y)
{  matrix _m;
   if(X.Size()!=2*m_train_set||Y.Size()!=2*m_train_set)
   {  return(0.0);
   }
   _m.Init(2,2);
   vector _x1,_x2,_y1,_y2;
   _x1.Init(m_train_set);_x1.Fill(0.0);
   _x2.Init(m_train_set);_x2.Fill(0.0);
   _y1.Init(m_train_set);_y1.Fill(0.0);
   _y2.Init(m_train_set);_y2.Fill(0.0);
   for(int i=0;i<m_train_set;i++)
   {  _x1[i] = X[i];
      _x2[i] = X[i+m_train_set];
      _y1[i] = Y[i];
      _y2[i] = Y[i+m_train_set];
   }
   _m[0][0] = _x1.CorrCoef(_x2);
   _m[0][1] = _x1.CorrCoef(_y2);
   _m[1][0] = _y1.CorrCoef(_x2);
   _m[1][1] = _y1.CorrCoef(_y2);
   return(_m.Det());
}
```

Note we return a single value that represents the matrix and not its individual readings as we using its determinant. Individual readings could also be used alternatively since correlation values are always normalized from -1.0 to +1.0 and this could lead to more accurate results for the model overall. For our purposes though we stuck to the determinant as efficiency is a bit more important, for preliminary testing anyway.

With all 4 data buffers mentioned it may be helpful to also talk about our very rudimentary model which is a simple neural network that is coded without using any libraries. We break down a network to its very basic components of: inputs, weights, biases, hidden-outputs, outputs, and target; and use only these in making forward feeds and back propagation. Our network interface is as follows:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class Cnetwork
{
protected:
   matrix            weights[];
   vector            biases[];

   vector            inputs;
   vector            hidden_outputs[];
   vector            target;

   int               hidden_layers;
   double            Softplus(double X);
   double            SoftplusDerivative(double X);

public:
   vector            output;

   void              Get(vector &Target)
   {                 target.Copy(Target);
   }

   void              Set(vector &Inputs)
   {                 inputs.Copy(Inputs);
   }

   bool              Get(string File, double &Criteria, datetime &Version);
   bool              Set(string File, double Criteria);

   void              Forward();
   void              Backward(double LearningRate);

   void              Cnetwork(int &Settings[],double InitialWeight,double InitialBias)
   {

                        ...
                        ...

   };
   void              ~Cnetwork(void) { };
};
```

Besides the common functions for setting and getting inputs and targets, also included is some export functions for weights and biases after training. The feed forward algorithm is ordinary, using soft-plus for activation at each layer and the back-propagation function is also old school relying on the chain-rule and gradient descent methods at adjusting network weights and biases. Initializing the network though requires some inputs and these go through a ‘validation’ before the class instance can be safely used. Inputs to create network are settings array, value for initial weights (throughout network) as well as initial bias values. The settings array determines the number of hidden layers the network will have by its own size and each integer at its indices sets the size of the respective layer.

```
   void              Cnetwork(int &Settings[],double InitialWeight,double InitialBias)
   {                 int _size =    ArraySize(Settings);
                     if(_size >= 2 && _size <= USHORT_MAX && Settings[ArrayMinimum(Settings)] > 0 && Settings[ArrayMaximum(Settings)] < USHORT_MAX)
                     {  ArrayResize(weights, _size - 1);
                        ArrayResize(biases, _size - 1);
                        ArrayResize(hidden_outputs, _size - 2);
                        hidden_layers = _size - 2;
                        for(int i = 0; i < _size - 1; i++)
                        {  weights[i].Init(Settings[i + 1], Settings[i]);
                           weights[i].Fill(InitialWeight);
                           biases[i].Init(Settings[i + 1]);
                           biases[i].Fill(InitialBias);
                           if(i < _size - 2)
                           {  hidden_outputs[i].Init(Settings[i + 1]);
                              hidden_outputs[i].Fill(0.0);
                           }
                        }
                        output.Init(Settings[_size - 1]);
                        target.Init(Settings[_size - 1]);
                     }
                     else
                     {  printf(__FUNCSIG__ + " invalid network settings. ");
                        //~Cnetwork(void);
                     }
   };
```

Since we are forecasting both demand and supply we will need two separate instances of our network one to handle each task. Our get output function which acts as a feed for the Check Open Long and Check Open Short functions will fill the respective input layers of each network with the data buffers already mentioned above. For demand since demand is only dependent on its prior values, plus the space and time parameters its input layer is sized to 3 however supply in addition to depending on its prior values can be influenced by previous demand so its input layer is sized 4 if you consider the similar space and time inputs. The filling of these layers is handled within the get output function on each new bar as follows:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CSignalNetwork::GetOutput()
{

        ...
        ...

   for(int i = 0; i < m_train_set; i++)
   {  for(int ii = 0; ii < __LONGS_INPUTS; ii++)
      {  if(ii==0)//time
         {  m_model_longs.x[i][ii] = T(m_time.GetData(i));
         }
         else if(ii==1)//spatial matrix
         {  vector _x,_y;
            _x.CopyRates(m_symbol.Name(),m_period,2,StartIndex() + ii + i,2*m_train_set);
            _y.CopyRates(m_symbol.Name(),m_period,4,StartIndex() + ii + i,2*m_train_set);
            m_model_longs.x[i][ii] = G(_x,_y);
         }
         else if(ii==2)//demand
         {  m_model_longs.x[i][ii] = (m_high.GetData(StartIndex() + ii + i) - m_open.GetData(StartIndex() + ii + i));
         }
      }
      if(i > 0) //assign classifier
      {  m_model_longs.y[i - 1] = (m_high.GetData(StartIndex() + i - 1) - m_open.GetData(StartIndex() + i - 1));
      }
   }
   for(int i = 0; i < m_train_set; i++)
   {  for(int ii = 0; ii < __SHORTS_INPUTS; ii++)
      {  if(ii==0)//time
         {  m_model_shorts.x[i][ii] = T(m_time.GetData(i));
         }
         else if(ii==1)//spatial matrix
         {  vector _x,_y;
            _x.CopyRates(m_symbol.Name(),m_period,4,StartIndex() + ii + i,2*m_train_set);
            _y.CopyRates(m_symbol.Name(),m_period,2,StartIndex() + ii + i,2*m_train_set);
            m_model_shorts.x[i][ii] = G(_x,_y);
         }
         else if(ii==2)//demand
         {  m_model_shorts.x[i][ii] = (m_high.GetData(StartIndex() + ii + i) - m_open.GetData(StartIndex() + ii + i));
         }
         else if(ii==3)//supply
         {  m_model_shorts.x[i][ii] = (m_open.GetData(StartIndex() + ii + i) - m_low.GetData(StartIndex() + ii + i));
         }
      }
      if(i > 0) //assign classifier
      {  m_model_shorts.y[i - 1] = (m_open.GetData(StartIndex() + i - 1) - m_low.GetData(StartIndex() + i - 1));
      }
   }

        ...
        ...

}
```

After this we assign input and target values for the two networks by retrieving this info from the model struct on within training loops where each loop iterates through the networks for a set number of epochs. From preliminary testing the ideal number of epochs was found to be about 10000 at a learning rate of 0.5, per training loop. This is certainly compute-intensive which is why test results presented here in the next section use very conservative values of about 250 epochs.

The set output function allows us to log the network weights and biases at each pass provided the pass results surpass the criteria set by the weights used at initialization. When initializing there is the option to read weights and the criteria set by those weights serves as a benchmark for the current test run.

### Test Runs

We perform test runs on EURUSD on the daily time frame for the year 2023 with very simple network settings. The demand forecasting network has 3 layers in total with one hidden and their sizes are 3, 5, and 1. The supply forecasting network also has 3 layers but as stated above the input layer has a different size so their sizes are 4, 5, and 1. The 1 at the end of both networks stores the outputs of each network.

As always assembling the code attached at the end of this article into an expert adviser is achieved via the MQL5 wizard so if you are new or unfamiliar please refer to articles [here](https://www.mql5.com/en/articles/171) and [here](https://www.mql5.com/en/articles/275) for guidance.

These two setups are very basic and arguably the simplest you can come up with given that the referenced network class can generate up to UCHAR\_MAX layers each with a size of also UCHAR\_MAX if the initialization settings define this. As more layers and even larger layer sizes are used compute resources do escalate none the less it is worth pointing out that layer sizes in the 100-unit range are typically considered sufficient even when set up with few layers.

We perform runs as always with no exit price targets where positions are held until the signal reverses as this approach tends to be more long-term and better assesses major price trends. A real-tick pass with some of the ideal settings gives us the report below:

![rep](https://c.mql5.com/2/73/report.png)

And the equity curve shown below:

![curv](https://c.mql5.com/2/73/curve.png)

If we dig into this report, it is clear too few trades are placed throughout the year which in the first place is not a bad thing because their quality was solid seeing as they are held over extended periods until signal reversals are trigged and for this tested period they have suffered minimum drawdowns. The issue though is in flat or whipsaw markets it is often prudent, especially in cases where leverage is involved, to be more sensitive to the micro movements of price. In the system we tested we were using the daily timeframe and this is okay as it tends to focus on the big picture but if were to make this a bit more pragmatic we would have to consider smaller time frames probably 4-hour or even 1-hour. When we do this, our compute resources just for testing will go up by a considerable magnitude in fact to say this relationship is exponential is not an exaggeration. And keep in mind this was a very rudimentary network of 3 layers with hidden layer having 5 points and the spatial matrix reduced to a determinant. All these factors are important because as one starts to deal with smaller time frames the need to sift through noise becomes even more important and one of the best ways of doing this is by being a bit pedantic.

In addition, within the referenced network class are functions that help import and export the network’s weight and bias settings at the beginning and end of each pass. I did not use them for these test runs as this was also slowing down the process a bit but they would need to be used when testing over extended periods (we have only considered one year for this test). When they are being used the criteria for which the test runs are being conducted needs to be weighed carefully as by default, in the network class, bigger is better. So, if one is more concerned about draw-downs for example then the code attached in the network class needs to be modified accordingly to reflect this such that on each pass, the weights written to file only get updated if the criteria value from that run is LESS than the previous value. This also implies that the start or default value at each run needs to be DBL\_MAX or a sufficiently high value to avoid unnecessary errors.

### Conclusion

STF’s ability to handle dual forecasting certainly presents an intriguing approach that is not very common and unquestionably has potential for refinement and improvement. The G matrix for spatial information used in our testing for example can be expanded to an n x n by splitting the input data vectors to smaller parts, and even each of its values could be input data points to a network as well as many other adjustments but these changes though come at a cost of compute resources.

In fact, in general implementing STF by neural network is essentially a compute intense endeavor that requires testing over fairly large amounts of data in order to get dependable cross-validation. And this presents its main limitation which is even more evident when implemented in the cited paper above where network transformers were used.

However, it is an aspect of this craft that is slowly being accepted by the industry as we see NVIDIA increasingly become more relevant in this landscape. Alternative more efficient models like random forests could be explored provided their attendant settings are also not overly complex but as compute costs start to decrease the cost-benefits of this may not be feasible.

MQL5 wizard though, remains a tool for rapid prototyping and testing ideas and this article on Spatial Temporal Fusion presents another illustration of this.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14552.zip "Download all attachments in the single ZIP archive")

[Network.mqh](https://www.mql5.com/en/articles/download/14552/network.mqh "Download Network.mqh")(10.87 KB)

[SignalWZ\_14\_aa.mqh](https://www.mql5.com/en/articles/download/14552/signalwz_14_aa.mqh "Download SignalWZ_14_aa.mqh")(14.03 KB)

[nn\_a.mq5](https://www.mql5.com/en/articles/download/14552/nn_a.mq5 "Download nn_a.mq5")(6.59 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Codex Pipelines, from Python to MQL5, for Indicator Selection: A Multi-Quarter Analysis of the XLF ETF with Machine Learning](https://www.mql5.com/en/articles/20595)
- [Codex Pipelines: From Python to MQL5 for Indicator Selection — A Multi-Quarter Analysis of the FXI ETF](https://www.mql5.com/en/articles/20550)
- [Market Positioning Codex for VGT with Kendall's Tau and Distance Correlation](https://www.mql5.com/en/articles/20271)
- [Markets Positioning Codex in MQL5 (Part 2): Bitwise Learning, with Multi-Patterns for Nvidia](https://www.mql5.com/en/articles/20045)
- [Markets Positioning Codex in MQL5 (Part 1): Bitwise Learning for Nvidia](https://www.mql5.com/en/articles/20020)
- [MQL5 Wizard Techniques you should know (Part 85): Using Patterns of Stochastic-Oscillator and the FrAMA with Beta VAE Inference Learning](https://www.mql5.com/en/articles/19948)
- [MQL5 Wizard Techniques you should know (Part 84): Using Patterns of Stochastic Oscillator and the FrAMA - Conclusion](https://www.mql5.com/en/articles/19890)

**[Go to discussion](https://www.mql5.com/en/forum/464638)**

![Neural networks made easy (Part 65): Distance Weighted Supervised Learning (DWSL)](https://c.mql5.com/2/61/Neural_Networks_Made_Easy_lPart_65q_DWSL_LOGO.png)[Neural networks made easy (Part 65): Distance Weighted Supervised Learning (DWSL)](https://www.mql5.com/en/articles/13779)

In this article, we will get acquainted with an interesting algorithm that is built at the intersection of supervised and reinforcement learning methods.

![Population optimization algorithms: Spiral Dynamics Optimization (SDO) algorithm](https://c.mql5.com/2/61/Spiral_Dynamics_Optimization_SDO_LOGO.png)[Population optimization algorithms: Spiral Dynamics Optimization (SDO) algorithm](https://www.mql5.com/en/articles/12252)

The article presents an optimization algorithm based on the patterns of constructing spiral trajectories in nature, such as mollusk shells - the spiral dynamics optimization (SDO) algorithm. I have thoroughly revised and modified the algorithm proposed by the authors. The article will consider the necessity of these changes.

![Python, ONNX and MetaTrader 5: Creating a RandomForest model with RobustScaler and PolynomialFeatures data preprocessing](https://c.mql5.com/2/61/Python_ONNX__MetaTrader_5____RandomForest____LOGO.png)[Python, ONNX and MetaTrader 5: Creating a RandomForest model with RobustScaler and PolynomialFeatures data preprocessing](https://www.mql5.com/en/articles/13725)

In this article, we will create a random forest model in Python, train the model, and save it as an ONNX pipeline with data preprocessing. After that we will use the model in the MetaTrader 5 terminal.

![Data Science and Machine Learning (Part 21): Unlocking Neural Networks, Optimization algorithms demystified](https://c.mql5.com/2/73/Data_Science_and_Machine_Learning_Part_21___LOGO.png)[Data Science and Machine Learning (Part 21): Unlocking Neural Networks, Optimization algorithms demystified](https://www.mql5.com/en/articles/14435)

Dive into the heart of neural networks as we demystify the optimization algorithms used inside the neural network. In this article, discover the key techniques that unlock the full potential of neural networks, propelling your models to new heights of accuracy and efficiency.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/14552&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070163465408024835)

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