---
title: MQL5 Wizard techniques you should know (Part 02): Kohonen Maps
url: https://www.mql5.com/en/articles/11154
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:30:25.187570
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/11154&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070340808902644766)

MetaTrader 5 / Trading systems


### 1\. Introduction

1.1Continuing with this series on the MQL5 wizard, we will delve into **Kohonen-Maps** for this one. These per[Wikipedia](https://en.wikipedia.org/wiki/Self-organizing_map "https://en.wikipedia.org/wiki/Self-organizing_map")are technique used to produce a low-dimensional (typically two-dimensional) representation of a higher dimensional data set while preserving the topological structure of the data. They were[put forward](https://www.mql5.com/go?link=https://www.sciencedirect.com/science/article/abs/pii/0893608088900202 "https://www.sciencedirect.com/science/article/abs/pii/0893608088900202")by Teuvo Kohonen in the 1980s.

In simple terms kohonen-maps (aka self-organizing maps) are away of summarizing complexity without losing the distinctness of what is summarized. The summary serves as a form of organizing hence the name self-organizing. With the re-organized data or maps we therefore have two sets of related data. The original high-dimensional data that serves as our input, and the summarized (lower-dimensional data) form that is usually, but not always, represented in two dimensions as the output. The inputs are the knowns while the outputs would be the unknown or in this case what is being ‘ _studied_’.

For a trader, if we are for the purpose of this article focus only on time-based price series, the knowns (that we’ll refer to as **feed** data) at any time are the prices left of that time with the _unknowns_(that we’ll call **functor** data) being those to the right. How we classify the knowns and unknowns determines the respective number of dimensions for both the feed and functor data. This is something traders should be critical-to because it is informed hugely by their outlook and approach to the markets.

1.2A Common misconception with these maps is that the functor data should be an image or 2 dimensional. Images such as the one below are all often shared as being representative of what Kohonen Maps are.

![typical_image](https://c.mql5.com/2/47/SOM_typical_image.png)

While not wrong, I want to highlight that the functor can and perhaps should (for traders) have a single dimension. So rather than reducing our high dimensional data to a 2D map, we will map it onto a single line. Kohonen maps by definition are meant to reduce dimensionality so I want us to take this to the next level for this article. The kohonen map is different from regular neural networks both in number of layers and the underlying algorithm. It is a single-layer (usually linear 2D grid as afore mentioned) _set_ of neurons, instead of multiple layers. All the neurons on this layer which we are referring to as the functor connect to the feed, but not to themselvesmeaning the neurons are not influenced by each other’s weights _directly_, and only update with respect to the feed data. The functor data layer is often a “map” that organizes itself at each training iteration depending on the feed data. As such, after training, each neuron has weight adjusted dimension in the functor layer and this allows one to calculate the Euclidean distance between any two such neurons.

### 2\. Creating the class

2.1_Class structure_

2.1.1**Dimension** abstract class is the first class we will define. This code would have been tidier if I had made most of it in a separate file and simply referenced it but I want to cover that together with the money and trailing classes in the next article, so for now like in the previous article all code will be in the signal file. The dimensions are always important in this network since they heavily influence the output. The feed data (the inputs) will be multi-dimensional as is typically the case. The functor data (the outputs) will have one dimension contrary to the typical x and y. Based on the multi-dimensionality of both the feed and functor data an ideal data type would be a double array.

However, keeping with the trend of exploring the MQL5 Library, we will use an [array list](https://www.mql5.com/en/docs/standardlibrary/generic/carraylist) of double type instead. The feed data will be changes in lows less changes in highs over a space of one bar as we had used in the previous article. As a rule inputs are better selected based on a trader’s insights on the market and should not be adopted and used by everyone on a live or even testing account. Every trader should modify this code to allow for his own input data. The functor data will be one dimensional as stated. However, since it is also a list, it can be customized to add more dimensions. For our purposes though, we will focus on the change between the most recent bar’s open and close. Once again, the MQL5 wizard allows you to set what a bar is by selecting your own timeframe. The dimension class will inherit from the list double interface in the MQL5 code library. Two functions will be added to this class, namely Get and Set. As their name suggests, they aid in retrieving and setting values in the list once an index is provided.

```
#include                        <Generic\ArrayList.mqh>
#include                        <Generic\HashMap.mqh>

#define                         SCALE 5

#define                         IN_WIDTH 2*SCALE
#define                         OUT_LENGTH 1

#define                         IN_RADIUS 100.0
#define                         OUT_BUFFER 10000

//
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class Cdimension                :  public CArrayList<double>
  {
    public:

      Cdimension()              {};
      ~Cdimension()             {};

     virtual double             Get(const int Index)
                                {
                                  double _value=0.0; TryGetValue(Index,_value); return(_value);
                                };
     virtual void               Set(const int Index,double Value)
                                {
                                  Insert(Index,Value);
                                };
  };
```

2.1.2**Feed** class will inherit from the dimension class just created above. No special functions will be added here. Only the constructor will specify the list capacity (analogous to array size) and default size of our feed data list will be 10.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class Cfeed             : public Cdimension
 {
   public:

     Cfeed()            { Clear(); Capacity(IN_WIDTH);  };
     ~Cfeed()           {                               };
 };
```

2.1.3**Functor** class will be similar to the feed class with only caveat being the size. As stated, we will consider one (not the usual two) dimensions for our functor data, so the set size will be 1.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class Cfunctor          : public Cdimension
 {
   public:

   Cfunctor()           { Clear(); Capacity(OUT_LENGTH); };
   ~Cfunctor()          {                                };
 };
```

2.1.4**Neuron** class is where our code gets interesting. We will declare it as a class that inherits from an interface in the MQL5 library that takes two custom data types. A key and a value. The template interface in question is the [HashMap](https://www.mql5.com/en/docs/standardlibrary/generic/chashmap).And the custom data typed we’ll use will be the two classes declared above. Namely the Feed class as our key and the Functor class as our value. We also have no functions but only pointers to the Feed class, Functor class and a ‘key-value’ class of the same. The purpose of this class as the name suggests is to define the neuron. The neuron is our unit of data since it includes both the inputs data-type (feed data) and output data-type (functor data). It is a neuron’s feed-data that is matched with already trained neurons in order to project what the functor could be. Also mapped neurons have their functor-data are adjusted whenever a new neuron is training.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class Cneuron           : public CHashMap<Cfeed*,Cfunctor*>
 {
   public:

    double              weight;

    Cfeed               *fd;
    Cfunctor            *fr;

    CKeyValuePair
    <
    Cfeed*,
    Cfunctor*
    >                   *ff;

    Cneuron()           {
                          weight=0.0;
                          fd = new Cfeed();
                          fr = new Cfunctor();
                          ff = new CKeyValuePair<Cfeed*,Cfunctor*>(fd,fr);
                          Add(ff);
                        };

   ~Cneuron()           {
                          ZeroMemory(weight);
                          delete fd;
                          delete fr;
                          delete ff;
                        };
 };
```

2.1.5**Layer** abstract class is what follows next. It inherits from a list template of the neuron class and has one object - a neuron pointer. Being an abstract class, this neuron pointer is meant to be used by classes that inherit from this class. There are two such classes, namely the input layer and the output layer. Strictly speaking, Kohonen maps should not be classified as neural networks as they do not have feed forward links with weights and back propagation. Some proponents though feel they are just a different type.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class Clayer            : public CArrayList<Cneuron*>
 {
   public:

    Cneuron             *n;

    Clayer()            { n = new Cneuron();     };
    ~Clayer()           { delete n;              };
 };
```

2.1.6**Input Layer** class inherits from the abstract layer class. It is where live and recent data feed values are stored when the network is running. Rather than being a typical layer with multiple neurons, it will feature a single neuron that has the most recent feed and functor data, therefore its size will be 1.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class Cinput_layer      : public Clayer
 {
   public:

   static const int     size;

    Cinput_layer()      {
                          Clear();
                          Capacity(Cinput_layer::size);
                          for(int s=0; s<size; s++)
                          {
                            n = new Cneuron();
                            Add(n);
                          }
                        }
    ~Cinput_layer()     {};
 };
 const int Cinput_layer::size=1;
```

2.1.7**Output Layer** class also inherits from the layer class but it serves as our map since ‘trained’ neurons are stored here. The functor-data portion of the neurons in this layer is equivalent to an image or map of your typical SOM. Its size will initially be 10,000 and will be incremented by the same amount as new neurons are trained.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class Coutput_layer      : public Clayer
 {
   public:

    int                  index;
    int                  size;

    Coutput_layer()      {
                           index=0;
                           size=OUT_BUFFER;
                           Clear();
                           Capacity(size);
                           for(int s=0; s<size; s++)
                           {
                             n = new Cneuron();
                             Add(n);
                           }
                         };

    ~Coutput_layer()     {
                           ZeroMemory(index);
                           ZeroMemory(size);
                         };
 };
```

2.1.8**Network** class like the neuron class also inherits from the HashMap template interface. Its key and value data types are the input layer class and the output layer class. It has the most functions (9) for not only getting list size but also retrieving and updating neurons on the respective layers.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class Cnetwork           : public CHashMap<Cinput_layer*,Coutput_layer*>
 {
   public:

     Cinput_layer        *i;
     Coutput_layer       *o;

     CKeyValuePair
     <
     Cinput_layer*,
     Coutput_layer*
     >                   *io;

     Cneuron             *i_neuron;
     Cneuron             *o_neuron;

     Cneuron             *best_neuron;

     Cnetwork()          {
                           i = new Cinput_layer();
                           o = new Coutput_layer();
                           io = new CKeyValuePair<Cinput_layer*,Coutput_layer*>(i,o);
                           Add(io);

                           i_neuron = new Cneuron();
                           o_neuron = new Cneuron();

                           best_neuron = new Cneuron();
                         };

     ~Cnetwork()         {
                           delete i;
                           delete o;
                           delete io;
                           delete i_neuron;
                           delete o_neuron;
                           delete best_neuron;
                         };

      virtual int        GetInputSize()
                         {
                           TryGetValue(i,o);
                           return(i.size);
                         };

      virtual int        GetOutputIndex()
                         {
                           TryGetValue(i,o);
                           return(o.index);
                         };

      virtual void       SetOutputIndex(const int Index)
                         {
                           TryGetValue(i,o);
                           o.index=Index;
                           TrySetValue(i,o);
                         };

      virtual int        GetOutputSize()
                         {
                           TryGetValue(i,o);
                           return(o.size);
                         };

      virtual void       SetOutputSize(const int Size)
                         {
                           TryGetValue(i,o);
                           o.size=Size;
                           o.Capacity(Size);
                           TrySetValue(i,o);
                         };

      virtual void       GetInNeuron(const int NeuronIndex)
                         {
                           TryGetValue(i,o);
                           i.TryGetValue(NeuronIndex,i_neuron);
                         };

      virtual void       GetOutNeuron(const int NeuronIndex)
                         {
                           TryGetValue(i,o);
                           o.TryGetValue(NeuronIndex,o_neuron);
                         };

      virtual void       SetInNeuron(const int NeuronIndex)
                         {
                           i.TrySetValue(NeuronIndex,i_neuron);
                         };

      virtual void       SetOutNeuron(const int NeuronIndex)
                         {
                           o.TrySetValue(NeuronIndex,o_neuron);
                         };
 };
```

2.1.9**Map** class is the final umbrella class. It calls an instance of the network class and includes other variables for training neurons and getting the best matching neuron for the network.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class Cmap
  {
    public:

      Cnetwork               *network;

      static const double     radius;
      static double           time;

      double                  QE;       //proxy for Quantization Error
      double                  TE;       //proxy for Topological Error

      datetime                refreshed;

      bool                    initialised;

      Cmap()                  {
                                network = new Cnetwork();

                                initialised=false;

                                time=0.0;

                                QE=0.50;
                                TE=5000.0;

                                refreshed=D'1970.01.05';
                               };

      ~Cmap()                  {
                                 ZeroMemory(initialised);

                                 ZeroMemory(time);

                                 ZeroMemory(QE);
                                 ZeroMemory(TE);

                                 ZeroMemory(refreshed);
                               };
 };
 const double Cmap::radius=IN_RADIUS;
 double Cmap::time=10000/fmax(1.0,log(IN_RADIUS));
```

### 2.2. Topology

2.2.1**Neuron training** is the [competitive learning](https://en.wikipedia.org/wiki/Competitive_learning "https://en.wikipedia.org/wiki/Competitive_learning") process of adjusting the functor weights of existing neurons in the output layer and adding a new trainer neuron. The rate, at which these weights are adjusted and most importantly the number of iterations it takes to adjust these weights, are very sensitive parameters in determining the efficacy of the network. At each iteration of adjusting the weights, a new smaller radius is calculated. I refer to this radius as the _functor-error_(not to be confused with the SOM Topological-error) but most refer to it as the neighborhood radius as measured by Euclidean distance. I choose ‘error’ as this is a parameter that needs to be minimized for better network results. The more iterations one performs, the smaller will be the Functor-error.Besides the number of iterations, the rate of learning needs to be reduced gradually from a number close to one towards zero.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CSignalKM::NetworkTrain(Cmap &Map,Cneuron &TrainNeuron)
  {
    Map.TE=0.0;

    int _iteration=0;
    double _training_rate=m_training_rate;

    int _err=0;
    double _functor_error=0.0;

    while(_iteration<m_training_iterations)
    {
      double _current_radius=GetTrainingRadius(Map,_iteration);

      for(int i=0; i<=Map.network.GetOutputIndex(); i++)
      {
        Map.network.GetOutNeuron(i);
        double _error = EuclideanFunctor(TrainNeuron,Map.network.o_neuron);

        if(_error<_current_radius)
        {
          _functor_error+=(_error);
          _err++;

          double _remapped_radius = GetRemappedRadius(_error, _current_radius);

          SetWeights(TrainNeuron,Map.network.o_neuron,_remapped_radius,_training_rate);

          Map.network.SetOutNeuron(i);
        }
      }

      _iteration++;
      _training_rate=_training_rate*exp(-(double)_iteration/m_training_iterations);
    }

    int
    _size=Map.network.GetOutputSize(),
    _index=Map.network.GetOutputIndex();
    Map.network.SetOutputIndex(_index+1);
    if(_index+1>=_size)
    {
      Map.network.SetOutputSize(_size+OUT_BUFFER);
    }

    Map.network.GetOutNeuron(_index+1);
    for(int w=0; w<IN_WIDTH; w++)
    {
      Map.network.o_neuron.fd.Set(w,TrainNeuron.fd.Get(w));
    }

    for(int l=0; l<OUT_LENGTH; l++)
    {
      Map.network.o_neuron.fr.Set(l,TrainNeuron.fr.Get(l));
    }

    Map.network.SetOutNeuron(_index+1);

    if(_err>0)
    {
      _functor_error/=_err;
      Map.TE=_functor_error*IN_RADIUS;
    }
  }
```

2.2.2  **Topological Error** is a key attribute in Kohonen maps. I take it as a measure of how close the output layer is to its long-term intended goal. Remember, with each training the output layer neurons get adapted to the true or intended result so the question becomes how we measure this progress. The answer to this is if we are preserving the output layer more, then we are closer to this target. For the purposes of this article, I will have the functor-error act as a proxy for it.

### 2.3. Quantization

2.3.1**Neuron mapping**is a process of finding the functor weights that would best fit a neuron for which only feed-data is present. This is done by finding the neuron in the output layer with the shortest feed-data Euclidean distance from the neuron, for which no functor data is known. Like with the training, I refer to this distance as the _feed-error._ Again, the smaller our value, the more reliable should be the network.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CSignalKM::NetworkMapping(Cmap &Map,Cneuron *MapNeuron)
  {
    Map.QE=0.0;

    Map.network.best_neuron = new Cneuron();

    int _random_neuron=rand()%Map.network.GetOutputIndex();

    Map.network.GetInNeuron(0);
    Map.network.GetOutNeuron(_random_neuron);

    double _feed_error = EuclideanFeed(Map.network.i_neuron,Map.network.o_neuron);

    for(int i=0; i<Map.network.GetOutputIndex(); i++)
    {
      Map.network.GetOutNeuron(i);

      double _error = EuclideanFeed(Map.network.i_neuron,Map.network.o_neuron);

      if(_error < _feed_error)
      {
        for(int w=0; w<IN_WIDTH; w++)
        {
          Map.network.best_neuron.fd.Set(w,Map.network.o_neuron.fd.Get(w));
        }

        for(int l=0; l<OUT_LENGTH; l++)
        {
          Map.network.best_neuron.fr.Set(l,Map.network.o_neuron.fr.Get(l));
        }

        _feed_error = _error;
      }
    }

    Map.QE=_feed_error/IN_RADIUS;
}
```

2.3.2   **Quantization Error** is another critical attribute in Kohonen maps that does not have a concise definition in my opinion. My take is its the error in translating high dimensional data to the low dimensional output. In our case, here it would be the error in converting the feed to the functor. For the purposes of this article I will have the feed-error act as a proxy for it.

### 3\. Assembling with MQL5 Wizard

3.1_Wizard assembly_ is straight forward. Only caution I have here is start testing on large timeframes first since the ideal 10,000 training iterations per bar will take some time when training over a significant period.

![wizard_1](https://c.mql5.com/2/47/wizard_1__2.png)

### 4\. Testing in Strategy Tester

4.1_Default inputs_ for our testing will investigate the sensitivity of our quantization error proxy (QE) and the topological error proxy (TE). We will look at the two scenarios. First, we will test with very conservative values with QE and TE at 0.5 and 12.5; then we will test these inputs at 0.75 and 25.0 respectively.

![criteria_1](https://c.mql5.com/2/47/criteria_1_1.png)

conservative options

![criteria_2](https://c.mql5.com/2/47/criteria_1_2.png)

aggressive options

The inputs are not that many. We have 'training read' which determines whether or not we should read a training file prior to initialization. If this file is absent, the expert will not validate. We also have 'training write' which, as name suggests, determines whether a learning file should be written once the expert de-initializes. Training always takes place once the expert is running. The option to train only and not trade is set by the 'training only' input parameter. The other two significant parameters to Kohonen maps are the 'training rate' (also known as learning rate) and the training iterations. Generally the higher these two are (training rate is capped at 1.0), the better performance should one expect, however this will come at a cost of time and CPU resources.

The expert was trained on EURJPY's V shaped period of 2018.10.01 to 2021.06.01 and forward tested from the training end date to present date.

The conservative option came to this report:

![report_1](https://c.mql5.com/2/47/report_1_1__1.png)

And this equity curve:

![curve_1](https://c.mql5.com/2/47/curve_1_1.png)

However, the more aggressive option had this report:

![report_2](https://c.mql5.com/2/47/report_2_1.png)

And this equity curve:

![curve_2](https://c.mql5.com/2/47/curve_2_1.png)

Clearly, more testing and fine-tuning is required regarding risk and position sizing but for a system that is trained over such a short period it is promising. Comparing the two scenarios above though, it appears the more conservative option is not sufficiently rewarded given its Sharpe ratio value of 0.43 is almost half the 0.85 value on more trades. More study is required here before use and as always besides customising the feed and functor data to one's style of how he trades; preliminary testing should always be done on your broker's real-ticks data over large significant periods of time before deployment.

### 5.  Conclusion

5.1**MQL5 Wizard** is clearly a very agile tool when it comes to assembling trading systems in a tight timeframe window. For this article, we explored the option of Kohonen maps that port multi-dimension feed data of price time series into a single dimension that ranges from -1.0 to 1.0. While not common practice, this approach does champion the very essence of Kohonen maps which is to reduce complexity and ease decision making. We have also done this while showcasing more code from the MQL library like [array lists](https://www.mql5.com/en/docs/standardlibrary/generic/carraylist) and [hash maps](https://www.mql5.com/en/docs/standardlibrary/generic/chashmap). I hope, you liked it. Thanks for reading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11154.zip "Download all attachments in the single ZIP archive")

[SignalKM\_article.mqh](https://www.mql5.com/en/articles/download/11154/signalkm_article.mqh "Download SignalKM_article.mqh")(33.23 KB)

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

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/428329)**
(3)


![YestheI](https://c.mql5.com/avatar/2022/11/6368bd57-cc11.png)

**[YestheI](https://www.mql5.com/en/users/yesthei)**
\|
29 Jan 2023 at 08:44

One question remains, how can we understand why the IN\_RADIUS value is taken modulo normalise the data:

double \_dimension=fabs(IN\_RADIUS)\*((Low(StartIndex()+w+Index)-Low(StartIndex()+w+Index+1))-(High(StartIndex()+w+Index)-High(StartIndex()+w+Index+1))))/fmax(m\_symbol. [Point()](https://www.mql5.com/en/docs/check/point "MQL5 documentation: Point function"),fmax(High(StartIndex()+w+Index),High(StartIndex()+w+Index+1))-fmin(Low(StartIndex()+w+Index),Low(StartIndex()+w+Index+1))));

because the cluster radius is a constant and has a positive value.

Perhaps this is an error and the whole numerator should be taken modulo?

![Oleg Pavlenko](https://c.mql5.com/avatar/2022/1/61DE8FA2-2774.png)

**[Oleg Pavlenko](https://www.mql5.com/en/users/ovpmusic)**
\|
26 Jun 2023 at 20:25

I assembled the Expert Advisor, but two parameters Stop Loss and Take Profit are not the same as on the screenshot in the article:

![In the article.](https://c.mql5.com/3/412/criteria_1_2.png)

[![I've got](https://c.mql5.com/3/412/2023-06-26_21-21-26__1.png)](https://c.mql5.com/3/412/2023-06-26_21-21-26.png "https://c.mql5.com/3/412/2023-06-26_21-21-26.png")

As a result, not a single trade...

Am I doing something wrong?

And how to use another indicator instead of ATR, for example MACD?

![YestheI](https://c.mql5.com/avatar/2022/11/6368bd57-cc11.png)

**[YestheI](https://www.mql5.com/en/users/yesthei)**
\|
15 Nov 2023 at 16:33

**Oleg Pavlenko [#](https://www.mql5.com/ru/forum/432734#comment_47769491):**

I built the Expert Advisor, but two parameters Stop Loss and Take Profit are not the same as on the screenshot in the article:

As a result, not a single trade...

Am I doing something wrong?

And how to use another indicator instead of ATR, for example MACD ?

You can do it this way:

```
bool CSignalMACD::InitMACD(CIndicators *indicators)
  {
//--- add object to collection
   if(!indicators.Add(GetPointer(m_MACD)))
     {
      printf(__FUNCTION__+": error adding object");
      return(false);
     }
//--- initialize object
   if(!m_MACD.Create(m_symbol.Name(),m_period,m_period_fast,m_period_slow,m_period_signal,m_applied))
     {
      printf(__FUNCTION__+": error initializing object");
      return(false);
     }
//--- ok
   return(true);
  }
```

also in protected:

```
protected:
   CiMACD            m_MACD;           // object-oscillator
   //--- adjusted parameters
   int               m_period_fast;    // the "period of fast EMA" parameter of the oscillator
   int               m_period_slow;    // the "period of slow EMA" parameter of the oscillator
   int               m_period_signal;  // the "period of averaging of difference" parameter of the oscillator
   ENUM_APPLIED_PRICE m_applied;       // the "price series" parameter of the oscillator
```

and in public:

```
 void              PeriodFast(int value)             { m_period_fast=value;           }
 void              PeriodSlow(int value)             { m_period_slow=value;           }
 void              PeriodSignal(int value)           { m_period_signal=value;         }
 void              Applied(ENUM_APPLIED_PRICE value) { m_applied=value;
```

and again in protected:

```
protected:
   //--- method of initialisation of the oscillator
   bool              InitMACD(CIndicators *indicators);
   //--- methods of getting data
   double            Main(int ind)                     { return(m_MACD.Main(ind));      }
   double            Signal(int ind)                   { return(m_MACD.Signal(ind));    }
```

and finally:

```
bool CSignalKM::OpenLongParams(double &price,double &sl,double &tp,datetime &expiration)
  {
   CExpertSignal *general=(m_general!=-1) ? m_filters.At(m_general) : NULL;
//---
   if(general==NULL)
     {
      m_MACD.Refresh(-1);
      //--- if a base price is not specified explicitly, take the current market price
      double base_price=(m_base_price==0.0) ? m_symbol.Ask() : m_base_price;

      //--- price overload that sets entry price to be based on MACD
      price      =base_price;
      double _range=m_MACD.Main(StartIndex())+((m_symbol.StopsLevel()+m_symbol.FreezeLevel())*m_symbol.Point());
      //
```

But what's the point?

![Developing a trading Expert Advisor from scratch (Part 14): Adding Volume At Price (II)](https://c.mql5.com/2/46/development__5.png)[Developing a trading Expert Advisor from scratch (Part 14): Adding Volume At Price (II)](https://www.mql5.com/en/articles/10419)

Today we will add some more resources to our EA. This interesting article can provide some new ideas and methods of presenting information. At the same time, it can assist in fixing minor flaws in your projects.

![Neural networks made easy (Part 14): Data clustering](https://c.mql5.com/2/48/Neural_networks_made_easy_014.png)[Neural networks made easy (Part 14): Data clustering](https://www.mql5.com/en/articles/10785)

It has been more than a year since I published my last article. This is quite a lot time to revise ideas and to develop new approaches. In the new article, I would like to divert from the previously used supervised learning method. This time we will dip into unsupervised learning algorithms. In particular, we will consider one of the clustering algorithms—k-means.

![DoEasy. Controls (Part 6): Panel control, auto resizing the container to fit inner content](https://c.mql5.com/2/46/MQL5-avatar-doeasy-library-2__5.png)[DoEasy. Controls (Part 6): Panel control, auto resizing the container to fit inner content](https://www.mql5.com/en/articles/10989)

In the article, I will continue my work on the Panel WinForms object and implement its auto resizing to fit the general size of Dock objects located inside the panel. Besides, I will add the new properties to the Symbol library object.

![Indicators with on-chart interactive controls](https://c.mql5.com/2/46/interactive-control.png)[Indicators with on-chart interactive controls](https://www.mql5.com/en/articles/10770)

The article offers a new perspective on indicator interfaces. I am going to focus on convenience. Having tried dozens of different trading strategies over the years, as well as having tested hundreds of different indicators, I have come to some conclusions I want to share with you in this article.

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=hzbqniumlqvuxphswzbfyhasamefcukd&ssn=1769185823763857528&ssn_dr=0&ssn_sr=0&fv_date=1769185823&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11154&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Wizard%20techniques%20you%20should%20know%20(Part%2002)%3A%20Kohonen%20Maps%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918582327350839&fz_uniq=5070340808902644766&sv=2552)

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