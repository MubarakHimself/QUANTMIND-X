---
title: Experiments with neural networks (Part 3): Practical application
url: https://www.mql5.com/en/articles/11949
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:27:05.926381
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/11949&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071841470475874167)

MetaTrader 5 / Expert Advisors


### Introduction

In the previous articles of the series [Revisiting geometry](https://www.mql5.com/en/articles/11077)and [Smart neural network optimization](https://www.mql5.com/en/articles/11186), I shared my observations and experiments with neural networks.Besides, I carried out the optimization of the resulting EAs and provided some explanations of their work. Despite all this, I barely touched the subject of the practical application of the obtained results. In this article,I will fix this unfortunate omission.

I will show the practical application of the obtained results and highlight a new algorithm allowing us to expand the capabilities of our EAs. As always, I will use only MetaTrader 5 tools without any third-party software. The article will most likely be similar to a step-by-step instruction. I will try to explain everything in the most accessible and simple way.

### 1\. Application idea

When optimizing the two previous systems, I received some results with the best values of the profit factor or complex criterion and also a set of weights for our perceptrons and neural networks. The tests of the obtained results have displayed quite tolerable values. The main idea of this improvement is to combine all the optimization results in one EA and make them work simultaneously. As you might imagine, it is not very convenient to keep 10 charts open with ten EAs. In addition, this will allow us to look at the results in an extended (comprehensive) way, using for example 10-20 parameters at the same time.

### 2\. Currency pair. Optimization and forward test range. Settings

Below are all optimization and test parameters:

- Forex;
- EURUSD;
- H1;
- Indicators: 2 TEMA indicators with the periods of 1 and 24. I had to abandon MA since the TEMA indicator turned out to be better in numerous tests.
- StopLoss and TakeProfit for the corresponding modifications of 600 and 60;

- "Open prices only" and "Complex Criterion max" optimization and testing modes. "Complex Criterion max" mode showed more stable and profitable results compared to "Maximum profit";
- Optimization range 3 years. 2018.12.09 - 2021.12.09. 3 years is not a reliable criterion. You can experiment with this parameter on your own;
- Forward test range is 1 year. 2021.12.09 - 2022.12.09;
- In all forward tests, 20 optimization results were used simultaneously;
- Optimization of EAs with the "Fast (genetic based algorithm)" perceptron;
- Optimization of EAs using the DeepNeuralNetwork.mqh "Slow complete algorithm" library;
- Initial deposit 10,000;
- Leverage 1:500.

### 3\. Perceptron-based EAs

According to numerous observations, it turned out that the weight depth of 200 is not required in EAs with a perceptron. 20 is sufficient. Therefore, the perceptron code itself and optimization parameters were changed. Now we optimize the weights from 0 in increments of 1 to 20.

The "Param" parameter has also been introduced, which is responsible for the depth of withdrawal to the positive or negative side of the perceptron. This parameter affected the number of trades and their accuracy. The number of trades has decreased, while their accuracy has improved.

Each of the systems uses 2 EAs. The first one is used for optimization, while the second one is meant for direct work. I decided to arrange the division of orders through order comments, since I believe this is the easiest and most convenient way. The unique order index is its serial number in the sample itself. The sample is an archive with the obtained optimization results. TheMaxSeries parameter is used to limit the amount of simultaneous work.

```
for(int i=0; i<=(ArraySize(EURUSD)/6)-1; i++){
 comm=IntegerToString(i);
 x1=(int)StringToInteger(EURUSD[i][0]);
 x2=(int)StringToInteger(EURUSD[i][1]);
 x3=(int)StringToInteger(EURUSD[i][2]);
 x4=(int)StringToInteger(EURUSD[i][3]);

 Param=(int)StringToInteger(EURUSD[i][4]);

//SELL++++++++++++++++++++++++++++++++++++++++++++++++
if (CalculateSeries(Magic)<MaxSeries && (perceptron1()<-Param) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment+" En_"+comm)==0) && (ind_In1[1]>ind_In2[1]) && (SpreadS1<=MaxSpread)){
  OpenSell(symbolS1.Name(), LotsXSell, TakeProfit, StopLoss, EAComment+" En_"+comm);
}

//BUY++++++++++++++++++++++++++++++++++++++++++++++++
if (CalculateSeries(Magic)<MaxSeries && (perceptron1()>Param) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment+" En_"+comm)==0) && (ind_In1[1]<ind_In2[1]) && (SpreadS1<=MaxSpread)){
  OpenBuy(symbolS1.Name(), LotsXBuy, TakeProfit, StopLoss, EAComment+" En_"+comm);
}

}
```

New perceptron code:

1perceptron4angleSLTP and 1 perceptron 4 angle

```
double perceptron1()
  {
   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   double a1 = (((ind_In1[1]-ind_In1[6])/Point())/6);
   double a2 = (((ind_In1[1]-ind_In1[11])/Point())/11);
   double a3 = (((ind_In2[1]-ind_In2[6])/Point())/6);
   double a4 = (((ind_In2[1]-ind_In2[11])/Point())/11);

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);
  }
```

1 perceptron 8 angle SL TP and 1 perceptron 8 angle

```
double perceptron1()
  {
   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   double v1 = y1 - 10.0;
   double v2 = y2 - 10.0;
   double v3 = y3 - 10.0;
   double v4 = y4 - 10.0;

   double a1 = (((ind_In1[1]-ind_In1[6])/Point())/6);
   double a2 = (((ind_In1[1]-ind_In1[11])/Point())/11);
   double a3 = (((ind_In2[1]-ind_In2[6])/Point())/6);
   double a4 = (((ind_In2[1]-ind_In2[11])/Point())/11);

   double b1 = (((ind_In1[1]-ind_In1[11])/Point())/11);
   double b2 = (((ind_In2[1]-ind_In1[11])/Point())/11);
   double b3 = (((ind_In1[1]-ind_In2[11])/Point())/11);
   double b4 = (((ind_In2[1]-ind_In2[11])/Point())/11);

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4   +   v1 * b1 + v2 * b2 + v3 * b3 + v4 * b4);
  }
```

2 perceptronа4 angle SL TP and 2 perceptronа4 angle

```
double perceptron1()
  {
   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   double a1 = (((ind_In1[1]-ind_In1[6])/Point())/6);
   double a2 = (((ind_In1[1]-ind_In1[11])/Point())/11);
   double a3 = (((ind_In2[1]-ind_In2[6])/Point())/6);
   double a4 = (((ind_In2[1]-ind_In2[11])/Point())/11);

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);
  }

double perceptron2()
  {
   double v1 = y1 - 10.0;
   double v2 = y2 - 10.0;
   double v3 = y3 - 10.0;
   double v4 = y4 - 10.0;

   double b1 = (((ind_In1[1]-ind_In1[11])/Point())/11);
   double b2 = (((ind_In2[1]-ind_In1[11])/Point())/11);
   double b3 = (((ind_In1[1]-ind_In2[11])/Point())/11);
   double b4 = (((ind_In2[1]-ind_In2[11])/Point())/11);

   return (v1 * b1 + v2 * b2 + v3 * b3 + v4 * b4);
  }
```

Entry activation code:

1perceptron4angleSLTP and 1 perceptron 4 angle

```
//SELL++++++++++++++++++++++++++++++++++++++++++++++++
if ((perceptron1()<-Param) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)==0) && (ind_In1[1]>ind_In2[1]) && (SpreadS1<=MaxSpread)){
  OpenSell(symbolS1.Name(), LotsXSell, TakeProfit, StopLoss, EAComment);
}

//BUY++++++++++++++++++++++++++++++++++++++++++++++++
if ((perceptron1()>Param) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)==0) && (ind_In1[1]<ind_In2[1]) && (SpreadS1<=MaxSpread)){
  OpenBuy(symbolS1.Name(), LotsXBuy, TakeProfit, StopLoss, EAComment);
}
```

1 perceptron 8 angle SL TP and 1 perceptron 8 angle

```
//SELL++++++++++++++++++++++++++++++++++++++++++++++++
if ((perceptron1()<-Param) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)==0) && (ind_In1[1]>ind_In2[1]) && (SpreadS1<=MaxSpread)){
  OpenSell(symbolS1.Name(), LotsXSell, TakeProfit, StopLoss, EAComment);
}

//BUY++++++++++++++++++++++++++++++++++++++++++++++++
if ((perceptron1()>Param) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)==0) && (ind_In1[1]<ind_In2[1]) && (SpreadS1<=MaxSpread)){
  OpenBuy(symbolS1.Name(), LotsXBuy, TakeProfit, StopLoss, EAComment);
}
```

2 perceptronа4 angle SL TP and 2 perceptronа4 angle

```
//SELL++++++++++++++++++++++++++++++++++++++++++++++++
if ((perceptron1()<-Param) && (perceptron2()<-Param) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)==0) && (ind_In1[1]>ind_In2[1]) && (SpreadS1<=MaxSpread)){
  OpenSell(symbolS1.Name(), LotsXSell, TakeProfit, StopLoss, EAComment);
}

//BUY++++++++++++++++++++++++++++++++++++++++++++++++
if ((perceptron1()>Param) && (perceptron2()>Param) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)==0) && (ind_In1[1]<ind_In2[1]) && (SpreadS1<=MaxSpread)){
  OpenBuy(symbolS1.Name(), LotsXBuy, TakeProfit, StopLoss, EAComment);
}
```

Optimization settings:

1perceptron4angleSLTP and 1 perceptron 4 angle

![Optimization settings 1](https://c.mql5.com/2/51/5__3.png)

1 perceptron 8 angle SL TP and 1 perceptron 8 angle

![Optimization settings](https://c.mql5.com/2/51/5__4.png)

2 perceptronа4 angle SL TP and 2 perceptronа4 angle

![Optimization settings](https://c.mql5.com/2/51/5__5.png)

### 3.1 EA 1 perceptron 4 angle SL TP

This EA modification usesstop loss and take profit to exit. Strategy 1 perceptron and 4 slope angles of TEMA indicators. Perform optimization 10 times. Structures of slope angles and optimization principles can be found in the first article. There is no point in repeating them here.

Optimization result:

![Optimization result](https://c.mql5.com/2/51/1__6.png)

![Optimization result](https://c.mql5.com/2/51/2__6.png)

A large number of results of the complex criterion 99.99. Profit factor at the high level of 4-8.

Next, export the obtained result toExcel. Leave the first 100 best results and delete everything else. Remove all columns except х1, х2, х3, х4 and Param.Save the file toCSV(commas are used as separators). I have named the fileEURUSDfor clarity.We can load this format into the EA code as a text array.It should look as shown in the picture below.

![Optimization result](https://c.mql5.com/2/51/4__3.png)

Insert the file into the code via the MetaEditor menu.

![Optimization result](https://c.mql5.com/2/51/6__1.png)

Get the ready-made text array with optimization results.

```
string EURUSD[][6]=
  {
   {"19","1","3","6","1100"},
   {"20","1","4","6","1000"},
   {"20","0","4","4","1200"},
   {"19","0","6","4","1100"},
   {"19","1","5","4","1100"},
   {"17","0","7","4","1100"},
   {"19","1","3","8","1000"},
   {"20","0","4","3","1300"},
   {"17","0","7","0","1400"}
  };
```

Compile and carry out the forward test.

![Forward Testing](https://c.mql5.com/2/51/3__7.png)

The results are good. We can see a steady upward movement throughout the year.

### 3.2 EA 1 perceptron 4 angle

This EA modification does not usestop loss and take profit. A closure is performed by the perceptron callback.

```
//SELL++++++++++++++++++++++++++++++++++++++++++++++++
if ((perceptron1()<-Param) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)==0) && (ind_In1[1]>ind_In2[1]) && (SpreadS1<=MaxSpread)){
  OpenSell(symbolS1.Name(), LotsXSell, 0, 0, EAComment);
}

if ((perceptron1()>0) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)>0)){
ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL);
}

//BUY++++++++++++++++++++++++++++++++++++++++++++++++
if ((perceptron1()>Param) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)==0) && (ind_In1[1]<ind_In2[1]) && (SpreadS1<=MaxSpread)){
  OpenBuy(symbolS1.Name(), LotsXBuy, 0, 0, EAComment);
}

if ((perceptron1()<0) && (CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)>0)){
ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY);
}
```

Optimization result:

![Optimization result](https://c.mql5.com/2/51/1__7.png)

![Optimization result](https://c.mql5.com/2/51/2__7.png)

The result of the complex criterion is somewhat lower than in the previous case. Profit factor is at the level of 2-2.5.

Forward test result:

![Forward test result](https://c.mql5.com/2/51/3__8.png)

The balance line repeats the previous result with deeper drawdowns.

### 3.3 EA 1 perceptron 8 angle SL TP

This EA modification uses stop loss and take profit to exit. Strategy 1 perceptron and 8 slope angles of TEMA indicators.

For this EA, we need to prepare anExcelfile as shown below. Here we optimize х1, х2, х3, х4,y1,y2,y3,y4 and Param parameters.

![Optimization result](https://c.mql5.com/2/51/3__9.png)

Optimization result:

![Optimization result](https://c.mql5.com/2/51/1__8.png)

![Optimization result](https://c.mql5.com/2/51/2__15.png)

The complex criterion result is high. Profit factor is at the level of 2.5-3.

Forward test result:

![Forward test result](https://c.mql5.com/2/51/4__4.png)

The balance line is not very stable. The drawdowns are considerable. However, the result is positive.

### 3.4 EA 1 perceptron 8 angle

Stop loss and take profitare not used. A closure is performed on the reverse signal of the perceptron. Strategy 1 perceptron and 8 slope angles ofTEMA indicators.

Optimization result:

![Optimization result](https://c.mql5.com/2/51/1__9.png)

![Optimization result](https://c.mql5.com/2/51/2__8.png)

The complex criterion result is high. Profit factor is at the level of 2.5-3.

Forward test result:

![Forward test result](https://c.mql5.com/2/51/3__10.png)

The balance line is stable. A good increase in the deposit throughout the year. The drawdowns are not that deep.

### 3.5 EA 2 perceptron 8 angle SL TP

Stop loss and take profit are used. Strategy 2 perceptron, 4 slope angles at the first and 4 slope angles at the second one which are different.

Optimization result:

![Optimization result](https://c.mql5.com/2/51/1__10.png)

![Optimization result](https://c.mql5.com/2/51/2__9.png)

The result of the complex criterion is at the level of 99.99. The profit factor of the results is almost the same 4.3.

Forward test result:

![Forward test result](https://c.mql5.com/2/51/3__11.png)

Sawtooth balance line. Profit throughout the year.

### 3.6 EA 2 perceptron 8 angle

No stop loss and take profit. Strategy 2 perceptron, 4 slope angles at the first and 4 slope angles at the second one which are different.Closure is performed by the perceptron reverse signal.

Optimization result:

![Optimization result](https://c.mql5.com/2/51/1__11.png)

![Optimization result](https://c.mql5.com/2/51/2__10.png)

The result of the complex criterion is at 99.8. The profit factor of the results is within 2.8-3.2.

Forward test result:

![Forward test result](https://c.mql5.com/2/51/3__12.png)

Sawtooth balance line, unstable. Profit throughout the year. Big drawdowns at the end of the year.

### 4\. EAs based on the DeepNeuralNetwork.mqh library

In the current article, I will use 4 EAs based on the DeepNeuralNetwork.mqh library - Angle 4-4-3 SL TP and Angle 8-4-3 SL TP applying stop loss and take profit for closure, as well as Angle 4-4-3 and Angle 8-4-3, in which the signal closure comes from the neural network. All of the EAs use slope angles as a strategy. The shapes presented in the second part of our experiments are not used here.

Stop loss and take profit code:

```
//SELL++++++++++++++++++++++++++++++++++++++++++++++++
if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment+" En_"+comm1)==0) && (yValues[1]>LL) && (SpreadS1<=MaxSpread)){
  if(CalculateSeries(Magic)<MaxSeries){
  OpenSell(symbolS1.Name(), LotsXSell, TP, SL, EAComment+" En_"+comm1);
  }
}

//BUY++++++++++++++++++++++++++++++++++++++++++++++++
if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment+" En_"+comm1)==0) && (yValues[0]>LL) && (SpreadS1<=MaxSpread)){
  if(CalculateSeries(Magic)<MaxSeries){
  OpenBuy(symbolS1.Name(), LotsXBuy, TP, SL, EAComment+" En_"+comm1);
  }
}
```

The code for closing from the neural network:

```
//SELL++++++++++++++++++++++++++++++++++++++++++++++++
if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment+" En_"+comm1)==0) && (yValues[1]>LL) && (SpreadS1<=MaxSpread)){
  ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment+" En_"+comm1);
  if(CalculateSeries(Magic)<MaxSeries){
  OpenSell(symbolS1.Name(), LotsXSell, TP, SL, EAComment+" En_"+comm1);
  }
}

//BUY++++++++++++++++++++++++++++++++++++++++++++++++
if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment+" En_"+comm1)==0) && (yValues[0]>LL) && (SpreadS1<=MaxSpread)){
  ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment+" En_"+comm1);
  if(CalculateSeries(Magic)<MaxSeries){
  OpenBuy(symbolS1.Name(), LotsXBuy, TP, SL, EAComment+" En_"+comm1);
  }
}

//CLOSE ALL++++++++++++++++++++++++++++++++++++++++++
if (yValues[2]>LL){
  ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment+" En_"+comm1);
  ClosePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment+" En_"+comm1);
}
```

Now let's apply a more complex scheme. We have 3 Expert Advisors in each set. The first one is meant for optimization, the second one is for converting the obtained results into a text array, while the third one is for testing and handlingthe resulting array.

As you might remember, during the optimization, the EA creates a CSV file with a set of optimized weights at C:\\Users\\Your username\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files. Copy the CSV optimization report file to the folder.

Launch Angle EA 4-4-3 convert on the currency pair chart.

![Expert Advisor](https://c.mql5.com/2/51/2__24.png)

EA parameters for conversion:

- Param– complex criterion value, below which the results are not copied to the array. I set 80;
- OptimizationFileName1 – CSV optimization report file
- OptimizationFileName2 – CSV file created by the EA during optimization and containing neural network weights;
- OptimizationFileName3 – array file to be inserted into the EA. The file is to be created automatically.

You can watch the process in the logs.

![Logs](https://c.mql5.com/2/51/4__5.png)

Insert the obtained file to the code of Angle EA 4-4-3 trade:

```
string Result[][37]=
  {
   {"17293","0.8","-0.1","-0.2","1.0","0.9","0.6","0.4","1.0","0.6","-0.4","0.9","-0.5","0.1","-0.5","-0.5","0.9","-0.1","-0.8","0.4","0.0","-0.1","0.1","0.2","-0.4","-0.7","-0.6","-0.9","-0.8","-0.9","-0.7","-0.5","0.4","0.4","0.8","-0.6"},
   {"18030","0.6","0.2","-0.4","0.9","-1.0","-0.9","-0.9","0.4","-0.9","-0.8","0.4","0.9","0.2","-0.8","0.9","-0.1","-0.6","0.3","0.5","-0.4","0.7","0.6","-0.4","-0.1","0.4","-0.8","0.4","0.9","-0.2","0.0","0.4","-0.6","-0.4","-0.7","0.7"},
   {"13128","0.7","-0.3","0.5","-0.5","-0.5","-0.1","0.8","0.0","0.6","0.9","-0.2","0.8","1.0","0.7","-0.7","-0.2","0.5","0.5","-0.6","0.5","-0.9","-0.5","-0.5","0.5","-0.3","0.5","0.8","0.2","-0.5","-0.2","0.1","-0.1","-0.4","-0.7","0.1"},
   {"10688","0.3","0.0","0.2","-0.1","0.6","0.1","0.1","-0.2","-1.0","0.3","0.2","0.5","-0.8","0.7","0.4","-0.5","-0.4","-0.3","-0.3","-0.9","-0.2","0.0","0.1","0.9","0.3","-0.9","-0.2","-0.2","0.1","0.9","0.8","0.1","0.4","0.8","0.6"},
   {"8356","0.8","0.8","0.7","0.2","0.0","-0.4","0.5","-0.8","0.0","0.9","0.2","-0.1","1.0","0.6","0.2","-0.8","-0.1","-0.5","-0.3","0.0","0.7","-0.5","-0.3","0.0","0.9","-1.0","-0.2","-0.6","-0.7","-0.5","-0.8","0.5","-0.3","-0.1","0.8"},
   {"18542","-0.8","0.9","-0.1","0.5","-0.5","0.3","0.8","-0.4","0.7","0.9","0.4","0.0","-0.2","0.0","0.2","0.5","0.9","0.4","1.0","0.7","0.1","0.1","-0.4","0.0","0.9","0.2","0.0","-0.8","0.1","-0.5","0.1","-0.1","0.1","-0.1","0.6"},
   {"18381","0.7","-1.0","-0.8","0.8","-0.8","-0.4","0.9","0.7","1.0","0.7","0.8","0.5","0.1","-0.3","-0.7","-0.9","-0.2","-0.4","0.8","-0.8","0.0","0.8","-0.5","-0.3","0.2","-0.3","-0.1","0.5","-0.1","0.3","0.0","-0.7","-0.2","-0.3","0.8"},
   {"13795","0.2","0.9","0.4","0.4","0.1","-0.6","-0.6","-0.3","0.7","0.9","0.7","0.0","-0.2","-0.9","-0.8","-0.6","-0.1","-0.4","-1.0","0.7","-0.7","-0.3","0.0","-0.3","-1.0","0.8","-0.9","-0.9","0.1","-0.5","-0.3","-0.7","-0.2","-0.7","-0.8"},
   {"4376","0.9","0.7","-0.6","-0.9","1.0","0.8","0.1","-0.8","0.7","-0.8","0.2","0.1","-0.9","0.8","0.9","-0.4","0.8","0.3","0.0","-0.3","-0.4","0.7","-0.2","0.4","-0.8","-0.2","0.9","0.9","0.2","0.0","0.1","0.5","-0.8","-0.1","0.6"},
   {"14503","0.1","-0.4","-0.7","0.1","-0.1","0.5","-0.7","-0.2","-0.9","0.0","0.2","-0.7","0.3","0.7","-0.7","0.1","0.4","0.3","0.3","-0.5","-0.8","-0.8","-0.7","0.2","-0.7","-0.1","-0.8","0.0","-0.4","0.0","0.1","0.5","-0.3","0.5","0.8"},
   {"12887","0.6","-0.1","0.4","0.6","-0.9","-0.3","0.7","0.2","-0.6","-1.0","0.0","-0.6","0.5","0.3","0.8","0.0","-0.5","-1.0","-0.6","0.6","-0.6","-0.9","-0.3","0.6","0.2","-0.5","0.6","0.2","-0.5","0.3","0.3","-0.9","-0.7","-0.8","0.8"},
   {"16285","0.3","0.3","-0.9","-0.7","-0.1","0.7","-0.7","-0.7","-0.2","-0.5","-0.8","-1.0","-0.1","-0.4","-0.6","1.0","0.3","-0.8","-0.6","1.0","-0.1","0.7","-0.1","0.5","-0.6","0.9","-0.5","0.6","0.2","0.5","-0.4","0.3","-0.6","-0.7","0.7"},
   {"13692","0.8","-0.9","0.6","0.3","-0.2","-0.8","-0.4","0.3","-0.6","0.7","0.7","-0.8","0.5","0.1","-0.2","0.7","-0.7","-0.2","0.7","-0.5","0.9","0.7","0.6","0.8","-0.1","-1.0","-0.8","-0.5","-0.1","-0.9","-0.5","0.2","-0.4","0.8","0.2"},
   {"1184","-0.1","0.1","0.6","-0.2","-0.3","0.0","-0.7","0.1","-0.5","0.1","-0.6","0.0","-0.9","-0.8","0.1","0.5","0.3","-1.0","0.1","-0.8","-0.6","0.0","-0.4","-0.1","-0.7","-0.8","0.6","0.5","0.0","0.9","-0.5","0.2","0.7","0.3","0.9"},
   {"9946","0.4","-0.5","0.9","-1.0","-0.4","-0.7","0.9","0.0","-0.2","0.7","0.7","0.1","0.7","0.4","-0.9","0.1","-0.6","-0.5","0.9","0.8","0.2","-0.9","0.0","0.1","0.9","0.7","0.3","0.6","-0.4","0.8","-0.1","0.2","-0.2","-0.4","0.7"},
   {"6104","0.5","-0.9","-0.1","0.7","-0.7","0.0","0.4","0.3","0.8","-0.7","-0.1","0.1","-0.1","-0.5","-0.5","1.0","-0.1","-0.5","0.5","0.7","-0.8","-0.7","-0.7","0.8","-0.2","-0.5","0.2","-0.6","-0.2","-0.1","-0.4","-0.9","-0.6","-0.1","0.9"},
   {"995","0.9","0.6","0.7","0.1","-0.8","0.3","-0.2","0.3","0.9","-0.1","0.2","0.5","0.9","-0.7","-0.7","-0.7","0.2","0.2","0.4","-0.7","-0.4","-0.2","0.0","-0.2","0.0","0.6","-0.3","-0.6","-0.9","0.8","-0.6","-0.2","0.2","0.5","0.9"},
   {"6922","0.5","0.9","0.1","-0.8","-1.0","-0.1","0.9","0.9","-0.2","0.8","0.8","0.5","-0.3","0.8","-0.2","0.9","-0.6","0.0","0.7","-0.9","0.4","0.7","0.6","-0.1","-0.4","0.5","-0.6","-0.2","-0.5","-0.9","-0.7","-0.6","0.5","-0.6","0.7"},
   {"3676","-0.9","-0.8","-0.5","0.8","0.4","-0.8","-0.4","0.6","0.9","0.9","-0.7","0.6","0.8","-0.9","0.3","0.7","-0.7","0.5","0.8","0.9","0.1","0.5","0.8","0.1","0.9","0.9","0.4","0.3","-0.1","0.4","-0.4","0.4","-0.3","-0.6","0.9"},
   {"6245","-0.1","-0.4","-0.6","0.7","0.6","-0.6","-0.2","0.2","0.0","-0.4","0.0","0.9","-0.3","0.5","-0.2","0.7","0.4","1.0","0.7","-0.1","-0.3","-0.9","-0.5","0.9","0.8","-0.1","-0.5","-1.0","0.3","0.9","-0.4","-0.2","-0.4","-0.3","0.9"},
   {"1039","-0.4","-0.3","-0.6","-0.7","-0.6","0.5","-0.2","-0.9","0.7","0.9","-0.2","-0.6","-0.2","-0.3","0.6","0.1","-0.9","-0.8","0.9","0.3","0.6","0.8","-0.8","0.8","0.6","0.1","-0.2","-0.7","0.6","-0.2","-0.6","0.4","-0.1","-0.2","0.1"},
   {"6615","-0.4","-0.1","-0.7","0.5","-0.9","0.4","-0.9","0.4","-0.4","-0.1","0.7","-0.4","0.4","0.4","-0.8","-0.2","-0.6","-0.1","-0.5","-0.7","0.6","0.0","1.0","0.9","-0.3","0.8","0.8","-0.1","-0.2","0.9","-0.2","0.9","-0.8","-0.6","0.5"},
   {"410","-0.3","0.2","-0.2","-0.2","0.2","-0.5","0.8","0.3","-0.9","-0.9","-0.4","0.3","-0.8","-0.8","0.0","0.9","-0.2","0.0","-0.2","-0.4","-0.1","0.1","-0.4","0.7","1.0","0.1","0.5","0.3","0.1","0.7","0.4","0.0","-0.2","-1.0","-0.1"},
   {"15027","-0.3","-0.4","-0.6","0.3","-0.5","-0.6","0.9","0.5","-0.2","0.0","-0.7","0.7","0.1","0.5","-0.4","-0.4","0.4","0.7","-0.1","0.9","-0.1","0.6","0.5","-0.3","0.6","0.8","0.4","0.1","0.9","-0.5","0.7","0.6","-0.8","-0.1","0.0"},
   {"14157","0.6","-0.7","0.7","0.5","0.8","-0.1","0.9","0.8","0.8","0.7","0.6","-0.3","-0.7","-0.5","-0.2","0.2","0.0","-0.8","0.6","0.9","-0.4","0.1","0.1","0.9","0.7","-0.8","-0.6","-0.5","-0.7","0.1","-0.3","0.9","0.5","0.8","-0.7"},
   {"11367","0.2","-1.0","-0.4","-0.4","-0.3","-0.2","0.2","-0.1","-0.4","0.7","-1.0","-0.5","-0.9","-0.7","-0.4","-0.8","-0.4","0.0","0.2","0.7","-0.2","0.4","0.1","0.0","-0.1","-0.9","0.2","-0.5","-0.6","-0.6","-0.7","-0.2","-0.3","-0.1","0.9"},
   {"3892","-0.7","-0.3","0.8","0.2","-0.3","0.4","0.0","0.3","-0.2","0.7","0.6","0.6","0.7","-0.4","-0.7","0.4","-0.3","-0.8","-0.2","0.0","0.9","0.9","0.3","0.0","0.7","0.1","-0.1","0.1","-0.8","-0.4","-0.5","0.9","-0.7","-0.6","0.2"}
  };
```

### 4.1 Angle 4-4-3 SL TP EA

The EA usesstop loss and take profit to exit. Strategy 4 slope angles of TEMA indicators.

Optimization result:

![Optimization result](https://c.mql5.com/2/51/1__12.png)

![Optimization result](https://c.mql5.com/2/51/3__13.png)

As you can see, there are many good results. The profit factor is within 1.6-5. There are 27 complex criterion values exceeding 80.

Forward test result:

![Forward test result](https://c.mql5.com/2/51/5__6.png)

Unfortunately, the EA failed the forward test. The results are negative and unstable.

### 4.2 Angle 4-4-3 EA

The EA uses the neural network to exit. Strategy 4 slope angles ofTEMA indicators.

Optimization result:

![Optimization result](https://c.mql5.com/2/51/1__13.png)

![Optimization result](https://c.mql5.com/2/51/2__12.png)

As you can see, there were only 6 good results for the complex criterion exceeding 80. The profit factor is within 1.6-1.9.

Forward test result:

![Forward test result](https://c.mql5.com/2/51/3__14.png)

With closing by the neural network, the EA showed profit throughout the year. The result is more stable than using Stop Loss and Take Profit.

### 4.3 Angle 8-4-3 SL TP EA

The EA usesstop loss and take profit to exit. Strategy 8 slope angles ofTEMA indicators.

Optimization result:

![Optimization result](https://c.mql5.com/2/51/1__14.png)

![Optimization result](https://c.mql5.com/2/51/2__13.png)

The profit factor of the results is lower compared to the neural network 4-4-3.There are 13 complex criterion results exceeding 80.

Forward test result:

![Forward test result](https://c.mql5.com/2/51/3__15.png)

As expected, the result is similar to the previous one using StopLoss and TakeProfit. The forward test has been failed.

### 4.4 Angle 8-4-3 EA

The EA uses the neural network to exit. Strategy 8 slope angles ofTEMA indicators.

Optimization result:

![Optimization result](https://c.mql5.com/2/51/1__15.png)

![Optimization result](https://c.mql5.com/2/51/2__14.png)

There are only 3 complex criterion results exceeding 80. The profit factor is at a low level compared to previous results.

Forward test result:

![Forward test result](https://c.mql5.com/2/51/3__16.png)

The forward result is not satisfactory. We see the gradual loss of the deposit.

### Conclusion

As we can see from the results of forward testing, none of the perceptron-based EAs went negative during the year, although there is a certain decline in profitability after 6 months of operation. This means that optimization is necessary at least once every 6 months.

As for the EAs based on the DeepNeuralNetwork.mqh library, all is complicated. The results are not as good as expected. Perhaps the strategy itself is affecting the things and it is necessary to pass something else to the neural network.

In most cases, profitability can be traced by the profit factor of optimized series. That gives us an additional food for thought.

For the future, I would like to point out2 tasks. It is necessary to check the best results obtained on other currency pairs and timeframes.

There are not as many trades as we would like to have, but no one forbids using other currency pairs and creating a portfolio based on these systems. This, in turn, is associated with unavoidable labor costs for optimization.

If you have any questions, ask them on the Forum or contact me via a private message. I will always be happy to help you.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/11949](https://www.mql5.com/ru/articles/11949)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11949.zip "Download all attachments in the single ZIP archive")

[EA.zip](https://www.mql5.com/en/articles/download/11949/ea.zip "Download EA.zip")(1056.16 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Experiments with neural networks (Part 7): Passing indicators](https://www.mql5.com/en/articles/13598)
- [Experiments with neural networks (Part 6): Perceptron as a self-sufficient tool for price forecast](https://www.mql5.com/en/articles/12515)
- [Experiments with neural networks (Part 5): Normalizing inputs for passing to a neural network](https://www.mql5.com/en/articles/12459)
- [Experiments with neural networks (Part 4): Templates](https://www.mql5.com/en/articles/12202)
- [Testing and optimization of binary options strategies in MetaTrader 5](https://www.mql5.com/en/articles/12103)
- [Experiments with neural networks (Part 2): Smart neural network optimization](https://www.mql5.com/en/articles/11186)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/442302)**
(45)


![CapeCoddah](https://c.mql5.com/avatar/avatar_na2.png)

**[CapeCoddah](https://www.mql5.com/en/users/capecoddah)**
\|
24 Mar 2023 at 16:53

Thanks for the speedy response.

I see I was doing my speculations on the wrong tab; I was using the orders tab when I should have been using the Deals tab for my analysis.  By using Deals and sorting on the profit column descending, I have immediately spotted the July 7th Deal that incurred a loss of $1395.

I have to chalk this boo boo of mine to inexperience.  I started using MQ5 6 months ago and just got into trade details from your EAs and tried using my MQ4 experience as a base.

Thanks for your assistance, I just learned a lot

Cape CVoddah

D

![Roman Poshtar](https://c.mql5.com/avatar/2019/12/5DE4DBC6-D492.jpg)

**[Roman Poshtar](https://www.mql5.com/en/users/romanuch)**
\|
24 Mar 2023 at 17:02

**CapeCoddah [#](https://www.mql5.com/en/forum/442302/page3#comment_45842482):**

Thanks for the speedy response.

I see I was doing my speculations on the wrong tab; I was using the orders tab when I should have been using the Deals tab for my analysis.  By using Deals and sorting on the profit column descending, I have immediately spotted the July 7th Deal that incurred a loss of $1395.

I have to chalk this boo boo of mine to inexperience.  I started using MQ5 6 months ago and just got into trade details from your EAs and tried using my MQ4 experience as a base.

Thanks for your assistance, I just learned a lot

Cape CVoddah

D

Nothing scary. We are all learning. If you're interested, I'm recruiting a team to develop neuro. Send me a private message. Participation is paid.

![Eric Ruvalcaba](https://c.mql5.com/avatar/2018/4/5AC4016D-F876.PNG)

**[Eric Ruvalcaba](https://www.mql5.com/en/users/ericruv)**
\|
26 Mar 2023 at 21:21

Hi Roman,

First let me thank you for this amazing contribution, very interesting, you are a genius. I gotta say I liked the 2 perceptron 4 angles options, so I added a 3rd perceptron for a parameter I have found to be quite valuable in my manual trading: RSI in different timeframes.

```
//+------------------------------------------------------------------+
//|  The PERCEPTRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron3() //RSI
{
   double v1 = z1 - 10.0;
   double v2 = z2 - 10.0;
   double v3 = z3 - 10.0;
   double v4 = z4 - 10.0;

   //In3 = RSI Current timeframe
   //In4 = RSI Higher timeframe

   double b1 = (((ind_In3[1]-ind_In3[2]))/2);
   double b2 = (((ind_In4[1]-ind_In3[2]))/2);
   double b3 = (((ind_In3[1]-ind_In4[2]))/2);
   double b4 = (((ind_In4[1]-ind_In4[2]))/2);

   return (v1 * b1 + v2 * b2 + v3 * b3 + v4 * b4);
}
```

Basically my 3rd perceptron looks for the slope relationship between 14 rsi on current timeframe vs the same on a higher level, my case being 4h and Daily respectively and the results show some promise, specially for the Pound.

**>1 year training, 2 years testing data set.**

**For the GBPUSD :**

Original 2 perceptron 4 angle, single model:

[![](https://c.mql5.com/3/404/3153514972757__1.png)](https://c.mql5.com/3/404/3153514972757.png "https://c.mql5.com/3/404/3153514972757.png")

3rd perceptron looking for RSI slope on different timeframe:

[![](https://c.mql5.com/3/404/1840972678868__1.png)](https://c.mql5.com/3/404/1840972678868.png "https://c.mql5.com/3/404/1840972678868.png")

For the GBPJPY:

Original 2 perceptron 4 angle, single model:

[![](https://c.mql5.com/3/404/4345454284469__1.png)](https://c.mql5.com/3/404/4345454284469.png "https://c.mql5.com/3/404/4345454284469.png")

3rd perceptron looking for RSI slope on different timeframe:

[![](https://c.mql5.com/3/404/3769943112205__1.png)](https://c.mql5.com/3/404/3769943112205.png "https://c.mql5.com/3/404/3769943112205.png")

Hope to keep sharing other findings.

Regards!

![Roman Poshtar](https://c.mql5.com/avatar/2019/12/5DE4DBC6-D492.jpg)

**[Roman Poshtar](https://www.mql5.com/en/users/romanuch)**
\|
26 Mar 2023 at 21:32

**Eric Ruvalcaba [#](https://www.mql5.com/en/forum/442302/page3#comment_45868958):**

Hi Roman,

First let me thank you for this amazing contribution, very interesting, you are a genius. I gotta say I liked the 2 perceptron 4 angles options, so I added a 3rd perceptron for a parameter I have found to be quite valuable in my manual trading: RSI in different timeframes.

Basically my 3rd perceptron looks for the slope relationship between 14 rsi on current timeframe vs the same on a higher level, my case being 4h and Daily respectively and the results show some promise, specially for the Pound.

**>1 year training, 2 years testing data set.**

**For the GBPUSD :**

Original 2 perceptron 4 angle, single model:

3rd perceptron looking for RSI slope on different timeframe:

For the GBPJPY:

Original 2 perceptron 4 angle, single model:

3rd perceptron looking for RSI slope on different timeframe:

Hope to keep sharing other findings.

Regards!

Hi! Thanks for your feedback. I'm glad I could help.

![CapeCoddah](https://c.mql5.com/avatar/avatar_na2.png)

**[CapeCoddah](https://www.mql5.com/en/users/capecoddah)**
\|
28 Mar 2023 at 13:13

Roman, Eric:

Here are some of my [testing results](https://www.mql5.com/en/docs/common/TesterStatistics "MQL5 Documentation: TesterStatistics function"), using Roman's original parameters, e.g. H1 2019 12/9-2021 12/9 and forward test 2021 12/09-2022 12/09 for 1 perceptron 4 angle tpsl

Original $2758. I then manually tried to optimize the tpsl and eventually found tp=150 & sl=550 producing $7915.  I tried variable lots and it produced identical results.  I also tried the GA optimization for TPSL around the manual optimized values.  Using the best Optimized  values TP=150 SL=524 which produced a very good optimization profit of $8973, the forward test bent bankrupt!  These results seem to indicate that for optimal profits, the TP & SL also need to be included in the GA optimization runs.  I will be switching my testing also to H4 as I believe the higher time frame "evens out" some of the loss producing reversals on the lower time frame.

Eric: its an interesting idea on using the higher RSI time frame, did you test the H6, H8 or H12 time frames?  And I'm interested to know if you used the MQ iRSI to obtain the results or if you used one of the MTF RSI's available with period interpolations?

Here's a tip for you:

Use the compiler directive #define and #ifdef and you can eliminate the optversions and gain save time replicating your efforts.

//#define OPTIMIZING    //COMMENT OUT FOR PRODUCTION TRADING

#ifdef OPTIMIZING

input int    x1 = 1;

input int    x2 = 1;

input int    x3 = 1;

input int    x4 = 1;

input int    z1 = 1;

input int    z2 = 1;

input int    z3 = 1;

input int    z4 = 1;

input int    Param = 1000;

#else

int    x1 = 1, x2 = 1,x3 = 1, x4 = 1;     //AnglePerceptron4

int    z1 = 1, z2 = 1,z3 = 1, z4 = 1;     //RsiPerceptron4

int    Param = 1000;

#endif

You will have to use this construct again in the OnTick function to comment out the for loop

Cheers to both of you

![Understand and efficiently use OpenCL API by recreating built-in support as DLL on Linux (Part 1): Motivation and validation](https://c.mql5.com/2/52/Recreating-built-in-OpenCL-API-002-avatar.png)[Understand and efficiently use OpenCL API by recreating built-in support as DLL on Linux (Part 1): Motivation and validation](https://www.mql5.com/en/articles/12108)

Bulit-in OpenCL support in MetaTrader 5 still has a major problem especially the one about device selection error 5114 resulting from unable to create an OpenCL context using CL\_USE\_GPU\_ONLY, or CL\_USE\_GPU\_DOUBLE\_ONLY although it properly detects GPU. It works fine with directly using of ordinal number of GPU device we found in Journal tab, but that's still considered a bug, and users should not hard-code a device. We will solve it by recreating an OpenCL support as DLL with C++ on Linux. Along the journey, we will get to know OpenCL from concept to best practices in its API usage just enough for us to put into great use later when we deal with DLL implementation in C++ and consume it with MQL5.

![Creating an EA that works automatically (Part 03): New functions](https://c.mql5.com/2/50/aprendendo_construindo_003_avatar.png)[Creating an EA that works automatically (Part 03): New functions](https://www.mql5.com/en/articles/11226)

Today we'll see how to create an Expert Advisor that simply and safely works in automatic mode. In the previous article, we started to develop an order system that we will use in our automated EA. However, we have created only one of the necessary functions.

![Population optimization algorithms: Invasive Weed Optimization (IWO)](https://c.mql5.com/2/51/invasive-weed-avatar.png)[Population optimization algorithms: Invasive Weed Optimization (IWO)](https://www.mql5.com/en/articles/11990)

The amazing ability of weeds to survive in a wide variety of conditions has become the idea for a powerful optimization algorithm. IWO is one of the best algorithms among the previously reviewed ones.

![Population optimization algorithms: Bat algorithm (BA)](https://c.mql5.com/2/51/Bat-algorithm-avatar.png)[Population optimization algorithms: Bat algorithm (BA)](https://www.mql5.com/en/articles/11915)

In this article, I will consider the Bat Algorithm (BA), which shows good convergence on smooth functions.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/11949&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071841470475874167)

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