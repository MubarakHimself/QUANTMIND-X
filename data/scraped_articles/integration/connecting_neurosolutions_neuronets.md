---
title: Connecting NeuroSolutions Neuronets
url: https://www.mql5.com/en/articles/236
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:08:47.805700
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/236&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083364464004176445)

MetaTrader 5 / Integration


### Introduction

I guess all traders that get acquainted with neuronets think of how great it would be to use them in market analysis. There are a lot of programs that allow to conveniently creating your own networks of any configuration, teaching and testing them in a visual mode. You can export necessary information from the client terminal to a neuronet program and analyze it there.

But what if you want to use the created neuronet in automatic trading? Is it possible to make an Expert Advisor connect to a neuronet and trade in the real time mode?

Yes, it is. Several neuronet programs have the required program interfaces. One of them is called **NeuroSolutions**. Its latest version is 6, however not everybody has it and the most popular version for now is the 5-th one. That's why this article describes interaction with the 5-th version. You need the full distributive of the program; it includes custom Solution Wizard that we need.

### Think of a Strategy

The strategy for our test example will be a simple one. Let's call it **WeekPattern**. It will predict the close price of a bar at its opening on the D1 timeframe using a neuronet. Depending on obtained information, it will make a Buy or Sell deal and hold it for all day long. The price prediction will be base on OHLC values of 5 previous bars. To increase the accuracy of neuronet operation, we are going to send it only the price changes relatively to the open price of the current (zero) bar, instead of prices themselves.

### Preparing Data for Training

Before we start creating a net, let's write a MQL5 script, which will export all the quotes from the client terminal in the required form. This information is required to train the neuronet. The data will be exported to a text file. List field names separated with a comma in the first list of the file. Next lines will be used for comma separated data. Each line is a combination of inputs and outputs of the neuronet. In our case, the script will move back by one bar of price history on each line and write the OHLC values of 6 bars in the line (5 bars from the past are inputs, and one current bar is the output).

The script скрипт **WeekPattern-Export.mq5** should be started at a required timeframe of a required symbol (in our example, it is D1 EURUSD). In the settings you should specify a file name and the required number of lines (260 lines fo D1 is about 1 year history). The full code of the script:

```
#property script_show_inputs
//+------------------------------------------------------------------+
input string    Export_FileName = "NeuroSolutions\\data.csv"; // File for exporting (in the folder "MQL5\Files")
input int       Export_Bars     = 260; // Number of lines to be exported
//+------------------------------------------------------------------+
void OnStart()
  {

   // Create the file
   int file = FileOpen(Export_FileName, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');

   if (file != INVALID_HANDLE)
     {
      // Write the heading of data

      string row="";
      for (int i=0; i<=5; i++)
        {
         if (StringLen(row)) row += ",";
         row += "Open"+i+",High"+i+",Low"+i+",Close"+i;
        }
      FileWrite(file, row);

      // Copy all required information from the history

      MqlRates rates[], rate;
      int count = Export_Bars + 5;
      if (CopyRates(Symbol(), Period(), 1, count, rates) < count)
        {
         Print("Error! Not enough history for exporting of data.");
         return;
        }
      ArraySetAsSeries(rates, true);

      // Write data

      for (int bar=0; bar<Export_Bars; bar++)
        {
         row="";
         double zlevel=0;
         for (int i=0; i<=5; i++)
           {
            if (StringLen(row)) row += ",";
            rate = rates[bar+i];
            if (i==0) zlevel = rate.open; // level for counting of prices
            row += NormalizeDouble(rate.open -zlevel, Digits()) + ","
                 + NormalizeDouble(rate.high -zlevel, Digits()) + ","
                 + NormalizeDouble(rate.low  -zlevel, Digits()) + ","
                 + NormalizeDouble(rate.close-zlevel, Digits());
           }
         FileWrite(file, row);
        }

      FileClose(file);
      Print("Export of data finished successfully.");
     }
   else Print("Error! Failed to create the file for data export. ", GetLastError());
  }
//+------------------------------------------------------------------+
```

After exporting the data, we obtain the file **data.csv**; its first lines (for example) look as following:

```
Open0,High0,Low0,Close0,Open1,High1,Low1,Close1,Open2,High2,Low2,Close2,Open3,High3,Low3,Close3,Open4,High4,Low4,Close4,Open5,High5,Low5,Close5
0,0.00463,-0.0041,0.00274,-0.00518,0.00182,-0.00721,-6e-005,0.00561,0.00749,-0.00413,-0.00402,0.02038,0.02242,0.00377,0.00565,0.03642,0.0379,0.01798,0.02028,0.0405,0.04873,0.03462,0.03647
0,0.007,-0.00203,0.00512,0.01079,0.01267,0.00105,0.00116,0.02556,0.0276,0.00895,0.01083,0.0416,0.04308,0.02316,0.02546,0.04568,0.05391,0.0398,0.04165,0.04504,0.05006,0.03562,0.0456
0,0.00188,-0.00974,-0.00963,0.01477,0.01681,-0.00184,4e-005,0.03081,0.03229,0.01237,0.01467,0.03489,0.04312,0.02901,0.03086,0.03425,0.03927,0.02483,0.03481,0.02883,0.04205,0.02845,0.03809
```

This is the format, which can be understood by NeuroSolutions. Now we can start creating and training a net.

### Creating Neuronet

In NeuroSolutions you can quickly create a neuronet, even if you see this program for the first time and know little about neuronets. To do it, select the wizard for beginners **NeuralExpert (Beginner)** at the program start:

![](https://c.mql5.com/2/2/Untitled-1_copy.gif)

In it you should specify a problem type that should be solved by the neuronet:

![](https://c.mql5.com/2/2/Untitled-2_copy.gif)

Then specify the file with training information, which we've created in the previous chapter:

![](https://c.mql5.com/2/2/Untitled-3_copy.gif)

As the inputs of the net, select all the fields of the file except the fields of the zero bar:

![](https://c.mql5.com/2/2/Untitled-4_copy.gif)

Since we don't have text fields, don't select anything:

![](https://c.mql5.com/2/2/Untitled-5_copy.gif)

Specify our file with information again:

![](https://c.mql5.com/2/2/Untitled-6_copy.gif)

Select only one output of our net:

![](https://c.mql5.com/2/2/Untitled-7_copy.gif)

The wizard suggests creating the simplest net on default. Let's do so:

![](https://c.mql5.com/2/2/Untitled-8_copy.gif)

The wizard has finished its work creating a neuronet for us (not a trained net, just a simple structure):

![](https://c.mql5.com/2/2/Untitled-9_copy.png)

Now we can work with it. We can train it, test and use for data analysis.

If you click the **Test** button, you'll be able to see how the untrained net will solve our problem. Answer the questions of the testing wizard:

![](https://c.mql5.com/2/2/Untitled-10_copy.gif)

Perform the test on the basis of information from the same file:

![](https://c.mql5.com/2/2/Untitled-11_copy.gif)

![](https://c.mql5.com/2/2/Untitled-12_copy.gif)

![](https://c.mql5.com/2/2/Untitled-13_copy.gif)

The test is over. In the window "Output vs. Desired Plot" you can see the chart that shows the values obtained from the net (the red color) on our history and the real values (the blue color). You can see that they pretty different:

![](https://c.mql5.com/2/2/Untitled-16_copy.gif)

Now let's train the net. In order to do it, click the green button **Start** on the toolbar below the menu. The training will be finished after a few second and the chart will change:

![](https://c.mql5.com/2/2/Untitled-17_copy.gif)

Now in the chart you can see that the net shows the results that seem to be true. Therefore, you can use it for trading. Save the net under the name **WeekPattern**.

### Export the neuronet in a DLL

Without exiting NeuroSolutions, click the **CSW** button that starts the **Custom Solution Wizard**. We need to generate a DLL from the current neuronet.

![](https://c.mql5.com/2/2/Untitled-18_copy.png)

The wizard can generate DLLs for different programs. As far as I understood, for compilation of the DLL you need Visual C++ of one of the following versions: 5.0/6.0/7.0 (.NET 2002)/7.1 (.NET 2003)/8.0 (.NET 2005). For some reason, it doesn't use the Express version (I've checked it).

There is no MetaTrader in the list of target applications. That's why select Visual C++.

![](https://c.mql5.com/2/2/Untitled-19_copy.gif)

Path to save the result:

![](https://c.mql5.com/2/2/Untitled-20_copy.gif)

If everything has passed successfully, the wizard tells about:

![](https://c.mql5.com/2/2/Untitled-21_copy.png)

A lot of files will appear in the folder specified in the wizard. The ones we need most are: **WeekPattern.dll**, it contains our neuronet with the program interface to it; and the file **WeekPattern.nsw** that contains the balance settings of the neuronet after its training. Among the other files you can find the one with an example of working with this DLL-neuronet. In this case it is the Visual C++ 6 project.

![](https://c.mql5.com/2/2/Untitled-22_copy.gif)

### Connecting DLL-Neuronet to MetaTrader

Created in the previous chapter DLL-neuronet is intended for using in Visual C++ projects. It operates with the objects of a complex structure that would be hard to describe on MQL5 or even impossible. That is why we are not going to connect this DLL to MetaTrader directly. Instead of it we are going to create a small DLL adapter. This adapter will contain one simple function for working with the neuronet. It will create the network, pass it the input information and return the output data.

This adapter will be easily called from MetaTrader 5. And the adapter will connect to the DLL-neuronet created in NeuroSolutions. Since the adapter will be written in Visual C++, it won't have any problems with objects of this DLL.

![](https://c.mql5.com/2/2/adapter-v__1.png)

There is no need to create the DLL adapter yourself. The ready-made DLL is attached to this article. The adapter works with any DLL-neuronet created in NeuroSolutions. You can skip further reading of this chapter.

But if you have an experience in programming in C++ and if you are interested how to create such adapter, read this chapter till the end. Probably, you will have an interest to improve it, since some other functions can be exported from a DLL-neuronet. For example, the training function (for an Expert Advisor do adapt to a changing market, re-training the net automatically). You can learn the full list of functions by analyzing the example generated by the Custom Solution Wizard, which is shown in the previous chapter.

We'll need only several files from that example.

In Visual C++ (the same version as used in Custom Solution Wizard) create an empty DLL project named **NeuroSolutionsAdapter** and copy the NSNetwork.h, NSNetwork.cpp and StdAfx.h files from example to it. Also create an empty main.cpp file:

![](https://c.mql5.com/2/2/Untitled-3_copy__1.gif)

Write the following code in the main.cpp file:

```
#include "stdafx.h"
#include "NSNetwork.h"

extern "C" __declspec(dllexport) int __stdcall CalcNeuralNet(
                LPCWSTR dllPath_u, LPCWSTR weightsPath_u,
                double* inputs, double* outputs)
{
    // Transform the lines from Unicode to normal ones
    CString dllPath     (dllPath_u);
    CString weightsPath (weightsPath_u);

    // Create neuronet
    NSRecallNetwork nn(dllPath);
    if (!nn.IsLoaded()) return (1);

    // Load balances
    if (nn.LoadWeights(weightsPath) != 0) return (2);

    // Pass input data and calculate the output
    if (nn.GetResponse(1, inputs, outputs) != 0) return (3);

    return 0;
}
```

Build. The DLL adapter is ready!

### Using Neuronet in Expert Advisor

Well, we have already created several files. Let me list the files, which are necessary for the Expert Advisor to work, and the folders where you should put them. All those files are attached to the article.

| File | Description | Where to put (in the terminal folder) |
| --- | --- | --- |
| WeekPattern.dll | our DLL-neuronet created in NeuroSolutions | MQL5\\Files\\NeuroSolutions\ |
| WeekPattern.nsw | balance settings of our neuronet | MQL5\\Files\\NeuroSolutions\ |
| NeuroSolutionsAdapter.dll | universal DLL-adapter for any DLL-neuronet | MQL5\\Libraries\ |

Here is the full code of the Expert Advisor **WeekPattern.mq5**. For convenience of searching and further modification, all the things concerning the neuronet are placed in the separate class CNeuroSolutionsNeuralNet.

```
input double    Lots = 0.1;
//+------------------------------------------------------------------+
// Connect the DLL adapter, using which we are going to use the DLL neuronet created in NeuroSolutions
#import "NeuroSolutionsAdapter.dll"
int CalcNeuralNet(string dllPath, string weightsPath, double& inputs[], double& outputs[]);
#import
//+------------------------------------------------------------------+
class CNeuroSolutionsNeuralNet
{
private:
   string dllPath;     // Path to a DLL neuronet created in NeuroSolutions
   string weightsPath; // Path to a file of the neuronet balances
public:
   double in[20]; // Neuronet inputs - OHLC of 5 bars
   double out[1]; // Neuronet outputs - Close of a current bar

   CNeuroSolutionsNeuralNet();
   bool Calc();
};
//+------------------------------------------------------------------+
void CNeuroSolutionsNeuralNet::CNeuroSolutionsNeuralNet()
{
   string terminal = TerminalInfoString(TERMINAL_PATH);
   dllPath     = terminal + "\\MQL5\\Files\\NeuroSolutions\\WeekPattern.dll";
   weightsPath = terminal + "\\MQL5\\Files\\NeuroSolutions\\WeekPattern.nsw";
}
//+------------------------------------------------------------------+
bool CNeuroSolutionsNeuralNet::Calc()
  {
   // Get current quotes for the neuronet
   MqlRates rates[], rate;
   CopyRates(Symbol(), Period(), 0, 6, rates);
   ArraySetAsSeries(rates, true);

   // Fill the array of input data of the neuronet
   double zlevel=0;
   for (int bar=0; bar<=5; bar++)
     {
      rate = rates[bar];
      // 0 bar is not taken for input
      if (bar==0) zlevel=rate.open; // level of price calculation
      // 1-5 bars are inputed
      else
        {
         int i=(bar-1)*4; // input number
         in[i  ] = rate.open -zlevel;
         in[i+1] = rate.high -zlevel;
         in[i+2] = rate.low  -zlevel;
         in[i+3] = rate.close-zlevel;
        }
     }

   // Calculate the neuronet in the NeuroSolutions DLL (though the DLL adapter)
   int res = CalcNeuralNet(dllPath, weightsPath, in, out);
   switch (res)
     {
      case 1: Print("Error of creating neuronet from DLL \"", dllPath, "\""); return (false);
      case 2: Print("Error of loading balances to neuronet from the file \"", weightsPath, "\""); return (false);
      case 3: Print("Error of calculation of neuronet");  return (false);
     }

   // Output of the neuronet has appeared in the array out, you shouldn't do anything with it

   return (true);
  }
//+------------------------------------------------------------------+

CNeuroSolutionsNeuralNet NN;
double Prognoze;

//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
//+------------------------------------------------------------------+
void OnTick()
  {
   // Get the price prediction from the neuronet
   if (NN.Calc()) Prognoze = NN.out[0];
   else           Prognoze = 0;

   // Perform necessary trade actions
   Trade();
  }
//+------------------------------------------------------------------+
void Trade()
  {

   // Close an open position if it is opposite to the prediction

   if(PositionSelect(_Symbol))
     {
      long type=PositionGetInteger(POSITION_TYPE);
      bool close=false;
      if((type == POSITION_TYPE_BUY)  && (Prognoze <= 0)) close = true;
      if((type == POSITION_TYPE_SELL) && (Prognoze >= 0)) close = true;
      if(close)
        {
         CTrade trade;
         trade.PositionClose(_Symbol);
        }
     }

   // If there is no positions, open one according to the prediction

   if((Prognoze!=0) && (!PositionSelect(_Symbol)))
     {
      CTrade trade;
      if(Prognoze > 0) trade.Buy (Lots);
      if(Prognoze < 0) trade.Sell(Lots);
     }
  }
//+------------------------------------------------------------------+
```

A good way to check, whether we have connected the neuronet correctly, is to run the Expert Advisor in the strategy tester on the same time period as the one used for training the neuronet.

Well, as experienced traders say, the neuronet is "adapter" for that period. So it is trained to recognize and inform about a profit signal for those exact data patterns, which dominate in this specific period. A profitability graph of an Expert Advisor drawn for such a period should be ascending.

Let's check it. In our case it will be the following beautiful chart:

![](https://c.mql5.com/2/2/Untitled-2_copy__2.gif)

That means that everything has been connected correctly.

And for the statistics, here are the other reports on testing of the Expert Advisor:

![](https://c.mql5.com/2/2/Untitled-1_copy__2.gif)

![](https://c.mql5.com/2/2/Untitled-3_copy__2.gif)

Just in case, let me give explanations for novice developers of trade strategies and neuronets.

The profitability of an Expert Advisor on a period, which was used for its optimization (training of its neuronet), doesn't tell about the total profitability of the EA. In other words, it doesn't guarantee its profitability on the other period. There can be other dominating patterns.

Creation of trade strategies that keep their profitability behind the training period is a complex and complicated task. You shouldn't count on the NeuroSolutions or any other neuronet application to solve this problem for you. It only creates a neuronet for you data.

Those are the reasons why I didn't give here the result of forward testing of the obtained Expert Advisor. Creation of a profitable trade strategy is not the aim of this article. The aim is to tell how to connect a neuronet to an Expert Advisor.

### Conclusion

Now traders have another powerful and easy tool for automatic trading analysis and trading. Using it together with a deep understanding of principles and possibilities of neuronets as well as the rules of training them will allow you following the road of creation of profitable Expert Advisors.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/236](https://www.mql5.com/ru/articles/236)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/236.zip "Download all attachments in the single ZIP archive")

[dll\_nsw.zip](https://www.mql5.com/en/articles/download/236/dll_nsw.zip "Download dll_nsw.zip")(383.37 KB)

[weekpattern.mq5](https://www.mql5.com/en/articles/download/236/weekpattern.mq5 "Download weekpattern.mq5")(3.78 KB)

[weekpattern-export.mq5](https://www.mql5.com/en/articles/download/236/weekpattern-export.mq5 "Download weekpattern-export.mq5")(2.04 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Trading strategy based on the improved Doji candlestick pattern recognition indicator](https://www.mql5.com/en/articles/12355)
- [Improved candlestick pattern recognition illustrated by the example of Doji](https://www.mql5.com/en/articles/9801)
- [Neural Networks Cheap and Cheerful - Link NeuroPro with MetaTrader 5](https://www.mql5.com/en/articles/830)
- [3 Methods of Indicators Acceleration by the Example of the Linear Regression](https://www.mql5.com/en/articles/270)
- [Decreasing Memory Consumption by Auxiliary Indicators](https://www.mql5.com/en/articles/259)
- [Parallel Calculations in MetaTrader 5](https://www.mql5.com/en/articles/197)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/3184)**
(75)


![Zhiqiang Zhu](https://c.mql5.com/avatar/2018/1/5A54757B-34DB.jpg)

**[Zhiqiang Zhu](https://www.mql5.com/en/users/zhiqiang_2016)**
\|
3 Jun 2019 at 10:20

It is also a variation of parameter optimisation. The essence is still to optimise historical data, profitability is only based on past data and is not decisive for the future.

What is the point of finding profit maximisation and risk minimisation in historical data for the future?

It's OK to learn, but it's a big mistake to rely on this for profit!

![mensz](https://c.mql5.com/avatar/avatar_na2.png)

**[mensz](https://www.mql5.com/en/users/mensz)**
\|
5 Jun 2019 at 19:56

If the combination of useful indicators of data, of course, will be useful, the problem is that this software can not smoothly generate DLL, option VC6 and too old, if not directly generate DLL that is too complicated, only to the masters to use, I installed the vc2008 and vc2010 programming software, tick vc6 option or prompt the computer does not have a C + + + function!


![Tengfei Xu](https://c.mql5.com/avatar/2018/12/5C07E9A2-ABE2.png)

**[Tengfei Xu](https://www.mql5.com/en/users/easytime_xtf)**
\|
8 Oct 2019 at 04:41

I can generate dll files! However, I would like to know how to get indicator data for multiple currency pairs to train on


![Yu Zhang](https://c.mql5.com/avatar/2022/2/620A27F9-FE06.jpg)

**[Yu Zhang](https://www.mql5.com/en/users/i201102053)**
\|
25 May 2021 at 12:26

The question is i don't konw what is modle used in NeuroSolutions.

![Ivan Butko](https://c.mql5.com/avatar/2017/1/58797422-0BFA.png)

**[Ivan Butko](https://www.mql5.com/en/users/capitalplus)**
\|
6 Sep 2022 at 00:14

The programme is no longer supported. It requires registration, gives an error and sends you to register on the site, which is no longer working.

If anyone has it on their computer, please share it with me.


![Charts and diagrams in HTML](https://c.mql5.com/2/0/html_Chart_MQL5.png)[Charts and diagrams in HTML](https://www.mql5.com/en/articles/244)

Today it is difficult to find a computer that does not have an installed web-browser. For a long time browsers have been evolving and improving. This article discusses the simple and safe way to create of charts and diagrams, based on the the information, obtained from MetaTrader 5 client terminal for displaying them in the browser.

![Trade Events in MetaTrader 5](https://c.mql5.com/2/0/trade_events.png)[Trade Events in MetaTrader 5](https://www.mql5.com/en/articles/232)

A monitoring of the current state of a trade account implies controlling open positions and orders. Before a trade signal becomes a deal, it should be sent from the client terminal as a request to the trade server, where it will be placed in the order queue awaiting to be processed. Accepting of a request by the trade server, deleting it as it expires or conducting a deal on its basis - all those actions are followed by trade events; and the trade server informs the terminal about them.

![Drawing Channels - Inside and Outside View](https://c.mql5.com/2/0/channels_MQL5.png)[Drawing Channels - Inside and Outside View](https://www.mql5.com/en/articles/200)

I guess it won't be an exaggeration, if I say the channels are the most popular tool for the analysis of market and making trade decisions after the moving averages. Without diving deeply into the mass of trade strategies that use channels and their components, we are going to discuss the mathematical basis and the practical implementation of an indicator, which draws a channel determined by three extremums on the screen of the client terminal.

![MQL5 Wizard: How to Create a Module of Trailing of Open Positions](https://c.mql5.com/2/0/MQL5_Wizard_Trailing_Stop__1.png)[MQL5 Wizard: How to Create a Module of Trailing of Open Positions](https://www.mql5.com/en/articles/231)

The generator of trade strategies MQL5 Wizard greatly simplifies the testing of trading ideas. The article discusses how to write and connect to the generator of trade strategies MQL5 Wizard your own class of managing open positions by moving the Stop Loss level to a lossless zone when the price goes in the position direction, allowing to protect your profit decrease drawdowns when trading. It also tells about the structure and format of the description of the created class for the MQL5 Wizard.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/236&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083364464004176445)

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