---
title: Practical application of neural networks in trading. It's time to practice
url: https://www.mql5.com/en/articles/7370
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:35:45.040082
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=whxbprbpymlgxwzjkgpswqmygulcjhqs&ssn=1769186141543657164&ssn_dr=0&ssn_sr=0&fv_date=1769186141&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7370&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Practical%20application%20of%20neural%20networks%20in%20trading.%20It%27s%20time%20to%20practice%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918614130859504&fz_uniq=5070411070272640348&sv=2552)

MetaTrader 5 / Examples


### Introduction

In my previous article ["Practical application of neural networks in trading"](https://www.mql5.com/en/articles/7031), I described the general points in creating a trading system using **Neural Network Modules** (NNM). In this article, we will test the NNM in practice. We will also try to create an NNM-based automated trading system.

All operations will be performed for the EURUSD pair. However, the neural network module is universal as a program, and it can work with different trading instruments and their combinations. Thus, it is possible to use one NNM integrating multiple neural networks trained for different currency pairs.

The technology concerning the preparation of historical data required to train neural networks is beyond the scope of this article. Our NNM will use neural networks trained on historical data close to the article publication time.

The resulting NNM is fully ready to be used for trading. In order to be able to introduce the complex within one article, I had to modify it so as to combine several neural network module functions in one program. Also, I moved to the NNM certain deal execution conditions from a trading Expert Advisor. The latter only concerns the NNM part intended for real trading. As for the neural network testing and its response optimization in the offline mode, all the relevant functions and deal execution conditions are still performed by the appropriate Expert Advisors. As for me, I prefer the NN module only having a trading function. I believe that the NNM used for trading must be as simple as possible, and it should not contain any additional functions. The functions themselves should be implemented outside the trading complex. In our case, these functions include training, testing and optimization. Deal execution conditions should better be implemented in the NNM, so that the signals could be received in a binary form. Although all the NNM execution options have confirmed their viability in practice.

The technology of NNM preparation and training on the Matlab platform will be described in more detail in the next article.

Also, the system might be translated to Python in the future. A relevant short video is available at the end of this article.

NNMs generate positive results in a longer testing period when tested using historical data received from the brokers with which you trade. This part could also be unified, but I don't think this would be useful.

### The EURUSD\_MT5 Neural Network Module

The below figure shows how the Neural Network Module looks like at the initial launch.

### ![НСМ EURUSD_MT5](https://c.mql5.com/2/37/EURUSD_MT5_2020-01-12.png)

01. The Online block is designed to start and stop neural networks during real trading and when testing in the visual mode.
02. Information fields with the conditions for the signal line crossing the response line of neural networks, when the Online block is activated.
03. "Train" is a demonstration block which is designed for training and "retraining" (?) of neural networks.
04. Fields for outputting the response values of neural networks. Left - neural network responses; right - the signal line. Lower - current bar, upper - previous bar.
05. The Offline block is designed for outputting neural network responses in a test sample to an array.
06. Field for entering the averaging value for the neural network response line when using the Online block. (Signal line period). Editable value.
07. Blocks "Net1,2,3" — three submodules of networks trained in different segments of a time series. Each block includes two neural networks.
08. NNM operation end time when using the Online block.
09. Field for entering the NNM operation period in hours, when using the Online block. Editable value.
10. Counting the time elapsed since NNM start, when using the Online block.

### Attachments

1. MyAppInstaller\_mcr.exe — installation file.
2. EURUSD\_MT5.exe — the neural network module itself.
3. EURUSD\_MT5var.exe — a variant of the Neural Network Module.
4. net1-net6.mat — neural networks of three subnets, Net1-Net3. To be used as an example of their training and testing.

1. ExpertMatlabPodkach\_MT5.mq5 and ExpertMatlab\_MT5.mq5 — two EAs required for preparing historical data for NNM offline testing.
2. NWI.mq5 — indicator which visually displays NNM responses.
3. Matlab\_MT5.mq5 — an EA for testing and optimizing NNM response in the strategy tester.
4. ExpertMatlabReal\_MT5.mq5 — an EA for the online operation on real or demo accounts, as well as for testing in the visualization mode.
5. EURUSDData.csv — file with training data.

### Program installation

YouTube

Before the first use of applications compiled by Matlab, you must use MyAppInstaller\_mcr.exe. This application will be used to install MATLAB Runtime and the neural network module itself.

After program installation, place the EURUSD\_MT5.exe shortcut to the ...\\Common\\Files folder in the data directory. By doing so, we provide a convenient way to launch the system. All Expert Advisors and NNMs write files to this folder. The NNM offers to search files from the directory where the shortcut is located.

![File EURUSD_MT5](https://c.mql5.com/2/37/Files_2020-01-15_15.08.12.png)

1\. Shortcut EURUSD\_MT5

Assign a working folder.

![Specify the path](https://c.mql5.com/2/38/3lsi3lxc__EURUSD_MT5_2020-03-01_11.10.03.png)

2\. Specify the path

### Practical use

Next, we have four options for using the NNM:

1. Online trading on demo or real accounts
2. NNM testing in the visual mode
3. Neural Network training
4. Receiving responses from neural network blocks to optimize a trading strategy

You might say that the logical order of the above points would be different. However, the Neural Network Nodule was initially created to execute the first point. Therefore, I prefer this implementation. The second, the third and the fourth options are beyond the NNM, because they can be performed during the general system setup. For this article, I have combined all these stages into one mono-block for a better understanding of the process of general system preparation for real work.

Let's consider these options in more detail.

1\. **Online trading on demo or real accounts**

1\. В онлайн режиме на демо или реальном счетах - YouTube

[Photo image of НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4](https://www.youtube.com/channel/UCScAAn_sRRaKHdNIxl0aI9A?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7370)

НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4

417 subscribers

[1\. В онлайн режиме на демо или реальном счетах](https://www.youtube.com/watch?v=VJwftJMeApU)

НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

0:00

0:00 / 4:13

•Live

•

As we can see, the system startup time takes less than five minutes.

Before that, it is necessary to configure Excel. The purpose of the configuration is to ensure that data from scripts and Expert Advisors are written in a numerical format. Otherwise Matlab will read this data incorrectly. Set a point as the separator of the integer and fractional parts. This can be done directly or using the system separator.

![Excel parameters](https://c.mql5.com/2/38/xoj62z3qc_Excel_2020-03-07_14.47.30.png)

3\. Excel parameters

Before the initial NNM launch, create a history downloading file using the Expert Advisor. Launch ExpertMatlabPodkach\_MT5.ex5 in the Strategy Tester.

![Chart](https://c.mql5.com/2/38/23476973_-_MetaQuotes.png)

![Launch ExpertMatlabPodkach_MT5](https://c.mql5.com/2/38/23476973_-_MetaQuotes_1.png)

4\. Launch ExpertMatlabPodkach\_MT5.ex5

As you can see, the EA launch time should be selected so that it covers the period of three days before the operation start.

As a result, we receive the EURUSDTestPodkach.csv file.

![File EURUSDPodkach_MT5](https://c.mql5.com/2/37/Files_2020-01-15_15.09.32.png)

5\. File EURUSDTestPodkach.csv

Open the file and edit it by deleting all rows except for the data concerning the opening of the last hour on the date preceding the system launch day.

![Delete unnecessary rows](https://c.mql5.com/2/38/Microsoft_Excel_-_EURUSDTestPodkach_2020-01-18.png)

6\. Delete unnecessary rows

Now we can launch ExpertMatlabReal\_MT5.ex5.

```
#include<Trade\Trade.mqh>
//--- An object for performing trading operations
CTrade  trade;

input int LossBuy;
input int ProfitBuy;
input int LossSell;
input int ProfitSell;

int BarMax;
int BarMin;
int handleInput;
int handleInputPodkach;
int handleBar;
int Con;
int Bar;

double DibMax;
double DibMin;

double in[32];

int Order01;
int Order1;

ulong TicketBuy1;
ulong TicketSell0;

bool send1;
bool send0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

   handleInputPodkach=FileOpen("EURUSDTestPodkach.csv",FILE_READ|FILE_CSV|FILE_ANSI|FILE_COMMON,";");
   if(handleInputPodkach==INVALID_HANDLE)
      Alert("No file EURUSDTestPodkach.csv");

   in[0]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[1]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[2]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[3]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[4]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[5]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[6]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[7]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[8]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[9]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[10]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[11]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[12]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[13]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[14]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[15]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[16]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[17]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[18]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[19]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[20]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[21]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[22]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[23]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[24]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[25]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[26]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[27]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[28]=1/StringToDouble(FileReadString(handleInputPodkach))-1;
   in[29]=1/StringToDouble(FileReadString(handleInputPodkach))-1;

   FileClose(handleInputPodkach);

//--- Setting MagicNumber to identify EA's orders
   int MagicNumber=123456;
   trade.SetExpertMagicNumber(MagicNumber);
//--- Setting allowable slippage in points for buying/selling
   int deviation=10;
   trade.SetDeviationInPoints(deviation);
//--- order filling mode, use the mode that is allowed by the server
   trade.SetTypeFilling(ORDER_FILLING_RETURN);
//--- The function to be used for trading: true - OrderSendAsync(), false - OrderSend()
   trade.SetAsyncMode(true);

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   FileClose(handleInput);
   FileClose(handleBar);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   MqlDateTime stm;
   TimeToStruct(TimeCurrent(),stm);

   if(stm.hour==1)
      DibMax=iHigh(NULL,PERIOD_H1,1);
   if(stm.hour>0)
     {
      if(iHigh(NULL,PERIOD_H1,1)>DibMax && iTime(NULL,PERIOD_H1,0)>1)
        {
         in[20]=iOpen(NULL,PERIOD_D1,0)-iLow(NULL,PERIOD_H1,1);
         in[21]=iHigh(NULL,PERIOD_H1,1)-iOpen(NULL,PERIOD_D1,0);
         in[22]=iHigh(NULL,PERIOD_D1,1)-iLow(NULL,PERIOD_D1,1);
         in[23]=iHigh(NULL,PERIOD_D1,1)-iOpen(NULL,PERIOD_H1,0);
         in[24]=iOpen(NULL,PERIOD_H1,0)-iLow(NULL,PERIOD_D1,1);
        }
     }
   if(iHigh(NULL,PERIOD_H1,1)>DibMax)
      DibMax=iHigh(NULL,PERIOD_H1,1);
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
   if(stm.hour==1)
      DibMin=iLow(NULL,PERIOD_H1,1);
   if(stm.hour>0)
     {
      if(iLow(NULL,PERIOD_H1,1)<DibMin && iTime(NULL,PERIOD_H1,0)>1)
        {
         in[25]=iOpen(NULL,PERIOD_D1,0)-iLow(NULL,PERIOD_H1,1);
         in[26]=iHigh(NULL,PERIOD_H1,1)-iOpen(NULL,PERIOD_D1,0);
         in[27]=iHigh(NULL,PERIOD_D1,1)-iLow(NULL,PERIOD_D1,1);
         in[28]=iHigh(NULL,PERIOD_D1,1)-iOpen(NULL,PERIOD_H1,0);
         in[29]=iOpen(NULL,PERIOD_H1,0)-iLow(NULL,PERIOD_D1,1);
        }
     }
   if(iLow(NULL,PERIOD_H1,1)<DibMin)
      DibMin=iLow(NULL,PERIOD_H1,1);

   in[30]=iHigh(NULL,PERIOD_D1,1)-iOpen(NULL,PERIOD_H1,0);
   in[31]=iOpen(NULL,PERIOD_H1,0)-iLow(NULL,PERIOD_D1,1);

   if(Bar<Bars(NULL,PERIOD_H1)&& stm.hour==0)
     {
      for(int i=19; i>=10; i--)
        {
         in[i-10]=in[i];
        }

      for(int i=29; i>=20; i--)
        {
         in[i-10]=in[i];
        }
     }

   handleInput=FileOpen("Input_mat.txt",FILE_TXT|FILE_WRITE|FILE_ANSI|FILE_SHARE_READ|FILE_COMMON,";");

   FileWrite(handleInput,

             1/(in[0]+1),1/(in[1]+1),1/(in[2]+1),1/(in[3]+1),1/(in[4]+1),1/(in[5]+1),1/(in[6]+1),1/(in[7]+1),1/(in[8]+1),1/(in[9]+1),1/(in[10]+1),1/(in[11]+1),1/(in[12]+1),1/(in[13]+1),1/(in[14]+1),1/(in[15]+1),
             1/(in[16]+1),1/(in[17]+1),1/(in[18]+1),1/(in[19]+1),1/(in[20]+1),1/(in[21]+1),1/(in[22]+1),1/(in[23]+1),1/(in[24]+1),1/(in[25]+1),1/(in[26]+1),1/(in[27]+1),1/(in[28]+1),1/(in[29]+1),1/(in[30]+1),1/(in[31]+1));

   FileClose(handleInput);

   handleBar=FileOpen("Bar.txt",FILE_TXT|FILE_WRITE|FILE_ANSI|FILE_SHARE_READ|FILE_COMMON,";");

   FileWrite(handleBar,stm.hour);

   FileClose(handleBar);

   Order01=FileOpen("Open1.txt",FILE_CSV|FILE_READ|FILE_ANSI|FILE_SHARE_READ|FILE_COMMON," ");

   Order1=StringToInteger(FileReadString(Order01));

   FileClose(Order01);

   int    digits=(int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
   double point=SymbolInfoDouble(_Symbol,SYMBOL_POINT);
   double PriceAsk=SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   double PriceBid=SymbolInfoDouble(_Symbol,SYMBOL_BID);

   double SL1=NormalizeDouble(PriceBid-LossBuy*point,digits);
   double TP1=NormalizeDouble(PriceAsk+ProfitBuy*point,digits);
   double SL0=NormalizeDouble(PriceAsk+LossSell*point,digits);
   double TP0=NormalizeDouble(PriceBid-ProfitSell*point,digits);

   if(Bar<Bars(NULL,PERIOD_H1))
      Con=0;

   Comment(Order1,"  ",Con);

   if(LossBuy==0)
      SL1=0;

   if(ProfitBuy==0)
      TP1=0;

   if(LossSell==0)
      SL0=0;

   if(ProfitSell==0)
      TP0=0;

   if(Order1==0 && Bar<Bars(NULL,PERIOD_H1) && Con==0)
      send0=true;

   if(Order1==1 && Bar<Bars(NULL,PERIOD_H1) && Con==0)
      send1=true;

//---------Buy0

   if(send1==false  &&  Bar==Bars(NULL,PERIOD_H1) && Order1==1 && Con>=1 && iLow(NULL,PERIOD_H1,1)<iLow(NULL,PERIOD_H1,2) && stm.hour>15 && stm.hour<20)
     {
      send1=trade.PositionOpen(_Symbol,ORDER_TYPE_BUY,1,PriceAsk,SL1,TP1);
      TicketBuy1 = trade.ResultDeal();
     }

   if(send1==true &&  Bar==Bars(NULL,PERIOD_H1) && Order1==0 && Con>=1 && iHigh(NULL,PERIOD_H1,1)>iHigh(NULL,PERIOD_H1,2))
     {
      trade.PositionClose(TicketBuy1);
      send1=false;
     }
//---------Sell0
   if(send0==false  &&  Bar==Bars(NULL,PERIOD_H1) && Order1==0 && Con>=1 && iHigh(NULL,PERIOD_H1,1)>iHigh(NULL,PERIOD_H1,2) && stm.hour>
        11 && stm.hour<14)
     {
      send0=trade.PositionOpen(_Symbol,ORDER_TYPE_SELL,1,PriceBid,SL0,TP0);
      TicketSell0 = trade.ResultDeal();
     }

   if(send0==true &&  Bar==Bars(NULL,PERIOD_H1) && Order1==1 && Con>=1 && iLow(NULL,PERIOD_H1,1)<iLow(NULL,PERIOD_H1,2))
     {
      trade.PositionClose(TicketSell0);
      send0=false;
     }
//-----------------------------------------------------------------------
   Bar=Bars(NULL,PERIOD_H1);
   Con++;
  }
//------------------------------------------------------------------------
```

As for the time limitations in the EA designed for trading, I prefer adding them straight into the code, after optimization. I do this to minimize the risk of an error which could occur with external variables.

![ExpertMatlabReal_MT5](https://c.mql5.com/2/38/ExpertMatlabReal_MT5_1.00_2020-03-14_19.17.45.png)

7\. Launch ExpertMatlabReal\_MT5.ex5

As for the Expert Advisor designed for optimization, they should be indicated as external variables.

![Matlab_MT5.ex5](https://c.mql5.com/2/38/Matlab_MT5_2__1.png)

7.1 Matlab\_MT5.ex5

If you forget to generate the history downloading file, you will receive a warning to prevent the EA from generating false signals.

![Warning](https://c.mql5.com/2/38/gq59r_2020-01-19_14.50.26.png)

8\. Warning

The EA will write two filed to the ...\\Common\\Files folder. The Input\_mat.txt file contains input data for the NNM.

![Expert Advisor response](https://c.mql5.com/2/38/Files_2020-01-19_15.05.25.png)

9\. Expert Advisor response

Launch the Neural Network Module EURUSD\_MT5.exe. See for the workspace to appear.

![Workspace](https://c.mql5.com/2/38/EURUSD_MT5_2020-01-19_15.27.24.png)

10\. NSM workspace

The modifiable parameters of blocks Net2 and Net3 in this program variant cannot be changed.

Press "Start" and select the net1.m file (or another neural network file).

![Press "Start"](https://c.mql5.com/2/38/Select_File_to_Open_2020-03-01_14.14.09.png)

11\. Press "Start"

You might have noticed that the NNM appearance has a bit changed since the first article. However, its functional capabilities remain the same.

The neural network module will start its work.

![NNM is trading](https://c.mql5.com/2/38/EURUSD_MT5_2020-01-19_15.49.32.png)

12\. NNM in the trading state

During the NNM operation, we cannot change any of its parameters.

The upper left corner of the chart shows the numbers received by the Expert Advisor from the NNM: 1.0, -1.0 and 0.0. This means that the EA has received a signal to buy (sell). From Net2 — stay out of the market. From Net3 — a signal to buy (sell). Since this program version is intended for evaluation purposes, we will not receive variable signals from Net1 and Net2.

In this example, our system uses the signal line of the 5hr-smoothed NNM response. In order to avoid receiving a false initial signal, the module must be launched five hours before trading time, based on the position opening conditions. The time depends on the smoothing parameter.

Please note that the use of history downloading file, data exchange using a disk and waiting for data are all connected with the need to control the incoming data in both direction. Data in the NNM window, on the chart and in files Open1,2,3 must be identical.

![Information control](https://c.mql5.com/2/37/EURUSD_Alpari_2019-11-11_07.39.00.png)

13\. Information control

The figure shows a variant of the Neural Network Module which passes to the EA only response from net1, net2 and net3. Position opening conditions are specified in the Expert Advisor. This is my preferred way. In our case, the NNM provides a ready signal, and the EA trades only subject to time limitation.

Such control is especially useful when debugging a system. Also, it is necessary to visually control the performance during tests and trading.

![NNM variant](https://c.mql5.com/2/38/Strategy_Tester_Visualization.png)

14\. NNM variant

The above figure shows another variant of a neural network module. In this variant, neural networks should better be integrated directly into the executable file. When clicking the green button, simply select the Input\_mat.txt file. I would recommend using a similar model for work. This variant only has a trading block, without training and testing blocks. Don't forget that the apparent complexity of the system preparation stage provides maximum simplicity in trading. The main market analysis is performed in NNM - this step is instant due to the use of neural networks. If you do not use other optimization conditions, the trading robot will only need to interpret two received digits.

```
 if(send1==false && Order1==1)
     {
      send1=trade.Buy(1);
      TicketBuy1 = trade.ResultDeal();
     }

 if(send1==true && Order1==0)
     {
      trade.PositionClose(TicketBuy1);
      send1=false;
     }
```

In conclusion, I would like to dwell on some differences in launching the ExpertMatlabReal\_MT4.ex4 for working in the MetaTrader 4 terminal. This is connected with the specific features of Strategy Testers, namely with the way they determine control points in testing. In MetaTrader 5, testing ends at the last bar of the previous day. In MetaTrader 4 the current bar is used. Therefore I introduced the "Hours" external variable in the EA for MetaTrader 4.

![External variable "Hours"](https://c.mql5.com/2/38/Hours.png)

14.1 External variable "Hours"

Using ExpertMatlabPodkach\_MT4.ex4, we generate a history downloading file with a current bar row, that is why the current time bar should be specified in the "Hours" variable during the first launch. After YA start, open its properties and set the variable back to 0. This ensures that further data shifts are performed at 00:00.

![Hours-0](https://c.mql5.com/2/38/Hours_0.png)

14.2 Hours -0

2\. **NNM testing in visual mode**

2\. Тестирование НСМ в режиме визуализации - YouTube

[Photo image of НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4](https://www.youtube.com/channel/UCScAAn_sRRaKHdNIxl0aI9A?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7370)

НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4

417 subscribers

[2\. Тестирование НСМ в режиме визуализации](https://www.youtube.com/watch?v=TTh4iXgKzdw)

НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

More videos

## More videos

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=TTh4iXgKzdw&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7370)

0:00

0:00 / 5:28

•Live

•

This test completes the preparation. Thus, the Neural Network Module will give us the desired result, according to **the preparation, neural network training, the received array of responses, visualization of responses and further optimization**. Now, we repeat actions from point 1, i.e. launch NNM as if for trading. Also, launch ExpertMatlabReal\_MT5.ex5 in the strategy tester, using visualization. Enable the "1 minute OHLC" mode in MetaTrader 5. In MetaTrader 4, enable "Control points". These models should be used in order to obtain reliable testing results, because deals will be executed on the tick following bar opening, when using the NNM. If we use "Open Price only" model, position opening will be delayed by one bar during testing. In this mode, the response delay can be easily considered using the EURUSD\_MT5var.exe module. Of course, this will not happen in real trading. Generally, this system shows almost identical results for all types of testing. This is another confirmation of its viability.

![Visualization](https://c.mql5.com/2/38/Strategy_Tester_Visualization___ExpertMatlabReal_M.png)

![Visualization](https://c.mql5.com/2/38/05z15jzneu9i.png)

15\. Visual Mode

3\. **Neural Network training**

3\. Обучение нейросетей - YouTube

[Photo image of НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4](https://www.youtube.com/channel/UCScAAn_sRRaKHdNIxl0aI9A?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7370)

НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4

417 subscribers

[3\. Обучение нейросетей](https://www.youtube.com/watch?v=TjVGfhhqLrQ)

НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

More videos

## More videos

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=TjVGfhhqLrQ&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7370)

0:00

0:00 / 4:04

•Live

•

This Neural Network Module mode, in the provided version, is used to demonstrate the neural network training process as well as its preparation for real work. It is a demonstration model, because these neural networks cannot be retrained, because the module uses already trained neural networks. There is no need to add such a submodule into your working program, because training is performed in the offline mode and the most efficient and reliable way is to train the neural network directly in MATLAB environment.

To test two neural network for the Net1 submodule, we need to have neural networks net1-2 and the EURUSDData.csv file with training data in the ...\\Common\\Files directory.

![Data](https://c.mql5.com/2/38/Files_2020-01-29_13.14.25.png)

16\. Data file

Training data is real - this is the data used to prepare the trading system using the Online block. I would like to note once again the advantage of using neural networks to assess the market situation. The EURUSDData.csv table has 90 columns which represent a set of sequential values which are used as the input data for the neural network. Simply put these are the inputs. Imagine that each column is a separate indicator which is preliminary calculated by the Expert Advisor and is then extracted as the data to train the neural network. All this is done in the offline mode, when preparing the system. Now imagine that we would try to analyze this impressive data set directly in a working Expert Advisor during trading.

When writing this article, I renamed button names for a better understanding of the order of actions.

![Updated button names](https://c.mql5.com/2/38/EURUSD_MT5_2020-01-29_15.10.52.png)

17\. Updated button names

We press "Train net1", open the net1.mat file and train two networks, net1 and net2, for the Net1 block. And so on: net3,4 for Net2 and net5,6 for Net3.

![Train net1](https://c.mql5.com/2/38/Select_File_to_Open_2020-01-29_15.14.37.png)

18\. Train net1

Again, the most convenient way is to place the program shortcut in the folder, where working files are located, and to change path to "Working folder" in its properties. Next, the NNM will train the networks.

4\. **Receiving responses from neural network blocks to optimize a trading strategy**

4\. Получение откликов от нейросетевых блоков для оптимизации торговой стратегии - YouTube

[Photo image of НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4](https://www.youtube.com/channel/UCScAAn_sRRaKHdNIxl0aI9A?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7370)

НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4

417 subscribers

[4\. Получение откликов от нейросетевых блоков для оптимизации торговой стратегии](https://www.youtube.com/watch?v=Vrgs7P-u7_o)

НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

More videos

## More videos

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=Vrgs7P-u7_o&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7370)

0:00

0:00 / 5:29

•Live

•

In the neural network module, I added the ability to test neural networks in the offline mode. **In other words, this is the possibility to receive neural network responses based on historical data, in an effort to create a signal indicator which would allow to optimize the system.** Here is an example of responses received from Net1.

Firstly, prepare the EURUSDTestPodkach\_MT5.csv file in the ...\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files folder. To do this, launch the ExpertMatlabPodkach\_MT5 EA in the Strategy Tester. This should start with the date preceding the testing start date (please make sure that the required history amount is downloaded).

For example:

![Podkach](https://c.mql5.com/2/38/23476973_-_MetaQuotes__1.png)

19\. Launch ExpertMatlabPodkach\_MT5.ex5

Delete all lines except one in the resulting file.

![EURUSDTestPodkach](https://c.mql5.com/2/37/Microsoft_Excel_-_EURUSDTestPodkach_2019-11-12_13..png)

20\. Leave only one line

To generate a file with the testing period data, launch ExpertMatlab\_MT5.ex5 in the Strategy Tester.

![ExpertMatlab](https://c.mql5.com/2/38/23476973_-_MetaQuotes_1__1.png)

21\. Launch ExpertMatlab\_MT5.ex5

Pay attention to how the date for the period beginning is selected.

![EURUSDTest, EURUSDDate](https://c.mql5.com/2/38/Files_2020-02-06_13.21.48.png)

22\. We obtain EURUSDTest.csv and EURUSDDate.csv

ExpertMatlab\_MT5.ex5 will generate two test files EURUSDDate.csv and EURUSDTest.csv. Launch the neural network module, click "Test net1" and select net1.mat.

![Test net1](https://c.mql5.com/2/38/Select_File_to_Open_2020-02-06_13.29.41.png)

23\. Press "Test net1"

Wait for some time until the Indicator1.csv response file is generated.

![Indicator1](https://c.mql5.com/2/38/EURUSD_MT5_2020-02-06_19.07.08.png)

24\. The Indicator1.csv file with responses is generated

Save Indicator1 as Indicator. **If the neural network block has saved "Indicator1.csv" in the "Compatibility mode", you should save it in the csv format via the "Save As" tab.**

![Indicator](https://c.mql5.com/2/38/Microsoft_Excel_2020-02-06_19.34.21.png)

25\. Save Indicator1.csv as Indicator.csv

Let's check the NNM responses in the visual mode, on the EURUSD H1 chart. To do this, use the NWI.ex5 indicator.

![NWI](https://c.mql5.com/2/38/NWI_2020-02-10_12.28.27.png)

26\. Launch the NWI.ex5 indicator

The default period value is 5. According to my experiments, this period value for the signal line provides the best result for EURUSD H1 However, you can try to find your own value.

![NWI](https://c.mql5.com/2/38/NWI.png)

27\. NWI visualization

Using Matlab\_MT5.ex5, we can test and further optimize the received responses.

```
#include<Trade\Trade.mqh>

CTrade  trade;

input int Period=5;
input int H1;
input int H2;
input int H3;
input int H4;

input int LossBuy;
input int ProfitBuy;
input int LossSell;
input int ProfitSell;

ulong TicketBuy1;
ulong TicketSell0;

datetime Count;

double Per;
double Buf_0[];
double Buf_1[];

bool send1;
bool send0;

int h=4;
int k;
int K;
int bars;
int Handle;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

   Handle=FileOpen("Indicator.csv",FILE_CSV|FILE_SHARE_READ|FILE_ANSI|FILE_COMMON,";");

   while(!FileIsEnding(Handle)&& !IsStopped())
     {
      StringToTime(FileReadString(Handle));
      bars++;
     }
   FileClose(Handle);

   ArrayResize(Buf_0,bars);
   ArrayResize(Buf_1,bars);

   Handle=FileOpen("Indicator.csv",FILE_CSV|FILE_SHARE_READ|FILE_ANSI|FILE_COMMON,";");

   while(!FileIsEnding(Handle)&& !IsStopped())
     {
      Count=StringToTime(FileReadString(Handle));
      Buf_0[k]=StringToDouble(FileReadString(Handle));
      h=Период-1;
      if(k>=h)
        {
         while(h>=0)
           {
            Buf_1[k]=Buf_1[k]+Buf_0[k-h];
            h--;
           }
         Buf_1[k]=Buf_1[k]/Period;
        }
      k++;
     }
   FileClose(Handle);

   int deviation=10;
   trade.SetDeviationInPoints(deviation);
   trade.SetTypeFilling(ORDER_FILLING_RETURN);
   trade.SetAsyncMode(true);

//---
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   MqlDateTime stm;
   TimeToStruct(TimeCurrent(),stm);

   int    digits=(int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
   double point=SymbolInfoDouble(_Symbol,SYMBOL_POINT);
   double PriceAsk=SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   double PriceBid=SymbolInfoDouble(_Symbol,SYMBOL_BID);

   double SL1=NormalizeDouble(PriceBid-LossBuy*point,digits);
   double TP1=NormalizeDouble(PriceAsk+ProfitBuy*point,digits);
   double SL0=NormalizeDouble(PriceAsk+LossSell*point,digits);
   double TP0=NormalizeDouble(PriceBid-ProfitSell*point,digits);

   if(LossBuy==0)
      SL1=0;

   if(ProfitBuy==0)
      TP1=0;

   if(LossSell==0)
      SL0=0;

   if(ProfitSell==0)
      TP0=0;

//---------Buy1
   if(send1==false && K>0 && Buf_0[K-1]<Buf_1[K-1] && Buf_0[K]>Buf_1[K] && iLow(NULL,PERIOD_H1,1)<iLow(NULL,PERIOD_H1,2) && stm.hour>H1 && stm.hour<H2 && H1<H2)
     {
      send1=trade.PositionOpen(_Symbol,ORDER_TYPE_BUY,1,PriceAsk,SL1,TP1);
      TicketBuy1 = trade.ResultDeal();
     }

   if(send1==true && K>0 && Buf_0[K-1]>Buf_1[K-1] && Buf_0[K]<Buf_1[K] && iHigh(NULL,PERIOD_H1,1)>iHigh(NULL,PERIOD_H1,2))
     {
      trade.PositionClose(TicketBuy1);
      send1=false;
     }

//---------Sell0

   if(send0==false && K>0 && Buf_0[K-1]>Buf_1[K-1] && Buf_0[K]<Buf_1[K] && iHigh(NULL,PERIOD_H1,1)>iHigh(NULL,PERIOD_H1,2) && stm.hour>H3 && stm.hour<H4 && H3<H4)
     {
      send0=trade.PositionOpen(_Symbol,ORDER_TYPE_SELL,1,PriceBid,SL0,TP0);
      TicketSell0 = trade.ResultDeal();
     }

   if(send0==true && K>0 && Buf_0[K-1]<Buf_1[K-1] && Buf_0[K]>Buf_1[K] && iLow(NULL,PERIOD_H1,1)<iLow(NULL,PERIOD_H1,2))
     {
      trade.PositionClose(TicketSell0);
      send0=false;
     }
   K++;
  }

//+------------------------------------------------------------------+
//| Tester function                                                  |
//+------------------------------------------------------------------+
double OnTester()
  {
//---
   double ret=0.0;
//---

//---
   return(ret);
  }
//+------------------------------------------------------------------+
```

![Matlab_MT5](https://c.mql5.com/2/38/Matlab_MT5.png)

![Matlab_MT5](https://c.mql5.com/2/38/Matlab_MT5_2.png)

28\. Launch Matlab\_MT5.ex5

![Balance](https://c.mql5.com/2/38/hapqnk.png)

29\. Balance

There is no point in conducting a forward test. After a backtest, a forward test starts with the next date, however the file is loaded with the starting date, with which the backtest started. Thus, the results will not be correct.

Let's carry out optimization. The previous test was conducted without restrictive levels and according to the system preparation principle described in the book. More levels have been introduced for optimization.

![Optim](https://c.mql5.com/2/38/Optim1.png)

![Optim](https://c.mql5.com/2/38/Optim2.png)

![Optim](https://c.mql5.com/2/38/Optim.png)

30\. Conduct optimization

Optimization period is 2011-2013, before the red line. After that, the period of 2013-2016 is used for testing. Of course, in practice no one would trade for such a ling period after optimization, as neural networks must be retrained from time to time. From my practice, testing should be performed within a month. On the chart, the period between the red line and the blue line is 4 months.

Let's test this period separately.

![Test](https://c.mql5.com/2/38/Test1.png)

![Test](https://c.mql5.com/2/38/Test.png)

![Reference test](https://c.mql5.com/2/38/Test2__1.png)![Reference test](https://c.mql5.com/2/38/Test3__1.png)

![Reference test](https://c.mql5.com/2/38/Test3_3.png)

31\. Reference test

I had not conducted these tests previously, so all this was done in the process of writing this section of the article. Other options can of course be used. And it is also necessary to prepare hedging neural networks. I just want to emphasize how much neural networks simplify the preparation of the system and the trading process.

We can say that the trades emulated during this test had ideal conditions. Therefore, we will take the result of this test as a reference.

5\. **Correction of errors**

While working with this part, I decided to get back to the beginning and explain why this part is so lengthy. The fact is that "Correction of errors" was performed in real time, while writing this article section. This also concerns the previous section. Everything was prepared for MetaTrader 4, but for MetaTrader 5 I had to work in real-time mode. This was a very useful experience for me. Anyway, please note that the preparation step is extremely important and you should therefore be very careful.

It is necessary to get back to point 2 and to test the result (which we called a reference) in the visualization mode.

Here is what we have.

![Report](https://c.mql5.com/2/38/Microsoft_Excel_-_ReportTester-23476973_2020-02-13.png)

32\. Failed test

Although the result is positive, it is completely different from the previous one. This means that there is an error in the program code which we need to find. The useful feature is that we can trace all data transfer steps.

Let's compare the responses in the output fields (4) of the Net1 sub-module and the NNM responses received using the ExpertMtatlab\_MT5.ex5 EA. Use the same bar.

![Compare responses](https://c.mql5.com/2/38/EURUSD_MT5_2020-02-20_16.36.22.png)

33\. Compare responses

As you can see, the responses of the Net1 submodule and the NNM responses do not match. Since we use the same block of neural networks, we can conclude that the data input during these two tests is different.

Compare the data in files EURUSDTest.csv and Input\_mat.txt by selecting any bar.

![Input_mat.txt](https://c.mql5.com/2/38/Input_mat_2_89g8nf8_2020-02-20_16.42.46.png)

34\. Compare if the data match

We were right, and the data is really different. Thus, let's check the program code of ExpertMtatlab\_MT5.ex5 and ExpertMatlabReal\_MT5.ex5.

```
ExpertMtatlab_MT5.ex5
if(stm.hour==0)
     {
      for(int i=19; i>=10; i--)
        {
         in[i-10]=in[i];
        }

      for(int i=29; i>=20; i--)
        {
         in[i-10]=in[i];
        }
     }
ExpertMatlabReal_MT5.ex5
if(Bar<Bars(NULL,PERIOD_H1) && stm.hour==0)
     {
      for(int i=19; i>=10; i--)
        {
         in[i-10]=in[i];
        }

      for(int i=29; i>=20; i--)
        {
         in[i-10]=in[i];
        }
     }
```

An error has been found in the ExpertMatlabReal\_MT5.ex5 code. The file did not include the main condition for the data shift by 00 hours. In real work, our NNM would receive from the terminal the information that does not correspond to the one that the networks should receive. This would produce "false responses". I put the quotes because, in fact, the neural network generates a normal response, but it will be false for us, since we provided wrong information to the network.

Again, let's compare the responses in the output fields (4) of the Net1 sub-module and the NNM responses received using the ExpertMtatlab\_MT5.ex5 EA.

![Compare responses](https://c.mql5.com/2/38/EURUSD_MT5_2020-02-20_17.05.24.png)

35\. Compare responses

Now we see that the information input into the NNM is correct.

Test in the visual mode once again, after fixing the error.

![Test](https://c.mql5.com/2/38/Test4.png)

36\. The test differs from the reference

Even though the result is somewhat closer, it is still significantly different. Moreover, the neural network module terminated on its own.

After checking, an error was found in the sell position closing condition, in ExpertMatlabReal\_MT5.mq5. Test once again in 1 minute OHLC, visualization mode.

![No tick waiting](https://c.mql5.com/2/38/NoCount.png)

![No tick waiting](https://c.mql5.com/2/38/NoCount_1.png)![No tick waiting](https://c.mql5.com/2/38/NoCount_2.png)

![No tick waiting](https://c.mql5.com/2/38/NoCount_3.png)

37\. Test with no tick waiting

The result does not match the reference value.

![Reference test](https://c.mql5.com/2/38/Test3_33.png)

38\. Reference

After analysis, we can conclude that this is the result of emulated trading under unfavorable conditions. This has lead to a larger number of false signals and unnecessary trades. Such a situation occurs when a position open signal appeared on the previous bar, and a signal to open an opposite trade or to stay out of the market appeared on the next bar. Since testing is performed in an accelerated mode, information from the NNM can be late and thus the EA can perform a deal based on the previous signal. But in this case, such testing is useful, since it gives us the opportunity to see how the system would behave under unfavorable market conditions.

Let's see the result after we add to the EA position opening restrictions for the situations described above.

```
 if(Order1==0 && Bar<Bars(NULL,PERIOD_H1) && Con==0)
      send0=true;

 if(Order1==1 && Bar<Bars(NULL,PERIOD_H1) && Con==0)
      send1=true;

//---------Buy
```

The test will be performed using ticks and at the highest visualization speed. You can watch the process in the video.

![Test with conditions](https://c.mql5.com/2/38/23476973_-_MetaQuotes-Demo_1.png)

![Test with conditions](https://c.mql5.com/2/38/23476973_-_MetaQuotes-Demo_2.png)![Test with conditions](https://c.mql5.com/2/38/23476973_-_MetaQuotes-Demo_3.png)

![Test with conditions](https://c.mql5.com/2/38/Microsoft_Excel_-_ReportTester-23476973_4_2020-02-.png)

39\. The result is close to the reference

As you can see, the result is close to the reference. Although, I think that the previous test would also be close to the reference if we reduced the testing speed. You can try to do this yourself.

Тест по всем тикам \- YouTube

[Photo image of НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4](https://www.youtube.com/channel/UCScAAn_sRRaKHdNIxl0aI9A?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7370)

НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4

417 subscribers

[Тест по всем тикам](https://www.youtube.com/watch?v=8oSPb4hilfA)

НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

More videos

## More videos

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=8oSPb4hilfA&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7370)

0:00

0:00 / 3:40

•Live

•

Now, let us find out why the NNM terminated. I could skip this part. But it is good that we found this bug in the testing mode. It would be much worse if the problem were detected during real trading. If neural networks in the NNM are untrained or if we launch the module before the trading EA launch, the Input\_mat.txt file will not have been generated by that time, and the NNM will not provide responses after we click "Start". Initially, I added the warning window and a forced exit from the internal timer. I removed this logical error but left the warning. However, if the window pops up in the process of work, you should simply click OK to hide the window. The program will continue operation.

![Error window](https://c.mql5.com/2/38/frmRecordRegion_2020-02-18_14.48.12.png)

40\. Error window

There is another problem that you should pay attention to. In MetaTrader 5, if we write data using the Strategy Data, the number of generated bars is greater than is displayed on the chart. Because of this, the resulting indicator is somewhat "longer". However, this does not affect the quality of testing. The extra bar is taken into account in a testing pass but it is not displayed on the chart. This can be checked in a step-by-step pass.

![MetaTrader 5 indicator](https://c.mql5.com/2/38/23476973_-_Indicator_MT5.png)

41\. The NWI indicator in MetaTrader 5

![MetaTrader 5 indicator](https://c.mql5.com/2/38/Microsoft_Excel_-_Indicator_2020-03-10_15.49.27.png)

42\. There is a time shift

This problem does not exist in MetaTrader 4.

![MetaTrader 4 indicator](https://c.mql5.com/2/38/60377261__Indicator_MT4.png)

43\. Indicator 1\_MT4 in MetaTrader 4

![MetaTrader 4 indicator](https://c.mql5.com/2/38/Microsoft_Excel_-_Indicator_2020-03-10_15.44.29.png)

44\. No shift

The below figure shows why this happens.

![23:59](https://c.mql5.com/2/38/Microsoft_Excel_-_Indicator_2020-03-10_15.51.02.png)

45\. Finding the reason for the shift

### **Conclusion**

What is the purpose of using neural networks?

1. Neural networks allow the processing of a significant amount of information outside the market.
2. Because the information is processed in the offline mode, we can go ahead of the market rather than running after it. Or at least we can keep up with the market.
3. As a result, neural networks enable timely response to changes in the market situation, not only during trading, but even at the preparation stage. Thus, we can eliminate large losses in the case of a trend change or when other factors affect the market.
4. As for the specific neural network presented in the article, and its training method, the undoubted advantage of this combination is that we can optimize the EA using the responses received in the period, in which the neural network was trained. Neural networks based on other architectures can show good results in the training mode, while having significantly different results in a test period. In our cases the results are approximately the same. This enables us to test the NN near the current date and to take into account the market factors which are available in history. This part is not covered in the article, but you can explore it yourself.

When writing this article, I initially wanted to integrate into the NNM only the Net1 neural networks, which have been trained by the article publication time. However, upon completion, I decided to integrate neural networks trained till 31.12.2019, for Net2. As well as data till 31.12.2010 for Net3. It is interesting to test them in the current market conditions. For all other works within the NNM, only neural networks for Net1 will be used.

### After you download the files

I apologize for the lengthy video, the process was filmed nearly in one frame.

Действия после загрузки файлов \- YouTube

[Photo image of НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4](https://www.youtube.com/channel/UCScAAn_sRRaKHdNIxl0aI9A?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7370)

НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4

417 subscribers

[Действия после загрузки файлов](https://www.youtube.com/watch?v=DJOL7a5OO34)

НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

More videos

## More videos

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=DJOL7a5OO34&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7370)

0:00

0:00 / 4:51

•Live

•

### Python

Первые шаги реализации нейросетевого модуля автоматической торговли на Python - YouTube

[Photo image of НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4](https://www.youtube.com/channel/UCScAAn_sRRaKHdNIxl0aI9A?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7370)

НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4

417 subscribers

[Первые шаги реализации нейросетевого модуля автоматической торговли на Python](https://www.youtube.com/watch?v=tvVkfUa1hYw)

НЕЙРОННЫЕ СЕТИ MATLAB и MetaTrader 4

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=tvVkfUa1hYw&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7370)

0:00

0:00 / 3:39

•Live

•

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7370](https://www.mql5.com/ru/articles/7370)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7370.zip "Download all attachments in the single ZIP archive")

[EURUSDData.csv](https://www.mql5.com/en/articles/download/7370/eurusddata.csv "Download EURUSDData.csv")(8009.95 KB)

[ExpertMatlab\_MT5.mq5](https://www.mql5.com/en/articles/download/7370/expertmatlab_mt5.mq5 "Download ExpertMatlab_MT5.mq5")(11.93 KB)

[ExpertMatlabPodkach\_MT5.mq5](https://www.mql5.com/en/articles/download/7370/expertmatlabpodkach_mt5.mq5 "Download ExpertMatlabPodkach_MT5.mq5")(7.53 KB)

[ExpertMatlabReal\_MT5.mq5](https://www.mql5.com/en/articles/download/7370/expertmatlabreal_mt5.mq5 "Download ExpertMatlabReal_MT5.mq5")(16.18 KB)

[Matlab\_MT5.mq5](https://www.mql5.com/en/articles/download/7370/matlab_mt5.mq5 "Download Matlab_MT5.mq5")(9.36 KB)

[NWI.mq5](https://www.mql5.com/en/articles/download/7370/nwi.mq5 "Download NWI.mq5")(5.02 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Practical application of neural networks in trading (Part 2). Computer vision](https://www.mql5.com/en/articles/8668)
- [Practical application of neural networks in trading. Python (Part I)](https://www.mql5.com/en/articles/8502)
- [Practical application of neural networks in trading](https://www.mql5.com/en/articles/7031)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/352715)**
(14)


![Enrique Enguix](https://c.mql5.com/avatar/2025/9/68c108f2-b619.jpg)

**[Enrique Enguix](https://www.mql5.com/en/users/envex)**
\|
1 Oct 2020 at 13:25

**MetaQuotes:**

Published article [Practical application of neural networks in trading. Let's move on to practice](https://www.mql5.com/en/articles/7370):

Author: [Andrey Dibrov](https://www.mql5.com/en/users/tomcat66 "tomcat66")

Very good article


![TipMyPip](https://c.mql5.com/avatar/avatar_na2.png)

**[TipMyPip](https://www.mql5.com/en/users/pcwalker)**
\|
6 Oct 2020 at 01:53

This is outstanding ! You are for real, I salute. <Deleted>


![Chun Feng Yin](https://c.mql5.com/avatar/avatar_na2.png)

**[Chun Feng Yin](https://www.mql5.com/en/users/matiji66)**
\|
28 Dec 2020 at 12:29

Great job,I would like to try it.Btw,the english pannel will be easier to understand. After all,I don't know your mother tongue.


![FM2020](https://c.mql5.com/avatar/avatar_na2.png)

**[FM2020](https://www.mql5.com/en/users/fmarmy)**
\|
26 Feb 2021 at 22:36

This is a great contribution and more sophisticated approach to trading FX. I will spend some time to study your approach and try to replicate. Thank you!


![Jason Kisogloo](https://c.mql5.com/avatar/2021/12/61C2A348-5044.png)

**[Jason Kisogloo](https://www.mql5.com/en/users/lordjason)**
\|
16 Nov 2021 at 01:01

### Temporal Convolutional neural network TCN's versus RNN

Latest [Neural network](https://www.mql5.com/en/articles/497 "Article: Neural networks - from theory to practice ") design TCN neural nets explained in attached PDF...

(TCN's Faster and more effective... )

![Quick Manual Trading Toolkit: Basic Functionality](https://c.mql5.com/2/39/Frame_1.png)[Quick Manual Trading Toolkit: Basic Functionality](https://www.mql5.com/en/articles/7892)

Today, many traders switch to automated trading systems which can require additional setup or can be fully automated and ready to use. However, there is a considerable part of traders who prefer trading manually, in the old fashioned way. In this article, we will create toolkit for quick manual trading, using hotkeys, and for performing typical trading actions in one click.

![Practical application of neural networks in trading](https://c.mql5.com/2/37/neural_DLL.png)[Practical application of neural networks in trading](https://www.mql5.com/en/articles/7031)

In this article, we will consider the main aspects of integration of neural networks and the trading terminal, with the purpose of creating a fully featured trading robot.

![Calculating mathematical expressions (Part 1). Recursive descent parsers](https://c.mql5.com/2/39/MQL5-avatar-analysis.png)[Calculating mathematical expressions (Part 1). Recursive descent parsers](https://www.mql5.com/en/articles/8027)

The article considers the basic principles of mathematical expression parsing and calculation. We will implement recursive descent parsers operating in the interpreter and fast calculation modes, based on a pre-built syntax tree.

![Manual charting and trading toolkit (Part I). Preparation: structure description and helper class](https://c.mql5.com/2/39/MQL5-set_of_tools.png)[Manual charting and trading toolkit (Part I). Preparation: structure description and helper class](https://www.mql5.com/en/articles/7468)

This is the first article in a series, in which I am going to describe a toolkit which enables manual application of chart graphics by utilizing keyboard shortcuts. It is very convenient: you press one key and a trendline appears, you press another key — this will create a Fibonacci fan with the necessary parameters. It will also be possible to switch timeframes, to rearrange layers or to delete all objects from the chart.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=iyeycvkcsfrxyhwmvbyctmxbajqqxsnp&ssn=1769186141543657164&ssn_dr=0&ssn_sr=0&fv_date=1769186141&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7370&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Practical%20application%20of%20neural%20networks%20in%20trading.%20It%27s%20time%20to%20practice%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918614130745034&fz_uniq=5070411070272640348&sv=2552)

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