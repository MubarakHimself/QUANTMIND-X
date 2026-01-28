---
title: Practical application of neural networks in trading. Python (Part I)
url: https://www.mql5.com/en/articles/8502
categories: Trading, Trading Systems, Integration
relevance_score: 3
scraped_at: 2026-01-23T18:16:10.161732
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/8502&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069295714510504631)

MetaTrader 5 / Examples


### Introduction

In the previous article entitled " [Practical application of neural networks in trading. It's time to practice](https://www.mql5.com/en/articles/7370)", we considered the practical application of a neural network module implemented using Matlab neural networks. However, that article did not cover questions related to the preparation of input data and network training related operations. In this article, we will consider these questions using examples and will implement further code using neural networks of libraries working with Python. This time, the trading system implementation principles will be different. This variant was briefly described in paragraph 5 of the basic article " [Practical application of neural networks in trading](https://www.mql5.com/en/articles/7031)". For this implementation, we will use the TensorFlow machine learning library developed by Google. We will also use the Keras library for describing neural networks.

### 1\. Data preparation

Let us consider some points related to data preparation for neural network training.

- For decision making, we will use two neural networks for opening positions in one direction
- According to the previous point, the training data should be divided into two groups - one for each direction.
- As in the previous system, the first neural network will be trained to build indicators similar to standard technical indicators. We used this solution in the previous system because we used self-written indicators and we didn't want to overload the working Expert Advisor. Python is used because only quotes can be received from the terminal. To prepare data for the neural network, we need to build these indicators in a Python script. By teaching the neural network to build such indicators, we eliminate the need to duplicate them in the script.
- The second neural network builds the signal indicator, based on which we create a trading strategy.
- The neural network will be trained for the EURUSD H1 chart.
- As a result, to build the system, we will need to prepare two neural networks for buying and two networks for selling. Thus, four neural networks will work in the system.

Two scripts will be used for preparing data for training the networks: PythonPrices.mq5 and PythonIndicators.mq5.

```
//+------------------------------------------------------------------+
//|                                                 PythonPrices.mq5 |
//|                                   Copyright 2020, Andrey Dibrov. |
//|                           https://www.mql5.com/ru/users/tomcat66 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, Andrey Dibrov."
#property link      "https://www.mql5.com/ru/users/tomcat66"
#property version   "1.00"
#property strict
#property script_show_inputs

input string Date="2004.07.01 00:00";
input string DateOut="2010.12.31 23:00";
input int History=0;

double inB[22];
string Date1;

int HandleInpuNet1Min;
int HandleInpuNet1Max;

double DibMin1_1[];
double DibMax1_1 [];
int DibMin1_1Handle;
int DibMax1_1Handle;

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   int k=iBars(NULL,PERIOD_H1)-1;

   DibMin1_1Handle=iCustom(NULL,PERIOD_H1,"DibMin1-1",History);
   CopyBuffer(DibMin1_1Handle,0,0,k,DibMin1_1);
   ArraySetAsSeries(DibMin1_1,true);

   DibMax1_1Handle=iCustom(NULL,PERIOD_H1,"DibMax1-1",History);
   CopyBuffer(DibMax1_1Handle,0,0,k,DibMax1_1);
   ArraySetAsSeries(DibMax1_1,true);

   HandleInpuNet1Min=FileOpen(Symbol()+"InputNet1Min.csv",FILE_CSV|FILE_WRITE|FILE_SHARE_READ|FILE_ANSI|FILE_COMMON,";");
   HandleInpuNet1Max=FileOpen(Symbol()+"InputNet1Max.csv",FILE_CSV|FILE_WRITE|FILE_SHARE_READ|FILE_ANSI|FILE_COMMON,";");
   FileSeek(HandleInpuNet1Min,0,SEEK_END);
   FileSeek(HandleInpuNet1Max,0,SEEK_END);

   if(HandleInpuNet1Min>0)
     {
      Alert("Writing to the file InputNet1Min");

      for(int i=iBars(NULL,PERIOD_H1)-1; i>=0; i--)
        {

         Date1=TimeToString(iTime(NULL,PERIOD_H1,i));

         if(DateOut>=Date1 && Date<=Date1)
           {
            if((DibMin1_1[i]==-1 && DibMin1_1[i+1]==1 && DibMax1_1[i]==1) || (DibMin1_1[i]==1 && DibMax1_1[i]==1))
              {
               for(int m=0; m<=14; m++)
                 {
                  inB[m]=inB[m+5];
                 }

               inB[15]=(iOpen(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i)))-iLow(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i))))*100000;
               inB[16]=(iHigh(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i)))-iOpen(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i))))*100000;
               inB[17]=(iHigh(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i)))-iLow(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i))))*10000;
               inB[18]=(iHigh(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i)))-iOpen(NULL,PERIOD_H1,i+1))*10000;
               inB[19]=(iOpen(NULL,PERIOD_H1,i+1)-iLow(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i))))*10000;

               inB[20]=(iHigh(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i)))-iOpen(NULL,PERIOD_H1,i))*10000;
               inB[21]=(iOpen(NULL,PERIOD_H1,i)-iLow(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i))))*10000;

               FileWrite(HandleInpuNet1Min,

                         inB[0],inB[1],inB[2],inB[3],inB[4],inB[5],inB[6],inB[7],inB[8],inB[9],inB[10],inB[11],inB[12],inB[13],inB[14],inB[15],
                         inB[16],inB[17],inB[18],inB[19],inB[20],inB[21]);
              }
           }
        }

      FileClose(HandleInpuNet1Min);
     }
//------------------------------------------------------------------------------------------------------------------------------------------------

   if(HandleInpuNet1Max>0)
     {
      Alert("Writing the file InputNet1Max");

      for(int i=iBars(NULL,PERIOD_H1)-1; i>=0; i--)
        {

         Date1=TimeToString(iTime(NULL,PERIOD_H1,i));

         if(DateOut>=Date1 && Date<=Date1)
           {
            if((DibMax1_1[i]==-1 && DibMax1_1[i+1]==1 && DibMin1_1[i]==1)|| (DibMin1_1[i]==1 && DibMax1_1[i]==1))
              {
               for(int m=0; m<=14; m++)
                 {
                  inB[m]=inB[m+5];
                 }

               inB[15]=(iOpen(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i)))-iLow(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i))))*100000;
               inB[16]=(iHigh(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i)))-iOpen(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i))))*100000;
               inB[17]=(iHigh(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i)))-iLow(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i))))*10000;
               inB[18]=(iHigh(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i)))-iOpen(NULL,PERIOD_H1,i+1))*10000;
               inB[19]=(iOpen(NULL,PERIOD_H1,i+1)-iLow(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i))))*10000;

               inB[20]=(iHigh(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i)))-iOpen(NULL,PERIOD_H1,i))*10000;
               inB[21]=(iOpen(NULL,PERIOD_H1,i)-iLow(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i))))*10000;

               FileWrite(HandleInpuNet1Max,

                         inB[0],inB[1],inB[2],inB[3],inB[4],inB[5],inB[6],inB[7],inB[8],inB[9],inB[10],inB[11],inB[12],inB[13],inB[14],inB[15],
                         inB[16],inB[17],inB[18],inB[19],inB[20],inB[21]);
              }
           }
        }

      FileClose(HandleInpuNet1Max);
     }
   Alert("Files written");

  }
//+------------------------------------------------------------------+
```

```
//+------------------------------------------------------------------+
//|                                             PythonIndicators.mq5 |
//|                                   Copyright 2020, Andrey Dibrov. |
//|                           https://www.mql5.com/ru/users/tomcat66 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, Andrey Dibrov."
#property link      "https://www.mql5.com/ru/users/tomcat66"
#property version   "1.00"
#property strict
#property script_show_inputs

input string Date="2004.07.01 00:00";
input string DateOut="2010.12.31 23:00";
input int History=0;

double Stochastic0[];
double Stochastic1[];
double CCI_Open[];
double CCI_Low[];
double CCI_High[];
double Momentum_Open[];
double Momentum_Low[];
double Momentum_High[];
double RSI_Open[];
double RSI_Low[];
double RSI_High[];
double WPR[];
double MACD_Open[];
double MACD_Low[];
double MACD_High[];
double OsMA_Open[];
double OsMA_Low[];
double OsMA_High[];
double TriX_Open[];
double TriX_Low[];
double TriX_High[];
double BearsPower[];
double BullsPower[];
double ADX_MINUSDI[];
double ADX_PLUSDI[];
double StdDev_Open[];
double StdDev_Low[];
double StdDev_High[];

//--------------------------
double DibMin1_1[];
double DibMax1_1 [];

int DibMin1_1Handle;
int DibMax1_1Handle;
//--------------------------
double inB[60];
double inS[60];

string Date1;

int HandleInputNet2OutNet1Min;
int HandleOutNet2Min;
int HandleInputNet2OutNet1Max;
int HandleOutNet2Max;

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   int k=iBars(NULL,PERIOD_H1)-1;

//------ Daily Low

   DibMin1_1Handle=iCustom(NULL,PERIOD_H1,"DibMin1-1",History);
   CopyBuffer(DibMin1_1Handle,0,0,k,DibMin1_1);
   ArraySetAsSeries(DibMin1_1,true);

   DibMax1_1Handle=iCustom(NULL,PERIOD_H1,"DibMax1-1",History);
   CopyBuffer(DibMax1_1Handle,0,0,k,DibMax1_1);
   ArraySetAsSeries(DibMax1_1,true);

   int Stochastic_handle=iStochastic(NULL,PERIOD_H1,5,3,3,MODE_SMA,STO_LOWHIGH);
   CopyBuffer(Stochastic_handle,0,0,k,Stochastic0);
   CopyBuffer(Stochastic_handle,1,0,k,Stochastic1);
   ArraySetAsSeries(Stochastic0,true);
   ArraySetAsSeries(Stochastic1,true);

   int CCI_Open_handle=iCCI(NULL,PERIOD_H1,14,PRICE_OPEN);
   CopyBuffer(CCI_Open_handle,0,0,k,CCI_Open);
   ArraySetAsSeries(CCI_Open,true);

   int CCI_Low_handle=iCCI(NULL,PERIOD_H1,14,PRICE_LOW);
   CopyBuffer(CCI_Low_handle,0,0,k,CCI_Low);
   ArraySetAsSeries(CCI_Low,true);

   int Momentum_Open_handle=iMomentum(NULL,PERIOD_H1,14,PRICE_OPEN);
   CopyBuffer(Momentum_Open_handle,0,0,k,Momentum_Open);
   ArraySetAsSeries(Momentum_Open,true);

   int Momentum_Low_handle=iMomentum(NULL,PERIOD_H1,14,PRICE_LOW);
   CopyBuffer(Momentum_Low_handle,0,0,k,Momentum_Low);
   ArraySetAsSeries(Momentum_Low,true);

   int RSI_Open_handle=iRSI(NULL,PERIOD_H1,14,PRICE_OPEN);
   CopyBuffer(RSI_Open_handle,0,0,k,RSI_Open);
   ArraySetAsSeries(RSI_Open,true);

   int RSI_Low_handle=iRSI(NULL,PERIOD_H1,14,PRICE_LOW);
   CopyBuffer(RSI_Low_handle,0,0,k,RSI_Low);
   ArraySetAsSeries(RSI_Low,true);

   int WPR_handle=iWPR(NULL,PERIOD_H1,14);
   CopyBuffer(WPR_handle,0,0,k,WPR);
   ArraySetAsSeries(WPR,true);

   int MACD_Open_handle=iMACD(NULL,PERIOD_H1,12,26,9,PRICE_OPEN);
   CopyBuffer(MACD_Open_handle,0,0,k,MACD_Open);
   ArraySetAsSeries(MACD_Open,true);

   int MACD_Low_handle=iMACD(NULL,PERIOD_H1,12,26,9,PRICE_LOW);
   CopyBuffer(MACD_Low_handle,0,0,k,MACD_Low);
   ArraySetAsSeries(MACD_Low,true);

   int OsMA_Open_handle=iOsMA(NULL,PERIOD_H1,12,26,9,PRICE_OPEN);
   CopyBuffer(OsMA_Open_handle,0,0,k,OsMA_Open);
   ArraySetAsSeries(OsMA_Open,true);

   int OsMA_Low_handle=iOsMA(NULL,PERIOD_H1,12,26,9,PRICE_LOW);
   CopyBuffer(OsMA_Low_handle,0,0,k,OsMA_Low);
   ArraySetAsSeries(OsMA_Low,true);

   int TriX_Open_handle=iTriX(NULL,PERIOD_H1,14,PRICE_OPEN);
   CopyBuffer(TriX_Open_handle,0,0,k,TriX_Open);
   ArraySetAsSeries(TriX_Open,true);

   int TriX_Low_handle=iTriX(NULL,PERIOD_H1,14,PRICE_LOW);
   CopyBuffer(TriX_Low_handle,0,0,k,TriX_Low);
   ArraySetAsSeries(TriX_Low,true);

   int BearsPower_handle=iBearsPower(NULL,PERIOD_H1,13);
   CopyBuffer(BearsPower_handle,0,0,k,BearsPower);
   ArraySetAsSeries(BearsPower,true);

   int ADX_MINUSDI_handle=iADX(NULL,PERIOD_H1,14);
   CopyBuffer(ADX_MINUSDI_handle,2,0,k,ADX_MINUSDI);
   ArraySetAsSeries(ADX_MINUSDI,true);

   int StdDev_Open_handle=iStdDev(NULL,PERIOD_H1,20,0,MODE_SMA,PRICE_OPEN);
   CopyBuffer(StdDev_Open_handle,0,0,k,StdDev_Open);
   ArraySetAsSeries(StdDev_Open,true);

   int StdDev_Low_handle=iStdDev(NULL,PERIOD_H1,20,0,MODE_SMA,PRICE_LOW);
   CopyBuffer(StdDev_Low_handle,0,0,k,StdDev_Low);
   ArraySetAsSeries(StdDev_Low,true);
//---------------------------------------------------------------------------------------------------------------------------

   HandleInputNet2OutNet1Min=FileOpen(Symbol()+"InputNet2OutNet1Min.csv",FILE_CSV|FILE_WRITE|FILE_SHARE_READ|FILE_ANSI|FILE_COMMON,";");
   HandleOutNet2Min=FileOpen(Symbol()+"OutNet2Min.csv",FILE_CSV|FILE_WRITE|FILE_SHARE_READ|FILE_ANSI|FILE_COMMON,";");
   FileSeek(HandleInputNet2OutNet1Min,0,SEEK_END);
   FileSeek(HandleOutNet2Min,0,SEEK_END);

   if(HandleInputNet2OutNet1Min>0)
     {
      Alert("Writing the files InputNet2OutNet1Min and OutNet2Min");

      for(int i=iBars(NULL,PERIOD_H1)-1; i>=0; i--)
        {
         Date1=TimeToString(iTime(NULL,PERIOD_H1,i));

         if(DateOut>=Date1 && Date<=Date1)
           {
            if(((DibMin1_1[i]==-1 && DibMin1_1[i+1]==1 && DibMax1_1[i]==1)) || (DibMin1_1[i]==1 && DibMax1_1[i]==1))

              {
               for(int m=0; m<=35; m++)
                 {
                  inB[m]=inB[m+12];
                 }

               inB[36]=Stochastic0[i];
               inB[37]=Stochastic1[i];
               inB[38]=CCI_Low[i];
               inB[39]=Momentum_Low[i];
               inB[40]=RSI_Low[i];;
               inB[41]=WPR[i+1];
               inB[42]=MACD_Low[i]*10000;
               inB[43]=OsMA_Low[i]*100000;
               inB[44]=TriX_Low[i]*100000;;
               inB[45]=BearsPower[i+1]*1000;
               inB[46]=ADX_MINUSDI[i+1];
               inB[47]=StdDev_Low[i]*10000;

               inB[48]=Stochastic0[i];
               inB[49]=Stochastic1[i];
               inB[50]=CCI_Open[i];
               inB[51]=Momentum_Open[i];
               inB[52]=RSI_Open[i];;
               inB[53]=WPR[i];
               inB[54]=MACD_Open[i]*10000;
               inB[55]=OsMA_Open[i]*100000;
               inB[56]=TriX_Open[i]*100000;;
               inB[57]=BearsPower[i]*1000;
               inB[58]=ADX_MINUSDI[i];
               inB[59]=StdDev_Open[i]*10000;

               FileWrite(HandleInputNet2OutNet1Min,
                         inB[0],inB[1],inB[2],inB[3],inB[4],inB[5],inB[6],inB[7],inB[8],inB[9],inB[10],inB[11],inB[12],inB[13],
                         inB[14],inB[15],inB[16],inB[17],inB[18],inB[19],inB[20],inB[21],inB[22],inB[23],inB[24],inB[25],inB[26],
                         inB[27],inB[28],inB[29],inB[30],inB[31],inB[32],inB[33],inB[34],inB[35],inB[36],inB[37],inB[38],inB[39],
                         inB[40],inB[41],inB[42],inB[43],inB[44],inB[45],inB[46],inB[47],inB[48],inB[49],inB[50],inB[51],inB[52],
                         inB[53],inB[54],inB[55],inB[56],inB[57],inB[58],inB[59]);

               FileWrite(HandleOutNet2Min,
                         (iOpen(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i)))-iOpen(NULL,PERIOD_H1,i))*10000);
              }
           }
        }
     }

   //------ Daily High

   int CCI_High_handle=iCCI(NULL,PERIOD_H1,14,PRICE_HIGH);
   CopyBuffer(CCI_High_handle,0,0,k,CCI_High);
   ArraySetAsSeries(CCI_High,true);

   int Momentum_High_handle=iMomentum(NULL,PERIOD_H1,14,PRICE_HIGH);
   CopyBuffer(Momentum_High_handle,0,0,k,Momentum_High);
   ArraySetAsSeries(Momentum_High,true);

   int RSI_High_handle=iRSI(NULL,PERIOD_H1,14,PRICE_HIGH);
   CopyBuffer(RSI_High_handle,0,0,k,RSI_High);
   ArraySetAsSeries(RSI_High,true);

   int MACD_High_handle=iMACD(NULL,PERIOD_H1,12,26,9,PRICE_HIGH);
   CopyBuffer(MACD_High_handle,0,0,k,MACD_High);
   ArraySetAsSeries(MACD_High,true);

   int OsMA_High_handle=iOsMA(NULL,PERIOD_H1,12,26,9,PRICE_HIGH);
   CopyBuffer(OsMA_High_handle,0,0,k,OsMA_High);
   ArraySetAsSeries(OsMA_High,true);

   int TriX_High_handle=iTriX(NULL,PERIOD_H1,14,PRICE_HIGH);
   CopyBuffer(TriX_High_handle,0,0,k,TriX_High);
   ArraySetAsSeries(TriX_High,true);

   int BullsPower_handle=iBullsPower(NULL,PERIOD_H1,13);
   CopyBuffer(BullsPower_handle,0,0,k,BullsPower);
   ArraySetAsSeries(BullsPower,true);

   int ADX_PLUSDI_handle=iADX(NULL,PERIOD_H1,14);
   CopyBuffer(ADX_PLUSDI_handle,1,0,k,ADX_PLUSDI);
   ArraySetAsSeries(ADX_PLUSDI,true);

   int StdDev_High_handle=iStdDev(NULL,PERIOD_H1,20,0,MODE_SMA,PRICE_HIGH);
   CopyBuffer(StdDev_High_handle,0,0,k,StdDev_High);
   ArraySetAsSeries(StdDev_High,true);
//---------------------------------------------------------------------------------------------------------------------------

   HandleInputNet2OutNet1Max=FileOpen(Symbol()+"InputNet2OutNet1Max.csv",FILE_CSV|FILE_WRITE|FILE_SHARE_READ|FILE_ANSI|FILE_COMMON,";");
   HandleOutNet2Max=FileOpen(Symbol()+"OutNet2Max.csv",FILE_CSV|FILE_WRITE|FILE_SHARE_READ|FILE_ANSI|FILE_COMMON,";");
   FileSeek(HandleInputNet2OutNet1Max,0,SEEK_END);
   FileSeek(HandleOutNet2Max,0,SEEK_END);

   if(HandleInputNet2OutNet1Max>0)
     {
      Alert("Writing the files InputNet2OutNet1Max and OutNet2Max");

      for(int i=iBars(NULL,PERIOD_H1)-1; i>=0; i--)
        {
         Date1=TimeToString(iTime(NULL,PERIOD_H1,i));

         if(DateOut>=Date1 && Date<=Date1)
           {
            if(((DibMax1_1[i]==-1 && DibMax1_1[i+1]==1 && DibMin1_1[i]==1)) || (DibMin1_1[i]==1 && DibMax1_1[i]==1))

              {
               for(int m=0; m<=35; m++)
                 {
                  inS[m]=inS[m+12];
                 }

               inS[36]=Stochastic0[i];
               inS[37]=Stochastic1[i];
               inS[38]=CCI_High[i];
               inS[39]=Momentum_High[i];
               inS[40]=RSI_High[i];;
               inS[41]=WPR[i+1];
               inS[42]=MACD_High[i]*10000;
               inS[43]=OsMA_High[i]*100000;
               inS[44]=TriX_High[i]*100000;;
               inS[45]=BullsPower[i+1]*1000;
               inS[46]=ADX_PLUSDI[i+1];
               inS[47]=StdDev_High[i]*10000;

               inS[48]=Stochastic0[i];
               inS[49]=Stochastic1[i];
               inS[50]=CCI_Open[i];
               inS[51]=Momentum_Open[i];
               inS[52]=RSI_Open[i];;
               inS[53]=WPR[i];
               inS[54]=MACD_Open[i]*10000;
               inS[55]=OsMA_Open[i]*100000;
               inS[56]=TriX_Open[i]*100000;;
               inS[57]=BullsPower[i]*1000;
               inS[58]=ADX_PLUSDI[i];
               inS[59]=StdDev_Open[i]*10000;

               FileWrite(HandleInputNet2OutNet1Max,
                         inS[0],inS[1],inS[2],inS[3],inS[4],inS[5],inS[6],inS[7],inS[8],inS[9],inS[10],inS[11],inS[12],inS[13],
                         inS[14],inS[15],inS[16],inS[17],inS[18],inS[19],inS[20],inS[21],inS[22],inS[23],inS[24],inS[25],inS[26],
                         inS[27],inS[28],inS[29],inS[30],inS[31],inS[32],inS[33],inS[34],inS[35],inS[36],inS[37],inS[38],inS[39],
                         inS[40],inS[41],inS[42],inS[43],inS[44],inS[45],inS[46],inS[47],inS[48],inS[49],inS[50],inS[51],inS[52],
                         inS[53],inS[54],inS[55],inS[56],inS[57],inS[58],inS[59]);

               FileWrite(HandleOutNet2Max,
                         (iOpen(NULL,PERIOD_H1,i)-iOpen(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i))))*10000);
              }
           }
        }
     }

   Alert("Files written");
  }
//+------------------------------------------------------------------+
```

One sample will be from the beginning of the working day until the first reach of the day's low. The second sample will be till the first reach of the day's high. For this purpose, two indicators will be used in scripts: DibMin1-1.mq5 and DibMax1-1.mq5

```
//+------------------------------------------------------------------+
//|                                                    DibMin1-1.mq5 |
//|                                   Copyright 2020, Andrey Dibrov. |
//|                           https://www.mql5.com/ru/users/tomcat66 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, Andrey Dibrov."
#property link      "https://www.mql5.com/ru/users/tomcat66"
#property version   "1.00"
#property indicator_separate_window
#property indicator_buffers 1
#property indicator_plots   1
#property indicator_type1   DRAW_LINE
#property indicator_minimum -2
#property indicator_maximum 2
#property indicator_color1 Red
#property indicator_label1  "DibMin1-1"
//---- input parameters
input int History=500;
double Buf[];
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping
   SetIndexBuffer(0,Buf,INDICATOR_DATA);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
//---
   int    i,z,Calc;
   double price;

   i=iBars(NULL,PERIOD_H1)-1;
   if(i>History-1)
      i=History-1;
   if(History==0)
      i=iBars(NULL,PERIOD_H1)-1;
   ArraySetAsSeries(Buf,true);
   ArraySetAsSeries(time,true);

   while(i>=0)
     {
      int min=0;
      Calc=(int)time[i]%86400/3600;
      double min1=iLow(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i)));
      for(z=0;z<=Calc;z++)
        {
          price=iLow(NULL,PERIOD_H1,i+z);
          if(min1<price)
          {
           min=1;
          }else
          {
           min=-1;
          break;
          }
         }
      Buf[i]=min;
      i--;
     }
   return(rates_total);
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//|                                                    DibMax1-1.mq5 |
//|                                   Copyright 2020, Andrey Dibrov. |
//|                           https://www.mql5.com/ru/users/tomcat66 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, Andrey Dibrov."
#property link      "https://www.mql5.com/ru/users/tomcat66"
#property version   "1.00"
#property indicator_separate_window
#property indicator_buffers 1
#property indicator_plots   1
#property indicator_type1   DRAW_LINE
#property indicator_minimum -2
#property indicator_maximum 2
#property indicator_color1 LightSeaGreen
#property indicator_label1  "DibMax1-1"
//---- input parameters
input int History=500;
double Buf[];
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping
   SetIndexBuffer(0,Buf,INDICATOR_DATA);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
//---
   int    i,z,Calc;
   double price;

   i=iBars(NULL,PERIOD_H1)-1;
   if (i>History-1)i=History-1;
   if (History==0) i=iBars(NULL,PERIOD_H1)-1;
   ArraySetAsSeries(Buf,true);
   ArraySetAsSeries(time,true);

   while(i>=0)
     {
      int max=0;
      Calc=(int)time[i]%86400/3600;
      double max1=iHigh(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_H1,i)));
      for(z=0;z<=Calc;z++)
        {
          price=iHigh(NULL,PERIOD_H1,i+z);//+1-1
          if(max1>price)
          {
           max=1;
          }else
          {
           max=-1;
           break;
          }
         }
      Buf[i]=max;
      i--;
      }
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

![Sample indicators](https://c.mql5.com/2/40/EURUSDH1.png)

When the price reaches daily extreme values, indicator values are set to -1.

After running the scripts on the EURUSD H1 chart, six CSV files will be created in the \\Common\\Files folder.

- EURUSDInputNet1Max and EURUSDInputNet1Min are files with price data. We can see from the file name that, for example EURUSDInputNet1Max contains input data for the Net1Max neural networks.
- EURUSDInputNet2OutNet1Max and EURUSDInputNet2OutNet1Min are files with indicator values. These values will be inputs for Net2 and outputs for Net1. **Please note that this part requires some experimenting: Net2 can be trained using standard technical indicators or using Net1 responses.**
- EURUSDOutNet2Max and EURUSDOutNet2Min are outputs for Net2: the price difference between hour opening and day opening (or day closing and hour opening).

1. Neural networks are trained by the price approaching extreme values, while we will interpret not the price itself, but the indicator values.
2. The neural network target is the difference between the hour opening and day opening(day closing and hour opening) prices. **You can also try here other targets, such as for example the difference between other prices.**
3. With this approach, we smooth out the probabilities of error responses in neural network modules, since neural networks are not trained to find specific daily High and Low prices, but they work with the probability of high/low approaching, taking into account the price amplitude to the daily extremum. If we then decide to use the price amplitude, we can use the difference between the daily closing and hour opening. The first option seems to be preferable, because in this case wetrain the neural network using achieved targets rather than events that should happen. This option is more logical, because assessing past events is easier than making predictions.

### 2\. Python neural network training

First of all, check the [Integration](https://www.mql5.com/en/docs/integration/python_metatrader5) section of the MQL5 documentation. After installing Python 3.8 and connecting the MetaTrader 5 integration module, connect TensorFlow, Keras, Numpy and Pandas libraries in the same way.

![Installing TensorFlow](https://c.mql5.com/2/40/tensor.png)

![TensorFlow setup window](https://c.mql5.com/2/40/TensorFlow.png)

Neural networks will be trained using the Python script EURUSDPyTren.py.

```
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

InputNet1=pd.read_csv('EURUSDInputNet1Min.csv', delimiter=';',header=None)
InputNet2OutNet1=pd.read_csv('EURUSDInputNet2OutNet1Min.csv', delimiter=';',header=None)
OutNet2=pd.read_csv('EURUSDOutNet2Min.csv', delimiter=';',header=None)

mean = InputNet1.mean(axis=0)
std = InputNet1.std(axis=0)
InputNet1 -= mean
InputNet1 /= std

mean = InputNet2OutNet1.mean(axis=0)
std = InputNet2OutNet1.std(axis=0)
InputNet2OutNet1 -= mean
InputNet2OutNet1 /= std

Net1Min = Sequential()
Net1Min.add(Dense(22, activation='relu', input_shape=(InputNet1.shape[1],)))
Net1Min.add(Dense(60))

Net1Min.compile(optimizer='adam', loss='mse', metrics=['mse'])
print(Net1Min.summary())

Net1Min.fit(InputNet1, InputNet2OutNet1, epochs=10, batch_size=10,verbose=2,validation_split=0.3)
Net1Min.save('net1Min.h5')

mean = OutNet2.mean(axis=0)
std = OutNet2.std(axis=0)
OutNet2 -= mean
OutNet2 /= std

Net2Min = Sequential()
Net2Min.add(Dense(60, activation='relu', input_shape=(InputNet2OutNet1.shape[1],)))
Net2Min.add(Dense(1))

Net2Min.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(Net2Min.summary())

Net2Min.fit(InputNet2OutNet1, OutNet2, epochs=100, batch_size=10,verbose=2,validation_split=0.3)
Net2Min.save('net2Min.h5')

InputNet1=pd.read_csv('EURUSDInputNet1Max.csv', delimiter=';',header=None)
InputNet2OutNet1=pd.read_csv('EURUSDInputNet2OutNet1Max.csv', delimiter=';',header=None)
OutNet2=pd.read_csv('EURUSDOutNet2Max.csv', delimiter=';',header=None)

mean = InputNet1.mean(axis=0)
std = InputNet1.std(axis=0)
InputNet1 -= mean
InputNet1 /= std

mean = InputNet2OutNet1.mean(axis=0)
std = InputNet2OutNet1.std(axis=0)
InputNet2OutNet1 -= mean
InputNet2OutNet1 /= std

Net1Max = Sequential()
Net1Max.add(Dense(22, activation='relu', input_shape=(InputNet1.shape[1],)))
Net1Max.add(Dense(60))

Net1Max.compile(optimizer='adam', loss='mse', metrics=['mse'])
print(Net1Max.summary())

Net1Max.fit(InputNet1, InputNet2OutNet1, epochs=10, batch_size=10,verbose=2,validation_split=0.3)
Net1Max.save('net1Max.h5')

mean = OutNet2.mean(axis=0)
std = OutNet2.std(axis=0)
OutNet2 -= mean
OutNet2 /= std

Net2Max = Sequential()
Net2Max.add(Dense(60, activation='relu', input_shape=(InputNet2OutNet1.shape[1],)))
Net2Max.add(Dense(1))

Net2Max.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(Net2Max.summary())

Net2Max.fit(InputNet2OutNet1, OutNet2, epochs=100, batch_size=10,verbose=2,validation_split=0.3)
Net2Max.save('net2Max.h5')

NetTest=pd.read_csv('EURUSDTest.csv', delimiter=';',header=None)
Date=pd.read_csv('EURUSDDate.csv', delimiter=';',header=None)

Net1Min = load_model('net1Min.h5')
Net2Min=  load_model('net2Min.h5')
Net1Max = load_model('net1Max.h5')
Net2Max=  load_model('net2Max.h5')
Net1Min = Net1Min.predict(NetTest)
Net2Min = Net2Min.predict(Net1Min)
Net1Max = Net1Max.predict(NetTest)
Net2Max = Net2Max.predict(Net1Max)

Date=pd.DataFrame(Date)
Date['0'] = Net2Min
Date.to_csv('IndicatorMin.csv',index=False, header=False,sep=';')
Date['0'] = Net2Max
Date.to_csv('IndicatorMax.csv',index=False, header=False,sep=';')

Date['0'] = Net2Min
Date['1'] = Net2Max
Date.to_csv('Indicator.csv',index=False, header=False,sep=';')

input('Press ENTER to exit')
```

Save this script at \\Common\\Files.

Let us take a closer look at the script.

```
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
```

Block for connecting libraries, packages and modules from Keras.

```
InputNet1=pd.read_csv('EURUSDInputNet1Min.csv', delimiter=';',header=None)
InputNet2OutNet1=pd.read_csv('EURUSDInputNet2OutNet1Min.csv', delimiter=';',header=None)
OutNet2=pd.read_csv('EURUSDOutNet2Min.csv', delimiter=';',header=None)
```

Form dataframe from data files.

```
mean = InputNet1.mean(axis=0)
std = InputNet1.std(axis=0)
InputNet1 -= mean
InputNet1 /= std
```

Standardize the data.

```
Net1Min = Sequential()
Net1Min.add(Dense(22, activation='relu', input_shape=(InputNet1.shape[1],)))
Net1Min.add(Dense(60))
```

The sequential network model, input layer has 22 neurons, output layer has 60 neurons.

```
Net1Min = Sequential()
Net1Min.add(Dense(22, activation='relu', input_shape=(InputNet1.shape[1],)))
Net1Min.add(Dense(11))
Net1Min.add(Dense(60))
```

**Further, you can try to add a hidden layer.**

```
Net1Max.compile(optimizer='adam', loss='mse', metrics=['mse'])
print(Net1Max.summary())
```

Compile the network and print its parameters.

```
Net1Min.fit(InputNet1, InputNet2OutNet1, epochs=10, batch_size=10,verbose=2,validation_split=0.3)
Net1Min.save('net1Min.h5')
```

Neural network training: 10 epochs; mini-sample size is 10; 30% of training data is allocated for validation. **These hyper parameters may also need further adjustment.** Verbose - training epoch visualization parameter. Save the trained neural network.

```
NetTest=pd.read_csv('EURUSDTest.csv', delimiter=';',header=None)
Date=pd.read_csv('EURUSDDate.csv', delimiter=';',header=None)
```

Form dataframe for testing.

```
Net1Min = load_model('net1Min.h5')
Net2Min=  load_model('net2Min.h5')
Net1Max = load_model('net1Max.h5')
Net2Max=  load_model('net2Max.h5')
Net1Min = Net1Min.predict(NetTest)
Net2Min = Net2Min.predict(Net1Min)
Net1Max = Net1Max.predict(NetTest)
Net2Max = Net2Max.predict(Net1Max)
```

Load the saved neural networks and obtain the result from them.

```
Date=pd.DataFrame(Date)
Date['0'] = Net2Min
Date.to_csv('IndicatorMin.csv',index=False, header=False,sep=';')
Date['0'] = Net2Max
Date.to_csv('IndicatorMax.csv',index=False, header=False,sep=';')

Date['0'] = Net2Min
Date['1'] = Net2Max
Date.to_csv('Indicator.csv',index=False, header=False,sep=';')
```

Save the obtained data to files.

```
input('Press ENTER to exit')
```

Wait or the window to close.

To ensure normal script operation, it is also required to prepare data for receiving neural network responses, which will be used to form the indicator and to analyze the efficiency of this indicator for a trading strategy.

This will be done using the PythonTestExpert Expert Advisors.

```
//+------------------------------------------------------------------+
//|                                             PythonTestExpert.mq5 |
//|                                   Copyright 2020, Andrey Dibrov. |
//|                           https://www.mql5.com/ru/users/tomcat66 |
//+------------------------------------------------------------------+
#property copyright " Copyright © 2019, Andrey Dibrov."
#property link      "https://www.mql5.com/ru/users/tomcat66"
#property version   "1.00"
#property strict

int handleInput;
int HandleDate;

double in[22];
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   handleInput=FileOpen(Symbol()+"Test.csv",FILE_CSV|FILE_WRITE|FILE_SHARE_READ|FILE_ANSI|FILE_COMMON,";");
   HandleDate=FileOpen(Symbol()+"Date.csv",FILE_CSV|FILE_READ|FILE_WRITE|FILE_ANSI|FILE_COMMON,";");
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
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   for(int i=0; i<=14; i++)
     {
      in[i]=in[i+5];
     }

   in[15]=((iOpen(NULL,PERIOD_D1,0)-iLow(NULL,PERIOD_D1,0))*100000);
   in[16]=((iHigh(NULL,PERIOD_D1,0)-iOpen(NULL,PERIOD_D1,0))*100000);
   in[17]=((iHigh(NULL,PERIOD_D1,0)-iLow(NULL,PERIOD_D1,0))*100000);
   in[18]=((iHigh(NULL,PERIOD_D1,0)-iOpen(NULL,PERIOD_H1,1))*10000);
   in[19]=((iOpen(NULL,PERIOD_H1,1)-iLow(NULL,PERIOD_D1,0))*10000);

   in[20]=((iHigh(NULL,PERIOD_D1,0)-iOpen(NULL,PERIOD_H1,0))*10000);
   in[21]=((iOpen(NULL,PERIOD_H1,0)-iLow(NULL,PERIOD_D1,0))*10000);

   FileWrite(handleInput,

             in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7],in[8],in[9],in[10],in[11],in[12],in[13],in[14],in[15],
             in[16],in[17],in[18],in[19],in[20],in[21]);

   FileWrite(HandleDate,TimeCurrent());

  }
//+------------------------------------------------------------------+
```

Run the Expert Advisor in the Strategy Tester on H1 chart, using the Open Prices mode. The data period for testing: from the beginning of 2011 to current date. This Expert Advisor will simulate data which the Python script will form based on prices received from MetaTrader 5 in real operation.

![](https://c.mql5.com/2/41/xpyare_h70sh5u6_qek_r5jf0adsgefe_peso1uh0_pvdf10.png)

The EA will create two files, EURUSDTest and EURUSDDate, in the \\Common folder.

Run the Python script EURUSDPyTren.py. (If it does not work, you might need to reinstall the earlier described additional packages and to restart your computer). If everything is correct, the script will run by double clicking.

![Script operation](https://c.mql5.com/2/40/script.png)

As a result, the following files will be created in the \\Common\\Files folder:

- net1Max.h5, net1Min.h5, net2Max.h5, net2Min.h5 — these are the trained networks that will be used in the basic script when trading in real time.
- IndicatorMax and IndicatorMin are two network responses for separate testing.
- Indicator is a combined response.

![Folder \Common\Files](https://c.mql5.com/2/40/Files.png)

Launch the 1\_MT5 indicator in the terminal.

```
//+------------------------------------------------------------------+
//|                                                        1_MT5.mq5 |
//|                                 Copyright © 2019, Andrey Dibrov. |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2019, Andrey Dibrov."
//--- indicator settings
#property indicator_separate_window
#property indicator_buffers 2
#property indicator_plots   2
#property indicator_type1   DRAW_LINE
#property indicator_type2   DRAW_LINE
#property indicator_color1  Red
#property indicator_color2  DodgerBlue

int Handle;
int i;

double    ExtBuffer[];
double    SignBuffer[];

datetime Date1;
datetime Date0;

string File_Name="Indicator.csv";

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
void OnInit()
  {
   SetIndexBuffer(0,ExtBuffer,INDICATOR_DATA);
   SetIndexBuffer(1,SignBuffer,INDICATOR_DATA);

   IndicatorSetInteger(INDICATOR_DIGITS,5);

  }
//+------------------------------------------------------------------+
//| Relative Strength Index                                          |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const int begin,
                const double &price[])
  {

   Handle=FileOpen(File_Name,FILE_CSV|FILE_SHARE_READ|FILE_ANSI|FILE_COMMON,";");

   Date0=StringToTime(FileReadString(Handle));

   FileClose(Handle);

   i=iBarShift(NULL,PERIOD_H1,Date0,false);

   Handle=FileOpen(File_Name,FILE_CSV|FILE_SHARE_READ|FILE_ANSI|FILE_COMMON,";");

   ArraySetAsSeries(ExtBuffer,true);
   ArraySetAsSeries(SignBuffer,true);

   while(!FileIsEnding(Handle) && !IsStopped())
     {
      Date1=StringToTime(FileReadString(Handle));
      ExtBuffer[i]=StringToDouble(FileReadString(Handle));
      SignBuffer[i]=StringToDouble(FileReadString(Handle));
      i--;
     }

   FileClose(Handle);

   return(rates_total);
  }
//+------------------------------------------------------------------+
```

![Indicator](https://c.mql5.com/2/41/y6hffssslb5.png)

Although we have not put much effort into the neural network training, there are some dependencies visible.

### 3\. Optimization of the results obtained.

Let us use the PythonOptimizExpert EA to optimize the resulting indicator.

```
//+------------------------------------------------------------------+
//|                                          PythonOptimizExpert.mq5 |
//|                                   Copyright 2020, Andrey Dibrov. |
//|                           https://www.mql5.com/ru/users/tomcat66 |
//+------------------------------------------------------------------+
#property copyright " Copyright © 2019, Andrey Dibrov."
#property link      "https://www.mql5.com/ru/users/tomcat66"
#property version   "1.00"
#property strict

#include<Trade\Trade.mqh>

CTrade  trade;

input int H1;
input int H2;
input int H3;
input int H4;

input double Buy;
input double Buy1;
input double Sell;
input double Sell1;

input int LossBuy;
input int ProfitBuy;
input int LossSell;
input int ProfitSell;

ulong TicketBuy1;
ulong TicketSell0;

datetime Count;

double Buf_0[];
double Buf_1[];

bool send1;
bool send0;

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
      Buf_1[k]=StringToDouble(FileReadString(Handle));
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
   if(send1==false && K>0 &&  Buf_0[K]<Buy && Buy<Buy1 && iLow(NULL,PERIOD_H1,1)<iLow(NULL,PERIOD_H1,2) && stm.hour>H1 && stm.hour<H2 && H1<H2)
     {
      send1=trade.PositionOpen(_Symbol,ORDER_TYPE_BUY,1,PriceAsk,SL1,TP1);//SL1,TP1
      TicketBuy1 = trade.ResultDeal();
     }

   if(send1==true && K>0 &&  Buf_0[K]>Buy1 &&  Buy<Buy1 && iHigh(NULL,PERIOD_H1,1)>iHigh(NULL,PERIOD_H1,2) )
     {
      trade.PositionClose(TicketBuy1);
      send1=false;
     }

//---------Sell0

   if(send0==false && K>0 &&  Buf_1[K]<Sell && Sell<Sell1 && iHigh(NULL,PERIOD_H1,1)>iHigh(NULL,PERIOD_H1,2) && stm.hour>H3 && stm.hour<H4 && H3<H4)
     {
      send0=trade.PositionOpen(_Symbol,ORDER_TYPE_SELL,1,PriceBid,SL0,TP0);//SL0,TP0
      TicketSell0 = trade.ResultDeal();
     }

   if(send0==true && K>0 &&  Buf_1[K]>Sell1 && Sell<Sell1 && iLow(NULL,PERIOD_H1,1)<iLow(NULL,PERIOD_H1,2) )
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

The following variables can be optimized:

- Н1, Н2 — the hours of the day between which buying will be performed.
- Н3 ,H4 — the hours of the day between which selling will be performed.
- Buy, Buy1 — these indicate the Buf\_0\[\] indicator's red line levels, which is the response of the neural network trained on the data up to day's low.
- Sell, Sell1 —these indicate the Buf\_1\[\] indicator's blue line levels, which is the response of the neural network trained on the data up to day's high.
- LossBuy, ProfitBuy, LossSell, ProfitSell — limiting levels.

Let us set the optimization parameters as shown in the figure. It means we will only optimize indicator levels.

![](https://c.mql5.com/2/41/xtg9nu8bquijkh_xqf0m0.png)

Optimized indicator levels

![](https://c.mql5.com/2/41/0gzcfyea0nj1.png)

![](https://c.mql5.com/2/41/019hux3vc_k8mvyuttezl.png)

Optimization parameters

Optimization results

![](https://c.mql5.com/2/41/3ql3sl9my3j2.png)

![](https://c.mql5.com/2/41/ne0apizlmzg3.png)

Testing results. The optimization period is up to the red line on the test chart. This is followed by testing using optimized parameters.

![](https://c.mql5.com/2/41/kolq_id_8ch66kmzfl_4nz3ee1o8i.png)![](https://c.mql5.com/2/41/xe8y_vq_778o8m8xd8_6hsum3lf9y2.png)

Using the obtained parameters, let us optimize further by time and stop orders.

![](https://c.mql5.com/2/41/urlct7hv7iq_mz_nlng01xu93_dthaw8io2s.png)

![](https://c.mql5.com/2/41/039lzbudf54_71_dj7a7dd6p7_96ugdrfhm71.png)

![](https://c.mql5.com/2/41/ne0apizlmzg3__1.png)

Now, let us test all the obtained parameters.

![](https://c.mql5.com/2/41/ohuy.png)

![](https://c.mql5.com/2/41/u92m1.png)

Optionally, you can try to optimize each trading direction separately.

### Conclusions

The video below will help you in understanding the Python script preparation.

Practical application of NN in trading - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8502)

MQL5.community

1.91K subscribers

[Practical application of NN in trading](https://www.youtube.com/watch?v=UYPmb0L8kSc)

MQL5.community

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

[Watch on](https://www.youtube.com/watch?v=UYPmb0L8kSc&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8502)

0:00

0:00 / 2:20

•Live

•

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8502](https://www.mql5.com/ru/articles/8502)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8502.zip "Download all attachments in the single ZIP archive")

[EURUSDPyTren.py](https://www.mql5.com/en/articles/download/8502/eurusdpytren.py "Download EURUSDPyTren.py")(3.47 KB)

[PythonTestExpert.mq5](https://www.mql5.com/en/articles/download/8502/pythontestexpert.mq5 "Download PythonTestExpert.mq5")(5.53 KB)

[DibMax1-1.mq5](https://www.mql5.com/en/articles/download/8502/dibmax1-1.mq5 "Download DibMax1-1.mq5")(5.11 KB)

[DibMin1-1.mq5](https://www.mql5.com/en/articles/download/8502/dibmin1-1.mq5 "Download DibMin1-1.mq5")(5.08 KB)

[PythonOptimizExpert.mq5](https://www.mql5.com/en/articles/download/8502/pythonoptimizexpert.mq5 "Download PythonOptimizExpert.mq5")(9.46 KB)

[PythonIndicators.mq5](https://www.mql5.com/en/articles/download/8502/pythonindicators.mq5 "Download PythonIndicators.mq5")(24.23 KB)

[PythonPrices.mq5](https://www.mql5.com/en/articles/download/8502/pythonprices.mq5 "Download PythonPrices.mq5")(11.46 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Practical application of neural networks in trading (Part 2). Computer vision](https://www.mql5.com/en/articles/8668)
- [Practical application of neural networks in trading. It's time to practice](https://www.mql5.com/en/articles/7370)
- [Practical application of neural networks in trading](https://www.mql5.com/en/articles/7031)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/359565)**
(19)


![Soso KbNuevon](https://c.mql5.com/avatar/2021/11/61818FE6-2712.jpg)

**[Soso KbNuevon](https://www.mql5.com/en/users/sofianebenslima)**
\|
29 Dec 2021 at 00:09

Thnaks fot his amazing content.

How to call the model from [mql5 script](https://www.mql5.com/en/docs/constants/environment_state/mql5_programm_info#enum_program_type "MQL5 documentation: Running MQL5 Program Properties") ?

![dustin  shiozaki](https://c.mql5.com/avatar/avatar_na2.png)

**[dustin shiozaki](https://www.mql5.com/en/users/dustovshio)**
\|
29 Jul 2022 at 09:09

I compiled the indicator 1\_MT5 from the code and it is in the indicator folder.

When I use the Optimizer EA It doesn't open any trades.  I have generated the indicator.csv and there is data in it.  The indicator 1\_MT5 draws on the chart.


![KrisnaMT5](https://c.mql5.com/avatar/2024/3/66062c29-abb6.jpg)

**[KrisnaMT5](https://www.mql5.com/en/users/krisnamt5)**
\|
2 Sep 2022 at 14:57

**dustovshio [#](https://www.mql5.com/en/forum/359565#comment_41093586) :** Saya mengkompilasi indikator 1\_MT5 dari kode dan ada di folder indikator.   Saat saya menggunakan EA Pengoptimal, EA ini tidak membuka perdagangan apa pun. Saya telah membuat indicator.csv dan ada data di dalamnya. Indikator 1\_MT5 mengacu pada grafik.

i am also having the same problem. have you found the solution? can you tell me, please?


![Pavel Komarovsky](https://c.mql5.com/avatar/2016/12/5857BC29-D4FF.jpeg)

**[Pavel Komarovsky](https://www.mql5.com/en/users/pavelk)**
\|
26 Jan 2023 at 21:00

Andrey hello,

1) I tried to do everything by steps, I reached the step when the python script worked and created files, then I run the indicator 1\_MT5 and it does not display anything.

The files Indicator, IndicatorMin, IndicatorMax contain only

03.01.2011 0:00

03.01.2011 1:00

03.01.2011 2:00

03.01.2011 0:00

03.01.2011 1:00

03.01.2011 2:00

is this the way to do it? what am I doing wrong?

2) what should I change if I just want to pass bars and standard indicators to neuronka, i.e. without the logic of "sampling from the beginning of the working day until the first one reaches the minimum of the day, the second one until the first one reaches the maximum of the day".

![brastor](https://c.mql5.com/avatar/avatar_na2.png)

**[brastor](https://www.mql5.com/en/users/brastor)**
\|
27 Sep 2023 at 08:03

**KrisnaMT5 [#](https://www.mql5.com/en/forum/359565#comment_41799413):**

i am also having the same problem. have you found the solution? can you tell me, please?

You need to set the initial values for H1, H2, H3, H4 (2, 22, 2, 22) on optimization settings.


![Timeseries in DoEasy library (part 56): Custom indicator object, get data from indicator objects in the collection](https://c.mql5.com/2/41/MQL5-avatar-doeasy-library.png)[Timeseries in DoEasy library (part 56): Custom indicator object, get data from indicator objects in the collection](https://www.mql5.com/en/articles/8646)

The article considers creation of the custom indicator object for the use in EAs. Let’s slightly improve library classes and add methods to get data from indicator objects in EAs.

![Neural networks made easy (Part 5): Multithreaded calculations in OpenCL](https://c.mql5.com/2/48/Neural_networks_made_easy_0065.png)[Neural networks made easy (Part 5): Multithreaded calculations in OpenCL](https://www.mql5.com/en/articles/8435)

We have earlier discussed some types of neural network implementations. In the considered networks, the same operations are repeated for each neuron. A logical further step is to utilize multithreaded computing capabilities provided by modern technology in an effort to speed up the neural network learning process. One of the possible implementations is described in this article.

![Timeseries in DoEasy library (part 57): Indicator buffer data object](https://c.mql5.com/2/41/MQL5-avatar-doeasy-library__1.png)[Timeseries in DoEasy library (part 57): Indicator buffer data object](https://www.mql5.com/en/articles/8705)

In the article, develop an object which will contain all data of one buffer for one indicator. Such objects will be necessary for storing serial data of indicator buffers. With their help, it will be possible to sort and compare buffer data of any indicators, as well as other similar data with each other.

![Timeseries in DoEasy library (part 55): Indicator collection class](https://c.mql5.com/2/40/MQL5-avatar-doeasy-library__7.png)[Timeseries in DoEasy library (part 55): Indicator collection class](https://www.mql5.com/en/articles/8576)

The article continues developing indicator object classes and their collections. For each indicator object create its description and correct collection class for error-free storage and getting indicator objects from the collection list.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=quoxjfelytvcmaovxlxmsngwpdbdvfov&ssn=1769181368592045560&ssn_dr=0&ssn_sr=0&fv_date=1769181368&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8502&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Practical%20application%20of%20neural%20networks%20in%20trading.%20Python%20(Part%20I)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918136828348405&fz_uniq=5069295714510504631&sv=2552)

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