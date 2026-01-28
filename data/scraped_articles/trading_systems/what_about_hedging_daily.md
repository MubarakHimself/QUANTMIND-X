---
title: What about Hedging Daily?
url: https://www.mql5.com/en/articles/1532
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:56:51.210900
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/1532&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083231878363748209)

MetaTrader 4 / Trading systems


### Introduction

I am giving away an idea of hedging the GBP/JPY & EUR/JPY daily. Yes Daily. This idea popped up while I was trying to generate a trading system that plays on the daily scheme, opening the trade only once a day, collecting the profit only around $100 each day, and of course, all the things do automatically. At the first time, I only tried to trade with the TD-Sequential System, owned by Tom Demark. I tried trading it daily, actually it dose a good system, but somehow give me an unacceptable loss, then the idea of hedging the two correlated pairs came out. "Why don't hedge ?, then you will lose less than today or maybe you can gain, sounds grate ! ". Then i tested it manually with 1 month past history, and found a good profit maker sign. So...no need to wait for anything, just make it automatically trade for you and test it live for a couple of months or longer. Now let's start making it comes true.

### Concept of Daily Hedge

Before we start the coding process, let's make a plan together. Including...

-\> What will we use for signaling the daily trading trend? : This will give us the estimate today's direction of GBP/JPY and EUR/JPY (these two pairs are always 90% correlated) . In this case i still choose the TD-Sequential System, an easy TD-Sequential I've found in a forum, to give me the daily signal.

-\> Which hedging pairs to hedge? : Just select your favorite pairs. Mine are GBP/JPY and EUR/JPY, with the reason above.

-\> Which pair will be the base pair? & Which one will be the hedge pair? : This will make it easier to code the EA. I decided to mark the EUR/JPY as my base pair and hedge GBP/JPY. ("Why Base & Hedge?", that's because of the system is hedge by the daily trend.) For example, today the TD-Sequential signal out the UP trend of EUR/JPY, then I will BUY EUR/JPY and hedge by selling GBP/JPY. Or maybe you can make sure by mark the UP day only when both EUR/JPY and GBP/JPY are showing the TD-Sequential UP, then buy the base pair & sell the hedge pair.

-\> What is the ... correlation ? : Of course, we need this factor, and you all know, it is an important factor of hedging system. In this case, I will only allow to hedge when the correlation of those two pairs is 0.9 or higher only. YES, please don't be astonished. Yes 0.9+ "WHY ?", I know every hedge professor suggests you to hedge when the correlation is low, but that is for a very very and very long - term . To me and my daily hedging system, hedging at high correlation is better. Please NOTE that, this is for my daily hedging system only. Because we need them to go the same way always, especially for today (our trading day), then we can get one positive and one negative always and then only collect the profit when they swing, even they were never swing in the profitable way, you still loss less than one way in negative trade.

O.K. Now let's start coding.

### Daily Hedge Expert Advisor

In this part, I will separate it into 5 major parts, that is .

1. The Input Parameters
2. The Daily Trend Signal Function
3. The Trade Function
4. The Trading Process
5. Showing The Hedging Status Function

And now let's begin with the input parameters.

### 1\. Input Parameters

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~External Input Parameters~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

externbool BlockOpening=false;

externbool ShowStatus=true;

externstring Auto\_Lot\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_="\_\_\_\_\_\_\_"; // Always Calculate The Lot Size Automatically

externint PercentMaxRisk=25; // With Max Risk Of 25% by default

externstring How\_Much\_You\_Xpect?\_\_\_\_\_\_\_\_\_\_\_\_ ="\_\_\_\_\_\_\_"; // The Getting Profit Part

externdouble Daily\_Percent\_ROI=7.98; // How many daily %ROI you wish.

externdouble AcceptableLoss\_ROI=3.08; // daily Acceptable loss calculated in ROI scheme.

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Internal Input Parameters~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

string BaseSymbol="GBPJPY";

string H\_Symbol="EURJPY";

int CorPeriod\_1=3; // just for checking that short-term

int CorPeriod\_2=5; //& long-term Correlation are the same concederation level

bool AutoLot=true;

double H\_B\_LotsRatio=1.50; // always hedge those 2 pairs by 1:1.5 ratio

int MMBase=3;

string ExpectCorrelation\_\_\_\_\_\_\_\_\_\_\_\_\_\_= "\_\_\_\_\_\_"; // the concederation level of their correlation.

double Between=1.05;

double And=0.9;

string TDSequential="\_\_\_\_\_\_"; // my easy TD-Sequential signal

int cntFrom=1; // only refer the today signal by yesterday candle

int cntTo=3; // count back to the 3rd candle

bool ClearTradeDaily?=true; // always clear the yesterday hedge

string MISC\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_= "\_\_\_\_\_\_";

int MagicNo=317;

bool PlayAudio=false;

int BSP

,HSP

,gsp

,BOP=-1

,HOP=-1

,up=0

,Hcnt=0

,u=0

,d=0

,day=0

,sent=0

,cntm

,curm

;

double Lot

,BaseOpen

,HOpen

,BaseLots

,HLots

,BUM //Base Used Margin

,GBUM //Get BUM

,HUM //Hedge Used Margin

,GHUM //Get HUM

,TUM //Total Used Margin

,BPt

,HPt

,midpt3;

bool SResult=false,BResult=false;

bool allmeetcor=false,BlockOpen=false,cleared=false;

string candletxt,tdstxt="";

double Min\_Lot;

double Max\_Lot;

double lot\_step;

//+------------------------------------------------------------------+

//\| expert initialization function                                   \|

//+------------------------------------------------------------------+

int init()

{

//----

BSP=MarketInfo(BaseSymbol,MODE\_SPREAD);

HSP=MarketInfo(H\_Symbol,MODE\_SPREAD);

BPt=MarketInfo(BaseSymbol,MODE\_POINT);

HPt=MarketInfo(H\_Symbol,MODE\_POINT);

lot\_step=MarketInfo(BaseSymbol, MODE\_LOTSTEP);

Min\_Lot=MarketInfo(BaseSymbol, MODE\_MINLOT);

if(Min\_Lot<=0)Min\_Lot=1\*lot\_step;

Max\_Lot=MarketInfo(BaseSymbol, MODE\_MAXLOT);

if(BSP>HSP)gsp=HSP;

else gsp=BSP;

//----

return(0);

}

### 2\. Daily Trend Signal Function

//+------------------------------------------------------------------+

//\|TOM DEMARK SEQUENTIAL : Return +Value for UP & -Value for DOWN Sig\|

//+------------------------------------------------------------------+

int DeMark(string sym,int s)

{

int i,pos=36,num=0,num1=0,Rnum,w,m;

for(i=pos; i>=0; i--)

     {

double midPt3=(iClose(sym,0,i+s+cntTo)+iOpen(sym,0,i+s+cntTo))/2;

if(iClose(sym,0,i+s+cntFrom)<midPt3)

        { w++;m=0;num++; num1=0;

         Rnum=-1\*num;

        }

else

if(iClose(sym,0,i+cntFrom)>midPt3)

           { m++;w=0;num1++;num=0;

            Rnum=num1;

           }

else {num1=0;num =0;Rnum=0;}

     }

return(Rnum);

}

//+------------------------------------------------------------------+

//\| CORRELATION : Calculate the correlation of the 2 pairs           \|

//+------------------------------------------------------------------+

double symboldif(string symbol,int shift,int CorPeriod)

{

return(iClose(symbol,1440,shift)-iMA(symbol,1440,CorPeriod,0,MODE\_SMA,PRICE\_CLOSE,shift));

}

//+------------------------------------------------------------------+

//\|                                                                  \|

//+------------------------------------------------------------------+

double powdif(double val)

{

return(MathPow(val,2));

}

//+------------------------------------------------------------------+

//\|                                                                  \|

//+------------------------------------------------------------------+

double u(double val1,double val2)

{

return((val1\*val2));

}

//+------------------------------------------------------------------+

//\|                                                                  \|

//+------------------------------------------------------------------+

double Cor(string base,string hedge,int CorPeriod)

{ double u1=0,l1=0,s1=0;

for(int i=CorPeriod-1 ;i >=0 ;i--)

     {

      u1 +=u(symboldif(base,i,CorPeriod),symboldif(hedge,i,CorPeriod));

      l1 +=powdif(symboldif(base,i,CorPeriod));

      s1 +=powdif(symboldif(hedge,i,CorPeriod));

     }

if(l1\*s1 >0) return(u1/MathSqrt(l1\*s1));

}

### 3\. Trade Function

//+------------------------------------------------------------------+

//\|  TOTAL PROFIT                                                    \|

//+------------------------------------------------------------------+

double TotalCurProfit(int magic)

{

double MyCurrentProfit=0;

for(int cnt=0;cnt < OrdersTotal();cnt++)

     {

OrderSelect(cnt,SELECT\_BY\_POS,MODE\_TRADES);

if (OrderMagicNumber()==magic)

        {

         MyCurrentProfit+= (OrderProfit()+OrderSwap());

        }

     }

return(MyCurrentProfit);

}

//+------------------------------------------------------------------+

//\|  CLOSE HEDGE                                                     \|

//+------------------------------------------------------------------+

bool CloseHedge(int magic)

{

//\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

for(int i=OrdersTotal()-1;i>=0;i--)

     {

if(OrderSelect(i,SELECT\_BY\_POS,MODE\_TRADES) && OrderMagicNumber()==magic)

        {

if(OrderClose(OrderTicket()

         ,OrderLots()

         ,OrderClosePrice()

         ,MarketInfo(OrderSymbol(),MODE\_SPREAD)

         ,CLR\_NONE)

         )SResult=true;

        }

     }

if(SResult\|\|BResult){return(true);if(PlayAudio){PlaySound("ok.wav");}}

elsePrint("CloseHedge Error: ",ErrorDescription(GetLastError()));

//\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

RefreshRates();

}

//+------------------------------------------------------------------+

//\| SEND HEDGE                                                       \|

//+------------------------------------------------------------------+

bool SendH(string symbol,int op,double lots,double price,int sp,string comment,int magic)

{

if(OrderSend(symbol

,op

,lots

,price

,sp

,0

,0

,comment

,magic

,0

,CLR\_NONE)

>0)

     { return(true);

if(PlayAudio)PlaySound("expert.wav");

     }

else {Print(symbol,": "

        ,magic," : "

        ,ErrorDescription(GetLastError()));

return(false);

     }

}

//+------------------------------------------------------------------+

//\|  EXISTING POSITION                                               \|

//+------------------------------------------------------------------+

int ExistPositions(string symbol,int magic)

{

int NumPos=0;

for(int i=0;i<OrdersTotal(); i++)

     {

if(OrderSelect(i,SELECT\_BY\_POS,MODE\_TRADES)

      &&OrderSymbol()==symbol

      &&OrderMagicNumber()==magic

      )

      { NumPos++;}

     }

return(NumPos);

}

//+------------------------------------------------------------------+

//\|  EXISTING OP POSITION                                            \|

//+------------------------------------------------------------------+

int ExistOP(string symbol,int magic)

{

int NumPos=-1;

for(int i=0;i<OrdersTotal(); i++)

     {

if(OrderSelect(i,SELECT\_BY\_POS,MODE\_TRADES)

      &&OrderSymbol()==symbol

      &&OrderMagicNumber()==magic

      )

      { NumPos=OrderType();}

     }

return(NumPos);

}

//+------------------------------------------------------------------+

//\| CLOSE SCRAP                                                      \|

//+------------------------------------------------------------------+

bool CloseScrap(string sym,int op,int magic)

{

//\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

for(int i=OrdersTotal()-1;i>=0;i--)

     {

if(OrderSelect(i,SELECT\_BY\_POS,MODE\_TRADES)

      && OrderMagicNumber()==magic

      &&OrderSymbol()==sym

      &&OrderType()==op)

        {

if(OrderClose(OrderTicket()

         ,OrderLots()

         ,OrderClosePrice()

         ,MarketInfo(OrderSymbol(),MODE\_SPREAD)

         ,CLR\_NONE)

         )BResult=true;

        }

     }

if(SResult\|\|BResult){return(true);if(PlayAudio){PlaySound("ok.wav");}}

elsePrint("CloseScrap Error: ",ErrorDescription(GetLastError()));

//\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

RefreshRates();

}

//+------------------------------------------------------------------+

//\|  Transform OP Value To string                                    \|

//+------------------------------------------------------------------+

string OP2Str(int op)

{

switch(op)

     {

caseOP\_BUY : return("BUY");

caseOP\_SELL: return("SELL");

default : return("~~");

     }

}

//+------------------------------------------------------------------+

//\| ET ORDERTIME OF EXISTING POSITION                                \|

//+------------------------------------------------------------------+

int GetTimeExistOP(string symbol,int magic)

{

int NumPos=-1;

for(int i=0;i<OrdersTotal(); i++)

     {

if(OrderSelect(i,SELECT\_BY\_POS,MODE\_TRADES)

      &&OrderSymbol()==symbol

      &&OrderMagicNumber()==magic

      )

      { NumPos=OrderOpenTime();}

     }

return(NumPos);

}

//+------------------------------------------------------------------+

//\| Translate bool to string                                         \|

//+------------------------------------------------------------------+

string bool2str( bool boolval)

{

if(boolval==true) return("Yes");

if(boolval==false)return("No");

}

//+------------------------------------------------------------------+

//\|AUTO LOT                                                          \|

//+------------------------------------------------------------------+

double Base(int MM)

{

switch(MM)

     {

case1: return(AccountBalance()); break;

case2: return(AccountEquity()); break;

case3: return(AccountFreeMargin());

     }

}

//+------------------------------------------------------------------+

//\|                                                                  \|

//+------------------------------------------------------------------+

double Lots(string symbol, double risk)

{

if(risk > 100) risk=100;

Lot=NormalizeDouble(Base(MMBase)\*(risk/100)/AccountLeverage()/10.0, 2);

Lot=NormalizeDouble(Lot/lot\_step, 0)\*lot\_step;

if(Lot < Min\_Lot) Lot=Min\_Lot;

if(Lot > Max\_Lot) Lot=Max\_Lot;

return(Lot);

}

//+------------------------------------------------------------------+

//\|                                                                  \|

//+------------------------------------------------------------------+

double AutoBLots()

{

double z=1+H\_B\_LotsRatio

,BLot=Lots(BaseSymbol,PercentMaxRisk)/z;

BLot=NormalizeDouble(BLot/lot\_step, 0)\*lot\_step;

if(BLot < Min\_Lot) BLot=Min\_Lot;

if(BLot > Max\_Lot) BLot=Max\_Lot;

return(BLot);

}

//+------------------------------------------------------------------+

//\|                                                                  \|

//+------------------------------------------------------------------+

double AutoHLots()

{

double HLot=AutoBLots()\*H\_B\_LotsRatio;

HLot=NormalizeDouble(HLot/lot\_step, 0)\*lot\_step;

if(HLot < Min\_Lot) HLot=Min\_Lot;

if(HLot > Max\_Lot) HLot=Max\_Lot;

return(HLot);

}

### 4\. Trading Process

//+------------------------------------------------------------------+

//\| expert start function                                            \|

//+------------------------------------------------------------------+

int start()

{

midpt3=(iClose(BaseSymbol,1440,3)+iOpen(BaseSymbol,1440,3))/2;

int hb=FileOpen("B317.csv", FILE\_CSV\|FILE\_READ) // Get the latest Used Margin

,hh=FileOpen("H317.csv", FILE\_CSV\|FILE\_READ); // for calculating the %ROI

if(hb>0)

     {

      GBUM=StrToDouble(FileReadString(hb));

FileClose(hb);

     }

if(hh>0)

     {

      GHUM=StrToDouble(FileReadString(hh));

FileClose(hh);

     }

TUM=GBUM+GHUM;

     {

if(Period()==1440) // only allow to atatch on D1 timeframe

        {

//----

if(day!=Day()) // the new day has come

           { sent=0;cleared=false;

if(ExistPositions(BaseSymbol,MagicNo)==1&&ExistPositions(H\_Symbol,MagicNo)==1)

              { //if the hedge exist.

if(Day()!= TimeDay(GetTimeExistOP(BaseSymbol,MagicNo))

                           && Day()!=TimeDay(GetTimeExistOP(H\_Symbol,MagicNo)))

//the order-time is not the same as today

                 {

if(ClearTradeDaily? && (TotalCurProfit(MagicNo)/TUM)\*100>AcceptableLoss\_ROI)

//allow to clear hedge daily and in acceptable loss

                    {

if(CloseHedge(MagicNo)){cleared=true;BUM=0;HUM=0;} //cleared.

                    }

else//in case the Demark's signal has changed

if((DeMark(BaseSymbol,0)>0&&DeMark(H\_Symbol,0)>0&&DeMark(BaseSymbol,1)<0&&

                                                                                DeMark(H\_Symbol,1)<0)

                     \|\|(DeMark(BaseSymbol,0)<0&&DeMark(H\_Symbol,0)<0&&DeMark(BaseSymbol,1)>0&&

                                                                                DeMark(H\_Symbol,1)>0)

                     )

                       {

if(CloseHedge(MagicNo)){cleared=true;BUM=0;HUM=0;} //cleared.

                       }

                 }

              }

else// in case there was any acident occure during clearing the hedge.

              {

if(ExistPositions(BaseSymbol,MagicNo)>=1&&Day()!= TimeDay(GetTimeExistOP(BaseSymbol,MagicNo)))

                 {

if(ExistOP(BaseSymbol,MagicNo)==OP\_SELL){CloseScrap(BaseSymbol,OP\_SELL,MagicNo);

                        {cleared=true;BUM=0;HUM=0;}}

if(ExistOP(BaseSymbol,MagicNo)==OP\_BUY ){CloseScrap(BaseSymbol,OP\_BUY,MagicNo);

                        {cleared=true;BUM=0;HUM=0;}}

                 }

else

if(ExistPositions(H\_Symbol,MagicNo)>=1&&Day()!= TimeDay(GetTimeExistOP(H\_Symbol,MagicNo)))

                    {

if(ExistOP(H\_Symbol,MagicNo)==OP\_BUY) {CloseScrap(H\_Symbol,OP\_BUY,MagicNo);

                                                                        {cleared=true;BUM=0;HUM=0;}}

if(ExistOP(H\_Symbol,MagicNo)==OP\_SELL){CloseScrap(H\_Symbol,OP\_SELL,MagicNo);

                                                                        {cleared=true;BUM=0;HUM=0;}}

                    }

              }

// block opening if the correlation are not in allowed level.

if(( Cor(BaseSymbol,H\_Symbol,CorPeriod\_1)>Between

            \|\| Cor(BaseSymbol,H\_Symbol,CorPeriod\_1)<And

            )

            \|\|( Cor(BaseSymbol,H\_Symbol,CorPeriod\_2)>Between

            \|\| Cor(BaseSymbol,H\_Symbol,CorPeriod\_2)<And

            )

            )

               BlockOpen=true;

else BlockOpen=false;

            day=Day(); // the new day process finished

           }

else// The intra-day tick comes.

if(TimeCurrent()>Time\[0\]&&ExistPositions(BaseSymbol,MagicNo)+ExistPositions(H\_Symbol,MagicNo)>1)

              { // there are the hedge exist

if((!cleared&&(TotalCurProfit(MagicNo)/TUM)\*100>AcceptableLoss\_ROI&&

Day()!= TimeDay(GetTimeExistOP(BaseSymbol,MagicNo)))

               \|\|((TotalCurProfit(MagicNo)/TUM)\*100>Daily\_Percent\_ROI)

               )

               {CloseHedge(MagicNo);BlockOpen=true;BUM=0;HUM=0;} // closed hedge when rich daily expected ROI.

              }

//~~~~~~~

double BMid=(MarketInfo(BaseSymbol,MODE\_ASK)+MarketInfo(BaseSymbol,MODE\_BID))/2

         ,HMid=(MarketInfo(H\_Symbol,MODE\_ASK)+MarketInfo(H\_Symbol,MODE\_BID))/2

         ,BLS,HLS

         ,BLST,HLST;

         BLS=AutoBLots(); // auto calculate the hedge lots

         HLS=AutoHLots();

//~~~~~~~

           {

if(MathAbs((BMid-iOpen(BaseSymbol,1440,0)))<=BPt\*gsp&&

MathAbs(HMid-iOpen(H\_Symbol,PERIOD\_D1,0))<=HPt\*gsp)

// only open trade when the prices are both near each daily open

              {

int handleB=FileOpen("B"+DoubleToStr(317,0)+".csv", FILE\_CSV\|FILE\_WRITE, ';')

               ,handleH=FileOpen("H"+DoubleToStr(317,0)+".csv", FILE\_CSV\|FILE\_WRITE, ';')

               ;// prepair to write the used margin to the files to recallable

if(DeMark(BaseSymbol,0)>0&&DeMark(H\_Symbol,0)>0

               && iClose(BaseSymbol,1440,1)>midpt3

              )// Demark signaled the UP TREND

                 {

                  up=1;

                  BaseOpen=MarketInfo(BaseSymbol,MODE\_ASK); // Buy Base Symbol

                  HOpen =MarketInfo(H\_Symbol,MODE\_BID); // Sell Hedge Symbol

if(MathAbs((BaseOpen-iOpen(MarketInfo(BaseSymbol,MODE\_BID),1440,0)))<=BPt\*gsp

                  &&MathAbs(MarketInfo(H\_Symbol,MODE\_BID)-iOpen(H\_Symbol,PERIOD\_D1,0))<=HPt\*gsp

                 )// if they both near daily open

                    {

if(!BlockOpen && !BlockOpening) // not both Manual blocking and Correlation blocking

                       {

if(ExistPositions(BaseSymbol,MagicNo)!=0 && ExistOP(BaseSymbol,MagicNo)==OP\_SELL)

                          { // there's one (or more) old base order exist

                           CloseScrap(BaseSymbol,OP\_SELL,MagicNo);BUM=0;HUM=0;

                          }

else

if(ExistPositions(BaseSymbol,MagicNo)==0// no base order exist

                           &&(ExistOP(H\_Symbol,MagicNo)==OP\_SELL

                           \|\| ExistOP(H\_Symbol,MagicNo)==-1

                           )

                           )

                             { BUM=((MarketInfo("EURUSD",MODE\_BID)+MarketInfo("EURUSD",MODE\_ASK))/2)\*BLS\*

                                                                (MarketInfo("EURJPY",MODE\_LOTSIZE)/100);

// calculate base used margin

if(handleB>0)

                                {

FileWrite(handleB,BUM); // write to a file

FileClose(handleB);

                                }

if(SendH(BaseSymbol,OP\_BUY,BLS,BaseOpen,BSP

                              ,"TDS UP : "+DoubleToStr(Cor(BaseSymbol,H\_Symbol,CorPeriod\_1),2)

                              +"\|"+DoubleToStr(Cor(BaseSymbol,H\_Symbol,CorPeriod\_2),2),MagicNo))

                              {sent++;} // sent base order

                              BLST=BLS;

                             }

if(ExistPositions(H\_Symbol,MagicNo)!=0 && ExistOP(H\_Symbol,MagicNo)==OP\_BUY)

                          { // there's one (or more) old hedge order exist

                           CloseScrap(H\_Symbol,OP\_BUY,MagicNo);BUM=0;HUM=0;

                          }

else// no hedge order exist

if(ExistPositions(H\_Symbol,MagicNo)==0

                           &&(ExistOP(BaseSymbol,MagicNo)==OP\_BUY

                           \|\| ExistOP(BaseSymbol,MagicNo)==-1

                           )

                           )

                             { HUM=((MarketInfo("GBPUSD",MODE\_BID)+MarketInfo("GBPUSD",MODE\_ASK))/2)\*HLS\*

                                                                (MarketInfo("GBPJPY",MODE\_LOTSIZE)/100);

// calculate the hedge used margin

if(handleH>0)

                                {

FileWrite(handleH,HUM); // write to a file

FileClose(handleH);

                                }

if(SendH(H\_Symbol,OP\_SELL,HLS,HOpen,HSP

                              ,"TDS UP : "+DoubleToStr(Cor(BaseSymbol,H\_Symbol,CorPeriod\_1),2)

                              +"\|"+DoubleToStr(Cor(BaseSymbol,H\_Symbol,CorPeriod\_2),2),MagicNo))

                              {sent++;} // sent hedge order

                              HLST=HLS;

                             }

                       }

                    }

                 }

//~~~~~~~~~~~~

if(DeMark(BaseSymbol,0)<0&&DeMark(H\_Symbol,0)<0

               && iClose(BaseSymbol,1440,1)<midpt3

              )// same thing but the DOWN signal came out

                 {

                  up=-1;

                  BaseOpen=MarketInfo(BaseSymbol,MODE\_BID);

                  HOpen =MarketInfo(H\_Symbol,MODE\_ASK);

if(MathAbs((BaseOpen-iOpen(MarketInfo(BaseSymbol,MODE\_BID),1440,0)))<=BPt\*gsp

                  &&MathAbs(MarketInfo(H\_Symbol,MODE\_BID)-iOpen(H\_Symbol,PERIOD\_D1,0))<=HPt\*gsp

                  )

                    {

if(!BlockOpen && !BlockOpening)

                       {

if(ExistPositions(BaseSymbol,MagicNo)!=0 && ExistOP(BaseSymbol,MagicNo)==OP\_BUY)

                          {

                           CloseScrap(BaseSymbol,OP\_BUY,MagicNo);BUM=0;HUM=0;

                          }

else

if(ExistPositions(BaseSymbol,MagicNo)==0

                           &&(ExistOP(H\_Symbol,MagicNo)==OP\_BUY

                           \|\| ExistOP(H\_Symbol,MagicNo)==-1

                           )

                           )

                             {BUM=((MarketInfo("EURUSD",MODE\_BID)+MarketInfo("EURUSD",MODE\_ASK))/2)\*BLS\*

                                                                (MarketInfo("EURJPY",MODE\_LOTSIZE)/100);

if(handleB>0)

                                {

FileWrite(handleB,BUM);

FileClose(handleB);

                                }

if(SendH(BaseSymbol,OP\_SELL,BLS,BaseOpen,BSP

                              ,"TDS DN : "+DoubleToStr(Cor(BaseSymbol,H\_Symbol,CorPeriod\_1),2)

                              +"\|"+DoubleToStr(Cor(BaseSymbol,H\_Symbol,CorPeriod\_2),2),MagicNo))

                              {sent++;}

                              BLST=BLS;

                             }

if(ExistPositions(H\_Symbol,MagicNo)!=0 && ExistOP(H\_Symbol,MagicNo)==OP\_SELL)

                          {

                           CloseScrap(H\_Symbol,OP\_SELL,MagicNo);BUM=0;HUM=0;

                          }

else

if(ExistPositions(H\_Symbol,MagicNo)==0

                           &&(ExistOP(BaseSymbol,MagicNo)==OP\_SELL

                           \|\| ExistOP(BaseSymbol,MagicNo)==-1

                           )

                           )

                             {HUM=((MarketInfo("GBPUSD",MODE\_BID)+MarketInfo("GBPUSD",MODE\_ASK))/2)\*HLS\*

                                                                (MarketInfo("GBPJPY",MODE\_LOTSIZE)/100);

if(handleH>0)

                                {

FileWrite(handleH,HUM);

FileClose(handleH);

                                }

if(SendH(H\_Symbol,OP\_BUY,HLS,HOpen,HSP

                              ,"TDS DN : "+DoubleToStr(Cor(BaseSymbol,H\_Symbol,CorPeriod\_1),2)

                              +"\|"+DoubleToStr(Cor(BaseSymbol,H\_Symbol,CorPeriod\_2),2),MagicNo))

                              {sent++;}

                              HLST=HLS;

                             }

                       }

                    }

                 }

//~~~~~~~~~~~~~~~~~

              }

else

if(day==Day() // just check if there still be any scrab orde left by any reason.

               &&TimeCurrent()>Time\[0\] // and clear it or them

               &&ExistPositions(BaseSymbol,MagicNo)+ExistPositions(H\_Symbol,MagicNo)!=0

               )

                 {

if((TotalCurProfit(MagicNo)/TUM)\*100>AcceptableLoss\_ROI)

                    {

if(ExistPositions(BaseSymbol,MagicNo)!=0

                     &&ExistPositions(H\_Symbol,MagicNo)==0

                     )

                       {

if(ExistOP(BaseSymbol,MagicNo)==OP\_SELL)

                        {CloseScrap(BaseSymbol,OP\_SELL,MagicNo);BlockOpen=true;BUM=0;HUM=0;}

else

if(ExistOP(BaseSymbol,MagicNo)==OP\_BUY)

                           {CloseScrap(BaseSymbol,OP\_BUY,MagicNo);BlockOpen=true;BUM=0;HUM=0;}

                       }

if(ExistPositions(BaseSymbol,MagicNo)==0

                     &&ExistPositions(H\_Symbol,MagicNo)!=0

                     )

                       {

if(ExistOP(H\_Symbol,MagicNo)==OP\_BUY)

                        {CloseScrap(H\_Symbol,OP\_BUY,MagicNo);BlockOpen=true;BUM=0;HUM=0;}

else

if(ExistOP(H\_Symbol,MagicNo)==OP\_SELL)

                           {CloseScrap(H\_Symbol,OP\_SELL,MagicNo);BlockOpen=true;BUM=0;HUM=0;}

                       }

                    }

                 }

           }

//----

        }

elseAlert("Please Attatch The EA On D1 Only.");

     }

### 5\. Showing the Hedging Status Function

//~~~~~~~~~~~~~~~~~~~~~~~For Showing Status Section~~~~~~~~~~~~~~~~~~~~~~//

if (DeMark(BaseSymbol,0)>0&&DeMark(H\_Symbol,0)>0) tdstxt="UP";

elseif(DeMark(BaseSymbol,0)<0&&DeMark(H\_Symbol,0)<0) tdstxt="DN";

else tdstxt="~~";

if(curm!=Minute())

     {

      cntm++;

      curm=Minute();

     }

if(cntm<=15)

string timetxt="\\n\\nThis text section will disappear in 15 minutes after this."

      +"\\n"

      +"\\n\\nIn order to run this EA you need to turn off every other EAs."

      +"\\nThis EA was created to be standed alone due to the AccountMargin() function."

      +"\\nRunning other EA at the same time will cause the WRONG calculation of your Daily ROI function."

      +"\\nPLS Strickly follow the instruction above to see the real performance of Daily Hedge Strategy."

      +"\\nThank You ^\_^."

      +"\\n~~~~~~~";

else timetxt="";

if(ShowStatus)

     {

Comment(

"\\n\\nDailyH : Daily GBPJPY ~ EURJPY Hedge."

      ,"\\nBy sexytrade.wordpress.com"

      ,"\\nWith A Static Magic No. of 317"

      ,timetxt

      ,"\\n\\nBlockOpen : "+bool2str(BlockOpen \|\| BlockOpening)

      ,"\\n\\nB/H \[sp\] : "+BaseSymbol+" \["+BSP+"\]"+" / "+H\_Symbol+" \["+HSP+"\]"

      ,"\\nCurOp \[Lots\]: "+OP2Str(ExistOP(BaseSymbol,MagicNo))+" \["+DoubleToStr(BLST,2)+"\]"

      +" ~ "+OP2Str(ExistOP(H\_Symbol,MagicNo))+" \["+DoubleToStr(HLST,2)+"\]"

      ,"\\nCurPF \[Expect\]: $"

      +DoubleToStr(TotalCurProfit(MagicNo),2)

      +" \[$"\
\
      +DoubleToStr(TUM\*(Daily\_Percent\_ROI/100),2)\
\
      +" / ROI: "\
\
      +DoubleToStr(Daily\_Percent\_ROI,2)\
\
      +"\]"

      );

     }

elseComment("");

### Let Me Show Off

My daily hedge system with some live testing results.

![](https://c.mql5.com/2/16/21stsmarn2007.png)

![](https://c.mql5.com/2/16/22vmaru2007.png)

![](https://c.mql5.com/2/16/23kmarh2007.png)

![](https://c.mql5.com/2/16/29tmard2007.png)

![](https://c.mql5.com/2/16/30imars2007.png)

![](https://c.mql5.com/2/16/4uapre2007.png)

![](https://c.mql5.com/2/16/5gaprf2007.png)

![](https://c.mql5.com/2/16/10xaprm2007pfull.png)

### Conclusion

From
my 1 month report and the back testing result that show a possibility
to make money in forex using this daily hedge concept, I think this system can help at least one idea of you to light up (" _Ping Pong!!!, Hey! What about doing this instead ?_") and generate a money maker system that is more qualify. Or maybe my style of coding can at least help one newbie to learn and practical for his/her traditional coding style.
I strongly hope that my article is useful for all readers, even the
system may fail later, and you all like it. Now I will follow my plan of testing it for at least 6 months and I will post the result if possible. GOOD LUCK.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1532.zip "Download all attachments in the single ZIP archive")

[DailyH.mq4](https://www.mql5.com/en/articles/download/1532/DailyH.mq4 "Download DailyH.mq4")(27.5 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Break Through The Strategy Tester Limit On Testing Hedge EA](https://www.mql5.com/en/articles/1493)
- [Sending Trade Signal Via RSS Feed](https://www.mql5.com/en/articles/1480)
- [The Basic of Coding A Hedge Expert Advisor](https://www.mql5.com/en/articles/1479)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39444)**
(25)


![zhen ding](https://c.mql5.com/avatar/avatar_na2.png)

**[zhen ding](https://www.mql5.com/en/users/charvo)**
\|
11 Sep 2008 at 21:04

how could we find out what exactly H317.csv and B317.csv are?

i didn't see the author mentioned it elsewhere.

![Ronaldo](https://c.mql5.com/avatar/avatar_na2.png)

**[Ronaldo](https://www.mql5.com/en/users/ronaldosim)**
\|
14 Sep 2008 at 04:10

**charvo:**

how could we find out what exactly H317.csv and B317.csv are?

i didn't see the author mentioned it elsewhere.

The csv files record the used margin for each of the hedged pairs; the system will create the files.

BlockOpening=false allows orders to be placed, if true orders sending will be blocked.

Anyway, this EA is not useable as it is: errors in firing orders, acceptable loss is computed wrongly,

too many [files opened](https://www.mql5.com/en/docs/files/fileopen "MQL5 documentation: FileOpen function") error, and sometimes divide by zero error

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
8 Jun 2009 at 12:22

I don't think it's a perfect idea though the thought's sound good.

As the author said "I decided to mark the EUR/JPY as my base pair and hedge GBP/JPY",no matter you buy/sell EUR/JPY and sell/buy GBP/JPY, it's the same effect as you buy or sell EUR/GBR in fact.

![sathish kumar](https://c.mql5.com/avatar/2015/12/56695935-C7CC.png)

**[sathish kumar](https://www.mql5.com/en/users/sathudx)**
\|
11 Nov 2015 at 13:17

better use [buy audjpy](https://www.mql5.com/en/quotes/currencies/audjpy "AUDJPY chart: techical analysis") short cadjpy,

we can get swap too.

![Rhona Rankin](https://c.mql5.com/avatar/avatar_na2.png)

**[Rhona Rankin](https://www.mql5.com/en/users/rhonamary)**
\|
22 Jul 2022 at 18:23

Thanks Chayutra. I updated the code to be able to read and understand it better, and updated the used [margin calculation](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_calc_mode "MQL5 documentation: Symbol properties") to account for leverage.

Looking forward to testing this now.

![Method of Determining Errors in Code by Commenting](https://c.mql5.com/2/16/697_3.gif)[Method of Determining Errors in Code by Commenting](https://www.mql5.com/en/articles/1547)

The article describes a method of searching the errors in the MQL4 code that is based on commenting. This method is found to be a useful one in case of problems occuring during the compilation caused by the errors in a reasonably large code.

![All about Automated Trading Championship: Additional Materials of Championships 2006-2007](https://c.mql5.com/2/16/704_13.gif)[All about Automated Trading Championship: Additional Materials of Championships 2006-2007](https://www.mql5.com/en/articles/1554)

We present to you a selection of these materials that are divided into several parts. The present one contains additional materials about automated trading, Expert Advisors development, etc.

![Grouped File Operations](https://c.mql5.com/2/16/677_43.gif)[Grouped File Operations](https://www.mql5.com/en/articles/1543)

It is sometimes necessary to perform identical operations with a group of files. If you have a list of files included into a group, then it is no problem. However, if you need to make this list yourself, then a question arises: "How can I do this?" The article proposes doing this using functions FindFirstFile() and FindNextFile() included in kernel32.dll.

![File Operations via WinAPI](https://c.mql5.com/2/16/668_76.gif)[File Operations via WinAPI](https://www.mql5.com/en/articles/1540)

Environment MQL4 is based on the conception of safe "sandbox": reading and saving files using the language are allowed in some predefined folders only. This protects the user of MetaTrader 4 from the potential danger of damaging important data on the HDD. However, it is sometimes necessary to leave that safe area. This article is devoted to the problem of how to do it easily and correctly.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=itetjbxaalwqgxwghfzvrhkigtzozsrj&ssn=1769252209950948983&ssn_dr=0&ssn_sr=0&fv_date=1769252209&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1532&back_ref=https%3A%2F%2Fwww.google.com%2F&title=What%20about%20Hedging%20Daily%3F%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925220992449364&fz_uniq=5083231878363748209&sv=2552)

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