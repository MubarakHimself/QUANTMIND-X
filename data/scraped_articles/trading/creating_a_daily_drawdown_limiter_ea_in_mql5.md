---
title: Creating a Daily Drawdown Limiter EA in MQL5
url: https://www.mql5.com/en/articles/15199
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T18:00:49.689232
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=dseljwoqqhxiolwumwrabzmswlawqiur&ssn=1769094048961072465&ssn_dr=0&ssn_sr=0&fv_date=1769094048&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15199&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20a%20Daily%20Drawdown%20Limiter%20EA%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909404810449230&fz_uniq=5049561026980064671&sv=2552)

MetaTrader 5 / Trading


### Introduction

In this article, we are creating a Daily Drawdown Limiter forex trading Expert Advisor (EA) in MetaQuotes Language (MQL5) for MetaTrader 5. The goal of this EA is to establish a daily withdrawal cap for trading accounts. The EA analyzes factors such as total bar count, starting balance, time of day, and daily balance and checks if trading occurs based on specific conditions. It also gathers information like balance, capital, and funds from the MetaTrader platform. This EA is specifically designed to operate with the MetaTrader trading platform and necessitates a trading account for proper functioning.

This journey will cover the following topics:

1. Drawdown Limiter explanation
2. Creation of EA in MQL5
3. Conclusion

### Drawdown limiter explanation

A drawdown limiter is a tool used in trading and investment for managing risk by limiting potential losses during a drawdown period. A drawdown period occurs when the value of an asset or a portfolio drops due to market volatility or economic conditions. During this period, the drawdown limiter helps to safeguard investors from substantial losses by automatically selling all or part of the investment if the value goes below a set level. This tool aims to reduce possible losses and safeguard the investor's capital. Managed futures accounts and other investment vehicles commonly utilize drawdown limiters to control risk and shield against significant market declines.

Traders suffer from controlling their drawdown when trading funded accounts. A daily drawdown limiter is meant for them. The Prop firms usually set a rule called 'Trader Daily Drawdown', and if it is not respected, the trader is disqualified. The Drawdown Limiter helps traders in:

1. Tracking the account drawdown
2. Alert the trader when he or she is taking  high-risk trades
3. Tracking the daily trader drawdown
4. Prevent the trader from overtrading by limiting open positions

### Creation of EA in MQL5

The Expert Advisor's essential function is to monitor all the activities on the account, whether it was a manual or automated trade by another Expert Advisor. The trader needs to add a chart, and it will take control. The traders will notice 'Traffic Lights'on their screen. Its 'Traffic Lights' feature will inform the traders about the four keys mentioned above in a simple graphical manner. The EA comments can be hidden and customizable to match their preferred colors and fonts. In one click, the details page can simply be hidden to gain some space on the charts. The position and style of the traffic lights can be super customizable to match their chart style.

We need to open trading positions. The easiest way to open positions is to include a trading instance, typically achieved via the inclusion of another file that is dedicated to open positions. We use the include directive to include the trade library, which contains functions for trading operations. First, we use the angle brackets to signify that the file we want to include is contained in the include folder to provide a trade folder, followed by a normal slash or backslash, and then the target file name, in this case, 'Trade.mqh'. cTrade is a class for handling trade operations, and obj-trade is an instance for this class, typically a pointer object created from the cTrade class to provide access to the member variables of the class.

```
#include <Trade/Trade.mqh>
CTrade obj-Trade;
```

Afterward, we need some control logic to generate signals to open positions. In our case, the function OnTick() is checking if the variable isTradeAllowed is true. In case it is, the function checkDailyProfit() is called, suggesting that the purpose of the OnTick() function is to check the daily profit and potentially allow or disallow trades based on the check. The _bars_ variable keeps track of the total number of bars on the chart, ensuring the trading logic executes only once per new bar, preventing multiple executions within a single bar. Together, these variables enable the Expert Advisor to generate trading signals based on values while maintaining proper execution timing. Since the function takes no parameters let us proceed  to the actions it performs as follows:

- It defines the variable  total\_day\_Profit and initializes it to 0.
- Furthermore, it receives the current time and converts it to a string using the TimeToString function stored in the date variable.
- Likewise, it calculates the initial hour of the day by adding 1 to the beginning of the day and saves it in a variable.
- Checks if the current time (daytime) is less than the time. If so, it sets the dayTime value and calculates the current balance using the Acc\_B function,  is stored in the dayBalance variable.
- Selects the historical data for the day using the History select function, with the beginning and end times set to the beginning and end of the day.
- Calculates the total number of deals for the day using the HistoryDealsTotal function and stores it in the TotalDeals variable.
- Not only that, but it goes through each transaction in the history and checks if the transaction entry type is DEAL\_ENTRY\_OUT, which means it is a closing transaction. If so, it calculates the trading profit by adding DEAL\_PROFIT, DEAL\_COMMISSION DEAL\_SWAP values and adds it to the total\_day\_profit variable.
- It calculates the opening balance of the day by subtracting the total\_day\_Profit from the current account balance using the AccountInfoDouble function with the ACCOUNT-BALANCE parameter.

The function returns the calculated opening balance as a Double value.

```
int totalBars = 0;
double initialBalance = 0;
datetime dayTime = 0;
double dayBalance = 0;
bool isTradeAllowed = true;
```

Next, we move on to the definition of functions. The function seems to be related to account information and returns different values based on the function name. The functions Acc\_B(), Acc\_E(), and  Acc\_S() are used to retrieve information about the account balance, equity, and currency respectively. These functions are used to monitor the financial status of the account.

```
double Acc_B(){return AccountInfoDouble(ACCOUNT_BALANCE);}
double Acc_E(){return AccountInfoDouble(ACCOUNT_EQUITY);}
string Acc_S(){return AccountInfoString(ACCOUNT_CURRENCY);}
```

The full code of opening positions is as follows:

```
#include <Trade/Trade.mqh>
CTrade obj-Trade;

int totalBars = 0;
double initialBalance = 0;
datetime dayTime = 0;
double dayBalance = 0;
bool isTradeAllowed=true;

double Acc_B(){return AccountInfoDouble(ACCOUNT_BALANCE);}
double Acc_E(){return AccountInfoDouble(ACCOUNT_EQUITY);}
string Acc_S(){return AccountInfoString(ACCOUNT_CURRENCY);}
```

The Onlnit event handler is called whenever the Expert Advisor is initialized. It is the instance that we need to initialize the indicator and create text to display the account's initial balance data for further analysis. To initialize the indicator, we use the built-in function to return its createText by providing the correct parameters. The text object is positioned at certain coordinates and uses colors with a font size. Here's a breakdown of what we achieve from this function:

1. It looks for the initial balance of the account with the Acc\_B()function and stores it in the variable 'initialBalance'.
2. This will create a text box with the text '\* PROP FIRM PROGRESS DASHBOARD \*' at position (30,30 on the screen, with a font size of  13 and a light blue color (clrAqual).
3. This will create several text boxes to display different messages and information to the user. These text boxes are positioned at different locations on the screen and have different font sizes and colors.

Here the main purpose is to create a user interface that displays various pieces of information related to account management and trading. Text boxes are used to display account information, messages, and other relevant data to the user.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {

   initialBalance = Acc_B();

   createText("0","***DAILY DRAWDOWN LIMITER ***",30,30,clrAqua,13);
   createText("00","______________________________________",30,30,clrAqua,13);
   createText("1","DrawDown Limiter is Active.",70,50,clrWhite,11);
   createText("2","Counters will be Reset on Next Day Start.",70,65,clrWhite,10);
   createText("3","From: ",70,80,clrWhite,10);
   createText("4",'Time Here',120,80,clrGray,10);
   createText("5","To: ",70,95,clrWhite,10);
   createText("6",'Time Here',120,95,clrGray,10);
   createText("7",'Current: ',70,110,clrWhite,10);
   createText("8",'Time Here',120,110,clrGray,10);

   createText("9",'ACCOUNT DRAWDOWN ============',70,130,clrPeru,11);
   createText("10",'Account Initial Balance: ',70,145,clrWhite,10);
   createText("11",DoubleToString(initialBalance,2)+" "+Acc_S(),250,145,clrWhite,10);
   createText("12",'Torelated DrawDown: ',70,160,clrWhite,10);
   createText("13","12.00 %",250,160,clrAqua,10);
   createText("14",'Current Account Equity: ',70,175,clrWhite,10);
   createText("15",DoubleToString(Acc_E(),2)+" "+Acc_S(),250,175,clrWhite,10);
   createText("16",'Current Balance Variation: ',70,190,clrWhite,10);
   createText("17",DoubleToString((Acc_E()-Acc_B())/Acc_B()*100,2)+" %",250,190,clrGray,10);

   createText("18",'DAILY DRAWDOWN ================',70,210,clrPeru,11);
   createText("19",'Starting Balance: ',70,225,clrWhite,10);
   createText("20",DoubleToString(Acc_B(),2)+" "+Acc_S(),270,225,clrWhite,10);
   createText("21",'DrawDowm Maximum Threshold: ',70,240,clrWhite,10);
   createText("22",'5.00 %"+" "+Acc_S(),270,240,clrAqua,10);
   createText("23",'DrawDown Maximum Amount: ',70,255,clrWhite,10);
   createText("24",'-"+DoubleToString(Acc_B()*5/100,2)+' "+Acc_S(),270,255,clrYellow,10);
   createText("25",'Current Closed Daily Profit: ',70,270,clrWhite,10);
   createText("26",'0.00"+" "+Acc_S(),270,270,clrGray,10);
   createText("27",'Current DrawDown Percent: ',70,285,clrWhite,10);
   createText("28",'0.00"+" %",270,285,clrGray,10);

   createText("29",'>>> Initializing The Program, Get Ready To Trade.",70,300,clrYellow,10);

   return(INIT_SUCCEEDED);
}
```

Here the OnTick function is called every time there is a new tick for the symbol the EA is attached to. For the checkDailyProfit function, one needs to ensure that it is implemented correctly. The isTradeAllowed is a boolean variable that controls whether trading is allowed. In case isTradeAllowed is false, it returns immediately, and no further code is executed within the OnTick function.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){

   checkDailyProfit();

   if (!isTradeAllowed) return;

```

We just define instances of the breakdown swing. This requires to be done on every tick, so we do it without restrictions. We first declare the Ask and Bid prices that we will use to open the positions once the respective conditions are met. Note that this needs also to be done on every tick so that we get the latest price quotes. Here, we declare the double data type variables for storing recent prices and normalize them to the digits of the symbol currency by rounding the floating-point number to maintain accuracy.

```
double ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
double bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);
```

After defining the instances of the breakdown swing, we now graduate to defining a function named 'iBars' which in our case takes two parameters: '\_Symbol' and '\_Period. The 'iBars' returns an integer value known as 'bars'. We then check if the variable 'totalBars' is equal to the value returned by the 'iBars' function. In case they are equal, the function returns without doing anything else . In case they are not equal, the value of 'totalBars' is set to the value of 'bars' returned by the 'iBars' function.

```
   int bars = iBars(_Symbol,_Period);
   if (totalBars == bars) return;
   totalBars = bars;
```

We now continue from defining the "iBars" function; here we check if the results of calling the 'positionsTotal()' function are greater than 1. In case it is, the function returns without doing anything else. In case it is not, the code goes to the next line. The line 'int number = MathRand()%' appears to be incomplete because there are no closing parenthesis or semicolons. Assuming the goal is to generate a random integer, the line would be completed as follows: " int number = MathRand()% totalBars;' This line generates a random between 0 and the value of 'totalBars' (inclusive), and assigns it to the variable  'number'.

```
 if (PositionsTotal() > 1) return;
   int number = MathRand()%
```

The full code of defining functions is as below:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){

   checkDailyProfit();

   if (!isTradeAllowed) return;

   double ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
   double bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);

   int bars = iBars(_Symbol,_Period);
   if (totalBars == bars) return;
   totalBars = bars;

   if (PositionsTotal() > 1) return;
   int number = MathRand()%
```

Here's the breakdown of the  key components of the article functions such as  placing trades, creating text labels on the chart, and checking daily profit:

1\. Trade Execution

In this component, we are examining the value of a variable named 'number' and taking different actions based on its value.

If the 'number' is 0, the code triggers a method named 'Buy' on an object identified as 'obj-Trade'. This method requires five parameters: the first one is 0.1, the second is variable '\_Symbol', the third is variable 'ask', the fourth is the 'ask' value minus 70 times the variable '\_Point', and the fifth is the 'ask' value plus 70 times the same variable '\_Point'. This indicates that the code is attempting to purchase an asset for a price slightly lower than the current ask price, considering the bid-ask spread.

If the 'number' is 1, the code executes a method named 'Sell' on the same object 'obj-Trade'. This method also takes five parameters: the first one is 0.1, the second is variable '\_Symbol', the third is the variable 'bid', the fourth is the 'bid' value plus 70 times the variable '\_Point', and the fifth is the 'bid' value minus 70 times the same variable '\_Point'. This implies that the code is trying to sell an asset for a price slightly higher than the current bid price, also considering the bid-ask price spread. This evaluates the value of a variable known as 'number' and performs different actions based on its value.

```
if (number == 0){
      obj_Trade.Buy(0.1,_Symbol,ask,ask-70*_Point,ask+70*_Point);
   }
   else if (number == 1){
      obj_Trade.Sell(0.1,_Symbol,bid,bid+70*_Point,bid-70*_Point);
   }
```

2\. Text Creation

In our prior code, there exists a function named createText which is responsible for creating a label object on a chart. To execute this function, certain parameters like object name (objName), text content (text), x and y coordinates for label placement (x and y), text color (clrTxt), and font size (font size) are required. Utilizing these inputs the function creates a label object on the chart, customizes its features, and updates the chart. Furthermore, a boolean value is returned by the function to signify the successful or unsuccessful creation of the label.

```

bool createText(string objName,string text,int x, int y,color clrTxt,int fontSize){
   ResetLastError();
   if (!ObjectCreate(0,objName,OBJ_LABEL,0,0,0)){
      Print(__FUNCTION__,": failed to create the Label! Error Code = ",GetLastError());
      return (false);
   }
   ObjectSetInteger(0,objName,OBJPROP_XDISTANCE,x);
   ObjectSetInteger(0,objName,OBJPROP_YDISTANCE,y);
   ObjectSetInteger(0,objName,OBJPROP_CORNER,CORNER_LEFT_UPPER);
   ObjectSetString(0,objName,OBJPROP_TEXT,text);
   ObjectSetInteger(0,objName,OBJPROP_COLOR,clrTxt);
   ObjectSetInteger(0,objName,OBJPROP_FONTSIZE,fontSize);

   ChartRedraw(0);
   return (true);
}
```

3\. Daily Profit Checking

In MQL function called checkDailyProfit calculates the total profit for a given day. The function takes no parameters and performs the following steps:

- Defines the variable total\_day\_profit and initializes it to 0.
- Gets the current time and converts it to a string using the TimeToString function, which is stored in the date variable.
- Calculate the initial hour of the day by adding 1 to the beginning of the day and saving it in a variable.
- Check if the daily time is less than the current time. If yes, set dayTime and calculate the current balance using the Acc\_B function which is stored in the dayBalance variable.
- Selects the historical data for the day using the HistorySelect function, with the beginning and end times set to the beginning and end of the day.
- Calculates the total number of deals for the day using the HistoryDealsTotal function and stores it in the TotalDeals variable.
- It goes through it transaction in the history and checks if the transaction entry type is DEAL\_ENTRY\_OUT, this means it is a closing transaction. If yes, it calculates the trading profit by adding DEAL\_PROFIT, DEAL\_COMMISSION, and DEAL\_SWAP values and adds to the total\_day\_profit variable.
- Calculate the opening balance of the day by subtracting the total\_day\_profit from the current account balance using the AccountInfoDouble function with the ACCOUNT\_BALANCE parameter.

The function returns the calculated opening balance as a Double value.

```
void checkDailyProfit(){

   double total_day_Profit = 0;
   datetime end = TimeCurrent();
   string sdate = TimeToString(TimeCurrent(),TIME_DATE);
   datetime start = StringToTime(sdate);
   datetime to = start + (1*24*60*60);

   if (dayTime < to){
      dayTime = to;
      dayBalance = Acc_B();
   }

   HistorySelect(start,end);
   int TotalDeals = HistoryDealsTotal();
   for (int i=0; i<TotalDeals; i++){
      ulong Ticket = HistoryDealGetTicket(i);
      if (HistoryDealGetInteger(Ticket,DEAL_ENTRY)==DEAL_ENTRY_OUT){
         double Latest_Day_Profit = (HistoryDealGetDouble(Ticket,DEAL_PROFIT)
                               +HistoryDealGetDouble(Ticket,DEAL_COMMISSION)
                               +HistoryDealGetDouble(Ticket,DEAL_SWAP));
         total_day_Profit += Latest_Day_Profit;
      }
   }
   double startingBalance = 0;
   startingBalance = AccountInfoDouble(ACCOUNT_BALANCE) - total_day_Profit;
   double daily_profit_or_drawdown = NormalizeDouble((total_day_Profit*100/startingBalance),2);
   string daily_profit_in_Text_Format = "";
   daily_profit_in_Text_Format = DoubleToString(daily_profit_or_drawdown,2)+" %";

   //Print(total_day_Profit, " >>> ",daily_profit_in_Text_Format);

   createText("4",TimeToString(start),120,80,clrYellow,10);
   createText("6",TimeToString(to),120,95,clrYellow,10);
   createText("8",TimeToString(end),120,110,clrWhite,10);

   if (Acc_E() > initialBalance){
      createText("15",DoubleToString(Acc_E(),2)+" "+Acc_S(),250,175,clrLime,10);
      createText("17",DoubleToString((Acc_E()-initialBalance)/initialBalance*100,2)+" %",250,190,clrLime,10);
   }
   else if (Acc_E() < initialBalance){
      createText("15",DoubleToString(Acc_E(),2)+" "+Acc_S(),250,175,clrRed,10);
      createText("17",DoubleToString((Acc_E()-initialBalance)/initialBalance*100,2)+" %",250,190,clrRed,10);
   }
   if (Acc_E() == initialBalance){
      createText("15",DoubleToString(Acc_E(),2)+" "+Acc_S(),250,175,clrWhite,10);
      createText("17",DoubleToString((Acc_E()-initialBalance)/initialBalance*100,2)+" %",250,190,clrWhite,10);
   }

   createText("20",DoubleToString(dayBalance,2)+" "+Acc_S(),270,225,clrWhite,10);
   createText("24","-"+DoubleToString(dayBalance*5/100,2)+" "+Acc_S(),270,255,clrYellow,10);

   if (Acc_B() > dayBalance){
      createText("26",DoubleToString(total_day_Profit,2)+" "+Acc_S(),270,270,clrLime,10);
      createText("28",daily_profit_in_Text_Format,270,285,clrLime,10);
   }
   else if (Acc_B() < dayBalance){
      createText("26",DoubleToString(total_day_Profit,2)+" "+Acc_S(),270,270,clrRed,10);
      createText("28",daily_profit_in_Text_Format,270,285,clrRed,10);
   }
   else if (Acc_B() == dayBalance){
      createText("26",DoubleToString(total_day_Profit,2)+" "+Acc_S(),270,270,clrWhite,10);
      createText("28",daily_profit_in_Text_Format,270,285,clrWhite,10);
   }

   if (daily_profit_or_drawdown <= -5.00 || ((Acc_E()-initialBalance)/initialBalance*100) < -12.00){
      createText("29",">>> Maximum Threshold Hit, Can't Trade.",70,300,clrRed,10);
      isTradeAllowed = false;
   }
   else {
      createText("29",">>> Maximum Threshold Not Hit, Can Trade.",70,300,clrL…
```

The full code of the key components of the article functions is as below:

```
2;
   if (number == 0){
      obj_Trade.Buy(0.1,_Symbol,ask,ask-70*_Point,ask+70*_Point);
   }
   else if (number == 1){
      obj_Trade.Sell(0.1,_Symbol,bid,bid+70*_Point,bid-70*_Point);
   }
}
//+------------------------------------------------------------------+
bool createText(string objName,string text,int x, int y,color clrTxt,int fontSize){
   ResetLastError();
   if (!ObjectCreate(0,objName,OBJ-LABEL,0,0,0)){
      Print(__FUNCTION__,": failed to create the Label! Error Code = ",GetLastError());
      return (false);
   }
   ObjectSetInteger(0,objName,OBJPROP-XDISTANCE,x);
   ObjectSetInteger(0,objName,OBJPROP-YDISTANCE,y);
   ObjectSetInteger(0,objName,OBJPROP-CORNER,CORNER_LEFT_UPPER);
   ObjectSetString(0,objName,OBJPROP-TEXT,text);
   ObjectSetInteger(0,objName,OBJPROP-COLOR,clrTxt);
   ObjectSetInteger(0,objName,OBJPROP-FONTSIZE,fontSize);

   ChartRedraw(0);
   return (true);
}

void checkDailyProfit(){

   double total_day_Profit = 0;
   datetime end = TimeCurrent();
   string sdate = TimeToString(TimeCurrent(),TIME_DATE);
   datetime start = StringToTime(sdate);
   datetime to = start + (1*24*60*60);

   if (dayTime < to){
      dayTime = to;
      dayBalance = Acc_B();
   }

   HistorySelect(start,end);
   int TotalDeals = HistoryDealsTotal();
   for (int i=0; i<TotalDeals; i++){
      ulong Ticket = HistoryDealGetTicket(i);
      if (HistoryDealGetInteger(Ticket,DEAL_ENTRY)==DEAL_ENTRY_OUT){
         double Latest_Day_Profit = (HistoryDealGetDouble(Ticket,DEAL_PROFIT)
                               +HistoryDealGetDouble(Ticket,DEAL_COMMISSION)
                               +HistoryDealGetDouble(Ticket,DEAL_SWAP));
         total_day_Profit += Latest_Day_Profit;
      }
   }
   double startingBalance = 0;
   startingBalance = AccountInfoDouble(ACCOUNT-BALANCE) - total_day_Profit;
   double daily_profit_or_drawdown = NormalizeDouble((total_day_Profit*100/startingBalance),2);
   string daily_profit_in_Text_Format = "";
   daily_profit_in_Text_Format = DoubleToString(daily_profit_or_drawdown,2)+" %";

   //Print(total_day_Profit, " >>> ",daily_profit_in_Text_Format);

   createText("4",TimeToString(start),120,80,clrYellow,10);
   createText("6",TimeToString(to),120,95,clrYellow,10);
   createText("8",TimeToString(end),120,110,clrWhite,10);

   if (Acc_E() > initialBalance){
      createText("15",DoubleToString(Acc_E(),2)+" "+Acc_S(),250,175,clrLime,10);
      createText("17",DoubleToString((Acc_E()-initialBalance)/initialBalance*100,2)+" %",250,190,clrLime,10);
   }
   else if (Acc_E() < initialBalance){
      createText("15",DoubleToString(Acc_E(),2)+" "+Acc_S(),250,175,clrRed,10);
      createText("17",DoubleToString((Acc_E()-initialBalance)/initialBalance*100,2)+" %",250,190,clrRed,10);
   }
   if (Acc_E() == initialBalance){
      createText("15",DoubleToString(Acc_E(),2)+" "+Acc_S(),250,175,clrWhite,10);
      createText("17",DoubleToString((Acc_E()-initialBalance)/initialBalance*100,2)+" %",250,190,clrWhite,10);
   }

   createText("20",DoubleToString(dayBalance,2)+" "+Acc_S(),270,225,clrWhite,10);
   createText("24","-"+DoubleToString(dayBalance*5/100,2)+" "+Acc_S(),270,255,clrYellow,10);

   if (Acc_B() > dayBalance){
      createText("26",DoubleToString(total_day_Profit,2)+" "+Acc_S(),270,270,clrLime,10);
      createText("28",daily_profit_in_Text_Format,270,285,clrLime,10);
   }
   else if (Acc_B() < dayBalance){
      createText("26",DoubleToString(total_day_Profit,2)+" "+Acc_S(),270,270,clrRed,10);
      createText("28",daily_profit_in_Text_Format,270,285,clrRed,10);
   }
   else if (Acc_B() == dayBalance){
      createText("26",DoubleToString(total_day_Profit,2)+" "+Acc_S(),270,270,clrWhite,10);
      createText("28",daily_profit_in_Text_Format,270,285,clrWhite,10);
   }

   if (daily_profit_or_drawdown <= -5.00 || ((Acc_E()-initialBalance)/initialBalance*100) < -12.00){
      createText("29",">>> Maximum Threshold Hit, Can't Trade.",70,300,clrRed,10);
      isTradeAllowed = false;
   }
   else {
      createText("29",">>> Maximum Threshold Not Hit, Can Trade.",70,300,clrRed);
```

The following is what we get.

Example of a far-threshold logic.

![FAR THRESHHOLD](https://c.mql5.com/2/111/Screenshot_2024-07-09_163502__2.png)

Example of a near-threshhold logic.

![NEAR THRESHOLD](https://c.mql5.com/2/111/Screenshot_2024-07-09_163532__2.png)

Example of a thresh-hold hit logic:

![THRESHOLD HIT](https://c.mql5.com/2/111/Screenshot_2024-07-09_164001__2.png)

The full code for creating a drawdown limiter is as below:

```
//+------------------------------------------------------------------+
//|                                       Daily Drawdown Limiter.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <Trade/Trade.mqh>
CTrade obj_Trade;

int totalBars = 0;
double initialBalance = 0;
double dayBalance = 0;
datetime dayTime = 0;
bool isTradeAllowed = true;

// Functions to get account balance, equity, and currency
double Acc_B() {return AccountInfoDouble(ACCOUNT_BALANCE);}
double Acc_E() {return AccountInfoDouble(ACCOUNT_EQUITY);}
string Acc_S() {return AccountInfoString(ACCOUNT_CURRENCY);}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Initialize initial balance
   initialBalance = Acc_B();

   // Create dashboard texts
   createText("0","*** Daily Drawdown Limiter ***",30,30,clrBlack,13);
   createText("00","______________________________________",30,30,clrBlack,13);
   createText("1","DrawDown Limiter is Active.",70,50,clrBlack,11);
   createText("2","Counters will be reset on Next Day Start.",70,65,clrBlack,10);
   createText("3","From: ",70,80,clrBlack,10);
   createText("4","Time Here",120,80,clrGray,10);
   createText("5","To: ",70,95,clrBlack,10);
   createText("6","Time Here",120,95,clrGray,10);
   createText("7","Current: ",70,110,clrBlack,10);
   createText("8","Time Here",120,110,clrGray,10);

   createText("9","ACCOUNT DRAWDOWN ============",70,130,clrPeru,11);
   createText("10","Account Initial Balance: ",70,145,clrBlack,10);
   createText("11",DoubleToString(initialBalance,2)+" "+Acc_S(),250,145,clrBlack,10);
   createText("12","Tolerated DrawDown: ",70,160,clrBlack,10);
   createText("13","12.00 %",250,160,clrBlack,10);
   createText("14","Current Account Equity: ",70,175,clrBlack,10);
   createText("15",DoubleToString(Acc_E(),2)+" "+Acc_S(),250,175,clrBlack,10);
   createText("16","Current Balance Variation: ",70,190,clrBlack,10);
   createText("17",DoubleToString((Acc_E()-Acc_B())/Acc_B()*100,2)+" %",250,190,clrGray,10);

   createText("18","DAILY DRAWDOWN ================",70,210,clrPeru,11);
   createText("19","Starting Balance: ",70,225,clrBlack,10);
   createText("20",DoubleToString(Acc_B(),2)+" "+Acc_S(),270,225,clrBlack,10);
   createText("21","DrawDown Maximum Threshold: ",70,240,clrBlack,10);
   createText("22","5.00 %",270,240,clrBlack,10);
   createText("23","DrawDown Maximum Amount: ",70,255,clrBlack,10);
   createText("24","-"+DoubleToString((Acc_B()*5/100),2)+" "+Acc_S(),270,255,clrBlue,10);
   createText("25","Current Closed Daily Profit: ",70,270,clrBlack,10);
   createText("26","0.00"+" "+Acc_S(),270,270,clrGray,10);
   createText("27","Current DrawDown Percent: ",70,285,clrBlack,10);
   createText("28","0.00 %",270,285,clrGray,10);
   createText("29",">>> Initializing The Program, Get Ready To Trade.",70,300,clrBlue,10);

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   // Deinitialization code here (if needed)
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Check daily profit and drawdown
   checkDailyProfit();

   // If trading is not allowed, exit function
   if (!isTradeAllowed) return;

   // Get current ask and bid prices
   double ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
   double bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);

   // Check for new bar
   int bars = iBars(_Symbol,_Period);
   if (totalBars == bars) return;
   totalBars = bars;

   // If more than one position, exit function
   if (PositionsTotal() > 1) return;

   // Random trade decision
   int number = MathRand()%2;
   Print(number);

   if (number == 0){
      obj_Trade.Buy(1,_Symbol,ask,ask-70*_Point,ask+70*_Point);
   }
   else if (number == 1){
      obj_Trade.Sell(1,_Symbol,bid,bid+70*_Point,bid-70*_Point);
   }
  }
//+------------------------------------------------------------------+
//| Check daily profit and drawdown                                  |
//+------------------------------------------------------------------+
void checkDailyProfit() {

   double total_day_Profit = 0.0;
   datetime end = TimeCurrent();
   string sdate = TimeToString (TimeCurrent(), TIME_DATE);
   datetime start = StringToTime(sdate);
   datetime to = start + (1*24*60*60);

   // Reset daily balance and time at start of new day
   if (dayTime < to){
      dayTime = to;
      dayBalance = Acc_B();
   }

   // Calculate total daily profit
   HistorySelect(start,end);
   int TotalDeals = HistoryDealsTotal();

   for(int i = 0; i < TotalDeals; i++){
      ulong Ticket = HistoryDealGetTicket(i);
      if(HistoryDealGetInteger(Ticket,DEAL_ENTRY) == DEAL_ENTRY_OUT){
         double Latest_Day_Profit = (HistoryDealGetDouble(Ticket,DEAL_PROFIT)
                                    + HistoryDealGetDouble(Ticket,DEAL_COMMISSION)
                                    + HistoryDealGetDouble(Ticket,DEAL_SWAP));
         total_day_Profit += Latest_Day_Profit;
      }
   }

   double startingBalance = 0.0;
   startingBalance = AccountInfoDouble(ACCOUNT_BALANCE) - total_day_Profit;
   string day_profit_in_TextFormat = "";
   double daily_Profit_or_Drawdown = NormalizeDouble(((total_day_Profit) * 100/startingBalance),2);
   day_profit_in_TextFormat = DoubleToString(daily_Profit_or_Drawdown,2) + " %";

   // Update dashboard texts with new data
   createText("4",TimeToString(start),120,80,clrBlue,10);
   createText("6",TimeToString(to),120,95,clrBlue,10);
   createText("8",TimeToString(end),120,110,clrBlack,10);

   createText("11",DoubleToString(initialBalance,2)+" "+Acc_S(),250,145,clrBlack,10);
   if (Acc_E() > initialBalance){
      createText("15",DoubleToString(Acc_E(),2)+" "+Acc_S(),250,175,clrMediumBlue,10);
      createText("17",DoubleToString(((Acc_E()-initialBalance)/initialBalance)*100,2)+" %",250,190,clrMediumBlue,10);
   }
   else if (Acc_E() < initialBalance){
      createText("15",DoubleToString(Acc_E(),2)+" "+Acc_S(),250,175,clrRed,10);
      createText("17",DoubleToString(((Acc_E()-initialBalance)/initialBalance)*100,2)+" %",250,190,clrRed,10);
   }
   else if (Acc_E() == initialBalance){
      createText("15",DoubleToString(Acc_E(),2)+" "+Acc_S(),250,175,clrBlack,10);
      createText("17",DoubleToString(((Acc_E()-initialBalance)/initialBalance)*100,2)+" %",250,190,clrBlack,10);
   }

   createText("20",DoubleToString(dayBalance,2)+" "+Acc_S(),270,225,clrBlack,10);
   createText("24","-"+DoubleToString((dayBalance*5/100),2)+" "+Acc_S(),270,255,clrBlue,10);
   if (Acc_B() > dayBalance){
      createText("26",DoubleToString(total_day_Profit,2)+" "+Acc_S(),270,270,clrMediumBlue,10);
      createText("28",day_profit_in_TextFormat,270,285,clrMediumBlue,10);
   }
   else if (Acc_B() < dayBalance){
      createText("26",DoubleToString(total_day_Profit,2)+" "+Acc_S(),270,270,clrRed,10);
      createText("28",day_profit_in_TextFormat,270,285,clrRed,10);
   }
   else if (Acc_B() == dayBalance){
      createText("26",DoubleToString(total_day_Profit,2)+" "+Acc_S(),270,270,clrBlack,10);
      createText("28",day_profit_in_TextFormat,270,285,clrBlack,10);
   }

   // Check if drawdown limits are hit and update trading permission
   if (daily_Profit_or_Drawdown <= -5.00 ||((Acc_E()-initialBalance)/initialBalance)*100 < -12.00){
      createText("29",">>> Max ThreshHold Hit, Can't Trade.",70,300,clrRed,10);
      isTradeAllowed = false;
   }
   else {
      createText("29",">>> Max ThresHold Not Hit, Can Trade.",70,300,clrMediumBlue,10);
      isTradeAllowed = true;
   }
}

//+------------------------------------------------------------------+
//| Create text label on the chart                                   |
//+------------------------------------------------------------------+
bool createText(string objName, string text, int x, int y, color clrTxt,int fontSize) {
 ResetLastError();
     if (!ObjectCreate(0,objName,OBJ_LABEL,0,0,0)){
        Print(__FUNCTION__,": failed to create the Label! Error code = ", GetLastError());
        return(false);
     }

   ObjectSetInteger(0,objName,OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0,objName,OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0,objName,OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetString(0,objName,OBJPROP_TEXT, text);
   ObjectSetInteger(0,objName,OBJPROP_FONTSIZE, fontSize);
   //ObjectSetString(0,objName,OBJPROP_FONT, "Calibri");
   ObjectSetInteger(0,objName,OBJPROP_COLOR, clrTxt);

   ChartRedraw(0);

   return(true);
}
```

Cheers to us! Now we created a daily Drawdown limit for Forex Trading Expert Advisor based on establishing a daily withdrawal cap for trading accounts.

### Conclusion

This article monitors trading activities and updates the chart with relevant text labels that indicate trade status, profit, and permission to trade based on certain thresholds. Using functions to place operations and update text labels on the chart helps to organize the article and simplify maintenance. We have looked at the basic steps that need to be implemented toward the automation of the famous daily drawdown limiter Forex trading strategy in MQL5. We have provided the basic definition and description of the strategy and shown how it can be created in MQL5. Traders can use the knowledge shown to develop a more complex daily drawdown limiter system that can, later on, be optimized to produce better results at the end.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15199.zip "Download all attachments in the single ZIP archive")

[Daily\_Drawdown\_Limiter.mq5](https://www.mql5.com/en/articles/download/15199/daily_drawdown_limiter.mq5 "Download Daily_Drawdown_Limiter.mq5")(9.31 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Implementing a Bollinger Bands Trading Strategy with MQL5: A Step-by-Step Guide](https://www.mql5.com/en/articles/15394)
- [Cascade Order Trading Strategy Based on EMA Crossovers for MetaTrader 5](https://www.mql5.com/en/articles/15250)
- [Developing Zone Recovery Martingale strategy in MQL5](https://www.mql5.com/en/articles/15067)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/469767)**
(2)


![atesz5870](https://c.mql5.com/avatar/avatar_na2.png)

**[atesz5870](https://www.mql5.com/en/users/atesz5870)**
\|
10 Jul 2024 at 13:29

nice!


![clueboard](https://c.mql5.com/avatar/avatar_na2.png)

**[clueboard](https://www.mql5.com/en/users/clueboard)**
\|
11 Jul 2024 at 09:40

Seems like you just used ForexAlgo-Trader ( [YouTube](https://www.youtube.com/@ForexAlgo-Trader "https://www.youtube.com/@ForexAlgo-Trader")) without giving him any credit at all and claiming his work as yours. From his video here, you can tell that this is his code line by line, variable by variable.

At least provide credit where it is due. Yes, the code is open source, but there's nothing wrong with providing credit where is it is rightfully deserved.

![How to Integrate Smart Money Concepts (BOS) Coupled with the RSI Indicator into an EA](https://c.mql5.com/2/83/Coupled_with_the_RSI_Indicator_into_an_EA____LOGO.png)[How to Integrate Smart Money Concepts (BOS) Coupled with the RSI Indicator into an EA](https://www.mql5.com/en/articles/15030)

Smart Money Concept (Break Of Structure) coupled with the RSI Indicator to make informed automated trading decisions based on the market structure.

![Neural networks made easy (Part 79): Feature Aggregated Queries (FAQ) in the context of state](https://c.mql5.com/2/71/Neural_networks_are_easy_Part_79____LOGO__2.png)[Neural networks made easy (Part 79): Feature Aggregated Queries (FAQ) in the context of state](https://www.mql5.com/en/articles/14394)

In the previous article, we got acquainted with one of the methods for detecting objects in an image. However, processing a static image is somewhat different from working with dynamic time series, such as the dynamics of the prices we analyze. In this article, we will consider the method of detecting objects in video, which is somewhat closer to the problem we are solving.

![Eigenvectors and eigenvalues: Exploratory data analysis in MetaTrader 5](https://c.mql5.com/2/83/Eigenvectors_and_eigenvalues__Exploratory_data_analysis_in_MetaTrader___LOGO.png)[Eigenvectors and eigenvalues: Exploratory data analysis in MetaTrader 5](https://www.mql5.com/en/articles/15229)

In this article we explore different ways in which the eigenvectors and eigenvalues can be applied in exploratory data analysis to reveal unique relationships in data.

![Neural networks made easy (Part 78): Decoder-free Object Detector with Transformer (DFFT)](https://c.mql5.com/2/70/Neural_networks_made_easy_Part_78____LOGO.png)[Neural networks made easy (Part 78): Decoder-free Object Detector with Transformer (DFFT)](https://www.mql5.com/en/articles/14338)

In this article, I propose to look at the issue of building a trading strategy from a different angle. We will not predict future price movements, but will try to build a trading system based on the analysis of historical data.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/15199&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049561026980064671)

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