---
title: The Optimal Method for Calculation of Total Position Volume by Specified Magic Number
url: https://www.mql5.com/en/articles/125
categories: Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:56:22.941831
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/125&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083226037208225621)

MetaTrader 5 / Trading systems


### Introduction

The MetaTrader 5 client terminal allows the parallel work of several Expert Advisors with one symbol. It is simple - just open several charts and attach Expert Advisors to them. It would be nice, if each Expert Advisor worked independently from the other Expert Advisors working with the same symbol (there is no such a problem for Expert Advisors working with different symbols).

First of all, it will allow an Expert Advisor to trade in full compliance with its testing performance and optimization in Strategy Tester. The conditions for opening the position may depend on the size or absence of already opened position. If several Expert Advisors work with the same symbol, they will affect each other.

The second and probably more important thing is to allow Expert Advisors to use different money management systems, depending on trading strategies implemented in Expert Advisors. And finally - the possibility to monitor the results of each Expert Advisor and turn it off if necessary.

### 1\. The General Principle of the Position Volume Calculation

When you open an order, you can mark it with a magic number by specifying the value of the magic variable in the [MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) structure, passed to the [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend) function. When the order is executed, the deal is also marked with order magic number. Further, analyzing the deals in the history, we can see deals, opened by different Expert Advisors.

The method of calculation of the total position is quite simple: for example, if you execute a buy deal with volume 0.1, then another one buy 0.1, and sell 0.1, the volume of the total position will be equal to 0.1+0.1-0.1=+0.1. We add the volumes of buy deals and subtract the volumes of sell deals and we get the volume of the total position.

It is important to start calculations when the volume of the total position is equal to 0. The first and the most obvious such point is the moment of account opening. In other words, you can request all the deals history of the account using the [HistorySelect()](https://www.mql5.com/en/docs/trading/historyselect) function with the first parameter, equal to 0 (the least possible time) and value of the second parameter [TimeCurrent()](https://www.mql5.com/en/docs/dateandtime/timecurrent) (the recent known time of a server):

```
HistorySelect(0,TimeCurrent()); // load all history
```

Then, go through the whole history from the beginning to the end, adding volumes of buy deals and subtracting the volumes of sell deals for each deal with the specified magic number. It's also a solution, but in practice, the history of deals can be quite large. This can significantly affect the speed of an Expert Advisor, especially during testing and optimization, up to the impossibility of the practical use of such an Expert Advisor. We need to find the very last moment in the deals history, when the volume of the total net position was equal to zero.

To do it, we have to go through the entire history first and find the last point, when the volume of the total net position was zero. Finding this point, we save it in a certain variable (fixed position time). Later, the Expert Advisor will go through the history of deals from the time of the saved point. The better solution is to save this point in a [global variable](https://www.mql5.com/en/docs/basis/variables/global) of the client terminal instead of variable of an Expert Advisor, because in such a case it will be destroyed when detaching the Expert Advisor.

In such a case even when the Expert Advisor is launched you need to load the minimum necessary history, instead of the entire history of deals. There are many Expert Advisors that can trade on the same symbol, so we will share this global variable (with the stored time of the recent point of the zero volume) with all Expert Advisors.

Let's digress from the main topic and consider the use of the global variables of the client terminal, that allow several Expert Advisors to work with same symbol (maybe with different parameters), and avoids the coincidence of the names, created by different instances of Expert Advisors.

### 2\. Using the Global Variables of the Client Terminal

The MQL5 language has the [MQLInfoString()](https://www.mql5.com/en/docs/check/mqlinfostring) function, that allows to obtain different information about a mql5-program.

To get information about the file name, call this function with the [MQL\_PROGRAM\_NAME](https://www.mql5.com/en/docs/constants/environment_state/mql5_programm_info) identifier:

```
MQL5InfoString(MQL_PROGRAM_NAME); // Expert Advisor name
```

Hence, we start the names of global variables with the name of an Expert Advisor. An Expert Advisor can work with several symbols; it means that we need to add the name of a symbol ( [Symbol](https://www.mql5.com/en/docs/predefined/_symbol)). Expert Advisors may work with the same symbol, but different timeframes (with different settings), for these cases we need to use the magic number. Therefore we also add the magic number.

For example, if the Expert Advisor has a magic number, stored in variable Magic\_N, we add it to the name of the global variable.

The names of all global variables will look as follows:

```
gvp=MQLInfoString(MQL_PROGRAM_NAME)+"_"+_Symbol+"_"+IntegerToString(Magic_N)+"_"; // name of an Expert Advisor and symbol name
                                                                            // and its magic number
```

where gvp (Global Variable Prefix) - is a string variable, declared in the section of common variables.

I would like to clarify the terminology to avoid the confusing of global variables, as they are used in programming (global variables are visible inside all functions, local variables of functions are visible inside the function only).

But here we have a different case - the "global variables" term means the global variables of the client terminal (special variables, stored in a file, they are available by the [GlobalVariable...()](https://www.mql5.com/en/docs/basis/variables/global) functions). When talking about global variables (as they are used in programming), we will use the "common variables" term. The term of local variables will mean local variables.

The global variables are useful, because they save their values after deinitialization of an Expert Advisor (restart of Expert Advisor, client terminal, computer), but in test mode it's necessary to clear all the variables (or a previous pass when optimization). The global variables, used in real operations should be separated from the global variables, created when testing, it's necessary to delete them after the testing. But you should not modify or delete the global variables, created by Expert Advisor.

Using the [AccountInfoInteger()](https://www.mql5.com/en/docs/account/accountinfointeger) function and calling it with the [ACCOUNT\_TRADE\_MODE](https://www.mql5.com/en/docs/constants/environment_state/accountinformation) identifier, you can distinguish the current mode: tester, demo, or real account.

Let's add a prefix to the global variables: "d" - when working on demo accounts, "r" - when working on real accounts, "t" - when working in Strategy Tester:

```
gvp=MQLInfoString(MQL_PROGRAM_NAME)+"_"+_Symbol+"_"+IntegerToString(Magic_N)+"_"; // name of an Expert Advisor, symbol name
                                                                                  // and the Expert Advisor magic number
if(AccountInfoInteger(ACCOUNT_TRADE_MODE)==ACCOUNT_TRADE_MODE_DEMO))
  {
   gvp=gvp+"d_"; // demo account
  }
if(AccountInfoInteger(ACCOUNT_TRADE_MODE)==ACCOUNT_TRADE_MODE_REAL)
  {
   gvp=gvp+"r_"; // real
  }
if(MQL5InfoInteger(MQL_TESTER))
  {
   gvp=gvp+"t_"; // testing
  }
```

The function should be called from the [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) function of the Expert Advisor.

As it is mentioned above, global variables should be deleted when testing, in other words, we need to add the function that deletes the global variables into the [OnDeinit()](https://www.mql5.com/en/docs/basis/function/events#ondeinit) function of an Expert Advisor:

```
void fDeleteGV()
  {
   if(MQL5InfoInteger(MQL_TESTER)) // Testing mode
     {
      for(int i=GlobalVariablesTotal()-1;i>=0;i--) // Check all global variables (from the end, not from the begin)
        {
         if(StringFind(GlobalVariableName(i),gvp,0)==0) // search for the specified prefix
           {
            GlobalVariableDel(GlobalVariableName(i)); // Delete variable
           }
        }
     }
  }
```

At present time it is impossible to interrupt testing in MetaTrader 5, in other words, the execution of [OnDeinit()](https://www.mql5.com/en/docs/basis/function/events#ondeinit) function isn't guaranteed, however it may appear in the future. We don't know whether the [OnDeinit()](https://www.mql5.com/en/docs/basis/function/events#ondeinit) function will be executed after the interruption of the Strategy Tester, therefore we delete the global variables at the beginning of the Expert Advisor running - inside the [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) function.

We will get the following code of the [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) and [OnDeinit()](https://www.mql5.com/en/docs/basis/function/events#ondeinit) functions:

```
int OnInit()
  {
   fCreateGVP(); // Creating a prefix for the names of global variables of the client terminal
   fDeleteGV();  // Delete global variables when working in Tester
   return(0);
  }

void OnDeinit(const int reason)
  {
   fDeleteGV();  // Delete global variables when working in tester
  }
```

Also we can simplify the use of [global variables](https://www.mql5.com/en/docs/basis/variables/global), by creating the functions with short names for the global variables creation (instead of the [GlobalVariableSet](https://www.mql5.com/en/docs/globals/globalvariableset)(gvp+...),.

The function to set the value of the global variable:

```
void fGVS(string aName,double aValue)
  {
   GlobalVariableSet(gvp+aName,aValue);
  }
```

The function to get the value of the global variable:

```
double fGVG(string aName)
  {
   return(GlobalVariableGet(gvp+aName));
  }
```

The function to delete the global variable:

```
void fGVD(string aName)
  {
   GlobalVariableDel(gvp+aName);
  }
```

We have discussed global variables, but that's not all.

We need to provide a possibility to create global variables for a symbol, and provide their different operation on the account and in the Strategy Tester. The names of these global variables should not depend on the name and magic number of an Expert Advisor.

Let's declare another variable for a global variable prefix, named as "Commom\_gvp". Then working with an account, it will have the value "COMMON", and it will have the same value, as a variable gvp when working with Strategy Tester (to delete the variable after or before the strategy backtesting process).

Finally, the function to prepare the global variables prefixes has the following form:

```
void fCreateGVP()
  {
   gvp=MQL5InfoString(MQL_PROGRAM_NAME)+"_"+_Symbol+"_"+IntegerToString(Magic_N)+"_";
   Commom_gvp="COMMOM_"; // Prefix for common variables for all Expert Advisors
   if(AccountInfoInteger(ACCOUNT_TRADE_MODE)==ACCOUNT_TRADE_MODE_DEMO)
     {
      gvp=gvp+"d_";
     }
   if(AccountInfoInteger(ACCOUNT_TRADE_MODE)==ACCOUNT_TRADE_MODE_REAL)
     {
      gvp=gvp+"r_";
     }
   if(MQLInfoInteger(MQL_TESTER))
     {
      gvp=gvp+"t_";
      Commom_gvp=gvp; // To be used in tester, the variables with such a prefix
                      // will be deleted after the testing
     }
  }
```

Someone may think that prefixes of global variables include extra information - the separation of demo and real accounts, and the "t" prefix when testing, although it could be done just by adding the "t" char, that indicates, that our Expert Advisor is working in Strategy Tester. But I have done it this way. We don't know the future and the things that may be required to analyze the work of the Expert Advisors.

Store is no sore they say.

The functions presented above means that the client terminal works with one account, there isn't any account changing during its work. Changing an account during the work of an Expert Advisor is prohibited. Of course, if it necessary, this problem can be solved by adding an account number to the names of global variables.

Another very important note! The length of the global variable name is limited to 63 symbols. Because of this fact, don't give long names to your Expert Advisors.

We have finished with global variables, now it's time to consider the major topic of the article - the calculation of position volume by a specified magic number.

### 3\. Calculating the Volume of a Position

First, let's check if there is a global variable with the information about the last time of the zero volume position using the [GlobalVariableCheck()](https://www.mql5.com/en/docs/globals/globalvariablecheck) function (for simplicity, if there isn't any opened position, we call it  a "zero position" case).

If there is such a variable - let's load the history of deals starting from the time, stored in the variable, otherwise we will load the whole history:

```
if(GlobalVariableCheck(Commom_gvp+sSymbol+"_HistStTm")) // Saved time of a "zero" total position
  {
   pLoadHistoryFrom=(datetime)GlobalVariableGet(Commom_gvp+pSymbol+"_HistStTm"); // initial date setting
                                                                             // select only the history needed
  }
else
 {
   GlobalVariableSet(Commom_gvp+sSymbol+"_HistStTm",0);
 }
if(!HistorySelect(sLoadHistoryFrom,TimeCurrent())) // Load the necessary part of the deal history
  {
   return(false);
  }
```

Next, we define the volume of the total net position for a symbol:

```
double CurrentVolume=fSymbolLots(pSymbol);
```

The volume of a position is determined using the fSymbolLots() function.

There are several ways to get the volume of a position: for example, it can be done using the [PositionSelect()](https://www.mql5.com/en/docs/trading/positionselect) function. If the function returns false, it means that there isn't any position (its volume is equal to zero). If the function returns true, the volume can be obtained using the [PositionGetDouble()](https://www.mql5.com/en/docs/trading/positiongetdouble) function with [POSITION\_VOLUME](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties) identifier. The position type (buy or sell), is determined using the [PositionGetInteger()](https://www.mql5.com/en/docs/trading/positiongetinteger) function with [POSITION\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties) identifier. The function returns a positive value for long positions and negative for short positions.

The complete function looks as follows:

```
double fSymbolLots(string aSymbol)
  {
   if(PositionSelect(aSymbol,1000)) // the position has been selected successfully, so it exists
     {
      switch(PositionGetInteger(POSITION_TYPE)) // It returns the positive or negative value dependent on the direction
        {
         case POSITION_TYPE_BUY:
            return(NormalizeDouble(PositionGetDouble(POSITION_VOLUME),2));
            break;
         case POSITION_TYPE_SELL:
            return(NormalizeDouble(-PositionGetDouble(POSITION_VOLUME),2));
            break;
        }
     }
   else
     {
      return(0);
     }
  }
```

Alternatively, you can determine the volume of the total position of the symbol by loop through all positions, the number of positions is determined by [PositionsTotal()](https://www.mql5.com/en/docs/trading/positionstotal) function. Then, find the necessary symbol using the [PositionGetSymbol()](https://www.mql5.com/en/docs/trading/positiongetsymbol) function, and determine the volume and direction of the position (the [PositionGetDouble()](https://www.mql5.com/en/docs/trading/positiongetdouble) with [POSITION\_VOLUME](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties) identifier and [PositionGetInteger()](https://www.mql5.com/en/docs/trading/positiongetinteger) function with [POSITION\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties) identifier).

In this case, the ready function will have the form:

```
double fSymbolLots(string aSymbol)
  {
   double TmpLots=0;
   for(int i=0;i<PositionsTotal();i++) // Go through all positions
     {
      if(PositionGetSymbol(i)==aSymbol) // we have found a position with specified symbol
        {
         TmpLots=PositionGetDouble(POSITION_VOLUME);
         if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL)
           {
            TmpLots*=-1; // the sign is dependent on the position type           }
         break;
        }
     }
   TmpLots=NormalizeDouble(TmpLots,2);
   return(TmpLots);
  }
```

After determination of the current volume, we will go through the history of deals from the end to the start, until the sum of the volumes becomes equal to the volume.

The length of the selected history ща deals is determined using the [HistoryDealsTotal()](https://www.mql5.com/en/docs/trading/historydealstotal) function, the ticket is determined for each deal using the [HistoryDealGetTicket()](https://www.mql5.com/en/docs/trading/historydealgetticket) function, the deal data is extracted using the [HistoryDealGetInteger()](https://www.mql5.com/en/docs/trading/historydealgetinteger) function (the [DEAL\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties) identifier for deal type) and [HistoryDealGetDouble()](https://www.mql5.com/en/docs/trading/historydealgetdouble) (the [DEAL\_VOLUME](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties) identifier for deal volume):

```
double Sum=0;
int FromI=0;
int FromTicket=0;
for(int i=HistoryDealsTotal()-1;i>=0;i--) // go through all the deals from the end to the beginning
  {
   ulong ticket=HistoryDealGetTicket(i); // Get ticket of the deal
   if(ticket!=0)
     {
      switch(HistoryDealGetInteger(ticket,DEAL_TYPE)) // We add or subtract the volume depending on deal direction
        {
         case DEAL_TYPE_BUY:
            Sum+=HistoryDealGetDouble(ticket,DEAL_VOLUME);
            Sum=NormalizeDouble(Sum,2);
            break;
         case DEAL_TYPE_SELL:
            Sum-=HistoryDealGetDouble(ticket,DEAL_VOLUME);
            Sum=NormalizeDouble(Sum,2);
            break;
        }
      if(CurrentVolume==Sum) // all the deals has scanned
        {
         sLoadHistoryFrom=HistoryDealGetInteger(ticket,DEAL_TIME); // Save the time of a "zero" position
         GlobalVariableSet(Commom_gvp+aSymbol+"_HistStTm",sLoadHistoryFrom);
         FromI=i; // Save the index
         break;
        }
     }
  }
```

When we have found this point, we store the time into the global variable, which will be used further when loading the history of deals (the deal index in the history is stored in the FromI variable).

Before the deal with an index FromI the total position on the symbol was equal to zero.

Now we go from the FromI towards the end of the history and count the volume of the deals with the specified magic number:

```
static double sVolume=0;
static ulong sLastTicket=0;
for(int i=FromI;i<HistoryDealsTotal();i++) // from the first deal until the end
  {
   ulong ticket=HistoryDealGetTicket(i);   // Get deal ticket
   if(ticket!=0)
     {
      if(HistoryDealGetString(ticket,DEAL_SYMBOL)==aSymbol) // Specified symbol
        {
         long PosMagic=HistoryDealGetInteger(ticket,DEAL_MAGIC);
         if(PosMagic==aMagic || aMagic==-1) // Specified magic
           {
            switch(HistoryDealGetInteger(ticket,DEAL_TYPE)) // add or subtract the deal volumes
                                                       // depending on the deal type
              {
               case DEAL_TYPE_BUY:
                  sVolume+=HistoryDealGetDouble(ticket,DEAL_VOLUME);
                  sLastTicket=ticket;
                  break;
               case DEAL_TYPE_SELL:
                  sVolume-=HistoryDealGetDouble(ticket,DEAL_VOLUME);
                  sLastTicket=ticket;
                  break;
              }
           }
        }
     }
  }
```

After the end of a loop we will have the volume of a current position by the specified magic number, the ticket of a last deal with the specified magic number will be stored in the sLastTicket variable, after the execution of the deal the total volume of a position with the specified magic number will be equal to sVolume. The preliminary work of the function is over.

The sLoadHistoryFrom, sLastTicket and sVolume variables are declared as static (they store their values after the completion of the function), this values will be used further for each call of the function.

We have the time (the starting point of the history of deals), the deal ticket, after its execution the volume of the total position (with specified symbol) will have the current value.

Because the time of the zero volume position, it's sufficient to go through the history from the current time to the saved time and perform the deal volumes summation and save the volume and ticket of the last deal.

Thus, the calculation of the total position of the Expert Advisor is the processing of a few last deals:

```
if(!HistorySelect(sLoadHistoryFrom,TimeCurrent())) // Request for the deals history up to the current time
  {
   return(false);
  }
for(int i=HistoryDealsTotal()-1;i>=0;i--) // Loop from the end
  {
   ulong ticket=HistoryDealGetTicket(i); // Get ticke
   if(ticket!=0)
     {
      if(ticket==sLastTicket) // We have found the already calculated deal, save the ticket and break
        {
         sLastTicket=HistoryDealGetTicket(HistoryDealsTotal()-1);
         break;
        }
      switch(HistoryDealGetInteger(ticket,DEAL_TYPE)) // Add or subtract deal volume depending on deal type
        {
         case DEAL_TYPE_BUY:
            sVolume+=HistoryDealGetDouble(ticket,DEAL_VOLUME);
            break;
         case DEAL_TYPE_SELL:
            sVolume-=HistoryDealGetDouble(ticket,DEAL_VOLUME);
            break;
        }
     }
  }
```

The algorithm of the function can be presented as follows:

![](https://c.mql5.com/2/1/Image.png)

The complete function:

```
bool fGetPositionVolume(string aSymbol,int aMagic,double aVolume)
  {
   static bool FirstStart=false;
   static double sVolume=0;
   static ulong sLastTicket=0;
   static datetime sLoadHistoryFrom=0;
   // First execution of function when Expert Advisor has started
   if(!FirstStart)
     {
      if(GlobalVariableCheck(Commom_gvp+aSymbol+"_HistStTm"))
        {
         sLoadHistoryFrom=(datetime)GlobalVariableGet(Commom_gvp+aSymbol+"_HistStTm");
        }
      else
        {
         GlobalVariableSet(Commom_gvp+aSymbol+"_HistStTm",0);
        }
      if(!HistorySelect(sLoadHistoryFrom,TimeCurrent())) // Return if unsuccessful,
                                                      // we will repeat on the next tick
        {
         return(false);
        }
      double CurrentVolume=fSymbolLots(aSymbol); // Total volume
      double Sum=0;
      int FromI=0;
      int FromTicket=0;
      // Search the last time when position volume was equal to zero
      for(int i=HistoryDealsTotal()-1;i>=0;i--)
        {
         ulong ticket=HistoryDealGetTicket(i);
         if(ticket!=0)
           {
            switch(HistoryDealGetInteger(ticket,DEAL_TYPE))
              {
               case DEAL_TYPE_BUY:
                  Sum+=HistoryDealGetDouble(ticket,DEAL_VOLUME);
                  Sum=NormalizeDouble(Sum,2);
                  break;
               case DEAL_TYPE_SELL:
                  Sum-=HistoryDealGetDouble(ticket,DEAL_VOLUME);
                  Sum=NormalizeDouble(Sum,2);
                  break;
              }
            if(CurrentVolume==Sum)
              {
               sLoadHistoryFrom=HistoryDealGetInteger(ticket,DEAL_TIME);
               GlobalVariableSet(Commom_gvp+aSymbol+"_HistStTm",sLoadHistoryFrom);
               FromI=i;
               break;
              }
           }
        }
      // Calculate the volume of position with specified magic number and symbol
      for(int i=FromI;i<HistoryDealsTotal();i++)
        {
         ulong ticket=HistoryDealGetTicket(i);
         if(ticket!=0)
           {
            if(HistoryDealGetString(ticket,DEAL_SYMBOL)==aSymbol)
              {
               long PosMagic=HistoryDealGetInteger(ticket,DEAL_MAGIC);
               if(PosMagic==aMagic || aMagic==-1)
                 {
                  switch(HistoryDealGetInteger(ticket,DEAL_TYPE))
                    {
                     case DEAL_TYPE_BUY:
                        sVolume+=HistoryDealGetDouble(ticket,DEAL_VOLUME);
                        sLastTicket=ticket;
                        break;
                     case DEAL_TYPE_SELL:
                        sVolume-=HistoryDealGetDouble(ticket,DEAL_VOLUME);
                        sLastTicket=ticket;
                        break;
                    }
                 }
              }
           }
        }
      FirstStart=true;
     }

   // Recalculate the volume of a position (with specified symbol and magic)
   // for the deals, after the zero position time
   if(!HistorySelect(sLoadHistoryFrom,TimeCurrent()))
     {
      return(false);
     }
   for(int i=HistoryDealsTotal()-1;i>=0;i--)
     {
      ulong ticket=HistoryDealGetTicket(i);
      if(ticket!=0)
        {
         if(ticket==sLastTicket)
           {
            sLastTicket=HistoryDealGetTicket(HistoryDealsTotal()-1);
            break;
           }
         switch(HistoryDealGetInteger(ticket,DEAL_TYPE))
           {
            case DEAL_TYPE_BUY:
               sVolume+=HistoryDealGetDouble(ticket,DEAL_VOLUME);
               break;
            case DEAL_TYPE_SELL:
               sVolume-=HistoryDealGetDouble(ticket,DEAL_VOLUME);
               break;
           }
        }
     }
   aVolume=NormalizeDouble(sVolume,2);;
   return(true);
  }
```

The symbol and magic number are passed to the function, that returns the position volume. It returns true if successful and false otherwise.

When successful, it returns the requested volume to the variable aVolume, passed to the function by reference. Static variables, declared in the function, do not allow to use this function with different parameters (symbol and magic number).

In case of MQL4, this problem could be solved by creation of a copy of this function with a different name and call it for the other pair "symbol-magic" or declare the variables FirstStart, sVolume, sLastTicket, sLoadHistoryFrom as common variables - for each pair "symbol-magic" and pass them into the function.

It also can be implemented in MQL5 the same way, but MQL5 has a much more convenient feature - the classes, it's the case where the use of classes is reasonable. When using classes, it's necessary to create a class instance for each pair of symbol-magic number, the data will be stored in each class instance.

Let's declare a PositionVolume class. All the variables, declared as static inside the function, will be declared as  private, we will not use them directly from the Expert Advisor, except the Volume variable. But we will need it only after the execution of volume calculation function. Also we declare the Symbol and Magic variables - it's impractical to pass them into the function, just do once when initializing the class instance.

The class will have two public functions: the initialization function and the function for calculation of position volume, and a private function to determine the total volume of the position:

```
class PositionVolume
  {
private:
   string            pSymbol;
   int               pMagic;
   bool              pFirstStart;
   ulong             pLastTicket;
   double            pVolume;
   datetime         pLoadHistoryFrom;
   double            SymbolLots();
public:
   void Init(string aSymbol,int aMagic)
     {
      pSymbol=aSymbol;
      pMagic=aMagic;
      pFirstStart=false;
      pLastTicket=0;
      pVolume=0;
     }
   bool              GetVolume(double  &aVolume);
  };
```

```
bool PositionVolume::GetVolume(double  &aVolume)
  {
   if(!pFirstStart)
     {
      if(GlobalVariableCheck(Commom_gvp+pSymbol+"_HistStTm"))
        {
         pLoadHistoryFrom=(datetime)GlobalVariableGet(Commom_gvp+pSymbol+"_HistStTm");
        }
      else
        {
         GlobalVariableSet(Commom_gvp+pSymbol+"_HistStTm",0);
        }
      if(!HistorySelect(pLoadHistoryFrom,TimeCurrent()))
        {
         return(false);
        }
      double CurrentVolume=fSymbolLots(pSymbol);
      double Sum=0;
      int FromI=0;
      int FromTicket=0;
      for(int i=HistoryDealsTotal()-1;i>=0;i--)
        {
         ulong ticket=HistoryDealGetTicket(i);
         if(ticket!=0)
           {
            switch(HistoryDealGetInteger(ticket,DEAL_TYPE))
              {
               case DEAL_TYPE_BUY:
                  Sum+=HistoryDealGetDouble(ticket,DEAL_VOLUME);
                  Sum=NormalizeDouble(Sum,2);
                  break;
               case DEAL_TYPE_SELL:
                  Sum-=HistoryDealGetDouble(ticket,DEAL_VOLUME);
                  Sum=NormalizeDouble(Sum,2);
                  break;
              }
            if(CurrentVolume==Sum)
              {
               pLoadHistoryFrom=HistoryDealGetInteger(ticket,DEAL_TIME);
               GlobalVariableSet(Commom_gvp+pSymbol+"_HistStTm",pLoadHistoryFrom);
               FromI=i;
               break;
              }
           }
        }
      for(int i=FromI;i<HistoryDealsTotal();i++)
        {
         ulong ticket=HistoryDealGetTicket(i);
         if(ticket!=0)
           {
            if(HistoryDealGetString(ticket,DEAL_SYMBOL)==pSymbol)
              {
               long PosMagic=HistoryDealGetInteger(ticket,DEAL_MAGIC);
               if(PosMagic==pMagic || pMagic==-1)
                 {
                  switch(HistoryDealGetInteger(ticket,DEAL_TYPE))
                    {
                     case DEAL_TYPE_BUY:
                        pVolume+=HistoryDealGetDouble(ticket,DEAL_VOLUME);
                        pLastTicket=ticket;
                        break;
                     case DEAL_TYPE_SELL:
                        pVolume-=HistoryDealGetDouble(ticket,DEAL_VOLUME);
                        pLastTicket=ticket;
                        break;
                    }
                 }
              }
           }
        }
      pFirstStart=true;
     }
   if(!HistorySelect(pLoadHistoryFrom,TimeCurrent()))
     {
      return(false);
     }
   for(int i=HistoryDealsTotal()-1;i>=0;i--)
     {
      ulong ticket=HistoryDealGetTicket(i);
      if(ticket!=0)
        {
         if(ticket==pLastTicket)
           {
            break;
           }
         if(HistoryDealGetString(ticket,DEAL_SYMBOL)==pSymbol)
           {
            long PosMagic=HistoryDealGetInteger(ticket,DEAL_MAGIC);
            if(PosMagic==pMagic || pMagic==-1)
              {
               switch(HistoryDealGetInteger(ticket,DEAL_TYPE))
                 {
                  case DEAL_TYPE_BUY:
                     pVolume+=HistoryDealGetDouble(ticket,DEAL_VOLUME);
                     break;
                  case DEAL_TYPE_SELL:
                     pVolume-=HistoryDealGetDouble(ticket,DEAL_VOLUME);
                     break;
                 }
              }
           }
        }
     }
   if(HistoryDealsTotal()>0)
     {
      pLastTicket=HistoryDealGetTicket(HistoryDealsTotal()-1);
     }
   pVolume=NormalizeDouble(pVolume,2);
   aVolume=pVolume;
   return(true);
  }
```

```
double PositionVolume::SymbolLots()
  {
   double TmpLots=0;
   for(int i=0;i<PositionsTotal();i++)
     {
      if(PositionGetSymbol(i)==pSymbol)
        {
         TmpLots=PositionGetDouble(POSITION_VOLUME);
         if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL)
           {
            TmpLots*=-1;
           }
         break;
        }
     }
   TmpLots=NormalizeDouble(TmpLots,2);
   return(TmpLots);
  }
```

When using this class for each pair of symbol-magic number, it's necessary to create a class instance:

```
PositionVolume PosVol11;
PositionVolume PosVol12;
PositionVolume PosVol21;
PositionVolume PosVol22;
```

It should be initialized in the [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) function of an Expert Advisor, for example:

```
PosVol11.Init(Symbol_1,Magic_1);
PosVol12.Init(Symbol_1,Magic_2);
PosVol21.Init(Symbol_2,Magic_1);
PosVol22.Init(Symbol_2,Magic_2);
```

After that it's possible to get the volume of a position by the specified symbol and magic number. Let's call the GetVolume function of the corresponding class instance.

If successful, it returns true and puts the value to the variable, passed by reference as parameter of the function:

```
double Vol11;
double Vol12;
double Vol21;
double Vol22;
PosVol11.GetVolume(Vol11);
PosVol12.GetVolume(Vol12);
PosVol21.GetVolume(Vol21);
PosVol22.GetVolume(Vol22);
```

Here, one can say, you are done, but the control test is left.

### 4\. Control Test

To test the work of the function we have used an Expert Advisor, which works simultaneously with four positions:

1. using the RSI indicator with period 14 on EURUSD with magic number 1;

2. using the RSI indicator with period 21 on EURUSD with magic number 2;

3. using the RSI indicator with period 14 on GBPUSD with magic number 1;

4. using the RSI indicator with period 21 on GBPUSD with magic number 2;


The Expert Advisor with magic number 1 traded 0.1 lot of the volume, the Expert Advisor with magic number 2 traded the volume equal to 0.2 lots.

The volume of a deal is added to the variables of the Expert Advisor when executing a deal, before and after the deal the volume of each position was determined using the function, presented above.

The function generates a message if there was an error in calculation of volumes.

The code of the Expert Advisor can be found in the attachment to the article (file name: ePosVolTest.mq5).

### Conclusion

Many functions are needed for an Expert Advisor, and they should be implemented the way, convenient to use at all the stages. These functions should be written in terms of the best use of computing resources.

The method of calculation of the position volume proposed in this article, satisfies these conditions - it loads only the required minimum of the history of deals when being launched. When working, it recalculates the current volume of the position using the latests deals.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/125](https://www.mql5.com/ru/articles/125)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/125.zip "Download all attachments in the single ZIP archive")

[eposvoltest.mq5](https://www.mql5.com/en/articles/download/125/eposvoltest.mq5 "Download eposvoltest.mq5")(17.98 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Color optimization of trading strategies](https://www.mql5.com/en/articles/5437)
- [Analyzing trading results using HTML reports](https://www.mql5.com/en/articles/5436)
- [Developing the oscillator-based ZigZag indicator. Example of executing a requirements specification](https://www.mql5.com/en/articles/4502)
- [Auto search for divergences and convergences](https://www.mql5.com/en/articles/3460)
- [The Flag Pattern](https://www.mql5.com/en/articles/3229)
- [Wolfe Waves](https://www.mql5.com/en/articles/3131)
- [Universal Trend with the Graphical Interface](https://www.mql5.com/en/articles/3018)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/1550)**
(32)


![SergeyNU](https://c.mql5.com/avatar/avatar_na2.png)

**[SergeyNU](https://www.mql5.com/en/users/sergeynu)**
\|
31 Jul 2019 at 14:58

Good afternoon.

Help me to understand how these classes and OOP work. Let's say we have connected this class to an [Expert Advisor](https://www.mql5.com/en/market/mt5/ "A Market of Applications for the MetaTrader 5 and MetaTrader 4"), is it executed only when accessing it from the Expert Advisor? Or it works in parallel and gives only results on request?

![Payman](https://c.mql5.com/avatar/2019/8/5D5713C0-2A4D.png)

**[Payman](https://www.mql5.com/en/users/paymanz)**
\|
5 Jul 2020 at 07:47

file cannot be compiled.


![jelagins](https://c.mql5.com/avatar/avatar_na2.png)

**[jelagins](https://www.mql5.com/en/users/jelagins)**
\|
10 Sep 2025 at 13:13

the following warnings and errors were noticed during compilation in mql5: [possible loss of data](https://www.mql5.com/en/docs/basis/types/casting "MQL5 Documentation: Type conversion") due to type conversion from 'long' to 'int' eposvoltest.mq5 426 20 , possible  loss of data due to type conversion from 'long' to 'datetime' eposvoltest.mq5 439 32, possible loss of data due to type conversion from 'long' to 'int' eposvoltest.mq5 456 26, possible loss of data due to type conversion from 'long' to 'int' eposvoltest.mq5 491 23, return value of 'OrderSend' should be checked eposvoltest.mq5 236 4, return value of 'OrderSend' should be checked eposvoltest.mq5 268 4, '-' - expression not boolean eposvoltest.mq5 279 14, 'MQL5\_TESTING' is deprecated, use 'MQL\_TESTER' instead eposvoltest.mq5 335 23, 'MQL5\_TESTING' is deprecated, use 'MQL\_TESTER' instead of eposvoltest.mq5 346 23

10.09.2025

![ceejay1962](https://c.mql5.com/avatar/avatar_na2.png)

**[ceejay1962](https://www.mql5.com/en/users/ceejay1962)**
\|
10 Sep 2025 at 16:49

**jelagins [#](https://www.mql5.com/en/forum/1550/page3#comment_58003381):**

the following warnings and errors were noticed during compilation in mql5: [possible loss of data](https://www.mql5.com/en/docs/basis/types/casting "MQL5 Documentation: Type conversion") due to type conversion from 'long' to 'int' eposvoltest.mq5 426 20 , possible  loss of data due to type conversion from 'long' to 'datetime' eposvoltest.mq5 439 32, possible loss of data due to type conversion from 'long' to 'int' eposvoltest.mq5 456 26, possible loss of data due to type conversion from 'long' to 'int' eposvoltest.mq5 491 23, return value of 'OrderSend' should be checked eposvoltest.mq5 236 4, return value of 'OrderSend' should be checked eposvoltest.mq5 268 4, '-' - expression not boolean eposvoltest.mq5 279 14, 'MQL5\_TESTING' is deprecated, use 'MQL\_TESTER' instead eposvoltest.mq5 335 23, 'MQL5\_TESTING' is deprecated, use 'MQL\_TESTER' instead of eposvoltest.mq5 346 23

10.09.2025

It is hardly surprising, considering that the code dates from 2010!

![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
10 Sep 2025 at 17:05

There was an error in the code.

```
bool fOpSell(string aSymbol,double aVolume=0.1,int aSlippage=0,int aMagic=0,string aComment="",string aMessage="",bool aSound=false)
  {
   request.symbol=aSymbol;
   request.action=TRADE_ACTION_DEAL;
   request.type=ORDER_TYPE_SELL;
   request.volume=aVolume;
   request.price=SymbolInfoDouble(aSymbol,SYMBOL_BID);
   request.sl=0;
   request.tp=0;
   request.deviation=aSlippage;
   request.type_filling=ORDER_FILLING_FOK;
   request.comment=aComment;
   request.magic=aMagic;
   if(aMessage!="")Print(aMessage);
   if(aSound)PlaySound("expert");
   OrderSend(request,result);
   if(result.retcode==TRADE_RETCODE_DONE)
     {
      Print("...lucky (#"+IntegerToString(result.order)+")");
      if(aSound)PlaySound("ok");
      return(1);
     }
   else
     {
      Print("...mistake "+IntegerToString(result.retcode)+" - "+fTradeRetCode(result.retcode));
      if(aSound)PlaySound("timeout");
      return(-1);
     }
  }
```

The corrected version is in the trailer.

![Transferring Indicators from MQL4 to MQL5](https://c.mql5.com/2/0/migrate_indicators_mql4_to_MQL5__1.png)[Transferring Indicators from MQL4 to MQL5](https://www.mql5.com/en/articles/66)

This article is dedicated to peculiarities of transferring price constructions written in MQL4 to MQL5. To make the process of transferring indicator calculations from MQL4 to MQL5 easier, the mql4\_2\_mql5.mqh library of functions is suggested. Its usage is described on the basis of transferring of the MACD, Stochastic and RSI indicators.

![Interview with Nikolay Kositsin: multicurrency EA are less risky (ATC 2010)](https://c.mql5.com/2/0/avatar__12.png)[Interview with Nikolay Kositsin: multicurrency EA are less risky (ATC 2010)](https://www.mql5.com/en/articles/524)

Nikolay Kositsin has told us about his developments. He believes multicurrency Expert Advisors are a promising direction; and he is an experienced developer of such robots. At the championships, Nikolay participates only with multicurrency EAs. His Expert Advisor was the only multicurrency EA among the prize winners of all the ATC contests.

![Creating and Publishing of Trade Reports and SMS Notification](https://c.mql5.com/2/0/trade_reports_SMS_MQL5.png)[Creating and Publishing of Trade Reports and SMS Notification](https://www.mql5.com/en/articles/61)

Traders don't always have ability and desire to seat at the trading terminal for hours. Especially, if trading system is more or less formalized and can automatically identify some of the market states. This article describes how to generate a report of trade results (using Expert Advisor, Indicator or Script) as HTML-file and upload it via FTP to WWW-server. We will also consider sending notification of trade events as SMS to mobile phone.

![How to Write an Indicator on the Basis of Another Indicator](https://c.mql5.com/2/0/indicator_based_on_other_MQL5__1.png)[How to Write an Indicator on the Basis of Another Indicator](https://www.mql5.com/en/articles/127)

In MQL5 you can write an indicator both from a scratch and on the basis of another already existing indicator, in-built in the client terminal or a custom one. And here you also have two ways - to improve an indicator by adding new calculations and graphical styles to it , or to use an indicator in-built in the client terminal or a custom one via the iCustom() or IndicatorCreate() functions.

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/125&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083226037208225621)

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