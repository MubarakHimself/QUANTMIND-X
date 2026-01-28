---
title: Continuous Walk-Forward Optimization (Part 2): Mechanism for creating an optimization report for any robot
url: https://www.mql5.com/en/articles/7452
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T17:24:57.372695
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/7452&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068197946639447735)

MetaTrader 5 / Examples


### Introduction

This is the next article within a series devoted to the creation of an automated optimizer, which can perform walk-through optimization of trading strategies. The [previous article](https://www.mql5.com/en/articles/7290) described the creation of a DLL to be used in our auto optimizer and in Expert Advisors. This new part is entirely devoted to the MQL5 language. We will consider optimization report generation methods and the application of this functionality within your algorithms.

The strategy tester does not allow access to its data from an Expert Advisor while the provided results lack details, therefor,e we will use the optimization report downloading functionality implemented in my previous articles. Since separate parts of this functionality have been modified, while others were not fully covered in earlier articles, let's consider these features once again as they constitute the key parts of our program. Let's start with one of the new features: addition of custom commission. All classes and functions described in this article are located under the Include/History manager directory.

### Implementation of custom commission and slippage

The MetaTrader 5 platform tester provides a lot of exciting possibilities. However, some brokers do not add trade commission to the history. Furthermore, sometimes you may want to add additional commission for extra strategy testing. For these purposes, I have added a class that saves commission for each separate symbol. Upon the call of an appropriate method, the class returns commission and specified slippage. The class itself is entitled as follows:

```
class CCCM
  {
private:
   struct Keeper
     {
      string            symbol;
      double            comission;
      double            shift;
     };

   Keeper            comission_data[];
public:

   void              add(string symbol,double comission,double shift);

   double            get(string symbol,double price,double volume);
   void              remove(string symbol);
  };
```

The Keeper structure has been created for this class, which stores commission and slippage for the specified asset. An array has been created to store all passed commissions and slippage values. The three declared methods add, receive and delete data. The asset adding method is implemented as follows:

```
void CCCM::add(string symbol,double comission,double shift)
{
 int s=ArraySize(comission_data);

 for(int i=0;i<s;i++)
   {
    if(comission_data[i].symbol==symbol)
        return;
   }

 ArrayResize(comission_data,s+1,s+1);

 Keeper keeper;
 keeper.symbol=symbol;
 keeper.comission=MathAbs(comission);
 keeper.shift=MathAbs(shift);

 comission_data[s]=keeper;
}
```

This method implements the addition of a new asset to the collection, after a preliminary check of whether the same asset has already been added earlier. Please note that slippage and commission are added modulo. Thus, when all costs are summed up, the sign will not affect the calculation. Another point to pay attention to is the calculation units.

- Commission: depending on the asset type, commission can be added in the profit currency or as a percentage of the volume traded.
- Slippage: always specified in points.

Also note that these values are not added per a complete position (i.e. opening + closing), but per each trade. Thus, the position will have the following value: n\*commission + n\*slippage, where n is the number of all deals within a position.

The _remove_ method deletes the selected asset. The symbol name is used for the key.

```
void CCCM::remove(string symbol)
{
 int total=ArraySize(comission_data);
 int ind=-1;
 for(int i=0;i<total;i++)
   {
    if(comission_data[i].symbol==symbol)
      {
       ind=i;
       break;
      }
   }
 if(ind!=-1)
    ArrayRemove(comission_data,ind,1);
}
```

If the appropriate symbol is not found, the method terminates without deleting any asset.

The _get_ method is used to obtain the selected shift and commission. The method implementation is different for different asset types.

```
double CCCM::get(string symbol,double price,double volume)
{

 int total=ArraySize(comission_data);
 for(int i=0;i<total;i++)
   {
    if(comission_data[i].symbol==symbol)
      {
       ENUM_SYMBOL_CALC_MODE mode=(ENUM_SYMBOL_CALC_MODE)SymbolInfoInteger(symbol,SYMBOL_TRADE_CALC_MODE);

       double shift=comission_data[i].shift*SymbolInfoDouble(symbol,SYMBOL_TRADE_TICK_VALUE);

       double ans;
       switch(mode)
         {
          case SYMBOL_CALC_MODE_FOREX :
             ans=(comission_data[i].comission+shift)*volume;
             break;
          case SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE :
             ans=(comission_data[i].comission+shift)*volume;
             break;
          case SYMBOL_CALC_MODE_FUTURES :
             ans=(comission_data[i].comission+shift)*volume;
             break;
          case SYMBOL_CALC_MODE_CFD :
             ans=(comission_data[i].comission+shift)*volume;
             break;
          case SYMBOL_CALC_MODE_CFDINDEX :
             ans=(comission_data[i].comission+shift)*volume;
             break;
          case SYMBOL_CALC_MODE_CFDLEVERAGE :
             ans=(comission_data[i].comission+shift)*volume;
             break;
          case SYMBOL_CALC_MODE_EXCH_STOCKS :
            {
             double trading_volume=price*volume*SymbolInfoDouble(symbol,SYMBOL_TRADE_CONTRACT_SIZE);
             ans=trading_volume*comission_data[i].comission/100+shift*volume;
            }
          break;
          case SYMBOL_CALC_MODE_EXCH_FUTURES :
             ans=(comission_data[i].comission+shift)*volume;
             break;
          case SYMBOL_CALC_MODE_EXCH_FUTURES_FORTS :
             ans=(comission_data[i].comission+shift)*volume;
             break;
          case SYMBOL_CALC_MODE_EXCH_BONDS :
            {
             double trading_volume=price*volume*SymbolInfoDouble(symbol,SYMBOL_TRADE_CONTRACT_SIZE);
             ans=trading_volume*comission_data[i].comission/100+shift*volume;
            }
          break;
          case SYMBOL_CALC_MODE_EXCH_STOCKS_MOEX :
            {
             double trading_volume=price*volume*SymbolInfoDouble(symbol,SYMBOL_TRADE_CONTRACT_SIZE);
             ans=trading_volume*comission_data[i].comission/100+shift*volume;
            }
          break;
          case SYMBOL_CALC_MODE_EXCH_BONDS_MOEX :
            {
             double trading_volume=price*volume*SymbolInfoDouble(symbol,SYMBOL_TRADE_CONTRACT_SIZE);
             ans=trading_volume*comission_data[i].comission/100+shift*volume;
            }
          break;
          case SYMBOL_CALC_MODE_SERV_COLLATERAL :
             ans=(comission_data[i].comission+shift)*volume;
             break;
          default: ans=0; break;
         }

       if(ans!=0)
          return -ans;

      }
   }

 return 0;
}
```

Search for the specified symbol in the array. Since different commission calculation types are used for different symbol types, the commission setting types are also different. For example, stock and bond commission is set as a percentage of turnover, while the turnover is calculated as the product of the number of lots by the number of contracts per lots and the deal price.

As a result, we get the monetary equivalent of the performed operation. The method execution result is always the sum of the commission and slippage in monetary terms. The slippage is calculated based on the tick value. Further, the described class will be used in the next class downloading reports. Commission parameters for each of the assets can be hard coded or automatically requested from a database; alternatively, it can be passed to the EA as inputs. In my algorithms, I prefer the latter method.

### Innovation in the CDealHistoryGetter class

The classes considered in this part further on, were mentioned in previous articles. That is why I will not go deep into derails for earlier discussed classes. However, I will try to provide comprehensive descriptions for new classes, because the key algorithm within the trading report downloading algorithm is the creation of the downloaded report.

Let us start with the CDealHistoryGetter class, which has been used with some modifications since the first [article](https://www.mql5.com/en/articles/4803). The first article was mainly devoted to describing this class. The latest version is attached below. It includes some new functionalities and minor fixes. The mechanism downloading the report in the easy-to-read form is described in detail in the first article. In this article, we will consider in more detail the addition of commission and slippage to the report. According to the OOP principle, which implies that one object must perform one specific designated purpose, this object is created to receive all types of trading report results. It contains the following public methods, each performing its specific role:

- getHistory — this method allows downloading the trading history grouped by positions. If we download the trading history in a cycle using standard methods, without any filter, we will receive the description of deals presented by the DealData structure:

```
struct DealData
  {
   long              ticket;        // Deal ticket
   long              order;         // The number of the order that opened the position
   datetime          DT;            // Position open date
   long              DT_msc;        // Position open date in milliseconds
   ENUM_DEAL_TYPE    type;          // Open position type
   ENUM_DEAL_ENTRY   entry;         // Position entry type
   long              magic;         // Unique position number
   ENUM_DEAL_REASON  reason;        // Order placing reason
   long              ID;            // Position ID
   double            volume;        // Position volume (lots)
   double            price;         // Position entry price
   double            comission;     // Commission paid
   double            swap;          // Swap
   double            profit;        // Profit / loss
   string            symbol;        // Symbol
   string            comment;       // Comment specified when at opening
   string            ID_external;   // External ID
  };
```

The received data will be sorted by position open time and will not be grouped in any other way. This [article](https://www.mql5.com/en/articles/4803) contains examples, showing the difficulty of reading the report in this form, because confusion between trades may occur when trading multiple algorithms. Especially if you use position increase techniques which additionally buy or sell an asset according to the underlying algorithms. As a result, we get a bulk of entry and exit deals which do not reflect the complete pictures.

Our method groups these deals by positions. Although there is confusion with orders, we eliminate unnecessary deals which do not refer to the analyzed position. The result is saved as a structure storing an array from the deal structure shown above.

```
struct DealKeeper
  {
   DealData          deals[]; /* List of all deals for this position
                              (or several positions in case of position reversal)*/
   string            symbol;  // Symbol
   long              ID;      // ID of the position (s)
   datetime          DT_min;  // Open date (or the date of the very first position)
   datetime          DT_max;  // Close date
  };
```

Note that this class does not take into account Magic numbers in grouping, because when two or more algorithms trade on one position, they often intersect. At least full separation is technically impossible on Moscow Exchange, for which I mainly write algorithms. Also, the tool is designed to download trading results or testing/optimization results. In the first case, statistics on the selected symbol is enough, while for the second case the Magic number does not matter, because the strategy tester runs one algorithm at a time.

The implementation of the method core has not changed since the first article. Now we add custom commission to it. For this task, the CCCM class discussed above is passed by reference to the class constructor and it is saved in the corresponding field. Then, at the time of DealData structure filling, namely at the time of commission filling, the custom commission stored in the passed CCCM class is added.

```
#ifndef ONLY_CUSTOM_COMISSION
               if(data.comission==0 && comission_manager != NULL)
                 {
                  data.comission=comission_manager.get(data.symbol,data.price,data.volume);
                 }
#else
               data.comission=comission_manager.get(data.symbol,data.price,data.volume);
#endif
```

The commission is added directively and conditionally. If before connecting a file with this class in the robot we define the ONLY\_CUSTOM\_COMISSION parameter, the commission field will always contain the passed commission instead of the broker provided value. If this parameter is not defined, the passed commission will be added conditionally: only if the broker does not provide it with the quotes. In all other cases the user commission value will be ignored.

- getIDArr — returns an array of the IDs of positions which were opened for all symbols during the requested time frame. Position IDs enable the combination of all deals into positions in our method. Actually, this is a unique list of the DealData.ID field.

- getDealsDetales — the method is similar to getHistory, however, it provides less details. The idea of the method is to provide a table of positions in an easy-to-read form, in which each row corresponds to one specific deal. Each position is described by the following structure:



```
struct DealDetales
    {
     string            symbol;        // Symbol
     datetime          DT_open;       // Open date
     ENUM_DAY_OF_WEEK  day_open;      // Open day
     datetime          DT_close;      // Cloe date
     ENUM_DAY_OF_WEEK  day_close;     // Close day
     double            volume;        // Volume (lots)
     bool              isLong;        // Long/Short
     double            price_in;      // Position entry price
     double            price_out;     // Position exit price
     double            pl_oneLot;     // Profit / loss is trading one lot
     double            pl_forDeal;    // Real profit/loss taking into account commission
     string            open_comment;  // Comment at the time of opening
     string            close_comment; // Comment at the time of closing
    };
```

They represent a table of positions sorted by position closing dates. The array of these values will be used to calculate coefficients in the next class. Also, we will receive the final testing report based on the presented data. Furthermore, based on such data the tester creates the PL graph line after trading.



As for the tester, note that in further calculations, the Recovery Factor calculated by the terminal will differ from that calculated based on received data. This is due to the fact that although the data downloading is correct and calculation formulas are the same, source data are different. The tester calculates the recovery factor using the green line, i.e. the detailed report, while we will calculate it using the blue line, i.e. data ignoring price fluctuations which occur between position opening and closing.

- getBalance — this method is designed to obtain balance data not taking into account trading operations on the specified date.



```
double CDealHistoryGetter::getBalance(datetime toDate)
    {
     if(HistorySelect(0,(toDate>0 ? toDate : TimeCurrent())))
       {
        int total=HistoryDealsTotal(); // Get the total number of positions
        double balance=0;
        for(int i=0; i<total; i++)
          {
           long ticket=(long)HistoryDealGetTicket(i);

           ENUM_DEAL_TYPE dealType=(ENUM_DEAL_TYPE)HistoryDealGetInteger(ticket,DEAL_TYPE);
           if(dealType==DEAL_TYPE_BALANCE ||
              dealType == DEAL_TYPE_CORRECTION ||
              dealType == DEAL_TYPE_COMMISSION)
             {
              balance+=HistoryDealGetDouble(ticket,DEAL_PROFIT);

              if(toDate<=0)
                 break;
             }
          }
        return balance;
       }
     else
        return 0;
    }
```


To achieve the task, the history of all deals from the very first time interval to the specified one is requested first. After that the balance is saved in a cycle, while all deposits and withdrawals are added to the original balance, taking into account commission and corrections provided by the broker. If a zero date was passed as an input, then only the balance as of the very first date was requested.

- getBalanceWithPL — the method is similar to the previous one, but in addition to balance changes it takes into account profit/loss of performed operations, including commissions according to the aforementioned principle.

### Class creating the optimization report — Structures used in calculations

Another class which was already mentioned in previous articles is CReportCreator. It was briefly described in the article [100 Best Optimization Passes](https://www.mql5.com/en/articles/5214) under section "Calculation Part". Now it is time to provide a more detailed description, because this class calculates all coefficients, based on which the auto optimizer will decide whether this combination of algorithm parameters corresponds to the requested criteria.

Let is firsts described the basic idea of the approach used in class implementation. A similar class with less functional possibilities was implemented in my [first article](https://www.mql5.com/en/articles/4803). But it was very slow, because in order to calculate the next group of requested parameters or the next chart, it had to download all the trading history anew and loop through it. This was done at each parameter request.

Sometimes, in case of too many data, the approach can take several seconds. To accelerate the calculations. I used another class implementation, which additionally provides much more data (including those not available in standard optimization results). You may notice that similar data are needed for the calculation of many coefficients, such as, for example, maximum profit/loss or accumulated profit/loss and the like.

Therefore, by calculating the coefficients in one loop and saving them in the class fields, we can further apply this data for calculating all other parameters in which these data are need. Thus, we obtain a class which loops once through the downloaded history, calculates all the required parameters and stores them till the next calculation. When we then need to obtain the required parameter, the class copies the saved data instead of recalculating it, which greatly speeds up operation.

Now let us view how the parameters are calculated. Let's start with the objects that store data used for further calculations. These objects are created as nested class objects declared in private scope. This is done for two reasons. First, to prevent their use in other classes which will use this functionality. The large number of declared structures and classes is confusing: some of them are needed for external calculations, others are technical, i.e. used for internal calculations. And thus, the second reason is to emphasize their purely technical purpose.

The PL\_Keeper structure:

```
struct PL_keeper
{
 PLChart_item      PL_total[];
 PLChart_item      PL_oneLot[];
 PLChart_item      PL_Indicative[];
};
```

This structure is created for storing all possible profit and loss graphs. They were described in detail in my first article (see the link above). Below the structure declaration, its instances are created:

```
PL_keeper         PL,PL_hist,BH,BH_hist;
```

Each instance stores 4 presented chart types for different source data. Data with the PL prefix are calculated based on the earlier mentioned blue line of the PL graph available in the terminal. Data with the BH prefix are calculated based on the profit and loss graph obtained by the Buy and Hold strategy. Data with the 'hist' postfix are calculated based on the profit and loss histogram.

DailyPL\_keeper structure:

```
// The structure of Daily PL graphs
struct DailyPL_keeper
{
 DailyPL           avarage_open,avarage_close,absolute_open,absolute_close;
};
```

This structure stores four possible daily profit/loss graph types. DailyPL structure instances with the 'average' prefix are calculated using average profit/loss data. Those with the 'absolute' prefix use total profit and loss values. Accordingly, differences between them are obvious. In the first case it reflects average daily profit for the entire trading period, in the second case the total profit is shown. Data of the 'open' prefix are sorted by days according to their opening date, while data with the 'close' prefix are sorted according to their closing date. The structure instance declaration is shown in the below code.

The RationTable\_keeper keeper:

```
// Table structure of extreme points
struct RatioTable_keeper
  {
   ProfitDrawdown    Total_max,Total_absolute,Total_percent;
   ProfitDrawdown    OneLot_max,OneLot_absolute,OneLot_percent;
  };
```

This structure consists of instances of the ProfitDrawdown structure.

```
struct ProfitDrawdown
  {
   double            Profit; // In some cases Profit, in other Profit / Loss
   double            Drawdown; // Drawdown
  };
```

It stores the profit and loss ratio according to certain criteria. Data with the 'Total' prefix are calculated using the profit/loss graph build taking into account lot changes. Data with 'OneLot' prefix are calculated as if one lot was traded all the time. The non-standard one-lot calculation idea is described in the aforementioned first article. In short, this method was created to evaluate the results of the trading system. It allows evaluating where the most result comes from: timely lot management or from the logic of the system itself. The 'max' postfix shows that the instance features data on the highest profit and drawdown encountered during the trading history. The 'absolute' postfix means that the instance contains total profit and drawdown data for the entire trading history. The 'percent' postfix means that profit and drawdown values are calculated as a percentage ratio to the maximum value on the PL curve within the tested time frame. The structure declaration is simple and is shown in the code attached to the article.

The next group of structures is not declared as a class field but is used as a local declaration in the main Create method. All the described structures are combined together, so let us view the declaration of all of them.

```
// Structures for calculating consecutive profits and losses
   struct S_dealsCounter
     {
      int               Profit,DD;
     };
   struct S_dealsInARow : public S_dealsCounter
     {
      S_dealsCounter    Counter;
     };
   // Structures for calculating auxiliary data
   struct CalculationData_item
     {
      S_dealsInARow     dealsCounter;
      int               R_arr[];
      double            DD_percent;
      double            Accomulated_DD,Accomulated_Profit;
      double            PL;
      double            Max_DD_forDeal,Max_Profit_forDeal;
      double            Max_DD_byPL,Max_Profit_byPL;
      datetime          DT_Max_DD_byPL,DT_Max_Profit_byPL;
      datetime          DT_Max_DD_forDeal,DT_Max_Profit_forDeal;
      int               Total_DD_numDeals,Total_Profit_numDeals;
     };
   struct CalculationData
     {
      CalculationData_item total,oneLot;
      int               num_deals;
      bool              isNot_firstDeal;
     };
```

The S\_dealsCounter and S\_dealsInARow structures are essentially a single entity. Such a strange combination of association and inheritance at the same time is connected with the specific calculation of its parameters. The S\_dealsInARow structure is created for storing and calculating the number of trades (actually, for calculating positions, i.e. from position opening to closing) in a row, either positive or negative. The nested instance of the S\_dealsCounter structure is declared for storing intermediate calculation results. Inherited fields store totals. We will get back to the operation counting profitable/losing deals later.

The CalculationData\_item structure contains fields required for calculating coefficients.

- R\_arr — array of consecutive profitable/losing deals series, shown as 1 / 0, respectively. The array is used for Z score calculation.
- DD\_percent — drawdown percentage.
- Accomulated\_DD, Accomulated\_Profit  — store total loss and profit values.
- PL — profit / loss.
- Max\_DD\_forDeal, Max\_Profit\_forDeal — as naming suggests, they store maximum drawdown and profit among all deals.
- Max\_DD\_byPL, Mаx\_Profit\_byPL — store maximum drawdown and profit calculated by PL graph.
- DT\_Max\_DD\_byPL, DT\_Max\_Profit\_byPL — store dates of highest drawdown and profit by PL graph.
- DT\_Max\_DD\_forDeal, DT\_Max\_Profit\_forDeal — dates of highest drawdown and profit by deals.
- Total\_DD\_numDeals, TotalProfit\_numDeals — total number of profitable and losing trades.

Further calculations are based on the above data.

CalculationData is an accumulating structure which combines all the described structures. It stores all the required data. It also contains the num\_deals field, which is actually the sum of CalculationData\_item::Total\_DD\_numDeals and CalculationData\_item::TotalProfit\_numDeals. The sNot\_firstDeal field is a technical flag which denotes that the calculation is performed not for the very first deal.

The CoefChart\_keeper structure:

```
struct CoefChart_keeper
     {
      CoefChart_item    OneLot_ShartRatio_chart[],Total_ShartRatio_chart[];
      CoefChart_item    OneLot_WinCoef_chart[],Total_WinCoef_chart[];
      CoefChart_item    OneLot_RecoveryFactor_chart[],Total_RecoveryFactor_chart[];
      CoefChart_item    OneLot_ProfitFactor_chart[],Total_ProfitFactor_chart[];
      CoefChart_item    OneLot_AltmanZScore_chart[],Total_AltmanZScore_chart[];
     };
```

It is intended to store coefficient charts. Since the class creates not only profit and lot graphs, but also some coefficient charts, another structure was created for the described data types. Prefix 'OneLot' shows that the instance will store data received from the profit/loss analysis if trading one lot. 'Total' means calculation taking into account lot management. If no lot management is used in the strategy, the two charts will be identical.

The СHistoryComparer class:

Similarly, a class to be used in data sorting is defined. The article "100 Best Optimization Passes" contains the description of the CGenericSorter class, which can sort any data type in descending and ascending order. It additionally needs a class which can compare passed types. Such a class is СHisoryComparer.

```
class CHistoryComparer : public ICustomComparer<DealDetales>
     {
   public:
      int               Compare(DealDetales &x,DealDetales &y);
     };
```

The method implementation is simple: it compares close dates as sorting is performed by close dates:

```
int CReportCreator::CHistoryComparer::Compare(DealDetales &x,DealDetales &y)
  {
   return(x.DT_close == y.DT_close ? 0 : (x.DT_close > y.DT_close ? 1 : -1));
  }
```

Also there is a similar class for sorting coefficient charts. These two classes and the sorter class are instantiated as a global field of the described CReportCreator class. In addition to the described objects, there are two other fields. Their types are described as separate not nested objects:

```
PL_detales        PL_detales_data;
DistributionChart OneLot_PDF_chart,Total_PDF_chart;
```

The PL\_detales structure contains brief trading information for profitable and losing positions:

```
//+------------------------------------------------------------------+
struct PL_detales_PLDD
  {
   int               orders; // Number of deals
   double            orders_in_Percent; // Number of orders as % of total number of orders
   int               dealsInARow; // Deals in a row
   double            totalResult; // Total result in money
   double            averageResult; // Average result in money
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
struct PL_detales_item
  {
   PL_detales_PLDD   profit; // Information on profitable deals
   PL_detales_PLDD   drawdown; // Information on losing deals
  };
//+-------------------------------------------------------------------+
//| A brief PL graph summary divided into 2 main blocks               |
//+-------------------------------------------------------------------+
struct PL_detales
  {
   PL_detales_item   total,oneLot;
  };
```

The second structure DistributionChart contains a number of VaR values as well as the distribution chart based on which these coefficients were calculated. Distribution is calculated as normal distribution.

```
//+------------------------------------------------------------------+
//| Structure used for saving distribution charts                    |
//+------------------------------------------------------------------+
struct Chart_item
  {
   double            y; // y axis
   double            x; // x axis
  };
//+------------------------------------------------------------------+
//| Structure contains the VaR value                                 |
//+------------------------------------------------------------------+
struct VAR
  {
   double            VAR_90,VAR_95,VAR_99;
   double            Mx,Std;
  };
//+------------------------------------------------------------------+
//| Structure - it is used to store distribution charts and          |
//| the VaR values                                                   |
//+------------------------------------------------------------------+
struct Distribution_item
  {
   Chart_item        distribution[]; // Distribution chart
   VAR               VaR; // VaR
  };
//+------------------------------------------------------------------+
//| Structure - Stores distribution data. Divided into 2 blocks      |
//+------------------------------------------------------------------+
struct DistributionChart
  {
   Distribution_item absolute,growth;
  };
```

The VaR coefficients are calculated according to a formula: Historical VaR, which might not be accurate enough, but it is quite suitable for the current implementation.

### Methods for calculating coefficients that describe trading results

Now that we have considered the data storing structures, you can imagine the huge amount of statistics calculated by this class. Let us view the specific methods for calculating the described values one by one, as they are named in the CReportCreator class.

The CalcPL is created for calculating the PL graph. It is implemented as follows:

```
void CReportCreator::CalcPL(const DealDetales &deal,CalculationData &data,PLChart_item &pl_out[],CalcType type)
  {
   PLChart_item item;
   ZeroMemory(item);
   item.DT=deal.DT_close; // Saving the date

   if(type!=_Indicative)
     {
      item.Profit=(type==_Total ? data.total.PL : data.oneLot.PL); // Saving the profit
      item.Drawdown=(type==_Total ? data.total.DD_percent : data.oneLot.DD_percent); // Saving the drawdown
     }
   else // Calculating the indicative chart
     {
      if(data.isNot_firstDeal)
        {
         if(data.total.PL!=0)
           {
            if(data.total.PL > 0 && data.total.Max_DD_forDeal < 0)
               item.Profit=data.total.PL/MathAbs(data.total.Max_DD_forDeal);
            else
               if(data.total.PL<0 && data.total.Max_Profit_forDeal>0)
                  item.Profit=data.total.PL/data.total.Max_Profit_forDeal;
           }
        }
     }
// Adding data to array
   int s=ArraySize(pl_out);
   ArrayResize(pl_out,s+1,s+1);
   pl_out[s]=item;
  }
```

As seen from the implementation, all its calculations are based on data of earlier described structures which are passed as an input.

If you need to calculate a non-indicative PL graph, simply copy the known data. Otherwise the calculation is subject to two conditions: the first iteration was not found in the cycle and the PL graph is non-zero. The calculation is performed according to the following logic:

- If PL is greater than zero and drawdown is less, divide the current PL value by the drawdown value. Thus, we obtain a coefficient indicating how many consecutive maximum drawdowns are required to reduce the current PL to zero.
- If PL is less than zero and the maximum profit for all deals is greater than zero, then we divide the PL value (which is currently the drawdown) by the maximum profit achieved. Thus, we obtain a coefficient showing how many maximum profits in a row would be required to bring the current drawdown to zero.

The next method CalcPLHist is based on a similar mechanism but it uses other structures fields for calculation: data.oneLot.Accomulated\_DD, data.total.Accomulated\_DD and data.oneLot.Accomulated\_Profit, data.total.Accomulated\_Profit. We have already considered its algorithm earlier, therefore, let's move on to the next two methods.

CalcData and CalcData\_item:

These methods calculate all auxiliary and main coefficients. Let's begin with the CalcData\_item. Its purpose is to calculate additional coefficients described above, based on which the main coefficients are calculated.

```
//+------------------------------------------------------------------+
//| Calculating auxiliary data                                       |
//+------------------------------------------------------------------+
void CReportCreator::CalcData_item(const DealDetales &deal,CalculationData_item &out,
                                   bool isOneLot)
  {
   double pl=(isOneLot ? deal.pl_oneLot : deal.pl_forDeal); //PL
   int n=0;
// Number of profits and losses
   if(pl>=0)
     {
      out.Total_Profit_numDeals++;
      n=1;
      out.dealsCounter.Counter.DD=0;
      out.dealsCounter.Counter.Profit++;
     }
   else
     {
      out.Total_DD_numDeals++;
      out.dealsCounter.Counter.DD++;
      out.dealsCounter.Counter.Profit=0;
     }
   out.dealsCounter.DD=MathMax(out.dealsCounter.DD,out.dealsCounter.Counter.DD);
   out.dealsCounter.Profit=MathMax(out.dealsCounter.Profit,out.dealsCounter.Counter.Profit);

// Series of profits and losses
   int s=ArraySize(out.R_arr);
   if(!(s>0 && out.R_arr[s-1]==n))
     {
      ArrayResize(out.R_arr,s+1,s+1);
      out.R_arr[s]=n;
     }

   out.PL+=pl; //Total PL
// Max Profit / DD
   if(out.Max_DD_forDeal>pl)
     {
      out.Max_DD_forDeal=pl;
      out.DT_Max_DD_forDeal=deal.DT_close;
     }
   if(out.Max_Profit_forDeal<pl)
     {
      out.Max_Profit_forDeal=pl;
      out.DT_Max_Profit_forDeal=deal.DT_close;
     }
// Accumulated Profit / DD
   out.Accomulated_DD+=(pl>0 ? 0 : pl);
   out.Accomulated_Profit+=(pl>0 ? pl : 0);
// Extreme profit values
   double maxPL=MathMax(out.Max_Profit_byPL,out.PL);
   if(compareDouble(maxPL,out.Max_Profit_byPL)==1/* || !isNot_firstDeal*/)// another check is needed to save the date
     {
      out.DT_Max_Profit_byPL=deal.DT_close;
      out.Max_Profit_byPL=maxPL;
     }
   double maxDD=out.Max_DD_byPL;
   double DD=0;
   if(out.PL>0)
      DD=out.PL-maxPL;
   else
      DD=-(MathAbs(out.PL)+maxPL);
   maxDD=MathMin(maxDD,DD);
   if(compareDouble(maxDD,out.Max_DD_byPL)==-1/* || !isNot_firstDeal*/)// another check is needed to save the date
     {
      out.Max_DD_byPL=maxDD;
      out.DT_Max_DD_byPL=deal.DT_close;
     }
   out.DD_percent=(balance>0 ?(MathAbs(DD)/(maxPL>0 ? maxPL : balance)) :(maxPL>0 ?(MathAbs(DD)/maxPL) : 0));
  }
```

Firstly, PL is calculated at the i-th iteration. Then, if there was profit at this iteration, increase the profitable deal counter and zero the counter of consecutive losses. Also, set value 1 for the n variable, which means that the deal was profitable. If PL was below zero, increase loss counters and zero the profitable deal counter. After that assign the maximum number of profitable and losing series in a row.

The next step is to calculate the series of profitable and losing deals. A series means consecutive winning or losing deals. In this array, zero is always followed by one, while one is always followed by zero. This shows the alternation of winning and losing trades, however 0 or 1 can mean multiple deals. This array will be used for calculating the Z score which shows the degree of trading randomness. The next step is to assign maximum profit/drawdown values and to calculated accumulated profit/loss. At the end of this method, extreme points are calculated, i.e. structures with the maximum profit and loss values are filled.

The CalcData data already uses the obtained intermediate data to calculate the required coefficients and updates calculations at each iteration. It is implemented as follows:

```
void CReportCreator::CalcData(const DealDetales &deal,CalculationData &out,bool isBH)
  {
   out.num_deals++; // Counting the number of deals
   CalcData_item(deal,out.oneLot,true);
   CalcData_item(deal,out.total,false);

   if(!isBH)
     {
      // Fill PL graphs
      CalcPL(deal,out,PL.PL_total,_Total);
      CalcPL(deal,out,PL.PL_oneLot,_OneLot);
      CalcPL(deal,out,PL.PL_Indicative,_Indicative);

      // Fill PL Histogram graphs
      CalcPLHist(deal,out,PL_hist.PL_total,_Total);
      CalcPLHist(deal,out,PL_hist.PL_oneLot,_OneLot);
      CalcPLHist(deal,out,PL_hist.PL_Indicative,_Indicative);

      // Fill PL graphs by days
      CalcDailyPL(DailyPL_data.absolute_close,CALC_FOR_CLOSE,deal);
      CalcDailyPL(DailyPL_data.absolute_open,CALC_FOR_OPEN,deal);
      CalcDailyPL(DailyPL_data.avarage_close,CALC_FOR_CLOSE,deal);
      CalcDailyPL(DailyPL_data.avarage_open,CALC_FOR_OPEN,deal);

      // Fill Profit Factor graphs
      ProfitFactor_chart_calc(CoefChart_data.OneLot_ProfitFactor_chart,out,deal,true);
      ProfitFactor_chart_calc(CoefChart_data.Total_ProfitFactor_chart,out,deal,false);

      // Fill Recovery Factor graphs
      RecoveryFactor_chart_calc(CoefChart_data.OneLot_RecoveryFactor_chart,out,deal,true);
      RecoveryFactor_chart_calc(CoefChart_data.Total_RecoveryFactor_chart,out,deal,false);

      // Fill winning coefficient graphs
      WinCoef_chart_calc(CoefChart_data.OneLot_WinCoef_chart,out,deal,true);
      WinCoef_chart_calc(CoefChart_data.Total_WinCoef_chart,out,deal,false);

      // Fill Sharpe Ration graphs
      ShartRatio_chart_calc(CoefChart_data.OneLot_ShartRatio_chart,PL.PL_oneLot,deal/*,out.isNot_firstDeal*/);
      ShartRatio_chart_calc(CoefChart_data.Total_ShartRatio_chart,PL.PL_total,deal/*,out.isNot_firstDeal*/);

      // Fill Z Score graphs
      AltmanZScore_chart_calc(CoefChart_data.OneLot_AltmanZScore_chart,(double)out.num_deals,
                              (double)ArraySize(out.oneLot.R_arr),(double)out.oneLot.Total_Profit_numDeals,
                              (double)out.oneLot.Total_DD_numDeals/*,out.isNot_firstDeal*/,deal);
      AltmanZScore_chart_calc(CoefChart_data.Total_AltmanZScore_chart,(double)out.num_deals,
                              (double)ArraySize(out.total.R_arr),(double)out.total.Total_Profit_numDeals,
                              (double)out.total.Total_DD_numDeals/*,out.isNot_firstDeal*/,deal);
     }
   else // Fill PL Buy and Hold graphs
     {
      CalcPL(deal,out,BH.PL_total,_Total);
      CalcPL(deal,out,BH.PL_oneLot,_OneLot);
      CalcPL(deal,out,BH.PL_Indicative,_Indicative);

      CalcPLHist(deal,out,BH_hist.PL_total,_Total);
      CalcPLHist(deal,out,BH_hist.PL_oneLot,_OneLot);
      CalcPLHist(deal,out,BH_hist.PL_Indicative,_Indicative);
     }

   if(!out.isNot_firstDeal)
      out.isNot_firstDeal=true; // Flag "It is NOT the first deal"
  }
```

Firstly, intermediate coefficients are calculated for one lot and managed lot trading systems, by calling the described method for both data types. Then calculation is split into coefficients for BH and opposite-type data. Interpretable coefficients are calculated inside each block. Only graphs are calculated for the Buy and Hold strategy, and thus coefficient calculating methods are not called.

The next group of methods calculates profit/loss split by days:

```
//+------------------------------------------------------------------+
//| Create a structure of trading during a day                       |
//+------------------------------------------------------------------+
void CReportCreator::CalcDailyPL(DailyPL &out,DailyPL_calcBy calcBy,const DealDetales &deal)
  {
   cmpDay(deal,MONDAY,out.Mn,calcBy);
   cmpDay(deal,TUESDAY,out.Tu,calcBy);
   cmpDay(deal,WEDNESDAY,out.We,calcBy);
   cmpDay(deal,THURSDAY,out.Th,calcBy);
   cmpDay(deal,FRIDAY,out.Fr,calcBy);
  }
//+------------------------------------------------------------------+
//| Save resulting PL/DD for the day                                 |
//+------------------------------------------------------------------+
void CReportCreator::cmpDay(const DealDetales &deal,ENUM_DAY_OF_WEEK etalone,PLDrawdown &ans,DailyPL_calcBy calcBy)
  {
   ENUM_DAY_OF_WEEK day=(calcBy==CALC_FOR_CLOSE ? deal.day_close : deal.day_open);
   if(day==etalone)
     {
      if(deal.pl_forDeal>0)
        {
         ans.Profit+=deal.pl_forDeal;
         ans.numTrades_profit++;
        }
      else
         if(deal.pl_forDeal<0)
           {
            ans.Drawdown+=MathAbs(deal.pl_forDeal);
            ans.numTrades_drawdown++;
           }
     }
  }
//+------------------------------------------------------------------+
//| Average resulting PL/DD for the day                              |
//+------------------------------------------------------------------+
void CReportCreator::avarageDay(PLDrawdown &day)
  {
   if(day.numTrades_profit>0)
      day.Profit/=day.numTrades_profit;
   if(day.numTrades_drawdown > 0)
      day.Drawdown/=day.numTrades_drawdown;
  }
```

The main work splitting profit/DD by days is performed in the cmpDay method, which first checks whether the day corresponds to the requested day or not, and then adds profit and loss values. Losses are summed modulo. CalcDailyPL is an aggregating method, in which an attempt is made to add the current passed PL to one of the five working days. The avarageDay method is called to average profits/losses in the main Create method. This method does not perform any specific actions, while it only calculates the average based on the earlier calculated absolute profit/loss values.

Profit Factor calculating method

```
//+------------------------------------------------------------------+
//| Calculate Profit Factor                                          |
//+------------------------------------------------------------------+
void CReportCreator::ProfitFactor_chart_calc(CoefChart_item &out[],CalculationData &data,const DealDetales &deal,bool isOneLot)
  {
   CoefChart_item item;
   item.DT=deal.DT_close;
   double profit=(isOneLot ? data.oneLot.Accomulated_Profit : data.total.Accomulated_Profit);
   double dd=MathAbs(isOneLot ? data.oneLot.Accomulated_DD : data.total.Accomulated_DD);
   if(dd==0)
      item.coef=0;
   else
      item.coef=profit/dd;
   int s=ArraySize(out);
   ArrayResize(out,s+1,s+1);
   out[s]=item;
  }
```

The method calculates a graph reflecting the change of Profit Factor throughout trading. The very last value is the one shown in the testing report. The formula is simple = accumulated profit / accumulated drawdown. If the drawdown is zero, then the coefficient will be equal to zero, since in classical arithmetic it is impossible to divide by zero without using limits, and the same rule applies in the language. Therefore, we will add divisor checks for all arithmetic operations.

The Recovery Factor calculation principle is similar:

```
//+------------------------------------------------------------------+
//| Calculate Recovery Factor                                        |
//+------------------------------------------------------------------+
void CReportCreator::RecoveryFactor_chart_calc(CoefChart_item &out[],CalculationData &data,const DealDetales &deal,bool isOneLot)
  {
   CoefChart_item item;
   item.DT=deal.DT_close;
   double pl=(isOneLot ? data.oneLot.PL : data.total.PL);
   double dd=MathAbs(isOneLot ? data.oneLot.Max_DD_byPL : data.total.Max_DD_byPL);
   if(dd==0)
      item.coef=0;//ideally it should be plus infinity
   else
      item.coef=pl/dd;
   int s=ArraySize(out);
   ArrayResize(out,s+1,s+1);
   out[s]=item;
  }
```

Coefficient calculation formula: profit as at the i-th iteration / drawdown as at the i-th iteration. Also note that since the profit can be zero or negative during the coefficient calculation, the coefficient itself can be zero or negative.

Win Rate

```
//+------------------------------------------------------------------+
//| Calculate Win Rate                                               |
//+------------------------------------------------------------------+
void CReportCreator::WinCoef_chart_calc(CoefChart_item &out[],CalculationData &data,const DealDetales &deal,bool isOneLot)
  {
   CoefChart_item item;
   item.DT=deal.DT_close;
   double profit=(isOneLot ? data.oneLot.Accomulated_Profit : data.total.Accomulated_Profit);
   double dd=MathAbs(isOneLot ? data.oneLot.Accomulated_DD : data.total.Accomulated_DD);
   int n_profit=(isOneLot ? data.oneLot.Total_Profit_numDeals : data.total.Total_Profit_numDeals);
   int n_dd=(isOneLot ? data.oneLot.Total_DD_numDeals : data.total.Total_DD_numDeals);
   if(n_dd == 0 || n_profit == 0)
      item.coef = 0;
   else
      item.coef=(profit/n_profit)/(dd/n_dd);
   int s=ArraySize(out);
   ArrayResize(out,s+1,s+1);
   out[s]=item;
  }
```

Win Rate calculation formula = (profit / number of profitable trades) / (drawdown / number of losing trades).  This coefficient can also be negative, if there is no profit at the time of calculation.

The calculation of Sharpe Ratio is a bit more complicated:

```
//+------------------------------------------------------------------+
//| Calculate Sharpe Ratio                                           |
//+------------------------------------------------------------------+
double CReportCreator::ShartRatio_calc(PLChart_item &data[])
  {
   int total=ArraySize(data);
   double ans=0;
   if(total>=2)
     {
      double pl_r=0;
      int n=0;
      for(int i=1; i<total; i++)
        {
         if(data[i-1].Profit!=0)
           {
            pl_r+=(data[i].Profit-data[i-1].Profit)/data[i-1].Profit;
            n++;
           }
        }
      if(n>=2)
         pl_r/=(double)n;
      double std=0;
      n=0;
      for(int i=1; i<total; i++)
        {
         if(data[i-1].Profit!=0)
           {
            std+=MathPow((data[i].Profit-data[i-1].Profit)/data[i-1].Profit-pl_r,2);
            n++;
           }
        }
      if(n>=2)
         std=MathSqrt(std/(double)(n-1));

      ans=(std!=0 ?(pl_r-r)/std : 0);
     }
   return ans;
  }
```

In the first cycle, an average profitability is calculated by the PL graph, in which each i-th profitability is calculated as the ratio of increase over PL to the previous PL value. The calculation is based on the example of price series normalization used for the evaluation of time series.

In the next cycle, volatility is calculated using the same normalized profitability series.

After that the coefficient itself is calculated using the formula (average profit - risk-free rate) / volatility (standard deviation of returns).

Maybe I applied a non-traditional approach in series normalization and probably even the formula, but this calculation seems pretty reasonable. If you find any error, please add a comment to the article.

Calculating VaR and normal distribution graph. This part consists of three methods. Two of them are calculating, the third one aggregates all calculations. Let's consider these methods.

```
//+------------------------------------------------------------------+
//| Distribution calculation                                         |
//+------------------------------------------------------------------+
void CReportCreator::NormalPDF_chart_calc(DistributionChart &out,PLChart_item &data[])
  {
   double Mx_absolute=0,Mx_growth=0,Std_absolute=0,Std_growth=0;
   int total=ArraySize(data);
   ZeroMemory(out.absolute);
   ZeroMemory(out.growth);
   ZeroMemory(out.absolute.VaR);
   ZeroMemory(out.growth.VaR);
   ArrayFree(out.absolute.distribution);
   ArrayFree(out.growth.distribution);

// Calculation of distribution parameters
   if(total>=2)
     {
      int n=0;
      for(int i=0; i<total; i++)
        {
         Mx_absolute+=data[i].Profit;
         if(i>0 && data[i-1].Profit!=0)
           {
            Mx_growth+=(data[i].Profit-data[i-1].Profit)/data[i-1].Profit;
            n++;
           }
        }
      Mx_absolute/=(double)total;
      if(n>=2)
         Mx_growth/=(double)n;

      n=0;
      for(int i=0; i<total; i++)
        {
         Std_absolute+=MathPow(data[i].Profit-Mx_absolute,2);
         if(i>0 && data[i-1].Profit!=0)
           {
            Std_growth+=MathPow((data[i].Profit-data[i-1].Profit)/data[i-1].Profit-Mx_growth,2);
            n++;
           }
        }
      Std_absolute=MathSqrt(Std_absolute/(double)(total-1));
      if(n>=2)
         Std_growth=MathSqrt(Std_growth/(double)(n-1));

      // Calculate VaR
      out.absolute.VaR.Mx=Mx_absolute;
      out.absolute.VaR.Std=Std_absolute;
      out.absolute.VaR.VAR_90=VaR(Q_90,Mx_absolute,Std_absolute);
      out.absolute.VaR.VAR_95=VaR(Q_95,Mx_absolute,Std_absolute);
      out.absolute.VaR.VAR_99=VaR(Q_99,Mx_absolute,Std_absolute);
      out.growth.VaR.Mx=Mx_growth;
      out.growth.VaR.Std=Std_growth;
      out.growth.VaR.VAR_90=VaR(Q_90,Mx_growth,Std_growth);
      out.growth.VaR.VAR_95=VaR(Q_95,Mx_growth,Std_growth);
      out.growth.VaR.VAR_99=VaR(Q_99,Mx_growth,Std_growth);

      // Calculate distribution
      for(int i=0; i<total; i++)
        {
         Chart_item  item_a,item_g;
         ZeroMemory(item_a);
         ZeroMemory(item_g);
         item_a.x=data[i].Profit;
         item_a.y=PDF_calc(Mx_absolute,Std_absolute,data[i].Profit);
         if(i>0)
           {
            item_g.x=(data[i-1].Profit != 0 ?(data[i].Profit-data[i-1].Profit)/data[i-1].Profit : 0);
            item_g.y=PDF_calc(Mx_growth,Std_growth,item_g.x);
           }
         int s=ArraySize(out.absolute.distribution);
         ArrayResize(out.absolute.distribution,s+1,s+1);
         out.absolute.distribution[s]=item_a;
         s=ArraySize(out.growth.distribution);
         ArrayResize(out.growth.distribution,s+1,s+1);
         out.growth.distribution[s]=item_g;
        }
      // Ascending
      sorter.Sort<Chart_item>(out.absolute.distribution,&chartComparer);
      sorter.Sort<Chart_item>(out.growth.distribution,&chartComparer);
     }
  }
//+------------------------------------------------------------------+
//| Calculate VaR                                                    |
//+------------------------------------------------------------------+
double CReportCreator::VaR(double quantile,double Mx,double Std)
  {
   return Mx-quantile*Std;
  }
//+------------------------------------------------------------------+
//| Distribution calculation                                         |
//+------------------------------------------------------------------+
double CReportCreator::PDF_calc(double Mx,double Std,double x)
  {
   if(Std!=0)
      return MathExp(-0.5*MathPow((x-Mx)/Std,2))/(MathSqrt(2*M_PI)*Std);
   else
      return 0;
  }
```

VaR calculation method is the simplest one. It uses the historic VaR model in calculations.

The normalized distribution calculation method is the one available in the Matlab statistical analysis package.

The normalized distribution calculation and graph building method is an aggregating one, in which the above described methods are applied. In the first cycle, the average profit value is calculated. In the second cycle , the standard deviation of returns is calculated. The returns for the graph and VaR calculated by growth are also calculated as a normalized time series. Further, after filling VaR value, normal distribution graph is calculated using the above method. As the x axis, we use the profitability for the growth-based graph and the absolute profit values for the profit-based graph.

To calculate the Z score, I used a formula form one of the articles in this site. Its full implementation is available in attached files.

Please note that all calculations start with the Calculate method with the following call signature

```
void CReportCreator::Create(DealDetales &history[],DealDetales &BH_history[],const double _balance,const string &Symb[],double _r);
```

Its implementation was described in the previously mentioned article "100 Best Optimization Passes". All public methods do not perform any logical operations, but they serve as getters forming the requested data in accordance with input parameters, indicating the type of the required information.

### Conclusion

In the previous article, we considered the process of library development in the C# language. In this article, we moved to the next step — the creation of a trading report, which we can obtain using the created methods. The report generation mechanism was already considered in earlier articles. But is has been improved and revised. This article presents the latest versions of these developments. The offered solution was tested on various optimizations and testing processes.

Two folders are available in the attached archive. Unzip both of them to MQL/Include directory.

The following files are included in the attachment:

1. CustomGeneric

   - GenericSorter.mqh
   - ICustomComparer.mqh

3. History manager

   - CustomComissionManager.mqh
   - DealHistoryGetter.mqh
   - ReportCreator.mqh

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7452](https://www.mql5.com/ru/articles/7452)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7452.zip "Download all attachments in the single ZIP archive")

[Include.zip](https://www.mql5.com/en/articles/download/7452/include.zip "Download Include.zip")(23.27 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Continuous walk-forward optimization (Part 8): Program improvements and fixes](https://www.mql5.com/en/articles/7891)
- [Continuous Walk-Forward Optimization (Part 7): Binding Auto Optimizer's logical part with graphics and controlling graphics from the program](https://www.mql5.com/en/articles/7747)
- [Continuous Walk-Forward Optimization (Part 6): Auto optimizer's logical part and structure](https://www.mql5.com/en/articles/7718)
- [Continuous Walk-Forward Optimization (Part 5): Auto Optimizer project overview and creation of a GUI](https://www.mql5.com/en/articles/7583)
- [Continuous Walk-Forward Optimization (Part 4): Optimization Manager (Auto Optimizer)](https://www.mql5.com/en/articles/7538)
- [Continuous Walk-Forward Optimization (Part 3): Adapting a Robot to Auto Optimizer](https://www.mql5.com/en/articles/7490)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/332982)**
(2)


![Aliaksei Karalkou](https://c.mql5.com/avatar/avatar_na2.png)

**[Aliaksei Karalkou](https://www.mql5.com/en/users/alekkar)**
\|
12 Jan 2020 at 11:48

The article is great, but I have not even started [learning mql5](https://www.mql5.com/en/book "Book \"MQL5 Programming for Traders\""). Once I tried to do the same thing on mql4, but the idea failed, although I did not give up until the end. So the question is: is it possible to realise something similar there?


![Andrey Azatskiy](https://c.mql5.com/avatar/2018/6/5B127D58-708F.jpg)

**[Andrey Azatskiy](https://www.mql5.com/en/users/andreykrivcov)**
\|
12 Jan 2020 at 15:24

**Aliaksei Karalkou:**

The article is great, but I have not even started learning mql5. Once I tried to do the same thing on mql4, but the idea failed, although I did not give up until the end. So the question is: is it possible to implement something similar there ?

I think you can, but I try not to write in MQL4, I think it is better to work on the latest version of the [product](https://www.metatrader5.com/en/terminal/help/fundamental/economic_indicators_usa/usa_productivity "Productivity"), which is MQL5 at the moment.

![SQLite: Native handling of SQL databases in MQL5](https://c.mql5.com/2/37/database-mql5.png)[SQLite: Native handling of SQL databases in MQL5](https://www.mql5.com/en/articles/7463)

The development of trading strategies is associated with handling large amounts of data. Now, you are able to work with databases using SQL queries based on SQLite directly in MQL5. An important feature of this engine is that the entire database is placed in a single file located on a user's PC.

![Library for easy and quick development of MetaTrader programs (part XXVII): Working with trading requests - placing pending orders](https://c.mql5.com/2/37/MQL5-avatar-doeasy__15.png)[Library for easy and quick development of MetaTrader programs (part XXVII): Working with trading requests - placing pending orders](https://www.mql5.com/en/articles/7418)

In this article, we will continue the development of trading requests, implement placing pending orders and eliminate detected shortcomings of the trading class operation.

![Library for easy and quick development of MetaTrader programs (part XXVIII): Closure, removal and modification of pending trading requests](https://c.mql5.com/2/37/MQL5-avatar-doeasy__16.png)[Library for easy and quick development of MetaTrader programs (part XXVIII): Closure, removal and modification of pending trading requests](https://www.mql5.com/en/articles/7438)

This is the third article about the concept of pending requests. We are going to complete the tests of pending trading requests by creating the methods for closing positions, removing pending orders and modifying position and pending order parameters.

![Library for easy and quick development of MetaTrader programs (part XXVI): Working with pending trading requests - first implementation (opening positions)](https://c.mql5.com/2/37/MQL5-avatar-doeasy__14.png)[Library for easy and quick development of MetaTrader programs (part XXVI): Working with pending trading requests - first implementation (opening positions)](https://www.mql5.com/en/articles/7394)

In this article, we are going to store some data in the value of the orders and positions magic number and start the implementation of pending requests. To check the concept, let's create the first test pending request for opening market positions when receiving a server error requiring waiting and sending a repeated request.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/7452&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068197946639447735)

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