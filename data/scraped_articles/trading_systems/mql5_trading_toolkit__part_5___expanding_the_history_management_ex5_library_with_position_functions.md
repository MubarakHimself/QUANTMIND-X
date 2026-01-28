---
title: MQL5 Trading Toolkit (Part 5): Expanding the History Management EX5 Library with Position Functions
url: https://www.mql5.com/en/articles/16681
categories: Trading Systems, Integration
relevance_score: 3
scraped_at: 2026-01-23T18:44:09.516794
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/16681&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069698174420977992)

MetaTrader 5 / Examples


### Introduction

In the [previous article](https://www.mql5.com/en/articles/16528), we began developing the primary functions of the **_HistoryManager EX5 library_**, which forms the core engine responsible for retrieving, sorting, and categorizing historical data into various types, including orders, deals, pending orders, and positions. Most of these functions were designed to operate in the background, unnoticed by the library's users, and were not directly accessible. The only exportable functions were the print functions, which allowed users to output simple descriptive lists of orders, deals, pending orders, or positions to the MetaTrader 5 log.

In this article, we will expand the _HistoryManager.mq5_ source code by introducing additional user-accessible functions that build upon the foundational ones we created in the previous article. These new functions will allow library users to effortlessly query trade history data. Users will be able to retrieve key details, such as the trade duration in seconds, opening and closing deal tickets of the last closed position, whether the position was initiated by a pending order or a direct market entry, pip-based metrics like profit, stop loss, and take profit, as well as the net profit after accounting for expenses such as commissions and swaps. All of this will allow you to import the EX5 library in your MQL5 projects and be able to query various positions' history with minimal effort through straightforward function calls.

To get started, we will open the _HistoryManager.mq5_ file from the previous article and begin by creating the _GetTotalDataInfoSize()_ function. The initial _HistoryManager.mq5_ source file is attached at the end of the previous article or can also be found at the end of this article ( _HistoryManager\_Part1.mq5)_. We will continue adding new code below the _PrintPendingOrdersHistory()_ function, which is where we left off previously.

### Get Total Data Info Size Function

The _GetTotalDataInfoSize()_ function is designed to retrieve and return the size of a specified historical data array. This function works in tandem with the _FetchHistoryByCriteria()_ function, ensuring that we can dynamically determine the amount of data available in a specific data structure. Its primary role is to identify which dataset we are interested in—deals, orders, positions, or pending orders—and return the total number of elements in that dataset.

The _GetTotalDataInfoSize()_ function will help streamline operations where dynamic access to various historical datasets is required. By passing the appropriate criteria as an argument, we will efficiently query the size of the relevant data structure.

We will start by defining the function signature. Since this function returns an _integer_ representing the size of a specified data array, it uses an _int_ return type. The input parameter or argument is an unsigned integer ( _uint_), which will allow us to pass predefined constants to specify the type of data we are querying.

```
int GetTotalDataInfoSize(uint dataToGet)
  {
//-- Our function's code will go here
  }
```

Next, we will declare a local variable, _totalDataInfo_, which will store the size of the requested dataset.

```
int totalDataInfo = 0;
```

We will use a _switch_ statement to check the value of the _dataToGet_ parameter. Depending on its value, we will identify the corresponding dataset and use the _ArraySize()_ function to determine its size.

- **Deals History Data:** If _dataToGet_ equals _GET\_DEALS\_HISTORY\_DATA_, we will calculate the size of the _dealInfo_ array and save it to _totalDataInfo_.
- **Orders History Data**: If _dataToGet_ equals _GET\_ORDERS\_HISTORY\_DATA_, we will calculate the size of the _orderInfo_ array and save it to _totalDataInfo_.
- **Positions History Data**: For _GET\_POSITIONS\_HISTORY\_DATA_, we will calculate the size of the _positionInfo_ array and save it to _totalDataInfo_.
- **Pending Orders Data**: When _dataToGet_ matches _GET\_PENDING\_ORDERS\_HISTORY\_DATA_, we will determine the size of the _pendingOrderInfo_ array and save it to _totalDataInfo_.
- **Default Case**: If none of the predefined constants match, we will set _totalDataInfo_ to _0_ as a fallback.

Finally, we will return the value stored in _totalDataInfo_ after we exit the _switch_. This will ensure that the function outputs the correct size for the specified data type or _0_ if no valid data type is provided.

```
switch(dataToGet)
  {
   case GET_DEALS_HISTORY_DATA:
      totalDataInfo = ArraySize(dealInfo);
      break;

   case GET_ORDERS_HISTORY_DATA:
      totalDataInfo = ArraySize(orderInfo);
      break;

   case GET_POSITIONS_HISTORY_DATA:
      totalDataInfo = ArraySize(positionInfo);
      break;

   case GET_PENDING_ORDERS_HISTORY_DATA:
      totalDataInfo = ArraySize(pendingOrderInfo);
      break;

   default:
      totalDataInfo = 0;
      break;
  }
return(totalDataInfo);
```

Below is the full implementation of the _GetTotalDataInfoSize()_ function with all the code segments in their proper sequence.

```
int GetTotalDataInfoSize(uint dataToGet)
  {
   int totalDataInfo = 0; //- Saves the total elements of the specified history found
   switch(dataToGet)
     {
      case GET_DEALS_HISTORY_DATA: //- Check if we have any available deals history data
         totalDataInfo = ArraySize(dealInfo); //- Save the total deals found
         break;

      case GET_ORDERS_HISTORY_DATA: //- Check if we have any available orders history data
         totalDataInfo = ArraySize(orderInfo); //- Save the total orders found
         break;

      case GET_POSITIONS_HISTORY_DATA: //- Check if we have any available positions history data
         totalDataInfo = ArraySize(positionInfo); //- Save the total positions found
         break;

      case GET_PENDING_ORDERS_HISTORY_DATA: //- Check if we have any available pending orders history data
         totalDataInfo = ArraySize(pendingOrderInfo); //- Save the total pending orders found
         break;

      default: //-- Unknown entry
         totalDataInfo = 0;
         break;
     }
   return(totalDataInfo);
  }
```

### Fetch History by Criteria Function

When querying recent history data with MQL5, such as retrieving the last five closed buy positions for a specific symbol, it is unnecessary to request the entire account history from the server, as this would waste valuable resources. Instead, you should adopt an optimal approach by first querying the most recent trade history, for example, within the current day. If the specific data you're looking for is not available within this period, you can then incrementally extend the time range until you find the targeted data. This method ensures efficiency and minimizes resource usage, providing the best performance while retrieving the necessary historical data.

The _FetchHistoryByCriteria()_ function systematically retrieves historical data based on specific criteria, starting from the last 24 hours and expanding the retrieval period if necessary. It begins by checking the most recent data, and if no relevant history is found, it progressively scans older data—first by week, up to a year. If no data is found after scanning the entire year, it will attempt to retrieve all available account history.

The _FetchHistoryByCriteria()_ function will serve as a key utility for fetching trade data from different periods and will allow the EX5 library to scan and retrieve history until the required data is found. If no relevant data is found, it will ensure the library can still attempt to retrieve older or complete account history.

Let us begin by defining the function signature. Since this function will be used internally by the library core modules, it will not be exportable.

```
bool FetchHistoryByCriteria(uint dataToGet)
  {
//-- Our function's code will go here
  }
```

We will define the interval variable, which modulates the history retrieval period. Initially, this will be set to 1 to start with a 24-hour time range.

```
int interval = 1;
```

We will calculate the time range starting from _24 hours_ ago to the _current time_.

```
datetime fromDateTime = TimeCurrent() - 1 * (PeriodSeconds(PERIOD_D1) * interval);
datetime toDateTime = TimeCurrent();
```

Next, we will use the _GetHistoryData()_ function to fetch the data within the defined time range.

```
GetHistoryData(fromDateTime, toDateTime, dataToGet);
```

If no data is found in the last _24 hours_, we will enter a loop where we increment the interval by _one week_ at a time. We will continue this process until we scan up to a full year ( _53 weeks_). During each iteration, the time range is updated to reflect the additional week.

```
while(GetTotalDataInfoSize(dataToGet) <= 0)
  {
   interval++;
   fromDateTime = TimeCurrent() - 1 * (PeriodSeconds(PERIOD_W1) * interval);
   toDateTime = TimeCurrent();
   GetHistoryData(fromDateTime, toDateTime, dataToGet);

   if(interval > 53)
     {
      break;
     }
  }
```

If the data is still not found after scanning for a _year_, we will reset the time range to cover the _entire account history_(from the _epoch_ to the _current time_). This ensures that all available history is checked.

```
fromDateTime = 0;
toDateTime = TimeCurrent();
GetHistoryData(fromDateTime, toDateTime, dataToGet);
```

Finally, we will check if any history has been successfully retrieved. If no data is found after scanning the entire account history, we will log the failure and return _false_. If data is found, we will return _true_ to indicate success.

```
if(GetTotalDataInfoSize(dataToGet) <= 0)
  {
   return(false);
  }
else
  {
   return(true);
  }
```

Here is the full implementation of the _FetchHistoryByCriteria()_ function with all the code segments included.

```
bool FetchHistoryByCriteria(uint dataToGet)
  {
   int interval = 1; //- Modulates the history period

//- Save the history period for the last 24 hours
   datetime fromDateTime = TimeCurrent() - 1 * (PeriodSeconds(PERIOD_D1) * interval);
   datetime toDateTime = TimeCurrent();

//- Get the specified history
   GetHistoryData(fromDateTime, toDateTime, dataToGet);

//- If we have no history in the last 24 hours we need to keep increasing the retrieval
//- period by one week untill we scan a full year (53 weeks)
   while(GetTotalDataInfoSize(dataToGet) <= 0)
     {
      interval++;
      fromDateTime = TimeCurrent() - 1 * (PeriodSeconds(PERIOD_W1) * interval);
      toDateTime = TimeCurrent();
      GetHistoryData(fromDateTime, toDateTime, dataToGet);

      //- If no history is found after a one year scanning period, we exit the while loop
      if(interval > 53)
        {
         break;
        }
     }

//- If we have not found any trade history in the last year, we scan and cache the intire account history
   fromDateTime = 0; //-- 1970 (Epoch)
   toDateTime = TimeCurrent(); //-- Time now
   GetHistoryData(fromDateTime, toDateTime, dataToGet);

//- If we still havent retrieved any history in the account, we log this info by
//- printing it and exit the function by returning false
   if(GetTotalDataInfoSize(dataToGet) <= 0)
     {
      return(false); //- Specified history not found, exit and return false
     }
   else
     {
      return(true); //- Specified history found, exit and return true
     }
  }
```

### Get The Last Closed Position Data Function

The _GetLastClosedPositionData()_ function is responsible for retrieving the properties of the most recent closed position and storing this data in the provided _lastClosedPositionInfo_ reference. This function will rely on the _FetchHistoryByCriteria()_ function to fetch the relevant trading history data, ensuring that it has access to the necessary position information. If no closed positions are found, the function will log an error message and return _false_. If successful, it will retrieve the data and return _true_.

Let us begin by defining the function signature. Since this function is intended for external use, it is _exported_ and can be imported by other MQL5 source files.

```
bool GetLastClosedPositionData(PositionData &lastClosedPositionInfo) export
  {
//-- Our function's code will go here
  }
```

We will first attempt to fetch the available position history data by calling the _FetchHistoryByCriteria()_ function with the _GET\_POSITIONS\_HISTORY\_DATA_ argument. This function call will search through available trading history to retrieve position-related data.

```
if(!FetchHistoryByCriteria(GET_POSITIONS_HISTORY_DATA))
  {
   Print(__FUNCTION__, ": No trading history available. Last closed position can't be retrieved.");
   return(false);
  }
```

If no position history data is available ( _i.e., the FetchHistoryByCriteria() function returns false_), we log an error message using the _Print()_ function, which helps in debugging by providing useful information about the failure. The function then returns _false_, indicating that it could not retrieve the last closed position.

If position history data is successfully retrieved, we will then save the information for the last closed position in the _lastClosedPositionInfo_ variable. This is done by assigning the first element in the _positionInfo_ array to _lastClosedPositionInfo_, as this array contains the history of all positions, with the most recent closed position being at the start of the array. To conclude the function, we will return _true_ to indicate that the last closed position data has been successfully retrieved and stored in the provided reference variable.

```
lastClosedPositionInfo = positionInfo[0];
return(true);
```

Here is the full implementation of the _GetLastClosedPositionData()_ function, with all the code segments included.

```
bool GetLastClosedPositionData(PositionData &lastClosedPositionInfo) export
  {
   if(!FetchHistoryByCriteria(GET_POSITIONS_HISTORY_DATA))
     {
      Print(__FUNCTION__, ": No trading history available. Last closed position can't be retrieved.");
      return(false);
     }

//-- Save the last closed position data in the referenced lastClosedPositionInfo variable
   lastClosedPositionInfo = positionInfo[0];
   return(true);
  }
```

### Last Closed Position Type Function

The _LastClosedPositionType()_ function will be responsible for determining the type of the most recently closed position in the trading account and storing it in the referenced variable _lastClosedPositionType_. This function is a logical extension of the _GetLastClosedPositionData()_ function, leveraging its capability to retrieve the last closed position and extract its specific type.

The _LastClosedPositionType()_ function is required for scenarios where you need to analyze the type of the most recent trade, such as distinguishing between buy and sell positions or identifying more complex strategies based on the position type.

Let us begin by defining the function signature. Since this function is intended for use by MQL5 files that _import_ the EX5 library, we will mark it as _exportable_. This will ensure the function is accessible externally, enhancing both modularity and usability.

```
bool LastClosedPositionType(ENUM_POSITION_TYPE &lastClosedPositionType) export
  {
//-- Our function's code will go here
  }
```

We begin by declaring a local variable _lastClosedPositionInfo_ of type _PositionData_. This variable will temporarily store the details of the last closed position, which we will extract using the _GetLastClosedPositionData()_ function.

```
PositionData lastClosedPositionInfo;
```

We invoke the _GetLastClosedPositionData()_ function, passing _lastClosedPositionInfo_ as an argument. If the function returns _true_, it means the data for the last closed position has been successfully retrieved.

```
if(GetLastClosedPositionData(lastClosedPositionInfo))
{
   //- Process the retrieved data
}
```

If _GetLastClosedPositionData()_ fails (returns _false_), this function immediately exits, returning _false_ to indicate that the operation could not be completed.

Inside the _if block_, we extract the _type property_ from _lastClosedPositionInfo_ and assign it to the referenced variable _lastClosedPositionType_. This ensures that the calling code has access to the type of the last closed position. After successfully saving the type, we return _true_ to indicate the operation was successful and exit the function.

```
lastClosedPositionType = lastClosedPositionInfo.type;
return(true);
```

If the retrieval of the last closed position fails, the function skips the _if block_ and directly returns _false_. This indicates that the position type could not be determined.

```
return(false);
```

Here is the complete implementation of the _LastClosedPositionType()_ function with all the code segments in their correct order.

```
bool LastClosedPositionType(ENUM_POSITION_TYPE &lastClosedPositionType) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionType = lastClosedPositionInfo.type;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Volume Function

The _LastClosedPositionVolume()_ function is responsible for retrieving the _volume_ of the most recently closed position in the trading history. It saves this volume into the referenced variable _lastClosedPositionVolume_. If the _LastClosedPositionVolume()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionVolume_ variable with the _volume_ value and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionVolume_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionVolume(double &lastClosedPositionVolume) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionVolume = lastClosedPositionInfo.volume;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Symbol Function

The _LastClosedPositionSymbol()_ function is responsible for retrieving the _symbol_ of the most recently closed position in the trading history. It saves this _symbol_ into the referenced variable _lastClosedPositionSymbol_. If the _LastClosedPositionSymbol()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionSymbol_ variable with the _symbol_ value and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionSymbol_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionSymbol(string &lastClosedPositionSymbol) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionSymbol = lastClosedPositionInfo.symbol;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Ticket Function

The _LastClosedPositionTicket()_ function is responsible for retrieving the _ticket_ number of the most recently closed position in the trading history. It saves this _ticket_ number into the referenced variable _lastClosedPositionTicket_. If the _LastClosedPositionTicket()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionTicket_ variable with the ticket number and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionTicket_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionTicket(ulong &lastClosedPositionTicket) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionTicket = lastClosedPositionInfo.ticket;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Profit Function

The _LastClosedPositionProfit()_ function is responsible for retrieving the _profit_ of the most recently closed position in the trading history. It saves this _profit_ into the referenced variable _lastClosedPositionProfit_. If the _LastClosedPositionProfit()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionProfit_ variable with the profit value and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionProfit_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionProfit(double &lastClosedPositionProfit) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionProfit = lastClosedPositionInfo.profit;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Net Profit Function

The _LastClosedPositionNetProfit()_ function is responsible for retrieving the _net profit_ of the most recently closed position in the trading history. The position _net profit_ is the final value after all charges like the _commission_ and _swaps_ have been deducted from the _position profit_. It saves this _net profit_ into the referenced variable _lastClosedPositionNetProfit_. If the _LastClosedPositionNetProfit()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionNetProfit_ variable with the _net profit_ value and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionNetProfit_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionNetProfit(double &lastClosedPositionNetProfit) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionNetProfit = lastClosedPositionInfo.netProfit;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Pip (Point) Profit Function

The _LastClosedPositionPipProfit()_ function is responsible for retrieving the _pip profit_ of the most recently closed position in the trading history. It saves this _pip profit_ into the referenced variable _lastClosedPositionPipProfit_. If the _LastClosedPositionPipProfit()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionPipProfit_ variable with the _pip profit_ value and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionPipProfit_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionPipProfit(int &lastClosedPositionPipProfit) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionPipProfit = lastClosedPositionInfo.pipProfit;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Close Price Function

The _LastClosedPositionClosePrice()_ function is responsible for retrieving the _closing price_ of the most recently closed position in the trading history. It saves this closing price into the referenced variable _lastClosedPositionClosePrice_. If the _LastClosedPositionClosePrice()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionClosePrice_ variable with the _closing price_ value and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionClosePrice_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionClosePrice(double &lastClosedPositionClosePrice) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionClosePrice = lastClosedPositionInfo.closePrice;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Open Price Function

The _LastClosedPositionOpenPrice()_ function is responsible for retrieving the _opening price_ of the most recently closed position in the trading history. It saves this opening price into the referenced variable _lastClosedPositionOpenPrice_. If the _LastClosedPositionOpenPrice()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionOpenPrice_ variable with the _opening price_ value and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionOpenPrice_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionOpenPrice(double &lastClosedPositionOpenPrice) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionOpenPrice = lastClosedPositionInfo.openPrice;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Stop Loss Price Function

The _LastClosedPositionSlPrice()_ function is responsible for retrieving the _stop loss price_ of the most recently closed position in the trading history. It saves this _stop loss price_ into the referenced variable _lastClosedPositionSlPrice_. If the _LastClosedPositionSlPrice()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionSlPrice_ variable with the _stop loss price_ value and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionSlPrice_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionSlPrice(double &lastClosedPositionSlPrice) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionSlPrice = lastClosedPositionInfo.slPrice;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Take Profit Price Function

The _LastClosedPositionTpPrice()_ function is responsible for retrieving the _take profit price_ of the most recently closed position in the trading history. It saves this _take profit price_ into the referenced variable _lastClosedPositionTpPrice_. If the _LastClosedPositionTpPrice()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionTpPrice_ variable with the _take profit price_ value and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionTpPrice_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionTpPrice(double &lastClosedPositionTpPrice) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionTpPrice = lastClosedPositionInfo.tpPrice;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Stop Loss Pips (Points) Function

The _LastClosedPositionSlPips()_ function is responsible for retrieving the _stop loss_ value of the most recently closed position in points ( _pips_). It saves this _stop loss_ value into the referenced variable _lastClosedPositionSlPips_. If the _LastClosedPositionSlPips()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionSlPips_ variable with the _stop loss_ value in _pips_ and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionSlPips_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionSlPips(int &lastClosedPositionSlPips) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionSlPips = lastClosedPositionInfo.slPips;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Take Profit Pips (Points) Function

The _LastClosedPositionTpPips()_ function is responsible for retrieving the _take profit_ value of the most recently closed position in points ( _pips_). It saves this _take profit_ value into the referenced variable _lastClosedPositionTpPips_. If the _LastClosedPositionTpPips()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionTpPips_ variable with the _take profit_ value in _pips_ and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionTpPips_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionTpPips(int &lastClosedPositionTpPips) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionTpPips = lastClosedPositionInfo.tpPips;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Open Time Function

The _LastClosedPositionOpenTime()_ function is responsible for retrieving the _open time_ of the most recently closed position. It saves this _open time_ into the referenced variable _lastClosedPositionOpenTime_. If the _LastClosedPositionOpenTime()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionOpenTime_ variable with the _open time_ value and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionOpenTime_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionOpenTime(datetime &lastClosedPositionOpenTime) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionOpenTime = lastClosedPositionInfo.openTime;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Close Time Function

The _LastClosedPositionCloseTime()_ function is responsible for retrieving the _close time_ of the most recently closed position. It saves this _close time_ into the referenced variable _lastClosedPositionCloseTime_. If the _LastClosedPositionCloseTime()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionCloseTime_ variable with the _close time_ value and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionCloseTime_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionCloseTime(datetime &lastClosedPositionCloseTime) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionCloseTime = lastClosedPositionInfo.closeTime;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Swap Function

The _LastClosedPositionSwap()_ function is responsible for retrieving the _swap_ value of the most recently closed position. It saves this _swap_ value into the referenced variable _lastClosedPositionSwap_. If the _LastClosedPositionSwap()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionSwap_ variable with the _swap_ value and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionSwap_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionSwap(double &lastClosedPositionSwap) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionSwap = lastClosedPositionInfo.swap;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Commission Function

The _LastClosedPositionCommission()_ function is responsible for retrieving the _commission_ value of the most recently closed position. It saves this _commission_ value into the referenced variable _lastClosedPositionCommission_. If the _LastClosedPositionCommission()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionCommission_ variable with the _commission_ value and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionCommission_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionCommission(double &lastClosedPositionCommission) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionCommission = lastClosedPositionInfo.commission;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Initiating Order Type Function

The _LastClosedPositionInitiatingOrderType()_ function is responsible for retrieving the _initiating order type_ of the most recently closed position. This enables us to know if the position was initiated by a pending order ( _Buy Stop, Buy Limit, Sell Stop, Sell Limit, Buy Stop Limit_, or _Sell Stop Limit_) or a direct _market entry_ order. It saves this _initiating order type_ into the referenced variable _lastClosedPositionInitiatingOrderType_. If the _LastClosedPositionInitiatingOrderType()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionInitiatingOrderType_ variable with the _initiating order type_ value and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionInitiatingOrderType_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionInitiatingOrderType(ENUM_ORDER_TYPE &lastClosedPositionInitiatingOrderType) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionInitiatingOrderType = lastClosedPositionInfo.initiatingOrderType;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position ID Function

The _LastClosedPositionId()_ function is responsible for retrieving the _ID_ of the most recently closed position. It saves this _position ID_ into the referenced variable _lastClosedPositionId_. If the _LastClosedPositionId()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionId_ variable with the _position ID_ value and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionId_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionId(ulong &lastClosedPositionId) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionId = lastClosedPositionInfo.positionId;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Initiated by Pending Order Function

The _LastClosedPositionInitiatedByPendingOrder()_ function is responsible for checking if the most recently closed position was initiated from a _pending order_. It saves this information into the referenced variable _lastClosedPositionInitiatedByPendingOrder_. If the _LastClosedPositionInitiatedByPendingOrder()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionInitiatedByPendingOrder_ variable with the result and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionInitiatedByPendingOrder_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionInitiatedByPendingOrder(bool &lastClosedPositionInitiatedByPendingOrder) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionInitiatedByPendingOrder = lastClosedPositionInfo.initiatedByPendingOrder;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Opening Order Ticket Function

The _LastClosedPositionOpeningOrderTicket()_ function is responsible for retrieving the _ticket_ number of the opening order for the most recently closed position. It saves this _ticket_ number into the referenced variable _lastClosedPositionOpeningOrderTicket_. If the _LastClosedPositionOpeningOrderTicket()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionOpeningOrderTicket_ variable with the _ticket_ number and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionOpeningOrderTicket_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionOpeningOrderTicket(ulong &lastClosedPositionOpeningOrderTicket) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionOpeningOrderTicket = lastClosedPositionInfo.openingOrderTicket;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Opening Deal Ticket Function

The _LastClosedPositionOpeningDealTicket()_ function is responsible for retrieving the _ticket_ number of the _opening deal_ for the most recently closed position. It saves this _deal ticket_ number into the referenced variable _lastClosedPositionOpeningDealTicket_. If the _LastClosedPositionOpeningDealTicket()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionOpeningDealTicket_ variable with the _deal ticket_ number and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the lastClosedPositionOpeningDealTicket variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionOpeningDealTicket(ulong &lastClosedPositionOpeningDealTicket) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionOpeningDealTicket = lastClosedPositionInfo.openingDealTicket;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Closing Deal Ticket Function

The _LastClosedPositionClosingDealTicket()_ function is responsible for retrieving the _ticket_ number of the _closing deal_ for the most recently closed position. It saves this _deal ticket_ number into the referenced variable _lastClosedPositionClosingDealTicket_. If the _LastClosedPositionClosingDealTicket()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionClosingDealTicket_ variable with the _deal ticket_ number and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionClosingDealTicket_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionClosingDealTicket(ulong &lastClosedPositionClosingDealTicket) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionClosingDealTicket = lastClosedPositionInfo.closingDealTicket;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Magic Function

The _LastClosedPositionMagic()_ function is responsible for retrieving the _magic number_ of the most recently closed position. It saves this _magic number_ into the referenced variable _lastClosedPositionMagic_. If the _LastClosedPositionMagic()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionMagic_ variable with the _magic number_ and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionMagic_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionMagic(ulong &lastClosedPositionMagic) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionMagic = lastClosedPositionInfo.magic;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Comment Function

The _LastClosedPositionComment()_ function is responsible for retrieving the _comment_ associated with the most recently closed position. It saves this _comment_ into the referenced variable _lastClosedPositionComment_. If the _LastClosedPositionComment()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionComment_ variable with the _comment_ and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionComment_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionComment(string &lastClosedPositionComment) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionComment = lastClosedPositionInfo.comment;
      return(true);
     }
   return(false);
  }
```

### Last Closed Position Duration Function

The _LastClosedPositionDuration()_ function is responsible for retrieving the _duration_ of the most recently closed position in _seconds_. It saves this _duration_ into the referenced variable _lastClosedPositionDuration_. If the _LastClosedPositionDuration()_ function successfully retrieves the data for the last closed position, it updates the _lastClosedPositionDuration_ variable with the _duration_ and confirms the operation was successful by returning _true_. If it fails to retrieve the data, the _lastClosedPositionDuration_ variable remains unchanged, and the function indicates the failure by returning _false_.

```
bool LastClosedPositionDuration(long &lastClosedPositionDuration) export
  {
   PositionData lastClosedPositionInfo;
   if(GetLastClosedPositionData(lastClosedPositionInfo))
     {
      lastClosedPositionDuration = lastClosedPositionInfo.duration;
      return(true);
     }
   return(false);
  }
```

### Conclusion

In this article, you have learned how smaller, focused utility functions, such as those we have designed to retrieve specific properties of the last closed positions, can work together to accomplish targeted tasks while maintaining clarity and modularity within the EX5 library codebase. By isolating the logic for extracting various properties of positions, these functions streamline the process of gathering specific data clearly and efficiently.

To keep this article concise and focused, we will defer the creation of library functions that retrieve the various properties of the last filled and canceled pending orders to the next article. In that article, we will explore these functions in detail, ensuring they are well-integrated with the existing framework. Following that, we will move on to developing a set of analytical reporting functions, which will enable users to generate insightful summaries and detailed reports based on the historical trade data. This step-by-step approach will ensure clarity and allow us to cover each topic comprehensively without overwhelming you with so much information all at once.

At the end of this article, you will find the latest version of the _**HistoryManager.mq5**_ library source code, which includes all the functions created in this article as well as those introduced in the previous one. Thank you for following along, and I look forward to connecting with you again in the next article.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16681.zip "Download all attachments in the single ZIP archive")

[HistoryManager\_Part1.mq5](https://www.mql5.com/en/articles/download/16681/historymanager_part1.mq5 "Download HistoryManager_Part1.mq5")(33.95 KB)

[HistoryManager.mq5](https://www.mql5.com/en/articles/download/16681/historymanager.mq5 "Download HistoryManager.mq5")(55.17 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Trading Toolkit (Part 8): How to Implement and Use the History Manager EX5 Library in Your Codebase](https://www.mql5.com/en/articles/17015)
- [MQL5 Trading Toolkit (Part 7): Expanding the History Management EX5 Library with the Last Canceled Pending Order Functions](https://www.mql5.com/en/articles/16906)
- [MQL5 Trading Toolkit (Part 6): Expanding the History Management EX5 Library with the Last Filled Pending Order Functions](https://www.mql5.com/en/articles/16742)
- [MQL5 Trading Toolkit (Part 4): Developing a History Management EX5 Library](https://www.mql5.com/en/articles/16528)
- [MQL5 Trading Toolkit (Part 3): Developing a Pending Orders Management EX5 Library](https://www.mql5.com/en/articles/15888)
- [MQL5 Trading Toolkit (Part 2): Expanding and Implementing the Positions Management EX5 Library](https://www.mql5.com/en/articles/15224)

**[Go to discussion](https://www.mql5.com/en/forum/478574)**

![Price Action Analysis Toolkit Development (Part 6): Mean Reversion Signal Reaper](https://c.mql5.com/2/107/Price_Action_Analysis_Toolkit_Development_Part_6_LOGO.png)[Price Action Analysis Toolkit Development (Part 6): Mean Reversion Signal Reaper](https://www.mql5.com/en/articles/16700)

While some concepts may seem straightforward at first glance, bringing them to life in practice can be quite challenging. In the article below, we'll take you on a journey through our innovative approach to automating an Expert Advisor (EA) that skillfully analyzes the market using a mean reversion strategy. Join us as we unravel the intricacies of this exciting automation process.

![Building a Candlestick Trend Constraint Model (Part 10): Strategic Golden and Death Cross (EA)](https://c.mql5.com/2/106/Building_A_Candlestick_Trend_Constraint_Model_Part_10_LOGO.png)[Building a Candlestick Trend Constraint Model (Part 10): Strategic Golden and Death Cross (EA)](https://www.mql5.com/en/articles/16633)

Did you know that the Golden Cross and Death Cross strategies, based on moving average crossovers, are some of the most reliable indicators for identifying long-term market trends? A Golden Cross signals a bullish trend when a shorter moving average crosses above a longer one, while a Death Cross indicates a bearish trend when the shorter average moves below. Despite their simplicity and effectiveness, manually applying these strategies often leads to missed opportunities or delayed trades.

![Automating Trading Strategies in MQL5 (Part 3): The Zone Recovery RSI System for Dynamic Trade Management](https://c.mql5.com/2/107/Automating_Trading_Strategies_in_MQL5_Part_3_LOGO.png)[Automating Trading Strategies in MQL5 (Part 3): The Zone Recovery RSI System for Dynamic Trade Management](https://www.mql5.com/en/articles/16705)

In this article, we create a Zone Recovery RSI EA System in MQL5, using RSI signals to trigger trades and a recovery strategy to manage losses. We implement a "ZoneRecovery" class to automate trade entries, recovery logic, and position management. The article concludes with backtesting insights to optimize performance and enhance the EA’s effectiveness.

![MQL5 Wizard Techniques you should know (Part 51): Reinforcement Learning with SAC](https://c.mql5.com/2/107/MQL_Wizard_Techniques_you_should_know_Part_51_LOGO.png)[MQL5 Wizard Techniques you should know (Part 51): Reinforcement Learning with SAC](https://www.mql5.com/en/articles/16695)

Soft Actor Critic is a Reinforcement Learning algorithm that utilizes 3 neural networks. An actor network and 2 critic networks. These machine learning models are paired in a master slave partnership where the critics are modelled to improve the forecast accuracy of the actor network. While also introducing ONNX in these series, we explore how these ideas could be put to test as a custom signal of a wizard assembled Expert Advisor.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=tgzarmjoxqaxyxkzsejokahjvrsvwybe&ssn=1769183048980355850&ssn_dr=0&ssn_sr=0&fv_date=1769183048&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16681&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Trading%20Toolkit%20(Part%205)%3A%20Expanding%20the%20History%20Management%20EX5%20Library%20with%20Position%20Functions%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918304842688379&fz_uniq=5069698174420977992&sv=2552)

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