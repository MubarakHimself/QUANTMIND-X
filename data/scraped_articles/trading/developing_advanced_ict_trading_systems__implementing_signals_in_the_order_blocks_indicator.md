---
title: Developing Advanced ICT Trading Systems: Implementing Signals in the Order Blocks Indicator
url: https://www.mql5.com/en/articles/16268
categories: Trading, Trading Systems, Indicators
relevance_score: 13
scraped_at: 2026-01-22T17:10:52.906226
---

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/16268&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5048832484562542475)

MetaTrader 5 / Examples


1. [Introduction](https://www.mql5.com/en/articles/16268#introduccion)

- [Order Block detection based on market depth](https://www.mql5.com/en/articles/16268#1)
- [Initialization and completion of the "Order Book" event and creation of arrays](https://www.mql5.com/en/articles/16268#2)
- [Collecting market depth data to determine volumes](https://www.mql5.com/en/articles/16268#3)
- [Startegy with marked depth-based Order Block detection](https://www.mql5.com/en/articles/16268#4)

- [Creating indicator buffers](https://www.mql5.com/en/articles/16268#5)
- [Modifying OnInit function to configure buffers](https://www.mql5.com/en/articles/16268#6)
- [Implementing buffers in the indicator (2)](https://www.mql5.com/en/articles/16268#7)

- [Updating input parametersp](https://www.mql5.com/en/articles/16268#8)
- [Logic of indicator signal generation](https://www.mql5.com/en/articles/16268#9)
- [Implementing a trading strategy](https://www.mql5.com/en/articles/16268#10)
- [Setting Take Profit (TP) and Stop Loss (SL) levels](https://www.mql5.com/en/articles/16268#11)
- [Visualizing TP and SL levels on the chart](https://www.mql5.com/en/articles/16268#12)
- [Adding buffers for TP and SL levels (4)](https://www.mql5.com/en/articles/16268#13)
- [Finalizing main code and cleanup](https://www.mql5.com/en/articles/16268#14)

- [Conclusion](https://www.mql5.com/en/articles/16268#conclusion)

### Introduction

Welcome to our article on MQL5! In this installment, we will focus on adding buffers and entry signals to our indicator, completing the essential functionalities required for use in automated trading strategies.

If you are new to this series, we recommend reviewing the [first part](https://www.mql5.com/en/articles/15899), where we explained how to create the indicator from scratch and covered the fundamental concepts.

### Order Block Detection Based on Market Depth

Our logic for identifying order blocks using market depth is structured as follows:

![ Algorithm](https://c.mql5.com/2/172/Example_logic__1.png)

1. **Array Creation**: We will create two arrays to store the volume of each candlestick in the market depth. This allows us to organize and analyze volume data efficiently.
2. **Market Depth Data Collection** — in the event.

```
void  OnBookEvent( )
```

In the OnBookEvent() function, we will capture every change in market depth, recording new volume data to ensure real-time updates.

      3\. **Rules to validate Order Blocks**: Once volumes are stored in the arrays, we will apply price action rules alongside this data to validate an order block.

**Rules for Identifying Order Blocks with Market Depth**

Initially, when building our indicator, we searched for order blocks within a specific candlestick range. However, in the case of market depth, we will not use a wide "x-range". Instead, we will focus specifically on the third candlestick (with candlestick 0 being the current one).

| Rules | Bullish Order Block | Bearish Order Block |
| --- | --- | --- |
| Volume peak on candlestick 3: | The buy volume of candlestick 3 must exceed, by a set ratio, the combined buy and sell volumes of candlesticks 2 and 4. | The sell volume of candlestick 3 must exceed, by a set ratio, the combined buy and sell volumes of candlesticks 2 and 4. |
| 3 consecutive candlesticks: | Must be three consecutive bullish candlesticks<br>(candlesticks 1, 2, and 3) | Must be three consecutive bearish candlesticks<br>(candlesticks 1, 2, and 3) |
| Body of candlestick 3: | The low of candlestick 2 must be greater than half of the body of candlestick 3. | The high of candlestick 2 must be lower than half of the body of candlestick 3. |
| Candlestick 3 high or low: | The high of candlestick 3 must be lower than the close of candlestick 2. | The low of candlestick 3 must be higher than the close of candlestick 2. |

With these rules, we ensure the following:

- **Buy/Sell Imbalance**: We verify a significant imbalance in buying or selling within a specific candlestick, where buy or sell orders surpass those of the previous and next candlesticks by a set ratio.
- **Candlestick Body Control in Imbalance**: We confirm that unexecuted orders caused by excess demand or supply are not absorbed by the subsequent candlestick, thereby validating the persistence of the order block.
- **Strong Bullish or Bearish Move**: We confirm that the pattern reflects a decisive upward or downward movement, highlighting the intensity of the imbalance in price action.

With all this in mind, we can now translate the logic into code.

### Initialization and completion of the "Order Book" event and creation of arrays

**Creating Arrays**

Before using the order book, you need to create dynamic arrays that will store volume data. These arrays will be of type:

```
long
```

They will be used to store buy and sell volumes, respectively.

1. Go to the global section of the program and declare the dynamic arrays:

```
long buy_volume[];
long sell_volume[];
```

      2\. Inside the OnInit event, resize the arrays to have an initial size of 1. Then, assign the value 0 to index 0 of each array:

```
  ArrayResize(buy_volume,1);
  ArrayResize(sell_volume,1);
  buy_volume[0] = 0.0;
  sell_volume[0] = 0.0;
```

**Initialization and Finalization of Market Depth Event**

Before starting market depth, we will create a global variable to indicate whether this function is available. This will allow us to avoid using:

```
INIT_FAILED
```

Not all symbols on every broker provide **traded volume** in market depth. With this approach, the indicator will not depend exclusively on a broker offering this function.

- To check whether the symbol you want to trade supports depth of market with traded volume, follow these steps:

1\. Click on the top left corner of your chart in the corresponding box:  ![](https://c.mql5.com/2/172/depth_market_2_h1c__1.png)

2\. Verify if the symbol has available depth of market volume. If supported, you will see a confirmation similar to the following images.

Symbol with Depth of Market volume:

![Depth of Market 2](https://c.mql5.com/2/172/Depth_Volume__1.PNG)

Symbol without Depth of Market volume:

![ ETHUSD Depth of Market](https://c.mql5.com/2/172/ETHUSD_Depth_of_Market__1.PNG)

As mentioned, depth of market volume is not available for all instruments and depends on your broker.

Let's move on to initialization and completion of the market depth.

1\. Global Control Variable

We define a global boolean variable to mark the availability of market depth:

```
bool use_market_book = true; //true by default
```

This variable starts as true but will be changed if market depth fails to initialize.

2\. Market Depth Initialization

To initialize market depth, we use the function:

```
MarketBookAdd()
```

This opens the Depth of Market for a specified symbol. The function requires the current symbol:

```
_Symbol
```

Inside the OnInit event, we check if initialization is successful:

```
 if(!MarketBookAdd(_Symbol)) //Verify initialization of the order book for the current symbol
     {
      Print("Error Open Market Book: ", _Symbol, " LastError: ", _LastError); //Print error in case of failure
      use_market_book = false; //Mark use_market_book as false if initialization fails
     }
```

3\. Market Depth Finalization

In the OnDeinit event, we release market depth using:

```
 MarketBookRelease()
```

We then check the closure and print a message accordingly:

```
//---
   if(MarketBookRelease(_Symbol)) //Verify if closure was successful
     Print("Order book successfully closed for: " , _Symbol); //Print success message if so
   else
     Print("Order book closed with errors for: " , _Symbol , "   Last error: " , GetLastError()); //Print error message with code if not
```

### Collecting Market Depth Data for Volume Detection in Arrays

With market depth initialized, we can begin collecting relevant data. For this, we will create the OnBookEvent event, which is triggered every time a change occurs in market depth.

1. **Creating** the OnBookEvent:

```
void OnBookEvent(const string& symbol)
```

     2. **Checking** the symbol and availability of the market depth:

```
 if(symbol !=_Symbol || use_market_book == false)
      return;
// Exit the event if conditions are not met
```

With this check in place, the complete **OnBookEvent** function can be structured as follows:

```
void OnBookEvent(const string& symbol)
  {
   if(symbol !=_Symbol || use_market_book == false)
      return;
// Define array to store Market Book data
   MqlBookInfo book_info[];

// Retrieve Market Book data
   bool book_count = MarketBookGet(_Symbol,book_info);

// Verify if data was successfully obtained
   if(book_count == true)
     {
      // Iterate through Market Book data
      for(int i = 0; i < ArraySize(book_info); i++)
        {
         // Check if the record is a buy order (BID)
         if(book_info[i].type == BOOK_TYPE_BUY  || book_info[i].type ==  BOOK_TYPE_BUY_MARKET)
           {

            buy_volume[0] += book_info[i].volume;
           }
         // Check if the record is a sell order (ASK)
         if(book_info[i].type == BOOK_TYPE_SELL || book_info[i].type == BOOK_TYPE_SELL_MARKET)
           {
            sell_volume[0] += book_info[i].volume;
           }
        }
     }
   else
     {
      Print("No Market Book data retrieved.");
     }
  }
```

Explanation of the Code:

- **Volume Retrieval**: Each time a change occurs in market depth, OnBookEvent collects the volume of the latest registered orders.
- **Array Updates**: It adds buy and sell volumes into index 0 of the buy\_volume and sell\_volume arrays, respectively.

To ensure that arrays accumulate market depth volume for each new candlestick and keep a rolling history (for example, 30 elements), the following adjustments are necessary.

1\. New Candlestick Verification and Counter Validation (more than 1)

To avoid false positives at program startup and ensure that arrays are only updated when a new candlestick opens (and after at least one prior opening), we implement a check combining the counter variable with new\_vela. This ensures that array updates occur only when there is genuinely new information available.

_Declaration of Static Variables_

We declare counter as a static variable so that it persists between calls to OnCalculate. The new\_vela variable should indicate whether a new candlestick has opened.

```
static int counter = 0;
```

_New Candlestick and Counter Verification Condition_

We verify that counter is greater than 1, that new\_vela is true, and that market depth is available. Only if all these conditions are satisfied will we resize the arrays and shift their elements. This prevents premature resizing and ensures that updates occur only when valid data is available and the market book provides traded volume for the current symbol.

```
if(counter > 1 && new_vela == true && use_market_book == true)
```

_Counter Update_

The counter is incremented by 1 each time a new candlestick is detected.

```
counter++;
```

2\. Array Size Control

We check that the arrays do not exceed a maximum size of 30 elements. If they do, we resize them back to 30, discarding the oldest element:

```
if(ArraySize(buy_volume) >= 30)
{
   ArrayResize(buy_volume, 30);  // Keep buy_volume size at 30
   ArrayResize(sell_volume, 30); // Keep sell_volume size at 30
}
```

3\. Resizing for New Values

We add an extra slot to the arrays to store the new volume at the initial position:

```
ArrayResize(buy_volume, ArraySize(buy_volume) + 1);
ArrayResize(sell_volume, ArraySize(sell_volume) + 1);
```

4\. Shifting Elements

We move all array elements one position forward. This ensures that the latest data is always stored at index 0, while older values are shifted to higher indices.

```
for(int i = ArraySize(buy_volume) - 1; i > 0; i--)
{
   buy_volume[i] = buy_volume[i - 1];
   sell_volume[i] = sell_volume[i - 1];
}
```

5\. Volume Verification

We print the buy and sell volumes at position 1 of the arrays to verify the volume recorded for the last candle:

```
Print("Buy volume of the last candle: ", buy_volume[1]);
Print("Sell volume of the last candle: ", sell_volume[1]);
```

6\. Volume Reset

We reset index 0 of both arrays to 0 so that they begin accumulating the volume for the new candle:

```
buy_volume[0] = 0;
sell_volume[0] = 0;
```

7\. Condition to Prevent Errors from Inconsistent Market Book Data

An additional safeguard is introduced to automatically disable use\_market\_book if the values of buy\_volume and sell\_volume at the most recent positions (indices 3, 2, and 1) are all zero. This adjustment is necessary because even if a symbol appears to have market book data in live trading, when running in the strategy tester it may also appear active, yet the arrays may not fill correctly due to the absence of market depth updates. This can result in zeros being stored, which may cause the indicator to handle incorrect information.

This verification ensures that the indicator does not process meaningless data and that use\_market\_book is only applied when the market book contains valid values.

```
if(ArraySize(buy_volume) > 4 && ArraySize(sell_volume) > 4)
        {
         if(buy_volume[3] == 0 && sell_volume[3] == 0 &&  buy_volume[2] == 0 && sell_volume[2] == 0 &&  buy_volume[1] == 0 && sell_volume[1] == 0)  use_market_book = false;
        }
```

Integrated Code Snippet

```
if(counter > 1 && new_vela == true && use_market_book == true)
     {
      if(ArraySize(buy_volume) > 4 && ArraySize(sell_volume) > 4)
        {
         if(buy_volume[3] == 0 && sell_volume[3] == 0 &&  buy_volume[2] == 0 && sell_volume[2] == 0 &&  buy_volume[1] == 0 && sell_volume[1] == 0)  use_market_book = false;
        }

       // If array size is greater than or equal to 30, resize to maintain a fixed length
     if(ArraySize(buy_volume) >= 30)
      {
      ArrayResize(buy_volume, 30);  // Ensure buy_volume does not exceed 30 elements
      ArrayResize(sell_volume, 30); // Ensure sell_volume does not exceed 30 elements
      }

      ArrayResize(buy_volume,ArraySize(buy_volume)+1);
      ArrayResize(sell_volume,ArraySize(sell_volume)+1);

      for(int i = ArraySize(buy_volume) - 1; i > 0; i--)
        {
         buy_volume[i] = buy_volume[i - 1];
         sell_volume[i] = sell_volume[i - 1];
        }

      // Reset volumes at index 0 to begin accumulating for the new candlestick
      buy_volume[0] = 0;
      sell_volume[0] = 0;
     }
```

### Strategy with Market-Depth Order Block Detection

The strategy will follow the same logic as used previously, with one key difference: instead of iterating through loops, we perform the checks directly on candlestick 3. The general logic remains the same: we verify specific conditions, identify the closest relevant candlestick (depending on the type of order block), assign the corresponding values to the structure, and then add the order block to the array. Here, we apply the same process, but in a simplified manner.

Let's begin by creating the structures that will store the order block information:

```
OrderBlocks newVela_Order_block_Book_bajista;
OrderBlocks newVela_Order_block_Book;
```

**1\. Initial Conditions**

First, we verify that the buy\_volume and sell\_volume arrays contain at least 5 elements. This ensures that sufficient historical data is available for analysis. We also confirm that use\_market\_book is active in order to process market depth.

```
if(ArraySize(buy_volume) >= 5 && ArraySize(sell_volume) >= 5 && use_market_book == true)
```

**2\. Control Variable Definition**

We define the variable case\_book to indicate whether a specific volume condition is met. The ratio is set to 1.4, which serves as a comparison factor to detect significant increases in buy volume.

```
bool case_book = false;
double ratio = 1.4;
```

**3\. Buy Volume Condition (Case Book)**

Here, we check if the buy volume at index 3 is significantly greater than the volumes at indices 2 and 4, for both buy and sell sides, using the ratio as a multiplier. If this condition is satisfied, case\_book is activated.

Bullish Case:

```
if(buy_volume[3] > buy_volume[4] * ratio && buy_volume[3] > buy_volume[2] * ratio &&
   buy_volume[3] > sell_volume[4] * ratio && buy_volume[3] > sell_volume[2] * ratio)
{
    case_book = true;
}
```

Bearish Case:

```
if(sell_volume[3] > buy_volume[4]*ratio && sell_volume[3] > buy_volume[2]*ratio &&
sell_volume[3] > sell_volume[4]*ratio && sell_volume[3] > sell_volume[2]*ratio)
{
case_book = true;
}
```

**4\. Candlestick Body Calculation**

We calculate the body size of the candlestick (body\_tree) at index 3 by subtracting its opening price from its closing price.

```
double body_tree = closeArray[3] - openArray[3];
```

```
double body_tree = openArray[3] - closeArray[3];
```

**5\. Price Condition Verification for Bullish Setup**

We evaluate the previously mentioned conditions (see the table above).

Bullish Case:

```
if(lowArray[2] > ((body_tree * 0.5) + openArray[3]) && highArray[3] < closeArray[2] &&
   closeArray[3] > openArray[3] && closeArray[2] > openArray[2] && closeArray[1] > openArray[1])
```

Bearish Case:

```
if(highArray[2] < (openArray[3]-(body_tree * 0.5)) && lowArray[3] > closeArray[2] &&
            closeArray[3] < openArray[3] && closeArray[2] < openArray[2] && closeArray[1] < openArray[1])
```

**6\. Identification of Previous Bullish Candlesticks**

We call the function FindFurthestAlcista, which searches for the furthest bullish candlestick within a 20-candlestick range from index 3. This helps to identify a reference candlestick for a strong bullish setup. If a bullish candlestick is found, its index is greater than 0, allowing the process to continue.

Bullish Case:

```
int furthestAlcista = FindFurthestAlcista(Time[3], 20);
if(furthestAlcista > 0)
```

**7\. Assigning Values to the Order Block**

If all conditions are satisfied, we define the order block (newVela\_Order\_block\_Book or newVela\_Order\_block\_Book\_bearish) with the values of the identified candlestick.

Bullish Case:

```
Print("Case Book Found");
datetime time1 = Time[furthestAlcista];
double price2 = openArray[furthestAlcista];
double price1 = lowArray[furthestAlcista];

//Assign the above variables to the structure
newVela_Order_block_Book.price1 = price1;
newVela_Order_block_Book.time1 = time1;
newVela_Order_block_Book.price2 = price2;
newVela_Order_block_Book.mitigated = false;
newVela_Order_block_Book.name = "Bullish Order Block Book " + TimeToString(newVela_Order_block_Book.time1);
AddIndexToArray_alcistas(newVela_Order_block_Book);
```

Bearish Case:

```
Print("Case Book Found");
datetime time1 = Time[furthestBajista];
double price1 = closeArray[furthestBajista];
double price2 = lowArray[furthestBajista];

//Assign the above variables to the structure
newVela_Order_block_Book_bajista.price1 = price1;
newVela_Order_block_Book_bajista.time1 = time1;
newVela_Order_block_Book_bajista.price2 = price2;
newVela_Order_block_Book_bajista.mitigated = false;
newVela_Order_block_Book_bajista.name = "Order Block Bajista Book " + TimeToString(newVela_Order_block_Book_bajista.time1);
AddIndexToArray_bajistas(newVela_Order_block_Book_bajista);
```

Complete code:

```
if(ArraySize(buy_volume) >= 5 && ArraySize(sell_volume) >= 5 && use_market_book == true)
  {

   bool case_book = false;
   double ratio = 1.4;

   if(sell_volume[3] > buy_volume[4]*ratio && sell_volume[3] > buy_volume[2]*ratio &&
      sell_volume[3] > sell_volume[4]*ratio && sell_volume[3] > sell_volume[2]*ratio)
     {
      case_book = true;
     }
   double body_tree =   openArray[3] - closeArray[3];

   if(highArray[2] < (openArray[3]-(body_tree * 0.5)) && lowArray[3] > closeArray[2] &&
      closeArray[3] < openArray[3] && closeArray[2] < openArray[2] && closeArray[1] < openArray[1])
     {
      int furthestBajista = FindFurthestBajista(Time[3],20); //We call the "FindFurthestAlcista" function to find out if there are bullish candlesticks before "one candle"
      if(furthestBajista  > 0) // Whether or not there is a furthest Bullish candle, it will be greater than 0 since if there is none, the previous candlestick returns to "one candle".
        {
         Print("Case Book Found");
         datetime time1 = Time[furthestBajista];
         double price1 = closeArray[furthestBajista];
         double price2 = lowArray[furthestBajista];

         //Assign the above variables to the structure
         newVela_Order_block_Book_bajista.price1 = price1;
         newVela_Order_block_Book_bajista.time1 = time1;
         newVela_Order_block_Book_bajista.price2 = price2;
         newVela_Order_block_Book_bajista.mitigated = false;
         newVela_Order_block_Book_bajista.name = "Order Block Bajista Book " + TimeToString(newVela_Order_block_Book_bajista.time1);

         AddIndexToArray_bajistas(newVela_Order_block_Book_bajista);

        }
     }
  }
//--------------------    Bullish   --------------------

if(ArraySize(buy_volume) >= 5 && ArraySize(sell_volume) >= 5 && use_market_book == true)
  {

   bool case_book = false;
   double ratio = 1.4;

   if(buy_volume[3] > buy_volume[4]*ratio && buy_volume[3] > buy_volume[2]*ratio &&
      buy_volume[3] > sell_volume[4]*ratio && buy_volume[3] > sell_volume[2]*ratio)
     {
      case_book = true;
     }
   double body_tree =  closeArray[3] - openArray[3];

   if(lowArray[2] > ((body_tree * 0.5)+openArray[3]) && highArray[3] < closeArray[2] &&
      closeArray[3] > openArray[3] && closeArray[2] > openArray[2] && closeArray[1] > openArray[1])
     {
      int furthestAlcista = FindFurthestAlcista(Time[3],20); //We call the "FindFurthestAlcista" function to find out if there are bullish candlessticks before "one candle"
      if(furthestAlcista > 0) // Whether or not there is a furthest Bullish candle, it will be greater than 0 since if there is none, the previous candlestick returns to "one candle".
        {
         Print("Case Book Found");
         datetime time1 = Time[furthestAlcista];     //assign the index time of furthestAlcista to the variable time1
         double price2 = openArray[furthestAlcista]; //assign the open of furthestAlcista as price 2 (remember that we draw it on a bearish candlestick most of the time)
         double price1 = lowArray[furthestAlcista];  //assign the low of furthestAlcista as price 1

         //Assign the above variables to the structure
         newVela_Order_block_Book.price1 = price1;
         newVela_Order_block_Book.time1 = time1;
         newVela_Order_block_Book.price2 = price2;
         newVela_Order_block_Book.mitigated = false;
         newVela_Order_block_Book.name = "Bullish Order Block Book " + TimeToString(newVela_Order_block_Book.time1);

         AddIndexToArray_alcistas(newVela_Order_block_Book);

        }
     }
  }
```

### Creating Indicator Buffers

To create and configure the buffers for our Order Block indicator in MQL5, we start by defining two buffers and two plots globally to store and display the price levels of bullish and bearish order blocks.

1\. Declaration of Buffers and Plots

We declare two buffers in the global section of the program to store the price data of the order blocks. Additionally, we define two plots to visualize the order blocks on the chart.

```
#property  indicator_buffers 2
#property  indicator_plots 2
#property indicator_label1 "Bullish Order Block"
#property indicator_label2 "Bearish Order Block"
```

2\. Create Dynamic Arrays for Buffers

We declare two dynamic arrays, buyOrderBlockBuffer and sellOrderBlockBuffer, to store the prices corresponding to bullish and bearish order blocks. These arrays are linked to the indicator buffers, allowing the order block data to be visualized on the chart.

```
//--- Define the buffers
double buyOrderBlockBuffer[];   // Buffer for bullish order blocks
double sellOrderBlockBuffer[];  // Buffer for bearish order blocks
```

Description:

- **buyOrderBlockBuffer**: Stores the price levels of bullish order blocks and represents points where price may find support.
- **sellOrderBlockBuffer**: Stores the price levels of bearish order blocks and represents points where price may encounter resistance.

### Modifying OnInit Function to Configure Buffers

In this section, we adjust the OnInit function to configure the indicator buffers, assigning the bullish and bearish order block arrays to the indicator buffers. This ensures that the indicator stores and displays the data correctly on the chart.

**Steps:**

1\. Assign Data Buffers Using SetIndexBuffer

In OnInit, we assign the arrays buyOrderBlockBuffer and sellOrderBlockBuffer to the indicator buffers using SetIndexBuffer. This ensures the arrays can store and display data on the chart.

```
//--- Assign data buffers to the indicator
   SetIndexBuffer(0, buyOrderBlockBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, sellOrderBlockBuffer, INDICATOR_DATA)
```

2\. Configure Buffers as Series and Fill with Empty Values

To display the data in reverse chronological order (like a time series), we set the arrays as series. We also initialize both buffers with EMPTY\_VALUE to prevent displaying incorrect data until the real values are calculated.

```
  ArraySetAsSeries(buyOrderBlockBuffer, true);
  ArraySetAsSeries(sellOrderBlockBuffer, true);
  ArrayFill(buyOrderBlockBuffer, 0,0, EMPTY_VALUE);  // Initialize to EMPTY_VALUE
  ArrayFill(sellOrderBlockBuffer, 0,0, EMPTY_VALUE); // Initialize to EMPTY_VALUE
```

### Implementing Buffers in the Indicator (2)

In this section, we assign the prices of bullish and bearish order blocks to the indicator buffers. These buffers make the data available at the corresponding index for the time (time1) of each order block.

**1\. Assign Prices for Bullish Order Blocks**

Inside the loop where each bullish block in ob\_alcistas is evaluated, we assign price2 to buyOrderBlockBuffer. We use iBarShift to get the exact index on the chart where time1 matches the order block time.

```
buyOrderBlockBuffer[iBarShift(_Symbol, _Period, ob_alcistas[i].time1)] = ob_alcistas[i].price2;
```

Here, price2 from the bullish block is assigned to the corresponding index in buyOrderBlockBuffer, so the buffer reflects the price level of the block on the chart.

**2\. Assign Prices for Bearish Order Blocks**

Similarly, we assign price2 from each bearish block to sellOrderBlockBuffer by iterating through the ob\_bajistas array and setting the price at the corresponding index.

```
sellOrderBlockBuffer[iBarShift(_Symbol, _Period, ob_bajistas[i].time1)] = ob_bajistas[i].price2;
```

Summary:

- **iBarShift** locates the exact index where the block time matches the chart position.
- **buyOrderBlockBuffer** and **sellOrderBlockBuffer** receive the price2 values, allowing prices to be recorded at the correct time for use in the chart and further indicator calculations.

### Updating Input Parameters

In this section, we configure the input parameters to allow the user to customize the Take Profit (TP) and Stop Loss (SL) calculation style. We create an enumeration to select between two options: **ATR** (Average True Range) or **POINT** (fixed points).

**ENUM\_TP\_SL\_STYLE**

The enumeration allows the user to choose between two TP and SL calculation modes.

- **АТR**: Sets TP and SL based on the average price movement range, automatically adjusting according to current market volatility.
- **POINT**: Sets TP and SL in fixed points defined by the user.

```
enum ENUM_TP_SL_STYLE
  {
   ATR,
   POINT
  };
```

Explanation:

- **АТR**: The user sets a multiplier to determine TP and SL distance relative to ATR. A higher multiplier increases the TP and SL distance based on market volatility.

- **POINT**: The user manually defines TP and SL in fixed points, allowing static levels regardless of volatility.


Now, continuing to work with the indicator parameters, we structure the indicator input parameters using sinput and group the settings into sections. This provides a more visual and organized display of parameters in the interface, making it easier for the user to configure.

**1\. Strategy Section**

We create a strategy group that includes the TP and SL calculation style:

```
sinput group "-- Strategy --"
input ENUM_TP_SL_STYLE tp_sl_style = POINT; // TP and SL style: ATR or fixed points
```

Here, tp\_sl\_style allows the user to choose whether to calculate TP and SL based on ATR (Average True Range) or in fixed points.

**2\. TP and SL Configuration by Selected Method**

To account for the specific settings of each method, we add two additional groups: one for the ATR method and one for fixed points.

ATR Group: here we include two double input variables that allow the user to specify ATR multipliers, thereby adjusting the TP and SL range based on volatility.

```
sinput group " ATR "
input double Atr_Multiplier_1 = 1.5; // Multiplier for TP
input double Atr_Multiplier_2 = 2.0; // Multiplier for SL
```

POINT Group: in this group, we add two input variables of type int to define fixed points TP and SL, which allows us to manually and precisely control these distances.

```
sinput group " POINT "
input int TP_POINT = 500; // Fixed points for TP
input int SL_POINT = 275; // Fixed points for SL
```

This structure keeps the parameters neatly organized and categorized, making them easier to use and increasing clarity when setting up the indicator. The user will be able to intuitively set the TP and SL style, choosing between automatic configurations based on ATR or manual settings in points.

The full code of parameters:

```
sinput group "--- Order Block Indicator settings ---"
sinput group "-- Order Block --"
input          int  Rango_universal_busqueda = 500;
input          int  Witdth_order_block = 1;

input          bool Back_order_block = true;
input          bool Fill_order_block = true;

input          color Color_Order_Block_Bajista = clrRed;
input          color Color_Order_Block_Alcista = clrGreen;

sinput group "-- Strategy --"
input          ENUM_TP_SL_STYLE tp_sl_style = POINT;

sinput group " ATR "
input          double Atr_Multiplier_1 = 1.5;
input          double Atr_Multiplier_2 = 2.0;
sinput group " POINT "
input          int TP_POINT = 500;
input          int SL_POINT = 275;
```

### Indicator Signal Generation Logic

To generate buy or sell signals, two static variables are used:

| Variable | Description |
| --- | --- |
| time\_ and time\_b | Stores the time when the order block is mitigated and adds a 5-candlestick margin (in seconds) for expiration. |
| buscar\_oba and buscar\_obb | Controls the search for newly mitigated order blocks. Activated or deactivated based on conditions. |

**Signal Generation Process**

Detection of a Mitigated Order Block:

- When an order block is mitigated, time\_ is set to the current time plus a 5-candlestick margin.
- The searcher variable is set to false to pause further searches while validating signal conditions.

Buy and Sell Signal Conditions:

- Signals are evaluated based on the 30-period Exponential Moving Average (EMA) and the mitigation time (time\_).

The following table summarizes the specific conditions:

| Signal Type | EMA conditions | Time conditions |
| --- | --- | --- |
| Buy | 30-period EMA is **below** the close of candlestick 1 | time\_ must be greater than current time |
| Entry for Sell | 30-period EMA is **above** the close of candlestick 1 | time\_b must be greater than current time |

**Note:** These conditions ensure the signal is generated within a 5-candlestick margin after order block mitigation.

Actions Upon Condition Fulfillment or Non-fulfillment:

| Status | Action |
| --- | --- |
| Fulfilled | Fill the TP and SL buffers to execute the corresponding trade. |
| Not fulfilled | Reset the searcher to true and time\_ and time\_b to 0, allowing the search for new order blocks to resume if the maximum time has elapsed. |

Block diagram:

Buy

![ Logic to Open Buy Position](https://c.mql5.com/2/172/Logic_Open_Buy_Position_1__1.png)

Sell

![Logic to Open Sell Position](https://c.mql5.com/2/172/Logic_Open_Sell_Position_1__1.png)

### Implementing a trading strategy

Before we begin, we will create an exponential moving average handle.

We create global variables (array and handle):

```
int hanlde_ma;
double ma[];
```

In OnInit, we initialize the handle and check if it has a value assigned to it.

```
hanlde_ma = iMA(_Symbol,_Period,30,0,MODE_EMA,PRICE_CLOSE);

if(hanlde_ma == INVALID_HANDLE)
{
Print("The EMA indicator is not available. Failure: ", _LastError);
return INIT_FAILED;
}
```

We declare static variables to control the search state and OB activation time, distinguishing between buy and sell scenarios.

```
//Variables for buy
static bool buscar_oba = true;
static datetime time_ = 0;

//Variables for sell
static bool buscar_obb = true;
static datetime time_b = 0;
```

We then loop through the soft order blocks (similar to what we did in the previous article for alerts).

We begin by adding conditions:

```
//Bullish case
 if(buscar_oba == true)
//Bearish case
 if(buscar_obb == true)
```

The next step is to determine whether the OB has been mitigated, that is, whether the price has interacted with it. If a mitigated OB is found, its time is recorded and the search is suspended. This is done for both bullish and bearish scenarios.

```
// Bearish case
for(int i = 0; i < ArraySize(ob_bajistas); i++) {
    if(ob_bajistas[i].mitigated == true &&
       !Es_Eliminado_PriceTwo(ob_bajistas[i].name, pricetwo_eliminados_obb) &&
       ObjectFind(ChartID(), ob_bajistas[i].name) >= 0) {

        Alert("The bearishorder block is being mitigated: ", TimeToString(ob_bajistas[i].time1));
        buscar_obb = false;  // Pause search
        time_b = iTime(_Symbol,_Period,1);  //  Record the mitigation time
        Agregar_Index_Array_1(pricetwo_eliminados_obb, ob_bajistas[i].name);
        break;
    }
}

// Bullish case
for(int i = 0; i < ArraySize(ob_alcistas); i++) {
    if(ob_alcistas[i].mitigated == true &&
       !Es_Eliminado_PriceTwo(ob_alcistas[i].name, pricetwo_eliminados_oba) &&
       ObjectFind(ChartID(), ob_alcistas[i].name) >= 0) {

        Alert("The bullish order block is mitigated: ", TimeToString(ob_alcistas[i].time1));
        time_ = iTime(_Symbol,_Period,0);
        Agregar_Index_Array_1(pricetwo_eliminados_oba, ob_alcistas[i].name);
        buscar_oba = false;  // Pause search
        break;
    }
}
```

This section ensures that the system stops searching once a mitigation is detected, avoiding duplicate signals.

Initial Condition for Executing a Trade

The strategy uses specific conditions to trigger the search for buy or sell signals once an OB has been mitigated and while the maximum waiting time has not been exceeded.

```
// Buy
if(buscar_oba == false && time_ > 0 && new_vela) { /* Code for Buy */ }

// Sell
if(buscar_obb == false && time_b > 0 && new_vela) { /* Code for Sell */ }
```

In these conditions:

1. buscar\_oba or buscar\_obb must be false (confirming a prior mitigation).
2. time\_ or time\_b must be greater than 0, indicating that a time has been recorded.
3. new\_vela ensures that the logic is applied only on a new candle, helping to prevent repeated actions.

Validation of Buy or Sell Conditions

To establish the necessary conditions, first we need a variable to store the maximum waiting time. Then, it is essential to know the closing price of candlestick 1 and its EMA (Exponential Moving Average). To obtain the close, we use the iClose function, and we store the EMA values in an array containing the full historical series of the moving average.

```
// Buy
double close_ = NormalizeDouble(iClose(_Symbol,_Period,1),_Digits);
datetime max_time_espera = time_ + (PeriodSeconds() * 5);

if(close_ > ma[1] && iTime(_Symbol,_Period,0) <= max_time_espera) {
    // Code for Buy...
}

// Sell
close_ = NormalizeDouble(iClose(_Symbol,_Period,1),_Digits);
max_time_espera = time_b + (PeriodSeconds() * 5);

if(close_ < ma[1] && iTime(_Symbol,_Period,0) <= max_time_espera) {
    // Code for Sell...
}
```

Resetting the Order Block Search

Finally, if the maximum waiting time is exceeded without the conditions being met, the code resets the search to allow detection of new OBs:

```
// Reset for Buy
if(iTime(_Symbol,_Period,0) > max_time_espera) {
    time_ = 0;
    buscar_oba = true;
}

// Reset for Sell
if(iTime(_Symbol,_Period,0) > max_time_espera) {
    time_b = 0;
    buscar_obb = true;
}
```

Now we are missing a function to draw tp and sl, as well as to add them to the buffers. We achieve this via the following code:

Let's proceed to new sections.

### Setting Take Profit (TP) and Stop Loss (SL) levels

In this section, we will develop the function GetTP\_SL, which calculates the TP and SL levels using two methods: Using either the ATR (Average True Range) or fixed points, as previously mentioned in the input configuration.

1: Function Definition

The GetTP\_SL function receives the following parameters: the position's opening price, the position type (ENUM\_POSITION\_TYPE), and references for the TP and SL levels (tp1, tp2, sl1, and sl2), where the calculated values will be stored.

```
void GetTP_SL(double price_open_position, ENUM_POSITION_TYPE type, double &tp1, double &tp2, double &sl1, double &sl2)
```

2: Obtaining the ATR

To calculate ATR-based levels, we first need an array that stores the ATR value of the latest candle. We use CopyBuffer to fill the atr array with the current value.

```
double atr[];
ArraySetAsSeries(atr, true);
CopyBuffer(atr_i, 0, 0, 1, atr);
```

3: Calculating TP and SL Based on ATR

When tp\_sl\_style is set to ATR, we calculate TP and SL levels by multiplying the ATR value by the defined multipliers (Atr\_Multiplier\_1 and Atr\_Multiplier\_2). These values are then added or subtracted from the opening price depending on the position type.

```
if (type == POSITION_TYPE_BUY) {
    sl1 = price_open_position - (atr[0] * Atr_Multiplier_1);
    sl2 = price_open_position - (atr[0] * Atr_Multiplier_2);
    tp1 = price_open_position + (atr[0] * Atr_Multiplier_1);
    tp2 = price_open_position + (atr[0] * Atr_Multiplier_2);
}

if (type == POSITION_TYPE_SELL) {
    sl1 = price_open_position + (atr[0] * Atr_Multiplier_1);
    sl2 = price_open_position + (atr[0] * Atr_Multiplier_2);
    tp1 = price_open_position - (atr[0] * Atr_Multiplier_1);
    tp2 = price_open_position - (atr[0] * Atr_Multiplier_2);
}
```

4: Calculating TP and SL Based on Points

When tp\_sl\_style is set to POINT, we add or subtract the specified points (TP\_POINT and SL\_POINT), multiplied by the current symbol's point value (\_Point), to the opening price. This provides a simpler alternative to ATR-based calculation.

```
if (type == POSITION_TYPE_BUY) {
    sl1 = price_open_position - (SL_POINT * _Point);
    sl2 = price_open_position - (SL_POINT * _Point * 2);
    tp1 = price_open_position + (TP_POINT * _Point);
    tp2 = price_open_position + (TP_POINT * _Point * 2);
}

if (type == POSITION_TYPE_SELL) {
    sl1 = price_open_position + (SL_POINT * _Point);
    sl2 = price_open_position + (SL_POINT * _Point * 2);
    tp1 = price_open_position - (TP_POINT * _Point);
    tp2 = price_open_position - (TP_POINT * _Point * 2);
}
```

### Visualizing TP and SL Levels on the Chart

We will create a function to draw the TP and SL levels on the chart using lines and text objects.

Creating Lines

```
bool TrendCreate(long            chart_ID,        // Chart ID
                 string          name,            // Line name
                 int             sub_window,      // Subwindow index
                 datetime              time1,           // Time of the first point
                 double                price1,          // Price of the first point
                 datetime              time2,           // Time of the second point
                 double                price2,          // Price of the second point
                 color           clr,         // Line color
                 ENUM_LINE_STYLE style,       // Line style
                 int             width,       // Line width
                 bool            back,        // in the background
                 bool            selection    // Selectable form moving
                )
  {
   ResetLastError();
   if(!ObjectCreate(chart_ID,name,OBJ_TREND,sub_window,time1,price1,time2,price2))
     {
      Print(__FUNCTION__,
            ": ¡Failed to create trend line! Error code = ",GetLastError());
      return(false);
     }

   ObjectSetInteger(chart_ID,name,OBJPROP_COLOR,clr);
   ObjectSetInteger(chart_ID,name,OBJPROP_STYLE,style);
   ObjectSetInteger(chart_ID,name,OBJPROP_WIDTH,width);
   ObjectSetInteger(chart_ID,name,OBJPROP_BACK,back);
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTABLE,selection);
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTED,selection);
   ChartRedraw(chart_ID);
   return(true);
  }
```

For texts:

```
bool TextCreate(long              chart_ID,                // Chart ID
                string            name,                    // Object name
                int               sub_window,              // Subwindow index
                datetime                time,                   // Anchor time
                double                  price,                  // Anchor price
                string            text,              // the text
                string            font,              // Font
                int               font_size,         // Font size
                color             clr,               // color
                double            angle,             // Text angle
                ENUM_ANCHOR_POINT anchor,            // Anchor point
                bool              back=false,               // font
                bool              selection=false)          // Selectable for moving

  {

//--- reset error value
   ResetLastError();
//--- create "Text" object
   if(!ObjectCreate(chart_ID,name,OBJ_TEXT,sub_window,time,price))
     {
      Print(__FUNCTION__,
            ": ¡Failed to create object \"Text\"! Error code = ",GetLastError());
      return(false);
     }
   ObjectSetString(chart_ID,name,OBJPROP_TEXT,text);
   ObjectSetString(chart_ID,name,OBJPROP_FONT,font);
   ObjectSetInteger(chart_ID,name,OBJPROP_FONTSIZE,font_size);
   ObjectSetDouble(chart_ID,name,OBJPROP_ANGLE,angle);
   ObjectSetInteger(chart_ID,name,OBJPROP_ANCHOR,anchor);
   ObjectSetInteger(chart_ID,name,OBJPROP_COLOR,clr);
   ObjectSetInteger(chart_ID,name,OBJPROP_BACK,back);
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTABLE,selection);
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTED,selection);

   ChartRedraw(chart_ID);
   return(true);
  }
```

Now let's move on to creating the function.

**Step 1. Input Details**

The function receives the following parameters:

- tp1 и tp2 — Values for the two Take Profit levels.
- sl1 и sl2 — Values for the two Stop Loss levels.

```
void DrawTP_SL( double tp1, double tp2, double sl1, double sl2)
```

**Step 2: Preparing Times**

First, create a string curr\_time to store the current date and time of the candlestick on the chart. Then, calculate extension\_time, which extends 15 periods ahead from the current time to project the TP and SL lines to the right. text\_time is used to adjust the text label positions slightly beyond extension\_time.

```
string curr_time = TimeToString(iTime(_Symbol, _Period, 0));
datetime extension_time = iTime(_Symbol, _Period, 0) + (PeriodSeconds(PERIOD_CURRENT) * 15);
datetime text_time = extension_time + (PeriodSeconds(PERIOD_CURRENT) * 2);
```

**Step 3: Drawing TP and SL Lines and Labels**

1. **Take Profit 1 (tp1)**:

- Draw a green dotted line (STYLE\_DOT) at tp1 using TrendCreate.
- Add a text label "TP1" at the tp1 position with TextCreate.

```
TrendCreate(ChartID(), curr_time + " TP1", 0, iTime(_Symbol, _Period, 0), tp1, extension_time, tp1, clrGreen, STYLE_DOT, 1, true, false);
TextCreate(ChartID(), curr_time + " TP1 - Text", 0, text_time, tp1, "TP1", "Arial", 8, clrGreen, 0.0, ANCHOR_CENTER);
```

2. **Take Profit 2 (tp2)**:

- Draw another green dotted line at tp2 and add a "TP2" text label.

```
TrendCreate(ChartID(), curr_time + " TP2", 0, iTime(_Symbol, _Period, 0), tp2, extension_time, tp2, clrGreen, STYLE_DOT, 1, true, false);
TextCreate(ChartID(), curr_time + " TP2 - Text", 0, text_time, tp2, "TP2", "Arial", 8, clrGreen, 0.0, ANCHOR_CENTER);
```

3. **Stop Loss 1 (sl1)**:

- Draw a red dotted line at sl1 and a "SL1" text label.

```
TrendCreate(ChartID(), curr_time + " SL1", 0, iTime(_Symbol, _Period, 0), sl1, extension_time, sl1, clrRed, STYLE_DOT, 1, true, false);
TextCreate(ChartID(), curr_time + " SL1 - Text", 0, text_time, sl1, "SL1", "Arial", 8, clrRed, 0.0, ANCHOR_CENTER);
```

4. **Stop Loss 2 (sl2)**:

- Similarly, draw a red line at sl2 and a "SL2" text label.

```
TrendCreate(ChartID(), curr_time + " SL2", 0, iTime(_Symbol, _Period, 0), sl2, extension_time, sl2, clrRed, STYLE_DOT, 1, true, false);
TextCreate(ChartID(), curr_time + " SL2 - Text", 0, text_time, sl2, "SL2", "Arial", 8, clrRed, 0.0, ANCHOR_CENTER);
```

Complete code:

```
void DrawTP_SL(double tp1, double tp2, double sl1, double sl2)
  {

   string  curr_time = TimeToString(iTime(_Symbol,_Period,0));
   datetime extension_time = iTime(_Symbol,_Period,0) + (PeriodSeconds(PERIOD_CURRENT) * 15);
   datetime   text_time = extension_time + (PeriodSeconds(PERIOD_CURRENT) * 2);

   TrendCreate(ChartID(),curr_time+" TP1",0,iTime(_Symbol,_Period,0),tp1,extension_time,tp1,clrGreen,STYLE_DOT,1,true,false);
   TextCreate(ChartID(),curr_time+" TP1 - Text",0,text_time,tp1,"TP1","Arial",8,clrGreen,0.0,ANCHOR_CENTER);

   TrendCreate(ChartID(),curr_time+" TP2",0,iTime(_Symbol,_Period,0),tp2,extension_time,tp2,clrGreen,STYLE_DOT,1,true,false);
   TextCreate(ChartID(),curr_time+" TP2 - Text",0,text_time,tp2,"TP2","Arial",8,clrGreen,0.0,ANCHOR_CENTER);

   TrendCreate(ChartID(),curr_time+" SL1",0,iTime(_Symbol,_Period,0),sl1,extension_time,sl1,clrRed,STYLE_DOT,1,true,false);
   TextCreate(ChartID(),curr_time+" SL1 - Text",0,text_time,sl1,"SL1","Arial",8,clrRed,0.0,ANCHOR_CENTER);

   TrendCreate(ChartID(),curr_time+" SL2",0,iTime(_Symbol,_Period,0),sl2,extension_time,sl2,clrRed,STYLE_DOT,1,true,false);
   TextCreate(ChartID(),curr_time+" SL2 - Text",0,text_time,sl2,"SL2","Arial",8,clrRed,0.0,ANCHOR_CENTER);

  }
```

### Adding buffers for TP and SL levels (4)

As we did for the two buffers storing price2, we create additional buffers for TP and SL:

```
#property indicator_label3 "Take Profit 1"
#property indicator_label4 "Take Profit 2"
#property indicator_label5 "Stop Loss 1"
#property indicator_label6 "Stop Loss 2"
```

We increase the number of plots and buffers from 2 to 6.

```
#property  indicator_buffers 6
#property  indicator_plots 6
```

Create an array of buffers:

```
double tp1_buffer[];
double tp2_buffer[];
double sl1_buffer[];
double sl2_buffer[];
```

Initialize the arrays and set them as series.

```
SetIndexBuffer(2, tp1_buffer, INDICATOR_DATA);
SetIndexBuffer(3, tp2_buffer, INDICATOR_DATA);

SetIndexBuffer(4, sl1_buffer, INDICATOR_DATA);
SetIndexBuffer(5, sl2_buffer, INDICATOR_DATA);

ArraySetAsSeries(buyOrderBlockBuffer, true);
ArraySetAsSeries(sellOrderBlockBuffer, true);
ArrayFill(buyOrderBlockBuffer, 0,0, EMPTY_VALUE); // Initialize to EMPTY_VALUE
ArrayFill(sellOrderBlockBuffer, 0,0, EMPTY_VALUE); // Initialize to EMPTY_VALUE

ArraySetAsSeries(tp1_buffer, true);
ArraySetAsSeries(tp2_buffer, true);
ArrayFill(tp1_buffer, 0, 0, EMPTY_VALUE); // Initialize to EMPTY_VALUE
ArrayFill(tp2_buffer, 0, 0, EMPTY_VALUE); // Initialize to EMPTY_VALUE

ArraySetAsSeries(sl1_buffer, true);
ArraySetAsSeries(sl2_buffer, true);
ArrayFill(sl1_buffer, 0, 0, EMPTY_VALUE); // Initialize to EMPTY_VALUE
ArrayFill(sl2_buffer, 0, 0, EMPTY_VALUE); // Initialize to EMPTY_VALUE
```

This ensures the TP and SL values are stored correctly and displayed on the chart.

### Finalizing Main Code and Cleanup

To complete the indicator, implement cleanup and optimization code. This improves backtesting performance and frees memory resources for arrays, such as those storing OrderBlocks, once they are no longer needed.

**1\. Clearing Arrays**

Within OnCalculate, monitor for a new daily candlestick. Use a **global variable** to store the last candlestick time.

```
datetime    tiempo_ultima_vela_1;
```

Each time a new daily candlestick opens, release memory from the arrays to prevent old data accumulation and optimize performance.

```
 if(tiempo_ultima_vela_1 != iTime(_Symbol,PERIOD_D1,  0))
     {
      Eliminar_Objetos();

      ArrayFree(ob_bajistas);
      ArrayFree(ob_alcistas);
      ArrayFree(pricetwo_eliminados_oba);
      ArrayFree(pricetwo_eliminados_obb);

      tiempo_ultima_vela_1 = iTime(_Symbol,PERIOD_D1,  0);
     }
```

**2\. Modifying OnDeinit**

In OnDeinit, release the EMA indicator handle and clear all arrays. This ensures no memory resources remain when the indicator is removed.

```
void OnDeinit(const int reason)
  {
   Eliminar_Objetos();

   ArrayFree(ob_bajistas);
   ArrayFree(ob_alcistas);
   ArrayFree(pricetwo_eliminados_oba);
   ArrayFree(pricetwo_eliminados_obb);

   if(atr_i  != INVALID_HANDLE)
      IndicatorRelease(atr_i);
   if(hanlde_ma != INVALID_HANDLE) //EMA
      IndicatorRelease(hanlde_ma);

   ResetLastError();

    if(MarketBookRelease(_Symbol)) //Verify if closure was successful
     Print("Order book successfully closed for: " , _Symbol); //Print success message if so
   else
     Print("Order book closed with errors for: " , _Symbol , "   Last error: " , GetLastError()); //Print error message with code if not
  }
```

**3\. Object Deletion Function**

The Eliminar\_Objetos function has been optimized to remove TP and SL lines along with order block rectangles, ensuring the chart remains clean.

```
void Eliminar_Objetos()
  {

   for(int i = 0 ; i < ArraySize(ob_alcistas) ; i++) // iterate through the array of bullish order blocks
     {
      ObjectDelete(ChartID(),ob_alcistas[i].name);   // delete the object using the order block's name
     }
   for(int n = 0 ; n < ArraySize(ob_bajistas) ; n++) // iterate through the array of bearish order blocks
     {
      ObjectDelete(ChartID(),ob_bajistas[n].name);   // delete the object using the order block's name
     }
 //Delete all TP and SL lines
   ObjectsDeleteAll(0," TP",-1,-1);
   ObjectsDeleteAll(0," SL",-1,-1);
  }
```

**4\. Initial Setup in OnInit**

Configure the indicator short name and chart plot labels to ensure proper labeling in the data window.

```
   string short_name = "Order Block Indicator";
   IndicatorSetString(INDICATOR_SHORTNAME,short_name);

// Set data precision for digits

// Assign labels for each plot
   PlotIndexSetString(0, PLOT_LABEL, "Bullish Order Block");
   PlotIndexSetString(1, PLOT_LABEL, "Bearish Order Block");
   PlotIndexSetString(2, PLOT_LABEL, "Take Profit 1");
   PlotIndexSetString(3, PLOT_LABEL, "Take Profit 2");
   PlotIndexSetString(4, PLOT_LABEL, "Stop Loss 1");
   PlotIndexSetString(5, PLOT_LABEL, "Stop Loss 2");
```

**5\. Setting TP and SL Levels When Opening Trades**

Finally, we set Take Profit and Stop Loss levels for buy and sell trades. For **buy** trades, use the Ask price; for **sell** trades, use the Bid price. Then draw the TP and SL lines on the chart for monitoring.

```
//Buy
double ask= NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);

double tp1;
double tp2;
double sl1;
double sl2;
GetTP_SL(ask,POSITION_TYPE_BUY,tp1,tp2,sl1,sl2);

DrawTP_SL(tp1,tp2,sl1,sl2);

tp1_buffer[iBarShift(_Symbol,PERIOD_CURRENT,iTime(_Symbol,_Period,0))] = tp1;
tp2_buffer[iBarShift(_Symbol,PERIOD_CURRENT,iTime(_Symbol,_Period,0))] = tp2;
sl1_buffer[iBarShift(_Symbol,PERIOD_CURRENT,iTime(_Symbol,_Period,0))] = sl1;
sl2_buffer[iBarShift(_Symbol,PERIOD_CURRENT,iTime(_Symbol,_Period,0))] = sl2;

time_ = 0;
buscar_oba = true;

//Sell

double bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);
double tp1;
double tp2;
double sl1;
double sl2;
GetTP_SL(bid,POSITION_TYPE_SELL,tp1,tp2,sl1,sl2);

DrawTP_SL(tp1,tp2,sl1,sl2);

tp1_buffer[iBarShift(_Symbol,PERIOD_CURRENT,iTime(_Symbol,_Period,0))] = tp1;
tp2_buffer[iBarShift(_Symbol,PERIOD_CURRENT,iTime(_Symbol,_Period,0))] = tp2;
sl1_buffer[iBarShift(_Symbol,PERIOD_CURRENT,iTime(_Symbol,_Period,0))] = sl1;
sl2_buffer[iBarShift(_Symbol,PERIOD_CURRENT,iTime(_Symbol,_Period,0))] = sl2;

time_b = 0;
buscar_obb = true;
```

| Step | Buy | Sell |
| --- | --- | --- |
| Price: | Get and normalize Ask. | Get and normalize Bid. |
| Variables: | Initialize variables to store Take Profit and Stop Loss values.<br>(tp1, tp2, sl1 и sl2). | The same variables are used to store Take Profit and Stop Loss levels.<br>(tp1, tp2, sl1 и sl2). |
| Calculation: | GetTP\_SL calculates TP and SL levels based on Ask price for a buy deal. | GetTP\_SL calculates TP and SL levels based on Bid price for a sell deal. |
| Drawing: | DrawTP\_SL visually displays on the chart TP and SL levels for buy deals. | DrawTP\_SL visually displays on the chart TP and SL levels for sell deals. |
| Buffer: | Use iBarShift to find current bar index and store TP/SL in buffers.<br>  (tp1\_buffer, tp2\_buffer, sl1\_buffer и sl2\_buffer). | to find current bar index and store TP/SL in the same buffers.<br> (tp1\_buffer, tp2\_buffer, sl1\_buffer и sl2\_buffer). |
| Static variables: | Reset static variables to search for new blocks of bullish orders in the next iteration.<br>(Static variables: "time\_" and "buscar\_oba"). | Reset static variables to search for new blocks of bearish orders in the next iteration.<br>(Static variables: "time\_b" and "search\_obb"). |

### Conclusion

In this article, we explored how to create an Order Block indicator based on market depth volume and optimized its functionality by adding additional buffers to the original indicator.

Our final result:

![Final Example GIF](https://c.mql5.com/2/172/Grabar_2024_11_04_21_58_19_130__1.gif)

With this section, we conclude the development of our Order Blocks indicator. In the upcoming articles, we will cover the creation of a risk management class from scratch and develop a trading bot that integrates this risk management, using the signal buffers from our indicator to make more precise and automated trading decisions.

Translated from Spanish by MetaQuotes Ltd.

Original article: [https://www.mql5.com/es/articles/16268](https://www.mql5.com/es/articles/16268)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16268.zip "Download all attachments in the single ZIP archive")

[Order\_Block\_Indicador\_New\_Part\_2.mq5](https://www.mql5.com/en/articles/download/16268/order_block_indicador_new_part_2.mq5 "Download Order_Block_Indicador_New_Part_2.mq5")(133.11 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Risk Management (Part 2): Implementing Lot Calculation in a Graphical Interface](https://www.mql5.com/en/articles/16985)
- [Risk Management (Part 1): Fundamentals for Building a Risk Management Class](https://www.mql5.com/en/articles/16820)
- [Developing Advanced ICT Trading Systems: Implementing Order Blocks in an Indicator](https://www.mql5.com/en/articles/15899)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/496759)**
(1)


![Vladislav Boyko](https://c.mql5.com/avatar/2025/12/692e1587-6181.png)

**[Vladislav Boyko](https://www.mql5.com/en/users/boyvlad)**
\|
4 Oct 2025 at 19:24

[https://www.mql5.com/en/articles/16268](https://www.mql5.com/en/articles/16268)

**5\. Setting TP and SL Levels When Opening Trades**

Finally, we set Take Profit and Stop Loss levels for buy and sell trades. For **buy** trades, use the Ask price; for **sell** trades, use the Bid price. Then draw the TP and SL lines on the chart for monitoring.

```
tp1_buffer[iBarShift(_Symbol,PERIOD_CURRENT,iTime(_Symbol,_Period,0))] = tp1;
tp2_buffer[iBarShift(_Symbol,PERIOD_CURRENT,iTime(_Symbol,_Period,0))] = tp2;
sl1_buffer[iBarShift(_Symbol,PERIOD_CURRENT,iTime(_Symbol,_Period,0))] = sl1;
sl2_buffer[iBarShift(_Symbol,PERIOD_CURRENT,iTime(_Symbol,_Period,0))] = sl2;
```

This looks like it could be a bit simplified.

![Automating Trading Strategies in MQL5 (Part 36): Supply and Demand Trading with Retest and Impulse Model](https://c.mql5.com/2/173/19674-automating-trading-strategies-logo__2.png)[Automating Trading Strategies in MQL5 (Part 36): Supply and Demand Trading with Retest and Impulse Model](https://www.mql5.com/en/articles/19674)

In this article, we create a supply and demand trading system in MQL5 that identifies supply and demand zones through consolidation ranges, validates them with impulsive moves, and trades retests with trend confirmation and customizable risk parameters. The system visualizes zones with dynamic labels and colors, supporting trailing stops for risk management.

![Price Action Analysis Toolkit Development (Part 43): Candlestick Probability and Breakouts](https://c.mql5.com/2/173/19738-price-action-analysis-toolkit-logo__1.png)[Price Action Analysis Toolkit Development (Part 43): Candlestick Probability and Breakouts](https://www.mql5.com/en/articles/19738)

Enhance your market analysis with the MQL5-native Candlestick Probability EA, a lightweight tool that transforms raw price bars into real-time, instrument-specific probability insights. It classifies Pinbars, Engulfing, and Doji patterns at bar close, uses ATR-aware filtering, and optional breakout confirmation. The EA calculates raw and volume-weighted follow-through percentages, helping you understand each pattern's typical outcome on specific symbols and timeframes. On-chart markers, a compact dashboard, and interactive toggles allow easy validation and focus. Export detailed CSV logs for offline testing. Use it to develop probability profiles, optimize strategies, and turn pattern recognition into a measurable edge.

![Neural Networks in Trading: A Hybrid Trading Framework with Predictive Coding (Final Part)](https://c.mql5.com/2/107/Neural_networks_in_trading_Hybrid_trading_framework_ending_LOGO.png)[Neural Networks in Trading: A Hybrid Trading Framework with Predictive Coding (Final Part)](https://www.mql5.com/en/articles/16713)

We continue our examination of the StockFormer hybrid trading system, which combines predictive coding and reinforcement learning algorithms for financial time series analysis. The system is based on three Transformer branches with a Diversified Multi-Head Attention (DMH-Attn) mechanism that enables the capturing of complex patterns and interdependencies between assets. Previously, we got acquainted with the theoretical aspects of the framework and implemented the DMH-Attn mechanisms. Today, we will talk about the model architecture and training.

![Building AI-Powered Trading Systems in MQL5 (Part 3): Upgrading to a Scrollable Single Chat-Oriented UI](https://c.mql5.com/2/173/19741-building-ai-powered-trading-logo__1.png)[Building AI-Powered Trading Systems in MQL5 (Part 3): Upgrading to a Scrollable Single Chat-Oriented UI](https://www.mql5.com/en/articles/19741)

In this article, we upgrade the ChatGPT-integrated program in MQL5 to a scrollable single chat-oriented UI, enhancing conversation history display with timestamps and dynamic scrolling. The system builds on JSON parsing to manage multi-turn messages, supporting customizable scrollbar modes and hover effects for improved user interaction.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/16268&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5048832484562542475)

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