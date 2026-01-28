---
title: Considering Orders in a Large Program
url: https://www.mql5.com/en/articles/1390
categories: Trading, Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:43:28.047828
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/1390&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083068094080881742)

MetaTrader 4 / Examples


If a trading strategy presupposes working with a small amount of orders (for example, with only one Buy order and one Sell order), the expert that realizes such strategy may do without recording of orders. It just makes trading decisions based on its own ideology, i.e., on that there cannot be more than one one-way order in the terminal.

This approach is basically acceptable, but it only suits for a small program and cannot qualify for being universal. Really, what shall this expert do if there are two or more orders of a kind for the same symbol? One may contradict that the trading strategy does not predict such a situation and the latter may not occur. This is right, but it also reveals how defective and limited this conception is.

As soon as the trader has enough experience to trade using several orders (including pending ones), it will be necessary to consider all orders available since pending orders can turn into opened ones. Moreover, it is known that traders can always open an order manually left behind that the expert works in the terminal. It is also possible that several traders trade on the same account. And in a more general case, the trading strategy itself can presuppose using of a great amount of orders.

Let us try and get into details of how we can better consider orders in a large program.

### 1\. Environment Characteristic and Generalities

Let us briefly describe characteristics of the environment our program to work in.

1\. First of all, we surely want to know (consider that the data will be collected by the program) how many and what orders we have. It is insufficient just to mention that we have, say, 3 orders Sell and 4 orders Buy. Most likely, any intelligence trading system contains an algorithm to control orders characteristics - StopLoss and TakeProfit levels, and OpenPrice for pending orders. Besides, we will need to know how much each order costs, what is its expiration time, and what program has been used to place it.

2\. Ticks. Any information makes sense if it is considered in time. Any expert deals with quotes that income in the terminal. Evidently, the state of orders can change on any price change, it means that orders must be "reconsidered" at every tick.

3\. The symbol, in the window of which the expert has been attached, can be subsumed under "general characteristic".

Based on these simple data, we can determine general principles of how to build the program to consider orders.

**First**, except for considering orders, the trading system can and must solve some other problems, for example, analyze the availability and quality of orders and, regarding these data, make trade decisions. Any analysis of the current situation is possible if the situation is known. This means the program code that is responsible for considering orders must be executed **before** other codes are. Best of all would be to form this code of consideration as a separate function. Let us name it Terminal() since its main task will be to watch orders in the terminal.

**Second**, data accumulated by the Terminal() function must be available to other functions. This is why the variables bearing useful information about orders must be declared at the **global** level (not to be confused with GlobalVariables).

**Third**, since we deal with information sets, it would be logical to organize the considering based on **arrays** of orders.

### 2\. Program's Structure

The preceding reasoning lets us design a general structure of the program containing orders considering:

```
// My_expert.mq4
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
#include ...
#include <Peremen.mq4>    // Description of the expert's variables.
#include <Terminal.mq4>   // Attach the Terminal function.
#include <Sobytiya.mq4>   // Attach the Sobytiya function.
#include ...
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
//
//
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int init()
   {
   ...                    // Function code init()
   return;
   }
//=================================================================
int start()
   {
   Terminal();            // This function is the first in
                          // the sequence of functions
   ...                    // The posterior start() function code
   Sobytiya();            // Events processing function
   ...                    // The posterior start() function code
   return;
   }
//=================================================================
int deinit()
   {
   ...                    // The deinit() function code
   return;
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
```

The Peremen.mq4 file describing variables must contain the description of arrays that give data about orders.

```
//  Peremen.mq4
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
//==================================================================
int
...
//==================================================================
double
...
Mas_Ord_Tek[31][8], Mas_Ord_Old[31][8],
// The current and the old array of orders
// the 1st index = the order number in this array
// [][0] not defined
// [][1] order open rate   (absolute value of the rate)
// [][2] StopLoss orders   (absolute value of the rate)
// [][3] TakeProfit orders (absolute value of the rate)
// [][4] order number
// [][5] lots in the order (absolute value of the rate)
// [][6] order type 1=B,2=S,3=BL,4=SL,5=BS,6=SS
// [][7] Magic Number of the order
...
//=================================================================
string
...
//=================================================================
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
```

When global variables are defined in small programs, they are usually written before the first coming function. In middle-sized and large programs, it is reasonable to collect all global variables in one file and attach this file to the expert using the **#include** instruction. In terms of the code efficiency, the differences between the methods are not important, but in terms of usability, attachment of the file is more preferable. This is the way in which the Peremen.mq4 file was attached to the expert in the above example.

### 3\. Arrays

In the Peremen.mq4 file, two arrays are defined as global variables:

- Mas\_Ord\_Tek\[31\]\[8\] - the current array of orders;
- Mas\_Ord\_Old\[31\]\[8\] - the old array of orders.

Why do we need two, not one array? To be able to monitor the events. What events can that be, we will discuss a bit later, and now, let us just mention that an event itself is identified based on comparison of its initial state to its current state. If there are no data about either state, it is impossible to identify the event. In this case, arrays of orders are used that inform about the trading space status before and after the tick.

Why do we use a 2D array? Because we will consider several orders simultaneously and keep information about each of them on several characteristics. The first index is the order number in the array, the second one is the order characteristic on the given criterion. In the above example, the first index does not allow more than 30 orders to be available in the terminal. You can change this amount in your discretion allowing to consider, for example, 50 or 100 orders in the terminal simultaneously. Of course, it makes sense only if your trading system is planned to work with such amount of orders. In standard situations, 1 or 2 orders are used, in rarer cases - 4. 30, to my opinion, are too much.

Why are 31 and 8 instead of 30 and 7 are given in the subscript brackets of the arrays? The matter is that arrays in MQL4 start index counting with zero. The use of zero cells for common elements is far from being always justified. In my opinion, it would be logical to make the order number correspond with the array element number, for example, the third order must be in the line with index 3. In fact, this line will be the forth, but its index is 3 as the very first line has index 0.

Let us study the table below that visualizes the inside of arrays. Suppose there are only 3 orders: Buy, Sell and BuyStop of different quality:

![](https://c.mql5.com/2/13/massiv_4_.gif)

Information about orders is located in the array cells numbered from \[1\]\[1\] to \[7\]\[30\]. Besides, we will place the total amount of orders in the array into the cell indexed as \[0\]\[0\]. In this case, 3. In the further calculations, this number will be used to organize loops, in which the current state will be analyzed.

Thus, we can store information about 30 orders, each being marked by 7 characteristics.

If your trading strategy presupposes simultaneous trading on several symbols, you can open a number of arrays and order them according to the currency pairs. This method is quite permissible, but not very convenient. If the considering of orders is organized in this way, you would have to open another array to store names of arrays that would contain information about orders.

A more convenient solution would be to use one large array (one sole and one new) to consider all the orders. This choice is better because, when processing this array in the further code, there will be no need to search for the names of symbol order arrays. It will just be enough to organize search in the loop of a formal numeric variable by one of the array indices.

The array that contains information about all orders can be organized as follows:

![](https://c.mql5.com/2/13/massiv_2_2_.gif)

Information about orders for each symbol is stored in the same index plane, like in the above example. The difference consists in that, in this case, there is a number of such planes (in the Figure, there are 3 of them: yellow, pink, and green). Their total amount is equal to the amount of currency pairs we are going to work with, plus one more, the zero one. In this index plane (gray), the only value, the total amount of orders is stored in the cell indexed as \[0\]\[0\]\[0\].

The dimension of the array for, say, 25 symbols will be 26 for the first index. In this case, the Peremen.mq4 that discribes arrays will look as follows:

```
//  Peremen.mq4
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
//===========================================================================
int
...
//===========================================================================
double
...
Mas_Ord_Tek[26][31][8], Mas_Ord_Old[26][31][8],
//The current and the new array of orders
// the 1st index = symbol number
// the 2nd index = order number ...
// ... on the plane of the symbol
// [][][0] not defined
// [][][1] order open rate     (absolute value of the rate)
// [][][2] StopLoss orders     (absolute value of the rate)
// [][][3] TakeProfit orders   (absolute value of the rate)
// [][][4] order number
// [][][5] lots of the order   (absolute value of the rate)
// [][][6] order type 1=B,2=S,3=BL,4=SL,5=BS,6=SS
// [][][7] MagicNumber of the order
//=================================================================
string
      Instrument[26];                   // Array of symbol names
...
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
//
//
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int Predopred()
   {
//================================================= Predefinitions =
   Instrument[1] = "EURUSD";
   Instrument[2] = "GBPUSD";
   ...
   Instrument[25]= "AUDCAD";
//==================================================================
   return();
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
```

In the bottom of the block opening variables of this file, the Instrument\[26\] array has been opened.

A large program usually has many variables, the values of which do not change during the program execution. In our case, these are elements of the Instrument\[\] array. For programming to be more comfortable, it would be better to collect such variables in one function. This function can also be formed as a separate file attached to the program using #include.

In the below example, we can see the predefinition of the Instrument\[\] array elements in the Predopred() function contained in the same Peremen.mq4 file. It is enough to launch this function only once, so it makes sense to include it into the program in the special function named **init()**:

```
int init()
   {
   Predopred();         // Predefinition of some variables
   return;

   }
```

Thus, some integers contained, in our case, in the Instrument\[\] array index values are made correspondent with symbols.

### 4\. Function for Considering Orders

Now let us study the Terminal() function to consider orders for one symbol. This function is also organized as a separate file named Terminal.mq4 and attached to the expert using the #include instruction.

```
// Terminal.mq4
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int Terminal()
   {
//================================================ Predefinition ==
   ArrayCopy(Mas_Ord_Old, Mas_Ord_Tek);// Save the preceding history
   int Kol=0;                       // Zeroization of orders counter
   ArrayInitialize(Mas_Ord_Tek,0);     // Zeroization of the array
//=============================================== Order analysis ==
   for (int i=0; i<OrdersTotal(); i++)
// For all orders in the terminal
     {
      if((OrderSelect(i, SELECT_BY_POS)==true) &&
                      (OrderSymbol()==Symbol()))
                        //If there is the next and our currency pair
       {
        Kol++;                   // Count the total amount of orders
//---------------------------  Forming of the new array of orders --
        Mas_Ord_Tek[Kol][1] = NormalizeDouble(OrderOpenPrice(),
                                              Digits);
// Order open rate
        Mas_Ord_Tek[Kol][2] = NormalizeDouble(OrderStopLoss(),
                                              Digits);
// SL rate
        Mas_Ord_Tek[Kol][3] = NormalizeDouble(OrderTakeProfit(),
                                              Digits);
// ТР rate
        Mas_Ord_Tek[Kol][4] = OrderTicket();      // Order number
        Mas_Ord_Tek[Kol][5] = OrderLots();        // Count of lots
        Mas_Ord_Tek[Kol][6] = OrderType();        // Order type
        Mas_Ord_Tek[Kol][7] = OrderMagicNumber(); // Magic number
//-----------------------------------------------------------------
       }
     }
   Mas_Ord_Tek[0][0] = Kol;     // Save to the zero cell
//=================================================================
   return();
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
```

The first to be done after the system has passed control to the Terminal() function is to save the preceding state of the orders. It is in the Mas\_Ord\_Tek\[\]\[\] array, but is not actual at the moment since it reflects the state of orders formed on the preceding tick. The current state has not yet been known by the initial moment.

In the line

```
ArrayCopy(Mas_Ord_Old, Mas_Ord_Tek);
// Save the preceding history
```

the latest known state of orders is passed to the Mas\_Ord\_Old\[\]\[\] array.

The function uses the local variable named **Kol** and reflecting the new, current amount of orders. The necessity to use this variable arose since MQL4 does not permit to use indexed indices. As you can see from the code, it is used as the index value. Let us zeroize this variable, as well as the whole array of the current state and take interest in the actual state of things.

To do so, let us organize a loop for the terminal, i.e., call all available orders consecutively and find out about their characteristics

```
for (int i=0; i<OrdersTotal(); i++)
// For all orders in the terminal
```

Please note: OrdersTotal() determines the amount of orders, and counting of order numbers in the terminal starts with 0. This is why sign " **<**" is used in the loop line and the change of the internal variable of the loop starts with zero: i=0.

In the subsequent code, the order of the symbol, to which the expert is attached, is emphasized. Using functions that determine various order characteristics, the obtained information is placed in the corresponding elements of the Mas\_Ord\_Tek\[\]\[\] array.

```
//--------------------------- Forming of the new array of orders --
       Mas_Ord_Tek[Kol][1] = NormalizeDouble(OrderOpenPrice(),
                                             Digits);
// Order open rate
       Mas_Ord_Tek[Kol][2] = NormalizeDouble(OrderStopLoss() ,
                                             Digits);
// SL rate
       Mas_Ord_Tek[Kol][3] = NormalizeDouble(OrderTakeProfit(),
                                             Digits);
// ТР rate
       Mas_Ord_Tek[Kol][4] = OrderTicket();      // Order number
       Mas_Ord_Tek[Kol][5] = OrderLots();        // Count
                                                 // of lots
       Mas_Ord_Tek[Kol][6] = OrderType();        // Order type
       Mas_Ord_Tek[Kol][7] = OrderMagicNumber(); // Magic number
//------------------------------------------------------------------
```

The counter that is filled in passing will pass its value to the array zero element at the end of the loop.

```
Mas_Ord_Tek[0][0] = Kol;        // Save to the zero cell
```

For simultaneous trading on several symbols, the Tеrminal() function can be constructed as follows:

```
// Terminal.mq4
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int Terminal()
   {
//================================================== Predefinitions =
   ArrayCopy(Mas_Ord_Old, Mas_Ord_Tek);
// Save the preceding history
   ArrayInitialize(Mas_Ord_Tek,0);
// Zeroization of array of the current orders
//================================================== Order analysis =
   for (int i=0; i<OrdersTotal(); i++)
// For all orders of the terminal
      {
      if(OrderSelect(i, SELECT_BY_POS)==true)
// If there is a "living" order
         {
//--------------------------------------- Symbol index definition -
         string Symb=OrderSymbol();
// Determine the current order currency
         for (int ind=1; ind<=25; ind++)
// Search in the array of symbols
            {
            if (Symb==Instrument[ind])
// Index found, the search is complete
               break;            // Quit the loop by ind
            }
//------------------------ Forming of the new array of orders ----
         Mas_Ord_Tek[0][0][0]++;
// Count the total amount of orders
         Mas_Ord_Tek[ind][0][0]++;
// Count the amount of orders for a symbol
         int k=Mas_Ord_Tek[ind][0][0];   // Formal variable

         Mas_Ord_Tek[ind][k][1] = NormalizeDouble(OrderOpenPrice(),
                                                  Digits);
// Order open rate
         Mas_Ord_Tek[ind][k][2] = NormalizeDouble(OrderStopLoss(),
                                                  Digits);
// SL rate
         Mas_Ord_Tek[ind][k][3] = NormalizeDouble(OrderTakeProfit(),
                                                  Digits);
// ТР rate
         Mas_Ord_Tek[ind][k][4] = OrderTicket(); // Order number
         Mas_Ord_Tek[ind][k][5] = OrderLots();   // Count of lots
         Mas_Ord_Tek[ind][k][6] = OrderType();   // Order type
         Mas_Ord_Tek[ind][k][7] = OrderMagicNumber(); // Magic
                                                      // number

//-----------------------------------------------------------------
         }
      }
//=================================================================
   return();
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
```

Algorithm of considering orders for several symbols differs from that for only one symbol through one little block determining the symbol index.

```
//----------------------------------- Symbol index definition ----
         string Symb=OrderSymbol();
// Determine the current order currency
         for (int ind=1; ind<=25; ind++)
// Search in the array of symbols
            {
            if (Symb==Instrument[ind])
// Index found, the search is complete
               break;              // Quit the loop by ind
            }
```

In the consecutive lines of the Terminal() function, the found symbol index value **ind** determines the plane index of the array containing data about orders for the corresponding symbol.

In the above examples of the Terminal() function realization, normalization is used in calculation of some variables:

```
Mas_Ord_Tek[ind][k][1] = NormalizeDouble(OrderOpenPrice(),
                                         Digits);
// Order open rate
Mas_Ord_Tek[ind][k][2] = NormalizeDouble(OrderStopLoss(),
                                         Digits);
// SL rate
Mas_Ord_Tek[ind][k][3] = NormalizeDouble(OrderTakeProfit(),
                                         Digits);
// ТР rate
```

The necessity of these calculations is determined by the intention to use normalized values of variables in operators in future calculations. At that, the amount of valid digits is determined based on the predefined variable Digits.

A trading system can take into consideration other features and characteristics of orders. But in the examples above, only general principles of considering orders in a large program are given. For example, considering of closed and deleted orders is not discussed here, neither some characteristics of orders are taken into consideration, for example, expiration time for a pending order. If you need to process the above features, you can easily reconfigure arrays in order to store additional data. For instance, you can increase the amount of the last index elements to store expiration time values for pending orders.

### 5\. Events Processing

Well, a new tick is here and the Tеrminal() function has triggered. Control will be passed to operators that are localized immediately after the Terminal() function. One cannot predict in what part of the special function start() some or other event processing custom function must be placed - it depends on the idea on which the trading system is based. In a simple case, some events can be processed immediately.

Before giving the example of how to process events, I would like to remind that there are specific cases where the instinctively expected order of events will not be kept.

First, there are some dealers (banks, for example, with their specific accounting) that forcedly close all open orders at the end of day and open the same orders at the beginning of the next day, but at rates differing from the current one by some tenths of a point. These tenths of a point lay the swap into the order economy. The swap itself is shown as zero in the terminal. In terms of economy, dealers do not break any rules. For us, this situation is important for the newly opened orders get new numbers, which are absolutely different from the previous ones.

Second, partial closing of a separate order (at all dealers) is performed like in two stages. At the first stage, the order is completely closed. At the second stage, a new order with a decreased price and a new number is opened. At that, the initial order number is written in the new order's comments.

Third, similar specificity occurs when one order is closed using another if these orders have different prices. Practically, in this case, the situation of partial closing of one order is repeated.

The only opportunity to identify the order is to initialize such its characteristic that does not change or repeat. Of all the opportunities provided by MQL4, I could find only one such parameter - MagicNumber. But, even in this case, the programmer cannot reckon on creation of full control over the situation. The matter is that MagicNumber cannot be changed by programming. On the one hand, it bears an uncontested advantage: We can be sure that this parameter will not change, even accidentally, as long as the order exists. But, on the other hand, it is desirable to have an adjustable parameter, which can be accessed by a program. In the current situation, there is no such an opportunity, so, if an order (one or more) has not been opened by our trading system, but by another program or manually, there is no opportunity to mark these orders somehow in order to have an identifier that would differ them from "ours" and from each other.

A weak alternative to a unique identifier can be the time of order opening. But this criterion cannot be considered to be reliable and universal since it is theoretically possible that several orders will be opened simultaneously (at the same second). For example, two pending orders on different symbols can open at the same time if the price moves rapidly.

Let us study the realization example of the Sobytiya() function contained in the Sobytiya.mq4 file. We will watch a simple event - deletion or closing of an order. We have already known that order number does not always suit for this, so we will analyze MagicNumber.

```
// Sobytiya.mq4
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int Sobytiya()
   {
//==================================== Search for lost orders ====
   for (int oo=1;oo<=Mas_Ord_Old[0][0];oo++)
// Take the order number from an old array
      {
      int Sovpadenie=0;
// Begin with zeroizing of the match condition mark
      for (int i=1;i<=Mas_Ord_Tek[0][0];i++)
// Try to find this order in the current array
         {
         if (Mas_Ord_Tek[i][7]==Mas_Ord_Old[oo][7])
// If the order MagicNumber matches, then remember, i.e., it's
// still there and quit the internal loop
            {
            Sovpadenie=1;
            break;
            }
         }
      if (Sovpadenie==1) continue;
// Go to the next old array
      Alert("Order closed ",Mas_Ord_Old[oo][4]);
// Inform about this with a screen text
      PlaySound( "Close_order.wav" );  // And music.
      }
//=================================================================
   return();
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
```

As you can see, you can easily watch the event if you have criteria describing it at your disposal. It is enough to compare sequentially orders available before the tick with those available after the tick - and here it is, the difference, as well as the fact confirming that the event has happened.

In the next example, below, we consider the search for another event - order type modification. Here, the same principle is used, but it applies to another criterion. It is quite acceptable to apply order numbers and types analysis.

```
// Sobytiya_2.mq4
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
int Sobytiya_2()
   {
//============ Search for orders that have changed their types ====
   for(int oo=1;oo<=Mas_Ord_Old[0][0]; oo++)
// Go through the old array
      {
      for (int i=1;i<=Mas_Ord_Tek[0][0];i++)
// Search for the order in the current array
         {
         if (Mas_Ord_Tek[i][4] == Mas_Ord_Old[oo][4] &&
// If the old array contained the same order, but of a different type,
// then: The pending order becomes actual
             Mas_Ord_Tek[i][6] != Mas_Ord_Old[oo][6])
            {
            Alert("Order transformed",Mas_Ord_Tek[i][4]);
// Inform about this with a screen text
            PlaySound("Transform.wav" );   // And music
            }
         }
      }
//=================================================================
   return();
   }
//жжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжжж
```

Depending on the idea programmed in the expert by the programmer, events can be watched in different parts of the program. If necessary, analysis of all events of interest can be formed in one file. At that, the code should be modified several times in order not to run search of arrays oftentimes separately for each criterion. It is also easy to compose an algorithm to extract the initial number from the comments of the partly closed order for further analysis. To do so, it is enough to use the OrderComment() function.

Except for the above features that concern orders consideration, there are some others, the close look at which would help us to partly enlarge our reality awareness and save us from undue works and sad mistakes. Let us return to dealers that close orders at the end of trade day.

The swap value is not usually divisible by the point, which results in that the amount of significance in the order open rate value will be increased by 1. To take this characteristic into consideration, it is necessary to normalize this value with accuracy to 1 more than Digits:

```
Mas_Ord_Tek[ind][k][1] = NormalizeDouble(OrderOpenPrice(),
                                         Digits+1);
// Order open rate
```

Besides, this situation is remarkable for that the open order rate can essentially change its status. If a trader uses a trading system that makes trading decisions regarding order open rates, such a system can just go to pieces at the end of day. Upon a closer view, we can come to the conclusion that the order open rate is of no consequence at all! The important factor is the situation development trend and, if it is against us, we have to close the order wherever it is.

Discussion about criteria influencing trading decisions grows far out of this article. Here, however, it is still necessary to mention that a proper choice of criteria to be used in the program is the root factor for any strategy.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1390](https://www.mql5.com/ru/articles/1390)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [My First "Grail"](https://www.mql5.com/en/articles/1413)
- [Synchronization of Expert Advisors, Scripts and Indicators](https://www.mql5.com/en/articles/1393)
- [Graphic Expert Advisor: AutoGraf](https://www.mql5.com/en/articles/1378)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39219)**
(1)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
11 Mar 2009 at 04:23

```
Good article, could the author pls attach the 3 mq4 files for us to test.
Peremen.mq4
Terminal.mq4
Sobytiya.mq4
```

![My First "Grail"](https://c.mql5.com/2/13/144_2.png)[My First "Grail"](https://www.mql5.com/en/articles/1413)

Examined are the most frequent mistakes that lead the first-time programmers to creation of a "super-moneymaking" (when tested) trading systems. Exemplary experts that show fantastic results in tester, but result in losses during real trading are presented.

![A Pause between Trades](https://c.mql5.com/2/12/103_1.gif)[A Pause between Trades](https://www.mql5.com/en/articles/1355)

The article deals with the problem of how to arrange pauses between trade operations when a number of experts work on one МТ 4 Client Terminal. It is intended for users who have basic skills in both working with the terminal and programming in MQL 4.

![Secrets of MetaTrader 4 Client Terminal](https://c.mql5.com/2/13/158_10.png)[Secrets of MetaTrader 4 Client Terminal](https://www.mql5.com/en/articles/1415)

21 way to ease the life: Latent features in MetaTrader 4 Client Terminal.
Full screen; hot keys; Fast Navigation bar; minimizing windows; favorites; traffic reduction; disabling of news; symbol sets; Market Watch; templates for testing and independent charts; profiles; crosshair; electronic ruler; barwise chart paging; account history in the chart; types of pending orders; modifying of StopLoss and TakeProfit; undo deletion; chart print.

![MagicNumber: "Magic" Identifier of the Order](https://c.mql5.com/2/13/105_2.gif)[MagicNumber: "Magic" Identifier of the Order](https://www.mql5.com/en/articles/1359)

The article deals with the problem of conflict-free trading of several experts on the same МТ 4 Client Terminal. It "teaches" the expert to manage only "its own" orders without modifying or closing "someone else's" positions (opened manually or by other experts). The article was written for users who have basic skills of working with the terminal and programming in MQL 4.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=vgulucnprzrefoiandryvzxzpsrbugpc&ssn=1769251406219795847&ssn_dr=0&ssn_sr=0&fv_date=1769251406&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1390&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Considering%20Orders%20in%20a%20Large%20Program%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925140696252544&fz_uniq=5083068094080881742&sv=2552)

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