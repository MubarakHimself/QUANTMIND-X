---
title: Balancing risk when trading multiple instruments simultaneously
url: https://www.mql5.com/en/articles/14163
categories: Trading
relevance_score: 0
scraped_at: 2026-01-24T13:28:55.652607
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ixmmpkeshcwvbynksqlwyhpuylthexzh&ssn=1769250534470488628&ssn_dr=0&ssn_sr=0&fv_date=1769250534&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14163&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Balancing%20risk%20when%20trading%20multiple%20instruments%20simultaneously%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925053458593429&fz_uniq=5082889986082083046&sv=2552)

MetaTrader 5 / Examples


This article will touch upon the topic of balancing risk when trading multiple instruments intraday at the same time. The purpose of this article is to enable the user to write a code for balancing instruments from scratch, and to introduce experienced users to other, perhaps previously unused, implementations of old ideas. To do this, we will consider the definition of the concept of risk, select criteria for optimization, focus on the technical aspects of implementing our solution, analyze the set of standard terminal capabilities for such an implementation, and also touch on other possible ways to integrate this algorithm into your software infrastructure.

### Criteria for balancing trading instruments by risk

When trading several financial instruments simultaneously, we will take into account two main factors as risk balancing criteria.

- Symbol tick price
- Symbol average daily volatility

Tick price is a value in currency of the minimum price change on a symbol with a standard symbol lot. We take this criterion into account because the tick value may vary significantly on different instruments. For example, from the 1.27042 for EURGBP to 0.61374 for AUDNZD.

Average daily volatility is the characteristic change in the symbol price over one day. This value is less constant on different symbols than the previous selected criterion and can change over time depending on the market stage. For example, EURGBP usually moves on average about 336 points, while CHFJPY can move 1271 points on the same day, which is almost four times more. The data presented here characterizes the "usual" and "most probable" values of price volatility without taking into account the abnormally high volatility of symbol at certain moments when the price begins to move very strongly in one direction without a rollback. Below is an example of such a movement on USDJPY.

![Figure 1. Increased symbol volatility on D1 chart](https://c.mql5.com/2/80/USDJPYzDaily1.png)

Figure 1. Increased symbol volatility on D1 chart

This behavior can cause very serious risks to the deposit, which are described in sufficient detail in the article " [How to reduce trader's risks](https://www.mql5.com/en/articles/4233#n12)". In this article, we will propose the thesis that it is impossible to protect against such a risk by balancing instruments. This is a completely different category of risk. By balancing, we can protect the deposit against the risk that the market will go against our position within the framework of average volatility. If you want to protect your funds from abnormal movements, or "black swans", then use the following principles.

- do not take a large percentage of risk on a single symbol,
- do not strive to be in an open position constantly,
- do not put all funds under management into a single account with a single broker
- do not trade the same entries on different accounts and brokers simultaneously.

By following these principles, you can minimize losses if it so happens that the symbol price goes against you when you are in an open position. Now let's get back to considering the risks associated with standard volatility.

Simultaneous consideration of these two factors will allow you to balance the risks and normalize the expected profits for each currency pair while trading simultaneously without overweighting the risks on any single instrument. This approach will subsequently provide more homogeneous trading statistics for further analysis in the history of transactions and will reduce the error when the strategy optimizer works with this data and, accordingly, reduce the standard deviation of the sample from the average data. To understand in more detail how the value of the standard deviation affects the quality of analysis of a set, you can read the article " [Mathematics in trading: How to estimate trade results](https://www.mql5.com/en/articles/1492)". Now let's move on to choosing a container for storing data.

### Selecting containers for storing data

When selecting containers for storing data in our project, we will take into account the following factors:

- container performance
- memory requirement for its initialization
- availability of built-in functionality for data analysis
- ease of initialization via the user interface

The most common criteria when choosing containers for storing data are the containers performance and the need for computer memory to store them. Different types of storage can usually provide better performance when processing data, or a gain in the amount of memory occupied.

To perform a check, let's declare a simple array and a special container of the [vector](https://www.mql5.com/en/docs/basis/types/matrix_vector) type, while preliminarily initializing them with the [double](https://www.mql5.com/en/docs/basis/types/double) data type and identical values.

```
   double arr[] = {1.5, 2.3};
   vector<double> vect = {1.5, 2.3};
```

Use the [sizeof](https://www.mql5.com/en/docs/basis/operations/other) operation and determine the memory size that corresponds to the above types at the compilation stage.

```
Print(sizeof(arr));
Print(sizeof(vect));
```

As a result, we get 16 and 128 bytes. This difference in memory requirements for the [vector](https://www.mql5.com/en/docs/basis/types/matrix_vector) data type is determined by the presence of built-in functionality, including additional redundant memory allocation to ensure better performance.

Based on this, we will use a simple storage type as an array considering our tasks for containers that only require storing previously selected data. For types of homogeneous data that we will subsequently handle during our algorithm, it would be advisable to use the special [vector](https://www.mql5.com/en/docs/basis/types/matrix_vector) data type. Using this type will also save development time in terms of writing custom functions for standard operations, which are already implemented in [vector](https://www.mql5.com/en/docs/basis/types/matrix_vector) out of the box.

As a result, the data storage required for the calculation will look like this.

```
   string symbols[];       // symbols used for balancing

   double tick_val[],      // symbol tick price
          atr[],           // symbol volatility
          volume[],        // calculated position volume for symbols taking into account balancing
          point[];         // value of one price change point

   vector<double> risk_contract; // risk amount for a standard contract
```

Now let's move on to considering options for implementing solutions for entering data for balanced symbols.

### Selecting a symbol entering method

There are many solutions when choosing data entry methods in MetaTrader 5. Globally, they are divided into solutions aimed at interacting directly with the user through a standard terminal dialog box, or at interacting with other applications through files saved on disk, or remote interaction interfaces.

Given the need to enter many symbols stored in the terminal in the [string](https://www.mql5.com/en/docs/basis/types/stringconst) format, we can highlight the following possible options for implementing data entry into our script:

1. reading data from the \*.csv table file
2. reading data from the \*.bin binary file
3. reading data from the .sqlite database file
4. use third-party methods for interaction of the terminal with remote databases
5. using web api solutions
6. standard for the terminal use of a variable(s) of the appropriate type with a memory class modifier of the [input](https://www.mql5.com/en/docs/basis/variables/inputvariables) type

Before selecting a data entry method in our project, we will briefly consider the pros and cons of each option for implementing this task. When choosing the method described in point 1, reading data from a table file of the \*.csv type can be implemented through the standard terminal functions for handling files. In general, the code might look like this:

```
   string file_name = "Inputs.csv";                         // file name

   int handle=FileOpen(file_name,FILE_CSV|FILE_READ,";");   // attempt to find and open the file

   if(handle!=INVALID_HANDLE)                               // if the file is found, then
     {
      while(FileIsEnding(handle)==false)                    // start reading the file
        {
         string str_follow = FileReadString(handle);        // reading

        // here we implement filling the container depending on its type
        }
     }
```

Other options for implementing the functionality of working with files are described in sufficient detail in the terminal documentation. In this option, the user only needs to prepare an input parameter file using a third-party spreadsheet application, such as MS Excel or OpenOffice. Even the standard Windows Notepad will do.

In the second point, the [FileLoad()](https://www.mql5.com/en/docs/files/fileload) standard terminal function is suitable for using \*.bin files. To use it, you will need to know in advance what data structure the application used when saving this file in order to read data from a file with this extension. The implementation of such an idea might look like this.

```
   struct InputsData                   // take the structure, according to which the binary file was created
     {
      int                  symbol_id;  // id of a balanced symbol
      ENUM_POSITION_TYPE   type;       // position type
     };

   InputsData inputsData[];            // storage of inputs

   string  filename="Inputs.bin";      // file name

   ArrayFree(inputsData);              // array released

   long count=FileLoad(filename,inputsData,FILE_COMMON); // load file
```

The main disadvantage of this approach is that the [FileLoad()](https://www.mql5.com/en/docs/files/fileload) function does not work with data structures that contain object data types. Accordingly, it will not work with a structure if it contains the [string](https://www.mql5.com/en/docs/basis/types/stringconst) data type. In this case, you will have to additionally use custom container dictionaries so that to convert the id of the characters in the [int](https://www.mql5.com/en/docs/basis/types/integer/integertypes) integer value to the appropriate [string](https://www.mql5.com/en/docs/basis/types/stringconst) data type or make an additional request to the corresponding database. In general, this method will not be the most successful for our implementation due to the excessive complexity in performing fairly simple operations.

The third point specifically suggests using the terminal built-in functionality for working with a database of .sqlite files. This is a built-in terminal option for working with a relational database built on interaction with a file saved on the hard drive.

```
   string filename="Inputs.sqlite"; // file name with inputs prepared in advance

   int db=DatabaseOpen(filename, DATABASE_OPEN_READWRITE |
                       DATABASE_OPEN_CREATE | DATABASE_OPEN_COMMON); // open the database

   if(db!=INVALID_HANDLE)                                            // if opened
     {
      // implement queries to the database using the DatabaseExecute() function
      // the query structure will depend on the structure of the database tables
     }
```

In implementing this approach, it will be important to initially build the structure of the database tables. The structure of queries to obtain the necessary data will depend on this. The main advantage of this approach will be the possibility of relational data storage, in which the ids of tools will be stored in tables in an integer format rather than in a string format, which can provide a very significant gain in optimizing computer disk space. It is also worth noting that under certain conditions, this version of the database can be very productive. See more details in the article " [SQLite: Native handling of SQL databases in MQL5](https://www.mql5.com/en/articles/7463#transactions_speedup)".

The fourth paragraph describes the option of applying remote databases, which will require the use of additional third-party libraries. This option will be more labor-intensive than the method described in the previous paragraph, since it cannot be fully implemented through the standard terminal functionality. There are many publications on the subject, including a good option for implementing the interaction of the terminal with the MySQL database, described in the article " [How to access the MySQL database from MQL5 (MQL4)](https://www.mql5.com/en/articles/932)".

The use of web api requests to obtain inputs, described in the fifth paragraph, can probably be considered the most universal and cross-platform solution for our task. The functionality is built into the terminal via the [WebRequest()](https://www.mql5.com/en/docs/network/webrequest) predefined function and will be perfect if you already have an infrastructure for back-end and front-end applications due to its versatility. Otherwise, it may take quite a lot of time and resources to develop these applications from scratch, even though these solutions can be written in many modern programming languages and interpreters.

In the current implementation, we will use variables with the [input](https://www.mql5.com/en/docs/basis/variables/inputvariables) type memory class modifier of the [string](https://www.mql5.com/en/docs/basis/types/stringconst) data type from point six, simply because this option is able to provide all the necessary functionality without the need for additional development of custom programs. Naturally, we will not declare many variables for each symbol, since we cannot know in advance how many symbols we will balance, and there is a more elegant and flexible solution for this. We will work with one line containing all the values with further separation of the data in it. To do this, we declare a variable at the global level in the following form:

```
input string input_symbols = "EURCHFz USDJPYz";
```

Make sure to initialize it with the default value so that a user who is not a developer understands exactly how to enter symbols for the application to work correctly. In this case, we will use a regular space as a line separator for user convenience.

We will arrange obtaining data from this variable and filling our symbols\[\] array using a predefined terminal function for handling [StringSplit()](https://www.mql5.com/en/docs/strings/stringsplit) strings in the following form:

```
   string symbols[];                         // storage of user-entered symbols

   StringSplit(input_symbols,' ',symbols);   // split the string into the symbols we need

   int size = ArraySize(symbols);            // remember the size of the resulting array right away
```

When executing the [StringSplit()](https://www.mql5.com/en/docs/strings/stringsplit) function, the symbols\[\] array passed to it by reference is filled with data extracted from the string using the (' ') delimiter in the form of a space.

Now that we have a filled array with the values of symbol names for risk balancing, let’s move on to requesting the necessary data on the selected terminal symbols for further calculations.

### Obtaining the necessary symbol data through predefined terminal functions

For calculations, we will need to know the value of the minimum change in the price of each instrument and how much this change will cost us in the deposit currency. We can implement this via the predefined [SymbolInfoDouble()](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) terminal function with the necessary [ENUM\_SYMBOL\_INFO\_DOUBLE](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) enumeration as one of the function parameters. The implementation of data enumeration will be arranged via the popular 'for' loop as follows:

```
for(int i=0; i<size; i++)  // loop through previously entered symbols
     {
      point[i] = SymbolInfoDouble(symbols[i],SYMBOL_POINT);                	// requested the minimum price change size (tick)
      tick_val[i] = SymbolInfoDouble(symbols[i],SYMBOL_TRADE_TICK_VALUE_LOSS);  // request tick price in currency
     }
```

Note that the [ENUM\_SYMBOL\_INFO\_DOUBLE](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) enumeration contains not only the [SYMBOL\_TRADE\_TICK\_VALUE\_LOSS](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) value for requesting the tick price in currency but also the [SYMBOL\_TRADE\_TICK\_VALUE](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) value and the [SYMBOL\_TRADE\_TICK\_VALUE\_PROFIT](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) value equal to it. Actually, a request for any of the specified values can be used in our calculation since the difference in these values is not very significant. For example, the values of the specified arguments for the AUDNZD cross are presented in the following table:

| Function parameter | Tick price value returned by the function |
| --- | --- |
| SYMBOL\_TRADE\_TICK\_VALUE | 0.6062700000000001 |
| SYMBOL\_TRADE\_TICK\_VALUE\_LOSS | 0.6066200000000002 |
| SYMBOL\_TRADE\_TICK\_VALUE\_PROFIT | 0.6062700000000001 |

Table 1. Difference in tick price values returned for different parameters of the SymbolInfoDouble() function for AUDNZD

Despite the fact that it would be most correct to use the [SYMBOL\_TRADE\_TICK\_VALUE\_LOSS](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) parameter in our case, I believe that we can use any of the options proposed here. As [Richard Hamming](https://ru.wikipedia.org/wiki/%D0%A5%D1%8D%D0%BC%D0%BC%D0%B8%D0%BD%D0%B3,_%D0%A0%D0%B8%D1%87%D0%B0%D1%80%D0%B4_%D0%A3%D1%8D%D1%81%D0%BB%D0%B8 "https://ru.wikipedia.org/wiki/%D0%A5%D1%8D%D0%BC%D0%BC%D0%B8%D0%BD%D0%B3,_%D0%A0%D0%B8%D1%87%D0%B0%D1%80%D0%B4_%D0%A3%D1%8D%D1%81%D0%BB%D0%B8") noted back in 1962:

"The purpose of computing is insight, not numbers"

Let's move on to requesting the necessary data on volatility.

### Retrieving the required volatility data through the standard custom CiATR class

In previous chapters, we have already mentioned the concept of volatility as an indicator characterizing a typical price change for a symbol over a certain period of time, in our case during a trading day. Despite all the obviousness of this indicator, the methods for calculating it can differ greatly, primarily due to the following aspects:

- taking into account unclosed gaps on daily charts
- using averaging and a period it is applied to
- excluding bars with "abnormal" (exceptionally rare) volatility from calculation
- calculating based on high/low of the daily bar or on opens/close only
- or we generally trade Renko bars and the averaging time is of no importance to us at all

Taking into account gaps when calculating volatility is very often used in work on stock exchanges, in contrast to the foreign exchange market, simply because market gaps that are not closed during the day are very rare in the foreign exchange market and the final calculation of average daily volatility does not change. The calculation here will differ only in that we take the maximum value from the two values. The first is the difference between the high and low of each bar, and the second is the difference between the high and low of the current bar and the close of the previous one. We can use the built-in [MathMax()](https://www.mql5.com/en/docs/math/mathmax) function to achieve that.

When applying averaging of the obtained values, it is necessary to take into account that the longer the averaging period, the slower this indicator becomes for changes in market volatility. As a rule, a period of 3 to 5 days is used to average daily volatility in the foreign exchange market. Also, it is advisable to exclude abnormal movements in the market when calculating the volatility indicator. This can be done automatically using the median value of the sample. To do this, we can use the built-in Median() method called for an instance of the [vector](https://www.mql5.com/en/docs/basis/types/matrix_vector) type class.

Many traders prefer to do averaging by taking into account volatility only through opening and closing prices without taking into account the wicks of candles. This method is not recommended, as it can produce a value much lower than the actual market volatility. When using Renko bars, the logic changes very much and here the aggregation of volatility does not come from the trading period, but the trading period is determined by volatility. Therefore, this approach will not suit us in our article.

In our implementation, we will use the volatility query through the ATR terminal indicator using the standard custom CiATR class stored in the terminal library in the open source format. We will use the fast value 3 to average the indicator. In general, the volatility request code will look like as follows. At the global level, declare the name of a class variable with a call of the default constructor.

```
CiATR indAtr[];
```

Here we use the array of indicators to store data simultaneously, and not to overload one existing indicator, mainly for convenience and the possibility of further expanding the functionality of the code. Next, we add the following code to our symbol iteration loop, where we are already requesting data on symbols to request indicator values.

```
indAtr[i].Create(symbols[i],PERIOD_D1, atr_period);   // create symbol and period

indAtr[i].Refresh();          // be sure to update data

atr[i] = indAtr[i].Main(1);   // request data on closed bars on D1
```

Now that all the necessary initial data for the calculation have been obtained, proceed directly to the calculations.

### Two options for the logic of risk calculation for different methods of exiting a position

When writing the logic for balancing risks when trading several instruments simultaneously, it remains to take into account the following important points. How exits from balanced positions are provided in our trading system and how we take into account the correlation of the symbols that we will balance. Even at first glance, a clear criterion for the correlation of crosses in foreign exchange markets does not always provide a guaranteed level of correlation in the period of time under consideration and can often be considered as separate symbols. For example, if at the beginning of the trading day we decided on the set of symbols we will trade today and the direction of entries, we should know the following in advance. Will we close all positions at the end of the trading day or not? Will we consider exiting positions during the day separately for each position, or will we simply close all positions over time? There cannot be some kind of universal recipe for everyone, simply because each trader makes decisions about entry and exit from a fairly large set of criteria based on their own knowledge and experience.

In this implementation, we will make a universal calculation that will allow us to flexibly determine the volume of entry into a position based on the fact that you are trading simply by changing the input risk parameters for the instrument and including/excluding instruments that, in the trader’s opinion, are correlated at the moment. To do this, we declare the input risk parameter for one instrument as follows.

```
input double risk_per_instr = 50;
```

Also, for universality of use, we will provide simultaneous output of balancing results when regressing the risk entered by the user onto the trading symbol taken separately and taking into account the correlation of these symbols. This will enable the trader to obtain a range of varying position volumes and, most importantly, the proportions of these instruments for simultaneous trading. To do this, we first need to add the following entry to our main calculation cycle.

```
risk_contract[i] = tick_val[i]*atr[i]/point[i]; // calculate the risk for a standard contract taking into account volatility and point price
```

Next, outside the specified loop, we find the instrument in our set with the maximum risk value in order to build a proportion from it to balance trading volumes throughout the set. The built-in functionality of our container is meant exactly for this.

```
double max_risk = risk_contract.Max();          // call the built-in container method to find the maximum value
```

Now that we know the maximum risk in our set, we arrange another loop to calculate the balanced volume for each instrument in our sample, provided there is no correlation between them, and immediately display the results in the journal.

```
for(int i=0; i<size; i++)	// loop through the size of our symbol array
     {
      volume[i] = NormalizeDouble((max_risk / risk_contract[i]) * (risk_per_instr / max_risk),calc_digits); // calculate the balanced volume
     }

Print("Separate");		// display the header in the journal preliminarily

for(int i=0; i<size; i++)	// loop through the array again
     {
      Print(symbols[i]+"\t"+DoubleToString(volume[i],calc_digits));	// display the resulting volume values
     }
```

This turned out to be the maximum volume for intraday trading on our positions, balanced by the average daily most probable and expected volatility, taking into account the risk we assigned to the symbol.

Next, calculate the volume of our positions based on the premise that the symbols may begin to correlate on the trading day, which may in fact significantly increase the expected risk on the symbol relative to what we entered in the input parameters. To do this, add the following code where we simply divide the resulting value by the number of instruments being traded.

```
Print("Complex");		// display the header in the journal preliminarily

for(int i=0; i<size; i++)	// loop through the array
     {
      Print(symbols[i]+"\t"+DoubleToString(volume[i]/size,calc_digits));	// calculate the minimum volume for entry
     }
```

Now we have obtained the maximum and minimum limits for the volumes of risk-balanced positions. The trader is able to independently determine the volume of entries within a given range based on them. The main thing is to adhere to the proportions indicated in the calculation. It should also be noted that logging results is the simplest method, but not the only one. You can also use other standard terminal functions to display information to the user. In this case, the terminal provides very wide functionality, including the usage of such functions as [MessageBox()](https://www.mql5.com/en/docs/common/messagebox), [Alert()](https://www.mql5.com/en/docs/common/alert), [SendNotification()](https://www.mql5.com/en/docs/network/sendnotification), [SendMail()](https://www.mql5.com/en/docs/network/sendmail) and many others. We move on to the full code of the EA in the compiled file.

### Final implementation of the solution in the script

As a result, our implementation code will look like this.

```
#property strict

#include <Indicators\Oscilators.mqh>

//---
input string input_symbols = "EURCHFz USDJPYz";
input double risk_per_instr = 50;
input int atr_period = 3;
input int calc_digits = 3;
CiATR indAtr[];

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   string symbols[];
   StringSplit(input_symbols,' ',symbols);

   int size = ArraySize(symbols);
   double tick_val[], atr[], volume[], point[];
   vector<double> risk_contract;

   ArrayResize(tick_val,size);
   ArrayResize(atr,size);
   ArrayResize(volume,size);
   ArrayResize(point,size);
   ArrayResize(indAtr,size);
   risk_contract.Resize(size);

   for(int i=0; i<size; i++)
     {
      indAtr[i].Create(symbols[i],PERIOD_D1, atr_period);
      indAtr[i].Refresh();

      point[i] = SymbolInfoDouble(symbols[i],SYMBOL_POINT);
      tick_val[i] = SymbolInfoDouble(symbols[i],SYMBOL_TRADE_TICK_VALUE);

      atr[i] = indAtr[i].Main(1);
      risk_contract[i] = tick_val[i]*atr[i]/point[i];
     }

   double max_risk = risk_contract.Max();
   Print("Max risk in set\t"+symbols[risk_contract.ArgMax()]+"\t"+DoubleToString(max_risk));

   for(int i=0; i<size; i++)
     {
      volume[i] = NormalizeDouble((max_risk / risk_contract[i]) * (risk_per_instr / max_risk),calc_digits);
     }

   Print("Separate");
   for(int i=0; i<size; i++)
     {
      Print(symbols[i]+"\t"+DoubleToString(volume[i],calc_digits));
     }

   Print("Complex");
   for(int i=0; i<size; i++)
     {
      Print(symbols[i]+"\t"+DoubleToString(volume[i]/size,calc_digits));
     }
  }
//+------------------------------------------------------------------+
```

After compilation, the inputs window appears. For example, we want to balance three symbols and the maximum risk is USD 500.

![Figure 2. Trader inputs](https://c.mql5.com/2/80/input.PNG)

Figure 2. Trader inputs

As a result of running the script, the following data will be obtained for the volume of each symbol, taking into account the risk balance.

![Figure 3. Output data](https://c.mql5.com/2/80/output.PNG)

Figure 3. Output data

Here is a full-featured code for calculating simultaneous trading volumes of several risk-balanced symbols using the simplest, but minimally necessary functionality provided by the terminal. If you wish, we can scale this algorithm using the additional features indicated in the article, as well as our own developments.

### Conclusion

We received a fully functional script that will allow traders who do not trade through algorithmic trading to quickly and accurately adjust position volumes when trading intraday. I hope that even for those who trade algorithmically, there will be new ideas here to improve their software infrastructure and possibly improve their current trading results. Thanks for reading and feedback!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14163](https://www.mql5.com/ru/articles/14163)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14163.zip "Download all attachments in the single ZIP archive")

[RiskBallance.mq5](https://www.mql5.com/en/articles/download/14163/riskballance.mq5 "Download RiskBallance.mq5")(2.27 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Visualizing deals on a chart (Part 2): Data graphical display](https://www.mql5.com/en/articles/14961)
- [Visualizing deals on a chart (Part 1): Selecting a period for analysis](https://www.mql5.com/en/articles/14903)
- [Risk manager for algorithmic trading](https://www.mql5.com/en/articles/14634)
- [Risk manager for manual trading](https://www.mql5.com/en/articles/14340)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/468457)**
(8)


![Aleksandr Seredin](https://c.mql5.com/avatar/2022/4/62543FFE-A51A.jpg)

**[Aleksandr Seredin](https://www.mql5.com/en/users/al.s.capital)**
\|
9 Feb 2024 at 18:59

**Maxim Kuznetsov [#](https://www.mql5.com/ru/forum/462210#comment_52238595):**

nothing at all...

not the risks, not the multisymbol.

shame

Thank you for your comment

![Anatoliy Migachyov](https://c.mql5.com/avatar/2023/7/64c79c6d-c9e7.jpg)

**[Anatoliy Migachyov](https://www.mql5.com/en/users/339979)**
\|
14 Feb 2024 at 12:27

Many people don't want to hear about risk, and here is the balance, and in general it is a sore subject for many traders

![Aleksandr Seredin](https://c.mql5.com/avatar/2022/4/62543FFE-A51A.jpg)

**[Aleksandr Seredin](https://www.mql5.com/en/users/al.s.capital)**
\|
14 Feb 2024 at 17:10

**Anatoliy Migachyov [#](https://www.mql5.com/ru/forum/462210#comment_52283648):**

Many people don't want to hear about risk, and here is the balance, and in general it is a sore subject for many traders

I completely agree. In one sentence, but how many questions are touched upon:

\- They don't want to hear, hoping that if "if you don't call the bad guy, it will be quiet". But it doesn't work like that on the market, I think there is no place for pralogical mystical thinking.

\- As a rule, one starts to think about risks not earlier than after the first drained deposit :)

\- and even after the first drained deposit, everyone starts to "piss off" the word "balance", simply because it becomes clear that the profitability starts to fall from the balance.

\- hence the "painful topic" for those who could not understand the risks and could not build a stable trading system, and then these people start to write comments like "about nothing" and "shame" ))))

And respect to the author of this comment for such a succinct and informative statement. Thank you

![nowenn](https://c.mql5.com/avatar/avatar_na2.png)

**[nowenn](https://www.mql5.com/en/users/nowenn)**
\|
12 Jun 2024 at 02:40

Thanks for the code example, the main issue is depending on what timeframe your trading how can we define normal and abnormal risk. Right?


![Aleksandr Seredin](https://c.mql5.com/avatar/2022/4/62543FFE-A51A.jpg)

**[Aleksandr Seredin](https://www.mql5.com/en/users/al.s.capital)**
\|
12 Jun 2024 at 07:19

**nowenn [#](https://www.mql5.com/en/forum/468457#comment_53657488):**

Thanks for the code example, the main issue is depending on what timeframe your trading how can we define normal and abnormal risk. Right?

All right. It is usually the daily timeframe that is used to assess risk.

![Gain An Edge Over Any Market (Part II): Forecasting Technical Indicators](https://c.mql5.com/2/80/Gain_An_Edge_Over_Any_Market_Part_II___LOGO.png)[Gain An Edge Over Any Market (Part II): Forecasting Technical Indicators](https://www.mql5.com/en/articles/14936)

Did you know that we can gain more accuracy forecasting certain technical indicators than predicting the underlying price of a traded symbol? Join us to explore how to leverage this insight for better trading strategies.

![Using optimization algorithms to configure EA parameters on the fly](https://c.mql5.com/2/70/Using_optimization_algorithms_to_configure_EA_parameters_on_the_fly____LOGO.png)[Using optimization algorithms to configure EA parameters on the fly](https://www.mql5.com/en/articles/14183)

The article discusses the practical aspects of using optimization algorithms to find the best EA parameters on the fly, as well as virtualization of trading operations and EA logic. The article can be used as an instruction for implementing optimization algorithms into an EA.

![Integrating Hidden Markov Models in MetaTrader 5](https://c.mql5.com/2/80/Integrating_Hidden_Markov_Models_in_MetaTrader_5_____LOGO.png)[Integrating Hidden Markov Models in MetaTrader 5](https://www.mql5.com/en/articles/15033)

In this article we demonstrate how Hidden Markov Models trained using Python can be integrated into MetaTrader 5 applications. Hidden Markov Models are a powerful statistical tool used for modeling time series data, where the system being modeled is characterized by unobservable (hidden) states. A fundamental premise of HMMs is that the probability of being in a given state at a particular time depends on the process's state at the previous time slot.

![MQL5 Wizard Techniques you should know (Part 22): Conditional GANs](https://c.mql5.com/2/80/MQL5_Wizard_Techniques_you_should_know_Part_22____LOGO.png)[MQL5 Wizard Techniques you should know (Part 22): Conditional GANs](https://www.mql5.com/en/articles/15029)

Generative Adversarial Networks are a pairing of Neural Networks that train off of each other for more accurate results. We adopt the conditional type of these networks as we look to possible application in forecasting Financial time series within an Expert Signal Class.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/14163&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082889986082083046)

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