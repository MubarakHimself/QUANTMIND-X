---
title: Visualizing deals on a chart (Part 1): Selecting a period for analysis
url: https://www.mql5.com/en/articles/14903
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:48:18.279658
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/14903&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083122992352859504)

MetaTrader 5 / Trading systems


### Introduction

In this article, we will develop a script from scratch to visualize deals during retrospective analysis of trading decisions. Simply put, if we trade manually and analyze our past market entries in order to improve our trading performance, we want to spend as little time as possible on manual history analysis and the purely technical work associated with it: opening charts, searching for deals in history, manually saving print screens in the terminal, independently drawing Stop Loss and Take Profit for completed deals, as well as collecting information on paid commissions and swaps. The script we are going to develop here will help to significantly reduce all mechanical work. Using the script (having collected it from the attachments below), you will be able to significantly reduce the time spent on technical work in order to pay more attention to analyzing trading decisions. Those who do not want to spend time assembling the projects can download a ready-made version of the script [on the MQL5 Market](https://www.mql5.com/en/market/product/86223). The script development will be divided in two parts with the source code attached to each part in full.

### Why do we need retrospective analysis?

The main goal of any person entering the market is to make a long-term profit with controlled risk. This type of business can be compared to others where you also have to deal with risks. However, as practice shows, most new businesses in the real economy eventually go bankrupt. In financial markets, the funds can be lost much faster due to the use of leverage and the full regression of losses on all available funds. In real business, if something goes wrong, you can redirect your production facilities for other purposes (if they are not mortgaged) and start over. In trading, your entire account bears all the losses.

Multiple statistics confirm the thesis that investing in financial markets without risk control and with unreasonable use of leverage can make such an investment the most dangerous in terms of capital investment risks. According to [the research](https://www.mql5.com/go?link=https://www.sec.gov/ "https://www.sec.gov/comments/s7-30-11/s73011-10.pdf") conducted by the U.S. Securities and Exchange Commission (Release No. 34-64874, File Number: S7-30-11) within 17 years:

" Approximately 70% of customers lose money every quarter and on average 100% of a retail customer's investment is lost in less than 12 months"

In the previous series of articles on the importance of [risk balancing](https://www.mql5.com/en/articles/14163) and [risk management](https://www.mql5.com/en/articles/14340), I have already noted that uncontrolled risk always leads to loss of money and even an initially profitable strategy can be turned into an unprofitable one without risk management. In this series of articles, we will consider the aspect that the market is a very flexible entity that changes over time under the influence of various economic and political factors. For a clearer understanding, we can simplify this thesis by saying that the market can be in at least two stages - a flat movement and a trend. Therefore, it is very important to constantly analyze your trading activities to determine whether the chosen strategy is adequate to the market situation.

Many algorithmic traders find that certain strategies work well in a flat market, but start to lose money when the market moves. Similarly, strategies adapted to trends lose efficiency in a flat market. Creating algorithms that can recognize market phase changes, before losses start to eat into profits, requires significant computing resources and time.

These problems force traders to constantly ask themselves "Am I doing everything right?", "Is today's drawdown normal or do I need to adjust the algorithms?", "How can I improve my results?" Answers to these questions are important for long-term success in the market. The methods for finding them vary: some use strategy optimizers, some apply deep neural networks, some rely on mathematical models or many years of experience. All these approaches can be effective, since the market itself is the main teacher.

Monitoring the market often becomes a source of innovative strategies and investment ideas. In this article, we will create a script that will not only help improve the efficiency of trades, but also offer new ideas for algorithms based on data analysis. Analysis of historical deals is a must for me, despite the fact that I have been trading exclusively using EAs for a long time.

Let's move on to writing a script that will help significantly simplify the routine process of analyzing deals. The script should eventually output information to us on a print screen, as shown in Figure 1.

![Figure 1. Displaying script data](https://c.mql5.com/2/79/1_exa.png)

Figure 1. Displaying script data

We will start implementing the script by entering the user's input data.

### Script inputs

The script functionality should include the ability to automatically upload data on historical transactions to a chart print screen with the ability to set several timeframes for each deal in a separate file, as well as provide the user with the option to access data on one individual deal, or on all deals for a period specified by the user. I will also try to provide the maximum opportunity to customize the graphics on the resulting print screens.

To enable the user to switch the script to unload data for one deal or for a period of historical data, first of all we need to provide a custom data type of the [enum](https://www.mql5.com/en/docs/basis/types/integer/enumeration) enumerated type in the following form:

```
enum input_method
  {
   Select_one_deal,
   Select_period
  };
```

The implementation of our custom enumeration will consist of only two previously announced options: choosing one deal and choosing a period. Now we can move on to declaring the input at the global level of the [input](https://www.mql5.com/en/docs/basis/variables/inputvariables) memory class:

```
input group "Enter deal ticket or select the period"
input input_method inp_method = Select_period;                          // One deal or period?
```

For the user convenience, here we have provided a named block using the [group](https://www.mql5.com/en/docs/basis/variables/inputvariables#group) keyword to visually separate each important parameter for the user and provide the necessary explanations. Also, comment out the [input](https://www.mql5.com/en/docs/basis/variables/inputvariables) memory class variables so that the variable names are replaced with text comments that are understandable to the average user.

In the graphical interface, entering the variable values will look like the one shown in Figure 2.

![Figure 2. User interface for entering data unloading conditions](https://c.mql5.com/2/98/2Sel.jpg)

Figure 2. User interface for entering data loading conditions

In order for the user to quickly obtain information on a certain deal, they need to set the corresponding enumeration to Select\_one\_deal in the inp\_method variable. After this, the user will need to indicate the number (ticket) of the necessary deal. Declare the input block in the following form:

```
input group "For case 'Select_one_deal': Enter ticket of deal"
input long inp_d_ticket = 4339491;                                      // Ticket (global id)
```

Let's set a default value for the variable to serve as an example for the user, and just in case, indicate that this is an order ticket by global number. As a rule, the terminal displays this number in the deal history, so the user should not have any difficulties with this issue.

But if the user wants to select a period for analysis so that all deals are downloaded, it is necessary to provide inputs corresponding to the value of the beginning and end of the period of selecting deals in history. This can be implemented through the following entry at the global level:

```
input group "For case 'Select_period': Enter period for analys"
input datetime start_date = D'17.07.2023';                              // Start date
input datetime finish_date = D'19.07.2023';                             // Finish date
```

The user will enter the start date of the sample into the start\_date variable, and the end date of the sample period into the finish\_date variable. Let's also initialize these variables with default values for the user convenience.

Now that we have some clarity regarding the deals to be analyzed, it is time to implement a mechanism that allows the user to save data on a single deal on multiple charts. This will be very convenient if, for example, during manual trading, the trader uses a daily chart to determine trading levels and looks for an entry point on M30 charts. It will be much more convenient to analyze historical deals if our script immediately downloads a chart (both M30 and D1) with all the data.

We will also implement this idea through inputs and provide not two, but four charts for deals to expand user's capabilities, since some traders use more than two charts in their trading, but very few use more than four. In exceptional cases, it will be possible to run the script multiple times. To do this, declare four variables of the [ENUM\_TIMEFRAMES](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes) enumerated standard type, where the main\_graph variable denotes the main unloading timeframe, while the remaining are auxiliary ones. Let's write the code in the following form:

```
input group "Enter time frames. 'current' = don't use"
input ENUM_TIMEFRAMES main_graph = PERIOD_D1;                           // Period of main chart
input ENUM_TIMEFRAMES addition_graph = PERIOD_H1;                       // Period of addition chart
input ENUM_TIMEFRAMES addition_graph_2 = PERIOD_CURRENT;                // Period of addition chart #2
input ENUM_TIMEFRAMES addition_graph_3 = PERIOD_CURRENT;                // Period of addition chart #3
```

The periods selected by the user can vary greatly in time, in fact, this is what they were provided for in the settings above. This is why we also need a chart shift on the right so that the images are neatly positioned on the chart as convenient for the user. This means that if we make a print screen of the daily chart, a few bars after the deal may be enough for us to see where the price ended up going after the deal. For smaller timeframes, we will need a corresponding setting with a much higher value. Therefore, we will provide these shifts on the charts with the default values as in the code below.

```
input group "Navigate settings in bars."
input int bars_from_right_main = 15;                                    // Shift from right on main chart
input int bars_from_right_add = 35;                                     // Shift from right on addition chart
input int bars_from_right_add_2 = 35;                                   // Shift from right on addition chart #2
input int bars_from_right_add_3 = 35;                                   // Shift from right on addition chart #3
```

In order to customize the chart to save print screens in the MetaTrader 5 terminal, we can apply [the template usage functionality](https://www.metatrader5.com/en/terminal/help/charts_advanced/templates_profiles "https://www.metatrader5.com/en/terminal/help/charts_advanced/templates_profiles") or configure all chart properties by implementing the [chart operation](https://www.mql5.com/en/docs/chart_operations) code and allocate each value into a separate script input. I believe, in case of my solution, it will be more elegant and correct not to "inflate" the section with the script inputs when using the chart settings, but to use ready-made templates for working with the presentation of the chart in the terminal. As a result, we will simply write the name of a pre-prepared template in the input for each timeframe, and the script will work with it. You can also use previously created templates to make the display more familiar.

The presentation of charts in the MetaTrader 5 terminal can be customized to suit almost any taste and preference. By pressing F8 on the chart, we can customize display modes, objects, color palette and much more. Any chart setting can be quickly and conveniently changed by creating templates for various display configurations. The context menu item Charts -> Templates -> Save Template/Load Template allows us to quickly change the price chart display setting even without changing the active chart window. As a result, multiple chart display settings fit into several variables according to the number of analyzed timeframes.

```
input group "Properties of charts. Enter the template name:"
input string main_template = "dailyHistorytemp";                        // Template name of main chart
input string addition_template = "hourHistoryTemp";                     // Template name of addition chart
input string addition_template_2 = "hourHistoryTemp";                   // Template name of addition chart #2
input string addition_template_3 = "hourHistoryTemp";                   // Template name of addition chart #3
```

In the user interface of the terminal inputs, it will look like the one displayed in Figure 3:

![Figure 3. User interface of template inputs](https://c.mql5.com/2/98/3Temp.jpg)

Figure 3. User interface of template inputs

Now that we have decided on all the standard settings, let's add the settings related specifically to displaying full information on deals in order to fully provide the user with information for analyzing their trading operations. These are mainly objects corresponding to the position open price, Stop Loss, Take Profit and the connecting line. The corresponding [color](https://www.mql5.com/en/docs/basis/types/integer/color) type variables will look like this:

```
input group "Colors of deals line"
input color clr_price_open = clrWhiteSmoke;                             // Color of price open label
input color clr_price_close = clrWhiteSmoke;                            // Color of price close label
input color clr_stop = clrRed;                                          // Color of stop loss label
input color clr_take = clrLawnGreen;                                    // Color of take profit label
input color clr_main = clrWhiteSmoke;                                   // Color of deals trendline
```

In general, all inputs of our script will look as described below:

```
#property copyright "Visit product page"
#property link      "https://www.mql5.com/ru/market/product/86223"
#property version   "1.00"
#property description "Make an automatic printscreen with a full description of all transactions for the period or
			specify the ticket of the desired transaction."
#property script_show_inputs

enum input_method
  {
   Select_one_deal,
   Select_period
  };

input group "Enter deal ticket or select the period"
input input_method inp_method = Select_period;                          // One deal or period?

input group "For case 'Select_one_deal': Enter ticket of deal"
input long inp_d_ticket = 4339491;                                      // Ticket (global id)

input group "For case 'Select_period': Enter period for analys"
input datetime start_date = D'17.07.2023';                              // Start date
input datetime finish_date = D'19.07.2023';                             // Finish date

input group "Enter time frames. 'current' = don't use"
input ENUM_TIMEFRAMES main_graph = PERIOD_D1;                           // Period of main chart
input ENUM_TIMEFRAMES addition_graph = PERIOD_H1;                       // Period of addition chart
input ENUM_TIMEFRAMES addition_graph_2 = PERIOD_CURRENT;                // Period of addition chart #2
input ENUM_TIMEFRAMES addition_graph_3 = PERIOD_CURRENT;                // Period of addition chart #3

input group "Navigate settings in bars."
input int bars_from_right_main = 15;                                    // Shift from right on main chart
input int bars_from_right_add = 35;                                     // Shift from right on addition chart
input int bars_from_right_add_2 = 35;                                   // Shift from right on addition chart #2
input int bars_from_right_add_3 = 35;                                   // Shift from right on addition chart #3

input group "Properties of charts. Enter the template name:"
input string main_template = "dailyHistorytemp";                        // Template name of main chart
input string addition_template = "hourHistoryTemp";                     // Template name of addition chart
input string addition_template_2 = "hourHistoryTemp";                   // Template name of addition chart #2
input string addition_template_3 = "hourHistoryTemp";                   // Template name of addition chart #3

input group "Colors of deals line"
input color clr_price_open = clrWhiteSmoke;                             // Color of price open label
input color clr_price_close = clrWhiteSmoke;                            // Color of price close label
input color clr_stop = clrRed;                                          // Color of stop loss label
input color clr_take = clrLawnGreen;                                    // Color of take profit label
input color clr_main = clrWhiteSmoke;                                   // Color of deals trendline
```

We have defined all the variables at the global level and can now proceed to implementing the code at the script entry point in [OnStart()](https://www.mql5.com/en/docs/event_handlers/onstart). Let's start by defining all the necessary variables for storing, handling and displaying the data to be sent to the saved print screen file. We will notify the user of each step when handling the script.

Let's start by informing the user that the script has started, reset the error variable so that we can correctly check the error return code if something goes wrong, and provide variables for all position properties, as well as provide appropriate storage for collecting information on all deals.

```
   Print("Script starts its work.");                                    // notified

   ResetLastError();                                                    // reset error

   string brok_name = TerminalInfoString(TERMINAL_COMPANY);             // get broker name
   long account_num = AccountInfoInteger(ACCOUNT_LOGIN);                // get account number

//---
   ulong    ticket = 0;                                                 // ticket
   ENUM_DEAL_ENTRY entry = -1;                                          // entry or exit
   long     position_id = 0,  PositionID[];                             // main id
   int      type = -1,        arr_type[];                               // deal type
   int      magic = -1,       arr_magic[];                              // magic number
   ENUM_DEAL_REASON      reason = -1,      arr_reason[];                // reason

   datetime time_open = 0,    arr_time_open[];                          // deal open time
   datetime time_close = 0,   arr_time_close[];                         // close time

   string   symbol,           arr_symbol[];                             // symbol
   string   comment,          arr_comment[];                            // comment
   string   externalID,       arr_extermalID[];                         // external id

   double   stop_loss = 0,    arr_stop_loss[];                          // deal Stop Loss
   double   take_profit = 0,  arr_take_profit[];                        // deal Take Profit
   double   open = 0,         arr_open[];                               // open price
   double   close = 0,        arr_close[];                              // close price
   double   volume = 0,       arr_volume[];                             // position volume
   double   commission = 0,   arr_commission[];                         // commission
   double   swap = 0,         arr_swap[];                               // swap
   double   profit = 0,       arr_profit[];                             // profit
   double   fee = 0,          arr_fee[];                                // fee

   int res = -1;                                                        // user command
```

Now we can implement collecting deal data depending on whether the user wants to receive a print screen for one deal or for all deals within a certain period.

### Selecting historical data by period

Let's implement the user selection through the 'switch' logical choice operator, which gets the value of the entered inp\_method global variable and start handling from the Select\_period case variant for collecting data on completed deals within a certain period.

First of all, inform the user that the option of analyzing deals within a period has been selected in the inputs. Informing is implemented using the [MessageBox()](https://www.mql5.com/en/docs/common/messagebox) predefined function for calling the message window. The third parameter is the [MB\_OKCANCEL](https://www.mql5.com/en/docs/constants/io_constants/messbconstants#messageboxflags) constant allowing the terminal to interrupt the script execution after clicking "cancel". This is convenient since the user can prematurely terminate the script and not wait for its execution if they accidentally entered the wrong option in the inp\_method input. The full code is presented below.

```
         res = MessageBox("You have selected analysis for period. Continue?","",MB_OKCANCEL); // wait for user confirmation
```

We will place the result of handling the button pressing event in the 'res' variable in order to implement the mechanism for interrupting the script. Technically, the easiest way to interrupt is through the 'return' statement. [return](https://www.mql5.com/en/docs/basis/operators/return), if the variable 'res' contains the [IDCANCEL](https://www.mql5.com/en/docs/constants/io_constants/messbconstants#messageboxflags) value, which means that the user pressed the corresponding button. The block is represented through the [if](https://www.mql5.com/en/docs/basis/operators/if) conditional logical choice operator in the following form.

```
         if(res == IDCANCEL)                                            // if interrupted by user
           {
            printf("%s - %d -> Scrypt was stoped by user.",__FUNCTION__,__LINE__); // notify
            return;                                                     // do not continue
           }
```

If at this stage the user has confirmed the option choice validity, start collecting information on completed deals for the specified historical period. We will perform the selection of historical deals using the [HistorySelect()](https://www.mql5.com/en/docs/trading/historyselect) predefined function. It is this function that will receive the values of the period start and end values, entered and confirmed by the user above.

After requesting historical deals, it would be very appropriate to arrange a check for the presence of deals on the account within the period specified by the user for code optimization and user convenience. Place the number of obtained historical deals to the 'total' variable via the predefined [HistoryDealsTotal()](https://www.mql5.com/en/docs/trading/historydealstotal) terminal function:

```
            int total = HistoryDealsTotal();                            // got the total number of deals
```

If there is nothing to analyze within the specified period and not a single deal was found, then notify the user of that and stop the script. The event is also handled by the [if](https://www.mql5.com/en/docs/basis/operators/if) conditional logical choice operator, in which we inform the user about the absence of deals within the specified period via the EA log and the information window. Interrupt the script using the [return](https://www.mql5.com/en/docs/basis/operators/return) operator as shown below:

```
            if(total <= 0)                                              // if nothing found
              {
               printf("%s - %d -> No deals were found for the specified period.",__FUNCTION__,__LINE__); // notify
               MessageBox("No deals were found for the specified period: "+TimeToString(start_date)+"-"+TimeToString(finish_date)+". Script is done.");
               return;
              }
```

If deals have been found within the period, we can start collecting data. Iterate through all deals obtained in history using the [for](https://www.mql5.com/en/docs/basis/operators/for) loop as shown below:

```
            for(int i=0; i<total; i++)                                  // iterate through the number of deals
```

Select and request data for each individual historical deal using its unique ID - ticket. use the predefined HistoryDealGetTicket() terminal function to get it. Its parameters will receive serial numbers from 0 to 'total' and get the return value as a unique deal ID as shown below. Do not forget to check the validity of the returned value.

```
               //--- try to get deals ticket
               if((ticket=HistoryDealGetTicket(i))>0)                   // took the ticket
```

After receiving the historical deal ticket, request three main features required to collect general position data from the deals. These features include: ID of a position a historical deal belongs to, the [ENUM\_DEAL\_ENTRY](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_entry) feature, which informs us what exactly the deal initialted: opening or closing a position and a deal type with the [DEAL\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_type) feature defining the order direction and type. All three requests are executed through the [HistoryDealGetInteger()](https://www.mql5.com/en/docs/trading/historydealgetinteger) predefined terminal function as shown below:

```
                  //--- get deals properties
                  position_id = HistoryDealGetInteger(ticket,DEAL_POSITION_ID);        // took the main id
                  entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(ticket,DEAL_ENTRY);   // entry or exit?
                  type = (int)HistoryDealGetInteger(ticket,DEAL_TYPE);  	       // deal type
```

The primary request for this data occurs due to the fact that their values will determine the values that we will use to obtain data for collecting data on the entire position. Let us recall that data on the general position is aggregated from a set of data on several orders related specifically to this position and compared with each other based on the position ID.

First, we need to sort out all historical deals that are not directly related to trading operations, such as: balance replenishment, withdrawal of funds, accrual of bonuses from the broker, etc. To do this, arrange the following check in the code:

```
                  //--- check the deal type
                  if(type == DEAL_TYPE_BUY            ||                // if it is buy
                     type == DEAL_TYPE_SELL           ||                // if it is sell
                     type == DEAL_TYPE_BUY_CANCELED   ||                // if canceled buy
                     type == DEAL_TYPE_SELL_CANCELED                    // if canceled sell
                    )
```

Once we have verified that the deal we have received is a trading operation, i.e. a buy or sell, or the corresponding closes, which may vary depending on the type of a broker account or a market, we can begin collecting data on the position as a whole. Considering the peculiarities of storing data on positions in the corresponding orders, we will take some position features from orders related to opening, while some features are taken from orders related to closing positions. For example, data on the financial result of a position can be expectedly and logically found in the position close orders. To present this data more visually and clearly, collect it in the following table:

| # | Entering a position (DEAL\_ENTRY\_IN) | Exiting a position (DEAL\_ENTRY\_OUT) |
| --- | --- | --- |
| 1 | open (DEAL\_PRICE) | close (DEAL\_PRICE) |
| 2 | time\_open (DEAL\_TIME) | time\_close (DEAL\_TIME) |
| 3 | symbol (DEAL\_SYMBOL) | reason (DEAL\_REASON) |
| 4 | stop\_loss (DEAL\_SL) | swap (DEAL\_SWAP) |
| 5 | take\_profit (DEAL\_TP) | profit (DEAL\_PROFIT) |
| 6 | magic (DEAL\_MAGIC) | fee (DEAL\_FEE) |
| 7 | comment (DEAL\_COMMENT) | - |
| 8 | externalID (DEAL\_EXTERNAL\_ID) | - |
| 9 | volume (DEAL\_VOLUME) | - |
| 10 | commission (DEAL\_COMMISSION) | - |

Table 1. Sources of obtaining data on the entire position depending on the deal type

In the code, the request and sorting of data shown in Table 1 would look like this:

```
                     if(entry == DEAL_ENTRY_IN)                         		// if this is an entry
                       {
                        open = HistoryDealGetDouble(ticket,DEAL_PRICE);                 // take open price
                        time_open  =(datetime)HistoryDealGetInteger(ticket,DEAL_TIME);  // take open time
                        symbol=HistoryDealGetString(ticket,DEAL_SYMBOL);   		// take symbol
                        stop_loss = HistoryDealGetDouble(ticket,DEAL_SL);  		// take Stop Loss
                        take_profit = HistoryDealGetDouble(ticket,DEAL_TP);		// take Take Profit

                        magic = (int)HistoryDealGetInteger(ticket,DEAL_MAGIC);   	// take magic number
                        comment=HistoryDealGetString(ticket,DEAL_COMMENT);       	// take comment
                        externalID=HistoryDealGetString(ticket,DEAL_EXTERNAL_ID);	// take external id
                        volume = HistoryDealGetDouble(ticket,DEAL_VOLUME);          	// take volume
                        commission = HistoryDealGetDouble(ticket,DEAL_COMMISSION);  	// take commission value
                       }

                     if(entry == DEAL_ENTRY_OUT)                        	 	// if this is an exit
                       {
                        close = HistoryDealGetDouble(ticket,DEAL_PRICE);               	// take close price
                        time_close  =(datetime)HistoryDealGetInteger(ticket,DEAL_TIME);	// take close time

                        reason = (ENUM_DEAL_REASON)HistoryDealGetInteger(ticket,DEAL_REASON); //
                        swap = HistoryDealGetDouble(ticket,DEAL_SWAP);        		// swap
                        profit = HistoryDealGetDouble(ticket,DEAL_PROFIT);    		// profit
                        fee = HistoryDealGetDouble(ticket,DEAL_FEE);          		// fee
                       }
```

Once the position data has been preliminarily obtained, a container should be organized to store the relevant information. When implementing this functionality, we will use standard one-dimensional arrays for each feature. In order to check for the presence of a position in the storage, we will define a small Find() template function. This function will be used to check if a position is present in the container. The logic is that we pass the container and the value we want to find in it to the function parameters. As I have already mentioned, we will look for the ID of the position the deal belongs to. If the position is found, the function should return the corresponding index. If not, it returns -1.

Considering that each property of a single position needs to be stored in different formats as a string, integer values, or fractional values, it makes sense to declare the Find() function as overloadable via a template. The MQL5 programming language allows us to flexibly and quite conveniently implement this functionality through the [template](https://www.mql5.com/en/docs/basis/oop/templates) keyword. This will allow us to declare our function template with the [typename](https://www.mql5.com/en/docs/basis/oop/templates) overloadable data type once, while the compiler will automatically substitute the required implementation for each data type. Since we are not going to pass custom data types there, there will not be any problems with implicit casting of different types, and there will be no need to do any operator overloading. The implementation of the Find() custom function template is displayed below.

```
template<typename A>
int               Find(A &aArray[],A aValue)
  {
   for(int i=0; i<ArraySize(aArray); i++)
     {
      if(aArray[i]==aValue)
        {
         return(i);                                                     // The element exists, return the element index
        }
     }
   return(-1);                                                          // No such element, return -1
  }
```

Using the declared Find() function template, complete the logic by checking if the current position is in the storage. If the function returned -1, then there is no position in the storage and it needs to be added there. The storage dimension should be changed first:

```
                     //--- enter data into the main storage
                     //--- check if there is such id
                     if(Find(PositionID,position_id)==-1)               // if there is no such deal yet,                       {
```

If such a number exists in the storage, then we get access to the position data using the index returned by the Find() function. This is due to the fact that orders in the trading history may be located in different orders if several instruments are traded on the account at the same time. For example, a position can be opened on one instrument later than some earlier order on another instrument. Accordingly, it can be closed earlier than the first symbol position. In general, the logic of searching and collecting information on positions for the period is presented and summarized below.

```
      case  Select_period:                                              			// search within a period

         res = MessageBox("You have selected analysis for period. Continue?","",MB_OKCANCEL); 	// wait for user confirmation

         if(res == IDCANCEL)                                            			// if interrupted by user
           {
            printf("%s - %d -> Scrypt was stoped by user.",__FUNCTION__,__LINE__); 		// notify
            return;                                                     			// stop
           }

         MessageBox("Please press 'Ok' and wait for the next message until script will be done."); // notify

         //--- select history data
         if(HistorySelect(start_date,finish_date))                      // select the necessary period in history
           {
            int total = HistoryDealsTotal();                            // got the total number of deals

            if(total <= 0)                                              // if nothing found
              {
               printf("%s - %d -> No deals were found for the specified period.",__FUNCTION__,__LINE__); // notify
               MessageBox("No deals were found for the specified period: "+TimeToString(start_date)+"-"+TimeToString(finish_date)+". Script is done.");
               return;
              }

            for(int i=0; i<total; i++)                                  // iterate through the number of deals
              {
               //--- try to get deals ticket
               if((ticket=HistoryDealGetTicket(i))>0)                   // took the ticket
                 {
                  //--- get deals properties
                  position_id = HistoryDealGetInteger(ticket,DEAL_POSITION_ID);        // took the main id
                  entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(ticket,DEAL_ENTRY);   // entry or exit?
                  type = (int)HistoryDealGetInteger(ticket,DEAL_TYPE);  	       // entry or exit?

                  //--- check the deal type
                  if(type == DEAL_TYPE_BUY            ||                // if it is buy
                     type == DEAL_TYPE_SELL           ||                // if it is sell
                     type == DEAL_TYPE_BUY_CANCELED   ||                // if canceled buy
                     type == DEAL_TYPE_SELL_CANCELED                    // if canceled sell
                    )
                    {
                     //--- is it entry or exit?
                     if(entry == DEAL_ENTRY_IN)                         // if this is an entry
                       {
                        open = HistoryDealGetDouble(ticket,DEAL_PRICE);                	// take open price
                        time_open  =(datetime)HistoryDealGetInteger(ticket,DEAL_TIME); 	// take open time
                        symbol=HistoryDealGetString(ticket,DEAL_SYMBOL);   		// take symbol
                        stop_loss = HistoryDealGetDouble(ticket,DEAL_SL);  		// take Stop Loss
                        take_profit = HistoryDealGetDouble(ticket,DEAL_TP);		// take Take Profit

                        magic = (int)HistoryDealGetInteger(ticket,DEAL_MAGIC);   	// take magic number
                        comment=HistoryDealGetString(ticket,DEAL_COMMENT);       	// take comment
                        externalID=HistoryDealGetString(ticket,DEAL_EXTERNAL_ID);	// take external id
                        volume = HistoryDealGetDouble(ticket,DEAL_VOLUME);          	// take volume
                        commission = HistoryDealGetDouble(ticket,DEAL_COMMISSION);  	// take commission value
                       }

                     if(entry == DEAL_ENTRY_OUT)                        		// if this is an exit
                       {
                        close = HistoryDealGetDouble(ticket,DEAL_PRICE);               	// take close price
                        time_close  =(datetime)HistoryDealGetInteger(ticket,DEAL_TIME);	// take close time

                        reason = (ENUM_DEAL_REASON)HistoryDealGetInteger(ticket,DEAL_REASON); // reason
                        swap = HistoryDealGetDouble(ticket,DEAL_SWAP);        		// swap
                        profit = HistoryDealGetDouble(ticket,DEAL_PROFIT);    		// profit
                        fee = HistoryDealGetDouble(ticket,DEAL_FEE);          		// fee
                       }

                     //--- enter data into the main storage
                     //--- check if there is such id
                     if(Find(PositionID,position_id)==-1)               		// if there is no such deal yet,
                       {
                        //--- change the size of containers
                        ArrayResize(arr_time_open,ArraySize(arr_time_open)+1);
                        ArrayResize(arr_time_close,ArraySize(arr_time_close)+1);
                        ArrayResize(arr_symbol,ArraySize(arr_symbol)+1);
                        ArrayResize(arr_stop_loss,ArraySize(arr_stop_loss)+1);
                        ArrayResize(arr_take_profit,ArraySize(arr_take_profit)+1);
                        ArrayResize(arr_open,ArraySize(arr_open)+1);
                        ArrayResize(arr_close,ArraySize(arr_close)+1);
                        ArrayResize(PositionID,ArraySize(PositionID)+1);

                        ArrayResize(arr_magic,ArraySize(arr_magic)+1);
                        ArrayResize(arr_extermalID,ArraySize(arr_extermalID)+1);
                        ArrayResize(arr_comment,ArraySize(arr_comment)+1);
                        ArrayResize(arr_volume,ArraySize(arr_volume)+1);
                        ArrayResize(arr_commission,ArraySize(arr_commission)+1);
                        ArrayResize(arr_reason,ArraySize(arr_reason)+1);
                        ArrayResize(arr_swap,ArraySize(arr_swap)+1);
                        ArrayResize(arr_profit,ArraySize(arr_profit)+1);
                        ArrayResize(arr_fee,ArraySize(arr_fee)+1);

                        PositionID[ArraySize(arr_time_open)-1]=position_id;

                        if(entry == DEAL_ENTRY_IN)                      		  // if this is an entry,
                          {
                           arr_time_open[    ArraySize(arr_time_open)-1]   = time_open;   // deal time
                           arr_symbol[       ArraySize(arr_symbol)-1]      = symbol;      // instrument symbol
                           arr_stop_loss[    ArraySize(arr_stop_loss)-1]   = stop_loss;   // deal Stop Loss
                           arr_take_profit[  ArraySize(arr_take_profit)-1] = take_profit; // deal Take Profit
                           arr_open[         ArraySize(arr_open)-1]        = open;        // open price
                           //---
                           arr_magic[        ArraySize(arr_magic)-1]       = magic;       // magic number
                           arr_comment[      ArraySize(arr_comment)-1]     = comment;     // comment
                           arr_extermalID[   ArraySize(arr_extermalID)-1]  = externalID;  // external id
                           arr_volume[       ArraySize(arr_volume)-1]      = volume;      // volume
                           arr_commission[   ArraySize(arr_commission)-1]  = commission;  // commission
                          }

                        if(entry == DEAL_ENTRY_OUT)                     		  // if this is an exit
                          {
                           arr_time_close[   ArraySize(arr_time_close)-1]  = time_close;  // close time
                           arr_close[        ArraySize(arr_close)-1]       = close;       // close price
                           //---
                           arr_reason[       ArraySize(arr_reason)-1]      = reason;      // reason
                           arr_swap[         ArraySize(arr_swap)-1]        = swap;        // swap
                           arr_profit[       ArraySize(arr_profit)-1]      = profit;      // profit
                           arr_fee[          ArraySize(arr_fee)-1]         = fee;         // fee
                          }
                       }
                     else
                       {
                        int index = Find(PositionID,position_id);       // if found, search for the index

                        if(entry == DEAL_ENTRY_IN)                      // if this is an entry
                          {
                           arr_time_open[index]   = time_open;          // deal time
                           arr_symbol[index]      = symbol;             // symbol
                           arr_stop_loss[index]   = stop_loss;          // deal Stop Loss
                           arr_take_profit[index] = take_profit;        // deal Take Profit
                           arr_open[index]        = open;               // close price
                           //---
                           arr_magic[index]       = magic;              // magic number
                           arr_comment[index]     = comment;            // comment
                           arr_extermalID[index]  = externalID;         // external id
                           arr_volume[index]      = volume;             // volume
                           arr_commission[index]  = commission;         // commission
                          }

                        if(entry == DEAL_ENTRY_OUT)                     // if this is an exit
                          {
                           arr_time_close[index]  = time_close;         // deal close time
                           arr_close[index]       = close;              // deal close price
                           //---
                           arr_reason[index]      = reason;             // reason
                           arr_swap[index]        = swap;               // swap
                           arr_profit[index]      = profit;             // profit
                           arr_fee[index]         = fee;                // fee
                          }
                       }
                    }
                 }
              }
           }
         else
           {
            printf("%s - %d -> Error of selecting history deals: %d",__FUNCTION__,__LINE__,GetLastError()); // notify
            printf("%s - %d -> No deals were found for the specified period.",__FUNCTION__,__LINE__);       // notify
            MessageBox("No deals were found for the specified period: "+TimeToString(start_date)+"-"+TimeToString(finish_date)+". Script is done.");
            return;
           }
         break;
```

### First part conclusion

In this article, we have considered the importance of historical trading analysis for safe and long-term trading in the financial markets. A key element of this analysis is the study of historical deals, which we have started to implement in the script. We have considered the formation of inputs and the implementation of the algorithm for selecting historical data on deals within the selected period. We have also implemented an overloadable function template to simplify handling data containers.

In the next article, we will complete the script while considering the algorithm for selecting data for a single deal, as well as drawing charts and implementing the code for displaying data objects on charts.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14903](https://www.mql5.com/ru/articles/14903)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14903.zip "Download all attachments in the single ZIP archive")

[DealsPrintScreen.mq5](https://www.mql5.com/en/articles/download/14903/dealsprintscreen.mq5 "Download DealsPrintScreen.mq5")(104.52 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Visualizing deals on a chart (Part 2): Data graphical display](https://www.mql5.com/en/articles/14961)
- [Risk manager for algorithmic trading](https://www.mql5.com/en/articles/14634)
- [Risk manager for manual trading](https://www.mql5.com/en/articles/14340)
- [Balancing risk when trading multiple instruments simultaneously](https://www.mql5.com/en/articles/14163)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/474861)**
(8)


![Aleksandr Seredin](https://c.mql5.com/avatar/2022/4/62543FFE-A51A.jpg)

**[Aleksandr Seredin](https://www.mql5.com/en/users/al.s.capital)**
\|
30 May 2024 at 08:26

**Yuriy Bykov [#](https://www.mql5.com/ru/forum/467806#comment_53525913):**

Thanks for the article!

Look at using [structures](https://www.mql5.com/ru/docs/basis/types/classes) to store similar sets of information. Instead of separate arrays for each transaction parameter, you could define a different data type for each parameter that includes all parameters, and then store the elements of our new type in a single array. This might be more convenient.

Thanks a lot for your comment! I agree, through custom structure data types or classes it will look more concise and probably more correct. Will definitely use this advice in the future :)

![Aleksandr Slavskii](https://c.mql5.com/avatar/2017/4/58E88E5E-2732.jpg)

**[Aleksandr Slavskii](https://www.mql5.com/en/users/s22aa)**
\|
30 May 2024 at 16:50

**Yevgeniy Koshtenko [#](https://www.mql5.com/ru/forum/467806#comment_53519628):**

Wow I wonder if it's realistic to upload deals to say, a Telegram channel? Screens)

Dak [here is a](https://www.mql5.com/ru/articles/2355) bible, very convenient for sending messages or screenshots to the Telegram channel. I have been using it for a long time.

But now ChartScreenShot is broken, the objects that LABEL are moved to the side.

I have to think how to make screens with some third-party software instead of the standard one.

![Aleksandr Seredin](https://c.mql5.com/avatar/2022/4/62543FFE-A51A.jpg)

**[Aleksandr Seredin](https://www.mql5.com/en/users/al.s.capital)**
\|
30 May 2024 at 17:55

**Aleksandr Slavskii [#](https://www.mql5.com/ru/forum/467806#comment_53533420):**

So [here](https://www.mql5.com/ru/articles/2355) is a bible, very convenient for sending messages or screenshots to the cart. I've been using it for a long time.

But now ChartScreenShot is broken, objects that LABEL are moved aside.

I have to think how to make screens with some third-party software instead of the standard one.

Thanks for the link. I will definitely read it.

![Retail Trading Realities LTD](https://c.mql5.com/avatar/2025/4/68116106-adc9.png)

**[Philip Kym Sang Nelson](https://www.mql5.com/en/users/rtr_ltd)**
\|
26 Oct 2024 at 23:05

Brilliant


![Aleksandr Seredin](https://c.mql5.com/avatar/2022/4/62543FFE-A51A.jpg)

**[Aleksandr Seredin](https://www.mql5.com/en/users/al.s.capital)**
\|
28 Oct 2024 at 11:43

**Philip Kym Sang Nelson [#](https://www.mql5.com/en/forum/474861#comment_54945388):**

Brilliant

Thank you, Sir!

![MQL5 Trading Toolkit (Part 3): Developing a Pending Orders Management EX5 Library](https://c.mql5.com/2/98/MQL5_Trading_Toolkit_Part_3___LOGO.png)[MQL5 Trading Toolkit (Part 3): Developing a Pending Orders Management EX5 Library](https://www.mql5.com/en/articles/15888)

Learn how to develop and implement a comprehensive pending orders EX5 library in your MQL5 code or projects. This article will show you how to create an extensive pending orders management EX5 library and guide you through importing and implementing it by building a trading panel or graphical user interface (GUI). The expert advisor orders panel will allow users to open, monitor, and delete pending orders associated with a specified magic number directly from the graphical interface on the chart window.

![Neural Network in Practice: Least Squares](https://c.mql5.com/2/76/Rede_neural_na_protica_Manimos_Quadrados___LOGO.png)[Neural Network in Practice: Least Squares](https://www.mql5.com/en/articles/13670)

In this article, we'll look at a few ideas, including how mathematical formulas are more complex in appearance than when implemented in code. In addition, we will consider how to set up a chart quadrant, as well as one interesting problem that may arise in your MQL5 code. Although, to be honest, I still don't quite understand how to explain it. Anyway, I'll show you how to fix it in code.

![Developing a Replay System (Part 48): Understanding the concept of a service](https://c.mql5.com/2/76/Desenvolvendo_um_sistema_de_Replay_9Parte_480___LOGO.png)[Developing a Replay System (Part 48): Understanding the concept of a service](https://www.mql5.com/en/articles/11781)

How about learning something new? In this article, you will learn how to convert scripts into services and why it is useful to do so.

![Reimagining Classic Strategies (Part X): Can AI Power The MACD?](https://c.mql5.com/2/97/Reimagining_Classic_Strategies_Part_X___LOGO.png)[Reimagining Classic Strategies (Part X): Can AI Power The MACD?](https://www.mql5.com/en/articles/16066)

Join us as we empirically analyzed the MACD indicator, to test if applying AI to a strategy, including the indicator, would yield any improvements in our accuracy on forecasting the EURUSD. We simultaneously assessed if the indicator itself is easier to predict than price, as well as if the indicator's value is predictive of future price levels. We will furnish you with the information you need to decide whether you should consider investing your time into integrating the MACD in your AI trading strategies.

[![](https://www.mql5.com/ff/si/0nfwvn6yhmgzf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F117%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dorder.expert%26utm_content%3Dorder.freelance%26utm_campaign%3D0622.MQL5.com.Internal&a=tunpwtbhegzufrqocbwiszessdutnobs&s=d9e7484e15300021b4066b1df77a94a1352f9e7c326d5113006bb4f6476bafeb&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=bcbhkenzqrmfebtxtvgskmfexldntwqv&ssn=1769251697818467845&ssn_dr=0&ssn_sr=0&fv_date=1769251697&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14903&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Visualizing%20deals%20on%20a%20chart%20(Part%201)%3A%20Selecting%20a%20period%20for%20analysis%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925169718183951&fz_uniq=5083122992352859504&sv=2552)

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