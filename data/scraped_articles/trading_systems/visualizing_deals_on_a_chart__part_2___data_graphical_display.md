---
title: Visualizing deals on a chart (Part 2): Data graphical display
url: https://www.mql5.com/en/articles/14961
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:48:08.887853
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/14961&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083119208486671718)

MetaTrader 5 / Trading systems


### Introduction

In this article, we will complete the script for visualizing deals on the chart we started implementing in " [Visualizing deals on a chart (Part 1): Selecting a period for analysis](https://www.mql5.com/en/articles/14903)". We will write the code to select data for a single user-selected deal, as well as to draw the necessary data objects on the chart, which we will then save to a file as a print screen of the corresponding charts. The script will allow us to save a significant amount of time on technical work related to the formation of deal graphs, as well as on saving them in print screens for retrospective analysis. Those who do not want to spend time assembling the projects can download a ready-made version of the script on the [Market](https://www.mql5.com/en/market/product/86223).

### Selecting data for one deal

Unlike selecting data on deals for a certain period, selecting data on a single deal will significantly simplify the implementation of the historical order selection case. The main difference here is the fact that instead of the [HistorySelect()](https://www.mql5.com/en/docs/trading/historyselect) predefined terminal function, we are going to use the [HistorySelectByPosition()](https://www.mql5.com/en/docs/trading/historyselectbyposition) method to request historical data. The method parameters should receive [POSITION\_IDENTIFIER](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_integer) we can find in the MetaTrader 5 terminal (View -> Toolbox -> History -> Ticket column). The value is to be passed to the script via the inp\_d\_ticket input global variable.

In all other aspects, the logic of the Select\_one\_deal case completely repeats the implementation of the previous case logic, and is presented in full in the code below with the same information inserts for users.

```
      //--- if one deal is needed
      case Select_one_deal:

         res = MessageBox("You have selected analysis of one deal. Continue?","",MB_OKCANCEL); // informed in the message

         if(res == IDCANCEL)                                            // if interrupted by user
           {
            printf("%s - %d -> Scrypt was stoped by user.",__FUNCTION__,__LINE__);  // informed in the journal
            return;                                                     // interrupted
           }

         MessageBox("Please press 'Ok' and wait for the next message until script will be done."); // informed in the message

         //--- select by one position
         if(HistorySelectByPosition(inp_d_ticket))                      // select position by id
           {
            int total = HistoryDealsTotal();                            // total deals

            if(total <= 0)                                              // if nothing found
              {
               printf("%s - %d -> Deal was not found.",__FUNCTION__,__LINE__); // notify
               MessageBox("Deal was not found with this tiket: "+IntegerToString(inp_d_ticket)+". Script is done."); // informed in the message
               return;
              }

            for(int i=0; i<total; i++)                                  // iterate through the number of deals
              {
               //--- try to get deals ticket
               if((ticket=HistoryDealGetTicket(i))>0)                   // took the deal number
                 {
                  //--- get deals properties
                  position_id = HistoryDealGetInteger(ticket,DEAL_POSITION_ID);     // took the main id
                  entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(ticket,DEAL_ENTRY);// entry or exit?

                  if(entry == DEAL_ENTRY_IN)                                        // if this is an entry
                    {
                     open = HistoryDealGetDouble(ticket,DEAL_PRICE);                // take open price
                     time_open  =(datetime)HistoryDealGetInteger(ticket,DEAL_TIME); // take open time
                     symbol=HistoryDealGetString(ticket,DEAL_SYMBOL);   	    // take symbol
                     stop_loss = HistoryDealGetDouble(ticket,DEAL_SL);  	    // take Stop Loss
                     take_profit = HistoryDealGetDouble(ticket,DEAL_TP);	    // take Take Profit
                     //---
                     magic = (int)HistoryDealGetInteger(ticket,DEAL_MAGIC);   	    // take Magic
                     comment=HistoryDealGetString(ticket,DEAL_COMMENT);       	    // take comment
                     externalID=HistoryDealGetString(ticket,DEAL_EXTERNAL_ID); 	    // take external id
                     volume = HistoryDealGetDouble(ticket,DEAL_VOLUME);             // take volume
                     commission = HistoryDealGetDouble(ticket,DEAL_COMMISSION);     // take commission value
                    }

                  if(entry == DEAL_ENTRY_OUT)                           	    // if this is an exit
                    {
                     close = HistoryDealGetDouble(ticket,DEAL_PRICE);               // take close price
                     time_close  =(datetime)HistoryDealGetInteger(ticket,DEAL_TIME);// take close time
                     //---
                     reason = (ENUM_DEAL_REASON)HistoryDealGetInteger(ticket,DEAL_REASON); // take reason
                     swap = HistoryDealGetDouble(ticket,DEAL_SWAP);     // take swap
                     profit = HistoryDealGetDouble(ticket,DEAL_PROFIT); // take profit
                     fee = HistoryDealGetDouble(ticket,DEAL_FEE);       // take fee
                    }

                  //--- enter data into the main storage
                  //--- check if there is such id
                  if(Find(PositionID,position_id)==-1)                         // if there is no such deal,
                    {
                     //--- change the dimensions of the arrays
                     ArrayResize(arr_time_open,ArraySize(arr_time_open)+1);    // open time
                     ArrayResize(arr_time_close,ArraySize(arr_time_close)+1);  // close time
                     ArrayResize(arr_symbol,ArraySize(arr_symbol)+1);          // symbols
                     ArrayResize(arr_stop_loss,ArraySize(arr_stop_loss)+1);    // stop levels
                     ArrayResize(arr_take_profit,ArraySize(arr_take_profit)+1);// profits
                     ArrayResize(arr_open,ArraySize(arr_open)+1);              // entries
                     ArrayResize(arr_close,ArraySize(arr_close)+1);            // exits
                     ArrayResize(PositionID,ArraySize(PositionID)+1);          // position id
                     //---
                     ArrayResize(arr_magic,ArraySize(arr_magic)+1);            // Magic
                     ArrayResize(arr_extermalID,ArraySize(arr_extermalID)+1);  // external id
                     ArrayResize(arr_comment,ArraySize(arr_comment)+1);        // comment
                     ArrayResize(arr_volume,ArraySize(arr_volume)+1);          // volume
                     ArrayResize(arr_commission,ArraySize(arr_commission)+1);  // commission
                     ArrayResize(arr_reason,ArraySize(arr_reason)+1);          // reason
                     ArrayResize(arr_swap,ArraySize(arr_swap)+1);              // swap
                     ArrayResize(arr_profit,ArraySize(arr_profit)+1);          // profit
                     ArrayResize(arr_fee,ArraySize(arr_fee)+1);                // fee

                     PositionID[ArraySize(arr_time_open)-1]=position_id;       // id

                     if(entry == DEAL_ENTRY_IN)                         	       // if this is an entry
                       {
                        arr_time_open[    ArraySize(arr_time_open)-1]   = time_open;   // deal time
                        arr_symbol[       ArraySize(arr_symbol)-1]      = symbol;      // instrument symbol
                        arr_stop_loss[    ArraySize(arr_stop_loss)-1]   = stop_loss;   // deal stop loss
                        arr_take_profit[  ArraySize(arr_take_profit)-1] = take_profit; // deal take profit
                        arr_open[         ArraySize(arr_open)-1]        = open;        // open price
                        //---
                        arr_magic[        ArraySize(arr_magic)-1]       = magic;       // Magic
                        arr_comment[      ArraySize(arr_comment)-1]     = comment;     // comment
                        arr_extermalID[   ArraySize(arr_extermalID)-1]  = externalID;  // external id
                        arr_volume[       ArraySize(arr_volume)-1]      = volume;      // volume
                        arr_commission[   ArraySize(arr_commission)-1]  = commission;  // commission
                       }

                     if(entry == DEAL_ENTRY_OUT)                        	       // if this is an exit
                       {
                        arr_time_close[   ArraySize(arr_time_close)-1]  = time_close;  // close time
                        arr_close[        ArraySize(arr_close)-1]       = close;       // close prices
                        //---
                        arr_reason[       ArraySize(arr_reason)-1]      = reason;      // reason
                        arr_swap[         ArraySize(arr_swap)-1]        = swap;        // swap
                        arr_profit[       ArraySize(arr_profit)-1]      = profit;      // profit
                        arr_fee[          ArraySize(arr_fee)-1]         = fee;         // fee
                       }
                    }
                  else
                    {
                     int index = Find(PositionID,position_id);          // if there was a record already,

                     if(entry == DEAL_ENTRY_IN)                         // if this was an entry
                       {
                        arr_time_open[index]   = time_open;             // deal time
                        arr_symbol[index]      = symbol;                // symbol
                        arr_stop_loss[index]   = stop_loss;             // deal stop loss
                        arr_take_profit[index] = take_profit;           // deal take profit
                        arr_open[index]        = open;                  // open price
                        //---
                        arr_magic[index]       = magic;                 // Magic
                        arr_comment[index]     = comment;               // comment
                        arr_extermalID[index]  = externalID;            // external id
                        arr_volume[index]      = volume;                // volume
                        arr_commission[index]  = commission;            // commission
                       }

                     if(entry == DEAL_ENTRY_OUT)                        // if this is an exit
                       {
                        arr_time_close[index]  = time_close;            // deal close time
                        arr_close[index]       = close;                 // deal close price
                        //---
                        arr_reason[index]      = reason;                // reason
                        arr_swap[index]        = swap;                  // swap
                        arr_profit[index]      = profit;                // profit
                        arr_fee[index]         = fee;                   // fee
                       }
                    }
                 }
              }
           }
         else
           {
            printf("%s - %d -> Error of selecting history deals: %d",__FUNCTION__,__LINE__,GetLastError());	// informed in the journal
            printf("%s - %d -> Deal was not found.",__FUNCTION__,__LINE__); 					// informed in the journal
            MessageBox("Deal was not found with this tiket: "+IntegerToString(inp_d_ticket)+". Script is done."); // informed in the message
            return;
           }
         break;
```

Now that both options have been described, and all storages have been filled with the necessary data during the program execution, we can start displaying this data on the terminal charts.

### Displaying required charts

To save deals on the chart, we need to first open a new window with the required symbol at the program level, make the necessary design settings, including individual shift of the indent on the right so that the entire deal is clearly visible, and call a predefined function that will save a print screen to the required folder.

First, let's declare the local variables we need to open the window of the desired chart. The 'bars' variable will store the offset value for the chart on the right, the 'chart\_width' and 'chart\_height' variables will store the corresponding sizes for saving, and the handle of the new chart, when it is opened, will be stored in the 'handle' variable for accessing the chart in the future.

```
//--- data collected, moving on to printing
   int bars = -1;                                                       // number of bars in a shift
   int chart_width = -1;                                                // chart width
   int chart_height =-1;                                                // chart height
   long handle =-1;                                                     // chart handle
```

Before starting a request to open new symbol windows, we should make requests for the validity of these symbols from history. This check is absolutely necessary to avoid the error of opening a "non-existent symbol" on the account. I think it is necessary to explain here where a "non-existent symbol" can come from if it has been saved in trading history, which means it once existed.

First of all, this may be related to the broker account types. Today, most brokers provide traders with several account options in order to make their use as profitable and convenient as possible in terms of the trading strategies used. Some accounts charge a commission to open deals but have a very low spread, while some account types have a high spread but no per-deal fee. Therefore, those traders who trade medium-term may not pay a commission for a deal, while the size of the spread in medium-term trading is not so important. Conversely, traders who trade small impulses intraday would rather pay a commission to open a deal than take a loss just because the spread "suddenly" widened. Typically, brokers package such conditions into account types, such as Standard, Gold, Platinum, ESN, and enter a symbol name for each account. For example, in case of the EURUSD pair on a standard account, the symbol on another account type might look like EURUSDb, EURUSDz or EURUSD\_i depending on the broker.

Also, symbol names may change depending on the expiration date of certain instruments not related to trading currency pairs on Forex, but we will not consider this point in detail here, since the article is still devoted specifically to currency pairs.

Another condition for the need to check the symbol validity is the purely technical absence of a subscription to the necessary instruments in the Market Watch window of the terminal. Even if a symbol name exists on the authorized account, but is not selected in the terminal context menu (View -> Market Watch), we will not be able to open its chart with a subsequent error of the calling function.

We will start implementing the check by arranging a loop for iterating over each tool in our storage, as shown below.

```
   for(int i=0; i<ArraySize(arr_symbol); i++)                           // iterate through all deal symbols
```

In order to check the validity of a symbol saved in our container, we will use the [SymbolSelect()](https://www.mql5.com/en/docs/marketinformation/symbolselect) predefined terminal function. The first parameter we will pass to it is the symbol name in the [string](https://www.mql5.com/en/docs/basis/types/stringconst) format. This is a symbol whose validity we want to check. The logical value of 'true' comes second. Passing 'true' as the second parameter means that if the given instrument is valid but not selected in the "Market Watch", it should be selected there automatically. The complete check logic looks as follows.

```
//--- check for symbol availability

   for(int i=0; i<ArraySize(arr_symbol); i++)                           // iterate through all deal symbols
     {
      if(!SymbolSelect(arr_symbol[i],true))                             // check if the symbol is in the book and add if not
        {
         printf("%s - %d -> Failed to add a symbol %s to the marketbook. Error: %d",
			__FUNCTION__,__LINE__,arr_symbol[i],GetLastError()); // informed in the journal
         MessageBox("Failed to add a symbol to the marketbook: "+arr_symbol[i]+
			". Please select 'show all' in the your market book and try again. Script is done."); // informed in the message
         return;                                                        // if failed, abort
        }
     }
```

Accordingly, if the validity check of the symbol is not passed, we terminate the program execution with appropriate notifications for the user. Once all validity checks have been passed, we can proceed to opening the necessary symbol charts directly in the terminal.

First of all, let's provide the deal\_close\_date auxiliary variable of the [MqlDateTime](https://www.mql5.com/en/docs/constants/structures/mqldatetime) data type, which will further help us to conveniently sort all saved charts into the corresponding time period folders. For explicit [datetime](https://www.mql5.com/en/docs/standardlibrary/controls/cdatetime/cdatetimedatetime) data type reduction to [MqlDateTime](https://www.mql5.com/en/docs/constants/structures/mqldatetime) data type in our storage, we will use the [TimeToStruct()](https://www.mql5.com/en/docs/dateandtime/timetostruct) predefined terminal function as shown below.

```
      MqlDateTime deal_close_date;                                      // deal closure date in the structure
      TimeToStruct(arr_time_close[i],deal_close_date);                  // pass date to the structure
```

The charts are to be drawn according to the user-defined data in the main\_graph, addition\_graph, addition\_graph\_2 and addition\_graph\_3 variables. If the variable contains the [PERIOD\_CURRENT](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes) enumeration value, we do not draw any chart. If a specific value is entered into the variable (for example [PERIOD\_D1](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes)), we take this chart for drawing. We will perform this check for all the entered variables in the following form (the main variable is shown below as an example):

```
      //--- check the main one
      if(main_graph != PERIOD_CURRENT)                                  // if the main one selected
```

Drawing each chart will begin with opening a new chart of the required symbol. The symbol chart will be opened using the [ChartOpen()](https://www.mql5.com/en/docs/chart_operations/chartopen) predefined terminal function, while passing the required symbol and timeframe from the storage, as shown below.

```
         //--- open the required chart
         handle = ChartOpen(arr_symbol[i],main_graph);                  // open the necessary symbol chart
```

Once the chart is open, we apply to it all the standard user settings I mentioned above. To do this, we will use the [ChartApplyTemplate()](https://www.mql5.com/en/docs/chart_operations/chartapplytemplate) predefined terminal function, which will help us a lot with this and save us from writing the code ourselves. The [ChartApplyTemplate()](https://www.mql5.com/en/docs/chart_operations/chartapplytemplate) function parameters get the chart handle obtained from calling the [ChartOpen()](https://www.mql5.com/en/docs/chart_operations/chartopen) function, as well as the name of the template specified by the user for the deal timeframe in the dailyHistorytemp format. The code for calling the template application function is shown below.

```
         ChartApplyTemplate(handle,main_template);                      // apply template
```

Let's make a small digression here for those who have not used templates in the MetaTrader 5 terminal until now. If we use an "ugly" template, the saved print screen of the deal may turn out to be "irritating" or even "useless". Follow these steps to create your own dailyHistorytemp template:

- Open the chart of any symbol via File - New Chart.
- Once the chart is open, press F8 to open the Properties window, for example "PropertiesGBPAUD,Daily".
- The Properties window contains several tabs: Common, Show and Colors. On each of them, make the settings that are more familiar to you - for example, for a daily chart, and click OK. Find the details here - [Chart Settings (official terminal Help)](https://www.metatrader5.com/en/terminal/help/charts_advanced/charts_settings "https://www.metatrader5.com/en/terminal/help/charts_advanced/charts_settings").
- After clicking OK, the Properties window is closed and the chart takes the form you need.
- Now in the context menu, select Charts - Templates - Save Template. The template saving window appears and the template saving window appears. Enter dailyHistorytemp.tpl in the "File name" and click Save.
- After that, the ..MQL5\\Profiles\\Templates terminal folder will feature the dailyHistorytemp.tpl file, which you can use in the script. The main thing to note is that the template name is entered into the script without the .tpl extension.

Now let's get back to our code. Once the desired template has been applied, we need to make a small delay in the code execution to give the chart time to load in the desired quality. Otherwise, the chart may not be displayed correctly due to the time required to load the required historical price data into the terminal. For example, if you have not opened a chart for quite a while, the terminal needs time to display it correctly. We will announce the time delay through the [Sleep()](https://www.mql5.com/en/docs/common/sleep) predefined terminal function, as shown below.

```
         Sleep(2000);                                                   // wait for the chart to load
```

As a delay, we will use the value of 2000 milliseconds or 2 seconds, taken purely from practice, so that the chart is guaranteed to have time to load, and the execution of the script does not reach long minutes with a large number of deals. To customize this value yourself, you can independently enter this value into the script settings to speed up or slow down the process, depending on the performance of your equipment or Internet connection. As practice shows, two seconds will be enough for most cases.

Now we need to disable scrolling the charts to the most recent bar values, since we are analyzing history and we do not need new ticks to shift our chart to the right all the time. This can be done by setting the [CHART\_AUTOSCROLL](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer "https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer") property to 'false' of the necessary chart via the [ChartSetInteger()](https://www.mql5.com/en/docs/chart_operations/chartsetinteger) predefined function, as shown below.

```
         ChartSetInteger(handle,CHART_AUTOSCROLL,false);                // disable auto scroll
```

Now that auto scroll is disabled, we first need to count the number of bars on the chart of the corresponding timeframe to the left in order to shift the chart towards history for the period of closing the deal in question. We can get the value via the [iBarShift()](https://www.mql5.com/en/docs/series/ibarshift) predefined terminal function, passing the symbol, chart timeframe and deal closure time as parameters, since we want to see the entire deal on the print screen from start to finish. In the 'exact' parameter, we pass 'false' just in case the history is really deep. However, it is not so critical for our implementation in this case. The full method call with the parameters is shown below.

```
         bars = iBarShift(arr_symbol[i],main_graph,arr_time_close[i],false); // get the shift for the deal time
```

Once we know the chart shift we need, we can display exactly the period that will capture the deal we need in history. We can shift the chart in the desired direction by the distance we need using the [ChartNavigate()](https://www.mql5.com/en/docs/chart_operations/chartnavigate) predefined terminal variable by passing the following parameters to it, as shown below.

```
         ChartNavigate(handle,CHART_CURRENT_POS,-bars+bars_from_right_main); // shifted the chart with a custom margin
```

To shift the chart, we passed the chart handle, the [CHART\_CURRENT\_POS](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_position) current position value of the [ENUM\_CHART\_POSITION](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_position) enumeration, as well as the shift to the deal we obtained earlier in the 'bars' variable with the offset entered by the user to assess the potential for price movement after exiting the position.

After the described chart transformations, call the [ChartRedraw()](https://www.mql5.com/en/docs/chart_operations/chartredraw) method just in case and start drawing additional data on the chart to analyze historical deals.

In order to draw custom information panel elements and lines indicating position opening and closing, as well as Stop Loss and Take Profit levels, we will use the corresponding paintDeal() and paintPanel() custom functions. We will define them ourselves based on standard patterns of behavior for working with terminal charts, where paintDeal() will draw lines of opening and closing deal prices, as well as Take Profit and Stop Loss, while the paintPanel() method will contain a table containing the full deal information in the corner of the screen.

The detailed definition of the methods is provided in the next section. Here we will simply indicate that the methods will be called in this code segment. This is also done from the point of view that you do not necessarily have to use the implementation given in the current article to draw these two groups of elements. You can redefine them yourself, while keeping the desired signature. The implementation of these methods in the current article is an example of the optimal ratio of beauty and information content of graphics at the time of writing it in the code. The main objective here is to keep the position of the method calls in the main code.

```
         //--- draw the deal
         paintDeal(handle,PositionID[i],arr_stop_loss[i],arr_take_profit[i],arr_open[i],arr_close[i],arr_time_open[i],arr_time_close[i]);

         //--- draw the information panel
         paintPanel(handle,PositionID[i],arr_stop_loss[i],arr_take_profit[i],arr_open[i],
                    arr_close[i],arr_time_open[i],arr_time_close[i],arr_magic[i],arr_comment[i],
                    arr_extermalID[i],arr_volume[i],arr_commission[i],arr_reason[i],arr_swap[i],
                    arr_profit[i],arr_fee[i],arr_symbol[i],(int)SymbolInfoInteger(arr_symbol[i],SYMBOL_DIGITS));
```

After the methods have drawn the deal lines and the information panel on the chart, we can proceed to the implementation of saving a print screen of everything that happened on the current chart. To do this, we first determine the future dimensions of the print screen in width and height by simply requesting this data from the open chart using the [ChartSetInteger()](https://www.mql5.com/en/docs/chart_operations/chartsetinteger) redefined terminal function, as shown below.

```
         //--- get data by screen size
         chart_width = (int) ChartGetInteger(handle,CHART_WIDTH_IN_PIXELS);   // look at the chart width
         chart_height = (int) ChartGetInteger(handle,CHART_HEIGHT_IN_PIXELS); // look at the chart height
```

We passed the [ENUM\_CHART\_PROPERTY\_INTEGER](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer "https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer") enumeration values for the [CHART\_WIDTH\_IN\_PIXELS](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer "https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer") width and the [CHART\_HEIGHT\_IN\_PIXELS](https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer "https://www.mql5.com/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer") height respectively as the corresponding parameters for displaying the chart.

Having received the size data, we will need to create a path for saving the deal print screen chart in the standard terminal folder. To prevent the EA from placing all files in one folder, but instead sorting them for the user's convenience, we automate this process through the file name in the next string.

```
         string name_main_screen = brok_name+"/"+
                                   IntegerToString(account_num)+"/"+
                                   IntegerToString(deal_close_date.year)+"-"+IntegerToString(deal_close_date.mon)+
				   "-"+IntegerToString(deal_close_date.day)+"/"+
                                   IntegerToString(PositionID[i])+"/"+
                                   EnumToString(main_graph)+IntegerToString(PositionID[i])+".png"; // assign the name
```

Graphically, the structure of sorting files into folders in a standard directory will be as shown in Figure 1.

![Figure 1. Structure of folder addresses of saved print screens by deals](https://c.mql5.com/2/80/2024-06-10_10h16_33.png)

Figure 1. Structure of folder addresses of saved print screens by deals

As we can see, the chart files will be sorted by broker name, account number, year, month and day of execution, so that the user can easily find the desired deal without looking for the file name in one general list. Different timeframes will be located in the folder of the corresponding position number from the terminal.

We will directly save the information by calling the ChartScreenShot() predefined terminal function, passing to it the handle of the required chart, the print screen sizes we obtained earlier, which correspond to the chart sizes, and also the name of the file containing the entire structure of folder addresses, as parameters as shown in Figure 1 and in the code below.

```
         ChartScreenShot(handle,name_main_screen,chart_width,chart_height,ALIGN_LEFT);             // make a screenshot
```

If the folders specified in the hierarchy do not exist in the standard terminal folder, the terminal will create them automatically without user intervention.

After saving the file, we can close the chart so as not to clutter the terminal view, especially if the download contains a large number of historical deals on the account. We will close the chart using the ChartClose() predefined terminal function passing the handle of the required chart to it, so as not to close anything unnecessary. The function call is shown below.

```
         ChartClose(handle);                                            // closed the chart
```

We will repeat this operation similarly for all timeframes specified by the user in the inputs. Now, in order to complete our script, we need to define the behavior of the paintDeal() and paintPanel() methods outside the main program code.

### Drawing data objects on charts

For convenient placement of information on the print screen chart, we only need to redefine two methods, which will determine how exactly the data required by the user will be drawn.

Let's start with a description of the paintDeal() method. Its objective is to draw graphics for a position associated with the location of the opening and closing prices, stop loss and take profit positions. To do this, declare the method description with the following signature outside the main code body:

```
void paintDeal(long handlE,
               ulong tickeT,
               double stop_losS,
               double take_profiT,
               double opeN,
               double closE,
               datetime timE,
               datetime time_closE)
```

The following values are specified in the method parameters: handlE - handle of the chart we will draw on, tickeT - deal ticket, stop\_losS - price of Stop Loss if present, take\_profiT - Take Profit if present, open price - opeN and close price - closE, deal open time - timE and deal close time - time\_closE.

Let's start drawing with the name of the object, which will correspond to a unique name that should not be repeated. Therefore, in the name we will implement a feature that this object corresponds to a stop in the form of "name\_sl\_". In order to make the name unique, we will also add the ticket number of the deal, as shown below.

```
   string name_sl = "name_sl_"+IntegerToString(tickeT);                    // assign the name
```

Now we can create the graphic object itself using the [ObjectCreate()](https://www.mql5.com/en/docs/objects/objectcreate) predefined terminal function, which draws the Stop Loss level by historical position on the chart. The passed parameters is the chart handle and unique name from the name\_sl variable. Specify the [OBJ\_ARROW\_LEFT\_PRICE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_arrow_left_price) value as an object type, which means the left price label from the [ENUM\_OBJECT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) enumeration, as well as the actual price value and the time the label was placed on the chart, as shown below.

```
   ObjectCreate(handlE,name_sl,OBJ_ARROW_LEFT_PRICE,0,timE,stop_losS);     // create the left label object
```

Now that the object has been created, let's set the values of its OBJPROP\_COLOR and OBJPROP\_TIMEFRAMES fields. Set OBJPROP\_COLOR to clrRed, since Stop Loss is usually colored red, while OBJPROP\_TIMEFRAMES is set to OBJ\_ALL\_PERIODS to display on all timeframes. Although, the second condition is not critical in this implementation. In general, the Stop Loss drawing block will look as follows.

```
//--- draw stop loss
   string name_sl = "name_sl_"+IntegerToString(tickeT);                    // assign the name
   ObjectCreate(handlE,name_sl,OBJ_ARROW_LEFT_PRICE,0,timE,stop_losS);     // create the left label object
   ObjectSetInteger(handlE,name_sl,OBJPROP_COLOR,clrRed);                  // add color
   ObjectSetInteger(handlE,name_sl,OBJPROP_TIMEFRAMES,OBJ_ALL_PERIODS);    // set visibility
   ChartRedraw(handlE);                                                    // redraw
```

After drawing each block, call the [ChartRedraw()](https://www.mql5.com/en/docs/chart_operations/chartredraw) method.

Drawing the Take Profit block will be similar to drawing Stop Loss with the following exceptions. First of all, add "name\_tp\_" plus the deal ticket to the unique object name, and set the color from the green palette, which corresponds to the traditional designation of the received profit, via the clrLawnGreen color. Otherwise, the logic is similar to the Stop Loss block and is presented in full here.

```
//--- draw take profit
   string name_tp = "name_tp_"+IntegerToString(tickeT);                    // assign the name
   ObjectCreate(handlE,name_tp,OBJ_ARROW_LEFT_PRICE,0,timE,take_profiT);   // create the left label object
   ObjectSetInteger(handlE,name_tp,OBJPROP_COLOR,clrLawnGreen);            // add color
   ObjectSetInteger(handlE,name_tp,OBJPROP_TIMEFRAMES,OBJ_ALL_PERIODS);    // set visibility
   ChartRedraw(handlE);                                                    // redraw
```

Let's move on to the implementation of drawing the entry price via the left price label as well. The difference from the previous blocks is, first of all, again in the unique name of the object. We will add "name\_open\_" at the beginning. Another difference is the clrWhiteSmoke line color, so that it does not stand out too much on the chart, but otherwise everything is the same.

```
//--- draw entry price
   string name_open = "name_open_"+IntegerToString(tickeT);                // assign the name
   ObjectCreate(handlE,name_open,OBJ_ARROW_LEFT_PRICE,0,timE,opeN);        // create the left label object
   ObjectSetInteger(handlE,name_open,OBJPROP_COLOR,clrWhiteSmoke);         // add color
   ObjectSetInteger(handlE,name_open,OBJPROP_TIMEFRAMES,OBJ_ALL_PERIODS);  // set visibility
   ChartRedraw(handlE);                                                    // redraw
```

The line connecting the deal opening and closing price labels is displayed using the same color. The line type will be different. When creating an object in the [ObjectCreate()](https://www.mql5.com/en/docs/objects/objectcreate) method parameter, we will pass the [OBJ\_TREND](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_arrow_left_price) value of the [ENUM\_OBJECT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) enumeration as the third parameter to create a trend line. To correctly position the trend line on the chart, we will need to specify additional parameters for the position of two points, where each point will have two attributes: price and time. To do this, we will pass the opening and closing prices opeN and closE to the subsequent parameters together with closing and opening times in the timE and time\_closE variables, as shown below.

```
//--- deal line
   string name_deal = "name_deal_"+IntegerToString(tickeT);                // assign the name
   ObjectCreate(handlE,name_deal,OBJ_TREND,0,timE,opeN,time_closE,closE);  // create the left label object
   ObjectSetInteger(handlE,name_deal,OBJPROP_COLOR,clrWhiteSmoke);         // add color
   ObjectSetInteger(handlE,name_deal,OBJPROP_TIMEFRAMES,OBJ_ALL_PERIODS);  // set visibility
   ChartRedraw(handlE);                                                    // redraw
```

To fully display the deal on the chart, it remains to draw the price label for closing the deal. To achieve this, we will use the right price label, so that the information is displayed on the print screen in a more visually pleasing way. To draw the right label, the [ObjectCreate()](https://www.mql5.com/en/docs/objects/objectcreate) method should receive the [OBJ\_ARROW\_RIGHT\_PRICE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_arrow_left_price) value as the third parameter, which means the right price label from the [ENUM\_OBJECT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) enumeration. For the rest of the drawing we only need the price and time, which we are going to pass via the corresponding time\_closE,closE variables as shown below.

```
//--- draw exit price
   string name_close = "name_close"+IntegerToString(tickeT);                // assign the name
   ObjectCreate(handlE,name_close,OBJ_ARROW_RIGHT_PRICE,0,time_closE,closE);// create the left label object
   ObjectSetInteger(handlE,name_close,OBJPROP_COLOR,clrWhiteSmoke);         // add color
   ObjectSetInteger(handlE,name_close,OBJPROP_TIMEFRAMES,OBJ_ALL_PERIODS);  // set visibility
   ChartRedraw(handlE);                                                     // redraw
```

This completes the description of our custom paintDeal() method for drawing the position entry and exit lines. Now we can proceed to describing the method for drawing the panel of full deal information in the paintPanel() method.

Describing the method for drawing the panel will require us to have a more complex structure of methods responsible for drawing text labels, such as [OBJ\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_arrow_left_price) type text labels, [ENUM\_OBJECT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) enumeration and [OBJ\_RECTANGLE\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_arrow_left_price) object for creating and designing user graphical interfaces. Let's declare the corresponding custom methods called LabelCreate() to create text labels and RectLabelCreate() to create a rectangular label. We will start the description with auxiliary methods, and then move on to the description of the main paintPanel() method, in which we will use auxiliary methods.

In general, the structure of our script methods will look like Figure 2.

![Figure 2. Structure of custom methods for drawing graphics](https://c.mql5.com/2/78/r2__1.JPG)

Figure 2. Structure of custom methods for drawing graphics

Declare the LabelCreate() method with the following signature as parameters:

```
bool LabelCreate(const long              chart_ID=0,               // chart ID
                 const string            name="Label",             // label name
                 const int               sub_window=0,             // subwindow number
                 const long              x=0,                      // X coordinate
                 const long              y=0,                      // Y coordinate
                 const ENUM_BASE_CORNER  corner=CORNER_LEFT_UPPER, // chart corner for anchoring
                 const string            text="Label",             // text
                 const string            font="Arial",             // font
                 const int               font_size=10,             // font size
                 const color             clr=clrRed,               // color
                 const double            angle=0.0,                // text angle
                 const ENUM_ANCHOR_POINT anchor=ANCHOR_LEFT_UPPER, // anchor type
                 const bool              back=false,               // in the background
                 const bool              selection=false,          // select to move
                 const bool              hidden=true,              // hidden in the list of objects
                 const long              z_order=0)                // priority for clicking with a mouse
```

The chart\_ID parameter receives the handle of the chart, on which we need to draw the object: 'name' is a unique object name, while the value of 0 of the sub\_window parameter means that we want to draw the object in the main chart window. The coordinates of the object upper left corner will be passed via the X and Y parameters respectively. We can change the binding of the object corner to the chart from the standard left corner by passing the corresponding value to the 'corner' parameter, but we will leave the default value of ANCHOR\_LEFT\_UPPER there. Pass the string value of the information to be displayed in the 'text' parameter. The display type, such as the font type and its size, color and angle, will be passed in the corresponding 'font', 'font\_size', 'clr' and 'angle' parameters. We will also make our object hidden in the list of objects for the user and not selectable with the mouse, using the 'selection' and 'hidden' parameters. The z\_order parameter will be responsible for the priority order of mouse clicks.

Let's start the method description by resetting the error variable so that it is possible to correctly control the result of creating an object in the future through the [ResetLastError()](https://www.mql5.com/en/docs/common/resetlasterror) predefined terminal function. Creating the [OBJ\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_arrow_left_price) type object creation result is handled via the 'if' logical operator while calling the [ObjectCreate()](https://www.mql5.com/en/docs/objects/objectcreate) function in it, as shown below. If the object is not created, inform the user about that in the EA log and interrupt the method execution via the return statement as usual.

```
//--- reset the error value
   ResetLastError();
//--- create a text label
   if(!ObjectCreate(chart_ID,name,OBJ_LABEL,sub_window,0,0))
     {
      Print(__FUNCTION__,
            ": failed to create the text label! Error code = ",GetLastError());
      return(false);
     }
```

If the object was created successfully, initialize the property fields of the object via the [ObjectSetInteger()](https://www.mql5.com/en/docs/objects/objectsetinteger), [ObjectSetString()](https://www.mql5.com/en/docs/objects/objectsetstring) and [ObjectSetDouble()](https://www.mql5.com/en/docs/objects/objectsetdouble) predefined terminal functions to give it the required appearance. Use the [ObjectSetInteger()](https://www.mql5.com/en/docs/objects/objectsetinteger) function to set the values of the corresponding coordinates, object anchor angle, font size, object anchor method, color, display mode, as well as the properties related to the object visibility to the user. Set the font angle values using the [ObjectSetDouble()](https://www.mql5.com/en/docs/objects/objectsetdouble) function, while the [ObjectSetString()](https://www.mql5.com/en/docs/objects/objectsetstring) function is used to define the content of the passed text and the font type for display. The complete implementation of the method body is presented below.

```
//--- reset the error value
   ResetLastError();
//--- create a text label
   if(!ObjectCreate(chart_ID,name,OBJ_LABEL,sub_window,0,0))
     {
      Print(__FUNCTION__,
            ": failed to create the text label! Error code = ",GetLastError());
      return(false);
     }
//--- set label coordinates
   ObjectSetInteger(chart_ID,name,OBJPROP_XDISTANCE,x);
   ObjectSetInteger(chart_ID,name,OBJPROP_YDISTANCE,y);
//--- set the chart's corner, relative to which point coordinates are defined
   ObjectSetInteger(chart_ID,name,OBJPROP_CORNER,corner);
//--- set the text
   ObjectSetString(chart_ID,name,OBJPROP_TEXT,text);
//--- set the text font
   ObjectSetString(chart_ID,name,OBJPROP_FONT,font);
//--- set font size
   ObjectSetInteger(chart_ID,name,OBJPROP_FONTSIZE,font_size);
//--- set the text angle
   ObjectSetDouble(chart_ID,name,OBJPROP_ANGLE,angle);
//--- set anchor type
   ObjectSetInteger(chart_ID,name,OBJPROP_ANCHOR,anchor);
//--- set the color
   ObjectSetInteger(chart_ID,name,OBJPROP_COLOR,clr);
//--- display in the foreground (false) or background (true)
   ObjectSetInteger(chart_ID,name,OBJPROP_BACK,back);
//--- enable (true) or disable (false) the mode of moving the label by mouse
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTABLE,selection);
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTED,selection);
//--- hide (true) or display (false) graphical object name in the object list
   ObjectSetInteger(chart_ID,name,OBJPROP_HIDDEN,hidden);
//--- set the priority for receiving the event of a mouse click on the chart
   ObjectSetInteger(chart_ID,name,OBJPROP_ZORDER,z_order);
//--- successful execution
   return(true);
```

Declare the RectLabelCreate() method with the following signature as object creation parameters:

```
bool RectLabelCreate(const long             chart_ID=0,               // chart ID
                     const string           name="RectLabel",         // label name
                     const int              sub_window=0,             // subwindow number
                     const int              x=19,                     // X coordinate
                     const int              y=19,                     // Y coordinate
                     const int              width=150,                // width
                     const int              height=20,                // height
                     const color            back_clr=C'236,233,216',  // background color
                     const ENUM_BORDER_TYPE border=BORDER_SUNKEN,     // border type
                     const ENUM_BASE_CORNER corner=CORNER_LEFT_UPPER, // chart corner for anchoring
                     const color            clr=clrRed,               // flat border color (Flat)
                     const ENUM_LINE_STYLE  style=STYLE_SOLID,        // flat border style
                     const int              line_width=1,             // flat border width
                     const bool             back=true,                // 'true' in the background
                     const bool             selection=false,          // select to move
                     const bool             hidden=true,              // hidden in the list of objects
                     const long             z_order=0)                // priority for clicking with a mouse
```

The parameters of the RectLabelCreate() method are very similar to the parameters of the previously declared LabelCreate() method, except for additional settings for the rectangular label border, which will serve as a background for displaying the data of the previous method object. Additional parameters for configuring the object border: 'border' - border type defined by the [ENUM\_BORDER\_TYPE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_border_type) enumeration with the default value of [BORDER\_SUNKEN](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_border_type), 'style' - border style defined by the [ENUM\_LINE\_STYLE](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles#enum_line_style) enumeration using the default value of [STYLE\_SOLID](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles#enum_line_style) and line\_width - border line width as an integer value.

The definition of the method body will look similar to the previous one and, similarly, will consist of two global sections: the creation of the object and the definition of its properties through the corresponding predefined terminal methods, as shown below.

```
//--- reset the error value
   ResetLastError();                                                    // reset error
//--- create a rectangle label
   if(ObjectCreate(chart_ID,name,OBJ_RECTANGLE_LABEL,sub_window,0,0))   // create object
     {
      //--- set label coordinates
      ObjectSetInteger(chart_ID,name,OBJPROP_XDISTANCE,x);              // assign x coordinate
      ObjectSetInteger(chart_ID,name,OBJPROP_YDISTANCE,y);              // assign y coordinate
      //--- set label size
      ObjectSetInteger(chart_ID,name,OBJPROP_XSIZE,width);              // width
      ObjectSetInteger(chart_ID,name,OBJPROP_YSIZE,height);             // height
      //--- set the background color
      ObjectSetInteger(chart_ID,name,OBJPROP_BGCOLOR,back_clr);         // background color
      //--- set border type
      ObjectSetInteger(chart_ID,name,OBJPROP_BORDER_TYPE,border);       // border type
      //--- set the chart corner, relative to which point coordinates are defined
      ObjectSetInteger(chart_ID,name,OBJPROP_CORNER,corner);            // anchor corner
      //--- set flat border color (in Flat mode)
      ObjectSetInteger(chart_ID,name,OBJPROP_COLOR,clr);                // frame
      //--- set flat border line style
      ObjectSetInteger(chart_ID,name,OBJPROP_STYLE,style);              // style
      //--- set flat border width
      ObjectSetInteger(chart_ID,name,OBJPROP_WIDTH,line_width);         // width
      //--- display in the foreground (false) or background (true)
      ObjectSetInteger(chart_ID,name,OBJPROP_BACK,back);                // default is background
      //--- enable (true) or disable (false) the mode of moving the label by mouse
      ObjectSetInteger(chart_ID,name,OBJPROP_SELECTABLE,selection);     // is it possible to select
      ObjectSetInteger(chart_ID,name,OBJPROP_SELECTED,selection);       //
      //--- hide (true) or display (false) graphical object name in the object list
      ObjectSetInteger(chart_ID,name,OBJPROP_HIDDEN,hidden);            // is it visible in the list
      //--- set the priority for receiving the event of a mouse click on the chart
      ObjectSetInteger(chart_ID,name,OBJPROP_ZORDER,z_order);           // no events
      //--- successful execution
     }
   return(true);
```

Now that all the auxiliary methods have been described, let's define the body of the main method that will draw the entire panel - paintPanel(). The inputs will contain the fields required to display the full deal information to the user as shown below.

```
void paintPanel(long handlE,
                ulong tickeT,
                double stop_losS,
                double take_profiT,
                double opeN,
                double closE,
                datetime timE,
                datetime time_closE,
                int magiC,
                string commenT,
                string externalIDD,
                double volumE,
                double commissioN,
                ENUM_DEAL_REASON reasoN,
                double swaP,
                double profiT,
                double feE,
                string symboL,
                int digitS
               )
```

As in the previous methods, the first parameter will be responsible for defining the handle of the chart, on which all objects associated with the information panel will be created. All other parameters will repeat the fields of the historical deal object.

We will start the implementation of the method for drawing the deal data panel by defining the variables for storing the size of the panel, as well as the coordinates for anchoring the columns with the name of the displayed information and the previously obtained values, as shown below.

```
int height=20, max_height =0, max_width = 0;	// column height and max values for indent
int x_column[2] = {10, 130};			// columns X coordinates
int y_column[17];				// Y coordinates
```

The 'height' variable will store a static value of 20 as the height of each column to ensure that each row is drawn evenly, and the max\_height and max\_width values will store the maximum values of each column to ensure even drawing. The required coordinates along the X and Y axes will be stored in the x\_column\[\] and y\_column\[\] arrays, respectively.

Now we need to declare two arrays that will store the row values for displaying the header column and the value column. We will declare the header column via the [string](https://www.mql5.com/en/docs/basis/types/stringconst) type data array, as shown in the following code.

```
   string column_1[17] =
     {
      "Symbol",
      "Position ID",
      "External ID",
      "Magic",
      "Comment",
      "Reason",
      "Open",
      "Close",
      "Time open",
      "Time close",
      "Stop loss",
      "Take profit",
      "Volume",
      "Commission",
      "Swap",
      "Profit",
      "Fee"
     };
```

All array values are declared and initialized statically, since the panel will not change, and the data will always be displayed in the same sequence. This should be convenient in terms of getting used to viewing information on different deals. It is possible to implement functionality that would allow excluding data from the panel that does not contain values or is equal to zero, but this would be inconvenient for the eyes when quickly searching for information. It is still more familiar to search for information in a well-known display pattern than to look at the column values every time.

In the same data sequence, declare a second array, which will already contain the values of the columns declared in the array above. Describe the array declaration as follows:

```
   string column_2[17] =
     {
      symboL,
      IntegerToString(tickeT),
      externalIDD,
      IntegerToString(magiC),
      commenT,
      EnumToString(reasoN),
      DoubleToString(opeN,digitS),
      DoubleToString(closE,digitS),
      TimeToString(timE),
      TimeToString(time_closE),
      DoubleToString(stop_losS,digitS),
      DoubleToString(take_profiT,digitS),
      DoubleToString(volumE,2),
      DoubleToString(commissioN,2),
      DoubleToString(swaP,2),
      DoubleToString(profiT,2),
      DoubleToString(feE,2)
     };
```

The array will be declared locally, at the method level, with the fields being initialized at the same time, directly from the method parameters, using the corresponding predefined terminal functions.

Now that we have declared the containers with the required data, we need to calculate the anchor values of each cell coordinates, while considering searching for the maximum value in each of them. We can implement this with the following code:

```
   int count_rows = 1;
   for(int i=0; i<ArraySize(y_column); i++)
     {
      y_column[i] = height * count_rows;
      max_height = y_column[i];
      count_rows++;

      int width_curr = StringLen(column_2[i]);

      if(width_curr>max_width)
        {
         max_width = width_curr;
        }
     }

   max_width = max_width*10;
   max_width += x_column[1];
   max_width += x_column[0];
```

Here we find the anchor coordinates of each object by looping over the number of rows, multiplying them by a fixed Y-height value for each. We also check each value against the maximum width value to get the X coordinate.

Once we have all the values with their coordinates, we can start drawing the information using the previously declared custom LabelCreate() method, which we will call cyclically, according to the number of our rows to display, as shown below.

```
   color back_Color = clrWhiteSmoke;
   color font_Color = clrBlueViolet;

   for(int i=0; i<ArraySize(column_1); i++)
     {
      //--- draw 1
      string name_1 = column_1[i]+"_1_"+IntegerToString(tickeT);
      LabelCreate(handlE,name_1,0,x_column[0],y_column[i],CORNER_LEFT_UPPER,column_1[i],"Arial",10,font_Color,0,ANCHOR_LEFT_UPPER,false);
      //--- draw 2
      string name_2 = column_1[i]+"_2_"+IntegerToString(tickeT);
      LabelCreate(handlE,name_2,0,x_column[1],y_column[i],CORNER_LEFT_UPPER,column_2[i],"Arial",10,font_Color,0,ANCHOR_LEFT_UPPER,false);
     }
```

At the end of the method, we just need to draw the background using the previously declared and described custom method RectLabelCreate() to these values and update the displayed chart, as shown below.

```
//--- draw the background
   RectLabelCreate(handlE,"RectLabel",0,1,height,max_width,max_height,back_Color);

   ChartRedraw(handlE);
```

This completes the description of all methods. The project is ready for assembly and use.

As a result, the chart file will look as shown in Figure 3 after using the script.

![Figure 3. The result of the script operation with the deal data displayed](https://c.mql5.com/2/80/PERIOD_M1531030020.png)

Figure 3. The result of the script operation with the deal data displayed

As we can see, all information on the deal is presented in a generalized form on a single chart, which makes the analysis and evaluation of the trading operations performed by the user more convenient. The script arranges such files into the appropriate folders, which also enables the user to find the required information on any trading operation in the account history at any time.

### Conclusion

With this article, we have completed writing a script for automated visualization of deals on a chart. Using this solution, you can significantly improve your trading by correcting possible errors when choosing an entry point, as well as increase the mathematical expectation of your entire strategy by choosing the right symbols and the expected price impulse location. Using the script will allow you to significantly save time on preparing chart files, which you can instead spend on analysis and searching for new trading ideas. The main thing to remember is that the market is constantly changing, and to ensure stable operation, you need to constantly stay in touch and monitor changes. This tool can help you with this. I wish you success in your work. Please leave your feedback in the comments.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14961](https://www.mql5.com/ru/articles/14961)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14961.zip "Download all attachments in the single ZIP archive")

[DealsPrintScreen.mq5](https://www.mql5.com/en/articles/download/14961/dealsprintscreen.mq5 "Download DealsPrintScreen.mq5")(104.52 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Visualizing deals on a chart (Part 1): Selecting a period for analysis](https://www.mql5.com/en/articles/14903)
- [Risk manager for algorithmic trading](https://www.mql5.com/en/articles/14634)
- [Risk manager for manual trading](https://www.mql5.com/en/articles/14340)
- [Balancing risk when trading multiple instruments simultaneously](https://www.mql5.com/en/articles/14163)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/476817)**
(2)


![Alexander P.](https://c.mql5.com/avatar/avatar_na2.png)

**[Alexander P.](https://www.mql5.com/en/users/alepie)**
\|
1 Dec 2024 at 15:17

A great idea and first-class explanation to understand. Thank you.


![Aleksandr Seredin](https://c.mql5.com/avatar/2022/4/62543FFE-A51A.jpg)

**[Aleksandr Seredin](https://www.mql5.com/en/users/al.s.capital)**
\|
1 Dec 2024 at 17:01

**Alexander Piechotta [#](https://www.mql5.com/de/forum/477293#comment_55269043):**

A great idea and first-class explanation to understand. Thank you very much.

Thank you very much for your comment. I would be delighted if this helps you in your work. Servus! :)

![Chemical reaction optimization (CRO) algorithm (Part I): Process chemistry in optimization](https://c.mql5.com/2/81/Algorithm_for_optimization_by_chemical_reactions__LOGO___2.png)[Chemical reaction optimization (CRO) algorithm (Part I): Process chemistry in optimization](https://www.mql5.com/en/articles/15041)

In the first part of this article, we will dive into the world of chemical reactions and discover a new approach to optimization! Chemical reaction optimization (CRO) uses principles derived from the laws of thermodynamics to achieve efficient results. We will reveal the secrets of decomposition, synthesis and other chemical processes that became the basis of this innovative method.

![Developing a Replay System (Part 52): Things Get Complicated (IV)](https://c.mql5.com/2/80/Desenvolvendo_um_sistema_de_Replay_Parte_52___LOGO.png)[Developing a Replay System (Part 52): Things Get Complicated (IV)](https://www.mql5.com/en/articles/11925)

In this article, we will change the mouse pointer to enable the interaction with the control indicator to ensure reliable and stable operation.

![Connexus Observer (Part 8): Adding a Request Observer](https://c.mql5.com/2/101/http60x60__1.png)[Connexus Observer (Part 8): Adding a Request Observer](https://www.mql5.com/en/articles/16377)

In this final installment of our Connexus library series, we explored the implementation of the Observer pattern, as well as essential refactorings to file paths and method names. This series covered the entire development of Connexus, designed to simplify HTTP communication in complex applications.

![MQL5 Wizard Techniques you should know (Part 48): Bill Williams Alligator](https://c.mql5.com/2/101/MQL5_Wizard_Techniques_you_should_know_Part_48__LOGO.png)[MQL5 Wizard Techniques you should know (Part 48): Bill Williams Alligator](https://www.mql5.com/en/articles/16329)

The Alligator Indicator, which was the brain child of Bill Williams, is a versatile trend identification indicator that yields clear signals and is often combined with other indicators. The MQL5 wizard classes and assembly allow us to test a variety of signals on a pattern basis, and so we consider this indicator as well.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/14961&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083119208486671718)

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