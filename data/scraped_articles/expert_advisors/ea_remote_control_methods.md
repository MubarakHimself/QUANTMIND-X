---
title: EA remote control methods
url: https://www.mql5.com/en/articles/5166
categories: Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T14:13:05.449562
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/5166&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083414027926772543)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/5166#para1)
- [1\. Setting a task](https://www.mql5.com/en/articles/5166#para2)
- [2\. Analyzing the template file](https://www.mql5.com/en/articles/5166#para3)
- [3\. Defining commands](https://www.mql5.com/en/articles/5166#para4)
- [4\. Creating the master EA](https://www.mql5.com/en/articles/5166#para5)

  - [4.1. Command manager](https://www.mql5.com/en/articles/5166#para51)
  - [4.2. Data on launched EAs](https://www.mql5.com/en/articles/5166#para52)
  - [4.3. EA status change function](https://www.mql5.com/en/articles/5166#para53)
  - [4.4. Functions for deleting orders and closing positions](https://www.mql5.com/en/articles/5166#para54)

- [Conclusion](https://www.mql5.com/en/articles/5166#para6)

### Introduction

Various automated systems for trading in financial markets have become quite common in our digital age. The main advantages of electronic Expert Advisors (EAs) are considered to be flawless algorithm execution and working 24 hours a day. The virtual hosting allows using EAs autonomously around the clock.

Unfortunately, not all EAs work equally well in any market situations. In these cases, traders often need to enable and disable them manually. This is easy when a user has direct access to the terminal. But what if you do not have quick access to the terminal with a working EA? In such cases, it would be good to be able to remotely control the EA operation. Let's consider one of the possible methods of EA remote control in the terminal.

### 1\. Setting a task

At first glance, the task seems clear: you need to create a program sending instructions to EAs upon receipt of external commands. But after considering the issue more deeply, we can see that MetaTrader 5 features no possibility to directly impact the work of a third-party EA by programming means. Each EA works in a separate flow, and it is impossible to detect a running EA on an opened chart. The user [fxsaber](https://www.mql5.com/en/users/fxsaber) offered a solution in the " [Expert - library for MetaTrader 5](https://www.mql5.com/en/code/19003)".

In this code, the author suggests using the ability to save templates. At first glance, saving templates does not affect the operation of programs running on a chart. This is true, but when saving the chart template, all graphical objects applied to the chart, as well as all launched applications along with their parameters are saved to the file. Subsequent application of the saved template to the chart restores all graphical objects and programs together with the saved parameters.

Another side of this impact is that all objects and programs that existed before loading the template are removed from the chart. In other words, if an EA is launched on the chart, it is removed in case the template contains no such EA, and vice versa. Thus, uploading and deleting an EA from the chart is reduced to editing a template file.

Of course, it is possible to prepare the necessary templates in advance and upload the necessary ones without the need to edit them. However, in this case, the number of required templates is growing twice as fast as the number of EAs used. Besides, if different EA settings are used for various symbols, adding each working symbol increases the number of templates. The preparation of templates becomes a routine work, and it becomes necessary not to get confused when applying them.

Editing templates also has its own nuances. By default, the templates are saved to "data\_folder\\Profiles\\Templates\\". But MQL5 allows working only with "sandbox" files. [fxsaber](https://www.mql5.com/en/users/fxsaber) again came out with a solution: he suggested adding the path to the sandbox when specifying the template file name. As a result, it is possible to access the template file without applying third-party libraries.

Let's thank [fxsaber](https://www.mql5.com/en/users/fxsaber) for his keen mind and extraordinary thinking.

After defining EA management methods, let's work on program-user communication model. The world of today forces people to remain mobile and carry their smartphones with them at all times. The MetaTrader 5 platform features versions both for iPhone and Android. After connecting the terminal to their accounts, users are able to analyze price movements on the chart and trade manually. Currently, mobile terminals do not allow using EAs and third-party indicators. Desktop terminals and virtual hosting are the only options if you want to use EAs.

The mobile terminal connected to an account allows viewing the entire account, but there are no direct connection channels between the mobile and desktop terminals. The only thing we can influence is placing and deleting orders. Placed orders are immediately displayed on the account and can be tracked by an EA running in the desktop terminal. Thus, together with placing orders, we can transfer control commands to our master EA. We only need to define the list of commands and the code to be used to pass them. The following sections will deal with these issues.

### 2\. Analyzing the template file

First, let's view the template file structure. The example below provides the EURUSD M30 chart template with the applied ExpertMACD EA and MACD indicator.

```
<chart>
id=130874249669663027
symbol=EURUSD
period_type=0
period_size=30
digits=5
.....
.....
windows_total=2

<expert>
name=ExpertMACD
path=Experts\Advisors\ExpertMACD.ex5
expertmode=0
<inputs>
Inp_Expert_Title=ExpertMACD
Inp_Signal_MACD_PeriodFast=12
Inp_Signal_MACD_PeriodSlow=24
Inp_Signal_MACD_PeriodSignal=9
Inp_Signal_MACD_TakeProfit=50
Inp_Signal_MACD_StopLoss=20
</inputs>
</expert>

<window>
height=162.766545
objects=103

<indicator>
name=Main
........
</indicator>
<object>
.......
</object>

<object>
........
</object>
........
........
<object>
........
</object>

</window>

<window>
height=50.000000
objects=0

<indicator>
name=MACD
path=
apply=1
show_data=1
scale_inherit=0
scale_line=0
scale_line_percent=50
scale_line_value=0.000000
scale_fix_min=0
scale_fix_min_val=-0.001895
scale_fix_max=0
scale_fix_max_val=0.001374
expertmode=0
fixed_height=-1

<graph>
name=
draw=2
style=0
width=1
color=12632256
</graph>

<graph>
name=
draw=1
style=2
width=1
color=255
</graph>
fast_ema=12
slow_ema=24
macd_sma=9
</indicator>
</window>
</chart>
```

As can be seen from the code, the information in the template file is structured and divided by tags. The file starts with the <chart> tag containing the main chart data, including chart ID, symbol and timeframe. The info we are interested in is located between the <expert> and </expert> tags.

The beginning of the block provides data on the EA: its short name displayed on the chart and the file path. The expertmode flag comes next. Its status indicates if the EA is allowed to trade. The flag is followed by EA parameters highlighted by the <inputs> </inputs> tags. Then we can find data on the chart subwindows. Each subwindow is highlighted by the <window> and </window> tags containing descriptions of launched indicators (<indicator> ... </indicator>) and applied graphical objects (<object> ... </object>).

Also, keep in mind that if an EA and/or an indicator launched on the chart create graphical objects, it would be wise to remove them from the template. Otherwise, the objects are applied to the chart from the template, and when launching such an EA and/or indicator, they re-create the same objects.

This may cause uncontrolled creation of a large number of identical objects cluttering up the chart, as well as excessive consumption of computing resources. In worst-case scenario, the necessary program shows the object creation error and is unloaded from the chart.

Programmatically created objects are usually hidden in the graphical object list by assigning the hidden property to them to protect them against user modification. The 'hidden' flag is provided to set this property in the template. It is equal to 1 for hidden objects.

Thus, to enable/disable the EA, we simply need to rewrite the template file changing the expertmode flag value and deleting hidden objects along the way. Applying the new template re-launches the EA with the necessary property.

### 3\. Defining commands

After defining the main principles of the developed EA, it is time to create the communication system. Previously, we have already decided to use orders for sending commands. But how do we send commands using orders without damaging the financial result? For these purposes, we will use pending orders, which will be deleted by the EA after receiving the command.

When placing a pending order, we should be sure that it is not activated within its working time interval. The solution here is pretty obvious - we need to place an order at a sufficient distance from the current price.

To my surprise, when working with a mobile terminal, I found it impossible to leave comments on the opened orders. At the same time, users can see comments to orders created in the desktop terminal. Thus, we can arrange one-way communication from the master EA to a user. However, users will not be able to issue commands this way.

Thus, we will be able to place pending orders to be read by the EA, but we cannot leave comments in them. In this case, we can use the price and order symbol for passing the command. Here we should define the price for coding the command. It should be located at a sufficient distance from the current price so that the placed order is not activated under any circumstances.

I believe, these are the prices close to zero. In theory, the probability of any symbol approaching zero is negligibly small. Thus, if we set the order price to 1-5 ticks, it will most probably never be activated. Of course, in this case, we are limited to sell stop orders, since the use of stop orders does not affect the free margin.

Keep in mind that we are able to read order comments in the mobile application, i.e. our master EA is able to send us the data in this way.

Summarizing the above, the following command codes are proposed:

| # | Price | Volume | Command |
| --- | --- | --- | --- |
| 1 | 1 tick | any | Request EAs' status.<br> The master EA sets pending orders by symbols with EAs, while the EA name and permission to trade are specified in the order comment. |
| 2 | 2 ticks | any | Stopping trading by the EA.<br> If there is no order comment, all EAs are stopped. If the command has been issued as a result of order modification after the first command, a certain EA on a certain symbol is stopped. |
| 3 | 3 ticks | any | Launching trading using the EA.<br> Operation principles are the same as in the command 2. |
| 4 | 4 ticks | 1 min. BUY lot<br> 2 min. SELL lot<br> 3 min. lot ALL | Remove pending orders. |
| 5 | 5 ticks | 1 min. BUY lot<br> 2 min. SELL lot<br> 3 min. lot ALL | Close positions. |

### 4\. Creating the master EA

Now that we have full understanding of the operation methods and the communication channel, let's start developing the EA. Our EA code will apply methods from the " [Expert - library for MetaTrader 5](https://www.mql5.com/en/code/19003)" with a minor change - all library methods will be made public for more ease of use.

First, let's assign mnemonic names to the applied tags for more convenience. Some of them are already declared in the applied library.

```
#define FILENAME (__FILE__ + ".tpl")
#define PATH "\\Files\\"
#define STRING_END "\r\n"
#define EXPERT_BEGIN ("<expert>" + STRING_END)
#define EXPERT_END ("</expert>" + STRING_END)
#define EXPERT_INPUT_BEGIN ("<inputs>" + STRING_END)
#define EXPERT_INPUT_END ("</inputs>" + STRING_END)
#define EXPERT_CHART_BEGIN ("<chart>" + STRING_END)
#define EXPERT_NAME "name="
#define EXPERT_PATH "path="
#define EXPERT_STOPLEVEL "stops_color="
```

Additional ones are declared in our EA code.

```
#define OBJECT_BEGIN             ("<object>" + STRING_END)
#define OBJECT_END               ("</object>" + STRING_END)
#define OBJECTS_NUMBER           ("objects=")
#define OBJECT_HIDDEN            ("hidden=1")
#define EXPERT_EXPERTMODE        ("expertmode=")
```

It should also be noted that [fxsaber](https://www.mql5.com/en/users/fxsaber) took care of the compatibility of his library with other code and canceled mnemonic names at the end of the library. This approach eliminates errors of reassigning the same name to another macro. Although that does not allow the use of such declarations outside the library. In order not to repeat similar declarations of macros in the EA code, lets comment out the #undef directive in the library code.

```
//#undef EXPERT_STOPLEVEL
//#undef EXPERT_PATH
//#undef EXPERT_NAME
//#undef EXPERT_CHART_BEGIN
//#undef EXPERT_INPUT_END
//#undef EXPERT_INPUT_BEGIN
//#undef EXPERT_END
//#undef EXPERT_BEGIN
//#undef STRING_END
//#undef PATH
//#undef FILENAME
```

Then declare two external variables of our master EA: the lifetime of info orders in minutes and the magic number for their identification.

```
sinput int      Expirations =  5;
sinput ulong    Magic       =  88888;
```

Add the library mentioned above and the library of trading operations to the EA code.

```
#include <fxsaber\Expert.mqh>
#include <Trade\Trade.mqh>
```

In the global variables block, declare an instance of the trading operations class and the variables for storing the working chart ID and the last command order ticket.

```
CTrade   *Trade;
ulong     chart;
ulong     last_command;
```

In the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) function, initialize the global variables.

```
int OnInit()
  {
//---
   Trade =  new CTrade();
   if(CheckPointer(Trade)==POINTER_INVALID)
      return INIT_FAILED;
   Trade.SetDeviationInPoints(0);
   Trade.SetExpertMagicNumber(Magic);
   Trade.SetMarginMode();
   Trade.SetTypeFillingBySymbol(_Symbol);
//---
   chart=ChartID();
   last_command=0;
//---
   return(INIT_SUCCEEDED);
  }
```

Do not forget to remove the trading operations class instance in the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) function.

```
void OnDeinit(const int reason)
  {
//---
   if(CheckPointer(Trade)!=POINTER_INVALID)
      delete Trade;
  }
```

#### 4.1. Command manager

The launch of the master EA functionality is performed from the [OnTradeTransaction](https://www.mql5.com/en/docs/event_handlers/ontradetransaction) transaction. Keep in mind that calling the function may be processed multiple times for one order. Therefore, before running the program code, we will conduct a series of checks.

We will check the availability of the order ticket for the event being processed. We will also check the order status in order not to handle the event of deleting orders. Besides, we should check the order magic number to avoid processing EA orders and verify the order type, since our commands arrive only with sell stop orders. In addition, we should check the type of a trading operation, as it should be adding or modifying an order.

After successful completion of all the checks, launch the command manager arranged in the CheckCommand function.

```
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
  {
//---
   if(trans.order>0 && trans.order_state!=ORDER_STATE_REQUEST_CANCEL && request.magic==0 &&
      trans.order_type==ORDER_TYPE_SELL_STOP &&
      (trans.type==TRADE_TRANSACTION_ORDER_ADD || trans.type==TRADE_TRANSACTION_ORDER_UPDATE))
      CheckCommand(trans,request);
  }
```

When launching the CheckCommand function, it receives trading operation and request structures in its parameters. First, let's check whether this request has been processed before. If the command has already been processed, exit the function.

```
void CheckCommand(const MqlTradeTransaction &trans,
                  const MqlTradeRequest &request)
  {
   if(last_command==trans.order)
      return;
```

If not, decode the order price into the command.

```
   double tick=SymbolInfoDouble(trans.symbol,SYMBOL_TRADE_TICK_SIZE);
   uint command=(uint)NormalizeDouble(trans.price/tick,0);
```

Then, use the [switch](https://www.mql5.com/en/docs/basis/operators/switch) operator to call the function that corresponds to the incoming command.

```
   switch(command)
     {
      case 1:
        if(StringLen(request.comment)>0 || trans.type==TRADE_TRANSACTION_ORDER_UPDATE)
           return;
        if(trans.order<=0 || (OrderSelect(trans.order) && StringLen(OrderGetString(ORDER_COMMENT))>0))
           return;
        GetExpertsInfo();
        break;
      case 2:
        ChangeExpertsMode(trans,request,false);
        break;
      case 3:
        ChangeExpertsMode(trans,request,true);
        break;
      case 4:
        DeleteOrders(trans);
        break;
      case 5:
        ClosePosition(trans);
        break;
      default:
        return;
     }
```

In conclusion, save the ticket of the last command order and delete the order from the account.

```
   last_command=trans.order;
   Trade.OrderDelete(last_command);
  }
```

The full code of all the functions is provided in the attachment.

#### 4.2. Data on launched EA

The GetExpertsInfo function is responsible for obtaining data on all EAs launched in the terminal. As decided earlier, this function places info stop orders for the symbol displaying on what symbol the EA is launched, while the EA name and its status are displayed in the order comments.

In the beginning of the function, delete previously set orders by calling the DeleteOrdersByMagic function.

```
void GetExpertsInfo(void)
  {
   DeleteOrdersByMagic(Magic);
```

Further on, arrange the loop for iterating over all active charts in the terminal. The loop begins with the check if the analyzed chart is a working chart of our master EA. If yes, move on to the next chart.

```
   long i_chart=ChartFirst();
   while(i_chart>=0 && !IsStopped())
     {
      if(i_chart==0 || i_chart==chart)
        {
         i_chart=ChartNext(i_chart);
         continue;
        }
```

At the next stage, upload the chart template with a preliminary check of the EA presence on the chart. If the EA is absent or the template is not uploaded, move on to the next chart.

```
      string temp=EXPERT::TemplateToString(i_chart,true);
      if(temp==NULL)
        {
         i_chart=ChartNext(i_chart);
         continue;
        }
```

Next, find the EA block in the template. If the block is not found, move on to the next chart.

```
      temp=EXPERT::StringBetween(temp,EXPERT_BEGIN,EXPERT_END);
      if(temp==NULL)
        {
         i_chart=ChartNext(i_chart);
         continue;
        }
```

After that, retrieve the EA name and its status. They are used to form the comment of the future info order. If the EA is allowed to trade, "T" letter is put before its name, otherwise, "S" letter is used.

```
      string name =  EXPERT::StringBetween(temp,EXPERT_NAME,STRING_END);
      bool state  =  (bool)StringToInteger(EXPERT::StringBetween(temp,EXPERT_EXPERTMODE,STRING_END));
      string comment =  (state ? "T " : "S ")+name;
```

At the end of the loop, determine the chart symbol and the info order validity period. Send the order and move on to the next chart.

```
      string symbol=ChartSymbol(i_chart);
      ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC;
      datetime expir=0;
      if(Expirations>0)
        {
         expir=TimeCurrent()+Expirations*60;
         type=ORDER_TIME_SPECIFIED;
        }
      Trade.SellStop(SymbolInfoDouble(symbol,SYMBOL_VOLUME_MIN),SymbolInfoDouble(symbol,SYMBOL_TRADE_TICK_SIZE),symbol,0,0,type,expir,comment);
      i_chart=ChartNext(i_chart);
     }
  }
```

In the mentioned function, we have used the methods from the [fxsaber's](https://www.mql5.com/en/users/fxsaber) library — getting the chart template to a string variable and getting a substring between specified tags.

If necessary, check if the EA is present on the chart to obtain the template to a string variable for the specified chart. Then save the template of the specified chart and read the obtained template as the binary array. Convert the numeric array to the string and return the calling functions. If there are errors at any verification stage, the function returns NULL.

```
  static string TemplateToString( const long Chart_ID = 0, const bool CheckExpert = false )
  {
    short Data[];
    return(((!CheckExpert || EXPERT::Is(Chart_ID)) && ::ChartSaveTemplate((ulong)Chart_ID, PATH + FILENAME) && (::FileLoad(FILENAME, Data) > 0)) ?
           ::ShortArrayToString(Data) : NULL);
  }
```

To obtain a substring between the specified tags, first define positions of the substring beginning and end.

```
  static string StringBetween( string &Str, const string StrBegin, const string StrEnd = NULL )
  {
    int PosBegin = ::StringFind(Str, StrBegin);
    PosBegin = (PosBegin >= 0) ? PosBegin + ::StringLen(StrBegin) : 0;

    const int PosEnd = ::StringFind(Str, StrEnd, PosBegin);
```

Then cut the substring for returning and decrease the original string down to the raw remainder.

```
    const string Res = ::StringSubstr(Str, PosBegin, (PosEnd >= 0) ? PosEnd - PosBegin : -1);
    Str = (PosEnd >= 0) ? ::StringSubstr(Str, PosEnd + ::StringLen(StrEnd)) : NULL;

    if (Str == "")
      Str = NULL;

    return(Res);
  }
```

Find the entire code of all the functions and classes in the attachment.

#### 4.3. EA status change function

EAs states are changed in the ChangeExpertsMode function. In its parameters, the specified function gets trading operation and request structures, as well as a new status for placing the EA.

```
void ChangeExpertsMode(const MqlTradeTransaction &trans,
                       const MqlTradeRequest &request,
                       bool  ExpertMode)
  {
   string comment=request.comment;
   if(StringLen(comment)<=0 && OrderSelect(trans.order))
      comment=OrderGetString(ORDER_COMMENT);
   string exp_name=(StringLen(comment)>2 ? StringSubstr(comment,2) : NULL);
```

Then arrange the loop for iterating over all charts and uploading templates as described above.

```
   long i_chart=ChartFirst();
   while(i_chart>=0 && !IsStopped())
     {
      if(i_chart==0 || i_chart==chart || (StringLen(exp_name)>0 && ChartSymbol()!=trans.symbol))
        {
         i_chart=ChartNext(i_chart);
         continue;
        }
      string temp=EXPERT::TemplateToString(i_chart,true);
      if(temp==NULL)
        {
         i_chart=ChartNext(i_chart);
         continue;
        }
```

At the next stage, check the availability of the necessary EA on the chart if necessary. If the EA launched on the chart does not correspond to the request, move on to the next chart. Also, check the current EA status. If it corresponds to the one set for placement, move on to the next chart.

```
      string NewTemplate   =  NULL;
      if(exp_name!=NULL)
        {
         NewTemplate=EXPERT::StringBetween2(temp,NULL,EXPERT_NAME);
         string name=EXPERT::StringBetween(temp,NULL,STRING_END);
         if(name!=exp_name)
           {
            i_chart=ChartNext(i_chart);
            continue;
           }
         NewTemplate+=name+STRING_END;
        }
//---
      NewTemplate+=EXPERT::StringBetween2(temp,NULL,EXPERT_EXPERTMODE);
      bool state  =  (bool)StringToInteger(EXPERT::StringBetween(temp,NULL,STRING_END));
      if(state==ExpertMode)
        {
         i_chart=ChartNext(i_chart);
         continue;
        }
```

After passing all necessary checks, create a new template with the EA's status specified. Delete hidden objects from the template and apply the new template to the chart. After performing all operations, move on to the next chart.

```
      NewTemplate+=IntegerToString(ExpertMode)+STRING_END+temp;
      NewTemplate=DeleteHiddenObjects(NewTemplate);
      EXPERT::TemplateApply(i_chart,NewTemplate,true);
//---
      i_chart=ChartNext(i_chart);
     }
```

After completing the chart iteration loop, launch the active EAs data collection function to demonstrate the EAs' new status to users.

```
   GetExpertsInfo();
  }
```

The full code of all the functions is provided in the attachment.

#### 4.4. Functions for deleting orders and closing positions

The functions for closing orders and open positions are based on the same algorithm differing only in targeted objects. Therefore, let's consider only one of them in this article. When calling the DeleteOrders function, trading operation structure is passed to it in the parameters. This structure is used to decode the order volume in the direction of closed orders (according to the table of commands from the section 3).

```
void DeleteOrders(const MqlTradeTransaction &trans)
  {
   int direct=(int)NormalizeDouble(trans.volume/SymbolInfoDouble(trans.symbol,SYMBOL_VOLUME_MIN),0);
```

The loop for iterating over all orders placed on the account is arranged afterwards. In this loop, the order type is checked for compliance with the incoming command. If there is a match, an order removal request is sent.

```
   for(int i=total-1;i>=0;i--)
     {
      ulong ticket=OrderGetTicket((uint)i);
      if(ticket<=0)
         continue;
//---
      switch((ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE))
        {
         case ORDER_TYPE_BUY_LIMIT:
         case ORDER_TYPE_BUY_STOP:
         case ORDER_TYPE_BUY_STOP_LIMIT:
           if(direct==2)
              continue;
           Trade.OrderDelete(ticket);
           break;
         case ORDER_TYPE_SELL_LIMIT:
         case ORDER_TYPE_SELL_STOP:
         case ORDER_TYPE_SELL_STOP_LIMIT:
           if(direct==1)
              continue;
           Trade.OrderDelete(ticket);
           break;
        }
     }
  }
```

The full code of all the functions is provided in the attachment.

YouTube

### Conclusion

This article proposed the method for the EA remote control in the MetaTrader 5 platform. This solution increases the mobility of traders using trading robots in their activity. The applied non-standard approach to the use of standard functions allows solving broader issues without using various dlls.

### References

[Expert - library for MetaTrader 5](https://www.mql5.com/en/code/19003)

### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | Expert.mqh | Class library | Expert - library for MetaTrader 5 |
| --- | --- | --- | --- |
| 2 | Master.mq5 | Expert Advisor | Master EA for managing other EAs launched in the terminal |
| --- | --- | --- | --- |
| 3 | Master.mqproj | Project file | Master EA project |
| --- | --- | --- | --- |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/5166](https://www.mql5.com/ru/articles/5166)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/5166.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/5166/mql5.zip "Download MQL5.zip")(88.19 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Neural Networks in Trading: Hybrid Graph Sequence Models (GSM++)](https://www.mql5.com/en/articles/17279)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Final Part)](https://www.mql5.com/en/articles/17241)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Chimera)](https://www.mql5.com/en/articles/17210)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model (Final Part)](https://www.mql5.com/en/articles/17157)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model](https://www.mql5.com/en/articles/17142)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Final Part)](https://www.mql5.com/en/articles/17104)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Hidformer)](https://www.mql5.com/en/articles/17069)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/291384)**
(15)


![sic20101](https://c.mql5.com/avatar/2017/11/5A0E02AB-C891.JPG)

**[sic20101](https://www.mql5.com/en/users/sic20101)**
\|
17 Sep 2018 at 19:59

There is only one difficulty in this, well or I'm wrong, I haven't checked, there is no Push notification in the [mobile terminal](https://www.metatrader5.com/en/mobile-trading "Mobile Trading Platform MetaTrader 5").

I just have never encountered such a problem as lack of internet. I just have a couple of neighbour's hacked networks in addition to my own, just in case.

But if there were notifications we would lose maximum opening of one deal (not always bad) as we don't know that internet is available and we need to do something there.

![Hao-Wei Lee](https://c.mql5.com/avatar/2015/7/55A63672-96CE.png)

**[Hao-Wei Lee](https://www.mql5.com/en/users/haowei)**
\|
14 Jan 2019 at 14:41

I think it's possible to create a EA to monitor a specific price of [pending order](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type "MQL5 documentation:"), for example price 1.00000 doing disable auto trading with #import "user32.dll"


![Keith Watford](https://c.mql5.com/avatar/avatar_na2.png)

**[Keith Watford](https://www.mql5.com/en/users/forexample)**
\|
16 Mar 2021 at 02:18

Comments that do not relate to this topic, have been moved to " [Off Topic Posts](https://www.mql5.com/en/forum/339471)".

![Anatolii Zverev](https://c.mql5.com/avatar/avatar_na2.png)

**[Anatolii Zverev](https://www.mql5.com/en/users/madeinussr77)**
\|
17 Sep 2022 at 07:33

Personally, I have implemented the management of my Expert Advisors through pending orders set in the area known to be unattainable for the price. For example, I create a pending order BUYLIMIT price 0.01 lot 0.01 - the Expert Advisor checks if there is such an order, and if there is, performs certain actions. By deleting this order in the [mobile terminal](https://www.metatrader5.com/en/mobile-trading "Mobile Trading Platform MetaTrader 5"), you will disable the execution of actions by the Expert Advisor.


![Dmitriy Gizlyk](https://c.mql5.com/avatar/2014/8/53E8CB77-1C48.png)

**[Dmitriy Gizlyk](https://www.mql5.com/en/users/dng)**
\|
18 Sep 2022 at 09:39

**Anatolii Zverev mobile terminal, you will disable the execution of actions by the Expert Advisor.**

The article also suggests managing Expert Advisors through [pending orders](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type "MQL5 documentation:"). In this case, after the execution of the command, the Expert Advisor deletes the order. And you will be able to control the execution.

![Movement continuation model - searching on the chart and execution statistics](https://c.mql5.com/2/34/wave_movie.png)[Movement continuation model - searching on the chart and execution statistics](https://www.mql5.com/en/articles/4222)

This article provides programmatic definition of one of the movement continuation models. The main idea is defining two waves — the main and the correction one. For extreme points, I apply fractals as well as "potential" fractals - extreme points that have not yet formed as fractals.

![100 best optimization passes (part 1). Developing optimization analyzer](https://c.mql5.com/2/34/TOP100passes.png)[100 best optimization passes (part 1). Developing optimization analyzer](https://www.mql5.com/en/articles/5214)

The article dwells on the development of an application for selecting the best optimization passes using several possible options. The application is able to sort out the optimization results by a variety of factors. Optimization passes are always written to a database, therefore you can always select new robot parameters without re-optimization. Besides, you are able to see all optimization passes on a single chart, calculate parametric VaR ratios and build the graph of the normal distribution of passes and trading results of a certain ratio set. Besides, the graphs of some calculated ratios are built dynamically beginning with the optimization start (or from a selected date to another selected date).

![Using limit orders instead of Take Profit without changing the EA's original code](https://c.mql5.com/2/34/Limit_TP.png)[Using limit orders instead of Take Profit without changing the EA's original code](https://www.mql5.com/en/articles/5206)

Using limit orders instead of conventional take profits has long been a topic of discussions on the forum. What is the advantage of this approach and how can it be implemented in your trading? In this article, I want to offer you my vision of this topic.

![Modeling time series using custom symbols according to specified distribution laws](https://c.mql5.com/2/33/Custom_series_modelling.png)[Modeling time series using custom symbols according to specified distribution laws](https://www.mql5.com/en/articles/4566)

The article provides an overview of the terminal's capabilities for creating and working with custom symbols, offers options for simulating a trading history using custom symbols, trend and various chart patterns.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/5166&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083414027926772543)

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