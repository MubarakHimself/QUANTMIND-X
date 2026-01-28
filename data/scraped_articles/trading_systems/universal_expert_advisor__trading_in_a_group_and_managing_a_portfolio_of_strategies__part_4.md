---
title: Universal Expert Advisor: Trading in a Group and Managing a Portfolio of Strategies (Part 4)
url: https://www.mql5.com/en/articles/2179
categories: Trading Systems, Integration
relevance_score: 6
scraped_at: 2026-01-23T11:50:31.489009
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=tkgqgcuarmdcdxgqruivvcfvovxschuh&ssn=1769158229827201494&ssn_dr=0&ssn_sr=0&fv_date=1769158229&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2179&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Universal%20Expert%20Advisor%3A%20Trading%20in%20a%20Group%20and%20Managing%20a%20Portfolio%20of%20Strategies%20(Part%204)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915822997527740&fz_uniq=5062758413508716629&sv=2552)

MetaTrader 5 / Examples


### Table of Contents

- [Introduction](https://www.mql5.com/en/articles/2179#intro)
- [CStrategyList Strategy Manager](https://www.mql5.com/en/articles/2179#c1)
- [Loading Strategies from an XML list. A Portfolio of Strategies](https://www.mql5.com/en/articles/2179#c2)
- [Managing Strategies Using a Custom Panel](https://www.mql5.com/en/articles/2179#c3)
- [Expert Advisors Trading in a Group](https://www.mql5.com/en/articles/2179#c4)
- [Analyzing Expert Advisor Operation in the Strategy Tester](https://www.mql5.com/en/articles/2179#c5)
- [Conclusion](https://www.mql5.com/en/articles/2179#exit)

### Introduction

We often need to create algorithms that should get along with one another, i.e. operation of an algorithm should not be influenced by the actions of other algorithms performed at the same time. This situation occurs when you need to combine several algorithms into one executable ex5 module. Despite its apparent simplicity, these tasks have some significant "pitfalls" — algorithmic features that must be considered when building the engine of trading strategies.

The CStrategy trading engine includes a set of algorithms that implement co-operation of two and more trading strategies. We will discuss them in detail in the fourth part of this series. Also we will create a _trading profile_ — a group of Expert Advisors trading simultaneously in order to diversify trading risks. The CStrategyList class — a container of CStrategy type strategies — belongs to the algorithms providing simultaneous operation of strategies. The class allows uploading the XML based presentation of the strategies, as well as create them dynamically using the corresponding method — _a factory of strategies_.

The attached video demonstrates the process of testing multiple strategies in the MetaTrader 5 Strategy Tester. All strategies based on the described trading engine have a default custom panel, which help you easily control separate strategies straight from the chart.

### CStrategyList Strategy Manager

The [second article](https://www.mql5.com/en/articles/2169) of the "Universal Expert Advisor" series described the CStrategy class and its main modules. Through the use of this class and its functionality implemented in the modules, every inherited strategy maintains a unified trading logic. However, organizing a trading process using robots is more than just a mere execution of trade requests. It is important to ensure their cooperation, including operation of several algorithms in one executable ex5 module.

The special **CStrategyList** class is used for this particular purpose. As you might guess from its name, this class provides a list of CStrategy type strategies, but its operation is somewhat more complicated than the operation of a usual data container. The module solves the following tasks:

- ensuring simultaneous operation of several trading strategies;
- delivering trade events to each strategy instance;
- creating strategy objects from the unified XML list of strategies (data deserializing);
- interaction with the custom panel used for EA configuration.

Here is the header of the CStrategyList class:

```
//+------------------------------------------------------------------+
//| Container class to manage strategies of the CStrategy type       |
//+------------------------------------------------------------------+
class CStrategyList
  {
private:
   CLog*       Log;                 // Logging
   CArrayObj   m_strategies;        // Strategies of the CStrategy type
   CLimits*    m_limits;
   void        ParseStrategies(CXmlElement* xmlStrategies, bool load_curr_symbol);
   void        ParseLimits(CXmlElement* xmlLimits);
   CStrBtn     StrButton;
public:
   CStrategyList(void);
   ~CStrategyList(void);
   void LoadStrategiesFromXML(string xml_name, bool load_curr_symbol);
   bool AddStrategy(CStrategy* strategy);
   int  Total();
   CStrategy* At(int index);
   void OnTick();
   void OnTimer();
   void OnBookEvent(string symbol);
   void OnDeinit(const int reason);
   void OnChartEvent(const int id,
                     const long &lparam,
                     const double &dparam,
                     const string &sparam);


  };
```

As you can see, most of the methods presented are handlers of trade events. They have contents of the same type. Let's analyze one of them, OnBookEvent:

```
//+------------------------------------------------------------------+
//| Sends OnBookEvent to all listed strategies                       |
//+------------------------------------------------------------------+
void CStrategyList::OnBookEvent(string symbol)
  {
   for(int i=0; i<m_strategies.Total(); i++)
     {
      CStrategy *strategy=m_strategies.At(i);
      strategy.OnBookEvent(symbol);
     }
  }
```

As seen from the class contents, it searches for CStrategy strategies in the list and calls an appropriate event in each of the strategies. The operation of other event methods is similar.

In addition to passing of events, CStrategyList performs special procedures loading strategies from the XML file. For more information about how it works, please read the next section.

### Loading Strategies from an XML list. A Portfolio of Strategies

If an executable ex5 module contains multiple trading algorithms, we need tools to generate a _portfolio_ _of strategies_. Suppose that two algorithms with different parameters trade in one executable module. How to configure these parameters? The simplest thing is to output the parameters of each strategy in the EA properties window. But what to do when many strategies are used, each of which has many parameters? In this case, the list of parameters with different modifiers, flags, strings and comments would be huge. That's what the parameters window of an Expert Advisor trading three strategies would look like:

![](https://c.mql5.com/2/21/5._Bollinger_Bands_strategy_MT5__1.png)

Fig. 1. The list of parameters of the EA trading three strategies

AN Expert Advisor can use even more strategies. In this case, the list of parameters could have unimaginable size. The second important aspect of portfolio trading is _creating strategies "on the flow"_. Suppose that we want to run the same strategy with two different sets of parameters. What should we do? Obviously, despite the different sets of parameters, these two strategies are one and the same strategy, although with different settings. Instead of creating each of the strategies manually, we can entrust this task to a separate class. The class can automatically create a strategy object and configure it properly.

Before creating a strategy "on the flow", it is necessary to provide its complete description. The description must contain the following details:

- the name of the strategy;
- a unique strategy ID or its Magic number;
- the symbol the strategy is running on;
- working timeframe of the strategy;
- a list of unique parameters of strategies (an individual list for each strategy).

Strategy description may contain other properties in addition to the above list. The best way to provide such a description is using XML. The language has been created as a special description tool. It allows to conveniently describe complex objects, so that an object like a trading strategy can be converted to a text XML document, and a text document can be converted to a strategy. For example, based on an XML document, the trading engine can create a strategy and properly configure its parameters. To work with this type of documents directly from MQL5, we should use a special [XML-Parser](https://www.mql5.com/en/code/712) library available in the Code Base.

Here is an example of the XML description of a portfolio that loads three MovingAverage strategies with different parameters:

```
<Global>
        <Strategies>
                <Strategy Name="MovingAverage" Magic="100" Timeframe="PERIOD_M1" Symbol="Si">
                        <TradeStateStart>Stop</TradeStateStart>
                        <Params>
                                <FastMA>1</FastMA>
                                <SlowMA>3</SlowMA>
                                <Shift>0</Shift>
                                <Method>MODE_SMA</Method>
                                <AppliedPrice>PRICE_CLOSE</AppliedPrice>
                        </Params>
                </Strategy>
                <Strategy Name="MovingAverage" Magic="101" Timeframe="PERIOD_M5" Symbol="SBRF">
                        <TradeStateStart>BuyOnly</TradeStateStart>
                        <Params>
                                <FastMA>15</FastMA>
                                <SlowMA>21</SlowMA>
                                <Shift>0</Shift>
                                <Method>MODE_SMA</Method>
                                <AppliedPrice>PRICE_CLOSE</AppliedPrice>
                        </Params>
                </Strategy>
                <Strategy Name="MovingAverage" Magic="102" Timeframe="PERIOD_M15" Symbol="GAZR">
                        <TradeStateStart>BuyAndSell</TradeStateStart>
                        <Params>
                                <FastMA>12</FastMA>
                                <SlowMA>45</SlowMA>
                                <Shift>1</Shift>
                                <Method>MODE_EMA</Method>
                                <AppliedPrice>PRICE_CLOSE</AppliedPrice>
                        </Params>
                </Strategy>
        </Strategies>
</Global>
```

Each of the strategies forms the <Strategy> unit. The following attributes are specified in it: Symbol, Timeframe, Magic and StrategyName. From the above example, we see that each of the three strategies has its own symbol, magic number and timeframe. In addition to these required parameters, other strategy properties are specified in the XML list. Section <TradeStateStart> specifies the trading mode at the time of the strategy launch. Section <Params> contains the parameters of the strategy.

At start up, the trading engine will attempt to load the trading strategies from the above XML file. A strategy is loaded and create based on this document in the CStrategyList class in its LoadStrategiesFromXML method. Below are the contents of this method, as well as of all related methods:

```
//+------------------------------------------------------------------+
//| Loads strategies from the passed XML file "xml_name"             |
//| If the load_curr_symbol flag is set to true, it will only load   |
//| the strategies in which symbol corresponds to the current        |
//| symbol CurrentSymbol()                                           |
//+------------------------------------------------------------------+
void CStrategyList::LoadStrategiesFromXML(string xml_name,bool load_curr_symbol)
  {
   CXmlDocument doc;
   string err;
   bool res=doc.CreateFromFile(xml_name,err);
   if(!res)
      printf(err);
   CXmlElement *global=GetPointer(doc.FDocumentElement);
   for(int i=0; i<global.GetChildCount(); i++)
     {
      CXmlElement* child = global.GetChild(i);
      if(child.GetName() == "Strategies")
         ParseStrategies(child,load_curr_symbol);
     }
  }
//+------------------------------------------------------------------+
//| Parses XML strategies                                            |
//+------------------------------------------------------------------+
void CStrategyList::ParseStrategies(CXmlElement *xmlStrategies,bool load_curr_symbol)
  {
   CParamsBase *params=NULL;
   for(int i=0; i<xmlStrategies.GetChildCount(); i++)
     {
      CXmlElement *xStrategy=xmlStrategies.GetChild(i);
      if(CheckPointer(params)!=POINTER_INVALID)
         delete params;
      params=new CParamsBase(xStrategy);
      if(!params.IsValid() || (params.Symbol()!=Symbol() && load_curr_symbol))
         continue;
      CStrategy *str=CStrategy::GetStrategy(params.Name());
      if(str==NULL)
         continue;
      str.ExpertMagic(params.Magic());
      str.ExpertSymbol(params.Symbol());
      str.Timeframe(params.Timeframe());
      str.ExpertName(params.Name());
      string name=str.ExpertName();
      CXmlElement *xml_params=xStrategy.GetChild("Params");
      if(xml_params!=NULL)
         str.ParseXmlParams(xml_params);
      CXmlElement *xml_mm=xStrategy.GetChild("MoneyManagment");
      if(xml_mm!=NULL)
        {
         if(!str.MM.ParseByXml(xml_mm))
           {
            string text="Strategy "+str.ExpertName()+" (Magic: "+(string)str.ExpertMagic()+") load MM from XML failed";
            CMessage *msg=new CMessage(MESSAGE_WARNING,__FUNCTION__,text);
            Log.AddMessage(msg);
           }
        }
      CXmlElement *xml_regim=xStrategy.GetChild("TradeStateStart");
      if(xml_regim!=NULL)
        {
         string regim=xml_regim.GetText();
         if(regim=="BuyAndSell")
            str.TradeState(TRADE_BUY_AND_SELL);
         else if(regim=="BuyOnly")
            str.TradeState(TRADE_BUY_ONLY);
         else if(regim=="SellOnly")
            str.TradeState(TRADE_SELL_ONLY);
         else if(regim=="Stop")
            str.TradeState(TRADE_STOP);
         else if(regim=="Wait")
            str.TradeState(TRADE_WAIT);
         else if(regim=="NoNewEntry")
            str.TradeState(TRADE_NO_NEW_ENTRY);
         else
           {
            string text="For strategy "+str.ExpertName()+" (Magic: "+(string)str.ExpertMagic()+
                        ") set not correctly trade state: "+regim;
            CMessage *msg=new CMessage(MESSAGE_WARNING,__FUNCTION__,text);
            Log.AddMessage(msg);
           }
        }
      AddStrategy(str);
     }
   if(CheckPointer(params)!=POINTER_INVALID)
      delete params;
  }
```

The most interesting part of the methods is creation of a strategy using the special static method CStrategy::GetStrategy. The name of the strategy should be passed to it as a parameter. The method returns a particular instance of the strategy associated with this name. The method has been made static to enable access to it before a strategy object is created. GetStrategy is written in a separate header file, because unlike other parts of the trading engine you will need to edit it from time to time adding new strategies to it. If you want your strategy to be loaded from XML, its creation procedure must be added directly to this method. The source code of this header file is as follows:

```
//+------------------------------------------------------------------+
//|                                              StrategyFactory.mqh |
//|                                 Copyright 2015, Vasiliy Sokolov. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Vasiliy Sokolov."
#property link      "http://www.mql5.com"
/*
   GetStrategy is a factory of strategies. It creates a strategy object corresponding to a certain name.
   The method is included in a separate file for automation purposes.
*/
#include <Strategy\Strategy.mqh>
#include <Strategy\Samples\MovingAverage.mqh>
#include <Strategy\Samples\ChannelSample.mqh>

CStrategy *CStrategy::GetStrategy(string name)
  {
   if(name=="MovingAverage")
      return new CMovingAverage();
   if(name=="BollingerBands")
      return new CChannel();
   CLog *mlog=CLog::GetLog();
   string text="Strategy with name "+name+" not defined in GetStrategy method. Please define strategy in 'StrategyFactory.mqh'";
   CMessage *msg=new CMessage(MESSAGE_ERROR,__FUNCTION__,text);
   mlog.AddMessage(msg);
   return NULL;
  }
```

Once the strategy has been created, it should be initialized with the required parameters from the <Params> section. Since the parameters of each strategy are unique, it is not possible to initialize these parameters at the level of the trading engine. Instead, the base class of the strategy can call the virtual method **ParseXmlParams**. If the strategy then overrides this method and properly parses the list of parameters as an XML node to it, it will be able to specify the required values of its own parameters. As an example, look at the ParseXmlParams method of the **CMovingAverage** strategy that trades based on two moving averages (its algorithm is described in the first chapter of this article).

```
//+------------------------------------------------------------------+
//| An example of a classical strategy based on two Moving Averages. |
//| If the fast MA crosses the slow one from upside down             |
//| we buy, if from top down - we sell.                              |
//+------------------------------------------------------------------+
class CMovingAverage : public CStrategy
  {
   ...
public:
   virtual bool      ParseXmlParams(CXmlElement *params);
  };
//+------------------------------------------------------------------+
//| The strategy's specific parameters are parsed inside it in       |
//| this method overridden from CStrategy                            |
//+------------------------------------------------------------------+
bool CMovingAverage::ParseXmlParams(CXmlElement *params)
  {
   bool res=true;
   for(int i=0; i<params.GetChildCount(); i++)
     {
      CXmlElement *param=params.GetChild(i);
      string name=param.GetName();
      if(name=="FastMA")
        {
         int fastMA=(int)param.GetText();
         if(fastMA == 0)
           {
            string text="Parameter 'FastMA' must be a number";
            CMessage *msg=new CMessage(MESSAGE_WARNING,SOURCE,text);
            Log.AddMessage(msg);
            res=false;
           }
         else
            FastMA.MaPeriod(fastMA);
        }
      else if(name=="SlowMA")
        {
         int slowMA=(int)param.GetText();
         if(slowMA == 0)
           {
            string text="Parameter 'SlowMA' must be a number";
            CMessage *msg=new CMessage(MESSAGE_WARNING,SOURCE,text);
            Log.AddMessage(msg);
            res=false;
           }
         else
            SlowMA.MaPeriod(slowMA);
        }
      else if(name=="Shift")
        {
         FastMA.MaShift((int)param.GetText());
         SlowMA.MaShift((int)param.GetText());
        }
      else if(name=="Method")
        {
         string smethod=param.GetText();
         ENUM_MA_METHOD method=MODE_SMA;
         if(smethod== "MODE_SMA")
            method = MODE_SMA;
         else if(smethod=="MODE_EMA")
            method=MODE_EMA;
         else if(smethod=="MODE_SMMA")
            method=MODE_SMMA;
         else if(smethod=="MODE_LWMA")
            method=MODE_LWMA;
         else
           {
            string text="Parameter 'Method' must be type of ENUM_MA_METHOD";
            CMessage *msg=new CMessage(MESSAGE_WARNING,SOURCE,text);
            Log.AddMessage(msg);
            res=false;
           }
         FastMA.MaMethod(method);
         SlowMA.MaMethod(method);
        }
      else if(name=="AppliedPrice")
        {
         string price=param.GetText();
         ENUM_APPLIED_PRICE a_price=PRICE_CLOSE;
         if(price=="PRICE_CLOSE")
            a_price=PRICE_CLOSE;
         else if(price=="PRICE_OPEN")
            a_price=PRICE_OPEN;
         else if(price=="PRICE_HIGH")
            a_price=PRICE_HIGH;
         else if(price=="PRICE_LOW")
            a_price=PRICE_LOW;
         else if(price=="PRICE_MEDIAN")
            a_price=PRICE_MEDIAN;
         else if(price=="PRICE_TYPICAL")
            a_price=PRICE_TYPICAL;
         else if(price=="PRICE_WEIGHTED")
            a_price=PRICE_WEIGHTED;
         else
           {
            string text="Parameter 'AppliedPrice' must be type of ENUM_APPLIED_PRICE";
            CMessage *msg=new CMessage(MESSAGE_WARNING,SOURCE,text);
            Log.AddMessage(msg);
            res=false;
           }
         FastMA.AppliedPrice(a_price);
         SlowMA.AppliedPrice(a_price);
        }
     }
   return res;
  }
```

The details of this strategy are described in the [third article](https://www.mql5.com/en/articles/2170) of the series, which covers the development of custom strategies.

Using the mechanism of strategy creation from a file, it is possible to configure a set of strategies once, and then load it from a file each time. You can go even further and write a self-optimizing algorithm that saves the sets of parameters of its best runs to an XML file. The trading engine will read this file at startup and will form a set of strategies on its basis.

### Managing Strategies Using a Custom Panel

From the point of view of the user, strategies can be conveniently controlled using a special _custom panel_. This panel would be displayed on a chart after the EA launch and would allowed performing simple operations with each of the trading algorithms:

- changing the strategy trading mode;
- buying or selling the required volume instead of the strategy.

The latter option is useful if the EA has failed to execute the appropriate action for some reason, and you need to synchronize its state with the current market situation.

Description of classes that create custom panels and dialog boxes is beyond the scope of the subject discussed, and requires a separate article. We will only describe the basic aspects related to the panel connection.

The Expert Advisor control panel is implemented in a separate **CPanel** class that includes various controls, such as lists, buttons and text labels. All classes for gui creation are available in <data\_folder>\\MQL5\\Include\\Panel. To ensure panel operation, it is necessary to handle the [OnChartEvent](https://www.mql5.com/en/docs/basis/function/events#onchartevent) event directly in the EA's mq5 file. The handler of chart events is located in the CStrategyList class, so it is enough to call this handler in OnChartEvent:

```
CStrategyList Manager;
...

void OnChartEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
   Manager.OnChartEvent(id,lparam,dparam,sparam);
  }
```

The handler of these events in CStrategyList sends them directly to the panel.

Upon a click on any button on the panel, it defines the action to be performed and performs it. For example, if we select a strategy from the list of strategies, the index of the current strategy will be equal to the selected one, then you can perform further trading actions. For example, you can change the trading mode of the elected strategy by selecting the appropriate option from the drop-down list of the strategy modes:

![Fig. 2. The list of modes of a selected strategy](https://c.mql5.com/2/21/fig2_list_of_modes.png)

Fig. 2. The list of modes of a selected strategy

Buying and selling on behalf of the selected strategy is performed the same way. A pointer to the strategy calls the Buy and Sell methods of the CStrategy base class. These methods buy and sell the volume passed in them. In this case, the magic number in the operations performed corresponds to the magic number of the strategy, so it is impossible to distinguish manual trading from the EA's actions.

It should be noted that the EA's trading logic is implemented so that all positions opened by a user are then maintained by this Expert Advisor in the normal mode. It manages such positions like its own automatically opened positions.

### Expert Advisors Trading in a Group

We can assemble a portfolio of trading strategies. The strategies must contain methods responsible for the parsing of XML parameters, i.e. we need to override the ParseXmlParams method. It is also necessary to add creation of the appropriate type of strategy to the CStrategy::GetStrategy method. Finally, we will need to create an XML file with a list of strategies and their parameters. After that the CStrategyList class will create instances of strategies and will add them to its list of strategies. The custom panel will display these strategies after that.

Let us create a portfolio of strategies consisting of the Expert Advisors described above. Examples of parsing of XML settings for the CMovingAverage and CChannel strategies are available in sections 3.5 and 4.3.

The contents of CStrategy::GetStrategy for the creation of the two strategies will be as follows:

```
//+------------------------------------------------------------------+
//|                                              StrategyFactory.mqh |
//|                                 Copyright 2015, Vasiliy Sokolov. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Vasiliy Sokolov."
#property link      "http://www.mql5.com"
/*
   GetStrategy is a factory of strategies. It creates a strategy object corresponding to a certain name.
   The method is included in a separate file for automation purposes.
*/
#include <Strategy\Strategy.mqh>
#include <Strategy\Samples\MovingAverage.mqh>
#include <Strategy\Samples\ChannelSample.mqh>

CStrategy *CStrategy::GetStrategy(string name)
  {
   if(name=="MovingAverage")
      return new CMovingAverage();
   if(name=="BollingerBands")
      return new CChannel();
   CLog *mlog=CLog::GetLog();
   string text="Strategy with name "+name+" not defined in GetStrategy method. Please define strategy in 'StrategyFactory.mqh'";
   CMessage *msg=new CMessage(MESSAGE_ERROR,__FUNCTION__,text);
   mlog.AddMessage(msg);
   return NULL;
  }
```

The final touch is to override the method responsible for the EA's full name. Perform the overriding for the CMovingAverage strategy:

```
//+------------------------------------------------------------------+
//| The full unique name of the Expert Advisor                       |
//+------------------------------------------------------------------+
string CMovingAverage::ExpertNameFull(void)
  {
   string name=ExpertName();
   name += "[" + ExpertSymbol();\
   name += "-" + StringSubstr(EnumToString(Timeframe()), 7);\
   name += "-" + (string)FastMA.MaPeriod();\
   name += "-" + (string)SlowMA.MaPeriod();\
   name += "-" + StringSubstr(EnumToString(SlowMA.MaMethod()), 5);\
   name += "]";
   return name;
  }
```

Now everything is ready for creating a portfolio of strategies. Our portfolio will include four trading systems. Each of them will trade its own symbol. Two strategies will be based on MovingAverage, and two others will use BollingerBands. A more detailed description of these strategies is available in the previous article: " [Universal Expert Advisor: Custom Strategies and Auxiliary Trade Classes (part 3)](https://www.mql5.com/en/articles/2170)".

Our XML portfolio will be as follows:

```
<Global>
        <Strategies>
                <Strategy Name="MovingAverage" Magic="100" Timeframe="PERIOD_M1" Symbol="Si">
                        <TradeStateStart>Stop</TradeStateStart>
                        <Params>
                                <FastMA>1</FastMA>
                                <SlowMA>3</SlowMA>
                                <Shift>0</Shift>
                                <Method>MODE_SMA</Method>
                                <AppliedPrice>PRICE_CLOSE</AppliedPrice>
                        </Params>
                </Strategy>
                <Strategy Name="MovingAverage" Magic="101" Timeframe="PERIOD_M5" Symbol="SBRF">
                        <TradeStateStart>BuyAndSell</TradeStateStart>
                        <Params>
                                <FastMA>15</FastMA>
                                <SlowMA>21</SlowMA>
                                <Shift>0</Shift>
                                <Method>MODE_SMA</Method>
                                <AppliedPrice>PRICE_CLOSE</AppliedPrice>
                        </Params>
                </Strategy>
                <Strategy Name="BollingerBands" Magic="102" Timeframe="PERIOD_M15" Symbol="GAZR">
                        <TradeStateStart>BuyAndSell</TradeStateStart>
                        <Params>
                                <Period>30</Period>
                                <StdDev>1.5</StdDev>
                        </Params>
                </Strategy>
                <Strategy Name="BollingerBands" Magic="103" Timeframe="PERIOD_M30" Symbol="ED">
                        <TradeStateStart>BuyAndSell</TradeStateStart>
                        <Params>
                                <Period>20</Period>
                                <StdDev>2.0</StdDev>
                        </Params>
                </Strategy>
        </Strategies>
</Global>
```

This file should be saved a common data folder of the MetaTrader platform as **Strategies.xml**.

Here is the source code of the mq5 module that creates an Expert Advisor:

```
//+------------------------------------------------------------------+
//|                                                       Expert.mq5 |
//|                                 Copyright 2015, Vasiliy Sokolov. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, Vasiliy Sokolov."
#property link      "http://www.mql5.com"
#property version   "1.00"
#include <Strategy\StrategiesList.mqh>

CStrategyList Manager;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   Manager.LoadStrategiesFromXML(StrategiesXMLFile,LoadOnlyCurrentSymbol);
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   Manager.OnTick();
  }
//+------------------------------------------------------------------+
//| BookEvent function                                               |
//+------------------------------------------------------------------+
void OnBookEvent(const string &symbol)
  {
   Manager.OnBookEvent(symbol);
  }

void OnChartEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
   Manager.OnChartEvent(id,lparam,dparam,sparam);
  }
```

Custom variables _StrategiesXMLFile_ and _LoadOnlyCurrentSymbol_ are defined in the CStrategyList class. They are used inside this class for specifying the list of strategies to load, and the mode that allows to only load the strategies with the symbol equal to the name of the instrument the Expert Advisor is running on. Also note that some events, such as OnBookEvent and OnTimer, are not used. This means that they will not be used in custom strategies.

The compilation should be successful. After that the Expert Advisor (named Agent.ex5 in the project) is ready for use. Let's try to run it on the chart. Before that, we must make sure that all used symbols are available in the MetaTrader Market Watch. After successful start, the Expert Advisor icon will appear in the upper right corner of the chart. Another button is added to the upper left corner of the chart; it maximizes the custom panel. If we select the list of EAs (named Agent) on the panel, a list of four Expert Advisors will open:

![Fig. 3. List of loaded Expert Advisors](https://c.mql5.com/2/21/fig3_list_of_experts.png)

Fig. 3. List of loaded Expert Advisors

The screenshot features the list of Expert Advisors formed by our XML file Strategies.xml. After a while, the strategies will start trading — each strategy on its individual symbol.

### Analyzing Expert Advisor Operation in the Strategy Tester

Having generated a portfolio of strategies, we can test it in the Strategy Tester to make sure it works properly. No additional specific action is required, because the XML list of strategies is located in the global data folder, accessible through the Strategy Tester. After launching the Agent.ex5 EA module in it, all the required symbols will be loaded automatically. Each Expert Advisor will perform trading operations following its individual trading rules, and will additionally draw its own set of indicators. The below video shows testing of a portfolio of strategies on four different instruments:

Tester 1 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2179)

MQL5.community

1.91K subscribers

[Tester 1](https://www.youtube.com/watch?v=eeTIGTibM8s)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

0:00

0:00 / 1:08

•Live

•

Simulation of CStrategy based strategies in the Strategy Tester is similar to realtime trading using these strategies. The visual testing option allows you to easily check the accuracy of the entries and exits of the strategies.

### Conclusion

We have considered algorithms allowing to create random sets of trading strategies. With these sets or portfolios of strategies, you can flexibly and efficiently scale the trading process, while managing multiple trading algorithms located in the same executable module. The algorithms are particularly useful for the strategies that use multiple trading instruments simultaneously. Using the proposed approach, creating similar trading algorithms is as easy as developing conventional trading strategies.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2179](https://www.mql5.com/ru/articles/2179)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2179.zip "Download all attachments in the single ZIP archive")

[strategyarticle.zip](https://www.mql5.com/en/articles/download/2179/strategyarticle.zip "Download strategyarticle.zip")(97.88 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing graphical interfaces based on .Net Framework and C# (part 2): Additional graphical elements](https://www.mql5.com/en/articles/6549)
- [Developing graphical interfaces for Expert Advisors and indicators based on .Net Framework and C#](https://www.mql5.com/en/articles/5563)
- [Custom Strategy Tester based on fast mathematical calculations](https://www.mql5.com/en/articles/4226)
- [R-squared as an estimation of quality of the strategy balance curve](https://www.mql5.com/en/articles/2358)
- [Universal Expert Advisor: CUnIndicator and Use of Pending Orders (Part 9)](https://www.mql5.com/en/articles/2653)
- [Implementing a Scalping Market Depth Using the CGraphic Library](https://www.mql5.com/en/articles/3336)
- [Universal Expert Advisor: Accessing Symbol Properties (Part 8)](https://www.mql5.com/en/articles/3270)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/85091)**
(18)


![Vasiliy Sokolov](https://c.mql5.com/avatar/2017/9/59C3C7E4-C9E1.png)

**[Vasiliy Sokolov](https://www.mql5.com/en/users/c-4)**
\|
15 Apr 2016 at 10:41

**Kuzmich:**

It still does not compile. Error: 'OnChartEvent' - function must have a body Agent.mq5 68 12

Comment out the [OnChartEvent function](https://www.mql5.com/en/docs/basis/function/events#oninit "MQL5 Documentation: Event Handling Functions") in Agent.mq5. For now in the current version of the compiler we will have to do without the panel and events from the chart.

```
//+------------------------------------------------------------------+
//||
//+------------------------------------------------------------------+
/*void OnChartEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
 {
 Manager.OnChartEvent(id,lparam,dparam,sparam);
 }*/
```

![Igor Nistor](https://c.mql5.com/avatar/2017/7/597A54A5-D950.jpg)

**[Igor Nistor](https://www.mql5.com/en/users/netmstnet)**
\|
15 Apr 2016 at 23:21

It worked :

MT5 Build 1301 from 15.04.16

P.S. hint where to dig: does not react to MM settings, that I do not specify, always trades only 1 lot, and also did not find anything about stops - loss, profit, trall, or it is not in the code?

![Vasiliy Sokolov](https://c.mql5.com/avatar/2017/9/59C3C7E4-C9E1.png)

**[Vasiliy Sokolov](https://www.mql5.com/en/users/c-4)**
\|
17 Apr 2016 at 18:56

**netmstnet:**

It worked :

MT5 Build 1301 from 15.04.16

P.S. hint where to dig: it does not react to MM settings that I do not specify, always trades only 1 lot, and also did not find anything about stops - loss, profit, trall, or it is not in the code?

Working with [pending orders](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type "MQL5 Documentation: Order Properties") will be described in the fifth part of the article. In order to react to the MM, it is necessary to explicitly specify in the Expert Advisor's logic what MM to use. Trails are not supported at the level of the engine itself, so to use them, you have to explicitly code a trawl in the Expert Advisor itself.


![Alexander](https://c.mql5.com/avatar/avatar_na2.png)

**[Alexander](https://www.mql5.com/en/users/kuva)**
\|
13 Aug 2016 at 18:31

For some reason in MT5 "Open" the panel in the tester does not work, but in MT5 MetaQuotes of the same bild 1375 the panel in the tester works, but all experts work only on the current instrument. Why? I can't find the file "Strategies.xml" in the attached archive.


![igorbel](https://c.mql5.com/avatar/avatar_na2.png)

**[igorbel](https://www.mql5.com/en/users/igorbel)**
\|
8 Sep 2016 at 09:02

Hello. Expert trading in a group is good, but there is a question of evaluating the results of each strategy, because a standard MT report will show the results for the whole portfolio. In principle, it is only necessary to [parse](https://www.mql5.com/en/articles/5638 "Article: Parsing MQL Using MQL ") all trades by magic number and evaluate financial results in this way. Do you have a solution in mind? Perhaps there is something already ready.


![Self-organizing feature maps (Kohonen maps) - revisiting the subject](https://c.mql5.com/2/20/jursbu_z7z.png)[Self-organizing feature maps (Kohonen maps) - revisiting the subject](https://www.mql5.com/en/articles/2043)

This article describes techniques of operating with Kohonen maps. The subject will be of interest to both market researchers with basic level of programing in MQL4 and MQL5 and experienced programmers that face difficulties with connecting Kohonen maps to their projects.

![Graphical Interfaces V: The List View Element (Chapter 2)](https://c.mql5.com/2/22/v-avatar.png)[Graphical Interfaces V: The List View Element (Chapter 2)](https://www.mql5.com/en/articles/2380)

In the previous chapter, we wrote classes for creating vertical and horizontal scrollbars. In this chapter, we will implement them. We will write a class for creating the list view element, a compound part of which will be a vertical scrollbar.

![Graphical Interfaces V: The Combobox Control (Chapter 3)](https://c.mql5.com/2/22/v-avatar__1.png)[Graphical Interfaces V: The Combobox Control (Chapter 3)](https://www.mql5.com/en/articles/2381)

In the first two chapters of the fifth part of the series, we developed classes for creating a scrollbar and a view list. In this chapter, we will speak about creating a class for the combobox control. This is also a compound control containing, among others, elements considered in the previous chapters of the fifth part.

![Graphical Interfaces V: The Vertical and Horizontal Scrollbar (Chapter 1)](https://c.mql5.com/2/22/v-avatar__2.png)[Graphical Interfaces V: The Vertical and Horizontal Scrollbar (Chapter 1)](https://www.mql5.com/en/articles/2379)

We are still discussing the development of the library for creating graphical interfaces in the MetaTrader environment. In the first article of the fifth part of the series, we will write classes for creating vertical and horizontal scrollbars.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/2179&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062758413508716629)

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