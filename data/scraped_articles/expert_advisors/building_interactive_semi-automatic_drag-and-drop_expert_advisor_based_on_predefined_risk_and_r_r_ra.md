---
title: Building interactive semi-automatic drag-and-drop Expert Advisor based on predefined risk and R/R ratio
url: https://www.mql5.com/en/articles/192
categories: Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:30:54.228997
---

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/192&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068324407656511516)

MetaTrader 5 / Examples


### Introduction

Some traders execute all their trades automatically, and some mix automatic and manual trades based on the output of several indicators. Being a member of the latter group I needed an interactive tool to asses dynamically risk and reward price levels directly from the chart.

Having declared maximum risk on my equity I wanted to calculate real-time parameters based on the stop-loss level I put on the chart and I needed to execute my trade directly from the EA based on calculated SL and TP levels.

This article will present a way to implement an interactive semi-automatic Expert Advisor with predefined equity risk and R/R ratio. The Expert Advisor risk, R/R and lot size parameters can be changed during runtime on the EA panel.

### 1\. Requirements

The requirements for the EA were as follows:

- ability to predefine risk level at startup and to change it during runtime to see how it affects position size
- ability to predefine risk to reward ratio and change it during runtime
- ability to calculate real-time maximum lot size for given risk and stop-loss level
- ability to change lot size at runtime to see how it affect equity risk and reward
- ability to execute buy/sell market order directly from EA
- drag and drop interface to set stop-loss and to see price level for predefined risk to reward level

### 2\. Design

Due to requirements for the EA of displaying and changing parameters during runtime I decided I would use [CChartObject](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/cchartobject)  classes and its descendands to display GUI on the chart window and handle incoming chart events for user interaction. Therefore, the EA needed user interface with labels, buttons and edit fields.

At first I wanted to use CChartObjectPanel object for grouping other objects on panel, but I decided to try a different approach, I designed a class that holds labels, edit fields and buttons and displays it on an image background. The background image of the interface was made using [GIMP](https://www.mql5.com/go?link=https://www.gimp.org/ "http://www.gimp.org/") software. MQL5 generated objects are edit fields, red labels updated real-time and buttons.

I simply put label objects on the chart and recorded their position and constructed CRRDialog class that handles all functions of displaying calculated output, receiving parameters of [CChartObjectEdit](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectedit) fields and recording button states. Color risk and reward rectangles are objects of [CChartObjectRectangle](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_shapes/cchartobjectrectangle) class and draggable stop loss pointer is a bitmap object of [CChartObjectBitmap](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectbitmap) class.

![Figure 1. Visual EA screenshot](https://c.mql5.com/2/2/Fig_1__2.png)

Figure 1. Visual EA screenshot

### 3\. Implementation of EA dialog class

The CRRDialog class handles all user interface of the EA. It contains a number of variables that are displayed, objects that are used to display the variables and methods to get/set variable values and refresh the dialog.

I am using [CChartObjectBmpLabel](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectbmplabel) object for the background, [CChartObjectEdit](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectedit) objects for edit fields, [CChartObjectLabel](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectlabel) objects for displaying labels and [CChartObjectButton](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectbutton) objects for buttons:

```
class CRRDialog
  {
private:

   int               m_baseX;
   int               m_baseY;
   int               m_fontSize;

   string            m_font;
   string            m_dialogName;
   string            m_bgFileName;

   double            m_RRRatio;
   double            m_riskPercent;
   double            m_orderLots;
   double            m_SL;
   double            m_TP;
   double            m_maxAllowedLots;
   double            m_maxTicksLoss;
   double            m_orderEquityRisk;
   double            m_orderEquityReward;
   ENUM_ORDER_TYPE   m_orderType;

   CChartObjectBmpLabel m_bgDialog;

   CChartObjectEdit  m_riskRatioEdit;
   CChartObjectEdit  m_riskValueEdit;
   CChartObjectEdit  m_orderLotsEdit;

   CChartObjectLabel m_symbolNameLabel;
   CChartObjectLabel m_tickSizeLabel;
   CChartObjectLabel m_maxEquityLossLabel;
   CChartObjectLabel m_equityLabel;
   CChartObjectLabel m_profitValueLabel;
   CChartObjectLabel m_askLabel;
   CChartObjectLabel m_bidLabel;
   CChartObjectLabel m_tpLabel;
   CChartObjectLabel m_slLabel;
   CChartObjectLabel m_maxAllowedLotsLabel;
   CChartObjectLabel m_maxTicksLossLabel;
   CChartObjectLabel m_orderEquityRiskLabel;
   CChartObjectLabel m_orderEquityRewardLabel;
   CChartObjectLabel m_orderTypeLabel;

   CChartObjectButton m_switchOrderTypeButton;
   CChartObjectButton m_placeOrderButton;
   CChartObjectButton m_quitEAButton;

public:

   void              CRRDialog(); // CRRDialog constructor
   void             ~CRRDialog(); // CRRDialog destructor

   bool              CreateCRRDialog(int topX,int leftY);
   int               DeleteCRRDialog();
   void              Refresh();
   void              SetRRRatio(double RRRatio);
   void              SetRiskPercent(double riskPercent);
   double            GetRiskPercent();
   double            GetRRRRatio();
   void              SetSL(double sl);
   void              SetTP(double tp);
   double            GetSL();
   double            GetTP();
   void              SetMaxAllowedLots(double lots);
   void              SetMaxTicksLoss(double ticks);
   void              SetOrderType(ENUM_ORDER_TYPE);
   void              SwitchOrderType();
   void              ResetButtons();
   ENUM_ORDER_TYPE   GetOrderType();
   void              SetOrderLots(double orderLots);
   double            GetOrderLots();
   void              SetOrderEquityRisk(double equityRisk);
   void              SetOrderEquityReward(double equityReward);
  };
```

Since get/set variables methods are straightforward, I will concentrate on CreateCRRDialog() and Refresh() methods. CreateCRRDialog() method initializes background image, labels, buttons and edit fields.

For initializing labels and edit fields I use: Create() method with coordinate parameters to locate the object on the chart, Font() and FontSize() method to setup font and Description() method to put text on the label.

For buttons: [Create()](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectedit/cchartobjecteditcreate) method additional parameters specify button size and  [BackColor()](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectedit/cchartobjecteditbackcolor) method specifies background color of the button.

```
bool CRRDialog::CreateCRRDialog(int topX,int leftY)
  {
   bool isCreated=false;

   MqlTick current_tick;
   SymbolInfoTick(Symbol(),current_tick);

   m_baseX = topX;
   m_baseY = leftY;

   m_bgDialog.Create(0, m_dialogName, 0, topX, leftY);
   m_bgDialog.BmpFileOn(m_bgFileName);

   m_symbolNameLabel.Create(0, "symbolNameLabel", 0, m_baseX + 120, m_baseY + 40);
   m_symbolNameLabel.Font("Verdana");
   m_symbolNameLabel.FontSize(8);
   m_symbolNameLabel.Description(Symbol());

   m_tickSizeLabel.Create(0, "tickSizeLabel", 0, m_baseX + 120, m_baseY + 57);
   m_tickSizeLabel.Font("Verdana");
   m_tickSizeLabel.FontSize(8);
   m_tickSizeLabel.Description(DoubleToString(SymbolInfoDouble(Symbol(), SYMBOL_TRADE_TICK_SIZE), Digits()));

   m_riskRatioEdit.Create(0, "riskRatioEdit", 0, m_baseX + 120, m_baseY + 72, 35, 15);
   m_riskRatioEdit.Font("Verdana");
   m_riskRatioEdit.FontSize(8);
   m_riskRatioEdit.Description(DoubleToString(m_RRRatio, 2));

   m_riskValueEdit.Create(0, "riskValueEdit", 0, m_baseX + 120, m_baseY + 90, 35, 15);
   m_riskValueEdit.Font("Verdana");
   m_riskValueEdit.FontSize(8);
   m_riskValueEdit.Description(DoubleToString(m_riskPercent, 2));

   m_equityLabel.Create(0, "equityLabel", 0, m_baseX + 120, m_baseY + 107);
   m_equityLabel.Font("Verdana");
   m_equityLabel.FontSize(8);
   m_equityLabel.Description(DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY),2));

   m_maxEquityLossLabel.Create(0, "maxEquityLossLabel", 0, m_baseX + 120, m_baseY + 122);
   m_maxEquityLossLabel.Font("Verdana");
   m_maxEquityLossLabel.FontSize(8);
   m_maxEquityLossLabel.Description(DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY)*m_riskPercent/100.0,2));

   m_askLabel.Create(0, "askLabel", 0, m_baseX + 120, m_baseY + 145);
   m_askLabel.Font("Verdana");
   m_askLabel.FontSize(8);
   m_askLabel.Description("");

   m_bidLabel.Create(0, "bidLabel", 0, m_baseX + 120, m_baseY + 160);
   m_bidLabel.Font("Verdana");
   m_bidLabel.FontSize(8);
   m_bidLabel.Description("");

   m_slLabel.Create(0, "slLabel", 0, m_baseX + 120, m_baseY + 176);
   m_slLabel.Font("Verdana");
   m_slLabel.FontSize(8);
   m_slLabel.Description("");

   m_tpLabel.Create(0, "tpLabel", 0, m_baseX + 120, m_baseY + 191);
   m_tpLabel.Font("Verdana");
   m_tpLabel.FontSize(8);
   m_tpLabel.Description("");

   m_maxAllowedLotsLabel.Create(0, "maxAllowedLotsLabel", 0, m_baseX + 120, m_baseY + 208);
   m_maxAllowedLotsLabel.Font("Verdana");
   m_maxAllowedLotsLabel.FontSize(8);
   m_maxAllowedLotsLabel.Description("");

   m_maxTicksLossLabel.Create(0, "maxTicksLossLabel", 0, m_baseX + 120, m_baseY + 223);
   m_maxTicksLossLabel.Font("Verdana");
   m_maxTicksLossLabel.FontSize(8);
   m_maxTicksLossLabel.Description("");

   m_orderLotsEdit.Create(0, "orderLotsEdit", 0, m_baseX + 120, m_baseY + 238, 35, 15);
   m_orderLotsEdit.Font("Verdana");
   m_orderLotsEdit.FontSize(8);
   m_orderLotsEdit.Description("");

   m_orderEquityRiskLabel.Create(0, "orderEquityRiskLabel", 0, m_baseX + 120, m_baseY + 255);
   m_orderEquityRiskLabel.Font("Verdana");
   m_orderEquityRiskLabel.FontSize(8);
   m_orderEquityRiskLabel.Description("");

   m_orderEquityRewardLabel.Create(0, "orderEquityRewardLabel", 0, m_baseX + 120, m_baseY + 270);
   m_orderEquityRewardLabel.Font("Verdana");
   m_orderEquityRewardLabel.FontSize(8);
   m_orderEquityRewardLabel.Description("");

   m_switchOrderTypeButton.Create(0, "switchOrderTypeButton", 0, m_baseX + 20, m_baseY + 314, 160, 20);
   m_switchOrderTypeButton.Font("Verdana");
   m_switchOrderTypeButton.FontSize(8);
   m_switchOrderTypeButton.BackColor(LightBlue);

   m_placeOrderButton.Create(0, "placeOrderButton", 0, m_baseX + 20, m_baseY + 334, 160, 20);
   m_placeOrderButton.Font("Verdana");
   m_placeOrderButton.FontSize(8);
   m_placeOrderButton.BackColor(LightBlue);
   m_placeOrderButton.Description("Place Market Order");

   m_quitEAButton.Create(0, "quitEAButton", 0, m_baseX + 20, m_baseY + 354, 160, 20);
   m_quitEAButton.Font("Verdana");
   m_quitEAButton.FontSize(8);
   m_quitEAButton.BackColor(LightBlue);
   m_quitEAButton.Description("Quit");

   return isCreated;
  }
```

Refresh() method refreshes all labels and buttons description with the CRRDialog variables and current bid/ask levels, account equity and equity risk values:

```
void CRRDialog::Refresh()
  {
   MqlTick current_tick;
   SymbolInfoTick(Symbol(),current_tick);

   m_equityLabel.Description(DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY),2));
   m_maxEquityLossLabel.Description(DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY)*
                                         StringToDouble(m_riskValueEdit.Description())/100.0,2));
   m_askLabel.Description(DoubleToString(current_tick.ask, Digits()));
   m_bidLabel.Description(DoubleToString(current_tick.bid, Digits()));
   m_slLabel.Description(DoubleToString(m_SL, Digits()));
   m_tpLabel.Description(DoubleToString(m_TP, Digits()));
   m_maxAllowedLotsLabel.Description(DoubleToString(m_maxAllowedLots,2));
   m_maxTicksLossLabel.Description(DoubleToString(m_maxTicksLoss,0));
   m_orderEquityRiskLabel.Description(DoubleToString(m_orderEquityRisk,2));
   m_orderEquityRewardLabel.Description(DoubleToString(m_orderEquityReward,2));

   if(m_orderType==ORDER_TYPE_BUY) m_switchOrderTypeButton.Description("Order Type: BUY");
   else if(m_orderType==ORDER_TYPE_SELL) m_switchOrderTypeButton.Description("Order Type: SELL");
  }
```

### 4\. Chart Events

Since EA is designed to be interactive, it shall handle chart events.

The events that are handled include:

- dragging S/L pointer (SL\_arrow object of [CChartObjectBitmap](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectbitmap) class) on the chart - this will allow to collect S/L level and calculate T/P level based on R/R ratio
- switching order type (buy/sell) button
- pressing 'place market order' button
- editing risk, R/R and order lot fields
- closing EA after pressing 'Exit' button

Events that are handled are [CHARTEVENT\_OBJECT\_CLICK](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents)for pointer selection and buttons, [CHARTEVENT\_OBJECT\_DRAG](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents)for dragging S/L pointer, and [CHARTEVENT\_OBJECT\_ENDEDIT](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) after edit fields are updated by the trader.

At first implementation of [OnChartEvent()](https://www.mql5.com/en/docs/basis/function/events#onchartevent) function took a few pages of code, but I decided to split it into several event handlers, this converted the [OnChartEvent()](https://www.mql5.com/en/docs/basis/function/events#onchartevent) function to human readeable form:

```
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//--- Check the event by pressing a mouse button
   if(id==CHARTEVENT_OBJECT_CLICK)
     {
      string clickedChartObject=sparam;

      if(clickedChartObject==slButtonID)
         SL_arrow.Selected(!SL_arrow.Selected());

      if(clickedChartObject==switchOrderTypeButtonID)
        {
         EA_switchOrderType();
        };

      if(clickedChartObject==placeOrderButtonID)
        {
         EA_placeOrder();
        }

      if(clickedChartObject==quitEAButtonID) ExpertRemove();

      ChartRedraw();
     }

   if(id==CHARTEVENT_OBJECT_DRAG)
     {
      // BUY
      if(visualRRDialog.GetOrderType()==ORDER_TYPE_BUY)
        {
         EA_dragBuyHandle();
        };

      // SELL
      if(visualRRDialog.GetOrderType()==ORDER_TYPE_SELL)
        {
         EA_dragSellHandle();
        };
      ChartRedraw();
     }

   if(id==CHARTEVENT_OBJECT_ENDEDIT)
     {
      if((sparam==riskRatioEditID || sparam==riskValueEditID || sparam==orderLotsEditID) && orderPlaced==false)
        {
         EA_editParamsUpdate();
        }
     }
  }
```

Event handlers implementation will be described in more detail in next sections. Worth noticing is the trick I used for selecting SL\_arrow object. Normally, to select an object on the chart one has to click twice on it. But it can be selected by clicking once and invoking [Selected()](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/cchartobject/cchartobjectselected) method of [CChartObject](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/cchartobject) object or its descendant in [OnChartEvent()](https://www.mql5.com/en/docs/basis/function/events#onchartevent) function inside [CHARTEVENT\_OBJECT\_CLICK](https://www.mql5.com/en/docs/basis/function/events#onchartevent) event handler:

```
      if(clickedChartObject==slButtonID)
         SL_arrow.Selected(!SL_arrow.Selected());
```

Object it is selected or deselected depending on its previous state after one click.

### 5\. Extended Money Management class based on CMoneyFixedRisk

Before I will be able to describe ChartEvent handlers I need to go through money management class.

For money management I reused CMoneyFixedRisk class provided by MetaQuotes and implemented CMoneyFixedRiskExt class.

Original CMoneyFixedRisk class methods return allowed order lot amounts for given price, stop-loss level and equity risk between minimum and maximum lot size allowed by broker. I changed CheckOpenLong() and CheckOpenShort() methods to return 0.0 lot size if risk requirements are not met and extended it with four methods: GetMaxSLPossible(), CalcMaxTicksLoss(), CalcOrderEquityRisk() and CalcOrderEquityReward():

```
class CMoneyFixedRiskExt : public CExpertMoney
  {
public:
   //---
   virtual double    CheckOpenLong(double price,double sl);
   virtual double    CheckOpenShort(double price,double sl);

   double GetMaxSLPossible(double price, ENUM_ORDER_TYPE orderType);
   double CalcMaxTicksLoss();
   double CalcOrderEquityRisk(double price, double sl, double lots);
   double CalcOrderEquityReward(double price, double sl, double lots, double rrratio);
  };
```

GetMaxSLPossible() method calculates maximum stop-loss price value for given equity risk and minimum allowed trade size.

For example if account balance is 10 000 of account base currency and risk is 2%, we may put maximum 200 of account currency at risk. If the minimum trade lot size is 0.1 lot, this method returns price level for [ORDER\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties)or [ORDER\_TYPE\_SELL](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties)order that will meet value of equity risk for position of 0.1 lot. This helps to estimate of what is the maximum stop-loss level we can afford for minimum lot size trade. This is a price level we cannot cross for given equity risk level.

```
double CMoneyFixedRiskExt::GetMaxSLPossible(double price, ENUM_ORDER_TYPE orderType)
{
   double maxEquityLoss, tickValLoss, maxTicksLoss;
   double minvol=m_symbol.LotsMin();
   double orderTypeMultiplier;

   if(m_symbol==NULL) return(0.0);

   switch (orderType)
   {
   case ORDER_TYPE_SELL: orderTypeMultiplier = -1.0; break;
   case ORDER_TYPE_BUY: orderTypeMultiplier = 1.0; break;
   default: orderTypeMultiplier = 0.0;
   }

   maxEquityLoss = m_account.Balance()*m_percent/100.0; // max loss
   tickValLoss = minvol*m_symbol.TickValueLoss(); // tick val loss
   maxTicksLoss = MathFloor(maxEquityLoss/tickValLoss);

   return (price - maxTicksLoss*m_symbol.TickSize()*orderTypeMultiplier);
}
```

CalcMaxTickLoss() method returns maximum number of ticks we can afford to loose for given risk and minimum allowed lot size.

At first maximum equity loss is calculated as percentage of the current balance, then tick value loss for change of one tick for minimum allowed lot size for given symbol is calculated. Then maximum equity loss is divided by tick value loss and the result is.  rounding it to integer value with [MathFloor()](https://www.mql5.com/en/docs/math/mathfloor) function:

```
double CMoneyFixedRiskExt::CalcMaxTicksLoss()
{
   double maxEquityLoss, tickValLoss, maxTicksLoss;
   double minvol=m_symbol.LotsMin();

   if(m_symbol==NULL) return(0.0);

   maxEquityLoss = m_account.Balance()*m_percent/100.0; // max loss
   tickValLoss = minvol*m_symbol.TickValueLoss(); // tick val loss
   maxTicksLoss = MathFloor(maxEquityLoss/tickValLoss);

   return (maxTicksLoss);
}
```

CalcOrderEquityRisk() method returns equity risk for given price, stop loss level and amount of lots. It is calculated by multiplying tick loss value by number of lots and price then multiplying by difference between current price and stop-loss level:

```
double CMoneyFixedRiskExt::CalcOrderEquityRisk(double price,double sl, double lots)
{
   double equityRisk;

   equityRisk = lots*m_symbol.TickValueLoss()*(MathAbs(price-sl)/m_symbol.TickSize());

   if (dbg) Print("calcEquityRisk: lots = " + DoubleToString(lots) +
                 " TickValueLoss = " + DoubleToString(m_symbol.TickValueLoss()) +
                 " risk = " + DoubleToString(equityRisk));

   return equityRisk;
}
```

CalcOrderEquityReward() method is analogic to CalcOrderEquityRisk() method but it uses TickValueProfit() instead of TickValueLoss() method and the result is multiplied by given risk to reward ratio:

```
double CMoneyFixedRiskExt::CalcOrderEquityReward(double price,double sl, double lots, double rrratio)
{
   double equityReward;
   equityReward = lots*m_symbol.TickValueProfit()*(MathAbs(price-sl)/m_symbol.TickSize())*rrratio;

   if (dbg) Print("calcEquityReward: lots = " + DoubleToString(lots) +
                   " TickValueProfit = " + DoubleToString(m_symbol.TickValueProfit()) +
                 " reward = " + DoubleToString(equityReward));
   return equityReward;
}
```

Those methods are sufficient for calculating maximum stop-loss levels and returning real-time equity risk and reward. CalcMaxTickLoss() method is used to correct drawing of risk rectangle - if trader wants to put a trade that crosses the boundary of number of ticks he can afford to loose, the rectangle is drawn only to the maximum number of ticks he can loose.

It makes life easier to see it directly on the chart. You can see it in demo at the end of the article.

### 6\. Chart Event handlers implementation

EA\_switchOrderType() handler is triggered after receiving [CHARTEVENT\_OBJECT\_CLICK](https://www.mql5.com/en/docs/basis/function/events#onchartevent)event on m\_switchOrderTypeButtonobject. It switches order type between [ORDER\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type) and [ORDER\_TYPE\_SELL](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type), resets buttons state, dialog's variables and deletes risk and reward rectangle objects on the chart:

```
void EA_switchOrderType()
  {
   symbolInfo.RefreshRates();

   visualRRDialog.SwitchOrderType();
   visualRRDialog.ResetButtons();
   visualRRDialog.SetSL(0.0);
   visualRRDialog.SetTP(0.0);
   visualRRDialog.SetMaxAllowedLots(0.0);
   visualRRDialog.SetOrderLots(0.0);
   visualRRDialog.SetMaxTicksLoss(0);
   visualRRDialog.SetOrderEquityRisk(0.0);
   visualRRDialog.SetOrderEquityReward(0.0);

   if(visualRRDialog.GetOrderType()==ORDER_TYPE_BUY) SL_arrow.SetDouble(OBJPROP_PRICE,symbolInfo.Ask());
   else if(visualRRDialog.GetOrderType()==ORDER_TYPE_SELL) SL_arrow.SetDouble(OBJPROP_PRICE,symbolInfo.Bid());
   SL_arrow.SetInteger(OBJPROP_TIME,0,TimeCurrent());

   rectReward.Delete();
   rectRisk.Delete();

   visualRRDialog.Refresh();

  }
```

EA\_dragBuyHandle() handler is triggered after SL\_arrow object is dragged and dropped on the chart. At first it reads SL\_arrow object drop point time and price parameters from the chart and sets price level as a hypotetical stop-loss for our trade.

Then it calculates how many lots can we open for given risk on the equity. If stop loss value cannot guarantee risk objective for the lowest trading lot possible on that symbol, it is automatically moved to the maximum SL level possible. This helps to assess how much space we have for stop loss for given risk.

After calculating risk and reward, rectangle objects are updated on the chart.

```
void EA_dragBuyHandle()
  {
   SL_arrow.GetDouble(OBJPROP_PRICE,0,SL_price);
   SL_arrow.GetInteger(OBJPROP_TIME,0,startTime);

   symbolInfo.RefreshRates();
   currentTime=TimeCurrent();

// BUY
   double allowedLots=MM.CheckOpenLong(symbolInfo.Ask(),SL_price);
   Print("Allowed lots = "+DoubleToString(allowedLots,2));
   double lowestSLAllowed=MM.GetMaxSLPossible(symbolInfo.Ask(),ORDER_TYPE_BUY);

   if(SL_price<lowestSLAllowed)
     {
      SL_price=lowestSLAllowed;
      ObjectSetDouble(0,slButtonID,OBJPROP_PRICE,lowestSLAllowed);
     }

   visualRRDialog.SetSL(SL_price);
   visualRRDialog.SetTP(symbolInfo.Ask()+(symbolInfo.Ask()-SL_price)*visualRRDialog.GetRRRRatio());

   if(visualRRDialog.GetTP()<SL_price)
     {
      visualRRDialog.SetSL(0.0);
      visualRRDialog.SetTP(0.0);
      SL_arrow.SetDouble(OBJPROP_PRICE,symbolInfo.Ask());
      rectReward.Delete();
      rectRisk.Delete();
      return;
     }

   double lotSize=MM.CheckOpenLong(symbolInfo.Ask(),SL_price);

   visualRRDialog.SetMaxAllowedLots(lotSize);
   visualRRDialog.SetOrderLots(lotSize);
   visualRRDialog.SetMaxTicksLoss(MM.CalcMaxTicksLoss());
   visualRRDialog.SetOrderEquityRisk(MM.CalcOrderEquityRisk(symbolInfo.Ask(), SL_price, lotSize));
   visualRRDialog.SetOrderEquityReward(MM.CalcOrderEquityReward(symbolInfo.Ask(),
                                       SL_price, lotSize, visualRRDialog.GetRRRRatio()));
   visualRRDialog.Refresh();

   rectUpdate(visualRRDialog.GetOrderType());

  }
```

EA\_dragSellHandle() is triggered for sell order configuration.

Calculations are based on [symbolInfo.Bid()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo/csymbolinfobid) price, and rectangles are drawn accordingly, that is green zone marking profit is below current price level.

```
void EA_dragSellHandle()
  {
   SL_arrow.GetDouble(OBJPROP_PRICE,0,SL_price);
   SL_arrow.GetInteger(OBJPROP_TIME,0,startTime);

   symbolInfo.RefreshRates();
   currentTime=TimeCurrent();

   double allowedLots=MM.CheckOpenShort(symbolInfo.Bid(),SL_price);
   Print("Allowed lots = "+DoubleToString(allowedLots,2));
   double maxSLAllowed=MM.GetMaxSLPossible(symbolInfo.Bid(),ORDER_TYPE_SELL);

   if(SL_price>maxSLAllowed)
     {
      SL_price=maxSLAllowed;
      SL_arrow.SetDouble(OBJPROP_PRICE,0,maxSLAllowed);
     }

   visualRRDialog.SetSL(SL_price);
   visualRRDialog.SetTP(symbolInfo.Bid()-(SL_price-symbolInfo.Bid())*visualRRDialog.GetRRRRatio());

   if(visualRRDialog.GetTP()>SL_price)
     {
      visualRRDialog.SetSL(0.0);
      visualRRDialog.SetTP(0.0);
      SL_arrow.SetDouble(OBJPROP_PRICE,symbolInfo.Bid());
      rectReward.Delete();
      rectRisk.Delete();
      return;
     }

   double lotSize=MM.CheckOpenShort(symbolInfo.Bid(),SL_price);

   visualRRDialog.SetMaxAllowedLots(lotSize);
   visualRRDialog.SetOrderLots(lotSize);
   visualRRDialog.SetMaxTicksLoss(MM.CalcMaxTicksLoss());
   visualRRDialog.SetOrderEquityRisk(MM.CalcOrderEquityRisk(symbolInfo.Bid(), SL_price, lotSize));
   visualRRDialog.SetOrderEquityReward(MM.CalcOrderEquityReward(symbolInfo.Bid(),
                                       SL_price, lotSize, visualRRDialog.GetRRRRatio()));
   visualRRDialog.Refresh();

   rectUpdate(visualRRDialog.GetOrderType());

  }
```

EA\_placeOrder() is triggered after m\_placeOrderButton object was pressed. It places buy or sell market order for calculated SL and TP levels and given lot size.

Please notice how easy it is to place market order using CExpertTrade class.

```
bool EA_placeOrder()
  {
   symbolInfo.RefreshRates();
   visualRRDialog.ResetButtons();

   if(visualRRDialog.GetOrderType()==ORDER_TYPE_BUY)
      orderPlaced=trade.Buy(visualRRDialog.GetOrderLots(),symbolInfo.Ask(),
                            visualRRDialog.GetSL(),visualRRDialog.GetTP(),TimeToString(TimeCurrent()));
   else if(visualRRDialog.GetOrderType()==ORDER_TYPE_SELL)
      orderPlaced=trade.Sell(visualRRDialog.GetOrderLots(),symbolInfo.Bid(),
                            visualRRDialog.GetSL(),visualRRDialog.GetTP(),TimeToString(TimeCurrent()));

   return orderPlaced;
  }
```

EA\_editParamsUpdate() handler is triggered when Enter key is pressed after editing one of the edit fields: riskRatioEdit, riskValueEdit and orderLotsEdit.

When this happens allowed lot size, TP level, max tick loss, equity risk and reward need to be recalculated:

```
void EA_editParamsUpdate()
  {
   MM.Percent(visualRRDialog.GetRiskPercent());

   SL_arrow.GetDouble(OBJPROP_PRICE, 0, SL_price);
   SL_arrow.GetInteger(OBJPROP_TIME, 0, startTime);

   symbolInfo.RefreshRates();
   currentTime=TimeCurrent();

   double allowedLots=MM.CheckOpenLong(symbolInfo.Ask(),SL_price);

   double lowestSLAllowed=MM.GetMaxSLPossible(symbolInfo.Ask(),ORDER_TYPE_BUY);
   if(SL_price<lowestSLAllowed)
     {
      SL_price=lowestSLAllowed;
      ObjectSetDouble(0,slButtonID,OBJPROP_PRICE,lowestSLAllowed);
     }

   visualRRDialog.SetSL(SL_price);
   visualRRDialog.SetTP(symbolInfo.Ask()+(symbolInfo.Ask()-SL_price)*visualRRDialog.GetRRRRatio());

   visualRRDialog.SetMaxTicksLoss(MM.CalcMaxTicksLoss());
   visualRRDialog.SetOrderEquityRisk(MM.CalcOrderEquityRisk(symbolInfo.Ask(),
                                     SL_price, visualRRDialog.GetOrderLots()));
   visualRRDialog.SetOrderEquityReward(MM.CalcOrderEquityReward(symbolInfo.Ask(), SL_price,
                                       visualRRDialog.GetOrderLots(), visualRRDialog.GetRRRRatio()));
   visualRRDialog.Refresh();
   rectUpdate(visualRRDialog.GetOrderType());

   ChartRedraw();
  }
```

EA\_onTick() is invoked every time new tick arrives. Calculations are performed only if order was not placed yet and stop loss level was already chosen by dragging SL\_arrow pointer.

After order is placed, risk and reward and TP level as well as redrawing of the risk and reward are not needed.

```
void EA_onTick()
  {
   if(SL_price!=0.0 && orderPlaced==false)
     {
      double lotSize=0.0;
      SL_price=visualRRDialog.GetSL();
      symbolInfo.RefreshRates();

      if(visualRRDialog.GetOrderType()==ORDER_TYPE_BUY)
         lotSize=MM.CheckOpenLong(symbolInfo.Ask(),SL_price);
      else if(visualRRDialog.GetOrderType()==ORDER_TYPE_SELL)
         lotSize=MM.CheckOpenShort(symbolInfo.Ask(),SL_price);

      visualRRDialog.SetMaxAllowedLots(lotSize);
      if(visualRRDialog.GetOrderLots()>lotSize) visualRRDialog.SetOrderLots(lotSize);

      visualRRDialog.SetMaxTicksLoss(MM.CalcMaxTicksLoss());

      if(visualRRDialog.GetOrderType()==ORDER_TYPE_BUY)
        {
         visualRRDialog.SetTP(symbolInfo.Ask()+(symbolInfo.Ask()-SL_price)*visualRRDialog.GetRRRRatio());
         visualRRDialog.SetOrderEquityRisk(MM.CalcOrderEquityRisk(symbolInfo.Ask(),
                                           SL_price, visualRRDialog.GetOrderLots()));
         visualRRDialog.SetOrderEquityReward(MM.CalcOrderEquityReward(symbolInfo.Ask(), SL_price,
                                             visualRRDialog.GetOrderLots(), visualRRDialog.GetRRRRatio()));
        }
      else if(visualRRDialog.GetOrderType()==ORDER_TYPE_SELL)
        {
         visualRRDialog.SetTP(symbolInfo.Bid()-(SL_price-symbolInfo.Bid())*visualRRDialog.GetRRRRatio());
         visualRRDialog.SetOrderEquityRisk(MM.CalcOrderEquityRisk(
                                           symbolInfo.Bid(), SL_price, visualRRDialog.GetOrderLots()));
         visualRRDialog.SetOrderEquityReward(MM.CalcOrderEquityReward(symbolInfo.Bid(), SL_price,
                                             visualRRDialog.GetOrderLots(), visualRRDialog.GetRRRRatio()));
        }
      visualRRDialog.Refresh();
      rectUpdate(visualRRDialog.GetOrderType());
     }

   ChartRedraw(0);
  }
```

Function rectUpdate() is responsible for redrawing color risk and reward rectangles. The control points are SL\_arrow object start time, current Ask or Bid price value depending on order type, and SL and TP levels. Light pink rectangle shows price range between current price and SL level and light green rectangle shows price range between current price and TP level.

Both rectangles are a great tool to observe risk to reward ratio impact on SL and TP price levels and help to adjust risk before entering the trade.

```
void rectUpdate(ENUM_ORDER_TYPE orderType)
  {
   symbolInfo.RefreshRates();
   currentTime=TimeCurrent();
   SL_arrow.GetInteger(OBJPROP_TIME,0,startTime);

   if(orderType==ORDER_TYPE_BUY)
     {
      rectReward.Create(0,rewardRectID,0,startTime,symbolInfo.Ask(),currentTime,symbolInfo.Ask()+
                       (symbolInfo.Ask()-visualRRDialog.GetSL())*visualRRDialog.GetRRRRatio());
      rectReward.Color(LightGreen);
      rectReward.Background(true);

      rectRisk.Create(0,riskRectID,0,startTime,visualRRDialog.GetSL(),currentTime,symbolInfo.Ask());
      rectRisk.Color(LightPink);
      rectRisk.Background(true);
     }
   else if(orderType==ORDER_TYPE_SELL)
     {
      rectReward.Create(0,rewardRectID,0,startTime,symbolInfo.Bid(),currentTime,symbolInfo.Bid()-
                        (visualRRDialog.GetSL()-symbolInfo.Bid())*visualRRDialog.GetRRRRatio());
      rectReward.Color(LightGreen);
      rectReward.Background(true);

      rectRisk.Create(0,riskRectID,0,startTime,visualRRDialog.GetSL(),currentTime,symbolInfo.Bid());
      rectRisk.Color(LightPink);
      rectRisk.Background(true);
     }
  }
```

### 7\. Demo

Please observe below demo of the working Expert Advisor in action. I am doing sell order after big rebounce shortly after market opened on Monday 01/11/2010.

For best viewing experience please set the video to full screen and quality to 480p. Comments are included in the video:

### **Conclusion**

In the following article I presented a way to build interactive Expert Advisor for manual trading based on predefined risk and risk to reward ratio.

I showed how to use standard classes for displaying content on the chart and how to handle chart events for entering new data and handling drag-and-drop objects. I hope that the ideas I presented will serve as basis for building other configurable visual tools in MQL5.

All source files and bitmaps are attached to the article.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/192.zip "Download all attachments in the single ZIP archive")

[visualrrea.mq5](https://www.mql5.com/en/articles/download/192/visualrrea.mq5 "Download visualrrea.mq5")(13.84 KB)

[crrdialog.mqh](https://www.mql5.com/en/articles/download/192/crrdialog.mqh "Download crrdialog.mqh")(13.95 KB)

[visualrrids.mqh](https://www.mql5.com/en/articles/download/192/visualrrids.mqh "Download visualrrids.mqh")(0.8 KB)

[moneyfixedriskext.mqh](https://www.mql5.com/en/articles/download/192/moneyfixedriskext.mqh "Download moneyfixedriskext.mqh")(7.51 KB)

[images.zip](https://www.mql5.com/en/articles/download/192/images.zip "Download images.zip")(159.05 KB)

[interactive\_expert\_mql5\_doc.zip](https://www.mql5.com/en/articles/download/192/interactive_expert_mql5_doc.zip "Download interactive_expert_mql5_doc.zip")(1445.52 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Securing MQL5 code: Password Protection, Key Generators, Time-limits, Remote Licenses and Advanced EA License Key Encryption Techniques](https://www.mql5.com/en/articles/359)
- [MQL5-RPC. Remote Procedure Calls from MQL5: Web Service Access and XML-RPC ATC Analyzer for Fun and Profit](https://www.mql5.com/en/articles/342)
- [Applying The Fisher Transform and Inverse Fisher Transform to Markets Analysis in MetaTrader 5](https://www.mql5.com/en/articles/303)
- [Advanced Adaptive Indicators Theory and Implementation in MQL5](https://www.mql5.com/en/articles/288)
- [Using MetaTrader 5 Indicators with ENCOG Machine Learning Framework for Timeseries Prediction](https://www.mql5.com/en/articles/252)
- [Exposing C# code to MQL5 using unmanaged exports](https://www.mql5.com/en/articles/249)
- [Moving Mini-Max: a New Indicator for Technical Analysis and Its Implementation in MQL5](https://www.mql5.com/en/articles/238)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/2499)**
(7)


![Jake](https://c.mql5.com/avatar/avatar_na2.png)

**[Jake](https://www.mql5.com/en/users/jake)**
\|
21 Aug 2012 at 13:59

Please help with installing this EA, got it working partly...


![Izzatilla Ikramov](https://c.mql5.com/avatar/2014/10/54392D7D-372B.png)

**[Izzatilla Ikramov](https://www.mql5.com/en/users/izzatilla)**
\|
9 Jan 2015 at 05:40

**MetaQuotes:**

Published article [Creating an interactive advisor for semi-automatic trading with a given risk](https://www.mql5.com/en/articles/192):

Author: [investeo](https://www.mql5.com/en/users/investeo)

Generally super! For a long time I have been thinking about creating an assistant-advisor for my manual trading. Part of it has already been implemented.

Thanks to the author for the work done.

![Alexey Volchanskiy](https://c.mql5.com/avatar/2018/8/5B70B603-444A.png)

**[Alexey Volchanskiy](https://www.mql5.com/en/users/vdev)**
\|
9 Jan 2015 at 15:22

Yes, it's massive, I'd love to explore the code, especially in terms of graphics. It's a pity that the video has no sound, it's hard to perceive.

Kudos to the author!

![apirakkamjan](https://c.mql5.com/avatar/avatar_na2.png)

**[apirakkamjan](https://www.mql5.com/en/users/apirakkamjan)**
\|
12 Oct 2019 at 11:20

**Jake:**

Please help with installing this EA, got it working partly...

Put both Images in the same folder with your EA.

![Tobias Johannes Zimmer](https://c.mql5.com/avatar/2022/3/6233327A-D1E7.JPG)

**[Tobias Johannes Zimmer](https://www.mql5.com/en/users/pennyhunter)**
\|
12 Feb 2022 at 17:44

It's amazing how long it's been around and then it's free.


![Alexander Anufrenko: "A danger foreseen is half avoided" (ATC 2010)](https://c.mql5.com/2/0/anufrenko_ava.png)[Alexander Anufrenko: "A danger foreseen is half avoided" (ATC 2010)](https://www.mql5.com/en/articles/535)

The risky development of Alexander Anufrenko (Anufrenko321) had been featured among the top three of the Championship for three weeks. Having suffered a catastrophic Stop Loss last week, his Expert Advisor lost about $60,000, but now once again he is approaching the leaders. In this interview the author of this interesting EA is describing the operating principles and characteristics of his application.

![Vladimir Tsyrulnik: The Essense of my program is improvisation! (ATC 2010)](https://c.mql5.com/2/0/ustas_ava.png)[Vladimir Tsyrulnik: The Essense of my program is improvisation! (ATC 2010)](https://www.mql5.com/en/articles/533)

Vladimir Tsyrulnik is the holder of one of the brightest highs of the current Championship. By the end of the third trading week Vladimir's Expert Advisor was on the sixth position. The IMEX algorithm the Expert Advisor is based on was developed by Vladimir. To learn more about this algorithm, we had an interview with Vladimir.

![The Use of the MQL5 Standard Trade Class libraries in writing an Expert Advisor](https://c.mql5.com/2/0/Trade_Classes_Stdlib_MQL5.png)[The Use of the MQL5 Standard Trade Class libraries in writing an Expert Advisor](https://www.mql5.com/en/articles/138)

This article explains how to use the major functionalities of the MQL5 Standard Library Trade Classes in writing Expert Advisors which implements position closing and modifying, pending order placing and deletion and verifying of Margin before placing a trade. We have also demonstrated how Trade classes can be used to obtain order and deal details.

![Growing Neural Gas: Implementation in MQL5](https://c.mql5.com/2/0/neural_gas_MQL5.png)[Growing Neural Gas: Implementation in MQL5](https://www.mql5.com/en/articles/163)

The article shows an example of how to develop an MQL5-program implementing the adaptive algorithm of clustering called Growing neural gas (GNG). The article is intended for the users who have studied the language documentation and have certain programming skills and basic knowledge in the area of neuroinformatics.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/192&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068324407656511516)

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