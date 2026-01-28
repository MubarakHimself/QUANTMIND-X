---
title: Multiple indicators on one chart (Part 05): Turning MetaTrader 5 into a RAD system (I)
url: https://www.mql5.com/en/articles/10277
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T20:48:16.316224
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/10277&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5051695630021154014)

MetaTrader 5 / Trading


### Introduction

There are a lot of people who do not know how to program but they are quite creative and have great ideas. However, the lack of programming knowledge prevents them from implementing these ideas. Today we will create our own Chart Trade interface to send market orders or to set up parameters used in pending orders. We will do this without programming, just using the functions that will be inside the Expert Advisor. We are curious, so let's see how it will look on our monitors:

![](https://c.mql5.com/2/44/01__4.png)

You might think, "But how do you do it? I don't know anything about programming, or what I know won't be enough for us to do it." Chart Trade which you see in the image above was created in the MetaTrader 5 platform itself and was designed as shown in the image below:

![](https://c.mql5.com/2/44/02__3.png)

Now that we know what this article is about, we should be full of enthusiasm and ideas for creating our own chart. But we will need to complete a few steps to make it all work. Once the auxiliary code is set up, our creativity will be the only limitation for the design of our own Chart Trade IDE. This article is a continuation of the previous ones, so for a complete and comprehensive understanding, I recommend reading the previous articles in this series.

So, let's get to work.

### Planning

To begin with, you should edit the propertied of the chart that you will be using as an IDE. This is done to reduce potential side effects. The point is that by leaving the chart clean, it will be easier to build and design the Chart Trade interface. So, open the chart properties and set the properties as shown in the figure below.

![](https://c.mql5.com/2/44/03__1.png)![](https://c.mql5.com/2/44/04.1.png)

Thus, the screen will be absolutely clean and free of everything that may interfere with the development of our IDE. Now pay attention to the following explanation. Our IDE will be saved as a settings file, i.e. as a TEMPLATE, so we can use any of the objects provided by MetaTrader 5, but for practical reasons we will only use some of them. For all available objects please see [Types of objects in MetaTrader 5](https://www.mql5.com/en/docs/constants/objectconstants/enum_object).

| Object | Type of coordinates used for positioning | Interesting for IDE |
| --- | --- | --- |
| Text | Date and price | NO |
| Label | X and Y location | YES |
| Button | X and Y location | YES |
| Graph | X and Y location | YES |
| Bitmap | Date and price | NO |
| Bitmap label | X and Y location | YES |
| Edit | X and Y location | YES |
| Event | Only the date is used | NO |
| Rectangle Label | X and Y location | YES |

We are going to use a system that can be located in any region of the screen, which is why it would not be practical to use an object that does not use the X and Y coordinate system for positioning, as such objects can make the IDE look completely different. Therefore, we will limit the system to six objects, which are more than enough to create an interface.

The idea is to arrange objects in a logical order, similar to how you draw something on the screen. We start with creating the background first, and then we lay the objects on top of each other, placing and adjusting the objects as we develop the interface. Here is how it goes:

![](https://c.mql5.com/2/44/01.gif)![](https://c.mql5.com/2/44/02.gif)

![](https://c.mql5.com/2/44/03__1.gif)![](https://c.mql5.com/2/44/04.gif)

It's all very simple, it just takes a little practice to master this way of designing and creating your own IDE. The idea here is very similar to the one used in [RAD](https://en.wikipedia.org/wiki/Rapid_application_development "https://en.wikipedia.org/wiki/Rapid_application_development") programs which are used to create programming interfaces in cases when the user interface development through code can be very complex. It's not that we can't create an interface directly through code. But the use of this method makes further modifications much faster and easier, which is ideal for those who want an interface with their own style.

Once we finish, we might end up with an interface like the one below, or even cooler. But here I tried to use as many objects as possible so that you can try them out. You can create your own preferred interface.

![](https://c.mql5.com/2/44/05.png)

This is the first step in creating our IDE. Now we need to create code that actually supports this interface and makes it functional. Although the simple fact that you can create your own user interface should also be the source of motivation, and this motivation will be embodied in the code.

The next step is to save this interface as a settings file. Now we can save it and use the code from the previous version to display it as a pointer. This means that we will not need to make significant changes to the source code. However, if we wanted to test the possibility of receiving events or sending events to our IDE, we would see that this is not possible. But if the interface was created using objects from MetaTrader 5, why isn't it possible to send and receive events from these objects? The answer to this question is easier to show than to explain. We can check it by adding the following code to the original version of the EA.

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
        switch (id)
        {
                case CHARTEVENT_OBJECT_CLICK:
                        Print(sparam);
                        break;
// .... The rest of the code...
        }
}
```

This code reports the name of the object that receives the click and generates the event. In this case, the event is [CHARTEVENT\_OBJECT\_CLICK](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents). However, the printed message will be the name of the object created by the EA, not the name of the objects in the IDE. This may seem to be a big problem that makes it impossible to use our IDE, but there is a very simple solution: read the settings file and then create the objects as specified in this file. This will create our IDE right on the chart. So, by analyzing the settings file (TPL), we can find the data that we need to use.

| KEY | Description |
| --- | --- |
| <chart> | Starts the settings file |
| </chart> | Ends the settings file |
| <window> | Starts the structure of elements present on the chart. |
| </window> | Ends the structure of elements present on the chart |
| <indicator> | Starts the structure that provides data related to some indicator |
| </indicator> | Ends the structure that provides data related to some indicator |
| <object> | Starts the structure that provides data about some object. |
| </object> | Ends the structure that provides object data. |

This structure looks as follows inside the TPL file.

```
<chart>

.... DATA

<window>

... DATA

<indicator>

... DATA

</indicator>

<object>

... DATA

</object>

</window>
</chart>
```

The part we are interested in is between **<object>** and **</object>**. There can be several such structures, each indicating a unique object. So, first we need to change the location of the file — we should add it to a place from which it can be read. This is the FILES directory. You can change the location, but in any case the file must be inside the FILE tree.

**An important detail**: although the system has received a modification that allows to clear the chart when using the IDE configuration file, ideally you should also have a clean file with the same name in the Profiles\\Templates directory. This minimizes any leftovers which can be present in the default template, as we have seen in previous articles. The main changes are highlighted below:

```
#include <Auxiliar\Chart IDE\C_Chart_IDE.mqh>
//+------------------------------------------------------------------+
class C_TemplateChart : public C_Chart_IDE
{

 .... Other parts from code ....

//+------------------------------------------------------------------+
void AddTemplate(const eTypeChart type, const string szTemplate, int scale, int iSize)
{
        if (m_Counter >= def_MaxTemplates) return;
        if (type == SYMBOL) SymbolSelect(szTemplate, true);
        SetBase(szTemplate, (type == INDICATOR ? _Symbol : szTemplate), scale, iSize);
        if (!ChartApplyTemplate(m_handle, szTemplate + ".tpl")) if (type == SYMBOL) ChartApplyTemplate(m_handle, "Default.tpl");
        if (szTemplate == "IDE") C_Chart_IDE::Create(m_IdSubWin);
        ChartRedraw(m_handle);
}
//+------------------------------------------------------------------+
void Resize(void)
{
#define macro_SetInteger(A, B) ObjectSetInteger(Terminal.Get_ID(), m_Info[c0].szObjName, A, B)
        int x0 = 0, x1, y = (int)(ChartGetInteger(Terminal.Get_ID(), CHART_HEIGHT_IN_PIXELS, m_IdSubWin));
        x1 = (int)((ChartGetInteger(Terminal.Get_ID(), CHART_WIDTH_IN_PIXELS, m_IdSubWin) - m_Aggregate) / (m_Counter > 0 ? (m_CPre == m_Counter ? m_Counter : (m_Counter - m_CPre)) : 1));
        for (char c0 = 0; c0 < m_Counter; x0 += (m_Info[c0].width > 0 ? m_Info[c0].width : x1), c0++)
        {
                macro_SetInteger(OBJPROP_XDISTANCE, x0);
                macro_SetInteger(OBJPROP_XSIZE, (m_Info[c0].width > 0 ? m_Info[c0].width : x1));
                macro_SetInteger(OBJPROP_YSIZE, y);
                if (m_Info[c0].szTemplate == "IDE") C_Chart_IDE::Resize(x0);
        }
        ChartRedraw();
#undef macro_SetInteger
}
//+------------------------------------------------------------------+

... The rest of the code

}
```

Note that we are adding the IDE interface as a new class, and it is inherited by our original class. This means that the functionality of the original class will be extended, and it won't cause any side effects in the original code.

So far this has been the easy part. Now we need to do something more complicated that will support our IDE. First, let's create a message protocol that the system will use. This protocol will allow the system to work as shown below:

![](https://c.mql5.com/2/44/05__1.gif)

Note that we can change the system data, which is currently not possible, but by adding a messaging protocol, it will be possible to make our IDE functional. So, let's define a few things:

| Message | Purpose |
| --- | --- |
| MSG\_BUY\_MARKET | Sends a market BUY order |
| MSG\_SELL\_MARKET | Sends a market SELL order |
| MSG\_LEVERAGE\_VALUE | Leverage data |
| MSG\_TAKE\_VALUE | Trade take profit data |
| MSG\_STOP\_VALUE | Trade stop loss data |
| MSG\_RESULT | Data on the current result of the open position |
| MSG\_DAY\_TRADE | Informs if the trade will be closed at the end of the day or not |

This protocol is a very important step. After defining it, make changes to the settings file. When you open the list of objects, you need to change it so that it looks like this:

![](https://c.mql5.com/2/44/06__1.png)

The interface I am showing will have a list of objects like in the image. Please pay attention to the following fact. The **_NAME_** of the objects corresponds to each of the messages that we are going to use. The names of other objects do not matter, as they will be used to help in the modeling of the IDE, but the objects with the names of the messages will receive and send messages. If you want to use more messages or a different type of messages, just make the necessary changes to the class code, and MetaTrader 5 itself will provide the means to exchange messages between the IDE and the EA code.

But we still need to study the TPL file to learn how to create our object class. Let's now find out how objects are declared inside the TPL file. It is true that we will have less access to the object properties in the TPL file than through programming, since the terminal interface itself gives less access to object properties. But even the access we have will be enough to make our IDE work.

So, inside the TPL file there is the structure that we need: from **<object>** to **</object>**. Based on the data inside the structure it may seem unclear how to find out what type of object it is. But if you take a closer look, you can see that the object type is determined by the **_type_** variable. It takes different values for each of the objects. The below table shows the objects that we want to use:

| The value of the TYPE variable | Referenced object |
| --- | --- |
| 102 | [OBJ\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_label) |
| 103 | [OBJ\_BUTTON](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_button) |
| 106 | [OBJ\_BITMAP\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_bitmap_label) |
| 107 | [OBJ\_EDIT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_edit) |
| 110 | [OBJ\_RECTANGLE\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_rectangle_label) |

Our class is already starting to take shape. Here is the first function code:

```
bool Create(int nSub)
{
        m_CountObject = 0;
        if ((m_fp = FileOpen("Chart Trade\\IDE.tpl", FILE_BIN | FILE_READ)) == INVALID_HANDLE) return false;
        FileReadInteger(m_fp, SHORT_VALUE);

        for (m_CountObject = eRESULT; m_CountObject <= eEDIT_STOP; m_CountObject++) m_ArrObject[m_CountObject].szName = "";
        m_SubWindow = nSub;
        m_szLine = "";
        while (m_szLine != "</chart>")
        {
                if (!FileReadLine()) return false;
                if (m_szLine == "<object>")
                {
                        if (!FileReadLine()) return false;
                        if (m_szLine == "type")
                        {
                                if (m_szValue == "102") if (!LoopCreating(OBJ_LABEL)) return false;
                                if (m_szValue == "103") if (!LoopCreating(OBJ_BUTTON)) return false;
                                if (m_szValue == "106") if (!LoopCreating(OBJ_BITMAP_LABEL)) return false;
                                if (m_szValue == "107") if (!LoopCreating(OBJ_EDIT)) return false;
                                if (m_szValue == "110") if (!LoopCreating(OBJ_RECTANGLE_LABEL)) return false;
                        }
                }
        }
        FileClose(m_fp);
        return true;
}
```

Please note that the first thing to do is open the file in read mode and as a binary file. This is done so as not to miss anything. When using the HEXA editor, the TPL file looks as follows. Note that it starts with a very interesting value.

![](https://c.mql5.com/2/44/07__1.png)

Sounds confusing? In fact, it is not. The file uses the [UTF-16](https://en.wikipedia.org/wiki/UTF-16 "https://en.wikipedia.org/wiki/UTF-16") encoding. We know that data is organized by line, so let's create a function to read the entire line at once. For this purpose, let's write the following code:

```
bool FileReadLine(void)
{
        int utf_16 = 0;
        bool b0 = false;
        m_szLine = m_szValue = "";
        for (int c0 = 0; c0 < 500; c0++)
        {
                utf_16 = FileReadInteger(m_fp, SHORT_VALUE);
                if (utf_16 == 0x000D) { FileReadInteger(m_fp, SHORT_VALUE); return true; } else
                if (utf_16 == 0x003D) b0 = true; else
                if (b0) m_szValue = StringFormat("%s%c", m_szValue, (char)utf_16); else m_szLine = StringFormat("%s%c", m_szLine, (char)utf_16);
                if (FileIsEnding(m_fp)) break;
        }
        return (utf_16 == 0x003E);
}
```

Reading tries to be as efficient as possible, so when we meet an equals sign ( = ), we separate already during the reading, so as not to do this later. The loop limits the [string](https://www.mql5.com/en/docs/basis/types/stringconst) to a maximum of 500 characters, but this value is arbitrary and can be changed if necessary. With each new string found, the function will return providing the contents of the string so that we can proceed with the appropriate analysis.

We will need certain variables to support the message protocol. They are shown in the code below:

```
class C_Chart_IDE
{
        protected:
                enum eObjectsIDE {eRESULT, eBTN_BUY, eBTN_SELL, eCHECK_DAYTRADE, eBTN_CANCEL, eEDIT_LEVERAGE, eEDIT_TAKE, eEDIT_STOP};
//+------------------------------------------------------------------+
#define def_HeaderMSG "IDE_"
#define def_MaxObject eEDIT_STOP + 32
//+------------------------------------------------------------------+
        private :
                int             m_fp,
                                m_SubWindow,
                                m_CountObject;
                string          m_szLine,
                                m_szValue;
                bool            m_IsDayTrade;
                struct st0
                        {
                                string  szName;
                                int     iPosX;
                        }m_ArrObject[def_MaxObject];

// ... The rest of the class code....
```

The **def\_MaxObject** definition indicates the maximum number of objects that we can keep. This number is obtained based on the number of messages plus an extra number of objects that we are going to use. In our case we have the maximum of 40 objects, but it can be changed if necessary. The first 8 objects will be used to send messages between the IDE and MetaTrader 5. The alias of these messages can be seen in the **_eObjectsIDE_** enumeration. It's important to keep this in mind in case you want to expand the system or adapt it for something else.

This is just the first part of the support system. There is another point to pay attention to: the constant which deals with the message system. In fact, the way MQL5 deals with constants can be a little confusing for those who program in C / C++. In C/C++, a constant is declared in the variable declaration itself. In MQL5 the way it is created can make the code a little more complicated. However, you can live with this, since constants are used quite rarely. Shown in bold below is how you can do this.

```
        public  :
                static const string szMsgIDE[];

// ... The rest of the class code....

};
//+------------------------------------------------------------------+
static const string C_Chart_IDE::szMsgIDE[] = {
                                                "MSG_RESULT",
                                                "MSG_BUY_MARKET",
                                                "MSG_SELL_MARKET",
                                                "MSG_DAY_TRADE",
                                                "MSG_CLOSE_POSITION",
                                                "MSG_LEVERAGE_VALUE",
                                                "MSG_TAKE_VALUE",
                                                "MSG_STOP_VALUE"
                                             };
//+------------------------------------------------------------------+
```

The defined constants are exactly the same values used in object names in the interface. The system was designed to be **case insensitive**. You can change this behavior if you want, but I don't recommend doing that.

After completing all these steps, it's time to move on to the next one. So, let's go back to the TPL file. Look at the file fragment below:

![](https://c.mql5.com/2/44/08__1.png)

After defining the type of object to use, we have a series of data that indicates the properties of the object, such as name, position, color, font, and so on. These properties should be passed to internal objects. Since it is a repetitive thing, we can create a general function for this. It will be as follows:

```
bool LoopCreating(ENUM_OBJECT type)
{
#define macro_SetInteger(A, B) ObjectSetInteger(Terminal.Get_ID(), m_ArrObject[c0].szName, A, B)
#define macro_SetString(A, B) ObjectSetString(Terminal.Get_ID(), m_ArrObject[c0].szName, A, B)
        int c0;
        bool b0;
        string sz0 = m_szValue;
        while (m_szLine != "</object>") if (!FileReadLine()) return false; else
        {
                if (m_szLine == "name")
                {
                        b0 = false;
                        StringToUpper(m_szValue);
                        for(c0 = eRESULT; (c0 <= eEDIT_STOP) && (!(b0 = (m_szValue == szMsgIDE[c0]))); c0++);
                        c0 = (b0 ? c0 : m_CountObject);
                        m_ArrObject[c0].szName = StringFormat("%s%04s>%s", def_HeaderMSG, sz0, m_szValue);
                        ObjectDelete(Terminal.Get_ID(), m_ArrObject[c0].szName);
                        ObjectCreate(Terminal.Get_ID(), m_ArrObject[c0].szName, type, m_SubWindow, 0, 0);
                }
                if (m_szLine == "pos_x"                 ) m_ArrObject[c0].iPosX = (int) StringToInteger(m_szValue);
                if (m_szLine == "pos_y"                 ) macro_SetInteger(OBJPROP_YDISTANCE    , StringToInteger(m_szValue));
                if (m_szLine == "size_x"                ) macro_SetInteger(OBJPROP_XSIZE        , StringToInteger(m_szValue));
                if (m_szLine == "size_y"                ) macro_SetInteger(OBJPROP_YSIZE        , StringToInteger(m_szValue));
                if (m_szLine == "offset_x"              ) macro_SetInteger(OBJPROP_XOFFSET      , StringToInteger(m_szValue));
                if (m_szLine == "offset_y"              ) macro_SetInteger(OBJPROP_YOFFSET      , StringToInteger(m_szValue));
                if (m_szLine == "bgcolor"               ) macro_SetInteger(OBJPROP_BGCOLOR      , StringToInteger(m_szValue));
                if (m_szLine == "color"                 ) macro_SetInteger(OBJPROP_COLOR        , StringToInteger(m_szValue));
                if (m_szLine == "bmpfile_on"            ) ObjectSetString(Terminal.Get_ID()     , m_ArrObject[c0].szName, OBJPROP_BMPFILE, 0, m_szValue);
                if (m_szLine == "bmpfile_off"           ) ObjectSetString(Terminal.Get_ID()     , m_ArrObject[c0].szName, OBJPROP_BMPFILE, 1, m_szValue);
                if (m_szLine == "fontsz"                ) macro_SetInteger(OBJPROP_FONTSIZE     , StringToInteger(m_szValue));
                if (m_szLine == "fontnm"                ) macro_SetString(OBJPROP_FONT          , m_szValue);
                if (m_szLine == "descr"                 ) macro_SetString(OBJPROP_TEXT          , m_szValue);
                if (m_szLine == "readonly"              ) macro_SetInteger(OBJPROP_READONLY     , StringToInteger(m_szValue) == 1);
                if (m_szLine == "state"                 ) macro_SetInteger(OBJPROP_STATE        , StringToInteger(m_szValue) == 1);
                if (m_szLine == "border_type"           ) macro_SetInteger(OBJPROP_BORDER_TYPE  , StringToInteger(m_szValue));
        }
        m_CountObject += (b0 ? 0 : (m_CountObject < def_MaxObject ? 1 : 0));
        return true;

#undef macro_SetString
#undef macro_SetInteger
}
```

Each object will get a name and will be stored in the appropriate location, but the highlighted line shows something different. When we create an IDE, it must start at the top left corner of the chart, but this X position is not necessarily the top left corner of the subwindow. This position must correspond to the top left corner of the [OBJ\_CHART](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_chart) object, to which the IDE will be bound. This object is indicated when loading the IDE template, so it can be anywhere inside the subwindow. If this is not corrected, then the IDE will not appear in the correct location. Therefore, save the X value and use it later to display the object in the correct place. The function that renders the IDE correctly is shown below.

The basic information used in the objects is already defined, but if you need to add any other information, just add it to the set of commands and change the property with the appropriate value.

```
void Resize(int x)
{
        for (int c0 = 0; c0 < m_CountObject; c0++)
                ObjectSetInteger(Terminal.Get_ID(), m_ArrObject[c0].szName, OBJPROP_XDISTANCE, x + m_ArrObject[c0].iPosX);
};
```

Before looking at how messages are processed, let's analyze two other equally important functions. The system can receive values from the EA that are received during initialization. These values must be correctly represented and adjusted so that when using Chart Trade, orders could be configured directly in it to either send a market order or a pending order, without the need to call the EA. Both functions are shown below:

```
void UpdateInfos(bool bSwap = false)
{
        int nContract, FinanceTake, FinanceStop;

        nContract       = (int) StringToInteger(ObjectGetString(Terminal.Get_ID(), m_ArrObject[eEDIT_LEVERAGE].szName, OBJPROP_TEXT));
        FinanceTake = (int) StringToInteger(ObjectGetString(Terminal.Get_ID(), m_ArrObject[eEDIT_TAKE].szName, OBJPROP_TEXT));
        FinanceStop = (int) StringToInteger(ObjectGetString(Terminal.Get_ID(), m_ArrObject[eEDIT_STOP].szName, OBJPROP_TEXT));
        m_IsDayTrade = (bSwap ? (m_IsDayTrade ? false : true) : m_IsDayTrade);
        ObjectSetInteger(Terminal.Get_ID(), m_ArrObject[eCHECK_DAYTRADE].szName, OBJPROP_STATE, m_IsDayTrade);
        NanoEA.Initilize(nContract, FinanceTake, FinanceStop, clrNONE, clrNONE, clrNONE, m_IsDayTrade);
}
//+------------------------------------------------------------------+
void InitilizeChartTrade(int nContracts, int FinanceTake, int FinanceStop, color cp, color ct, color cs, bool b1)
{
        NanoEA.Initilize(nContracts, FinanceTake, FinanceStop, cp, ct, cs, b1);
        if (m_CountObject < eEDIT_STOP) return;
        ObjectSetString(Terminal.Get_ID(), m_ArrObject[eEDIT_LEVERAGE].szName, OBJPROP_TEXT, IntegerToString(nContracts));
        ObjectSetString(Terminal.Get_ID(), m_ArrObject[eEDIT_TAKE].szName, OBJPROP_TEXT, IntegerToString(FinanceTake));
        ObjectSetString(Terminal.Get_ID(), m_ArrObject[eEDIT_STOP].szName, OBJPROP_TEXT, IntegerToString(FinanceStop));
        ObjectSetInteger(Terminal.Get_ID(), m_ArrObject[eCHECK_DAYTRADE].szName, OBJPROP_STATE, m_IsDayTrade = b1);
}
```

Please note that the IDE is linked to the order system, so changes made to the system will be reflected in the order system. This way we will not have to change data in the EA like we did before. Now we can do this directly in the IDE or in our Chart Trade - this is done with these two functions mentioned above related to the messaging system.

```
void DispatchMessage(int iMsg, string szArg, double dValue = 0.0)
{
        if (m_CountObject < eEDIT_STOP) return;
        switch (iMsg)
        {
                case CHARTEVENT_CHART_CHANGE:
                        if (szArg == szMsgIDE[eRESULT])
                        {
                                ObjectSetInteger(Terminal.Get_ID(), m_ArrObject[eRESULT].szName, OBJPROP_BGCOLOR, (dValue < 0 ? clrLightCoral : clrLightGreen));
                                ObjectSetString(Terminal.Get_ID(), m_ArrObject[eRESULT].szName, OBJPROP_TEXT, DoubleToString(dValue, 2));
                        }
                        break;
                case CHARTEVENT_OBJECT_CLICK:
                        if (StringSubstr(szArg, 0, StringLen(def_HeaderMSG)) != def_HeaderMSG) return;
                        szArg = StringSubstr(szArg, 9, StringLen(szArg));
                        StringToUpper(szArg);
                        if ((szArg == szMsgIDE[eBTN_SELL]) || (szArg == szMsgIDE[eBTN_BUY])) NanoEA.OrderMarket(szArg == szMsgIDE[eBTN_BUY]);
                        if (szArg == szMsgIDE[eBTN_CANCEL])
                        {
                                NanoEA.ClosePosition();
                                ObjectSetInteger(Terminal.Get_ID(), m_ArrObject[eBTN_CANCEL].szName, OBJPROP_STATE, false);
                        }
                        if (szArg == szMsgIDE[eCHECK_DAYTRADE]) UpdateInfos(true);
                        break;
                case CHARTEVENT_OBJECT_ENDEDIT:
                        UpdateInfos();
                        break;
        }
}
```

And the question arises: Is that all? Yes, it is the message system that allows the MetaTrader 5 platform to interact with the IDE. It is very simple, I must admit, but without this function, the IDE wouldn't work, and it wouldn't be possible to build the system. It may seem a little complicated how to make this work in an EA, but actually thanks to OOP the EA code will remain super simple. What will be a little tricky is to get update the result that will appear in the IDE. Values are updated in the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) function, but for simplicity, I used the data provided in MetaTrader 5, so the function looks like this. This part is the most important — this function is the most requested of all, so it should also be the fastest one.

```
void OnTick()
{
        SubWin.DispatchMessage(CHARTEVENT_CHART_CHANGE, C_Chart_IDE::szMsgIDE[C_Chart_IDE::eRESULT], NanoEA.CheckPosition());
}
```

In other words, with each new quote, a message is sent to the class and the resulting value is updated in the operation. But please do not forget that this function must be well optimized, otherwise we can have serious problems.

### Conclusion

Sometimes it seems impossible to do some things, but I like challenges. And this one, which shows how to make a RAD system inside a platform that was not originally developed for this, was quite interesting. I hope that this system that started with something simple can motivate you to try to explore something new, which few people dare to.

Soon I will add something new to this Expert Advisor, so stay tuned!

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10277](https://www.mql5.com/pt/articles/10277)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10277.zip "Download all attachments in the single ZIP archive")

[EA\_1.04.zip](https://www.mql5.com/en/articles/download/10277/ea_1.04.zip "Download EA_1.04.zip")(3275.47 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Market Simulation (Part 09): Sockets (III)](https://www.mql5.com/en/articles/12673)
- [Market Simulation (Part 08): Sockets (II)](https://www.mql5.com/en/articles/12672)
- [Market Simulation (Part 07): Sockets (I)](https://www.mql5.com/en/articles/12621)
- [Market Simulation (Part 06): Transferring Information from MetaTrader 5 to Excel](https://www.mql5.com/en/articles/11794)
- [Market Simulation (Part 05): Creating the C\_Orders Class (II)](https://www.mql5.com/en/articles/12598)
- [Market Simulation (Part 04): Creating the C\_Orders Class (I)](https://www.mql5.com/en/articles/12589)
- [Market Simulation (Part 03): A Matter of Performance](https://www.mql5.com/en/articles/12580)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/425163)**
(2)


![Guilherme Mendonca](https://c.mql5.com/avatar/2018/9/5B98163A-29AC.jpg)

**[Guilherme Mendonca](https://www.mql5.com/en/users/billy-gui)**
\|
13 May 2022 at 14:48

Congratulations Daniel,

I've been following your articles closely and I'm very grateful to you for sharing your knowledge and code.

I have been an MQL5 programmer for 4 years and your article will be very useful for everyone in the community, especially since this [graphic](https://www.mql5.com/en/articles/2866 "Article: Visualise! MQL5 graphics library as an analogue to the R chart") part (GUI) in MQL5, we have little support and little code base to help us.

Abs,

Guilherme.

![Pau Hean Yap](https://c.mql5.com/avatar/2022/5/627E74B3-4328.jpg)

**[Pau Hean Yap](https://www.mql5.com/en/users/yappauhean-gmail)**
\|
27 Jun 2022 at 15:43

**MetaQuotes:**

New article [Multiple Indicators on One Chart (Part 05): Converting MetaTrader 5 to a RAD System(I)](https://www.mql5.com/en/articles/10277) has been released:

By [Daniel Jose](https://www.mql5.com/en/users/DJ_TLoG_831 "DJ_TLoG_831")

![Learn how to design a trading system by Parabolic SAR](https://c.mql5.com/2/46/why-and-how__5.png)[Learn how to design a trading system by Parabolic SAR](https://www.mql5.com/en/articles/10920)

In this article, we will continue our series about how to design a trading system using the most popular indicators. In this article, we will learn about the Parabolic SAR indicator in detail and how we can design a trading system to be used in MetaTrader 5 using some simple strategies.

![Graphics in DoEasy library (Part 99): Moving an extended graphical object using a single control point](https://c.mql5.com/2/45/MQL5-avatar-doeasy-library3-2__3.png)[Graphics in DoEasy library (Part 99): Moving an extended graphical object using a single control point](https://www.mql5.com/en/articles/10584)

In the previous article, I implemented the ability to move pivot points of an extended graphical object using control forms. Now I am going to implement the ability to move a composite graphical object using a single graphical object control point (form).

![Data Science and Machine Learning (Part 03): Matrix Regressions](https://c.mql5.com/2/48/matrix_regression__1.png)[Data Science and Machine Learning (Part 03): Matrix Regressions](https://www.mql5.com/en/articles/10928)

This time our models are being made by matrices, which allows flexibility while it allows us to make powerful models that can handle not only five independent variables but also many variables as long as we stay within the calculations limits of a computer, this article is going to be an interesting read, that's for sure.

![Multiple indicators on one chart (Part 04): Advancing to an Expert Advisor](https://c.mql5.com/2/45/variety_of_indicators__2.png)[Multiple indicators on one chart (Part 04): Advancing to an Expert Advisor](https://www.mql5.com/en/articles/10241)

In my previous articles, I have explained how to create an indicator with multiple subwindows, which becomes interesting when using custom indicators. This time we will see how to add multiple windows to an Expert Advisor.

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/10277&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5051695630021154014)

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