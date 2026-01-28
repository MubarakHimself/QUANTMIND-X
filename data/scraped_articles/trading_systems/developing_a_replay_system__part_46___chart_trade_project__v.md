---
title: Developing a Replay System (Part 46): Chart Trade Project (V)
url: https://www.mql5.com/en/articles/11737
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:09:14.396387
---

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/11737&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070052775510871848)

MetaTrader 5 / Examples


### Introduction

In the previous article " [Developing a Replay System (Part 45): Chart Trade Project (IV)](https://www.mql5.com/en/articles/11701), I explained how to launch the functionality of the Chart Trade indicator. You may have noticed that in the attachment to that article there were a lot of files to transfer. The risk of making some mistake, such as forgetting to add a file or accidentally deleting something important, makes this method not very attractive.

I know that many people use this form of distribution and storage, but there is a much more suitable way. At least when it comes to distributing and storing executable files.

The method that will be presented here can be very useful, since you can use MetaTrader 5 itself as an excellent assistant, as well as MQL5. Furthermore, it is not that difficult to understand. What I will explain in this article can help you avoid problems with properly saving executable files and everything that is necessary for their correct operation.

### Understanding the idea

Before we begin, we need to understand that there are some problems and limitations in implementing our plans. These limitations arise from the fact that both the MQL5 language and the MetaTrader 5 platform are not intended to be used in certain ways.

Both the language and the platform are designed to be used as a market visualization tool. The fact that we want or try to make them work differently does not necessarily mean that they can't do something.

This is the first of the limitations that exist and that you should be aware of. The next thing to note is that you can add almost anything to executable files. Notice I said **ALMOST**. In theory, you can add almost anything if you know how to add or include it in the executable. Naturally, the way to include information in an executable file is to make it an internal resource of the executable file itself.

I have noticed that many people add images, sounds and similar things, but can we add a template file so that it becomes part of the executable file? After reading the documentation, you will find out that this is not possible. In fact, if you try to do this, the compiler will return an error when trying to compile the program.

```
#define def_IDE_RAD  "Files\\Chart Trade\\IDE_RAD.tpl"
#resource "\\" + def_IDE_RAD;
```

If you try to compile an executable containing the above code, you will see that the compiler will generate an error because the definition refers to a template. This is generally prohibited for an executable file that will be used on the MetaTrader 5 platform.

I must admit that these kinds of restrictions are quite unpleasant, but they can be easily circumvented, although not without costs. There are limitations and problems that need to be solved. Actually you can, and I'll show you how, add a template to the executable compiled by MetaEditor. But the main problem is not how to include the template in the executable file, but how to use this very template.

Why do I want to explain how to do this, isn't it easier to just use things as usual? Well, it is much easier to use everything the way we are used to. However, as I mentioned at the beginning of this article, it is much more practical to have everything included in the executable. Think about having to pack all the necessary files to run a particular function in MetaTrader 5, and transferring them to the machine that will be used to trade. If you forget something, you will have to search for the lost file or files again. You must place them in the correct location in the directory tree so that the application can access them.

In my opinion, this is quite tiring. Many people use the same device to develop and use MetaTrader 5. I would like to tell these people that this is a big mistake, to say the least. **NEVER**, remember **NEVER** use one and the same machine or MetaTrader5 installation for trading and development. By doing this, you can end up causing some kind of glitch or vulnerability in the application that could affect the correct operation of the automatic EA if you use it.

I've seen a lot of strange things happen during the development stage. Sometimes a program or application will simply start interacting with another application running on the chart, causing things to behave very strangely. This may cause mistakes or some unexplainable things. Some people might say it's because I'm constantly testing things. Well, this is true. Sometimes this may actually be caused by testing, but other times it just doesn't make sense.

This is why it is extremely important to have two installations for using MetaTrader 5: one for development and one exclusively for working in the market. Of course, this refers to developers. In any other case, there will be no need for this.

I've thought a lot about whether I should show you how to do certain things. Most people can't understand the basics of MQL5 or how MetaTrader 5 works, so imagine what will happen if they try to understand what I'm about to show you. Some people will think I'm crazy. But in fact, you can compare the attachments in this article and in the previous one, both in functionality and portability. The functionality is the same. As far as portability goes, I think this option is much better. You just need to send one file without worrying any directory structures.

**Important:** While I say not to worry about the directory structure, that only applies to storing the executable in the specific directory it is in. Moving it from one directory to another or even renaming it will cause the system to work incorrectly.

So let's see how this was implemented and how you can use it in your own codes.

### Resources, resources and resources

To use this model, we will need to make some changes to the code. Despite this, the source code of the indicator remained unchanged, so it will not be here in the article. The code in the C\_ChartFloatingRAD class has undergone some minor changes, but since they haven't affected the entire class, I'll focus only on the part where the change actually occurred.

Below you can see this part and the explanation.

```
068. inline void AdjustTemplate(const bool bFirst = false)
069.                    {
070. #define macro_AddAdjust(A) {                   \
071.            (*Template).Add(A, "size_x", NULL); \
072.            (*Template).Add(A, "size_y", NULL); \
073.            (*Template).Add(A, "pos_x", NULL);  \
074.            (*Template).Add(A, "pos_y", NULL);  \
075.                            }
076. #define macro_GetAdjust(A) {                                                                        \
077.            m_Info.Regions[A].x = (int) StringToInteger((*Template).Get(EnumToString(A), "pos_x"));  \
078.            m_Info.Regions[A].y = (int) StringToInteger((*Template).Get(EnumToString(A), "pos_y"));  \
079.            m_Info.Regions[A].w = (int) StringToInteger((*Template).Get(EnumToString(A), "size_x")); \
080.            m_Info.Regions[A].h = (int) StringToInteger((*Template).Get(EnumToString(A), "size_y")); \
081.                            }
082. #define macro_PointsToFinance(A) A * (GetInfoTerminal().VolumeMinimal + (GetInfoTerminal().VolumeMinimal * (m_Info.Leverage - 1))) * GetInfoTerminal().AdjustToTrade
083.
084.                            C_AdjustTemplate *Template;
085.
086.                            if (bFirst)
087.                            {
088.                                    Template = new C_AdjustTemplate(m_Info.szFileNameTemplate = IntegerToString(GetInfoTerminal().ID) + ".tpl", true);
089.                                    for (eObjectsIDE c0 = 0; c0 <= MSG_CLOSE_POSITION; c0++) macro_AddAdjust(EnumToString(c0));
090.                                    AdjustEditabled(Template, true);
091.                            }else Template = new C_AdjustTemplate(m_Info.szFileNameTemplate);
092.                            m_Info.Leverage = (m_Info.Leverage <= 0 ? 1 : m_Info.Leverage);
093.                            m_Info.FinanceTake = macro_PointsToFinance(FinanceToPoints(MathAbs(m_Info.FinanceTake), m_Info.Leverage));
094.                            m_Info.FinanceStop = macro_PointsToFinance(FinanceToPoints(MathAbs(m_Info.FinanceStop), m_Info.Leverage));
095.                            (*Template).Add("MSG_NAME_SYMBOL", "descr", GetInfoTerminal().szSymbol);
096.                            (*Template).Add("MSG_LEVERAGE_VALUE", "descr", IntegerToString(m_Info.Leverage));
097.                            (*Template).Add("MSG_TAKE_VALUE", "descr", DoubleToString(m_Info.FinanceTake, 2));
098.                            (*Template).Add("MSG_STOP_VALUE", "descr", DoubleToString(m_Info.FinanceStop, 2));
099.                            (*Template).Add("MSG_DAY_TRADE", "state", (m_Info.IsDayTrade ? "1" : "0"));
100.                            (*Template).Add("MSG_MAX_MIN", "state", (m_Info.IsMaximized ? "1" : "0"));
101.                            (*Template).Execute();
102.                            if (bFirst)
103.                            {
104.                                    for (eObjectsIDE c0 = 0; c0 <= MSG_CLOSE_POSITION; c0++) macro_GetAdjust(c0);
105.                                    m_Info.Regions[MSG_TITLE_IDE].w = m_Info.Regions[MSG_MAX_MIN].x;
106.                                    AdjustEditabled(Template, false);
107.                            };
108.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YSIZE, (m_Info.IsMaximized ? 210 : m_Info.Regions[MSG_TITLE_IDE].h + 6));
109.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XDISTANCE, (m_Info.IsMaximized ? m_Info.x : m_Info.minx));
110.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YDISTANCE, (m_Info.IsMaximized ? m_Info.y : m_Info.miny));
111.
112.                            delete Template;
113.
114.                            ChartApplyTemplate(m_Info.WinHandle, "/Files/" + m_Info.szFileNameTemplate);
115.                            ChartRedraw(m_Info.WinHandle);
116.
117. #undef macro_PointsToFinance
118. #undef macro_GetAdjust
119. #undef macro_AddAdjust
120.                    }
```

C\_ChartFloatingRAD class code fragment

You may not notice any differences here, but they are there. The difference can be seen in line 88, although it is not so obvious. So if you take the C\_ChartFloatingRAD class code from the previous article and change it in line 88 exactly as shown in this part, you will be able to get the new data model that we are going to use.

You can notice that, unlike the original code, here we only define a string. Why? The reason is that we will no longer use the template as before. We will use a resource. In other words, the template will be embedded into the indicator's executable file. So we no longer need to communicate a lot of data.

However, this change will soon raise another question. How are we going to use a template as a resource if we can't actually compile the template as part of the executable, i.e. include it as a resource? In fact, we can include anything in the executable file. The question is how exactly we should do it.

To understand this, let's look at the code of the C\_AdjustTemplate class to understand why the change was made to line 88 of the C\_ChartFloatingRAD class. The complete code of the C\_AdjustTemplate class is shown below. Since there were more changes made to it, although not as significant, it will be interesting to understand what actually happened.

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "../Auxiliar/C_Terminal.mqh"
005. //+------------------------------------------------------------------+
006. #define def_PATH_BTN "Images\\Market Replay\\Chart Trade"
007. #define def_BTN_BUY  def_PATH_BTN + "\\BUY.bmp"
008. #define def_BTN_SELL def_PATH_BTN + "\\SELL.bmp"
009. #define def_BTN_DT   def_PATH_BTN + "\\DT.bmp"
010. #define def_BTN_SW   def_PATH_BTN + "\\SW.bmp"
011. #define def_BTN_MAX  def_PATH_BTN + "\\MAX.bmp"
012. #define def_BTN_MIN  def_PATH_BTN + "\\MIN.bmp"
013. #define def_IDE_RAD  "Files\\Chart Trade\\IDE_RAD.tpl"
014. //+------------------------------------------------------------------+
015. #resource "\\" + def_BTN_BUY
016. #resource "\\" + def_BTN_SELL
017. #resource "\\" + def_BTN_DT
018. #resource "\\" + def_BTN_SW
019. #resource "\\" + def_BTN_MAX
020. #resource "\\" + def_BTN_MIN
021. #resource "\\" + def_IDE_RAD as string IdeRad;
022. //+------------------------------------------------------------------+
023. class C_AdjustTemplate
024. {
025.    private :
026.            string m_szName[],
027.                   m_szFind[],
028.                   m_szReplace[],
029.                   m_szFileName;
030.            int    m_maxIndex,
031.                   m_FileIn,
032.                   m_FileOut;
033.            bool   m_bFirst;
034. //+------------------------------------------------------------------+
035.    public  :
036. //+------------------------------------------------------------------+
037.            C_AdjustTemplate(const string szFile, const bool bFirst = false)
038.                    :m_maxIndex(0),
039.                     m_szFileName(szFile),
040.                     m_bFirst(bFirst),
041.                     m_FileIn(INVALID_HANDLE),
042.                     m_FileOut(INVALID_HANDLE)
043.                    {
044.                            ResetLastError();
045.                            if (m_bFirst)
046.                            {
047.                                    int handle = FileOpen(m_szFileName, FILE_TXT | FILE_WRITE);
048.                                    FileWriteString(handle, IdeRad);
049.                                    FileClose(handle);
050.                            }
051.                            if ((m_FileIn = FileOpen(m_szFileName, FILE_TXT | FILE_READ)) == INVALID_HANDLE)        SetUserError(C_Terminal::ERR_FileAcess);
052.                            if ((m_FileOut = FileOpen(m_szFileName + "_T", FILE_TXT | FILE_WRITE)) == INVALID_HANDLE) SetUserError(C_Terminal::ERR_FileAcess);
053.                    }
054. //+------------------------------------------------------------------+
055.            ~C_AdjustTemplate()
056.                    {
057.                            FileClose(m_FileIn);
058.                            FileClose(m_FileOut);
059.                            FileMove(m_szFileName + "_T", 0, m_szFileName, FILE_REWRITE);
060.                            ArrayResize(m_szName, 0);
061.                            ArrayResize(m_szFind, 0);
062.                            ArrayResize(m_szReplace, 0);
063.                    }
064. //+------------------------------------------------------------------+
065.            void Add(const string szName, const string szFind, const string szReplace)
066.                    {
067.                            m_maxIndex++;
068.                            ArrayResize(m_szName, m_maxIndex);
069.                            ArrayResize(m_szFind, m_maxIndex);
070.                            ArrayResize(m_szReplace, m_maxIndex);
071.                            m_szName[m_maxIndex - 1] = szName;
072.                            m_szFind[m_maxIndex - 1] = szFind;
073.                            m_szReplace[m_maxIndex - 1] = szReplace;
074.                    }
075. //+------------------------------------------------------------------+
076.            string Get(const string szName, const string szFind)
077.                    {
078.                            for (int c0 = 0; c0 < m_maxIndex; c0++) if ((m_szName[c0] == szName) && (m_szFind[c0] == szFind)) return m_szReplace[c0];
079.
080.                            return NULL;
081.                    }
082. //+------------------------------------------------------------------+
083.            void Execute(void)
084.                    {
085.                            string sz0, tmp, res[];
086.                            int count0 = 0, i0;
087.
088.                            if ((m_FileIn != INVALID_HANDLE) && (m_FileOut != INVALID_HANDLE)) while ((!FileIsEnding(m_FileIn)) && (_LastError == ERR_SUCCESS))
089.                            {
090.                                    sz0 = FileReadString(m_FileIn);
091.                                    if (sz0 == "<object>") count0 = 1;
092.                                    if (sz0 == "</object>") count0 = 0;
093.                                    if (count0 > 0) if (StringSplit(sz0, '=', res) > 1)
094.                                    {
095.                                            if ((m_bFirst) && ((res[0] == "bmpfile_on") || (res[0] == "bmpfile_off")))
096.                                                    sz0 = res[0] + "=\\Indicators\\Replay\\Chart Trade.ex5::" + def_PATH_BTN + res[1];
097.                                            i0 = (count0 == 1 ? 0 : i0);
098.                                            for (int c0 = 0; (c0 < m_maxIndex) && (count0 == 1); i0 = c0, c0++) count0 = (res[1] == (tmp = m_szName[c0]) ? 2 : count0);
099.                                            for (int c0 = i0; (c0 < m_maxIndex) && (count0 == 2); c0++) if ((res[0] == m_szFind[c0]) && (tmp == m_szName[c0]))
100.                                            {
101.                                                    if (StringLen(m_szReplace[c0])) sz0 =  m_szFind[c0] + "=" + m_szReplace[c0];
102.                                                    else m_szReplace[c0] = res[1];
103.                                            }
104.                                    }
105.                                    FileWriteString(m_FileOut, sz0 + "\r\n");
106.                            };
107.                    }
108. //+------------------------------------------------------------------+
109. };
110. //+------------------------------------------------------------------+
```

C\_AdjustTemplate class source code

The code may seem strange to beginners. Although the code is quite extravagant, it saves us from having to port files around as we did in the previous article. Here we include everything in the executable file. But it's not free of course, everything has its price. The main difficulty is that we cannot change the directory or name of the executable file after it is compiled. We can do this, but it will require doing other things, and I will not explain the whole process now.

Look carefully at the definitions in lines 6 to 13. Everything here is basically already familiar to us. Look closely at line 13. This is a template. The same template that we previously used. Now it will no longer be a separate file but will become part of the executable file. Below is how to do this.

In lines 15 to 20 we make definitions as usual. This is done when we want to add images or sounds. But in line 21 we have something different, something that is really rare in codes. This is because we usually do not use aliases in MQL5 programming when we are going to include resources in executable files. You can get an idea of what will happen by looking at the [RESOURCES](https://www.mql5.com/en/docs/runtime/resources) in documentation. But besides this, we still need to sort out some other details. Only in this way can you understand everything.

Aliases are very common in some types of languages. For example, in Visual Basic or VBA (Visual Basic for Applications), which is often used in Excel, these aliases are used to access things in a slightly different way. Usually when we access a resource we use "::" which is a scope resolver. When we use it to access a resource, we operate on the resource definition name. This may seem complicated, but in reality it is much simpler. To understand this, look at the following code:

```
01. #define def_BTN_BUY  "Images\\Market Replay\\Chart Trade\\BUY.bmp"
02. #define def_BTN_SELL "Images\\Market Replay\\Chart Trade\\SELL.bmp"
03. #resource "\\" + def_BTN_BUY
04. #resource "\\" + def_BTN_SELL
05. //+------------------------------------------------------------------+
06. int OnInit()
07. {
08.     long id;
09.     string sz;
10.
11.     ObjectCreate(id = ChartID(), sz = IntegerToString(ObjectsTotal(id)), OBJ_BITMAP_LABEL, 0, 0, 0);
12.     ObjectSetString(id, sz, OBJPROP_BMPFILE, 0, "::" + def_BTN_BUY);
13.     ResourceSave("::" + def_BTN_SELL, "BTN_SELL.bmp");
14.     ObjectSetString(id, sz, OBJPROP_BMPFILE, 1, "\\Files\\BTN_SELL.bmp");
15.
16.     return INIT_SUCCEEDED;
17. }
18. //+------------------------------------------------------------------+
```

Fragment 01 – Example of use

In this fragment 01, we can see and better understand how everything works at the first level of resource usage. Forget about the C\_AdjustTemplate class at this point. Let's first learn how to use resources correctly to make it easier to port our already compiled codes.

In lines 1 and 2, we define two strings. The content pointed by these strings will be compiled and built into the executable by the compiler because of lines 3 and 4. Up to this point, we have nothing complicated, we work in the usual way.

In line 11, we indicate that we are going to create an object. In this case, it is a Bitmap. Again, nothing special. But now comes the first phase, where we will use the resources embedded in the executable file. Look at line 12, where we use "::" to indicate that we are going to use a resource. In this case, it is the resource present in the executable file. We could turn to another program, but to keep things simple, let's first deal with this simpler concept. By reading line 12, the compiler will understand the following:

```
12.     ObjectSetString(id, sz, OBJPROP_BMPFILE, 0, "::Images\\Market Replay\\Chart Trade\\BUY.bmp");
```

However, if you look at the contents of the text strings in the object, you will not see what is shown above, but something else. For now, let's get back to the basics, as it is necessary to understand the essence of things.

So, this line 12 was easy to understand, but what about the following lines? This is where the differences start. You won't usually see these things in the codes, but it's important to understand them to really understand the Chart Trade indicator.

Line 13 will take the specified resource and save it under the specified name in the specified location. The compiler will perceive line 13 as follows:

```
13.     ResourceSave("::Images\\Market Replay\\Chart Trade\\SELL.bmp", "\\Files\\BTN_SELL.bmp");
```

Again, I'm using the most basic concept here. This time things are a little more complicated. Pay attention to the following in the [ResourceSave](https://www.mql5.com/en/docs/common/resourcesave) call:  the resource available in the executable file will be saved as a regular file. This would essentially be equivalent to using the FileMove function. Thus, line 13 can be understood as:

```
FileMove("::Images\\Market Replay\\Chart Trade\\SELL.bmp", 0, "\\Files\\BTN_SELL.bmp", FILE_REWRITE);
```

This is how it will work in practice.

In line 14, instead of accessing the executable's internal resource, we will indirectly use that resource. This is because we saved it to a file in line 13 and are now pointing to it in line 14. Please note the following. Unlike line 12, in line 14 we actually use a physical file present on the disk. It will remain there until it is removed or changed.

The approach shown in lines 13 and 14 is not very common in MQL5 programs. We usually run a resource in an executable file and use it directly in it. This is how we usually do it. However, as I mentioned earlier, the situation is a little more complicated here. But at the same time it's more interesting. You don't actually need to put resources in each of your executables. This would make standardization much more difficult. We can use a very common technique in programming: use a file for resources. Typically, in the case of Windows, it would be located in a DLL. This would allows some sort of standardization.

Something similar can be done in MQL5. But keep in mind that some type of code needs to be standardized. Otherwise, you will generate a monstrous amount of useless data.

So, how to do this in MQL5? It is quite easy. Simply specify the file name at the moment when you are going to use the resource. Remember how I said things are much more complicated? This is exactly what happened. The name of the executable file was ignored. So, suppose you have an executable file called ICONS.LIB, it contains the same images from fragment 01, and this executable file ICONS.LIB is located in the root of the "Indicators" folder. To do the same thing but using ICONS.LIB, we need the following code:

```
01. //+------------------------------------------------------------------+
02. #define def_BTN_BUY  "Images\\Market Replay\\Chart Trade\\BUY.bmp"
03. #define def_BTN_SELL "Images\\Market Replay\\Chart Trade\\SELL.bmp"
04. #define def_LIB      "\\Indicators\\Icons.Lib"
05. //+------------------------------------------------------------------+
06. int OnInit()
07. {
08.     long id;
09.     string sz;
10.
11.     ObjectCreate(id = ChartID(), sz = IntegerToString(ObjectsTotal(id)), OBJ_BITMAP_LABEL, 0, 0, 0);
12.     ObjectSetString(id, sz, OBJPROP_BMPFILE, 0, def_LIB + "::" + def_BTN_BUY);
13.     ResourceSave(def_LIB + "::" + def_BTN_SELL, "BTN_SELL.bmp");
14.     ObjectSetString(id, sz, OBJPROP_BMPFILE, 1, "\\Files\\BTN_SELL.bmp");
15.
16.     return INIT_SUCCEEDED;
17. }
18. //+------------------------------------------------------------------+
```

Fragment 02

Notice that we now define the executable file. Therefore, all the explanations given above apply to this fragment 02. Of course, we also need to understand how the compiler sees the code. So, let's look at the following code.

```
01. //+------------------------------------------------------------------+
02. int OnInit()
03. {
04.     long id;
05.     string sz;
06.
07.     ObjectCreate(id = ChartID(), sz = IntegerToString(ObjectsTotal(id)), OBJ_BITMAP_LABEL, 0, 0, 0);
08.     ObjectSetString(id, sz, OBJPROP_BMPFILE, 0, "\\Indicators\\Icons.Lib::Images\\Market Replay\\Chart Trade\\BUY.bmp");
09.     ResourceSave("\\Indicators\\Icons.Lib::Images\\Market Replay\\Chart Trade\\SELL.bmp", "\\Files\\BTN_SELL.bmp");
10.     ObjectSetString(id, sz, OBJPROP_BMPFILE, 1, "\\Files\\BTN_SELL.bmp");
11.
12.     return INIT_SUCCEEDED;
13. }
14. //+------------------------------------------------------------------+
```

Fragment 02 (Extended code)

You are unlikely to see anyone writing such extended code. This is because maintaining such code is much more labor-intensive. But for the compiler everything will work the same.

Now that I've explained that easier part, let's get back to aliases. When you use aliases here in MQL5, you "force" the compiler to act a little differently. It will use a data compression system to reduce the size of the files. At the same time, it will ignore any direct access to the resource. That is, you will NOT be able to use the definition directly. It is very important that you understand this. You can add anything to the executable, but yoг will **NOT** be able to directly use what has been added.

In fact, to use something that was added to the executable via an alias, you will need to use that alias, not the resource. Seems confusing? Let's try to figure this out.

```
01. #resource "\\Images\\euro.bmp" as bitmap euro[][]
02. #resource "\\Images\\dollar.bmp"
03. //+------------------------------------------------------------------+
04. void Image(string name,string rc,int x,int y)
05. {
06.     ObjectCreate(0, name, OBJ_BITMAP_LABEL, 0, 0, 0);
07.     ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
08.     ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
09.     ObjectSetString(0, name, OBJPROP_BMPFILE, rc);
10. }
11. //+------------------------------------------------------------------+
12. void OnStart()
13. {
14.     for(int x = 0; x < ArrayRange(euro, 1); x++)
15.             euro[ArrayRange(euro, 1) / 2][x] = 0xFFFF0000;
16.     ResourceCreate("euro_icon", euro, ArrayRange(euro, 1), ArrayRange(euro, 0), 0, 0, ArrayRange(euro, 1), COLOR_FORMAT_ARGB_NORMALIZE);
17.     Image("Euro" , "::euro_icon", 10, 40);
18.     Image("USD", "::Images\\dollar.bmp", 15 + ArrayRange(euro,1), 40);
19.     Image("E2", "::Images\\euro.bmp", 20 + ArrayRange(euro, 1) * 2, 40);
20. }
21. //+------------------------------------------------------------------+
```

Source code provided in the documentation

The source code above demonstrates exactly one of the ways to use resources when we use aliases. The alias was defined on line 1. From that moment, we should no longer use the resource name. We should use its alias. So when we execute line 19, we will get an error because we are trying to use the resource name when we should actually be using its alias.

With the code shown above, the alias tells us which structure to use to model the data. It could be any of the possible ones. We could also use tools to convert one data type to another. This way you can do many more things than many people usually imagine. The difference between this code, borrowed from the documentation, and the one I use is precisely the nature of the information.

In this code from the documentation, we use the system to demonstrate an example of using aliases. However, in this case there is no need to save the existing data as resources of the executable file. The reason is that this is data that we can use directly. But what if the resource is a template? According to the rules, MetaTrader 5 does not allow using templates as resources. Therefore, it is necessary to extract the template from the executable file so that it stops being a resource and becomes a regular file.

This can be achieved in the same way as shown in the code in fragment 01. In line 13, we convert the image included in the executable as a resource into a file so that it can be used in the object in line 14.

This works in theory, but in practice it's a little more complicated. If you notice, in line 13 of code 01 we use the resource name and not the alias. The way we work with aliases is a little different from working with resources, but that's not our main problem. Our main problem is that we cannot include templates as resources in executables. Therefore, let's focus on the code of the C\_AdjustTemplate class. This will help you understand how I managed to overcome these two problems: the inability to include templates in executables and the ability to use a template stored in an executable.

So, in line 13 of the C\_AdjustTemplate class, I define the template I'm going to use. This is the same template that appeared in previous articles on the Chart Trade. But notice that in line 21, where I convert the definition to a resource, I don't convert it to use the resource. However, I do this using an alias. This alias is based on the string data type, so all code present in the template file is added to the executable as if it were a long string of characters. However, this line is like a big constant. It is very important to understand this well. In other words, to make it even clearer: the template file will be perceived by the system as a constant, although in fact it is not.

The template file is assumed to be a text string with the **IdeRad** alias inside the executable file. We can start thinking about how to work with it.

The first thing to understand is that we can't use this template with the **IdeRad** alias directly in the **OBJ\_CHART** object. This is impossible. We need to convert this data back into a file. However, the **ResourceSave** function can't handle such a case because the template **IS NOT A RESOURCE**. But we can do something else. That is why we have a code in lines 45 to 50.

Note: I use this method not because it is required, but because I don't want to change the class code that is already working. We could directly read the template content using the IdeRad alias and make the necessary changes. However, this would complicate the logic that has already been created and tested.

So let's figure out what happens in lines 45 to 50. When calling a constructor to manipulate a template, we specify the call level as well as the name of the file being accessed. If this is the first call, line 45 will create the template file. The file will be created in line 47 and closed in line 49. If only these two lines existed, the template would be empty. But line 48 is where the magic happens.

Here we put the entire contents of the line inside the file. What is this line? It is the one in the **IdeRad** variable. Oops, wait a minute. Do you mean to say that we store the template inside the executable file as a resource? To avoid problems, we assign it an alias. Then when we want to restore its contents, we take the contents of that alias and put it into a file. Really? Yes, exactly. Now you may be wondering: Why hasn't anyone thought of doing this before, or showing how to do it? Well, I don't know. Maybe because no one has actually tried to do it. Or maybe no one could just imagine how to do it.

After the file is closed in line 49, the rest of the process is the same as explained earlier. This is because now MetaTrader 5 will have to deal not with a resource, but with a file on the disk.

Finally, there is one small detail that I cannot ignore. Many people may imagine something or try to manipulate the data to understand what is happening when interacting with the Chart Trade indicator. The question is about the buttons. If you run the indicator on a chart, you will not understand how to buttons are accessed. This will only happen if you have little experience in MQL5. You will not find the images that were stated as the buttons anywhere. If you look at the template contents, you will see something like:

> bmpfile\_on=\\Indicators\\Replay\\Chart Trade.ex5::Images\\Market Replay\\Chart Trade\\BUY.bmp
>
> bmpfile\_off=\\Indicators\\Replay\\Chart Trade.ex5::Images\\Market Replay\\Chart Trade\\BUY.bmp

It doesn't make any sense. None if you are just starting to learn MQL5. But if you reproduce the same pattern on a chart, you will notice that it is displayed correctly. How is this possible? Where are the images being referenced located? That is the question. The images are located inside the executable file. That is, the template directly specifies the image that should be used. This is the same as what was seen in the previously presented code and in the extension of fragment 02. The code in question is precisely line 08, which I show again below to make it easier for you to understand:

```
08.     ObjectSetString(id, sz, OBJPROP_BMPFILE, 0, "\\Indicators\\Icons.Lib::Images\\Market Replay\\Chart Trade\\BUY.bmp");
```

Notice how similar the lines are in both the code and the template. This demonstrates that MetaTrader 5 and the MQL5 language have been underexplored. Believe me, MetaTrader 5 and MQL5 can do much more than you can imagine. Even with all the limitations that many people talk about, we can do much, much more than we think. Without resorting to other languages.

### Conclusion

In this article, I showed how to do what many people think is impossible: use templates inside an executable file. Although this knowledge is presented here as something easy to understand, it actually allows you to do much more. Well, using a DLL, we can do the same work shown here much more extensive.

It all depends on the creativity, abilities and personality of the programmer. There are people who say that something cannot be done. Others say there are no means. But there are people who try and achieve success. Don't be one of those people who gives up when faced with a challenge. Be the one who solves problems. My motto is:

_A true professional programmer: Solve a problem while others only see it._

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11737](https://www.mql5.com/pt/articles/11737)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11737.zip "Download all attachments in the single ZIP archive")

[Indicators.zip](https://www.mql5.com/en/articles/download/11737/indicators.zip "Download Indicators.zip")(149.64 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/473282)**
(3)


![Aleksey Vyazmikin](https://c.mql5.com/avatar/2024/6/6678986f-2caa.png)

**[Aleksey Vyazmikin](https://www.mql5.com/en/users/-aleks-)**
\|
18 Sep 2024 at 05:48

Thank you. It was interesting to read. Does this mean you can pass any file (say dll) through string conversion, or are there limitations?


![Daniel Jose](https://c.mql5.com/avatar/2021/1/5FF740FF-26B8.jpg)

**[Daniel Jose](https://www.mql5.com/en/users/dj_tlog_831)**
\|
19 Sep 2024 at 17:27

**Aleksey Vyazmikin [#](https://www.mql5.com/en/forum/473282#comment_54605919):**

Thank you. It was interesting to read. Does this mean you can pass any file (say dll) through string conversion, or are there limitations?

You can even pass files by converting them into strings. However, there is a limitation due to the NULL symbol. Once this symbol terminates what is a string in that governs MQL5 code. Other than that, we have no other limitations. At least as far as I know. 🙂👍

![Donaldo Sande Angiela](https://c.mql5.com/avatar/2024/8/66b22ded-9a11.jpg)

**[Donaldo Sande Angiela](https://www.mql5.com/en/users/donaldoangiela)**
\|
20 Sep 2024 at 09:55

Awesome article! Been wondering how to fix such issues, thanks


![Self Optimizing Expert Advisor With MQL5 And Python (Part IV): Stacking Models](https://c.mql5.com/2/94/Self_Optimizing_Expert_Advisor_With_MQL5_And_Python_Part_IV___LOGO__1.png)[Self Optimizing Expert Advisor With MQL5 And Python (Part IV): Stacking Models](https://www.mql5.com/en/articles/15886)

Today, we will demonstrate how you can build AI-powered trading applications capable of learning from their own mistakes. We will demonstrate a technique known as stacking, whereby we use 2 models to make 1 prediction. The first model is typically a weaker learner, and the second model is typically a more powerful model that learns the residuals of our weaker learner. Our goal is to create an ensemble of models, to hopefully attain higher accuracy.

![MQL5 Wizard Techniques you should know (Part 39): Relative Strength Index](https://c.mql5.com/2/94/MQL5_Wizard_Techniques_you_should_know_Part_39____LOGO__1.png)[MQL5 Wizard Techniques you should know (Part 39): Relative Strength Index](https://www.mql5.com/en/articles/15850)

The RSI is a popular momentum oscillator that measures pace and size of a security’s recent price change to evaluate over-and-under valued situations in the security’s price. These insights in speed and magnitude are key in defining reversal points. We put this oscillator to work in another custom signal class and examine the traits of some of its signals. We start, though, by wrapping up what we started previously on Bollinger Bands.

![Scalping Orderflow for MQL5](https://c.mql5.com/2/94/Scalping_Orderflow_for_MQL5__LOGO2.png)[Scalping Orderflow for MQL5](https://www.mql5.com/en/articles/15895)

This MetaTrader 5 Expert Advisor implements a Scalping OrderFlow strategy with advanced risk management. It uses multiple technical indicators to identify trading opportunities based on order flow imbalances. Backtesting shows potential profitability but highlights the need for further optimization, especially in risk management and trade outcome ratios. Suitable for experienced traders, it requires thorough testing and understanding before live deployment.

![Using PSAR, Heiken Ashi, and Deep Learning Together for Trading](https://c.mql5.com/2/94/Using_PSARc_Heiken_Ashik_and_Deep_Learning_Together_for_Trading__LOGO.png)[Using PSAR, Heiken Ashi, and Deep Learning Together for Trading](https://www.mql5.com/en/articles/15868)

This project explores the fusion of deep learning and technical analysis to test trading strategies in forex. A Python script is used for rapid experimentation, employing an ONNX model alongside traditional indicators like PSAR, SMA, and RSI to predict EUR/USD movements. A MetaTrader 5 script then brings this strategy into a live environment, using historical data and technical analysis to make informed trading decisions. The backtesting results indicate a cautious yet consistent approach, with a focus on risk management and steady growth rather than aggressive profit-seeking.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=kbyhfsctgzovkxnqriwlayxxaobpfrys&ssn=1769184553186425751&ssn_dr=0&ssn_sr=0&fv_date=1769184553&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11737&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20(Part%2046)%3A%20Chart%20Trade%20Project%20(V)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918455310370736&fz_uniq=5070052775510871848&sv=2552)

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