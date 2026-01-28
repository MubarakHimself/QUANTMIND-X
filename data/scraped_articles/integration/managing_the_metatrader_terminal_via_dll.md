---
title: Managing the MetaTrader Terminal via DLL
url: https://www.mql5.com/en/articles/1903
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:19:21.393470
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/1903&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071745697000140200)

MetaTrader 4 / Examples


### Task Definition

We have a MetaQuotes ID list containing more than four delivery addresses. As we know, the [SendNotification](https://docs.mql4.com/en/common/sendnotification) function uses only the IDs set in the Notifications tab of the Options window. Thus, you can send push notifications solely to the previously specified IDs no more than four at a time by means of MQL. Let's try to fix that.

The issue can be solved in two ways – we can either develop the push notification delivery function from scratch, or change the terminal settings and use the standard function. The first option is quite time-consuming and lacks universality. Therefore, I have selected the second one. In their turn, the terminal settings can also be changed in various ways. According to my experience, this can be done via the user interface or by substituting values in the process memory. Working with memory looks much better since it allows users to avoid flashing windows. However, it can disrupt the operation of the entire terminal at the slightest mistake. The worst thing that may happen when working via the UI is a disappearance of a window or a button.

In this article, we will look at managing the terminal via the user interface using an auxiliary DLL library. In particular, we will consider changing the settings. Interaction with the terminal will be performed in usual way, which means using its windows and components. No interference with the terminal process will take place. This method can be applied to solve other issues as well.

### 1\. Creating a DLL

Here, we will focus mainly on working with WinAPI. So, let's briefly examine how a dynamic library can be developed in Delphi.

```
library Set_Push;

uses
  Windows,
  messages,
  Commctrl,
  System.SysUtils;

var
   windows_name:string;
   class_name:string;
   Hnd:hwnd;

{$R *.res}
{$Warnings off}
{$hints on}

function FindFunc(h:hwnd; L:LPARAM): BOOL; stdcall;
begin
  ...
end;

function FindFuncOK(h:hwnd; L:LPARAM): BOOL; stdcall;
begin
  ...
end;

function Find_Tab(AccountNumber:integer):Hwnd;
begin
  ...
end;

function Set_Check(AccountNumber:integer):boolean; export; stdcall;
var
  HndButton, HndP:HWnd;
  i:integer;
  WChars:array[0..255] of WideChar;
begin
   ...
end;

function Set_MetaQuotesID(AccountNumber:integer; Str_Set:string):boolean; export; stdcall;
begin
  ...
end;

//--------------------------------------------------------------------------/
Exports Set_Check, Set_MetaQuotesID;

begin
end.
```

As we can see, the Set\_Check and Set\_MetaQuotesID functions are to be exported, while others are intended for internal use. FindFunc looks for a necessary window (described below), while Find\_Tab looks for a necessary tab. Windows, Messages, and Commctrl libraries are enabled for using WinAPI.

**1.1. Applied Tools**

The basic principle of solving this task is using WinAPI in Delphi XE4 environment. С++ may also be used, since WinAPI syntax is almost identical. The search for component names and classes can be performed either using [Spy++](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/dd460725.aspx "https://msdn.microsoft.com/en-us/library/dd460725.aspx") utility included in the **Visual Studio** delivery, or by a simple enumeration described below.

**1.2. Looking for MetaTrader Windows**

Any program window can be found by its title (see Fig. 1).

![Fig. 1. Window title](https://c.mql5.com/2/19/fig1_window_title__2.png)

Fig. 1. Window title

As we can see, the MetaTrader window title contains the account number, while the title itself is changed depending on the selected symbol and timeframe. Thus, the search will be performed only by an account number. We should also find the Options window that appears afterwards. It has a permanent title.

For the first case, we will use the EnumWindows function allowing us to enumerate all available windows. The function for processing the enumerated windows is passed as the EnumWindows parameter. In our case, it is the FindFunc function.

```
function FindFunc(h:hwnd; L:LPARAM): BOOL; stdcall;
var buff: ARRAY [0..255] OF WideChar;
begin
   result:=true;
   Hnd := 0;
   GetWindowText(h, buff, sizeof(buff));
   if((windows_name='') or (pos(windows_name, StrPas(buff))> 0)) then begin
      GetClassName(h, buff, sizeof(buff));
      if ((class_name='') or (pos(class_name, StrPas(buff))> 0)) then begin
         Hnd := h;
         result:=false;
      end;
   end;
end;
```

Let's consider this function in more details. The function header remains unchanged except for the function and variables' names. When a new window is detected, the EnumWindows function calls the specified function and passes the window handle to it. If the specified function returns true, the enumeration continues. Otherwise, it is complete.

Using the received handle, we can examine the window title (GetWindowText) and class name (GetClassName) by copying it to the buffer. Next, we should compare the window title and class with the necessary ones. If there is a match, we remember the handle (which is the most important thing) and exit the enumeration by returning _false_.

Now, let's consider the EnumWindows function call.

```
windows_name:=IntToStr(AccountNumber);
class_name:='MetaTrader';
EnumWindows(@FindFunc, 0);
```

Here we should assign the necessary class and partial window title values. Now, let's call the function for enumerating all available windows. As a result, we receive the main window handle in the _Hnd_ global variable.

Looking ahead, let's examine yet another windows searching function. Since we need to change the terminal settings, we will certainly have to face the new Options window that appears after the appropriate menu option is selected. There is another way to find the window.

```
hnd:=FindWindow(nil, 'Options');
```

Class name and window title are used as the function parameters, while the returned value is a necessary handle or 0 if none found. Unlike the previous case, the function looks for exactly matching names instead of a string occurrence.

**1.3. Working with a Menu**

Like all other components, working with a menu starts after a parent handle (a certain window) is found. Then, we should find corresponding menu item and sub-item and perform a selection.

Please note: **the amount of the terminal menu items changes** depending on whether a chart window is expanded or not (see Fig. 2). Item enumeration starts from 0.

![Fig. 2. Changing the amount of the menu items](https://c.mql5.com/2/19/fig2_menu_items__1.png)

Fig. 2. Changing the amount of the menu items

If the amount of menu items is changed, the Tools item index number is changed as well. Therefore, we should consider the total amount of points using the GetMenuItemCount(Hnd:HMenu) function the menu handle is passed to.

Let's examine the following example:

```
function Find_Tab(AccountNumber:integer; Language:integer):Hwnd;
var
  HndMen :HMenu;
  idMen:integer;
  ...
begin
   ...
   //_____working in the menu________
   HndMen:=GetMenu(Hnd);
   if (GetMenuItemCount(HndMen)=7) then
      HndMen:=GetSubMenu(HndMen,4)
   else
      HndMen:=GetSubMenu(HndMen,5);
   idMen:=GetMenuItemID(HndMen,6);
   if idMen<>0 then begin
      PostMessage(Hnd,WM_COMMAND,idMen,0);
      ...
```

In this example, we find the main menu handle via its parent. Then, we find the appropriate sub-menu by the menu handle. The sub-menu index number is used as the second parameter of the GetSubMenu function. After that, we find the appropriate sub-menu item. To perform a selection, we need to send an appropriate message. After sending the message, we have to wait for the Options window.

```
for i := 0 to 10000 do
   hnd:=FindWindow(nil, 'Options');
```

Setting an infinite loop is not recommended, since it may cause the terminal crash after closing the window even despite the program's fast operation.

**1.4. Searching for Components**

We have obtained the options window, and now we need to address its components, or (using WinAPI terms) child windows. But first, we should find them using the handle. The "child window" term is used for a reason, since we search for them the same way as when looking for windows.

```
windows_name:='ОК';
class_name:='Button';
EnumChildWindows(HndParent, @FindFunc, 0);
```

or

```
Hnd:=FindWindowEx(HndParent, 0, 'Button', 'OK');
```

Thus, we have observed the main examples of searching for the components. At this stage, we have not faced any particular complications apart from changed function names and additional passing of the parent handle. Difficulties usually arise if you need to consider component peculiarities and know component titles or classes, by which the search is performed. In that case, the Spy++ utility can be of help, as well as the enumeration of all parent window components followed by displaying all the values. To achieve this, we need to slightly change the passed function (FindFunc) – set the returned value to _true_ in all cases and save window names and their classes (for example, write them to a file).

Let's examine one of the component search features: **ОК is a system button.** It means that the button text is written in Latin characters in the English Windows OS, while it is written in Cyrillic characters in the Russian version. Therefore, this solution is not universal.

The search is based on the fact that the name length (at least for languages using Latin and Cyrillic characters) consists of two characters. This already makes the library more versatile. The search function for this case looks as follows:

```
function FindFuncOK(h:hwnd; L:LPARAM): BOOL; stdcall;
var buff: ARRAY [0..255] OF WideChar;
begin
   result:=true;
   Hnd := 0;
   GetClassName(h, buff, sizeof(buff));
   if (pos('Button', StrPas(buff))> 0) then begin
      GetWindowText(h, buff, sizeof(buff));
      if(Length(StrPas(buff))=2) then  begin
         Hnd := h;
         result:=false;
      end;
   end;
end;
```

Accordingly, the search for the OK button is performed the following way:

```
EnumChildWindows(HndParent, @FindFuncOK, 0);
```

**1.5. Working with Components**

We should receive the following window as a result of all our actions (Fig. 3):

![Fig. 3. Options window](https://c.mql5.com/2/19/fig3_settings_window__2.png)

Fig. 3. Options window

**TabControl**

The window contains multiple tabs, and we cannot be sure that the required one is selected. The component responsible for the set of tabs is TabControl, or in this case, SysTabControl32, as indicated in its class. Let's search for its handle. The options window is used as its parent:

```
Hnd:=FindWindowEx(Hnd, 0, 'SysTabControl32', nil);
```

Then, we send a tab change message to this component:

```
SendMessage(Hnd, TCM_SETCURFOCUS, 5, 0);
```

In the example above, 5 is an index number of the necessary tab (Notifications). Now, we can search for the necessary tab:

```
Hnd:=GetParent(Hnd);
Hnd:=FindWindowEx(Hnd, 0, '#32770', 'Notifications');
```

The Options window is used as a parent for an active tab. Since we had the TabControl handle, we take the handle of its parent (the window). After that, the search for the required tab is performed. Accordingly, the class of the tab is "#32770".

**CheckBox**

As we can see, the options window has the "Enable Push Notifications" option. Of course, we should not expect that a user has set everything correctly. The component responsible for enabling/disabling has the Button class, and there are messages designed specifically for this type of component.

First, let's search for the component. Notifications tab acts as its parent. If the component is found, check whether notifications are allowed (whether the option is checked or not). If it is not, check it. All actions with the object are performed by sending messages.

```
Hnd:=FindWindowEx(Hnd, 0, 'Button', 'Enable Push Notifications');
if(Hnd<>0) then begin
   if (SendMessage(Hnd,BM_GETCHECK,0,0)<>BST_CHECKED) then
      SendMessage(Hnd,BM_SETCHECK,BST_CHECKED,0);
         ...
```

**Edit**

This component is a field for entering MetaQuotes ID addresses. Its parent is the Notifications tab as well, while the class is Edit. The operation principle is the same – find the component and send a message.

```
Hnd:=FindWindowEx(Hnd, 0, 'Edit', nil);
if (Hnd<>0) then begin
   SendMessage(Hnd, WM_Settext,0,Integer(Str_Set));
```

where Str\_Set is a list of _string_ addresses.

**Button**

Now, let's examine the standard OK button at the bottom of the Options window. This component does not belong to any tab, which means that its parent is the window itself. After completing all necessary actions, we should send a button pressing message to it.

```
EnumChildWindows(HndParent, @FindFuncOK, 0);
I:=GetDlgCtrlID(HndButton);
if I<>0 then begin
   SendMessage(GetParent(HndButton),WM_Command,MakeWParam(I,BN_CLICKED),HndButton);
   ...
```

### 2\. Creating a Script in MQL4

The result of our work is a DLL with the two external Set\_Check and Set\_MetaQuotesID functions that enable sending Push notifications and fill the field with MetaQuotes ID addresses from the list accordingly. If all the terminal windows and components are found in the functions, they return _true_. Now, let's see how they may be used in the script.

```
//+------------------------------------------------------------------+
//|                                                    Send_Push.mq4 |
//|                        Copyright 2015, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
#property show_inputs

#import "Set_Push.dll"
bool Set_Check(int);
bool Set_MetaQuotesID(int,string);
#import

extern string text="test";
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnStart()
  {
   if(StringLen(text)<1)
     {
      Alert("Error: No text to send"); return;
     }
   if(!Set_Check(AccountNumber()))
     {
      Alert("Error: Failed to enable sending push. Check the terminal language"); return;
     }
   string str_id="1C2F1442,2C2F1442,3C2F1442,4C2F1442";
   if(!Set_MetaQuotesID(AccountNumber(),str_id))
     {
      Alert("Error: dll execution error! Possible interference with the process"); return;
     }
   if(!SendNotification(text))
     {
      int Err=GetLastError();
      switch(Err)
        {
         case 4250: Alert("Waiting: Failed to send ", str_id); break;
         case 4251: Alert("Err: Invalid message text ", text); return; break;
         case 4252: Alert("Waiting: Invalid ID list ", str_id); break;
         case 4253: Alert("Err: Too frequent requests! "); return; break;
        }
     }
  }
//+------------------------------------------------------------------+
```

### Conclusion

We have observed the basic principles of managing the terminal windows via DLL, which can enable you to use all the terminal features more efficiently. However, please note that this method should be used only as a last resort in case an issue cannot be solved by conventional methods, since it has several drawbacks, including dependence on the selected terminal language, user intervention and implementation complexity. If misused, it may cause fatal errors or even the program crash.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1903](https://www.mql5.com/ru/articles/1903)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1903.zip "Download all attachments in the single ZIP archive")

[send\_push.mq4](https://www.mql5.com/en/articles/download/1903/send_push.mq4 "Download send_push.mq4")(4.71 KB)

[set\_push.zip](https://www.mql5.com/en/articles/download/1903/set_push.zip "Download set_push.zip")(3.91 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/62144)**
(1)


![jiaming](https://c.mql5.com/avatar/avatar_na2.png)

**[jiaming](https://www.mql5.com/en/users/jiaming)**
\|
13 Feb 2016 at 09:10

Nice article! I would also like to add in the section **1.2. Looking for MetaTrader Windows**, instead of finding the handle using the [account number](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_info_integer "MQL5 documentation: Account Properties"), I would pass it into my DLL from MQL.

You can use the following code in MQL to get the window handle:  **int handle = WindowHandle(Symbol(), Period());**

Then in the DLL, use a function similar to the one below to find the parent handle:

**HWND GetTerminalHandle(int chart\_handle)**

**{**

**HWND child = (HWND)chart\_handle;**

**HWND parent;**

**while (true)**

**{**

**parent = GetParent((HWND)child);**

**if (parent == NULL)**

**break;**

**else**

**child = parent;**

**}**

**return child;**

**}**

![Price Action. Automating the Inside Bar Trading Strategy](https://c.mql5.com/2/19/PA.png)[Price Action. Automating the Inside Bar Trading Strategy](https://www.mql5.com/en/articles/1771)

The article describes the development of a MetaTrader 4 Expert Advisor based on the Inside Bar strategy, including Inside Bar detection principles, as well as pending and stop order setting rules. Test and optimization results are provided as well.

![Drawing Dial Gauges Using the CCanvas Class](https://c.mql5.com/2/19/gg_cases.png)[Drawing Dial Gauges Using the CCanvas Class](https://www.mql5.com/en/articles/1699)

We can find dial gauges in cars and airplanes, in industrial production and everyday life. They are used in all spheres which require quick response to behavior of a controlled value. This article describes the library of dial gauges for MetaTrader 5.

![Using Layouts and Containers for GUI Controls: The CGrid Class](https://c.mql5.com/2/20/avatar.png)[Using Layouts and Containers for GUI Controls: The CGrid Class](https://www.mql5.com/en/articles/1998)

This article presents an alternative method of GUI creation based on layouts and containers, using one layout manager — the CGrid class. The CGrid class is an auxiliary control that acts as a container for other containers and controls using a grid layout.

![Statistical Verification of the Labouchere Money Management System](https://c.mql5.com/2/18/labouchere.png)[Statistical Verification of the Labouchere Money Management System](https://www.mql5.com/en/articles/1800)

In this article, we test the statistical properties of the Labouchere money management system. It is considered to be a less aggressive kind of Martingale, since bets are not doubled, but are raised by a certain amount instead.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=vjgokdblplobnpelihaajcymedyzxxle&ssn=1769192360241976859&ssn_dr=0&ssn_sr=0&fv_date=1769192360&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1903&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Managing%20the%20MetaTrader%20Terminal%20via%20DLL%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176919236054098576&fz_uniq=5071745697000140200&sv=2552)

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