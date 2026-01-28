---
title: Using AutoIt With MQL5
url: https://www.mql5.com/en/articles/10130
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:12:57.597898
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/10130&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071661824878783526)

MetaTrader 5 / Examples


### Introduction

In the article [Managing the MetaTrader Terminal via DLL](https://www.mql5.com/en/articles/1903 "Managing the Metatrader Terminal Via DLL") the author details how to manage Metatrader 4 (MetaTrader 4) by programmatically controlling the application's graphical user interface (gui) with a custom made [dynamically linked library (dll)](https://en.wikipedia.org/wiki/Dynamic-link_library "https://en.wikipedia.org/wiki/Dynamic-link_library"). In this article we will consider something similar but instead of writing a custom dll, we will leverage functionality provided by AutoIt. Using AutoIt we will create a class that enables the automation of tasks that cannot be accomplished with pure MQL5.

These tasks include:

- The ability to add and remove scripts or expert advisors from charts.
- Reading from the terminal's experts and journal tabs in the toolbox window.
- The ability to automatically handle alerts (read alert messages and auto close alert dialogues).
- Finally we will implement the same functionality that was demonstrated in the MetaTrader 4 article, the ability to set the MetaQuotes IDs in the terminals settings.


### AutoIt

[AutoIt](https://www.mql5.com/go?link=https://www.autoitscript.com/site/autoit-tools/ "https://www.autoitscript.com/site/autoit-tools/") is a scripting language for automating the Microsoft Windows graphical user interface. It provides amongst other features window management, direct access to user interface components, and input simulation for both mouse and keyboard strokes. Part of AutoIt is [AutoItX](https://www.mql5.com/go?link=https://documentation.help/AutoItX/introduction.htm "https://documentation.help/AutoItX/introduction.htm"),  a dll which implements some of the features of the AutoIt scripting language. By interfacing with this dll we can endow MQL5 with similar gui scripting capabilities.

**Installation**

AutoIt software is freely available at this [link](https://www.mql5.com/go?link=https://www.autoitscript.com/site/autoit/downloads/ "https://www.autoitscript.com/site/autoit/downloads/"). It functions on all x86 versions of Microsoft Windows up to Windows 10. It is a 32 bit application that ships with various 64 bit components, including a 64 bit version of AutoItX. If you wish to learn more about AutoIt, the help file that is part of the application install, is essential for reading. Before we go any further we need to be aware of the limitations of AutoIt.

**Limitations**

AutoIt is limited in that it can only reliably work with standard Microsoft Windows components as provided by the Win32 API. If a program uses custom made componets, AutoIt  will not work. This is mostly true for software created using cross platform frameworks. We can check if a program's user interface is compatible with AutoIt by using the AutoIt Window Info Tool.  Another limitation to be aware of is brought on by AutoItX only implementing part of what the AutoIt scripting language is capable of. It is possible that a component can be manipulated by AutoIt but the same functionality may not be available via AutoItX.

**AutoIt Window Info Tool**

AutoIt ships with an application called the [AutoIt Window Info Tool](https://www.mql5.com/go?link=https://documentation.help/AutoItX/au3spy.htm "https://documentation.help/AutoItX/au3spy.htm") that is used to get information about application windows.

By dragging the Finder Tool over any part of a target application, we can get the properties of a particular component. These components are refered to as controls. A control can be a button, dropdown menu, or a tab. These are just a few examples, there are many types of controls used to build applications. Each control is associated with a window. An application can be made up of a number of windows. Usually there is a main window, onto which other child windows are attached or docked. If the child windows are attached or docked to the main application window, then all the controls enclosed in those child windows become part of the main application window. When using AutoIt to accurately locate a control, the window the control is associated with is important, whether it is a child window or the main application window.

Looking at the graphic below, we can see the finder tool being dragged over different regions of a Metatrader 5 (MetaTrader 5) application. Take note of the settings of the Window Info tool when Options menu is selected. Freeze, Always On Top and Use Spy++ Control Detection Logic options are ticked.

![](https://c.mql5.com/2/49/WindoInfoTool.gif)

The window tab will show the properties of an application window in focus, here the title and class of the window are listed. These properties can be used to uniquely identify it. Moving to the control tab, we can see the properties of the control, of interest here is the ClassnameNN property. (Note the classname in the AutoIt context refers to the type of a control). This property combines the control type with an instance identifier in the form of a number. Knowing the type of a control is essential as this will determine what AutoIt function calls will work on it.

### AutoItX Integration

To ensure successfull integration with MetaTrader 5 we need to make sure that the terminal can find the required dll during runtime. To achieve this, we need to simply copy the required dll to the Libraries folder of our MetaTrader 5 install. The default AutoIt3 install directory will be in the Program files (x86) folder. The AutoItX folder within it contains all AutoItX related components, including the AutoItX3\_Dll header which lists all the function prototypes exposed in the dll. It is important to get the appropriate dll for your build of MetaTrader 5 [(64 or 32 bit)](https://www.mql5.com/go?link=https://support.microsoft.com/en-us/windows/32-bit-and-64-bit-windows-frequently-asked-questions-c6ca9541-8dce-4d48-0415-94a3faa2e13d "https://support.microsoft.com/en-us/windows/32-bit-and-64-bit-windows-frequently-asked-questions-c6ca9541-8dce-4d48-0415-94a3faa2e13d").

```
#pragma once

///////////////////////////////////////////////////////////////////////////////
//
// AutoItX v3
//
// Copyright (C)1999-2013:
//    - Jonathan Bennett <jon at autoitscript dot com>
//    - See "AUTHORS.txt" for contributors.
//
// This file is part of AutoItX.  Use of this file and the AutoItX DLL is subject
// to the terms of the AutoItX license details of which can be found in the helpfile.
//
// When using the AutoItX3.dll as a standard DLL this file contains the definitions,
// and function declarations required to use the DLL and AutoItX3_DLL.lib file.
//
///////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
#define AU3_API extern "C"
#else
#define AU3_API
#endif

// Definitions
#define AU3_INTDEFAULT        (-2147483647)  // "Default" value for _some_ int parameters (largest negative number)

//
// nBufSize
// When used for specifying the size of a resulting string buffer this is the number of CHARACTERS
// in that buffer, including the null terminator.  For example:
//
// WCHAR szBuffer[10];
// AU3_ClipGet(szBuffer, 10);
//
// The resulting string will be truncated at 9 characters with the the terminating null in the 10th.
//

///////////////////////////////////////////////////////////////////////////////
// Exported functions
///////////////////////////////////////////////////////////////////////////////

#include <windows.h>
AU3_API void WINAPI AU3_Init(void);
AU3_API int AU3_error(void);

AU3_API int WINAPI AU3_AutoItSetOption(LPCWSTR szOption, int nValue);

AU3_API void WINAPI AU3_ClipGet(LPWSTR szClip, int nBufSize);
AU3_API void WINAPI AU3_ClipPut(LPCWSTR szClip);
AU3_API int WINAPI AU3_ControlClick(LPCWSTR szTitle, LPCWSTR szText, LPCWSTR szControl, LPCWSTR szButton, int nNumClicks, int nX = AU3_INTDEFAULT, int nY = AU3_INTDEFAULT);
AU3_API int WINAPI AU3_ControlClickByHandle(HWND hWnd, HWND hCtrl, LPCWSTR szButton, int nNumClicks, int nX = AU3_INTDEFAULT, int nY = AU3_INTDEFAULT);
AU3_API void WINAPI AU3_ControlCommand(LPCWSTR szTitle, LPCWSTR szText, LPCWSTR szControl, LPCWSTR szCommand, LPCWSTR szExtra, LPWSTR szResult, int nBufSize);
AU3_API void WINAPI AU3_ControlCommandByHandle(HWND hWnd, HWND hCtrl, LPCWSTR szCommand, LPCWSTR szExtra, LPWSTR szResult, int nBufSize);
AU3_API void WINAPI AU3_ControlListView(LPCWSTR szTitle, LPCWSTR szText, LPCWSTR szControl, LPCWSTR szCommand, LPCWSTR szExtra1, LPCWSTR szExtra2, LPWSTR szResult, int nBufSize);
AU3_API void WINAPI AU3_ControlListViewByHandle(HWND hWnd, HWND hCtrl, LPCWSTR szCommand, LPCWSTR szExtra1, LPCWSTR szExtra2, LPWSTR szResult, int nBufSize);
AU3_API int WINAPI AU3_ControlDisable(LPCWSTR szTitle, LPCWSTR szText, LPCWSTR szControl);
AU3_API int WINAPI AU3_ControlDisableByHandle(HWND hWnd, HWND hCtrl);
AU3_API int WINAPI AU3_ControlEnable(LPCWSTR szTitle, LPCWSTR szText, LPCWSTR szControl);
AU3_API int WINAPI AU3_ControlEnableByHandle(HWND hWnd, HWND hCtrl);
AU3_API int WINAPI AU3_ControlFocus(LPCWSTR szTitle, LPCWSTR szText, LPCWSTR szControl);
AU3_API int WINAPI AU3_ControlFocusByHandle(HWND hWnd, HWND hCtrl);
AU3_API void WINAPI AU3_ControlGetFocus(LPCWSTR szTitle, LPCWSTR szText, LPWSTR szControlWithFocus, int nBufSize);
AU3_API void WINAPI AU3_ControlGetFocusByHandle(HWND hWnd, LPWSTR szControlWithFocus, int nBufSize);
AU3_API HWND WINAPI AU3_ControlGetHandle(HWND hWnd, LPCWSTR szControl);
AU3_API void WINAPI AU3_ControlGetHandleAsText(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText, LPCWSTR szControl, LPWSTR szRetText, int nBufSize);
AU3_API int WINAPI AU3_ControlGetPos(LPCWSTR szTitle, LPCWSTR szText, LPCWSTR szControl, LPRECT lpRect);
AU3_API int WINAPI AU3_ControlGetPosByHandle(HWND hWnd, HWND hCtrl, LPRECT lpRect);
AU3_API void WINAPI AU3_ControlGetText(LPCWSTR szTitle, LPCWSTR szText, LPCWSTR szControl, LPWSTR szControlText, int nBufSize);
AU3_API void WINAPI AU3_ControlGetTextByHandle(HWND hWnd, HWND hCtrl, LPWSTR szControlText, int nBufSize);
AU3_API int WINAPI AU3_ControlHide(LPCWSTR szTitle, LPCWSTR szText, LPCWSTR szControl);
AU3_API int WINAPI AU3_ControlHideByHandle(HWND hWnd, HWND hCtrl);
AU3_API int WINAPI AU3_ControlMove(LPCWSTR szTitle, LPCWSTR szText, LPCWSTR szControl, int nX, int nY, int nWidth = -1, int nHeight = -1);
AU3_API int WINAPI AU3_ControlMoveByHandle(HWND hWnd, HWND hCtrl, int nX, int nY, int nWidth = -1, int nHeight = -1);
AU3_API int WINAPI AU3_ControlSend(LPCWSTR szTitle, LPCWSTR szText, LPCWSTR szControl, LPCWSTR szSendText, int nMode = 0);
AU3_API int WINAPI AU3_ControlSendByHandle(HWND hWnd, HWND hCtrl, LPCWSTR szSendText, int nMode = 0);
AU3_API int WINAPI AU3_ControlSetText(LPCWSTR szTitle, LPCWSTR szText, LPCWSTR szControl, LPCWSTR szControlText);
AU3_API int WINAPI AU3_ControlSetTextByHandle(HWND hWnd, HWND hCtrl, LPCWSTR szControlText);
AU3_API int WINAPI AU3_ControlShow(LPCWSTR szTitle, LPCWSTR szText, LPCWSTR szControl);
AU3_API int WINAPI AU3_ControlShowByHandle(HWND hWnd, HWND hCtrl);
AU3_API void WINAPI AU3_ControlTreeView(LPCWSTR szTitle, LPCWSTR szText, LPCWSTR szControl, LPCWSTR szCommand, LPCWSTR szExtra1, LPCWSTR szExtra2, LPWSTR szResult, int nBufSize);
AU3_API void WINAPI AU3_ControlTreeViewByHandle(HWND hWnd, HWND hCtrl, LPCWSTR szCommand, LPCWSTR szExtra1, LPCWSTR szExtra2, LPWSTR szResult, int nBufSize);

AU3_API void WINAPI AU3_DriveMapAdd(LPCWSTR szDevice, LPCWSTR szShare, int nFlags, /*[in,defaultvalue("")]*/LPCWSTR szUser, /*[in,defaultvalue("")]*/LPCWSTR szPwd, LPWSTR szResult, int nBufSize);
AU3_API int WINAPI AU3_DriveMapDel(LPCWSTR szDevice);
AU3_API void WINAPI AU3_DriveMapGet(LPCWSTR szDevice, LPWSTR szMapping, int nBufSize);

AU3_API int WINAPI AU3_IsAdmin(void);

AU3_API int WINAPI AU3_MouseClick(/*[in,defaultvalue("LEFT")]*/LPCWSTR szButton, int nX = AU3_INTDEFAULT, int nY = AU3_INTDEFAULT, int nClicks = 1, int nSpeed = -1);
AU3_API int WINAPI AU3_MouseClickDrag(LPCWSTR szButton, int nX1, int nY1, int nX2, int nY2, int nSpeed = -1);
AU3_API void WINAPI AU3_MouseDown(/*[in,defaultvalue("LEFT")]*/LPCWSTR szButton);
AU3_API int WINAPI AU3_MouseGetCursor(void);
AU3_API void WINAPI AU3_MouseGetPos(LPPOINT lpPoint);
AU3_API int WINAPI AU3_MouseMove(int nX, int nY, int nSpeed = -1);
AU3_API void WINAPI AU3_MouseUp(/*[in,defaultvalue("LEFT")]*/LPCWSTR szButton);
AU3_API void WINAPI AU3_MouseWheel(LPCWSTR szDirection, int nClicks);

AU3_API int WINAPI AU3_Opt(LPCWSTR szOption, int nValue);

AU3_API unsigned int WINAPI AU3_PixelChecksum(LPRECT lpRect, int nStep = 1);
AU3_API int WINAPI AU3_PixelGetColor(int nX, int nY);
AU3_API void WINAPI AU3_PixelSearch(LPRECT lpRect, int nCol, /*default 0*/int nVar, /*default 1*/int nStep, LPPOINT pPointResult);
AU3_API int WINAPI AU3_ProcessClose(LPCWSTR szProcess);
AU3_API int WINAPI AU3_ProcessExists(LPCWSTR szProcess);
AU3_API int WINAPI AU3_ProcessSetPriority(LPCWSTR szProcess, int nPriority);
AU3_API int WINAPI AU3_ProcessWait(LPCWSTR szProcess, int nTimeout = 0);
AU3_API int WINAPI AU3_ProcessWaitClose(LPCWSTR szProcess, int nTimeout = 0);

AU3_API int WINAPI AU3_Run(LPCWSTR szProgram, /*[in,defaultvalue("")]*/LPCWSTR szDir, int nShowFlag = SW_SHOWNORMAL);
AU3_API int WINAPI AU3_RunWait(LPCWSTR szProgram, /*[in,defaultvalue("")]*/LPCWSTR szDir, int nShowFlag = SW_SHOWNORMAL);
AU3_API int WINAPI AU3_RunAs(LPCWSTR szUser, LPCWSTR szDomain, LPCWSTR szPassword, int nLogonFlag, LPCWSTR szProgram, /*[in,defaultvalue("")]*/LPCWSTR szDir, int nShowFlag = SW_SHOWNORMAL);
AU3_API int WINAPI AU3_RunAsWait(LPCWSTR szUser, LPCWSTR szDomain, LPCWSTR szPassword, int nLogonFlag, LPCWSTR szProgram, /*[in,defaultvalue("")]*/LPCWSTR szDir, int nShowFlag = SW_SHOWNORMAL);

AU3_API void WINAPI AU3_Send(LPCWSTR szSendText, int nMode = 0);
AU3_API int WINAPI AU3_Shutdown(int nFlags);
AU3_API void WINAPI AU3_Sleep(int nMilliseconds);
AU3_API int WINAPI AU3_StatusbarGetText(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText, /*[in,defaultvalue(1)]*/int nPart, LPWSTR szStatusText, int nBufSize);
AU3_API int WINAPI AU3_StatusbarGetTextByHandle(HWND hWnd, /*[in,defaultvalue(1)]*/int nPart, LPWSTR szStatusText, int nBufSize);

AU3_API void WINAPI AU3_ToolTip(LPCWSTR szTip, int nX = AU3_INTDEFAULT, int nY = AU3_INTDEFAULT);

AU3_API int WINAPI AU3_WinActivate(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText);
AU3_API int WINAPI AU3_WinActivateByHandle(HWND hWnd);
AU3_API int WINAPI AU3_WinActive(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText);
AU3_API int WINAPI AU3_WinActiveByHandle(HWND hWnd);
AU3_API int WINAPI AU3_WinClose(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText);
AU3_API int WINAPI AU3_WinCloseByHandle(HWND hWnd);
AU3_API int WINAPI AU3_WinExists(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText);
AU3_API int WINAPI AU3_WinExistsByHandle(HWND hWnd);
AU3_API int WINAPI AU3_WinGetCaretPos(LPPOINT lpPoint);
AU3_API void WINAPI AU3_WinGetClassList(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText, LPWSTR szRetText, int nBufSize);
AU3_API void WINAPI AU3_WinGetClassListByHandle(HWND hWnd, LPWSTR szRetText, int nBufSize);
AU3_API int WINAPI AU3_WinGetClientSize(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText, LPRECT lpRect);
AU3_API int WINAPI AU3_WinGetClientSizeByHandle(HWND hWnd, LPRECT lpRect);
AU3_API HWND WINAPI AU3_WinGetHandle(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText);
AU3_API void WINAPI AU3_WinGetHandleAsText(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText, LPWSTR szRetText, int nBufSize);
AU3_API int WINAPI AU3_WinGetPos(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText, LPRECT lpRect);
AU3_API int WINAPI AU3_WinGetPosByHandle(HWND hWnd, LPRECT lpRect);
AU3_API DWORD WINAPI AU3_WinGetProcess(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText);
AU3_API DWORD WINAPI AU3_WinGetProcessByHandle(HWND hWnd);
AU3_API int WINAPI AU3_WinGetState(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText);
AU3_API int WINAPI AU3_WinGetStateByHandle(HWND hWnd);
AU3_API void WINAPI AU3_WinGetText(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText, LPWSTR szRetText, int nBufSize);
AU3_API void WINAPI AU3_WinGetTextByHandle(HWND hWnd, LPWSTR szRetText, int nBufSize);
AU3_API void WINAPI AU3_WinGetTitle(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText, LPWSTR szRetText, int nBufSize);
AU3_API void WINAPI AU3_WinGetTitleByHandle(HWND hWnd, LPWSTR szRetText, int nBufSize);
AU3_API int WINAPI AU3_WinKill(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText);
AU3_API int WINAPI AU3_WinKillByHandle(HWND hWnd);
AU3_API int WINAPI AU3_WinMenuSelectItem(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText, LPCWSTR szItem1, LPCWSTR szItem2, LPCWSTR szItem3, LPCWSTR szItem4, LPCWSTR szItem5, LPCWSTR szItem6, LPCWSTR szItem7, LPCWSTR szItem8);
AU3_API int WINAPI AU3_WinMenuSelectItemByHandle(HWND hWnd, LPCWSTR szItem1, LPCWSTR szItem2, LPCWSTR szItem3, LPCWSTR szItem4, LPCWSTR szItem5, LPCWSTR szItem6, LPCWSTR szItem7, LPCWSTR szItem8);
AU3_API void WINAPI AU3_WinMinimizeAll();
AU3_API void WINAPI AU3_WinMinimizeAllUndo();
AU3_API int WINAPI AU3_WinMove(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText, int nX, int nY, int nWidth = -1, int nHeight = -1);
AU3_API int WINAPI AU3_WinMoveByHandle(HWND hWnd, int nX, int nY, int nWidth = -1, int nHeight = -1);
AU3_API int WINAPI AU3_WinSetOnTop(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText, int nFlag);
AU3_API int WINAPI AU3_WinSetOnTopByHandle(HWND hWnd, int nFlag);
AU3_API int WINAPI AU3_WinSetState(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText, int nFlags);
AU3_API int WINAPI AU3_WinSetStateByHandle(HWND hWnd, int nFlags);
AU3_API int WINAPI AU3_WinSetTitle(LPCWSTR szTitle,/*[in,defaultvalue("")]*/ LPCWSTR szText, LPCWSTR szNewTitle);
AU3_API int WINAPI AU3_WinSetTitleByHandle(HWND hWnd, LPCWSTR szNewTitle);
AU3_API int WINAPI AU3_WinSetTrans(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText, int nTrans);
AU3_API int WINAPI AU3_WinSetTransByHandle(HWND hWnd, int nTrans);
AU3_API int WINAPI AU3_WinWait(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText, int nTimeout = 0);
AU3_API int WINAPI AU3_WinWaitByHandle(HWND hWnd, int nTimeout);
AU3_API int WINAPI AU3_WinWaitActive(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText, int nTimeout = 0);
AU3_API int WINAPI AU3_WinWaitActiveByHandle(HWND hWnd, int nTimeout);
AU3_API int WINAPI AU3_WinWaitClose(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText, int nTimeout = 0);
AU3_API int WINAPI AU3_WinWaitCloseByHandle(HWND hWnd, int nTimeout);
AU3_API int WINAPI AU3_WinWaitNotActive(LPCWSTR szTitle, /*[in,defaultvalue("")]*/LPCWSTR szText, int nTimeout);
AU3_API int WINAPI AU3_WinWaitNotActiveByHandle(HWND hWnd, int nTimeout = 0);

///////////////////////////////////////////////////////////////////////////////
```

Any MetaTrader 5 program that uses the AutoItX library will first have to import its functions. For convenience we will create an include file autoIt.mqh whose sole purpose is to import all the function prototypes exposed by the library.

AutoIt is strictly a Microsoft Windows Operating System tool that leverages the Windows [API](https://en.wikipedia.org/wiki/API "https://en.wikipedia.org/wiki/API"). This is the reason why the AuotItX library makes extensive use of Win32 API specific data types. To ensure compatibility with MQL5 we could implement these data types ourselves in MQL5 but this is not necessary. Instead we can use the windef.mqh file which is part of MetaQuotes' efforts to integrate the Win32 API in MQL5. The file contains most data type definitions used in the Windows api.

We will include windef.mqh in our autoit.mqh file that will contain all the function prototypes imported from the dll. The autoit.mqh code file now looks as follows:

```
//+------------------------------------------------------------------+
//|                                                       autoIt.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                                             https://www.mql5.com  |
//+------------------------------------------------------------------+
#include <WinAPI\windef.mqh>

#define LPPOINT POINT&
#define LPRECT  RECT&
#define WORD    ushort
#define DWORD   int
#define HWND    int

#import "AutoItX3_x64.dll"
void AU3_Init(void);
int AU3_error(void);
int AU3_AutoItSetOption(string, int);
void AU3_ClipGet(string&, int);
void AU3_ClipPut(string);
int AU3_ControlClick(string, string, string, string, int, int, int);
int AU3_ControlClickByHandle(HWND, HWND, string, int, int, int);
void AU3_ControlCommand(string, string, string, string, string, string&, int);
void AU3_ControlCommandByHandle(HWND, HWND, string, string, string&, int);
void AU3_ControlListView(string, string, string, string, string, string, string&, int);
void AU3_ControlListViewByHandle(HWND, HWND, string, string, string, string&, int);
int AU3_ControlDisable(string, string, string);
int AU3_ControlDisableByHandle(HWND, HWND);
int AU3_ControlEnable(string, string, string);
int AU3_ControlEnableByHandle(HWND, HWND);
int AU3_ControlFocus(string, string, string);
int AU3_ControlFocusByHandle(HWND, HWND);
void AU3_ControlGetFocus(string, string, string&, int);
void AU3_ControlGetFocusByHandle(HWND, string&, int);
HWND AU3_ControlGetHandle(HWND, string);
void AU3_ControlGetHandleAsText(string, string, string, string&, int);
int AU3_ControlGetPos(string, string, string, LPRECT);
int AU3_ControlGetPosByHandle(HWND, HWND, LPRECT);
void AU3_ControlGetText(string, string, string, string&, int);
void AU3_ControlGetTextByHandle(HWND, HWND, string&, int);
int AU3_ControlHide(string, string, string);
int AU3_ControlHideByHandle(HWND, HWND);
int AU3_ControlMove(string, string, string, int, int, int, int);
int AU3_ControlMoveByHandle(HWND, HWND, int, int, int, int);
int AU3_ControlSend(string, string, string, string, int);
int AU3_ControlSendByHandle(HWND, HWND, string, int);
int AU3_ControlSetText(string, string, string, string);
int AU3_ControlSetTextByHandle(HWND, HWND, string);
int AU3_ControlShow(string, string, string);
int AU3_ControlShowByHandle(HWND, HWND);
void AU3_ControlTreeView(string, string, string, string, string, string, string&, int);
void AU3_ControlTreeViewByHandle(HWND, HWND, string, string, string, string&, int);
void AU3_DriveMapAdd(string,  string, int, string, string, string&, int);
int AU3_DriveMapDel(string);
void AU3_DriveMapGet(string, string&, int);
int AU3_IsAdmin(void);
int AU3_MouseClick(string, int, int, int, int);
int AU3_MouseClickDrag(string, int, int, int, int, int);
void AU3_MouseDown(string);
int AU3_MouseGetCursor(void);
void AU3_MouseGetPos(LPPOINT);
int AU3_MouseMove(int, int, int);
void AU3_MouseUp(string);
void AU3_MouseWheel(string, int);
int AU3_Opt(string, int);
unsigned int AU3_PixelChecksum(LPRECT, int);
int AU3_PixelGetColor(int, int);
void AU3_PixelSearch(LPRECT, int, int, int, LPPOINT);
int AU3_ProcessClose(string);
int AU3_ProcessExists(string);
int AU3_ProcessSetPriority(string, int);
int AU3_ProcessWait(string, int);
int AU3_ProcessWaitClose(string, int);
int AU3_Run(string, string, int);
int AU3_RunWait(string, string, int);
int AU3_RunAs(string, string, string, int, string, string, int);
int AU3_RunAsWait(string, string, string, int, string, string, int);
void AU3_Send(string, int);
int AU3_Shutdown(int);
void AU3_Sleep(int);
int AU3_StatusbarGetText(string, string, int, string&, int);
int AU3_StatusbarGetTextByHandle(HWND, int, string&, int);
void AU3_ToolTip(string, int, int);
int AU3_WinActivate(string, string);
int AU3_WinActivateByHandle(HWND);
int AU3_WinActive(string, string);
int AU3_WinActiveByHandle(HWND);
int AU3_WinClose(string, string);
int AU3_WinCloseByHandle(HWND);
int AU3_WinExists(string, string);
int AU3_WinExistsByHandle(HWND);
int AU3_WinGetCaretPos(LPPOINT);
void AU3_WinGetClassList(string, string, string&, int);
void AU3_WinGetClassListByHandle(HWND, string&, int);
int AU3_WinGetClientSize(string, string, LPRECT);
int AU3_WinGetClientSizeByHandle(HWND, LPRECT);
HWND AU3_WinGetHandle(string, string);
void AU3_WinGetHandleAsText(string, string, string&, int);
int AU3_WinGetPos(string, string, LPRECT);
int AU3_WinGetPosByHandle(HWND, LPRECT);
DWORD AU3_WinGetProcess(string, string);
DWORD AU3_WinGetProcessByHandle(HWND);
int AU3_WinGetState(string, string);
int AU3_WinGetStateByHandle(HWND);
void AU3_WinGetText(string, string, string&, int);
void AU3_WinGetTextByHandle(HWND, string&, int);
void AU3_WinGetTitle(string, string, string&, int);
void AU3_WinGetTitleByHandle(HWND, string&, int);
int AU3_WinKill(string, string);
int AU3_WinKillByHandle(HWND);
int AU3_WinMenuSelectItem(string, string, string, string, string, string, string, string, string, string);
int AU3_WinMenuSelectItemByHandle(HWND, string, string, string, string, string, string, string, string);
void AU3_WinMinimizeAll();
void AU3_WinMinimizeAllUndo();
int AU3_WinMove(string, string, int, int, int, int);
int AU3_WinMoveByHandle(HWND, int, int, int, int);
int AU3_WinSetOnTop(string, string, int);
int AU3_WinSetOnTopByHandle(HWND, int);
int AU3_WinSetState(string, string, int);
int AU3_WinSetStateByHandle(HWND, int);
int AU3_WinSetTitle(string, string, string);
int AU3_WinSetTitleByHandle(HWND, string);
int AU3_WinSetTrans(string, string, int);
int AU3_WinSetTransByHandle(HWND, int);
int AU3_WinWait(string, string, int);
int AU3_WinWaitByHandle(HWND, int);
int AU3_WinWaitActive(string, string, int);
int AU3_WinWaitActiveByHandle(HWND, int);
int AU3_WinWaitClose(string, string, int);
int AU3_WinWaitCloseByHandle(HWND, int);
int AU3_WinWaitNotActive(string, string, int);
int AU3_WinWaitNotActiveByHandle(HWND, int);
#import

//+------------------------------------------------------------------+
```

**Usage preliminaries**

Using the library in MQL5 requires initialization of the dll by first calling the  AU3\_Init function. This should be done before any of the other imported functions are called. The function signatures are analogous to those used in the AutoIt scripting language, which means they have similar function parameters and return types. Familiarity with the AutoIt scripting language is necessary in order to understand how the functions work, again all this information is available in the help file that is part of the application install.

The functions that return a value, return positive numerical values, usually 1 on success and 0 on failure. The void functions output to either a string or struct reference. For these types of functions you can call the AU3\_error() function to confirm whether an error has occured. A  function will also simply output either a string with "0" or an empty struct when the function fails. Functions that output string references also specify a corresponding buffer size parameter. This means the buffer length for the string passed by reference should be explicitly set before calling the function. Otherwise if the string  buffer length is not set, an error will be flagged. If the allocated size of the buffer is insufficient, the function will output the characters that fit in the specified space, leaving out the rest, producing a truncated string. There is no way to know the amount of space required for output, so it is necessary to be aware of this quirk. To set the buffer length for a string, we can use the built in MQL5 function [StringInit](https://www.mql5.com/en/docs/strings/stringinit). There are some functions that have ...ByHandle suffix in their names, for exampleAU3\_WinCloseByHandle(). This function does the same thing as AU3\_WinClose(), the difference is that the ByHandle suffixed function works by identifying a control or window by its handle. Using these functions is preferable as it helps with debugging, it will be easier to find errors relating to identification of the correct window or control.

**An initial example**

The example below demonstrates the use of AutoItX functions in MQL5. In the script below, we will use the AU3\_WinGetHandle function to get the handle of the terminal's main window. The terminal can be uniquely identified by the active account number displayed in the window title bar.

```
//+------------------------------------------------------------------+
//|                                         TerminalUIComponents.mq5 |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                                             https://www.mql5.com  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://www.mql5.com "
#property version   "1.00"
#include<autoIt.mqh>

string sbuffer;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   AU3_Init();//initialize the library

   if(!StringInit(sbuffer,1000))// set buffer length for string
      Print("Failed to set string bufferlength with error - "+string(GetLastError()));

   HWND window_handle=AU3_WinGetHandle(IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN)),"");// get the terminal's handle

   if(!window_handle)
      Print("Could not find main app window. Error is "+string(AU3_error()));

   AU3_WinGetClassListByHandle(window_handle,sbuffer,1000);// get classnames of all user interface componets of the terminal

   Print(sbuffer);

  }
//+------------------------------------------------------------------+
```

With the main window handle retrieved we can get more information about the gui components we can interact with by calling AU3\_WinGetClassListByHandle function. It will return a list of strings that are the classnames of all the user interface components associated with the supplied window handle.

The result of running the script is shown below

![TerminalUIComponents result](https://c.mql5.com/2/49/TerminalUIComponents.PNG)

Before we move on to some more robust examples, let's create a class that wraps most of the imported AutoIt functions.

### **CAutoit class**

CAutoit will be the base class from which other classes that use the AutoItX library will be derived. It will have a single static property that specifies whether the initialization library function AU3\_Init() has been called or not.

The only other property m\_buffer\_size determines the buffer length of the strings used in the method calls.

```
//+------------------------------------------------------------------+
//|                                                   autoitbase.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://www.mql5.com "
#include<autoIt.mqh>

enum ENUM_WINDOW_STATE
  {
   SW_HIDE=0,
   SW_SHOW,
   SW_MINIMIZE,
   SW_MAXIMIZE,
   SW_RESTORE
  };

//+------------------------------------------------------------------+
//| Class CAutoIt.                                                   |
//| Purpose:  class for working with autoit.                         |
//+------------------------------------------------------------------+

class CAutoIt
  {
private:
   int               m_buffer_size;                                     //string buffer size
   // --- static property common to all object instances
   static bool       m_initialized;                              //flag for checking initialization state

   //--- private methods
   void              ResetBuffer(string &buffer)      { StringFill(buffer,0); }

public:
   //--- constructor, destructor
                     CAutoIt(void);
                    ~CAutoIt(void);

   void              Init(void);
   //--- static method
   static bool       IsInitialized(void)             { return(m_initialized);     }
   //--- general purpose methods
   int               Error(void)                           { return (AU3_error());      }
   void              Sleep(const int milliseconds)         { AU3_Sleep(milliseconds);   }
   void              SetBufferLen(const int n_bufferlen)   { m_buffer_size=(n_bufferlen>0)?n_bufferlen:1000; }
   int               GetBufferLen(void)                    { return(m_buffer_size);     }

protected:
   int               Run(const string exefile,const string path,const ENUM_WINDOW_STATE w_state) { return(AU3_Run(exefile,path,int(w_state)));  }
   //--- system clipboard manipulation
   void              ClipGet(string & out);
   void              ClipPut(string copy);
   //---
   void              ControlPosition(HWND window_handle,HWND control_handle, LPRECT rect) { AU3_ControlGetPosByHandle(window_handle,control_handle, rect);}
   //--- methods for simulating clicks
   int               ControlRightClick(HWND window_handle, HWND control_handle,int number_of_clicks=1,int x=1,int y=1);
   int               ControlLeftClick(HWND window_handle, HWND control_handle,int number_of_clicks=1,int x=1,int y=1);
   //--- general Control Command method
   bool              ControlCommand(HWND window_handle, HWND control_handle, string command, string command_option);

   //--- methods for interacting with comboboxes, buttons,radio buttons
   string            GetCurrentComboBoxSelection(HWND window_handle, HWND control_handle);
   bool              IsControlEnabled(HWND window_handle, HWND control_handle);
   bool              IsControlVisible(HWND window_handle, HWND control_handle);
   int               EnableControl(HWND window_handle, HWND control_handle) { return(AU3_ControlEnableByHandle(window_handle,control_handle));}
   int               ShowControl(HWND window_handle, HWND control_handle)  { return(AU3_ControlShowByHandle(window_handle,control_handle));  }
   int               ControlFocus(HWND window_handle,HWND control_handle)  { return(AU3_ControlFocusByHandle(window_handle,control_handle)); }
   bool              IsButtonChecked(HWND window_handle, HWND control_handle);
   bool              CheckButton(HWND window_handle, HWND control_handle);
   bool              UnCheckButton(HWND window_handle, HWND control_handle);
   //--- methods for interacting with system32tab control
   string            GetCurrentTab(HWND window_handle, HWND control_handle);
   bool              TabRight(HWND window_handle, HWND control_handle);
   bool              TabLeft(HWND window_handle, HWND control_handle);
   long              TotalTabs(HWND window, HWND systab);
   //--- methods for interacting with syslistview32 control
   bool              ControlListView(HWND window_handle, HWND control_handle, string command, string command_option1, string command_option2);

   long              GetListViewItemCount(HWND window_handle, HWND control_handle);
   long              FindListViewItem(HWND window_handle, HWND control_handle, string find_item,string sub_item);
   string            GetSelectedListViewItem(HWND window_handle, HWND control_handle);
   long              GetSelectedListViewCount(HWND window_handle, HWND control_handle);
   long              GetListViewSubItemCount(HWND window_handle, HWND control_handle);
   string            GetListViewItemText(HWND window_handle, HWND control_handle,string item_index,string sub_item_index);
   bool              IsListViewItemSelected(HWND window_handle, HWND control_handle, string item_index);
   bool              SelectListViewItem(HWND window_handle, HWND control_handle, string from_item_index,string to_item_index);
   bool              SelectAllListViewItems(HWND window_handle, HWND control_handle);
   bool              ClearAllListViewItemSelections(HWND window_handle, HWND control_handle);
   bool              ViewChangeListView(HWND window_handle, HWND control_handle);
   //--- general methods for various types of controls
   HWND              ControlGetHandle(HWND window_handle, string control_id);
   string            ControlGetText(HWND window_handle, HWND control_handle);
   int               ControlSetText(HWND window_handle,HWND control_handle,string keys)   { return(AU3_ControlSetTextByHandle(window_handle,control_handle,keys));}
   int               ControlSend(HWND window_handle, HWND control_handle, string keys, int mode);
   bool              SetFocus(HWND window_handle,HWND control_handle) { return(AU3_ControlFocusByHandle(window_handle,control_handle)>0); }
   //--- methods for interacting with systreeview32 control
   bool              ControlTreeView(HWND window_handle, HWND control_handle, string command, string command_option1, string command_option2);
   long              GetTreeViewItemCount(HWND window_handle, HWND control_handle, string item);
   string            GetSelectedTreeViewItem(HWND window_handle, HWND control_handle);
   string            GetTreeViewItemText(HWND window_handle, HWND control_handle,string item);
   bool              SelectTreeViewItem(HWND window_handle, HWND control_handle,string item);
   //--- general methods for application windows, subwindows and dialogues
   int               WinClose(string window_title, string window_text) { return(AU3_WinClose(window_title,window_text)); }
   int               WinClose(HWND window_handle)                        { return(AU3_WinCloseByHandle(window_handle));    }
   string            WinGetText(HWND window_handle);
   int               WinMenuSelectItem(HWND window_handle, string menu_name_1, string menu_name_2, string menu_name_3, string menu_name_4, string menu_name_5, string menu_name_6, string menu_name_7, string menu_name_8);
   int               WinSetState(HWND window_handle, ENUM_WINDOW_STATE new_state);
   string            WinGetTitle(HWND window_handle);
   ENUM_WINDOW_STATE WinGetState(HWND window_handle)         { return((ENUM_WINDOW_STATE)AU3_WinGetStateByHandle(window_handle)); }
   HWND              WinGetHandle(string window_title, string window_text);
   void              WinGetPosition(HWND window_handle, LPRECT winpos)   { AU3_WinGetPosByHandle(window_handle,winpos);                       }
   void              WinClientSize(HWND window_handle, LPRECT winsize)    { AU3_WinGetClientSizeByHandle(window_handle,winsize);               }
   int               WinExists(HWND window_handle)                       { return(AU3_WinExistsByHandle(window_handle));                      }
   int               WinExists(string window_title, string window_text)   { return(AU3_WinExists(window_title,window_text));                   }
  };

bool CAutoIt::m_initialized=false;
//+------------------------------------------------------------------+
//| Constructor without parameters                                   |
//+------------------------------------------------------------------+
CAutoIt::CAutoIt(void): m_buffer_size(1000)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CAutoIt::~CAutoIt(void)
  {
  }

//+------------------------------------------------------------------+
//|Initializes AutoIt library                                        |
//+------------------------------------------------------------------+

void CAutoIt::Init(void)
  {
   if(!m_initialized)
     {
      AU3_Init();
      m_initialized=true;
     }

  }

//+------------------------------------------------------------------+
//|Reads and outputs textual contents of system clipboard            |
//+------------------------------------------------------------------+
void CAutoIt::ClipGet(string &out)
  {
   if(StringBufferLen(out)<m_buffer_size)
      StringInit(out,m_buffer_size);

   ResetBuffer(out);

   AU3_ClipGet(out,m_buffer_size);

  }
//+------------------------------------------------------------------+
//|Writes text to system clipboard                                   |
//+------------------------------------------------------------------+
void CAutoIt::ClipPut(string copy)
  {
   AU3_ClipPut(copy);
  }
//+------------------------------------------------------------------+
//|Simulates a left click                                            |
//+------------------------------------------------------------------+
int CAutoIt::ControlLeftClick(int window_handle,int control_handle,int number_of_clicks=1,int x=1,int y=1)
  {
   return(AU3_ControlClickByHandle(window_handle,control_handle,"left",number_of_clicks,x,y));
  }
//+------------------------------------------------------------------+
//|Simulates a right click                                           |
//+------------------------------------------------------------------+
int CAutoIt::ControlRightClick(int window_handle,int control_handle,int number_of_clicks=1,int x=1,int y=1)
  {
   return(AU3_ControlClickByHandle(window_handle,control_handle,"right",number_of_clicks,x,y));
  }
//+------------------------------------------------------------------+
//|Sends a command to a control                                      |
//+------------------------------------------------------------------+
bool CAutoIt::ControlCommand(int window_handle,int control_handle,string command,string command_option)
  {
   string m_buffer;
   StringInit(m_buffer,m_buffer_size);

   AU3_ControlCommandByHandle(window_handle,control_handle,command,command_option,m_buffer,m_buffer_size);

   if(StringFind(m_buffer,"0")>=0)
      return(false);

   return(true);
  }
//+------------------------------------------------------------------+
//|Retrieves text of currently selected option of a Combobox Control |
//+------------------------------------------------------------------+
string CAutoIt::GetCurrentComboBoxSelection(int window_handle,int control_handle)
  {
   string m_buffer;
   StringInit(m_buffer,m_buffer_size);

   AU3_ControlCommandByHandle(window_handle,control_handle,"GetCurrentSelection","",m_buffer,m_buffer_size);

   return(m_buffer);

  }
//+------------------------------------------------------------------+
//|Checks if a button is clickable                                   |
//+------------------------------------------------------------------+
bool CAutoIt::IsControlEnabled(int window_handle,int control_handle)
  {
   return(ControlCommand(window_handle,control_handle,"IsEnabled",""));
  }
//+------------------------------------------------------------------+
//|Checks if a control is visible                                    |
//+------------------------------------------------------------------+
bool CAutoIt::IsControlVisible(int window_handle,int control_handle)
  {
   return(ControlCommand(window_handle,control_handle,"IsVisible",""));

  }
//+------------------------------------------------------------------+
//|Checks if tickbox is ticked                                       |
//+------------------------------------------------------------------+
bool CAutoIt::IsButtonChecked(int window_handle,int control_handle)
  {

   return(ControlCommand(window_handle,control_handle,"IsChecked",""));

  }
//+------------------------------------------------------------------+
//| Ticks a tick box                                                 |
//+------------------------------------------------------------------+
bool CAutoIt::CheckButton(int window_handle,int control_handle)
  {
   return(ControlCommand(window_handle,control_handle,"Check",""));
  }
//+------------------------------------------------------------------+
//|Unticks a tick box                                                |
//+------------------------------------------------------------------+
bool CAutoIt::UnCheckButton(int window_handle,int control_handle)
  {

   return(ControlCommand(window_handle,control_handle,"UnCheck",""));
  }
//+------------------------------------------------------------------+
//|Gets text of currently enabled tab of SysTabControl32 control     |
//+------------------------------------------------------------------+
string CAutoIt::GetCurrentTab(int window_handle,int control_handle)
  {
   string m_buffer;
   StringInit(m_buffer,m_buffer_size);

   AU3_ControlCommandByHandle(window_handle,control_handle,"CurrentTab","",m_buffer,m_buffer_size);

   return(m_buffer);
  }
//+------------------------------------------------------------------+
//|Enables the tab to the left in SysTabControl32 control            |
//+------------------------------------------------------------------+
bool CAutoIt::TabLeft(int window_handle,int control_handle)
  {
   return(ControlCommand(window_handle,control_handle,"TabLeft",""));
  }
//+------------------------------------------------------------------+
//|Enables the tab to the right in SysTabControl32 control           |
//+------------------------------------------------------------------+
bool CAutoIt::TabRight(int window_handle,int control_handle)
  {
   return(ControlCommand(window_handle,control_handle,"TabRight",""));
  }
//+------------------------------------------------------------------+
//|Sends a command to a ListView32 control                           |
//+------------------------------------------------------------------+
bool CAutoIt::ControlListView(int window_handle,int control_handle,string command,string command_option1,string command_option2)
  {
   string m_buffer;
   StringInit(m_buffer,m_buffer_size);

   AU3_ControlListViewByHandle(window_handle,control_handle,command,command_option1,command_option2,m_buffer,m_buffer_size);

   if(StringFind(m_buffer,"1")>=0)
      return(true);

   return(false);
  }
//+------------------------------------------------------------------+
//|Gets number of items in ListView32 control                        |
//+------------------------------------------------------------------+
long CAutoIt::GetListViewItemCount(int window_handle,int control_handle)
  {
   string m_buffer;
   StringInit(m_buffer,m_buffer_size);

   AU3_ControlListViewByHandle(window_handle,control_handle,"GetItemCount","","",m_buffer,m_buffer_size);

   return(StringToInteger(m_buffer));
  }
//+-------------------------------------------------------------------------------------------+
//|retrievs the index of a ListView32 control item that matches find_item and sub_item strings|
//+-------------------------------------------------------------------------------------------+
long CAutoIt::FindListViewItem(int window_handle,int control_handle,string find_item,string sub_item)
  {
   string m_buffer;
   StringInit(m_buffer,m_buffer_size);

   AU3_ControlListViewByHandle(window_handle,control_handle,"FindItem",find_item,sub_item,m_buffer,m_buffer_size);

   return(StringToInteger(m_buffer));
  }
//+------------------------------------------------------------------+
//|gets the string list of all selected ListView32 control items     |
//+------------------------------------------------------------------+
string CAutoIt::GetSelectedListViewItem(int window_handle,int control_handle)
  {
   string m_buffer;
   StringInit(m_buffer,m_buffer_size);

   AU3_ControlListViewByHandle(window_handle,control_handle,"GetSelected","1","",m_buffer,m_buffer_size);

   return(m_buffer);
  }
//+------------------------------------------------------------------+
//|returns number of currently selected ListView32 control items     |
//+------------------------------------------------------------------+
long CAutoIt::GetSelectedListViewCount(int window_handle,int control_handle)
  {
   string m_buffer;
   StringInit(m_buffer,m_buffer_size);

   AU3_ControlListViewByHandle(window_handle,control_handle,"GetSelectedCount","","",m_buffer,m_buffer_size);

   return(StringToInteger(m_buffer));
  }
//+------------------------------------------------------------------+
//|gets number of sub items in ListView32 control                    |
//+------------------------------------------------------------------+
long CAutoIt::GetListViewSubItemCount(int window_handle,int control_handle)
  {
   string m_buffer;
   StringInit(m_buffer,m_buffer_size);

   AU3_ControlListViewByHandle(window_handle,control_handle,"GetSubItemCount","","",m_buffer,m_buffer_size);

   return(StringToInteger(m_buffer));
  }
//+---------------------------------------------------------------------+
//|returns text of single ListView32 control item referenced by an index|
//+---------------------------------------------------------------------+
string CAutoIt::GetListViewItemText(int window_handle,int control_handle,string item_index,string sub_item_index)
  {
   string m_buffer;
   StringInit(m_buffer,m_buffer_size);

   AU3_ControlListViewByHandle(window_handle,control_handle,"GetText",item_index,sub_item_index,m_buffer,m_buffer_size);

   return(m_buffer);
  }
//+------------------------------------------------------------------+
//|Checks if a certain ListView32 control item is selected           |
//+------------------------------------------------------------------+
bool CAutoIt::IsListViewItemSelected(int window_handle,int control_handle,string item_index)
  {
   return(ControlListView(window_handle,control_handle,"IsSelected",item_index,""));
  }
//+------------------------------------------------------------------+
//|Selects a certain ListView32 control                              |
//+------------------------------------------------------------------+
bool CAutoIt::SelectListViewItem(int window_handle,int control_handle,string from_item_index,string to_item_index)
  {
   return(ControlListView(window_handle,control_handle,"Select",from_item_index,to_item_index));
  }
//+------------------------------------------------------------------+
//|selects all listview items                                        |
//+------------------------------------------------------------------+
bool CAutoIt::SelectAllListViewItems(int window_handle,int control_handle)
  {
   return(ControlListView(window_handle,control_handle,"SelectAll","",""));
  }

//+------------------------------------------------------------------+
//|Deselects all currently selected item in a ListView32 control     |
//+------------------------------------------------------------------+
bool CAutoIt::ClearAllListViewItemSelections(int window_handle,int control_handle)
  {
   return(ControlListView(window_handle,control_handle,"SelectClear","",""));
  }
//+------------------------------------------------------------------+
//| Change listview control view to details mode                     |
//+------------------------------------------------------------------+
bool CAutoIt::ViewChangeListView(int window_handle,int control_handle)
  {
   string m_buffer;
   StringInit(m_buffer,m_buffer_size);

   AU3_ControlListViewByHandle(window_handle,control_handle,"ViewChange","list","",m_buffer,m_buffer_size);

   if(StringFind(m_buffer,"1")>=0)
      return(true);

   return(false);
  }
//+------------------------------------------------------------------+
//|returns handle for a control                                      |
//+------------------------------------------------------------------+
HWND CAutoIt::ControlGetHandle(int window_handle,string control_id)
  {
   return(AU3_ControlGetHandle(window_handle,control_id));
  }
//+------------------------------------------------------------------+
//|returns visible text from a control                               |
//+------------------------------------------------------------------+
string CAutoIt::ControlGetText(int window_handle,int control_handle)
  {
   string m_buffer;
   StringInit(m_buffer,m_buffer_size);

   AU3_ControlGetTextByHandle(window_handle,control_handle,m_buffer,m_buffer_size);

   return(m_buffer);
  }
//+------------------------------------------------------------------+
//|directs keyboard presses to a certain control                     |
//+------------------------------------------------------------------+
int CAutoIt::ControlSend(int window_handle,int control_handle,string keys,int mode)
  {
   return(AU3_ControlSendByHandle(window_handle,control_handle,keys,mode));
  }
//+------------------------------------------------------------------+
//|Sends a command to a TreeView32 control                           |
//+------------------------------------------------------------------+
bool CAutoIt::ControlTreeView(int window_handle,int control_handle,string command,string command_option1,string command_option2)
  {
   string m_buffer;
   StringInit(m_buffer,m_buffer_size);

   AU3_ControlTreeViewByHandle(window_handle,control_handle,command,command_option1,command_option2,m_buffer,m_buffer_size);

   if(StringFind(m_buffer,"1")>=0)
      return(true);

   return(false);
  }
//+-----------------------------------------------------------------------------+
//|returns number of children on a a TreeView32 control item with selected index|
//+-----------------------------------------------------------------------------+
long CAutoIt::GetTreeViewItemCount(HWND window_handle, HWND control_handle, string item_index)
  {
   string m_buffer;
   StringInit(m_buffer,m_buffer_size);

   AU3_ControlTreeViewByHandle(window_handle,control_handle,"GetItemCount",item_index,"",m_buffer,m_buffer_size);

   return(StringToInteger(m_buffer));
  }
//+------------------------------------------------------------------+
//|gets index in string format of selected TreeView32 control item   |
//+------------------------------------------------------------------+
string CAutoIt::GetSelectedTreeViewItem(HWND window_handle, HWND control_handle)
  {
   string m_buffer;
   StringInit(m_buffer,m_buffer_size);

   AU3_ControlTreeViewByHandle(window_handle,control_handle,"GetSelected","","",m_buffer,m_buffer_size);

   return(m_buffer);
  }
//+------------------------------------------------------------------+
//|retrieves the text of a a TreeView32 control item                 |
//+------------------------------------------------------------------+
string CAutoIt::GetTreeViewItemText(HWND window_handle, HWND control_handle,string item)
  {
   string m_buffer;
   StringInit(m_buffer,m_buffer_size);

   AU3_ControlTreeViewByHandle(window_handle,control_handle,"GetText",item,"",m_buffer,m_buffer_size);

   return(m_buffer);
  }
//+------------------------------------------------------------------+
//|selects a a TreeView32 control item                               |
//+------------------------------------------------------------------+
bool CAutoIt::SelectTreeViewItem(HWND window_handle, HWND control_handle,string item)
  {
   return(ControlTreeView(window_handle,control_handle,"Select",item,""));
  }

//+------------------------------------------------------------------+
//|returns handle of window by its window title                      |
//+------------------------------------------------------------------+
HWND CAutoIt::WinGetHandle(string window_title, string window_text="")
  {
   return(AU3_WinGetHandle(window_title,window_text));
  }
//+------------------------------------------------------------------+
//|return all text that is visible within a window                   |
//+------------------------------------------------------------------+
string CAutoIt::WinGetText(HWND window_handle)
  {
   string m_buffer;
   StringInit(m_buffer,m_buffer_size);

   AU3_WinGetTextByHandle(window_handle,m_buffer,m_buffer_size);

   return(m_buffer);
  }
//+------------------------------------------------------------------+
//|Invokes a menu item of a window                                   |
//+------------------------------------------------------------------+
int CAutoIt::WinMenuSelectItem(HWND window_handle, string menu_name_1, string menu_name_2, string menu_name_3, string menu_name_4, string menu_name_5, string menu_name_6, string menu_name_7, string menu_name_8)
  {
   return(AU3_WinMenuSelectItemByHandle(window_handle,menu_name_1,menu_name_2,menu_name_3,menu_name_4,menu_name_5,menu_name_6,menu_name_7,menu_name_8));
  }
//+------------------------------------------------------------------+
//|Shows, hides, minimizes, maximizes, or restores a window          |
//+------------------------------------------------------------------+
int CAutoIt::WinSetState(HWND window_handle, ENUM_WINDOW_STATE new_state)
  {
   return(AU3_WinSetStateByHandle(window_handle,(int)new_state));
  }

//+------------------------------------------------------------------+
//|retrieves window title                                            |
//+------------------------------------------------------------------+
string CAutoIt::WinGetTitle(int window_handle)
  {
   string m_buffer;
   StringInit(m_buffer,m_buffer_size);

   AU3_WinGetTitleByHandle(window_handle,m_buffer,m_buffer_size);

   return(m_buffer);
  }

//+------------------------------------------------------------------+
//|Gets total number of tabs in systab32 control                     |
//+------------------------------------------------------------------+
long CAutoIt::TotalTabs(const int window,const int systab)
  {
   if(!WinExists(window))
     {
      return(0);
     }

   string index=GetCurrentTab(window,systab);
   long shift=-1;

   while(TabRight(window,systab))
     {
      index="";
      index=GetCurrentTab(window,systab);
     }

   shift=StringToInteger(index);

   return(shift);

  }
//+------------------------------------------------------------------+
```

### CTerminalController class

Next let us create the CTerminalController class in the terminalcontroller.mqh file.

First we include the autoitbase.mqh file which contains the CAutoIt class. We use preprocessor directives to define string constants that identify user inteface components the class will work with.

Before specifying the class, we declare a custom enumeration which classifies MetaTrader 5 programs as either being a script or an Expert Advistor.

```
//+------------------------------------------------------------------+
//|                                           terminalcontroller.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://www.mql5.com "
#include<autoItbase.mqh>

#define JOURNAL_TAB              "SysListView321"
#define EXPERTS_TAB              "SysListView327"

#define NAVIGATOR                "SysTreeView321"
#define NAVIGATOR_TAB_BAR        "AfxWnd140su3"

#define PROGRAM_WINDOW_TAB       "SysTabControl321"

#define EXPERTS_WINDOW_LISTVIEW  "SysListView321"
#define EXPERTS_REMOVE_BUTTON    "Button2"

#define DLLCHECK_BUTTON          "Button1"
#define ALGOTRADECHECK_BUTTON    "Button4"
#define OK_BUTTON_1              "Button7"
#define OK_BUTTON_2              "Button3"
#define OK_BUTTON_3              "Button5"
#define LOAD_BUTTON              "Button1"
#define YES_BUTTON               "Button1"

#define FILENAME_EDIT            "Edit1"

#define OPTWINDOW_TAB            "SysTabControl321"
#define OPTWINDOW_CHECK_1        "Button1"
#define OPTWINDOW_CHECK_2        "Button2"
#define OPTWINDOW_CHECK_3        "Button6"
#define OPTWINDOW_OK             "Button7"
#define OPTWINDOW_EDIT           "Edit1"

#define EXPERTS_WINDOW           "Experts"

#define NAVIGATORWINDOW          "Navigator"
#define TOOLBOXWINDOW            "Toolbox"

#define FILEDIALOGE_WINDOW       "Open"
#define WINDOWTEXT_DLL           "Allow DLL"
#define WINDOWTEXT_EA            "Expert Advisor settings"
#define WINDOWTEXT_INPUT         "Input"

#define OPTWINDOW                "Options"
#define NOTIFICATIONS_TEXT       "Enable Push notifications"

#define ALERTWINDOW              "Alert"

#define MENU_TOOLS             "&Tools"
#define MENU_OPTIONS           "&Options"
#define MENU_WINDOW            "&Window"
#define MENU_TILEWINDOWS       "&Tile Windows"

#define MENU_VIEW              "&View"
#define MENU_NAVIGATOR         "&Navigator"

#define MENU_CHARTS            "&Charts"
#define MENU_EXPERTS_LIST      "&Expert List"

enum ENUM_TYPE_PROGRAM
  {
   ENUM_TYPE_EXPERT=0,//EXPERT_ADVISOR
   ENUM_TYPE_SCRIPT//SCRIPT
  };
```

The private properties of type HWND are the window and control handles that are used in the class.

```
//+------------------------------------------------------------------+
//|Class CTerminalController                                         |
//| Purpose: class for scripting the terminal                        |
//+------------------------------------------------------------------+

class CTerminalController:public CAutoIt
  {
private:

   HWND              m_terminal;                 //terminal window handle
   HWND              m_journaltab;               //journal tab handle
   HWND              m_expertstab;               //experts tab handle

   HWND              m_navigator;                //navigator systreeview
   HWND              m_navigatortabs;            //navigator tab header
   HWND              m_navigatorwindow;          //navigator window
   HWND              m_systab32;                 //handle to inputs tabbed control with in inputs dialogue
   HWND              m_program;                  //window handle for user inputs dialogue
   HWND              m_expwindow;                //handle to window showing list of experts
   HWND              m_explistview;              //list view control for experts list in m_expwindow
   long              m_chartid;                  //chart id
   long              m_accountNum;               //account number
   string            m_buffer;                   //string buffer
```

m\_chartid refers to the chart identifier as defined by calling the MQL5 function chartID().

m\_accountNum is the active account number displayed on the terminal's application title bar. This value will be used to discern one active terminal window from another.

m\_buffer string is where detailed error messages will be written to in case something goes wrong.

```
public:
   //default constructor
                     CTerminalController(void): m_terminal(0),
                     m_journaltab(0),
                     m_expertstab(0),
                     m_navigator(0),
                     m_navigatortabs(0),
                     m_navigatorwindow(0),
                     m_systab32(0),
                     m_program(0),
                     m_expwindow(0),
                     m_explistview(0),
                     m_chartid(-1),
                     m_buffer("")
     {
      m_accountNum=AccountInfoInteger(ACCOUNT_LOGIN);
      StringInit(m_buffer,1000);
     }
   //destructor
                    ~CTerminalController(void)
     {

     }
```

CTerminalController will have a single constructor. The default constructor initializes m\_accountNum with the currently active terminal account returned by a call to [AccountInfoInteger](https://www.mql5.com/en/docs/account/accountinfointeger) function.

**The methods**

```
//public methods
   string            GetErrorDetails(void)  {     return(m_buffer);      }
   bool              ChartExpertAdd(const ENUM_TYPE_PROGRAM p_type,const string ea_name,const string ea_relative_path,const string ea_set_file,const string chart_symbol,const ENUM_TIMEFRAMES chart_tf);
   bool              ChartExpertRemove(const string ea_name,const string chart_symbol,const ENUM_TIMEFRAMES chart_tf);
   bool              ChartExpertRemoveAll(void);

   void              CloseAlertDialogue(void);

   void              GetLastExpertsLogEntry(string &entry);
   void              GetLastExpertsLogEntryByText(const string text_to_search_for,const int max_items_to_search_in,string &entry);
   void              GetLastJournalEntry(string &entry);
   void              GetLastJournalEntryByText(const string text_to_search_for,const int max_items_to_search_in,string &entry);
   bool              Initialize(const long accountNumber=0);
   bool              SwitchToNewChart(string n_symbol, const ENUM_TIMEFRAMES n_tf);
   bool              SetNotificationId(const string MetaQuotes_id);
   bool              ToggleAutoTrading(void);

private:
   //helper methods
   void              SetErrorDetails(const string text);
   void              Findbranch(const long childrenOnBranch,const string index,const string pname,string& sbuffer);
   bool              Findprogram(const ENUM_TYPE_PROGRAM pr_type, const string program_name,const string relative_path,string& sbuffer);
   string            PeriodToString(const ENUM_TIMEFRAMES chart_tf);
   bool              RemoveExpert(const string program_name,const string chart_symbol,const ENUM_TIMEFRAMES chart_tf);
   string            BrokerName(void);

  };
```

For retrieving error messages there is GetErrorDetails(). The function returns the contents of private member m\_buffer. SetErrorDetails() on the other hand is a private method that is used internally to set the contents of m\_buffer string member.

```
//+------------------------------------------------------------------+
//| Set error details                                                |
//+------------------------------------------------------------------+
void CTerminalController::SetErrorDetails(const string _text)
  {
   StringFill(m_buffer,0);

   m_buffer=_text;

   return;
  }
```

The Initialize() method should be called at least once at program initialization before any other method. The method has a single default parameter, accountNumber - if the argument is non zero, it resets the value of m\_accountNum class property which in turn resets all other class properties triggering the search for the main terminal window identified by m\_accountNum property. Once the main window is found, all other control handles are retrieved and used to set the remaining class properties.

```
//+------------------------------------------------------------------+
//| sets or resets window and control handles used in the class      |
//+------------------------------------------------------------------+
bool CTerminalController::Initialize(const long accountNumber=0)
  {
   Init();

   if(accountNumber>0 && accountNumber!=m_accountNum)
     {
      m_accountNum=accountNumber;
      m_program=m_expwindow=m_systab32=m_explistview=m_navigatorwindow=0;
      m_terminal=m_journaltab=m_expertstab=m_navigator=m_navigatortabs=0;
      m_chartid=0;
     }

   if(m_terminal)
      return(true);

   m_terminal=WinGetHandle(IntegerToString(m_accountNum));
   if(!m_terminal)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Terminal window "+IntegerToString(m_accountNum)+" not found");
      return(false);
     }

   m_journaltab=ControlGetHandle(m_terminal,JOURNAL_TAB);
   if(!m_journaltab)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+JOURNAL_TAB+" not found");
      return(false);
     }

   m_expertstab=ControlGetHandle(m_terminal,EXPERTS_TAB);
   if(!m_expertstab)
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+EXPERTS_TAB+" not found");
      return(false);
     }

   m_navigatorwindow=ControlGetHandle(m_terminal,NAVIGATORWINDOW);
   if(!m_navigatorwindow)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+NAVIGATORWINDOW+" not found");
      return(false);
     }

   m_navigator=ControlGetHandle(m_terminal,NAVIGATOR);
   if(!m_navigator)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+NAVIGATOR+" not found");
      return(false);
     }

   m_navigatortabs=ControlGetHandle(m_terminal,NAVIGATOR_TAB_BAR);
   if(!m_navigatortabs)
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+NAVIGATOR_TAB_BAR+" not found");
      return(false);
     }

   StringInit(m_buffer,1000);

   return(true);
  }
```

ChartExpertAdd() is used to attach expert advisors or scripts to charts. The  function parameters are:

- p\_type - the program type, either an expert advisor or a script,
- ea\_name - the name of the script or expert advisor exactly as it appears in the navigator window, file extension names must not be included.
- ea\_relative\_path - this is the relative path to a subfolder in either the Scripts or Experts folders that contains the program given by ea\_name parameter, take for example the expert advisor Controls shown in the picture below, it is contained in a folder named Controls in the Examples subfolder within the Experts folder, so its path would be Examples\\Controls. If the program is not in any subfolder of either the Scripts or Experts folder, then this argument should be set NULL or "".
![Navigator](https://c.mql5.com/2/49/NavigatorFolder.PNG)


- ea\_set\_file is the filename of a .set file. For this argument the file extension .set should be specified.
- chart\_symbol and chart tf arguments specify the properties of the chart the program will be added to.

First the SwitchToNewChart() method is called. This method searches for the requested chart amongst currently open chart windows. If found, focus is set to it, otherwise the ChartOpen() function is used to open a new chart. The method is also responsible for setting the m\_chartid property of the class.

The m\_chartid property is then used to check if the program has already been added to the chart. If the program exists, then the method returns 'true'.

```
//+------------------------------------------------------------------+
//|sets focus to existing or new chart window                        |
//+------------------------------------------------------------------+
bool CTerminalController::SwitchToNewChart(string n_symbol,const ENUM_TIMEFRAMES n_tf)
  {
   if(!IsInitialized())
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return(false);
     }

   string sarray[];

   StringToUpper(n_symbol);
   string newchartname=n_symbol+","+PeriodToString(n_tf);

   if(!WinMenuSelectItem(m_terminal,MENU_WINDOW,MENU_TILEWINDOWS,"","","","","",""))
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Failed to select menu item: Tile Windows");
      return(false);
     }

   Sleep(200);

   string windowtext=WinGetText(m_terminal);

   int find=StringFind(windowtext,ChartSymbol(ChartFirst())+","+PeriodToString(ChartPeriod(ChartFirst())));

   string chartstring=StringSubstr(windowtext,find);

   StringReplace(chartstring,"\n\n","\n");

   int sarraysize=StringSplit(chartstring,StringGetCharacter("\n",0),sarray);

   bool found=false;
   int i;

   long prevChart=0;
   m_chartid=ChartFirst();

   for(i=0; i<sarraysize; i++,)
     {
      if(sarray[i]=="")
         continue;

      if(i>0)
         m_chartid=ChartNext(prevChart);

      if(StringFind(sarray[i],newchartname)>=0)
        {
         found=true;
         break;
        }

      prevChart=m_chartid;
     }

   ArrayFree(sarray);

   HWND frameview=0;

   if(found)
      frameview=ControlGetHandle(m_terminal,newchartname);
   else
      if(ChartOpen(n_symbol,n_tf))
         frameview=ControlGetHandle(m_terminal,newchartname);

   if(frameview)
     {
      if(!WinSetState(frameview,SW_MAXIMIZE))
        {

         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Could not maximize "+newchartname+" chart window");
         return(false);
        }
     }
   else
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Chart window "+newchartname+" not found");
      return(false);
     }

   return(true);
  }
```

If the program is not found, preparations are made to search the Navigator pane. If the Common tab of the Navigator pane is not visible, it is enabled.

Once enabled the Findprogram() method is called to search for the MetaTrader 5 program on the systreeview32 control. The method enlists the help of the recursive method FindBranch().

```
//+---------------------------------------------------------------------+
//|searches navigator for a program and outputs its location on the tree|
//+---------------------------------------------------------------------+
bool CTerminalController::Findprogram(const ENUM_TYPE_PROGRAM pr_type,const string program_name,const string relative_path,string &sbuffer)
  {

   long listsize=GetTreeViewItemCount(m_terminal,m_navigator,"#0");

   if(!listsize)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Root treeview control is empty");
      return(false);
     }
   else
     {
      string rpath="";

      if(relative_path=="")
         rpath=relative_path;
      else
        {
         if(StringFind(relative_path,"\\")==0)
            rpath=StringSubstr(relative_path,1);
         else
            rpath=relative_path;
         if(StringFind(rpath,"\\",StringLen(rpath)-1)<0)
            rpath+="\\";
        }

      switch(pr_type)
        {
         case ENUM_TYPE_EXPERT:
           {
            string fullpath="Expert Advisors\\"+rpath+program_name;
            Findbranch(listsize,"#0",fullpath,sbuffer);
            break;
           }
         case ENUM_TYPE_SCRIPT:
           {
            string fullpath="Scripts\\"+rpath+program_name;
            Findbranch(listsize,"#0",fullpath,sbuffer);
            break;
           }
         default:
            Findbranch(listsize,"#0","",sbuffer);
            break;
        }

     }

   if(sbuffer=="")
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Operation failed could not find "+program_name);
      return(false);
     }
   else
      return(true);

  }
```

FindBranch() traverses the control and also builds the string reference ID which is the path to the programs location on the systreeview32 control. The program's reference ID is needed so AutoIt can be utilised to select it and add it to the chart.

```
//+-----------------------------------------------------------------------+
//|recursively searches systreeview for program and builds location string|
//+-----------------------------------------------------------------------+
void CTerminalController::Findbranch(const long childrenOnBranch,const string index,const string pname,string &sbuffer)
  {
   if(pname=="" ||  index=="")
     {
      sbuffer=index;
      return;
     }
   else
     {
      if(childrenOnBranch<=0)
         return;
      else
        {
         int find=StringFind(pname,"\\");

         long ss=0;
         long i;

         for(i=0; i<childrenOnBranch; i++)
           {

            ss=GetTreeViewItemCount(m_terminal,m_navigator,index+"|#"+IntegerToString(i));

            string search=(find>=0)?StringSubstr(pname,0,find):pname;

            string treebranchtext=GetTreeViewItemText(m_terminal,m_navigator,index+"|#"+IntegerToString(i));

            if(StringFind(treebranchtext,search)>=0 && StringLen(treebranchtext)==StringLen(search))
               break;

           }

         string npath=(find>=0)?StringSubstr(pname,find+1):"";

         Findbranch(ss,(i<childrenOnBranch)?index+"|#"+IntegerToString(i):"",npath,sbuffer);
        }
     }

   return;
  }
```

At this point if the program is an expert advisor and there is another EA already on the chart, then a dialogue will appear confirming the replacement of the currently running expert advisor.

If the program launched has inputs the do... while loop cycles through the tabs, sets the .set file and ticks check buttons enabling auto trading and dll exports if needed.

Finally at the end the method returns the result of checking the chart again for the program to confirm if it has been added to the chart.

```
//+------------------------------------------------------------------+
//| Adds EA,Script,Service to a chart                                |
//+------------------------------------------------------------------+
bool CTerminalController::ChartExpertAdd(const ENUM_TYPE_PROGRAM p_type,const string ea_name,const string ea_relative_path,const string ea_set_file,const string chart_symbol,const ENUM_TIMEFRAMES chart_tf)
  {

   if(!SwitchToNewChart(chart_symbol,chart_tf))
     {
      return(false);
     }

   if(StringFind((p_type==ENUM_TYPE_EXPERT)?ChartGetString(m_chartid,CHART_EXPERT_NAME):ChartGetString(m_chartid,CHART_SCRIPT_NAME),ea_name)>=0)
     {
      return(true);
     }

   if(p_type==ENUM_TYPE_EXPERT && StringFind(ChartGetString(m_chartid,CHART_EXPERT_NAME),ChartGetString(ChartID(),CHART_EXPERT_NAME))>=0)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Cannot replace currenlty running Expert");
      return(false);
     }

   if(StringLen(ea_set_file)>0)
     {
      if(StringFind(ea_relative_path,".set")<0)
        {
         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Incorrect parameter setting, set file name should contain .set file extension");
         return(false);
        }
     }

   if(!IsControlVisible(m_terminal,m_navigator))
     {
      if(!IsControlVisible(m_terminal,m_navigatorwindow))
        {
         if(!WinMenuSelectItem(m_terminal,MENU_VIEW,MENU_NAVIGATOR,"","","","","",""))
           {
            SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Failed to select menu item : Navigator");
            return(false);
           }
        }

      if(!ControlLeftClick(m_terminal,m_navigatortabs,1,5,5))
        {
         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Failed to click on Navigator, Common tab");
         return(false);
        }
     }

   string treepath="";

   if(!Findprogram(p_type,ea_name,ea_relative_path,treepath))
      return(false);

   if(!SelectTreeViewItem(m_terminal,m_navigator,treepath))
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Failed to select treeview item: "+treepath);
      return(false);
     }

   if(!ControlSend(m_terminal,m_navigator,"{SHIFTDOWN}{F10}{SHIFTUP}c",0))
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Control Send failure");
      return(false);
     }

   Sleep(500);

   m_program=WinGetHandle(ea_name);

   HWND dialogue=WinGetHandle(BrokerName()+" - MetaTrader 5",ea_name);

   if(!m_program && !dialogue)
     {
      if(p_type==ENUM_TYPE_EXPERT && StringFind(ChartGetString(m_chartid,CHART_EXPERT_NAME),ea_name)<0)
        {
         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Could not add program to chart");
         return(false);
        }
      else
        {
         return(true);
        }
     }

   if(!m_program && dialogue)// replace current ea
     {
      HWND button = ControlGetHandle(dialogue,YES_BUTTON);

      if(button)
        {
         if(ControlLeftClick(dialogue,button))
           {
            Sleep(200);
            m_program=WinGetHandle(ea_name);
           }
         else
           {

            SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Failed left click on yes button");
            WinClose(dialogue);
            return(false);
           }
        }
      else
        {

         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Yes button for dialogue not found");
         WinClose(dialogue);
         return(false);
        }

     }

   m_systab32=ControlGetHandle(m_program,PROGRAM_WINDOW_TAB);
   if(!m_systab32)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+PROGRAM_WINDOW_TAB+ "not found");
      return(false);
     }

   long totaltabs=TotalTabs(m_program,m_systab32);
   bool algoenabled=false;
   bool inputmod=false;
   bool dllenabled=(totaltabs<3)?true:false;

   do
     {
      string windowtext=WinGetText(m_program);

      if(StringFind(windowtext,WINDOWTEXT_DLL)>=0)
        {
         StringFill(windowtext,0);
         HWND button=ControlGetHandle(m_program,DLLCHECK_BUTTON);

         if(!button)
           {
            WinClose(m_program);
              {
               SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"DLL check button not found");
               return(false);
              }
           }

         if(!IsButtonChecked(m_program,button))
           {
            if(CheckButton(m_program,button))
               dllenabled=true;
           }
         else
            dllenabled=true;

         if(dllenabled)
           {
            if(TabLeft(m_program,m_systab32))
               if(!TabLeft(m_program,m_systab32))
                 {
                  inputmod=true;
                 }
           }
         else
           {
            WinClose(m_program);
            SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Could not enable DLL loading");
            return(false);
           }
        }
      else
         if(StringFind(windowtext,WINDOWTEXT_EA)==0)
           {
            StringFill(windowtext,0);
            HWND button=ControlGetHandle(m_program,ALGOTRADECHECK_BUTTON);

            if(!button)
              {
               WinClose(m_program);
               SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Algo check button not found");
               return(false);
              }

            if(!IsButtonChecked(m_program,button))
              {
               if(CheckButton(m_program,button))
                  algoenabled=true;
              }
            else
               algoenabled=true;

            if(algoenabled)
              {
               if(!inputmod)
                 {
                  if(!TabRight(m_program,m_systab32))
                    {
                     HWND okbutton=ControlGetHandle(m_program,OK_BUTTON_3);
                     if(!okbutton)
                       {
                        WinClose(m_program);
                        SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Failed find Ok button");
                        return(false);
                       }

                     if(!ControlLeftClick(m_program,okbutton))
                       {
                        WinClose(m_program);
                        SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Left Click failure");
                        return(false);
                       }
                     else
                       {
                        break;
                       }
                    }
                 }
               else
                 {
                  HWND okbutton=ControlGetHandle(m_program,OK_BUTTON_3);
                  if(!okbutton)
                    {
                     WinClose(m_program);
                     SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Failed find Ok button");
                     return(false);
                    }

                  if(!ControlLeftClick(m_program,okbutton))
                    {
                     WinClose(m_program);
                     SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Left Click failure");
                     return(false);
                    }
                 }
              }
            else
              {
               WinClose(m_program);
               SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Unable to enable autotrading");
               return(false);
              }
           }
         else
            if(StringFind(windowtext,WINDOWTEXT_INPUT)==0)
              {
               StringFill(windowtext,0);

               HWND button=ControlGetHandle(m_program,LOAD_BUTTON);

               HWND okbutton=ControlGetHandle(m_program,(totaltabs>2)?OK_BUTTON_1:OK_BUTTON_2);

               if(!okbutton)
                 {
                  WinClose(m_program);
                  SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Failed to find OK button");
                  return(false);
                 }

               if(StringLen(ea_set_file)>0)
                 {
                  if(!button)
                    {
                     WinClose(m_program);
                     SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Failed to find .set file load button");
                     return(false);
                    }

                  if(!ControlLeftClick(m_program,button))
                    {
                     WinClose(m_program);
                     SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Left click failure");
                     return(false);
                    }

                  HWND filewin=0;
                  int try
                        =50;
                  while(!filewin && try
                           >0)
                       {
                        filewin=WinGetHandle(FILEDIALOGE_WINDOW);
                        try
                           --;
                        Sleep(200);
                       }

                  HWND filedit=ControlGetHandle(filewin,FILENAME_EDIT);

                  if(!filedit || !filewin)
                    {
                     if(!filedit)
                       {
                        if(filewin)
                           WinClose(filewin);
                       }
                     WinClose(m_program);
                     SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" File dialogue failure");
                     return(false);
                    }

                  if(ControlSetText(filewin,filedit,""))
                    {
                     if(ControlSend(filewin,filedit,ea_set_file,1))
                       {
                        Sleep(200);
                        if(ControlSend(filewin,filedit,"{ENTER}",0))
                           Sleep(300);
                       }
                    }

                  if(WinExists(filewin))
                    {
                     WinClose(filewin);
                     WinClose(m_program);
                     SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Failed to set .set file through file dialogue");
                     return(false);
                    }
                 }

               inputmod=true;

               if(algoenabled)
                 {
                  if(ControlLeftClick(m_program,okbutton))
                     break;
                  else
                    {
                     WinClose(m_program);
                     SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Failed to click OK button");
                     return(false);
                    }
                 }
               else
                 {
                  if(!TabLeft(m_program,m_systab32))
                    {
                     if(ControlLeftClick(m_program,okbutton))
                        break;
                     else
                       {
                        WinClose(m_program);
                        SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Failed to click OK button");
                        return(false);
                       }
                    }
                 }
              }
     }
   while(!inputmod||!dllenabled||!algoenabled);

   int try
         =50;

   while(WinExists(m_program) && try
            >0)
        {
         Sleep(500);
         try
            --;
        }

   if(WinExists(m_program) && !try)
     {
      WinClose(m_program);
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Failed to add program to chart");
      return(false);
     }

   if(p_type==ENUM_TYPE_EXPERT && StringFind(ChartGetString(m_chartid,CHART_EXPERT_NAME),ea_name)<0)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Operation failed could not add program to chart");
      return(false);
     }

   return(true);

  }
```

ChartExpertRemove() detaches either a script or Expert Advisor from a chart. It does this by invoking the Experts window dialogue via the main application's menu. The method then iterates through the list of programs, and finds a match based on program name and chart properties (i.e. chart symbol and timeframe).

The ChartExpertRemoveAll() function removes all EA's and scripts except for the program that is actually doing the removals.

```
//+------------------------------------------------------------------+
//|Removes EA,Script from a chart                                    |
//+------------------------------------------------------------------+
bool CTerminalController::ChartExpertRemove(const string ea_name,const string chart_symbol,const ENUM_TIMEFRAMES chart_tf)
  {
   return(RemoveExpert(ea_name,chart_symbol,chart_tf));
  }
```

```
//+------------------------------------------------------------------+
//|Helper function detaches program from a chart                     |
//+------------------------------------------------------------------+
bool CTerminalController::RemoveExpert(const string program_name,const string chart_symbol,const ENUM_TIMEFRAMES chart_tf)
  {
   if(!IsInitialized())
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return(false);
     }

   if(!WinMenuSelectItem(m_terminal,MENU_CHARTS,MENU_EXPERTS_LIST,"","","","","",""))
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Menu selection failure.");
      return(false);
     }

   Sleep(200);

   m_expwindow=WinGetHandle(EXPERTS_WINDOW);
   if(!m_expwindow)
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+EXPERTS_WINDOW+" window not found");
      return(false);
     }

   m_explistview=ControlGetHandle(m_expwindow,EXPERTS_WINDOW_LISTVIEW);
   if(!m_explistview)
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+EXPERTS_WINDOW_LISTVIEW+" control not found");
      return(false);
     }

   HWND remove_button=ControlGetHandle(m_expwindow,EXPERTS_REMOVE_BUTTON);
   if(!remove_button)
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Remove button not found");
      return(false);
     }

   long listsize=GetListViewItemCount(m_expwindow,m_explistview);

   if(listsize<=0)
     {
      return(true);
     }

   string newchartname=chart_symbol;
   StringToUpper(newchartname);
   newchartname+=","+PeriodToString(chart_tf);

   bool found=false;

   ClearAllListViewItemSelections(m_expwindow,m_explistview);

   for(int i=0; i<int(listsize); i++)
     {
      if(!SelectListViewItem(m_expwindow,m_explistview,IntegerToString(i),""))
         continue;

      string pname=GetListViewItemText(m_expwindow,m_explistview,IntegerToString(i),"0");

      string chartname=GetListViewItemText(m_expwindow,m_explistview,IntegerToString(i),"1");

      if(StringFind(pname,program_name)>=0 && StringFind(chartname,newchartname)>=0)
        {
         if(IsControlEnabled(m_expwindow,remove_button))
            if(ControlLeftClick(m_expwindow,remove_button))
               found=true;
        }

      if(found)
         break;

      ClearAllListViewItemSelections(m_expwindow,m_explistview);

     }

   WinClose(m_expwindow);

   return(found);
  }
```

GetLastExpertsLogEntry() and GetLastJournalEntry() methods output to a string reference the last entry made to the experts log and the journal log respectively.

The methods interact with either the journal or experts tabs and do not read from the actual log files, they basically scrape text from the listview32 control. If the tabs have been manually cleared, the methods cannot retrieve data directly from the log files.

```
//+------------------------------------------------------------------+
//|Gets the last journal entry                                       |
//+------------------------------------------------------------------+
void CTerminalController::GetLastJournalEntry(string &entry)
  {
   entry="";

   if(!IsInitialized())
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return;
     }

   long listsize=GetListViewItemCount(m_terminal,m_journaltab);

   if(listsize<=0)
      return;

   ClipPut("");

   if(!ClearAllListViewItemSelections(m_terminal,m_journaltab))
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to deselect all listview items");
      return;
     }

   if(!SelectListViewItem(m_terminal,m_journaltab,IntegerToString(listsize-1),""))
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to select last listview item");
      return;
     }

   if(!ControlSend(m_terminal,m_journaltab,"{LCTRL down}c{LCTRL up}",0))
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to send keys to control");
      return;
     }

   ClipGet(entry);

   StringTrimRight(entry);

   ClipPut("");

   ClearAllListViewItemSelections(m_terminal,m_journaltab);

   return;
  }
//+------------------------------------------------------------------+
//|Gets last entry made to experts log file                          |
//+------------------------------------------------------------------+
void CTerminalController::GetLastExpertsLogEntry(string &entry)
  {
   entry="";

   if(!IsInitialized())
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return;
     }

   long listsize=GetListViewItemCount(m_terminal,m_expertstab);

   if(listsize<=0)
      return;

   ClipPut("");

   if(!ClearAllListViewItemSelections(m_terminal,m_expertstab))
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to deselect all listview items");
      return;
     }

   if(!SelectListViewItem(m_terminal,m_expertstab,IntegerToString(listsize-1),""))
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to select last listview item");
      return;
     }

   if(!ControlSend(m_terminal,m_expertstab,"{LCTRL down}c{LCTRL up}",0))
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to send keys to control");
      return;
     }

   ClipGet(entry);

   StringTrimRight(entry);

   ClipPut("");

   ClearAllListViewItemSelections(m_terminal,m_expertstab);

   return;
  }
```

GetLastExpertsLogEntryByText() and GetLastJournalEntryByText() functions return the last log entry that matches _text\_to\_search\_for_ parameter. The functions start the search from the most recent log entry going backwards in time. The _max\_items\_to\_search\_in_ parameter sets the maximum number log entries to iterate through. For example if set to 10, the function will iterate through the last 10 log entries looking for a match. If this parameter is set to 0 or a negative number, then the function iterates through all the log entries.

```
//+------------------------------------------------------------------+
//|Gets last entry made to experts log file containg certain string  |
//+------------------------------------------------------------------+
void CTerminalController::GetLastExpertsLogEntryByText(const string text_to_search_for,const int max_items_to_search_in,string &entry)
  {
   entry="";

   if(!IsInitialized())
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return;
     }

   string items;
   string itemsarray[];

   long listsize=GetListViewItemCount(m_terminal,m_expertstab);

   if(listsize<=0)
      return;

   long stop=(max_items_to_search_in>0)? listsize-max_items_to_search_in:0;

   if(!ClearAllListViewItemSelections(m_terminal,m_expertstab))
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to deselect all listview items");
      return;
     }

   if(stop<=0)
     {
      if(!SelectAllListViewItems(m_terminal,m_expertstab))
        {
         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to select all listview items");
         return;
        }

      StringInit(items,int(listsize)*1000);
     }
   else
     {
      if(!SelectListViewItem(m_terminal,m_expertstab,IntegerToString(stop),IntegerToString(listsize-1)))
        {
         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to select "+IntegerToString(stop)+" listview items");
         return;
        }

      StringInit(items,int(stop)*1000);
     }

   ClipPut("");

   if(!ControlSend(m_terminal,m_expertstab,"{LCTRL down}c",0))
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to send keys to control");
      return;
     }

   ClipGet(items);

   ClipPut("");

   ClearAllListViewItemSelections(m_terminal,m_expertstab);

   int a_size=StringSplit(items,StringGetCharacter("\n",0),itemsarray);

   for(int i=(a_size-1); i>=0; i--)
     {
      if(itemsarray[i]=="")
         continue;

      if(StringFind(itemsarray[i],text_to_search_for,24)>=24)
        {
         entry=itemsarray[i];
         StringTrimRight(entry);
         break;;
        }
     }

   ArrayFree(itemsarray);

   return;

  }

//+------------------------------------------------------------------+
//|Gets last entry made to journal containing certain string         |
//+------------------------------------------------------------------+
void CTerminalController::GetLastJournalEntryByText(const string text_to_search_for,const int max_items_to_search_in,string &entry)
  {

   entry="";

   if(!IsInitialized())
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return;
     }

   string items;
   string itemsarray[];

   long listsize=GetListViewItemCount(m_terminal,m_journaltab);

   if(listsize<=0)
      return;

   long stop=(max_items_to_search_in>0)? listsize-max_items_to_search_in:0;

   if(!ClearAllListViewItemSelections(m_terminal,m_journaltab))
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to deselect all listview items");
      return;
     }

   if(stop<=0)
     {
      if(!SelectAllListViewItems(m_terminal,m_journaltab))
        {
         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to select all listview items");
         return;
        }

      StringInit(items,int(listsize)*1000);
     }
   else
     {
      if(!SelectListViewItem(m_terminal,m_journaltab,IntegerToString(stop),IntegerToString(listsize-1)))
        {
         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to select "+IntegerToString(stop)+" listview items");
         return;
        }

      StringInit(items,int(stop)*1000);
     }

   ClipPut("");

   if(!ControlSend(m_terminal,m_journaltab,"{LCTRL down}c",0))
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to send keys to control");
      return;
     }

   ClipGet(items);

   ClipPut("");

   ClearAllListViewItemSelections(m_terminal,m_journaltab);

   int a_size=StringSplit(items,StringGetCharacter("\n",0),itemsarray);

   for(int i=(a_size-1); i>=0; i--)
     {
      if(itemsarray[i]=="")
         continue;

      if(StringFind(itemsarray[i],text_to_search_for,24)>=24)
        {
         entry=itemsarray[i];
         StringTrimRight(entry);
         break;;
        }
     }

   ArrayFree(itemsarray);

   return;

  }
```

The SetNotificationId() method is a throwback to the original demonstration of this concept. It takes a string parameter of a maximum of four MetaQuotes IDs.

```
//+------------------------------------------------------------------+
//|set the MetaQuotes id                                             |
//+------------------------------------------------------------------+
bool CTerminalController::SetNotificationId(const string MetaQuotes_id)
  {
   if(!IsInitialized())
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return(false);
     }

   string itemsarray[];

   int totalids=StringSplit(MetaQuotes_id,StringGetCharacter(",",0),itemsarray);

   if(totalids>4)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Invalid parameter settings, Only maximum of 4 MetaQuotes ID's allowed");
      return(false);
     }

   HWND opt_window,opt_tab,edit,ok,checkbutton1,checkbutton2,checkbutton3;

   if(!WinMenuSelectItem(m_terminal,MENU_TOOLS,MENU_OPTIONS,"","","","","",""))
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Menu selection failure.");
      return(false);
     }

   Sleep(200);

   opt_window=WinGetHandle(OPTWINDOW);
   if(!opt_window)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Options Window not found.");
      return(false);
     }

   opt_tab=ControlGetHandle(opt_window,OPTWINDOW_TAB);
   if(!opt_tab)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Options Window tab control not found");
      WinClose(opt_window);
      return(false);
     }

   RECT wnsize;

   WinClientSize(opt_window,wnsize);

   int y=5;
   int x=5;

   string wintext=WinGetText(opt_window);
   while(StringFind(wintext,NOTIFICATIONS_TEXT)<0)
     {
      if(x<wnsize.right && ControlLeftClick(opt_window,opt_tab,1,x,5))
        {
         wintext=WinGetText(opt_window);
         x+=y;
        }
      else
        {
         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Notification settings tab not found");
         WinClose(opt_window);
         return(false);
        }
     }

   checkbutton1=ControlGetHandle(opt_window,OPTWINDOW_CHECK_1);
   if(!checkbutton1)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Enable Notifications check button not found");
      WinClose(opt_window);
      return(false);
     }

   checkbutton2=ControlGetHandle(opt_window,OPTWINDOW_CHECK_2);
   if(!checkbutton2)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Enable Notifications from terminal check button not found");
      WinClose(opt_window);
      return(false);
     }

   checkbutton3=ControlGetHandle(opt_window,OPTWINDOW_CHECK_3);
   if(!checkbutton3)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Enable Notifications from trade server check button not found");
      WinClose(opt_window);
      return(false);
     }

   ok=ControlGetHandle(opt_window,OPTWINDOW_OK);
   if(!ok)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"OK button not found");
      WinClose(opt_window);
      return(false);
     }

   edit=ControlGetHandle(opt_window,OPTWINDOW_EDIT);
   if(!checkbutton1)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Notification Ids edit control not found");
      WinClose(opt_window);
      return(false);
     }

   string current_id=ControlGetText(opt_window,edit);

   if(!StringCompare(current_id,MetaQuotes_id))
     {
      WinClose(opt_window);
      return(true);
     }

   if(!IsButtonChecked(opt_window,checkbutton1))
      CheckButton(opt_window,checkbutton1);

   if(IsControlEnabled(opt_window,checkbutton2) && !IsButtonChecked(opt_window,checkbutton2))
      CheckButton(opt_window,checkbutton2);

   if(IsControlEnabled(opt_window,checkbutton3) && !IsButtonChecked(opt_window,checkbutton3))
      CheckButton(opt_window,checkbutton3);

   if(ControlSetText(opt_window,edit,""))
      ControlSend(opt_window,edit,MetaQuotes_id,1);

   if(ControlLeftClick(opt_window,ok))
      Sleep(200);

   if(WinExists(opt_window))
      WinClose(opt_window);

   return(true);

  }
```

The CloseAlertDialogue() when called looks for any alert pop-up and closes.

```
//+------------------------------------------------------------------+
//| closes any pop up alert window                                   |
//+------------------------------------------------------------------+
void CTerminalController::CloseAlertDialogue(void)
  {
   static datetime lastcheck;

   if(!IsInitialized())
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return;
     }

   if(WinExists(ALERTWINDOW,""))
     {
      string alertmessage;
      StringInit(alertmessage,200);

      GetLastExpertsLogEntryByText(ALERTWINDOW,0,alertmessage);

      if(StringLen(alertmessage)>0)
        {
         datetime check=StringToTime(StringSubstr(alertmessage,0,24));
         if(check>lastcheck && check>iTime(NULL,PERIOD_D1,0))
           {
            WinClose(ALERTWINDOW,"");
            lastcheck=check;
           }
        }

     }

   return;
  }
```

The code for the whole class is shown below

```
//+------------------------------------------------------------------+
//|                                           terminalcontroller.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                                             https://www.mql5.com  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://www.mql5.com "
#include<autoItbase.mqh>

#define JOURNAL_TAB              "SysListView321"
#define EXPERTS_TAB              "SysListView327"

#define NAVIGATOR                "SysTreeView321"
#define NAVIGATOR_TAB_BAR        "AfxWnd140su3"

#define PROGRAM_WINDOW_TAB       "SysTabControl321"

#define EXPERTS_WINDOW_LISTVIEW  "SysListView321"
#define EXPERTS_REMOVE_BUTTON    "Button2"

#define DLLCHECK_BUTTON          "Button1"
#define ALGOTRADECHECK_BUTTON    "Button4"
#define OK_BUTTON_1              "Button7"
#define OK_BUTTON_2              "Button3"
#define OK_BUTTON_3              "Button5"
#define LOAD_BUTTON              "Button1"
#define YES_BUTTON               "Button1"

#define FILENAME_EDIT            "Edit1"

#define OPTWINDOW_TAB            "SysTabControl321"
#define OPTWINDOW_CHECK_1        "Button1"
#define OPTWINDOW_CHECK_2        "Button2"
#define OPTWINDOW_CHECK_3        "Button6"
#define OPTWINDOW_OK             "Button7"
#define OPTWINDOW_EDIT           "Edit1"

#define EXPERTS_WINDOW           "Experts"

#define NAVIGATORWINDOW          "Navigator"
#define TOOLBOXWINDOW            "Toolbox"

#define FILEDIALOGE_WINDOW       "Open"
#define WINDOWTEXT_DLL           "Allow DLL"
#define WINDOWTEXT_EA            "Expert Advisor settings"
#define WINDOWTEXT_INPUT         "Input"

#define OPTWINDOW                "Options"
#define NOTIFICATIONS_TEXT       "Enable Push notifications"

#define ALERTWINDOW              "Alert"

#define MENU_TOOLS             "&Tools"
#define MENU_OPTIONS           "&Options"
#define MENU_WINDOW            "&Window"
#define MENU_TILEWINDOWS       "&Tile Windows"

#define MENU_VIEW              "&View"
#define MENU_NAVIGATOR         "&Navigator"

#define MENU_CHARTS            "&Charts"
#define MENU_EXPERTS_LIST      "&Expert List"

enum ENUM_TYPE_PROGRAM
  {
   ENUM_TYPE_EXPERT=0,//EXPERT_ADVISOR
   ENUM_TYPE_SCRIPT//SCRIPT
  };

//+------------------------------------------------------------------+
//|Class CTerminalController                                         |
//| Purpose: class for scripting the terminal                        |
//+------------------------------------------------------------------+

class CTerminalController:public CAutoIt
  {
private:

   HWND              m_terminal;                 //terminal window handle
   HWND              m_journaltab;               //journal tab handle
   HWND              m_expertstab;               //experts tab handle

   HWND              m_navigator;                //navigator systreeview
   HWND              m_navigatortabs;            //navigator tab header
   HWND              m_navigatorwindow;          //navigator window
   HWND              m_systab32;                 //handle to inputs tabbed control with in inputs dialogue
   HWND              m_program;                  //window handle for user inputs dialogue
   HWND              m_expwindow;                //handle to window showing list of experts
   HWND              m_explistview;              //list view control for experts list in m_expwindow
   long              m_chartid;                  //chart id
   long              m_accountNum;               //account number
   string            m_buffer;                   //string buffer

public:
   //default constructor
                     CTerminalController(void): m_terminal(0),
                     m_journaltab(0),
                     m_expertstab(0),
                     m_navigator(0),
                     m_navigatortabs(0),
                     m_navigatorwindow(0),
                     m_systab32(0),
                     m_program(0),
                     m_expwindow(0),
                     m_explistview(0),
                     m_chartid(-1),
                     m_buffer("")
     {
      m_accountNum=AccountInfoInteger(ACCOUNT_LOGIN);
      StringInit(m_buffer,1000);
     }
   //destructor
                    ~CTerminalController(void)
     {

     }

   //public methods
   string            GetErrorDetails(void)  {     return(m_buffer);      }
   bool              ChartExpertAdd(const ENUM_TYPE_PROGRAM p_type,const string ea_name,const string ea_relative_path,const string ea_set_file,const string chart_symbol,const ENUM_TIMEFRAMES chart_tf);
   bool              ChartExpertRemove(const string ea_name,const string chart_symbol,const ENUM_TIMEFRAMES chart_tf);
   bool              ChartExpertRemoveAll(void);

   void              CloseAlertDialogue(void);

   void              GetLastExpertsLogEntry(string &entry);
   void              GetLastExpertsLogEntryByText(const string text_to_search_for,const int max_items_to_search_in,string &entry);
   void              GetLastJournalEntry(string &entry);
   void              GetLastJournalEntryByText(const string text_to_search_for,const int max_items_to_search_in,string &entry);
   bool              Initialize(const long accountNumber=0);
   bool              SwitchToNewChart(string n_symbol, const ENUM_TIMEFRAMES n_tf);
   bool              SetNotificationId(const string MetaQuotes_id);
   bool              ToggleAutoTrading(void);

private:
   //helper methods
   void              SetErrorDetails(const string text);
   void              Findbranch(const long childrenOnBranch,const string index,const string pname,string& sbuffer);
   bool              Findprogram(const ENUM_TYPE_PROGRAM pr_type, const string program_name,const string relative_path,string& sbuffer);
   string            PeriodToString(const ENUM_TIMEFRAMES chart_tf);
   bool              RemoveExpert(const string program_name,const string chart_symbol,const ENUM_TIMEFRAMES chart_tf);
   string            BrokerName(void);

  };

//+------------------------------------------------------------------+
//| sets or resets window and control handles used in the class      |
//+------------------------------------------------------------------+
bool CTerminalController::Initialize(const long accountNumber=0)
  {
   Init();

   if(accountNumber>0 && accountNumber!=m_accountNum)
     {
      m_accountNum=accountNumber;
      m_program=m_expwindow=m_systab32=m_explistview=m_navigatorwindow=0;
      m_terminal=m_journaltab=m_expertstab=m_navigator=m_navigatortabs=0;
      m_chartid=0;
     }

   if(m_terminal)
      return(true);

   m_terminal=WinGetHandle(IntegerToString(m_accountNum));
   if(!m_terminal)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Terminal window "+IntegerToString(m_accountNum)+" not found");
      return(false);
     }

   m_journaltab=ControlGetHandle(m_terminal,JOURNAL_TAB);
   if(!m_journaltab)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+JOURNAL_TAB+" not found");
      return(false);
     }

   m_expertstab=ControlGetHandle(m_terminal,EXPERTS_TAB);
   if(!m_expertstab)
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+EXPERTS_TAB+" not found");
      return(false);
     }

   m_navigatorwindow=ControlGetHandle(m_terminal,NAVIGATORWINDOW);
   if(!m_navigatorwindow)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+NAVIGATORWINDOW+" not found");
      return(false);
     }

   m_navigator=ControlGetHandle(m_terminal,NAVIGATOR);
   if(!m_navigator)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+NAVIGATOR+" not found");
      return(false);
     }

   m_navigatortabs=ControlGetHandle(m_terminal,NAVIGATOR_TAB_BAR);
   if(!m_navigatortabs)
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+NAVIGATOR_TAB_BAR+" not found");
      return(false);
     }

   StringInit(m_buffer,1000);

   return(true);
  }

//+------------------------------------------------------------------+
//|sets focus to existing or new chart window                        |
//+------------------------------------------------------------------+
bool CTerminalController::SwitchToNewChart(string n_symbol,const ENUM_TIMEFRAMES n_tf)
  {
   if(!IsInitialized())
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return(false);
     }

   string sarray[];

   StringToUpper(n_symbol);
   string newchartname=n_symbol+","+PeriodToString(n_tf);

   if(!WinMenuSelectItem(m_terminal,MENU_WINDOW,MENU_TILEWINDOWS,"","","","","",""))
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Failed to select menu item: Tile Windows");
      return(false);
     }

   Sleep(200);

   string windowtext=WinGetText(m_terminal);

   int find=StringFind(windowtext,ChartSymbol(ChartFirst())+","+PeriodToString(ChartPeriod(ChartFirst())));

   string chartstring=StringSubstr(windowtext,find);

   StringReplace(chartstring,"\n\n","\n");

   int sarraysize=StringSplit(chartstring,StringGetCharacter("\n",0),sarray);

   bool found=false;
   int i;

   long prevChart=0;
   m_chartid=ChartFirst();

   for(i=0; i<sarraysize; i++,)
     {
      if(sarray[i]=="")
         continue;

      if(i>0)
         m_chartid=ChartNext(prevChart);

      if(StringFind(sarray[i],newchartname)>=0)
        {
         found=true;
         break;
        }

      prevChart=m_chartid;
     }

   ArrayFree(sarray);

   HWND frameview=0;

   if(found)
      frameview=ControlGetHandle(m_terminal,newchartname);
   else
      if(ChartOpen(n_symbol,n_tf))
         frameview=ControlGetHandle(m_terminal,newchartname);

   if(frameview)
     {
      if(!WinSetState(frameview,SW_MAXIMIZE))
        {

         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Could not maximize "+newchartname+" chart window");
         return(false);
        }
     }
   else
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Chart window "+newchartname+" not found");
      return(false);
     }

   return(true);
  }
//+------------------------------------------------------------------+
//| Adds EA,Script,Service to a chart                                |
//+------------------------------------------------------------------+
bool CTerminalController::ChartExpertAdd(const ENUM_TYPE_PROGRAM p_type,const string ea_name,const string ea_relative_path,const string ea_set_file,const string chart_symbol,const ENUM_TIMEFRAMES chart_tf)
  {

   if(!SwitchToNewChart(chart_symbol,chart_tf))
     {
      return(false);
     }

   if(StringFind((p_type==ENUM_TYPE_EXPERT)?ChartGetString(m_chartid,CHART_EXPERT_NAME):ChartGetString(m_chartid,CHART_SCRIPT_NAME),ea_name)>=0)
     {
      return(true);
     }

   if(p_type==ENUM_TYPE_EXPERT && StringFind(ChartGetString(m_chartid,CHART_EXPERT_NAME),ChartGetString(ChartID(),CHART_EXPERT_NAME))>=0)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Cannot replace currenlty running Expert");
      return(false);
     }

   if(StringLen(ea_set_file)>0)
     {
      if(StringFind(ea_relative_path,".set")<0)
        {
         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Incorrect parameter setting, set file name should contain .set file extension");
         return(false);
        }
     }

   if(!IsControlVisible(m_terminal,m_navigator))
     {
      if(!IsControlVisible(m_terminal,m_navigatorwindow))
        {
         if(!WinMenuSelectItem(m_terminal,MENU_VIEW,MENU_NAVIGATOR,"","","","","",""))
           {
            SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Failed to select menu item : Navigator");
            return(false);
           }
        }

      if(!ControlLeftClick(m_terminal,m_navigatortabs,1,5,5))
        {
         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Failed to click on Navigator, Common tab");
         return(false);
        }
     }

   string treepath="";

   if(!Findprogram(p_type,ea_name,ea_relative_path,treepath))
      return(false);

   if(!SelectTreeViewItem(m_terminal,m_navigator,treepath))
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Failed to select treeview item: "+treepath);
      return(false);
     }

   if(!ControlSend(m_terminal,m_navigator,"{SHIFTDOWN}{F10}{SHIFTUP}c",0))
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Control Send failure");
      return(false);
     }

   Sleep(500);

   m_program=WinGetHandle(ea_name);

   HWND dialogue=WinGetHandle(BrokerName()+" - MetaTrader 5",ea_name);

   if(!m_program && !dialogue)
     {
      if(p_type==ENUM_TYPE_EXPERT && StringFind(ChartGetString(m_chartid,CHART_EXPERT_NAME),ea_name)<0)
        {
         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Could not add program to chart");
         return(false);
        }
      else
        {
         return(true);
        }
     }

   if(!m_program && dialogue)// replace current ea
     {
      HWND button = ControlGetHandle(dialogue,YES_BUTTON);

      if(button)
        {
         if(ControlLeftClick(dialogue,button))
           {
            Sleep(200);
            m_program=WinGetHandle(ea_name);
           }
         else
           {

            SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Failed left click on yes button");
            WinClose(dialogue);
            return(false);
           }
        }
      else
        {

         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Yes button for dialogue not found");
         WinClose(dialogue);
         return(false);
        }

     }

   m_systab32=ControlGetHandle(m_program,PROGRAM_WINDOW_TAB);
   if(!m_systab32)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+PROGRAM_WINDOW_TAB+ "not found");
      return(false);
     }

   long totaltabs=TotalTabs(m_program,m_systab32);
   bool algoenabled=false;
   bool inputmod=false;
   bool dllenabled=(totaltabs<3)?true:false;

   do
     {
      string windowtext=WinGetText(m_program);

      if(StringFind(windowtext,WINDOWTEXT_DLL)>=0)
        {
         StringFill(windowtext,0);
         HWND button=ControlGetHandle(m_program,DLLCHECK_BUTTON);

         if(!button)
           {
            WinClose(m_program);
              {
               SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"DLL check button not found");
               return(false);
              }
           }

         if(!IsButtonChecked(m_program,button))
           {
            if(CheckButton(m_program,button))
               dllenabled=true;
           }
         else
            dllenabled=true;

         if(dllenabled)
           {
            if(TabLeft(m_program,m_systab32))
               if(!TabLeft(m_program,m_systab32))
                 {
                  inputmod=true;
                 }
           }
         else
           {
            WinClose(m_program);
            SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Could not enable DLL loading");
            return(false);
           }
        }
      else
         if(StringFind(windowtext,WINDOWTEXT_EA)==0)
           {
            StringFill(windowtext,0);
            HWND button=ControlGetHandle(m_program,ALGOTRADECHECK_BUTTON);

            if(!button)
              {
               WinClose(m_program);
               SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Algo check button not found");
               return(false);
              }

            if(!IsButtonChecked(m_program,button))
              {
               if(CheckButton(m_program,button))
                  algoenabled=true;
              }
            else
               algoenabled=true;

            if(algoenabled)
              {
               if(!inputmod)
                 {
                  if(!TabRight(m_program,m_systab32))
                    {
                     HWND okbutton=ControlGetHandle(m_program,OK_BUTTON_3);
                     if(!okbutton)
                       {
                        WinClose(m_program);
                        SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Failed find Ok button");
                        return(false);
                       }

                     if(!ControlLeftClick(m_program,okbutton))
                       {
                        WinClose(m_program);
                        SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Left Click failure");
                        return(false);
                       }
                     else
                       {
                        break;
                       }
                    }
                 }
               else
                 {
                  HWND okbutton=ControlGetHandle(m_program,OK_BUTTON_3);
                  if(!okbutton)
                    {
                     WinClose(m_program);
                     SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Failed find Ok button");
                     return(false);
                    }

                  if(!ControlLeftClick(m_program,okbutton))
                    {
                     WinClose(m_program);
                     SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Left Click failure");
                     return(false);
                    }
                 }
              }
            else
              {
               WinClose(m_program);
               SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Unable to enable autotrading");
               return(false);
              }
           }
         else
            if(StringFind(windowtext,WINDOWTEXT_INPUT)==0)
              {
               StringFill(windowtext,0);

               HWND button=ControlGetHandle(m_program,LOAD_BUTTON);

               HWND okbutton=ControlGetHandle(m_program,(totaltabs>2)?OK_BUTTON_1:OK_BUTTON_2);

               if(!okbutton)
                 {
                  WinClose(m_program);
                  SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Failed to find OK button");
                  return(false);
                 }

               if(StringLen(ea_set_file)>0)
                 {
                  if(!button)
                    {
                     WinClose(m_program);
                     SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Failed to find .set file load button");
                     return(false);
                    }

                  if(!ControlLeftClick(m_program,button))
                    {
                     WinClose(m_program);
                     SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Left click failure");
                     return(false);
                    }

                  HWND filewin=0;
                  int try
                        =50;
                  while(!filewin && try
                           >0)
                       {
                        filewin=WinGetHandle(FILEDIALOGE_WINDOW);
                        try
                           --;
                        Sleep(200);
                       }

                  HWND filedit=ControlGetHandle(filewin,FILENAME_EDIT);

                  if(!filedit || !filewin)
                    {
                     if(!filedit)
                       {
                        if(filewin)
                           WinClose(filewin);
                       }
                     WinClose(m_program);
                     SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" File dialogue failure");
                     return(false);
                    }

                  if(ControlSetText(filewin,filedit,""))
                    {
                     if(ControlSend(filewin,filedit,ea_set_file,1))
                       {
                        Sleep(200);
                        if(ControlSend(filewin,filedit,"{ENTER}",0))
                           Sleep(300);
                       }
                    }

                  if(WinExists(filewin))
                    {
                     WinClose(filewin);
                     WinClose(m_program);
                     SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Failed to set .set file through file dialogue");
                     return(false);
                    }
                 }

               inputmod=true;

               if(algoenabled)
                 {
                  if(ControlLeftClick(m_program,okbutton))
                     break;
                  else
                    {
                     WinClose(m_program);
                     SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Failed to click OK button");
                     return(false);
                    }
                 }
               else
                 {
                  if(!TabLeft(m_program,m_systab32))
                    {
                     if(ControlLeftClick(m_program,okbutton))
                        break;
                     else
                       {
                        WinClose(m_program);
                        SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Failed to click OK button");
                        return(false);
                       }
                    }
                 }
              }
     }
   while(!inputmod||!dllenabled||!algoenabled);

   int try
         =50;

   while(WinExists(m_program) && try
            >0)
        {
         Sleep(500);
         try
            --;
        }

   if(WinExists(m_program) && !try)
     {
      WinClose(m_program);
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Failed to add program to chart");
      return(false);
     }

   if(p_type==ENUM_TYPE_EXPERT && StringFind(ChartGetString(m_chartid,CHART_EXPERT_NAME),ea_name)<0)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+" Operation failed could not add program to chart");
      return(false);
     }

   return(true);

  }
//+------------------------------------------------------------------+
//|Removes EA,Script from a chart                                    |
//+------------------------------------------------------------------+
bool CTerminalController::ChartExpertRemove(const string ea_name,const string chart_symbol,const ENUM_TIMEFRAMES chart_tf)
  {
   return(RemoveExpert(ea_name,chart_symbol,chart_tf));
  }
//+------------------------------------------------------------------+
//| Removes all scripts and experts from all charts                  |
//+------------------------------------------------------------------+
bool CTerminalController::ChartExpertRemoveAll(void)
  {
   if(!IsInitialized())
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return(false);
     }

   if(!WinMenuSelectItem(m_terminal,MENU_CHARTS,MENU_EXPERTS_LIST,"","","","","",""))
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Menu selection failure.");
      return(false);
     }

   Sleep(200);

   m_expwindow=WinGetHandle(EXPERTS_WINDOW);
   if(!m_expwindow)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+EXPERTS_WINDOW+" window not found");
      return(false);
     }

   m_explistview=ControlGetHandle(m_expwindow,EXPERTS_WINDOW_LISTVIEW);
   if(!m_explistview)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+EXPERTS_WINDOW_LISTVIEW+" control not found");
      WinClose(m_expwindow);
      return(false);
     }

   HWND remove_button=ControlGetHandle(m_expwindow,EXPERTS_REMOVE_BUTTON);
   if(!remove_button)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Remove button not found");
      WinClose(m_expwindow);
      return(false);
     }

   long listsize=GetListViewItemCount(m_expwindow,m_explistview);

   if(listsize<=1)
     {
      WinClose(m_expwindow);
      return(true);
     }

   string prgname;

   ENUM_PROGRAM_TYPE mql_program=(ENUM_PROGRAM_TYPE)MQLInfoInteger(MQL_PROGRAM_TYPE);
   switch(mql_program)
     {
      case PROGRAM_SCRIPT:
        {
         prgname=ChartGetString(ChartID(),CHART_SCRIPT_NAME);
         break;
        }
      case PROGRAM_EXPERT:
        {
         prgname=ChartGetString(ChartID(),CHART_EXPERT_NAME);
         break;
        }
      default:
         prgname="";
     }

   if(prgname=="")
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Unable to determine name of current program");
      WinClose(m_expwindow);
      return(false);
     }

   do
     {
      ClearAllListViewItemSelections(m_expwindow,m_explistview);

      for(int i=0; i<int(listsize); i++)
        {
         if(!SelectListViewItem(m_expwindow,m_explistview,IntegerToString(i),""))
            continue;

         string pname=GetListViewItemText(m_expwindow,m_explistview,IntegerToString(i),"0");

         if(StringFind(pname,prgname)<0)
           {
            if(IsControlEnabled(m_expwindow,remove_button))
              {
               if(ControlLeftClick(m_expwindow,remove_button))
                 {
                  listsize--;
                  Sleep(500);
                  break;
                 }
               else
                 {
                  SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Failed to click remove button");
                  break;
                 }
              }
            else
              {
               SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Remove button is disabled");
               break;
              }
           }
         else
           {
            ClearAllListViewItemSelections(m_expwindow,m_explistview);
            continue;
           }
        }
     }
   while(listsize>1);

   while(WinExists(m_expwindow))
      WinClose(m_expwindow);

   return(true);

  }
//+------------------------------------------------------------------+
//|Gets the last journal entry                                       |
//+------------------------------------------------------------------+
void CTerminalController::GetLastJournalEntry(string &entry)
  {
   entry="";

   if(!IsInitialized())
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return;
     }

   long listsize=GetListViewItemCount(m_terminal,m_journaltab);

   if(listsize<=0)
      return;

   ClipPut("");

   if(!ClearAllListViewItemSelections(m_terminal,m_journaltab))
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to deselect all listview items");
      return;
     }

   if(!SelectListViewItem(m_terminal,m_journaltab,IntegerToString(listsize-1),""))
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to select last listview item");
      return;
     }

   if(!ControlSend(m_terminal,m_journaltab,"{LCTRL down}c{LCTRL up}",0))
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to send keys to control");
      return;
     }

   ClipGet(entry);

   StringTrimRight(entry);

   ClipPut("");

   ClearAllListViewItemSelections(m_terminal,m_journaltab);

   return;
  }
//+------------------------------------------------------------------+
//|Gets last entry made to experts log file                          |
//+------------------------------------------------------------------+
void CTerminalController::GetLastExpertsLogEntry(string &entry)
  {
   entry="";

   if(!IsInitialized())
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return;
     }

   long listsize=GetListViewItemCount(m_terminal,m_expertstab);

   if(listsize<=0)
      return;

   ClipPut("");

   if(!ClearAllListViewItemSelections(m_terminal,m_expertstab))
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to deselect all listview items");
      return;
     }

   if(!SelectListViewItem(m_terminal,m_expertstab,IntegerToString(listsize-1),""))
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to select last listview item");
      return;
     }

   if(!ControlSend(m_terminal,m_expertstab,"{LCTRL down}c{LCTRL up}",0))
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to send keys to control");
      return;
     }

   ClipGet(entry);

   StringTrimRight(entry);

   ClipPut("");

   ClearAllListViewItemSelections(m_terminal,m_expertstab);

   return;
  }
//+------------------------------------------------------------------+
//|Gets last entry made to experts log file containg certain string  |
//+------------------------------------------------------------------+
void CTerminalController::GetLastExpertsLogEntryByText(const string text_to_search_for,const int max_items_to_search_in,string &entry)
  {
   entry="";

   if(!IsInitialized())
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return;
     }

   string items;
   string itemsarray[];

   long listsize=GetListViewItemCount(m_terminal,m_expertstab);

   if(listsize<=0)
      return;

   long stop=(max_items_to_search_in>0)? listsize-max_items_to_search_in:0;

   if(!ClearAllListViewItemSelections(m_terminal,m_expertstab))
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to deselect all listview items");
      return;
     }

   if(stop<=0)
     {
      if(!SelectAllListViewItems(m_terminal,m_expertstab))
        {
         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to select all listview items");
         return;
        }

      StringInit(items,int(listsize)*1000);
     }
   else
     {
      if(!SelectListViewItem(m_terminal,m_expertstab,IntegerToString(stop),IntegerToString(listsize-1)))
        {
         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to select "+IntegerToString(stop)+" listview items");
         return;
        }

      StringInit(items,int(stop)*1000);
     }

   ClipPut("");

   if(!ControlSend(m_terminal,m_expertstab,"{LCTRL down}c",0))
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to send keys to control");
      return;
     }

   ClipGet(items);

   ClipPut("");

   ClearAllListViewItemSelections(m_terminal,m_expertstab);

   int a_size=StringSplit(items,StringGetCharacter("\n",0),itemsarray);

   for(int i=(a_size-1); i>=0; i--)
     {
      if(itemsarray[i]=="")
         continue;

      if(StringFind(itemsarray[i],text_to_search_for,24)>=24)
        {
         entry=itemsarray[i];
         StringTrimRight(entry);
         break;;
        }
     }

   ArrayFree(itemsarray);

   return;

  }

//+------------------------------------------------------------------+
//|Gets last entry made to journal containing certain string         |
//+------------------------------------------------------------------+
void CTerminalController::GetLastJournalEntryByText(const string text_to_search_for,const int max_items_to_search_in,string &entry)
  {

   entry="";

   if(!IsInitialized())
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return;
     }

   string items;
   string itemsarray[];

   long listsize=GetListViewItemCount(m_terminal,m_journaltab);

   if(listsize<=0)
      return;

   long stop=(max_items_to_search_in>0)? listsize-max_items_to_search_in:0;

   if(!ClearAllListViewItemSelections(m_terminal,m_journaltab))
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to deselect all listview items");
      return;
     }

   if(stop<=0)
     {
      if(!SelectAllListViewItems(m_terminal,m_journaltab))
        {
         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to select all listview items");
         return;
        }

      StringInit(items,int(listsize)*1000);
     }
   else
     {
      if(!SelectListViewItem(m_terminal,m_journaltab,IntegerToString(stop),IntegerToString(listsize-1)))
        {
         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to select "+IntegerToString(stop)+" listview items");
         return;
        }

      StringInit(items,int(stop)*1000);
     }

   ClipPut("");

   if(!ControlSend(m_terminal,m_journaltab,"{LCTRL down}c",0))
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"failed to send keys to control");
      return;
     }

   ClipGet(items);

   ClipPut("");

   ClearAllListViewItemSelections(m_terminal,m_journaltab);

   int a_size=StringSplit(items,StringGetCharacter("\n",0),itemsarray);

   for(int i=(a_size-1); i>=0; i--)
     {
      if(itemsarray[i]=="")
         continue;

      if(StringFind(itemsarray[i],text_to_search_for,24)>=24)
        {
         entry=itemsarray[i];
         StringTrimRight(entry);
         break;;
        }
     }

   ArrayFree(itemsarray);

   return;

  }

//+------------------------------------------------------------------+
//|Helper function detaches program from a chart                     |
//+------------------------------------------------------------------+
bool CTerminalController::RemoveExpert(const string program_name,const string chart_symbol,const ENUM_TIMEFRAMES chart_tf)
  {
   if(!IsInitialized())
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return(false);
     }

   if(!WinMenuSelectItem(m_terminal,MENU_CHARTS,MENU_EXPERTS_LIST,"","","","","",""))
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Menu selection failure.");
      return(false);
     }

   Sleep(200);

   m_expwindow=WinGetHandle(EXPERTS_WINDOW);
   if(!m_expwindow)
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+EXPERTS_WINDOW+" window not found");
      return(false);
     }

   m_explistview=ControlGetHandle(m_expwindow,EXPERTS_WINDOW_LISTVIEW);
   if(!m_explistview)
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+EXPERTS_WINDOW_LISTVIEW+" control not found");
      return(false);
     }

   HWND remove_button=ControlGetHandle(m_expwindow,EXPERTS_REMOVE_BUTTON);
   if(!remove_button)
     {

      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Remove button not found");
      return(false);
     }

   long listsize=GetListViewItemCount(m_expwindow,m_explistview);

   if(listsize<=0)
     {
      return(true);
     }

   string newchartname=chart_symbol;
   StringToUpper(newchartname);
   newchartname+=","+PeriodToString(chart_tf);

   bool found=false;

   ClearAllListViewItemSelections(m_expwindow,m_explistview);

   for(int i=0; i<int(listsize); i++)
     {
      if(!SelectListViewItem(m_expwindow,m_explistview,IntegerToString(i),""))
         continue;

      string pname=GetListViewItemText(m_expwindow,m_explistview,IntegerToString(i),"0");

      string chartname=GetListViewItemText(m_expwindow,m_explistview,IntegerToString(i),"1");

      if(StringFind(pname,program_name)>=0 && StringFind(chartname,newchartname)>=0)
        {
         if(IsControlEnabled(m_expwindow,remove_button))
            if(ControlLeftClick(m_expwindow,remove_button))
               found=true;
        }

      if(found)
         break;

      ClearAllListViewItemSelections(m_expwindow,m_explistview);

     }

   WinClose(m_expwindow);

   return(found);
  }
//+------------------------------------------------------------------+
//|helper method converts period names to string format              |
//+------------------------------------------------------------------+
string CTerminalController::PeriodToString(const ENUM_TIMEFRAMES chart_tf)
  {
   string strper="";

   switch(chart_tf)
     {
      case PERIOD_MN1:
         strper="Monthly";
         break;
      case PERIOD_W1:
         strper="Weekly";
         break;
      case PERIOD_D1:
         strper="Daily";
         break;
      default:
         strper=StringSubstr(EnumToString(chart_tf),StringFind(EnumToString(chart_tf),"_")+1);
         break;
     }

   return strper;
  }
//+---------------------------------------------------------------------+
//|searches navigator for a program and outputs its location on the tree|
//+---------------------------------------------------------------------+
bool CTerminalController::Findprogram(const ENUM_TYPE_PROGRAM pr_type,const string program_name,const string relative_path,string &sbuffer)
  {

   long listsize=GetTreeViewItemCount(m_terminal,m_navigator,"#0");

   if(!listsize)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Root treeview control is empty");
      return(false);
     }
   else
     {
      string rpath="";

      if(relative_path=="")
         rpath=relative_path;
      else
        {
         if(StringFind(relative_path,"\\")==0)
            rpath=StringSubstr(relative_path,1);
         else
            rpath=relative_path;
         if(StringFind(rpath,"\\",StringLen(rpath)-1)<0)
            rpath+="\\";
        }

      switch(pr_type)
        {
         case ENUM_TYPE_EXPERT:
           {
            string fullpath="Expert Advisors\\"+rpath+program_name;
            Findbranch(listsize,"#0",fullpath,sbuffer);
            break;
           }
         case ENUM_TYPE_SCRIPT:
           {
            string fullpath="Scripts\\"+rpath+program_name;
            Findbranch(listsize,"#0",fullpath,sbuffer);
            break;
           }
         default:
            Findbranch(listsize,"#0","",sbuffer);
            break;
        }

     }

   if(sbuffer=="")
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Operation failed could not find "+program_name);
      return(false);
     }
   else
      return(true);

  }
//+-----------------------------------------------------------------------+
//|recursively searches systreeview for program and builds location string|
//+-----------------------------------------------------------------------+
void CTerminalController::Findbranch(const long childrenOnBranch,const string index,const string pname,string &sbuffer)
  {
   if(pname=="" ||  index=="")
     {
      sbuffer=index;
      return;
     }
   else
     {
      if(childrenOnBranch<=0)
         return;
      else
        {
         int find=StringFind(pname,"\\");

         long ss=0;
         long i;

         for(i=0; i<childrenOnBranch; i++)
           {

            ss=GetTreeViewItemCount(m_terminal,m_navigator,index+"|#"+IntegerToString(i));

            string search=(find>=0)?StringSubstr(pname,0,find):pname;

            string treebranchtext=GetTreeViewItemText(m_terminal,m_navigator,index+"|#"+IntegerToString(i));

            if(StringFind(treebranchtext,search)>=0 && StringLen(treebranchtext)==StringLen(search))
               break;

           }

         string npath=(find>=0)?StringSubstr(pname,find+1):"";

         Findbranch(ss,(i<childrenOnBranch)?index+"|#"+IntegerToString(i):"",npath,sbuffer);
        }
     }

   return;
  }
//+------------------------------------------------------------------+
//| Get the broker name from the terminal window title               |
//+------------------------------------------------------------------+
string CTerminalController::BrokerName(void)
  {
   string full_title=WinGetTitle(m_terminal);
   int find=StringFind(full_title,"-");
   string m_brokername=StringSubstr(full_title,find+1,StringFind(full_title,"-",find+1)-find-1);

   StringTrimLeft(m_brokername);
   StringTrimRight(m_brokername);

   return(m_brokername);
  }

//+------------------------------------------------------------------+
//| Set error details                                                |
//+------------------------------------------------------------------+
void CTerminalController::SetErrorDetails(const string _text)
  {
   StringFill(m_buffer,0);

   m_buffer=_text;

   return;
  }
//+------------------------------------------------------------------+
//|set the MetaQuotes id                                             |
//+------------------------------------------------------------------+
bool CTerminalController::SetNotificationId(const string MetaQuotes_id)
  {
   if(!IsInitialized())
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return(false);
     }

   string itemsarray[];

   int totalids=StringSplit(MetaQuotes_id,StringGetCharacter(",",0),itemsarray);

   if(totalids>4)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Invalid parameter settings, Only maximum of 4 MetaQuotes ID's allowed");
      return(false);
     }

   HWND opt_window,opt_tab,edit,ok,checkbutton1,checkbutton2,checkbutton3;

   if(!WinMenuSelectItem(m_terminal,MENU_TOOLS,MENU_OPTIONS,"","","","","",""))
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Menu selection failure.");
      return(false);
     }

   Sleep(200);

   opt_window=WinGetHandle(OPTWINDOW);
   if(!opt_window)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Options Window not found.");
      return(false);
     }

   opt_tab=ControlGetHandle(opt_window,OPTWINDOW_TAB);
   if(!opt_tab)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Options Window tab control not found");
      WinClose(opt_window);
      return(false);
     }

   RECT wnsize;

   WinClientSize(opt_window,wnsize);

   int y=5;
   int x=5;

   string wintext=WinGetText(opt_window);
   while(StringFind(wintext,NOTIFICATIONS_TEXT)<0)
     {
      if(x<wnsize.right && ControlLeftClick(opt_window,opt_tab,1,x,5))
        {
         wintext=WinGetText(opt_window);
         x+=y;
        }
      else
        {
         SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Notification settings tab not found");
         WinClose(opt_window);
         return(false);
        }
     }

   checkbutton1=ControlGetHandle(opt_window,OPTWINDOW_CHECK_1);
   if(!checkbutton1)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Enable Notifications check button not found");
      WinClose(opt_window);
      return(false);
     }

   checkbutton2=ControlGetHandle(opt_window,OPTWINDOW_CHECK_2);
   if(!checkbutton2)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Enable Notifications from terminal check button not found");
      WinClose(opt_window);
      return(false);
     }

   checkbutton3=ControlGetHandle(opt_window,OPTWINDOW_CHECK_3);
   if(!checkbutton3)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Enable Notifications from trade server check button not found");
      WinClose(opt_window);
      return(false);
     }

   ok=ControlGetHandle(opt_window,OPTWINDOW_OK);
   if(!ok)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"OK button not found");
      WinClose(opt_window);
      return(false);
     }

   edit=ControlGetHandle(opt_window,OPTWINDOW_EDIT);
   if(!checkbutton1)
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"Notification Ids edit control not found");
      WinClose(opt_window);
      return(false);
     }

   string current_id=ControlGetText(opt_window,edit);

   if(!StringCompare(current_id,MetaQuotes_id))
     {
      WinClose(opt_window);
      return(true);
     }

   if(!IsButtonChecked(opt_window,checkbutton1))
      CheckButton(opt_window,checkbutton1);

   if(IsControlEnabled(opt_window,checkbutton2) && !IsButtonChecked(opt_window,checkbutton2))
      CheckButton(opt_window,checkbutton2);

   if(IsControlEnabled(opt_window,checkbutton3) && !IsButtonChecked(opt_window,checkbutton3))
      CheckButton(opt_window,checkbutton3);

   if(ControlSetText(opt_window,edit,""))
      ControlSend(opt_window,edit,MetaQuotes_id,1);

   if(ControlLeftClick(opt_window,ok))
      Sleep(200);

   if(WinExists(opt_window))
      WinClose(opt_window);

   return(true);

  }
//+------------------------------------------------------------------+
//| closes any pop up alert window                                   |
//+------------------------------------------------------------------+
void CTerminalController::CloseAlertDialogue(void)
  {
   static datetime lastcheck;

   if(!IsInitialized())
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return;
     }

   if(WinExists(ALERTWINDOW,""))
     {
      string alertmessage;
      StringInit(alertmessage,200);

      GetLastExpertsLogEntryByText(ALERTWINDOW,0,alertmessage);

      if(StringLen(alertmessage)>0)
        {
         datetime check=StringToTime(StringSubstr(alertmessage,0,24));
         if(check>lastcheck && check>iTime(NULL,PERIOD_D1,0))
           {
            WinClose(ALERTWINDOW,"");
            lastcheck=check;
           }
        }

     }

   return;
  }
//+------------------------------------------------------------------+
//|Enable and disable autotrading                                    |
//+------------------------------------------------------------------+
bool CTerminalController::ToggleAutoTrading(void)
  {
   if(!IsInitialized())
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return(false);
     }

   if(!ControlSend(m_terminal,m_navigatortabs,"{LCTRL down}e{LCTRL up}",1))
     {
      SetErrorDetails(__FUNCTION__+" "+string(__LINE__)+". "+"AutoIt library not Initialized");
      return(false);
     }

   return(true);
  }

//+------------------------------------------------------------------+
```

### Using the CTerminalController class

The first example demonstrates adding programs to charts and also removing them. Add the name of a program and specify the chart. The last input parameter dictates how long the added program will run in seconds. After the specified time, the program will be removed if it is still running.

```
//+------------------------------------------------------------------+
//|                                           TerminalController.mq5 |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://www.mql5.com "
#property version   "1.00"
#include<terminalcontroller.mqh>

//--- input parameters
input string   ProgramName="Controls";
input string   Path="Examples\\Controls";
input string   Symbolname="BTCUSD";
input ENUM_TIMEFRAMES    Timeframe=PERIOD_D1;
input ENUM_TYPE_PROGRAM  Type=0;
input string   SetFileName="";
input int      RemoveProgramTimer=10;//Max seconds added program will run

CTerminalController terminal;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   if(!terminal.Initialize())
     {
      Print("Service failed to initialize CTerminalController instance");
      return(INIT_FAILED);
     }

   if(!terminal.ChartExpertAdd(Type,ProgramName,Path,SetFileName,Symbolname,Timeframe))
      Print("Failed to add "+ProgramName+" to chart. Error > "+terminal.GetErrorDetails());

   if(RemoveProgramTimer>0)
      EventSetTimer(RemoveProgramTimer);
   else
      EventSetTimer(10);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert timer function                                            |
//+------------------------------------------------------------------+
void OnTimer(void)
  {
   if(!terminal.ChartExpertRemove(ProgramName,Symbolname,Timeframe))
     {
      Print("Failed to remove "+ProgramName+". Error > "+terminal.GetErrorDetails());
     }
   else
     {
      string comment;
      terminal.GetLastJournalEntry(comment);
      Print(comment);
      ExpertRemove();
     }

  }
//+------------------------------------------------------------------+
```

The HandleAlerts expert advisor sends out a visual pop-up alert for each new bar. Then the CloseAlertDialogue() method is used to close any pop-up dialogues generated by the terminal.

```
//+------------------------------------------------------------------+
//|                                                 HandleAlerts.mq5 |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                                             https://www.mql5.com  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://www.mql5.com "
#property version   "1.00"
#include<terminalcontroller.mqh>
//--- input parameters
input int      Seconds=5;

CTerminalController terminal;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   if(!terminal.Initialize())
     {
      Print("Service failed to initialize CTerminalController instance");
      return(INIT_FAILED);
     }
   if(Seconds>0)
      EventSetTimer(Seconds);
   else
      EventSetTimer(10);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   static datetime lastsend;

   if(iTime(NULL,0,0)!=lastsend)
     {
      Alert("New Alert message sent at "+TimeToString(TimeCurrent()));
     }
  }
//+------------------------------------------------------------------+
//| Expert Timer functions                                           |
//+------------------------------------------------------------------+
void OnTimer()
  {
   terminal.CloseAlertDialogue();
  }
//+------------------------------------------------------------------+
```

The last example is Send Push script, the script works similarly to the example provided in the article managing the MetaTrader Terminal via DLL.

```
//+------------------------------------------------------------------+
//|                                                    Send_Push.mq5 |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://www.mql5.com "
#property version   "1.00"
#property script_show_inputs
#include<terminalcontroller.mqh>
//--- input parameters
input string     message_text="test";
input string     Mq_ID="1C2F1442,2C2F1442,3C2F1442,4C2F1442";

CTerminalController terminal;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   if(!terminal.Initialize())
     {
      Print("Service failed to initialize CTerminalController instance");
      return;
     }

   if(!terminal.SetNotificationId(Mq_ID))
     {
      Print("Failed to set MetaQuotes id's. Error > "+terminal.GetErrorDetails());
      return;
     }

   if(!SendNotification(message_text))
     {
      int Err=GetLastError();
      switch(Err)
        {
         case 4515:
            Alert("Waiting: Failed to send ", Mq_ID);
            break;
         case 4516:
            Alert("Err: Invalid message text ", message_text);
            return;
            break;
         case 4517:
            Alert("Waiting: Invalid ID list ", Mq_ID);
            break;
         case 4518:
            Alert("Err: Too frequent requests! ");
            return;
            break;
        }
     }

  }
//+------------------------------------------------------------------+
```

### Conclusion

The article provided a brief description of how to work with AutoIt. We observed how to use the AutoItX library by integrating it with MQL5 and also documented the creation of a class that uses the AutoIt dll. The only limitation found when using the library is that access violation errors will occur if two or more MetaTrader 5 programs that use AutoItX are run simultaneously. It is therefore recommended to run single instances of MetaTrader 5 programs that reference any AutoItX code.

| Folder | Contents | Description |
| --- | --- | --- |
| MetaTrader 5zip\\MQL5\\include | autoIt.mqh, autoItbase.mqh,terminalcontroller.mqh | autoIt.mqh contains AutoItX function import statements, autoItbase.mqh contains CAutoIt class and terminalcontroller.mqh contains CTerminalController class |
| MetaTrader 5zip\\MQL5\\Experts | TerminalController.mq5,HandleAlerts.mq5 | TerminalController EA demonstrates the auto removal and addition of either expert advisors or scripts from a chart.<br>HandleAlerts EA shows how to automate the handling of pop-up alert windows |
| MetaTrader 5zip\\MQL5\\Scripts | TerminalUIComponents.mq5,Send\_Push.mq5 | TerminalUICommpents demonstrates the use of AutoItX by calling the imported functions directly, Send\_Push script adds new MetaQuotes IDs to the terminal before sending a notification |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10130.zip "Download all attachments in the single ZIP archive")

[Mt5.zip](https://www.mql5.com/en/articles/download/10130/mt5.zip "Download Mt5.zip")(15.46 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Building Volatility models in MQL5 (Part I): The Initial Implementation](https://www.mql5.com/en/articles/20589)
- [Bivariate Copulae in MQL5 (Part 2): Implementing Archimedean copulae in MQL5](https://www.mql5.com/en/articles/19931)
- [Bivariate Copulae in MQL5 (Part 1): Implementing Gaussian and Student's t-Copulae for Dependency Modeling](https://www.mql5.com/en/articles/18361)
- [Dynamic mode decomposition applied to univariate time series in MQL5](https://www.mql5.com/en/articles/19188)
- [Singular Spectrum Analysis in MQL5](https://www.mql5.com/en/articles/18777)
- [Websockets for MetaTrader 5: Asynchronous client connections with the Windows API](https://www.mql5.com/en/articles/17877)
- [Resampling techniques for prediction and classification assessment in MQL5](https://www.mql5.com/en/articles/17446)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/383090)**
(9)


![Denis Nikolaev](https://c.mql5.com/avatar/avatar_na2.png)

**[Denis Nikolaev](https://www.mql5.com/en/users/zaqcde)**
\|
7 Apr 2022 at 15:57

Very interesting and useful article. Thank you to the author!


![Pavel Shestakov](https://c.mql5.com/avatar/avatar_na2.png)

**[Pavel Shestakov](https://www.mql5.com/en/users/grimdragon)**
\|
7 Jun 2022 at 12:23

And please tell me how to set the third parameter in this function int AU3\_WinGetPos(string, string, LPRECT); and get these coordinates?


![Maxim Kuznetsov](https://c.mql5.com/avatar/2016/1/56935A91-AF51.png)

**[Maxim Kuznetsov](https://www.mql5.com/en/users/nektomk)**
\|
7 Jun 2022 at 16:53

there is a more useful and Microsoft spy in the VS kit.

AutoIt is not free software (it's shareware), you have to pay for it and it's unclear how you missed its promotional article.

![Jan4654](https://c.mql5.com/avatar/avatar_na2.png)

**[Jan4654](https://www.mql5.com/en/users/jan4654)**
\|
11 May 2023 at 00:17

Yeah AutoIt is great I actually wrote AutoMTF in Autoit


![jsteamid -](https://c.mql5.com/avatar/2023/11/654FA398-946B.png)

**[jsteamid -](https://www.mql5.com/en/users/jsteamid)**
\|
14 Nov 2023 at 15:05

Is there any way to set the color of the red and blue in Market Watch to solid color and not gradation ? I use Autoit to read the color but everytime I rerun the [trading app](https://www.mql5.com/en/market "A Market of Applications for the MetaTrader 5 and MetaTrader 4") it's changing making the pixel color checker also changed.

I also need a way to read profit of each open position using autoit

![Graphics in DoEasy library (Part 88): Graphical object collection — two-dimensional dynamic array for storing dynamically changing object properties](https://c.mql5.com/2/44/MQL5-avatar-doeasy-library3-2.png)[Graphics in DoEasy library (Part 88): Graphical object collection — two-dimensional dynamic array for storing dynamically changing object properties](https://www.mql5.com/en/articles/10091)

In this article, I will create a dynamic multidimensional array class with the ability to change the amount of data in any dimension. Based on the created class, I will create a two-dimensional dynamic array to store some dynamically changed properties of graphical objects.

![Fix PriceAction Stoploss or Fixed RSI (Smart StopLoss)](https://c.mql5.com/2/44/price_action.png)[Fix PriceAction Stoploss or Fixed RSI (Smart StopLoss)](https://www.mql5.com/en/articles/9827)

Stop-loss is a major tool when it comes to money management in trading. Effective use of stop-loss, take profit and lot size can make a trader more consistent in trading and overall more profitable. Although stop-loss is a great tool, there are challenges that are encountered when being used. The major one being stop-loss hunt. This article looks on how to reduce stop-loss hunt in trade and compare with the classical stop-loss usage to determine its profitability.

![An attempt at developing an EA constructor](https://c.mql5.com/2/43/carpenter-3572804_640.png)[An attempt at developing an EA constructor](https://www.mql5.com/en/articles/9717)

In this article, I offer my set of trading functions in the form of a ready-made EA. This method allows getting multiple trading strategies by simply adding indicators and changing inputs.

![Use MQL5.community channels and group chats](https://c.mql5.com/2/43/chats.png)[Use MQL5.community channels and group chats](https://www.mql5.com/en/articles/8586)

The MQL5.com website brings together traders from all over the world. Users publish articles, share free codes, sell products in the Market, perform Freelance orders and copy trading signals. You can communicate with them on the Forum, in trader chats and in MetaTrader channels.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/10130&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071661824878783526)

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