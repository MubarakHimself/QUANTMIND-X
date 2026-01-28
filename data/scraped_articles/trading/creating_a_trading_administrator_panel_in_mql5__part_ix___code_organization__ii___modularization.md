---
title: Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (II): Modularization
url: https://www.mql5.com/en/articles/16562
categories: Trading
relevance_score: 6
scraped_at: 2026-01-22T17:58:37.078767
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/16562&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049529115373055259)

MetaTrader 5 / Examples


### Contents:

- [Introduction](https://www.mql5.com/en/articles/16562#para1)
- [Discussion Overview](https://www.mql5.com/en/articles/16562#para2)
- [Modularization](https://www.mql5.com/en/articles/16562#para3)
- [Code Implementation](https://www.mql5.com/en/articles/16562#para4)
- [The New Trading Admin Panel](https://www.mql5.com/en/articles/16562#5)
- [Testing](https://www.mql5.com/en/articles/16562#para6)
- [Conclusion](https://www.mql5.com/en/articles/16562#para7)

### Introduction

This discussion marks a breakthrough in creating a maintainable Admin Panel Expert Advisor. The code organization introduced in the previous article significantly enhanced our main code, and today, we take it a step further by modularizing key components into external files. This approach ensures that future updates focus on improving individual components without disrupting other parts of the code.

A clear example of why this is beneficial occurred when I needed to refine the Communication Panel. Scrolling through a large, monolithic codebase to locate the relevant section was overwhelming. By breaking the code into structured modules, we simplify navigation, making development and maintenance much more efficient.

Our inspiration comes from well-structured projects that exemplify best practices in code organization. Today, we will implement modularization by introducing custom classes for essential functionalities that define our program. Below, we have tabulated the complete list of potential modules we plan to develop.

| Module File | Description |
| --- | --- |
| AdminHomeDialog.mqh | This declares the central section of the Trading Administrator Panel, providing access to other utilities within the program. |
| Aunthentication.mqh | This module manages user authentication, including password verification and two-factor authentication. |
| ThemeManager.mqh | Responsible for managing the appearance and styling of your admin panel. |
| Telegram.mqh | Contains functions and classes for interacting with Telegram, usually for sending messages, notifications. |
| CommunicationsDialog.mqh | This will be responsible for handling the user interface (UI) and interactions related to communication features within your Admin Panel |
| AnalyticsDialog.mqh | For displaying and managing analytical data, such as trade statistics, performance metrics, or visual charts within a dialog panel. |
| TradeManagementDialog.mqh | This will handle the UI Creation of trade related tasks, where users can execute and manage trades efficiently. |

After a successful creation of these files, they can be included in the main code

```
#include <Telegram.mqh>
#include <Authentication.mqh>
#include <AdminHomeDialog.mqh>
#include  <AnalyticsDialog.mqh>
#include <TradeManagementDialog.mqh>
#include <CommunicationDialog.mqh>
```

All declarations for the panel components will be placed in the include files, while the main code will primarily contain definitions. Since definitions are generally smaller than declarations, this approach keeps the main program clean and uncluttered, improving readability and maintainability.

I'm sure you can already envision how our project is evolving with these innovations. In the next section, we will provide a detailed explanation of modularization, followed by its implementation in this project.

![Declarations and Implementation](https://c.mql5.com/2/118/CODE_STRUCTURE_i1g.png)

Relationship between main code and header file

### Discussion Overview

With a brief overview from the introduction above, we will now explore modularization in more detail before diving into the development and implementation of our code components. Each module will be explained thoroughly, with a breakdown of the functionality of each line of code.

Finally, we will integrate all the modules into a new main codebase for the Trading Administrator Panel, essentially rebuilding it from the ground up with enhanced structure and efficiency.

At the end of this discussion, we will have developed, integrated, and tested the following files:

- AdminHomeDialog.mqh
- Authentication.mqh
- Telegram.mqh

### Modularization

In MQL5 programming, modularization refers to the practice of breaking down a program into smaller, independent, and reusable pieces, mainly through the use of classes, functions, and include files. This approach allows developers to encapsulate specific functionalities into modules or classes, like creating UI components or trading logic, which can be included or instantiated as needed across different parts of an application or even in multiple applications. By doing so, code becomes more manageable, easier to maintain, and less error-prone since changes to one module don't necessarily impact others, promoting code reuse, improving readability, and facilitating collaborative development in the MetaTrader 5 environment.

In this context, we have already outlined the subcomponents of our new program in the introduction above. Additionally, there are other resources available for further reading on this topic, and I have come across various approaches to applying modularization in different articles.

In the next steps, I will guide you through the development of each module in detail, ensuring a clear understanding of their implementation and integration.

### Code Implementation

Now is the time to apply our MQL5 knowledge to develop the key components of our Trading Administrator Panel EA. The great news is that these files are designed to be easily adapted and integrated into your own projects.

Main Structure of a Header File in MQL5:

A header file in MQL5, typically with the _**.mqh**_ extension, serves as a place to define classes, constants, enumerations, and function prototypes that can be included in other MQL5 scripts or expert advisors. Here's the typical structure based on other inbuilt header file code:

1. File Metadata:  Includes copyright information, links, and version control.
2. Include Statements: Lists other header files or libraries that the current header depends on.
3. Defines/Constants: Defines macros or constants used within the class or by other parts of the code for consistent values.
4. Class Declaration: Declares the class with its inheritance (if any), private members, public methods, and protected methods.
5. Event Mapping: Uses macros to define how events are mapped to member functions for event-driven programming.
6. Method Implementations: While not strictly necessary for all header files, in this case, method implementations are included, which is common for smaller classes where encapsulation isn't critical. For performance reasons in MQL5 including the implementation can reduce function call overhead.
7. Constructors and Destructors: These are part of the class definition, specifying how objects of the class are created and destroyed.

Based on the outline above, here is a sample code template:

```
// Meta Data here on top.
#include   // Include other necessary libraries or headers
#include

//+------------------------------------------------------------------+
//| Defines                                                          |
//+------------------------------------------------------------------+
#define CONSTANT_NAME1    (value1)  // Constants or macro definitions
#define CONSTANT_NAME2    (value2)

//+------------------------------------------------------------------+
//| Class CClass                                                     |
//| Usage: Description of class purpose                              |
//+------------------------------------------------------------------+
class CClass : public CParentClass  // Inherits from another class if needed
  {
private:
   // Private member variables
   CSomeControl   m_control;  // Example control member

public:
   CClass(void);              // Constructor
   ~CClass(void);             // Destructor
   virtual bool   Create(/* parameters */);  // Virtual method for polymorphism
   virtual bool   OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam);

protected:
   // Protected methods or members
   bool           CreateSomeControl(void);
   void           SomeEventHandler(void);

  };
//+------------------------------------------------------------------+
//| Event Handling                                                   |
//+------------------------------------------------------------------+
EVENT_MAP_BEGIN(CClass)
   ON_EVENT(SOME_EVENT,m_control,SomeEventHandler)
EVENT_MAP_END(CParentClass)

// Constructor implementation
CClass::CClass(void)
  {
  }

// Destructor implementation
CClass::~CClass(void)
  {
  }

// Method implementations if included in the header
bool CClass::Create(/* parameters */)
  {
   // Implementation of create method
  }

// Event handler examples
void CClass::SomeEventHandler(void)
  {
   // Handle the event
  }

//+------------------------------------------------------------------+
```

To create header files in MetaEditor, open a new file by pressing **Ctrl + N** or navigating manually through the menu. In the pop-up window, select Include (\*.mqh) and proceed to start editing. By default, the generated template includes some comments as notes to guide you.

Refer to the image below:

![create an include file](https://c.mql5.com/2/117/Creating_a_new_header_file_in_MQL5.PNG)

Create a new header file in MetaEditor

Here is the default header file template, which includes some commented notes for guidance.

```
//+------------------------------------------------------------------+
//|                                                     Telegram.mqh |
//|                                Copyright 2024, Clemence Benjamin |
//|             https://www.mql5.com/en/users/billionaire2024/seller |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
// #define MacrosHello   "Hello, world!"
// #define MacrosYear    2010
//+------------------------------------------------------------------+
//| DLL imports                                                      |
//+------------------------------------------------------------------+
// #import "user32.dll"
//   int      SendMessageA(int hWnd,int Msg,int wParam,int lParam);
// #import "my_expert.dll"
//   int      ExpertRecalculate(int wParam,int lParam);
// #import
//+------------------------------------------------------------------+
//| EX5 imports                                                      |
//+------------------------------------------------------------------+
// #import "stdlib.ex5"
//   string ErrorDescription(int error_code);
// #import
//+------------------------------------------------------------------+
```

**Admin Home Header File**

In this section, we develop the _CAdminHomeDialog_ class, which serves as the main interface for the admin panel in our MQL5 program. It integrates essential header files for dialog and button controls while utilizing predefined constants to maintain consistent panel dimensions and spacing.

```
//+------------------------------------------------------------------+
//| AdminHomeDialog.mqh                                              |
//+------------------------------------------------------------------+
#include <Controls\Dialog.mqh>
#include <Controls\Button.mqh>

//+------------------------------------------------------------------+
//| Defines                                                          |
//+------------------------------------------------------------------+
#define ADMIN_PANEL_WIDTH   (335)
#define ADMIN_PANEL_HEIGHT  (350)
#define INDENT_LEFT         (11)
#define INDENT_TOP          (11)
#define INDENT_RIGHT        (11)
#define INDENT_BOTTOM       (11)
#define CONTROLS_GAP_X      (5)
#define CONTROLS_GAP_Y      (5)
#define BUTTON_WIDTH        (250)
#define BUTTON_HEIGHT       (40)
```

The _CAdminHomeDialog_ class extends _CAppDialog_ and includes four key buttons, _m\_tradeMgmtButton_, _m\_commButton_, _m\_analyticsButton_, and _m\_showAllButton_—that provide seamless navigation to different sections of the admin panel. The class structure remains streamlined, with a minimal constructor and destructor, while the Create method ensures all buttons are properly initialized for a smooth user experience.

```
//+------------------------------------------------------------------+
//| CAdminHomeDialog class                                           |
//+------------------------------------------------------------------+
class CAdminHomeDialog : public CAppDialog
{
private:
    CButton m_tradeMgmtButton;
    CButton m_commButton;
    CButton m_analyticsButton;
    CButton m_showAllButton;

public:
    CAdminHomeDialog(void) {}
    ~CAdminHomeDialog(void) {}

    virtual bool Create(const long chart, const string name, const int subwin,
                       const int x1, const int y1, const int x2, const int y2);
    virtual bool OnEvent(const int id, const long &lparam, const double &dparam, const string &sparam);
};

//+------------------------------------------------------------------+
//| Create                                                           |
//+------------------------------------------------------------------+
bool CAdminHomeDialog::Create(const long chart, const string name, const int subwin,
                             const int x1, const int y1, const int x2, const int y2)
{
    if(!CAppDialog::Create(chart, name, subwin, x1, y1, x2, y2))
        return false;

    if(!CreateTradeMgmtButton()) return false;
    if(!CreateCommButton()) return false;
    if(!CreateAnalyticsButton()) return false;
    if(!CreateShowAllButton()) return false;

    return true;
}
```

User interactions are handled within the _OnEvent_ method, where button clicks trigger debug messages and call their respective event handlers: _OnClickTradeManagement_, _OnClickCommunications_, _OnClickAnalytics_, and _OnClickShowAll_. These handlers currently log interactions but will be expanded as we enhance functionality.

```
//+------------------------------------------------------------------+
//| Event Handling (Enhanced Debugging)                              |
//+------------------------------------------------------------------+
bool CAdminHomeDialog::OnEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
    if(id == CHARTEVENT_OBJECT_CLICK)
    {
        Print("Clicked object: ", sparam); // Debug which object was clicked

        if(sparam == m_tradeMgmtButton.Name())
        {
            Print("Trade Management button detected");
            OnClickTradeManagement();
            return true;
        }
        else if(sparam == m_commButton.Name())
        {
            Print("Communications button detected");
            OnClickCommunications();
            return true;
        }
        else if(sparam == m_analyticsButton.Name())
        {
            Print("Analytics button detected");
            OnClickAnalytics();
            return true;
        }
        else if(sparam == m_showAllButton.Name())
        {
            Print("Show All button detected");
            OnClickShowAll();
            return true;
        }
    }
    return CAppDialog::OnEvent(id, lparam, dparam, sparam);
}

//+------------------------------------------------------------------+
//| Button Click Handlers                                            |
//+------------------------------------------------------------------+
void CAdminHomeDialog::OnClickTradeManagement() { Print("Trade Management Panel clicked"); }
void CAdminHomeDialog::OnClickCommunications()  { Print("Communications Panel clicked"); }
void CAdminHomeDialog::OnClickAnalytics()       { Print("Analytics Panel clicked"); }
void CAdminHomeDialog::OnClickShowAll()         { Print("Show All clicked"); }
```

The button creation methods— _CreateTradeMgmtButton, CreateCommButton, CreateAnalyticsButton,_ and _CreateShowAllButton_—dynamically generate buttons with unique identifiers, precise positioning, and well-defined labels. The "Show All" button even incorporates an emoji to enhance the UI. As we continue development, additional improvements and refinements will be introduced to optimize performance and usability.

```
//+------------------------------------------------------------------+
//| Control Creation Methods                                         |
//+------------------------------------------------------------------+
bool CAdminHomeDialog::CreateTradeMgmtButton()
{
    int x = INDENT_LEFT;
    int y = INDENT_TOP;
    return m_tradeMgmtButton.Create(m_chart_id, m_name+"_TradeBtn", m_subwin,
                                  x, y, x+BUTTON_WIDTH, y+BUTTON_HEIGHT)
        && m_tradeMgmtButton.Text("Trade Management Panel")
        && Add(m_tradeMgmtButton);
}

bool CAdminHomeDialog::CreateCommButton()
{
    int x = INDENT_LEFT;
    int y = INDENT_TOP + BUTTON_HEIGHT + CONTROLS_GAP_Y;
    return m_commButton.Create(m_chart_id, m_name+"_CommBtn", m_subwin,
                             x, y, x+BUTTON_WIDTH, y+BUTTON_HEIGHT)
        && m_commButton.Text("Communications Panel")
        && Add(m_commButton);
}

bool CAdminHomeDialog::CreateAnalyticsButton()
{
    int x = INDENT_LEFT;
    int y = INDENT_TOP + (BUTTON_HEIGHT + CONTROLS_GAP_Y) * 2;
    return m_analyticsButton.Create(m_chart_id, m_name+"_AnalyticsBtn", m_subwin,
                                  x, y, x+BUTTON_WIDTH, y+BUTTON_HEIGHT)
        && m_analyticsButton.Text("Analytics Panel")
        && Add(m_analyticsButton);
}

bool CAdminHomeDialog::CreateShowAllButton()
{
    int x = INDENT_LEFT;
    int y = INDENT_TOP + (BUTTON_HEIGHT + CONTROLS_GAP_Y) * 3;
    return m_showAllButton.Create(m_chart_id, m_name+"_ShowAllBtn", m_subwin,
                                x, y, x+BUTTON_WIDTH, y+BUTTON_HEIGHT)
        && m_showAllButton.Text("Show All 💥")
        && Add(m_showAllButton);
}
```

**Implementation of the AdminHomeDialog.mqh in main program:**

1\. Include via **#include** _"AdminHomeDialog.mqh"_

```
#include "AdminHomeDialog.mqh"
```

Including AdminHomeDialog.mqh makes the _CAdminHomeDialog_ class available in the main script. Without this inclusion, the compiler wouldn't recognize _CAdminHomeDialog_, leading to errors. This modular approach allows the main script to remain clean while keeping the dialog’s implementation in a separate file for better organization and maintainability.

2\. Declare as _CAdminHomeDialog ExtDialog_;

```
CAdminHomeDialog ExtDialog;
```

Declaring _ExtDialog_ as an instance of _CAdminHomeDialog_ allows the script to reference and control the Admin Home Panel throughout the program. This object handles the creation, visibility, and event management of the panel, making it accessible across different functions.

3\. Create using _ExtDialog_ inside _CreateHiddenPanels()_

```
bool CreateHiddenPanels()
{
    bool success = ExtDialog.Create(0, "Admin Home", 0,
                    MAIN_DIALOG_X, MAIN_DIALOG_Y,
                    MAIN_DIALOG_X + MAIN_DIALOG_WIDTH,
                    MAIN_DIALOG_Y + MAIN_DIALOG_HEIGHT);
    if(success)
    {
        ExtDialog.Hide();
        ChartRedraw();
    }
    return success;
}
```

The _Create()_ method initializes the panel with specific dimensions and positions it correctly on the chart. Placing this inside _CreateHiddenPanels()_ ensures that the panel is only created once during initialization, keeping the setup process organized and preventing unnecessary reinitialization.

4\. Shown or hidden based on authentication status in _OnChartEvent()_

```
if(authManager.IsAuthenticated())
{
    if(!ExtDialog.IsVisible())
    {
        ExtDialog.Show();
        ChartRedraw();
    }
    ExtDialog.ChartEvent(id, lparam, dparam, sparam);
}
else
{
    if(ExtDialog.IsVisible())
    {
        ExtDialog.Hide();
    }
}
```

The Admin Home Panel should only be accessible after successful authentication. Checking _authManager. IsAuthenticated()_ ensures unauthorized users cannot interact with the panel. If authentication is valid, the panel is shown; otherwise, it remains hidden, enhancing security and access control.

5\. Destroyed in _OnDeinit()_ when the script is removed

```
void OnDeinit(const int reason)
{
    ExtDialog.Destroy(reason);
}
```

When the expert is removed from the chart, calling _ExtDialog. Destroy()_ ensures that resources allocated for the panel are thoroughly cleaned up. This prevents potential memory leaks or orphaned graphical objects that could interfere with future instances of the script.

**Telegram Header File**

To create a Telegram header file, the Telegram function is copied directly into the header source due to its simplicity and straightforward operation. However, this approach may differ for other files that require a more structured setup, involving classes, methods, constructors, and destructors, as mentioned earlier. Therefore, this is the simplest header file in our list of creations. By modularizing the function, we reduce the length of the main code, and the function can be easily reused in other projects where it's needed.

```
//+------------------------------------------------------------------+
//|                                                     Telegram.mqh |
//|                                Copyright 2024, Clemence Benjamin |
//|             https://www.mql5.com/en/users/billionaire2024/seller |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
//+------------------------------------------------------------------+
//| Telegram.mqh - Telegram Communication Include File               |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Send the message to Telegram                                     |
//+------------------------------------------------------------------+
bool SendMessageToTelegram(string message, string chatId, string botToken)
  {
   string url = "https://api.telegram.org/bot" + botToken + "/sendMessage";
   string jsonMessage = "{\"chat_id\":\"" + chatId + "\", \"text\":\"" + message + "\"}";

   char postData[];
   ArrayResize(postData, StringToCharArray(jsonMessage, postData) - 1);

   int timeout = 5000;
   char result[];
   string responseHeaders;
   int responseCode = WebRequest("POST", url, "Content-Type: application/json\r\n", timeout, postData, result, responseHeaders);

   if (responseCode == 200)
     {
      Print("Message sent successfully: ", message);
      return true;
     }
   else
     {
      Print("Failed to send message. HTTP code: ", responseCode, " Error code: ", GetLastError());
      Print("Response: ", CharArrayToString(result));
      return false;
     }
  }
//+------------------------------------------------------------------+
```

**Implementation of Telegram header in the main code**

This is simply done in 2 steps:

1\. Including the file in the main code as follows.

```
#include<Telegram.mqh>
```

The above works only if it is stored in the MQL5/Include directory, if otherwise the sub-folder name must be stated as follows

```
#include <FolderName\Telegram.mqh> // Replace FolderName with actual location name
```

2\. Finally, we need to call the function in your main code when it is required, as follows:

```
SendMessageToTelegram("Your verification code: " + ActiveTwoFactorAuthCode,
                         TwoFactorAuthChatId, TwoFactorAuthBotToken)
```

**Developing an Authentication Header File**

From previous developments, you have seen the evolution of security prompt logic, which has been consistently used in every version of the Admin Panel. This logic can also be adapted for other panel-related projects, especially when developing a classified module for its functionality. At this stage, we are developing _Authentication.mqh_, which consolidates all the security logic used in the former Admin Panel. I will share the code below and then provide an explanation of how it works.

```
//+------------------------------------------------------------------+
//|                                          authentication.mqh      |
//|                           Copyright 2024, Clemence Benjamin      |
//|        https://www.mql5.com/en/users/billionaire2024/seller      |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
#property version   "1.0"
#property strict

// Authentication Dialog Coordinates
#define AUTH_DIALOG_X         100
#define AUTH_DIALOG_Y         100
#define AUTH_DIALOG_WIDTH     300
#define AUTH_DIALOG_HEIGHT    200

#define PASS_INPUT_X          20
#define PASS_INPUT_Y          50
#define PASS_INPUT_WIDTH      260  // Wider input field
#define PASS_INPUT_HEIGHT     30

#define PASS_LABEL_X          20
#define PASS_LABEL_Y          20
#define PASS_LABEL_WIDTH      200
#define PASS_LABEL_HEIGHT     20

#define FEEDBACK_LABEL_X      20
#define FEEDBACK_LABEL_Y      100
#define FEEDBACK_LABEL_WIDTH  260
#define FEEDBACK_LABEL_HEIGHT 40

// Button spacing adjustments
#define LOGIN_BTN_X           20
#define LOGIN_BTN_Y           130
#define LOGIN_BTN_WIDTH       120
#define LOGIN_BTN_HEIGHT      30

#define CANCEL_BTN_X          160  // Added 20px spacing from login button
#define CANCEL_BTN_Y          130
#define CANCEL_BTN_WIDTH      120
#define CANCEL_BTN_HEIGHT     30

// Two-Factor Authentication Dialog Coordinates
#define TWOFA_DIALOG_X        100
#define TWOFA_DIALOG_Y        100
#define TWOFA_DIALOG_WIDTH    300
#define TWOFA_DIALOG_HEIGHT   200

#define TWOFA_INPUT_X         20
#define TWOFA_INPUT_Y         50
#define TWOFA_INPUT_WIDTH     180
#define TWOFA_INPUT_HEIGHT    30

#define TWOFA_LABEL_X         20
#define TWOFA_LABEL_Y         20
#define TWOFA_LABEL_WIDTH     260
#define TWOFA_LABEL_HEIGHT    20

#define TWOFA_FEEDBACK_X      20
#define TWOFA_FEEDBACK_Y      100
#define TWOFA_FEEDBACK_WIDTH  260
#define TWOFA_FEEDBACK_HEIGHT 40

#define TWOFA_VERIFY_BTN_X    60
#define TWOFA_VERIFY_BTN_Y    130
#define TWOFA_VERIFY_WIDTH    120
#define TWOFA_VERIFY_HEIGHT   30

#define TWOFA_CANCEL_BTN_X    140
#define TWOFA_CANCEL_BTN_Y    130
#define TWOFA_CANCEL_WIDTH    60
#define TWOFA_CANCEL_HEIGHT   30

#include <Controls\Dialog.mqh>
#include <Controls\Button.mqh>
#include <Controls\Edit.mqh>
#include <Controls\Label.mqh>
#include <Telegram.mqh>

class CAuthenticationManager {
private:
    CDialog m_authDialog;
    CDialog m_2faDialog;
    CEdit m_passwordInput;
    CEdit m_2faCodeInput;
    CLabel m_passwordLabel;
    CLabel m_feedbackLabel;
    CLabel m_2faLabel;
    CLabel m_2faFeedback;
    CButton m_loginButton;
    CButton m_closeAuthButton;
    CButton m_2faLoginButton;
    CButton m_close2faButton;

    string m_password;
    string m_2faChatId;
    string m_2faBotToken;
    int m_failedAttempts;
    bool m_isAuthenticated;
    string m_active2faCode;

public:
    CAuthenticationManager(string password, string twoFactorChatId, string twoFactorBotToken) :
        m_password(password),
        m_2faChatId(twoFactorChatId),
        m_2faBotToken(twoFactorBotToken),
        m_failedAttempts(0),
        m_isAuthenticated(false),
        m_active2faCode("")
    {
    }

    ~CAuthenticationManager()
    {
        m_authDialog.Destroy();
        m_2faDialog.Destroy();
    }

    bool Initialize() {
        if(!CreateAuthDialog() || !Create2FADialog()) {
            Print("Authentication initialization failed");
            return false;
        }
        m_2faDialog.Hide();  // Ensure 2FA dialog starts hidden
        return true;
    }

    bool IsAuthenticated() const { return m_isAuthenticated; }

    void HandleEvent(const int id, const long &lparam, const double &dparam, const string &sparam) {
        if(id == CHARTEVENT_OBJECT_CLICK) {
            if(sparam == "LoginButton") HandleLoginAttempt();
            else if(sparam == "2FALoginButton") Handle2FAAttempt();
            else if(sparam == "CloseAuthButton") m_authDialog.Hide();
            else if(sparam == "Close2FAButton") m_2faDialog.Hide();
        }
    }

private:
    bool CreateAuthDialog() {
    if(!m_authDialog.Create(0, "Authentication", 0,
       AUTH_DIALOG_X, AUTH_DIALOG_Y,
       AUTH_DIALOG_X + AUTH_DIALOG_WIDTH,
       AUTH_DIALOG_Y + AUTH_DIALOG_HEIGHT))
       return false;

    if(!m_passwordInput.Create(0, "PasswordInput", 0,
       PASS_INPUT_X, PASS_INPUT_Y,
       PASS_INPUT_X + PASS_INPUT_WIDTH,
       PASS_INPUT_Y + PASS_INPUT_HEIGHT) ||
       !m_passwordLabel.Create(0, "PasswordLabel", 0,
       PASS_LABEL_X, PASS_LABEL_Y,
       PASS_LABEL_X + PASS_LABEL_WIDTH,
       PASS_LABEL_Y + PASS_LABEL_HEIGHT) ||
       !m_feedbackLabel.Create(0, "AuthFeedback", 0,
       FEEDBACK_LABEL_X, FEEDBACK_LABEL_Y,
       FEEDBACK_LABEL_X + FEEDBACK_LABEL_WIDTH,
       FEEDBACK_LABEL_Y + FEEDBACK_LABEL_HEIGHT) ||
       !m_loginButton.Create(0, "LoginButton", 0,
       LOGIN_BTN_X, LOGIN_BTN_Y,
       LOGIN_BTN_X + LOGIN_BTN_WIDTH,
       LOGIN_BTN_Y + LOGIN_BTN_HEIGHT) ||
       !m_closeAuthButton.Create(0, "CloseAuthButton", 0,
       CANCEL_BTN_X, CANCEL_BTN_Y,
       CANCEL_BTN_X + CANCEL_BTN_WIDTH,
       CANCEL_BTN_Y + CANCEL_BTN_HEIGHT))
       return false;

        m_passwordLabel.Text("Enter Password:");
        m_feedbackLabel.Text("");
        m_feedbackLabel.Color(clrRed);
        m_loginButton.Text("Login");
        m_closeAuthButton.Text("Cancel");

        m_authDialog.Add(m_passwordInput);
        m_authDialog.Add(m_passwordLabel);
        m_authDialog.Add(m_feedbackLabel);
        m_authDialog.Add(m_loginButton);
        m_authDialog.Add(m_closeAuthButton);

        m_authDialog.Show();
        return true;
    }

    bool Create2FADialog() {
        if(!m_2faDialog.Create(0, "2FA Verification", 0,
           TWOFA_DIALOG_X, TWOFA_DIALOG_Y,
           TWOFA_DIALOG_X + TWOFA_DIALOG_WIDTH,
           TWOFA_DIALOG_Y + TWOFA_DIALOG_HEIGHT))
            return false;

        if(!m_2faCodeInput.Create(0, "2FAInput", 0,
           TWOFA_INPUT_X, TWOFA_INPUT_Y,
           TWOFA_INPUT_X + TWOFA_INPUT_WIDTH,
           TWOFA_INPUT_Y + TWOFA_INPUT_HEIGHT) ||
           !m_2faLabel.Create(0, "2FALabel", 0,
           TWOFA_LABEL_X, TWOFA_LABEL_Y,
           TWOFA_LABEL_X + TWOFA_LABEL_WIDTH,
           TWOFA_LABEL_Y + TWOFA_LABEL_HEIGHT) ||
           !m_2faFeedback.Create(0, "2FAFeedback", 0,
           TWOFA_FEEDBACK_X, TWOFA_FEEDBACK_Y,
           TWOFA_FEEDBACK_X + TWOFA_FEEDBACK_WIDTH,
           TWOFA_FEEDBACK_Y + TWOFA_FEEDBACK_HEIGHT) ||
           !m_2faLoginButton.Create(0, "2FALoginButton", 0,
           TWOFA_VERIFY_BTN_X, TWOFA_VERIFY_BTN_Y,
           TWOFA_VERIFY_BTN_X + TWOFA_VERIFY_WIDTH,
           TWOFA_VERIFY_BTN_Y + TWOFA_VERIFY_HEIGHT) ||
           !m_close2faButton.Create(0, "Close2FAButton", 0,
           TWOFA_CANCEL_BTN_X, TWOFA_CANCEL_BTN_Y,
           TWOFA_CANCEL_BTN_X + TWOFA_CANCEL_WIDTH,
           TWOFA_CANCEL_BTN_Y + TWOFA_CANCEL_HEIGHT))
            return false;

        m_2faLabel.Text("Enter verification code:");
        m_2faFeedback.Text("");
        m_2faFeedback.Color(clrRed);
        m_2faLoginButton.Text("Verify");
        m_close2faButton.Text("Cancel");

        m_2faDialog.Add(m_2faCodeInput);
        m_2faDialog.Add(m_2faLabel);
        m_2faDialog.Add(m_2faFeedback);
        m_2faDialog.Add(m_2faLoginButton);
        m_2faDialog.Add(m_close2faButton);

        return true;
    }

    void HandleLoginAttempt() {
        if(m_passwordInput.Text() == m_password) {
            m_isAuthenticated = true;
            m_authDialog.Hide();
            m_2faDialog.Hide();  // Ensure both dialogs are hidden
        } else {
            if(++m_failedAttempts >= 3) {
                Generate2FACode();
                m_authDialog.Hide();
                m_2faDialog.Show();
            } else {
                m_feedbackLabel.Text(StringFormat("Invalid password (%d attempts left)",
                                                 3 - m_failedAttempts));
            }
        }
    }

    void Handle2FAAttempt() {
        if(m_2faCodeInput.Text() == m_active2faCode) {
            m_isAuthenticated = true;
            m_2faDialog.Hide();
            m_authDialog.Hide();  // Hide both dialogs on success
        } else {
            m_2faFeedback.Text("Invalid code - please try again");
            m_2faCodeInput.Text("");
        }
    }

    void Generate2FACode() {
        m_active2faCode = StringFormat("%06d", MathRand() % 1000000);
        SendMessageToTelegram("Your verification code: " + m_active2faCode,
                             m_2faChatId, m_2faBotToken);
    }
};
//+------------------------------------------------------------------+
```

The _CAuthenticationManager_ class in the provided module handles a multistep user authentication process for an MQL5-based application. It manages both password authentication and two-factor authentication (2FA) through a dialog-based interface. The authentication process starts with the user inputting a password into a dialog box. If the password is correct, access is granted immediately. However, if the password fails, the system tracks failed attempts, and after three incorrect entries, it triggers a second dialog for 2FA. The 2FA process involves generating a six-digit verification code, which is sent via Telegram to the user, who must enter the correct code to proceed.

The module uses predefined coordinates for dialog elements to ensure a consistent layout, and it includes feedback mechanisms to inform users of errors or successful authentication. The class also integrates the _Telegram.mqh_ library to handle the sending of 2FA codes. It is designed with extensibility in mind, allowing easy modification of the password and 2FA settings, and ensuring that both dialogs are hidden when authentication is successful. This design streamlines the user experience while maintaining robust security features.

### The New Trading Admin Panel

At this stage, we are integrating all the previously developed modules to build a more structured and efficient Admin Panel. This improved version enhances organization and modularity, allowing its components to be easily shared and reused across other applications within the terminal.

The New Admin Panel includes _AdminHomeDialog.mq_ h for the graphical interface and _Authentication.mqh_ for authentication management. The EA defines input parameters for a Telegram chat ID, and a bot token for 2FA verification. During initialization _(OnInit)_, it attempts to initialize authentication and create a hidden panel ( _CreateHiddenPanels_), failing if either process is unsuccessful.

The _OnChartEvent_ function processes chart events, handling authentication before showing or hiding the _AdminHomeDialog_ panel based on authentication status. If authenticated, it ensures the panel is visible and forwards events to the panel. Otherwise, it hides the panel. The deinitialization function ( _OnDeinit_) ensures the dialog is destroyed properly. This design ensures secure access to the Admin Panel Home, requiring authentication before granting control over its features.

Here is the complete code for the new program:

```
//+------------------------------------------------------------------+
//|                                              New Admin Panel.mq5 |
//|                                Copyright 2024, Clemence Benjamin |
//|             https://www.mql5.com/en/users/billionaire2024/seller |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
#property version   "1.00"

// Panel coordinate defines
#define MAIN_DIALOG_X        30
#define MAIN_DIALOG_Y        80
#define MAIN_DIALOG_WIDTH    335
#define MAIN_DIALOG_HEIGHT   350

#include "AdminHomeDialog.mqh"
#include <Authentication.mqh>

// Input parameters for authentication
input string TwoFactorChatID = "YOUR_CHAT_ID";
input string TwoFactorBotToken = "YOUR_BOT_TOKEN";
string AuthPassword = "2024";

CAdminHomeDialog ExtDialog;
CAuthenticationManager authManager(AuthPassword, TwoFactorChatID, TwoFactorBotToken);

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    if(!authManager.Initialize() || !CreateHiddenPanels())
    {
        Print("Initialization failed");
        return INIT_FAILED;
    }
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    ExtDialog.Destroy(reason);
}

//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam)
{
    authManager.HandleEvent(id, lparam, dparam, sparam);

    if(authManager.IsAuthenticated())
    {
        if(!ExtDialog.IsVisible())
        {
            ExtDialog.Show();
            ChartRedraw();
        }
        // Handle dialog events only when authenticated
        ExtDialog.ChartEvent(id, lparam, dparam, sparam);
    }
    else
    {
        if(ExtDialog.IsVisible())
        {
            ExtDialog.Hide();

        }
    }
}

//+------------------------------------------------------------------+
//| Create hidden panels                                             |
//+------------------------------------------------------------------+
bool CreateHiddenPanels()
{
    bool success = ExtDialog.Create(0, "Admin Home", 0,
                    MAIN_DIALOG_X, MAIN_DIALOG_Y,
                    MAIN_DIALOG_X + MAIN_DIALOG_WIDTH,
                    MAIN_DIALOG_Y + MAIN_DIALOG_HEIGHT);
    if(success)
    {
        ExtDialog.Hide();
        ChartRedraw();
    }
    return success;
}
```

### Testing

Here, we present the testing results after developing individual components and integrating them logically into the new Admin Panel. We successfully compiled and ran the program on the chart, as shown below.

![Testing the New Admin Panel](https://c.mql5.com/2/118/terminal64_EQ2zcJS5zA.gif)

Testing the New Admin Panel on EURUSD chart

### Conclusion

I'm sure you can appreciate how we've streamlined the organization of our Admin Panel. By inheriting from CAppDialog, we've significantly improved the application's responsiveness, allowing us to drag the panel freely across the chart for the first time. Additionally, the minimize button, inherited from the base class, enables us to minimize the application, providing an unobstructed chart view while keeping it running in the background. The panel can be maximized at any time when needed, ensuring seamless usability.

Our focus on readability, scalability, and modularity has driven us to develop each component separately. This approach enhances code reusability, allowing us to integrate these modules into other programs by simply including the relevant files and calling essential methods. Moving forward, we plan to complete the remaining components that will make the new Admin Panel even more powerful. These include CommunicationsDialog.mqh, TradeManagementDialog.mqh, and AnalyticsDialog.mqh, each designed to be reusable across different applications.

As a starting point, you can try implementing Telegram.mqh in your own program to see how easily it integrates. I've attached all the necessary files below—enjoy testing and further developing them!

| File | Description |
| --- | --- |
| New Admin Panel .mqh | The latest Admin Panel incorporates the concept of modularity. |
| Telegram.mqh | For transmitting messages and notifications via Telegram. |
| Authentication.mqh | This file contains all security declarations and the complete logic. |
| AdminHomeDialog.mqh | For admin home dialog creation. It contains all the coordinates declarations. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16562.zip "Download all attachments in the single ZIP archive")

[New\_Admin\_Panel.mq5](https://www.mql5.com/en/articles/download/16562/new_admin_panel.mq5 "Download New_Admin_Panel.mq5")(3.05 KB)

[Telegram.mqh](https://www.mql5.com/en/articles/download/16562/telegram.mqh "Download Telegram.mqh")(1.62 KB)

[Authentication.mqh](https://www.mql5.com/en/articles/download/16562/authentication.mqh "Download Authentication.mqh")(8.92 KB)

[AdminHomeDialog.mqh](https://www.mql5.com/en/articles/download/16562/adminhomedialog.mqh "Download AdminHomeDialog.mqh")(12.02 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [From Novice to Expert: Higher Probability Signals](https://www.mql5.com/en/articles/20658)
- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)
- [From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)

**[Go to discussion](https://www.mql5.com/en/forum/481439)**

![Price Action Analysis Toolkit Development (Part 13): RSI Sentinel Tool](https://c.mql5.com/2/119/Price_Action_Analysis_Toolkit_Development_Part_13___LOGO.png)[Price Action Analysis Toolkit Development (Part 13): RSI Sentinel Tool](https://www.mql5.com/en/articles/17198)

Price action can be effectively analyzed by identifying divergences, with technical indicators such as the RSI providing crucial confirmation signals. In the article below, we explain how automated RSI divergence analysis can identify trend continuations and reversals, thereby offering valuable insights into market sentiment.

![Developing a Replay System (Part 59): A New Future](https://c.mql5.com/2/87/Desenvolvendo_um_sistema_de_Replay_Parte_59__LOGO__3.png)[Developing a Replay System (Part 59): A New Future](https://www.mql5.com/en/articles/12075)

Having a proper understanding of different ideas allows us to do more with less effort. In this article, we'll look at why it's necessary to configure a template before the service can interact with the chart. Also, what if we improve the mouse pointer so we can do more things with it?

![Automating Trading Strategies in MQL5 (Part 7): Building a Grid Trading EA with Dynamic Lot Scaling](https://c.mql5.com/2/119/Automating_Trading_Strategies_in_MQL5_Part_7__LOGO.png)[Automating Trading Strategies in MQL5 (Part 7): Building a Grid Trading EA with Dynamic Lot Scaling](https://www.mql5.com/en/articles/17190)

In this article, we build a grid trading expert advisor in MQL5 that uses dynamic lot scaling. We cover the strategy design, code implementation, and backtesting process. Finally, we share key insights and best practices for optimizing the automated trading system.

![From Basic to Intermediate: Variables (III)](https://c.mql5.com/2/87/Do_b9sico_ao_intermediwrio_Varicveis_III____LOGO.png)[From Basic to Intermediate: Variables (III)](https://www.mql5.com/en/articles/15304)

Today we will look at how to use predefined MQL5 language variables and constants. In addition, we will analyze another special type of variables: functions. Knowing how to properly work with these variables can mean the difference between an application that works and one that doesn't. In order to understand what is presented here, it is necessary to understand the material that was discussed in previous articles.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/16562&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049529115373055259)

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