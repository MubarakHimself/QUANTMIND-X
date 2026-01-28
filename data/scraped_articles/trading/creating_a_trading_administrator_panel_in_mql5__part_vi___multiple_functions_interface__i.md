---
title: Creating a Trading Administrator Panel in MQL5 (Part VI): Multiple Functions Interface (I)
url: https://www.mql5.com/en/articles/16240
categories: Trading, Integration
relevance_score: 6
scraped_at: 2026-01-22T17:59:58.809727
---

[![](https://www.mql5.com/ff/sh/a27a2kwmtszm2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Subscribe to traders' channels or create your own.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=vpcudokyepxfrcxrpjcktglhsjlemtza&s=f08ad2c1289e29bd5630f1ef977aef297d5cdbfcb686faed4a4b0f1e276d3c4a&uid=&ref=https://www.mql5.com/en/articles/16240&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049547248724979051)

MetaTrader 5 / Examples


### Core Content:

- [Introduction.](https://www.mql5.com/en/articles/16240#para1)
- [Exploring other possible panel features and Administrator roles.](https://www.mql5.com/en/articles/16240#para2)
- [Implementation of MQL5 for Multiple Functionality Interface.](https://www.mql5.com/en/articles/16240#para3)
- [Code Development](https://www.mql5.com/en/articles/16240#para4)
- [Testing and Results](https://www.mql5.com/en/articles/16240#para5)
- [Conclusion](https://www.mql5.com/en/articles/16240#para6)

### Introduction

Based on my experience with MetaTrader 5, a single chart can only support one Expert Advisor, multiple indicators, and a single script at a time. This limitation of having only one Expert Advisor per session highlights the need to create a highly versatile panel that can handle a range of tasks without requiring the EA to be changed on the chart. Below are some of the operations that this panel can be made to perform.

- Order Management
- Position Management

- Market Analysis and Data Display

- Graphical and Chart Features

- Risk Management and Reporting etc.

As you can see from the list, there are currently no features incorporated into the existing Admin Panel. Today, our goal is to redesign the panel's logic to support multiple features that were previously missing. I've included an image that illustrates the limitation in MetaTrader 5, where it is not possible to add multiple Expert Advisors to a single chart. In the case presented below, you will be overwriting the currently running Expert Advisor to add a new one.

![Adding an EA to the chart overight the one already added before.](https://c.mql5.com/2/126/terminal64_YoSRiKdDjk__2.gif)

Expert Advisors cannot share a single chart

Before we begin designing our program to incorporate new features, let's first take a look at the current offerings of the available Admin Panel, from which we'll build upon.

Here’s a diagram illustrating the basic flow, starting from the initialization of the program until all its features are fully realized. The process begins with the login dialog prompt, and upon successful login, the user is prompted for 2FA verification. During this step, a 6-digit verification code is sent to the user’s Telegram account. Once the authentication is successful, the Admin Panel and its features are unlocked for use.

![Current Admin Panel](https://c.mql5.com/2/126/current_admin_panel__2.png)

The basic processes in our current Panel

To incorporate the features I have planned, I intend to leverage multiple dialog class panels, similar to how websites work. After logging in, users will be directed to the homepage, which provides an overall summary of everything the website has to offer. On this main page, buttons and links will lead to other panels or features of the platform. Theoretically, using the block image I have inserted below, the offerings of the new panel will be significantly enhanced, giving users a streamlined and dynamic experience.

![New Panel Imagined](https://c.mql5.com/2/126/current_admin_panel1__2.png)

New Admin Panel

Our previous Admin Panel was simple and primarily focused on communication between the Trading Administrator and the Telegram channel audience, including both groups and channels. As per the design outlined above, it is now much easier to expand our program. We will introduce an Admin Home Panel, which will feature buttons to access other functionalities within their respective panels. What was once known as the Admin Panel will now be referred to as the Communications Panel. Additionally, we will add a separate panel for Trade Management, which will allow the admin to manage trades directly without leaving the chart. This Trade Management Panel will include sub-features such as opening new orders and managing existing ones.

### Exploring other possible panel features and Administrator roles

Under Trade management, let's focus on two most important areas outlined below briefly detail for the sake of easily grabbing the expansion process, though we could do for more than that.

Order Management

- Quick Entry and Exit Buttons: Provide options to quickly enter and exit trades, adjust order sizes, or place pending orders with pre-set parameters.
- Order Modification: Allows modifications to Stop Loss (SL), Take Profit (TP), or Trailing Stop levels directly from the panel.
- Order Overview and Filtering: Show open, pending, and closed orders with filters by order type, symbol, or status.

Position Management

- Quick Position Actions: Buttons for closing all positions, closing specific types (e.g., all longs or shorts), and reversing positions.
- Risk Management Tools: Display live metrics like exposure, margin usage, profit/loss, and account equity.
- Auto-Close Conditions: Options to auto-close positions at specific times, on certain news events, or if market conditions meet certain criteria.

### Implementation of MQL5 for Multiple Functionality Interface

We will modify the former Admin Panel, which originally served as the main panel, into a sub-panel titled "Communications Panel" accessible through the Admin Home Panel. This approach will keep all communication-related elements in one place for better organization. Additionally, as previously mentioned, we will introduce a separate panel for Trade Management, which will also be accessible from the Admin Home Panel. To ensure a smooth user experience, we will allow users to easily switch back and forth between the Admin Home Panel and the sub-panels.

It is important to note that at this stage, we are using modified library files for Dialog, Edit, and Button to manage the theme effectively. If you use the unmodified versions, you may encounter compilation errors. To avoid this, ensure that your access to include/control folder within your MQL5 directory and overwrite the existing files with the extended versions we are using. This step is crucial for ensuring compatibility and proper functionality in your project

Up next, we will focus on developing the code to introduce the new features, starting with the implementation of the Admin Home Panel, followed by the transition of the previous Admin Panel into the Communications Panel. We will also work on the Trade Management Panel, which will allow seamless trade operations directly from the chart. Our goal is to create a smooth user experience with easy navigation between the home panel and the sub-panels, ensuring that all features work efficiently and without conflicts.

### Code Development

The previous source code, which amounted to 602 lines, is now about to expand as we incorporate additional features. However, the modification methods we're using are designed to maintain clarity and organization, no matter how long the code becomes.

It’s crucial to avoid altering parts of the code that are functioning well, focusing only on the areas that require updates. The block diagram above illustrates a streamlined approach to incorporating new features.

Assuming you now have a good understanding of the _Dialog_ class and the coordinate system for creating panel windows in MQL5, we will now introduce a new dialog window called the Admin Home Panel. This panel will be displayed during initialization and, after successful password and 2FA verification, will grant access to other sub-windows with different functionalities.

This approach ensures that the program remains well-structured and easy to navigate, while expanding its capabilities.

After including the necessary library files, we begin by declaring our global variables.

```
// Global variables
CDialog adminHomePanel, tradeManagementPanel, communicationsPanel;
```

On initialization function _(OnInit())_  we want the program to first display an authentication prompt by calling _ShowAuthenticationPrompt()_, and if authentication fails, it prints an error message and exits with _INIT\_FAILED_.

It will attempt to create three dialog panels (Admin Home Panel, Communications Panel, and Trade Management Panel) linked to the current chart using the Create method, checking for success at each step; if any panel creation fails, it prints the corresponding error message and exits.

Next, it sets up controls for the admin home panel through _CreateAdminHomeControls()_ and for other panels with _CreateControls()_, again ensuring each operation completes successfully.

Finally, it hides all created panels to prevent them from displaying immediately except for the authentication panel for verification process to take place and returns _INIT\_SUCCEEDED_ if all operations are successful, indicating that the Expert Advisor is ready to function.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    if (!ShowAuthenticationPrompt())
    {
        Print("Authorization failed. Exiting...");
        return INIT_FAILED;
    }

    if (!adminHomePanel.Create(ChartID(), "Admin Home Panel", 0, 30, 30, 500, 500))
    {
        Print("Failed to create Admin Home Panel");
        return INIT_FAILED;
    }

    if (!CreateAdminHomeControls())
    {
        Print("Home panel control creation failed");
        return INIT_FAILED;
    }
    if (!communicationsPanel.Create(ChartID(), "Communications Panel", 0, 30, 30, 500, 500))
    {
        Print("Failed to create Communications panel dialog");
        return INIT_FAILED;
    }

    if (!CreateControls())
    {
        Print("Control creation failed");
        return INIT_FAILED;
    }
    if (!tradeManagementPanel.Create(ChartID(), "Trade Management Panel", 0, 30, 30, 500, 500))
    {
        Print("Failed to create Communictions panel dialog");
        return INIT_FAILED;
    }


    adminHomePanel.Hide(); // Hide home panel by default on initialization
    communicationsPanel.Hide(); // Hide the Communications Panel
    tradeManagementPanel.Hide();// Hide the Trade Management Panel
    return INIT_SUCCEEDED;
}
```

The _CreateAdminHomeControls()_ function is responsible for creating and adding various control buttons to the Admin Home Panel in our Admin Panel expert advisor. It will first retrieve the current chart ID using ChartID() and then attempts to create a "Trade Management Access" button; if creation fails, it prints an error message and returns false. If successful, it sets the button's text and adds it to the _adminHomePanel_.

This process is repeated for a "Communications Panel Access" button, a minimize button, a maximize button, and a close button, each time checking for successful creation, setting their corresponding display texts, and adding them to the panel. If all buttons are created and added successfully, the function returns true, indicating that the control creation was successful.

```
//+------------------------------------------------------------------+
//| Admin Home Panel controls creation                               |
//+------------------------------------------------------------------+
bool CreateAdminHomeControls()
{
    long chart_id = ChartID();

    if (!tradeMgmtAccessButton.Create(chart_id, "TradeMgmtAccessButton", 0, 50, 50, 250, 90))
    {
        Print("Failed to create Trade Management Access button");
        return false;
    }
    tradeMgmtAccessButton.Text("Trade Management Panel");
    adminHomePanel.Add(tradeMgmtAccessButton);

    if (!communicationsPanelAccessButton.Create(chart_id, "CommunicationsPanelAccessButton", 0, 50, 100, 250, 140))
    {
        Print("Failed to create Communications Panel Access button");
        return false;
    }
    communicationsPanelAccessButton.Text("Communications Panel");
    adminHomePanel.Add(communicationsPanelAccessButton);

    if (!minimizeButton.Create(chart_id, "MinimizeButton", 0, 375, -22, 405, 0))
    {
        Print("Failed to create minimize button");
        return false;
    }
    minimizeButton.Text("_");
    adminHomePanel.Add(minimizeButton);

    if (!maximizeButton.Create(chart_id, "MaximizeButton", 0, 405, -22, 435, 0))
    {
        Print("Failed to create maximize button");
        return false;
    }
    maximizeButton.Text("[ ]");
    adminHomePanel.Add(maximizeButton);

    if (!closeButton.Create(chart_id, "CloseButton", 0, 435, -22, 465, 0))
    {
        Print("Failed to create close button");
        return false;
    }
    closeButton.Text("X");
    adminHomePanel.Add(closeButton);

    return true;
}
```

The other crucial function is the _OnChartEvent_ function designed to handle various chart events in a MetaTrader 5, for user interaction with the graphical elements of the application. When a user clicks on an object on the chart, the function checks the type of the event _(CHARTEVENT\_OBJECT\_CLICK)_ and then identifies which specific object was clicked using the sparam string parameter.

For instance, if the " _TradeMgmtAccessButton_" is clicked, the function shows the _tradeManagementPanel_ while hiding the _adminHomePanel_, ensuring that users have a seamless navigation experience within the application. Similar logic is applied for the " _CommunicationsPanelAccessButton_," which displays the communications panel, and the other buttons handle minimizing, maximizing, or closing the application.

This structured event handling is essential for maintaining an intuitive interface, allowing users to interact with the application efficiently and effectively based on user actions.

```
//+------------------------------------------------------------------+
//| Handle chart events                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
    if (id == CHARTEVENT_OBJECT_CLICK)
    {
        if (sparam == "TradeMgmtAccessButton")
        {
           tradeManagementPanel.Show();
           adminHomePanel.Hide();
        }
        else if (sparam == "CommunicationsPanelAccessButton")
        {
            communicationsPanel.Show();
            adminHomePanel.Hide();
        }
        else if (sparam == "MinimizeButton")
        {
            OnMinimizeButtonClick();
        }
        else if (sparam == "MaximizeButton")
        {
            OnMaximizeButtonClick();
        }
        else if (sparam == "CloseButton")
        {
            ExpertRemove();
        }
    }
    {
        if (sparam == "LoginButton")
        {
            OnLoginButtonClick();
        }
        else if (sparam == "CloseAuthButton")
        {
            OnCloseAuthButtonClick();
        }
        else if (sparam == "TwoFALoginButton")
        {
            OnTwoFALoginButtonClick();
        }
        else if (sparam == "Close2FAButton")
        {
            OnClose2FAButtonClick();
        }
    }
```

For clarity, I have renamed everything that was previously referred to as the Admin Panel to "Communications Panel." The lines of code to be modified are highlighted in red in the snippet below.

```
// Global variables
CDialog adminPanel;    //To rename it to communicationsPanel
CDialog authentication, twoFactorAuth;
CButton sendButton, clearButton, changeFontButton, toggleThemeButton;
CButton loginButton, closeAuthButton, twoFALoginButton, close2FAButton;
CButton quickMessageButtons[8], minimizeButton, maximizeButton, closeButton;
CEdit inputBox, passwordInputBox, twoFACodeInput;
CLabel charCounter, passwordPromptLabel, feedbackLabel, twoFAPromptLabel, twoFAFeedbackLabel;
bool minimized = false;
bool darkTheme = false;
int MAX_MESSAGE_LENGTH = 4096;
string availableFonts[] = { "Arial", "Courier New", "Verdana", "Times New Roman" };
int currentFontIndex = 0;
string Password = "2024"; // Hardcoded password
string twoFACode = "";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    if (!ShowAuthenticationPrompt())
    {
        Print("Authorization failed. Exiting...");
        return INIT_FAILED;
    }

    if (!adminPanel.Create(ChartID(), "Admin Panel", 0, 30, 30, 500, 500))
    {
        Print("Failed to create admin panel dialog");
        return INIT_FAILED;
    }

    if (!CreateControls())
    {
        Print("Control creation failed");
        return INIT_FAILED;
    }

    adminPanel.Hide();
    Print("Initialization complete");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Show authentication input dialog                                 |
//+------------------------------------------------------------------+
bool ShowAuthenticationPrompt()
{
    if (!authentication.Create(ChartID(), "Authentication", 0, 100, 100, 500, 300))
    {
        Print("Failed to create authentication dialog");
        return false;
    }

    if (!passwordInputBox.Create(ChartID(), "PasswordInputBox", 0, 20, 70, 260, 95))
    {
        Print("Failed to create password input box");
        return false;
    }
    authentication.Add(passwordInputBox);

    if (!passwordPromptLabel.Create(ChartID(), "PasswordPromptLabel", 0, 20, 20, 260, 40))
    {
        Print("Failed to create password prompt label");
        return false;
    }
    passwordPromptLabel.Text("Enter password: Access Admin Panel");
    authentication.Add(passwordPromptLabel);

    if (!feedbackLabel.Create(ChartID(), "FeedbackLabel", 0, 20, 140, 380, 160))
    {
        Print("Failed to create feedback label");
        return false;
    }
    feedbackLabel.Text("");
    feedbackLabel.Color(clrRed); // Red color for incorrect attempts
    authentication.Add(feedbackLabel);

    if (!loginButton.Create(ChartID(), "LoginButton", 0, 20, 120, 100, 140))
    {
        Print("Failed to create login button");
        return false;
    }
    loginButton.Text("Login");
    authentication.Add(loginButton);

    if (!closeAuthButton.Create(ChartID(), "CloseAuthButton", 0, 120, 120, 200, 140))
    {
        Print("Failed to create close button for authentication");
        return false;
    }
    closeAuthButton.Text("Close");
    authentication.Add(closeAuthButton);

    authentication.Show();
    ChartRedraw();

    return true;
}

//+------------------------------------------------------------------+
//| Show two-factor authentication input dialog                      |
//+------------------------------------------------------------------+
void ShowTwoFactorAuthPrompt()
{
    if (!twoFactorAuth.Create(ChartID(), "Two-Factor Authentication", 0, 100, 100, 500, 300))
    {
        Print("Failed to create 2FA dialog");
        return;
    }

    if (!twoFACodeInput.Create(ChartID(), "TwoFACodeInput", 0, 20, 70, 260, 95))
    {
        Print("Failed to create 2FA code input box");
        return;
    }
    twoFactorAuth.Add(twoFACodeInput);

    if (!twoFAPromptLabel.Create(ChartID(), "TwoFAPromptLabel", 0, 20, 20, 380, 40))
    {
        Print("Failed to create 2FA prompt label");
        return;
    }
    twoFAPromptLabel.Text("Enter the 2FA code sent to your Telegram:");
    twoFactorAuth.Add(twoFAPromptLabel);

    if (!twoFAFeedbackLabel.Create(ChartID(), "TwoFAFeedbackLabel", 0, 20, 140, 380, 160))
    {
        Print("Failed to create 2FA feedback label");
        return;
    }
    twoFAFeedbackLabel.Text("");
    twoFAFeedbackLabel.Color(clrRed); // Red color for incorrect 2FA attempts
    twoFactorAuth.Add(twoFAFeedbackLabel);

    if (!twoFALoginButton.Create(ChartID(), "TwoFALoginButton", 0, 20, 120, 100, 140))
    {
        Print("Failed to create 2FA login button");
        return;
    }
    twoFALoginButton.Text("Verify");
    twoFactorAuth.Add(twoFALoginButton);

    if (!close2FAButton.Create(ChartID(), "Close2FAButton", 0, 120, 120, 200, 140))
    {
        Print("Failed to create close button for 2FA");
        return;
    }
    close2FAButton.Text("Close");
    twoFactorAuth.Add(close2FAButton);

    twoFactorAuth.Show();
    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Handle chart events                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
    if (id == CHARTEVENT_OBJECT_CLICK)
    {
        if (sparam == "LoginButton")
        {
            OnLoginButtonClick();
        }
        else if (sparam == "CloseAuthButton")
        {
            OnCloseAuthButtonClick();
        }
        else if (sparam == "TwoFALoginButton")
        {
            OnTwoFALoginButtonClick();
        }
        else if (sparam == "Close2FAButton")
        {
            OnClose2FAButtonClick();
        }
    }

    switch (id)
    {
        case CHARTEVENT_OBJECT_CLICK:
            if (sparam == "SendButton") OnSendButtonClick();
            else if (sparam == "ClearButton") OnClearButtonClick();
            else if (sparam == "ChangeFontButton") OnChangeFontButtonClick();
            else if (sparam == "ToggleThemeButton") OnToggleThemeButtonClick();
            else if (sparam == "MinimizeButton") OnMinimizeButtonClick();
            else if (sparam == "MaximizeButton") OnMaximizeButtonClick();
            else if (sparam == "CloseButton") OnCloseButtonClick();
            else if (StringFind(sparam, "QuickMessageButton") != -1)
            {
                long index = StringToInteger(StringSubstr(sparam, 18));
                OnQuickMessageButtonClick(index - 1);
            }
            break;

        case CHARTEVENT_OBJECT_ENDEDIT:
            if (sparam == "InputBox") OnInputChange();
            break;
    }
}

//+------------------------------------------------------------------+
//| Handle login button click                                        |
//+------------------------------------------------------------------+
void OnLoginButtonClick()
{
    string enteredPassword = passwordInputBox.Text();
    if (enteredPassword == Password)
    {
        twoFACode = GenerateRandom6DigitCode();
        SendMessageToTelegram("A login attempt was made on the Admin Panel. Please use this code to verify your identity: " + twoFACode, Hardcoded2FAChatId, Hardcoded2FABotToken);

        authentication.Destroy();
        ShowTwoFactorAuthPrompt();
        Print("Password authentication successful. A 2FA code has been sent to your Telegram.");
    }
    else
    {
        feedbackLabel.Text("Wrong password. Try again.");
        passwordInputBox.Text("");
    }
}

//+------------------------------------------------------------------+
//| Handle 2FA login button click                                    |
//+------------------------------------------------------------------+
void OnTwoFALoginButtonClick()
{
    string enteredCode = twoFACodeInput.Text();
    if (enteredCode == twoFACode)
    {
        twoFactorAuth.Destroy();
        adminPanel.Show();
        Print("2FA authentication successful.");
    }
    else
    {
        twoFAFeedbackLabel.Text("Wrong code. Try again.");
        twoFACodeInput.Text("");
    }
}

//+------------------------------------------------------------------+
//| Handle close button for authentication                           |
//+------------------------------------------------------------------+
void OnCloseAuthButtonClick()
{
    authentication.Destroy();
    ExpertRemove(); // Exit the expert
    Print("Authentication dialog closed.");
}

//+------------------------------------------------------------------+
//| Handle close button for 2FA                                      |
//+------------------------------------------------------------------+
void OnClose2FAButtonClick()
{
    twoFactorAuth.Destroy();
    ExpertRemove();
    Print("2FA dialog closed.");
}

//+------------------------------------------------------------------+
//| Create necessary UI controls                                     |
//+------------------------------------------------------------------+
bool CreateControls()
{
    long chart_id = ChartID();

    if (!inputBox.Create(chart_id, "InputBox", 0, 5, 25, 460, 95))
    {
        Print("Failed to create input box");
        return false;
    }
    adminPanel.Add(inputBox);

    if (!charCounter.Create(chart_id, "CharCounter", 0, 380, 5, 460, 25))
    {
        Print("Failed to create character counter");
        return false;
    }
    charCounter.Text("0/" + IntegerToString(MAX_MESSAGE_LENGTH));
    adminPanel.Add(charCounter);

    if (!clearButton.Create(chart_id, "ClearButton", 0, 235, 95, 345, 125))
    {
        Print("Failed to create clear button");
        return false;
    }
    clearButton.Text("Clear");
    adminPanel.Add(clearButton);

    if (!sendButton.Create(chart_id, "SendButton", 0, 350, 95, 460, 125))
    {
        Print("Failed to create send button");
        return false;
    }
    sendButton.Text("Send");
    adminPanel.Add(sendButton);

    if (!changeFontButton.Create(chart_id, "ChangeFontButton", 0, 95, 95, 230, 115))
    {
        Print("Failed to create change font button");
        return false;
    }
    changeFontButton.Text("Font<>");
    adminPanel.Add(changeFontButton);

    if (!toggleThemeButton.Create(chart_id, "ToggleThemeButton", 0, 5, 95, 90, 115))
    {
        Print("Failed to create toggle theme button");
        return false;
    }
    toggleThemeButton.Text("Theme<>");
    adminPanel.Add(toggleThemeButton);

    if (!minimizeButton.Create(chart_id, "MinimizeButton", 0, 375, -22, 405, 0))
    {
        Print("Failed to create minimize button");
        return false;
    }
    minimizeButton.Text("_");
    adminPanel.Add(minimizeButton);

    if (!maximizeButton.Create(chart_id, "MaximizeButton", 0, 405, -22, 435, 0))
    {
        Print("Failed to create maximize button");
        return false;
    }
    maximizeButton.Text("[ ]");
    adminPanel.Add(maximizeButton);

    if (!closeButton.Create(chart_id, "CloseButton", 0, 435, -22, 465, 0))
    {
        Print("Failed to create close button");
        return false;
    }
    closeButton.Text("X");
    adminPanel.Add(closeButton);

    return CreateQuickMessageButtons();
}

//+------------------------------------------------------------------+
//| Create quick message buttons                                     |
//+------------------------------------------------------------------+
bool CreateQuickMessageButtons()
{
    string quickMessages[] = { QuickMessage1, QuickMessage2, QuickMessage3, QuickMessage4, QuickMessage5, QuickMessage6, QuickMessage7, QuickMessage8 };
    int startX = 5, startY = 160, width = 222, height = 65, spacing = 5;

    for (int i = 0; i < ArraySize(quickMessages); i++)
    {
        bool created = quickMessageButtons[i].Create(ChartID(), "QuickMessageButton" + IntegerToString(i + 1), 0,
            startX + (i % 2) * (width + spacing), startY + (i / 2) * (height + spacing),
            startX + (i % 2) * (width + spacing) + width, startY + (i / 2) * (height + spacing) + height);

        if (!created)
        {
            Print("Failed to create quick message button ", i + 1);
            return false;
        }
        quickMessageButtons[i].Text(quickMessages[i]);
        adminPanel.Add(quickMessageButtons[i]);
    }
    return true;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    adminPanel.Destroy();
    Print("Deinitialization complete");
}

//+------------------------------------------------------------------+
//| Handle custom message send button click                          |
//+------------------------------------------------------------------+
void OnSendButtonClick()
{
    string message = inputBox.Text();
    if (StringLen(message) > 0)
    {
        if (SendMessageToTelegram(message, InputChatId, InputBotToken))
            Print("Custom message sent: ", message);
        else
            Print("Failed to send custom message.");
    }
    else
    {
        Print("No message entered.");
    }
}

//+------------------------------------------------------------------+
//| Handle clear button click                                        |
//+------------------------------------------------------------------+
void OnClearButtonClick()
{
    inputBox.Text("");
    OnInputChange();
    Print("Input box cleared.");
}

//+------------------------------------------------------------------+
//| Handle quick message button click                                |
//+------------------------------------------------------------------+
void OnQuickMessageButtonClick(long index)
{
    string quickMessages[] = { QuickMessage1, QuickMessage2, QuickMessage3, QuickMessage4, QuickMessage5, QuickMessage6, QuickMessage7, QuickMessage8 };
    string message = quickMessages[(int)index];

    if (SendMessageToTelegram(message, InputChatId, InputBotToken))
        Print("Quick message sent: ", message);
    else
        Print("Failed to send quick message.");
}

//+------------------------------------------------------------------+
//| Update character counter                                         |
//+------------------------------------------------------------------+
void OnInputChange()
{
    int currentLength = StringLen(inputBox.Text());
    charCounter.Text(IntegerToString(currentLength) + "/" + IntegerToString(MAX_MESSAGE_LENGTH));
    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Handle toggle theme button click                                 |
//+------------------------------------------------------------------+
void OnToggleThemeButtonClick()
{
    darkTheme = !darkTheme;
    UpdateThemeColors();
    Print("Theme toggled: ", darkTheme ? "Dark" : "Light");
}

//+------------------------------------------------------------------+
//| Update theme colors for the panel                                |
//+------------------------------------------------------------------+
void UpdateThemeColors()
{
    color textColor = darkTheme ? clrWhite : clrBlack;
    color buttonBgColor = darkTheme ? clrDarkSlateGray : clrGainsboro;
    color borderColor = darkTheme ? clrSlateGray : clrGray;
    color bgColor = darkTheme ? clrDarkBlue : clrWhite;


    UpdateButtonTheme(clearButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(sendButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(toggleThemeButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(changeFontButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(minimizeButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(maximizeButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(closeButton, textColor, buttonBgColor, borderColor);

    for (int i = 0; i < ArraySize(quickMessageButtons); i++)
    {
        UpdateButtonTheme(quickMessageButtons[i], textColor, buttonBgColor, borderColor);
    }

    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Apply theme settings to a button                                 |
//+------------------------------------------------------------------+
void UpdateButtonTheme(CButton &button, color textColor, color bgColor, color borderColor)
{
    button.SetTextColor(textColor);
    button.SetBackgroundColor(bgColor);
    button.SetBorderColor(borderColor);
}

//+------------------------------------------------------------------+
//| Handle change font button click                                  |
//+------------------------------------------------------------------+
void OnChangeFontButtonClick()
{
    currentFontIndex = (currentFontIndex + 1) % ArraySize(availableFonts);
    SetFontForAll(availableFonts[currentFontIndex]);
    Print("Font changed to: ", availableFonts[currentFontIndex]);
    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Set font for all input boxes and buttons                         |
//+------------------------------------------------------------------+
void SetFontForAll(string fontName)
{
    inputBox.Font(fontName);
    clearButton.Font(fontName);
    sendButton.Font(fontName);
    toggleThemeButton.Font(fontName);
    changeFontButton.Font(fontName);
    minimizeButton.Font(fontName);
    maximizeButton.Font(fontName);
    closeButton.Font(fontName);

    for (int i = 0; i < ArraySize(quickMessageButtons); i++)
    {
        quickMessageButtons[i].Font(fontName);
    }
}

//+------------------------------------------------------------------+
//| Generate a random 6-digit code for 2FA                           |
//+------------------------------------------------------------------+
string GenerateRandom6DigitCode()
{
    int code = MathRand() % 1000000; // Produces a 6-digit number
    return StringFormat("%06d", code); // Ensures leading zeros
}

//+------------------------------------------------------------------+
//| Handle minimize button click                                     |
//+------------------------------------------------------------------+
void OnMinimizeButtonClick()
{
    minimized = true;
    adminPanel.Hide();
    minimizeButton.Hide();
    maximizeButton.Show();
    closeButton.Show();
    Print("Panel minimized.");
}

//+------------------------------------------------------------------+
//| Handle maximize button click                                     |
//+------------------------------------------------------------------+
void OnMaximizeButtonClick()
{
    if (minimized)
    {
        adminPanel.Show();
        minimizeButton.Show();
        maximizeButton.Hide();
        closeButton.Hide();
        minimized = false;
        Print("Panel maximized.");
    }
}

//+------------------------------------------------------------------+
//| Handle close button click for admin panel                        |
//+------------------------------------------------------------------+
void OnCloseButtonClick()
{
    ExpertRemove();
    Print("Admin panel closed.");
}

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

Without altering much of the code’s functionality, I’ve simply renamed and edited all references to the former Admin Panel. Below is the combined code, now incorporating new features for initial testing.

```
//+------------------------------------------------------------------+
//|                                             Admin Panel.mq5      |
//|                           Copyright 2024, Clemence Benjamin      |
//|        https://www.mql5.com/en/users/billionaire2024/seller      |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
#property description "A secure and responsive Admin Panel. Send messages to your telegram clients without leaving MT5"
#property version   "1.21"

#include <Trade\Trade.mqh>
#include <Controls\Dialog.mqh>
#include <Controls\Button.mqh>
#include <Controls\Edit.mqh>
#include <Controls\Label.mqh>

// Input parameters for quick messages
input string QuickMessage1 = "Updates";
input string QuickMessage2 = "Close all";
input string QuickMessage3 = "In deep profits";
input string QuickMessage4 = "Hold position";
input string QuickMessage5 = "Swing Entry";
input string QuickMessage6 = "Scalp Entry";
input string QuickMessage7 = "Book profit";
input string QuickMessage8 = "Invalid Signal";
input string InputChatId = "YOUR_CHAT_ID";
input string InputBotToken = "YOUR_BOT_TOKEN";

// Constants for 2FA
const string Hardcoded2FAChatId = "Replace with your chat ID from telegram";
const string Hardcoded2FABotToken = "Replace with your bot token from telegram";

// Global variables
CDialog adminHomePanel, tradeManagementPanel, communicationsPanel;
CDialog authentication, twoFactorAuth;
CButton sendButton, clearButton, changeFontButton, toggleThemeButton;
CButton loginButton, closeAuthButton, twoFALoginButton, close2FAButton;
CButton quickMessageButtons[8], minimizeButton, maximizeButton, closeButton;
CButton tradeMgmtAccessButton, communicationsPanelAccessButton;
CEdit inputBox, passwordInputBox, twoFACodeInput;
CLabel charCounter, passwordPromptLabel, feedbackLabel, twoFAPromptLabel, twoFAFeedbackLabel;
bool minimized = false;
bool darkTheme = false;
int MAX_MESSAGE_LENGTH = 4096;
string availableFonts[] = { "Arial", "Courier New", "Verdana", "Times New Roman" };
int currentFontIndex = 0;
string Password = "2024"; // Hardcoded password
string twoFACode = "";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    if (!ShowAuthenticationPrompt())
    {
        Print("Authorization failed. Exiting...");
        return INIT_FAILED;
    }

    if (!adminHomePanel.Create(ChartID(), "Admin Home Panel", 0, 30, 30, 500, 500))
    {
        Print("Failed to create Admin Home Panel");
        return INIT_FAILED;
    }

    if (!CreateAdminHomeControls())
    {
        Print("Home panel control creation failed");
        return INIT_FAILED;
    }
    if (!communicationsPanel.Create(ChartID(), "Communications Panel", 0, 30, 30, 500, 500))
    {
        Print("Failed to create Communications panel dialog");
        return INIT_FAILED;
    }

    if (!CreateControls())
    {
        Print("Control creation failed");
        return INIT_FAILED;
    }
    if (!tradeManagementPanel.Create(ChartID(), "Trade Management Panel", 0, 30, 30, 500, 500))
    {
        Print("Failed to create Communictions panel dialog");
        return INIT_FAILED;
    }


    adminHomePanel.Hide(); // Hide home panel by default on initialization
    communicationsPanel.Hide(); // Hide the Communications Panel
    tradeManagementPanel.Hide();// Hide the Trade Management Panel
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Show authentication input dialog                                 |
//+------------------------------------------------------------------+
bool ShowAuthenticationPrompt()
{
    if (!authentication.Create(ChartID(), "Authentication", 0, 100, 100, 500, 300))
    {
        Print("Failed to create authentication dialog");
        return false;
    }

    if (!passwordInputBox.Create(ChartID(), "PasswordInputBox", 0, 20, 70, 260, 95))
    {
        Print("Failed to create password input box");
        return false;
    }
    authentication.Add(passwordInputBox);

    if (!passwordPromptLabel.Create(ChartID(), "PasswordPromptLabel", 0, 20, 20, 260, 40))
    {
        Print("Failed to create password prompt label");
        return false;
    }
    passwordPromptLabel.Text("Enter password: Access Admin Panel");
    authentication.Add(passwordPromptLabel);

    if (!feedbackLabel.Create(ChartID(), "FeedbackLabel", 0, 20, 140, 380, 160))
    {
        Print("Failed to create feedback label");
        return false;
    }
    feedbackLabel.Text("");
    feedbackLabel.Color(clrRed); // Red color for incorrect attempts
    authentication.Add(feedbackLabel);

    if (!loginButton.Create(ChartID(), "LoginButton", 0, 20, 120, 100, 140))
    {
        Print("Failed to create login button");
        return false;
    }
    loginButton.Text("Login");
    authentication.Add(loginButton);

    if (!closeAuthButton.Create(ChartID(), "CloseAuthButton", 0, 120, 120, 200, 140))
    {
        Print("Failed to create close button for authentication");
        return false;
    }
    closeAuthButton.Text("Close");
    authentication.Add(closeAuthButton);

    authentication.Show();
    ChartRedraw();
    return true;
}

//+------------------------------------------------------------------+
//| Show two-factor authentication input dialog                      |
//+------------------------------------------------------------------+
void ShowTwoFactorAuthPrompt()
{
    if (!twoFactorAuth.Create(ChartID(), "Two-Factor Authentication", 0, 100, 100, 500, 300))
    {
        Print("Failed to create 2FA dialog");
        return;
    }

    if (!twoFACodeInput.Create(ChartID(), "TwoFACodeInput", 0, 20, 70, 260, 95))
    {
        Print("Failed to create 2FA code input box");
        return;
    }
    twoFactorAuth.Add(twoFACodeInput);

    if (!twoFAPromptLabel.Create(ChartID(), "TwoFAPromptLabel", 0, 20, 20, 380, 40))
    {
        Print("Failed to create 2FA prompt label");
        return;
    }
    twoFAPromptLabel.Text("Enter the 2FA code sent to your Telegram:");
    twoFactorAuth.Add(twoFAPromptLabel);

    if (!twoFAFeedbackLabel.Create(ChartID(), "TwoFAFeedbackLabel", 0, 20, 140, 380, 160))
    {
        Print("Failed to create 2FA feedback label");
        return;
    }
    twoFAFeedbackLabel.Text("");
    twoFAFeedbackLabel.Color(clrRed); // Red color for incorrect 2FA attempts
    twoFactorAuth.Add(twoFAFeedbackLabel);

    if (!twoFALoginButton.Create(ChartID(), "TwoFALoginButton", 0, 20, 120, 100, 140))
    {
        Print("Failed to create 2FA login button");
        return;
    }
    twoFALoginButton.Text("Verify");
    twoFactorAuth.Add(twoFALoginButton);

    if (!close2FAButton.Create(ChartID(), "Close2FAButton", 0, 120, 120, 200, 140))
    {
        Print("Failed to create close button for 2FA");
        return;
    }
    close2FAButton.Text("Close");
    twoFactorAuth.Add(close2FAButton);

    twoFactorAuth.Show();
    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Admin Home Panel controls creation                               |
//+------------------------------------------------------------------+
bool CreateAdminHomeControls()
{
    long chart_id = ChartID();

    if (!tradeMgmtAccessButton.Create(chart_id, "TradeMgmtAccessButton", 0, 50, 50, 250, 90))
    {
        Print("Failed to create Trade Management Access button");
        return false;
    }
    tradeMgmtAccessButton.Text("Trade Management Panel");
    adminHomePanel.Add(tradeMgmtAccessButton);

    if (!communicationsPanelAccessButton.Create(chart_id, "CommunicationsPanelAccessButton", 0, 50, 100, 250, 140))
    {
        Print("Failed to create Communications Panel Access button");
        return false;
    }
    communicationsPanelAccessButton.Text("Communications Panel");
    adminHomePanel.Add(communicationsPanelAccessButton);

    if (!minimizeButton.Create(chart_id, "MinimizeButton", 0, 375, -22, 405, 0))
    {
        Print("Failed to create minimize button");
        return false;
    }
    minimizeButton.Text("_");
    adminHomePanel.Add(minimizeButton);

    if (!maximizeButton.Create(chart_id, "MaximizeButton", 0, 405, -22, 435, 0))
    {
        Print("Failed to create maximize button");
        return false;
    }
    maximizeButton.Text("[ ]");
    adminHomePanel.Add(maximizeButton);

    if (!closeButton.Create(chart_id, "CloseButton", 0, 435, -22, 465, 0))
    {
        Print("Failed to create close button");
        return false;
    }
    closeButton.Text("X");
    adminHomePanel.Add(closeButton);

    return true;
}

//+------------------------------------------------------------------+
//| Handle chart events                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
    if (id == CHARTEVENT_OBJECT_CLICK)
    {
        if (sparam == "TradeMgmtAccessButton")
        {
           tradeManagementPanel.Show();
           adminHomePanel.Hide();
        }
        else if (sparam == "CommunicationsPanelAccessButton")
        {
            communicationsPanel.Show();
            adminHomePanel.Hide();
        }
        else if (sparam == "MinimizeButton")
        {
            OnMinimizeButtonClick();
        }
        else if (sparam == "MaximizeButton")
        {
            OnMaximizeButtonClick();
        }
        else if (sparam == "CloseButton")
        {
            ExpertRemove();
        }
    }
    {
        if (sparam == "LoginButton")
        {
            OnLoginButtonClick();
        }
        else if (sparam == "CloseAuthButton")
        {
            OnCloseAuthButtonClick();
        }
        else if (sparam == "TwoFALoginButton")
        {
            OnTwoFALoginButtonClick();
        }
        else if (sparam == "Close2FAButton")
        {
            OnClose2FAButtonClick();
        }
    }

    switch (id)
    {
        case CHARTEVENT_OBJECT_CLICK:
            if (sparam == "SendButton") OnSendButtonClick();
            else if (sparam == "ClearButton") OnClearButtonClick();
            else if (sparam == "ChangeFontButton") OnChangeFontButtonClick();
            else if (sparam == "ToggleThemeButton") OnToggleThemeButtonClick();
            else if (sparam == "MinimizeButton") OnMinimizeButtonClick();
            else if (sparam == "MaximizeButton") OnMaximizeButtonClick();
            else if (sparam == "CloseButton") OnCloseButtonClick();
            else if (StringFind(sparam, "QuickMessageButton") != -1)
            {
                long index = StringToInteger(StringSubstr(sparam, 18));
                OnQuickMessageButtonClick(index - 1);
            }
            break;

        case CHARTEVENT_OBJECT_ENDEDIT:
            if (sparam == "InputBox") OnInputChange();
            break;
    }
}

//+------------------------------------------------------------------+
//| Handle login button click                                        |
//+------------------------------------------------------------------+
void OnLoginButtonClick()
{
    string enteredPassword = passwordInputBox.Text();
    if (enteredPassword == Password)
    {
        twoFACode = GenerateRandom6DigitCode();
        SendMessageToTelegram("A login attempt was made on the Admin Panel. Please use this code to verify your identity: " + twoFACode, Hardcoded2FAChatId, Hardcoded2FABotToken);

        authentication.Destroy();
        ShowTwoFactorAuthPrompt();
        Print("Password authentication successful. A 2FA code has been sent to your Telegram.");
    }
    else
    {
        feedbackLabel.Text("Wrong password. Try again.");
        passwordInputBox.Text("");
    }
}

//+------------------------------------------------------------------+
//| Handle 2FA login button click                                    |
//+------------------------------------------------------------------+
void OnTwoFALoginButtonClick()
{
    // If 2FA is successful, show the trade management panel
    string enteredCode = twoFACodeInput.Text();
    if (enteredCode == twoFACode)
    {
        twoFactorAuth.Destroy();
        adminHomePanel.Show();
        Print("2FA authentication successful. Access granted to Trade Management Panel.");
    }
    else
    {
        twoFAFeedbackLabel.Text("Wrong code. Try again.");
        twoFACodeInput.Text("");
    }
}

//+------------------------------------------------------------------+
//| Handle close button for authentication                           |
//+------------------------------------------------------------------+
void OnCloseAuthButtonClick()
{
    authentication.Destroy();
    ExpertRemove(); // Exit the expert
    Print("Authentication dialog closed.");
}

//+------------------------------------------------------------------+
//| Handle close button for 2FA                                      |
//+------------------------------------------------------------------+
void OnClose2FAButtonClick()
{
    twoFactorAuth.Destroy();
    ExpertRemove();
    Print("2FA dialog closed.");
}

//+------------------------------------------------------------------+
//| Create necessary UI controls                                     |
//+------------------------------------------------------------------+
bool CreateControls()
{
    long chart_id = ChartID();

    if (!inputBox.Create(chart_id, "InputBox", 0, 5, 25, 460, 95))
    {
        Print("Failed to create input box");
        return false;
    }
    communicationsPanel.Add(inputBox);

    if (!charCounter.Create(chart_id, "CharCounter", 0, 380, 5, 460, 25))
    {
        Print("Failed to create character counter");
        return false;
    }
    charCounter.Text("0/" + IntegerToString(MAX_MESSAGE_LENGTH));
    communicationsPanel.Add(charCounter);

    if (!clearButton.Create(chart_id, "ClearButton", 0, 235, 95, 345, 125))
    {
        Print("Failed to create clear button");
        return false;
    }
    clearButton.Text("Clear");
    communicationsPanel.Add(clearButton);

    if (!sendButton.Create(chart_id, "SendButton", 0, 350, 95, 460, 125))
    {
        Print("Failed to create send button");
        return false;
    }
    sendButton.Text("Send");
    communicationsPanel.Add(sendButton);

    if (!changeFontButton.Create(chart_id, "ChangeFontButton", 0, 95, 95, 230, 115))
    {
        Print("Failed to create change font button");
        return false;
    }
    changeFontButton.Text("Font<>");
    communicationsPanel.Add(changeFontButton);

    if (!toggleThemeButton.Create(chart_id, "ToggleThemeButton", 0, 5, 95, 90, 115))
    {
        Print("Failed to create toggle theme button");
        return false;
    }
    toggleThemeButton.Text("Theme<>");
    communicationsPanel.Add(toggleThemeButton);

    if (!minimizeButton.Create(chart_id, "MinimizeButton", 0, 375, -22, 405, 0))
    {
        Print("Failed to create minimize button");
        return false;
    }
    minimizeButton.Text("_");
    communicationsPanel.Add(minimizeButton);

    if (!maximizeButton.Create(chart_id, "MaximizeButton", 0, 405, -22, 435, 0))
    {
        Print("Failed to create maximize button");
        return false;
    }
    maximizeButton.Text("[ ]");
    communicationsPanel.Add(maximizeButton);

    if (!closeButton.Create(chart_id, "CloseButton", 0, 435, -22, 465, 0))
    {
        Print("Failed to create close button");
        return false;
    }
    closeButton.Text("X");
    communicationsPanel.Add(closeButton);

    return CreateQuickMessageButtons();
}

//+------------------------------------------------------------------+
//| Create quick message buttons                                     |
//+------------------------------------------------------------------+
bool CreateQuickMessageButtons()
{
    string quickMessages[] = { QuickMessage1, QuickMessage2, QuickMessage3, QuickMessage4, QuickMessage5, QuickMessage6, QuickMessage7, QuickMessage8 };
    int startX = 5, startY = 160, width = 222, height = 65, spacing = 5;

    for (int i = 0; i < ArraySize(quickMessages); i++)
    {
        bool created = quickMessageButtons[i].Create(ChartID(), "QuickMessageButton" + IntegerToString(i + 1), 0,
            startX + (i % 2) * (width + spacing), startY + (i / 2) * (height + spacing),
            startX + (i % 2) * (width + spacing) + width, startY + (i / 2) * (height + spacing) + height);

        if (!created)
        {
            Print("Failed to create quick message button ", i + 1);
            return false;
        }
        quickMessageButtons[i].Text(quickMessages[i]);
        communicationsPanel.Add(quickMessageButtons[i]);
    }
    return true;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    communicationsPanel.Destroy();
    Print("Deinitialization complete");
}

//+------------------------------------------------------------------+
//| Handle custom message send button click                          |
//+------------------------------------------------------------------+
void OnSendButtonClick()
{
    string message = inputBox.Text();
    if (StringLen(message) > 0)
    {
        if (SendMessageToTelegram(message, InputChatId, InputBotToken))
            Print("Custom message sent: ", message);
        else
            Print("Failed to send custom message.");
    }
    else
    {
        Print("No message entered.");
    }
}

//+------------------------------------------------------------------+
//| Handle clear button click                                        |
//+------------------------------------------------------------------+
void OnClearButtonClick()
{
    inputBox.Text("");
    OnInputChange();
    Print("Input box cleared.");
}

//+------------------------------------------------------------------+
//| Handle quick message button click                                |
//+------------------------------------------------------------------+
void OnQuickMessageButtonClick(long index)
{
    string quickMessages[] = { QuickMessage1, QuickMessage2, QuickMessage3, QuickMessage4, QuickMessage5, QuickMessage6, QuickMessage7, QuickMessage8 };
    string message = quickMessages[(int)index];

    if (SendMessageToTelegram(message, InputChatId, InputBotToken))
        Print("Quick message sent: ", message);
    else
        Print("Failed to send quick message.");
}

//+------------------------------------------------------------------+
//| Update character counter                                         |
//+------------------------------------------------------------------+
void OnInputChange()
{
    int currentLength = StringLen(inputBox.Text());
    charCounter.Text(IntegerToString(currentLength) + "/" + IntegerToString(MAX_MESSAGE_LENGTH));
    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Handle toggle theme button click                                 |
//+------------------------------------------------------------------+
void OnToggleThemeButtonClick()
{
    darkTheme = !darkTheme;
    UpdateThemeColors();
    Print("Theme toggled: ", darkTheme ? "Dark" : "Light");
}

//+------------------------------------------------------------------+
//| Update theme colors for the panel                                |
//+------------------------------------------------------------------+
void UpdateThemeColors()
{
    color textColor = darkTheme ? clrWhite : clrBlack;
    color buttonBgColor = darkTheme ? clrDarkSlateGray : clrGainsboro;
    color borderColor = darkTheme ? clrSlateGray : clrGray;
    color bgColor = darkTheme ? clrDarkBlue : clrWhite;


    UpdateButtonTheme(clearButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(sendButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(toggleThemeButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(changeFontButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(minimizeButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(maximizeButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(closeButton, textColor, buttonBgColor, borderColor);

    for (int i = 0; i < ArraySize(quickMessageButtons); i++)
    {
        UpdateButtonTheme(quickMessageButtons[i], textColor, buttonBgColor, borderColor);
    }

    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Apply theme settings to a button                                 |
//+------------------------------------------------------------------+
void UpdateButtonTheme(CButton &button, color textColor, color bgColor, color borderColor)
{
    button.SetTextColor(textColor);
    button.SetBackgroundColor(bgColor);
    button.SetBorderColor(borderColor);
}

//+------------------------------------------------------------------+
//| Handle change font button click                                  |
//+------------------------------------------------------------------+
void OnChangeFontButtonClick()
{
    currentFontIndex = (currentFontIndex + 1) % ArraySize(availableFonts);
    SetFontForAll(availableFonts[currentFontIndex]);
    Print("Font changed to: ", availableFonts[currentFontIndex]);
    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Set font for all input boxes and buttons                         |
//+------------------------------------------------------------------+
void SetFontForAll(string fontName)
{
    inputBox.Font(fontName);
    clearButton.Font(fontName);
    sendButton.Font(fontName);
    toggleThemeButton.Font(fontName);
    changeFontButton.Font(fontName);
    minimizeButton.Font(fontName);
    maximizeButton.Font(fontName);
    closeButton.Font(fontName);

    for (int i = 0; i < ArraySize(quickMessageButtons); i++)
    {
        quickMessageButtons[i].Font(fontName);
    }
}

//+------------------------------------------------------------------+
//| Generate a random 6-digit code for 2FA                           |
//+------------------------------------------------------------------+
string GenerateRandom6DigitCode()
{
    int code = MathRand() % 1000000; // Produces a 6-digit number
    return StringFormat("%06d", code); // Ensures leading zeros
}

//+------------------------------------------------------------------+
//| Handle minimize button click                                     |
//+------------------------------------------------------------------+
void OnMinimizeButtonClick()
{
    minimized = true;
    communicationsPanel.Hide();
    minimizeButton.Hide();
    maximizeButton.Show();
    closeButton.Show();
    Print("Panel minimized.");
}

//+------------------------------------------------------------------+
//| Handle maximize button click                                     |
//+------------------------------------------------------------------+
void OnMaximizeButtonClick()
{
    if (minimized)
    {
        communicationsPanel.Show();
        minimizeButton.Show();
        maximizeButton.Hide();
        closeButton.Hide();
        minimized = false;
        Print("Panel maximized.");
    }
}

//+------------------------------------------------------------------+
//| Handle close button click for admin panel                        |
//+------------------------------------------------------------------+
void OnCloseButtonClick()
{
    ExpertRemove();
    Print("Admin panel closed.");
}

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

This code compilation resulted in the initial testing of the program. Next, I will share an image of the outcome, followed by some feedback and critiques below.

### **Initial Testing**

Here is the initial test of the newly integrated features in the Admin Panel:

![Initial Testing of the Admin Panel](https://c.mql5.com/2/126/terminal64_U8GZ2g7JKQ__1.gif)

Initial Testing of the Admin Panel with multiple features

Overall, our initial launch has been promising, but we encountered an issue where we got stuck in a single window, unable to switch back to the Home Panel to access other features. The solution is clear, we need to add navigation buttons to seamlessly switch between windows and features. Next, we will incorporate the necessary buttons and briefly touch on key features of the Trade Management Panel before conducting our final test for today's development. As you can see, the Trade Management Panel is currently empty, with no features implemented yet.

### Inter-Panel Switching

This is a simple and yet important step. We will incorporate buttons in sub panels to simply switch back to the Home Panel. To add a "Home" button for easy navigation back to the main Admin Home Panel, we need to create a dedicated button on each sub-panel (thus, Communications and Trade Management). These buttons will be recognized within the _OnChartEvent_ function, allowing them to trigger the main panel’s visibility while hiding the sub-panel when clicked. This approach involved extending the control creation function in each panel setup to include a "Home" button and updating the _OnChartEvent_ handler to manage the visibility of panels based on user interaction. Follow the steps below to easily understand.

Step 1:  We define _homeButton_ for Each Sub-panel.

```
CButton homeButtonComm, homeButtonTrade; // Home buttons for each sub-panel
```

Step 2: We create the Home Button in _CreateControls_ for Each Panel.

In our _CreateControls_ function, we will add the following code to create and position the homeButton:

```
bool CreateControls()
{
    // Create Home Button for Communications Panel
    if (!homeButtonComm.Create(ChartID(), "HomeButtonComm", 0, 20, 400, 120, 420))
    {
        Print("Failed to create Home button for Communications Panel");
        return false;
    }
    homeButtonComm.Text("Home");
    communicationsPanel.Add(homeButtonComm);

    // Create Home Button for Trade Management Panel
    if (!homeButtonTrade.Create(ChartID(), "HomeButtonTrade", 0, 20, 400, 120, 420))
    {
        Print("Failed to create Home button for Trade Management Panel");
        return false;
    }
    homeButtonTrade.Text("Home");
    tradeManagementPanel.Add(homeButtonTrade);

    return true;
}
```

Step 3: Update _OnChartEvent_ to Handle the Home Button Clicks.

Here we will add cases to handle clicks on each _homeButton_ and switch back to the _adminHomePanel_ when they are clicked.

```
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
    if (id == CHARTEVENT_OBJECT_CLICK)
    {
        if (sparam == "HomeButtonComm")
        {
            adminHomePanel.Show();
            communicationsPanel.Hide();
        }
        else if (sparam == "HomeButtonTrade")
        {
            adminHomePanel.Show();
            tradeManagementPanel.Hide();
        }
        else if (sparam == "TradeMgmtAccessButton")
        {
           tradeManagementPanel.Show();
           adminHomePanel.Hide();
        }
        else if (sparam == "CommunicationsPanelAccessButton")
        {
            communicationsPanel.Show();
            adminHomePanel.Hide();
        }
        else if (sparam == "MinimizeButton")
        {
            OnMinimizeButtonClick();
        }
        else if (sparam == "MaximizeButton")
        {
            OnMaximizeButtonClick();
        }
        else if (sparam == "CloseButton")
        {
            ExpertRemove();
        }
    }
}
```

There is still a lot of work to be done on the Trade Management Panel to add more trade-related functionality, aside from just the navigation buttons. For today, let's focus on adding some unique buttons for position management. We will later integrate the logic and necessary classes that will respond to these button clicks.

To start, we will first declare our global buttons.

```
CButton buyButton, sellButton, closePosButton, modifyPosButton, setSLButton, setTPButton;/// Position management buttons at global level declaration
```

Next, we will create buttons specifically for trade and position management.

```
// Create the Trade Management Panel controls
bool CreateTradeManagementControls()
{
    long chart_id = ChartID();

    // Buy Button
    if (!buyButton.Create(chart_id, "BuyButton", 0, 50, 50, 150, 90))
    {
        Print("Failed to create Buy button");
        return false;
    }
    buyButton.Text("Buy");
    tradeManagementPanel.Add(buyButton);

    // Sell Button
    if (!sellButton.Create(chart_id, "SellButton", 0, 160, 50, 260, 90))
    {
        Print("Failed to create Sell button");
        return false;
    }
    sellButton.Text("Sell");
    tradeManagementPanel.Add(sellButton);

    // Close Position Button
    if (!closePosButton.Create(chart_id, "ClosePosButton", 0, 50, 100, 190, 140))
    {
        Print("Failed to create Close Position button");
        return false;
    }
    closePosButton.Text("Close Position");
    tradeManagementPanel.Add(closePosButton);

    // Modify Position Button
    if (!modifyPosButton.Create(chart_id, "ModifyPosButton", 0, 200, 100, 340, 140))
    {
        Print("Failed to create Modify Position button");
        return false;
    }
    modifyPosButton.Text("Modify Position");
    tradeManagementPanel.Add(modifyPosButton);

    // Set Stop-Loss Button
    if (!setSLButton.Create(chart_id, "SetSLButton", 0, 50, 150, 150, 190))
    {
        Print("Failed to create Set Stop-Loss button");
        return false;
    }
    setSLButton.Text("Set SL");
    tradeManagementPanel.Add(setSLButton);

    // Set Take-Profit Button
    if (!setTPButton.Create(chart_id, "SetTPButton", 0, 160, 150, 260, 190))
    {
        Print("Failed to create Set Take-Profit button");
        return false;
    }
    setTPButton.Text("Set TP");
    tradeManagementPanel.Add(setTPButton);

    return true;
}
```

Finally, here's the complete source code after integrating all the features together.

```
//+------------------------------------------------------------------+
//|                                             Admin Panel.mq5      |
//|                           Copyright 2024, Clemence Benjamin      |
//|        https://www.mql5.com/en/users/billionaire2024/seller      |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
#property description "A secure and responsive Admin Panel. Send messages to your telegram clients without leaving MT5"
#property version   "1.21"

#include <Trade\Trade.mqh>
#include <Controls\Dialog.mqh>
#include <Controls\Button.mqh>
#include <Controls\Edit.mqh>
#include <Controls\Label.mqh>

// Input parameters for quick messages
input string QuickMessage1 = "Updates";
input string QuickMessage2 = "Close all";
input string QuickMessage3 = "In deep profits";
input string QuickMessage4 = "Hold position";
input string QuickMessage5 = "Swing Entry";
input string QuickMessage6 = "Scalp Entry";
input string QuickMessage7 = "Book profit";
input string QuickMessage8 = "Invalid Signal";
input string InputChatId = "YOUR_CHAT_ID";
input string InputBotToken = "YOUR_BOT_TOKEN";

// Constants for 2FA
const string Hardcoded2FAChatId = "7049213628";
const string Hardcoded2FABotToken = "7491148147:AAHjzHVL1S74RG0Ib-pN2bgG7wEKD2Rd2MU";

// Global variables
CDialog adminHomePanel, tradeManagementPanel, communicationsPanel;
CDialog authentication, twoFactorAuth;
CButton homeButtonComm, homeButtonTrade;
CButton buyButton, sellButton, closePosButton, modifyPosButton, setSLButton, setTPButton;/// Position management buttons at global level declaration.
CButton sendButton, clearButton, changeFontButton, toggleThemeButton;
CButton loginButton, closeAuthButton, twoFALoginButton, close2FAButton;
CButton quickMessageButtons[8], minimizeButton, maximizeButton, closeButton;
CButton tradeMgmtAccessButton, communicationsPanelAccessButton;
CEdit inputBox, passwordInputBox, twoFACodeInput;
CLabel charCounter, passwordPromptLabel, feedbackLabel, twoFAPromptLabel, twoFAFeedbackLabel;
bool minimized = false;
bool darkTheme = false;
int MAX_MESSAGE_LENGTH = 4096;
string availableFonts[] = { "Arial", "Courier New", "Verdana", "Times New Roman" };
int currentFontIndex = 0;
string Password = "2024"; // Hardcoded password
string twoFACode = "";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    if (!ShowAuthenticationPrompt())
    {
        Print("Authorization failed. Exiting...");
        return INIT_FAILED;
    }

    if (!adminHomePanel.Create(ChartID(), "Admin Home Panel", 0, 30, 30, 500, 500))
    {
        Print("Failed to create Admin Home Panel");
        return INIT_FAILED;
    }

    if (!CreateAdminHomeControls())
    {
        Print("Home panel control creation failed");
        return INIT_FAILED;
    }
    if (!communicationsPanel.Create(ChartID(), "Communications Panel", 0, 30, 30, 500, 500))
    {
        Print("Failed to create Communications panel dialog");
        return INIT_FAILED;
    }
    if (!tradeManagementPanel.Create(ChartID(), "Trade Management Panel", 0, 30, 30, 500, 500))
    {
        Print("Failed to create Communictions panel dialog");
        return INIT_FAILED;
    }

    if (!CreateControls())
    {
        Print("Control creation failed");
        return INIT_FAILED;
    }
    if (!CreateTradeManagementControls())
    {
        Print("Control creation failed");
        return INIT_FAILED;
    }


    adminHomePanel.Hide(); // Hide home panel by default on initialization
    communicationsPanel.Hide(); // Hide the Communications Panel
    tradeManagementPanel.Hide();// Hide the Trade Management Panel
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Show authentication input dialog                                 |
//+------------------------------------------------------------------+
bool ShowAuthenticationPrompt()
{
    if (!authentication.Create(ChartID(), "Authentication", 0, 100, 100, 500, 300))
    {
        Print("Failed to create authentication dialog");
        return false;
    }

    if (!passwordInputBox.Create(ChartID(), "PasswordInputBox", 0, 20, 70, 260, 95))
    {
        Print("Failed to create password input box");
        return false;
    }
    authentication.Add(passwordInputBox);

    if (!passwordPromptLabel.Create(ChartID(), "PasswordPromptLabel", 0, 20, 20, 260, 40))
    {
        Print("Failed to create password prompt label");
        return false;
    }
    passwordPromptLabel.Text("Enter password: Access Admin Panel");
    authentication.Add(passwordPromptLabel);

    if (!feedbackLabel.Create(ChartID(), "FeedbackLabel", 0, 20, 140, 380, 160))
    {
        Print("Failed to create feedback label");
        return false;
    }
    feedbackLabel.Text("");
    feedbackLabel.Color(clrRed); // Red color for incorrect attempts
    authentication.Add(feedbackLabel);

    if (!loginButton.Create(ChartID(), "LoginButton", 0, 20, 120, 100, 140))
    {
        Print("Failed to create login button");
        return false;
    }
    loginButton.Text("Login");
    authentication.Add(loginButton);

    if (!closeAuthButton.Create(ChartID(), "CloseAuthButton", 0, 120, 120, 200, 140))
    {
        Print("Failed to create close button for authentication");
        return false;
    }
    closeAuthButton.Text("Close");
    authentication.Add(closeAuthButton);

    authentication.Show();
    ChartRedraw();
    return true;
}

//+------------------------------------------------------------------+
//| Show two-factor authentication input dialog                      |
//+------------------------------------------------------------------+
void ShowTwoFactorAuthPrompt()
{
    if (!twoFactorAuth.Create(ChartID(), "Two-Factor Authentication", 0, 100, 100, 500, 300))
    {
        Print("Failed to create 2FA dialog");
        return;
    }

    if (!twoFACodeInput.Create(ChartID(), "TwoFACodeInput", 0, 20, 70, 260, 95))
    {
        Print("Failed to create 2FA code input box");
        return;
    }
    twoFactorAuth.Add(twoFACodeInput);

    if (!twoFAPromptLabel.Create(ChartID(), "TwoFAPromptLabel", 0, 20, 20, 380, 40))
    {
        Print("Failed to create 2FA prompt label");
        return;
    }
    twoFAPromptLabel.Text("Enter the 2FA code sent to your Telegram:");
    twoFactorAuth.Add(twoFAPromptLabel);

    if (!twoFAFeedbackLabel.Create(ChartID(), "TwoFAFeedbackLabel", 0, 20, 140, 380, 160))
    {
        Print("Failed to create 2FA feedback label");
        return;
    }
    twoFAFeedbackLabel.Text("");
    twoFAFeedbackLabel.Color(clrRed); // Red color for incorrect 2FA attempts
    twoFactorAuth.Add(twoFAFeedbackLabel);

    if (!twoFALoginButton.Create(ChartID(), "TwoFALoginButton", 0, 20, 120, 100, 140))
    {
        Print("Failed to create 2FA login button");
        return;
    }
    twoFALoginButton.Text("Verify");
    twoFactorAuth.Add(twoFALoginButton);

    if (!close2FAButton.Create(ChartID(), "Close2FAButton", 0, 120, 120, 200, 140))
    {
        Print("Failed to create close button for 2FA");
        return;
    }
    close2FAButton.Text("Close");
    twoFactorAuth.Add(close2FAButton);

    twoFactorAuth.Show();
    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Admin Home Panel controls creation                               |
//+------------------------------------------------------------------+
bool CreateAdminHomeControls()
{
    long chart_id = ChartID();

    if (!tradeMgmtAccessButton.Create(chart_id, "TradeMgmtAccessButton", 0, 50, 50, 250, 90))
    {
        Print("Failed to create Trade Management Access button");
        return false;
    }
    tradeMgmtAccessButton.Text("Trade Management Panel");
    adminHomePanel.Add(tradeMgmtAccessButton);

    if (!communicationsPanelAccessButton.Create(chart_id, "CommunicationsPanelAccessButton", 0, 50, 100, 250, 140))
    {
        Print("Failed to create Communications Panel Access button");
        return false;
    }
    communicationsPanelAccessButton.Text("Communications Panel");
    adminHomePanel.Add(communicationsPanelAccessButton);

    if (!minimizeButton.Create(chart_id, "MinimizeButton", 0, 375, -22, 405, 0))
    {
        Print("Failed to create minimize button");
        return false;
    }
    minimizeButton.Text("_");
    adminHomePanel.Add(minimizeButton);

    if (!maximizeButton.Create(chart_id, "MaximizeButton", 0, 405, -22, 435, 0))
    {
        Print("Failed to create maximize button");
        return false;
    }
    maximizeButton.Text("[ ]");
    adminHomePanel.Add(maximizeButton);

    if (!closeButton.Create(chart_id, "CloseButton", 0, 435, -22, 465, 0))
    {
        Print("Failed to create close button");
        return false;
    }
    closeButton.Text("X");
    adminHomePanel.Add(closeButton);

    return true;
}
// Create the Trade Management Panel controls
bool CreateTradeManagementControls()
{
    long chart_id = ChartID();

    // Buy Button
    if (!buyButton.Create(chart_id, "BuyButton", 0, 50, 50, 150, 90))
    {
        Print("Failed to create Buy button");
        return false;
    }
    buyButton.Text("Buy");
    tradeManagementPanel.Add(buyButton);

    // Sell Button
    if (!sellButton.Create(chart_id, "SellButton", 0, 160, 50, 260, 90))
    {
        Print("Failed to create Sell button");
        return false;
    }
    sellButton.Text("Sell");
    tradeManagementPanel.Add(sellButton);

    // Close Position Button
    if (!closePosButton.Create(chart_id, "ClosePosButton", 0, 50, 100, 190, 140))
    {
        Print("Failed to create Close Position button");
        return false;
    }
    closePosButton.Text("Close Position");
    tradeManagementPanel.Add(closePosButton);

    // Modify Position Button
    if (!modifyPosButton.Create(chart_id, "ModifyPosButton", 0, 200, 100, 340, 140))
    {
        Print("Failed to create Modify Position button");
        return false;
    }
    modifyPosButton.Text("Modify Position");
    tradeManagementPanel.Add(modifyPosButton);

    // Set Stop-Loss Button
    if (!setSLButton.Create(chart_id, "SetSLButton", 0, 50, 150, 150, 190))
    {
        Print("Failed to create Set Stop-Loss button");
        return false;
    }
    setSLButton.Text("Set SL");
    tradeManagementPanel.Add(setSLButton);

    // Set Take-Profit Button
    if (!setTPButton.Create(chart_id, "SetTPButton", 0, 160, 150, 260, 190))
    {
        Print("Failed to create Set Take-Profit button");
        return false;
    }
    setTPButton.Text("Set TP");
    tradeManagementPanel.Add(setTPButton);

    return true;
}

//+------------------------------------------------------------------+
//| Handle chart events                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
    if (id == CHARTEVENT_OBJECT_CLICK)
    {
    if (sparam == "HomeButtonComm")
        {
            adminHomePanel.Show();
            communicationsPanel.Hide();
        }
        else if (sparam == "HomeButtonTrade")
        {
            adminHomePanel.Show();
            tradeManagementPanel.Hide();
        }

        if (sparam == "TradeMgmtAccessButton")
        {
           tradeManagementPanel.Show();
           adminHomePanel.Hide();
        }
        else if (sparam == "CommunicationsPanelAccessButton")
        {
            communicationsPanel.Show();
            adminHomePanel.Hide();
        }
        else if (sparam == "MinimizeButton")
        {
            OnMinimizeButtonClick();
        }
        else if (sparam == "MaximizeButton")
        {
            OnMaximizeButtonClick();
        }
        else if (sparam == "CloseButton")
        {
            ExpertRemove();
        }
    }
    {
        if (sparam == "LoginButton")
        {
            OnLoginButtonClick();
        }
        else if (sparam == "CloseAuthButton")
        {
            OnCloseAuthButtonClick();
        }
        else if (sparam == "TwoFALoginButton")
        {
            OnTwoFALoginButtonClick();
        }
        else if (sparam == "Close2FAButton")
        {
            OnClose2FAButtonClick();
        }
    }

    switch (id)
    {
        case CHARTEVENT_OBJECT_CLICK:
            if (sparam == "SendButton") OnSendButtonClick();
            else if (sparam == "ClearButton") OnClearButtonClick();
            else if (sparam == "ChangeFontButton") OnChangeFontButtonClick();
            else if (sparam == "ToggleThemeButton") OnToggleThemeButtonClick();
            else if (sparam == "MinimizeButton") OnMinimizeButtonClick();
            else if (sparam == "MaximizeButton") OnMaximizeButtonClick();
            else if (sparam == "CloseButton") OnCloseButtonClick();
            else if (StringFind(sparam, "QuickMessageButton") != -1)
            {
                long index = StringToInteger(StringSubstr(sparam, 18));
                OnQuickMessageButtonClick(index - 1);
            }
            break;

        case CHARTEVENT_OBJECT_ENDEDIT:
            if (sparam == "InputBox") OnInputChange();
            break;
    }
}

//+------------------------------------------------------------------+
//| Handle login button click                                        |
//+------------------------------------------------------------------+
void OnLoginButtonClick()
{
    string enteredPassword = passwordInputBox.Text();
    if (enteredPassword == Password)
    {
        twoFACode = GenerateRandom6DigitCode();
        SendMessageToTelegram("A login attempt was made on the Admin Panel. Please use this code to verify your identity: " + twoFACode, Hardcoded2FAChatId, Hardcoded2FABotToken);

        authentication.Destroy();
        ShowTwoFactorAuthPrompt();
        Print("Password authentication successful. A 2FA code has been sent to your Telegram.");
    }
    else
    {
        feedbackLabel.Text("Wrong password. Try again.");
        passwordInputBox.Text("");
    }
}

//+------------------------------------------------------------------+
//| Handle 2FA login button click                                    |
//+------------------------------------------------------------------+
void OnTwoFALoginButtonClick()
{
    // If 2FA is successful, show the trade management panel
    string enteredCode = twoFACodeInput.Text();
    if (enteredCode == twoFACode)
    {
        twoFactorAuth.Destroy();
        adminHomePanel.Show();
        Print("2FA authentication successful. Access granted to Trade Management Panel.");
    }
    else
    {
        twoFAFeedbackLabel.Text("Wrong code. Try again.");
        twoFACodeInput.Text("");
    }
}

//+------------------------------------------------------------------+
//| Handle close button for authentication                           |
//+------------------------------------------------------------------+
void OnCloseAuthButtonClick()
{
    authentication.Destroy();
    ExpertRemove(); // Exit the expert
    Print("Authentication dialog closed.");
}

//+------------------------------------------------------------------+
//| Handle close button for 2FA                                      |
//+------------------------------------------------------------------+
void OnClose2FAButtonClick()
{
    twoFactorAuth.Destroy();
    ExpertRemove();
    Print("2FA dialog closed.");
}

//+------------------------------------------------------------------+
//| Create necessary UI controls                                     |
//+------------------------------------------------------------------+
bool CreateControls()
{
    long chart_id = ChartID();


    if (!inputBox.Create(chart_id, "InputBox", 0, 5, 25, 460, 95))
    {
        Print("Failed to create input box");
        return false;
    }
    communicationsPanel.Add(inputBox);

    // Create Home Button for Communications Panel
    if (!homeButtonComm.Create(chart_id,  "HomeButtonComm", 0, 20, 120, 120,150))
    {
        Print("Failed to create Home button for Communications Panel");
        return false;
    }
    homeButtonComm.Text("Home 🏠");
    communicationsPanel.Add(homeButtonComm);

    // Create Home Button for Trade Management Panel
    if (!homeButtonTrade.Create(chart_id, "HomeButtonTrade", 0, 20, 10, 120, 30))
    {
        Print("Failed to create Home button for Trade Management Panel");
        return false;
    }
    homeButtonTrade.Text("Home 🏠");
    tradeManagementPanel.Add(homeButtonTrade);

    if (!charCounter.Create(chart_id, "CharCounter", 0, 380, 5, 460, 25))
    {
        Print("Failed to create character counter");
        return false;
    }
    charCounter.Text("0/" + IntegerToString(MAX_MESSAGE_LENGTH));
    communicationsPanel.Add(charCounter);

    if (!clearButton.Create(chart_id, "ClearButton", 0, 235, 95, 345, 125))
    {
        Print("Failed to create clear button");
        return false;
    }
    clearButton.Text("Clear");
    communicationsPanel.Add(clearButton);

    if (!sendButton.Create(chart_id, "SendButton", 0, 350, 95, 460, 125))
    {
        Print("Failed to create send button");
        return false;
    }
    sendButton.Text("Send");
    communicationsPanel.Add(sendButton);

    if (!changeFontButton.Create(chart_id, "ChangeFontButton", 0, 95, 95, 230, 115))
    {
        Print("Failed to create change font button");
        return false;
    }
    changeFontButton.Text("Font<>");
    communicationsPanel.Add(changeFontButton);

    if (!toggleThemeButton.Create(chart_id, "ToggleThemeButton", 0, 5, 95, 90, 115))
    {
        Print("Failed to create toggle theme button");
        return false;
    }
    toggleThemeButton.Text("Theme<>");
    communicationsPanel.Add(toggleThemeButton);

    if (!minimizeButton.Create(chart_id, "MinimizeButton", 0, 375, -22, 405, 0))
    {
        Print("Failed to create minimize button");
        return false;
    }
    minimizeButton.Text("_");
    communicationsPanel.Add(minimizeButton);

    if (!maximizeButton.Create(chart_id, "MaximizeButton", 0, 405, -22, 435, 0))
    {
        Print("Failed to create maximize button");
        return false;
    }
    maximizeButton.Text("[ ]");
    communicationsPanel.Add(maximizeButton);

    if (!closeButton.Create(chart_id, "CloseButton", 0, 435, -22, 465, 0))
    {
        Print("Failed to create close button");
        return false;
    }
    closeButton.Text("X");
    communicationsPanel.Add(closeButton);

    return CreateQuickMessageButtons();

}

//+------------------------------------------------------------------+
//| Create quick message buttons                                     |
//+------------------------------------------------------------------+
bool CreateQuickMessageButtons()
{
    string quickMessages[] = { QuickMessage1, QuickMessage2, QuickMessage3, QuickMessage4, QuickMessage5, QuickMessage6, QuickMessage7, QuickMessage8 };
    int startX = 5, startY = 160, width = 222, height = 65, spacing = 5;

    for (int i = 0; i < ArraySize(quickMessages); i++)
    {
        bool created = quickMessageButtons[i].Create(ChartID(), "QuickMessageButton" + IntegerToString(i + 1), 0,
            startX + (i % 2) * (width + spacing), startY + (i / 2) * (height + spacing),
            startX + (i % 2) * (width + spacing) + width, startY + (i / 2) * (height + spacing) + height);

        if (!created)
        {
            Print("Failed to create quick message button ", i + 1);
            return false;
        }
        quickMessageButtons[i].Text(quickMessages[i]);
        communicationsPanel.Add(quickMessageButtons[i]);
    }

    return true;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    communicationsPanel.Destroy();
    Print("Deinitialization complete");
}

//+------------------------------------------------------------------+
//| Handle custom message send button click                          |
//+------------------------------------------------------------------+
void OnSendButtonClick()
{
    string message = inputBox.Text();
    if (StringLen(message) > 0)
    {
        if (SendMessageToTelegram(message, InputChatId, InputBotToken))
            Print("Custom message sent: ", message);
        else
            Print("Failed to send custom message.");
    }
    else
    {
        Print("No message entered.");
    }
}

//+------------------------------------------------------------------+
//| Handle clear button click                                        |
//+------------------------------------------------------------------+
void OnClearButtonClick()
{
    inputBox.Text("");
    OnInputChange();
    Print("Input box cleared.");
}

//+------------------------------------------------------------------+
//| Handle quick message button click                                |
//+------------------------------------------------------------------+
void OnQuickMessageButtonClick(long index)
{
    string quickMessages[] = { QuickMessage1, QuickMessage2, QuickMessage3, QuickMessage4, QuickMessage5, QuickMessage6, QuickMessage7, QuickMessage8 };
    string message = quickMessages[(int)index];

    if (SendMessageToTelegram(message, InputChatId, InputBotToken))
        Print("Quick message sent: ", message);
    else
        Print("Failed to send quick message.");
}

//+------------------------------------------------------------------+
//| Update character counter                                         |
//+------------------------------------------------------------------+
void OnInputChange()
{
    int currentLength = StringLen(inputBox.Text());
    charCounter.Text(IntegerToString(currentLength) + "/" + IntegerToString(MAX_MESSAGE_LENGTH));
    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Handle toggle theme button click                                 |
//+------------------------------------------------------------------+
void OnToggleThemeButtonClick()
{
    darkTheme = !darkTheme;
    UpdateThemeColors();
    Print("Theme toggled: ", darkTheme ? "Dark" : "Light");
}

//+------------------------------------------------------------------+
//| Update theme colors for the panel                                |
//+------------------------------------------------------------------+
void UpdateThemeColors()
{
    color textColor = darkTheme ? clrWhite : clrBlack;
    color buttonBgColor = darkTheme ? clrDarkSlateGray : clrGainsboro;
    color borderColor = darkTheme ? clrSlateGray : clrGray;
    color bgColor = darkTheme ? clrDarkBlue : clrWhite;


    UpdateButtonTheme(clearButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(sendButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(toggleThemeButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(changeFontButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(minimizeButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(maximizeButton, textColor, buttonBgColor, borderColor);
    UpdateButtonTheme(closeButton, textColor, buttonBgColor, borderColor);

    for (int i = 0; i < ArraySize(quickMessageButtons); i++)
    {
        UpdateButtonTheme(quickMessageButtons[i], textColor, buttonBgColor, borderColor);
    }

    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Apply theme settings to a button                                 |
//+------------------------------------------------------------------+
void UpdateButtonTheme(CButton &button, color textColor, color bgColor, color borderColor)
{
    button.SetTextColor(textColor);
    button.SetBackgroundColor(bgColor);
    button.SetBorderColor(borderColor);
}

//+------------------------------------------------------------------+
//| Handle change font button click                                  |
//+------------------------------------------------------------------+
void OnChangeFontButtonClick()
{
    currentFontIndex = (currentFontIndex + 1) % ArraySize(availableFonts);
    SetFontForAll(availableFonts[currentFontIndex]);
    Print("Font changed to: ", availableFonts[currentFontIndex]);
    ChartRedraw();
}

//+------------------------------------------------------------------+
//| Set font for all input boxes and buttons                         |
//+------------------------------------------------------------------+
void SetFontForAll(string fontName)
{
    inputBox.Font(fontName);
    clearButton.Font(fontName);
    sendButton.Font(fontName);
    toggleThemeButton.Font(fontName);
    changeFontButton.Font(fontName);
    minimizeButton.Font(fontName);
    maximizeButton.Font(fontName);
    closeButton.Font(fontName);

    for (int i = 0; i < ArraySize(quickMessageButtons); i++)
    {
        quickMessageButtons[i].Font(fontName);
    }
}

//+------------------------------------------------------------------+
//| Generate a random 6-digit code for 2FA                           |
//+------------------------------------------------------------------+
string GenerateRandom6DigitCode()
{
    int code = MathRand() % 1000000; // Produces a 6-digit number
    return StringFormat("%06d", code); // Ensures leading zeros
}

//+------------------------------------------------------------------+
//| Handle minimize button click                                     |
//+------------------------------------------------------------------+
void OnMinimizeButtonClick()
{
    minimized = true;
    communicationsPanel.Hide();
    minimizeButton.Hide();
    maximizeButton.Show();
    closeButton.Show();
    Print("Panel minimized.");
}

//+------------------------------------------------------------------+
//| Handle maximize button click                                     |
//+------------------------------------------------------------------+
void OnMaximizeButtonClick()
{
    if (minimized)
    {
        communicationsPanel.Show();
        minimizeButton.Show();
        maximizeButton.Hide();
        closeButton.Hide();
        minimized = false;
        Print("Panel maximized.");
    }
}

//+------------------------------------------------------------------+
//| Handle close button click for admin panel                        |
//+------------------------------------------------------------------+
void OnCloseButtonClick()
{
    ExpertRemove();
    Print("Admin panel closed.");
}

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

### Testing and Results

Indeed, it is entirely possible to add multiple functionalities to a single Expert Advisor program. We have significantly enhanced our Admin Panel Expert Advisor by incorporating new, exciting features that greatly expand its capabilities. Below is a screen capture showing the outcome after successfully compiling and running the updated code.

![Launching Admin Panel v1,21](https://c.mql5.com/2/126/ShareX_HmilLcYrxO__1.gif)

Launching Admin Panel V1.21 on MetaTrader 5

### Conclusion

In this project, we started with a simple Admin Panel focused on Telegram communication and expanded it into a more comprehensive, user-friendly interface. Now at Version 1.21, our setup includes an Admin Home Panel that serves as the main dashboard, making it easy to access the Communications Panel and the new Trade Management Panel. This approach keeps everything well-organized, secure with 2FA, and easy to navigate. Next, we’ll dive deeper into the Trade Management Panel’s features to keep enhancing trading and communication in one cohesive tool.

For the Home button icon, I simply copied an icon from Telegram and pasted it in. We'll explore more details about icons and design choices in future updates. You are welcome to share your thoughts in the comments section below.

[Back to contents](https://www.mql5.com/en/articles/16240#para0)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16240.zip "Download all attachments in the single ZIP archive")

[Admin\_Panel\_V1.21.mq5](https://www.mql5.com/en/articles/download/16240/admin_panel_v1.21.mq5 "Download Admin_Panel_V1.21.mq5")(56.4 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/476315)**

![Client in Connexus (Part 7): Adding the Client Layer](https://c.mql5.com/2/101/http60x60.png)[Client in Connexus (Part 7): Adding the Client Layer](https://www.mql5.com/en/articles/16324)

In this article we continue the development of the connexus library. In this chapter we build the CHttpClient class responsible for sending a request and receiving an order. We also cover the concept of mocks, leaving the library decoupled from the WebRequest function, which allows greater flexibility for users.

![MQL5 Wizard Techniques you should know (Part 47): Reinforcement Learning with Temporal Difference](https://c.mql5.com/2/101/MQL5_Wizard_Techniques_you_should_know_Part_47__LOGO.png)[MQL5 Wizard Techniques you should know (Part 47): Reinforcement Learning with Temporal Difference](https://www.mql5.com/en/articles/16303)

Temporal Difference is another algorithm in reinforcement learning that updates Q-Values basing on the difference between predicted and actual rewards during agent training. It specifically dwells on updating Q-Values without minding their state-action pairing. We therefore look to see how to apply this, as we have with previous articles, in a wizard assembled Expert Advisor.

![Developing a multi-currency Expert Advisor (Part 13): Automating the second stage — selection into groups](https://c.mql5.com/2/80/Developing_a_multi-currency_advisor_Part_13__LOGO.png)[Developing a multi-currency Expert Advisor (Part 13): Automating the second stage — selection into groups](https://www.mql5.com/en/articles/14892)

We have already implemented the first stage of the automated optimization. We perform optimization for different symbols and timeframes according to several criteria and store information about the results of each pass in the database. Now we are going to select the best groups of parameter sets from those found at the first stage.

![Price Action Analysis Toolkit Development (Part 1): Chart Projector](https://c.mql5.com/2/101/Price_Action_Analysis_Toolkit_Development_Part_1____LOGO__2.png)[Price Action Analysis Toolkit Development (Part 1): Chart Projector](https://www.mql5.com/en/articles/16014)

This project aims to leverage the MQL5 algorithm to develop a comprehensive set of analysis tools for MetaTrader 5. These tools—ranging from scripts and indicators to AI models and expert advisors—will automate the market analysis process. At times, this development will yield tools capable of performing advanced analyses with no human involvement and forecasting outcomes to appropriate platforms. No opportunity will ever be missed. Join me as we explore the process of building a robust market analysis custom tools' chest. We will begin by developing a simple MQL5 program that I have named, Chart Projector.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=qttftujndfdldpuyiwozcfkdbfmzlkkw&ssn=1769093997891405201&ssn_dr=0&ssn_sr=0&fv_date=1769093997&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16240&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20a%20Trading%20Administrator%20Panel%20in%20MQL5%20(Part%20VI)%3A%20Multiple%20Functions%20Interface%20(I)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909399707564946&fz_uniq=5049547248724979051&sv=2552)

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