---
title: Creating a Trading Administrator Panel in MQL5 (Part V): Two-Factor Authentication (2FA)
url: https://www.mql5.com/en/articles/16142
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:38:19.755224
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/16142&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062610275791709561)

MetaTrader 5 / Examples


### Contents:

- [Introduction](https://www.mql5.com/en/articles/16142#para1)
- [What Two-Factor Authentication (2FA) is](https://www.mql5.com/en/articles/16142#para2)
- [Implementing Two-Factor Authentication (2FA) in the Admin Panel Using MQL5](https://www.mql5.com/en/articles/16142#para4)

  - [Implementing GUI Elements for Password Input and 2FA Code Validation](https://www.mql5.com/en/articles/16142#para5)
  - [Verification Code Generation Algorithm](https://www.mql5.com/en/articles/16142#para6)
  - [Understanding MathRand() Function](https://www.mql5.com/en/articles/16142#para7)
  - [Telegram API for 2FA verification](https://www.mql5.com/en/articles/16142#para8)

- [Testing and Results](https://www.mql5.com/en/articles/16142#para9)
- [Conclusion](https://www.mql5.com/en/articles/16142#para10)

### Introduction

[Previously](https://www.mql5.com/en/articles/16079), we explored the implementation of passcode authentication within the Admin Panel, a crucial first step in safeguarding communications between administrators and traders. This basic form of security, while essential for initial protection, can potentially expose these systems to vulnerabilities due to its reliance on a single verification factor. Incorporating two-factor authentication (2FA) becomes a vital upgrade, providing a more robust security framework for our application.

This approach significantly mitigates the risk of unauthorized access, as it demands access to the secondary verification method along with the knowledge of the password. 2FA ensures that all communications originate from legitimate sources, thereby protecting against the potential consequences of compromised data, misinformation, and market manipulation.

By integrating 2FA into our Admin Panel, we can provide users with a higher level of trust and security. This dual authentication system acts as a powerful deterrent against potential breaches, giving both administrators and traders peace of mind as they operate in a dynamic financial landscape. Today, we will discuss a concept that I have successfully transformed into a working solution. This initiative has reinforced the security of our Admin Panel in response to the vulnerabilities associated with the passcode security implemented in the previous article of the series.

### What Two-Factor Authentication Is

Two-factor authentication (2FA) is a security mechanism that requires two different forms of verification before granting access to an account or system. It is a subset of multifactor authentication (MFA), which can involve two or more verification factors. The primary goal of 2FA is to add another layer of security beyond just a username and password, making it more difficult for unauthorized users to gain access.

According to some sources, 2FA typically involves two of the following three categories of authentication factors:

1. Something You Know (Knowledge Factor): This is typically a password or a PIN that the user must enter to access their account. It serves as the first line of defense.
2. Something You Have (Possession Factor): This involves a physical device or token that the user possesses, such as a security token, smart card, or mobile phone. Many systems use apps like Google Authenticator or Authy to generate one-time passwords (OTPs) that are valid for generally, a short period (usually 15 minutes).
3. Something You Are (Biometric Factor): This aspect may include fingerprint recognition, facial recognition, or other biometric data. While not always used in tandem with the first two factors for 2FA, biometrics can provide an additional layer of security.

### Implementing Two-Factor Authentication (2FA) in the Admin Panel Using MQL5

By integrating 2FA, we are moving beyond the traditional single-factor authentication (SFA) that typically relies solely on usernames and passwords. This shift is crucial because while passwords remain the most common form of initial security, they are inherently vulnerable to various types of attacks, including social engineering, brute-force, and dictionary attacks. The multifactor approach of 2FA effectively mitigates these risks by mandating that users provide two distinct forms of authentication that belong to different categories, thereby increasing confidence that access is truly granted to legitimate users.

To implement 2FA in our Admin Panel project, I leveraged the Dialog library, which allows us to create multiple window layers controlled by specific logic. At the beginning of this series, we integrated Telegram communication, primarily focused on transmitting messages from the Admin Panel to user channels or groups on Telegram. However, its potential extends beyond that; we aim to utilize it for OTP delivery as well.

In this project, we will adjust our code to generate a random six-digit code, which the program will securely store and subsequently send to the Admin's Telegram for verification purposes. I've noticed that many companies use Telegram for verification, albeit with approaches that differ from what we will implement here. For instance, in the image below, you can see the [MQL5 Verification Bot](https://www.mql5.com/go?link=https://t.me/mql5_verification_bot "https://t.me/mql5_verification_bot"), which serves as an example of such usage.

![MQL5 Verifcation Bot](https://c.mql5.com/2/98/photo_2024-10-21_12-36-25.jpg)

MQL5 Verification Bot

### Implementing GUI Elements for Password Input and 2FA Code Validation

When implementing the password input dialog, we begin by defining the _ShowAuthenticationPrompt()_ function, which encapsulates all the steps required to create a user interface for password entry. The process commences with the instantiation of the authentication dialog using the _Create()_ method, specifying its dimensions and position on the chart. For user input, we create a _passwordInputBox_ to securely capture the user's password. Following this, a _passwordPromptLabe_ l is added to provide clear instructions for the user, guiding them on the purpose of the dialog. To handle user feedback, especially for incorrect entries, a _feedbackLabel_ is implemented, where error messages will be displayed in red text, enhancing the user's ability to understand what went wrong. Next, we set up two buttons:

- a _loginButton_ that users can click to submit their password for authentication and
- A closeAuthButton that allows them to exit the dialog if they decide not to proceed.

Finally, we call s _how()_ on the authentication dialog to present it to the user and invoke _ChartRedraw()_ to ensure that all components render correctly on the screen. This systematic approach ensures a secure and user-friendly interface for password entry in the Admin Panel. See the next code snippet.

**Password Input Dialog Creation**

```
// Show authentication input dialog
bool ShowAuthenticationPrompt()
{
    if (!authentication.Create(ChartID(), "Authentication", 0, 100, 100, 500, 300))
    {
        Print("Failed to create authentication dialog");
        return false;
    }

    // Create password input box
    if (!passwordInputBox.Create(ChartID(), "PasswordInputBox", 0, 20, 70, 260, 95))
    {
        Print("Failed to create password input box");
        return false;
    }
    authentication.Add(passwordInputBox);

    // Create password prompt label
    if (!passwordPromptLabel.Create(ChartID(), "PasswordPromptLabel", 0, 20, 20, 260, 40))
    {
        Print("Failed to create password prompt label");
        return false;
    }
    passwordPromptLabel.Text("Enter password: Access Admin Panel");
    authentication.Add(passwordPromptLabel);

    // Create feedback label for wrong attempts
    if (!feedbackLabel.Create(ChartID(), "FeedbackLabel", 0, 20, 140, 380, 40))
    {
        Print("Failed to create feedback label");
        return false;
    }
    feedbackLabel.Text("");
    feedbackLabel.Color(clrRed); // Set color for feedback
    authentication.Add(feedbackLabel);

    // Create login button
    if (!loginButton.Create(ChartID(), "LoginButton", 0, 20, 120, 100, 40))
    {
        Print("Failed to create login button");
        return false;
    }
    loginButton.Text("Login");
    authentication.Add(loginButton);

    // Create close button for authentication dialog
    if (!closeAuthButton.Create(ChartID(), "CloseAuthButton", 0, 120, 120, 200, 40))
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
```

### 2FA Code Validation Dialog Creation

To create the 2FA code validation dialog, we define the _ShowTwoFactorAuthPrompt()_ function, which handles all the components necessary for verifying the user's two-factor authentication code.

We start by creating the _twoFactorAuth_ dialog, again using the _Create()_ method to set its properties. The first component added is _twoFACodeInput_, an input field designed to securely capture the 2FA code sent to the user's Telegram.

To guide the user, we implement a _twoFAPromptLabel_ that clearly instructs them to enter the 2FA code they received. Enhancing user experience further, we include a _twoFAFeedbackLabel_ to display real-time feedback; this label will show messages in red if the entered code does not match the expected value, thereby informing the user of incorrect entries.

For submission, a _twoFALoginButton_ is created, enabling the user to verify their code, while a _close2FAButton_ is provided for exiting the dialog if necessary. Once the components are set up, we invoke _Show()_ on the _twoFactorAuth_ dialog to make it visible to the user and call _ChartRedraw()_ to refresh the interface.

This structured approach ensures a secure method for validating 2FA codes and a streamlined user interaction in the Admin Panel. Below is a code snippet for better comprehension.

```
// Show two-factor authentication input dialog
void ShowTwoFactorAuthPrompt()
{
    if (!twoFactorAuth.Create(ChartID(), "Two-Factor Authentication", 0, 100, 100, 500, 300))
    {
        Print("Failed to create 2FA dialog");
        return;
    }

    // Create input box for 2FA code
    if (!twoFACodeInput.Create(ChartID(), "TwoFACodeInput", 0, 20, 70, 260, 95))
    {
        Print("Failed to create 2FA code input box");
        return;
    }
    twoFactorAuth.Add(twoFACodeInput);

    // Create prompt label for 2FA
    if (!twoFAPromptLabel.Create(ChartID(), "TwoFAPromptLabel", 0, 20, 20, 380, 40))
    {
        Print("Failed to create 2FA prompt label");
        return;
    }
    twoFAPromptLabel.Text("Enter the 2FA code sent to your Telegram:");
    twoFactorAuth.Add(twoFAPromptLabel);

    // Create feedback label for wrong attempts
    if (!twoFAFeedbackLabel.Create(ChartID(), "TwoFAFeedbackLabel", 0, 20, 140, 380, 40))
    {
        Print("Failed to create 2FA feedback label");
        return;
    }
    twoFAFeedbackLabel.Text("");
    twoFAFeedbackLabel.Color(clrRed); // Set color for feedback
    twoFactorAuth.Add(twoFAFeedbackLabel);

    // Create login button for 2FA code submission
    if (!twoFALoginButton.Create(ChartID(), "TwoFALoginButton", 0, 20, 120, 100, 40))
    {
        Print("Failed to create 2FA login button");
        return;
    }
    twoFALoginButton.Text("Verify");
    twoFactorAuth.Add(twoFALoginButton);

    // Create close button for 2FA dialog
    if (!close2FAButton.Create(ChartID(), "Close2FAButton", 0, 120, 120, 200, 40))
    {
        Print("Failed to create close button for 2FA");
        return;
    }
    close2FAButton.Text("Close");
    twoFactorAuth.Add(close2FAButton);

    twoFactorAuth.Show();
    ChartRedraw();
}
```

### Verification Code Generation Algorithm

After successfully entering the password for Admin Panel access, we established a second layer of protection involving OTP code generation. This code is securely stored and forwarded to a unique hard-coded chat ID linked to the Telegram app, either on mobile or desktop, where the legitimate owner can retrieve it for further entry in the prompt. If the entered code matches, the application grants access to the full features of the Admin Panel for operations and communication.

Regarding the code generation algorithm, I will explain the various components in the code snippets below:

The line of code, below, serves as an essential variable declaration for managing two-factor authentication (2FA) within our program.

```
string twoFACode = "";
```

This line initializes the variable _twoFACode_ as an empty string, which will be used to store a randomly generated 6-digit two-factor authentication code. Throughout the authentication process, this variable plays a critical role of holding the actual code that is sent to the user via Telegram after they successfully enter the correct password to access the Admin Panel.

When the user passes the initial password check, the _twoFACode_ variable is populated with a new value generated by the _GenerateRandom6DigitCode()_ function, which produces a 6-digit numerical string. This value is then sent to the user’s Telegram through the _SendMessageToTelegram()_ function.

Later on, when the user is prompted to enter their _2FA code_, the program compares the user's input against this stored value in _twoFACode_. If the user’s input matches the value held in _twoFACode_, access to the admin panel is granted; otherwise, an error message is displayed.

1\. Password Authentication:

```
string Password = "2024"; // Hardcoded password

// Handle login button click
void OnLoginButtonClick()
{
    string enteredPassword = passwordInputBox.Text();
    if (enteredPassword == Password)
    {
        twoFACode = GenerateRandom6DigitCode();
        SendMessageToTelegram("Your 2FA code is: " + twoFACode, Hardcoded2FAChatId, Hardcoded2FABotToken);
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
```

A hard-coded password ("2024") serves as the access control mechanism for the Admin Panel. When the user submits the entered password through the designated input box, the code checks if the input matches the hard-coded password. If it matches, a random 6-digit code is generated using the _GenerateRandom6DigitCode()_ function and sent to the user's Telegram via the _SendMessageToTelegram()_ function. This indicates successful authentication and prompts the application to transition to the two-factor authentication phase. If the password is incorrect, an error message prompts the user to try again.

2\. Two-Factor Authentication Code Generation and Delivery:

```
// Generate a random 6-digit code for 2FA
string GenerateRandom6DigitCode()
{
    int code = MathRand() % 1000000; // Produces a 6-digit number
    return StringFormat("%06d", code); // Ensures leading zeros
}

// Handle 2FA login button click
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
```

This section manages the two-factor authentication (2FA) by generating a unique 6-digit code and validating user input against this code. The _GenerateRandom6DigitCode()_ function generates a 6-digit number by utilizing the _[MathRand()](https://www.mql5.com/en/docs/math/mathrand?utm_campaign=search&utm_medium=special&utm_source=mt5editor)_ function, ensuring that it remains in the required format even when leading zeros are present. After the initial password is validated, this 6-digit code is sent via Telegram to the user's specified chat to enhance security. In the following function, _OnTwoFALoginButtonClick()_, the user’s input is checked against the generated code. If the entered code matches the one sent to Telegram, access to the admin panel is granted; otherwise, the user is notified of the incorrect code and prompted to try again

### Understanding [MathRand()](https://www.mql5.com/en/docs/math/mathrand?utm_campaign=search&utm_medium=special&utm_source=mt5editor) Function

This function in MQL5 is used to generate a pseudo-random integer. In case of our project, the _MathRand()_ function generates a random integer within the range of _0_ to _MathRandMax()_, which is typically defined as 0 to _32767_.

To generate a random 6-digit number, you can limit the output by applying the modulo operator (%). This operator calculates the remainder of a division, allowing you to restrict the range of random numbers to fit within valid 6-digit values, which span from _000000 (0)_ to _999999_.

Specifically, using the expression _MathRand() % 1000000_ will yield a result between _0_ and _999999_, ensuring that all possible 6-digit combinations, including those with leading zeros, are covered.

### Telegram API for 2FA verification

The _SendMessageToTelegram()_ function is crucial for ensuring that our 2FA codes and other messages are securely delivered to the user’s Telegram chat. This function constructs an HTTP POST request to the Telegram Bot API, including the _bot token_ and the destination _chat ID_. The message is formatted in JSON to adhere to the API's requirements.

The _WebRequest()_ function executes the request with a specified timeout, and the function checks the response code to confirm whether the message was sent successfully (HTTP code 200).

If the message delivery fails, the function logs an error, including the response code, error message, and any relevant response content, helping identify potential issues in sending messages for further diagnostics.

For more about [Telegram API](https://www.mql5.com/en/articles/14968), you can visit their [website](https://www.mql5.com/go?link=https://telegram.org/ "https://telegram.org/") and read some past articles where we explained about it.

```
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
```

As part of Telegram API, we have the following constants that we incorporated into our project to facilitate the two-factor authentication (2FA) process via Telegram.

```
// Constants for 2FA
const string Hardcoded2FAChatId = "REPLACE WITH YOUR CHAT ID";
const string Hardcoded2FABotToken = "REPLACE WITH YOUR ACTUAL BOT TOKEN";
```

In this portion of the code, the variable _Hardcoded2FAChatId_ represents the unique chat identifier for the Telegram chat where the authentication messages will be sent, while _Hardcoded2FABotToken_ holds the token for the Telegram bot used to send messages.

The bot token is essential for authenticating requests made to the Telegram API, ensuring that only a legitimate bot with the correct permissions can send messages to the specified chat. By hard-coding these constants, the program streamlines the process of sending 2FA codes, as the same chat ID and bot token are used each time without the need for user input or configuration.

However, it's important to note that hard-coding sensitive information like bot tokens can pose security risks if the code is exposed, so alternative secure storage methods should be considered for production environments.

The overall code is presented here, and we can observe that it is expanding significantly as new features are being implemented.

```
//+------------------------------------------------------------------+
//|                                             Admin Panel.mq5      |
//|                           Copyright 2024, Clemence Benjamin      |
//|        https://www.mql5.com/en/users/billionaire2024/seller      |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
#property description "A secure and  responsive Admin Panel. Send messages to your telegram clients without leaving MT5"
#property version   "1.20"

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
const string Hardcoded2FAChatId = "ENTER YOUR REAL CHAT ID";
const string Hardcoded2FABotToken = "ENTER YOUR Telegram Bot Token";

// Global variables
CDialog adminPanel;
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
        SendMessageToTelegram("A login attempt was made on the Admin Panel. Please use this code to verify your identity. " + twoFACode, Hardcoded2FAChatId, Hardcoded2FABotToken);

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

### Testing and Results

Finally, we have successfully implemented robust password security and two-factor authentication (2FA) for the Admin Panel. Below are images showcasing the responses.

![Wrong Password Attempt](https://c.mql5.com/2/98/login_error.PNG)

Wrong password attempt

![Wrong verification code attempt.](https://c.mql5.com/2/98/2FA_error.PNG)

Wrong verification code attempt

The following image illustrates the complete login and verification processes. We also managed to synchronize the Telegram app with the login procedure to capture the delivery of the verification code.

![Full login testing password and 2FA.](https://c.mql5.com/2/98/ShareX_fKP0dcDYtV.gif)

Full login password and 2FA testing

![2FA code delivery on telegram](https://c.mql5.com/2/98/ShareX_e8IWHxTmEd.gif)

2FA code delivery on Telegram for the above login attempt.

### Conclusion

Incorporating two-factor authentication (2FA) features into this MQL5 project significantly enhanced the security of the Admin Panel by adding a crucial layer of verification for user access. Leveraging Telegram for code delivery ensured that user receive real-time notifications. The implementation includes error handling for incorrect entries, prompting users to retry entering their passwords or 2FA codes, which minimizes unauthorized access while keeping users informed of any mistakes.

However, it is important to acknowledge that risks remain, particularly if the Telegram application is installed and logged in on the same computer at risk. An infiltrator may exploit vulnerabilities in such setups, especially if the device is compromised or if unauthorized users gain access to the phone being used for Telegram by the system admin. Therefore, maintaining strict security practices—such as staying logged out of the app when not in use and ensuring that devices are secure—becomes paramount in safeguarding sensitive data and communications.

I hope you gained valuable insights on implementing MQL5 for 2FA, particularly in the context of real-time code generation and secure messaging. Understanding these concepts will enhance the security of your applications and also highlight the importance of proactive measures in protecting your applications against potential threats. Attached is the final work below. Happy developing! Traders.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16142.zip "Download all attachments in the single ZIP archive")

[Admin\_\_Panel.mq5](https://www.mql5.com/en/articles/download/16142/admin__panel.mq5 "Download Admin__Panel.mq5")(21.85 KB)

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

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/475245)**
(1)


![joaopedrodev](https://c.mql5.com/avatar/2024/9/66da07c2-0125.png)

**[joaopedrodev](https://www.mql5.com/en/users/joaopedrodev)**
\|
23 Aug 2025 at 13:32

Amazing! The article's content is very interesting. Adding 2FA to MQL5 was brilliant!

Thank you for this contribution.

![MQL5 Wizard Techniques you should know (Part 44): Average True Range (ATR) technical indicator](https://c.mql5.com/2/99/MQL5_Wizard_Techniques_you_should_know_Part_44___LOGO.png)[MQL5 Wizard Techniques you should know (Part 44): Average True Range (ATR) technical indicator](https://www.mql5.com/en/articles/16213)

The ATR oscillator is a very popular indicator for acting as a volatility proxy, especially in the forex markets where volume data is scarce. We examine this, on a pattern basis as we have with prior indicators, and share strategies & test reports thanks to the MQL5 wizard library classes and assembly.

![Feature selection and dimensionality reduction using principal components](https://c.mql5.com/2/98/Feature_selection_and_dimensionality_reduction_using_principal_components____LOGO.png)[Feature selection and dimensionality reduction using principal components](https://www.mql5.com/en/articles/16190)

The article delves into the implementation of a modified Forward Selection Component Analysis algorithm, drawing inspiration from the research presented in “Forward Selection Component Analysis: Algorithms and Applications” by Luca Puggini and Sean McLoone.

![Neural Network in Practice: Straight Line Function](https://c.mql5.com/2/78/Rede_neural_na_prdtica_Fundso_de_reta____LOGO2.png)[Neural Network in Practice: Straight Line Function](https://www.mql5.com/en/articles/13696)

In this article, we will take a quick look at some methods to get a function that can represent our data in the database. I will not go into detail about how to use statistics and probability studies to interpret the results. Let's leave that for those who really want to delve into the mathematical side of the matter. Exploring these questions will be critical to understanding what is involved in studying neural networks. Here we will consider this issue quite calmly.

![Developing a Replay System (Part 49): Things Get Complicated (I)](https://c.mql5.com/2/77/Desenvolvendo_um_sistema_de_Replay_oParte_49q_____LOGO.png)[Developing a Replay System (Part 49): Things Get Complicated (I)](https://www.mql5.com/en/articles/11820)

In this article, we'll complicate things a little. Using what was shown in the previous articles, we will start to open up the template file so that the user can use their own template. However, I will be making changes gradually, as I will also be refining the indicator to reduce the load on MetaTrader 5.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/16142&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062610275791709561)

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