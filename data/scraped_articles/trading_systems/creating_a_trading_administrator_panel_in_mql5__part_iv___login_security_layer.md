---
title: Creating a Trading Administrator Panel in MQL5 (Part IV): Login Security Layer
url: https://www.mql5.com/en/articles/16079
categories: Trading Systems, Integration
relevance_score: 6
scraped_at: 2026-01-23T11:38:39.385203
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/16079&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062613840614565261)

MetaTrader 5 / Examples


### Contents Table:

- [Introduction](https://www.mql5.com/en/articles/16079#para2)
- [Overview of security in MQL5](https://www.mql5.com/en/articles/16079#para3)
- [Brief Recap of (Part III)](https://www.mql5.com/en/articles/16079#para4)
- [Integration of password protection in Admin Panel](https://www.mql5.com/en/articles/16079#para5)
- [Testing and Results](https://www.mql5.com/en/articles/16079#para6)
- [Conclusion](https://www.mql5.com/en/articles/16079#para7)

### Introduction

Security is paramount in any field of expertise, and we cannot afford to overlook its importance. With the persistent threat of unauthorized access, it’s crucial to safeguard our Admin Panel from potential intruders. If unauthorized individuals gain access, they could easily manipulate the panel and jeopardize our communication efforts with the broadcast community. The primary purpose of this system is to facilitate reliable communication, and while we can enhance functionality at the Expert Advisor level, the risks of intrusion remain significant.

An attacker accessing the dashboard could send misleading messages to our users, causing confusion and damaging the reputation of the system administrator. To mitigate these risks, I believe it is essential to implement a security layer that restricts access to key features without the correct credentials. This straightforward approach to security protects our panel and also helps maintain the integrity of our communications and the trust of our community.

![Login Panel](https://c.mql5.com/2/97/ShareX_1YG56ww0jC.gif)

Login Panel

### Overview of Security in MQL5

MQL5 offers a comprehensive range of security features designed to protect both source code and compiled files (EX5), safeguarding intellectual property and preventing unauthorized use. Key mechanisms include the encryption of compiled files, account-based and time-based licensing, and integration with external DLLs for additional protection. The platform supports digital signatures to verify code authenticity, while MetaQuotes provides code protection through compilation and obfuscation to deter reverse engineering. For products distributed via the [MQL5 Market](https://www.mql5.com/en/market), additional encryption ensures that only licensed users can access and use the software, establishing a robust security framework for developers.

In 2012, [Investeo](https://www.mql5.com/en/users/investeo), an author at MQL5, discussed various methods for securing MQL5 programs and code, sharing valuable insights on implementing techniques such as password protection, key generation, single account licensing, time-limit protection, remote licenses, secure license encryption, and advanced anti-decompilation methods. His work serves as a foundational reference for enhancing program security.

**Discussion Goal:**

Recognizing the importance of security, we aim to discuss the implementation of password protection for accessing the features of the Admin Panel. We will delve into the techniques used to achieve the results demonstrated in the earlier image, focusing on how we can safeguard user access effectively.

**Areas of Security Concern in Our Admin Panel:**

As our program evolves and incorporates new features, we acknowledge the growing complexity, particularly for novice developers. We identify several key areas of interest regarding security:

**- **Secured Access to the Admin Panel:****

To ensure that no unauthorized users can access the Admin Panel without the correct passcode, we implement password protection. Unauthorized access could lead to unintended messages being disseminated to the community of traders who rely on admin insights. Random clicks on quick buttons could occur without genuine intent, making a secure passcode essential. While many applications employ two-factor authentication (2FA) for additional verification, our current focus remains on implementing foundational security features, with plans to incorporate more advanced options as we progress.

**- **Security of Telegram API Messages:****

We also prioritize the security of communication via the Telegram API by securely inputting the Chat ID and bot token during program launch. This approach ensures that sensitive data remains protected in the hands of the user. Telegram employs robust security features to safeguard user communications, including transport layer security through the [MTProto protocol](https://www.mql5.com/go?link=https://core.telegram.org/mtproto "https://core.telegram.org/mtproto") for standard chats and end-to-end encryption for Secret Chats. Additionally, Telegram supports 2FA, allowing users to manage active sessions and enhance account security. While Telegram's security protocols are strong, users must also ensure their devices are secure, as compromised devices can undermine these protections.

### Brief Recap of (Part III)

In our previous discussion, we touched on incorporating methods for theme management. However, we were working with files that are subject to change during updates to the MetaTrader 5 platform. Each time an update is released, it is automatically downloaded and installed upon relaunch. Below is a code snippet that illustrates the errors I encountered when I attempted to compile it after the updates.

```
'UpdateThemeColors' - undeclared identifier     Admin Panel .mq5        390     16
'darkTheme' - some operator expected    Admin Panel .mq5        390     34
'SetTextColor' - undeclared identifier  Admin Panel .mq5        397     14
'textColor' - some operator expected    Admin Panel .mq5        397     27
'SetBackgroundColor' - undeclared identifier    Admin Panel .mq5        398     14
'bgColor' - some operator expected      Admin Panel .mq5        398     33
'SetBorderColor' - undeclared identifier        Admin Panel .mq5        399     14
'borderColor' - some operator expected  Admin Panel .mq5        399     29
'SetTextColor' - undeclared identifier  Admin Panel .mq5        424     12
'textColor' - some operator expected    Admin Panel .mq5        424     25
'SetBackgroundColor' - undeclared identifier    Admin Panel .mq5        425     12
'bgColor' - some operator expected      Admin Panel .mq5        425     31
'SetBorderColor' - undeclared identifier        Admin Panel .mq5        426     12
'borderColor' - some operator expected  Admin Panel .mq5        426     27
14 errors, 1 warnings           15      2
```

**Temporary Solution**

To resolve the issue, it's essential to first understand the source of the problem. As previously explained, platform updates reset the libraries we were using to their default states. Consequently, the methods we implemented for theme management are no longer valid, which is why we are now encountering errors. To address this, we need to overwrite the updated files [(Dialog.mqh, Edit.mqh, and Button.mqh)](https://www.mql5.com/en/articles/16045) with the [extended versions](https://www.mql5.com/en/articles/16045) that I attached in the previous article. You can locate the folder for the include files as shown in the image below.

![Loacate the root folder folder for includes files](https://c.mql5.com/2/97/ShareX_tZ8DfL93NK.gif)

Locating the dialog.mqh root folder easily

**Permanent Solution:**

We can rename the [Dialog.mqh](https://www.mql5.com/en/docs/standardlibrary/controls/cdialog?utm_campaign=search&utm_medium=special&utm_source=mt5editor) and other related files at use to [Extended\_Dialog.mqh](https://www.mql5.com/en/articles/16045) and adjust our code accordingly, but be sure to update any [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) statements that reference the old file name to reflect the new name. Additionally, we need to check for any other dependencies that might reference it and update those as needed. After making these changes, we recompile our project to identify if any potential errors and thoroughly test the functionality to ensure everything works correctly. This will save it separately with the new name but retain the original file.

For example, if we have already saved the file as [Extended\_Dialog.mqh](https://www.mql5.com/en/articles/16045), we can then navigate to our Admin Panel and adjust the code as follows:

```
#include <Controls\Extended_Dialog.mqh>
#include <Controls\Extended_Button.mqh>
#include <Controls\Extended_Edit.mqh>
#include <Controls\Label.mqh>
```

**Advantages of saving with a different name**

It offers the ability to tailor functionality specifically to your needs by adding or adjusting features not present in the built-in version. This customization enables you to create a unique interface that meets your requirements. Additionally, using custom file names helps avoid conflicts with built-in libraries or third-party libraries, reducing the risk of unexpected behaviors due to overlapping names. Isolating your enhancements in a renamed file further protects your customizations from being affected by other inbuilt features that might utilize the original Dialog, ensuring that you can develop and maintain your project without interference from external changes.

### Integration of Password Protection on the Admin Panel

In this project, we will implement conditional password protection using a string-type password that can include both letters and numbers, enhancing complexity. While a four-digit PIN may seem simple, it remains difficult to guess. On the Admin Panel, we utilize the Dialog class to prompt the user for a password upon login, with conditions set to display the main panel features only after successful password entry.

As we continue to develop the Admin Panel program, our primary focus is on establishing robust login security to ensure that only authorized users can access sensitive administrative functionalities. We have recognized the critical need to protect our system against unauthorized access and discuss how to implement MQL5 to secure our products.

**Authentication Mechanism**

To secure the admin panel, we implement a straightforward password-based authentication mechanism that prompts users for a password before granting access to any functionalities. This choice reflects our commitment to validating user identity as a prerequisite for accessing critical components of the program.

```
// Show authentication input dialog
bool ShowAuthenticationPrompt()
{
    if (!authentication.Create(ChartID(), "Authentication", 0, 100, 100, 500, 300))
    {
        Print("Failed to create authentication dialog");
        return false;
    }

}
```

In the **ShowAuthenticationPrompt** function, we design a user-friendly interface that effectively guides our users through the authentication process. By creating a dedicated dialog for password entry, we ensure that the primary access point to the admin panel remains secure while being intuitive.

To enhance understanding, I’ve outlined the code for dialog creation in the snippet below, complete with comments to explain how it works. If you’d like to refresh your memory on axes and coordinates, please refer to ( [Part I](https://www.mql5.com/en/articles/15417)).

```
// This condition checks if the authentication object is created successfully
if (!authentication.Create(          // Function call to create authentication
    ChartID(),                       // Retrieve the ID of the current chart
    "Authentication",                // Label of the dialog window in this case it is 'Authentication'
    0,                               // Initial X position on the chart also X_1
    100,                             // Initial Y position on the chart also Y_1
    500,                             // Width of the authentication window also X_2
    300                              // Height of the authentication window also Y_2
))
```

Having established the authentication dialog, we proceed to arrange other UI elements similarly, albeit with different values. The process starts with the creation of a password input box where users can type in their credentials, followed by essential buttons. Specifically, we focus on two main buttons: the "Login" button and the "Close" button. The "Login" button is used for submitting the entered password, while the "Close" button provides users with the option to exit the dialog if they do not know the password. Below is a code snippet illustrating the logic for creating these buttons, and the password prompt label.

```
 // Create password input
    if (!passwordInputBox.Create(ChartID(), "PasswordInputBox", 0, 20, 70, 260, 95))
    {
        Print("Failed to create password input box");
        return false;
    }
    authentication.Add(passwordInputBox);

    // Create prompt label
    if (!passwordPromptLabel.Create(ChartID(), "PasswordPromptLabel", 0, 20, 20, 260, 20))
    {
        Print("Failed to create password prompt label");
        return false;
    }
    passwordPromptLabel.Text("Enter password: Access Admin Panel");
    authentication.Add(passwordPromptLabel);

    // Create login button
    if (!loginButton.Create(ChartID(), "LoginButton", 0, 20, 120, 100, 140))
    {
        Print("Failed to create login button");
        return false;
    }
    loginButton.Text("Login");
    authentication.Add(loginButton);

    // Create close button for authentication
    if (!closeAuthButton.Create(ChartID(), "CloseAuthButton", 0, 120, 120, 200, 140)) // Adjusted position
    {
        Print("Failed to create close button for authentication");
        return false;
    }
    closeAuthButton.Text("Close");
    authentication.Add(closeAuthButton);

    authentication.Show(); // Show the authentication dialog
    ChartRedraw(); // Redraw the chart to reflect changes

    return true; // Prompt shown successfully
}
```

**Password Management**

Currently, we used a simple hard-coded password for initial testing, which allows us to prototype functionality rapidly. However, we fully understand that this approach carries risks, such as vulnerability to brute force attacks if the code is compromised.

```
// Default password for authentication
string Password = "2024";
```

While we recognize that using a hard-coded password expedites our development, we need to transition towards a more secure solution in future updates—specifically, implementing encrypted configuration files or utilizing a more sophisticated user account management system to enhance security.

**Handling User Input**

To bolster security, we need to ensure that the password input field is clearly defined in the authentication dialog. By guiding users to enter their passwords and validating those inputs against the stored password, we aim for a seamless and secure login experience.

```
// Handle login button click
void OnLoginButtonClick()
{
    string enteredPassword = passwordInputBox.Text();
    if (enteredPassword == Password) // Check the entered password
    {
        authentication.Destroy(); // Hide the authentication dialog
        Print("Authentication successful.");
        adminPanel.Show(); // Show the admin panel after successful authentication
    }
    else
    {
        Print("Incorrect password. Please try again.");
        passwordInputBox.Text(""); // Clear the password input
    }
}
```

In the **OnLoginButtonClick** function, the program checks whether the entered password matches the stored password. Upon successful entry, it hided the authentication dialog and present the admin panel to the user. If the password is incorrect, it clears the input field and prompt the user to try again, ensuring they clearly understand the process while feeling secure during their login.

We also have a handler for the "Close" button, which is responsible for the exit logic. When this button is clicked, it closes the authentication dialog and also completely removes the expert from the chart, ensuring that there is no lingering access to the admin functionalities. This action reinforces security and provides a clear exit pathway for users who choose not to proceed with authentication. Here’s how the handler is defined:

```
//+------------------------------------------------------------------+
//| Handle close button click for authentication                     |
//+------------------------------------------------------------------+
void OnCloseAuthButtonClick()
{
    authentication.Destroy();
    ExpertRemove(); // Remove the expert if user closes the authentication dialog
    Print("Authentication dialog closed.");
}
```

In this handler, the **authentication. Destroy()** method effectively closes the dialog, while **ExpertRemove()** ensures that the expert advisor is completely removed from view, enhancing the overall security of the application

Fully incorporated into the main program:

```
//+------------------------------------------------------------------+
//|                                             Admin Panel.mq5      |
//|                     Copyright 2024, Clemence Benjamin            |
//|     https://www.mql5.com/en/users/billionaire2024/seller         |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
#property version   "1.19"

#include <Trade\Trade.mqh>
#include <Controls\Dialog.mqh>
#include <Controls\Button.mqh>
#include <Controls\Edit.mqh>
#include <Controls\Label.mqh>

// Input parameters
input string QuickMessage1 = "Updates";
input string QuickMessage2 = "Close all";
input string QuickMessage3 = "In deep profits";
input string QuickMessage4 = "Hold position";
input string QuickMessage5 = "Swing Entry";
input string QuickMessage6 = "Scalp Entry";
input string QuickMessage7 = "Book profit";
input string QuickMessage8 = "Invalid Signal";
input string InputChatId = "Enter Chat ID from Telegram bot API";
input string InputBotToken = "Enter BOT TOKEN from your Telegram bot";

// Global variables
CDialog adminPanel;
CDialog authentication; // Renamed from passwordPanel
CButton sendButton, clearButton, changeFontButton, toggleThemeButton, loginButton, closeAuthButton;
CButton quickMessageButtons[8], minimizeButton, maximizeButton, closeButton;
CEdit inputBox, passwordInputBox;
CLabel charCounter, passwordPromptLabel;
bool minimized = false;
bool darkTheme = false;
int MAX_MESSAGE_LENGTH = 4096;
string availableFonts[] = { "Arial", "Courier New", "Verdana", "Times New Roman", "Britannic Bold", "Dubai Medium", "Impact", "Ink Tree", "Brush Script MT"};
int currentFontIndex = 0;

// Default password for authentication
string Password = "2024";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    if (!ShowAuthenticationPrompt())
    {
        Print("Authorization failed. Exiting...");
        return INIT_FAILED; // Exit if the authorization fails
    }

    // Initialize the main admin panel
    if (!adminPanel.Create(ChartID(), "Admin Panel", 0, 30, 30, 500, 500))
    {
        Print("Failed to create admin panel dialog");
        return INIT_FAILED;
    }

    // Create controls for the admin panel
    if (!CreateControls())
    {
        Print("Control creation failed");
        return INIT_FAILED;
    }

    // Initially hide the admin panel
    adminPanel.Hide();

    Print("Initialization complete");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Show authentication input dialog                                 |
//+------------------------------------------------------------------+
bool ShowAuthenticationPrompt()
{
    if (!authentication.Create(ChartID(), "Authentication", 0, 100, 100, 500, 300))
    {
        Print("Failed to create authentication dialog");
        return false;
    }

    // Create password input
    if (!passwordInputBox.Create(ChartID(), "PasswordInputBox", 0, 20, 70, 260, 95))
    {
        Print("Failed to create password input box");
        return false;
    }
    authentication.Add(passwordInputBox);

    // Create prompt label
    if (!passwordPromptLabel.Create(ChartID(), "PasswordPromptLabel", 0, 20, 20, 260, 20))
    {
        Print("Failed to create password prompt label");
        return false;
    }
    passwordPromptLabel.Text("Enter password: Access Admin Panel");
    authentication.Add(passwordPromptLabel);

    // Create login button
    if (!loginButton.Create(ChartID(), "LoginButton", 0, 20, 120, 100, 140))
    {
        Print("Failed to create login button");
        return false;
    }
    loginButton.Text("Login");
    authentication.Add(loginButton);

    // Create close button for authentication
    if (!closeAuthButton.Create(ChartID(), "CloseAuthButton", 0, 120, 120, 200, 140)) // Adjusted position
    {
        Print("Failed to create close button for authentication");
        return false;
    }
    closeAuthButton.Text("Close");
    authentication.Add(closeAuthButton);

    authentication.Show(); // Show the authentication dialog
    ChartRedraw(); // Redraw the chart to reflect changes

    return true; // Prompt shown successfully
}

//+------------------------------------------------------------------+
//| Handle chart events                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
    if (id == CHARTEVENT_OBJECT_CLICK)
    {
        // Handle button clicks inside the authentication dialog
        if (sparam == "LoginButton")
        {
            OnLoginButtonClick(); // Call the login button handler
        }
        else if (sparam == "CloseAuthButton") // Made sure this matches the ID
        {
            OnCloseAuthButtonClick(); // Call the close button handler
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
    if (enteredPassword == Password) // Check the entered password
    {
        authentication.Destroy(); // Hide the authentication dialog
        Print("Authentication successful.");
        adminPanel.Show(); // Show the admin panel after successful authentication
    }
    else
    {
        Print("Incorrect password. Please try again.");
        passwordInputBox.Text(""); // Clear the password input
    }
}

//+------------------------------------------------------------------+
//| Handle close button click for authentication                     |
//+------------------------------------------------------------------+
void OnCloseAuthButtonClick()
{
    authentication.Destroy();
    ExpertRemove(); // Remove the expert if user closes the authentication dialog
    Print("Authentication dialog closed.");
}

//+------------------------------------------------------------------+
//| Create necessary UI controls                                     |
//+------------------------------------------------------------------+
bool CreateControls()
{
    long chart_id = ChartID();

    // Create the input box
    if (!inputBox.Create(chart_id, "InputBox", 0, 5, 25, 460, 95))
    {
        Print("Failed to create input box");
        return false;
    }
    adminPanel.Add(inputBox);

    // Character counter
    if (!charCounter.Create(chart_id, "CharCounter", 0, 380, 5, 460, 25))
    {
        Print("Failed to create character counter");
        return false;
    }
    charCounter.Text("0/" + IntegerToString(MAX_MESSAGE_LENGTH));
    adminPanel.Add(charCounter);

    // Clear button
    if (!clearButton.Create(chart_id, "ClearButton", 0, 235, 95, 345, 125))
    {
        Print("Failed to create clear button");
        return false;
    }
    clearButton.Text("Clear");
    adminPanel.Add(clearButton);

    // Send button
    if (!sendButton.Create(chart_id, "SendButton", 0, 350, 95, 460, 125))
    {
        Print("Failed to create send button");
        return false;
    }
    sendButton.Text("Send");
    adminPanel.Add(sendButton);

    // Change font button
    if (!changeFontButton.Create(chart_id, "ChangeFontButton", 0, 95, 95, 230, 115))
    {
        Print("Failed to create change font button");
        return false;
    }
    changeFontButton.Text("Font<>");
    adminPanel.Add(changeFontButton);

    // Toggle theme button
    if (!toggleThemeButton.Create(chart_id, "ToggleThemeButton", 0, 5, 95, 90, 115))
    {
        Print("Failed to create toggle theme button");
        return false;
    }
    toggleThemeButton.Text("Theme<>");
    adminPanel.Add(toggleThemeButton);

    // Minimize button
    if (!minimizeButton.Create(chart_id, "MinimizeButton", 0, 375, -22, 405, 0)) // Adjusted Y-coordinate for visibility
    {
        Print("Failed to create minimize button");
        return false;
    }
    minimizeButton.Text("_");
    adminPanel.Add(minimizeButton);

    // Maximize button
    if (!maximizeButton.Create(chart_id, "MaximizeButton", 0, 405, -22, 435, 0)) // Adjusted Y-coordinate for visibility
    {
        Print("Failed to create maximize button");
        return false;
    }
    maximizeButton.Text("[ ]");
    adminPanel.Add(maximizeButton);

    // Close button for admin panel
    if (!closeButton.Create(chart_id, "CloseButton", 0, 435, -22, 465, 0)) // Adjusted Y-coordinate for visibility
    {
        Print("Failed to create close button");
        return false;
    }
    closeButton.Text("X");
    adminPanel.Add(closeButton);

    // Quick messages
    return CreateQuickMessageButtons();
}

//+------------------------------------------------------------------+
//| Create quick message buttons                                     |
//+------------------------------------------------------------------+
bool CreateQuickMessageButtons()
{
    string quickMessages[8] = { QuickMessage1, QuickMessage2, QuickMessage3, QuickMessage4, QuickMessage5, QuickMessage6, QuickMessage7, QuickMessage8 };
    int startX = 5, startY = 160, width = 222, height = 65, spacing = 5;

    for (int i = 0; i < 8; i++)
    {
        if (!quickMessageButtons[i].Create(ChartID(), "QuickMessageButton" + IntegerToString(i + 1), 0, startX + (i % 2) * (width + spacing), startY + (i / 2) * (height + spacing), startX + (i % 2) * (width + spacing) + width, startY + (i / 2) * (height + spacing) + height))
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
    if (message != "")
    {
        if (SendMessageToTelegram(message))
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
void OnQuickMessageButtonClick(int index)
{
    string quickMessages[8] = { QuickMessage1, QuickMessage2, QuickMessage3, QuickMessage4, QuickMessage5, QuickMessage6, QuickMessage7, QuickMessage8 };
    string message = quickMessages[index];

    if (SendMessageToTelegram(message))
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
    // Use the dialog's theme update method as a placeholder.
    adminPanel.UpdateThemeColors(darkTheme);

    color textColor = darkTheme ? clrWhite : clrBlack;
    color buttonBgColor = darkTheme ? clrDarkSlateGray : clrGainsboro;
    color borderColor = darkTheme ? clrSlateGray : clrGray;
    color bgColor     = darkTheme ? clrDarkBlue : clrWhite;

    inputBox.SetTextColor(textColor);
    inputBox.SetBackgroundColor(bgColor);
    inputBox.SetBorderColor(borderColor);

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

    charCounter.Color(textColor);

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

    inputBox.Font(availableFonts[currentFontIndex]);
    clearButton.Font(availableFonts[currentFontIndex]);
    sendButton.Font(availableFonts[currentFontIndex]);
    toggleThemeButton.Font(availableFonts[currentFontIndex]);
    changeFontButton.Font(availableFonts[currentFontIndex]);

    for (int i = 0; i < ArraySize(quickMessageButtons); i++)
    {
        quickMessageButtons[i].Font(availableFonts[currentFontIndex]);
    }

    Print("Font changed to: ", availableFonts[currentFontIndex]);
    ChartRedraw();
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
        Print("Panel maximized.");
    }
}

//+------------------------------------------------------------------+
//| Handle close button click for admin panel                        |
//+------------------------------------------------------------------+
void OnCloseButtonClick()
{
    ExpertRemove();
    Print("Admin Panel closed.");
}

//+------------------------------------------------------------------+
//| Send the message to Telegram                                     |
//+------------------------------------------------------------------+
bool SendMessageToTelegram(string message)
{
    string url = "https://api.telegram.org/bot" + InputBotToken + "/sendMessage";
    string jsonMessage = "{\"chat_id\":\"" + InputChatId + "\", \"text\":\"" + message + "\"}";
    char post_data[];
    ArrayResize(post_data, StringToCharArray(jsonMessage, post_data, 0, WHOLE_ARRAY) - 1);

    int timeout = 5000;
    char result[];
    string responseHeaders;

    int res = WebRequest("POST", url, "Content-Type: application/json\r\n", timeout, post_data, result, responseHeaders);

    if (res == 200)
    {
        Print("Message sent successfully: ", message);
        return true;
    }
    else
    {
        Print("Failed to send message. HTTP code: ", res, " Error code: ", GetLastError());
        Print("Response: ", CharArrayToString(result));
        return false;
    }
}
```

### Testing and Results

Our code compiled successfully, and upon launching the application, we observed that the panel's full features remain inaccessible until the correct PIN is entered. This behavior ensures that only authorized users can access the administrative functions. At this stage, we are proud of our progress, yet we recognize that we have not yet reached the limits of our development. We understand that our security measures still need enhancement, as they may be vulnerable to advanced hackers. We know that every step we take is an opportunity to learn more about implementing the MQL5 language, and as we advance in our skills, we can achieve more robust security levels. Below is an image showcasing the launch of the application along with the desired outcome.

![Secure Admin Panel](https://c.mql5.com/2/97/ShareX_d49S3DCb4y__1.gif)

Panel Launch

### Conclusion

In this project, the implementation of a login authentication mechanism significantly enhanced the security of the Admin Panel, which is vital for protecting sensitive functionalities. By requiring a password before granting access to the admin features, the program mitigates unauthorized use and ensures that only verified users can manage crucial settings and operations. The design is fortified by a clearly defined global password and a user-friendly interface for entering credentials.

As we advance our Admin Panel, we will focus on critical improvements such as transitioning from hard-coded passwords to securely managed credentials to prevent vulnerabilities, incorporating multifactor authentication for added security, and continuously optimizing the login experience.

On the other hand, we recognize that once the code is compiled, it becomes challenging for anyone without access to the source code to gain entry, thanks to the anti-decompilation security features offered by MQL5. This added layer of protection helps safeguard our application from unauthorized access and reverse engineering.

Please don't hesitate to try it out in your projects! I welcome comments and feedback, as your insights can help us improve and refine our work. Your perspectives are valuable to us as we continue to develop and enhance our applications. Check the attachment below.

[Back to Content Page](https://www.mql5.com/en/articles/16079#para0)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16079.zip "Download all attachments in the single ZIP archive")

[Admin\_Panel\_.mq5](https://www.mql5.com/en/articles/download/16079/admin_panel_.mq5 "Download Admin_Panel_.mq5")(18.66 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/474928)**
(1)


![SERGEI NAIDENOV](https://c.mql5.com/avatar/2019/11/5DE2C04F-1280.jpg)

**[SERGEI NAIDENOV](https://www.mql5.com/en/users/leonsi)**
\|
23 May 2025 at 19:53

**MetaQuotes:**

The article [Creating an Administrator's Trading Panel in MQL5 (Part IV)](https://www.mql5.com/ru/articles/16079) has been published: [Login Security](https://www.mql5.com/ru/articles/16079):

Author: [Clemence Benjamin](https://www.mql5.com/ru/users/Billionaire2024 "Billionaire2024")

**When trying to compile:**

'Admin\_Panel.mq5' 1

Trade.mqh

Object.mqh

StdLibErr.mqh

OrderInfo.mqh

HistoryOrderInfo.mqh

PositionInfo.mqh

DealInfo.mqh

Dialog.mqh

WndContainer.mqh

Wnd.mqh

Rect.mqh

Defines.mqh

ArrayObj.mqh

Array.mqh

WndClient.mqh

Panel.mqh

WndObj.mqh

ChartObjectsTxtControls.mqh

ChartObject.mqh

Scrolls.mqh

BmpButton.mqh

ChartObjectsBmpControls.mqh

Edit.mqh

Chart.mqh

Button.mqh

Label.mqh

'Up.bmp' as resource "::res\\Up.bmp" 1

'ThumbVert.bmp' as resource "::res\\ThumbVert.bmp" 1

'Down.bmp' as resource "::res\\Down.bmp" 1

'Left.bmp' as resource "::res\\Left.bmp" 1

'ThumbHor.bmp' as resource "::res\\ThumbHor.bmp" 1

'Right.bmp' as resource "::res\\Right.bmp" 1

'Close.bmp' as resource "::res\\Close.bmp" 1

'Restore.bmp' as resource "::res\\Restore.bmp" 1

'Turn.bmp' as resource "::res\\Turn.bmp" 1

possible loss of data due to type conversion from 'long' to 'int' Admin\_Panel(4)\_.mq5 161 49

'UpdateThemeColors' - undeclared identifier Admin\_Panel(4)\_.mq5 390 16

'darkTheme' - some operator expected Admin\_Panel(4)\_.mq5 390 34

'SetTextColor' - undeclared identifier Admin\_Panel(4)\_.mq5 397 14

'textColor' - some operator expected Admin\_Panel(4)\_.mq5 397 27

'SetBackgroundColor' - undeclared identifier Admin\_Panel(4)\_.mq5 398 14

'bgColor' - some operator expected Admin\_Panel(4)\_.mq5 398 33

'SetBorderColor' - undeclared identifier Admin\_Panel(4)\_.mq5 399 14

'borderColor' - some operator expected Admin\_Panel(4)\_.mq5 399 29

'SetTextColor' - undeclared identifier Admin\_Panel(4)\_.mq5 424 12

'textColor' - some operator expected Admin\_Panel(4)\_.mq5 424 25

'SetBackgroundColor' - undeclared identifier Admin\_Panel(4)\_.mq5 425 12

'bgColor' - some operator expected Admin\_Panel(4)\_.mq5 425 31

'SetBorderColor' - undeclared identifier Admin\_Panel(4)\_.mq5 426 12

'borderColor' - some operator expected Admin\_Panel(4)\_.mq5 426 27

14 errors, 1 warnings 15 2

![MQL5 Wizard Techniques you should know (Part 43): Reinforcement Learning with SARSA](https://c.mql5.com/2/98/MQL5_Wizard_Techniques_you_should_know_Part_43___LOGO.png)[MQL5 Wizard Techniques you should know (Part 43): Reinforcement Learning with SARSA](https://www.mql5.com/en/articles/16143)

SARSA, which is an abbreviation for State-Action-Reward-State-Action is another algorithm that can be used when implementing reinforcement learning. So, as we saw with Q-Learning and DQN, we look into how this could be explored and implemented as an independent model rather than just a training mechanism, in wizard assembled Expert Advisors.

![Developing a Replay System (Part 48): Understanding the concept of a service](https://c.mql5.com/2/76/Desenvolvendo_um_sistema_de_Replay_9Parte_480___LOGO.png)[Developing a Replay System (Part 48): Understanding the concept of a service](https://www.mql5.com/en/articles/11781)

How about learning something new? In this article, you will learn how to convert scripts into services and why it is useful to do so.

![Integrating MQL5 with data processing packages (Part 3): Enhanced Data Visualization](https://c.mql5.com/2/98/Integrating_MQL5_with_data_processing_packages_Part_3___LOGO.png)[Integrating MQL5 with data processing packages (Part 3): Enhanced Data Visualization](https://www.mql5.com/en/articles/16083)

In this article, we will perform Enhanced Data Visualization by going beyond basic charts by incorporating features like interactivity, layered data, and dynamic elements, enabling traders to explore trends, patterns, and correlations more effectively.

![MQL5 Trading Toolkit (Part 3): Developing a Pending Orders Management EX5 Library](https://c.mql5.com/2/98/MQL5_Trading_Toolkit_Part_3___LOGO.png)[MQL5 Trading Toolkit (Part 3): Developing a Pending Orders Management EX5 Library](https://www.mql5.com/en/articles/15888)

Learn how to develop and implement a comprehensive pending orders EX5 library in your MQL5 code or projects. This article will show you how to create an extensive pending orders management EX5 library and guide you through importing and implementing it by building a trading panel or graphical user interface (GUI). The expert advisor orders panel will allow users to open, monitor, and delete pending orders associated with a specified magic number directly from the graphical interface on the chart window.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=fasohaaucevzhhkekoblhmdltrcpstgc&ssn=1769157517460369577&ssn_dr=0&ssn_sr=0&fv_date=1769157517&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16079&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20a%20Trading%20Administrator%20Panel%20in%20MQL5%20(Part%20IV)%3A%20Login%20Security%20Layer%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915751794860758&fz_uniq=5062613840614565261&sv=2552)

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