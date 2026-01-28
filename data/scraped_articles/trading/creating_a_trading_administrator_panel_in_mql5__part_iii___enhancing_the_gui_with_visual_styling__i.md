---
title: Creating a Trading Administrator Panel in MQL5 (Part III): Enhancing the GUI with Visual Styling (I)
url: https://www.mql5.com/en/articles/15419
categories: Trading, Integration
relevance_score: 6
scraped_at: 2026-01-22T18:00:18.964346
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/15419&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049553669701086592)

MetaTrader 5 / Examples


### Contents:

1. [Introduction](https://www.mql5.com/en/articles/15419#para1)
2. [Importance of a Visually appealing GUI](https://www.mql5.com/en/articles/15419#para2)
3. [Applying MQL5 GUI Styling Features](https://www.mql5.com/en/articles/15419#para3)
4. [Customizing Colors and Fonts](https://www.mql5.com/en/articles/15419#para4)
5. [Theme Management Logic](https://www.mql5.com/en/articles/15419#para5)
6. [Adjusting New Buttons Layout](https://www.mql5.com/en/articles/15419#para6)
7. [Advanced Enhancement of the GUI](https://www.mql5.com/en/articles/15419#para7)
8. [Conclusion](https://www.mql5.com/en/articles/15419#para8)

### Introduction

Looking at the goals outlined in our previous article, can we confidently say we have done enough? In my opinion, what I see inspires a drive for advancement beyond our current offerings. Imagine how beneficial it would be to implement a toggle between dark and light themes for our Admin Panel. Additionally, we could enhance the user experience by adding stylish buttons, offering a diverse selection of fonts, and enabling language switching among major languages. This would make our panel more user-friendly for everyone.

Our goal is to provide trading Administrators with a comprehensive communication solution integrated within the trading platform. The concepts we aim to incorporate are inspired by influential research and developments in graphical user interfaces (GUIs) since the 1970s. Notable contributors include [Alan Kay](https://en.wikipedia.org/wiki/Alan_Kay "https://en.wikipedia.org/wiki/Alan_Kay"), [Xerox PARC](https://en.wikipedia.org/wiki/PARC_(company) "https://en.wikipedia.org/wiki/PARC_(company)"), [Apple (macOS](https://en.wikipedia.org/wiki/MacOS "https://en.wikipedia.org/wiki/MacOS")), [Microsoft (Windows)](https://en.wikipedia.org/wiki/Microsoft_Windows "https://en.wikipedia.org/wiki/Microsoft_Windows"), [CSS (Cascading Style Sheets)](https://en.wikipedia.org/wiki/CSS "https://en.wikipedia.org/wiki/CSS"), and [Material Design](https://en.wikipedia.org/wiki/Material_Design "https://en.wikipedia.org/wiki/Material_Design") from Google. By leveraging these insights, we can create an Admin Panel meets users’ needs and enhances their overall experience.

![Simple Admin with Quick Message buttons](https://c.mql5.com/2/92/Simple_Panel.PNG)

The Basic Admin Panel we have developed so far.

Recap of what we have accomplished so far **:**

-  Creating the Admin Panel with a [Messaging Interface and Telegram Integration](https://www.mql5.com/en/articles/15417).
-  Adding essential buttons to the interface such as minimize, maximize, close, and quick message buttons.

By the end of this article, we will have a fully customized and visually styled trading administrator panel in MQL5. You’ll learn how to implement various styling techniques that improve both the appearance and functionality of the interface, creating a professional and user-friendly environment for traders.

Here are the main objectives of this article:

- Applying basic styling techniques using MQL5
- Customizing fonts, colors, and layouts
- Enhancing user interaction with visual elements
- Incorporating customizability between light and dark theme mode.
- Adding dynamic features like animations and transitions

### Applying MQL5 GUI Styling Features

MQL5 provides various functions and features to style the GUI of your trading application. These include options for customizing colors, fonts, and layouts to suit the needs of your users and the overall design aesthetic you want to achieve.

Styling the GUI in MQL5 involves using several key functions and techniques.  We will write discuss the functions which allow us to change the properties of graphical objects, such as buttons, labels, and panels. With these, we can customize the background color, border style, font size, and other visual aspects to create a cohesive look and feel.

> 1. Customizing Colors and Fonts
> 2. Theme Management logic
> 3. Adjusting New Buttons Layout

### Customizing Colors and Fonts:

> **Font Array and Index:**
>
> We start by defining an **availableFonts** array and a **currentFontIndex** to manage font selections for the Admin Panel. The **availableFonts** array includes font names like "Arial," "Courier New," "Verdana," and "Times New Roman," giving users a range of options to customize the panel’s appearance. The **currentFontIndex** keeps track of the selected font by indexing into this array. This setup allows us to easily cycle through the fonts and apply them to the UI components whenever the user changes the font, making sure the user experience remains both dynamic and cohesive.
>
> ```
> // Array of available fonts
> string availableFonts[] = {"Arial", "Courier New", "Verdana", "Times New Roman"};
> // Index of the current font in use
> int currentFontIndex = 0;
> ```
>
> **Creating the Change Font Button:**
>
> Let’s create a button labeled " **Font<>**," which we position strategically within the Admin Panel. This button is not just any button; it’s a key feature for changing fonts. We ensure it fits well within the panel’s layout and handle any issues with its creation. By adding this button, we provide users with an intuitive way to cycle through different fonts, enhancing the panel’s usability and aesthetic flexibility. If there’s a hiccup in creating the button, we print an error message to keep track of any issues.
>
> ```
> // Create a button for changing the font
> CButton changeFontButton;
> changeFontButton.Create(panel, "ChangeFontButton", 0, 10, 10, 100, 30);
> changeFontButton.Text("Font<>");
>
> // Verify button creation and handle errors
> if(!changeFontButton.IsCreated())
> {
>     Print("Error creating Font<> button.");
> }
> ```
>
> **Handling Font Change Button Click:**
>
> When we implement the **OnChangeFontButtonClick** function, our goal is to manage the font-changing process smoothly. This function updates the **currentFontIndex** to select the next font in the **availableFonts** array, wrapping around to the beginning if needed. After updating the index, we apply the new font to all relevant UI components, such as the input box, clear button, and send button, ensuring a consistent look across the panel. To finalize the changes, we use **ChartRedraw** to refresh the display and print a confirmation message, letting users know the font change was successful.
>
> ```
> // Function to handle the font change button click
> void OnChangeFontButtonClick()
> {
>     // Update the font index, wrapping around if necessary
>     currentFontIndex = (currentFontIndex + 1) % ArraySize(availableFonts);
>     string newFont = availableFonts[currentFontIndex];
>
>     // Apply the new font to UI components
>     inputBox.Font(newFont);
>     clearButton.Font(newFont);
>     sendButton.Font(newFont);
>
>     // Refresh the display to apply the changes
>     ChartRedraw();
>
>     // Print confirmation of the font change
>     Print("Font changed to ", newFont);
> }
> ```
>
> **OnChartEvent to Handle Button Clicks:**
>
> In the **OnChartEvent** function, we handle user interactions with various chart objects, including our font change button. This function listens for button click events and checks which button was clicked by inspecting the sparam string. When the " **ChangeFontButton**" is clicked, we call the **OnChangeFontButtonClick** function to manage the font change. This event-driven approach keeps our UI responsive and interactive, ensuring that user actions trigger the right responses and maintain an engaging interface.
>
> ```
> // Function to handle chart events
> void OnChartEvent(const int id, const int sub_id, const int type, const int x, const int y, const int state)
> {
>     // Handle button clicks
>     if(type == CHARTEVENT_OBJECT_CLICK)
>     {
>         string buttonName = ObjectGetString(0, "ChangeFontButton", OBJPROP_TEXT);
>         if(buttonName == "Font<>")
>         {
>             OnChangeFontButtonClick();
>         }
>     }
> }
> ```

![Font Switching Working fine](https://c.mql5.com/2/93/terminal64_LtO2zbQIrF.gif)

Font Switching working

### Theme Management Logic:

> **Theme Switching Logic:**
>
>  We start by setting up the theme management system with two distinct themes: light and dark. To manage the switch, we use a boolean variable, **isDarkMode**, that keeps track of which theme is currently active. The toggle is simple: when the user clicks the theme button, the **isDarkMode** value flips, changing the entire look and feel of the Admin Panel. By defining the colors for each theme separately, we streamline the process, making it easy to maintain and apply new styles whenever necessary.
>
> ```
> bool isDarkMode = false; // Tracks the current theme mode (light or dark)
> color lightBackgroundColor = clrWhite;  // Background color for light mode
> color darkBackgroundColor = clrBlack;   // Background color for dark mode
> color lightTextColor = clrBlack;        // Text color for light mode
> color darkTextColor = clrWhite;         // Text color for dark mode
> ```
>
> **Create the Theme Switch Button:**
>
>  Let’s move on to creating a button labeled **"Theme<>.**" This button is placed within the Admin Panel, and it provides users with an easy way to swap between light and dark modes. If something goes wrong during its creation, we make sure to handle the error with a printed message. This makes troubleshooting easier and ensures that the interface remains intuitive and responsive.
>
> ```
> //Creating the theme switch button
> if(!CreateButton("ToggleThemeButton", "Theme<>", 50, 220, 100, 30))
> {
>    Print("Error: Failed to create theme toggle button"); // Error handling if button creation fails
> }
> ```
>
> **Handle Theme Toggle Button Click:**
>
>  Next, we handle the actual theme change by implementing the **OnToggleModeButtonClick** function. This function flips the **isDarkMode** variable, switching between the light and dark themes. Once we know which theme is active, we apply the corresponding background and text colors to all UI elements, such as the panel, buttons, and text. The theme change happens in real-time thanks to a quick refresh, making the interface feel smooth and responsive. We also print a confirmation message for users so they know when the mode has changed.
>
> ```
> //Theme switching handler
> void OnToggleModeButtonClick()
> {
>     isDarkMode = !isDarkMode; // Toggle the theme mode
>     if(isDarkMode)
>     {
>         ApplyTheme(darkBackgroundColor, darkTextColor); // Apply dark mode colors
>     }
>     else
>     {
>         ApplyTheme(lightBackgroundColor, lightTextColor); // Apply light mode colors
>     }
>     Print("Theme has been switched"); // Inform the user that the theme has changed
> }
> ```
>
> **OnChartEvent to Handle Theme Toggle Button Clicks:**
>
>  In the **OnChartEvent** function, we detect when a user clicks the "Toggle Theme" button and trigger the **OnToggleModeButtonClick** function. This event-driven approach ensures that the panel instantly responds to user actions. By listening for button click events, we make sure the Admin Panel stays interactive and engaging, letting users easily switch between light and dark themes as needed.
>
> ```
> //The OneChartEvent for  the theme
> void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
> {
>     if(id == CHARTEVENT_OBJECT_CLICK) // Check if the event is a button click
>     {
>         if(sparam == "ToggleThemeButton") // Check if the clicked button is the theme toggle button
>         {
>             OnToggleModeButtonClick(); // Call the function to handle the theme change
>         }
>     }
> }
> ```
>
> **Applying Theme Without Recreating Objects:**
>
>  One of our key design decisions is to update the theme without recreating any of the panel’s objects. Instead of tearing down and building new UI components, we simply apply the new color scheme to the existing elements. This keeps the system efficient, reducing lag and maintaining a smooth user experience. It also ensures that the panel remains responsive as we apply the new colors dynamically.
>
> ```
>
> //Applying theme
> void ApplyTheme(color backgroundColor, color textColor)
> {
>     // Update background and text colors of existing objects
>     ObjectSetInteger(0, "AdminPanelBackground", OBJPROP_COLOR, backgroundColor); // Change background color
>     ObjectSetInteger(0, "ClearButton", OBJPROP_COLOR, textColor);                // Change text color of clear button
>     ObjectSetInteger(0, "SendButton", OBJPROP_COLOR, textColor);                 // Change text color of send button
>     ObjectSetInteger(0, "InputBox", OBJPROP_COLOR, textColor);                   // Change text color of input box
>     ChartRedraw(); // Redraw the chart to reflect the changes
> }
> ```

### Adjusting New Buttons Layout.

> **Font Change button:**

> The Font Change button, we positioned it in the admin panel with its top-left corner at (95, 95) and its bottom-right corner at (230, 115). This places it to the left of the Send and Clear buttons. Its dimensions make it wide enough for the "Font<>" label and user-friendly interaction. The button allows users to cycle through different font options for all text elements within the panel.
>
> ```
> if (!changeFontButton.Create(chart_id, "ChangeFontButton", 0, 95, 95, 230, 115))
> ```
>
> **Theme Switch button:**
>
>  As for the Theme Switch button, we placed it at coordinates (5, 95) for the top-left and (90, 115) for the bottom-right. This positions the button at the far-left side of the panel, slightly above the Font Change button, providing a clear separation. The compact size and proximity to other buttons make it easy for users to switch between dark and light themes without cluttering the interface.
>
> ```
> if (!toggleThemeButton.Create(chart_id, "ToggleThemeButton", 0, 5, 95, 90, 115))
> ```
>
>  Here's our full program with all the new features perfectly integrated.
>
> ```
> //+------------------------------------------------------------------+
> //|                                             Admin Panel.mq5      |
> //|                     Copyright 2024, Clemence Benjamin            |
> //|     https://www.mql5.com/en/users/billionaire2024/seller         |
> //+------------------------------------------------------------------+
> #property copyright "Copyright 2024, Clemence Benjamin"
> #property link      "https://www.mql5.com/en/users/billionaire2024/seller"
> #property description "A responsive Admin Panel. Send messages to your telegram clients without leaving MT5"
> #property version   "1.11"
>
> #include <Trade\Trade.mqh>
> #include <Controls\Dialog.mqh>
> #include <Controls\Button.mqh>
> #include <Controls\Edit.mqh>
> #include <Controls\Label.mqh>
>
> // Input parameters
> input string QuickMessage1 = "Updates";
> input string QuickMessage2 = "Close all";
> input string QuickMessage3 = "In deep profits";
> input string QuickMessage4 = "Hold position";
> input string QuickMessage5 = "Swing Entry";
> input string QuickMessage6 = "Scalp Entry";
> input string QuickMessage7 = "Book profit";
> input string QuickMessage8 = "Invalid Signal";
> input string InputChatId = "Enter Chat ID from Telegram bot API";
> input string InputBotToken = "Enter BOT TOKEN from your Telegram bot";
>
> // Global variables
> CDialog adminPanel;
> CButton sendButton, clearButton, changeFontButton, toggleThemeButton;
> CButton quickMessageButtons[8], minimizeButton, maximizeButton, closeButton;
> CEdit inputBox;
> CLabel charCounter;
> #define BG_RECT_NAME "BackgroundRect"
> bool minimized = false;
> bool darkTheme = false;
> int MAX_MESSAGE_LENGTH = 4096;
> string availableFonts[] = { "Arial", "Courier New", "Verdana", "Times New Roman" };
> int currentFontIndex = 0;
>
> //+------------------------------------------------------------------+
> //| Expert initialization function                                   |
> //+------------------------------------------------------------------+
> int OnInit()
> {
>     // Initialize the Dialog
>     if (!adminPanel.Create(ChartID(), "Admin Panel", 0, 30, 30, 500, 500))
>     {
>         Print("Failed to create dialog");
>         return INIT_FAILED;
>     }
>
>     // Create controls
>     if (!CreateControls())
>     {
>         Print("Control creation failed");
>         return INIT_FAILED;
>     }
>
>     adminPanel.Show();
>     // Initialize with the default theme
>     CreateOrUpdateBackground(ChartID(), darkTheme ? clrBlack : clrWhite);
>
>     Print("Initialization complete");
>     return INIT_SUCCEEDED;
> }
>
> //+------------------------------------------------------------------+
> //| Create necessary UI controls                                    |
> //+------------------------------------------------------------------+
> bool CreateControls()
> {
>     long chart_id = ChartID();
>
>     // Create the input box
>     if (!inputBox.Create(chart_id, "InputBox", 0, 5, 25, 460, 95))
>     {
>         Print("Failed to create input box");
>         return false;
>     }
>     adminPanel.Add(inputBox);
>
>     // Character counter
>     if (!charCounter.Create(chart_id, "CharCounter", 0, 380, 5, 460, 25))
>     {
>         Print("Failed to create character counter");
>         return false;
>     }
>     charCounter.Text("0/" + IntegerToString(MAX_MESSAGE_LENGTH));
>     adminPanel.Add(charCounter);
>
>     // Clear button
>     if (!clearButton.Create(chart_id, "ClearButton", 0, 235, 95, 345, 125))
>     {
>         Print("Failed to create clear button");
>         return false;
>     }
>     clearButton.Text("Clear");
>     adminPanel.Add(clearButton);
>
>     // Send button
>     if (!sendButton.Create(chart_id, "SendButton", 0, 350, 95, 460, 125))
>     {
>         Print("Failed to create send button");
>         return false;
>     }
>     sendButton.Text("Send");
>     adminPanel.Add(sendButton);
>
>     // Change font button
>     if (!changeFontButton.Create(chart_id, "ChangeFontButton", 0, 95, 95, 230, 115))
>     {
>         Print("Failed to create change font button");
>         return false;
>     }
>     changeFontButton.Text("Font<>");
>     adminPanel.Add(changeFontButton);
>
>     // Toggle theme button
>     if (!toggleThemeButton.Create(chart_id, "ToggleThemeButton", 0, 5, 95, 90, 115))
>     {
>         Print("Failed to create toggle theme button");
>         return false;
>     }
>     toggleThemeButton.Text("Theme<>");
>     adminPanel.Add(toggleThemeButton);
>
>     // Minimize button
>     if (!minimizeButton.Create(chart_id, "MinimizeButton", 0, 375, -22, 405, 0))
>     {
>         Print("Failed to create minimize button");
>         return false;
>     }
>     minimizeButton.Text("_");
>     adminPanel.Add(minimizeButton);
>
>     // Maximize button
>     if (!maximizeButton.Create(chart_id, "MaximizeButton", 0, 405, -22, 435, 0))
>     {
>         Print("Failed to create maximize button");
>         return false;
>     }
>     maximizeButton.Text("[ ]");
>     adminPanel.Add(maximizeButton);
>
>     // Close button
>     if (!closeButton.Create(chart_id, "CloseButton", 0, 435, -22, 465, 0))
>     {
>         Print("Failed to create close button");
>         return false;
>     }
>     closeButton.Text("X");
>     adminPanel.Add(closeButton);
>
>     // Quick messages
>     return CreateQuickMessageButtons();
> }
>
> //+------------------------------------------------------------------+
> //| Create quick message buttons                                     |
> //+------------------------------------------------------------------+
> bool CreateQuickMessageButtons()
> {
>     string quickMessages[8] = { QuickMessage1, QuickMessage2, QuickMessage3, QuickMessage4, QuickMessage5, QuickMessage6, QuickMessage7, QuickMessage8 };
>     int startX = 5, startY = 160, width = 222, height = 65, spacing = 5;
>
>     for (int i = 0; i < 8; i++)
>     {
>         if (!quickMessageButtons[i].Create(ChartID(), "QuickMessageButton" + IntegerToString(i + 1), 0, startX + (i % 2) * (width + spacing), startY + (i / 2) * (height + spacing), startX + (i % 2) * (width + spacing) + width, startY + (i / 2) * (height + spacing) + height))
>         {
>             Print("Failed to create quick message button ", i + 1);
>             return false;
>         }
>         quickMessageButtons[i].Text(quickMessages[i]);
>         adminPanel.Add(quickMessageButtons[i]);
>     }
>     return true;
> }
>
> //+------------------------------------------------------------------+
> //| Expert deinitialization function                                 |
> //+------------------------------------------------------------------+
> void OnDeinit(const int reason)
> {
>     adminPanel.Destroy();
>     ObjectDelete(ChartID(), BG_RECT_NAME);
>     Print("Deinitialization complete");
> }
>
> //+------------------------------------------------------------------+
> //| Handle chart events                                              |
> //+------------------------------------------------------------------+
> void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
> {
>     switch (id)
>     {
>         case CHARTEVENT_OBJECT_CLICK:
>             if (sparam == "SendButton") OnSendButtonClick();
>             else if (sparam == "ClearButton") OnClearButtonClick();
>             else if (sparam == "ChangeFontButton") OnChangeFontButtonClick();
>             else if (sparam == "ToggleThemeButton") OnToggleThemeButtonClick();
>             else if (sparam == "MinimizeButton") OnMinimizeButtonClick();
>             else if (sparam == "MaximizeButton") OnMaximizeButtonClick();
>             else if (sparam == "CloseButton") OnCloseButtonClick();
>             else if (StringFind(sparam, "QuickMessageButton") != -1)
>             {
>                 int index = StringToInteger(StringSubstr(sparam, 18));
>                 OnQuickMessageButtonClick(index - 1);
>             }
>             break;
>
>         case CHARTEVENT_OBJECT_ENDEDIT:
>             if (sparam == "InputBox") OnInputChange();
>             break;
>     }
> }
>
> //+------------------------------------------------------------------+
> //| Handle custom message send button click                          |
> //+------------------------------------------------------------------+
> void OnSendButtonClick()
> {
>     string message = inputBox.Text();
>     if (message != "")
>     {
>         if (SendMessageToTelegram(message))
>             Print("Custom message sent: ", message);
>         else
>             Print("Failed to send custom message.");
>     }
>     else
>     {
>         Print("No message entered.");
>     }
> }
>
> //+------------------------------------------------------------------+
> //| Handle clear button click                                        |
> //+------------------------------------------------------------------+
> void OnClearButtonClick()
> {
>     inputBox.Text(""); // Clear the text in the input box
>     OnInputChange();   // Update the character counter
>     Print("Input box cleared.");
> }
>
> //+------------------------------------------------------------------+
> //| Handle quick message button click                                |
> //+------------------------------------------------------------------+
> void OnQuickMessageButtonClick(int index)
> {
>     string quickMessages[8] = { QuickMessage1, QuickMessage2, QuickMessage3, QuickMessage4, QuickMessage5, QuickMessage6, QuickMessage7, QuickMessage8 };
>     string message = quickMessages[index];
>
>     if (SendMessageToTelegram(message))
>         Print("Quick message sent: ", message);
>     else
>         Print("Failed to send quick message.");
> }
>
> //+------------------------------------------------------------------+
> //| Update character counter                                         |
> //+------------------------------------------------------------------+
> void OnInputChange()
> {
>     int currentLength = StringLen(inputBox.Text());
>     charCounter.Text(IntegerToString(currentLength) + "/" + IntegerToString(MAX_MESSAGE_LENGTH));
>     ChartRedraw();
> }
>
> //+------------------------------------------------------------------+
> //| Handle toggle theme button click                                 |
> //+------------------------------------------------------------------+
> void OnToggleThemeButtonClick()
> {
>     darkTheme = !darkTheme;
>     color bgColor = darkTheme ? clrBlack : clrWhite;
>     color textColor = darkTheme ? clrWhite : clrBlack;
>
>     // Set text color appropriate to the theme
>     inputBox.Color(textColor);
>     clearButton.Color(textColor);
>     sendButton.Color(textColor);
>     toggleThemeButton.Color(textColor);
>     changeFontButton.Color(textColor);
>
>     for(int i = 0; i < ArraySize(quickMessageButtons); i++)
>     {
>         quickMessageButtons[i].Color(textColor);
>     }
>
>     charCounter.Color(textColor);
>
>     CreateOrUpdateBackground(ChartID(), bgColor);
>
>     ChartRedraw();
> }
>
> //+------------------------------------------------------------------+
> //| Create and update background rectangle                           |
> //+------------------------------------------------------------------+
> void CreateOrUpdateBackground(long chart_id, color bgColor)
> {
>     if (!ObjectFind(chart_id, BG_RECT_NAME))
>     {
>         if (!ObjectCreate(chart_id, BG_RECT_NAME, OBJ_RECTANGLE, 0, 0, 0))
>             Print("Failed to create background rectangle");
>     }
>
>     ObjectSetInteger(chart_id, BG_RECT_NAME, OBJPROP_COLOR, bgColor);
>     ObjectSetInteger(chart_id, BG_RECT_NAME, OBJPROP_BACK, true);
>     ObjectSetInteger(chart_id, BG_RECT_NAME, OBJPROP_SELECTABLE, false);
>     ObjectSetInteger(chart_id, BG_RECT_NAME, OBJPROP_SELECTED, false);
>     ObjectSetInteger(chart_id, BG_RECT_NAME, OBJPROP_HIDDEN, false);
>     ObjectSetInteger(chart_id, BG_RECT_NAME, OBJPROP_CORNER, CORNER_LEFT_UPPER);
>     ObjectSetInteger(chart_id, BG_RECT_NAME, OBJPROP_XOFFSET, 25);
>     ObjectSetInteger(chart_id, BG_RECT_NAME, OBJPROP_YOFFSET, 25);
> }
>
> //+------------------------------------------------------------------+
> //| Handle change font button click                                  |
> //+------------------------------------------------------------------+
> void OnChangeFontButtonClick()
> {
>     currentFontIndex = (currentFontIndex + 1) % ArraySize(availableFonts);
>
>     inputBox.Font(availableFonts[currentFontIndex]);
>     clearButton.Font(availableFonts[currentFontIndex]);
>     sendButton.Font(availableFonts[currentFontIndex]);
>     toggleThemeButton.Font(availableFonts[currentFontIndex]);
>     changeFontButton.Font(availableFonts[currentFontIndex]);
>
>     for(int i = 0; i < ArraySize(quickMessageButtons); i++)
>     {
>         quickMessageButtons[i].Font(availableFonts[currentFontIndex]);
>     }
>
>     Print("Font changed to: ", availableFonts[currentFontIndex]);
>     ChartRedraw();
> }
>
> //+------------------------------------------------------------------+
> //| Handle minimize button click                                     |
> //+------------------------------------------------------------------+
> void OnMinimizeButtonClick()
> {
>     minimized = true;
>     adminPanel.Hide();
>     minimizeButton.Hide();
>     maximizeButton.Show();
>     closeButton.Show();
>     Print("Panel minimized.");
> }
>
> //+------------------------------------------------------------------+
> //| Handle maximize button click                                     |
> //+------------------------------------------------------------------+
> void OnMaximizeButtonClick()
> {
>     if (minimized)
>     {
>         adminPanel.Show();
>         minimizeButton.Show();
>         maximizeButton.Hide();
>         closeButton.Hide();
>         Print("Panel maximized.");
>     }
> }
>
> //+------------------------------------------------------------------+
> //| Handle close button click                                        |
> //+------------------------------------------------------------------+
> void OnCloseButtonClick()
> {
>     ExpertRemove(); // Completely remove the EA
>     Print("Admin Panel closed.");
> }
>
> //+------------------------------------------------------------------+
> //| Send the message to Telegram                                     |
> //+------------------------------------------------------------------+
> bool SendMessageToTelegram(string message)
> {
>     string url = "https://api.telegram.org/bot" + InputBotToken + "/sendMessage";
>     string jsonMessage = "{\"chat_id\":\"" + InputChatId + "\", \"text\":\"" + message + "\"}";
>     char post_data[];
>     ArrayResize(post_data, StringToCharArray(jsonMessage, post_data, 0, WHOLE_ARRAY) - 1);
>
>     int timeout = 5000;
>     char result[];
>     string responseHeaders;
>
>     int res = WebRequest("POST", url, "Content-Type: application/json\r\n", timeout, post_data, result, responseHeaders);
>
>     if (res == 200) // HTTP 200 OK
>     {
>         Print("Message sent successfully: ", message);
>         return true;
>     }
>     else
>     {
>         Print("Failed to send message. HTTP code: ", res, " Error code: ", GetLastError());
>         Print("Response: ", CharArrayToString(result));
>         return false;
>     }
> }
> ```

> ![New Features Testing](https://c.mql5.com/2/93/terminal64_6lzmliwgzh__1.gif)
>
> New Features Tested on XAUUSD

With these, foundational styling techniques in place, we can now explore more advanced customization options that can bring even greater interactivity and visual appeal to the GUI. From the above image, we can see that our theme is only working on foreground text, yet we want it to also affect the panel background. In the next segment, we will address ways to resolve this issue.

### Advanced Enhancement of the GUI

> **Extending the Dialog Class for Theme Management:**
>
> To extend the **Dialog** for theme management, we can customize the existing dialog class to support dynamic theme changes, similar to how we manage themes in the Admin Panel. This would involve modifying or subclassing the **CDialog** class to include properties for background and text colors, as well as methods for applying different themes (light or dark). By overriding the constructor or adding methods like ApplyTheme, we can ensure that dialog boxes created using this class respond to theme changes without recreating the dialog objects.

![Customizing the Dialog class](https://c.mql5.com/2/93/ShareX_ncbUIzQKWq__1.gif)

Customizing the colors on Dialog class

**Why is it important?**

Extending the Dialog class for theme management allows for a more seamless and cohesive user experience across all UI elements, not just the Admin Panel. It ensures that all parts of the application—including dialog boxes—adhere to the chosen theme, enhancing both usability and visual consistency. This feature becomes particularly important in trading applications where users may spend extended periods interacting with the interface, and customizable themes can reduce eye strain and improve overall user satisfaction.

### ![Background theme after modifying Dialog class](https://c.mql5.com/2/93/New_theme.PNG)

Admin Panel: Background theme after modifying the Dialog class

**Other options:**

While extending Dialog class is a direct and flexible approach, another option is to apply theme management at a higher level. For example, we could create a global theme management system that automatically updates the properties of all UI elements, including dialogs, without requiring changes to individual components. Additionally, leveraging external libraries or designing a custom dialog framework might offer more granular control over UI elements if specific styling needs arise.

### **CEdit Class**

According to search on Google, the maximum length of a Telegram message is 4096 characters and it must be **UTF-8** encoded. When trying to implement the value in this project, we were limited to a maximum of 63 characters and the problem must be within the limitation of the **CEdit** class that we will address in the next article.

When editing MQL5 library files, exercise caution, as improper modifications can lead to compilation errors or runtime issues due to the strict syntax and structure of the language, which is object-oriented. Always back up original files before making changes, and ensure you understand the class relationships involved to avoid breaking functionality. After editing, compile your code to check for errors and thoroughly test the changes in a demo environment to verify stability. If you encounter significant issues that you cannot resolve, consider reinstalling the MetaTrader 5 platform to restore default settings and files.

### Conclusion

In conclusion, our implementation of font and theme management in the Admin Panel program demonstrated promising results. While we faced limitations with the static background of the dialog class, the text foreground successfully adapted to theme changes, providing an improved user experience. The dynamic font management also worked well, allowing users to switch between different fonts with ease.

Moving forward, our next endeavor will be to extend the dialog class to fully support theme changes, including dynamic background updates. This enhancement aims to overcome the current limitations and provide a more cohesive and visually appealing interface. Stay tuned as we tackle these challenges in our upcoming articles!

Give these styling techniques a try on your own trading panels, and explore additional customization options in MQL5. I’d love to hear about your experiences and insights, so feel free to share them in the comments below as we dive into more advanced GUI design challenges. The source file for this project is attached—feel free to review it.

[Contents](https://www.mql5.com/en/articles/15419#para)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15419.zip "Download all attachments in the single ZIP archive")

[Admin\_Panel.mq5](https://www.mql5.com/en/articles/download/15419/admin_panel.mq5 "Download Admin_Panel.mq5")(14.67 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/473066)**
(1)


![SERGEI NAIDENOV](https://c.mql5.com/avatar/2019/11/5DE2C04F-1280.jpg)

**[SERGEI NAIDENOV](https://www.mql5.com/en/users/leonsi)**
\|
23 May 2025 at 19:09

when trying to compile it generates a warning:

[possible loss of data](https://www.mql5.com/en/docs/basis/types/casting "MQL5 Documentation: Type conversion") due to type conversion from 'long' to 'int' Admin\_Panel.mq5 208 27

line of code:

```
int index = StringToInteger(StringSubstr(sparam, 18));
```

if you do it this way, everything is without errors:

```
int index = int(StringToInteger(StringSubstr(sparam, 18)));
```

if you attach to a chart and try to minimise it, then expand it... the button of closing "X" (deleting the Expert Advisor) does not work in the expanded state of the panel. If you minimise it, it works.

question: moving the window on the chart is not implemented?


![Example of CNA (Causality Network Analysis), SMOC (Stochastic Model Optimal Control) and Nash Game Theory with Deep Learning](https://c.mql5.com/2/94/Example_of_CNA_b_SMOC_and_Nash_Game__LOGO2.png)[Example of CNA (Causality Network Analysis), SMOC (Stochastic Model Optimal Control) and Nash Game Theory with Deep Learning](https://www.mql5.com/en/articles/15819)

We will add Deep Learning to those three examples that were published in previous articles and compare results with previous. The aim is to learn how to add DL to other EA.

![Developing a multi-currency Expert Advisor (Part 10): Creating objects from a string](https://c.mql5.com/2/77/Developing_a_multi-currency_advisor_2Part_101___LOGO.png)[Developing a multi-currency Expert Advisor (Part 10): Creating objects from a string](https://www.mql5.com/en/articles/14739)

The EA development plan includes several stages with intermediate results being saved in the database. They can only be retrieved from there again as strings or numbers, not objects. So we need a way to recreate the desired objects in the EA from the strings read from the database.

![Turtle Shell Evolution Algorithm (TSEA)](https://c.mql5.com/2/77/Turtle_Shell_Evolution_Algorithm___LOGO.png)[Turtle Shell Evolution Algorithm (TSEA)](https://www.mql5.com/en/articles/14789)

This is a unique optimization algorithm inspired by the evolution of the turtle shell. The TSEA algorithm emulates the gradual formation of keratinized skin areas, which represent optimal solutions to a problem. The best solutions become "harder" and are located closer to the outer surface, while the less successful solutions remain "softer" and are located inside. The algorithm uses clustering of solutions by quality and distance, allowing to preserve less successful options and providing flexibility and adaptability.

![How to Implement Auto Optimization in MQL5 Expert Advisors](https://c.mql5.com/2/93/Implementing_Auto_Optimization_in_MQL5_Expert_Advisors__LOGO.png)[How to Implement Auto Optimization in MQL5 Expert Advisors](https://www.mql5.com/en/articles/15837)

Step by step guide for auto optimization in MQL5 for Expert Advisors. We will cover robust optimization logic, best practices for parameter selection, and how to reconstruct strategies with back-testing. Additionally, higher-level methods like walk-forward optimization will be discussed to enhance your trading approach.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=pjmlstqlyxhhysctrqbkyqohzinwgmjt&ssn=1769094017244793155&ssn_dr=0&ssn_sr=0&fv_date=1769094017&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15419&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creating%20a%20Trading%20Administrator%20Panel%20in%20MQL5%20(Part%20III)%3A%20Enhancing%20the%20GUI%20with%20Visual%20Styling%20(I)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909401704455447&fz_uniq=5049553669701086592&sv=2552)

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