---
title: Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (III): Communication Module
url: https://www.mql5.com/en/articles/17044
categories: Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:28:19.763640
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/17044&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068276286842926978)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/17044#para1)
- [Developing the CommunicationsDialog class.](https://www.mql5.com/en/articles/17044#para2)
- [Integration of CommunicationsDialog with other header files and the main program](https://www.mql5.com/en/articles/17044#para3)
- [Testing and Results](https://www.mql5.com/en/articles/17044#para4)
- [Conclusion](https://www.mql5.com/en/articles/17044#para5)

### Introduction

Today, we aim to expand our New Admin Panel from where we left off in the [previous article](https://www.mql5.com/en/articles/16562), where we introduced modularization as a key aspect of broader code organization. We introduced the _AdminHomeDialog_ class, responsible for creating the Admin Home interface. This home panel serves as the central hub for accessing various features and consists of access control buttons leading to three main component panels:

- Trade Management Panel
- Communications Panel
- Analytics Panel

These are not the final limits of the system, as new features remain feasible as we continue refining and expanding the existing foundation. In this discussion, we focus specifically on the Communications Panel as a module, enhancing it further from its previous version within the monolithic Admin Panel.

Key takeaways from this discussion:

- Understanding classes in MQL5
- Developing header files
- Inheriting from built-in classes
- Utilizing the ListView header file
- Applying colors for better UI design

To recap from the previous article, modularization refers to breaking an application into distinct, independent units (modules), each defined in its own file. These modules can be developed and maintained separately. In this case, the main program is an **.mq5** file ( _New\_Admin\_Panel.mq5_), while the dialogs are classes defined in **.mqh** header files ( _AdminHomeDialog.mqh_ and _CommunicationsDialog.mqh_). The structure involves the main program creating an instance of _AdminHomeDialog_, which, in turn, can create instances of _CommunicationsDialog_ and any potential future dialogs. See the diagram below.

![Modular (New_Admin_Panel) Flow.](https://c.mql5.com/2/123/New_Admin_Panel_Flow.png)

New Admin Panel modular flow

1\. Main Program (New\_Admin\_Panel.mq5)

- This is the entry point of the application.
- It initializes the system and creates an instance of _AdminHomeDialog,_ which serves as the primary user interface.

2\. AdminHomeDialog

- Acts as the central hub for user interaction.
- Responds to events, such as button clicks, to create instances of other dialogs like CommunicationsDialog.
- Designed to be extensible, allowing it to spawn additional dialogs (e.g., FutureDialog) as needed.

3\. CommunicationsDialog

- A specialized dialog responsible for specific functionality, such as sending messages via the Telegram API.
- Created dynamically by _AdminHomeDialog_ when triggered by a user action (e.g., clicking a "Send Message" button).

4\. Event-Driven Interactions

- The system uses an event-driven approach, common in MQL5 applications. For example, a button click in _AdminHomeDialog_ triggers the creation and display of _CommunicationsDialog_, which then performs its task (e.g., interacting with Telegram).

5\. Modular Design

- Each component has a distinct role: the main program initializes, _AdminHomeDialog_ manages the interface, and _CommunicationsDialog_ handles communication tasks. This modularity allows for easy expansion or modification.

With the above introduction and overview of our new program, we can now dive into the details of developing our pioneering module, CommunicationsDialog. In Part (I), this was just a basic communication interface, but now we have expanded its functionality by incorporating new features. With this foundational knowledge, we can better visualize our direction. In the next subsections, we will explore the essential building blocks that make our custom classes possible.

### Developing the CommunicationsDialog class

To enhance understanding, I have included an image below that illustrates the hierarchy of flow from the base class to our custom classes. This approach is a powerful way to promote code reusability. Throughout the project, there are many panel components, each serving a distinct purpose. This uniqueness makes each code component adaptable for integration into other projects as well.

In MQL5, the Dialog class typically refers to _CAppDialog_ or _CDialog_ from the (Dialog.mqh) include file, as these serve as the foundational dialog classes in the standard library.

Both _CommunicationsDialog_ and _CAdminHomeDialog_ inherit from _CAppDialog_, creating a structured hierarchy where multiple dialog classes share a common base for dialog functionality. This structure will be represented in the hierarchy flowchart below.

![Building Custom classes from Dialog](https://c.mql5.com/2/123/inheritence.png)

Relationship of the base class and custom classes

To get started, open MetaEditor 5 from your desktop or launch it from the terminal by pressing F4.

In the Navigator, locate Dialog.mqh in the Includes folder and open it for reference.

Follow the image below for guidance.

![Locating the Dialog.mqh](https://c.mql5.com/2/123/MetaEditor64_dLbJb0sH47.gif)

Locating the Dialog class in MetaEditor 5

Next, create a new file for developing the new class.

Generally, our program is composed of sections like header setup, layout and color definitions, class declaration, and method implementations. We will start with the basics of our file. The header comments at the top tell us what this file is (CommunicationsDialog.mqh), who owns it (MetaQuotes Ltd., though you can change this to your name), and where it’s from (MQL5 community). These comments are like a title page for your code, helping others identify its purpose.

Next, we use _#ifndef_ and _#define_ to create a "guard" called _COMMUNICATIONS\_DIALOG\_MQH_. This prevents the file from being included multiple times in a project, which could cause errors. Think of it as a lock that says, "If I’ve already been opened, don’t open me again."

The #include lines bring in tools we need from the MQL5 library. The (Dialog.mqh) gives us the base dialog class (CAppDialog), while (Button.mqh) and (Edit.mqh) provide button and text box classes. (Label.mqh) and (ListView.mqh) add a label and a list for quick messages. Finally, (Telegram.mqh) is a custom file (assumed to exist) that handles Telegram messaging. These are like borrowing tools from a toolbox to build our dialog. Below is the code snippet for this section.

**Section 1: File Header and Includes**

```
//+------------------------------------------------------------------+
//|                                         CommunicationsDialog.mqh |
//|                             Copyright 2000-2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#ifndef COMMUNICATIONS_DIALOG_MQH
#define COMMUNICATIONS_DIALOG_MQH

#include <Controls\Dialog.mqh>
#include <Controls\Button.mqh>
#include <Controls\Edit.mqh>
#include <Controls\Label.mqh>
#include <Controls\ListView.mqh>
#include "Telegram.mqh"
```

**Section 2: Layout and Color Definitions**

Before we build the dialog, we need to plan its size and look. The #define statements are like setting up a blueprint with measurements and colors. For layout, COMMS\_PANEL\_WIDTH (300 pixels) and COMMS\_PANEL\_HEIGHT (350 pixels) decide how big our dialog will be on the screen. Margins (COMMS\_MARGIN\_LEFT, COMMS\_MARGIN\_TOP, COMMS\_MARGIN\_RIGHT) give padding around the edges, while COMMS\_GAP\_VERTICAL adds space between items vertically. Each control has its own size: the input box is tall (COMMS\_INPUT\_HEIGHT), buttons are small rectangles (COMMS\_BUTTON\_WIDTH and HEIGHT), and the list view and label have their own dimensions.

Colors make the dialog pretty and readable. We use hexadecimal numbers (like 0x808080 for dark gray) because that’s how computers understand colors in MQL5. CLR\_PANEL\_BG sets the dialog’s main background, CLR\_CLIENT\_BG colors the area where controls sit, and CLR\_CAPTION\_BG and CLR\_CAPTION\_TEXT style the title bar. Borders get CLR\_BORDER\_BG and CLR\_BORDER, while controls like the input box (CLR\_INPUT\_BG, CLR\_INPUT\_TEXT) and buttons (CLR\_SEND\_BG, CLR\_CLEAR\_BG) get their own colors. Comments next to each define explain what they do, making it easy to tweak later.

```
// **Layout Defines**
#define COMMS_PANEL_WIDTH       300
#define COMMS_PANEL_HEIGHT      350
#define COMMS_MARGIN_LEFT       10
#define COMMS_MARGIN_TOP        10
#define COMMS_MARGIN_RIGHT      10
#define COMMS_GAP_VERTICAL      10
#define COMMS_INPUT_HEIGHT      30
#define COMMS_BUTTON_WIDTH      80
#define COMMS_BUTTON_HEIGHT     30
#define COMMS_LISTVIEW_WIDTH    280
#define COMMS_LISTVIEW_HEIGHT   80
#define COMMS_LABEL_HEIGHT      20

// **Color Defines (Hexadecimal Values)**
#define CLR_PANEL_BG      0x808080  // Dark Gray (Dialog background)
#define CLR_CLIENT_BG     0xD3D3D3  // Light Gray (Client area background)
#define CLR_CAPTION_BG    0x404040  // Darker Gray (Caption background)
#define CLR_CAPTION_TEXT  0xFFFFFF  // White (Caption text)
#define CLR_BORDER_BG     0xFFFFFF  // White (Border background)
#define CLR_BORDER        0xA9A9A9  // Gray (Border color)
#define CLR_INPUT_BG      0xFFFFFF  // White (Input box background)
#define CLR_INPUT_TEXT    0x000000  // Black (Input box text)
#define CLR_SEND_BG       0x00FF00  // Lime Green (Send button background)
#define CLR_CLEAR_BG      0xF08080  // Light Coral (Clear button background)
#define CLR_BUTTON_TEXT   0x000000  // Black (Button text)
#define CLR_LABEL_TEXT    0xFFFFFF  // White (Label text)
#define CLR_LIST_BG       0xFFFFFF  // White (List view background)
#define CLR_LIST_TEXT     0x000000  // Black (List view text)
```

**Section 3: Class Declaration**

Now we’re defining the heart of our dialog: the CCommunicationDialog class. Think of a class as a recipe for making a dialog object. We say : public _CAppDialog_ because our dialog builds on _CAppDialog_, a ready-made dialog class from MQL5 that gives us basic dialog features like a title bar and borders.

The private section lists the ingredients we’ll use inside the dialog. The _m\_inputBox_ is a text field where users type messages, _m\_sendButton_ and _m\_clearButton_ are buttons to send or clear the message, _m\_quickMsgLabel_ is a text label, and _m\_quickMessageList_ is a list of preset messages. We also store _m\_chatId_ and _m\_botToken_ as strings for Telegram, and m\_quickMessages as an array to hold eight quick message options.

In the public section, we list functions anyone can use to interact with our dialog. The constructor ( _CCommunicationDialog_) sets up the dialog with a chat ID and bot token, and the destructor _(~CCommunicationDialog)_ cleans up when we’re done. Create builds the dialog on the chart, OnEvent handles clicks and actions, and Toggle shows or hides it.

Back in private, we have helper functions to create each control (CreateInputBox, etc.) and event handlers (OnClickSend, OnClickClear) to decide what happens when buttons are clicked. These are private because they’re internal details of how the dialog works.

```
//+------------------------------------------------------------------+
//| Class CCommunicationDialog                                       |
//| Purpose: A dialog for sending Telegram messages with controls    |
//+------------------------------------------------------------------+
class CCommunicationDialog : public CAppDialog
{
private:
   CEdit         m_inputBox;           // Field to edit/send message
   CButton       m_sendButton;         // Send message button
   CButton       m_clearButton;        // Clear edit box button
   CLabel        m_quickMsgLabel;      // Label for "QuickMessages"
   CListView     m_quickMessageList;   // ListView for quick messages
   string        m_chatId;             // Telegram chat ID
   string        m_botToken;           // Telegram bot token
   string        m_quickMessages[8];   // Array of quick messages

public:
   CCommunicationDialog(const string chatId, const string botToken);
   ~CCommunicationDialog();

   virtual bool Create(const long chart, const string name, const int subwin,
                       const int x1, const int y1, const int x2, const int y2);
   virtual bool OnEvent(const int id, const long &lparam, const double &dparam, const string &sparam);
   void Toggle();                      // Toggle dialog visibility

private:
   //--- Create dependent controls
   bool CreateInputBox(void);
   bool CreateClearButton(void);
   bool CreateSendButton(void);
   bool CreateQuickMsgLabel(void);
   bool CreateQuickMessageList(void);

   //--- Handlers of dependent controls events
   void OnClickSend(void);      // Handler for Send button
   void OnClickClear(void);     // Handler for Clear button
};
```

**Section 4: Constructor and Destructor**

The constructor is like setting up a new toy before playing with it. When someone creates a _CCommunicationDialog_, they must give it a _chatId_ and botToken (strings for Telegram). The m\_chatId(chatId), and m\_botToken(botToken) part copies these into our private variables so the dialog knows where to send messages. Inside the curly braces, we fill the _m\_quickMessages_ array with eight handy phrases users can pick from. This happens when the dialog is born, so it’s ready to go.

The destructor is the cleanup crew. It runs when the dialog is deleted (like when you close the program). Currently, it’s empty because CAppDialog handles most cleanup for us, and we don’t have anything extra to tidy up. It’s here as a placeholder in case we add special cleanup later, like freeing memory.

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CCommunicationDialog::CCommunicationDialog(const string chatId, const string botToken)
   : m_chatId(chatId), m_botToken(botToken)
{
   // Initialize quick messages
   m_quickMessages[0] = "Updates";
   m_quickMessages[1] = "Close all";
   m_quickMessages[2] = "In deep profits";
   m_quickMessages[3] = "Hold position";
   m_quickMessages[4] = "Swing Entry";
   m_quickMessages[5] = "Scalp Entry";
   m_quickMessages[6] = "Book profit";
   m_quickMessages[7] = "Invalid Signal";
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CCommunicationDialog::~CCommunicationDialog()
{
}
```

**Section 5: Create Method**

The Create method is where we build the dialog on the screen, like assembling a toy from parts. It takes inputs like chart (where it appears), name (a unique ID), subwin (which chart window), and coordinates (x1, y1, x2, y2) for position and size. We call CAppDialog::Create first to set up the basic dialog structure. If that fails, we return false to say something went wrong.

Next, Caption("Communications Panel") sets the title at the top of the dialog. Then, we color it up! ObjectSetInteger changes colors by talking directly to the dialog’s parts (like “Back” for background, “Client” for the control area). We use our color defines (e.g., CLR\_PANEL\_BG) to make it look nice. The if(!m\_panel\_flag) check adds borders only if the dialog isn’t a special type (controlled by m\_panel\_flag, a variable from CAppDialog).

After that, we call helper functions to create each control (input box, buttons, etc.). If any fail, we stop and return false. Finally, we put the first quick message in the input box with m\_inputBox.Text and call ChartRedraw() to show everything on the chart. Returning true means success.

```
//+------------------------------------------------------------------+
//| Create Method                                                    |
//| Initializes the dialog and its controls with full color styling  |
//+------------------------------------------------------------------+
bool CCommunicationDialog::Create(const long chart, const string name, const int subwin,
                                  const int x1, const int y1, const int x2, const int y2)
{
   // Create the base dialog
   if(!CAppDialog::Create(chart, name, subwin, x1, y1, x2, y2))
      return(false);

   Caption("Communications Panel"); // Set the title

   // Set dialog background color
   ObjectSetInteger(m_chart_id, m_name + "Back", OBJPROP_BGCOLOR, CLR_PANEL_BG);

   // Set client area background color
   ObjectSetInteger(m_chart_id, m_name + "Client", OBJPROP_BGCOLOR, CLR_CLIENT_BG);

   // Set caption colors
   ObjectSetInteger(m_chart_id, m_name + "Caption", OBJPROP_BGCOLOR, CLR_CAPTION_BG);
   ObjectSetInteger(m_chart_id, m_name + "Caption", OBJPROP_COLOR, CLR_CAPTION_TEXT);

   // Set border colors (if border exists, i.e., m_panel_flag is false)
   if(!m_panel_flag)
   {
      ObjectSetInteger(m_chart_id, m_name + "Border", OBJPROP_BGCOLOR, CLR_BORDER_BG);
      ObjectSetInteger(m_chart_id, m_name + "Border", OBJPROP_BORDER_COLOR, CLR_BORDER);
   }

   // Create all controls
   if(!CreateInputBox())
      return(false);
   if(!CreateClearButton())
      return(false);
   if(!CreateSendButton())
      return(false);
   if(!CreateQuickMsgLabel())
      return(false);
   if(!CreateQuickMessageList())
      return(false);

   // Set initial text in input box
   m_inputBox.Text(m_quickMessages[0]);
   ChartRedraw();
   return(true);
}
```

**Section 6: Control Creation Methods**

These methods are like assembling the pieces of our dialog. Each one creates a control (input box, buttons, label, list) and places it on the dialog. They all return true if successful or false if something fails, so we can stop if there’s a problem.

For CreateInputBox, we calculate positions using our layout defines (e.g., COMMS\_MARGIN\_LEFT). The input box is wide (using ClientAreaWidth()) and tall (three times COMMS\_INPUT\_HEIGHT). We call m\_inputBox.Create with the chart ID, a unique name (m\_name + "\_InputBox"), and coordinates. Add puts it on the dialog, and ObjectSetInteger sets its colors.

CreateClearButton and CreateSendButton make buttons below the input box. We stack them vertically with COMMS\_GAP\_VERTICAL and position Send next to Clear using clear\_button\_x2. Each gets a name, text ("Clear" or "Send"), and colors from our defines.

CreateQuickMsgLabel adds a label below the buttons, using more spacing calculations. It’s just text, so it only needs a text color. CreateQuickMessageList makes a list view even lower, filling it with our quick messages from the constructor. Each control uses the same pattern: create, add, color, and check for errors with Print messages to help debug.

```
//+------------------------------------------------------------------+
//| CreateInputBox                                                   |
//+------------------------------------------------------------------+
bool CCommunicationDialog::CreateInputBox(void)
{
   int x1 = COMMS_MARGIN_LEFT;
   int y1 = COMMS_MARGIN_TOP;
   int x2 = ClientAreaWidth() - COMMS_MARGIN_RIGHT;
   int y2 = y1 + 3 * COMMS_INPUT_HEIGHT;

   if(!m_inputBox.Create(m_chart_id, m_name + "_InputBox", m_subwin, x1, y1, x2, y2))
   {
      Print("Failed to create InputBox");
      return(false);
   }
   if(!Add(m_inputBox))
      return(false);

   ObjectSetInteger(m_chart_id, m_name + "_InputBox", OBJPROP_BGCOLOR, CLR_INPUT_BG);
   ObjectSetInteger(m_chart_id, m_name + "_InputBox", OBJPROP_COLOR, CLR_INPUT_TEXT);
   return(true);
}

//+------------------------------------------------------------------+
//| CreateClearButton                                                |
//+------------------------------------------------------------------+
bool CCommunicationDialog::CreateClearButton(void)
{
   int button_y1 = COMMS_MARGIN_TOP + 3 * COMMS_INPUT_HEIGHT + COMMS_GAP_VERTICAL;
   int x1 = COMMS_MARGIN_LEFT;
   int x2 = x1 + COMMS_BUTTON_WIDTH;
   int y2 = button_y1 + COMMS_BUTTON_HEIGHT;

   if(!m_clearButton.Create(m_chart_id, m_name + "_ClearButton", m_subwin, x1, button_y1, x2, y2))
   {
      Print("Failed to create ClearButton");
      return(false);
   }
   m_clearButton.Text("Clear");
   if(!Add(m_clearButton))
      return(false);

   ObjectSetInteger(m_chart_id, m_name + "_ClearButton", OBJPROP_BGCOLOR, CLR_CLEAR_BG);
   ObjectSetInteger(m_chart_id, m_name + "_ClearButton", OBJPROP_COLOR, CLR_BUTTON_TEXT);
   return(true);
}

//+------------------------------------------------------------------+
//| CreateSendButton                                                 |
//+------------------------------------------------------------------+
bool CCommunicationDialog::CreateSendButton(void)
{
   int button_y1 = COMMS_MARGIN_TOP + 3 * COMMS_INPUT_HEIGHT + COMMS_GAP_VERTICAL;
   int clear_button_x2 = COMMS_MARGIN_LEFT + COMMS_BUTTON_WIDTH;
   int x1 = clear_button_x2 + COMMS_GAP_VERTICAL;
   int x2 = x1 + COMMS_BUTTON_WIDTH;
   int y2 = button_y1 + COMMS_BUTTON_HEIGHT;

   if(!m_sendButton.Create(m_chart_id, m_name + "_SendButton", m_subwin, x1, button_y1, x2, y2))
   {
      Print("Failed to create SendButton");
      return(false);
   }
   m_sendButton.Text("Send");
   if(!Add(m_sendButton))
      return(false);

   ObjectSetInteger(m_chart_id, m_name + "_SendButton", OBJPROP_BGCOLOR, CLR_SEND_BG);
   ObjectSetInteger(m_chart_id, m_name + "_SendButton", OBJPROP_COLOR, CLR_BUTTON_TEXT);
   return(true);
}

//+------------------------------------------------------------------+
//| CreateQuickMsgLabel                                              |
//+------------------------------------------------------------------+
bool CCommunicationDialog::CreateQuickMsgLabel(void)
{
   int label_y1 = COMMS_MARGIN_TOP + 3 * COMMS_INPUT_HEIGHT + COMMS_GAP_VERTICAL +
                  COMMS_BUTTON_HEIGHT + COMMS_GAP_VERTICAL;
   int x1 = COMMS_MARGIN_LEFT;
   int x2 = x1 + COMMS_LISTVIEW_WIDTH;
   int y2 = label_y1 + COMMS_LABEL_HEIGHT;

   if(!m_quickMsgLabel.Create(m_chart_id, m_name + "_QuickMsgLabel", m_subwin, x1, label_y1, x2, y2))
   {
      Print("Failed to create QuickMessages Label");
      return(false);
   }
   m_quickMsgLabel.Text("QuickMessages");
   if(!Add(m_quickMsgLabel))
      return(false);

   ObjectSetInteger(m_chart_id, m_name + "_QuickMsgLabel", OBJPROP_COLOR, CLR_LABEL_TEXT);
   return(true);
}

//+------------------------------------------------------------------+
//| CreateQuickMessageList                                           |
//+------------------------------------------------------------------+
bool CCommunicationDialog::CreateQuickMessageList(void)
{
   int list_y1 = COMMS_MARGIN_TOP + 3 * COMMS_INPUT_HEIGHT + COMMS_GAP_VERTICAL +
                 COMMS_BUTTON_HEIGHT + COMMS_GAP_VERTICAL +
                 COMMS_LABEL_HEIGHT + COMMS_GAP_VERTICAL;
   int x1 = COMMS_MARGIN_LEFT;
   int x2 = x1 + COMMS_LISTVIEW_WIDTH;
   int y2 = list_y1 + COMMS_LISTVIEW_HEIGHT;

   if(!m_quickMessageList.Create(m_chart_id, m_name + "_QuickMsgList", m_subwin, x1, list_y1, x2, y2))
   {
      Print("Failed to create ListView");
      return(false);
   }
   if(!Add(m_quickMessageList))
      return(false);

   ObjectSetInteger(m_chart_id, m_name + "_QuickMsgList", OBJPROP_BGCOLOR, CLR_LIST_BG);
   ObjectSetInteger(m_chart_id, m_name + "_QuickMsgList", OBJPROP_COLOR, CLR_LIST_TEXT);

   for(int i = 0; i < ArraySize(m_quickMessages); i++)
   {
      if(!m_quickMessageList.AddItem("Message: " + m_quickMessages[i]))
         return(false);
   }

   return(true);
}
```

**Section 7: Toggle and Event Handling**

Toggle is a simple switch to show or hide the dialog. IsVisible() checks if it’s on the screen. If it is, Hide() makes it disappear; if not, Show() brings it back. ChartRedraw() updates the chart so you see the change right away. It’s like flipping a light switch for the dialog.

OnEvent is the brain that listens for user actions, like clicks. It gets an id (what happened), lparam (details), dparam (more details), and sparam (which object). If id is CHARTEVENT\_OBJECT\_CLICK, it means something was clicked. We check if sparam (the clicked object’s name) matches m\_sendButton.Name() or m\_clearButton.Name(), then call OnClickSend or OnClickClear. Returning true says we handled it.

If id is ON\_CHANGE and sparam is the list’s name, the user picked a quick message. lparam tells us which one (as a number), and we put that message in the input box with m\_inputBox.Text. If nothing matches, we let CAppDialog::OnEvent handle it, passing the event up the chain.

```
//+------------------------------------------------------------------+
//| Toggle                                                           |
//+------------------------------------------------------------------+
void CCommunicationDialog::Toggle()
{
   if(IsVisible())
      Hide();
   else
      Show();
   ChartRedraw();
}

//+------------------------------------------------------------------+
//| OnEvent                                                          |
//+------------------------------------------------------------------+
bool CCommunicationDialog::OnEvent(const int id, const long &lparam,
                                   const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_OBJECT_CLICK)
   {
      if(sparam == m_sendButton.Name())
      {
         OnClickSend();
         return true;
      }
      else if(sparam == m_clearButton.Name())
      {
         OnClickClear();
         return true;
      }
   }
   else if(id == ON_CHANGE && sparam == m_quickMessageList.Name())
   {
      int selectedIndex = (int)lparam;
      if(selectedIndex >= 0 && selectedIndex < ArraySize(m_quickMessages))
      {
         m_inputBox.Text(m_quickMessages[selectedIndex]);
      }
      return true;
   }
   return CAppDialog::OnEvent(id, lparam, dparam, sparam);
}
```

**Section 8: Button Event Handlers**

OnClickSend runs when the "Send" button is clicked. It grabs the text from m\_inputBox.Text() and checks if it’s not empty (""). If there’s a message, it tries to send it using SendMessageToTelegram (a function from Telegram.mqh) with our m\_chatId and m\_botToken. If it works, we print a success message; if not, a failure one. If the box is empty, we just print a note. This is the dialog’s main job—sending messages!

OnClickClear is simpler. When "Clear" is clicked, it sets the input box text to nothing ("") with m\_inputBox.Text(""), redraws the chart to show the change, and prints a confirmation. It’s like hitting reset on a form.

```
//+------------------------------------------------------------------+
//| OnClickSend                                                      |
//+------------------------------------------------------------------+
void CCommunicationDialog::OnClickSend()
{
   string message = m_inputBox.Text();
   if(message != "")
   {
      if(SendMessageToTelegram(message, m_chatId, m_botToken))
         Print("Message sent to Telegram: ", message);
      else
         Print("Failed to send message to Telegram");
   }
   else
   {
      Print("No message to send - input box is empty");
   }
}

//+------------------------------------------------------------------+
//| OnClickClear                                                     |
//+------------------------------------------------------------------+
void CCommunicationDialog::OnClickClear()
{
   m_inputBox.Text("");  // Clear the input box
   ChartRedraw();
   Print("Input box cleared.");
}
```

At the end of the file, add #endif to close the header guard:

```
#endif // COMMUNICATIONS_DIALOG_MQH
```

This matches the #ifndef at the start, wrapping everything up neatly.

### Integration of CommunicationsDialog with other header files and the main program

The _CommunicationsDialog_ is handled within the _AdminHomeDialog_. In this section, I will explain how it works. I won’t cover the full code of _AdminHomeDialog_ since we discussed it thoroughly in the [previous article](https://www.mql5.com/en/articles/16562).

Step 1

To connect _CommunicationsDialog_ to _AdminHomeDialog_, we need to bring it into our file. The #include (CommunicationsDialog.mqh) line is like opening a door between the two. It tells MQL5 to load the CommunicationsDialog.mqh file, making its _CCommunicationDialog_ class available for us to use. Place this near the top of (AdminHomeDialog.mqh), after other includes like (Dialog.mqh) and (Button.mqh). Without this, _AdminHomeDialog_ wouldn’t know about the communication panel, so it’s the first step to linking them.

```
#include <CommunicationsDialog.mqh>  // Use the enhanced Communications dialog
```

Step 2

Inside the CAdminHomeDialog class, we need a way to hold our communication panel. We add CCommunicationDialog \*m\_commPanel in the private section, using a pointer (the \* means it’s a reference to an object we’ll create later). This is like reserving a spot for a toy we’ll unpack when needed. We also add m\_chatId and m\_botToken as string variables to store Telegram details, which we’ll pass to CCommunicationDialog. These are private because only this class needs to manage them, setting the stage for the integration.

```
class CAdminHomeDialog : public CAppDialog
{
private:
   CCommunicationDialog *m_commPanel; // Pointer to the Communications panel
   string              m_chatId;      // Telegram Chat ID
   string              m_botToken;    // Telegram Bot Token

///.................Space for other members e.g. buttons
};
```

Step 3

The constructor sets up CAdminHomeDialog when it’s created. We update it to take chatId and botToken as inputs, which we copy into m\_chatId and m\_botToken using the : m\_chatId(chatId), m\_botToken(botToken) part. We also set m\_commPanel to NULL (nothing yet) with : m\_commPanel(NULL). This means we won’t make the communication panel right away—we’ll wait until the user asks for it. It’s like keeping a box closed until you’re ready to play with what’s inside.

The destructor cleans up when the dialog is done. We check if(m\_commPanel) to see if we’ve made a communication panel. If so, delete m\_commPanel frees its memory (like throwing away a used toy), and m\_commPanel = NULL ensures we don’t try to use it again by mistake. This keeps our program neat and prevents crashes when connecting to CommunicationsDialog.

```
CAdminHomeDialog::CAdminHomeDialog(string chatId, string botToken)
   : m_commPanel(NULL), m_chatId(chatId), m_botToken(botToken)
{
}

CAdminHomeDialog::~CAdminHomeDialog(void)
{
   if(m_commPanel)
   {
      delete m_commPanel;
      m_commPanel = NULL;
   }
}
```

Step 4

This is where the magic happens— _OnClickCommunications_ launches the communication panel when the "Communications" button is clicked. First, we check if(m\_commPanel == NULL) to see if it’s not created yet. If it’s NULL, we use new CCommunicationDialog(m\_chatId, m\_botToken) to make a new one, passing our stored Telegram details. This is like opening that toy box and setting it up with the right instructions.

If new fails (maybe the computer ran out of space), m\_commPanel stays NULL, and we print an error and stop. Otherwise, we call m\_commPanel.Create to build it on the chart. We use m\_chart\_id (the chart we’re on), "CommPanel" as a name, m\_subwin (which window), and coordinates (20, 435 for the top-left corner, 300 wide and 350 tall for size). If Create fails, we print an error, delete it, and reset m\_commPanel to NULL.

If it’s already made or just created successfully, m\_commPanel.Toggle() flips it on or off—showing it if hidden, hiding it if shown. This “lazy creation” means we only build it when the user clicks, saving resources until it’s needed.

```
void CAdminHomeDialog::OnClickCommunications()
{
   if(m_commPanel == NULL)
   {
      m_commPanel = new CCommunicationDialog(m_chatId, m_botToken); // Pass chatId and botToken
      if(m_commPanel == NULL)
      {
         Print("Error: Failed to allocate Communications panel");
         return;
      }
      if(!m_commPanel.Create(m_chart_id, "CommPanel", m_subwin, 20, 435, 20 + 300, 435 + 350))
      {
         Print("Error: Failed to create Communications panel");
         delete m_commPanel;
         m_commPanel = NULL;
         return;
      }
   }
   m_commPanel.Toggle();
}
```

Step 5

OnEvent listens for user actions like clicks and connects _CommunicationsDialog_ by passing events to it. When id is CHARTEVENT\_OBJECT\_CLICK, we check _if sparam_ (the clicked object’s name) matches _m\_commButton.Name()_. If it does, we print a message and call _OnClickCommunications_ to open the panel, returning true to say we handled it.

The key integration part is the _else if(m\_commPanel != NULL && m\_commPanel.IsVisible())_ block. If the communication panel exists _(!= NULL)_ and is on-screen _(IsVisible())_, we send the event (id, lparam, etc.) to _m\_commPanel.OnEvent_. This lets CommunicationsDialog handle its own clicks, like "Send" or "Clear". We return whatever _m\_commPanel.OnEvent_ returns, linking the two dialogs’ event systems. If no match, _CAppDialog::OnEvent_ takes over. This teamwork lets both dialogs respond to the user smoothly.

```
bool CAdminHomeDialog::OnEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_OBJECT_CLICK)
   {
      Print("Clicked object: ", sparam);
      if(sparam == m_commButton.Name())
      {
         Print("Communications button detected");
         OnClickCommunications();
         return true;
      }
      // ... other button checks ...
   }
   // Forward remaining events to CommPanel if visible
   else if(m_commPanel != NULL && m_commPanel.IsVisible())
   {
      return m_commPanel.OnEvent(id, lparam, dparam, sparam);
   }
   return CAppDialog::OnEvent(id, lparam, dparam, sparam);
}
```

### Testing and Results

After successfully compiling the communications module and integrating it to work with the main program, we launched the program on the terminal chart without issues. All panels respond to clicks, as confirmed by the entries in the Experts log. Clicking the 'Communications Panel' button in _AdminHomeDialog_ initiates the creation of the _CommunicationsDialog_, but the initial logic hides it, requiring a second click to make it visible. When the _CommunicationsDialog_ is visible on the chart, clicking the button again hides it, effectively toggling it on and off.

However, I noticed a significant challenge with the list view: the click and scroll events for selecting quick messages to send aren’t working as expected, and we need to identify where we went wrong. I believe this is a minor issue we can fix. For now, the core concept is functional and visible, and our next steps are to refine it and add more features.

![Testing the New Communications Panel](https://c.mql5.com/2/123/terminal64_F2Jgy0Og6h.gif)

Testing the Communications Module

### Conclusion

The development of the _CommunicationsDialog_ module, integrated into _AdminHomeDialog_.mqh, has resulted in a functional and modular system for Telegram messaging within our New Admin Panel, MQL5 trading application. We’ve achieved a responsive admin interface that successfully launches and toggles the communication panel on demand, as evidenced by the Experts log showing click events like "Clicked object: _m\_SendButton_." The lazy creation and toggle approach optimizes resource use, while the event-driven design ensures scalability, proving the core concept is visible and operational on the chart. This modular structure, with _CAdminHomeDialog_ as a hub and _CCommunicationDialog_ as a specialized tool, sets a solid foundation for future expansions.

However, minor challenges remain, such as the list view’s click and scroll events not working for quick message selection and button events of the communications panel need further refinement.  Despite these, the system’s strengths—efficiency, responsiveness, and a clear integration—outweigh the gaps. With targeted fixes to event propagation and plans to refine it, and add more features, we’re well-positioned to transform this prototype into a polished, feature-rich tool for traders. Good news! The [Telegram.mqh](https://www.mql5.com/en/code/56583) file is now available in the Codebase.

Table of file attachments

| File Name | Description |
| --- | --- |
| CommunicationsDialog.mqh | Defines a dialog for sending Telegram messages, featuring a text input and a list of quick message options. |
| AdminHomeDialog.mqh | For admin home dialog creation. It contains all the coordinates declarations. |
| New\_Admin\_Panel.mqh | The latest Admin Panel incorporates the concept of modularity. |
| Telegram.mqh | For transmitting messages and notifications via Telegram. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17044.zip "Download all attachments in the single ZIP archive")

[CommunicationsDialog.mqh](https://www.mql5.com/en/articles/download/17044/communicationsdialog.mqh "Download CommunicationsDialog.mqh")(28.92 KB)

[AdminHomeDialog.mqh](https://www.mql5.com/en/articles/download/17044/adminhomedialog.mqh "Download AdminHomeDialog.mqh")(16.81 KB)

[New\_Admin\_Panel.mq5](https://www.mql5.com/en/articles/download/17044/new_admin_panel.mq5 "Download New_Admin_Panel.mq5")(3.4 KB)

[Telegram.mqh](https://www.mql5.com/en/articles/download/17044/telegram.mqh "Download Telegram.mqh")(1.35 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/482869)**

![Price Action Analysis Toolkit Development (Part 17): TrendLoom EA Tool](https://c.mql5.com/2/125/Price_Action_Analysis_Toolkit_Development_Part_17.png)[Price Action Analysis Toolkit Development (Part 17): TrendLoom EA Tool](https://www.mql5.com/en/articles/17329)

As a price action observer and trader, I've noticed that when a trend is confirmed by multiple timeframes, it usually continues in that direction. What may vary is how long the trend lasts, and this depends on the type of trader you are, whether you hold positions for the long term or engage in scalping. The timeframes you choose for confirmation play a crucial role. Check out this article for a quick, automated system that helps you analyze the overall trend across different timeframes with just a button click or regular updates.

![Developing a multi-currency Expert Advisor (Part 17): Further preparation for real trading](https://c.mql5.com/2/90/logo-midjourney_image_15360_419_4065__3.png)[Developing a multi-currency Expert Advisor (Part 17): Further preparation for real trading](https://www.mql5.com/en/articles/15360)

Currently, our EA uses the database to obtain initialization strings for single instances of trading strategies. However, the database is quite large and contains a lot of information that is not needed for the actual EA operation. Let's try to ensure the EA's functionality without a mandatory connection to the database.

![MQL5 Wizard Techniques you should know (Part 57): Supervised Learning with Moving Average and Stochastic Oscillator](https://c.mql5.com/2/125/MQL5_Wizard_Techniques_you_should_know_Part_57___LOGO.png)[MQL5 Wizard Techniques you should know (Part 57): Supervised Learning with Moving Average and Stochastic Oscillator](https://www.mql5.com/en/articles/17479)

Moving Average and Stochastic Oscillator are very common indicators that some traders may not use a lot because of their lagging nature. In a 3-part ‘miniseries' that considers the 3 main forms of machine learning, we look to see if this bias against these indicators is justified, or they might be holding an edge. We do our examination in wizard assembled Expert Advisors.

![Neural Networks in Trading: A Complex Trajectory Prediction Method (Traj-LLM)](https://c.mql5.com/2/89/logo-midjourney_image_15595_398_3845__1.png)[Neural Networks in Trading: A Complex Trajectory Prediction Method (Traj-LLM)](https://www.mql5.com/en/articles/15595)

In this article, I would like to introduce you to an interesting trajectory prediction method developed to solve problems in the field of autonomous vehicle movements. The authors of the method combined the best elements of various architectural solutions.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/17044&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068276286842926978)

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