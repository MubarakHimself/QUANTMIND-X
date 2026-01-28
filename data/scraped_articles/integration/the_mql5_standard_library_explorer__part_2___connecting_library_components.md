---
title: The MQL5 Standard Library Explorer (Part 2): Connecting Library Components
url: https://www.mql5.com/en/articles/19834
categories: Integration, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:04:59.568165
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/19834&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071558754253613641)

MetaTrader 5 / Examples


### Contents

1. [Introduction](https://www.mql5.com/en/articles/19834#para1)
2. [Implementation](https://www.mql5.com/en/articles/19834#para2)
3. [Testing](https://www.mql5.com/en/articles/19834#para3)
4. [Conclusion](https://www.mql5.com/en/articles/19834#para4)

### Introduction

Many developers entering the MQL5 ecosystem—from enthusiastic newcomers to seasoned programmers coming from other languages—encounter a common and frustrating obstacle: they can write code, but they struggle to build robust, professional-grade Expert Advisors. The [MQL5 Standard Library](https://www.mql5.com/en/docs/standardlibrary) is a powerhouse of pre-built, tested, and optimized classes designed to eliminate that struggle. Yet, its very richness can feel overwhelming. The real challenge is not the lack of components but the absence of a clear roadmap—a practical guide on how to integrate these sophisticated building blocks into a cohesive, functioning trading system.

Developers often find themselves with a toolbox full of precision instruments but no blueprint for assembling them into a complete solution. To simplify this complexity, we’ll use an analogy drawn from real life: just as the human body relies on specialized organs working under the coordination of the brain, an Expert Advisor relies on specialized modules coordinated by its main program.

In this analogy, the Expert Advisor (EA) acts as the brain—the central decision-maker. Supporting library classes serve as its organs, performing specialized functions:

- The CTrade class executes trades—like the hands performing actions.
- The CIndicators class analyzes market data—like the eyes perceiving the environment.
- The CAccountInfo class monitors account health—like the nervous system collecting and transmitting feedback.

![](https://c.mql5.com/2/175/Photos_NRommVTCf6.png)

Fig 1. Conceptual Software System

![](https://c.mql5.com/2/175/Photos_3CgMYFsShu.png)

Fig 2. Conceptual Biological System

The analogy here is only a conceptual aspect to make the abstract structure of [object-oriented programming (OOP)](https://en.wikipedia.org/wiki/Object-oriented_programming "https://en.wikipedia.org/wiki/Object-oriented_programming") more intuitive, helping readers visualize how each part of the library interacts to form a coordinated, efficient trading system.

All these are benefits of OOP—modularity, reusability, and abstraction—but in this series, we deliberately skip the deep theoretical explanation to focus on practical application first. Many learners understand concepts better by doing, and once you start building and integrating classes hands-on, the underlying OOP principles begin to make natural sense.

For readers who prefer to explore OOP theory before continuing, there are many excellent [references](https://www.mql5.com/en/articles/12813) on the [MQL5 Community](https://www.mql5.com/en/forum) and beyond that explain encapsulation, inheritance, and polymorphism in depth. However, our approach here is guided by practice: through real examples, diagrams, and analogies that make your journey with the MQL5 Standard Library both enjoyable and productive.

By the end of this discussion, we will move beyond isolated examples to reveal a master blueprint—the definitive guide to interweaving the powerful threads of the MQL5 Standard Library into the solid framework of a professional Expert Advisor, while also discovering ways to expand the library itself.

![](https://c.mql5.com/2/175/chrome_jUNmdf0OxG.png)

Fig 3. EA as the Brain and Classes as Organs

While the Expert Advisor is rightly treated as the brain or main program, we must also understand that modules themselves can be built by including other modules—what we call dependencies. Just as we're learning to integrate entire libraries within EA programs, we need the same expertise when intertwining modules or implementing one module inside another.

The above analogy reveals that interrelationships exist between modules in two key ways: generally through the main program's coordination, but also through direct dependencies. For example, the [CPanel](https://www.mql5.com/en/docs/standardlibrary/controls/cpanel) class doesn't stand alone—it directly includes and depends on [WndObj.mqh](https://www.mql5.com/en/docs/standardlibrary/controls/cwndobj) and [ChartObjectsTxtControls.mqh.](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjecttext)

![](https://c.mql5.com/2/175/chrome_z67c6xx8yT.png)

Fig 4. Class-to-Class Communication

The good news is that this relationship network isn't complicated once understood. When using any module in an EA, whether it's a simple class or a complex dependent module, there are consistent, repeatable steps that ensure successful integration every time. See the partial code snippet below—it highlights the key dependencies used by Panel.mqh and where they are included.

```
//+------------------------------------------------------------------+
//|                                                        Panel.mqh |
//|                             Copyright 2000-2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#include "WndObj.mqh"
#include <ChartObjects\ChartObjectsTxtControls.mqh>
```

In the next section, I’ll share the steps I normally follow when integrating library classes into my projects. This will help you build a solid understanding before we move on to the implementation stage, where we’ll explore practical examples in more detail.

**Integrating Any Class into an Expert Advisor**

In most cases, selecting which classes to use comes after defining your EA’s overall strategy or goal—there’s no need to overload your project with unnecessary components. In this section, however, our focus isn’t on strategy design but on understanding the integration process itself.

Our goal is to share the essential knowledge and practical steps that will enable you to confidently integrate any library module into your future projects whenever the need arises. The following section outlines these steps, accompanied by a flow diagram to illustrate the process.

Process Flow:

1. Analyze - Study class purpose & inheritance chain
2. Include - Add #include directive with correct path
3. Declare - Create object instance in EA
4. Initialize - Call Create() in OnInit() with parameters
5. Runtime - Update in OnTick(), handle events
6. Cleanup - Release resources in OnDeinit()

Error Handling:

- Compilation errors → Check dependencies
- Creation failures → Verify parameters
- Runtime issues → Monitor object state

![](https://c.mql5.com/2/175/Photos_elmFlNbWu4.png)

Fig 5. MQL5 Library Integration Roadmap

**Module-to-Module Integration Flow**

Dependency Analysis

- Parent-Child Relationships: e.g. CPanel → WndObj → ChartObjectsTxtControls
- Interface Contracts: What methods must be implemented
- Data Flow: How information passes between modules

![Module Inheritance Hierarchy](https://c.mql5.com/2/175/Photos_QaW1GkAKEk.png)

Fig 6. Module Inheritance Hierarchy

Implementation Steps

1. Include Chain: Each module includes its direct dependencies
2. Inheritance Setup: Class hierarchies established through extends
3. Method Implementation: Override virtual methods as needed
4. Isolated Testing: Test modules independently before EA integration

Common Patterns

- Layered Architecture: Higher-level modules depend on lower-level ones
- Interface Segregation: Modules expose clean, focused APIs
- Dependency Injection: Pass required objects rather than hard-coding

![](https://c.mql5.com/2/175/Photos_7cjgB6yi4K.png)

Fig 7. Module-to-Module Integration Flow

With our architectural foundation firmly established and the integration roadmap clearly defined, we now move into the implementation phase, where theory becomes practice. This is where we apply the structured integration process to a working project, transforming conceptual understanding into a functional Expert Advisor.

Our implementation plan is to construct an InfoPanel EA—an information dashboard that demonstrates how to bring the CPanel class and its dependencies to life within a trading environment. This example will showcase exactly how to apply each step of our integration methodology to produce a clean, professional, and extensible system.

### Implementation

When approaching any library class in MQL5, integration begins with understanding what the class truly is—its purpose, its lineage, and how it fits into the ecosystem of chart controls. CPanel is more than just a rectangular area; it’s a self-contained interface layer that inherits from CWndObj and manages its own lifecycle on the chart. It serves as the architectural skeleton for complex UI components. Before any code appears in an Expert Advisor, this understanding sets the mental framework for seamless integration.

**Analyzing the CPanel Class—in Panel.mqh**

The first encounter with a class should always be investigative. Reading through the Panel.mqh header reveals key relationships and extension points. The class inherits from CWndObj, meaning it inherits the ability to handle events and manage chart-based windows. Inside, it encapsulates CChartObjectRectLabel, the graphical rectangle that gives the panel its visible frame.

By identifying its public entry points—notably Create()  and BorderType()—you locate the control handles that your EA will interact with. This inspection replaces trial-and-error with informed design.

```
//+------------------------------------------------------------------+
//|                                                        Panel.mqh |
//|                             Copyright 2000-2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#include "WndObj.mqh"
#include <ChartObjects\ChartObjectsTxtControls.mqh>
//+------------------------------------------------------------------+
//| Class CPanel                                                     |
//| Usage: control that is displayed by                              |
//|             the CChartObjectRectLabel object                     |
//+------------------------------------------------------------------+
class CPanel : public CWndObj
  {
private:
   CChartObjectRectLabel m_rectangle;       // chart object
   //--- parameters of the chart object
   ENUM_BORDER_TYPE  m_border;              // border type

public:
                     CPanel(void);
                    ~CPanel(void);
   //--- create
   virtual bool      Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2);
   //--- parameters of the chart object
   ENUM_BORDER_TYPE  BorderType(void)       const { return(m_border);                                  }
   bool              BorderType(const ENUM_BORDER_TYPE type);

protected:
   //--- handlers of object settings
   virtual bool      OnSetText(void)              { return(m_rectangle.Description(m_text));           }
   virtual bool      OnSetColorBackground(void)   { return(m_rectangle.BackColor(m_color_background)); }
   virtual bool      OnSetColorBorder(void)       { return(m_rectangle.Color(m_color_border));         }
   //--- internal event handlers
   virtual bool      OnCreate(void);
   virtual bool      OnShow(void);
   virtual bool      OnHide(void);
   virtual bool      OnMove(void);
   virtual bool      OnResize(void);
   virtual bool      OnChange(void);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CPanel::CPanel(void) : m_border(BORDER_FLAT)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CPanel::~CPanel(void)
  {
  }
//+------------------------------------------------------------------+
//| Create a control                                                 |
//+------------------------------------------------------------------+
bool CPanel::Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2)
  {
//--- call method of the parent class
   if(!CWndObj::Create(chart,name,subwin,x1,y1,x2,y2))
      return(false);
//--- create the chart object
   if(!m_rectangle.Create(chart,name,subwin,x1,y1,Width(),Height()))
      return(false);
//--- call the settings handler
   return(OnChange());
  }
//+------------------------------------------------------------------+
//| Set border type                                                  |
//+------------------------------------------------------------------+
bool CPanel::BorderType(const ENUM_BORDER_TYPE type)
  {
//--- save new value of parameter
   m_border=type;
//--- set up the chart object
   return(m_rectangle.BorderType(type));
  }
//+------------------------------------------------------------------+
//| Create object on chart                                           |
//+------------------------------------------------------------------+
bool CPanel::OnCreate(void)
  {
//--- create the chart object by previously set parameters
   return(m_rectangle.Create(m_chart_id,m_name,m_subwin,m_rect.left,m_rect.top,m_rect.Width(),m_rect.Height()));
  }
//+------------------------------------------------------------------+
//| Display object on chart                                          |
//+------------------------------------------------------------------+
bool CPanel::OnShow(void)
  {
   return(m_rectangle.Timeframes(OBJ_ALL_PERIODS));
  }
//+------------------------------------------------------------------+
//| Hide object from chart                                           |
//+------------------------------------------------------------------+
bool CPanel::OnHide(void)
  {
   return(m_rectangle.Timeframes(OBJ_NO_PERIODS));
  }
//+------------------------------------------------------------------+
//| Absolute movement of the chart object                            |
//+------------------------------------------------------------------+
bool CPanel::OnMove(void)
  {
//--- position the chart object
   return(m_rectangle.X_Distance(m_rect.left) && m_rectangle.Y_Distance(m_rect.top));
  }
//+------------------------------------------------------------------+
//| Resize the chart object                                          |
//+------------------------------------------------------------------+
bool CPanel::OnResize(void)
  {
//--- resize the chart object
   return(m_rectangle.X_Size(m_rect.Width()) && m_rectangle.Y_Size(m_rect.Height()));
  }
//+------------------------------------------------------------------+
//| Set up the chart object                                          |
//+------------------------------------------------------------------+
bool CPanel::OnChange(void)
  {
//--- set up the chart object
   return(CWndObj::OnChange() && m_rectangle.BorderType(m_border));
  }
//+------------------------------------------------------------------+

```

Each of these lines will later translate directly into an initialization step inside your EA.

**Integration Steps**

Step 1. Including the Header and Dependencies

Once the intent is clear, the next move is inclusion. The EA must gain visibility of CPanel and its base classes through proper #include directives. MQL5’s file structure is strict about include paths, so always check that your Panel.mqh and its supporting headers (WndObj.mqh, ChartObjectsTxtControls.mqh, etc.) are accessible within the project or global Include directory.

```
#include <Controls\Panel.mqh>  // gives access to CPanel and its internals
```

At this point, the EA “knows” the panel class exists—it’s ready for instantiation.

Step 2. Declaring the Object

Declaration defines where the panel lives within the EA’s memory scope. Most developers choose to declare CPanel as a pointer at the global level to ensure it persists between event calls and can be safely deleted during cleanup.

```
CPanel *mainPanel = NULL;
```

This makes the object accessible in all event handlers (OnInit, OnTick, OnDeinit, etc.), ensuring a consistent lifecycle.

Step 3. Initialization—Bringing the Panel to Life

Creation happens inside OnInit(), and this is where the header’s design meets execution. The Create() method requires six arguments: the chart ID, a unique name, the target subwindow, and four coordinates defining the panel’s rectangle. Always verify that the call returns true; a false result often signals a parameter or dependency issue.

```
int OnInit()
{
   mainPanel = new CPanel();
   if(!mainPanel.Create(ChartID(), "InfoPanel", 0, 10, 10, 400, 120))
   {
      Print("Panel creation failed.");
      delete mainPanel;
      mainPanel = NULL;
      return(INIT_FAILED);
   }

   mainPanel.BorderType(BORDER_RAISED);
   mainPanel.Text("Information Panel");
   return(INIT_SUCCEEDED);
}
```

This snippet demonstrates minimal but complete initialization. The panel now exists visually on the chart, occupying the specified rectangle.

Step 4. Runtime Management

The panel itself may not need continuous updates if it’s static, but this is where your integration logic extends. You may wish to embed child controls, such as labels or buttons, or update panel contents periodically.

For dynamic updates, a OnTimer() event provides a stable refresh mechanism without burdening the OnTick() loop.

```
void OnTimer()
{
   // Placeholder for runtime UI updates
}
```

Each update can refresh data shown inside the panel without redrawing the object.

Step 5. Cleanup and Error Handling

Every object created dynamically must be destroyed to avoid leaks. In OnDeinit(), call delete and set the pointer to NULL. This ensures proper deallocation and prevents orphaned chart objects.

```
void OnDeinit(const int reason)
{
   if(mainPanel)
   {
      mainPanel.OnHide();
      delete mainPanel;
      mainPanel = NULL;
   }
}
```

If compilation fails, inspect missing includes or undefined references. Creation errors typically relate to incorrect coordinates or name conflicts, while runtime flicker or event issues often stem from overlapping chart objects.

**Analyzing the CChartObjectText in ChartObjectsTxtControls.mqh**

In this section, we focus on the interface layer of the [ChartObjectsTxtControls.mqh](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjecttext) header. For clarity and brevity, we’ve only extracted the public and top-level portion of the class—the parts that define how we interact with it. The full source is much longer, but for our analysis, we only need the public interface, which represents the “front door” of the class.

This selective view is intentional. The essence of Object-Oriented Programming (OOP) is abstraction—we don’t need to know how the internals work as long as we understand the interface and how to use it correctly. The interface defines the available methods, their parameters, and their expected behavior—that’s all we need to build integrations effectively.

Our goal here is to analyze the header to understand its purpose and identify the grip points—the key methods we’ll call from our main EA to describe or manipulate text on the chart. In this case, we included this header not for panel construction (as when it served as a CPanel dependency) but purely for text description within our Expert Advisor.

When it was previously used as a dependency, we didn’t need to focus much on its internal logic—the parent CPanel handled that. However, in this educational context, we deliberately study it to understand how each class is built and how to use it independently.

The excerpt below shows the interface of CChartObjectText, which provides all essential methods for working with text objects directly on the chart. Among these are Create(), FontSize(), and Anchor()—our primary grip points when integrating text elements into a panel or EA. These are the methods that let us define pixel positions, set font styles, control alignment, and manage visibility—everything needed to produce the text description layer of our InfoPanel.

```
//+------------------------------------------------------------------+
//|                                      ChartObjectsTxtControls.mqh |
//|                             Copyright 2000-2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
//| All text objects.                                                |
//+------------------------------------------------------------------+
#include "ChartObject.mqh"
//+------------------------------------------------------------------+
//| Class CChartObjectText.                                          |
//| Purpose: Class of the "Text" object of chart.                    |
//|          Derives from class CChartObject.                        |
//+------------------------------------------------------------------+
class CChartObjectText : public CChartObject
  {
public:
                     CChartObjectText(void);
                    ~CChartObjectText(void);
   //--- method of creating the object
   bool              Create(long chart_id,const string name,const int window,const datetime time,const double price);
   //--- method of identifying the object
   virtual int       Type(void) const override { return(OBJ_TEXT); }
   //--- methods of access to properties of the object
   double            Angle(void) const;
   bool              Angle(const double angle) const;
   string            Font(void) const;
   bool              Font(const string font) const;
   int               FontSize(void) const;
   bool              FontSize(const int size) const;
   ENUM_ANCHOR_POINT Anchor(void) const;
   bool              Anchor(const ENUM_ANCHOR_POINT anchor) const;
   //--- methods for working with files
   virtual bool      Save(const int file_handle) override;
   virtual bool      Load(const int file_handle) override;
  };
```

**Integration Steps of CChartObjectText into the EA**

Step 1. Include

We include the ChartObjectsTxtControls.mqh near the top of the EA so the class definitions are visible. We use the same path as the library.

If you already include Panel.mqh, this header is often pulled in, but including it explicitly in the EA makes intent and dependencies clear.

```
#include <ChartObjects\ChartObjectsTxtControls.mqh>  // provides CChartObjectLabel et al.
```

Step 2. Declare

We declare label objects where EA lifecycle functions can access them—global scope. Use stack objects for chart text classes (they manage attachment internally) or pointers if you prefer new/delete semantics. Name labels carefully (unique per chart). If you expect multiple labels, use clear name patterns: "InfoPanel\_123\_txt1".

```
// global declarations
CChartObjectLabel txtDesc;    // pixel-based label (good for panel body)
```

Step 3. Initialize (OnInit)

Next, we create the label inside OnInit() using the signature shown in the ChartObjectsTxtControls.mqh header: Create(long chart\_id, const string name, const int window, const int X, const int Y). Position it relative to your panel rectangle to keep layout consistent.

If the label fails to create, log helpful info (chart id, name, coords) and abort initialization.

```
int OnInit()
{
  string txt_name = panel_name + "_txtDesc";
  int left = PANEL_X1 + PADDING;
  int top  = PANEL_Y1 + PADDING + 18;

  if(!txtDesc.Create(ChartID(), txt_name, 0, left, top))
  {
    Print("txtDesc.Create failed");
    return(INIT_FAILED);
  }

  txtDesc.Description("Hello trader! I am InfoPanel EA");
  txtDesc.FontSize(11);
  // txtDesc.Anchor(ANCHOR_LEFT_UPPER); // if API available
  return(INIT_SUCCEEDED);
}
```

Step 4. Runtime (OnTick/OnTimer/OnChartEvent)

Keep runtime work minimal for static text. If you need dynamic fields, update Description() from OnTimer() (set EventSetTimer() in OnInit()), not every tick. If you add interactivity (buttons or edits from the same header), forward OnChartEvent() and inspect sparam for object clicks.

For multiple lines, either use multiple CChartObjectLabel instances stacked vertically, or generate \\n and see if Description() accepts line breaks; if not, prefer separate labels.

```
void OnTimer()
{
  // e.g., update price line inside the panel
  txtDesc.Description("Bid: " + DoubleToString(SymbolInfoDouble(_Symbol, SYMBOL_BID), _Digits));
}
```

Step 5. Cleanup (OnDeinit)

Remove the chart object when the EA exits. Use the class Delete() method (or ObjectDelete() as fallback). Do not call internal protected methods of other classes — just delete your label and free any panel pointer.

```
void OnDeinit(const int reason)
{
  txtDesc.Delete();        // removes the chart object
  // delete panel pointer here if allocated
  EventKillTimer();        // if used
}
```

**Connecting the pieces together into an EA**

The final integration of the CPanel class into our InfoPanel EA completes the foundational stage of our Standard Library study. This panel isn’t an interactive dashboard or a trading controller—it’s a static information display, a simple but professional mark on the chart that openly describes the product in use. Many developers might choose a similar design to display their EA’s identity and details directly on the chart, avoiding the limitation of the default chart name or the need to dig through properties. It acts as a subtle yet practical signature—a visual tag that brings clarity to the environment.

While the concept may appear modest, its significance lies in the method, not the magnitude. Here we’ve practiced the complete class integration cycle: from including the Standard Library header, to creating and customizing a CPanel instance, and pairing it with chart text elements for meaningful display. This process reflects the very core of how professional MQL5 components are structured and managed. It’s about understanding the architectural rhythm—include, declare, create, and control—that you’ll later reuse for far more advanced systems.

This first project is intentionally simple because it sets the stage for expansion. Once you’re comfortable with CPanel, adding other elements such as CChartObjectLabel for richer text output or even CPicture for branding becomes natural. Together, these static elements form a small but complete ecosystem of information visualization, ready to grow in complexity as our study progresses.

The broader vision is to train the mind to think modularly, to see the Standard Library not as a rigid framework, but as a flexible foundation that can be extended indefinitely. Each class—CPanel, CChartObjectLabel, CPicture, and beyond—is a building block. What starts as a simple display panel today can later evolve into a comprehensive workspace of data visualization, custom controls, and adaptive layouts.

Our journey with CPanel therefore marks more than the creation of a static information panel; it’s the beginning of mastering the logic and craftsmanship required to build entire systems from the Standard Library upward—step by step, layer by layer.

```
//+------------------------------------------------------------------+
//| InfoPanel_EA.mq5                                                 |
//| Minimal example: creating a CPanel and a label text showing      |
//| a short EA description with panel background & border. We will   |
//| also add a logo to explore CPicture.                             |
//+------------------------------------------------------------------+
#property copyright "Clemence Benjamin"
#property version   "1.0"

#include <Controls/Panel.mqh>                        // Panel header (correct path)
#include <ChartObjects\ChartObjectsTxtControls.mqh>  // for CChartObjectLabel

//---- Globals --------------------------------------------------------
CPanel             *infoPanel = NULL;   // pointer to panel (allocate/free)
CChartObjectLabel   txtDesc;            // label text inside the panel

// layout (pixels)
int PANEL_X1 = 10;
int PANEL_Y1 = 30;
int PANEL_X2 = 300;   // widened to give text room
int PANEL_Y2 = 100;
int PADDING  = 10;

// short description that fits comfortably on one line
string EA_DESCRIPTION = "Hello trader! I am InfoPanel EA";

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   // allocate the panel object
   infoPanel = new CPanel();
   if(infoPanel == NULL)
   {
      Print("InfoPanel: allocation failed");
      return(INIT_FAILED);
   }

   // unique panel name
   string panel_name = "InfoPanel_" + IntegerToString((int)ChartID());

   // create the panel on the current chart (subwindow 0 = main)
   if(!infoPanel.Create(ChartID(), panel_name, 0, PANEL_X1, PANEL_Y1, PANEL_X2, PANEL_Y2))
   {
      Print("InfoPanel: Create() failed");
      delete infoPanel; infoPanel = NULL;
      return(INIT_FAILED);
   }

   // appearance: border type, header text, background and border colors
   infoPanel.BorderType(BORDER_RAISED);
   infoPanel.Text("InfoPanel");

   // Set a visible background and border color.
   // Use standard color constants; change them if you prefer another palette.
   // If your library expects ARGB or different color encoding, adapt accordingly.
   infoPanel.ColorBackground(clrSilver); // panel background
   infoPanel.ColorBorder(clrBlack);      // panel border

   // compute child label position inside panel (some padding below header)
   int left = PANEL_X1 + PADDING;
   int top  = PANEL_Y1 + PADDING + 18; // leave header area

   // create a CChartObjectLabel as panel body (unique name)
   string txt_name = panel_name + "_txtDesc";
   if(!txtDesc.Create(ChartID(), txt_name, 0, left, top))
   {
      Print("InfoPanel: txtDesc.Create() failed");
      delete infoPanel; infoPanel = NULL;
      return(INIT_FAILED);
   }

   // set the short description and font size using the correct API
   txtDesc.Description(EA_DESCRIPTION);
   txtDesc.FontSize(11);

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // remove the label object
   txtDesc.Delete();

   // free the panel (do NOT call protected methods)
   if(infoPanel != NULL)
   {
      delete infoPanel;
      infoPanel = NULL;
   }

   // safe kill of timer if later added
   EventKillTimer();
}

//+------------------------------------------------------------------+
//| Expert tick function (kept intentionally light)                  |
//+------------------------------------------------------------------+
void OnTick()
{
   // static info panel — no per-tick UI updates
}

//+------------------------------------------------------------------+
//| Chart event handler (optional extension point)                   |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
// placeholder for future interactive controls
}
```

![](https://c.mql5.com/2/175/infopanel.png)

Fig 8. Initial Testing

Now, taking an extra step, we’ll add a BMP logo to the panel to give it a clear identity. I’ve always preferred using open-source tools for image creation— [GIMP](https://www.mql5.com/go?link=https://www.gimp.org/downloads/ "https://www.gimp.org/downloads/") is an excellent choice, as it makes exporting images in BMP format quick and straightforward. Of course, you can use any alternative software you’re comfortable with. Once your logo is created and saved in the Images folder of your terminal for easy access, we can proceed to explore the CPicture class and plan how to integrate it into our panel.

Below are two images. Fig 9 illustrates where the logo will be positioned within our panel, while Fig 10 shows the logo I created using [GIMP.](https://www.mql5.com/go?link=https://www.gimp.org/downloads/ "https://www.gimp.org/downloads/") When exporting the image in [BMP format](https://www.mql5.com/en/book/advanced/resources/resources_variables), make sure to use 24-bit color depth, a single layer, and no alpha channel. These settings worked perfectly during testing—other export configurations produced images that failed to display correctly on the panel. The logo image shown here is not in the correct BMP format, so it will not function properly. I used a PNG version, which is supported for publication purposes. You can find the attached logo at the bottom of the article.

![LOGO](https://c.mql5.com/2/175/LOGO_SP_PNG.png)

Fig 9. EA logo space

![BMP logo](https://c.mql5.com/2/175/Logogram.png)

Fig 10. 56x59 pixel logo

**Understanding and Preparing the CPicture Class for Integration**

The CPicture class extends the foundation we have already laid with CPanel and CChartObjectLabel. Just as the panel gave us a container and the label offered text, this class introduces a simple way to display bitmap images on the chart—ideal for our next improvement, where we want to add a logo thumbnail to the information panel.

At its core, CPicture acts as a small wrapper around the CChartObjectBmpLabel chart object. It inherits from CWndObj, meaning it follows the same lifecycle methods (Create, OnShow, OnHide, etc.) we’ve seen before. This design ensures that we can integrate it into our EA using the same familiar steps we used with previous classes—allocate, create, configure, and clean up.

When we look closely, a few key points stand out. The main creation entry, Create(chart, name, subwin, x1, y1, x2, y2), defines the object’s area on the chart in pixel coordinates, just like CPanel. Once created, we can assign the actual image through BmpName(name), which points to a .bmp file located in the terminal’s Images folder. The method Border(value) adjusts the visible width or border thickness of the image area, though for a clean logo display, we will likely set this to zero. Internally, these two methods update the underlying chart object through the OnChange() handler.

Our Grip Points for Integration

For our purposes, we only need to focus on a few practical grip points:

- Create() – initializes the object on the chart using a unique name and pixel coordinates.
- BmpName() – assigns the bitmap file to display, such as our EA logo.
- Border() – defines border or width values if needed (optional for logos).
- Destructor or delete – safely frees memory when the EA is removed.

```
//+------------------------------------------------------------------+
//|                                                      Picture.mqh |
//|                             Copyright 2000-2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#include "WndObj.mqh"
#include <ChartObjects\ChartObjectsBmpControls.mqh>
//+------------------------------------------------------------------+
//| Class CPicture                                                   |
//| Note: image displayed by                                         |
//|             the CChartObjectBmpLabel object                      |
//+------------------------------------------------------------------+
class CPicture : public CWndObj
  {
private:
   CChartObjectBmpLabel m_picture;          // chart object
   //--- parameters of the chart object
   int               m_border;              // border width
   string            m_bmp_name;            // filename

public:
                     CPicture(void);
                    ~CPicture(void);
   //--- create
   virtual bool      Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2);
   //--- parameters of the chart object
   int               Border(void) const { return(m_border); }
   bool              Border(const int value);
   string            BmpName(void) const { return(m_bmp_name); }
   bool              BmpName(const string name);

protected:
   //--- internal event handlers
   virtual bool      OnCreate(void);
   virtual bool      OnShow(void);
   virtual bool      OnHide(void);
   virtual bool      OnMove(void);
   virtual bool      OnChange(void);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CPicture::CPicture(void) : m_border(0),
                           m_bmp_name(NULL)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CPicture::~CPicture(void)
  {
  }
//+------------------------------------------------------------------+
//| Create a control                                                 |
//+------------------------------------------------------------------+
bool CPicture::Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2)
  {
//--- call method of the parent class
   if(!CWndObj::Create(chart,name,subwin,x1,y1,x2,y2))
      return(false);
//--- create the chart object
   if(!m_picture.Create(chart,name,subwin,x1,y1))
      return(false);
//--- call the settings handler
   return(OnChange());
  }
//+------------------------------------------------------------------+
//| Set border width                                                 |
//+------------------------------------------------------------------+
bool CPicture::Border(const int value)
  {
//--- save new value of parameter
   m_border=value;
//--- set up the chart object
   return(m_picture.Width(value));
  }
//+------------------------------------------------------------------+
//| Set image                                                        |
//+------------------------------------------------------------------+
bool CPicture::BmpName(const string name)
  {
//--- save new value of parameter
   m_bmp_name=name;
//--- set up the chart object
   return(m_picture.BmpFileOn(name));
  }
//+------------------------------------------------------------------+
//| Create object on chart                                           |
//+------------------------------------------------------------------+
bool CPicture::OnCreate(void)
  {
//--- create the chart object by previously set parameters
   return(m_picture.Create(m_chart_id,m_name,m_subwin,m_rect.left,m_rect.top));
  }
//+------------------------------------------------------------------+
//| Display object on chart                                          |
//+------------------------------------------------------------------+
bool CPicture::OnShow(void)
  {
   return(m_picture.Timeframes(OBJ_ALL_PERIODS));
  }
//+------------------------------------------------------------------+
//| Hide object from chart                                           |
//+------------------------------------------------------------------+
bool CPicture::OnHide(void)
  {
   return(m_picture.Timeframes(OBJ_NO_PERIODS));
  }
//+------------------------------------------------------------------+
//| Absolute movement of the chart object                            |
//+------------------------------------------------------------------+
bool CPicture::OnMove(void)
  {
//--- position the chart object
   return(m_picture.X_Distance(m_rect.left) && m_picture.Y_Distance(m_rect.top));
  }
//+------------------------------------------------------------------+
//| Set up the chart object                                          |
//+------------------------------------------------------------------+
bool CPicture::OnChange(void)
  {
//--- set up the chart object
   return(m_picture.Width(m_border) && m_picture.BmpFileOn(m_bmp_name));
  }
//+------------------------------------------------------------------+
```

With these simple handles, we have everything required to integrate an image component into our panel.

Planning the Integration

Now that we understand how CPicture works, we can plan how to bring it into our InfoPanel\_EA. The image file—our logo—should first be prepared in BMP format. As mentioned earlier, tools like GIMP make exporting to BMP straightforward. Once we have the file, we place it in the terminal’s Images folder (MQL5/Images/) so that the EA can easily access it by name.

In our code, we will:

1. Include the Picture.mqh header.
2. Declare a global pointer for our CPicture instance, much like we did for the panel.
3. Create the object in OnInit(), positioning it neatly inside or just above the panel.
4. Set the image name with BmpName("logo.bmp") to load our chosen file.
5. Release resources in OnDeinit() to ensure a clean exit.

By doing so, our panel will evolve from a simple text display into a branded information area—a subtle but meaningful step that makes the EA’s presentation more professional and identifiable.

Justification of this step

Although this addition may appear minor, it deepens our understanding of how any visual class from the Standard Library can be combined in practice. Each integration reinforces a pattern: create, configure, and manage lifecycle safely. By the time we complete the logo integration, we’ll have connected three different standard classes—CPanel, CChartObjectLabel, and CPicture—into one cohesive interface.

This prepares us for the next section, where we will take the full integration live and attach our EA’s logo onto the panel itself. That will complete our static information panel and serve as a visual revision of everything we’ve learned so far. The upcoming section will demonstrate how these classes can coexist harmoniously, making our EA both informative and visually distinctive.

```
//+------------------------------------------------------------------+
//| InfoPanel_EA.mq5                                                 |
//| Info panel with logo placed inside the panel and on foreground   |
//+------------------------------------------------------------------+
#property copyright "Clemence Benjamin"
#property version   "1.0"

#include <Controls\Panel.mqh>
#include <ChartObjects\ChartObjectsTxtControls.mqh>
#include <Controls\Picture.mqh>

//---- Globals --------------------------------------------------------
CPanel             *infoPanel = NULL;
CChartObjectLabel   txtDesc;
CPicture           *logo = NULL;

// layout (pixels)
int PANEL_X1 = 10;
int PANEL_Y1 = 30;
int PANEL_X2 = 300;
int PANEL_Y2 = 100;
int PADDING  = 10;

// logo size and filename
int LOGO_SIZE = 56;
string LOGO_FILE = "Logo.bmp";

// short description
string EA_DESCRIPTION = "Hello trader! I am InfoPanel EA";

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   // 1. Create panel
   infoPanel = new CPanel();
   if(infoPanel == NULL)
   {
      Print("InfoPanel: allocation failed");
      return(INIT_FAILED);
   }

   string panel_name = "InfoPanel_" + IntegerToString((int)ChartID());

   if(!infoPanel.Create(ChartID(), panel_name, 0, PANEL_X1, PANEL_Y1, PANEL_X2, PANEL_Y2))
   {
      Print("InfoPanel: Create() failed");
      delete infoPanel; infoPanel = NULL;
      return(INIT_FAILED);
   }

   infoPanel.BorderType(BORDER_RAISED);
   infoPanel.Text("InfoPanel");
   infoPanel.ColorBackground(clrSilver);
   infoPanel.ColorBorder(clrBlack);

   // 2. Create label inside panel
   int label_left = PANEL_X1 + PADDING;
   int label_top = PANEL_Y1 + PADDING + 18;
   string txt_name = panel_name + "_txtDesc";
   if(!txtDesc.Create(ChartID(), txt_name, 0, label_left, label_top))
   {
      Print("InfoPanel: txtDesc.Create() failed");
      delete infoPanel; infoPanel = NULL;
      return(INIT_FAILED);
   }
   txtDesc.Description(EA_DESCRIPTION);
   txtDesc.FontSize(11);

   // 3. Set panel as background
   if(!ObjectSetInteger(ChartID(), panel_name, OBJPROP_BACK, true))
      PrintFormat("InfoPanel: unable to set OBJPROP_BACK for '%s' (non-fatal).", panel_name);

   // 4. Compute logo coordinates
   int logo_right = PANEL_X2 - PADDING;
   int logo_left = logo_right - LOGO_SIZE;
   int logo_top = PANEL_Y1 + PADDING;
   int logo_bottom = logo_top + LOGO_SIZE;
   string logo_name = panel_name + "_logo";

   // 5. Create logo
   logo = new CPicture();
   if(logo == NULL)
   {
      Print("InfoPanel: logo allocation failed (continuing without logo)");
   }
   else
   {
      if(!logo.Create(ChartID(), logo_name, 0, logo_left, logo_top, logo_right, logo_bottom))
      {
         Print("InfoPanel: logo.Create() failed (continuing without logo)");
         delete logo; logo = NULL;
      }
      else
      {
         // Ensure logo is in foreground
         if(!ObjectSetInteger(ChartID(), logo_name, OBJPROP_BACK, false))
            Print("InfoPanel: Failed to set OBJPROP_BACK=false for logo (non-fatal).");

         // Log coordinates
         PrintFormat("InfoPanel: Logo coords: left=%d, top=%d, right=%d, bottom=%d", logo_left, logo_top, logo_right, logo_bottom);

         // Load built-in logo.bmp with correct relative path (no fullpath needed).You can use custom logo names as long you update the code.
         string bmp_path = "\\Images\\Logo.bmp";
         bool ok = logo.BmpName(bmp_path);
         PrintFormat("InfoPanel: logo.BmpName('%s') returned %s", bmp_path, ok ? "true" : "false");
         if(!ok)
         {
            Print("InfoPanel: LOGO load failed. Verify MT5 installation has MQL5\\Images\\dollar.bmp.");
         }
         else
         {
            Print("InfoPanel: Logo loaded successfully!");
         }

         logo.Border(LOGO_SIZE);
         ChartRedraw(ChartID()); // Force redraw
      }
   }

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(logo != NULL) { delete logo; logo = NULL; }
   txtDesc.Delete();
   if(infoPanel != NULL) { delete infoPanel; infoPanel = NULL; }
   EventKillTimer();
}

//+------------------------------------------------------------------+
//| OnTick                                                          |
//+------------------------------------------------------------------+
void OnTick()
{
}

//+------------------------------------------------------------------+
//| Chart events                                                    |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
}
```

By placing the CPicture logo inside the CPanel’s rectangle and forcing it to the foreground, we completed a small but powerful demonstration: combining three separate Standard Library classes—CPanel, CChartObjectLabel, and CPicture—into one coherent UI element. We created the panel first, then added the descriptive label, and finally computed precise pixel coordinates to insert the logo so it sits neatly on the right side of the panel. Calling ObjectSetInteger(..., OBJPROP\_BACK, true) for the panel and OBJPROP\_BACK = false for the logo gave us control over drawing order so the image appears above the background but behind other potential interactive objects. We also used ChartRedraw() to ensure the terminal repainted immediately after loading the bitmap. All of these steps followed the same integration rhythm: include, declare, create, configure, check return values, and clean up.

Several concrete lessons about class integration emerged from this exercise. First, coordinate math and naming discipline matter—compute positions relative to the panel’s rectangle and use unique, predictable object names (we used panel\_name + "\_logo") to avoid collisions and simplify debugging. Second, external resources require careful handling: we saved the bitmap in MQL5\\Images, passed its relative path ("\\Images\\Logo.bmp") to BmpName(), and logged success/failure so the EA remains resilient if the file is missing. Third, respect the lifecycle contract of each class: we allocated objects with new, validated every Create() and BmpName() call, and always deleted objects in OnDeinit() to avoid leaving stray chart objects or leaking memory. Fourth, layering and redraw control are practical skills—OBJPROP\_BACK and ChartRedraw() let us manage visibility and refresh deterministically.

Finally, this work reinforces a repeatable pattern that applies to any Standard Library class: identify the public grip points (create, configure setters like Border()/BmpName() or Text()), integrate them inside the EA lifecycle (OnInit → runtime → OnDeinit), validate every step with clear logging, and compute geometry relative to existing elements so the layout remains maintainable. What looks like a tiny feature—a logo inside a panel—is actually a compact training ground for the broader skillset we’re building: composing small, well-behaved modules into larger, professional interfaces. From here, we can confidently extend the panel (responsive layout, additional static metadata lines, or optional non-interactive icons), knowing that any class from the Standard Library can be integrated using the exact same disciplined approach.

### Testing

After compiling the code in MetaEditor 5, we deployed it on a chart to observe our small information panel, complete with its description and logo—as shown in the image below. This panel and its components can be viewed as a visual identity for an Expert Advisor. The same idea can be applied to give your future products a unique and recognizable presence on the chart. For now, we can take pride in having successfully demonstrated the concept and deepened our understanding of the integration process.

![Testing the InfoPanel_EA](https://c.mql5.com/2/175/terminal64_SZarJJfXa6.gif)

Fig 11. InfoPanel\_EA testing on chart

### Conclusion

Through this exercise, we successfully established a repeatable routine for integrating Standard Library classes into an Expert Advisor. From the creation of a CPanel to the addition of a descriptive label and logo via CChartObjectLabel and CPicture, we explored how different UI components can be combined and controlled to produce a polished and functional result. This small yet complete demonstration reflects the same architectural discipline required for larger, more complex systems.

The goal here was not only to create a static information panel but to internalize a working pattern—include, declare, initialize, configure, and clean up—that applies universally across MQL5’s class ecosystem. As we proceed further in the MQL5 Standard Library Explorer series, this foundational exercise will simplify our future work with more advanced controls and composite modules.

While certain refinements and creative variations remain open, the essential idea was achieved: understanding how modular classes interact and how small reusable parts can evolve into complete applications. Repetition is strongly encouraged—recreate this example, experiment with custom colors, alternative layouts, or new classes such as CButton, CDialog, or CCheckBox. Each iteration deepens your intuition for how MQL5’s object model behaves.

Remember, modularization is the key to progress. Many powerful building blocks already exist in the Standard Library; our task as developers is to assemble them into intelligent systems. If you aspire to master this craft, dedicate time to studying [Object-Oriented Programming (OOP)](https://www.mql5.com/en/articles/53)—it will give you the insight to design, extend, and maintain your own scalable frameworks.

This exercise may appear simple on the surface, but it’s a vital brushstroke in your journey toward professional MQL5 development. We’ve touched on a lot, yet there’s much more to come. The attached file contains the complete implementation—feel free to test, refine, and share your thoughts. Let’s make this discussion even more insightful with your comments and creative extensions below.

| File name | Description |
| --- | --- |
| InfoPanel\_EA.mq5 | Demonstrates how to integrate multiple Standard Library classes within an Expert Advisor. The EA creates an informational display panel on the chart using CPanel, adds a descriptive text label with CChartObjectLabel, and embeds a logo image through the CPicture class. This example serves as a foundation for understanding modular UI construction and class interaction in MQL5. |
| Logo.zip | It contains a logo in the supported BMP format, which you can extract to the terminal’s Images directory. However, you’re encouraged to customize your own version using open-source tools like GIMP for a more personalized development experience. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19834.zip "Download all attachments in the single ZIP archive")

[InfoPanel\_EA.mq5](https://www.mql5.com/en/articles/download/19834/InfoPanel_EA.mq5 "Download InfoPanel_EA.mq5")(10 KB)

[Logo.zip](https://www.mql5.com/en/articles/download/19834/Logo.zip "Download Logo.zip")(1.55 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/498120)**

![Dynamic Swing Architecture: Market Structure Recognition from Swings to Automated Execution](https://c.mql5.com/2/176/19793-dynamic-swing-architecture-logo.png)[Dynamic Swing Architecture: Market Structure Recognition from Swings to Automated Execution](https://www.mql5.com/en/articles/19793)

This article introduces a fully automated MQL5 system designed to identify and trade market swings with precision. Unlike traditional fixed-bar swing indicators, this system adapts dynamically to evolving price structure—detecting swing highs and swing lows in real time to capture directional opportunities as they form.

![Royal Flush Optimization (RFO)](https://c.mql5.com/2/117/Royal_Flush_Optimization___LOGO.png)[Royal Flush Optimization (RFO)](https://www.mql5.com/en/articles/17063)

The original Royal Flush Optimization algorithm offers a new approach to solving optimization problems, replacing the classic binary coding of genetic algorithms with a sector-based approach inspired by poker principles. RFO demonstrates how simplifying basic principles can lead to an efficient and practical optimization method. The article presents a detailed analysis of the algorithm and test results.

![Neural Networks in Trading: A Multimodal, Tool-Augmented Agent for Financial Markets (FinAgent)](https://c.mql5.com/2/108/Neural_Networks_in_Trading_Multimodal_Agent_Augmented_with_Instruments____LOGO.png)[Neural Networks in Trading: A Multimodal, Tool-Augmented Agent for Financial Markets (FinAgent)](https://www.mql5.com/en/articles/16850)

We invite you to explore FinAgent, a multimodal financial trading agent framework designed to analyze various types of data reflecting market dynamics and historical trading patterns.

![MQL5 Wizard Techniques you should know (Part 85): Using Patterns of Stochastic-Oscillator and the FrAMA with Beta VAE Inference Learning](https://c.mql5.com/2/175/19948-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 85): Using Patterns of Stochastic-Oscillator and the FrAMA with Beta VAE Inference Learning](https://www.mql5.com/en/articles/19948)

This piece follows up ‘Part-84’, where we introduced the pairing of Stochastic and the Fractal Adaptive Moving Average. We now shift focus to Inference Learning, where we look to see if laggard patterns in the last article could have their fortunes turned around. The Stochastic and FrAMA are a momentum-trend complimentary pairing. For our inference learning, we are revisiting the Beta algorithm of a Variational Auto Encoder. We also, as always, do the implementation of a custom signal class designed for integration with the MQL5 Wizard.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/19834&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071558754253613641)

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