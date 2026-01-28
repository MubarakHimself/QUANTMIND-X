---
title: MQL as a Markup Tool for the Graphical Interface of MQL Programs. Part 2
url: https://www.mql5.com/en/articles/7739
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:05:35.913340
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=rggwtutnmcdrejwrhtjjqobxpjbdkpjt&ssn=1769252734903062601&ssn_dr=0&ssn_sr=0&fv_date=1769252734&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7739&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL%20as%20a%20Markup%20Tool%20for%20the%20Graphical%20Interface%20of%20MQL%20Programs.%20Part%202%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925273458249019&fz_uniq=5083330628251818366&sv=2552)

MetaTrader 5 / Examples


In [Part 1](https://www.mql5.com/en/articles/7734), we considered the basic principles of describing the graphical interface layout of MQL programs in MQL. To implement them, we had to create some classes that are directly responsible for intitializing the interface elements, combining them in a common hierarchy, and adjusting their properties. Now we are going to some more complicated examples and, in order not to be distracted by practical things, briefly pay our attention to the library of standard components, using which we will build our examples.

### Customizing the Standard Control Library

In elaborating the window interface of the earlier articles on OLAP, also based on the standard library and CBox containers, we had to correct the components of the standard library. As it turned out, to integrate the proposed layout system, the Controls library needed even more corrections — partly regarding the extension of capabilities and partly regarding error corrections. For this reason, we decided to make the full copy (the version branch) of all classes, place them in the ControlsPlus folder, and then work with them only.

Here are the main updates.

Practically in all classes, the private access level is changed for the protected one to ensure the library augmentability.

To facilitate debugging the projects containing GUI elements, string field \_rtti was added to the CWind class, and it is filled out with the name of a specific class in the constructor of each derived class, using the RTTI macro.

```
  #define RTTI _rtti = StringFormat("%s %d", typename(this), &this);
```

It allows seeing in the debugger window a real class of objectsto be dereferenced by the base class link (in this case, the debugger displays the base class).

Information on the fields and alignment of the element in class CWnd was made accessible using two new overloaded methods. Moreover, it has become possible to separately change alignment and fields.

```
    ENUM_WND_ALIGN_FLAGS Alignment(void) const
    {
      return (ENUM_WND_ALIGN_FLAGS)m_align_flags;
    }
    CRect Margins(void) const
    {
      CRectCreator rect(m_align_left, m_align_top, m_align_right, m_align_bottom);
      return rect;
    }
    void Alignment(const int flags)
    {
      m_align_flags = flags;
    }
    void Margins(const int left, const int top, const int right, const int bottom)
    {
      m_align_left = left;
      m_align_top = top;
      m_align_right = right;
      m_align_bottom = bottom;
    }
```

Method CWnd::Align was overridden in accordance with the expected behavior of all alignment modes. Standard implementation does not ensure a shift to the boundary of the pre-defined field, if the stretch is defined (both dimensions are prone thereto).

Method DeleteAll is added to the CWndContainer class to delete all child elements when deleting a container. It is called from Delete(CWnd \*control), if the pointer to the passed "control" contains a container object.

In different places of class CWndClient, we added strings that regulate the visibility of scroll bars, which may change due to resizing.

Class CAppDialog now considers the window instance\_id when assigning identifiers to the interface elements. Without this correction, controls conflicted (affected each other) in different windows having the same names.

In the groups of "controls," i.e., CRadioGroup, CCheckGroup, and CListView, the Redraw method was made virtual for the "rubber" child classes to be able to correctly respond to resizing. We also slightly corrected the recalculation of the width of their child elements.

For the same purpose, virtual method OnResize was added to classes CDatePicker, CCheckBox, and CRadioButton. In class CDatePicker, the low priority error was fixed for the pop-up calendar (mouse clicks passed through it).

Method CEdit::OnClick does not "eat" mouse clicks.

Moreover, we had already developed some classes of "controls" before, which supported resizing; and the number of "rubber" classes was extended within this specific project. Their files are located in the Layouts folder.

- ComboBoxResizable
- SpinEditResizable
- ListViewResizable
- CheckGroupResizable
- RadioGroupResizable

It should be reminded that some "controls," such as button or entry field, support stretching originally.

General structure of the standard elements library, considering the adapted versions supporting the "rubber" nature and third-party containers, is given in the classes diagram.

![Hierarchy of Controls](https://c.mql5.com/2/38/class_hierarchy__2.png)

**Hierarchy of Controls**

### Generating and Caching Elements

So far, elements were constructed as automatic instances inside the object window. In fact, these are "dummies" that are then initialized by methods, such as Create. The GUI elements layout system can independently create these elements, rather than get them from the window. For this, you only need a storage. Let us name it LayoutCache.

```
  template<typename C>
  class LayoutCache
  {
    protected:
      C *cache[];   // autocreated controls and boxes

    public:
      virtual void save(C *control)
      {
        const int n = ArraySize(cache);
        ArrayResize(cache, n + 1);
        cache[n] = control;
      }

      virtual C *get(const long m)
      {
        if(m < 0 || m >= ArraySize(cache)) return NULL;
        return cache[(int)m];
      }

      virtual C *get(const string name) = 0;
      virtual bool find(C *control);
      virtual int indexOf(C *control);
      virtual C *findParent(C *control) = 0;
      virtual bool revoke(C *control) = 0;
      virtual int cacheSize();
  };
```

In fact, this is an array (common for all elements) of the base class pointers, where they can be placed using the "save" method. In the interface, we also implement (if it is possible at this abstract level) or declare (for further re-defining) methods to search elements by number, name, link, or the fact of "parential" relations (feedback from nested elements to the container).

Let us add cache as a static member to class LayoutBase.

```
  template<typename P,typename C>
  class LayoutBase: public LayoutData
  {
    protected:
      ...
      static LayoutCache<C> *cacher;

    public:
      static void setCache(LayoutCache<C> *c)
      {
        cacher = c;
      }
```

Each window will have to create for itself a cache instance and set it as a working one using setCache at the very beginning of the method, such as CreateLayout. Since MQL programs are single-threaded, we are guaranteed that windows (if more than one are needed) won't be formed in parallel or compete on the "cacher" pointer. We are going to clean the pointer automatically in destructor LayoutBase; when the stack is finished, it means that we have left the last external container in the layout description and there is no need to save anything else.

```
      ~LayoutBase()
      {
        ...
        if(stack.size() == 0)
        {
          cacher = NULL;
        }
      }
```

Resetting a link does not mean that we are clearing cache. This is just the way to ensure that the potential next layout won't add there the "controls" of another window by mistake.

To fill the cache, we will add a new type of method init to LayoutBase — this time, without a pointer or a link to a "third-party" elements of the GUI in parameters.

```
      // nonbound layout, control T is implicitly stored in internal cache
      template<typename T>
      T *init(const string name, const int m = 1, const int x1 = 0, const int y1 = 0, const int x2 = 0, const int y2 = 0)
      {
        T *temp = NULL;
        for(int i = 0; i < m; i++)
        {
          temp = new T();
          if(save(temp))
          {
            init(temp, name + (m > 1 ? (string)(i + 1) : ""), x1, y1, x2, y2);
          }
          else return NULL;
        }
        return temp;
      }

      virtual bool save(C *control)
      {
        if(cacher != NULL)
        {
          cacher.save(control);
          return true;
        }
        return false;
      }
```

With the template, we can write new T and generate objects in laying out (by default, 1 object per time, but we can also do several ones optionally).

For the standard library elements, we have written a specific cache implementation, StdLayoutCache (it is shown here abridged, the full code is attached hereto).

```
  // CWnd implementation specific!
  class StdLayoutCache: public LayoutCache<CWnd>
  {
    public:
      ...
      virtual CWnd *get(const long m) override
      {
        if(m < 0)
        {
          for(int i = 0; i < ArraySize(cache); i++)
          {
            if(cache[i].Id() == -m) return cache[i];
            CWndContainer *container = dynamic_cast<CWndContainer *>(cache[i]);
            if(container != NULL)
            {
              for(int j = 0; j < container.ControlsTotal(); j++)
              {
                if(container.Control(j).Id() == -m) return container.Control(j);
              }
            }
          }
          return NULL;
        }
        else if(m >= ArraySize(cache)) return NULL;
        return cache[(int)m];
      }

      virtual CWnd *findParent(CWnd *control) override
      {
        for(int i = 0; i < ArraySize(cache); i++)
        {
          CWndContainer *container = dynamic_cast<CWndContainer *>(cache[i]);
          if(container != NULL)
          {
            for(int j = 0; j < container.ControlsTotal(); j++)
            {
              if(container.Control(j) == control)
              {
                return container;
              }
            }
          }
        }
        return NULL;
      }
      ...
  };
```

Note that method get searches the "control" by either its indexing number (if the input is positive) or identifier (it is signed with the minus symbol). Here, identifier shall mean a unique number assigned by the standard components library to dispatch events. In events, it is passed in parameter lparam.

In the application class of the window, we can use directly this class StdLayoutCache or write one derived from it.

How caching allows reducing the window class description, we will see in the example below. However, before going to it, let us consider some additional opportunities opened by cache. We will also use them in our examples.

### Styler

Since cache is an object that processes elements in a centralized manner, it is convenient to use it to solve many other tasks, other than laying out. Particularly, for elements, we can unify using the single style rules, such as color, font, or indents. At the same time, it is sufficient to set up this style at one location, not write the same properties for each "control" separately. Moreover, cache can undertake processing messages for cached elements. Potentially, we can dynamically construct, cache, and interact with absolutely all elements. Then there is no need at all to declare any "explicit" elements. A bit later, we will see what obvious advantage the dynamically created elements have over automated ones.

To support the centralized styles in class StdLayoutCache, a stub method is provided:

```
    virtual LayoutStyleable<C> *getStyler() const
    {
      return NULL;
    }
```

If you do not want to use styles, then no additionally coding is required. However, if you realize the advantages of centralizing the style management, you can implement the descendant class, LayoutStyleable. Interface is very simple.

```
  enum STYLER_PHASE
  {
    STYLE_PHASE_BEFORE_INIT,
    STYLE_PHASE_AFTER_INIT
  };

  template<typename C>
  class LayoutStyleable
  {
    public:
      virtual void apply(C *control, const STYLER_PHASE phase) {};
  };
```

Method apply will be called for each "control" two times: At the initialization stage (STYLE\_PHASE\_BEFORE\_INIT) and at the stage of registering in container (STYLE\_PHASE\_AFTER\_INIT). Thus, in methods LayoutBase::init, a call is added at the first stage:

```
      if(cacher != NULL)
      {
        LayoutStyleable<C> *styler = cacher.getStyler();
        if(styler != NULL)
        {
          styler.apply(object, STYLE_PHASE_BEFORE_INIT);
        }
      }
```

while into destructor, we add similar strings, but with STYLE\_PHASE\_AFTER\_INIT for the second stage.

Two phases are required, since styling goals may differ. In some elements, it is sometimes necessary to set individual properties having a higher priority over those common ones that have been set in the styler. At the initialization stage, the "control" is still empty, i.e., no settings are made in the layout. At the registration stage, all properties have already been set in it, and we can additionaly modify the style, based on them. The most obvious example is as follows. All entry fields flagged "read only" should preferably be displayed in gray. However, the "read only" property is only assigned to the "control" while laying out, after initialization. Therefore, the first stage does not suit here, and the second one is required. On the other hand, no all the fields will usually have this flag; in all other cases, it is necessary to set the default color, beofre the layout language performs the selective customization.

By the way, a similar technology can be used in the centralized localization of the MQL program interfaces into various languages.

### Handling the Events

The second function to be logically assigned to cache is event processing. For them, a stub method (C is the class template parameter) is added in class LayoutCache:

```
    virtual bool onEvent(const int event, C *control)
    {
      return false;
    }
```

Again, we can implement it in a derived class, but it is not necessary. Event codes are defined by the specific library.

For this method to start working, we need the event interception macrodefinitions similar to those available in the standard library and written in the map, as follows:

```
  EVENT_MAP_BEGIN(Dialog)
    ON_EVENT(ON_CLICK, m_button1, OnClickButton1)
    ...
  EVENT_MAP_END(AppDialog)
```

New macros will redirect the events into the cache object. One of them:

```
  #define ON_EVENT_LAYOUT_ARRAY(event, cache)  if(id == (event + CHARTEVENT_CUSTOM) && cache.onEvent(event, cache.get(-lparam))) { return true; }
```

Here we can see search inside cache by identifier that comes in lparam (but with the sign reversed), after which the element found is sent to the onEvent processor considered above. Basically, we can omit searching the element when processing each event and memorize the element index in cache, and then link the specific processor to the index.

The current cache size is the index, for which the new element has just been saved. We can save the index of the "controls" required while laying out.

```
          _layout<CButton> button1("Button");
          button1index = cache.cacheSize() - 1;
```

Here, button1index is an integer variable in the window class. It should be used in another macro defined for processing elements by the cache index:

```
  #define ON_EVENT_LAYOUT_INDEX(event, cache, controlIndex, handler)  if(id == (event + CHARTEVENT_CUSTOM) && lparam == cache.get(controlIndex).Id()) { handler(); return(true); }
```

Additionally, we can send the events directly into elements themselves, not into cache. For this purpose, the element must implement in itself interface Notifiable templated by the required "control" class.

```
  template<typename C>
  class Notifiable: public C
  {
    public:
      virtual bool onEvent(const int event, void *parent) = 0;
  };
```

In the parent parameter, any object can be passed, including a dialog box. Based on Notifiable, for example, it is easy to create a button, the CButton descendant.

```
  class NotifiableButton: public Notifiable<CButton>
  {
    public:
      virtual bool onEvent(const int event, void *anything) override
      {
        this.StateFlagsReset(7);
        return true;
      }
  };
```

There are 2 macros to work with the "notifiable" elements. They only differ in the number of parameters: ON\_EVENT\_LAYOUT\_CTRL\_ANY enables passing a random object to the last parameters, while ON\_EVENT\_LAYOUT\_CTRL\_DLG does not have this parameter, since it always sends the "this" of the dialog as an object.

```
  #define ON_EVENT_LAYOUT_CTRL_ANY(event, cache, type, anything)  if(id == (event + CHARTEVENT_CUSTOM)) {type *ptr = dynamic_cast<type *>(cache.get(-lparam)); if(ptr != NULL && ptr.onEvent(event, anything)) { return true; }}
  #define ON_EVENT_LAYOUT_CTRL_DLG(event, cache, type)  if(id == (event + CHARTEVENT_CUSTOM)) {type *ptr = dynamic_cast<type *>(cache.get(-lparam)); if(ptr != NULL && ptr.onEvent(event, &this)) { return true; }}
```

We are going to consider various options for processing events in the context of the second example.

### Case 2. Dialog with Controls

Demo project contains class CControlsDialog with the main types of the "controls" of the Standard Library. Similarly with the first case, we will delete all methods of creating them and replace them with the only one, CreateLayout. By the way, there were as many as 17 methods in the old project, and they were called one from another using compound conditional operators.

To save "controls" in cache when generating them, we will add a simple cache class and also a styling class. Here is cache first.

```
  class MyStdLayoutCache: public StdLayoutCache
  {
    protected:
      MyLayoutStyleable styler;
      CControlsDialog *parent;

    public:
      MyStdLayoutCache(CControlsDialog *owner): parent(owner) {}

      virtual StdLayoutStyleable *getStyler() const override
      {
        return (StdLayoutStyleable *)&styler;
      }

      virtual bool onEvent(const int event, CWnd *control) override
      {
        if(control != NULL)
        {
          parent.SetCallbackText(__FUNCTION__ + " " + control.Name());
          return true;
        }
        return false;
      }
  };
```

In the cache class, the event processor, onEvent, is declared, which we will connect via an event map. Here, the processor sends a message to the parent window, where it is displayed in the information field, like in the preceding case versions.

In the styler class, we provide setting identical fields for all elements, a non-standard font on all buttons, and displaying CEdit with the "read only" attribute in gray (we only have one like this, but, if any other one is added, it will automatically fall within the common setting).

```
  class MyLayoutStyleable: public StdLayoutStyleable
  {
    public:
      virtual void apply(CWnd *control, const STYLER_PHASE phase) override
      {
        CButton *button = dynamic_cast<CButton *>(control);
        if(button != NULL)
        {
          if(phase == STYLE_PHASE_BEFORE_INIT)
          {
            button.Font("Arial Black");
          }
        }
        else
        {
          CEdit *edit = dynamic_cast<CEdit *>(control);
          if(edit != NULL && edit.ReadOnly())
          {
            if(phase == STYLE_PHASE_AFTER_INIT)
            {
              edit.ColorBackground(clrLightGray);
            }
          }
        }

        if(phase == STYLE_PHASE_BEFORE_INIT)
        {
          control.Margins(DEFAULT_MARGIN);
        }
      }
  };
```

Link to cache is saved in the window; it is created and deleted, respectively, in constructor and destructor, a link to the window being passed in creating as a parameter to ensure feedback.

```
  class CControlsDialog: public AppDialogResizable
  {
    private:
      ...
      MyStdLayoutCache *cache;
    public:
      CControlsDialog(void)
      {
        cache = new MyStdLayoutCache(&this);
      }
```

Now let us consider method CreateLayout in stages. Due to reading the detailed descriptions, the method may seem to be very long and complicated. but this is not the case, indeed. If the informative comments (that are not used in the real project) are removed, the method will fit within one screen and it does not contain any complex logic.

At the very beginning, cache is activated by calling setCache. Then the main container, CControlsDialog, is described in the first block. It won't be in cache, since we pass the link to the "this" already created.

```
  bool CControlsDialog::CreateLayout(const long chart, const string name, const int subwin, const int x1, const int y1, const int x2, const int y2)
  {
    StdLayoutBase::setCache(cache); // assign the cache object to store implicit objects

    {
      _layout<CControlsDialog> dialog(this, name, x1, y1, x2, y2);
```

Then an implicit instance of the nested container of class CBox is created for the client area of the window. It is oriented vertically, so the nested containers will fill out the space from top to bottom. We save the link to the object in variable m\_main, since we will have to call its method Pack upon resizing the window. If your dialog is not a "rubber" one, you don't need to do so. Finally, for the client area, zero fields and alignment in all directions are set for the panel to fill the entire window, even upon resizing.

```
      {
        // example of implicit object in the cache
        _layout<CBox> clientArea("main", ClientAreaWidth(), ClientAreaHeight(), LAYOUT_STYLE_VERTICAL);
        m_main = clientArea.get(); // we can get the pointer to the object from cache (if required)
        clientArea <= WND_ALIGN_CLIENT <= 0.0; // double type is important
```

At the next level, the container follows as the first, which will fill the entire window width, but it will be just slightly higher than the entry field. Moreover, it will be "glued" to the upper edge of the window, using alignment WND\_ALIGN\_TOP (along with WND\_ALIGN\_WIDTH).

```
        {
          // another implicit container (we need no access it directly)
          _layout<CBox> editRow("editrow", ClientAreaWidth(), EDIT_HEIGHT * 1.5, (ENUM_WND_ALIGN_FLAGS)(WND_ALIGN_TOP|WND_ALIGN_WIDTH));
```

There is the only "control" of class CEdit in the "read only" mode inside. Explicit variable m\_edit is reserved for it, so it won't get to cache.

```
          {
            // for editboxes default boolean property is ReadOnly
            _layout<CEdit> edit(m_edit, "Edit", ClientAreaWidth(), EDIT_HEIGHT, true);
          }
        }
```

By this time, we have already initialized 3 elements. After the closed bracket, the "edit" layout object will be destructed, and in the course of executing its destructor, m\_edit, will be added to container "editrow." However, another closing bracket follows immediately. It destructs the context, in which the layout object, editRow, "lived." So this container, in turn, is added to the client area container that remains on the stack. Thus, the first row is formed for the vertical layout in m\_main.

Then we have a row with three buttons. First, a container is created for it.

```
        {
          _layout<CBox> buttonRow("buttonrow", ClientAreaWidth(), BUTTON_HEIGHT * 1.5);
          buttonRow["align"] <= (WND_ALIGN_CONTENT|WND_ALIGN_WIDTH);
```

Here, you should note the non-standard way of aligning WND\_ALIGN\_CONTENT. It means the following.

To class CBox, algorithm is added to scale the nested elements for the container size. It is executed in method AdjustFlexControls and only comes into effect, if a spacial value of WND\_ALIGN\_CONTENT is specified in the flags of container alignment. It is not a part of the standard enumeration, ENUM\_WND\_ALIGN\_FLAGS. Container analyzes "controls" regarding which of them have a fixed size and which ones don't. "Controls" with a fixed size are those, for which no alignment is specified by the container sides (in a specific dimension). For all such "controls," the container calculates the sum of their sizes, subtracts it from the total container size, and divides the remainder proportionally among the remaining "controls." For example, if there are two "controls" in the container and none of them has binding, then they go halves with each other in the entire container area.

It is a very convenient mode, but you should not misuse it on a set of interleaved containers — due to the single-pass algorithm of calculating the sizes, internal elements are aligned over the area of the container, which, in turn, adjusts to the contents and generates uncertainty (for this reason, a special event, ON\_LAYOUT\_REFRESH, is made in layout classes, which the window can send to itself to repeat the calculation of sizes).

In case of our row with three buttons, they all will proportionally change their lengths when the window width is resized. First button of class CButton is created implicitly and stored in cache.

```
          { // 1
            _layout<CButton> button1("Button1");
            button1index = cache.cacheSize() - 1;
            button1["width"] <= BUTTON_WIDTH;
            button1["height"] <= BUTTON_HEIGHT;
          } // 1
```

Second button has class NotifiableButton (it has already been described above). The button will process messages by itself.

```
          { // 2
            _layout<NotifiableButton> button2("Button2", BUTTON_WIDTH, BUTTON_HEIGHT);
          } // 2
```

Third button is created based on the explicitly defined window variable, m\_button3, and has a "sticking" property.

```
          { // 3
            _layout<CButton> button3(m_button3, "Button3", BUTTON_WIDTH, BUTTON_HEIGHT, "Locked");
            button3 <= true; // for buttons default boolean property is Locking
          } // 3
        }
```

Please note that all buttons are enframed in their own blocks of braces. due to this, they are added into the row in the order, in which closing braces appear, which are marked as 1, 2, and 3; i.e., in a natural order. We could omit making these "personal" blocks for each button and get limited with the general block of the container. But then the buttons would be added in a reversed order, because the destructors of the objects are always called in the order reversed to that of creating them. We could "fix" the situation by inversing the order of describing the buttons in the layout.

In the third row, there is a container with the controls, spinner and calendar. The container is created "anonymously" and stored in cache.

```
        {
          _layout<CBox> spinDateRow("spindaterow", ClientAreaWidth(), BUTTON_HEIGHT * 1.5);
          spinDateRow["align"] <= (WND_ALIGN_CONTENT|WND_ALIGN_WIDTH);

          {
            _layout<SpinEditResizable> spin(m_spin_edit, "SpinEdit", GROUP_WIDTH, EDIT_HEIGHT);
            spin["min"] <= 10;
            spin["max"] <= 1000;
            spin["value"] <= 100; // can set value only after limits (this is how SpinEdits work)
          }

          {
            _layout<CDatePicker> date(m_date, "Date", GROUP_WIDTH, EDIT_HEIGHT, TimeCurrent());
          }
        }
```

Finally, the last container fills all the remaining area of the window and contains two columns with elements. Bright colors are exclusively assigned to clearly demonstrate which container is where in the window.

```
        {
          _layout<CBox> listRow("listsrow", ClientAreaWidth(), LIST_HEIGHT);
          listRow["top"] <= (int)(EDIT_HEIGHT * 1.5 * 3);
          listRow["align"] <= (WND_ALIGN_CONTENT|WND_ALIGN_CLIENT);
          (listRow <= clrMagenta)["border"] <= clrBlue;

          createSubList(&m_lists_column1, LIST_OF_OPTIONS);
          createSubList(&m_lists_column2, LIST_LISTVIEW);
          // or vice versa (changed order gives swapped left/right side location)
          // createSubList(&m_lists_column1, LIST_LISTVIEW);
          // createSubList(&m_lists_column2, LIST_OF_OPTIONS);
        }
```

Here, it should be specially noted that two columns, m\_lists\_column1 and m\_lists\_column2, are filled out not in method CreateLayout itself, but using the helper method, createSubList. In terms of layout, the function is called in a manner that does not differ from entering into the next block of braces. It means that the layout does not necessarily consist of a long static list, but it may include fragments that are modified by condition. Or you can include the same fragment into different dialogs.

In our case, we can change the order of columns in the window, by changing the second parameter of the function.

```
      }
    }
```

Upon closing all braces, all the GUI elements are initialized and connected to each other. We call method Pack (directly or via SelfAdjustment, where it is also called as a response to requesting a "rubber" dialog).

```
    // m_main.Pack();
    SelfAdjustment();
    return true;
  }
```

We are not going to get into details of method createSubList. Inside, the possibilities are implemented that allow generating a set of 3 "controls" (combo-box, group of options, and group of radiocolumns) or a list (ListView), all being made as "rubber" ones. Of interest is that "controls" are filled using another class of generators, ItemGenerator.

```
  template<typename T>
  class ItemGenerator
  {
    public:
      virtual bool addItemTo(T *object) = 0;
  };
```

The only method of this class is called from layout for the object "control", until the method returns false (a sign of the data end).

By default, some simple generators are provided for the standard library (they use the method of "controls", AddItem): StdItemGenerator, StdGroupItemGenerator, SymbolsItemGenerator, and ArrayItemGenerator. Particularly, SymbolsItemGenerator allows filling the "control" with the symbols from Market Watch.

```
  template<typename T>
  class SymbolsItemGenerator: public ItemGenerator<T>
  {
    protected:
      long index;

    public:
      SymbolsItemGenerator(): index(0) {}

      virtual bool addItemTo(T *object) override
      {
        object.AddItem(SymbolName((int)index, true), index);
        index++;
        return index < SymbolsTotal(true);
      }
  };
```

In the layout, it is specified in the same manner, as the generators of "controls." Alternatively, it is allowed to pass to the layout object the link to a pointer to the dynamically distributed object of generator, rather than to an automated or static one (that must be described somewhere earlier in the code).

```
        _layout<ListViewResizable> list(m_list_view, "ListView", GROUP_WIDTH, LIST_HEIGHT);
        list <= WND_ALIGN_CLIENT < new SymbolsItemGenerator<ListViewResizable>();
```

For this purpose, operator < is used. Dynamically distributed generator will be deleted automatically upon completing the work.

To connect new events, the relevant macros are added to the map.

```
  EVENT_MAP_BEGIN(CControlsDialog)
    ...
    ON_EVENT_LAYOUT_CTRL_DLG(ON_CLICK, cache, NotifiableButton)
    ON_EVENT_LAYOUT_INDEX(ON_CLICK, cache, button1index, OnClickButton1)
    ON_EVENT_LAYOUT_ARRAY(ON_CLICK, cache)
  EVENT_MAP_END(AppDialogResizable)
```

Macro ON\_EVENT\_LAYOUT\_CTRL\_DLG connects notifications on mouse clicks for any buttons of class NotifiableButton (in our case, it is a single one). Macro ON\_EVENT\_LAYOUT\_INDEX sends the same event into the button with the specified index in cache. However, we could omit writing this macro, since macro ON\_EVENT\_LAYOUT\_ARRAY will send with the last string the mouse click to any element in cache, provided that its identifier coincides with lparam.

Basically, all elements could be passed to cache, and their events could be processed in a new manner; however, the old one works, too, and they can be combined.

In the following animated image, the response to the events is shown.

![Controls-Containing Dialog Formed Using the MQL Markup Language](https://c.mql5.com/2/38/layout4__1.gif)

**Controls-Containing Dialog Formed Using the MQL Markup Language**

Please note that the way of translating an event can be indirectly identified by the signature of the function displayed in the information field. You can also see that the events come in both the "controls" and containers. Red frames of containers are displayed for debugging, and you can disable them using macro LAYOUT\_BOX\_DEBUG.

### Case 3. Dynamic Layouts of DynamicForm

In this last example, we are going to consider the form, in which all elements will be dynamically created in cache. This will give us a couple of new important opportunities.

Like in previous case, cache will support styling the elements. The only style setting is identical distinctive fields that allow seeing the nesting of containers and select them using your mouse, if so desired.

The following simple interface structure is described inside method CreateLayout. As usual, the main container fills the entire client area of the window. In the upper part, there is a block with two buttons: Inject and Export. All the space below them is filled with the container divided into the left and right columns. Left column marked in gray is originally empty. In the right column, a group of radiobuttons is located, which allows selecting the control type.

```
      {
        // example of implicit object in the cache
        _layout<CBoxV> clientArea("main", ClientAreaWidth(), ClientAreaHeight());
        m_main = clientArea.get();
        clientArea <= WND_ALIGN_CLIENT <= PackedRect(10, 10, 10, 10);
        clientArea["background"] <= clrYellow <= VERTICAL_ALIGN_TOP;

        {
          _layout<CBoxH> buttonRow("buttonrow", ClientAreaWidth(), BUTTON_HEIGHT * 5);
          buttonRow <= 5.0 <= (ENUM_WND_ALIGN_FLAGS)(WND_ALIGN_TOP|WND_ALIGN_WIDTH);
          buttonRow["background"] <= clrCyan;

          {
            // these 2 buttons will be rendered in reverse order (destruction order)
            // NB: automatic variable m_button3
            _layout<CButton> button3(m_button3, "Export", BUTTON_WIDTH, BUTTON_HEIGHT);
            _layout<NotifiableButton> button2("Inject", BUTTON_WIDTH, BUTTON_HEIGHT);
          }
        }

        {
          _layout<CBoxH> buttonRow("buttonrow2", ClientAreaWidth(), ClientAreaHeight(),
            (ENUM_WND_ALIGN_FLAGS)(WND_ALIGN_CONTENT|WND_ALIGN_CLIENT));
          buttonRow["top"] <= BUTTON_HEIGHT * 5;

          {
            {
              _layout<CBoxV> column("column1", GROUP_WIDTH, 100, WND_ALIGN_HEIGHT);
              column <= clrGray;
              {
                // dynamically created controls will be injected here
              }
            }

            {
              _layout<CBoxH> column("column2", GROUP_WIDTH, 100, WND_ALIGN_HEIGHT);

              _layout<RadioGroupResizable> selector("selector", GROUP_WIDTH, CHECK_HEIGHT);
              selector <= WND_ALIGN_HEIGHT;
              string types[3] = {"Button", "CheckBox", "Edit"};
              ArrayItemGenerator<RadioGroupResizable,string> ctrls(types);
              selector <= ctrls;
            }
          }
        }
      }
```

It is supposed that, upon having selected the element type in a radiogroup, the user pushes the Inject button, and the relevant "control" is created in the left part of the window. Of course, you can create several different "controls" one by one. The will be centered automatically according to the container settings. To implement this logic, the Inject button has class NotifiableButton with processor onEvent.

```
  class NotifiableButton: public Notifiable<CButton>
  {
      static int count;

      StdLayoutBase *getPtr(const int value)
      {
        switch(value)
        {
          case 0:
            return new _layout<CButton>("More" + (string)count++, BUTTON_WIDTH, BUTTON_HEIGHT);
          case 1:
            return new _layout<CCheckBox>("More" + (string)count++, BUTTON_WIDTH, BUTTON_HEIGHT);
          case 2:
            return new _layout<CEdit>("More" + (string)count++, BUTTON_WIDTH, BUTTON_HEIGHT);
        }
        return NULL;
      }

    public:
      virtual bool onEvent(const int event, void *anything) override
      {
        DynamicForm *parent = dynamic_cast<DynamicForm *>(anything);
        MyStdLayoutCache *cache = parent.getCache();
        StdLayoutBase::setCache(cache);
        CBox *box = cache.get("column1");
        if(box != NULL)
        {
          // put target box to the stack by retrieving it from the cache
          _layout<CBox> injectionPanel(box, box.Name());

          {
            CRadioGroup *selector = cache.get("selector");
            if(selector != NULL)
            {
              const int value = (int)selector.Value();
              if(value != -1)
              {
                AutoPtr<StdLayoutBase> base(getPtr(value));
                (~base).get().Id(rand() + (rand() << 32));
              }
            }
          }
          box.Pack();
        }

        return true;
      }
  };
```

Container, into which new elements should be inserted, is first searched for in cache by name "column1". This container goes as the first parameter when creating object injectionPanel. The fact that the element to be passed is in cache already is specifically considered in the layout algorithm — it is not cached again, but usually put into the container stack. This allows adding elements to "old" containers.

Based on the user's choice, an object of the required type is created using operator "new" in helper method getPtr. For the "controls" added to work correctly, unique identifiers are generated for them randomly. Special class, AutoPtr ensures deleting the pointer when exiting from the code block.

If too many elements are added, they will go beyond the container boundaries. This happens, because our available container classes have not learned yet how to respond to overflow. In this case, we could, for example, show the scroll bar, while the elements beyond the boundaries could be hidden.

It is not importnat, though. The point of this case is that we can generate dynamic contents by setting up the form and ensure the necessary contents and sizes of containers.

Along with adding elements, this dialog can delete them. Any element in the form can be selected by a mouse click. At the same time, the class and name of the element are logged, while the element itself is highlighted with a red frame. If you click on an element already selected, the dialog will display a request for confirming the deletion and, if it is confirmed, delete the element. All this is implemented in our cache class.

```
  class MyStdLayoutCache: public StdLayoutCache
  {
    protected:
      DynamicForm *parent;
      CWnd *selected;

      bool highlight(CWnd *control, const color clr)
      {
        CWndObj *obj = dynamic_cast<CWndObj *>(control);
        if(obj != NULL)
        {
          obj.ColorBorder(clr);
          return true;
        }
        else
        {
          CWndClient *client = dynamic_cast<CWndClient *>(control);
          if(client != NULL)
          {
            client.ColorBorder(clr);
            return true;
          }
        }
        return false;
      }

    public:
      MyStdLayoutCache(DynamicForm *owner): parent(owner) {}

      virtual bool onEvent(const int event, CWnd *control) override
      {
        if(control != NULL)
        {
          highlight(selected, CONTROLS_BUTTON_COLOR_BORDER);

          CWnd *element = control;
          if(!find(element)) // this is an auxiliary object, not a compound control
          {
            element = findParent(control); // get actual GUI element
          }

          if(element == NULL)
          {
            Print("Can't find GUI element for ", control._rtti + " / " + control.Name());
            return true;
          }

          if(selected == control)
          {
            if(MessageBox("Delete " + element._rtti + " / " + element.Name() + "?", "Confirm", MB_OKCANCEL) == IDOK)
            {
              CWndContainer *container;
              container = dynamic_cast<CWndContainer *>(findParent(element));
              if(container)
              {
                revoke(element); // deep remove of all references (with subtree) from cache
                container.Delete(element); // delete all subtree of wnd-objects

                CBox *box = dynamic_cast<CBox *>(container);
                if(box) box.Pack();
              }
              selected = NULL;
              return true;
            }
          }
          selected = control;

          const bool b = highlight(selected, clrRed);
          Print(control.Name(), " -> ", element._rtti, " / ", element.Name(), " / ", b);

          return true;
        }
        return false;
      }
  };
```

We can delete any interface element available in cache, i.e., not only those added by the Inject button. In this manner, you can, for example, delete the entire left half or the right "radiobox." Most interesting thing will happen, if we try to delete the upper container with two buttons. This will result in that the Export button won't be bound to the dialog anymore and will stay in the chart.

![Editable Form: Adding and Deleting Elements](https://c.mql5.com/2/38/dynamic1__1.gif)

**Editable Form: Adding and Deleting Elements**

This happens, since it is the only element that is intentionally described as an automatic, not dynamic variable (in the form class, there is an instance of CButton, m\_button3).

When the standard library tries to delete interface elements, it delegates this to array class CArrayObj, which, in turn, checks the pointer type and only deletes objects with POINTER\_DYNAMIC. Thus, it becomes clear that, to construct an adaptive interface where elements can replace each other or be deleted completely, it is desirable to use dynamic placement, and cache offers an ready solution for this.

Finally, let us refer to the second button of the dialog, Export. As we can see from its name, it is designed to save the current state of the dialog as a text file in the MQL-layout syntax considered. Of course, the form allows setting up its appearance to a limited extent only. But the possibility itself to export the appearance into the ready MQL code, which you can then easily copy to the program and get the same interface, potentially represents quite a valuable technology. Of course, only interface is exported, while you have to independently enable the event processing code or general settings.

Exporting is ensured by class LayoutExporter; we are not going to consider it in all details, and the source codes are attached hereto.

### Conclusions

In this article, we have checked the implementability of the concept of describing the graphical interface layout of MQL programs in the MQL itself. Using the dynamic generation of elements with the centralized storage in cache allows facilitating the creation of and control over the hierarchy of components. Based on cache, you can implement the majority of tasks related to designing interface, particularly, unified restyling, event processing, editing the layout on-the-fly, and saving it in a format suitable for subsequent use.

If we fit these functions together, it will turn out that practically everything is available for a simple visual form editor. It could support just the most important properties that are common for many "controls," but, nevertheless, it would allow forming interface templates. However, we can see that even the initial stage of assessing this new concept has taken much work. Therefore, the practical implementation of the new editor represents quite a complex problem. And this is another story.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7739](https://www.mql5.com/ru/articles/7739)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7739.zip "Download all attachments in the single ZIP archive")

[MQL5GUI2.zip](https://www.mql5.com/en/articles/download/7739/mql5gui2.zip "Download MQL5GUI2.zip")(98.72 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Backpropagation Neural Networks using MQL5 Matrices](https://www.mql5.com/en/articles/12187)
- [Parallel Particle Swarm Optimization](https://www.mql5.com/en/articles/8321)
- [Custom symbols: Practical basics](https://www.mql5.com/en/articles/8226)
- [Calculating mathematical expressions (Part 2). Pratt and shunting yard parsers](https://www.mql5.com/en/articles/8028)
- [Calculating mathematical expressions (Part 1). Recursive descent parsers](https://www.mql5.com/en/articles/8027)
- [MQL as a Markup Tool for the Graphical Interface of MQL Programs (Part 3). Form Designer](https://www.mql5.com/en/articles/7795)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/347482)**
(16)


![Dmitry Fedoseev](https://c.mql5.com/avatar/2014/9/54056F23-4E95.png)

**[Dmitry Fedoseev](https://www.mql5.com/en/users/integer)**
\|
4 Apr 2020 at 12:13

**Koldun Zloy:**

It's actually pretty clear there for someone who knows OOP.

Those who don't know it are to blame.

What's the connection?

![Dmitry Fedoseev](https://c.mql5.com/avatar/2014/9/54056F23-4E95.png)

**[Dmitry Fedoseev](https://www.mql5.com/en/users/integer)**
\|
4 Apr 2020 at 12:13

**Koldun Zloy:**

The point of this markup language is that it does not require a separate interpreter. It is embedded directly into the programme code.

But without OOP knowledge, you cannot see a lot of things in the articles.

And since you don't plan to [study OOP](https://www.mql5.com/en/articles/703 "Getting acquainted with object-oriented programming in MQL5"), why are you here at all?

Who is "he" embedding?

![Dmitry Fedoseev](https://c.mql5.com/avatar/2014/9/54056F23-4E95.png)

**[Dmitry Fedoseev](https://www.mql5.com/en/users/integer)**
\|
4 Apr 2020 at 12:15

As Wini Pooh used to say: markup language is a very strange subject, it's either there or it's not.))


![Реter Konow](https://c.mql5.com/avatar/avatar_na2.png)

**[Реter Konow](https://www.mql5.com/en/users/peterkonow)**
\|
4 Apr 2020 at 12:18

**Dmitry Fedoseev:**

Who "he" is embedding?

The point)))))

![xthomasm](https://c.mql5.com/avatar/2020/6/5ED9272E-F583.png)

**[xthomasm](https://www.mql5.com/en/users/xthomasm)**
\|
24 Jul 2020 at 15:45

Decided to come to the original article just to thank you for the work that you're putting on developing this library.

Amazing job, I really appreciated.


![Continuous Walk-Forward Optimization (Part 6): Auto optimizer's logical part and structure](https://c.mql5.com/2/38/MQL5-avatar-continuous_optimization__3.png)[Continuous Walk-Forward Optimization (Part 6): Auto optimizer's logical part and structure](https://www.mql5.com/en/articles/7718)

We have previously considered the creation of automatic walk-forward optimization. This time, we will proceed to the internal structure of the auto optimizer tool. The article will be useful for all those who wish to further work with the created project and to modify it, as well as for those who wish to understand the program logic. The current article contains UML diagrams which present the internal structure of the project and the relationships between objects. It also describes the process of optimization start, but it does not contain the description of the optimizer implementation process.

![Timeseries in DoEasy library (part 39): Library-based indicators - preparing data and timeseries events](https://c.mql5.com/2/38/MQL5-avatar-doeasy-library__4.png)[Timeseries in DoEasy library (part 39): Library-based indicators - preparing data and timeseries events](https://www.mql5.com/en/articles/7724)

The article deals with applying DoEasy library for creating multi-symbol multi-period indicators. We are going to prepare the library classes to work within indicators and test creating timeseries to be used as data sources in indicators. We will also implement creating and sending timeseries events.

![Developing a cross-platform grid EA: testing a multi-currency EA](https://c.mql5.com/2/38/mql5_ea_adviser_grid.png)[Developing a cross-platform grid EA: testing a multi-currency EA](https://www.mql5.com/en/articles/7777)

Markets dropped down by more that 30% within one month. It seems to be the best time for testing grid- and martingale-based Expert Advisors. This article is an unplanned continuation of the series "Creating a Cross-Platform Grid EA". The current market provides an opportunity to arrange a stress rest for the grid EA. So, let's use this opportunity and test our Expert Advisor.

![Multicurrency monitoring of trading signals (Part 4): Enhancing functionality and improving the signal search system](https://c.mql5.com/2/38/Article_Logo__1.png)[Multicurrency monitoring of trading signals (Part 4): Enhancing functionality and improving the signal search system](https://www.mql5.com/en/articles/7678)

In this part, we expand the trading signal searching and editing system, as well as introduce the possibility to use custom indicators and add program localization. We have previously created a basic system for searching signals, but it was based on a small set of indicators and a simple set of search rules.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/7739&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083330628251818366)

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