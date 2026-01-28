---
title: MQL as a Markup Tool for the Graphical Interface of MQL Programs. Part 1
url: https://www.mql5.com/en/articles/7734
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:05:45.637943
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/7734&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083332625411611015)

MetaTrader 5 / Examples


Whether MQL-based programs need a graphical window interface? Agreement is lacking about it. On the one hand, the trader's dream is the simplest way of communicating with a trading robot — a button that enables trading and starts "coining money" magically. On the other hand, because it's a dream, it's far from reality, since you usually have to select a whole mess of settings painstakingly and for a long time, before the system starts working; however, even after that, you have to control it and correct it manually, if necessary. I say nothing of allegiants of completely manual trading — in their case, selecting a comfortable intuitive trading panel is half the battle. Generally, it can be said that window interface, in one form or another, would sooner be necessary than not.

### Introduction to the GUI Markup Technology

To construct a graphical interface, MetaTrader provides some highly demanded control elements both as independent objects to be placed on charts and as the ones wrapped in the "controls" of the standard library, which can be organized as a single interactive window. There are also some alternative solutions for constructing a GUI. However, all these libraries very rarely touch upon the layout of elements, i.e., somewhat of interface design automation.

Of course, it is a rare occasion when somebody hits upon the idea of drawing in the chart a window that would equal to MetaTrader itself; however, even a seemingly simple trading panel can consist of dozens of "controls," controlling which from MQL turns into a true monotony.

Layout is a unified way to describe the arrangements and attributes of interface elements, based on which we can ensure automatically creating the windows and linking them to the control code.

Let us remember how interface is created in the standard instances of MQL.

```
  bool CPanelDialog::Create(const long chart, const string name, const int subwin, const int x1, const int y1, const int x2, const int y2)
  {
    if(!CAppDialog::Create(chart, name, subwin, x1, y1, x2, y2)) return(false);
    // create dependent controls
    if(!CreateEdit()) return(false);
    if(!CreateButton1()) return(false);
    if(!CreateButton2()) return(false);
    if(!CreateButton3()) return(false);
    ...
    if(!CreateListView()) return(false);
    return(true);
  }

  bool CPanelDialog::CreateButton2(void)
  {
    // coordinates
    int x1 = ClientAreaWidth() - (INDENT_RIGHT + BUTTON_WIDTH);
    int y1 = INDENT_TOP + BUTTON_HEIGHT + CONTROLS_GAP_Y;
    int x2 = x1 + BUTTON_WIDTH;
    int y2 = y1 + BUTTON_HEIGHT;

    if(!m_button2.Create(m_chart_id, m_name + "Button2", m_subwin, x1, y1, x2, y2)) return(false);
    if(!m_button2.Text("Button2")) return(false);
    if(!Add(m_button2)) return(false);
    m_button2.Alignment(WND_ALIGN_RIGHT, 0, 0, INDENT_RIGHT, 0);
    return(true);
  }
  ...
```

Everything is done in an imperative style, using many calls of the same type. MQL code comes out to be long and non-efficient, in terms of repeating it for each element, the own constants (the so-called "magic numbers" that are considered a potential source of errors) being used in each case. Writing such a code is a thankless task (particularly, the [Copy&Paste errors](https://en.wikipedia.org/wiki/Copy-and-paste_programming "https://en.wikipedia.org/wiki/Copy-and-paste_programming") have become proverbial among developers), and where you need to insert a new element and shift the older ones, you will most likely have to manually recalculate and modify many "magic numbers."

Below is how the description of interface elements looks in the dialog class.

```
  CEdit        m_edit;          // the display field object
  CButton      m_button1;       // the button object
  CButton      m_button2;       // the button object
  CButton      m_button3;       // the fixed button object
  CSpinEdit    m_spin_edit;     // the up-down object
  CDatePicker  m_date;          // the datepicker object
  CListView    m_list_view;     // the list object
  CComboBox    m_combo_box;     // the dropdown list object
  CRadioGroup  m_radio_group;   // the radio buttons group object
  CCheckGroup  m_check_group;   // the check box group object
```

This flat list of "controls" may be very long, and it is hard to perceive and maintain it without any visual "hints" a layout could provide.

In other programming languages, interface design is normally separated from coding. Declarative languages, such as XML or JSON, are used to describe the layout of elements.

In particular, basic principles of describing interface elements for Android projects can be found in the documents or in [tutorials](https://www.mql5.com/go?link=https://www.tutorialspoint.com/android/android_user_interface_layouts.htm "https://www.tutorialspoint.com/android/android_user_interface_layouts.htm"). To get the gist of it, you have just to have a general idea of XML. In such files, hierarchy is clearly in evidence, container elements, such as LinearLayout or RelativeLayout, and single "controls," such as ImageView, TextView, or CheckBox, are defined, automatically adjusting the sizes to the content, such as match\_parent or wrap\_content, and linkes to the centralized style descriptions are defined in the settings, and event processors are specified optionally, although all elements can surely be additionally adjusted, and other event processors can be attached to them from the executable code.

If we remember the .Net platform, they also use a similar declarative description of interfaces using [XAML](https://www.mql5.com/go?link=https://docs.microsoft.com/en-us/dotnet/desktop-wpf/fundamentals/xaml "https://docs.microsoft.com/en-us/dotnet/desktop-wpf/fundamentals/xaml"). Even for those who have never coded in C# or any other [managed code](https://en.wikipedia.org/wiki/Managed_code "https://en.wikipedia.org/wiki/Managed_code") infrastructure languages (the concept of which, as a matter of fact, is very similar to the MetaTrader platform and its "managed" MQL), the core elements, such as "controls," containers, properties, and responses to the user's actions, are also visible here, like all-in-one.

Why is layout separated from code and described in a special language? Here are the basic benefits of such approach.

- Visual presentation of hierarchic relations among elements and containers;
- Logical grouping;
- Unified definition of layout and alignment;
- Easily writing the properties and their values;
- Declarations allow implementing the automatic generation of the code maintaining the lifecycle and control of elements, such as creating, setting up, interactiing, and deleting;
- Generalized abstraction level, i.e., general properties, states, and initialization/processing phases, which allows developing the GUI independently on coding;
- Repeated (multiple) uses of layouts, i.e., the same fragment can be included in different dialogs several times;
- Dynamic content implementation/generation on-the-fly, in a manner similar to switching among tabs, a specific set of elements being used for each of them;
- Dynamic creation of "controls" inside the layout, saving them in a single array of pointers to the basic class, such as CWnd, in case of the standard MQL library; and
- Using a specific graphic editor for the interactive interface design — in this case, the special format of describing the layouts acts as a connecting link between the external representation of the program and its executive part in the programming language.

For the MQL environment, just a few shots have been made at solving some of these problems. Particularly, a visual dialog designer is presented in the article [How to Design and Construct Object Classes](https://www.mql5.com/en/articles/53). It works based on the [MasterWindows library](https://www.mql5.com/en/code/15883). However, the ways of arranging layouts and the list of element types supported are considerably limited in it.

A more advanced layout system, although without a visual designer, is proposed in the articles [Using Layouts and Containers for GUI Controls: The CBox Class](https://www.mql5.com/en/articles/1867) and [The CGrid Class](https://www.mql5.com/en/articles/1998). It supports all standard control elements and other ones, inherited from CWndObj or CWndContainer, but still leaves the routine coding aimed at creating and arranging components to the user.

Conceptually, this approach with containers is very advanced (if suffices to mention its popularity in practically all markup languages). Therefore, we are going to take heed of it. In one of my earlier articles ( [Applying OLAP in Trading (Part 2): Visualizing the Interactive Multidimensional Data Analysis Results](https://www.mql5.com/en/articles/6603)), I proposed a modification of containers CBox and CGrid, as well as some control elements to support the "rubber" properties. Below, we're going to use those developments and improve them to solve the problem of automatically arranging elements, exemplified by the objects of a standard library.

### Graphic Editor of Interface: Pros and Contras

The main function of the graphic interface editor is to creat and set up the properties of elements in the window in the on-the-fly manner, by the user's commands. This suggests using input fields to select properties; for them to work, you should know the list of properties and their types for each class. Thus, every "control" must have two interrelated versions: The so-called run-time ones (for standard operations) and the design-time ones (for interactively designing your interface). "Controls" have the first one by default — it is the class that operates in windows. The second version is the wrap of the "control," intended for vewing and changing its available properties. It would be a hard job to write such a wrap for each type of elements. Therefore, it would be desirable to automate this process. Theoretically, you can use for this pupose the MQL parser described in the article titled [MQL Parsing by means of MQL](https://www.mql5.com/en/articles/5638). In many programming languages, the concept of property is put in the language syntax and combines a "setter" and a "getter" of a certain internal field of the object. MQL does not have this so far, but a similar principle is used in the window classes of the standard library: To set and read the same field, a pair of "mirror" methods with the same name are used — the one takes the value of a specific type, and the other one returns it. For instance, this is how the "ReadOnly" property is defined for the CEdit input field:

```
    bool ReadOnly(void) const;
    bool ReadOnly(const bool flag);
```

And this is how it enables working with the upper limit of CSpinEdit:

```
    int  MaxValue(void) const;
    void MaxValue(const int value);
```

Using the MQL parser, you can find these pairs of methods in each class and then include them into a general list, considering the inheritance hierarchy, after which you can generat a wrapper class to interactively set and read the properties found. You have to do so only once for each class of "controls" (provided that the class will not change its public properties).

An implementable project, if even a large-scale one. Before tackling it, you should consider all its pros and contras.

Let us emphasize 2 core design goals: Identifying the hierarchic dependences of elements and their properties. If any alternative ways were found to achieve them, we could omit the visual editor.

Upon conscious reflection, it becomes clear that the basic properties of all elements are standard, i.e., type, size, alignment, text, and style (color). You can also set specific properties in the MQL code. Thankfully, these are single operations that are usually associated with the business logic. As to the type, size, and alignment, they are implicitly set by the objects hierarchy itself.

Thus, we are coming to the conclusion that, in most cases, instead of a full-fledged editor, it is sufficient to have a convenient way to desctibe the hierarchy of the interface elements.

Imagine that all control elements and containers within the dialog class are described not by a continuous list, but with an indent simulating a tree structure of nesting/dependence.

```
    CBox m_main;                       // main client window

        CBox m_edit_row;                   // top level container/group
            CEdit m_edit;                      // control

        CBox m_button_row;                 // top level container/group
            CButton m_button1;                 // control
            CButton m_button2;                 // control
            CButton m_button3;                 // control

        CBox m_spin_date_row;              // top level container/group
            SpinEdit m_spin_edit;              // control
            CDatePicker m_date;                // control

        CBox m_lists_row;                  // top level container/group

            CBox m_lists_column1;              // nested container/group
                ComboBox m_combo_box;              // control
                CRadioGroup m_radio_group;         // control
                CCheckGroup m_check_group;         // control

            CBox m_lists_column2;              // nested container/group
                CListView m_list_view;             // control
```

In this way, the structure is much more visible. However, the changed formatting does not, of course, affect in any way the program ability to interpret these objects in a special manner.

Ideally, we would like to have a method to describe the interface, based on which the control elements would be created by themselves, in accordance with a defined hierarchy, find a right place on the screen, and compute the suitable size.

### Designing the Markup Language

Thus, we have to develop a markup language describing the general structure of the window interface and the properties of its individual elements. Here, we could rely on the widely used XML format and reserve a set of relevant tags. We could even borrow them from another framework, such as those mentioned above. But then we would have to parse XML and then interpret it into MQL, translating it into the operations of creating and adjusting objects. Moreover, since there is no need for a visual editor anymore, the "external" markup language has also become unnecessary as a means of communication between the editor and the runtime environment.

In such conditions, an idea appears: Can MQL itself be used as the markup language? It can, indeed.

Hierarchy is initially incorporated in MQL. Classes inherited one from another come to mind immediately. But classes describe the static hierarchy formed before executing the code. However, we need a hierarchy that could be interpreted as the MQL code is getting executed. In some other programming languages, for this purpose, i.e., analyzing the hierarchy and internal structure of classes from the program itself, there is an embedded tool, the so-called [run-time type information, RTTI](https://en.wikipedia.org/wiki/Run-time_type_information "https://en.wikipedia.org/wiki/Run-time_type_information"), also known as [reflections](https://en.wikipedia.org/wiki/Reflection_(computer_programming) "https://en.wikipedia.org/wiki/Reflection_(computer_programming)"). But MQL does not have such tools.

However, MQL has another hierarchy, like in most programming languages: Hierarchy of the contexts of executing the code fragments. Each pair of braces in a function/method (that is, excluding those used to describe classes and structures) forms a context, i.e., a life area of local variables. Since the unit nesting is not limited, we can use them to describe random hierarchies.

A similar approach has already been used in MQL, particularly to implement a self-made profiler measuring the code execution velocity (see [MQL's OOP notes: Self-made profiler on static and automatic objects](https://www.mql5.com/en/blogs/post/682673)). Its operation principles are quite simple. If, along with the operations solving an applied problem, we declare a local variable in the code unit:

```
  {
    ProfilerObject obj;

    ... // code lines of your actual algorithm
  }
```

then it will be created immediately upon entering the unit and deleted before exiting from it. This is true of the objects of any classes, including those being able to consider this behavior. Particularly, you can note the time of these instructions in the constructor and destructor and thereby calculate the duration of the applied algorithm. Naturally, to accumulate these measurements, another, more superior object is required, i.e., the profiler itself. However, an exchange device between them is not very important here (see more details in the blog). The matter is to apply the same principle to describing the layout. In other words, it will look as follows:

```
  container<Dialog> dialog(&this);
  {
    container<classA> main; // create classA internal object 1

    {
      container<classB> top_level(name, property, ...); // create classB internal object 2

      {
        container<classC> next_level_1(name, property, ...); // create classC internal object 3

        {
          control<classX> ctrl1(object4, name, property, ...); // create classX object 4
          control<classX> ctrl2(object5, name, property, ...); // create classX object 5
        } // register objects 4&5 in object 3 (via ctrl1, ctrl2 in next_level_1)
      } // register object 3 in object 2 (via next_level_1 in top_level)

      {
        container<classC> next_level2(name, property, ...); // create classC internal object 6

        {
          control<classY> ctrl3(object7, name, property, ...); // create classY object 7
          control<classY> ctrl4(object8, name, property, ...); // create classY object 8
        } // register objects 7&8 in object 6 (via ctrl3, ctrl4 in next_level_2)
      } // register object 6 in object 2 (via next_level_2 in top_level)
    } // register object 2 in object 1 (via top_level in main)
  } // register object 1 (main) in the dialog (this)
```

As this codes executes, the objects of some class (notionally named "container") will be created, with a template parameter defining the class of a specific GUI element to be generated within the dialog. All container objects are placed in a special arrayin the stack mode: Each next nesting level adds a container to the array, the current context unit being available on the top of the stack, while window is always at the very bottom, i.e., number one. At closing each unit, all child elements created in it will be automatically bound to the immediate parent (which is exactly on the top of the stack).

All this "magic" must be ensured by the inside of the "container" and "control" classes. In fact, this will be the same class, "layout," but, for the sake of better visibility, the difference between containers and controls is emphasized in the graph above. In the reality, the difference only relies on which classes are specified by the template parameters. Thus, classes Dialog, classA, classB, and classC in the example above must be window containers, i.e., support storing "controls" in them.

We should distinguish short-living ancillary objects of layout (they are named above as main, top\_level, next\_level\_1, ctrl1, ctrl2, next\_level2, ctrl3, and ctrl4) and the objects of interface classes (object 1 ... object 8) controlled by them, which will stay bound to each other and to the window. All this code will be executed as the dialog method (similar to method Create). Therefore, the dialog object is available as "this."

To some layout objects, we send the GUI objects as the class variables (object 4, 5, 7, 8), while to some of them, we don't (name and properties are specified). In any case, the GUI object must exist, but we don't always need it explicitly. If the "control" is used to subsequently interact with the algorithm, it is convenient to have a link to it. Containers are not usually related to the program logic and only fulfill the functions of placing the "controls," therefore, they are created non-explicitly inside the layout system.

We will develop the specific syntax of recording the properties and list them a bit later.

### Classes for interface layout: Abstract level

Let us write classes that allow implementing the formation of the interface elements hierarchy. Potentially, this approach can apply to any libraries of "controls." Therefore, we will divide the set of classes into 2 parts: Abstract ones (with general functionality) and applied ones related to the specific aspects of a specific library of standard control elements (CWnd descendant classes). We will verify the viability of the conception on standard dialogs, and those wishing can apply it to other libraries, guided by the abstract layer.

Class LayoutData is central.

```
  class LayoutData
  {
    protected:
      static RubbArray<LayoutData *> stack;
      static string rootId;
      int _x1, _y1, _x2, _y2;
      string _id;

    public:
      LayoutData()
      {
        _x1 = _y1 = _x2 = _y2 = 0;
        _id = NULL;
      }
  };
```

The minimum amount of information is stored in it, inherent to any layout element: Unique name \_id and coordinates. For your information, this field \_id is defined at an abstract level, and GUI can be "displayed" onto its own "control" property in each specific library. Particularly, in the standard library, this field is named m\_name, and it is available via public method CWnd::Name. Names cannot coincide for two objects. In CWnd, the m\_id field of type "long" is also defined — it is used for message dispatching. When we come to the applied implementation, it should not be confused with our \_id.

Besides, class LayoutData provides a static storage of its one instances as a stack and a window instance identifier (rootId). Statics of the two last members is not an issue, since each MQL program is executed within a single thread. Even if several windows will be in it, only one of them can be created at a time. As soon as a window is drawn, the stack will already become empty and ready to work with another window. Windwo identifier, rootId, is known for the standard library as field m\_instance\_id in class CAppDialog. For other libraries, there must be something similar (not necessary a string, but something unique, reducible to a string), since otherwise windows may conflict. We will address this issue again later.

Typed LayoutBase will be the descendant of class LayoutData. It is the prototype of that very layout class generating the interface elements by the MQL code with units of braces as instructions.

```
  template<typename P,typename C>
  class LayoutBase: public LayoutData
  {
    ...
```

Its two template parameters, P and C, are relevant to the classes of elements that work as containers and "controls."

Containers include by design the "controls" and/or other containers, while "controls" are percieved as a whole and cannot contain anything. Here it may be specifically noted that a "control" shall mean a logically monolithic unit of the interface, which can indeed consist of many ancillary objects. Particularly, classes CListView or CComboBox of the standard library are "controls," but they are implemented inside using several objects. These are the technicalities of implementation, while the similar types of control elements can be implemented in other libraries as a single outline, on which buttons and texts are drawn. In the context of abstract layout classes, we should not dig into it, breaking the principles of encapsulation, but the applied implementation designed for a specific library will, of course, have to consider this nuance (and distinguish real containers from compound "controls").

For the standard library, the best candidates for being the parameters of template, P and C, are CWndContainer and CWnd. Jumping ahead a bit, we should note that CWndObj may not be used as a class of "controls," since many "controls" are inherited from CWndContainer. These, for example, include CComboBox, CListView, CSpinEdit, CDatePicker, etc. However, as parameter C, we should select the nearest common class of all "controls," and CWnd is this for the standard library. As we can see, a class of containers, such as CWndContainer, can in practice meet simple elements; therefore, we will further have to ensure a more accurate check of whether a specific instance is a container or not. Similarly, the nearest common class of all containers must be selected as parameter P. In the standard library, window class is CDialog, the descendant of CWndContainer. However, along with it, we are going to use the classes of the CBox branch to group elements inside dialogs, and it descends from CWndClient that, in turn, descends from CWndContainer. Thus, the nearest common ancestor is CWndContainer.

Fields of class LayoutBase will store pointers to the interface element generated by the layout object.

```
    protected:
      P *container; // not null if container (can be used as flag)
      C *object;
      C *array[];
    public:
      LayoutBase(): container(NULL), object(NULL) {}
```

Here, container and object are pointing to the same thing; however, container is not NULL, provided that the element is really a container.

The array allows using one layout object to create a group of elements of the same type, such as buttons. In this case, pointers container and object will be equal to NULL. For all members, there are trivial "getter" methods, we won't present them all. For instance, it is easy to get a link to object, using method get().

The next three methods declare abstract operations over the bound element that must be able to execute the layout object.

```
    protected:
      virtual bool setContainer(C *control) = 0;
      virtual string create(C *object, const string id = NULL) = 0;
      virtual void add(C *object) = 0;
```

Method setContainer allows distinguishing a container from a normal "control" in the parameter passed. It s in this method, where we suggest to fill out the container field. If it is not NULL, then true is returned.

Method create initiates the element (a similar method, Create, is in all classes in the standard library; but, in my opinion, other libraries, such as EasyAndFastGUI, include similar methods; yet, in case of EasyAndFastGUI, they are named differently in different classes for some reason; therefore, those willing to connect the layout mechanism described to it, we will have to write adapter classes unifying the program interface of the "controls" of different times; but there is more: It is much more important to write classes similar to CBox and CGrid for EasyAndFastGUI). You can pass the desired identifier of the element to the method, but it is not necessarily the case that the executive algorithm will consider this desire in full or in part (particularly, instance\_id can be added). Therefore, you can get to know the real identifier from the string to be returned.

Method "add" adds an element to the parent container element (in the standard library, this operation is executed by method Add; while in EasyAndFastGUI, apparently, by MainPointer).

Now let us see how these 3 methods are involved at the abstract level. We have each element of the interface bound to the layout object and goes through 2 phases: Creation (at initiating the local variable in the code unit) and deletion (at exiting from the code unit and calling the destructor of the local variable). For the first phase, we will write method init that will be called from the constructors of descendant classes.

```
      template<typename T>
      void init(T *ref, const string id = NULL, const int x1 = 0, const int y1 = 0, const int x2 = 0, const int y2 = 0)
      {
        object = ref;
        setContainer(ref);

        _x1 = x1;
        _y1 = y1;
        _x2 = x2;
        _y2 = y2;
        if(stack.size() > 0)
        {
          if(_x1 == 0 && _y1 == 0 && _x2 == 0 && _y2 == 0)
          {
            _x1 = stack.top()._x1;
            _y1 = stack.top()._y1;
            _x2 = stack.top()._x2;
            _y2 = stack.top()._y2;
          }

          _id = rootId + (id == NULL ? typename(T) + StringFormat("%d", object) : id);
        }
        else
        {
          _id = (id == NULL ? typename(T) + StringFormat("%d", object) : id);
        }

        string newId = create(object, _id);

        if(stack.size() == 0)
        {
          rootId = newId;
        }
        if(container)
        {
          stack << &this;
        }
      }
```

The first parameter is the pointer to the element of the relevant class. Here, we will restrict ourselves so far to considering a case where the element is passed from the outside. But in the draft layout syntax above, we had some implicit elements (only names were specified for them). We will turn back to this operation scheme a bit later.

The method stores the pointer to the element into object, checks using setContainer, whether it is a container (suggesting that, if yes, then the container field will also be filled out), and takes the specified coordinates from inputs or, optionally, from the parent container, provided that it is already in the stack. Calling "create" initiates the interface element. If the stack is still empty, we will save the indentifier in rootId (in case of the standard library, it will be instance\_id), since the first element on the stack will always be the foremost container, i.e., the window responsible for all descending elements (in the standard library, it is class CDialog or a derived one). Finally, if the current element is a container, we will put it into the stack (stack << &this).

Method init is a template one. This allows automatically generating the names of "controls" by types; moreover, we will soon add other similar methods init. One of them will generate elements inside, rather than take them ready from outside, and, in this case, we need a specific type. Another version of init is designed to register in the layout several elements of the same type at a time (remember the array\[\] member), while arrays are passed by links, and the links do not support the conversion of types ("parameter conversion not allowed", "no one of the overloads can be applied to the function call," depending on the code structure), by virtue whereof we need again to point to a specific type via the template parameter. Thus, all methods init will have the same "template" contract, i.e., rules for the use.

The most interesting things happen in destructor LayoutBase.

```
      ~LayoutBase()
      {
        if(container)
        {
          stack.pop();
        }

        if(object)
        {
          LayoutBase *up = stack.size() > 0 ? stack.top() : NULL;
          if(up != NULL)
          {
            up.add(object);
          }
        }
      }
  };
```

If the current bound element is a container, we will delete it from the stack, since we are exiting from the relevant unit of braces (container is over). The matter is that, inside each unit, it is the top of the stack that contains the highest-nesting container, where the elements occurring inside the unit are added (in fact, have already been added), which elements can be both "controls" and smaller containers. Then the current elements is added using the method of "add" into the container that, in turn, has got to the top of the stack.

### Classes for the Interface Layout: Applied Level for the Elements of the Standard Library

Let us go to more specific things — implementing the classes for the layout of the interface elements of the standard library. Using classes CWndContainer and CWnd as the template parameters, let us define the intermediate class, StdLayoutBase.

```
  class StdLayoutBase: public LayoutBase<CWndContainer,CWnd>
  {
    public:
      virtual bool setContainer(CWnd *control) override
      {
        CDialog *dialog = dynamic_cast<CDialog *>(control);
        CBox *box = dynamic_cast<CBox *>(control);
        if(dialog != NULL)
        {
          container = dialog;
        }
        else if(box != NULL)
        {
          container = box;
        }
        return true;
      }
```

Method setContainer uses dynamic casts to check whether element CWnd descends from CDialog or CBox and, if yes, then it is a container.

```
      virtual string create(CWnd *child, const string id = NULL) override
      {
        child.Create(ChartID(), id != NULL ? id : _id, 0, _x1, _y1, _x2, _y2);
        return child.Name();
      }
```

Method "create" initiates the element and returns its name. Note that we are only working with the current chart (ChartID()) and in the main window (subwindows were not considered within this project, but you can adapt the code for your needs, if you want).

```
      virtual void add(CWnd *child) override
      {
        CDialog *dlg = dynamic_cast<CDialog *>(container);
        if(dlg != NULL)
        {
          dlg.Add(child);
        }
        else
        {
          CWndContainer *ptr = dynamic_cast<CWndContainer *>(container);
          if(ptr != NULL)
          {
            ptr.Add(child);
          }
          else
          {
            Print("Can't add ", child.Name(), " to ", container.Name());
          }
        }
      }
  };
```

Method "add" adds a child element to the parent one, preliminarily making as much "upcasting" as possible, since the Add method in the standard library is not virtual (technically, we could make a relevant change in the standard library, but we will talk about modifying it later).

Based on class StdLayoutBase, we will create work class \_layout that will be present in the code with the description of the layout in MQL. Name starts with an underscore to draw attention to the non-standard purpose of the objects of this class. Let us consider a simplified version of the class. We are going to add some more functionality to it later. In fact, all activities are started by constructors, inside which one method init or another is called from LayoutBase.

```
  template<typename T>
  class _layout: public StdLayoutBase
  {
    public:

      _layout(T &ref, const string id, const int dx, const int dy)
      {
        init(&ref, id, 0, 0, dx, dy);
      }

      _layout(T *ptr, const string id, const int dx, const int dy)
      {
        init(ptr, id, 0, 0, dx, dy);
      }

      _layout(T &ref, const string id, const int x1, const int y1, const int x2, const int y2)
      {
        init(&ref, id, x1, y1, x2, y2);
      }

      _layout(T *ptr, const string id, const int x1, const int y1, const int x2, const int y2)
      {
        init(ptr, id, x1, y1, x2, y2);
      }

      _layout(T &refs[], const string id, const int x1, const int y1, const int x2, const int y2)
      {
        init(refs, id, x1, y1, x2, y2);
      }
  };
```

You can glance over the overall picture, using the following class diagram. There is something on it what we have to get to know, but most classes are familiar to us.

![Diagram of the GUI Layout Classes](https://c.mql5.com/2/38/layout_classes.png)

**Diagram of the GUI Layout Classes**

Now we could practically check how the description of an object, such as \_layout<CButton> button(m\_button, 100, 20), initiates and registers object m\_button in a dialog, provided that it is described in an external unit like this: \_layout<CAppDialog> dialog(this, name, x1, y1, x2, y2). However, elements have many other properties, other than sizes. Some properties, such as alignment by sides, are of no lesser importance for the layout than coordinates. Indeed, if the element has horizontal alignment, in terms of the standard library "alignment," then it will be stretched over the entire width of the parent container area, minus the pre-defined fields on the left and on the right. Thus, alignment takes priority over coordinates. Moreover, in the CBox class containers, the orientation (direction) is important, in which the child elements are placed, i.e., horizontal (by default) or vertical. It would also be right to support other properties that affect the external representation, such as font size or color, and the operation mode, such as read only, "sticky" buttons, etc.

Where a GUI object is described in a window class and passed to the layout, we could use the "native" methods of setting the properties, such as edit.Text("text"). Layout system supports this old technique, but it is not a single or optimal one. In many cases, creating object would be convenient to assign to the layout system, then they won't be directly available from the window. Thus, it is necessary to somehow extend the capabilities of class \_layout regarding adjusting the elements.

Since there are many properties, it is recommended not to saddle the same class with working on them, but to share the responsibility between it and a special helper class. At the same time, \_layout is still the starting point for registering the elements, but it delegates all setup details to the new class. That is all the more important for making the layout technique as independent as possible on the specific library of controls.

### Classes for Configuring the Properties of Elements

At the abstract level, the set of properties is divided by their value types. We are going to support the basic embedded types of MQL, as well as some other ones that will be discussed later. Syntactically, it would be convenient to assign properties by a call chain of the known pattern, builder:

```
  _layout<CBox> column(...);
  column.style(LAYOUT_STYLE_VERTICAL).color(clrGray).margin(5);
```

However, this syntax implies a very long set of methods within one class, the latter one having to be the layout class, since the dereference operator (dot) cannot be overridden. In class \_layout, a method could be reserved to return an instance of the helper object for properties, like this:

```
  _layout<CBox> column(...);
  column.properties().style(LAYOUT_STYLE_VERTICAL).color(clrGray).margin(5);
```

But it would not be out of place to define many proxy classes — each for its own type of elements, to verify the correctness of the assigned properties at the compilation stage. This would complicate the project, but we would like to do everything as simply as possible for the first test implementation. Well, this approach is now left for further extension.

It should also be noted that the names of methods in the "builder" template are redundant in some sense, since values, such as LAYOUT\_STYLE\_VERTICAL or clrGray, are self-explanatory and other types do not often require any detailed description — thus, for the CEdit "control," the bool-type value usually means the "read only" flag, while it is the "sticking" sign for CButton. As a result, it is tempting just to assign values using an overloaded operator. However, strange to say, assignment operator does not suit us, since it does not allow threading the call chain.

```
  _layout<CBox> column(...);
  column = LAYOUT_STYLE_VERTICAL = clrGray = 5; // 'clrGray' - l-value required ...
```

Single-line assignment operators are executed from right to left, i.e., not starting from the object, in which the overloaded assignment is introduced. This would work as follows:

```
  ((column = LAYOUT_STYLE_VERTICAL) = clrGray) = 5;
```

But it looks a bit cumbersome.

Version:

```
  column = LAYOUT_STYLE_VERTICAL; // orientation
  column = clrGray;               // color
  column = 5;                     // margin
```

is too long, too. Therefore, we decided to overload the operator <= and use as follows:

```
  column <= LAYOUT_STYLE_VERTICAL <= clrGray <= 5.0;
```

For this purpose, there is a stub in class LayoutBase:

```
    template<typename V>
    LayoutBase<P,C> *operator<=(const V value) // template function cannot be virtual
    {
      Print("Please, override " , __FUNCSIG__, " in your concrete Layout class");
      return &this;
    }
```

Its double goal is to declare the intention to use the operator overload and to remind about overriding the method in the derived class. In theory, the mediator class object must be used there with the following interface (shown not in full).

```
  template<typename T>
  class ControlProperties
  {
    protected:
      T *object;
      string context;

    public:
      ControlProperties(): object(NULL), context(NULL) {}
      ControlProperties(T *ptr): object(ptr), context(NULL) {}
      void assign(T *ptr) { object = ptr; }
      T *get(void) { return object; }
      virtual ControlProperties<T> *operator[](const string property) { context = property; StringToLower(context); return &this; };
      virtual T *operator<=(const bool b) = 0;
      virtual T *operator<=(const ENUM_ALIGN_MODE align) = 0;
      virtual T *operator<=(const color c) = 0;
      virtual T *operator<=(const string s) = 0;
      virtual T *operator<=(const int i) = 0;
      virtual T *operator<=(const long l) = 0;
      virtual T *operator<=(const double d) = 0;
      virtual T *operator<=(const float f) = 0;
      virtual T *operator<=(const datetime d) = 0;
  };
```

As we can see, a link to the element (object) to be set up is stored in the mediator class. Binding is performed in the constructor or using the assign method. If we assume that we have written a specific mediator of class MyControlProperties:

```
  template<typename T>
  class MyControlProperties: public ControlProperties<T>
  {
    ...
  };
```

then, in class \_layout, we can use its object according to the following scheme (strings and method added are commented):

```
  template<typename T>
  class _layout: public StdLayoutBase
  {
    protected:
      C *object;
      C *array[];

      MyControlProperties helper;                                          // +

    public:
      ...
      _layout(T *ptr, const string id, const int dx, const int dy)
      {
        init(ptr, id, 0, 0, dx, dy); // this will save ptr in the 'object'
        helper.assign(ptr);                                                // +
      }
      ...

      // non-virtual function override                                     // +
      template<typename V>                                                 // +
      _layout<T> *operator<=(const V value)                                // +
      {
        if(object != NULL)
        {
          helper <= value;
        }
        else
        {
          for(int i = 0; i < ArraySize(array); i++)
          {
            helper.assign(array[i]);
            helper <= value;
          }
        }
        return &this;
      }
```

Due to the fact that operator <= in \_layout is a template one, it will automatically generate a call for a correct parameter type from the interface of ControlProperties (it is, of course, not about the abstract methods of interface, but about implementing them in the derived class MyControlProperties; we are going to write one soon for a specific window library).

In some cases, the same data type is used to define several different properties. For example, the same bool is used in CWnd when setting the flags of the visibility and active state of elements, along with the above-mentioned modes of "read only" (for CEdit) and "sticking" (for CButton). To be able to explicitly specify a property name, operator \[\] with the string type parameter is provided in interface ControlProperties. It sets the "context" field, based on which the derived class will be able to modify the required characteristic.

For each combination of the inputs type and the element class, one of the properties (the most frequently used one) will be considered the by-default property (their examples for CEdit and CButton are shown above). Other properties need a context to be specified.

For example, for CButton, it will look as follows:

```
  button1 <= true;
  button2["visible"] <= false;
```

In the first string, no context is specified; therefore, the "locking" property (a two-position button) is implied. In the second one, the button is initially created as invisible, which is normally a rare case.

Let us consider the basic details of implementing mediator StdControlProperties for the library of standard elements. The complete code can be found in the files attached hereto. At the beginning, you can see how operator <= is overridden for type "bool."

```
  template<typename T>
  class StdControlProperties: public ControlProperties<T>
  {
    public:
      StdControlProperties(): ControlProperties() {}
      StdControlProperties(T *ptr): ControlProperties(ptr) {}

      // we need dynamic_cast throughout below, because control classes
      // in the standard library does not provide a set of common virtual methods
      // to assign specific properties for all of them (for example, readonly
      // is available for edit field only)
      virtual T *operator<=(const bool b) override
      {
        if(StringFind(context, "enable") > -1)
        {
          if(b) object.Enable();
          else  object.Disable();
        }
        else
        if(StringFind(context, "visible") > -1)
        {
          object.Visible(b);
        }
        else
        {
          CEdit *edit = dynamic_cast<CEdit *>(object);
          if(edit != NULL) edit.ReadOnly(b);

          CButton *button = dynamic_cast<CButton *>(object);
          if(button != NULL) button.Locking(b);
        }

        return object;
      }
```

The following rule is used for strings: Any text gets into the "control" header, if only no "font" context is specified, which means the font name:

```
      virtual T *operator<=(const string s) override
      {
        CWndObj *ctrl = dynamic_cast<CWndObj *>(object);
        if(ctrl != NULL)
        {
          if(StringFind(context, "font") > -1)
          {
            ctrl.Font(s);
          }
          else // default
          {
            ctrl.Text(s);
          }
        }
        return object;
      }
```

In class StdControlProperties, we additionally introduced the <== overrides for the types that are only inherent to the standard library. Particularly, it can take enumeration ENUM\_WND\_ALIGN\_FLAGS that describes an alignment version. Please note that, in this enumeration, along with four sides (left, right, up, and down), there are the descriptions of not all combinations, but the most frequently used ones, such as aligning width (WND\_ALIGN\_WIDTH = WND\_ALIGN\_LEFT\|WND\_ALIGN\_RIGHT) or the entire client area (WND\_ALIGN\_CLIENT = WND\_ALIGN\_WIDTH\|WND\_ALIGN\_HEIGHT). However, if you need to align an element by width and by upper edge, this combination of flags will not be a part of the enumeration any more. Therefore, we will have to explicitly specify the type conversion to it ((ENUM\_WND\_ALIGN\_FLAGS)(WND\_ALIGN\_WIDTH\|WND\_ALIGN\_TOP)). Otherwise, the bit-by-bit OR operation will produce the int type, and the wrong overload of setting up the integer properties will be called. Alternative solution is to specify the "align" context.

No surprise that an override for the int type is the most laborous. Particularly, there can be set properties, such as width, height, margins, font size, etc. To facilitate this situation, it has been made possible to specify sizes directly in the layout object constructor, while margins can be alternatively specified by using double-type numbers or by a special packing named PackedRect. Of course, the operator overload has also been added for it; it is convenient to use it where unsymmetric margins are required:

```
  button <= PackedRect(5, 100, 5, 100); // left, top, right, bottom
```

because it is easier to sepcify equal-sided fields with only one double-type value:

```
  button <= 5.0;
```

However, the user may choose an alternative, i.e., the "margin" context; then you don't need double, and the equivalent record will be as follows:

```
  button["margin"] <= 5;
```

As to margins and indents, you should pay attention to just one caveat. There is the alignment term in the standard library, which includes margins to be automatically added around the "control." At the same time, in the CBox classes, their own padding mechanism is implemented, which represents a gap inside a container between its external boundary and the childe "controls" (contents). Thus, fields, in terms of "controls," and indents, in terms of containers, mean essentially the same. Since, unfortunately, two positioning algorithms do not consider each other, the simultaneous use of both margins and indents may cause issues (the most obvious of them is the shift of elements, which does not meet your expectations). General recommendation is to keep indents at zero and manipulate with margins. However, where necessary, you could try to include indents, too, especially if it is about a specific container, rather than general settings.

This paper is a proof-of-concept (POC) study and does not provide a ready-made solution. Its purpose is to try the technique proposed on the standard library classes and containers available as of writing it, with the minimal modifications of all those components. Ideally, containers (not necessary the CBox ones) must be written as an integral part of the GUI elements library and work considering all the possible combinations of modes.

Below is the table of the supported properties and elements. Class CWnd means the applicability of the properties to all elements, while class CWndObj is for simple "controls" (two of them, CEdit and CButton, are also given in the table). Class CWndClient generalizes "controls" (CCheckGroup, CRadioGroup, and CListView), and it is the parent class for containers CBox/CGrid.

**Table of Properties Supported by Data Types and by Classes of Elements**

|     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| type/control | CWnd | CWndObj | CWndClient | CEdit | CButton | CSpinEdit | CDatePicker | CBox/CGrid |
| bool | visible<br> enable | visible<br> enable | visible<br> enable | **(readonly)**<br> visible<br> enable | **(locking)**<br> visible<br> enable | visible<br> enable | visible<br> enable | visible<br> enable |
| color |  | **(text)**<br> background<br> border | **(background)**<br> border | **(text)**<br> background<br> border | **(text)**<br> background<br> border |  |  | **(background)**<br> border |
| string |  | **(text)**<br> font |  | **(text)**<br> font | **(text)**<br> font |  |  |  |
| int | width<br> height<br> margin<br> left<br> top<br> right<br> bottom<br> align | width<br> height<br> margin<br> left<br> top<br> right<br> bottom<br> align<br> fontsize | width<br> height<br> margin<br> left<br> top<br> right<br> bottom<br> align | width<br> height<br> margin<br> left<br> top<br> right<br> bottom<br> align<br> fontsize | width<br> height<br> margin<br> left<br> top<br> right<br> bottom<br> align<br> fontsize | **(value)**<br> width<br> height<br> margin<br> left<br> top<br> right<br> bottom<br> align<br> min<br> max | width<br> height<br> margin<br> left<br> top<br> right<br> bottom<br> align | width<br> height<br> margin<br> left<br> top<br> right<br> bottom<br> align |
| long | **(id)** | **(id)**<br> zorder | **(id)** | **(id)**<br> zorder | **(id)**<br> zorder | **(id)** | **(id)** | **(id)** |
| double | **(margin)** | **(margin)** | **(margin)** | **(margin)** | **(margin)** | **(margin)** | **(margin)** | **(margin)** |
| float |  |  |  |  |  |  |  | **(padding)**<br> left \*<br> top \*<br> right \*<br> bottom \* |
| datetime |  |  |  |  |  |  | **(value)** |  |
| PackedRect | **(margin\[4\])** | **(margin\[4\])** | **(margin\[4\])** | **(margin\[4\])** | **(margin\[4\])** | **(margin\[4\])** | **(margin\[4\])** | **(margin\[4\])** |
| ENUM\_ALIGN\_MODE |  |  |  | (text align) |  |  |  |  |
| ENUM\_WND\_ALIGN\_FLAGS | **(alignment)** | **(alignment)** | **(alignment)** | **(alignment)** | **(alignment)** | **(alignment)** | **(alignment)** | **(alignment)** |
| LAYOUT\_STYLE |  |  |  |  |  |  |  | **(style)** |
| VERTICAL\_ALIGN |  |  |  |  |  |  |  | **(vertical align)** |
| HORIZONTAL\_ALIGN |  |  |  |  |  |  |  | **(horizonal align)** |

Full source code of class StdControlProperties is attached hereto, which ensures translating the properties of the layout elements and calling the methods of the standard components library.

Let us try to test out the layout classes. We can finally start studying the instances, moving from simple to complex. According to the tradition that has developed since publishing the two original articles on laying out the GUI using containers, let us adapt to the new technique the sliding puzzle (SlidingPuzzle4) and a standard demo for working with "controls" (ControlsDialog4). Indexes correspond with the stages of updating these projects. In the [article](https://www.mql5.com/en/articles/6603), the same programs are presented with indexes 3, and you can compare the source codes if you wish. Examples can be found in folder MQL5/Experts/Examples/Layouts/.

### The example 1. SlidingPuzzle

The only considerable modification in the public interface of the main form of CSlidingPuzzleDialog is the new method, CreateLayout. It should be called from handler OnInit instead of conventional Create. Both methods have the same lists of parameters. This substitution was required, since the dialog itself is a layout object (the outermost level) and its method Create will be automatically called by the new framework (method StdLayoutBase::create does this, which we have considered above). All information for the framework on the form and its contents is specifically defined in method CreateLayout using the MQL-based markup language. Here is the method itself:

```
  bool CSlidingPuzzleDialog::CreateLayout(const long chart, const string name, const int subwin, const int x1, const int y1, const int x2, const int y2)
  {
    {
      _layout<CSlidingPuzzleDialog> dialog(this, name, x1, y1, x2, y2);
      {
        _layout<CGridTkEx> clientArea(m_main, NULL, 0, 0, ClientAreaWidth(), ClientAreaHeight());
        {
          SimpleSequenceGenerator<long> IDs;
          SimpleSequenceGenerator<string> Captions("0", 15);

          _layout<CButton> block(m_buttons, "block");
          block["background"] <= clrCyan <= IDs <= Captions;

          _layout<CButton> start(m_button_new, "New");
          start["background;font"] <= clrYellow <= "Arial Black";

          _layout<CEdit> label(m_label);
          label <= "click new" <= true <= ALIGN_CENTER;
        }
        m_main.Init(5, 4, 2, 2);
        m_main.SetGridConstraints(m_button_new, 4, 0, 1, 2);
        m_main.SetGridConstraints(m_label, 4, 2, 1, 2);
        m_main.Pack();
      }
    }
    m_empty_cell = &m_buttons[15];

    SelfAdjustment();
    return true;
  }
```

Here, two nested containers are formed successively, each being controlled by its own layout object:

- dialog for an instance of CSlidingPuzzleDialog (variable "this");
- clientArea for element CGridTkEx m\_main;

Then, in the client area, the set of buttons, CButton m\_buttons\[16\], is initialized, bound to the single layout object, block, as well as the game starting button (CButton m\_button\_new in the "start" object) and the informing label (CEdit m\_label, object "label"). All local variables, i.e., dialog, clientArea, block, start, and label, ensure automatically calling Create for the interface elements as the code is executed, assign them with the defined additional parameters (parameters will be discussed a bit later below), and register the interface elements bound to them in the higher-level container when deleting, i.e., when it goes beyond the visibility of the next block of braces. Thus, the m\_main client area will be included in the "this" window, while all "controls" will be in the client area. In this case, it is executed in the reversed order, though, since the blocks are closed starting with the most nested one. but it's not all that important. Practically the same happens when you use the conventional method of creating dialogs: The larger interface groups create the smaller ones, and the latter ones, in turn, create even smaller ones, down to the level of individual "controls," and start adding the initialized elements in the reversed (ascending) order: First, "controls" are added into the medium blocks, and then the medium ones are added into the larger ones.

For a dialog and for the client area, all parameters are passed via the constructor parameters (it's like the standard Create method). We don't need to pass sizes to "controls," since class GridTkEx allocates them automatically correctly, while other parameters are passed using operator <=.

A block of 16 buttons is initialized without any visible loop (it is hidden in the layout object now). Background color of all buttons is defined by string block\["background"\] <= clrCyan. Then, helper objects that we have not known yet are passed to the same layout object (SimpleSequenceGenerator).

When forming a user interface, it is often necessary to generate several elements of the same type and fill them out with some known data in batch mode. For this purpose, it is convenient to use the so-called generator.

Generator is a class with the method that can be called in a loop to get the next element from a certain list.

```
  template<typename T>
  class Generator
  {
    public:
      virtual T operator++() = 0;
  };
```

Normally, generator must know the number of the elements required, and it stores a cursor (index of the current element). Particularly, if you need to create the sequences of the values of a certain embedded type, such as integer or string, the following simple implementation of SimpleSequenceGenerator will suit you.

```
  template<typename T>
  class SimpleSequenceGenerator: public Generator<T>
  {
    protected:
      T current;
      int max;
      int count;

    public:
      SimpleSequenceGenerator(const T start = NULL, const int _max = 0): current(start), max(_max), count(0) {}

      virtual T operator++() override
      {
        ulong ul = (ulong)current;
        ul++;
        count++;
        if(count > max) return NULL;
        current = (T)ul;
        return current;
      }
  };
```

Generators are added for the convenience of batch operations (file Generators.mqh), while there is the override of operator <= for generators in the layout class. This allows us to fill out16 buttones with identifiers and captions in one line.

In the following strings of method CreateLayout, the m\_button\_new button is created.

```
        _layout<CButton> start(m_button_new, "New");
        start["background;font"] <= clrYellow <= "Arial Black";
```

String "New" is both an identifier and a caption. If we needed another caption to be assigned, we could do this as follows: start <= "Caption". Generally, it is not necessary to define an identifier, either (if we don't need it). The system will generate it itself.

In the second string, context is defined, which contains two tooltips at once: background and font. The former one is required to correctly interpret color clrYellow. Since the button is the descendant of CWndObj, "unnamed" color means the text color for it. The second tooltip ensures changing the used font by string "Arial Black" (without any context, the string would change the caption). If you wish, you may write in more details:

```
        start["background"] <= clrYellow;
        start["font"] <= "Arial Black";
```

Of course, the button still has its methods available, i.e., you can write as before:

```
        m_button_new.ColorBackground(clrYellow);
        m_button_new.Font("Arial Black");
```

However, to do so, you have to have a button object, which will not always be the case — later on, we will come to a scheme where the layout system will be responsible for everything, including constructing and storing your elements.

To set up a label, the following strings are used:

```
        _layout<CEdit> label(m_label);
        label <= "click new" <= true <= ALIGN_CENTER;
```

It is here where the object with an automatic identifier is created (if you open the window listing the objects on the chart, you will see the unique number of the instance). In the second string, we define the label text, the "read only" attribute, and the center alignment of the text.

Then follow the strings of adjusting the m\_main object of class CGridTKEx:

```
      m_main.Init(5, 4, 2, 2);
      m_main.SetGridConstraints(m_button_new, 4, 0, 1, 2);
      m_main.SetGridConstraints(m_label, 4, 2, 1, 2);
      m_main.Pack();
```

CGridTKEx is the slightly improved CGridTk (known from the preceding articles). In CGridTkEx, we have implemented the way of defining limitations for child "controls", using the new method, SetGridConstraints. In GridTk, this can only be done with simultaneously adding an element, inside method Grid. This is intrinsically bad, since it mixes two essentially different operations within one method: Establishing relations between objects and adjusting the properties. Moreover, it turns out that you should not use Add to add elements to the grid, but you only must use this method (since it is the only way to define limitations, without which GridTk cannot work). This violates the general approach of the library, where Add is always used for this purpose. And the operation of the automatic markup system is, in turn, tied with it. In class CGridTkEx, we separated 2 operations — now each of them has its own method.

It should be reminded that, for the main containers (including the entire window) of classes CBox/CGridTk, it is important to call method Pack — it is this method that performs the layout, calling Pack in the nested containers, if necessary.

If we compare the source codes of SlidingPuzzle3.mqh and SlidingPuzzle4.mqh, we will easily notice that the source code has become considerably more compact. Methods Create, CreateMain, CreateButton, CreateButtonNew, and CreateLabel have "left" the class. The only CreateLayout works instead of all them now.

Having started the program, we can see that elements are created and work as expected.

Well, we are still having the list declaring all the "controls" and containers in the class. As the programs get more complex and the number of components increases, it will not be very convenient to duplicate their descriptions in the window class and in the layout. Could everything be done using the layout? It is easy to guess that it could. This, however, will be discussed in the second part.

### Conclusions

This paper presents the theoretical bases and goals of the graphical interface markup languages. We have developed the concept of implementing a markup language in MQL and considered the core classes that embody this idea. But there are more complex and constructive examples to come.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7734](https://www.mql5.com/ru/articles/7734)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7734.zip "Download all attachments in the single ZIP archive")

[MQL5GUI1.zip](https://www.mql5.com/en/articles/download/7734/mql5gui1.zip "Download MQL5GUI1.zip")(86.86 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Backpropagation Neural Networks using MQL5 Matrices](https://www.mql5.com/en/articles/12187)
- [Parallel Particle Swarm Optimization](https://www.mql5.com/en/articles/8321)
- [Custom symbols: Practical basics](https://www.mql5.com/en/articles/8226)
- [Calculating mathematical expressions (Part 2). Pratt and shunting yard parsers](https://www.mql5.com/en/articles/8028)
- [Calculating mathematical expressions (Part 1). Recursive descent parsers](https://www.mql5.com/en/articles/8027)
- [MQL as a Markup Tool for the Graphical Interface of MQL Programs (Part 3). Form Designer](https://www.mql5.com/en/articles/7795)
- [MQL as a Markup Tool for the Graphical Interface of MQL Programs. Part 2](https://www.mql5.com/en/articles/7739)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/346032)**
(54)


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
7 Apr 2020 at 12:47

**Aliaksandr Hryshyn:**

Doesn't the right mouse click event come up in this situation?

That depends. Events come to the object based on coordinates. When the cursor is outside the button or window, they get nothing. Even drag'n'drop works on this principle - a constantly [moving](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "MetaTrader 5 Help: Moving Average indicator") object is created under the cursor. A slightly different edit is needed there.

![Andrey Khatimlianskii](https://c.mql5.com/avatar/2022/10/63528ae6-0b8c.jpg)

**[Andrey Khatimlianskii](https://www.mql5.com/en/users/komposter)**
\|
7 Apr 2020 at 22:36

**Stanislav Korotky:**

I've never had the urge to specifically press a button and drag without pushing it. It's not a very obvious use case after all.

It happens spontaneously from time to time.

![Dmitry Fedoseev](https://c.mql5.com/avatar/2014/9/54056F23-4E95.png)

**[Dmitry Fedoseev](https://www.mql5.com/en/users/integer)**
\|
7 Apr 2020 at 23:11

**Andrey Khatimlianskii:**

Happens spontaneously from time to time.

Moreover, it should be provided in case the user pressed the mouse, but then changed his mind to press the button, in this case he takes the mouse away from the button and releases it, the button is not pressed.

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
7 Apr 2020 at 23:14

**Dmitry Fedoseev:**

Moreover, it should be provided in case the user pressed the mouse, but then changed his mind to press the button, in this case he takes the mouse away from the button and releases it, the button is not pressed.

This can be done several times. Spinning around the object of the desired. The main thing is to change the logic of actions constantly) Not to be "readable". So that the exploit is "backwards" )

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
8 Apr 2020 at 00:19

This is the kind of fix for the button.


![Multicurrency monitoring of trading signals (Part 4): Enhancing functionality and improving the signal search system](https://c.mql5.com/2/38/Article_Logo__1.png)[Multicurrency monitoring of trading signals (Part 4): Enhancing functionality and improving the signal search system](https://www.mql5.com/en/articles/7678)

In this part, we expand the trading signal searching and editing system, as well as introduce the possibility to use custom indicators and add program localization. We have previously created a basic system for searching signals, but it was based on a small set of indicators and a simple set of search rules.

![Timeseries in DoEasy library (part 38): Timeseries collection - real-time updates and accessing data from the program](https://c.mql5.com/2/38/MQL5-avatar-doeasy-library__3.png)[Timeseries in DoEasy library (part 38): Timeseries collection - real-time updates and accessing data from the program](https://www.mql5.com/en/articles/7695)

The article considers real-time update of timeseries data and sending messages about the "New bar" event to the control program chart from all timeseries of all symbols for the ability to handle these events in custom programs. The "New tick" class is used to determine the need to update timeseries for the non-current chart symbol and periods.

![Timeseries in DoEasy library (part 39): Library-based indicators - preparing data and timeseries events](https://c.mql5.com/2/38/MQL5-avatar-doeasy-library__4.png)[Timeseries in DoEasy library (part 39): Library-based indicators - preparing data and timeseries events](https://www.mql5.com/en/articles/7724)

The article deals with applying DoEasy library for creating multi-symbol multi-period indicators. We are going to prepare the library classes to work within indicators and test creating timeseries to be used as data sources in indicators. We will also implement creating and sending timeseries events.

![Multicurrency monitoring of trading signals (Part 3): Introducing search algorithms](https://c.mql5.com/2/38/Article_Logo.png)[Multicurrency monitoring of trading signals (Part 3): Introducing search algorithms](https://www.mql5.com/en/articles/7600)

In the previous article, we developed the visual part of the application, as well as the basic interaction of GUI elements. This time we are going to add internal logic and the algorithm of trading signal data preparation, as well us the ability to set up signals, to search them and to visualize them in the monitor.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/7734&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083332625411611015)

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