---
title: Developing a Replay System (Part 27): Expert Advisor project — C_Mouse class (I)
url: https://www.mql5.com/en/articles/11337
categories: Trading, Trading Systems
relevance_score: 6
scraped_at: 2026-01-22T18:02:20.003584
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/11337&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049578898338983418)

MetaTrader 5 / Tester


### Introduction

In the previous article " [Developing a Replay System (Part 26): Expert Advisor project (I)](https://www.mql5.com/en/articles/11328)", we considered in detail the beginning of the construction of the first class. Now let's expand on these ideas and make them more useful. This brings us to the creation of the C\_Mouse class. It provides the ability to program at the highest level. However, talking about high-level or low-level programming languages is not about including obscene words or jargon in the code. It's the other way around. When we talk about high-level or low-level programming, we mean how easy or difficult the code is for other programmers to understand. In fact, the difference between high-level and low-level programming shows how simple or complex code can be for other developers. Thus, code is considered high-level if it is similar to natural language, and low-level if it is less similar to natural language and closer to how the processor interprets instructions.

Our goal is to keep the class code as high as possible, while avoiding as much as possible certain types of modeling that can make it difficult for less experienced people to understand. This is the goal, although I cannot guarantee that it will be fully achieved.

### C\_Mouse class: Starting interaction with the user

The mouse and keyboard are the most common means of interaction between the user and the platform. Therefore, it is very important that this interaction is simple and effective, so that the user does not have to relearn how to perform the actions. The code starts with the following lines:

```
#property copyright "Daniel Jose"
//+------------------------------------------------------------------+
#include "C_Terminal.mqh"
//+------------------------------------------------------------------+
#define def_MousePrefixName "MOUSE_"
#define def_NameObjectLineH def_MousePrefixName + "H"
#define def_NameObjectLineV def_MousePrefixName + "TV"
#define def_NameObjectLineT def_MousePrefixName + "TT"
#define def_NameObjectBitMp def_MousePrefixName + "TB"
#define def_NameObjectText  def_MousePrefixName + "TI"
//+------------------------------------------------------------------+
#define def_Fillet      "Resource\\Fillet.bmp"
#resource def_Fillet
//+------------------------------------------------------------------+
```

We have included a header file that contains the C\_Terminal class. As mentioned in the previous article, this C\_Mouse class file is located in the same directory as the C\_Terminal class file, which allows us to use this syntax without any problems. We define the name of the resource to be included in the executable, allowing you to port it without having to download the resource separately. This is very useful in many cases, especially when the resource is critical and its availability during use is essential. We usually place the resource in a specific directory for easier access. This way, it will always be compiled along with the header file. We have added a directory called 'Resource' to the folder where the C\_Mouse.mqh file is located. The Fillet.bmp file is inside this 'Resource' directory. If we change the directory structure while keeping the same modeling, the compiler will know exactly where to find the Fillet.bmp file. Once the code is compiled, we can load the executable without worrying about the resource not being found since it will be embedded in the executable itself.

In this step, we first define a name, and in fact a prefix for other names that we will define later. The use of definitions makes development and maintenance much easier as is common practice in professional code. The programmer defines various names and elements to be used in the code, which is usually done in a Defines.mqh file or another similar file. With this file, it is easy to change the definitions. However, since these definitions exist only in this file, there is no need to declare them anywhere else.

```
#undef def_MousePrefixName
#undef def_NameObjectLineV
#undef def_NameObjectBitMp
#undef def_NameObjectLineH
#undef def_NameObjectLineT
#undef def_NameObjectText
#undef def_Fillet
```

This code tells the compiler that all the symbols and names defined and visible by the C\_Mouse.mqh file should no longer be visible form this moment. It is generally not recommended to remove or change definitions in other files - this is not common practice. That's why we announce names where they actually appear and come into use. After that these definitions are removed. Changing or deleting definitions without criteria is also not good practice. If we need to use a definition in multiple files, it is best to create a separate file for it.

Let's now proceed to the first few lines of class code. This is where things get interesting. It all starts here:

```
class C_Mouse : public C_Terminal
{

   protected:
      enum eEventsMouse {ev_HideMouse, ev_ShowMouse};
      enum eBtnMouse {eKeyNull = 0x01, eClickLeft = 0x01, eClickRight = 0x02, eSHIFT_Press = 0x04, eCTRL_Press = 0x08, eClickMiddle = 0x10};
      struct st_Mouse
      {
         struct st00
         {
            int      X,
                     Y;
            double   Price;
            datetime dt;
         }Position;
         uint    ButtonStatus;
      };
```

In this fragment, we see that the C\_Terminal class is publicly inherited by the C\_Mouse class, which means that using the C\_Mouse class, we will have access to all public methods of the C\_Terminal class. Thus, the C\_Mouse class will have much more functionality than if it were limited to just the code in the C\_Mouse.mqh file. Inheritance does not only provide this benefit, there are other benefits for making classes more efficient, which we will discuss in future articles. Let's continue working with this code part. Inside the protected part of the code, we have two enumeration declarations that allow us to program at a slightly higher level. The first enumeration is quite simple and follows the same concept and rules that we covered in the previous article. On the other hand, the second list may seem a little confusing and complex, but we are going to explore the reason for its complexity and the reason for its existence in the first place.

This enumeration gives us an opportunity that would otherwise be much more difficult to maintain; that is, it will save us a lot of work. This particular enumeration creates name definitions, which is equivalent to the **#define** compiler directive. However, we decided to use enumeration instead of definitions. This will allow us to use a slightly different technique, but at the same time it will be much easier to understand in code. Using this enumeration, we will see how the code becomes much more readable. This becomes crucial in complex code. If you think that this enumeration has a declaration that at first glance is very confusing and complex, then you probably do not fully understand how enumerations work. From the compiler's point of view, an enumeration is just a sequence of definitions, where by default the first element starts at index zero. However, we can set the desired starting index of the enumeration, from which the compiler will begin to increment the values of subsequent indexes. This is very useful in many scenarios where the value of a certain index serves as the starting value of a sequence. Often programs use long lists of enumerations in which error values are set based on some specific criterion. If you define a name and assign it a specific value, the compiler will automatically increment the values of all subsequent names. This makes it much easier to create large lists of definitions, without the risk of duplicate values at some point.

In fact, it's surprising that many programmers don't use this technique, since it can go a long way toward avoiding mistakes when programming certain types of projects. Now that you understand this, you can experiment and find that using enumerations greatly simplifies the process of creating a large list of related elements, whether sequential or not. The approach we are exploring aims to improve programming by making code easier to read and understand.

The next part is a structure that is responsible for informing the rest of the code about what the mouse is doing. At this point, many might expect to declare a variable, but declaring variables within a class outside of a private clause is not considered good programming practice. Others may think that it would be more appropriate to place these declarations in the public part of the code. However, I prefer to start with a more limited level of access, allowing public access only as a last resort. We must ensure that functions and methods have public access, except those that are of direct interest to the class. Otherwise, we always recommend starting by granting minimal privileges to the elements.

Continuing with this idea, let's see the variables present in our C\_Mouse class:

```

   private :
      enum eStudy {eStudyNull, eStudyCreate, eStudyExecute};
      struct st01
      {
         st_Mouse Data;
         color    corLineH,
                  corTrendP,
                  corTrendN;
         eStudy   Study;
      }m_Info;
      struct st_Mem
      {
         bool    CrossHair;
      }m_Mem;
```

It's especially interesting to note that we already have an enumeration that won't be visible to any other part of the code outside of the class. This is because this enumeration is only useful within that class, and there is no point for other parts of the code to know about its existence. This concept is known as encapsulation, which is based on the principle that no other piece of code will know how the code performing the given task actually works. This type of approach is highly valued by library developers who allow other programmers to access procedures without revealing how the library code actually works.

Next, we find the structure. This uses another structure that can be accessed outside the class body, and which we will describe in detail when explaining access procedures and functions. At this point, it is important to understand that this variable refers to a private class structure. There is another structure, and in this case I prefer to use this approach because it makes it clear that the content is special and will only be accessed at very specific points in the code. However, nothing would prevent you from declaring the same data in the previous structure. You just need to be careful while writing code not to change this data, since those included in the first structure are for general use and can be changed at any time.

We have already introduced the parts related to variables, now we can move on to the analysis of functions and methods. Take a look at the following code:

```
inline void CreateObjectBase(const string szName, const ENUM_OBJECT obj, const color cor)
   {
      ObjectCreate(GetInfoTerminal().ID, szName, obj, 0, 0, 0);
      ObjectSetString(GetInfoTerminal().ID, szName, OBJPROP_TOOLTIP, "\n");
      ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_BACK, false);
      ObjectSetInteger(GetInfoTerminal().ID, szName, OBJPROP_COLOR, cor);
   }
```

This code facilitates software reuse because throughout the development of the C\_Mouse class, we will need to create various elements that must adhere to some standardization. Therefore, to facilitate the process, we will centralize this creation within a single method. The practice that can be often seen in declarations, especially when performance is a critical factor, is the use of a specific keyword. I made this choice because I want the compiler to include the code directly where it is declared, acting similarly to a macro. Well, this may lead to an increase in the size of the executable file, but in return we will get an improvement, albeit minor, in the overall runtime performance. Sometimes the performance gain is minimal due to various factors that may not justify increasing the size of the executable file.

Here we have a situation that may seem minor to many, but will become a recurring aspect throughout the code. This function refers to a structure declared in the C\_Terminal class. However, the compiler interprets it not as a function, but as a constant variable. But how is this possible? How can the compiler treat a declaration as a constant variable, which looks like a function? At first glance, this does not make much sense. However, look at the code for this call and its implementation in the C\_Terminal class in more detail:

```
inline const st_Terminal GetInfoTerminal(void) const
   {
      return m_Infos;
   }
```

This code returns a reference to a structure that belongs to the C\_Terminal class. The variable we return a reference to is private to the class and under no circumstances should its values be modified by any code other than that already present in the C\_Terminal class. To ensure that code does not make any changes to this private variable when it accesses it, we decided to include a special declaration. This way, the compiler ensures that any code that receives a reference to the constant cannot change its value. This measure is used to avoid accidental changes or programming errors. Thus, even if inside the C\_Terminal class there is an attempt to change a value in the function in an inappropriate manner, the compiler will recognize this process as an error, since, according to the compiler, no information there can be changed. This happens because this place is unsuitable or wrong for such changes.

This type of programming, although more labor intensive, improves code robustness and reliability. However, there is a flaw in this context that will be addressed later. This is because explaining this decision now would complicate the overall interpretation. Let's now look at the following method in the C\_Mouse class:

```
inline void CreateLineH(void)
   {
      CreateObjectBase(def_NameObjectLineH, OBJ_HLINE, m_Info.corLineH);
   }
```

It creates a horizontal line that represents the price line. It's important to note that by delegating all the complexity to another procedure, we only need to write one line. This type of approach is usually replaced by a macro. However, I prefer to try to do this without using macros. Another way is to paste the same content, given that it is one line, in the places where the call will occur. Personally, I don't recommend this practice, not because it is wrong, but because it requires to change all lines, while actually only one line changes. This can be a tedious task and prone to errors. So while it may seem more practical to place code directly at referenced points, it is safer to do so using a macro or code with the word 'inline' in its declaration.

The methods that we'll see below may not make much sense right now, but it's important to know them before we begin the explanation. The first method is shown below:

```
void CreateStudy(void)
{
   CreateObjectBase(def_NameObjectLineV, OBJ_VLINE, m_Info.corLineH);
   CreateObjectBase(def_NameObjectLineT, OBJ_TREND, m_Info.corLineH);
   CreateObjectBase(def_NameObjectBitMp, OBJ_BITMAP, clrNONE);
   CreateObjectBase(def_NameObjectText, OBJ_TEXT, clrNONE);
   ObjectSetString(GetInfoTerminal().ID, def_NameObjectText, OBJPROP_FONT, "Lucida Console");
   ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectText, OBJPROP_FONTSIZE, 10);
   ObjectSetString(GetInfoTerminal().ID, def_NameObjectBitMp, OBJPROP_BMPFILE, "::" + def_Fillet);
   ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineT, OBJPROP_WIDTH, 2);
   m_Info.Study = eStudyCreate;
}
```

It creates objects required to conduct research on the chart, allowing us to create our own style of analysis. This assists in highlighting the information that is considered most relevant and necessary for effective analysis. The research model presented here is quite simple, but a more varied methodology can be developed that is faster and does not visually clutter the chart. As an example of the simplest of methodologies, we will create a study to test the number of points between one price and another, visually indicating whether the value is negative or positive. Although it is not a complex system, it serves as a basis for the development of other, more sophisticated models.

Many studies use a wide variety of objects and their combinations, which sometimes require calculations or finding a position in a price range (high-low studies). Doing all this manually is not only slow, but also tedious, since you have to constantly add and remove objects from the chart. Otherwise, the chart may become crowded and confusing, making it difficult to identify the information you need. So use this method as a basis for creating something more refined and tailored to your needs. There is no need to rush now, as there will be an opportunity to improve it later.

The only thing you need to do is make sure that when you want to add text (which is very likely), it will be the last object in the creation sequence. This is visible from the above code, where the text displaying the information is the last object created. This prevents it from being hidden by other objects. Usually, several objects are created first, and one of them may hide a really important one. **Remember:** the most important element in which you are most interested should always be the last one in the creation queue.

Do not be confused by the fact that the object's colors are initially set to **clrNONE** as these colors will subsequently change as the analysis progresses. If the previous method is responsible for creating the analysis, then the next method actually performs that analysis on the chart.

```
void ExecuteStudy(const double memPrice)
{
   if (CheckClick(eClickLeft))
   {
      ObjectMove(GetInfoTerminal().ID, def_NameObjectLineT, 1, m_Info.Data.Position.dt, m_Info.Data.Position.Price);
      ObjectMove(GetInfoTerminal().ID, def_NameObjectBitMp, 0, m_Info.Data.Position.dt, m_Info.Data.Position.Price);
      ObjectMove(GetInfoTerminal().ID, def_NameObjectText, 0, m_Info.Data.Position.dt, m_Info.Data.Position.Price);
      ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectLineT, OBJPROP_COLOR, (memPrice > m_Info.Data.Position.Price ? m_Info.corTrendN : m_Info.corTrendP));
      ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectText, OBJPROP_COLOR, (memPrice > m_Info.Data.Position.Price ? m_Info.corTrendN : m_Info.corTrendP));
      ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectBitMp, OBJPROP_ANCHOR, (memPrice > m_Info.Data.Position.Price ? ANCHOR_RIGHT_UPPER : ANCHOR_RIGHT_LOWER));
      ObjectSetInteger(GetInfoTerminal().ID, def_NameObjectText, OBJPROP_ANCHOR, (memPrice > m_Info.Data.Position.Price ? ANCHOR_RIGHT_UPPER : ANCHOR_RIGHT_LOWER));
      ObjectSetString(GetInfoTerminal().ID, def_NameObjectText, OBJPROP_TEXT, StringFormat("%." + (string)GetInfoTerminal().nDigits + "f ", m_Info.Data.Position.Price - memPrice));
   } else {
      m_Info.Study equal eStudyNull;
      ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, true);
      ObjectsDeleteAll(GetInfoTerminal().ID, def_MousePrefixName + "T");
   }
   m_Info.Data.ButtonStatus equal eKeyNull;
}
```

Conducting analysis on the chart may seem pointless at first glance, especially if you look at the code in isolation. However, if you study the interaction code, its purpose becomes much clearer and more understandable. Let's now try to understand what's going on here. Unlike the code used to create research objects, this segment contains some interesting elements. Essentially, we distinguish two main parts in the code. In the first one we check the condition - and even without programming knowledge you can understand what exactly is being checked, since **with high-level programming the code is similar to natural language**. In the second one, we complete the analysis. Both parts are very simple and can be quickly understood. At this stage we move objects on the screen to indicate the analysis interval. In the next lines, we change the colors of the objects to indicate whether the movement is upward or downward, although this is usually obvious. However, situations may arise, such as when using some type of curve, where determining whether the values are positive or negative by simply observing becomes difficult. It is important to remember that analysis can be conducted in several ways based on different criteria. Perhaps the most important part is the line where we present values based on some calculation or analysis. Here we have the freedom to present many different information in different ways, depending on each case.

What we really need is a method to properly complete the research presentation, which is what the second code part does. Although at first glance this part does not seem particularly noteworthy, there is one aspect that deserves special attention: the use of the the [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/objectdeleteall) function. Why is this moment important and requires attention? The answer is in the C\_Terminal class. The constructor of the C\_Terminal class has the following line:

```
ChartSetInteger(m_Infos.ID, CHART_EVENT_OBJECT_DELETE, 0, true);
```

This line tells the platform that every time an object is removed from the chart, it should generate an event notifying which object was removed. Using the ObjectsDeleteAll function, we delete all elements or objects used in the analysis. This causes MetaTrader 5 to generate an event for each object deleted from the chart. The platform will do just that, and it's up to our code to decide whether those objects will be created again or not. The problem occurs when there objects fail to be deleted (because the code creates them again) or they are deleted without the relevant notification of the code. In this situation, for the **CHART\_EVENT\_OBJECT\_DELETE** property will be set to **false**. Although this doesn't happen initially, as the code expands there are times when this property can be accidentally changed and we may forget to re-enable it. As a result, the platform will not create an event to notify our code that objects have been removed from the chart, which can lead to inaccuracies and errors in the management of objects.

Let's now look at the C\_Mouse class constructor.

```
C_Mouse(color corH, color corP, color corN)
   :C_Terminal()
{
   m_Info.corLineH  = corH;
   m_Info.corTrendP = corP;
   m_Info.corTrendN = corN;
   m_Info.Study = eStudyNull;
   m_Mem.CrossHair  = (bool)ChartGetInteger(GetInfoTerminal().ID, CHART_CROSSHAIR_TOOL);
   ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_MOUSE_MOVE, true);
   ChartSetInteger(GetInfoTerminal().ID, CHART_CROSSHAIR_TOOL, false);
   CreateLineH();
}
```

At this point, we explicitly call the constructor of the C\_Terminal class. It is critical that this constructor is called before any other execution within the C\_Mouse class, ensuring that the necessary values of the C\_Terminal class are already properly initialized. After this initialization, we configure certain aspects and save the state of others. Pay attention to the two lines, in which we communicate to the platform our desire to receive mouse events and our intention not to use the standard analytics tools offered by the platform. When these definitions are applied, the platform will comply with our requirements by reporting mouse events upon request. On the other hand, when we try to use analytical tools using the mouse, we will have to provide the means to conduct such analyzes using our code.

The next code is the class destructor. It is very important to implement a destructor so that we can restore the original functionality of the platform when we are done using the C\_Mouse class.

```
~C_Mouse()
{
   ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_OBJECT_DELETE, 0, false);
   ChartSetInteger(GetInfoTerminal().ID, CHART_EVENT_MOUSE_MOVE, false);
   ChartSetInteger(GetInfoTerminal().ID, CHART_CROSSHAIR_TOOL, m_Mem.CrossHair);
   ObjectsDeleteAll(GetInfoTerminal().ID, def_MousePrefixName);
}
```

This question, although it may seem counterintuitive, is important. The reason for including this particular line is that when we try to delete objects created by the C\_Mouse class, in particular the price line, the platform generates an event notifying us that the object has been deleted from the chart. Our code will then attempt to return this object to the chart, even if it is in the process of being deleted. To prevent the platform from generating such an event, we must clearly indicate that we do not want this to happen. One might wonder, "But wouldn't the C\_Terminal class take care of this by telling us that we no longer want to receive events related to objects being removed from the chart?" Yes, the C\_Terminal class would do this, but since we still need some of the data present in the C\_Terminal class, we allow the compiler to implicitly make a call to the C\_Terminal class destructor, which occurs only after the last line of the C\_Mouse class destructor has been executed. Without adding a dedicated line of code, the platform will continue to generate the event, so even if the price line is initially removed, it can be put back before the code is fully completed. The remaining lines of the destructor are simpler, since all we do is return the chart to its original state.

We have come to the last functions discussed in this article.

```
inline bool CheckClick(const eBtnMouse value) { return (m_Info.Data.ButtonStatus & value) == value; }
```

In this line, using the enumeration we defined for the events received from the mouse, we check if the value provided by the platform matches what a particular point in the code expects. If the match was confirmed, the function will return true; otherwise it will return false. Although this may seem trivial at the moment, such a check will be very useful when we begin to interact more intensively with the system. There is a special trick to declaring this function that makes it easier to use, but since that's not important at this point, I won't go into detail.

The next function is just as simple as the previous one and follows the same principle as the GetInfoTerminal function of the C\_Terminal class.

```
inline const st_Mouse GetInfoMouse(void) const { return m_Info.Data; }
```

The goal is to return the information contained in the variable storing the mouse data structure, ensuring that this data cannot be changed without explicit permission of the C\_Mouse class. Since we have already talked about this method, I do not see the need to repeat its explanation. Essentially, both operate in a similar way.

Finally, we come to the culmination of the C\_Mouse class code. However, I think it is important to leave the last function present in the C\_Mouse class for discussion in our next article. Then we will explain the reason for this.

### Conclusion

We are in the process of creating something very promising, although it is still very far from being final. We will continue in the next article.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11337](https://www.mql5.com/pt/articles/11337)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Market Simulation (Part 09): Sockets (III)](https://www.mql5.com/en/articles/12673)
- [Market Simulation (Part 08): Sockets (II)](https://www.mql5.com/en/articles/12672)
- [Market Simulation (Part 07): Sockets (I)](https://www.mql5.com/en/articles/12621)
- [Market Simulation (Part 06): Transferring Information from MetaTrader 5 to Excel](https://www.mql5.com/en/articles/11794)
- [Market Simulation (Part 05): Creating the C\_Orders Class (II)](https://www.mql5.com/en/articles/12598)
- [Market Simulation (Part 04): Creating the C\_Orders Class (I)](https://www.mql5.com/en/articles/12589)
- [Market Simulation (Part 03): A Matter of Performance](https://www.mql5.com/en/articles/12580)

**[Go to discussion](https://www.mql5.com/en/forum/463000)**

![Developing a Replay System (Part 28): Expert Advisor project — C_Mouse class (II)](https://c.mql5.com/2/58/Replay-p28_II_avatar.png)[Developing a Replay System (Part 28): Expert Advisor project — C\_Mouse class (II)](https://www.mql5.com/en/articles/11349)

When people started creating the first systems capable of computing, everything required the participation of engineers, who had to know the project very well. We are talking about the dawn of computer technology, a time when there were not even terminals for programming. As it developed and more people got interested in being able to create something, new ideas and ways of programming emerged which replaced the previous-style changing of connector positions. This is when the first terminals appeared.

![Developing a Replay System (Part 26): Expert Advisor project — C_Terminal class](https://c.mql5.com/2/58/replay-p26-avatar.png)[Developing a Replay System (Part 26): Expert Advisor project — C\_Terminal class](https://www.mql5.com/en/articles/11328)

We can now start creating an Expert Advisor for use in the replay/simulation system. However, we need something improved, not a random solution. Despite this, we should not be intimidated by the initial complexity. It's important to start somewhere, otherwise we end up ruminating about the difficulty of a task without even trying to overcome it. That's what programming is all about: overcoming obstacles through learning, testing, and extensive research.

![Neural networks made easy (Part 60): Online Decision Transformer (ODT)](https://c.mql5.com/2/59/Online_Decision_Transformer_logo_up.png)[Neural networks made easy (Part 60): Online Decision Transformer (ODT)](https://www.mql5.com/en/articles/13596)

The last two articles were devoted to the Decision Transformer method, which models action sequences in the context of an autoregressive model of desired rewards. In this article, we will look at another optimization algorithm for this method.

![Data Science and Machine Learning (Part 20): Algorithmic Trading Insights, A Faceoff Between LDA and PCA in MQL5](https://c.mql5.com/2/70/Data_Science_and_Machine_Learning_Part_20__LOGO.png)[Data Science and Machine Learning (Part 20): Algorithmic Trading Insights, A Faceoff Between LDA and PCA in MQL5](https://www.mql5.com/en/articles/14128)

Uncover the secrets behind these powerful dimensionality reduction techniques as we dissect their applications within the MQL5 trading environment. Delve into the nuances of Linear Discriminant Analysis (LDA) and Principal Component Analysis (PCA), gaining a profound understanding of their impact on strategy development and market analysis.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=xzwxhsrnthzjextqihzfoatsodfylxlb&ssn=1769094138668801611&ssn_dr=0&ssn_sr=0&fv_date=1769094138&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11337&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20(Part%2027)%3A%20Expert%20Advisor%20project%20%E2%80%94%20C_Mouse%20class%20(I)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17690941387537361&fz_uniq=5049578898338983418&sv=2552)

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