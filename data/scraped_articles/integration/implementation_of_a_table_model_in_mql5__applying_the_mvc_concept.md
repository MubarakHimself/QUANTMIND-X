---
title: Implementation of a table model in MQL5: Applying the MVC concept
url: https://www.mql5.com/en/articles/17653
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T11:55:32.139807
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/17653&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062820286807583105)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/17653#node01)
- [A little bit about MVC concept (Model-View-Controller)](https://www.mql5.com/en/articles/17653#node02)
- [Writing classes to build a table model](https://www.mql5.com/en/articles/17653#node03)

1. [Linked lists as a basis for storing tabular data](https://www.mql5.com/en/articles/17653#node04)
2. [Table Cell Class](https://www.mql5.com/en/articles/17653#node05)
3. [Table Row Class](https://www.mql5.com/en/articles/17653#node06)
4. [Table Model Class](https://www.mql5.com/en/articles/17653#node07)

- [Testing the result](https://www.mql5.com/en/articles/17653#node08)
- [Conclusion](https://www.mql5.com/en/articles/17653#node09)

### Introduction

In programming, application architecture plays a key role in ensuring reliability, scalability, and ease of support. One of the approaches that helps achieve such goals is to leverage architecture pattern called **MVC (Model-View-Controller)**.

MVC concept allows you to divide an application into three interrelated components: **model** (data and logic management), **view** (data display), and **controller** (processing user actions). This separation simplifies code development, testing, and maintenance, making it more structured and flexible.

In this article, we consider how to apply MVC principles to implement a table model in the MQL5 language. Tables are an important tool for storing, processing, and displaying data, and properly organizing them can make working with information much easier. We will create classes for working with tables: table cells, rows, and table model. To store cells within rows and rows within the table model, we will use the [linked list](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist) classes from the MQL5 Standard Library that allow efficient storage and use of data.

### A little bit about MVC concept: what is it and why do we want it?

Imagine the application as a theater production. There is a scenario that describes what should happen (this is the **model**). There is the stage — what the viewer sees (this is **view**). And finally, there is the director who manages the entire process and connects other elements (this is the **controller**). This is the way the architectural pattern **MVC** — Model-View-Controller operates.

This concept helps to separate responsibilities within the application. The model is responsible for data and logic, the view is responsible for display and appearance, and the controller is responsible for processing user actions. Such separation makes the code clearer, more flexible, and more convenient for teamwork.

Let's say you are creating a table. The model knows which rows and cells it contains and knows how to change them. The view draws a table on the screen. And the controller reacts when the user clicks "Add row" and passes the task to the model, and then tells the view to update.

MVC is especially useful when the application becomes more complex: new features are added, the interface is changing, and several developers are working. With clear architecture, it is easier to make changes, test components individually, and reuse the code.

This approach also has some drawbacks. For very simple projects, MVC may be redundant — one will have to separate even what could fit into a couple of functions. However, for scalable, serious applications, this structure quickly pays off.

**In summary:**

MVC is a powerful architectural template that helps organize code, make it more understandable, testable, and scalable. It is especially useful for complex applications where separation of data logic, user interface, and management is important. For small projects, its use is redundant.

The Model-View-Controller paradigm fits our task very well. The table will be created from independent objects.:

- **Table cell.**

An object that stores a value of one of the types — real, integer, or string - is equipped with tools for managing the value, setting it, and retrieving it;
- **Table row.**

An object that stores a list of objects in table cells is equipped with tools for managing cells, their location, adding and deleting;
- **A table model.**

An object that stores a list of table string objects is equipped with tools for managing table strings and columns, their location, adding and deleting, and also has access to string and cell controls.


The figure below schematically shows the structure of a 4x4 table model:

![](https://c.mql5.com/2/129/Table4x4.png)

Fig.1 4x4 table model

Now let’s move from theory to practice.

### Writing classes to build a table model

We will use the [MQL5 Standard Library](https://www.mql5.com/en/docs/standardlibrary) to create all the objects.

Each object will be an inheritor of the library's [base class](https://www.mql5.com/en/docs/standardlibrary/cobject). This will allow you to store these objects in [object lists](https://www.mql5.com/en/docs/standardlibrary/datastructures).

We will write all classes in a single test script file so that everything is in one file, visible and quickly accessible. In the future, we will distribute the written classes into their separate include files.

### **1\. Linked lists as a basis for storing tabular data**

[CList linked list](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist) is very well suited for storing tabular data. Unlike the similar [CArrayObj](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) list, it implements access methods to neighboring list objects located to the left and right of the current one. This will make it easy to move cells in a row, or move rows in a table, add and delete them. At the same time, the list itself will take care of the correct indexing of moved, added or deleted objects in the list.

But there is one nuance here. If you refer to the methods for loading and saving a list to a file, you can see that when loading from a file, the list class must create a new object in the virtual CreateElement() method.

This method in this class simply returns NULL:

```
   //--- method of creating an element of the list
   virtual CObject  *CreateElement(void) { return(NULL); }
```

This means that in order to work with linked lists, and provided that we need file operations, we must inherit from the CList class and implement this method in our class.

If you look at the methods of saving Standard Library objects to a file, you can see the following algorithm for saving object properties:

1. The data start marker (-1) is written to the file,
2. The object type is written to the file,
3. All object properties are written to the file one by one.

The first and second points are inherent in all implemented save/load methods that Standard Library objects possess. Accordingly, following the same logic, we want to know the type of the object saved in the list, so that when reading from a file, we can create an object with this type in the virtual CreateElement() method of the list class inherited from CList.

Plus, all the objects that can be loaded into the list — their classes must be declared or created before the list class is implemented. In this case, the list will "know" which objects are "in question" and which need to be created.

In terminal directory \\MQL5\\Scripts\\, create a new folder TableModel\\, and in it — a new file of test script TableModelTest.mq5.

Connect the linked list file and declare future table model classes:

```
//+------------------------------------------------------------------+
//|                                               TableModelTest.mq5 |
//|                                  Copyright 2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| Включаемые библиотеки                                            |
//+------------------------------------------------------------------+
#include <Arrays\List.mqh>

//--- Форвард-декларация классов
class CTableCell;                   // Класс ячейки таблицы
class CTableRow;                    // Класс строки таблицы
class CTableModel;                  // Класс модели таблицы
```

The forward declaration of future classes is necessary here so that the linked-list class that inherits from CList, knows about these types of classes, as well as knows about the types of objects that it will have to create. To do this, we will write enumeration of object types, auxiliary macros, and enumeration of ways to sort lists:

```
//+------------------------------------------------------------------+
//| Включаемые библиотеки                                            |
//+------------------------------------------------------------------+
#include <Arrays\List.mqh>

//--- Форвард-декларация классов
class CTableCell;                   // Класс ячейки таблицы
class CTableRow;                    // Класс строки таблицы
class CTableModel;                  // Класс модели таблицы

//+------------------------------------------------------------------+
//| Макросы                                                          |
//+------------------------------------------------------------------+
#define  MARKER_START_DATA    -1    // Маркер начала данных в файле
#define  MAX_STRING_LENGTH    128   // Максимальная длина строки в ячейке

//+------------------------------------------------------------------+
//| Перечисления                                                     |
//+------------------------------------------------------------------+
enum ENUM_OBJECT_TYPE               // Перечисление типов объектов
  {
   OBJECT_TYPE_TABLE_CELL=10000,    // Ячейка таблицы
   OBJECT_TYPE_TABLE_ROW,           // Строка таблицы
   OBJECT_TYPE_TABLE_MODEL,         // Модель таблицы
  };

enum ENUM_CELL_COMPARE_MODE         // Режимы сравнения ячеек таблицы
  {
   CELL_COMPARE_MODE_COL,           // Сравнение по номеру колонки
   CELL_COMPARE_MODE_ROW,           // Сравнение по номеру строки
   CELL_COMPARE_MODE_ROW_COL,       // Сравнение по строке и колонке
  };

//+------------------------------------------------------------------+
//| Функции                                                          |
//+------------------------------------------------------------------+
//--- Возвращает тип объекта как строку
string TypeDescription(const ENUM_OBJECT_TYPE type)
  {
   string array[];
   int total=StringSplit(EnumToString(type),StringGetCharacter("_",0),array);
   string result="";
   for(int i=2;i<total;i++)
     {
      array[i]+=" ";
      array[i].Lower();
      array[i].SetChar(0,ushort(array[i].GetChar(0)-0x20));
      result+=array[i];
     }
   result.TrimLeft();
   result.TrimRight();
   return result;
  }
//+------------------------------------------------------------------+
//| Классы                                                           |
//+------------------------------------------------------------------+
```

The function that returns description of object type is based on the assumption that all names of object type constants begin with "OBJECT\_TYPE\_" substring. Then you can take the substring following this one, convert all the characters of the resulting row to lowercase, convert the first character to uppercase, and clear all spaces and control characters from the final string on the left and right.

Let us write our own linked list class. We will continue writing the code further in the same file:

```
//+------------------------------------------------------------------+
//| Классы                                                           |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Класс связанного списка объектов                                 |
//+------------------------------------------------------------------+
class CListObj : public CList
  {
protected:
   ENUM_OBJECT_TYPE  m_element_type;   // Тип создаваемого объекта в CreateElement()
public:
//--- Виртуальный метод (1) загрузки списка из файла, (2) создания элемента списка
   virtual bool      Load(const int file_handle);
   virtual CObject  *CreateElement(void);
  };
```

The CListObj class is our new linked list class, inherited from Standard Library [CList](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist) class.

The only variable in the class will be the one in which the type of the object being created will be written. Since [CreateElement()](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist/clistcreateelement) method is virtual and must have exactly the same signature as the parent class method, we cannot pass the type of the object being created to it. But we can write this type to a declared variable, and read the type of the object being created from it.

We must redefine two virtual methods of the parent class: method of uploading from a file and method of creating a new object. Let us consider them.

**Method uploading the list from a file:**

```
//+------------------------------------------------------------------+
//| Загрузка списка из файла                                         |
//+------------------------------------------------------------------+
bool CListObj::Load(const int file_handle)
  {
//--- Переменные
   CObject *node;
   bool     result=true;
//--- Проверяем хэндл
   if(file_handle==INVALID_HANDLE)
      return(false);
//--- Загрузка и проверка маркера начала списка - 0xFFFFFFFFFFFFFFFF
   if(::FileReadLong(file_handle)!=MARKER_START_DATA)
      return(false);
//--- Загрузка и проверка типа списка
   if(::FileReadInteger(file_handle,INT_VALUE)!=Type())
      return(false);
//--- Чтение размера списка (количество объектов)
   uint num=::FileReadInteger(file_handle,INT_VALUE);

//--- Последовательно заново создаём элементы списка с помощью вызова метода Load() объектов node
   this.Clear();
   for(uint i=0; i<num; i++)
     {
      //--- Читаем и проверяем маркер начала данных объекта - 0xFFFFFFFFFFFFFFFF
      if(::FileReadLong(file_handle)!=MARKER_START_DATA)
         return false;
      //--- Читаем тип объекта
      this.m_element_type=(ENUM_OBJECT_TYPE)::FileReadInteger(file_handle,INT_VALUE);
      node=this.CreateElement();
      if(node==NULL)
         return false;
      this.Add(node);
      //--- Сейчас файловый указатель смещён относительно начала маркера объекта на 12 байт (8 - маркер, 4 - тип)
      //--- Поставим указатель на начало данных объекта и загрузим свойства объекта из файла методом Load() элемента node.
      if(!::FileSeek(file_handle,-12,SEEK_CUR))
         return false;
      result &=node.Load(file_handle);
     }
//--- Результат
   return result;
  }
```

Here, the beginning of the list is first controlled, its type and size i.e. the number of elements in the list; and then, in a loop by the number of elements, beginning-of-data markers of each object and its type are read from the file. The resulting type is written to variable **m\_element\_type** and a method for creating a new element is called. In this method, a new element with the received type is created and written to **node** pointer variable, which, in turn, is added to the list. The entire logic of the method is explained in detail in the comments. Let us consider a method of creating a new list item.

**Method creating a list item:**

```
//+------------------------------------------------------------------+
//| Метод создания элемента списка                                   |
//+------------------------------------------------------------------+
CObject *CListObj::CreateElement(void)
  {
//--- В зависимости от типа объекта в m_element_type, создаём новый объект
   switch(this.m_element_type)
     {
      case OBJECT_TYPE_TABLE_CELL   :  return new CTableCell();
      case OBJECT_TYPE_TABLE_ROW    :  return new CTableRow();
      case OBJECT_TYPE_TABLE_MODEL  :  return new CTableModel();
      default                       :  return NULL;
     }
  }
```

This means that before calling the method, the type of the object being created is already written in **m\_element\_type** variable. Depending on item type, a new object of appropriate type is created, and a pointer to it is returned. In the future, when developing new controls, their types will be written into ENUM\_OBJECT\_TYPE enumeration. And new cases will be added here to create new types of objects. The linked list class based on the standard CList is ready. Now it can store all objects of known types, save lists to a file and upload them from the file and restore them correctly.

### **2\. Table Cell Class**

A table cell is the simplest element of a table that stores a certain value. Cells compose lists representing table rows. Each list represents one table row. In our table, cells will be able to store only one value of several types at a time — a real, integer, or string value.

In addition to a simple value, a cell can be assigned one object of a known type from ENUM\_OBJECT\_TYPE enumeration. In this case, the cell can store a value of any of the listed types, plus a pointer to an object, the type of which is written to a special variable. Thus, in the future, View component can be instructed to display such an object in a cell in order to interact with it using Controller component.

Since several different types of values can be stored in one cell, we will use union to write, store, and return them. [Union](https://www.mql5.com/en/docs/basis/types/classes#union) is a special type of data that stores several fields in the same memory area. A union is similar to a structure, but here, unlike in a structure, different terms of the union belong to the same memory area. While in the structure, each field is allocated its own memory area.

Let's continue writing the code in the already created file. Let us start writing a new class. In the protected section, we write a union and declare variables:

```
//+------------------------------------------------------------------+
//| Класс ячейки таблицы                                             |
//+------------------------------------------------------------------+
class CTableCell : public CObject
  {
protected:
//--- Объединение для хранения значений ячейки (double, long, string)
   union DataType
     {
      protected:
      double         double_value;
      long           long_value;
      ushort         ushort_value[MAX_STRING_LENGTH];

      public:
      //--- Установка значений
      void           SetValueD(const double value) { this.double_value=value;                   }
      void           SetValueL(const long value)   { this.long_value=value;                     }
      void           SetValueS(const string value) { ::StringToShortArray(value,ushort_value);  }

      //--- Возврат значений
      double         ValueD(void) const { return this.double_value; }
      long           ValueL(void) const { return this.long_value; }
      string         ValueS(void) const
                       {
                        string res=::ShortArrayToString(this.ushort_value);
                        res.TrimLeft();
                        res.TrimRight();
                        return res;
                       }
     };
//--- Переменные
   DataType          m_datatype_value;                      // Значение
   ENUM_DATATYPE     m_datatype;                            // Тип данных
   CObject          *m_object;                              // Объект в ячейке
   ENUM_OBJECT_TYPE  m_object_type;                         // Тип объекта в ячейке
   int               m_row;                                 // Номер строки
   int               m_col;                                 // Номер столбца
   int               m_digits;                              // Точность представления данных
   uint              m_time_flags;                          // Флаги отображения даты/времени
   bool              m_color_flag;                          // Флаг отображения наименования цвета
   bool              m_editable;                            // Флаг редактируемой ячейки

public:
```

In the public section, write access methods to protected variables, virtual methods, and class constructors for various types of data stored in a cell:

```
public:
//--- Возврат координат и свойств ячейки
   uint              Row(void)                           const { return this.m_row;                            }
   uint              Col(void)                           const { return this.m_col;                            }
   ENUM_DATATYPE     Datatype(void)                      const { return this.m_datatype;                       }
   int               Digits(void)                        const { return this.m_digits;                         }
   uint              DatetimeFlags(void)                 const { return this.m_time_flags;                     }
   bool              ColorNameFlag(void)                 const { return this.m_color_flag;                     }
   bool              IsEditable(void)                    const { return this.m_editable;                       }
//--- Возвращает (1) double, (2) long, (3) string значение
   double            ValueD(void)                        const { return this.m_datatype_value.ValueD();        }
   long              ValueL(void)                        const { return this.m_datatype_value.ValueL();        }
   string            ValueS(void)                        const { return this.m_datatype_value.ValueS();        }
//--- Возвращает значение в виде форматированной строки
   string            Value(void) const
                       {
                        switch(this.m_datatype)
                          {
                           case TYPE_DOUBLE  :  return(::DoubleToString(this.ValueD(),this.Digits()));
                           case TYPE_LONG    :  return(::IntegerToString(this.ValueL()));
                           case TYPE_DATETIME:  return(::TimeToString(this.ValueL(),this.m_time_flags));
                           case TYPE_COLOR   :  return(::ColorToString((color)this.ValueL(),this.m_color_flag));
                           default           :  return this.ValueS();
                          }
                       }
   string            DatatypeDescription(void) const
                       {
                        string type=::StringSubstr(::EnumToString(this.m_datatype),5);
                        type.Lower();
                        return type;
                       }
//--- Установка значений переменных
   void              SetRow(const uint row)                    { this.m_row=(int)row;                          }
   void              SetCol(const uint col)                    { this.m_col=(int)col;                          }
   void              SetDatatype(const ENUM_DATATYPE datatype) { this.m_datatype=datatype;                     }
   void              SetDigits(const int digits)               { this.m_digits=digits;                         }
   void              SetDatetimeFlags(const uint flags)        { this.m_time_flags=flags;                      }
   void              SetColorNameFlag(const bool flag)         { this.m_color_flag=flag;                       }
   void              SetEditable(const bool flag)              { this.m_editable=flag;                         }
   void              SetPositionInTable(const uint row,const uint col)
                       {
                        this.SetRow(row);
                        this.SetCol(col);
                       }
//--- Назначает объект в ячейку
   void              AssignObject(CObject *object)
                       {
                        if(object==NULL)
                          {
                           ::PrintFormat("%s: Error. Empty object passed",__FUNCTION__);
                           return;
                          }
                        this.m_object=object;
                        this.m_object_type=(ENUM_OBJECT_TYPE)object.Type();
                       }
//--- Снимает назначение объекта
   void              UnassignObject(void)
                       {
                        this.m_object=NULL;
                        this.m_object_type=-1;
                       }

//--- Устанавливает double-значение
   void              SetValue(const double value)
                       {
                        this.m_datatype=TYPE_DOUBLE;
                        if(this.m_editable)
                           this.m_datatype_value.SetValueD(value);
                       }
//--- Устанавливает long-значение
   void              SetValue(const long value)
                       {
                        this.m_datatype=TYPE_LONG;
                        if(this.m_editable)
                           this.m_datatype_value.SetValueL(value);
                       }
//--- Устанавливает datetime-значение
   void              SetValue(const datetime value)
                       {
                        this.m_datatype=TYPE_DATETIME;
                        if(this.m_editable)
                           this.m_datatype_value.SetValueL(value);
                       }
//--- Устанавливает color-значение
   void              SetValue(const color value)
                       {
                        this.m_datatype=TYPE_COLOR;
                        if(this.m_editable)
                           this.m_datatype_value.SetValueL(value);
                       }
//--- Устанавливает string-значение
   void              SetValue(const string value)
                       {
                        this.m_datatype=TYPE_STRING;
                        if(this.m_editable)
                           this.m_datatype_value.SetValueS(value);
                       }
//--- Очищает данные
   void              ClearData(void)
                       {
                        if(this.Datatype()==TYPE_STRING)
                           this.SetValue("");
                        else
                           this.SetValue(0.0);
                       }
//--- (1) Возвращает, (2) выводит в журнал описание объекта
   string            Description(void);
   void              Print(void);

//--- Виртуальные методы (1) сравнения, (2) сохранения в файл, (3) загрузки из файла, (4) тип объекта
   virtual int       Compare(const CObject *node,const int mode=0) const;
   virtual bool      Save(const int file_handle);
   virtual bool      Load(const int file_handle);
   virtual int       Type(void)                          const { return(OBJECT_TYPE_TABLE_CELL);}


//--- Конструкторы/деструктор
                     CTableCell(void) : m_row(0), m_col(0), m_datatype(-1), m_digits(0), m_time_flags(0), m_color_flag(false), m_editable(true), m_object(NULL), m_object_type(-1)
                       {
                        this.m_datatype_value.SetValueD(0);
                       }
                     //--- Принимает double-значение
                     CTableCell(const uint row,const uint col,const double value,const int digits) :
                        m_row((int)row), m_col((int)col), m_datatype(TYPE_DOUBLE), m_digits(digits), m_time_flags(0), m_color_flag(false), m_editable(true), m_object(NULL), m_object_type(-1)
                       {
                        this.m_datatype_value.SetValueD(value);
                       }
                     //--- Принимает long-значение
                     CTableCell(const uint row,const uint col,const long value) :
                        m_row((int)row), m_col((int)col), m_datatype(TYPE_LONG), m_digits(0), m_time_flags(0), m_color_flag(false), m_editable(true), m_object(NULL), m_object_type(-1)
                       {
                        this.m_datatype_value.SetValueL(value);
                       }
                     //--- Принимает datetime-значение
                     CTableCell(const uint row,const uint col,const datetime value,const uint time_flags) :
                        m_row((int)row), m_col((int)col), m_datatype(TYPE_DATETIME), m_digits(0), m_time_flags(time_flags), m_color_flag(false), m_editable(true), m_object(NULL), m_object_type(-1)
                       {
                        this.m_datatype_value.SetValueL(value);
                       }
                     //--- Принимает color-значение
                     CTableCell(const uint row,const uint col,const color value,const bool color_names_flag) :
                        m_row((int)row), m_col((int)col), m_datatype(TYPE_COLOR), m_digits(0), m_time_flags(0), m_color_flag(color_names_flag), m_editable(true), m_object(NULL), m_object_type(-1)
                       {
                        this.m_datatype_value.SetValueL(value);
                       }
                     //--- Принимает string-значение
                     CTableCell(const uint row,const uint col,const string value) :
                        m_row((int)row), m_col((int)col), m_datatype(TYPE_STRING), m_digits(0), m_time_flags(0), m_color_flag(false), m_editable(true), m_object(NULL), m_object_type(-1)
                       {
                        this.m_datatype_value.SetValueS(value);
                       }
                    ~CTableCell(void) {}
  };
```

In the methods for setting values, it is done so that the type of the value stored in the cell is set first, and then the flag of the feature to edit values in the cell is checked. And only when the flag is set, the new value is saved in the cell:

```
//--- Устанавливает double-значение
   void              SetValue(const double value)
                       {
                        this.m_datatype=TYPE_DOUBLE;
                        if(this.m_editable)
                           this.m_datatype_value.SetValueD(value);
                       }
```

Why is it done this way? When creating a new cell, it is created with the real type of the stored value. If you want to change the type of value, but at the same time the cell is not editable, you can call the method to set the value of the desired type with any value. The type of the stored value will be changed, but the value in the cell itself will not be affected.

The data cleaning method sets numeric values to zero, and enters a space in string values:

```
//--- Очищает данные
   void              ClearData(void)
                       {
                        if(this.Datatype()==TYPE_STRING)
                           this.SetValue("");
                        else
                           this.SetValue(0.0);
                       }
```

Later, we will do it differently — so that no data is displayed in cleared cells — to keep the cell empty, we will make an "empty" value for the cell, and then, when the cell is cleared, all the values recorded and displayed in it will be erased. After all, zero is also a full—fledged value, and now when the cell is cleared digital data is filled with zeros. This is incorrect.

In parametric constructors of the class, cell coordinates in the table are passed — number of row and column and the value of the required type (double, long, datetime, color, string). Some types of values require additional information:

- double— accuracy of the output value (number of decimal places),
- datetime— time output flags (date/hours-minutes/seconds),
- color— flag for displaying names of known standard colors.

In constructors with these types of values stored in cells additional parameters are passed to set the format of values displayed in cells:

```
 //--- Принимает double-значение
 CTableCell(const uint row,const uint col,const double value,const int digits) :
    m_row((int)row), m_col((int)col), m_datatype(TYPE_DOUBLE), m_digits(digits), m_time_flags(0), m_color_flag(false), m_editable(true), m_object(NULL), m_object_type(-1)
   {
    this.m_datatype_value.SetValueD(value);
   }
 //--- Принимает long-значение
 CTableCell(const uint row,const uint col,const long value) :
    m_row((int)row), m_col((int)col), m_datatype(TYPE_LONG), m_digits(0), m_time_flags(0), m_color_flag(false), m_editable(true), m_object(NULL), m_object_type(-1)
   {
    this.m_datatype_value.SetValueL(value);
   }
 //--- Принимает datetime-значение
 CTableCell(const uint row,const uint col,const datetime value,const uint time_flags) :
    m_row((int)row), m_col((int)col), m_datatype(TYPE_DATETIME), m_digits(0), m_time_flags(time_flags), m_color_flag(false), m_editable(true), m_object(NULL), m_object_type(-1)
   {
    this.m_datatype_value.SetValueL(value);
   }

 //--- Принимает color-значение
 CTableCell(const uint row,const uint col,const color value,const bool color_names_flag) :
    m_row((int)row), m_col((int)col), m_datatype(TYPE_COLOR), m_digits(0), m_time_flags(0), m_color_flag(color_names_flag), m_editable(true), m_object(NULL), m_object_type(-1)
   {
    this.m_datatype_value.SetValueL(value);
   }
 //--- Принимает string-значение
 CTableCell(const uint row,const uint col,const string value) :
    m_row((int)row), m_col((int)col), m_datatype(TYPE_STRING), m_digits(0), m_time_flags(0), m_color_flag(false), m_editable(true), m_object(NULL), m_object_type(-1)
   {
    this.m_datatype_value.SetValueS(value);
   }
```

**Method for comparing two objects:**

```
//+------------------------------------------------------------------+
//| Сравнение двух объектов                                          |
//+------------------------------------------------------------------+
int CTableCell::Compare(const CObject *node,const int mode=0) const
  {
   const CTableCell *obj=node;
   switch(mode)
     {
      case CELL_COMPARE_MODE_COL :  return(this.Col()>obj.Col() ? 1 : this.Col()<obj.Col() ? -1 : 0);
      case CELL_COMPARE_MODE_ROW :  return(this.Row()>obj.Row() ? 1 : this.Row()<obj.Row() ? -1 : 0);
      //---CELL_COMPARE_MODE_ROW_COL
      default                    :  return
                                      (
                                       this.Row()>obj.Row() ? 1 : this.Row()<obj.Row() ? -1 :
                                       this.Col()>obj.Col() ? 1 : this.Col()<obj.Col() ? -1 : 0
                                      );
     }
  }
```

The method allows you to compare parameters of two objects by one of three comparison criteria — by column number, by row number, and simultaneously by row and column numbers.

This method is necessary to be able to sort table rows by values of table columns. This will be handled by Controller in subsequent articles.

**Method for saving cell properties to file:**

```
//+------------------------------------------------------------------+
//| Сохранение в файл                                                |
//+------------------------------------------------------------------+
bool CTableCell::Save(const int file_handle)
  {
//--- Проверяем хэндл
   if(file_handle==INVALID_HANDLE)
      return(false);
//--- Сохраняем маркер начала данных - 0xFFFFFFFFFFFFFFFF
   if(::FileWriteLong(file_handle,MARKER_START_DATA)!=sizeof(long))
      return(false);
//--- Сохраняем тип объекта
   if(::FileWriteInteger(file_handle,this.Type(),INT_VALUE)!=INT_VALUE)
      return(false);

   //--- Сохраняем тип данных
   if(::FileWriteInteger(file_handle,this.m_datatype,INT_VALUE)!=INT_VALUE)
      return(false);
   //--- Сохраняем тип объекта в ячейке
   if(::FileWriteInteger(file_handle,this.m_object_type,INT_VALUE)!=INT_VALUE)
      return(false);
   //--- Сохраняем номер строки
   if(::FileWriteInteger(file_handle,this.m_row,INT_VALUE)!=INT_VALUE)
      return(false);
   //--- Сохраняем номер столбца
   if(::FileWriteInteger(file_handle,this.m_col,INT_VALUE)!=INT_VALUE)
      return(false);
   //--- Сохраняем точность представления данных
   if(::FileWriteInteger(file_handle,this.m_digits,INT_VALUE)!=INT_VALUE)
      return(false);
   //--- Сохраняем флаги отображения даты/времени
   if(::FileWriteInteger(file_handle,this.m_time_flags,INT_VALUE)!=INT_VALUE)
      return(false);
   //--- Сохраняем флаг отображения наименования цвета
   if(::FileWriteInteger(file_handle,this.m_color_flag,INT_VALUE)!=INT_VALUE)
      return(false);
   //--- Сохраняем флаг редактируемой ячейки
   if(::FileWriteInteger(file_handle,this.m_editable,INT_VALUE)!=INT_VALUE)
      return(false);
   //--- Сохраняем значение
   if(::FileWriteStruct(file_handle,this.m_datatype_value)!=sizeof(this.m_datatype_value))
      return(false);

//--- Всё успешно
   return true;
  }
```

After writing the starting data marker and the object type to the file, all the cell properties are saved in turn. The union must be saved as a structure using [FileWriteStruct()](https://www.mql5.com/en/docs/files/filewritestruct).

**Method for loading cell properties from file:**

```
//+------------------------------------------------------------------+
//| Загрузка из файла                                                |
//+------------------------------------------------------------------+
bool CTableCell::Load(const int file_handle)
  {
//--- Проверяем хэндл
   if(file_handle==INVALID_HANDLE)
      return(false);
//--- Загружаем и проверяем маркер начала данных - 0xFFFFFFFFFFFFFFFF
   if(::FileReadLong(file_handle)!=MARKER_START_DATA)
      return(false);
//--- Загружаем тип объекта
   if(::FileReadInteger(file_handle,INT_VALUE)!=this.Type())
      return(false);

   //--- Загружаем тип данных
   this.m_datatype=(ENUM_DATATYPE)::FileReadInteger(file_handle,INT_VALUE);
   //--- Загружаем тип объекта в ячейке
   this.m_object_type=(ENUM_OBJECT_TYPE)::FileReadInteger(file_handle,INT_VALUE);
   //--- Загружаем номер строки
   this.m_row=::FileReadInteger(file_handle,INT_VALUE);
   //--- Загружаем номер столбца
   this.m_col=::FileReadInteger(file_handle,INT_VALUE);
   //--- Загружаем точность представления данных
   this.m_digits=::FileReadInteger(file_handle,INT_VALUE);
   //--- Загружаем флаги отображения даты/времени
   this.m_time_flags=::FileReadInteger(file_handle,INT_VALUE);
   //--- Загружаем флаг отображения наименования цвета
   this.m_color_flag=::FileReadInteger(file_handle,INT_VALUE);
   //--- Загружаем флаг редактируемой ячейки
   this.m_editable=::FileReadInteger(file_handle,INT_VALUE);
   //--- Загружаем значение
   if(::FileReadStruct(file_handle,this.m_datatype_value)!=sizeof(this.m_datatype_value))
      return(false);

//--- Всё успешно
   return true;
  }
```

After reading and checking beginning-of-data markers and type object, all the properties of the object are loaded in turn in the same order as they were saved. Unions are read using [FileReadStruct()](https://www.mql5.com/en/docs/files/filereadstruct).

**Method that returns description of the object:**

```
//+------------------------------------------------------------------+
//| Возвращает описание объекта                                      |
//+------------------------------------------------------------------+
string CTableCell::Description(void)
  {
   return(::StringFormat("%s: Row %u, Col %u, %s <%s>Value: %s",
                         TypeDescription((ENUM_OBJECT_TYPE)this.Type()),this.Row(),this.Col(),
                         (this.m_editable ? "Editable" : "Uneditable"),this.DatatypeDescription(),this.Value()));
  }
```

Here, a row is created from some of cell parameters and returned, for example, for double, in this format:

```
  Table Cell: Row 2, Col 2, Uneditable <double>Value: 0.00
```

**Method that outputs object description to the log:**

```
//+------------------------------------------------------------------+
//| Выводит в журнал описание объекта                                |
//+------------------------------------------------------------------+
void CTableCell::Print(void)
  {
   ::Print(this.Description());
  }
```

Here, the object description is simply printed to the log.

```
//+------------------------------------------------------------------+
//| Класс строки таблицы                                             |
//+------------------------------------------------------------------+
class CTableRow : public CObject
  {
protected:
   CTableCell        m_cell_tmp;                            // Объект ячейки для поиска в списке
   CListObj          m_list_cells;                          // Список ячеек
   uint              m_index;                               // Индекс строки

//--- Добавляет указанную ячейку в конец списка
   bool              AddNewCell(CTableCell *cell);

public:
//--- (1) Устанавливает, (2) возвращает индекс строки
   void              SetIndex(const uint index)                { this.m_index=index;  }
   uint              Index(void)                         const { return this.m_index; }
//--- Устанавливает позиции строки и колонки всем ячейкам
   void              CellsPositionUpdate(void);

//--- Создаёт новую ячейку и добавляет в конец списка
   CTableCell       *CreateNewCell(const double value);
   CTableCell       *CreateNewCell(const long value);
   CTableCell       *CreateNewCell(const datetime value);
   CTableCell       *CreateNewCell(const color value);
   CTableCell       *CreateNewCell(const string value);

//--- Возвращает (1) ячейку по индексу, (2) количество ячеек
   CTableCell       *GetCell(const uint index)                 { return this.m_list_cells.GetNodeAtIndex(index);  }
   uint              CellsTotal(void)                    const { return this.m_list_cells.Total();                }

//--- Устанавливает значение в указанную ячейку
   void              CellSetValue(const uint index,const double value);
   void              CellSetValue(const uint index,const long value);
   void              CellSetValue(const uint index,const datetime value);
   void              CellSetValue(const uint index,const color value);
   void              CellSetValue(const uint index,const string value);
//--- (1) назначает в ячейку, (2) снимает с ячейки назначенный объект
   void              CellAssignObject(const uint index,CObject *object);
   void              CellUnassignObject(const uint index);

//--- (1) Удаляет (2) перемещает ячейку
   bool              CellDelete(const uint index);
   bool              CellMoveTo(const uint cell_index, const uint index_to);

//--- Обнуляет данные ячеек строки
   void              ClearData(void);

//--- (1) Возвращает, (2) выводит в журнал описание объекта
   string            Description(void);
   void              Print(const bool detail, const bool as_table=false, const int cell_width=10);

//--- Виртуальные методы (1) сравнения, (2) сохранения в файл, (3) загрузки из файла, (4) тип объекта
   virtual int       Compare(const CObject *node,const int mode=0) const;
   virtual bool      Save(const int file_handle);
   virtual bool      Load(const int file_handle);
   virtual int       Type(void)                          const { return(OBJECT_TYPE_TABLE_ROW); }

//--- Конструкторы/деструктор
                     CTableRow(void) : m_index(0) {}
                     CTableRow(const uint index) : m_index(index) {}
                    ~CTableRow(void){}
  };
```

### **3\. Table Row Class**

A table row is essentially a linked list of cells. The row class must support adding, deleting, and reordering cells in the list to a new location. The class must have a minimum-sufficient set of methods for managing cells in the list.

Let's continue writing the code in the same file. Only one variable is available in class parameters — the row index in the table. All the rest are methods for working with row properties and with a list of its cells.

Let us consider class methods.

**Method for comparing two table rows:**

```
//+------------------------------------------------------------------+
//| Сравнение двух объектов                                          |
//+------------------------------------------------------------------+
int CTableRow::Compare(const CObject *node,const int mode=0) const
  {
   const CTableRow *obj=node;
   return(this.Index()>obj.Index() ? 1 : this.Index()<obj.Index() ? -1 : 0);
  }
```

Since rows can only be compared by their single parameter - row index - this comparison is implemented here. This method will be required to sort out table rows.

**Overloaded methods for creating cells with different types of stored data:**

```
//+------------------------------------------------------------------+
//| Создаёт новую double-ячейку и добавляет в конец списка           |
//+------------------------------------------------------------------+
CTableCell *CTableRow::CreateNewCell(const double value)
  {
//--- Создаём новый объект ячейки, хранящей значение с типом double
   CTableCell *cell=new CTableCell(this.m_index,this.CellsTotal(),value,2);
   if(cell==NULL)
     {
      ::PrintFormat("%s: Error. Failed to create new cell in row %u at position %u",__FUNCTION__, this.m_index, this.CellsTotal());
      return NULL;
     }
//--- Добавляем созданную ячейку в конец списка
   if(!this.AddNewCell(cell))
     {
      delete cell;
      return NULL;
     }
//--- Возвращаем указатель на объект
   return cell;
  }
//+------------------------------------------------------------------+
//| Создаёт новую long-ячейку и добавляет в конец списка             |
//+------------------------------------------------------------------+
CTableCell *CTableRow::CreateNewCell(const long value)
  {
//--- Создаём новый объект ячейки, хранящей значение с типом long
   CTableCell *cell=new CTableCell(this.m_index,this.CellsTotal(),value);
   if(cell==NULL)
     {
      ::PrintFormat("%s: Error. Failed to create new cell in row %u at position %u",__FUNCTION__, this.m_index, this.CellsTotal());
      return NULL;
     }
//--- Добавляем созданную ячейку в конец списка
   if(!this.AddNewCell(cell))
     {
      delete cell;
      return NULL;
     }
//--- Возвращаем указатель на объект
   return cell;
  }
//+------------------------------------------------------------------+
//| Создаёт новую datetime-ячейку и добавляет в конец списка         |
//+------------------------------------------------------------------+
CTableCell *CTableRow::CreateNewCell(const datetime value)
  {
//--- Создаём новый объект ячейки, хранящей значение с типом datetime
   CTableCell *cell=new CTableCell(this.m_index,this.CellsTotal(),value,TIME_DATE|TIME_MINUTES|TIME_SECONDS);
   if(cell==NULL)
     {
      ::PrintFormat("%s: Error. Failed to create new cell in row %u at position %u",__FUNCTION__, this.m_index, this.CellsTotal());
      return NULL;
     }
//--- Добавляем созданную ячейку в конец списка
   if(!this.AddNewCell(cell))
     {
      delete cell;
      return NULL;
     }
//--- Возвращаем указатель на объект
   return cell;
  }
//+------------------------------------------------------------------+
//| Создаёт новую color-ячейку и добавляет в конец списка            |
//+------------------------------------------------------------------+
CTableCell *CTableRow::CreateNewCell(const color value)
  {
//--- Создаём новый объект ячейки, хранящей значение с типом color
   CTableCell *cell=new CTableCell(this.m_index,this.CellsTotal(),value,true);
   if(cell==NULL)
     {
      ::PrintFormat("%s: Error. Failed to create new cell in row %u at position %u",__FUNCTION__, this.m_index, this.CellsTotal());
      return NULL;
     }
//--- Добавляем созданную ячейку в конец списка
   if(!this.AddNewCell(cell))
     {
      delete cell;
      return NULL;
     }
//--- Возвращаем указатель на объект
   return cell;
  }
//+------------------------------------------------------------------+
//| Создаёт новую string-ячейку и добавляет в конец списка           |
//+------------------------------------------------------------------+
CTableCell *CTableRow::CreateNewCell(const string value)
  {
//--- Создаём новый объект ячейки, хранящей значение с типом string
   CTableCell *cell=new CTableCell(this.m_index,this.CellsTotal(),value);
   if(cell==NULL)
     {
      ::PrintFormat("%s: Error. Failed to create new cell in row %u at position %u",__FUNCTION__, this.m_index, this.CellsTotal());
      return NULL;
     }
//--- Добавляем созданную ячейку в конец списка
   if(!this.AddNewCell(cell))
     {
      delete cell;
      return NULL;
     }
//--- Возвращаем указатель на объект
   return cell;
  }
```

Five methods for creating a new cell (double, long, datetime, color, string) and adding it to the list end. Additional parameters of data output format into the cell are set with default values. They can be changed after the cell is created. First, a new cell object is created, and then added to the list end. If the object was not created, it is deleted to avoid memory leaks.

**Method that adds a cell to the list end:**

```
//+------------------------------------------------------------------+
//| Добавляет ячейку в конец списка                                  |
//+------------------------------------------------------------------+
bool CTableRow::AddNewCell(CTableCell *cell)
  {
//--- Если передан пустой объект - сообщаем и возвращаем false
   if(cell==NULL)
     {
      ::PrintFormat("%s: Error. Empty CTableCell object passed",__FUNCTION__);
      return false;
     }
//--- Устанавливаем индекс ячейки в списке и добавляем созданную ячейку в конец списка
   cell.SetPositionInTable(this.m_index,this.CellsTotal());
   if(this.m_list_cells.Add(cell)==WRONG_VALUE)
     {
      ::PrintFormat("%s: Error. Failed to add cell (%u,%u) to list",__FUNCTION__,this.m_index,this.CellsTotal());
      return false;
     }
//--- Успешно
   return true;
  }
```

Any newly created cell is always added to the list end. Next, it can be moved to the appropriate position using methods of the table model class, which will be created later.

**Overloaded methods for setting values to the specified cell:**

```
//+------------------------------------------------------------------+
//| Устанавливает double-значение в указанную ячейку                 |
//+------------------------------------------------------------------+
void CTableRow::CellSetValue(const uint index,const double value)
  {
//--- Получаем из списка нужную ячейку и записываем в неё новое значение
   CTableCell *cell=this.GetCell(index);
   if(cell!=NULL)
      cell.SetValue(value);
  }
//+------------------------------------------------------------------+
//| Устанавливает long-значение в указанную ячейку                   |
//+------------------------------------------------------------------+
void CTableRow::CellSetValue(const uint index,const long value)
  {
//--- Получаем из списка нужную ячейку и записываем в неё новое значение
   CTableCell *cell=this.GetCell(index);
   if(cell!=NULL)
      cell.SetValue(value);
  }
//+------------------------------------------------------------------+
//| Устанавливает datetime-значение в указанную ячейку               |
//+------------------------------------------------------------------+
void CTableRow::CellSetValue(const uint index,const datetime value)
  {
//--- Получаем из списка нужную ячейку и записываем в неё новое значение
   CTableCell *cell=this.GetCell(index);
   if(cell!=NULL)
      cell.SetValue(value);
  }
//+------------------------------------------------------------------+
//| Устанавливает color-значение в указанную ячейку                  |
//+------------------------------------------------------------------+
void CTableRow::CellSetValue(const uint index,const color value)
  {
//--- Получаем из списка нужную ячейку и записываем в неё новое значение
   CTableCell *cell=this.GetCell(index);
   if(cell!=NULL)
      cell.SetValue(value);
  }
//+------------------------------------------------------------------+
//| Устанавливает string-значение в указанную ячейку                 |
//+------------------------------------------------------------------+
void CTableRow::CellSetValue(const uint index,const string value)
  {
//--- Получаем из списка нужную ячейку и записываем в неё новое значение
   CTableCell *cell=this.GetCell(index);
   if(cell!=NULL)
      cell.SetValue(value);
  }
```

Using the index, we get the required cell from the list and set the value for it. If the cell is non-editable, SetValue() method of cell object will for the cell set only the type of value being set. The value itself will not be set.

**A method that assigns an object to a cell:**

```
//+------------------------------------------------------------------+
//| Назначает в ячейку объект                                        |
//+------------------------------------------------------------------+
void CTableRow::CellAssignObject(const uint index,CObject *object)
  {
//--- Получаем из списка нужную ячейку и записываем в неё указатель на объект
   CTableCell *cell=this.GetCell(index);
   if(cell!=NULL)
      cell.AssignObject(object);
  }
```

We get a cell object by its index and use its AssignObject() method to assign a pointer to the object to the cell.

**Method that cancels an assigned object for a cell:**

```
//+------------------------------------------------------------------+
//| Отменяет для ячейки назначенный объект                           |
//+------------------------------------------------------------------+
void CTableRow::CellUnassignObject(const uint index)
  {
//--- Получаем из списка нужную ячейку и отменяем в ней указатель на объект и его тип
   CTableCell *cell=this.GetCell(index);
   if(cell!=NULL)
      cell.UnassignObject();
  }
```

We get the cell object by its index and use its UnassignObject() method to remove the pointer to the object assigned to the cell.

**Method that deletes a cell:**

```
//+------------------------------------------------------------------+
//| Удаляет ячейку                                                   |
//+------------------------------------------------------------------+
bool CTableRow::CellDelete(const uint index)
  {
//--- Удаляем ячейку в списке по индексу
   if(!this.m_list_cells.Delete(index))
      return false;
//--- Обновляем индексы для оставшихся ячеек в списке
   this.CellsPositionUpdate();
   return true;
  }
```

Using [Delete()](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist/clistdelete) method of the CList class we delete the cell from the list. After a cell has been deleted from the list, indexes of remaining cells are changed. Using CellsPositionUpdate() method, we update indexes of all remaining cells in the list.

**Method that moves a cell to the specified position:**

```
//+------------------------------------------------------------------+
//| Перемещает ячейку на указанную позицию                           |
//+------------------------------------------------------------------+
bool CTableRow::CellMoveTo(const uint cell_index,const uint index_to)
  {
//--- Выбираем нужную ячейку по индексу в списке, делая её текущей
   CTableCell *cell=this.GetCell(cell_index);
//--- Перемещаем текущую ячейку на указанную позицию в списке
   if(cell==NULL || !this.m_list_cells.MoveToIndex(index_to))
      return false;
//--- Обновляем индексы всех ячеек в списке
   this.CellsPositionUpdate();
   return true;
  }
```

In order for the CList class to operate on an object, this object in the list must be the current one. It becomes current, for example, when it is selected. Therefore, here we first get the cell object from the list by index. The cell becomes the current one, and then, using [MoveToIndex()](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist/clistmovetoindex) method of CList class, we are moving the object to the required position in the list. After successfully moving an object to a new position, indexes of the remaining objects must be adjusted, which is done using CellsPositionUpdate() method.

**Method that sets row and column positions for all cells in the list:**

```
//+------------------------------------------------------------------+
//| Устанавливает позиции строки и колонки всем ячейкам              |
//+------------------------------------------------------------------+
void CTableRow::CellsPositionUpdate(void)
  {
//--- В цикле по всем ячейкам в списке
   for(int i=0;i<this.m_list_cells.Total();i++)
     {
      //--- получаем очередную ячейку и устанавливаем в неё индексы строки и столбца
      CTableCell *cell=this.GetCell(i);
      if(cell!=NULL)
         cell.SetPositionInTable(this.Index(),this.m_list_cells.IndexOf(cell));
     }
  }
```

The CList class allows you to find the current object index in the list. To do this, the object must be selected. Here we loop through all the cell objects in the list, select each one and find out its index using [IndexOf()](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist/clistindexof) method of the CList class. Row index and the found cell index are set to the cell object using its SetPositionInTable() method.

**Method that resets data of row cells:**

```
//+------------------------------------------------------------------+
//| Обнуляет данные ячеек строки                                     |
//+------------------------------------------------------------------+
void CTableRow::ClearData(void)
  {
//--- В цикле по всем ячейкам в списке
   for(uint i=0;i<this.CellsTotal();i++)
     {
      //--- получаем очередную ячейку и устанавливаем в неё пустое значение
      CTableCell *cell=this.GetCell(i);
      if(cell!=NULL)
         cell.ClearData();
     }
  }
```

In the loop, reset each next cell in the list using ClearData() cell object method. For string data, an empty row is written to the cell, and for numeric data, zero is written.

**Method that returns description of the object:**

```
//+------------------------------------------------------------------+
//| Возвращает описание объекта                                      |
//+------------------------------------------------------------------+
string CTableRow::Description(void)
  {
   return(::StringFormat("%s: Position %u, Cells total: %u",
                         TypeDescription((ENUM_OBJECT_TYPE)this.Type()),this.Index(),this.CellsTotal()));
  }
```

A row is collected from object's properties and data and returned in the following format, for example:

```
Table Row: Position 1, Cells total: 4:
```

**Method that outputs object description to the log:**

```
//+------------------------------------------------------------------+
//| Выводит в журнал описание объекта                                |
//+------------------------------------------------------------------+
void CTableRow::Print(const bool detail, const bool as_table=false, const int cell_width=10)
  {

//--- Количество ячеек
   int total=(int)this.CellsTotal();

//--- Если вывод в табличном виде
   string res="";
   if(as_table)
     {
      //--- создаём строку таблицы из значений всех ячеек
      string head=" Row "+(string)this.Index();
      string res=::StringFormat("|%-*s |",cell_width,head);
      for(int i=0;i<total;i++)
        {
         CTableCell *cell=this.GetCell(i);
         if(cell==NULL)
            continue;
         res+=::StringFormat("%*s |",cell_width,cell.Value());
        }
      //--- Выводим строку в журнал
      ::Print(res);
      return;
     }

//--- Выводим заголовок в виде описания строки
   ::Print(this.Description()+(detail ? ":" : ""));

//--- Если детализированное описание
   if(detail)
     {

      //--- Вывод не в табличном виде
      //--- В цикле по спискук ячеек строки
      for(int i=0; i<total; i++)
        {
         //--- получаем текущую ячейку и добавляем в итоговую строку её описание
         CTableCell *cell=this.GetCell(i);
         if(cell!=NULL)
            res+="  "+cell.Description()+(i<total-1 ? "\n" : "");
        }
      //--- Выводим в журнал созданную в цикле строку
      ::Print(res);
     }
  }
```

For non-tabular data display in the log, the header is first displayed in the log as row description. Then, if detailed display flag is set, descriptions of each cell are displayed in the log in a loop through the list of cells.

As a result, the detailed display of a table row to the log looks like this, for example (for a non-tabular view):

```
Table Row: Position 0, Cells total: 4:
  Table Cell: Row 0, Col 0, Editable <long>Value: 10
  Table Cell: Row 0, Col 1, Editable <long>Value: 21
  Table Cell: Row 0, Col 2, Editable <long>Value: 32
  Table Cell: Row 0, Col 3, Editable <long>Value: 43
```

For tabular display, the result will be, for example, as follows:

```
| Row 0     |         0 |         1 |         2 |         3 |
```

**Method that saves a table row to a file:**

```
//+------------------------------------------------------------------+
//| Сохранение в файл                                                |
//+------------------------------------------------------------------+
bool CTableRow::Save(const int file_handle)
  {
//--- Проверяем хэндл
   if(file_handle==INVALID_HANDLE)
      return(false);
//--- Сохраняем маркер начала данных - 0xFFFFFFFFFFFFFFFF
   if(::FileWriteLong(file_handle,MARKER_START_DATA)!=sizeof(long))
      return(false);
//--- Сохраняем тип объекта
   if(::FileWriteInteger(file_handle,this.Type(),INT_VALUE)!=INT_VALUE)
      return(false);

//--- Сохраняем индекс
   if(::FileWriteInteger(file_handle,this.m_index,INT_VALUE)!=INT_VALUE)
      return(false);
//--- Сохраняем список ячеек
   if(!this.m_list_cells.Save(file_handle))
      return(false);

//--- Успешно
   return true;
  }
```

Save beginning-of-data markers, then object type. This is the standard header of each object in the file. After that, an entry to the object's properties file follows. Here, the row index is saved to the file, and then the list of cells.

**Method uploading the row from a file:**

```
//+------------------------------------------------------------------+
//| Загрузка из файла                                                |
//+------------------------------------------------------------------+
bool CTableRow::Load(const int file_handle)
  {
//--- Проверяем хэндл
   if(file_handle==INVALID_HANDLE)
      return(false);
//--- Загружаем и проверяем маркер начала данных - 0xFFFFFFFFFFFFFFFF
   if(::FileReadLong(file_handle)!=MARKER_START_DATA)
      return(false);
//--- Загружаем тип объекта
   if(::FileReadInteger(file_handle,INT_VALUE)!=this.Type())
      return(false);

//--- Загружаем индекс
   this.m_index=::FileReadInteger(file_handle,INT_VALUE);
//--- Загружаем список ячеек
   if(!this.m_list_cells.Load(file_handle))
      return(false);

//--- Успешно
   return true;
  }
```

Here everything is in the same order as when saving. First, the beginning-of-data marker and the object type are loaded and checked. Then the row index and the entire list of cells are loaded.

### **4\. Table models class**

In its simplest form, the table model is a linked list of rows, which, in turn, contain linked lists of cells. Our model, which we create today, will receive a two-dimensional array of one of five types at the input (double, long, datetime, color, string), and build a virtual table from it. Further, we will extend this class to accept other arguments for creating tables from other input data. The same model will be considered the default model.

Let's continue writing the code in the same file \\MQL5\\Scripts\\TableModel\\TableModelTest.mq5.

The table model class is essentially a linked list of rows with methods for managing rows, columns, and cells. Write the class body with all the variables and methods, and then consider declared methods of the class:

```
//+------------------------------------------------------------------+
//| Класс модели таблицы                                             |
//+------------------------------------------------------------------+
class CTableModel : public CObject
  {
protected:
   CTableRow         m_row_tmp;                             // Объект строки для поиска в списке
   CListObj          m_list_rows;                           // Список строк таблицы
//--- Создаёт модель таблицы из двумерного массива
template<typename T>
   void              CreateTableModel(T &array[][]);
//--- Возвращает корректный тип данных
   ENUM_DATATYPE     GetCorrectDatatype(string type_name)
                       {
                        return
                          (
                           //--- Целочисленное значение
                           type_name=="bool" || type_name=="char"    || type_name=="uchar"   ||
                           type_name=="short"|| type_name=="ushort"  || type_name=="int"     ||
                           type_name=="uint" || type_name=="long"    || type_name=="ulong"   ?  TYPE_LONG      :
                           //--- Вещественное значение
                           type_name=="float"|| type_name=="double"                          ?  TYPE_DOUBLE    :
                           //--- Значение даты/времени
                           type_name=="datetime"                                             ?  TYPE_DATETIME  :
                           //--- Значение цвета
                           type_name=="color"                                                ?  TYPE_COLOR     :
                           /*--- Строковое значение */                                          TYPE_STRING    );
                       }

//--- Создаёт и добавляет новую пустую строку в конец списка
   CTableRow        *CreateNewEmptyRow(void);
//--- Добавляет строку в конец списка
   bool              AddNewRow(CTableRow *row);
//--- Устанавливает позиции строки и колонки всем ячейкам таблицы
   void              CellsPositionUpdate(void);

public:
//--- Возвращает (1) ячейку, (2) строку по индексу, количество (3) строк, ячеек (4) в указанной строке, (5) в таблице
   CTableCell       *GetCell(const uint row, const uint col);
   CTableRow        *GetRow(const uint index)                  { return this.m_list_rows.GetNodeAtIndex(index);   }
   uint              RowsTotal(void)                     const { return this.m_list_rows.Total();  }
   uint              CellsInRow(const uint index);
   uint              CellsTotal(void);

//--- Устанавливает (1) значение, (2) точность, (3) флаги отображения времени, (4) флаг отображения имён цветов в указанную ячейку
template<typename T>
   void              CellSetValue(const uint row, const uint col, const T value);
   void              CellSetDigits(const uint row, const uint col, const int digits);
   void              CellSetTimeFlags(const uint row, const uint col, const uint flags);
   void              CellSetColorNamesFlag(const uint row, const uint col, const bool flag);
//--- (1) Назначает, (2) отменяет объект в ячейке
   void              CellAssignObject(const uint row, const uint col,CObject *object);
   void              CellUnassignObject(const uint row, const uint col);
//--- (1) Удаляет (2) перемещает ячейку
   bool              CellDelete(const uint row, const uint col);
   bool              CellMoveTo(const uint row, const uint cell_index, const uint index_to);

//--- (1) Возвращает, (2) выводит в журнал описание ячейки, (3) назначенный в ячейку объект
   string            CellDescription(const uint row, const uint col);
   void              CellPrint(const uint row, const uint col);
   CObject          *CellGetObject(const uint row, const uint col);

public:
//--- Создаёт новую строку и (1) добавляет в конец списка, (2) вставляет в указанную позицию списка
   CTableRow        *RowAddNew(void);
   CTableRow        *RowInsertNewTo(const uint index_to);
//--- (1) Удаляет (2) перемещает строку, (3) очищает данные строки
   bool              RowDelete(const uint index);
   bool              RowMoveTo(const uint row_index, const uint index_to);
   void              RowResetData(const uint index);
//--- (1) Возвращает, (2) выводит в журнал описание строки
   string            RowDescription(const uint index);
   void              RowPrint(const uint index,const bool detail);

//--- (1) Удаляет (2) перемещает столбец, (3) очищает данные столбца
   bool              ColumnDelete(const uint index);
   bool              ColumnMoveTo(const uint row_index, const uint index_to);
   void              ColumnResetData(const uint index);

//--- (1) Возвращает, (2) выводит в журнал описание таблицы
   string            Description(void);
   void              Print(const bool detail);
   void              PrintTable(const int cell_width=10);

//--- (1) Очищает данные, (2) уничтожает модель
   void              ClearData(void);
   void              Destroy(void);

//--- Виртуальные методы (1) сравнения, (2) сохранения в файл, (3) загрузки из файла, (4) тип объекта
   virtual int       Compare(const CObject *node,const int mode=0) const;
   virtual bool      Save(const int file_handle);
   virtual bool      Load(const int file_handle);
   virtual int       Type(void)                          const { return(OBJECT_TYPE_TABLE_MODEL);  }

//--- Конструкторы/деструктор
                     CTableModel(void){}
                     CTableModel(double &array[][])   { this.CreateTableModel(array); }
                     CTableModel(long &array[][])     { this.CreateTableModel(array); }
                     CTableModel(datetime &array[][]) { this.CreateTableModel(array); }
                     CTableModel(color &array[][])    { this.CreateTableModel(array); }
                     CTableModel(string &array[][])   { this.CreateTableModel(array); }
                    ~CTableModel(void){}
  };
```

Basically, there is only one object for a linked list of table rows and methods for managing rows, cells, and columns. And constructors that accept different types of two-dimensional arrays.

An array is passed to the class constructor, and **a method to create a table model** is called:

```
//+------------------------------------------------------------------+
//| Создаёт модель таблицы из двумерного массива                     |
//+------------------------------------------------------------------+
template<typename T>
void CTableModel::CreateTableModel(T &array[][])
  {
//--- Получаем из свойств массива количество строк и столбцов таблицы
   int rows_total=::ArrayRange(array,0);
   int cols_total=::ArrayRange(array,1);
//--- В цикле по индексам строк
   for(int r=0; r<rows_total; r++)
     {
      //--- создаём новую пустую строку и добавляем её в конец списка строк
      CTableRow *row=this.CreateNewEmptyRow();
      //--- Если строка создана и добавлена в список,
      if(row!=NULL)
        {
         //--- В цикле по количеству ячеек в строке
         //--- создаём все ячейки, добавляя каждую новую в конец списка ячеек строки
         for(int c=0; c<cols_total; c++)
            row.CreateNewCell(array[r][c]);
        }
     }
  }
```

The method's logic is explained in the comments. The first dimension of the array is table rows, the second one is cells of each row. When creating cells, they use the same data type which array type is passed to the method.

Thus, we can create several table models, which cells initially store different types of data (double, long, datetime, color , and string). Later, after creating the table model, the types of data stored in the cells can be changed.

**A method that creates a new empty row and adds it to the end of the list:**

```
//+------------------------------------------------------------------+
//| Создаёт новую пустую строку и добавляет в конец списка           |
//+------------------------------------------------------------------+
CTableRow *CTableModel::CreateNewEmptyRow(void)
  {
//--- Создаём новый объект строки
   CTableRow *row=new CTableRow(this.m_list_rows.Total());
   if(row==NULL)
     {
      ::PrintFormat("%s: Error. Failed to create new row at position %u",__FUNCTION__, this.m_list_rows.Total());
      return NULL;
     }
//--- Если строку не удалось добавить в список - удаляем созданный новый объект и возвращаем NULL
   if(!this.AddNewRow(row))
     {
      delete row;
      return NULL;
     }

//--- Успешно - возвращаем указатель на созданный объект
   return row;
  }
```

The method creates a new object of the CTableRow class and adds it to the end of rows list using AddNewRow() method. If an addition error occurs, the created new object is deleted and NULL is returned. On success, the method returns a pointer to the row newly added to the list.

**Method that adds a row object to the list end:**

```
//+------------------------------------------------------------------+
//| Добавляет строку в конец списка                                  |
//+------------------------------------------------------------------+
bool CTableModel::AddNewRow(CTableRow *row)
  {
//--- Если передан пустой объект - сообщаем об этом и возвращаем false
   if(row==NULL)
     {
      ::PrintFormat("%s: Error. Empty CTableRow object passed",__FUNCTION__);
      return false;
     }
//--- Устанавливаем строке её индекс в списке и добавляем её в конец списка
   row.SetIndex(this.RowsTotal());
   if(this.m_list_rows.Add(row)==WRONG_VALUE)
     {
      ::PrintFormat("%s: Error. Failed to add row (%u) to list",__FUNCTION__,row.Index());
      return false;
     }

//--- Успешно
   return true;
  }
```

Both of the methods discussed above are located in the protected section of the class, work in pairs, and are used internally when adding new rows to the table.

**Method for creating a new row and adding it to the list end:**

```
//+------------------------------------------------------------------+
//| Создаёт новую строку и добавляет в конец списка                  |
//+------------------------------------------------------------------+
CTableRow *CTableModel::RowAddNew(void)
  {
//--- Создаём новую пустую строку и добавляем её в конец списка строк
   CTableRow *row=this.CreateNewEmptyRow();
   if(row==NULL)
      return NULL;

//--- Создаём ячейки по количеству ячеек первой строки
   for(uint i=0;i<this.CellsInRow(0);i++)
      row.CreateNewCell(0.0);
   row.ClearData();

//--- Успешно - возвращаем указатель на созданный объект
   return row;
  }
```

This is a public method. It is used to add a new row with cells to the table. The number of cells for the created row is taken from the very first row of the table.

**Method for creating and adding a new row to a specified list position:**

```
//+------------------------------------------------------------------+
//| Создаёт и добавляет новую строку в указанную позицию списка      |
//+------------------------------------------------------------------+
CTableRow *CTableModel::RowInsertNewTo(const uint index_to)
  {
//--- Создаём новую пустую строку и добавляем её в конец списка строк
   CTableRow *row=this.CreateNewEmptyRow();
   if(row==NULL)
      return NULL;

//--- Создаём ячейки по количеству ячеек первой строки
   for(uint i=0;i<this.CellsInRow(0);i++)
      row.CreateNewCell(0.0);
   row.ClearData();

//--- Смещаем строку на позицию index_to
   this.RowMoveTo(this.m_list_rows.IndexOf(row),index_to);

//--- Успешно - возвращаем указатель на созданный объект
   return row;
  }
```

Sometimes you need to insert a new row not at the end of the list of rows, but between the existing ones. This method first creates a new row at the end of the list, fills it with cells, clears them, and then moves the row to the desired position.

**Method that sets values to the specified cell:**

```
//+------------------------------------------------------------------+
//| Устанавливает значение в указанную ячейку                        |
//+------------------------------------------------------------------+
template<typename T>
void CTableModel::CellSetValue(const uint row,const uint col,const T value)
  {
//--- Получаем ячейку по индексам строки и столбца
   CTableCell *cell=this.GetCell(row,col);
   if(cell==NULL)
      return;
//--- Получаем корректный тип устанавливаемых данных (double, long, datetime, color, string)
   ENUM_DATATYPE type=this.GetCorrectDatatype(typename(T));
//--- В зависимости от типа данных вызываем соответствующий типу данных
//--- метод ячейки для установки значения, явно указывая требуемый тип
   switch(type)
     {
      case TYPE_DOUBLE  :  cell.SetValue((double)value);    break;
      case TYPE_LONG    :  cell.SetValue((long)value);      break;
      case TYPE_DATETIME:  cell.SetValue((datetime)value);  break;
      case TYPE_COLOR   :  cell.SetValue((color)value);     break;
      case TYPE_STRING  :  cell.SetValue((string)value);    break;
      default           :  break;
     }
  }
```

First, get a pointer to the desired cell by the coordinates of its row and column, and then set a value to it. Whatever the value passed to the method to install it in the cell is, the method will select only the correct type for installation — double, long, datetime, color, or string.

**Method that sets** **accuracy of displaying data in the specified cell:**

```
//+------------------------------------------------------------------+
//| Устанавливает точность отображения данных в указанную ячейку     |
//+------------------------------------------------------------------+
void CTableModel::CellSetDigits(const uint row,const uint col,const int digits)
  {
//--- Получаем ячейку по индексам строки и столбца и
//--- вызываем её соответствующий метод для установки значения
   CTableCell *cell=this.GetCell(row,col);
   if(cell!=NULL)
      cell.SetDigits(digits);
  }
```

The method is relevant only for cells that store the real value type. It is used to specify the number of decimal places for the value displayed by the cell.

**Method that sets** time display flags to the specified cell:

```
//+------------------------------------------------------------------+
//| Устанавливает флаги отображения времени в указанную ячейку       |
//+------------------------------------------------------------------+
void CTableModel::CellSetTimeFlags(const uint row,const uint col,const uint flags)
  {
//--- Получаем ячейку по индексам строки и столбца и
//--- вызываем её соответствующий метод для установки значения
   CTableCell *cell=this.GetCell(row,col);
   if(cell!=NULL)
      cell.SetDatetimeFlags(flags);
  }
```

Relevant for the cells displaying datetime values. Sets the time display format by cell (one of TIME\_DATE\|TIME\_MINUTES\|TIME\_SECONDS, or combinations thereof)

TIME\_DATE gets result as " yyyy.mm.dd " ,

TIME\_MINUTES gets result as " hh:mi " ,

TIME\_SECONDS gets result as " hh:mi:ss ".

**Method that sets** **color name display flags to the specified cell:**

```
//+------------------------------------------------------------------+
//| Устанавливает флаг отображения имён цветов в указанную ячейку    |
//+------------------------------------------------------------------+
void CTableModel::CellSetColorNamesFlag(const uint row,const uint col,const bool flag)
  {
//--- Получаем ячейку по индексам строки и столбца и
//--- вызываем её соответствующий метод для установки значения
   CTableCell *cell=this.GetCell(row,col);
   if(cell!=NULL)
      cell.SetColorNameFlag(flag);
  }
```

Relevant only for the cells displaying color values. It indicates the need to display color names if the color stored in the cell is present in the [color table](https://www.mql5.com/en/docs/constants/objectconstants/webcolors).

**Method that assigns an object to a cell:**

```
//+------------------------------------------------------------------+
//| Назначает объект в ячейку                                        |
//+------------------------------------------------------------------+
void CTableModel::CellAssignObject(const uint row,const uint col,CObject *object)
  {
//--- Получаем ячейку по индексам строки и столбца и
//--- вызываем её соответствующий метод для установки значения
   CTableCell *cell=this.GetCell(row,col);
   if(cell!=NULL)
      cell.AssignObject(object);
  }
```

**Method that cancels assignment of an object in a cell:**

```
//+------------------------------------------------------------------+
//| Отменяет назначение объекта в ячейке                             |
//+------------------------------------------------------------------+
void CTableModel::CellUnassignObject(const uint row,const uint col)
  {
//--- Получаем ячейку по индексам строки и столбца и
//--- вызываем её соответствующий метод для установки значения
   CTableCell *cell=this.GetCell(row,col);
   if(cell!=NULL)
      cell.UnassignObject();
  }
```

The two methods presented above allow you to assign an object to a cell, or remove its assignment. The object must be a descendant of the [CObject](https://www.mql5.com/en/docs/standardlibrary/cobject) class. In the context of articles about tables, an object can be, for example, one of the list of known objects from the ENUM\_OBJECT\_TYPE enumeration. At the moment, the list contains only cell objects, rows, and table models. Assigning them to a cell doesn't make sense. But enumeration will expand as we write articles about View component, where controls will be created. It is them that it would be expedient to assign to a cell, for example, the "drop-down list" control.

**Method that deletes the specified cell:**

```
//+------------------------------------------------------------------+
//| Удаляет ячейку                                                   |
//+------------------------------------------------------------------+
bool CTableModel::CellDelete(const uint row,const uint col)
  {
//--- Получаем строку по индексу и возвращаем результат удаления ячейки из списка
   CTableRow *row_obj=this.GetRow(row);
   return(row_obj!=NULL ? row_obj.CellDelete(col) : false);
  }
```

The method gets the row object by its index and calls its method for deleting the specified cell. For a single cell in a single row, the method does not make sense yet, as it will reduce the number of cells in only one row of the table. This will cause cells to become out of sync with neighboring rows. So far, there is no processing of such deletion, where it is necessary to "expand" the cell next to the deleted one to the size of two cells so that the table structure is not disrupted. However, this method is used as part of the table column deletion method, where cells in all rows are deleted at once without violating the integrity of the entire table.

**Method for moving a table cell:**

```
//+------------------------------------------------------------------+
//| Перемещает ячейку                                                |
//+------------------------------------------------------------------+
bool CTableModel::CellMoveTo(const uint row,const uint cell_index,const uint index_to)
  {
//--- Получаем строку по индексу и возвращаем результат перемещения ячейки на новую позицию
   CTableRow *row_obj=this.GetRow(row);
   return(row_obj!=NULL ? row_obj.CellMoveTo(cell_index,index_to) : false);
  }
```

Getting the row object by its index and calling its method for deleting the specified cell.

**Method that returns the number of cells in the specified row:**

```
//+------------------------------------------------------------------+
//| Возвращает количество ячеек в указанной строке                   |
//+------------------------------------------------------------------+
uint CTableModel::CellsInRow(const uint index)
  {
   CTableRow *row=this.GetRow(index);
   return(row!=NULL ? row.CellsTotal() : 0);
  }
```

Get the row by index and return the number of cells in it by calling CellsTotal() row method.

**Method that returns the** **number of cells in the table:**

```
//+------------------------------------------------------------------+
//| Возвращает количество ячеек в таблице                            |
//+------------------------------------------------------------------+
uint CTableModel::CellsTotal(void)
  {
//--- подсчёт ячеек в цикле по строкам (медленно при большом количестве строк)
   uint res=0, total=this.RowsTotal();
   for(int i=0; i<(int)total; i++)
     {
      CTableRow *row=this.GetRow(i);
      res+=(row!=NULL ? row.CellsTotal() : 0);
     }
   return res;
  }
```

The method goes through all the rows of the table and adds the number of cells of each row to the total result, which is returned. With a large number of rows in the table, such counting can be slow. After all the methods that affect the number of cells in the table are created, take them into account only when their number changes.

**Method that returns the** **specified table cell:**

```
//+------------------------------------------------------------------+
//| Возвращает указанную ячейку таблицы                              |
//+------------------------------------------------------------------+
CTableCell *CTableModel::GetCell(const uint row,const uint col)
  {
//--- Получаем строку по индексу row и возвращаем по индексу col ячейку строки
   CTableRow *row_obj=this.GetRow(row);
   return(row_obj!=NULL ? row_obj.GetCell(col) : NULL);
  }
```

Get the row by row index and return the pointer to the cell object by col index using GetCell() method of row object.

**Method that returns** **description of the cell:**

```
//+------------------------------------------------------------------+
//| Возвращает описание ячейки                                       |
//+------------------------------------------------------------------+
string CTableModel::CellDescription(const uint row,const uint col)
  {
   CTableCell *cell=this.GetCell(row,col);
   return(cell!=NULL ? cell.Description() : "");
  }
```

Get the row by index, get the cell from the row and return its description.

**Method that displays cell description to the log:**

```
//+------------------------------------------------------------------+
//| Выводит в журнал описание ячейки                                 |
//+------------------------------------------------------------------+
void CTableModel::CellPrint(const uint row,const uint col)
  {
//--- Получаем ячейку по индексу строки и колонки и возвращаем её описание
   CTableCell *cell=this.GetCell(row,col);
   if(cell!=NULL)
      cell.Print();
  }
```

Get a pointer to the cell by row and column indexes and, using Print() method of cell object, display its description in the log.

**Method that deletes the specified row:**

```
//+------------------------------------------------------------------+
//| Удаляет строку                                                   |
//+------------------------------------------------------------------+
bool CTableModel::RowDelete(const uint index)
  {
//--- Удаляем строку из списка по индексу
   if(!this.m_list_rows.Delete(index))
      return false;
//--- После удаления строки необходимо обновить все индексы всех ячеек таблицы
   this.CellsPositionUpdate();
   return true;
  }
```

Using [Delete()](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist/clistdelete) method of the CList class delete the row object by index from the list. After deleting the row, indexes of the remaining rows and cells in them do not correspond to reality, and they must be adjusted using CellsPositionUpdate() method.

**Method that moves a row to the specified position:**

```
//+------------------------------------------------------------------+
//| Перемещает строку на указанную позицию                           |
//+------------------------------------------------------------------+
bool CTableModel::RowMoveTo(const uint row_index,const uint index_to)
  {
//--- Получаем строку по индексу, делая её текущей
   CTableRow *row=this.GetRow(row_index);
//--- Перемещаем текущую строку на указанную позицию в списке
   if(row==NULL || !this.m_list_rows.MoveToIndex(index_to))
      return false;
//--- После перемещения строки необходимо обновить все индексы всех ячеек таблицы
   this.CellsPositionUpdate();
   return true;
  }
```

In the CList class, many methods work with the current list object. Get a pointer to the required row, making it the current one, and move it to the required position using [MoveToIndex()](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist/clistmovetoindex) method of the CList class. After shifting the row to a new position, it is necessary to update indexes for the remaining rows, which we do using CellsPositionUpdate() method.

**Method that sets row and column positions for all cells:**

```
//+------------------------------------------------------------------+
//| Устанавливает позиции строки и колонки всем ячейкам              |
//+------------------------------------------------------------------+
void CTableModel::CellsPositionUpdate(void)
  {
//--- В цикле по списку строк
   for(int i=0;i<this.m_list_rows.Total();i++)
     {
      //--- получаем очередную строку
      CTableRow *row=this.GetRow(i);
      if(row==NULL)
         continue;
      //--- устанавливаем строке индекс, найденный методом IndexOf() списка
      row.SetIndex(this.m_list_rows.IndexOf(row));
      //--- Обновляем индексы позиций ячеек строки
      row.CellsPositionUpdate();
     }
  }
```

Go through the list of all rows in the table, select each subsequent row and set a correct index for it, found using [IndexOf()](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist/clistindexof) method of the CList class. Then call CellsPositionUpdate() row method, which sets the correct index for each cell in the row.

**Method that clears data of all cells in a row:**

```
//+------------------------------------------------------------------+
//| Очищает строку (только данные в ячйках)                          |
//+------------------------------------------------------------------+
void CTableModel::RowResetData(const uint index)
  {
//--- Получаем строку из списка и очищаем данные ячеек строки методом ClearData()
   CTableRow *row=this.GetRow(index);
   if(row!=NULL)
      row.ClearData();
  }
```

Each cell in the row is reset to an "empty" value. For now, for simplification purposes, the empty value for numeric cells is zero, but this will be changed later, since zero is also a value that needs to be displayed in the cell. And resetting a value implies displaying an empty cell field.

**Method that clears data of** **all cells in the table:**

```
//+------------------------------------------------------------------+
//| Очищает таблицу (данные всех ячеек)                              |
//+------------------------------------------------------------------+
void CTableModel::ClearData(void)
  {
//--- В цикле по всем строкам таблицы очищаем данные каждой строки
   for(uint i=0;i<this.RowsTotal();i++)
      this.RowResetData(i);
  }
```

Go through all the rows in the table, and for each row call RowResetData() method discussed above.

**Method that returns description of the row:**

```
//+------------------------------------------------------------------+
//| Возвращает описание строки                                       |
//+------------------------------------------------------------------+
string CTableModel::RowDescription(const uint index)
  {
//--- Получаем строку по индексу и возвращаем её описание
   CTableRow *row=this.GetRow(index);
   return(row!=NULL ? row.Description() : "");
  }
```

Get a pointer to the row by index and return its description.

**Method that displays row description in the log:**

```
//+------------------------------------------------------------------+
//| Выводит в журнал описание строки                                 |
//+------------------------------------------------------------------+
void CTableModel::RowPrint(const uint index,const bool detail)
  {
   CTableRow *row=this.GetRow(index);
   if(row!=NULL)
      row.Print(detail);
  }
```

Get a pointer to the row and call Print() method of the received object.

**Method that deletes table column:**

```
//+------------------------------------------------------------------+
//| Удаляет столбец                                                  |
//+------------------------------------------------------------------+
bool CTableModel::ColumnDelete(const uint index)
  {
   bool res=true;
   for(uint i=0;i<this.RowsTotal();i++)
     {
      CTableRow *row=this.GetRow(i);
      if(row!=NULL)
         res &=row.CellDelete(index);
     }
   return res;
  }
```

In a loop through all rows of the table, get each next row and delete a required cell in it by column index. This deletes all cells in the table that have the same column indexes.

**Method that moves a table column:**

```
//+------------------------------------------------------------------+
//| Перемещает столбец                                               |
//+------------------------------------------------------------------+
bool CTableModel::ColumnMoveTo(const uint col_index,const uint index_to)
  {
   bool res=true;
   for(uint i=0;i<this.RowsTotal();i++)
     {
      CTableRow *row=this.GetRow(i);
      if(row!=NULL)
         res &=row.CellMoveTo(col_index,index_to);
     }
   return res;
  }
```

In a loop through all rows of the table, get each next row and move a required cell to a new position. This moves all cells in the table that have the same column indexes.

**Method that clears data of column cells:**

```
//+------------------------------------------------------------------+
//| Очищает данные столбца                                           |
//+------------------------------------------------------------------+
void CTableModel::ColumnResetData(const uint index)
  {
//--- В цикле по всем строкам таблицы
   for(uint i=0;i<this.RowsTotal();i++)
     {
      //--- получаем из каждой строки ячейку с индексом столбца и очищаем её
      CTableCell *cell=this.GetCell(i, index);
      if(cell!=NULL)
         cell.ClearData();
     }
  }
```

In a loop through all rows of the table, get each row and clear data in the required cell. This clears all cells in the table that have the same column indexes.

**Method that returns description of the object:**

```
//+------------------------------------------------------------------+
//| Возвращает описание объекта                                      |
//+------------------------------------------------------------------+
string CTableModel::Description(void)
  {
   return(::StringFormat("%s: Rows %u, Cells in row %u, Cells Total %u",
                         TypeDescription((ENUM_OBJECT_TYPE)this.Type()),this.RowsTotal(),this.CellsInRow(0),this.CellsTotal()));
  }
```

A row is created and returned from some parameters of the table model in this format:

```
Table Model: Rows 4, Cells in row 4, Cells Total 16:
```

**Method that outputs object description to the log:**

```
//+------------------------------------------------------------------+
//| Выводит в журнал описание объекта                                |
//+------------------------------------------------------------------+
void CTableModel::Print(const bool detail)
  {
//--- Выводим в журнал заголовок
   ::Print(this.Description()+(detail ? ":" : ""));
//--- Если детализированное описание,
   if(detail)
     {
      //--- В цикле по всем строкам таблицы
      for(uint i=0; i<this.RowsTotal(); i++)
        {
         //--- получаем очередную строку и выводим в журнал её детализированное описание
         CTableRow *row=this.GetRow(i);
         if(row!=NULL)
            row.Print(true,false);
        }
     }
  }
```

First, the header is printed as description of the model, and then, if the detailed output flag is set, detailed descriptions of all rows of the table model are printed in the loop.

**Method that outputs** **object description to the log in table form:**

```
//+------------------------------------------------------------------+
//| Выводит в журнал описание объекта в табличном виде               |
//+------------------------------------------------------------------+
void CTableModel::PrintTable(const int cell_width=10)
  {
//--- Получаем указатель на первую строку (индекс 0)
   CTableRow *row=this.GetRow(0);
   if(row==NULL)
      return;
   //--- По количеству ячеек первой строки таблицы создаём строку заголовка таблицы
   uint total=row.CellsTotal();
   string head=" n/n";
   string res=::StringFormat("|%*s |",cell_width,head);
   for(uint i=0;i<total;i++)
     {
      if(this.GetCell(0, i)==NULL)
         continue;
      string cell_idx=" Column "+(string)i;
      res+=::StringFormat("%*s |",cell_width,cell_idx);
     }
   //--- Выводим строку заголовка в журнал
   ::Print(res);

   //--- Пройдём в цикле по всем строкам таблицы и распечатаем их в табличном виде
   for(uint i=0;i<this.RowsTotal();i++)
     {
      CTableRow *row=this.GetRow(i);
      if(row!=NULL)
         row.Print(true,true,cell_width);
     }
  }
```

First, based on the number of cells in the very first row of the table, create and print the table header with names of table columns in the log. Then, go through all the rows of the table in a loop and print each one in tabular form.

**Method that destroys table model:**

```
//+------------------------------------------------------------------+
//| Уничтожает модель                                                |
//+------------------------------------------------------------------+
void CTableModel::Destroy(void)
  {
//--- Очищаем список строк
   m_list_rows.Clear();
  }
```

The list of table rows is simply cleared and all objects are destroyed using [Clear()](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist/clistclear) method of the CList class.

**Method for saving table model to file:**

```
//+------------------------------------------------------------------+
//| Сохранение в файл                                                |
//+------------------------------------------------------------------+
bool CTableModel::Save(const int file_handle)
  {
//--- Проверяем хэндл
   if(file_handle==INVALID_HANDLE)
      return(false);
//--- Сохраняем маркер начала данных - 0xFFFFFFFFFFFFFFFF
   if(::FileWriteLong(file_handle,MARKER_START_DATA)!=sizeof(long))
      return(false);
//--- Сохраняем тип объекта
   if(::FileWriteInteger(file_handle,this.Type(),INT_VALUE)!=INT_VALUE)
      return(false);

   //--- Сохраняем список строк
   if(!this.m_list_rows.Save(file_handle))
      return(false);

//--- Успешно
   return true;
  }
```

After saving beginning-of-data marker and the type of the list, save the list of rows to the file using [Save()](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist/clistsave) method of the CList class.

**Method for loading table model from file:**

```
//+------------------------------------------------------------------+
//| Загрузка из файла                                                |
//+------------------------------------------------------------------+
bool CTableModel::Load(const int file_handle)
  {
//--- Проверяем хэндл
   if(file_handle==INVALID_HANDLE)
      return(false);
//--- Загружаем и проверяем маркер начала данных - 0xFFFFFFFFFFFFFFFF
   if(::FileReadLong(file_handle)!=MARKER_START_DATA)
      return(false);
//--- Загружаем тип объекта
   if(::FileReadInteger(file_handle,INT_VALUE)!=this.Type())
      return(false);

   //--- Загружаем список строк
   if(!this.m_list_rows.Load(file_handle))
      return(false);

//--- Успешно
   return true;
  }
```

After loading and checking the beginning-of-data marker and the list type, load the list of rows from the file using Load() method of the CListObj class, discussed at the beginning of the article.

All classes for creating a table model are ready. Now, let's write a script to test model operation.

### Testing the result.

Continue writing the code in the same file. Write a script in which we will create a two-dimensional 4x4 array (4 rows of 4 cells each) with the type, for example, long. Next, open a file to write data of the table model into it and load data into the table from the file. Create a table model and check operation of some of its methods.

Each time the table is changed, we will log the received result using TableModelPrint() function, which selects how to print the table model. If value of macro PRINT\_AS\_TABLE is true, logging is done using PrintTable() method of CTableModel class, if value is false — using Print() method of the same class.

In script, create a table model, print it out in tabular form, and save the model to a file. Then add rows, delete columns, and change the edit permission...

Then download the initial original version of the table from the file again and print out the result.

```
#define  PRINT_AS_TABLE    true  // Распечатывать модель как таблицу
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- Объявляем и заполняем массив с размерностью 4x4
//--- Тип массива может быть double, long, datetime, color, string
   long array[4][4]={{ 1,  2,  3,  4},
                     { 5,  6,  7,  8},
                     { 9, 10, 11, 12},
                     {13, 14, 15, 16}};

//--- Создаём модель таблицы из вышесозданного long-массива array 4x4
   CTableModel *tm=new CTableModel(array);

//--- Если модель не создана - уходим
   if(tm==NULL)
      return;

//--- Распечатаем модель в табличном виде
   Print("The table model has been successfully created:");
   tm.PrintTable();

//--- Удалим объект модели таблицы
   delete tm;
  }
```

As a result, we get the following result of the script in the log:

```
The table model has been successfully created:
|       n/n |  Column 0 |  Column 1 |  Column 2 |  Column 3 |
| Row 0     |         1 |         2 |         3 |         4 |
| Row 1     |         5 |         6 |         7 |         8 |
| Row 2     |         9 |        10 |        11 |        12 |
| Row 3     |        13 |        14 |        15 |        16 |
```

To test working with the table model, adding, deleting and moving rows and columns, working with a file, complete the script:

```
#define  PRINT_AS_TABLE    true  // Распечатывать модель как таблицу
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- Объявляем и заполняем массив с размерностью 4x4
//--- Тип массива может быть double, long, datetime, color, string
   long array[4][4]={{ 1,  2,  3,  4},
                     { 5,  6,  7,  8},
                     { 9, 10, 11, 12},
                     {13, 14, 15, 16}};

//--- Создаём модель таблицы из вышесозданного long-массива array 4x4
   CTableModel *tm=new CTableModel(array);

//--- Если модель не создана - уходим
   if(tm==NULL)
      return;

//--- Распечатаем модель в табличном виде
   Print("The table model has been successfully created:");
   tm.PrintTable();


//--- Проверим работу с файлами и функционал модели таблицы
//--- Открываем файл для записи в него данных модели таблицы
   int handle=FileOpen(MQLInfoString(MQL_PROGRAM_NAME)+".bin",FILE_READ|FILE_WRITE|FILE_BIN|FILE_COMMON);
   if(handle==INVALID_HANDLE)
      return;

   //--- Сохраним в файл оригинальную созданную таблицу
   if(tm.Save(handle))
      Print("\nThe table model has been successfully saved to file.");

//--- Теперь вставим в таблицу новую строку в позицию 2
//--- Получим последнюю ячейку созданной строки и сделаем её нередактируемой
//--- Распечатаем в журнале изменённую модель таблицы
   if(tm.RowInsertNewTo(2))
     {
      Print("\nInsert a new row at position 2 and set cell 3 to non-editable");
      CTableCell *cell=tm.GetCell(2,3);
      if(cell!=NULL)
         cell.SetEditable(false);
      TableModelPrint(tm);
     }

//--- Теперь удалим столбец таблицы с индексом 1 и
//--- распечатаем в журнале полученную модель таблицы
   if(tm.ColumnDelete(1))
     {
      Print("\nRemove column from position 1");
      TableModelPrint(tm);
     }

//--- При сохранении данных таблицы файловый указатель был смещён на последние записанные данные
//--- Поставим указатель в начало файла, загрузим ранее сохранённую оригинальную таблицу и распечатаем её
   if(FileSeek(handle,0,SEEK_SET) && tm.Load(handle))
     {
      Print("\nLoad the original table view from the file:");
      TableModelPrint(tm);
     }

//--- Закроем открытый файл и удалим объект модели таблицы
   FileClose(handle);
   delete tm;
  }
//+------------------------------------------------------------------+
//| Распечатывает модель таблицы                                     |
//+------------------------------------------------------------------+
void TableModelPrint(CTableModel *tm)
  {
   if(PRINT_AS_TABLE)
      tm.PrintTable();  // Распечатать модель как таблицу
   else
      tm.Print(true);   // Распечатать детализированные данные таблицы
  }
```

Get this result in the log:

```
The table model has been successfully created:
|       n/n |  Column 0 |  Column 1 |  Column 2 |  Column 3 |
| Row 0     |         1 |         2 |         3 |         4 |
| Row 1     |         5 |         6 |         7 |         8 |
| Row 2     |         9 |        10 |        11 |        12 |
| Row 3     |        13 |        14 |        15 |        16 |

The table model has been successfully saved to file.

Insert a new row at position 2 and set cell 3 to non-editable
|       n/n |  Column 0 |  Column 1 |  Column 2 |  Column 3 |
| Row 0     |         1 |         2 |         3 |         4 |
| Row 1     |         5 |         6 |         7 |         8 |
| Row 2     |      0.00 |      0.00 |      0.00 |      0.00 |
| Row 3     |         9 |        10 |        11 |        12 |
| Row 4     |        13 |        14 |        15 |        16 |

Remove column from position 1
|       n/n |  Column 0 |  Column 1 |  Column 2 |
| Row 0     |         1 |         3 |         4 |
| Row 1     |         5 |         7 |         8 |
| Row 2     |      0.00 |      0.00 |      0.00 |
| Row 3     |         9 |        11 |        12 |
| Row 4     |        13 |        15 |        16 |

Load the original table view from the file:
|       n/n |  Column 0 |  Column 1 |  Column 2 |  Column 3 |
| Row 0     |         1 |         2 |         3 |         4 |
| Row 1     |         5 |         6 |         7 |         8 |
| Row 2     |         9 |        10 |        11 |        12 |
| Row 3     |        13 |        14 |        15 |        16 |
```

If you want to output to log the data that is not in tabular form from the table model, then set false in PRINT\_AS\_TABLE macro:

```
#define  PRINT_AS_TABLE    false  // Распечатывать модель как таблицу
```

In this case, the following is displayed in the log:

```
The table model has been successfully created:
|       n/n |  Column 0 |  Column 1 |  Column 2 |  Column 3 |
| Row 0     |         1 |         2 |         3 |         4 |
| Row 1     |         5 |         6 |         7 |         8 |
| Row 2     |         9 |        10 |        11 |        12 |
| Row 3     |        13 |        14 |        15 |        16 |

The table model has been successfully saved to file.

Insert a new row at position 2 and set cell 3 to non-editable
Table Model: Rows 5, Cells in row 4, Cells Total 20:
Table Row: Position 0, Cells total: 4:
  Table Cell: Row 0, Col 0, Editable <long>Value: 1
  Table Cell: Row 0, Col 1, Editable <long>Value: 2
  Table Cell: Row 0, Col 2, Editable <long>Value: 3
  Table Cell: Row 0, Col 3, Editable <long>Value: 4
Table Row: Position 1, Cells total: 4:
  Table Cell: Row 1, Col 0, Editable <long>Value: 5
  Table Cell: Row 1, Col 1, Editable <long>Value: 6
  Table Cell: Row 1, Col 2, Editable <long>Value: 7
  Table Cell: Row 1, Col 3, Editable <long>Value: 8
Table Row: Position 2, Cells total: 4:
  Table Cell: Row 2, Col 0, Editable <double>Value: 0.00
  Table Cell: Row 2, Col 1, Editable <double>Value: 0.00
  Table Cell: Row 2, Col 2, Editable <double>Value: 0.00
  Table Cell: Row 2, Col 3, Uneditable <double>Value: 0.00
Table Row: Position 3, Cells total: 4:
  Table Cell: Row 3, Col 0, Editable <long>Value: 9
  Table Cell: Row 3, Col 1, Editable <long>Value: 10
  Table Cell: Row 3, Col 2, Editable <long>Value: 11
  Table Cell: Row 3, Col 3, Editable <long>Value: 12
Table Row: Position 4, Cells total: 4:
  Table Cell: Row 4, Col 0, Editable <long>Value: 13
  Table Cell: Row 4, Col 1, Editable <long>Value: 14
  Table Cell: Row 4, Col 2, Editable <long>Value: 15
  Table Cell: Row 4, Col 3, Editable <long>Value: 16

Remove column from position 1
Table Model: Rows 5, Cells in row 3, Cells Total 15:
Table Row: Position 0, Cells total: 3:
  Table Cell: Row 0, Col 0, Editable <long>Value: 1
  Table Cell: Row 0, Col 1, Editable <long>Value: 3
  Table Cell: Row 0, Col 2, Editable <long>Value: 4
Table Row: Position 1, Cells total: 3:
  Table Cell: Row 1, Col 0, Editable <long>Value: 5
  Table Cell: Row 1, Col 1, Editable <long>Value: 7
  Table Cell: Row 1, Col 2, Editable <long>Value: 8
Table Row: Position 2, Cells total: 3:
  Table Cell: Row 2, Col 0, Editable <double>Value: 0.00
  Table Cell: Row 2, Col 1, Editable <double>Value: 0.00
  Table Cell: Row 2, Col 2, Uneditable <double>Value: 0.00
Table Row: Position 3, Cells total: 3:
  Table Cell: Row 3, Col 0, Editable <long>Value: 9
  Table Cell: Row 3, Col 1, Editable <long>Value: 11
  Table Cell: Row 3, Col 2, Editable <long>Value: 12
Table Row: Position 4, Cells total: 3:
  Table Cell: Row 4, Col 0, Editable <long>Value: 13
  Table Cell: Row 4, Col 1, Editable <long>Value: 15
  Table Cell: Row 4, Col 2, Editable <long>Value: 16

Load the original table view from the file:
Table Model: Rows 4, Cells in row 4, Cells Total 16:
Table Row: Position 0, Cells total: 4:
  Table Cell: Row 0, Col 0, Editable <long>Value: 1
  Table Cell: Row 0, Col 1, Editable <long>Value: 2
  Table Cell: Row 0, Col 2, Editable <long>Value: 3
  Table Cell: Row 0, Col 3, Editable <long>Value: 4
Table Row: Position 1, Cells total: 4:
  Table Cell: Row 1, Col 0, Editable <long>Value: 5
  Table Cell: Row 1, Col 1, Editable <long>Value: 6
  Table Cell: Row 1, Col 2, Editable <long>Value: 7
  Table Cell: Row 1, Col 3, Editable <long>Value: 8
Table Row: Position 2, Cells total: 4:
  Table Cell: Row 2, Col 0, Editable <long>Value: 9
  Table Cell: Row 2, Col 1, Editable <long>Value: 10
  Table Cell: Row 2, Col 2, Editable <long>Value: 11
  Table Cell: Row 2, Col 3, Editable <long>Value: 12
Table Row: Position 3, Cells total: 4:
  Table Cell: Row 3, Col 0, Editable <long>Value: 13
  Table Cell: Row 3, Col 1, Editable <long>Value: 14
  Table Cell: Row 3, Col 2, Editable <long>Value: 15
  Table Cell: Row 3, Col 3, Editable <long>Value: 16
```

This output provides more debugging information. For example, here we see that setting the edit ban flag to a cell is displayed in program logs.

All the stated functionality works as expected, rows are added, moved, columns are deleted, we can change cell properties and work with files.

### Conclusion

This is the first version of a simple table model that allows creating a table from a two-dimensional array of data. Next, we will create other, specialized table models, create View table component, and ahead we will have full-fledged work with tabular data as one of the controls of the user graphical interface.

The file of the script created today with the classes included in it is attached to the article. You can download it for self-study.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/17653](https://www.mql5.com/ru/articles/17653)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17653.zip "Download all attachments in the single ZIP archive")

[TableModelTest.mq5](https://www.mql5.com/en/articles/download/17653/tablemodeltest.mq5 "Download TableModelTest.mq5")(136.43 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Tables in the MVC Paradigm in MQL5: Customizable and sortable table columns](https://www.mql5.com/en/articles/19979)
- [How to publish code to CodeBase: A practical guide](https://www.mql5.com/en/articles/19441)
- [Tables in the MVC Paradigm in MQL5: Integrating the Model Component into the View Component](https://www.mql5.com/en/articles/19288)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Resizable elements](https://www.mql5.com/en/articles/18941)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Containers](https://www.mql5.com/en/articles/18658)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Simple controls](https://www.mql5.com/en/articles/18221)
- [The View component for tables in the MQL5 MVC paradigm: Base graphical element](https://www.mql5.com/en/articles/17960)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/500347)**
(9)


![Alexey Viktorov](https://c.mql5.com/avatar/2017/4/58E3DFDD-D3B2.jpg)

**[Alexey Viktorov](https://www.mql5.com/en/users/alexeyvik)**
\|
4 Apr 2025 at 15:31

**Artyom Trishkin [#](https://www.mql5.com/ru/forum/484047#comment_56361779):**

When a class of SomeObject is loaded from a file by calling the Load() method of this very SomeObject, it checks "did I really read myself from the file?" (that's what you're asking). If not, it means that something went wrong, so there is no point in loading further.

What I have here is a LIST (CListObj) reading an object type from a file. The list does not know what is there (what object) in the file. But it must know this object type to create it in its CreateElement() method. That's why it doesn't check the type of the loaded object from the file. After all, there will be a comparison with Type(), which in this method returns the type of a list, not an object.

Thanks, I've figured it out, I understand.

![Maxim Kuznetsov](https://c.mql5.com/avatar/2016/1/56935A91-AF51.png)

**[Maxim Kuznetsov](https://www.mql5.com/en/users/nektomk)**
\|
5 Apr 2025 at 08:05

I read it, and then I reread it again.

it's anything but a "model" in MVC. Some ListStorage for example

![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)

**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**
\|
5 Apr 2025 at 08:37

Let's get to the point. Keep your opinions to yourself.


![Aleksey Nikolayev](https://c.mql5.com/avatar/2018/8/5B813025-B4F2.jpeg)

**[Aleksey Nikolayev](https://www.mql5.com/en/users/alexeynikolaev2)**
\|
5 Apr 2025 at 09:38

I wonder. Is it possible to get some analogue of python and R dataframes in this way? These are tables where different columns can contain data of different types (from a limited set of types, but including string).


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
5 Apr 2025 at 11:29

**Aleksey Nikolayev [#](https://www.mql5.com/ru/forum/484047#comment_56368076):**

I wonder. Is it possible to get some analogue of python and R dataframes in this way? These are such tables where different columns can contain data of different types (from a limited set of types, but including string).

You can. If we are talking about different columns of one table, then in the described implementation each cell of the table can have a different data type.

![From Novice to Expert: Predictive Price Pathways](https://c.mql5.com/2/182/20160-from-novice-to-expert-predictive-logo.png)[From Novice to Expert: Predictive Price Pathways](https://www.mql5.com/en/articles/20160)

Fibonacci levels provide a practical framework that markets often respect, highlighting price zones where reactions are more likely. In this article, we build an expert advisor that applies Fibonacci retracement logic to anticipate likely future moves and trade retracements with pending orders. Explore the full workflow—from swing detection to level plotting, risk controls, and execution.

![Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Hidformer)](https://c.mql5.com/2/115/Neural_Networks_in_Trading_Hierarchical_Two-Tower_Transformer_Hidformer___LOGO__1.png)[Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Hidformer)](https://www.mql5.com/en/articles/17069)

We invite you to get acquainted with the Hierarchical Double-Tower Transformer (Hidformer) framework, which was developed for time series forecasting and data analysis. The framework authors proposed several improvements to the Transformer architecture, which resulted in increased forecast accuracy and reduced computational resource consumption.

![Developing Trading Strategy: Pseudo Pearson Correlation Approach](https://c.mql5.com/2/181/20065-developing-trading-strategy-logo.png)[Developing Trading Strategy: Pseudo Pearson Correlation Approach](https://www.mql5.com/en/articles/20065)

Generating new indicators from existing ones offers a powerful way to enhance trading analysis. By defining a mathematical function that integrates the outputs of existing indicators, traders can create hybrid indicators that consolidate multiple signals into a single, efficient tool. This article introduces a new indicator built from three oscillators using a modified version of the Pearson correlation function, which we call the Pseudo Pearson Correlation (PPC). The PPC indicator aims to quantify the dynamic relationship between oscillators and apply it within a practical trading strategy.

![Price Action Analysis Toolkit Development (Part 50): Developing the RVGI, CCI and SMA Confluence Engine in MQL5](https://c.mql5.com/2/181/20262-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 50): Developing the RVGI, CCI and SMA Confluence Engine in MQL5](https://www.mql5.com/en/articles/20262)

Many traders struggle to identify genuine reversals. This article presents an EA that combines RVGI, CCI (±100), and an SMA trend filter to produce a single clear reversal signal. The EA includes an on-chart panel, configurable alerts, and the full source file for immediate download and testing.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=iximasfymjwaaisjqymthavqdsvpxnkl&ssn=1769158529160493752&ssn_dr=0&ssn_sr=0&fv_date=1769158529&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F17653&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Implementation%20of%20a%20table%20model%20in%20MQL5%3A%20Applying%20the%20MVC%20concept%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691585297252106&fz_uniq=5062820286807583105&sv=2552)

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