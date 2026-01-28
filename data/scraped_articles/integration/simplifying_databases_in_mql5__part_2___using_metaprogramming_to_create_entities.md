---
title: Simplifying Databases in MQL5 (Part 2): Using metaprogramming to create entities
url: https://www.mql5.com/en/articles/19594
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T17:17:59.242287
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/19594&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068066198517642515)

MetaTrader 5 / Examples


### Introduction

In the previous article, we took the first steps toward understanding how MQL5 handles databases: we created tables, inserted, updated, and deleted records, and even explored transaction features, data import and export. All of this was done directly, writing raw SQL and calling the native functions offered by the language. This step was crucial because it laid the foundation upon which any abstraction layer will be built. But now an inevitable question arises: do we want to write SQL every time we need to handle data within our robots and indicators?

If we consider a more robust system, the answer is no. Working solely with SQL makes the code verbose, repetitive, and error-prone, especially as the application grows and begins to handle multiple tables, relationships, and validations. This is where an ORM (Object-Relational Mapping) comes in: a way to bridge the gap between the object-oriented world in which we program and the relational world in which the data lives. The first step in this direction is to create a way to represent database tables directly as classes in MQL5.

In this second article, we'll learn how to use an often-underestimated but extremely powerful language feature: #define. It will allow us to automate and standardize the creation of structures, avoiding duplication and facilitating future expansions. With it, we'll build our first entities (classes that represent tables) and also a mechanism for describing column metadata, such as data type, primary key, auto-increment, required fields, and default values.

This approach will be the foundation for everything that follows: repositories, query builders, and automatic table creation. In other words, we're beginning to shape the MQL5 ORM we envisioned from the beginning.

### What \#define is and how it works in MQL5

In C-family languages (and MQL5 is in this group), [#define](https://www.mql5.com/en/docs/basis/preprosessor/constant) is a preprocessing tool, that is, something the compiler interprets _even_ before compiling the final code. It doesn't generate functions or create variables; it intelligently replaces text, almost like a "shortcut" system or macro.

In practice, this means we can write a pattern once and reuse it in multiple places in the code, reducing duplication and errors. What's more, we can transform #define into a _metaprogramming_ tool, capable of generating complex structures, such as our database entities, from simple definitions.

Let's start by exploring the different uses, from the most basic to the most advanced.

**1\. The simplest #define: aliases and direct substitutions**

The most common use case is to use #define to create a text shortcut.

```
#define PI 3.14159
#define AUTHOR "João Pedro"

//--- Use
int OnInit()
  {
   double area = PI * 2 * 2;
   Print("Code written by", AUTHOR);
   return(INIT_SUCCEEDED);
  }
```

Here, wheneve the compiler encounters PI, it will replace it with the value 3.14159. There's no type or context checking; it's pure text substitution. This is useful, but still trivial.

**2\. Parameters in macros**

With #define, we can also create macros that receive parameters.

```
#define SQUARE(x) (x * x)

int OnInit()
  {
   Print("Square of 5: ", SQUARE(5));
   Print("Square of 10: ", SQUARE(10));
  }
```

When compiled, SQUARE(5) will be replaced literally by (5 \* 5). This gives you an idea of how we can encapsulate repetitive patterns into reusable forms.

**3\. The # operator: transforming arguments into strings**

A little-explored feature in MQL5 is the # operator, which turns the argument passed to the macro into a literal string.

```
#define META(name) Print("Variable name: ", #name)

int OnInit()
  {
   int value = 42;
   META(value);
  }
```

Note that it doesn't print the variable's contents, but rather its _name_. This trick is extremely useful for generating logs or creating metadata from code identifiers themselves.

**4\. The ## operator: concatenating identifiers**

Another advanced feature is the ## operator, which concatenates tokens (pieces of code).

```
#define META(name) Print("Concatenated: ", name##Id)

int OnInit()
  {
   int userId = 7;
   META(user);
  }
```

The compiler literally glues user with Id, forming userId. This technique allows for dynamically generating variable, method, or constant names, something that would otherwise be impossible in MQL5.

**5\. Macros as parameters for other macros**

So far, we've seen #define as a simple substitution or with parameters. But this is where things get interesting, when we realize that macros can also receive _other macros as arguments_. This opens up a kind of "repetition engine," where we can generate complex code blocks from a single definition.

See this example:

```
// Step 1 - Macro that describes a set of operations
#define MATH_OPERATIONS(OP) \
  OP(Add, +)                \
  OP(Sub, -)                \
  OP(Mul, *)                \
  OP(Div, /)

// Step 2 - Macro that generates functions from the list above
#define GENERATE_FUNCTION(name, symbol) \
  double name(double a, double b) { return a symbol b; }

// Step 3 - Expansion: Creates multiple functions at once
MATH_OPERATIONS(GENERATE_FUNCTION)

// Step 4 - Use
int OnInit()
  {
   Print("2 + 3 = ", Add(2,3));
   Print("10 - 7 = ", Sub(10,7));
   Print("6 * 4 = ", Mul(6,4));
   Print("20 / 5 = ", Div(20,5));
   return INIT_SUCCEEDED;
  }
```

What's happening in this example? Let's go step by step:

- Step 1: Here we create a macro that, by itself, doesn't generate anything useful. It simply lists four elements (Add, Sub, Mul, Div), each associated with the symbol of a mathematical operation. The detail is that each line calls an OP macro, which we don't yet know how it will be implemented. This means that MATH\_OPERATIONS works as a _template_, but it still needs to receive a "tool" that tells it what to do with each element in the list.

- Step 2: This macro already does something concrete: given a name and a symbol, it creates a function that applies the operation. For example, if we pass (Add, +), we will have the following function created:


```
double Add(double a, double b) { return a + b; }
```


And so on for the rest (Sub, Mul, and Div).

- Step 3: Now comes the magic: we pass GENERATE\_FUNCTION as a parameter to the operations list. This causes the compiler to expand each OP call within MATH\_OPERATIONS, replacing OP with GENERATE\_FUNCTION. The end result will be equivalent to writing manually:


```
double Add(double a, double b) { return a + b; }
double Sub(double a, double b) { return a - b; }
double Mul(double a, double b) { return a * b; }
double Div(double a, double b) { return a / b; }
```

- Step 4: Here we directly use the functions created by the compiler from the combination of macros. To the programmer, it seems as if there have always been functions called Add, Sub, Mul, and Div, but in fact they were _built in automatically_.


To summarize:

- A macro can list elements ( MATH\_OPERATIONS ).
- Another macro defines how each element should be transformed into code (GENERATE\_FUNCTION).
- When we combine the two, the compiler automatically generates a set of functions (or methods, properties, classes, etc.).

This technique of passing macros as parameters to other macros is an incredible tool that we will use to eliminate repetition, standardize structures, and create extensible blocks. Later, we will apply the same logic to database tables and columns, but here it is clear that the idea is not limited to databases: _we can use the feature in any situation where there is a repetition of patterns._

### Creating a class that represents a table (Entity)

Let's delve deeper into the concept of an _Entity,_ which is basically a class that mirrors a database table. In other words, if we have a table called Account in the database with columns like id, number, balance, and owner, then in the code we have an Account class with properties that represent each of these columns. This allows us to manipulate data as objects, without having to worry directly about SQL all the time.

Let's imagine we have the Account table in the database. To represent it in MQL5, we would create a class like this:

```
class Account
  {
public:
   ulong             id;        // unique identifier
   double            number;    // account number
   double            balance;   // available balance
   string            owner;     // account owner

                     Account(void);
                    ~Account(void);

   //--- Converts the data into a string
   string            ToString();
  };
Account::Account(void)
  {
  }
Account::~Account(void)
  {
  }
string Account::ToString(void)
  {
   return("Account[id="+ (string)id+ ", number="+ (string)number+ ", balance="+ (string)balance+ ", owner="+ (string)owner+ "]");
  }
```

This code works, but it has two classic problems:

1. Repetition: For each table, we need to manually rewrite all the properties and constructors. If the system has 20 tables, there will be 20 nearly identical classes, with only the fields varying.
2. Poor scalability: If a table field changes (for example, renaming number to account\_number), we need to manually change it in both the database and the class, risking inconsistencies.

This is where metaprogramming with #define comes in. Using macros to automatically generate the entity, the idea is very similar to what we saw in the previous section: we create a list of columns as a macro and, from it, automatically generate the corresponding class. Let's do this step by step:

**Step 1 - Defining the columns with a macro**

```
#define ACCOUNT_COLUMNS(COLUMN) \
  COLUMN(ulong,  id,      0)   \
  COLUMN(double, number,  0.0) \
  COLUMN(double, balance, 0.0) \
  COLUMN(string, owner,   "")
```

Here, ACCOUNT\_COLUMNS defines all the columns in the Account table. Note that each column is described by _type, name, and default value_.

The secret lies in the COLUMN parameter, which will be passed in another macro later to decide what to do with each item.

**Step 2 - Creating the macro that generates the class's attributes**

```
#define ENTITY_FIELD(type, name, default_value) type name;

#define ENTITY_DEFAULT(type, name, default_value) name = default_value;

#define ENTITY_TO_STRING(type, name, default_value) _s += #name+"="+(string)name+", ";
```

These macros are the building blocks of our entity. Each of them takes the triple (type, name, default value) defined in the column list and generates a specific piece of code:

- ENTITY\_FIELD → creates the class attribute.
  - Considering these parameters: ENTITY\_FIELD(ulong, id, 0) generates ulong id;
  - In other words, it only declares the variable with its type.
- ENTITY\_DEFAULT → initializes the attribute with a default value within the constructor.
  - Considering these parameters: ENTITY\_DEFAULT(double, balance, 0.0) generates balance = 0.0;
  - This ensures that every object in the class starts with consistent values.
- ENTITY\_TO\_STRING → generates a string to display the attribute values.
  - Considering these parameters: ENTITY\_TO\_STRING(string, owner, ""), it concatenates the name and value \_s += "owner="+(string)owner+", "; to the string "\_s".
  - This way, we can create a generic ToString method that prints all the fields of the class without having to manually write each attribute.

**Step 3 - Creating the main macro for the entity**

```
#define ENTITY(name, COLUMN) \
class name \
  { \
public: \
                     COLUMN(ENTITY_FIELD) \
                     name(void){COLUMN(ENTITY_DEFAULT)}; \
                    ~name(void){}; \
   string            ToString(void) \
     { \
      string _s = ""; \
      COLUMN(ENTITY_TO_STRING) \
      _s = StringSubstr(_s,0,StringLen(_s)-2); \
      return(#name+ "["+_s+"]"); \
     } \
  };
```

This is the macro that actually _assembles the entire class_. Let's break it down line by line:

1. class name {...};

   - Creates a class with the given name.
   - Example: ENTITY(Account, ACCOUNT\_COLUMNS) → class Account {...};
2. COLUMN(ENTITY\_FIELD)

   - For each column in the list, apply the ENTITY\_FIELD macro.
   - Result: declares all the attributes of the class.
3. Constructor name(void){COLUMN(ENTITY\_DEFAULT)};

   - The constructor calls COLUMN(ENTITY\_DEFAULT), which means it initializes all attributes with their default values.
4. Destructor ~name(void){};

   - Here we only make the empty destructor explicit.
5. ToString() Method


   - Assembles a string \_s by concatenating name=value from each field.
   - Uses COLUMN(ENTITY\_TO\_STRING) to apply this logic to all columns.
   - Removes the trailing comma with StringSubstr.
   - Prints something like this:

Account\[id=1, number=12345, balance=500.0, owner=João\]

**Step 4 - Creating the entity**

```
ENTITY(Account, ACCOUNT_COLUMNS)
```

This single line automatically generates the class equivalent to the one we wrote manually at the beginning. At the end of this, we can use the Account class normally:

![](https://c.mql5.com/2/170/5232218428116.png)

The difference between the manual version and the #define version is huge in terms of _scalability and maintainability_. Now, to create a new entity, simply:

1. Define its columns in a TABLE\_COLUMNS macro.
2. Call DEFINE\_ENTITY(Table, TABLE\_COLUMNS).

This eliminates repetition, makes updating easier (if you change the name of a column, you can change it in the macro), and opens up space for us to expand this concept, for example, by adding automatic constructors, serialization methods, SQL integration, and much more.

### Column metadata: encapsulating the properties

So far, we've created entities that have attributes and even managed to print their values automatically using macros. But there's a problem: these attributes are still _"silent"_.

They know how to store data, but they don't know how to _describe themselves_. For example, we can't ask an attribute: "Are you a primary key (PK)?", "Can you accept null values?", "Is it an auto-increment field?", or "What is the actual type in the database (INTEGER, TEXT, REAL, etc.)?"

This information is called metadata, or data about the data. If we want our ORM to be able to automatically generate SQL (such as creating tables or validating structures), we need an extra layer that stores these properties and allows the entity to describe itself.

The idea is to create a metadata class called IColumnMetadata. It will be responsible for storing all the information about a column:

- Field name (m\_name)
- Logical type in MQL5 (m\_type)
- Database type (m\_db\_type)
- Whether it can be nullable (m\_nullable)
- Whether it is auto\_increment (m\_auto\_increment)
- Whether it is a primary key (m\_primary\_key)
- Whether it is unique (m\_unique)

Thus, each column of the entity can carry a complete description, which will be used in the future to automatically generate CREATE TABLE statements, validate whether the database complies with the entity, and correctly map values between MQL5 and SQL.

We create a new folder called _TickORM_ inside includes, and inside it a folder called _metadata_, and a new file called _IColumnMetadata.mqh_, at the end this is the path: _<MQL5/Includes/TickORM/metadata/ColumnMetadata.mqh>_. And we create the following class:

```
//+------------------------------------------------------------------+
//| class abstract : IColumnMetadata                                 |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : IColumnMetadata                                    |
//| Heritage    : No heritage                                        |
//| Description : Stores all the information in a column.            |
//|                                                                  |
//+------------------------------------------------------------------+
class IColumnMetadata
  {
private:

   //--- Props
   string            m_name;
   string            m_type;
   string            m_db_type;
   bool              m_nullable;
   bool              m_auto_increment;
   bool              m_primary_key;
   bool              m_unique;

public:
                     IColumnMetadata(string name, string type,string db_type,bool nullable,bool auto_increment,bool primary_key,bool unique);
                     IColumnMetadata(void);
                    ~IColumnMetadata(void);

   //--- Get Props
   string            Name(void);
   string            Type(void);
   string            DbType(void);
   bool              Nullable(void);
   bool              AutoIncrement(void);
   bool              PrimaryKey(void);
   bool              Unique(void);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
IColumnMetadata::IColumnMetadata(string name,string type,string db_type,bool nullable,bool auto_increment,bool primary_key,bool unique)
  {
   m_name = name;
   m_type = type;
   m_db_type = db_type;
   m_nullable = nullable;
   m_auto_increment = auto_increment;
   m_primary_key = primary_key;
   m_unique = unique;
  }
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
IColumnMetadata::IColumnMetadata(void)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
IColumnMetadata::~IColumnMetadata(void)
  {
  }
//+------------------------------------------------------------------+
//| Get name                                                         |
//+------------------------------------------------------------------+
string IColumnMetadata::Name(void)
  {
   return m_name;
  }
//+------------------------------------------------------------------+
//| Get type                                                         |
//+------------------------------------------------------------------+
string IColumnMetadata::Type(void)
  {
   return m_type;
  };
//+------------------------------------------------------------------+
//| Get database type                                                |
//+------------------------------------------------------------------+
string IColumnMetadata::DbType(void)
  {
   return m_db_type;
  };
//+------------------------------------------------------------------+
//| Get is nullable                                                  |
//+------------------------------------------------------------------+
bool IColumnMetadata::Nullable(void)
  {
   return m_nullable;
  };
//+------------------------------------------------------------------+
//| Get is auto increment                                            |
//+------------------------------------------------------------------+
bool IColumnMetadata::AutoIncrement(void)
  {
   return m_auto_increment;
  };
//+------------------------------------------------------------------+
//| Get is primary key                                               |
//+------------------------------------------------------------------+
bool IColumnMetadata::PrimaryKey(void)
  {
   return m_primary_key;
  };
//+------------------------------------------------------------------+
//| Get is unique                                                    |
//+------------------------------------------------------------------+
bool IColumnMetadata::Unique(void)
  {
   return m_unique;
  };
//+------------------------------------------------------------------+
```

Note that the class has no intelligence or complex logic behind it; it simply stores the table column data. This level of abstraction may seem bureaucratic at first, but it's exactly what will allow us to reach the next step:

- An entity can expose its metadata.
- The ORM can scan this metadata and automatically generate SQL to create the corresponding table in the database.

Without this layer, every time we wanted to create a table, we would have to manually write the entire CREATE TABLE... statement, which is exactly the kind of repetition we want to eliminate.

Imagine we have a Trades table with three columns:

1. id: integer, primary key, auto\_increment.
2. symbol: required string, cannot be null.
3. volume: decimal number, also required.

With our metadata class, we can describe them like this:

```
IColumnMetadata id("id", "int", "INTEGER", false, true, true, true);
IColumnMetadata symbol("symbol", "string", "TEXT", false, false, false, false);
IColumnMetadata volume("volume", "double", "REAL", false, false, false, false);
```

Now it is not just a simple attribute: it is an _object that knows itself_, that knows how to explain its rules and restrictions.

Now it's not just a simple attribute: it's a self-knowledgeable object that can explain its rules and restrictions.

### Creating the ITableMetadata class: the complete description of the entity

If we previously worked at the column level, now we need to go a step further and think at the entire table level. Each table (or entity) isn't just a set of attributes: it also has a proper name, a primary key, and a collection of column metadata.

In other words, we need a structure that can store all of an entity's IColumnMetadata. If IColumnMetadata describes a field, the ITableMetadata we're going to create describes the entire entity. It needs to answer other questions like: "What is the table name?", "What is the primary key?", "How many columns does it have?", and "What are the properties of each column?"

Furthermore, it needs to be extensible, meaning each entity will have its own metadata version, but they can all follow the same "base interface."

Let's create a new file called TableMetadata.mqh in the <MQL5/Include/TickORM/metadata/TableMetadata.mqh> directory. We've already imported ColumnMetadata.mqh.

```
//+------------------------------------------------------------------+
//| Import                                                           |
//+------------------------------------------------------------------+
#include "PropertyMetadata.mqh"
//+------------------------------------------------------------------+
//| class abstract : IEntityMetadata                                 |
//|                                                                  |
//| [PROPERTY]                                                       |
//| Name        : IEntityMetadata                                    |
//| Heritage    : No heritage                                        |
//| Description : Stores all the information in a table.             |
//|                                                                  |
//+------------------------------------------------------------------+
class ITableMetadata
  {
protected:
   IColumnMetadata  *m_properties[];

public:
                     ITableMetadata(void);
                    ~ITableMetadata(void);

   //--- Add new column
   void              AddColumn(IColumnMetadata *column);
   IColumnMetadata   *Column(int index);
   int               ColumnSize(void);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
ITableMetadata::ITableMetadata(void)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
ITableMetadata::~ITableMetadata(void)
  {
   int size = ArraySize(m_columns);
   for(int i=0;i<size;i++)
     {
      delete m_columns[i];
     }
  }
//+------------------------------------------------------------------+
//| Add new column                                                   |
//+------------------------------------------------------------------+
void ITableMetadata::AddColumn(IColumnMetadata *column)
  {
   int size = ArraySize(m_columns);
   ArrayResize(m_columns,size+1);
   m_columns[size] = column;
  }
//+------------------------------------------------------------------+
//| Get property metadata                                            |
//+------------------------------------------------------------------+
IColumnMetadata *ITableMetadata::Column(int index)
  {
   return(m_columns[index]);
  }
//+------------------------------------------------------------------+
//| Get size columns                                                 |
//+------------------------------------------------------------------+
int ITableMetadata::ColumnSize(void)
  {
   return(ArraySize(m_columns));
  }
//+------------------------------------------------------------------+
```

Note that I've already added an array of the IColumnMetadata object, where each position in the array represents a table column. I've also added some basic methods for manipulating the array.

Finally, we've added two [virtual](https://www.mql5.com/en/docs/basis/oop/virtual) methods. This means that any child class can override TableName() and PrimaryKey() to describe itself, since only child classes know the table name. To avoid this, we've created a base implementation that returns NULL.

```
class ITableMetadata
  {
public:
   //--- Virtual methods (will be implemented in the child class)
   virtual string    TableName(void);
   virtual string    PrimaryKey(void);
  };
//+------------------------------------------------------------------+
//| Get table name                                                   |
//+------------------------------------------------------------------+
string ITableMetadata::TableName(void)
  {
   return(NULL);
  }
//+------------------------------------------------------------------+
//| Get is primary key                                               |
//+------------------------------------------------------------------+
string ITableMetadata::PrimaryKey(void)
  {
   return(NULL);
  }
//+------------------------------------------------------------------+
```

### Connecting everything

So far, we've gone through two fundamental steps: first, we manually created an entity class; then, we saw how to use macro metaprogramming to automatically generate these classes. But there's still one important piece missing: we need to centralize the definition of entities and their columns in a single location. This location must be able to translate native MQL5 types into SQL types and, at the same time, record all the table metadata in an organized manner.

This is precisely the role of the TickORM.mqh file, which will be located at <MQL5/TickORM/TickORM.mqh>. We can think of it as a bridge connecting the code written in MQL5 to the relational database model. The logic is simple: for each entity, we create a class derived from ITableMetadata that automatically records all the table columns.

This allows the ORM to not only understand the table structure, but also how to create, validate, and manipulate columns, without requiring developers to manually perform the same conversions for each new entity.

To better understand the usefulness of this, let's take a step-by-step approach: manually construct a metadata class for the Account table, which has four basic columns. We won't use macros here yet, precisely to clarify the work we want to automate:

```
class AccountMetadata : public ITableMetadata
  {
public:
                     AccountMetadata(void);
                    ~AccountMetadata(void);

   string            TableName();
   string            PrimaryKey(void);
  };
AccountMetadata::AccountMetadata(void)
  {
   this.AddColumn(new IColumnMetadata("id","ulong","INTEGER",false,true,true,true));
   this.AddColumn(new IColumnMetadata("number","ulong","REAL",false,false,false,false));
   this.AddColumn(new IColumnMetadata("balance","ulong","REAL",false,false,false,false));
   this.AddColumn(new IColumnMetadata("owner","ulong","TEXT",false,false,false,false));
  }
AccountMetadata::~AccountMetadata(void)
  {
  }
string AccountMetadata::TableName(void)
  {
   return("Account");
  }
string AccountMetadata::PrimaryKey(void)
  {
   int size = ArraySize(m_columns);
   for(int i=0;i<size;i++)
     {
      if(m_columns[i].PrimaryKey())
        {
         return(m_columns[i].Name());
        }
     }
   return(NULL);
  }
```

This implementation is sufficient for us to query the table's metadata. Here's how we can use it:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   AccountMetadata metadata;
   int size_cols = metadata.ColumnSize();
   Print("table name: ",metadata.TableName());
   Print("ptimary key: ",metadata.PrimaryKey());
   Print("size columns: ",size_cols);
   for(int i=0;i<size_cols;i++)
     {
      IColumnMetadata *column = metadata.Column(i);
      Print("===");
      Print("Column name: "+column.Name());
      Print("Type: "+column.Type());
      Print("DbType: "+column.DbType());
      Print("Nullable: "+column.Nullable());
      Print("PrimaryKey: "+column.PrimaryKey());
      Print("Unique: "+column.Unique());
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

The console output clearly shows all the captured information:

```
table name: Account
ptimary key: id
size columns: 4
===
Column name: id
Type: ulong
DbType: INTEGER
Nullable: false
PrimaryKey: true
Unique: true
===
Column name: number
Type: ulong
DbType: REAL
Nullable: false
PrimaryKey: false
Unique: false
===
Column name: balance
Type: ulong
DbType: REAL
Nullable: false
PrimaryKey: false
Unique: false
===
Column name: owner
Type: ulong
DbType: TEXT
Nullable: false
PrimaryKey: false
Unique: false
```

Note that we already have all the essential data for the table and its columns available at runtime. What's missing now is to automate the creation of this metadata class using #define, so we don't have to manually repeat this entire structure for each new entity. This will be the next step.

### Automating the creation of the metadata class

When we start connecting MQL5 entities to the database, a challenge immediately arises: the data types don't speak the same language. In MQL5 code, we use int, double, string, and others. In the database, the types are different: INTEGER, REAL, TEXT, etc.

If we didn't create a clear translation rule, each column would need to be mapped manually, which would be laborious and error-prone. To avoid this rework, we created a small "conversion dictionary" using #define:

```
#define MQL5_TO_SQL_int        "INTEGER"
#define MQL5_TO_SQL_double     "REAL"
#define MQL5_TO_SQL_float      "REAL"
#define MQL5_TO_SQL_long       "INTEGER"
#define MQL5_TO_SQL_ulong      "INTEGER"
#define MQL5_TO_SQL_datetime   "INTEGER"
#define MQL5_TO_SQL_string     "TEXT"
#define MQL5_TO_SQL_bool       "INTEGER"
#define DB_TYPE_FROM_MQL5(type) MQL5_TO_SQL_##type
```

It works simply: if we declare a column as int in the entity, the DB\_TYPE\_FROM\_MQL5 macro automatically converts it to "INTEGER". This ensures that each MQL5 type always has its corresponding type in the database, without having to remember or manually repeat this mapping.

Now, with the types resolved, we need a way to organize the metadata for each table. To do this, we dynamically create a class called _name##Metadata_ (e.g., AccountMetadata ). This class inherits from ITableMetadata and has two main functions:

- TableName(): returns the entity name (which will be used as the table name).
- PrimaryKey(): automatically identifies which column has been marked as the primary key.

```
#define ENTITY_META_DATA(name, COLUMNS) \
class name##Metadata : public ITableMetadata \
  { \
public: \
                     name##Metadata(void) \
     { \
      COLUMNS(ENTITY_META_DATA_COLUMNS); \
     } \
                    ~name##Metadata(void){}; \
   string            TableName() { return(#name); }; \
   string            PrimaryKey(void) \
     { \
      int size = ArraySize(m_columns); \
      for(int i=0;i<size;i++) \
        { \
         if(m_columns[i].PrimaryKey()) \
           { \
            return(m_columns[i].Name()); \
           } \
        } \
      return(NULL); \
     } \
  };
```

Finally, to create a complete entity (class + metadata), we use two macros: the first defines the columns, the second automatically generates the entity and its metadata class:

```
#define ACCOUNT_COLUMNS(COLUMN) \
  COLUMN(ulong,  id,      false, 0, true,  true,  false) \
  COLUMN(double, number,  false, 0, false, false, false) \
  COLUMN(double, balance, false, 0, false, false, false) \
  COLUMN(string, owner,   false,"", false, false, false)

ENTITY(Account, COLUMNS)
ENTITY_META_DATA(Account, COLUMNS)
```

Here, in just a few lines, we declare the entire structure of the Account table:

- id is the primary key (primary = true) and auto-increment (auto\_inc = true),
- number and balance are required numbers,
- owner is required text.

In other words, with a single point of definition, we were able to create the Account class and its metadata class AccountMetadata, ready for use by the ORM.

### Conclusion and next steps

We've reached the end of another stage in building our ORM. In this article, we've covered an important path:

- We started by better understanding how #define works in MQL5, not just for simple constants, but as a metaprogramming tool.
- We moved on to creating entities (the classes that represent our tables) and saw how to simplify their definition using macros.
- We enriched these entities with column metadata, describing attributes such as type, primary key, auto-increment, uniqueness, and nullability.
- Finally, we centralized everything in the TickORM.mqh file, connecting MQL5 types to SQL and automating the generation of metadata classes.

This foundation is crucial: now we not only have entities, but also the complete description of their properties, and this will be the engine that will allow the ORM to manipulate the database intelligently and automatically.

In the next article, we'll take another decisive step: we'll create the Repository layer. This layer will be responsible for manipulating the data without having to manually write SQL. Instead, we'll make calls like accountRepository.Save(account) or ordersRepository.FindById(1), and the ORM will take care of the rest.

In other words, if we've learned to describe the structure of the tables so far, in the next article we'll learn how to operate on the data in a clean, organized, and secure manner.

| File Name | Description |
| --- | --- |
| Include/TickORM/metadata/ColumnMetadata.mq5 | Interface that represents the column data |
| Include/TickORM/metadata/TableMetadata.mqh | Interface that represents the table data |
| Include/TickORM/TickORM.mqh | Main file |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19594.zip "Download all attachments in the single ZIP archive")

[TickORM.zip](https://www.mql5.com/en/articles/download/19594/TickORM.zip "Download TickORM.zip")(3.02 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Simplifying Databases in MQL5 (Part 1): Introduction to Databases and SQL](https://www.mql5.com/en/articles/19285)
- [Mastering Log Records (Part 10): Avoiding Log Replay by Implementing a Suppression](https://www.mql5.com/en/articles/19014)
- [Mastering Log Records (Part 9): Implementing the builder pattern and adding default configurations](https://www.mql5.com/en/articles/18602)
- [Mastering Log Records (Part 8): Error Records That Translate Themselves](https://www.mql5.com/en/articles/18467)
- [Mastering Log Records (Part 7): How to Show Logs on Chart](https://www.mql5.com/en/articles/18291)
- [Mastering Log Records (Part 6): Saving logs to database](https://www.mql5.com/en/articles/17709)

**[Go to discussion](https://www.mql5.com/en/forum/495959)**

![Developing a Volatility Based Breakout System](https://c.mql5.com/2/171/19459-developing-a-volatility-based-logo.png)[Developing a Volatility Based Breakout System](https://www.mql5.com/en/articles/19459)

Volatility based breakout system identifies market ranges, then trades when price breaks above or below those levels, filtered by volatility measures such as ATR. This approach helps capture strong directional moves.

![Price Action Analysis Toolkit Development (Part 41): Building a Statistical Price-Level EA in MQL5](https://c.mql5.com/2/171/19589-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 41): Building a Statistical Price-Level EA in MQL5](https://www.mql5.com/en/articles/19589)

Statistics has always been at the heart of financial analysis. By definition, statistics is the discipline that collects, analyzes, interprets, and presents data in meaningful ways. Now imagine applying that same framework to candlesticks—compressing raw price action into measurable insights. How helpful would it be to know, for a specific period of time, the central tendency, spread, and distribution of market behavior? In this article, we introduce exactly that approach, showing how statistical methods can transform candlestick data into clear, actionable signals.

![Developing Trading Strategies with the Parafrac and Parafrac V2 Oscillators: Single Entry Performance Insights](https://c.mql5.com/2/170/19439-developing-trading-strategies-logo.png)[Developing Trading Strategies with the Parafrac and Parafrac V2 Oscillators: Single Entry Performance Insights](https://www.mql5.com/en/articles/19439)

This article introduces the ParaFrac Oscillator and its V2 model as trading tools. It outlines three trading strategies developed using these indicators. Each strategy was tested and optimized to identify their strengths and weaknesses. Comparative analysis highlighted the performance differences between the original and V2 models.

![Neuro-symbolic systems in algorithmic trading: Combining symbolic rules and neural networks](https://c.mql5.com/2/112/Neurosymbolic_systems_in_algo-trading___LOGO.png)[Neuro-symbolic systems in algorithmic trading: Combining symbolic rules and neural networks](https://www.mql5.com/en/articles/16894)

The article describes the experience of developing a hybrid trading system that combines classical technical analysis with neural networks. The author provides a detailed analysis of the system architecture from basic pattern analysis and neural network structure to the mechanisms behind trading decisions, and shares real code and practical observations.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/19594&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068066198517642515)

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