---
title: Library for easy and quick development of MetaTrader programs (part XX): Creating and storing program resources
url: https://www.mql5.com/en/articles/7195
categories: Trading Systems
relevance_score: 6
scraped_at: 2026-01-23T11:48:40.383691
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/7195&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062736444750997477)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/7195#node01)
- [File generator class](https://www.mql5.com/en/articles/7195#node02)
- [Program resource collection class](https://www.mql5.com/en/articles/7195#node03)
- [Testing access to automatically created files](https://www.mql5.com/en/articles/7195#node04)
- [What's next?](https://www.mql5.com/en/articles/7195#node05)


### Concept

When developing an application, we often need audio and images. The MQL language features several methods of using such data and all of them
require

[downloading files from the terminal file sandbox](https://www.mql5.com/en/docs/files). If the final result is a
compiled file, you only have to

[connect the file as a resource](https://www.mql5.com/en/docs/runtime/resources) and get rid of the necessity to pass
additional files for the program operation. This method is well suited for sending programs to

[mql5.com Market](https://www.mql5.com/en/market) since only an executable file is required there. However, this method is
not suitable for placing a source code to

[mql5.com CodeBase](https://www.mql5.com/en/code) since audio \*.wav file and graphical \*.bmp files cannot be placed there.
Without them, the source code lacks some important features.

The solution is quite obvious: data of all required files should be stored in the source code of \*.mqh include files as binary arrays. When
launching the application, all the necessary files are generated in the necessary directories out of the existing data sets. Thus, when
launching such a code, the application generates all the necessary files, places them to folders and works correctly. All this happens
seamless for users with the only exception that the time of the first launch is slightly increased due to generating and writing missing
files.

We will create two classes for working with file data:

- file generator class out of prepared data,

- class for working with the list of created files — collection of objects describing files.


File generator class will have a set of static methods for creating files out of the appropriate data. These methods will be available for any part
of the library.

The class for working with already generated files will feature the list of all created files and methods for accessing these files by
data stored in the list. In fact, the list will contain simple descriptions of created files (file name and description). This data enables
us to always gain access to a physical file to work with it.

### File generator class

Let's start from the **Datas.mqh** file. We will add the necessary messages to it. They can be displayed during the operation of
created classes.

**Add constants specifying the location of new messages in the**
**array of the library text data to the enumeration of text message indices:**

```
//+------------------------------------------------------------------+
//| List of the library's text message indices                       |
//+------------------------------------------------------------------+
enum ENUM_MESSAGES_LIB
  {
   MSG_LIB_PARAMS_LIST_BEG=ERR_USER_ERROR_FIRST,      // Beginning of the parameter list
   MSG_LIB_PARAMS_LIST_END,                           // End of the parameter list
   MSG_LIB_PROP_NOT_SUPPORTED,                        // Property not supported
   MSG_LIB_PROP_NOT_SUPPORTED_MQL4,                   // Property not supported in MQL4
   MSG_LIB_PROP_NOT_SUPPORTED_POSITION,               // Property not supported for position
   MSG_LIB_PROP_NOT_SUPPORTED_PENDING,                // Property not supported for pending order
   MSG_LIB_PROP_NOT_SUPPORTED_MARKET,                 // Property not supported for market order
   MSG_LIB_PROP_NOT_SUPPORTED_MARKET_HIST,            // Property not supported for historical market order
   MSG_LIB_PROP_NOT_SET,                              // Value not set
   MSG_LIB_PROP_EMPTY,                                // Not set

   MSG_LIB_SYS_ERROR,                                 // Error
   MSG_LIB_SYS_NOT_SYMBOL_ON_SERVER,                  // Error. No such symbol on server
   MSG_LIB_SYS_FAILED_PUT_SYMBOL,                     // Failed to place to market watch. Error:
   MSG_LIB_SYS_NOT_GET_PRICE,                         // Failed to get current prices. Error:
   MSG_LIB_SYS_NOT_GET_MARGIN_RATES,                  // Failed to get margin ratios. Error:
   MSG_LIB_SYS_NOT_GET_DATAS,                         // Failed to get data

   MSG_LIB_SYS_FAILED_CREATE_STORAGE_FOLDER,          // Failed to create folder for storing files. Error:
   MSG_LIB_SYS_FAILED_ADD_ACC_OBJ_TO_LIST,            // Error. Failed to add current account object to collection list
   MSG_LIB_SYS_FAILED_CREATE_CURR_ACC_OBJ,            // Error. Failed to create account object with current account data
   MSG_LIB_SYS_FAILED_OPEN_FILE_FOR_WRITE,            // Could not open file for writing
   MSG_LIB_SYS_INPUT_ERROR_NO_SYMBOL,                 // Input error: no symbol
   MSG_LIB_SYS_FAILED_CREATE_SYM_OBJ,                 // Failed to create symbol object
   MSG_LIB_SYS_FAILED_ADD_SYM_OBJ,                    // Failed to add symbol

   MSG_LIB_SYS_NOT_GET_CURR_PRICES,                   // Failed to get current prices by event symbol
   MSG_LIB_SYS_EVENT_ALREADY_IN_LIST,                 // This event is already in the list
   MSG_LIB_SYS_FILE_RES_ALREADY_IN_LIST,              // This file already created and added to list:
   MSG_LIB_SYS_FAILED_CREATE_RES_LINK,                // Error. Failed to create object pointing to resource file
   MSG_LIB_SYS_ERROR_ALREADY_CREATED_COUNTER,         // Error. Counter with ID already created
   MSG_LIB_SYS_FAILED_CREATE_COUNTER,                 // Failed to create timer counter
   MSG_LIB_SYS_FAILED_CREATE_TEMP_LIST,               // Error creating temporary list
   MSG_LIB_SYS_ERROR_NOT_MARKET_LIST,                 // Error. This is not a market collection list
   MSG_LIB_SYS_ERROR_NOT_HISTORY_LIST,                // Error. This is not a history collection list
   MSG_LIB_SYS_FAILED_ADD_ORDER_TO_LIST,              // Could not add order to the list
   MSG_LIB_SYS_FAILED_ADD_DEAL_TO_LIST,               // Could not add deal to the list
   MSG_LIB_SYS_FAILED_ADD_CTRL_ORDER_TO_LIST,         // Failed to add control order
   MSG_LIB_SYS_FAILED_ADD_CTRL_POSITION_TO_LIST,      // Failed to add control position
   MSG_LIB_SYS_FAILED_ADD_MODIFIED_ORD_TO_LIST,       // Could not add modified order to the list of modified orders

   MSG_LIB_SYS_NO_TICKS_YET,                          // No ticks yet
   MSG_LIB_SYS_FAILED_CREATE_OBJ_STRUCT,              // Could not create object structure
   MSG_LIB_SYS_FAILED_WRITE_UARRAY_TO_FILE,           // Could not write uchar array to file
   MSG_LIB_SYS_FAILED_LOAD_UARRAY_FROM_FILE,          // Could not load uchar array from file
   MSG_LIB_SYS_FAILED_CREATE_OBJ_STRUCT_FROM_UARRAY,  // Could not create object structure from uchar array
   MSG_LIB_SYS_FAILED_SAVE_OBJ_STRUCT_TO_UARRAY,      // Failed to save object structure to uchar array, error
   MSG_LIB_SYS_ERROR_INDEX,                           // Error. "index" value should be within 0 - 3
   MSG_LIB_SYS_ERROR_FAILED_CONV_TO_LOWERCASE,        // Failed to convert string to lowercase, error

   MSG_LIB_SYS_ERROR_EMPTY_STRING,                    // Error. Predefined symbols string empty, to be used
   MSG_LIB_SYS_FAILED_PREPARING_SYMBOLS_ARRAY,        // Failed to prepare array of used symbols. Error
   MSG_LIB_SYS_INVALID_ORDER_TYPE,                    // Invalid order type:

```

**Write texts in English and Russian corresponding to the appropriate index**
**constants** **to the array of text messages:**

```
//+------------------------------------------------------------------+
string messages_library[][TOTAL_LANG]=
  {
   {"Начало списка параметров","Beginning of the event parameter list"},
   {"Конец списка параметров","End of the parameter list"},
   {"Свойство не поддерживается","Property not supported"},
   {"Свойство не поддерживается в MQL4","Property not supported in MQL4"},
   {"Свойство не поддерживается у позиции","Property not supported for position"},
   {"Свойство не поддерживается у отложенного ордера","Property not supported for pending order"},
   {"Свойство не поддерживается у маркет-ордера","Property not supported for market order"},
   {"Свойство не поддерживается у исторического маркет-ордера","Property not supported for historical market order"},
   {"Значение не задано","Value not set"},
   {"Отсутствует","Not set"},

   {"Ошибка ","Error "},
   {"Ошибка. Такого символа нет на сервере","Error. No such symbol on server"},
   {"Не удалось поместить в обзор рынка. Ошибка: ","Failed to put in market watch. Error: "},
   {"Не удалось получить текущие цены. Ошибка: ","Could not get current prices. Error: "},
   {"Не удалось получить коэффициенты взимания маржи. Ошибка: ","Failed to get margin rates. Error: "},
   {"Не удалось получить данные ","Failed to get data of "},

   {"Не удалось создать папку хранения файлов. Ошибка: ","Could not create file storage folder. Error: "},
   {"Ошибка. Не удалось добавить текущий объект-аккаунт в список-коллекцию","Error. Failed to add current account object to collection list"},
   {"Ошибка. Не удалось создать объект-аккаунт с данными текущего счёта","Error. Failed to create account object with current account data"},
   {"Не удалось открыть для записи файл ","Could not open file for writing: "},
   {"Ошибка входных данных: нет символа ","Input error: no "},
   {"Не удалось создать объект-символ ","Failed to create symbol object "},
   {"Не удалось добавить символ ","Failed to add "},

   {"Не удалось получить текущие цены по символу события ","Failed to get current prices by event symbol "},
   {"Такое событие уже есть в списке","This event already in the list"},
   {"Такой файл уже создан и добавлен в список: ","This file has already been created and added to list: "},
   {"Ошибка. Не удалось создать объект-указатель на файл ресурса","Error. Failed to create resource file link object"},

   {"Ошибка. Уже создан счётчик с идентификатором ","Error. Already created counter with id "},
   {"Не удалось создать счётчик таймера ","Failed to create timer counter "},

   {"Ошибка создания временного списка","Error creating temporary list"},
   {"Ошибка. Список не является списком рыночной коллекции","Error. The list is not a list of market collection"},
   {"Ошибка. Список не является списком исторической коллекции","Error. The list is not a list of history collection"},
   {"Не удалось добавить ордер в список","Could not add order to list"},
   {"Не удалось добавить сделку в список","Could not add deal to list"},
   {"Не удалось добавить контрольный ордер ","Failed to add control order "},
   {"Не удалось добавить контрольую позицию ","Failed to add control position "},
   {"Не удалось добавить модифицированный ордер в список изменённых ордеров","Could not add modified order to list of modified orders"},

   {"Ещё не было тиков","No ticks yet"},
   {"Не удалось создать структуру объекта","Could not create object structure"},
   {"Не удалось записать uchar-массив в файл","Could not write uchar array to file"},
   {"Не удалось загрузить uchar-массив из файла","Could not load uchar array from file"},
   {"Не удалось создать структуру объекта из uchar-массива","Could not create object structure from uchar array"},
   {"Не удалось сохранить структуру объекта в uchar-массив, ошибка ","Failed to save object structure to uchar array, error "},
   {"Ошибка. Значение \"index\" должно быть в пределах 0 - 3","Error. \"index\" value should be between 0 - 3"},
   {"Не удалось преобразовать строку в нижний регистр, ошибка ","Failed to convert string to lowercase, error "},

   {"Ошибка. Строка предопределённых символов пустая, будет использоваться ","Error. String of predefined symbols is empty, symbol will be used: "},
   {"Не удалось подготовить массив используемых символов. Ошибка ","Failed to create array of used characters. Error "},
   {"Неправильный тип ордера: ","Invalid order type: "},
```

Please note that the sequence of texts in the array should exactly match the sequence of declaring
index constants in the enumeration.

For the file generator class to work, we need to create a database, from which the class is to take data on audio and image files. Such data should
be in the form of

[unsigned char](https://www.mql5.com/en/docs/basis/types/integer/integertypes#uchar) arrays. To create them, we need
audio (\*.wav) and bitmap (\*.bmp) files to be stored in the library source. I have already prepared some test data as an example. Sound and
bitmap files are to be stored each in its own include file.

In the \\MQL5\\Include\ **DoEasy\** root directory of the library, create the **DataSND.mqh** include file and add the
names of audio files to be placed in the arrays (data array names can be added later since they are only needed for a quick search of the array
declaration places with the data of a certain file since the arrays cannot exceed 16 MB):

```
//+------------------------------------------------------------------+
//|                                                      DataSND.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Audio                                                            |
//+------------------------------------------------------------------+
/*
   sound_array_coin_01           // Falling coin 1
   sound_array_coin_02           // Falling coins
   sound_array_coin_03           // Coins
   sound_array_coin_04           // Falling coin 2
   //---
   sound_array_click_01          // Button click 1
   sound_array_click_02          // Button click 2
   sound_array_click_03          // Button click 3
   //---
   sound_array_cash_machine_01   // Cash register 1
*/
//+------------------------------------------------------------------+
```

To insert the file into the program source code, click **Edit**--\> **Insert**--\> **File** as Binary
Array":

![](https://c.mql5.com/2/37/metaeditor64_wySxk7RD9S.png)

This will open the file selection window where you should find a previously prepared file to upload its data to the array. The array is generated
automatically based on the name of the selected file (the example is not complete since the binary data are numerous):

```
//+------------------------------------------------------------------+
//|                                                      DataSND.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Audio                                                            |
//+------------------------------------------------------------------+
/*
   sound_array_coin_01           // Falling coin 1
   sound_array_coin_02           // Falling coins
   sound_array_coin_03           // Coins
   sound_array_coin_04           // Falling coin 2
   //---
   sound_array_click_01          // Button click 1
   sound_array_click_02          // Button click 2
   sound_array_click_03          // Button click 3
   //---
   sound_array_cash_machine_01   // Cash register 1
*/
//+------------------------------------------------------------------+
//| Falling coin 01                                                  |
//+------------------------------------------------------------------+
unsigned char sound_array_coin_01[]=
  {
   0x52,0x49,0x46,0x46,0x1E,0x50,0x09,0x00,0x57,0x41,0x56,0x45,0x66,0x6D,0x74,0x20,
   0x12,0x00,0x00,0x00,0x03,0x00,0x02,0x00,0x44,0xAC,0x00,0x00,0x20,0x62,0x05,0x00,
   0x08,0x00,0x20,0x00,0x00,0x00,0x64,0x61,0x74,0x61,0x28,0x35,0x09,0x00,0x11,0xBA,
   0x20,0xBB,0x11,0xBA,0x20,0xBB,0x74,0x81,0x62,0xBB,0x74,0x81,0x62,0xBB,0xE9,0x37,
   0x4D,0xBB,0xE9,0x37,0x4D,0xBB,0x7C,0x41,0x13,0xBB,0x7C,0x41,0x13,0xBB,0x8F,0xE4,
   0x84,0xBA,0x8F,0xE4,0x84,0xBA,0x51,0x6D,0x05,0x3A,0x51,0x6D,0x05,0x3A,0xE1,0xC7,
   0x85,0x3A,0xE1,0xC7,0x85,0x3A,0xF5,0xA9,0xA1,0x39,0xF5,0xA9,0xA1,0x39,0xD6,0xF7,
   0x25,0xBA,0xD6,0xF7,0x25,0xBA,0x88,0x38,0x5B,0xBA,0x88,0x38,0x5B,0xBA,0xC8,0x31,
   0x05,0x39,0xC8,0x31,0x05,0x39,0x62,0xA0,0xF8,0x39,0x62,0xA0,0xF8,0x39,0x62,0xE5,
   0x39,0xBA,0x62,0xE5,0x39,0xBA,0x8A,0xAA,0x8B,0xBA,0x8A,0xAA,0x8B,0xBA,0xDC,0xF9,
   0x0B,0xBA,0xDC,0xF9,0x0B,0xBA,0xEA,0x27,0x97,0xBA,0xEA,0x27,0x97,0xBA,0xA1,0x5E,
   0xDA,0xBA,0xA1,0x5E,0xDA,0xBA,0x96,0x5B,0x56,0xBA,0x96,0x5B,0x56,0xBA,0x13,0xB9,
   0xA6,0xB8,0x13,0xB9,0xA6,0xB8,0x3B,0xFD,0x39,0xB7,0x3B,0xFD,0x39,0xB7,0x05,0x39,
   0x37,0xB9,0x05,0x39,0x37,0xB9,0x7D,0xED,0x18,0xBA,0x7D,0xED,0x18,0xBA,0x73,0xDD,
   0x8A,0xBA,0x73,0xDD,0x8A,0xBA,0xD1,0x83,0xD4,0xBA,0xD1,0x83,0xD4,0xBA,0xFF,0x94,
   0x0C,0xBB,0xFF,0x94,0x0C,0xBB,0xAF,0x13,0xF6,0xBA,0xAF,0x13,0xF6,0xBA,0x5A,0x27,
   0x4A,0xBA,0x5A,0x27,0x4A,0xBA,0x68,0xA7,0xA1,0x39,0x68,0xA7,0xA1,0x39,0xBE,0x9B,
   0x32,0x3A,0xBE,0x9B,0x32,0x3A,0xFD,0x21,0x0C,0x3A,0xFD,0x21,0x0C,0x3A,0x36,0xFF,
   0x21,0x3A,0x36,0xFF,0x21,0x3A,0xB5,0xA2,0x36,0x3A,0xB5,0xA2,0x36,0x3A,0xBB,0x73,
   0xE2,0xB9,0xBB,0x73,0xE2,0xB9,0x16,0xDA,0x16,0xBB,0x16,0xDA,0x16,0xBB,0x41,0x70,
   0x23,0xBB,0x41,0x70,0x23,0xBB,0xA9,0xC6,0x34,0xBA,0xA9,0xC6,0x34,0xBA,0x78,0x88,
   0x35,0x37,0x78,0x88,0x35,0x37,0xFB,0x4C,0xA9,0xBA,0xFB,0x4C,0xA9,0xBA,0xDA,0xAA,
```

Since the array data fully repeats the file one, the resulting arrays are quite large. This is why I entered the names of the created arrays in
advance to quickly jump to the beginning of each one in the listing using

**Ctrl+F**.

Now it only remains to add the required number of audio data arrays to the file listing. I have already created several test sounds. Since the
file is large, there is no point in displaying its listing here. You can find it in the library files attached below.

**Create the file with bitmap data named DataIMG.mqh** in exactly the same way. The file already creates two arrays depicting
a two-color LED bulb: data with the image of the green LED in one and with the image of the red LED in another:

```
//+------------------------------------------------------------------+
//|                                                      DataIMG.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Images                                                           |
//+------------------------------------------------------------------+
/*
   img_array_spot_green             // Green LED 16x16, 32 bit
   img_array_spot_red               // Red LED 16x16, 32 bit
*/
//+------------------------------------------------------------------+
//| Green LED 32 bit, alpha                                          |
//+------------------------------------------------------------------+
unsigned char img_array_spot_green[]=
  {
   0x42,0x4D,0x38,0x04,0x00,0x00,0x00,0x00,0x00,0x00,0x36,0x00,0x00,0x00,0x28,0x00,
   0x00,0x00,0x10,0x00,0x00,0x00,0x10,0x00,0x00,0x00,0x01,0x00,0x20,0x00,0x00,0x00,
   0x00,0x00,0x02,0x04,0x00,0x00,0xC3,0x0E,0x00,0x00,0xC3,0x0E,0x00,0x00,0x00,0x00,
   0x00,0x00,0x00,0x00,0x00,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,
   0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,
   0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,
   0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,
   0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,
   0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,
```

...

```
   0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,
   0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,
   0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,
   0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,
   0xFF,0x00,0xFF,0xFF,0xFF,0x00,0x00,0x00
  };
//+------------------------------------------------------------------+
//| Red LED 32 bit, alpha                                            |
//+------------------------------------------------------------------+
unsigned char img_array_spot_red[]=
  {
   0x42,0x4D,0x38,0x04,0x00,0x00,0x00,0x00,0x00,0x00,0x36,0x00,0x00,0x00,0x28,0x00,
   0x00,0x00,0x10,0x00,0x00,0x00,0x10,0x00,0x00,0x00,0x01,0x00,0x20,0x00,0x00,0x00,
   0x00,0x00,0x02,0x04,0x00,0x00,0xC3,0x0E,0x00,0x00,0xC3,0x0E,0x00,0x00,0x00,0x00,
   0x00,0x00,0x00,0x00,0x00,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,
   0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,
   0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,
   0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,
   0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,0xFF,0xFF,
```

Like in the example with the audio data, I do not provide the full listing of the resulting file here.

**Include files with data to the Defines.mqh**
**file so that the binary data of files are available in the library:**

```
//+------------------------------------------------------------------+
//|                                                      Defines.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "DataSND.mqh"
#include "DataIMG.mqh"
#include "Datas.mqh"
#ifdef __MQL4__
#include "ToMQL4.mqh"
#endif
//+------------------------------------------------------------------+
```

In the Defines.mqh file's macro substitution block, **add the macro**
**substitution specifying the library resource data location folder:**

```
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
//--- Describe the function with the error line number
#define DFUN_ERR_LINE                  (__FUNCTION__+(TerminalInfoString(TERMINAL_LANGUAGE)=="Russian" ? ", Page " : ", Line ")+(string)__LINE__+": ")
#define DFUN                           (__FUNCTION__+": ")        // "Function description"
#define COUNTRY_LANG                   ("Russian")                // Country language
#define END_TIME                       (D'31.12.3000 23:59:59')   // End date for account history data requests
#define TIMER_FREQUENCY                (16)                       // Minimal frequency of the library timer in milliseconds
//--- Parameters of the orders and deals collection timer
#define COLLECTION_ORD_PAUSE           (250)                      // Orders and deals collection timer pause in milliseconds
#define COLLECTION_ORD_COUNTER_STEP    (16)                       // Increment of the orders and deals collection timer counter
#define COLLECTION_ORD_COUNTER_ID      (1)                        // Orders and deals collection timer counter ID
//--- Parameters of the account collection timer
#define COLLECTION_ACC_PAUSE           (1000)                     // Account collection timer pause in milliseconds
#define COLLECTION_ACC_COUNTER_STEP    (16)                       // Account timer counter increment
#define COLLECTION_ACC_COUNTER_ID      (2)                        // Account timer counter ID
//--- Symbol collection timer 1 parameters
#define COLLECTION_SYM_PAUSE1          (100)                      // Pause of the symbol collection timer 1 in milliseconds (for scanning market watch symbols)
#define COLLECTION_SYM_COUNTER_STEP1   (16)                       // Increment of the symbol timer 1 counter
#define COLLECTION_SYM_COUNTER_ID1     (3)                        // Symbol timer 1 counter ID
//--- Symbol collection timer 2 parameters
#define COLLECTION_SYM_PAUSE2          (300)                      // Pause of the symbol collection timer 2 in milliseconds (for events of the market watch symbol list)
#define COLLECTION_SYM_COUNTER_STEP2   (16)                       // Increment of the symbol timer 2 counter
#define COLLECTION_SYM_COUNTER_ID2     (4)                        // Symbol timer 2 counter ID
//--- Collection list IDs
#define COLLECTION_HISTORY_ID          (0x7779)                   // Historical collection list ID
#define COLLECTION_MARKET_ID           (0x777A)                   // Market collection list ID
#define COLLECTION_EVENTS_ID           (0x777B)                   // Event collection list ID
#define COLLECTION_ACCOUNT_ID          (0x777C)                   // Account collection list ID
#define COLLECTION_SYMBOLS_ID          (0x777D)                   // Symbol collection list ID
//--- Data parameters for file operations
#define DIRECTORY                      ("DoEasy\\")               // Library directory for storing object folders
#define RESOURCE_DIR                   ("DoEasy\\Resource\\")     // Library directory for storing resource folders
//--- Symbol parameters
#define CLR_DEFAULT                    (0xFF000000)               // Default color
#define SYMBOLS_COMMON_TOTAL           (1000)                     // Total number of working symbols
//+------------------------------------------------------------------+
```

Inside the **Resource\** library subfolder, **Sounds** and **Images** folders are automatically generated for creating
and storing audio and image files, respectively.

When creating files out of prepared arrays, we need to specify their extensions. In order to let the file generation method know what file is
exactly created and what folder it should be placed in, we need the enumeration of file types whose data are written to binary arrays

**.**

**Add the necessary enumeration at the very end of the**
**Defines.mqh listing:**

```
//+------------------------------------------------------------------+
//| Data for working with program resource data                      |
//+------------------------------------------------------------------+
enum ENUM_FILE_TYPE
  {
   FILE_TYPE_WAV,                                           // wav file
   FILE_TYPE_BMP,                                           // bmp file
  };
//+------------------------------------------------------------------+
```

Since all library resource files are located in Sounds and Images folders in MQL5\ **Files**\ of the terminal, we need to
correct the PlaySound() method of the

**CMessage** class. Open \\MQL5\\Include\\DoEasy\ **Services\\Message.mqh** and **adjust**
**the file path in the PlaySound() method:**

```
//+------------------------------------------------------------------+
//| Play an audio file                                               |
//+------------------------------------------------------------------+
bool CMessage::PlaySound(const string file_name)
  {
   bool res=::PlaySound("\\Files\\"+file_name);
   CMessage::m_global_error=(res ? ERR_SUCCESS : ::GetLastError());
   return res;
  }
//+------------------------------------------------------------------+
```

To play the file, we specify the **\\Files\** subfolder since we are to store all data relative to MQL5\ and this folder, while
the remaining file path is set and passed to the method using the

file\_name parameter when creating the file description object.

**Currently, this is enough for creating the necessary classes.**

In \\MQL5\\Include\\DoEasy\ **Services**\\, create the **CFileGen** new class in the **FileGen.mqh** file:

```
//+------------------------------------------------------------------+
//|                                                      FileGen.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\\Services\DELib.mqh"
//+------------------------------------------------------------------+
//| File generator class                                             |
//+------------------------------------------------------------------+
class CFileGen
  {
private:
   static string     m_folder_name;    // Name of a folder library resource files are stored in
   static string     m_subfolder;      // Name of a subfolder storing audio or bitmap files
   static string     m_name;           // File name
   static int        m_handle;         // File handle
//--- Set a (1) file, (2) subfolder name
   static void       SetName(const ENUM_FILE_TYPE file_type,const string file_name);
   static void       SetSubFolder(const ENUM_FILE_TYPE file_type);
//--- Return file extension by its type
   static string     Extension(const ENUM_FILE_TYPE file_type);
public:
//--- Return the (1) set name, (2) the flag of a file presence in the resource directory
   static string     Name(void)  { return CFileGen::m_name; }
   static bool       IsExist(const ENUM_FILE_TYPE file_type,const string file_name);
//--- Create a file out of the data array
   static bool       Create(const ENUM_FILE_TYPE file_type,const string file_name,const uchar &file_data_array[]);
  };
//+------------------------------------------------------------------+
```

The file immediately includes the DELib.mqh library of service functions,
since Defines.mqh and Message.mqh necessary for the class operation are already included in it:

```
//+------------------------------------------------------------------+
//|                                                        DELib.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property strict  // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\Defines.mqh"
#include "Message.mqh"
#include "TimerCounter.mqh"
//+------------------------------------------------------------------+
```

All class member variables and its methods have headers in the code, and there is no point in describing their purpose — everything is clear
there. Let's consider the implementation of methods.

**Since [class member variables are static](https://www.mql5.com/en/docs/basis/oop/staticmembers), they**
**require explicit initialization:**

```
//+------------------------------------------------------------------+
//| Initialization of static variables                               |
//+------------------------------------------------------------------+
string CFileGen::m_folder_name=RESOURCE_DIR;
string CFileGen::m_subfolder="\\";
string CFileGen::m_name=NULL;
int    CFileGen::m_handle=INVALID_HANDLE;
//+------------------------------------------------------------------+
```

**Method of creating a file out of a data array:**

```
//+------------------------------------------------------------------+
//| Create a file out of a data array                                |
//+------------------------------------------------------------------+
bool CFileGen::Create(const ENUM_FILE_TYPE file_type,const string file_name,const uchar &file_data_array[])
  {
   //--- Set a file name consisting of the file path, its name and extension
   CFileGen::SetName(file_type,file_name);
   //--- If such a file already exists, return 'false'
   if(::FileIsExist(CFileGen::m_name))
      return false;
   //--- Open the file with the generated name for writing
   CFileGen::m_handle=::FileOpen(CFileGen::m_name,FILE_WRITE|FILE_BIN);
   //--- If failed to create the file, receive an error code, display the file opening error message and return 'false'
   if(CFileGen::m_handle==INVALID_HANDLE)
     {
      int err=::GetLastError();
      ::Print(CMessage::Text(MSG_LIB_SYS_FAILED_OPEN_FILE_FOR_WRITE),"\"",CFileGen::m_name,"\". ",CMessage::Text(MSG_LIB_SYS_ERROR),"\"",CMessage::Text(err),"\" ",CMessage::Retcode(err));
      return false;
     }
   //--- Write the contents of the file_data_array[] array, close the file and return 'true'

   ::FileWriteArray(CFileGen::m_handle,file_data_array);
   ::FileClose(CFileGen::m_handle);
   return true;
  }
//+------------------------------------------------------------------+
```

The method uses the [FileWriteArray()](https://www.mql5.com/en/docs/files/filewritearray) file standard function
allowing to write the arrays of any data except string one to a binary file.

The method receives the type of a written file (audio or
image),

a name of a future file and the array
with a set of binary data of a created file.

All actions performed by the method are written in its listing — they are simple
and comprehensible. There is no need to dwell on them.

**The method returning the flag of a file presence in the resource directory:**

```
//+------------------------------------------------------------------+
//| The flag of a file presence in the resource directory            |
//+------------------------------------------------------------------+
bool CFileGen::IsExist(const ENUM_FILE_TYPE file_type,const string file_name)
  {
   CFileGen::SetName(file_type,file_name);
   return ::FileIsExist(CFileGen::m_name);
  }
//+------------------------------------------------------------------+
```

The method receives the type of a written file and the name
of a file whose presence should be checked. Then the file name consisting of the file path, name and extension is set, and the result of
the same-name file presence check is returned using the

[FileIsExist()](https://www.mql5.com/en/docs/files/fileisexist) function.

**The method setting a file name consisting of the file path, its name and extension:**

```
//+------------------------------------------------------------------+
//| Set a file name                                                  |
//+------------------------------------------------------------------+
void CFileGen::SetName(const ENUM_FILE_TYPE file_type,const string file_name)
  {
   CFileGen::SetSubFolder(file_type);
   CFileGen::m_name=CFileGen::m_folder_name+CFileGen::m_subfolder+file_name+CFileGen::Extension(file_type);
  }
//+------------------------------------------------------------------+
```

The method receives the written file type and a file
name, out of which the method assembles the complete file name including the
folder for storing library files, file extension subfolder
created by the

**SetSubFolder()** method passed to the name
(

**file\_name**) method and the file extension created
by the

**Extension()** method by the file
type (audio or image). The obtained result is written to the **m\_name**
class member variable.

**The method for setting the subfolder name depending on the file type (audio or image):**

```
//+------------------------------------------------------------------+
//| Set a subfolder name                                             |
//+------------------------------------------------------------------+
void CFileGen::SetSubFolder(const ENUM_FILE_TYPE file_type)
  {
   CFileGen::m_subfolder=(file_type==FILE_TYPE_BMP ? "Images\\" : file_type==FILE_TYPE_WAV ? "Sounds\\" : "");
  }
//+------------------------------------------------------------------+
```

The method receives a file type. Based on that type, the **m\_subfolder** class member variable receives the **Images\** or **Sounds\**
subfolder name.

The method returning a file extension by its type:

```
//+------------------------------------------------------------------+
//| Return file extension by its type                                |
//+------------------------------------------------------------------+
string CFileGen::Extension(const ENUM_FILE_TYPE file_type)
  {
   string ext=::StringSubstr(::EnumToString(file_type),10);
   if(!::StringToLower(ext))
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_CONV_TO_LOWERCASE),CMessage::Retcode(::GetLastError()));
   return "."+ext;
  }
//+------------------------------------------------------------------+
```

The method receives a file type. Next, from the text
representation (

[EnumToString()](https://www.mql5.com/en/docs/convert/enumtostring)) of the ENUM\_FILE\_TYPE enumeration constant passed
to the method,

retrieve the extension using the [StringSubstr()](https://www.mql5.com/en/docs/strings/stringsubstr)
function (for an audio file, the "WAV" substring is retrieved from the "FILE\_TYPE\_WAV" string, while in case of a bitmap file, it is "BMP"). Then all
symbols of a retrieved substring with the file extension are

converted to string ones using [StringToLower()](https://www.mql5.com/en/docs/strings/stringtolower),

the "full stop" character (.) is added to the string (in front of it)
and the obtained result is returned from the method.

**This concludes creation of the file generator class.**

Now we are able to store binary data of any audio and bitmap files in the source code and create full-fledged files out of this data
automatically during the program launch. The files are to be located in the appropriate folders the library is going to access to get the
files and use them for their intended purpose.

To get them conveniently, we need to somehow describe the existing files and have quick and convenient access to them. To do this, we will
create the program resource collection class. This will not be a collection we are going to create in the library (lists of pointers to
collection objects), but rather a collection of file description objects.

For each created physical file, we are going to generate an object describing this file where its name, path and description are to be
specified. We will use these descriptor objects to access physical files. A file description may consist of any text description we are
going to add to a descriptor object for each specific file.

For example, the sound of a mouse click may be described as "Mouse click 01", "Click 01" or anything you like. In order to get a descriptor of
a necessary file, we simply enter the description to the search parameters. The search method returns the file descriptor object index we
can use to get the properties of the physical file.

### Program resource collection class

In the \\MQL5\\Include\\DoEasy\ **Collections\** library folder, create the new class **CResourceCollection** in the **ResourceCollection.mqh**
file. In the new class listing, write yet another class — the descriptor object class whose instances are to be created for each new file and added to
the descriptor collection:

```
//+------------------------------------------------------------------+
//|                                           ResourceCollection.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Arrays\ArrayObj.mqh>
#include "..\\Services\\FileGen.mqh"
//+------------------------------------------------------------------+
//| Descriptor object class for the library resource file            |
//+------------------------------------------------------------------+
class CResObj : public CObject
  {
private:
   string            m_file_name;      // Path + file name + extension
   string            m_description;    // File text description
public:
//--- Set (1) file name, (2) file description
   void              FileName(const string name)                  { this.m_file_name=name;      }
   void              Description(const string descr)              { this.m_description=descr;   }
//--- Return (1) file name, (2) file description
   string            FileName(void)                         const { return this.m_file_name;    }
   string            Description(void)                      const { return this.m_description;  }
//--- Compare CResObj objects by properties (to search for equal resource objects)
   bool              IsEqual(const CResObj* compared_obj)   const { return this.Compare(compared_obj,0)==0; }
//--- Compare CResObj objects by all properties (for sorting)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Constructor
                     CResObj(void){;}
  };
//+------------------------------------------------------------------+
//| Compare CResObj objects                                          |
//+------------------------------------------------------------------+
int CResObj::Compare(const CObject *node,const int mode=0) const
  {
   const CResObj *obj_compared=node;
   if(mode==0)
      return(this.m_file_name>obj_compared.m_file_name ? 1 : this.m_file_name<obj_compared.m_file_name ? -1 : 0);
   else
      return(this.m_description>obj_compared.m_description ? 1 : this.m_description<obj_compared.m_description ? -1 : 0);
  }
//+------------------------------------------------------------------+
```

Include the necessary files to the listing right away — [the \\
class of the](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) [dynamic \\
array of pointers to CObject class instances](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj) and its descendantsof the standard library and the class
generating CFileGen library files.

To allow the descriptor object to be stored in the CArrayObj list, we need to make this object a descendant [of the CObject standard library base class](https://www.mql5.com/en/docs/standardlibrary/cobject).

All class member variables and methods are described in the code comments. There is no
point in dwelling on a detailed analysis of their purpose.

Please note that the Compare() virtual method compares the
fields of two descriptor objects by a file name by default. If the compared fields of two objects are equal, the method returns zero. If
objects should be compared by a file description,

mode should be set to 1.

The IsEqual()
method returning the object equality flag compares objects only by file name (since there cannot be two files with similar names in
one folder). By default, the method returns the Compare() method operation result with 'mode' (0), which corresponds to comparing by a file
name.

**Let's implement the collection class of descriptor objects describing program resource files:**

```
//+------------------------------------------------------------------+
//| Collection class of resource files descriptor objects            |
//+------------------------------------------------------------------+
class CResourceCollection
  {
private:
//--- List of pointers to descriptor objects
   CArrayObj         m_list_dscr_obj;
//--- Create a file descriptor object and add it to the list
   bool              CreateFileDescrObj(const string file_name,const string descript);
//--- Add a new object to the list of descriptor objects
   bool              AddToList(CResObj* element);
public:
//--- Create a file and add its description to the list
   bool              CreateFile(const ENUM_FILE_TYPE file_type,const string file_name,const string descript,const uchar &file_data_array[]);
//--- Return the (1) list of pointers to descriptor objects, (2) index of the file descriptor object by description
   CArrayObj        *GetList(void)  { return &this.m_list_dscr_obj;  }
   int               GetIndexResObjByDescription(const string file_description);
//--- Constructor
                     CResourceCollection(void);
  };
//+------------------------------------------------------------------+
```

The purpose of each method is set here in the listing as well, therefore let's move on to the class methods.

**Write the class constructor outside the class body:**

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CResourceCollection::CResourceCollection()
  {
   this.m_list_dscr_obj.Clear();
   this.m_list_dscr_obj.Sort();
  }
//+------------------------------------------------------------------+
```

Here the list of descriptor objects is cleared and the
flag of sorting by file name is set for the list (the default is 0).

**The method of creating a file and adding a descriptor object to the collection list:**

```
//+------------------------------------------------------------------+
//| Create a file and add its descriptor object to the list          |
//+------------------------------------------------------------------+
bool CResourceCollection::CreateFile(const ENUM_FILE_TYPE file_type,const string file_name,const string descript,const uchar &file_data_array[])
  {
   if(!CFileGen::Create(file_type,file_name,file_data_array))
     {
      if(!::FileIsExist(CFileGen::Name()))
         return false;
     }
   return this.CreateFileDescrObj(file_type,CFileGen::Name(),descript);
  }
//+------------------------------------------------------------------+
```

The method receives the type of a created file (audio or
image),

file name, file
description and a link to the binary array of file data
the file is to consist of.

If failed to create the file using the Create() method of the CFileGen class
and the file is actually absent (it may already exist and cannot be
re-created because of that),

return false.


If the file has never been there and has just been created,

return the operation result of the method generating a descriptor object and adding it
to the collection list of file descriptor objects from the method.

**The method of creating a file descriptor object and adding it to the collection list:**

```
//+------------------------------------------------------------------+
//| Create a file descriptor object and add it to the list           |
//+------------------------------------------------------------------+
bool CResourceCollection::CreateFileDescrObj(const string file_name,const string descript)
  {
   CResObj *res_dscr=new CResObj();
   if(res_dscr==NULL)
     {
      Print(DFUN,CMessage::Text(MSG_LIB_SYS_FAILED_CREATE_RES_LINK));
      return false;
     }
   res_dscr.FileName(file_name);
   res_dscr.Description(descript);
   if(!this.AddToList(res_dscr))
     {
      delete res_dscr;
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

The method receives the file name and its description.

Create the new file descriptor object. If
failed to create the object, display the appropriate message to the journal
and return

false.

Next, set a
file name for the newly created descriptor object, as well as the file
description and add it to the collection list. If failed
to add the object, it should be destroyed to avoid a memory leak —

remove an object and return

false.

Otherwise, return
true
— the object has been successfully created and added to the collection.

**The method adding a new descriptor object to the collection list:**

```
//+------------------------------------------------------------------+
//| Add a new object to the list of file descriptor objects          |
//+------------------------------------------------------------------+
bool CResourceCollection::AddToList(CResObj *element)
  {
   this.m_list_dscr_obj.Sort();
   if(this.m_list_dscr_obj.Search(element)>WRONG_VALUE)
      return false;
   return this.m_list_dscr_obj.Add(element);
  }
//+------------------------------------------------------------------+
```

The method receives a descriptor object and the
sorting flag by a file name is set for the collection list. If the [object \\
with the same name is already present in the list](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjsearch), return false.


Otherwise,

return the operation result of the [method \\
for adding an object to the list](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjadd).

**The method returning the descriptor object index in the collection list by the file description:**

```
//+----------------------------------------------------------------------+
//| Return the index of the file descriptor object by a file description |
//+----------------------------------------------------------------------+
int CResourceCollection::GetIndexResObjByDescription(const string file_description)
  {
   CResObj *obj=new CResObj();
   if(obj==NULL)
      return WRONG_VALUE;
   obj.Description(file_description);
   this.m_list_dscr_obj.Sort(1);
   int index=this.m_list_dscr_obj.Search(obj);
   delete obj;
   return index;
  }
//+------------------------------------------------------------------+
```

The method receives the file description to be used to find a
descriptor object.

Create a temporary instance of a descriptor object and set
the file description passed to the method in its description field.

Set
the sorting flag by the file description to the collection list (

**1**) and [get](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjsearch)
the descriptor object index whose description field contains the necessary text.

Make
sure to remove the temporary object and return the obtained index
(-1 in case there is no object with this description in the collection list).

**The class of file descriptor object collection is ready.**

Now we need to add a few methods to the [**CEngine** library base \\
object](https://www.mql5.com/en/articles/5687#node02).

Open the file \\MQL5\\Include\\DoEasy\ **Engine.mqh** and **include the**
**descriptor object collection file to it:**

```
//+------------------------------------------------------------------+
//|                                                       Engine.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Services\TimerCounter.mqh"
#include "Collections\HistoryCollection.mqh"
#include "Collections\MarketCollection.mqh"
#include "Collections\EventsCollection.mqh"
#include "Collections\AccountsCollection.mqh"
#include "Collections\SymbolsCollection.mqh"
#include "Collections\ResourceCollection.mqh"
//+------------------------------------------------------------------+
```

**Create the new collection object for application resources (collection of file descriptors):**

```
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine : public CObject
  {
private:
   CHistoryCollection   m_history;                       // Collection of historical orders and deals
   CMarketCollection    m_market;                        // Collection of market orders and deals
   CEventsCollection    m_events;                        // Event collection
   CAccountsCollection  m_accounts;                      // Account collection
   CSymbolsCollection   m_symbols;                       // Symbol collection
   CResourceCollection  m_resource;                      // Resource list
   CArrayObj            m_list_counters;                 // List of timer counters
   int                  m_global_error;                  // Global error code
   bool                 m_first_start;                   // First launch flag
   bool                 m_is_hedge;                      // Hedge account flag
   bool                 m_is_tester;                     // Flag of working in the tester
   bool                 m_is_market_trade_event;         // Account trading event flag
   bool                 m_is_history_trade_event;        // Account history trading event flag
   bool                 m_is_account_event;              // Account change event flag
   bool                 m_is_symbol_event;               // Symbol change event flag
   ENUM_TRADE_EVENT     m_last_trade_event;              // Last account trading event
   int                  m_last_account_event;            // Last event in the account properties
   int                  m_last_symbol_event;             // Last event in the symbol properties
//--- Return the counter index by id
```

and **add three new methods for working with program resource collection:**

```
//--- Timer
   void                 OnTimer(void);
//--- Set the list of used symbols
   bool                 SetUsedSymbols(const string &array_symbols[])   { return this.m_symbols.SetUsedSymbols(array_symbols);}

//--- Create a resource file
   bool                 CreateFile(const ENUM_FILE_TYPE file_type,const string file_name,const string descript,const uchar &file_data_array[])
                          {
                           return this.m_resource.CreateFile(file_type,file_name,descript,file_data_array);
                          }
//--- Return the list of links to resources
   CArrayObj           *GetListResource(void)                                 { return this.m_resource.GetList();                               }
   int                  GetIndexResObjByDescription(const string file_name)   { return this.m_resource.GetIndexResObjByDescription(file_name);  }

//--- Return event (1) milliseconds, (2) reason and (3) source from its 'long' value
   ushort               EventMSC(const long lparam)               const { return this.LongToUshortFromByte(lparam,0);         }
   ushort               EventReason(const long lparam)            const { return this.LongToUshortFromByte(lparam,1);         }
   ushort               EventSource(const long lparam)            const { return this.LongToUshortFromByte(lparam,2);         }

//--- Constructor/destructor
                        CEngine();
                       ~CEngine();
  };
//+------------------------------------------------------------------+
```

The methods have the same names as the methods in the collection class of descriptor objects and call the methods of the same name from this
class.

In order to test the created classes, create all files according to the available data in the DataSND.mqh and DataIMG.mqh include files in
binary arrays. In the Experts journal, display the results of creating files from binary arrays and display the contents of the resulting
collection list of file descriptor objects. Also, let's play one of the sounds and display the image consisting of the two created image
files depicting red and green LEDs in the lower right corner of the screen.

### Testing access to automatically created files

Let's use the TestDoEasyPart19.mq5 EA [from the previous article](https://www.mql5.com/en/articles/7176) and
save it to \\MQL5\\Experts\\TestDoEasy\

**Part20\** under the name **TestDoEasyPart20.mq5**.

Since files should be created during the program's first launch, the classes for creating the program resources should be handled in OnInit().
The obtained result should be tested there as well.

**Add the following code block to the very end of the OnInit()**
**handler:**

```
//--- Set CTrade trading class parameters
#ifdef __MQL5__
   trade.SetDeviationInPoints(slippage);
   trade.SetExpertMagicNumber(magic_number);
   trade.SetTypeFillingBySymbol(Symbol());
   trade.SetMarginMode();
   trade.LogLevel(LOG_LEVEL_NO);
#endif
//--- Create and check the resource files
   Print("\n",TextByLanguage("--- Проверка успешности создания файлов ---","--- Verifying files were created ---"));
   string dscr=TextByLanguage("Проверка существования файла: ","Checking existence of file: ");

   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_01",TextByLanguage("Звук упавшей монетки 1","Sound of falling coin 1"),sound_array_coin_01);
   if(CFileGen::IsExist(FILE_TYPE_WAV,"sound_array_coin_01"))
      Print(dscr+CFileGen::Name(),": OK");
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_02",TextByLanguage("Звук упавших монеток","Sound fallen coins"),sound_array_coin_02);
   if(CFileGen::IsExist(FILE_TYPE_WAV,"sound_array_coin_02"))
      Print(dscr+CFileGen::Name(),": OK");
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_03",TextByLanguage("Звук монеток","Sound of coins"),sound_array_coin_03);
   if(CFileGen::IsExist(FILE_TYPE_WAV,"sound_array_coin_03"))
      Print(dscr+CFileGen::Name(),": OK");
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_04",TextByLanguage("Звук упавшей монетки 2","Sound of falling coin 2"),sound_array_coin_04);
   if(CFileGen::IsExist(FILE_TYPE_WAV,"sound_array_coin_04"))
      Print(dscr+CFileGen::Name(),": OK");
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_01",TextByLanguage("Звук щелчка по кнопке 1","Click on button sound 1"),sound_array_click_01);
   if(CFileGen::IsExist(FILE_TYPE_WAV,"sound_array_click_01"))
      Print(dscr+CFileGen::Name(),": OK");
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_02",TextByLanguage("Звук щелчка по кнопке 2","Click on button sound 1"),sound_array_click_02);
   if(CFileGen::IsExist(FILE_TYPE_WAV,"sound_array_click_02"))
      Print(dscr+CFileGen::Name(),": OK");
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_03",TextByLanguage("Звук щелчка по кнопке 3","Click on button sound 1"),sound_array_click_03);
   if(CFileGen::IsExist(FILE_TYPE_WAV,"sound_array_click_03"))
      Print(dscr+CFileGen::Name(),": OK");
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_cash_machine_01",TextByLanguage("Звук кассового аппарата","Sound of cash machine"),sound_array_cash_machine_01);
   if(CFileGen::IsExist(FILE_TYPE_WAV,"sound_array_cash_machine_01"))
      Print(dscr+CFileGen::Name(),": OK");

   engine.CreateFile(FILE_TYPE_BMP,"img_array_spot_green",TextByLanguage("Изображение \"Зелёный светодиод\"","Image \"Green Spot lamp\""),img_array_spot_green);
   if(CFileGen::IsExist(FILE_TYPE_BMP,"img_array_spot_green"))
      Print(dscr+CFileGen::Name(),": OK");
   engine.CreateFile(FILE_TYPE_BMP,"img_array_spot_red",TextByLanguage("Изображение \"Красный светодиод\"","Image \"Red Spot lamp\""),img_array_spot_red);
   if(CFileGen::IsExist(FILE_TYPE_BMP,"img_array_spot_red"))
      Print(dscr+CFileGen::Name(),": OK");

//--- Check the file description list
   Print("\n",TextByLanguage("--- Проверка списка описания файлов ---","--- Checking file description list ---"));
   CArrayObj* list_res=engine.GetListResource();
   if(list_res!=NULL)
     {
      //--- Let's see the entire list of file descriptions
      for(int i=0;i<list_res.Total();i++)
        {
         CResObj *res=list_res.At(i);
         if(res==NULL)
            continue;
         //--- Display the paths to the files and the file description in the journal
         string type=(StringFind(res.FileName(),"\\Sounds\\")>0 ? TextByLanguage("Звук ","Sound ") : TextByLanguage("Изображение ","Image "));
         Print(type,string(i+1),": ",TextByLanguage("Имя файла :","File name: "),res.FileName()," (",res.Description(),")");

         //--- If the current description corresponds to the falling coin sound 1, play the appropriate sound
         if(res.Description()==TextByLanguage("Звук упавшей монетки 1","Sound of falling coin 1"))
           {
            CMessage::PlaySound(res.FileName());
           }
        }
      //--- Create the image of the red-green LED
      //--- Get the indices of red and green LEDs image descriptions
      int index_r=engine.GetIndexResObjByDescription(TextByLanguage("Изображение \"Красный светодиод\"","Image \"Red Spot lamp\""));
      int index_g=engine.GetIndexResObjByDescription(TextByLanguage("Изображение \"Зелёный светодиод\"","Image \"Green Spot lamp\""));
      if(index_g>WRONG_VALUE && index_r>WRONG_VALUE)
        {
         //--- Get two objects with files description from the list
         CResObj *res_g=list_res.At(index_g);
         CResObj *res_r=list_res.At(index_r);
         if(res_g==NULL || res_r==NULL)
           {
            Print(TextByLanguage("Не удалось получить данные с описанием файла изображения","Failed to get image file description data"));
            return(INIT_SUCCEEDED);
           }
         //--- Create a button based on image files
         long chart_ID=ChartID();
         string name=prefix+"RedGreenSpot";
         if(ObjectCreate(chart_ID,name,OBJ_BITMAP_LABEL,0,0,0))
           {
            ObjectSetString(chart_ID,name,OBJPROP_BMPFILE,0,"\\Files\\"+res_g.FileName());   // Изображение для нажатой кнопки
            ObjectSetString(chart_ID,name,OBJPROP_BMPFILE,1,"\\Files\\"+res_r.FileName());   // Released button image
            ObjectSetInteger(chart_ID,name,OBJPROP_CORNER,CORNER_RIGHT_LOWER);
            ObjectSetInteger(chart_ID,name,OBJPROP_ANCHOR,ANCHOR_RIGHT_LOWER);
            ObjectSetInteger(chart_ID,name,OBJPROP_STATE,true);
            ObjectSetInteger(chart_ID,name,OBJPROP_XSIZE,16);
            ObjectSetInteger(chart_ID,name,OBJPROP_YSIZE,16);
            ObjectSetInteger(chart_ID,name,OBJPROP_XOFFSET,0);
            ObjectSetInteger(chart_ID,name,OBJPROP_YOFFSET,0);
            ObjectSetInteger(chart_ID,name,OBJPROP_BACK,false);
            ObjectSetInteger(chart_ID,name,OBJPROP_TIMEFRAMES,OBJ_ALL_PERIODS);
            ChartRedraw(chart_ID);
           }
        }
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

Here we arranged creating files out of data located in binary arrays of the library source codes. After creating each file, check its existence
and display the result in the journal. After creating all files and their descriptors, check the full list of all file descriptors added to
the collection when creating the files.

The listing describes all the actions. You can analyze them on your own.

After compiling the EA, it displays file creation results in the journal, plays the falling coin sound and displays the LED picture consisting
of two images in the lower right corner of the screen. You can switch images by clicking on the LED. In fact, this is a button having two states
(on/off).

![](https://c.mql5.com/2/37/QbpSy0Nylk.gif)

![](https://c.mql5.com/2/37/tR9XlcS9DJ.gif)

As we can see, everything works as it is supposed to. The messages about successful file generation appear in the journal, the LED changes its
color when clicking on it, and if we open the terminal data folder (File --> Open Data Folder) and enter MQL5\\Files\\DoEasy\\Resource\\, we
can see the Images and Sounds subfolders where all the newly created files are located.

### What's next?

Starting from the next article, we open the new library section — trading classes and everything related to them.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7195#node00)

**Previous articles within the series:**

[Part 1. Concept, data management](https://www.mql5.com/en/articles/5654)

[Part \\
2\. Collection of historical orders and deals](https://www.mql5.com/en/articles/5669)

[Part 3. Collection of market orders \\
and positions, arranging the search](https://www.mql5.com/en/articles/5687)

[Part 4. Trading events. Concept](https://www.mql5.com/en/articles/5724)

[Part 5. Classes and collection of trading events. Sending events to the program](https://www.mql5.com/en/articles/6211)

[Part \\
6\. Netting account events](https://www.mql5.com/en/articles/6383)

[Part 7. StopLimit order activation events, preparing \\
the functionality for order and position modification events](https://www.mql5.com/en/articles/6482)

[Part 8. Order and \\
position modification events](https://www.mql5.com/en/articles/6595)

[Part 9. Compatibility with MQL4 - Preparing data](https://www.mql5.com/en/articles/6651)

[Part 10. Compatibility with MQL4 - Events of opening a position and activating pending orders](https://www.mql5.com/en/articles/6767)

[Part 11. Compatibility with MQL4 - Position closure events](https://www.mql5.com/en/articles/6921)

[Part \\
12\. Account object class and account object collection](https://www.mql5.com/en/articles/6952)

[Part 13. Account object \\
events](https://www.mql5.com/en/articles/6995)

[Part 14. Symbol object](https://www.mql5.com/en/articles/7014)

[Part \\
15\. Symbol object collection](https://www.mql5.com/en/articles/7041)

[Part 16. Symbol collection events](https://www.mql5.com/en/articles/7071)

[Part 17. Interactivity of library objects](https://www.mql5.com/en/articles/7124)

[Part \\
18\. Interactivity of account and any other library objects](https://www.mql5.com/en/articles/7149)

[Part 19. Class of \\
library messages](https://www.mql5.com/en/articles/7176)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7195](https://www.mql5.com/ru/articles/7195)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7195.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7195/mql5.zip "Download MQL5.zip")(3556.3 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7195/mql4.zip "Download MQL4.zip")(3553.08 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/327748)**
(4)


![Edgar Akhmadeev](https://c.mql5.com/avatar/avatar_na2.png)

**[Edgar Akhmadeev](https://www.mql5.com/en/users/dali)**
\|
2 Sep 2019 at 12:23

You have 1 byte encoded with 5 characters ("0xNN,").

Base64 is used to densely pack binary data into text. I got 1.36 characters per byte in my test.

Here is an example

```
void
OnStart() {
        uchar result[], result2[];
        uchar key[] = { 0 };
        uchar data[];
        int len1 = StringToCharArray("The quick brown  fox  jumps  over  the  lazy  dog", data);

        int len2 = CryptEncode(CRYPT_BASE64, data, key, result);
        Print("len1=", len1, ", len2=", len2, ", result=", CharArrayToString(result));

        CryptDecode(CRYPT_BASE64, result, key, result2);
        Print("result2=", CharArrayToString(result2));
}
```

len1=50, len2=68, result=VGhlIHF1aWNrIGJyb3duICBmb3ggIGp1bXBzICBvdmVyICB0aGUgIGxhenkgIGRvZwA=

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
2 Sep 2019 at 13:07

**Edgar:**

You have 1 byte encoded with 5 characters ("0xNNN,").

Base64 is used to densely pack binary data into text. In my test I got 1.36 characters per byte.

Here is an example

len1=50, len2=68, result=VGhlIHF1aWNrIGJyb3duICBmb3ggIGp1bXBzICBvdmVyICB0aGUgIGxhenkgIGRvZwA=

So that's what it's about here...


![Edgar Akhmadeev](https://c.mql5.com/avatar/avatar_na2.png)

**[Edgar Akhmadeev](https://www.mql5.com/en/users/dali)**
\|
2 Sep 2019 at 13:30

**Artyom Trishkin:**

That's not what we're talking about here...

Ah, you mean that you have data for compilation and in ex5 will occupy 1:1. Yes, there is no need to pack here.

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
2 Sep 2019 at 13:33

**Edgar:**

Ah, you mean that you have data for compilation and in ex5 will occupy 1:1. Yes, there is no need to pack it here.

Later, storage in programme resources will be added - the compiler compresses the data there.


![Library for easy and quick development of MetaTrader programs (part XXI): Trading classes - Base cross-platform trading object](https://c.mql5.com/2/37/MQL5-avatar-doeasy__3.png)[Library for easy and quick development of MetaTrader programs (part XXI): Trading classes - Base cross-platform trading object](https://www.mql5.com/en/articles/7229)

In this article, we will start the development of the new library section - trading classes. Besides, we will consider the development of a unified base trading object for MetaTrader 5 and MetaTrader 4 platforms. When sending a request to the server, such a trading object implies that verified and correct trading request parameters are passed to it.

![Building an Expert Advisor using separate modules](https://c.mql5.com/2/37/mql5_avatar_adviser_modules.png)[Building an Expert Advisor using separate modules](https://www.mql5.com/en/articles/7318)

When developing indicators, Expert Advisors and scripts, developers often need to create various pieces of code, which are not directly related to the trading strategy. In this article, we consider a way to create Expert Advisors using earlier created blocks, such as trailing, filtering and scheduling code, among others. We will see the benefits of this programming approach.

![Library for easy and quick development of MetaTrader programs (part XXII): Trading classes - Base trading class, verification of limitations](https://c.mql5.com/2/37/MQL5-avatar-doeasy__4.png)[Library for easy and quick development of MetaTrader programs (part XXII): Trading classes - Base trading class, verification of limitations](https://www.mql5.com/en/articles/7258)

In this article, we will start the development of the library base trading class and add the initial verification of permissions to conduct trading operations to its first version. Besides, we will slightly expand the features and content of the base trading class.

![Library for easy and quick development of MetaTrader programs (part XIX): Class of library messages](https://c.mql5.com/2/37/MQL5-avatar-doeasy__1.png)[Library for easy and quick development of MetaTrader programs (part XIX): Class of library messages](https://www.mql5.com/en/articles/7176)

In this article, we will consider the class of displaying text messages. Currently, we have a sufficient number of different text messages. It is time to re-arrange the methods of their storage, display and translation of Russian or English messages to other languages. Besides, it would be good to introduce convenient ways of adding new languages to the library and quickly switching between them.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/7195&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062736444750997477)

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