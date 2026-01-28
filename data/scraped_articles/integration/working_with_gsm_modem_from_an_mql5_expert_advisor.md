---
title: Working with GSM Modem from an MQL5 Expert Advisor
url: https://www.mql5.com/en/articles/797
categories: Integration, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:26:49.520642
---

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/797&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068241643636717351)

MetaTrader 5 / Integration


### Introduction

There is currently a fair number of means for a comfortable remote monitoring of a trading account: mobile terminals, push notifications, working with ICQ. But it all requires Internet connection. This article describes the process of creating an Expert Advisor that will allow you to stay in touch with your trading terminal even when mobile Internet is not available, through calls and text messaging. In addition, this Expert Advisor will be able to notify you of the lost or reestablished connection with the trade server.

For this purpose, virtually any GSM modem, as well as most phones with the modem function would do. For illustration, I have chosen **Huawei E1550**, as this modem is one of the most widely used devices of its kind. Further, at the end of the article, we will try to replace the modem with an old cellphone **Siemens M55** (released in 2003) and see what happens.

But first, a few words on how to send a byte of data from an Expert Advisor to a modem.

### 1\. Working with COM Port

Upon connecting the modem to your computer and installing all the necessary drivers, you will be able to see a virtual COM port in the system. All future operations with the modem are performed via this port. Consequently, in order to exchange data with the modem, you should first get access to the COM port.

![The modem as displayed in the device manager](https://c.mql5.com/2/6/en_01__3.png)

Fig. 1. Huawei modem is connected to the COM3 port

Here, we will need a DLL library **TrComPort.dll** which is freely distributed in the Internet [together with the source files](https://www.mql5.com/go?link=http://forum.sources.ru/index.php?s=68b8efb9b3903ab2cd7b39f5029c63e1&act=Attach&type=post&id=261305&attach_id=0 "http://forum.sources.ru/index.php?s=68b8efb9b3903ab2cd7b39f5029c63e1&act=Attach&type=post&id=261305&attach_id=0"). It will be used to configure the COM port, query its state, as well as to receive and send data. In order to get that done, we will use the following functions:

```
#import "TrComPort.dll"
   int TrComPortOpen(int portnum);
   int TrComPortClose(int portid);
   int TrComPortSetConfig(int portid, TrComPortParameters& parameters);
   int TrComPortGetConfig(int portid, TrComPortParameters& parameters);
   int TrComPortWriteArray(int portid, uchar& buffer[], uint length, int timeout);
   int TrComPortReadArray(int portid, uchar& buffer[], uint length, int timeout);
   int TrComPortGetQueue(int portid, uint& input_queue, uint& output_queue);
#import
```

Transmitted data types had to be slightly modified for compatibility with MQL5.

The **TrComPortParameters** structure is as follows:

```
struct TrComPortParameters
{
   uint   DesiredParams;
   int    BaudRate;           // Data rate
   int    DefaultTimeout;     // Default timeout (in milliseconds)
   uchar  ByteSize;           // Data size(4-8)
   uchar  StopBits;           // Number of stop bits
   uchar  CheckParity;        // Parity check(0-no,1-yes)
   uchar  Parity;             // Parity type
   uchar  RtsControl;         // Initial RTS state
   uchar  DtrControl;         // Initial DTR state
};
```

Most devices work with the following settings: 8 data bits, no parity check, 1 stop bit. Therefore, out of all COM port parameters it makes sense to only add to the parameters of the Expert Advisor the COM port number and the data rate:

```
input ComPortList    inp_com_port_index=COM3;   // Choosing the COM port
input BaudRateList   inp_com_baudrate=_9600bps; // Data rate
```

The COM port initialization function will then be as follows:

```
//+------------------------------------------------------------------+
//| COM port initialization                                          |
//+------------------------------------------------------------------+
bool InitComPort()
  {
   rx_cnt=0;
   tx_cnt=0;
   tx_err=0;
//--- attempt to open the port
   PortID=TrComPortOpen(inp_com_port_index);
   if(PortID!=inp_com_port_index)
     {
      Print("Error when opening the COM port"+DoubleToString(inp_com_port_index+1,0));
      return(false);
     }
   else
     {
      Print("The COM port"+DoubleToString(inp_com_port_index+1,0)+" opened successfully");
      //--- request all parameters, so set all flags
      com_par.DesiredParams=tcpmpBaudRate|tcpmpDefaultTimeout|tcpmpByteSize|tcpmpStopBits|tcpmpCheckParity|tcpmpParity|tcpmpEnableRtsControl|tcpmpEnableDtrControl;
      //--- read current parameters
      if(TrComPortGetConfig(PortID,com_par)==-1)
         ,bnreturn(false);//read error
      //
      com_par.ByteSize=8;                //8 bits
      com_par.Parity=0;                  //no parity check
      com_par.StopBits=0;                //1 stop bit
      com_par.DefaultTimeout=100;        //100 ms timeout
      com_par.BaudRate=inp_com_baudrate; //rate - from the parameters of the Expert Advisor
      //---
      if(TrComPortSetConfig(PortID,com_par)==-1)
         return(false);//write error
     }
   return(true);
  }
```

In case of successful initialization the **PortID** variable will store the identifier of the open COM port.

It should be noted here that identifiers are numbered from zero, so the COM3 port identifier will be equal to 2. Now that the port is open we can exchange data with the modem. And, by the way, not only with a modem. The access to the COM port from the Expert Advisor opens up great opportunities for creativity to those who is good at soldering: you can connect the Expert Advisor to an LED or a moving text display to show equity or market prices of certain currency pairs.

The **TrComPortGetQueue** function shall be used to get details of the data in the queue of the COM port receiver and transmitter:

```
int TrComPortGetQueue(
   int   portid,           // COM port identifier
   uint& input_queue,      // Number of bytes in the input buffer
   uint& output_queue      // Number of bytes in the output buffer
   );
```

In case of an error, it returns a negative value of the error code. A detailed description of error codes is available in the archive with the **TrComPort.dll** library source codes.

If the function returns a non-zero number of data in the receiving buffer, they need to be read. For this purpose, we use the **TrComPortReadArray** function:

```
int TrComPortReadArray(
   int portid,             // Port identifier
   uchar& buffer[],        // Pointer to the buffer to read
   uint length,            // Number of data bytes
   int timeout             // Execution timeout (in ms)
   );
```

In case of an error, it returns a negative value of the error code. The number of data bytes should correspond to the value returned by the **TrComPortGetQueue** function.

To use the default timeout (set at the COM port initialization), you need to pass the value of -1.

To transmit data to the COM port, we use the **TrComPortWriteArray** function:

```
int TrComPortWriteArray(
   int portid,             // Port identifier
   uchar& buffer[],        // Pointer to the initial buffer
   uint length,            // Number of data bytes
   int timeout             // Execution timeout (in ms)
   );
```

Application example. In reply to the message saying "Hello world!", we should send "Have a nice day!".

```
uchar rx_buf[1024];
uchar tx_buf[1024];
string rx_str;
int rxn, txn;
TrComPortGetQueue(PortID, rxn, txn);
if(rxn>0)
{  //--- data received in the receiving buffer
   //--- read
   TrComPortReadArray(PortID, rx_buf, rxn, -1);
   //--- convert to a string
   rx_str = CharArrayToString(rx_buf,0,rxn,CP_ACP);
   //--- check the received message (expected message "Hello world!"
   if(StringFind(rx_str,"Hello world!",0)!=-1)
   {//--- if we have a match, prepare the reply
      string tx_str = "Have a nice day!";
      int len = StringLen(tx_str);//get the length in characters
      //--- convert to uchar buffer
      StringToCharArray(tx_str, tx_buf, 0, len, CP_ACP);
      //--- send to the port
      if(TrComPortWriteArray(PortID, tx_buf, len, -1)<0) Print("Error when writing to the port");
   }
}
```

Special attention should be paid to the port closing function:

```
int TrComPortClose(
   int portid         // Port identifier
   );
```

This function must always be present in the Expert Advisor deinitialization process. In most cases the port that was left open will become available again only after restarting the system. In fact, even turning the modem off and on may not help.

### 2\. AT Commands and Working with Modem

Working with a modem is arranged using [АТ commands](https://en.wikipedia.org/wiki/Hayes_command_set "https://en.wikipedia.org/wiki/Hayes_command_set"). Those of you who have ever used mobile Internet from a computer must remember the so-called "modem initialization string", which roughly looks as follows: AT+CGDCONT=1,"IP","internet". This is one of the AT commands. Almost all of them start with the prefix AT and end with 0x0d (carriage return).

We will use the minimum set of AT commands required for the implementation of the desired functionality. This will reduce the effort for ensuring compatibility of the command set with various devices.

Below is the list of the AT commands used by our handler for working with the modem:

| Command | Description |
| --- | --- |
| ATE1 | Enable echo |
| AT+CGMI | Get the name of manufacturer |
| AT+CGMM | Get the model of device |
| AT^SCKS | Get SIM card status |
| AT^SYSINFO | Get system information |
| AT+CREG | Get network registration state |
| AT+COPS | Get the name of the current mobile operator |
| AT+CMGF | Switch between text/PDU modes |
| AT+CLIP | Enable calling line identification |
| AT+CPAS | Get modem status |
| AT+CSQ | Get signal quality |
| AT+CUSD | Send a USSD request |
| AT+CALM | Enable silent mode (applicable to phones) |
| AT+CBC | Get battery status (applicable to phones) |
| AT+CSCA | Get SMS service center number |
| AT+CMGL | Get list of SMS messages |
| AT+CPMS | Select memory for SMS messages |
| AT+CMGD | Delete SMS message from the memory |
| AT+CMGR | Read SMS message from the memory |
| AT+CHUP | Reject incoming call |
| AT+CMGS | Send a SMS message |

I am not going to get off topic, describing the subtleties of working with AT commands. There is plenty of relevant information on technical forums. Besides, everything has already been implemented and in order to create an Expert Advisor capable of working with a modem, all we need is to include a header file and start using ready-made functions and structures. This is what I am going to elaborate on.

**2.1. Functions**

COM port initialization:

```
bool InitComPort();
```

Returned value: if initialized successfully - true, otherwise - false. It is called from the **OnInit()** function before the modem initialization.

COM port deinitialization:

```
void DeinitComPort();
```

Returned value: none. It is called from the **OnDeinit()** function.

Modem initialization:

```
void InitModem();
```

Returned value: none. It is called from the **OnInit()** function following a successful initialization of the COM port.

Modem event handler:

```
void ModemTimerProc();
```

Returned value: none. It is called from the **OnTimer()** function at 1-second intervals.

Reading SMS message by index from the modem memory:

```
bool ReadSMSbyIndex(
   int index,             // SMS message index in the modem memory
   INCOMING_SMS_STR& sms  // Pointer to the structure where the message will be moved
   );
```

Returned value: if read successfully - true, otherwise - false.

Deleting SMS message by index from the modem memory:

```
bool DelSMSbyIndex(
   int index              // SMS message index in the modem memory
   );
```

Returned value: if deleted successfully - true, otherwise - false.

Conversion of the connection quality index to a string:

```
string rssi_to_str(
   int rssi               // Connection quality index, values 0..31, 99
   );
```

Returned value: a string, e.g. "-55 dBm".

Sending SMS message:

```
bool SendSMS(
   string da,      // Recipient's phone number in international format
   string text,    // Text of the message, Latin characters and numbers, maximum length - 158 characters
   bool flash      // Flash message flag
   );
```

Returned value: if sent successfully - true, otherwise - false. SMS messages can only be sent when written using Latin characters. Cyrillic characters are supported for incoming SMS messages only. If flash=true is set, a flash message will be sent.

**2.2. Events (functions called by the modem handler)**

Updating data in the modem state structure:

```
void ModemChState();
```

Passed parameters: none. When this function is called by the modem handler, it suggests that data has been updated in the modem structure (the description of the structure will be provided below).

Incoming call:

```
void IncomingCall(
   string number          // Caller number
   );
```

Passed parameters: caller number. When this function is called by the modem handler, it suggests that the incoming call from the 'number' number was accepted and rejected.

New incoming SMS message:

```
void IncomingSMS(
   INCOMING_SMS_STR& sms  // SMS message structure
   );
```

Passed parameters: SMS message structure (the description of the structure will be provided below). When this function is called by the modem handler, it suggests that there is one or more new unread SMS messages in the modem memory. If the number of unread messages is greater than one, the most recent message will be passed to this function.

SMS memory full:

```
void SMSMemoryFull(
   int n                  // Number of SMS messages in the modem memory
   );
```

Passed parameters: number of messages in the modem memory. When this function is called by the modem handler, it suggests that SMS memory is full and the modem will not accept any new messages until the memory is released.

**2.3. State structure of modem parameters**

```
struct MODEM_STR
{
   bool     init_ok;          // The required minimum initialized
   //
   string   manufacturer;     // Manufacturer
   string   device;           // Model
   int      sim_stat;         // SIM status
   int      net_reg;          // Network registration state
   int      status;           // Modem status
   string   op;               // Operator
   int      rssi;             // Signal quality
   string   sms_sca;          // SMS center number
   int      bat_stat;         // Battery state
   int      bat_charge;       // Battery charge in percent (applicability depends on bat_stat)
   //
   double   bal;              // Mobile account balance
   string   exp_date;         // Mobile number expiration date
   int      sms_free;         // Package SMS available
   int      sms_free_cnt;     // Counter of package SMS used
   //
   int      sms_mem_size;     // SMS memory size
   int      sms_mem_used;     // Used SMS memory size
   //
   string   incoming;         // Caller number
};

MODEM_STR modem;
```

This structure is filled exclusively by the modem event handler and should be used by other functions only for reading.

Below is the description of structure elements:

| Element | Description |
| --- | --- |
| modem.init\_ok | An indication that the modem has been initialized successfully. <br>  The initial value of false becomes true after the initialization is complete. |
| modem.manufacturer | Modem manufacturer, e.g. "huawei".<br>  The initial value is "n/a". |
| modem.device | Modem model, e.g. "E1550"<br>  The initial value is "n/a". |
| modem.sim\_stat | Sim card status. It can take on the following values:<br>  -1 - no data<br>   0 - the card is missing, blocked or out of order<br>   1 - the card is available |
| modem.net\_reg | Network registration state. It can take on the following values:<br>  -1 - no data<br>   0 - not registered<br>   1 - registered<br>   2 - searching<br>   3 - forbidden<br>   4 - state undefined<br>   5 - registered in roaming |
| modem.status | Modem status. It can take on the following values:<br>  -1 - initialization<br>   0 - ready<br>   1 - error<br>   2 - error<br>   3 - incoming call<br>   4 - active call |
| modem.op | Current mobile operator. <br>  It can be equal to either the operator name (e.g. "MTS UKR"), <br>  or the international operator code (e.g. "25501").<br>  The initial value is "n/a". |
| modem.rssi | Signal quality index. It can take on the following values:<br>  -1 - no data<br>   0 - signal -113 dBm or lower<br>   1 - signal -111 dBm<br>   2...30 - signal -109...-53 dBm<br>  31 - signal -51 dBm or higher<br>  99 - no data<br>  For conversion to a string, use the rssi\_to\_str() function. |
| modem.sms\_sca | SMS service center number. It is contained in the SIM card memory.<br>  It is necessary for generating an outgoing SMS message.<br>  In rare cases, if the number is not saved in the SIM card memory, it will be<br>  replaced with the number specified in the input parameters of the Expert Advisor. |
| modem.bat\_stat | Modem battery status (applicable to phones only).<br>  It can take on the following values:<br>  -1 - no data<br>   0 - the device works on battery power<br>   1 - the battery is available but the device is not battery powered<br>   2 - no battery<br>   3 - error |
| modem.bat\_charge | Battery charge in percent.<br>  It can take on values from 0 to 100. |
| modem.bal | Mobile account balance. The value is obtained<br>  from the operator response to the relevant USSD request.<br>  The initial value (prior to initialization): -10000. |
| modem.exp\_date | Mobile number expiration date. The value is obtained<br>  from the operator response to the relevant USSD request.<br>  The initial value is "n/a". |
| modem.sms\_free | Number of package SMS available. It is calculated as the difference between<br>  the initial number and the counter of package SMS used. |
| modem.sms\_free\_cnt | Counter of package SMS used. The value is obtained<br>  from the operator response to the relevant USSD request. The initial value is -1. |
| modem.sms\_mem\_size | Modem SMS memory size. |
| modem.sms\_mem\_used | Modem SMS memory used. |
| modem.incoming | Number of the last caller.<br>  The initial value is "n/a". |

**2.4. SMS message structure**

```
//+------------------------------------------------------------------+
//| SMS message structure                                            |
//+------------------------------------------------------------------+
struct INCOMING_SMS_STR
{
   int index;                //index in the modem memory
   string sca;               //sender's SMS center number
   string sender;            //sender's number
   INCOMING_CTST_STR scts;   //SMS center time label
   string text;              //text of the message
};
```

The SMS center time label is the time when a given message from the sender was received in the SMS center. The time label structure is as follows:

```
//+------------------------------------------------------------------+
//| Time label structure                                             |
//+------------------------------------------------------------------+
struct INCOMING_CTST_STR
{
   datetime time;            // time
   int gmt;                  // time zone
};
```

The time zone is expressed in 15-minute intervals. So, the value of 8 corresponds to GMT+02:00.

The text of received SMS message can be written using Latin, as well as Cyrillic characters. 7-bit and UCS2 encodings are supported for received messages. Merging of long messages is not implemented (in view of the fact that this operation is designed for short commands).

SMS messages can only be sent if written using Latin characters. Maximum message length is 158 characters. In case of a longer message, it will be sent without the characters in excess of the specified number.

### 3\. Developing an Expert Advisor

For a start, you need to copy the file **TrComPort.dll** to the **Libraries** folder and place **ComPort.mqh**, **modem.mqh** and **sms.mqh** in the **Include** folder.

Then using the Wizard, we create a new Expert Advisor and add the minimum required for working with the modem. That is:

Include **modem.mqh:**

```
#include <modem.mqh>
```

Add input parameters:

```
input string         str00="COM port settings";
input ComPortList    inp_com_port_index=COM3;   // Selecting the COM port
input BaudRateList   inp_com_baudrate=_9600bps; // Data rate
//
input string         str01="Modem";
input int            inp_refr_period=3;         // Modem query period, sec
input int            inp_ussd_request_tout=20;  // Timeout for response to a USSD request, sec
input string         inp_sms_service_center=""; // SMS service center number
//
input string         str02="Balance";
input int            inp_refr_bal_period=12;    // Query period, hr
input string         inp_ussd_get_balance="";   // Balance USSD request
input string         inp_ussd_bal_suffix="";    // Balance suffix
input string         inp_ussd_exp_prefix="";    // Prefix of the number expiration date
//
input string         str03="Number of package SMS";
input int            inp_refr_smscnt_period=6;  // Query period, hr
input string         inp_ussd_get_sms_cnt="";   // USSD request for the package service status
input string         inp_ussd_sms_suffix="";    // SMS counter suffix
input int            inp_free_sms_daily=0;      // Daily SMS limit
```

Functions called by the modem handler:

```
//+------------------------------------------------------------------+
//| Called when a new SMS message is received                        |
//+------------------------------------------------------------------+
void IncomingSMS(INCOMING_SMS_STR& sms)
{
}

//+------------------------------------------------------------------+
//| SMS memory is full                                               |
//+------------------------------------------------------------------+
void SMSMemoryFull(int n)
{
}

//+------------------------------------------------------------------+
//| Called upon receiving an incoming call                           |
//+------------------------------------------------------------------+
void IncomingCall(string number)
{
}

//+------------------------------------------------------------------+
//| Called after updating data in the modem status structure         |
//+------------------------------------------------------------------+
void ModemChState()
{
   static bool init_ok = false;
   if(modem.init_ok==true && init_ok==false)
   {
      Print("Modem initialized successfully");
      init_ok = true;
   }
}
```

The COM port and modem initialization along with a timer set at 1 second intervals should be added to the **OnInit()** function:

```
int OnInit()
{  //---COM port initialization
   if(InitComPort()==false)
   {
      Print("Error when initializing the COM"+DoubleToString(inp_com_port_index+1,0)+" port");
      return(INIT_FAILED);
   }
   //--- modem initialization
   InitModem();
   //--- setting the timer
   EventSetTimer(1); //1 second interval
   //
   return(INIT_SUCCEEDED);
}
```

In the **OnTimer()** function, we need to call the modem handler:

```
void OnTimer()
{
//---
   ModemTimerProc();
}
```

It is necessary to call the COM port deinitialization in the **OnDeinit()** function:

```
void OnDeinit(const int reason)
{
//--- destroy timer
   EventKillTimer();
   DeinitComPort();
}
```

We compile the code and see: 0 error(s).

Now run the Expert Advisor, but remember to allow DLL import and select the COM port associated with the modem. You should be able to see the following messages in the "Expert Advisors" tab:

![The first run](https://c.mql5.com/2/6/en_02__3.png)

Fig. 2. Messages of the Expert Advisor following a successful run

If you have got the same messages, it means that your modem (phone) is suitable for working with this Expert Advisor. In this case we move further.

Let's draw a table for visualization of modem parameters. It will be placed in the upper left corner of the terminal window, under the OHLC line. The text font to be used in the table will be monospaced, e.g. "Courier New".

```
//+------------------------------------------------------------------+
//| TextXY                                                           |
//+------------------------------------------------------------------+
void TextXY(string ObjName,string Text,int x,int y,color TextColor)
  {
//--- display the text string
   ObjectDelete(0,ObjName);
   ObjectCreate(0,ObjName,OBJ_LABEL,0,0,0,0,0);
   ObjectSetInteger(0,ObjName,OBJPROP_XDISTANCE,x);
   ObjectSetInteger(0,ObjName,OBJPROP_YDISTANCE,y);
   ObjectSetInteger(0,ObjName,OBJPROP_COLOR,TextColor);
   ObjectSetInteger(0,ObjName,OBJPROP_FONTSIZE,9);
   ObjectSetString(0,ObjName,OBJPROP_FONT,"Courier New");
   ObjectSetString(0,ObjName,OBJPROP_TEXT,Text);
  }
//+------------------------------------------------------------------+
//| Drawing the table of modem parameters                            |
//+------------------------------------------------------------------+
void DrawTab()
  {
   int   x=20, //horizontal indent
   y = 20,     //vertical indent
   dy = 15;    //step along the Y-axis
//--- draw the background
   ObjectDelete(0,"bgnd000");
   ObjectCreate(0,"bgnd000",OBJ_RECTANGLE_LABEL,0,0,0,0,0);
   ObjectSetInteger(0,"bgnd000",OBJPROP_XDISTANCE,x-10);
   ObjectSetInteger(0,"bgnd000",OBJPROP_YDISTANCE,y-5);
   ObjectSetInteger(0,"bgnd000",OBJPROP_XSIZE,270);
   ObjectSetInteger(0,"bgnd000",OBJPROP_YSIZE,420);
   ObjectSetInteger(0,"bgnd000",OBJPROP_BGCOLOR,clrBlack);
//--- port parameters
   TextXY("str0",  "Port:            ", x, y, clrWhite); y+=dy;
   TextXY("str1",  "Speed:           ", x, y, clrWhite); y+=dy;
   TextXY("str2",  "Rx:              ", x, y, clrWhite); y+=dy;
   TextXY("str3",  "Tx:              ", x, y, clrWhite); y+=dy;
   TextXY("str4",  "Err:             ", x, y, clrWhite); y+=(dy*3)/2;
//--- modem parameters
   TextXY("str5",  "Modem:           ", x, y, clrWhite); y+=dy;
   TextXY("str6",  "SIM:             ", x, y, clrWhite); y+=dy;
   TextXY("str7",  "NET:             ", x, y, clrWhite); y+=dy;
   TextXY("str8",  "Operator:        ", x, y, clrWhite); y+=dy;
   TextXY("str9",  "SMSC:            ", x, y, clrWhite); y+=dy;
   TextXY("str10", "RSSI:            ", x, y, clrWhite); y+=dy;
   TextXY("str11", "Bat:             ", x, y, clrWhite); y+=dy;
   TextXY("str12", "Modem status:    ", x, y, clrWhite); y+=(dy*3)/2;
//--- mobile account balance
   TextXY("str13", "Balance:         ", x, y, clrWhite); y+=dy;
   TextXY("str14", "Expiration date: ", x, y, clrWhite); y+=dy;
   TextXY("str15", "Free SMS:        ", x, y, clrWhite); y+=(dy*3)/2;
//--- number of the last incoming call
   TextXY("str16","Incoming:        ",x,y,clrWhite); y+=(dy*3)/2;
//--- parameters of the last received SMS message
   TextXY("str17", "SMS mem full:    ", x, y, clrWhite); y+=dy;
   TextXY("str18", "SMS number:      ", x, y, clrWhite); y+=dy;
   TextXY("str19", "SMS date/time:   ", x, y, clrWhite); y+=dy;
//--- text of the last received SMS message
   TextXY("str20", "                 ", x, y, clrGray); y+=dy;
   TextXY("str21", "                 ", x, y, clrGray); y+=dy;
   TextXY("str22", "                 ", x, y, clrGray); y+=dy;
   TextXY("str23", "                 ", x, y, clrGray); y+=dy;
   TextXY("str24", "                 ", x, y, clrGray); y+=dy;
//---
   ChartRedraw(0);
  }
```

To refresh data in the table, we will use the **RefreshTab()** function:

```
//+------------------------------------------------------------------+
//| Refreshing values in the table                                   |
//+------------------------------------------------------------------+
void RefreshTab()
  {
   string str;
//--- COM port index:
   str="COM"+DoubleToString(PortID+1,0);
   ObjectSetString(0,"str0",OBJPROP_TEXT,"Port:            "+str);
//--- data rate:
   str=DoubleToString(inp_com_baudrate,0)+" bps";
   ObjectSetString(0,"str1",OBJPROP_TEXT,"Speed:           "+str);
//--- number of bytes received:
   str=DoubleToString(rx_cnt,0)+" bytes";
   ObjectSetString(0,"str2",OBJPROP_TEXT,"Rx:              "+str);
//--- number of bytes transmitted:
   str=DoubleToString(tx_cnt,0)+" bytes";
   ObjectSetString(0,"str3",OBJPROP_TEXT,"Tx:              "+str);
//--- number of port errors:
   str=DoubleToString(tx_err,0);
   ObjectSetString(0,"str4",OBJPROP_TEXT,"Err:             "+str);
//--- modem manufacturer and model:
   str=modem.manufacturer+" "+modem.device;
   ObjectSetString(0,"str5",OBJPROP_TEXT,"Modem:           "+str);
//--- SIM card status:
   string sim_stat_str[2]={"Error","Ok"};
   if(modem.sim_stat==-1)
      str="n/a";
   else
      str=sim_stat_str[modem.sim_stat];
   ObjectSetString(0,"str6",OBJPROP_TEXT,"SIM:             "+str);
//--- network registration:
   string net_reg_str[6]={"No","Ok","Search...","Restricted","Unknown","Roaming"};
   if(modem.net_reg==-1)
      str="n/a";
   else
      str=net_reg_str[modem.net_reg];
   ObjectSetString(0,"str7",OBJPROP_TEXT,"NET:             "+str);
//--- name of mobile operator:
   ObjectSetString(0,"str8",OBJPROP_TEXT,"Operator:        "+modem.op);
//--- SMS service center number
   ObjectSetString(0,"str9",OBJPROP_TEXT,"SMSC:            "+modem.sms_sca);
//--- signal level:
   if(modem.rssi==-1)
      str="n/a";
   else
      str=rssi_to_str(modem.rssi);
   ObjectSetString(0,"str10",OBJPROP_TEXT,"RSSI:            "+str);
//--- battery status (applicable to phones):
   string bat_stats_str[4]={"Ok, ","Ok, ","No","Err"};
   if(modem.bat_stat==-1)
      str="n/a";
   else
      str=bat_stats_str[modem.bat_stat];
   if(modem.bat_stat==0 || modem.bat_stat==1)
      str+=DoubleToString(modem.bat_charge,0)+"%";
   ObjectSetString(0,"str11",OBJPROP_TEXT,"Bat:             "+str);
//--- modem status:
   string modem_stat_str[5]={"Ready","Err","Err","Incoming call","Active call"};
   if(modem.status==-1)
      str="init...";
   else
     {
      if(modem.status>4 || modem.status<0)
         Print("Unknown modem status: "+DoubleToString(modem.status,0));
      else
         str=modem_stat_str[modem.status];
     }
   ObjectSetString(0,"str12",OBJPROP_TEXT,"Modem status:    "+str);
//--- mobile account balance:
   if(modem.bal==-10000)
      str="n/a";
   else
      str=DoubleToString(modem.bal,2)+" "+inp_ussd_bal_suffix;
   ObjectSetString(0,"str13",OBJPROP_TEXT,"Balance:         "+str);
//--- mobile number expiration date:
   ObjectSetString(0,"str14",OBJPROP_TEXT,"Expiration date: "+modem.exp_date);
//--- package SMS available:
   if(modem.sms_free<0)
      str="n/a";
   else
      str=DoubleToString(modem.sms_free,0);
   ObjectSetString(0,"str15",OBJPROP_TEXT,"Free SMS:        "+str);
//--- SMS memory full:
   if(sms_mem_full==true)
      str="Yes";
   else
      str="No";
   ObjectSetString(0,"str17",OBJPROP_TEXT,"SMS mem full:    "+str);
//---
   ChartRedraw(0);
  }
```

The **DelTab()** function deletes the table:

```
//+------------------------------------------------------------------+
//| Deleting the table                                               |
//+------------------------------------------------------------------+
void DelTab()
  {
   for(int i=0; i<25; i++)
      ObjectDelete(0,"str"+DoubleToString(i,0));
   ObjectDelete(0,"bgnd000");
  }
```

Let's add functions for working with the table to event handlers **OnInit()** and **OnDeinit()**, as well as to the **ModemChState()** function:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- COM port initialization
   if(InitComPort()==false)
     {
      Print("Error when initializing the COM port"+DoubleToString(inp_com_port_index+1,0));
      return(INIT_FAILED);
     }
//---
   DrawTab();
//--- modem initialization
   InitModem();
//--- setting the timer
   EventSetTimer(1);//1 second interval
//---
   RefreshTab();
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- destroy timer
   EventKillTimer();
   DeinitComPort();
   DelTab();
  }
//+------------------------------------------------------------------+
//| ModemChState                                                     |
//+------------------------------------------------------------------+
void ModemChState()
  {
   static bool init_ok=false;
//Print("Modem status changed");
   if(modem.init_ok==true && init_ok==false)
     {
      Print("Modem initialized successfully");
      init_ok=true;
     }
//---
   RefreshTab();
  }
```

Further, we add the opportunity to refresh the number of the last incoming call in the table to the **IncomingCall()** function:

```
//+------------------------------------------------------------------+
//| Called upon receiving an incoming call                           |
//+------------------------------------------------------------------+
void IncomingCall(string number)
{
   //--- update the number of the last incoming call:
   ObjectSetString(0, "str16",OBJPROP_TEXT, "Incoming:        "+number);
}
```

Now, compile the code and run the Expert Advisor. You should be able to see the following report in the terminal window:

![Modem status parameters](https://c.mql5.com/2/6/tab_1.png)

Fig. 3. Modem parameters

Try calling the modem. The call will be rejected and your number will appear in the "Incoming" line.

### 4\. Working with USSD Requests

A mobile account not topped up in due time may disrupt the operation of an Expert Advisor at the least appropriate moment. So, the function that checks the account balance is one of the most important ones. To check the mobile account balance we usually use [USSD](https://en.wikipedia.org/wiki/Unstructured_Supplementary_Service_Data "https://en.wikipedia.org/wiki/Unstructured_Supplementary_Service_Data") requests. Further, we will use USSD requests to get information on the number of package SMS available.

Data for generating requests and processing responses received are located in the input parameters:

```
input string         str02="=== Balance ======";
input int            inp_refr_bal_period=12;  //query period, hr
input string         inp_ussd_get_balance=""; //balance USSD request
input string         inp_ussd_bal_suffix="";  //balance suffix
input string         inp_ussd_exp_prefix="";  //prefix of the number expiration date
//
input string         str03="= Number of package SMS ==";
input int            inp_refr_smscnt_period=6;//query period, hr
input string         inp_ussd_get_sms_cnt=""; //USSD request for the package service status
input string         inp_ussd_sms_suffix="";  //SMS counter suffix
input int            inp_free_sms_daily=0;    //daily SMS limit
```

If the request number is not specified, the request will not be processed. Alternatively, the request will be sent right after the modem initialization and will be repeatedly sent after the specified period of time. Further, the request will not be processed if your modem (phone) does not support the relevant IT command (it concerns old cellphone models).

Assume that following the balance request you receive the following response from your operator:

**7.13 UAH, expires on 22.05.2014. Phone Plan - Super MTS 3D Null 25.**

To ensure that the handler identifies the response correctly, the balance suffix must be set to " **UAH**" and the prefix of the number expiration date should be " **expires on**".

Since our Expert Advisor is expected to send SMS messages quite often, it would be good to buy an SMS package from your operator, that is, a service whereby you get a certain number of SMS messages for a small fee. In this case, it may be very useful to know how many package SMS are still available. This can also be done using a USSD request. The operator usually responds with the number of used SMS instead of the available ones.

Assume, you have received the following response from your operator:

**Balance: 69 minutes of local calls for today. Used today: 0 SMS and 0 MB.**

In this case, the SMS counter suffix shall be set to " **SMS**" and the daily limit shall be set in accordance with the SMS package terms and conditions. For instance, if you are given 30 text messages per day and the request returned the value of 10, it means that you have 30-10=20 SMS available. This number will be placed by the handler to the appropriate element of the modem status structure.

**ATTENTION!****Be very careful** with USSD request numbers! Sending a wrong request can have **undesirable consequences**, e.g. enabling some **unwanted paid service**!

In order for our Expert Advisor to start working with USSD requests, we just need to specify the relevant input parameters.

For example, the parameters for the Ukrainian mobile operator, MTS Ukraine, will be as follows:

![Parameters of the request for the available balance](https://c.mql5.com/2/6/en_04__1.png)

Fig. 4. Parameters of the USSD request for the available balance

![Parameters of the request for the number of package SMS available](https://c.mql5.com/2/6/en_05__1.png)

Fig. 5. Parameters of the USSD request for the number of package SMS available

Set the values relevant to your mobile operator. After that, the available balance in your mobile account and the number of available SMS will be displayed in the table of modem status:

![Available Balance](https://c.mql5.com/2/6/balance.PNG)

Fig. 6. Parameters obtained from USSD responses

At the time of writing this article, my mobile operator was sending Christmas advertisement instead of the number expiration date. Consequently, the handler could not obtain the date value, which is why we can see "n/a" in the "Expiration date" line. Please note that all operator responses are displayed in the "Expert Advisors" tab.

![Operator responses](https://c.mql5.com/2/6/en_07.png)

Fig. 7. Operator responses displayed in the "Expert Advisors" tab

### 5\. Sending SMS Messages

We are going to start adding useful functions, for instance, sending SMS messages stating the current profit, equity and the number of open positions. Sending will be initiated by an incoming call.

Such response is certainly expected only in case of the administrator number, so we will have another input parameter:

```
input string         inp_admin_number="+XXXXXXXXXXXX";//administrator's phone number
```

The number shall be specified in international format, including "+" before the number.

The number check, as well as SMS text generation and sending should be added to the incoming call handler:

```
//+------------------------------------------------------------------+
//| Called upon receiving an incoming call                           |
//+------------------------------------------------------------------+
void IncomingCall(string number)
{
   bool result;
   if(number==inp_admin_number)
   {
      Print("Administrator's phone number. Sending SMS.");
      //
      string mob_bal="";
      if(modem.bal!=-10000)//mobile account balance
         mob_bal = "\n(m.bal="+DoubleToString(modem.bal,2)+")";
      result = SendSMS(inp_admin_number, "Account: "+DoubleToString(AccountInfoInteger(ACCOUNT_LOGIN),0)
               +"\nProfit: "+DoubleToString(AccountInfoDouble(ACCOUNT_PROFIT),2)
               +"\nEquity: "+DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY),2)
               +"\nPositions: "+DoubleToString(PositionsTotal(),0)
               +mob_bal
               , false);
      if(result==true)
         Print("SMS sent successfully");
      else
         Print("Error when sending SMS");
   }
   else
      Print("Unauthorized number ("+number+")");
   //--- update the number of the last incoming call:
   ObjectSetString(0, "str16",OBJPROP_TEXT, "Incoming:        "+number);
}
```

Now, if there is a call to the modem from the administrator's number **inp\_admin\_number**, an SMS message will be sent in response:

![SMS in response to an incoming call](https://c.mql5.com/2/6/en_08.png)

Fig. 8. The SMS message sent by the Expert Advisor in response to the call received from the administrator's phone number

Here, we can see the current values of profit and equity, as well as the number of open positions and the mobile account balance.

### 6\. Monitoring Connection with the Trade Server

Let's add notifications in case of lost and reestablished connection with the trade server. For this purpose, we will check trade server connectivity once every 10 seconds using [TerminalInfoInteger()](https://www.mql5.com/en/docs/check/terminalinfointeger) with the [TERMINAL\_CONNECTED](https://www.mql5.com/en/docs/constants/environment_state/terminalstatus#enum_terminal_info_integer) property identifier.

To filter short-time connection losses, we will use hysteresis which should be added to the list of input parameters:

```
input int            inp_conn_hyst=6; //Hysteresis, х10 sec
```

The value of 6 means that connection will be considered lost if there is no connection for more than 6\*10=60 seconds. Similarly, connection will be considered reestablished if it is available for more than 60 seconds. The local time of the first registered lack of connectivity will be deemed the time of connection loss, while the first local time when the connection became available will be deemed the recovery time.

To implement this, we add the following code to the **OnTimer()** function:

```
   static int s10 = 0;//pre-divider by 10 seconds
   static datetime conn_time;
   static datetime disconn_time;
   if(++s10>=10)
   {//--- once every 10 seconds
      s10 = 0;
      //
      if((bool)TerminalInfoInteger(TERMINAL_CONNECTED)==true)
      {
         if(cm.conn_cnt==0)             //first successful query in the sequence
            conn_time = TimeLocal();    //save the time
         if(cm.conn_cnt<inp_conn_hyst)
         {
            if(++cm.conn_cnt>=inp_conn_hyst)
            {//--- connection has been stabilized
               if(cm.connected == false)
               {//--- if there was a long-standing connection loss prior to that
                  cm.connected = true;
                  cm.new_state = true;
                  cm.conn_time = conn_time;
               }
            }
         }
         cm.disconn_cnt = 0;
      }
      else
      {
         if(cm.disconn_cnt==0)          //first unsuccessful query in the sequence
            disconn_time = TimeLocal(); //save the time
         if(cm.disconn_cnt<inp_conn_hyst)
         {
            if(++cm.disconn_cnt>=inp_conn_hyst)
            {//--- long-standing connection loss
               if(cm.connected == true)
               {//--- if the connection was stable prior to that
                  cm.connected = false;
                  cm.new_state = true;
                  cm.disconn_time = disconn_time;
               }
            }
         }
         cm.conn_cnt = 0;
      }
   }
   //
   if(cm.new_state == true)
   {//--- connection status changed
      if(cm.connected == true)
      {//--- connection is available
         string str = "Connected "+TimeToString(cm.conn_time,TIME_DATE|TIME_SECONDS);
         if(cm.disconn_time!=0)
            str+= ", offline: "+dTimeToString((ulong)(cm.conn_time-cm.disconn_time));
         Print(str);
         SendSMS(inp_admin_number, str, false);//sending message
      }
      else
      {//--- no connection
         string str = "Disconnected "+TimeToString(cm.disconn_time,TIME_DATE|TIME_SECONDS);
         if(cm.conn_time!=0)
            str+= ", online: "+dTimeToString((ulong)(cm.disconn_time-cm.conn_time));
         Print(str);
         SendSMS(inp_admin_number, str, false);//sending message
      }
      cm.new_state = false;
   }
```

The **cm** structure is as follows:

```
//+------------------------------------------------------------------+
//| Structure of monitoring connection with the terminal             |
//+------------------------------------------------------------------+
struct CONN_MON_STR
  {
   bool              new_state;    //flag of change in the connection status
   bool              connected;    //connection status
   int               conn_cnt;     //counter of successful connection queries
   int               disconn_cnt;  //counter of unsuccessful connection queries
   datetime          conn_time;    //time of established connection
   datetime          disconn_time; //time of lost connection
  };

CONN_MON_STR cm;//structure of connection monitoring
```

In the text of the SMS message, we will state the time when connection with the trade server was lost (or reestablished), as well as the time during which connection was available (or not available) calculated as the difference between the time of established connection and the time of lost connection. To convert the time difference from seconds to dd hh:mm:ss, we will add the **dTimeToString()** function:

```
string dTimeToString(ulong sec)
{
   string str;
   uint days = (uint)(sec/86400);
   if(days>0)
   {
      str+= DoubleToString(days,0)+" days, ";
      sec-= days*86400;
   }
   uint hour = (uint)(sec/3600);
   if(hour<10) str+= "0";
   str+= DoubleToString(hour,0)+":";
   sec-= hour*3600;
   uint min = (uint)(sec/60);
   if(min<10) str+= "0";
   str+= DoubleToString(min,0)+":";
   sec-= min*60;
   if(sec<10) str+= "0";
   str+= DoubleToString(sec,0);
   //
   return(str);
}
```

To ensure that the Expert Advisor does not send a text message about the established connection every time the Expert Advisor is run, we add a **conn\_mon\_init()** function that sets the values to **cm** structure elements in such a way, as if the connection were already established. In this case, the connection will be deemed established at the local time of running the Expert Advisor. This function must be called from the **OnInit() function**.

```
void conn_mon_init()
{
   cm.connected = true;
   cm.conn_cnt = inp_conn_hyst;
   cm.disconn_cnt = 0;
   cm.conn_time = TimeLocal();
   cm.new_state = false;
}
```

Now, compile and run the Expert Advisor. Then try to disconnect your computer from the Internet. In 60 (give or take 10) seconds, you will receive a message saying that the connection with the server has been lost. Connect back to the Internet. In 60 seconds, you will get a message about reestablished connection, stating the total disconnection time:

![The message about lost connection](https://c.mql5.com/2/6/en_09a.png)![The message about reestablished connection](https://c.mql5.com/2/6/en_09b.png)

Fig. 9. Text messages notifying of the lost and reestablished connection with the server

### 7\. Sending Reports on Position Opening and Closing

To monitor the opening and closing of positions, let's add the following code to the **OnTradeTransaction()** function:

```
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
{
//---
   if(trans.type==TRADE_TRANSACTION_DEAL_ADD)
   {
      if(trans.deal_type==DEAL_TYPE_BUY ||
         trans.deal_type==DEAL_TYPE_SELL)
      {
         int i;
         for(i=0;i<POS_BUF_LEN;i++)
         {
            if(ps[i].new_event==false)
               break;
         }
         if(i<POS_BUF_LEN)
         {
            ps[i].new_event = true;
            ps[i].deal_type = trans.deal_type;
            ps[i].symbol = trans.symbol;
            ps[i].volume = trans.volume;
            ps[i].price = trans.price;
            ps[i].deal = trans.deal;
         }
      }
   }
}
```

where **ps** is the buffer of **POS\_STR** structures:

```
struct POS_STR
{
   bool new_event;
   string symbol;
   ulong deal;
   ENUM_DEAL_TYPE deal_type;
   double volume;
   double price;
};

#define POS_BUF_LEN  3

POS_STR ps[POS_BUF_LEN];
```

The buffer is required in case more than one position was closed (or opened) within a short period of time. When a position is opened or closed, after the deal is added to the history, we get all the necessary parameters and set the **new\_event** flag.

Below is the code that will be added to the **OnTimer()** function to watch for **new\_event** flags and generate SMS reports:

```
   //--- processing of the opening/closing of positions
   string posstr="";
   for(int i=0;i<POS_BUF_LEN;i++)
   {
      if(ps[i].new_event==true)
      {
         string str;
         if(ps[i].deal_type==DEAL_TYPE_BUY)
            str+= "Buy ";
         else if(ps[i].deal_type==DEAL_TYPE_SELL)
            str+= "Sell ";
         str+= DoubleToString(ps[i].volume,2)+" "+ps[i].symbol;
         int digits = (int)SymbolInfoInteger(ps[i].symbol,SYMBOL_DIGITS);
         str+= ", price="+DoubleToString(ps[i].price,digits);
         //
         long deal_entry;
         HistorySelect(TimeCurrent()-3600,TimeCurrent());//retrieve the history for the last hour
         if(HistoryDealGetInteger(ps[i].deal,DEAL_ENTRY,deal_entry)==true)
         {
            if(((ENUM_DEAL_ENTRY)deal_entry)==DEAL_ENTRY_IN)
               str+= ", entry: in";
            else if(((ENUM_DEAL_ENTRY)deal_entry)==DEAL_ENTRY_OUT)
            {
               str+= ", entry: out";
               double profit;
               if(HistoryDealGetDouble(ps[i].deal,DEAL_PROFIT,profit)==true)
               {
                  str+= ", profit = "+DoubleToString(profit,2);
               }
            }
         }
         posstr+= str+"\r\n";
         ps[i].new_event=false;
      }
   }
   if(posstr!="")
   {
      Print(posstr+"pos: "+DoubleToString(PositionsTotal(),0));
      SendSMS(inp_admin_number, posstr+"pos: "+DoubleToString(PositionsTotal(),0), false);
   }
```

Now, compile and run the Expert Advisor. Let's try to buy AUDCAD, with the lot size of 0.14. The Expert Advisor will send the following SMS message: "Buy 0.14 AUDCAD, price=0.96538, entry: in". After a little while we close the position and get the following text message regarding the position closing:

![The message about the position opening](https://c.mql5.com/2/6/en_10a.png)![The message about the position closing](https://c.mql5.com/2/6/en_10b.png)

Fig. 10. Text messages about the position opening (entry: in) and closing (entry: out)

### 8\. Processing Incoming SMS Messages for Open Position Management

Until now, our Expert Advisor has only sent messages to the administrator's phone number. Let's now teach it to receive and execute SMS commands. This may be useful in, e.g. closing all or some open positions. As we know, there is nothing like having your position closed on time.

But we should first make sure that SMS messages are received correctly. To do this, we add the display of the last received message to the **IncomingSMS()** function:

```
//+------------------------------------------------------------------+
//| Called when a new SMS message is received                        |
//+------------------------------------------------------------------+
void IncomingSMS(INCOMING_SMS_STR& sms)
{
   string str, strtmp;
   //Number from which the last received SMS message was sent:
   ObjectSetString(0, "str18", OBJPROP_TEXT, "SMS number:      "+sms.sender);
   //Date and time of sending the last received SMS message:
   str = TimeToString(sms.scts.time,TIME_DATE|TIME_SECONDS);
   ObjectSetString(0, "str19", OBJPROP_TEXT, "SMS date/time:   "+str);
   //Text of the last received SMS message:
   strtmp = StringSubstr(sms.text, 0, 32); str = " ";
   if(strtmp!="") str = strtmp;
   ObjectSetString(0, "str20", OBJPROP_TEXT, str);
   strtmp = StringSubstr(sms.text,32, 32); str = " ";
   if(strtmp!="") str = strtmp;
   ObjectSetString(0, "str21", OBJPROP_TEXT, str);
   strtmp = StringSubstr(sms.text,64, 32); str = " ";
   if(strtmp!="") str = strtmp;
   ObjectSetString(0, "str22", OBJPROP_TEXT, str);
   strtmp = StringSubstr(sms.text,96, 32); str = " ";
   if(strtmp!="") str = strtmp;
   ObjectSetString(0, "str23", OBJPROP_TEXT, str);
   strtmp = StringSubstr(sms.text,128,32); str = " ";
   if(strtmp!="") str = strtmp;
   ObjectSetString(0, "str24", OBJPROP_TEXT, str);
}
```

If we now send a SMS message to the modem, it will be displayed in the table:

![New SMS message](https://c.mql5.com/2/6/new_sms.PNG)

Fig. 11. The incoming SMS message as displayed in the terminal window

Please note that all incoming SMS messages are displayed in the "Expert Advisors" tab in the following form: <index\_in\_modem\_memory>text\_of\_the\_message:

![The SMS message displayed in the "Expert Advisors" tab](https://c.mql5.com/2/6/en_12.png)

Fig. 12. The incoming SMS message as displayed in the "Expert Advisors" tab

The word **"close"** will be used as a command for closing deals. It should be followed by the space and the **parameter** \- the symbol of the position that needs to be closed, or **"all"** in case you need to close all positions. The case is of no importance as prior to processing the text of the message, we use the [StringToUpper()](https://www.mql5.com/en/docs/strings/stringtoupper) function. When analyzing the message, make sure to check that the sender's phone number matches the set administrator's number.

Further, it should be noted that there might be cases where an SMS message is received with a considerable delay (due to technical glitches at the operator's side, etc.). In such cases, you cannot take into account the command received in the message as the market situation could have changed. In view of this, we introduce another input parameter:

```
input int            inp_sms_max_old=600; //SMS command expiration, sec
```

The value of 600 suggests that commands that took more than 600 seconds (10 minutes) to be delivered, will be ignored. Please note that the method of checking the delivery time used in the example implies that the SMS service center and the device on which the Expert Advisor is running are situated in the same time zone.

To process SMS commands, let's add the following code to the **IncomingSMS()** function:

```
   if(sms.sender==inp_admin_number)
   {
      Print("SMS from the administrator");
      datetime t = TimeLocal();
      //--- message expiration check
      if(t-sms.scts.time<=inp_sms_max_old)
      {//--- check if the message is a command
         string cmdstr = sms.text;
         StringToUpper(cmdstr);//convert everything to upper case
         int pos = StringFind(cmdstr, "CLOSE", 0);
         cmdstr = StringSubstr(cmdstr, pos+6, 6);
         if(pos>=0)
         {//--- command. send it for processing
            ClosePositions(cmdstr);
         }
      }
      else
         Print("The SMS command has expired");
   }
```

If the SMS message was delivered from the administrator, it has not expired and represents a command (contains the keyword "Close"), we send its parameter for processing by the **ClosePositions()** function:

```
uint ClosePositions(string sstr)
{//--- close the specified positions
   bool all = false;
   if(StringFind(sstr, "ALL", 0)>=0)
      all = true;
   uint res = 0;
   for(int i=0;i<PositionsTotal();i++)
   {
      string symbol = PositionGetSymbol(i);
      if(all==true || sstr==symbol)
      {
         if(PositionSelect(symbol)==true)
         {
            long pos_type;
            double pos_vol;
            if(PositionGetInteger(POSITION_TYPE,pos_type)==true)
            {
               if(PositionGetDouble(POSITION_VOLUME,pos_vol)==true)
               {
                  if(OrderClose(symbol, (ENUM_POSITION_TYPE)pos_type, pos_vol)==true)
                     res|=0x01;
                  else
                     res|=0x02;
               }
            }
         }
      }
   }
   return(res);
}
```

This function checks whether any open position is a match in terms of the parameter (symbol) received in the command. Positions that satisfy this condition are closed using the **OrderClose()** function:

```
bool OrderClose(string symbol, ENUM_POSITION_TYPE pos_type, double vol)
{
   MqlTick last_tick;
   MqlTradeRequest request;
   MqlTradeResult result;
   double price = 0;
   //
   ZeroMemory(request);
   ZeroMemory(result);
   //
   if(SymbolInfoTick(Symbol(),last_tick))
   {
      price = last_tick.bid;
   }
   else
   {
      Print("Error when getting current prices");
      return(false);
   }
   //
   if(pos_type==POSITION_TYPE_BUY)
   {//--- closing a BUY position - SELL
      request.type = ORDER_TYPE_SELL;
   }
   else if(pos_type==POSITION_TYPE_SELL)
   {//--- closing a SELL position - BUY
      request.type = ORDER_TYPE_BUY;
   }
   else
      return(false);
   //
   request.price = NormalizeDouble(price, _Digits);
   request.deviation = 20;
   request.action = TRADE_ACTION_DEAL;
   request.symbol = symbol;
   request.volume = NormalizeDouble(vol, 2);
   if(request.volume==0)
      return(false);
   request.type_filling = ORDER_FILLING_FOK;
   //
   if(OrderSend(request, result)==true)
   {
      if(result.retcode==TRADE_RETCODE_DONE || result.retcode==TRADE_RETCODE_DONE_PARTIAL)
      {
         Print("Order executed successfully");
         return(true);
      }
   }
   else
   {
      Print("Order parameter error: ", GetLastError(),", Trade server return code: ", result.retcode);
      return(false);
   }
   //
   return(false);
}
```

After successful processing of orders, the function for monitoring position changes will generate and send an SMS notification.

### 9\. Deleting Messages from the Modem Memory

Please note that the modem handler does not delete incoming SMS messages on its own. Therefore, when the SMS memory gets full over the course of time, the handler will call the **SMSMemoryFull()** function and pass to it the current number of messages in the modem memory. You can delete them all or do it on a selective basis. The modem will not accept any new messages until the memory is released.

```
//+------------------------------------------------------------------+
//| SMS memory is full                                               |
//+------------------------------------------------------------------+
void SMSMemoryFull(int n)
{
   sms_mem_full = true;
   for(int i=0; i<n; i++)
   {//delete all SMS messages
      if(DelSMSbyIndex(i)==false)
         break;
      else
         sms_mem_full = false;
   }
}
```

You can also delete SMS messages right after they have been processed. When the **IncomingSMS()** function is called by the modem handler, the **INCOMING\_SMS\_STR** structure passes the index of the message in the modem memory, which allows deleting the message right after processing, using the **DelSMSbyIndex()** function:

### Conclusion

This article has dealt with the development of the Expert Advisor that uses a GSM modem to remotely monitor the trading terminal. We have considered the methods for getting information about open positions, current profit and other data using SMS notifications. We have also implemented the basic functions for open position management using SMS commands. The example provided features commands in English but you can use the Russian commands equally well (not to waste time on switching between different keyboard layouts in your phone).

Finally, let's check the behavior of our Expert Advisor when dealing with an old mobile phone released to market over 10 years ago. Device - **Siemens M55**. Let's connect it:

![The parameters of Siemens M55](https://c.mql5.com/2/6/en_13.png)![Siemens M55](https://c.mql5.com/2/6/m55.png)

Fig. 13. Connecting Siemens M55

![Siemens M55, "Expert Advisors" tab](https://c.mql5.com/2/6/en_14.png)

Fig. 14. Successful initialization of Siemens M55, "Expert Advisors" tab

You can see that all the necessary parameters have been obtained. The only problem is the data we get from USSD requests. The thing is that Siemens M55 does not support the AT command for working with USSD requests. Apart from that, its functionality is as good as that of any present-day modem, so it can be used for working with our Expert Advisor.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/797](https://www.mql5.com/ru/articles/797)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/797.zip "Download all attachments in the single ZIP archive")

[trcomport.zip](https://www.mql5.com/en/articles/download/797/trcomport.zip "Download trcomport.zip")(30.99 KB)

[modem.mqh](https://www.mql5.com/en/articles/download/797/modem.mqh "Download modem.mqh")(44.74 KB)

[sms.mqh](https://www.mql5.com/en/articles/download/797/sms.mqh "Download sms.mqh")(9.46 KB)

[gsminformer.mq5](https://www.mql5.com/en/articles/download/797/gsminformer.mq5 "Download gsminformer.mq5")(25.28 KB)

[comport.mqh](https://www.mql5.com/en/articles/download/797/comport.mqh "Download comport.mqh")(5.18 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Applying network functions, or MySQL without DLL: Part II - Program for monitoring changes in signal properties](https://www.mql5.com/en/articles/7495)
- [Applying network functions, or MySQL without DLL: Part I - Connector](https://www.mql5.com/en/articles/7117)
- [Using OpenCL to test candlestick patterns](https://www.mql5.com/en/articles/4236)
- [Drawing Dial Gauges Using the CCanvas Class](https://www.mql5.com/en/articles/1699)
- [Liquid Chart](https://www.mql5.com/en/articles/1208)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/20670)**
(14)


![Igor Makanu](https://c.mql5.com/avatar/2018/10/5BB56740-A283.jpg)

**[Igor Makanu](https://www.mql5.com/en/users/igorm)**
\|
17 Feb 2014 at 11:38

**decanium:** You send an SMS command to the Expert Advisor and thus stop it manually, until you find out.

That's why I have always respected real programmers - it's their unwillingness to look for a circuit solution.

Buy a GSM Switch and switch off the PC. ;)

Thanks for the article - informative!

![onewithzachy](https://c.mql5.com/avatar/2009/12/4B1A2389-5CDC.jpg)

**[onewithzachy](https://www.mql5.com/en/users/onewithzachy)**
\|
17 Feb 2014 at 16:45

How about CDMA, dude ? Same principle apply ? What need to be change ?

I used to have SMS sender standalone software that only works on GSM but not on CDMA.

![](https://c.mql5.com/3/30/smiley-bangheadonwall.gif)

![Hongming Huo](https://c.mql5.com/avatar/2020/4/5EA3B78B-00A7.jpg)

**[Hongming Huo](https://www.mql5.com/en/users/hmhuo)**
\|
6 Apr 2014 at 06:40

OMG! that's great!


![Matthew Todorovski](https://c.mql5.com/avatar/2014/9/5423F64B-633A.jpg)

**[Matthew Todorovski](https://www.mql5.com/en/users/bluepanther)**
\|
14 Nov 2014 at 07:44

Haha! Neato.

I also like the comment: "Welcome to Forex - the world of financial independence." I have yet to believe this...

![Sergey Kozlov](https://c.mql5.com/avatar/2020/5/5EC6F6F0-DE92.png)

**[Sergey Kozlov](https://www.mql5.com/en/users/gektor52)**
\|
28 Dec 2015 at 09:33

Hello, I don't work with mt5, so my question is: is there an EA for mt4?


![Upgrade to MetaTrader 4 Build 600 and Higher](https://c.mql5.com/2/13/1145_130.png)[Upgrade to MetaTrader 4 Build 600 and Higher](https://www.mql5.com/en/articles/1389)

The new version of the MetaTrader 4 terminal features the updated structure of user data storage. In earlier versions all programs, templates, profiles etc. were stored directly in terminal installation folder. Now all necessary data required for a particular user are stored in a separate directory called data folder. Read the article to find answers to frequently asked questions.

![MQL5 Programming Basics: Lists](https://c.mql5.com/2/0/Linked_List_MQL5.png)[MQL5 Programming Basics: Lists](https://www.mql5.com/en/articles/709)

The new version of the programming language for trading strategy development, MQL \[MQL5\], provides more powerful and effective features as compared with the previous version \[MQL4\]. The advantage essentially lies in the object-oriented programming features. This article looks into the possibility of using complex custom data types, such as nodes and lists. It also provides an example of using lists in practical programming in MQL5.

![Common Errors in MQL4 Programs and How to Avoid Them](https://c.mql5.com/2/13/1152_84.png)[Common Errors in MQL4 Programs and How to Avoid Them](https://www.mql5.com/en/articles/1391)

To avoid critical completion of programs, the previous version compiler handled many errors in the runtime environment. For example, division by zero or array out of range are critical errors and usually lead to program crash. The new compiler can detect actual or potential sources of errors and improve code quality. In this article, we discuss possible errors that can be detected during compilation of old programs and see how to fix them.

![Data Structure in MetaTrader 4 Build 600 and Higher](https://c.mql5.com/2/13/1143_38.png)[Data Structure in MetaTrader 4 Build 600 and Higher](https://www.mql5.com/en/articles/1388)

MetaTarder 4 build 600 features the new structure and location of the client terminal files. Now, MQL4 applications are placed in separate directories according to the program type (Expert Advisors, indicators or scripts). In most cases, the terminal data is now stored in a special data folder separated from the terminal installation location. In this article, we will describe in details how data is transferred, as well as the reasons for introducing the new storage system.

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/797&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068241643636717351)

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