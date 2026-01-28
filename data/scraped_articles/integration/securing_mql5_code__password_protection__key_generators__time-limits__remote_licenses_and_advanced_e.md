---
title: Securing MQL5 code: Password Protection, Key Generators, Time-limits, Remote Licenses and Advanced EA License Key Encryption Techniques
url: https://www.mql5.com/en/articles/359
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:20:49.330450
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/359&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071762846804553215)

MetaTrader 5 / Examples


### Introduction

Most developers need to have their code secured. This article will present a few different ways to protect MQL5 software. All examples in the article will refer to Expert Advisors but the same rules can be applied to Scripts and Indicators. The article starts with simple password protection and follows with key generators, licensing a given brokers account and time-limit protection. Then it introduces a remote license server concept. My [last article on MQL5-RPC framework](https://www.mql5.com/en/articles/342) described Remote Procedure Calls from [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/") to any XML-RPC server.

I will make use of this solution to provide an example of a remote license. I will also describe how to enhance this solution with base64 encoding and provide advice for PGP support to make ultra-secure protection for MQL5 Expert Advisors and Indicators. I am aware that [MetaQuotes Software Corp.](https://www.metaquotes.net/ "https://www.metaquotes.net/") is providing some options for licensing the code directly from the MQL5.com [Market section](https://www.mql5.com/en/market). This is really good for all developers and will not invalidate ideas presented in this article. Both solutions used together can only make the protection stronger and more secure against software theft.

### 1\. Password protection

Let's start with something simple. The first most used solution for a protection of computer software is password or license key protection. During first run after installation the user is queried with a dialog box to insert a password tied with a software copy (like the Microsoft Windows or Microsoft Office serial key) and if the entered password matches the user is allowed to use a single registered copy of a software. We can use an input variable or a direct textbox to enter the code. An example stub code is shown below.

The code below initializes a [CChartObjectEdit](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/obj_controls/cchartobjectedit) field that is be used to insert a password. There is a predefined array of allowed passwords that is matched against a password inserted by a user. Password is checked in [OnChartEvent()](https://www.mql5.com/en/docs/basis/function/events#onchartevent) method after receiving [CHARTEVENT\_OBJECT\_ENDEDIT](https://www.mql5.com/en/docs/constants/chartconstants/enum_chartevents) event.

```
//+------------------------------------------------------------------+
//|                                          PasswordProtectedEA.mq5 |
//|                                      Copyright 2012, Investeo.pl |
//|                                           http://www.investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, Investeo.pl"
#property link      "http://www.investeo.pl"
#property version   "1.00"

#include <ChartObjects/ChartObjectsTxtControls.mqh>

CChartObjectEdit password_edit;

const string allowed_passwords[] = { "863H-6738725-JG76364",
                             "145G-8927523-JG76364",
                             "263H-7663233-JG76364" };

int    password_status = -1;
string password_message[] = { "WRONG PASSWORD. Trading not allowed.",
                         "EA PASSWORD verified." };

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   password_edit.Create(0, "password_edit", 0, 10, 10, 260, 25);
   password_edit.BackColor(White);
   password_edit.BorderColor(Black);
   password_edit.SetInteger(OBJPROP_SELECTED, 0, true);
//---
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   password_edit.Delete();
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
  if (password_status>0)
  {
    // password correct
  }
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   if (id == CHARTEVENT_OBJECT_ENDEDIT && sparam == "password_edit" )
      {
         password_status = -1;

         for (int i=0; i<ArraySize(allowed_passwords); i++)
            if (password_edit.GetString(OBJPROP_TEXT) == allowed_passwords[i])
            {
               password_status = i;
               break;
            }

         if (password_status == -1)
            password_edit.SetString(OBJPROP_TEXT, 0, password_message[0]);
         else
            password_edit.SetString(OBJPROP_TEXT, 0, password_message[1]);
      }
  }
//+------------------------------------------------------------------+
```

This method is simple but is vurnelable for someone to publish the password on a website with hacked serial numbers. EA author cannot do anything until a new Expert Advisor is released and the stolen password is blacklisted.

### 2\. Key generator

Key generators are a mechanism that allows to use a set of passwords based on predefined rules. I will give an overview by providing a stub for a keygenerator below. In the example presented below the key must consist of three numbers separated by two hyphens. Therefore the allowed format for a password is XXXXX-XXXXX-XXXXX.

First number must be divisible by 3, second number must be divisible by 4 and third number must be divisible by 5. Therefore the allowed passwords may be 3-4-5, 18000-20000-20000 or the more complicated one 3708-102792-2844770.

```
//+------------------------------------------------------------------+
//|                                      KeyGeneratorProtectedEA.mq5 |
//|                                      Copyright 2012, Investeo.pl |
//|                                           http://www.investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, Investeo.pl"
#property link      "http://www.investeo.pl"
#property version   "1.00"

#include <ChartObjects/ChartObjectsTxtControls.mqh>
#include <Strings/String.mqh>

CChartObjectEdit password_edit;
CString user_pass;

const double divisor_sequence[] = { 3.0, 4.0, 5.0 };

int    password_status = -1;
string password_message[] = { "WRONG PASSWORD. Trading not allowed.",
                         "EA PASSWORD verified." };

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   password_edit.Create(0, "password_edit", 0, 10, 10, 260, 25);
   password_edit.BackColor(White);
   password_edit.BorderColor(Black);
   password_edit.SetInteger(OBJPROP_SELECTED, 0, true);
//---
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   password_edit.Delete();
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
  if (password_status==3)
  {
    // password correct
  }
  }

//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   if (id == CHARTEVENT_OBJECT_ENDEDIT && sparam == "password_edit" )
      {
         password_status = 0;

         user_pass.Assign(password_edit.GetString(OBJPROP_TEXT));

         int hyphen_1 = user_pass.Find(0, "-");
         int hyphen_2 = user_pass.FindRev("-");

         if (hyphen_1 == -1 || hyphen_2 == -1 || hyphen_1 == hyphen_2) {
            password_edit.SetString(OBJPROP_TEXT, 0, password_message[0]);
            return;
         } ;

         long pass_1 = StringToInteger(user_pass.Mid(0, hyphen_1));
         long pass_2 = StringToInteger(user_pass.Mid(hyphen_1 + 1, hyphen_2));
         long pass_3 = StringToInteger(user_pass.Mid(hyphen_2 + 1, StringLen(user_pass.Str())));

         // PrintFormat("%d : %d : %d", pass_1, pass_2, pass_3);

         if (MathIsValidNumber(pass_1) && MathMod((double)pass_1, divisor_sequence[0]) == 0.0) password_status++;
         if (MathIsValidNumber(pass_2) && MathMod((double)pass_2, divisor_sequence[1]) == 0.0) password_status++;
         if (MathIsValidNumber(pass_3) && MathMod((double)pass_3, divisor_sequence[2]) == 0.0) password_status++;

         if (password_status != 3)
            password_edit.SetString(OBJPROP_TEXT, 0, password_message[0]);
         else
            password_edit.SetString(OBJPROP_TEXT, 0, password_message[1]);
      }
  }
//+------------------------------------------------------------------+
```

Of course the number of digits in a number can be set to a given value and the calculations can be more complicated. One might also add a variable that is valid only with a given hardware by adding HDD serial number or CPU ID to calculation. In such case person to run the EA would have to run additional generator calculated on basis of the hardware.

The output would be an input to a keygen and the generated password would be valid only for a given hardware. This has a limitation of someone changing computer hardware or using VPS for running EA but this could be resolved by giving away two or three valid passwords. This is also the case in the [Market](https://www.mql5.com/en/market) section of MQL5 website.

### 3\. Single Account License

Since the Account number of the terminal of any given broker is unique this can be used to allow usage of the EA on one or a set of account numbers. In such case it is enough to use [AccountInfoString(ACCOUNT\_COMPANY)](https://www.mql5.com/en/docs/account/accountinfostring) and [AccountInfoInteger(ACCOUNT\_LOGIN)](https://www.mql5.com/en/docs/account/accountinfointeger) methods to fetch the account data and compare it against precompiled allowed values:

```
//+------------------------------------------------------------------+
//|                                           AccountProtectedEA.mq5 |
//|                                      Copyright 2012, Investeo.pl |
//|                                           http://www.investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, Investeo.pl"
#property link      "http://www.investeo.pl"
#property version   "1.00"

const string allowed_broker = "MetaQuotes Software Corp.";
const long allowed_accounts[] = { 979890, 436290, 646490, 225690, 279260 };

int password_status = -1;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   string broker = AccountInfoString(ACCOUNT_COMPANY);
   long account = AccountInfoInteger(ACCOUNT_LOGIN);

   printf("The name of the broker = %s", broker);
   printf("Account number =  %d", account);

   if (broker == allowed_broker)
      for (int i=0; i<ArraySize(allowed_accounts); i++)
       if (account == allowed_accounts[i]) {
         password_status = 1;
         Print("EA account verified");
         break;
       }
   if (password_status == -1) Print("EA is not allowed to run on this account.");

//---
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
  if (password_status == 1)
  {
    // password correct
  }
  }
```

This is a simple but quite powerful protection. The disadvantage is that it is necessary to recompile the EA for each new account number added to account database.

### 4\. Time-limit Protection

Time-limit protection is useful when the license is granted on temporary basis, for example using trial version of the software or when the license is granted on monthly or yearly basis. Since this is obvious that this can be applied for Expert Advisors and Indicators.

First idea is to check server time and based on that let the user use the Indicator or Expert Advisor within given period of time. After it expires the licensor is able to partially or totally disable its functionality to the licensee.

```
//+------------------------------------------------------------------+
//|                                         TimeLimitProtectedEA.mq5 |
//|                                      Copyright 2012, Investeo.pl |
//|                                           http://www.investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, Investeo.pl"
#property link      "http://www.investeo.pl"
#property version   "1.00"

datetime allowed_until = D'2012.02.11 00:00';

int password_status = -1;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   printf("This EA is valid until %s", TimeToString(allowed_until, TIME_DATE|TIME_MINUTES));
   datetime now = TimeCurrent();

   if (now < allowed_until)
         Print("EA time limit verified, EA init time : " + TimeToString(now, TIME_DATE|TIME_MINUTES));


//---
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
  if (TimeCurrent() < allowed_until)
    {
    }
   else Print("EA expired.");
  }
```

The only drawback is that that the solution would need to be have to be compiled separately for each licensee.

### 5\. Remote licenses

Wouldn't it be good to have a total control wheter to disable the license or extend the trial period on per user basis? This can simply be done using MQL5-RPC call that would send a query with the account name and receive the value whether to run the script in trial mode or disable it.

Please see the code below for a sample implementation:

```
from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler

class RequestHandler( SimpleXMLRPCRequestHandler ):
    rpc_path = ( '/RPC2', )


class RemoteLicenseExample(SimpleXMLRPCServer):

    def __init__(self):
        SimpleXMLRPCServer.__init__( self, ("192.168.2.103", 9099), requestHandler=RequestHandler, logRequests = False)

        self.register_introspection_functions()
        self.register_function( self.xmlrpc_isValid, "isValid" )

        self.licenses = {}

    def addLicense(self, ea_name, broker_name, account_number):
        if ea_name in self.licenses:
            self.licenses[ea_name].append({ 'broker_name': broker_name, 'account_number' : account_number })
        else:
            self.licenses[ea_name] = [ { 'broker_name': broker_name, 'account_number' : account_number } ]

    def listLicenses(self):
        print self.licenses

    def xmlrpc_isValid(self, ea_name, broker_name, account_number):
        isValidLicense = False

        ea_name = str(ea_name)
        broker_name = str(broker_name)

        print "Request for license", ea_name, broker_name, account_number

        try:
            account_number = int(account_number)
        except ValueError as error:
            return isValidLicense

        if ea_name in self.licenses:
            for license in self.licenses[ea_name]:
                if license['broker_name'] == broker_name and license['account_number'] == account_number:
                    isValidLicense = True
                    break

        print "License valid:", isValidLicense

        return isValidLicense

if __name__ == '__main__':
    server = RemoteLicenseExample()
    server.addLicense("RemoteProtectedEA", "MetaQuotes Software Corp.", 1024221)
    server.addLicense("RemoteProtectedEA", "MetaQuotes Software Corp.", 1024223)

    server.listLicenses()
    server.serve_forever()
```

This is a simple XML-RPC server implemented in Python with two predefined licenses for [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/"). The licenses are set for "RemoteProtectedEA" expert advisor running on default MetaQuotes demo server (access.metatrader5.com:443) with account numbers 1024221 and 1024223. An industrial solution would probable make use of a license database in Postgresql or any other database but the example above is more than enough for this article since it handles the remote licenses very well.

If you need a short explanation on how to install Python please read ["MQL5-RPC. Remote Procedure Calls from MQL5: Web Service Access and XML-RPC ATC Analyzer for Fun and Profit"](https://www.mql5.com/en/articles/342).

The EA that uses the remote license simply needs to prepare a remote MQL5-RPC call to isValid() method that returns true or false boolean values depending if the license is valid. The example below shows a sample EA that is based on account protection:

```
//+------------------------------------------------------------------+
//|                                            RemoteProtectedEA.mq5 |
//|                                      Copyright 2012, Investeo.pl |
//|                                           http://www.investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, Investeo.pl"
#property link      "http://www.investeo.pl"
#property version   "1.00"

#include <MQL5-RPC.mqh>
#include <Arrays\ArrayObj.mqh>
#include <Arrays\ArrayInt.mqh>
#include <Arrays\ArrayString.mqh>
#include <Arrays\ArrayBool.mqh>

bool license_status=false;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
/* License proxy server */
   CXMLRPCServerProxy s("192.168.2.103:9099");
   if(s.isConnected()==true)
     {

      CXMLRPCResult *result;

/* Get account data */
      string broker= AccountInfoString(ACCOUNT_COMPANY);
      long account = AccountInfoInteger(ACCOUNT_LOGIN);

      printf("The name of the broker = %s",broker);
      printf("Account number =  %d",account);

/* Get remote license status */
      CArrayObj* params= new CArrayObj;
      CArrayString* ea = new CArrayString;
      CArrayString* br = new CArrayString;
      CArrayInt *ac=new CArrayInt;

      ea.Add("RemoteProtectedEA");
      br.Add(broker);
      ac.Add((int)account);

      params.Add(ea); params.Add(br); params.Add(ac);

      CXMLRPCQuery query("isValid",params);

      result=s.execute(query);

      CArrayObj *resultArray=result.getResults();
      if(resultArray!=NULL && resultArray.At(0).Type()==TYPE_BOOL)
        {
         CArrayBool *stats=resultArray.At(0);

         license_status=stats.At(0);
        }
      else license_status=false;

      if(license_status==true) printf("License valid.");
      else printf("License invalid.");

      delete params;
      delete result;
     }
   else Print("License server not connected.");
//---
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   if(license_status==true)
     {
      // license valid
     }
  }
//+------------------------------------------------------------------+
```

If you execute both scripts you should be able to add a remote license for your account number. The remote license can be used as well for time-limit license or a password license that can be remotely deactivated after a trial period. For example you would give an EA for someone for 10 days testing, if he does not satisifed with the product you deactivate the license or in the case he is satisfied you can activate the license for any given period of time.

### 6\. Secure License Encryption

The ideas presented in the last paragraph used Remote Procedure Calls to exchange information between license server and client terminal. This could be possibly hacked by using sniffer packages on a registered copy of the EA. By using sniffer application hacker is able to capture all TCP packets that are sent between two machines. We will overcome this problem by using base64 encoding for sending account data and receive encrypted message.

For a skilled person it would be also possible to use PGP and/or put all the code in a DLL for further protection. I came up with the idea that the message will be in fact another RPC message (like in [Russian Matryoshka doll](https://en.wikipedia.org/wiki/Matryoshka_doll "https://en.wikipedia.org/wiki/Matryoshka_doll")) that will be further converted into MQL5 data.

The first step is to add base64 encoding and decoding support for MQL5-RPC. Luckily this was already done for [MetaTrader 4](https://www.metatrader4.com/ "https://www.metatrader4.com/") at [https://www.mql5.com/en/code/8098](https://www.mql5.com/en/code/8098) by Renat therefore I only needed to convert it to MQL5.

```
//+------------------------------------------------------------------+
//|                                                       Base64.mq4 |
//|                      Copyright © 2006, MetaQuotes Software Corp. |
//|                                  MT5 version © 2012, Investeo.pl |
//|                                        https://www.metaquotes.net |
//+------------------------------------------------------------------+
#property copyright "Copyright Â© 2006, MetaQuotes Software Corp."
#property link      "https://www.metaquotes.net"

static uchar ExtBase64Encode[64]={ 'A','B','C','D','E','F','G','H','I','J','K','L','M',
                                 'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                                 'a','b','c','d','e','f','g','h','i','j','k','l','m',
                                 'n','o','p','q','r','s','t','u','v','w','x','y','z',
                                 '0','1','2','3','4','5','6','7','8','9','+','/'      };

static uchar ExtBase64Decode[256]={
                    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
                    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
                    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  62,  -1,  -1,  -1,  63,
                    52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  -1,  -1,  -1,  -2,  -1,  -1,
                    -1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
                    15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  -1,  -1,  -1,  -1,  -1,
                    -1,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,
                    41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  -1,  -1,  -1,  -1,  -1,
                    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
                    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
                    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
                    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
                    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
                    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
                    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
                    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1 };


void Base64Encode(string in,string &out)
  {
   int i=0,pad=0,len=StringLen(in);

   while(i<len)
     {

      int b3,b2,b1=StringGetCharacter(in,i);
      i++;
      if(i>=len) { b2=0; b3=0; pad=2; }
      else
        {
         b2=StringGetCharacter(in,i);
         i++;
         if(i>=len) { b3=0; pad=1; }
         else       { b3=StringGetCharacter(in,i); i++; }
        }
      //----
      int c1=(b1 >> 2);
      int c2=(((b1 & 0x3) << 4) | (b2 >> 4));
      int c3=(((b2 & 0xf) << 2) | (b3 >> 6));
      int c4=(b3 & 0x3f);

      out=out+CharToString(ExtBase64Encode[c1]);
      out=out+CharToString(ExtBase64Encode[c2]);
      switch(pad)
        {
         case 0:
           out=out+CharToString(ExtBase64Encode[c3]);
           out=out+CharToString(ExtBase64Encode[c4]);
           break;
         case 1:
           out=out+CharToString(ExtBase64Encode[c3]);
           out=out+"=";
           break;
         case 2:
           out=out+"==";
           break;
        }
     }
//----
  }

void Base64Decode(string in,string &out)
  {
   int i=0,len=StringLen(in);
   int shift=0,accum=0;

   while(i<len)
     {
      int value=ExtBase64Decode[StringGetCharacter(in,i)];
      if(value<0 || value>63) break;

      accum<<=6;
      shift+=6;
      accum|=value;
      if(shift>=8)
        {
         shift-=8;
         value=accum >> shift;
         out=out+CharToString((uchar)(value & 0xFF));
        }
      i++;
     }
//----
  }
//+------------------------------------------------------------------+
```

For a detailed description of base64 encoding you may want to visit a [Wikipedia article](https://en.wikipedia.org/wiki/Base64 "https://en.wikipedia.org/wiki/Base64").

A sample test of MQL5 base64 coding and decoding script is presented below:

```
//+------------------------------------------------------------------+
//|                                                   Base64Test.mq5 |
//|                                      Copyright 2012, Investeo.pl |
//|                                           http://www.investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, Investeo.pl"
#property link      "http://www.investeo.pl"
#property version   "1.00"

#include <Base64.mqh>

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   string to_encode = "<test>Abrakadabra</test>";

   string encoded;
   string decoded;

   Base64Encode(to_encode, encoded);

   Print(encoded);

   Base64Decode(encoded, decoded);

   Print(decoded);

  }
//+------------------------------------------------------------------+
```

The script produces the following result.

```
DK      0       Base64Test (EURUSD,H1)  16:21:13        Original string: <test>Abrakadabra</test>
PO      0       Base64Test (EURUSD,H1)  16:21:13        Base64 encoded string: PHRlc3Q+QWJyYWthZGFicmE8L3Rlc3Q+
FM      0       Base64Test (EURUSD,H1)  16:21:13        Base64 decoded string: <test>Abrakadabra</test>
```

The validity of encoding can be simply checked in Python in 4 lines of code:

```
import base64

encoded = 'PHRlc3Q+QWJyYWthZGFicmE8L3Rlc3Q+'
decoded = base64.b64decode(encoded)
print decoded

<test>Abrakadabra</test>
```

The seconds step is to encrypt XMLRPC result in base64 (aka Matryoshka technique):

```
import base64
from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler

class RequestHandler( SimpleXMLRPCRequestHandler ):
    rpc_path = ( '/RPC2', )


class RemoteLicenseExampleBase64(SimpleXMLRPCServer):

    def __init__(self):
        SimpleXMLRPCServer.__init__( self, ("192.168.2.103", 9099), requestHandler=RequestHandler, logRequests = False)

        self.register_introspection_functions()
        self.register_function( self.xmlrpc_isValid, "isValid" )

        self.licenses = {}

    def addLicense(self, ea_name, broker_name, account_number):
        if ea_name in self.licenses:
            self.licenses[ea_name].append({ 'broker_name': broker_name, 'account_number' : account_number })
        else:
            self.licenses[ea_name] = [ { 'broker_name': broker_name, 'account_number' : account_number } ]

    def listLicenses(self):
        print self.licenses

    def xmlrpc_isValid(self, ea_name, broker_name, account_number):
        isValidLicense = False

        ea_name = str(ea_name)
        broker_name = str(broker_name)

        print "Request for license", ea_name, broker_name, account_number

        try:
            account_number = int(account_number)
        except ValueError as error:
            return isValidLicense

        if ea_name in self.licenses:
            for license in self.licenses[ea_name]:
                if license['broker_name'] == broker_name and license['account_number'] == account_number:
                    isValidLicense = True
                    break

        print "License valid:", isValidLicense

        # additional xml encoded with base64
        xml_response = "<?xml version='1.0'?><methodResponse><params><param><value><boolean>%d</boolean></value></param></params></methodResponse>"

        retval = xml_response % int(isValidLicense)

        return base64.b64encode(retval)

if __name__ == '__main__':
    server = RemoteLicenseExampleBase64()
    server.addLicense("RemoteProtectedEA", "MetaQuotes Software Corp.", 1024221)
    server.addLicense("RemoteProtectedEA", "MetaQuotes Software Corp.", 1024223)

    server.listLicenses()
    server.serve_forever()
```

After the license is encrypted we can use MQL5-RPC method for converting decrypted message back into MQL5 data:

```
//+------------------------------------------------------------------+
//|                                      RemoteProtectedEABase64.mq5 |
//|                                      Copyright 2012, Investeo.pl |
//|                                           http://www.investeo.pl |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, Investeo.pl"
#property link      "http://www.investeo.pl"
#property version   "1.00"

#include <MQL5-RPC.mqh>
#include <Arrays\ArrayObj.mqh>
#include <Arrays\ArrayInt.mqh>
#include <Arrays\ArrayString.mqh>
#include <Arrays\ArrayBool.mqh>
#include <Base64.mqh>

bool license_status=false;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
/* License proxy server */
   CXMLRPCServerProxy s("192.168.2.103:9099");

   if(s.isConnected()==true)
     {
      CXMLRPCResult *result;

/* Get account data */
      string broker= AccountInfoString(ACCOUNT_COMPANY);
      long account = AccountInfoInteger(ACCOUNT_LOGIN);

      printf("The name of the broker = %s",broker);
      printf("Account number =  %d",account);

/* Get remote license status */
      CArrayObj* params= new CArrayObj;
      CArrayString* ea = new CArrayString;
      CArrayString* br = new CArrayString;
      CArrayInt *ac=new CArrayInt;

      ea.Add("RemoteProtectedEA");
      br.Add(broker);
      ac.Add((int)account);

      params.Add(ea); params.Add(br); params.Add(ac);

      CXMLRPCQuery query("isValid",params);

      result=s.execute(query);

      CArrayObj *resultArray=result.getResults();
      if(resultArray!=NULL && resultArray.At(0).Type()==TYPE_STRING)
        {
         CArrayString *stats=resultArray.At(0);

         string license_encoded=stats.At(0);

         printf("encoded license: %s",license_encoded);

         string license_decoded;

         Base64Decode(license_encoded,license_decoded);

         printf("decoded license: %s",license_decoded);

         CXMLRPCResult license(license_decoded);
         resultArray=license.getResults();

         CArrayBool *bstats=resultArray.At(0);

         license_status=bstats.At(0);
        }
      else license_status=false;

      if(license_status==true) printf("License valid.");
      else printf("License invalid.");

      delete params;
      delete result;
     }
   else Print("License server not connected.");

//---
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   if(license_status==true)
     {
      // license valid
     }
  }
//+------------------------------------------------------------------+
```

The result of running the script provided that RemoteLicenseExampleBase64 server is running is as follows:

```
KI  0  RemoteProtectedEABase64 (EURUSD,H1) 19:47:57  The name of the broker = MetaQuotes Software Corp.
GP  0  RemoteProtectedEABase64 (EURUSD,H1) 19:47:57  Account number =  1024223
EM  0  RemoteProtectedEABase64 (EURUSD,H1) 19:47:57  <?xml version='1.0'?><methodResponse><params><param><value><string>PD94bWwgdmVyc2lvbj0nMS4wJz8+PG1ldGhvZFJlc3BvbnNlPjxwYXJhbXM+PHBhcmFtPjx2YWx1ZT48Ym9vbGVhbj4xPC9ib29sZWFuPjwvdmFsdWU+PC9wYXJhbT48L3BhcmFtcz48L21ldGhvZFJlc3BvbnNlPg==</string></value></param></params></methodResponse>
DG  0  RemoteProtectedEABase64 (EURUSD,H1) 19:47:57  encoded license: PD94bWwgdmVyc2lvbj0nMS4wJz8+PG1ldGhvZFJlc3BvbnNlPjxwYXJhbXM+PHBhcmFtPjx2YWx1ZT48Ym9vbGVhbj4xPC9ib29sZWFuPjwvdmFsdWU+PC9wYXJhbT48L3BhcmFtcz48L21ldGhvZFJlc3BvbnNlPg==
FL  0  RemoteProtectedEABase64 (EURUSD,H1) 19:47:57  decoded license: <?xml version='1.0'?><methodResponse><params><param><value><boolean>1</boolean></value></param></params></methodResponse>
QL  0  RemoteProtectedEABase64 (EURUSD,H1) 19:47:57  License valid.
```

As you can see, the XML-RPC payload contains a string that is in fact a XML-RPC message encoded by base64. This base64 encoded message is decoded into XML string and later decoded into MQL5 data.

### 7\. Advanced Anti-Decompilation Guidelines

As soon as MQL5 code gets decompiled even the most secure protections that are exposed to skilled reverse-engineer will be vulnerable for being cracked. After some googling I found a website that offers MQL5 decompiler but I simply suspect that this is a fake one made to take away money from naive people that would like to steal someone's code. Anyway, I did not try it and I might be wrong. Even if such solution existed you should be able to make a stronger protection by sending encrypted EA/indicator input parameters or object indexes passing.

It will be very hard for a hacker to obtain correct input parameters for the protected EA or see correct input values of the protected indicator which in turn will make it useless. It is also possible to send correct parameters if account ID match or send unencrypted fake parameters if account ID is not valid. For that solution one may want to use PGP (Pretty Good privacy). Even if the code is decompiled, data will be sent encrypted with private PGP key and EA parameters will be decrypted only when account ID and PGP key match.

### Conclusion

In this article I presented a few ways to protect MQL5 code. I also introduced remote license concept via MQL5-RPC call and added base64 encoding support. I hope the article will serve as a basis to further ideas on how to secure MQL5 code. All source code is attached to the article.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/359.zip "Download all attachments in the single ZIP archive")

[simpleremotelicenseservers.zip](https://www.mql5.com/en/articles/download/359/simpleremotelicenseservers.zip "Download simpleremotelicenseservers.zip")(2.01 KB)

[license\_examples\_mql5.zip](https://www.mql5.com/en/articles/download/359/license_examples_mql5.zip "Download license_examples_mql5.zip")(24.45 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5-RPC. Remote Procedure Calls from MQL5: Web Service Access and XML-RPC ATC Analyzer for Fun and Profit](https://www.mql5.com/en/articles/342)
- [Applying The Fisher Transform and Inverse Fisher Transform to Markets Analysis in MetaTrader 5](https://www.mql5.com/en/articles/303)
- [Advanced Adaptive Indicators Theory and Implementation in MQL5](https://www.mql5.com/en/articles/288)
- [Using MetaTrader 5 Indicators with ENCOG Machine Learning Framework for Timeseries Prediction](https://www.mql5.com/en/articles/252)
- [Exposing C# code to MQL5 using unmanaged exports](https://www.mql5.com/en/articles/249)
- [Moving Mini-Max: a New Indicator for Technical Analysis and Its Implementation in MQL5](https://www.mql5.com/en/articles/238)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/6204)**
(51)


![Carl Schreiber](https://c.mql5.com/avatar/2018/2/5A745EEE-EB76.PNG)

**[Carl Schreiber](https://www.mql5.com/en/users/gooly)**
\|
15 Dec 2022 at 11:15

Deepl says this: https://www.deepl.com/translator#en/de/License%20server%20not%20connected

The question now is why? Is it running? Protected? ...

![MENAHEL1](https://c.mql5.com/avatar/2023/10/651b935e-9dcc.jpg)

**[MENAHEL1](https://www.mql5.com/en/users/menahel1)**
\|
15 Dec 2023 at 21:46

Hello, how can I [put](https://www.mql5.com/en/articles/1171 "Why Virtual Hosting on the MetaTrader 4 and MetaTrader 5 platforms is better than the usual VPSs") item 7 in my EA to avoid decompiling the ex.5 file?

Do you have code examples to avoid this?

Could you explain in detail?

![Winged Trading](https://c.mql5.com/avatar/2023/9/64f751e2-5777.png)

**[Winged Trading](https://www.mql5.com/en/users/wingedtrading)**
\|
20 Aug 2025 at 20:13

Thank you for this article.

In the end, this debate is about: "how much time can I gain, before the product gets cracked". Aiming to make it so expensive to crack, that its not worth it anymore.

A method one could implement, is code obfuscation. Renaming every variable and method with some random name. double Signal would become double AB1234, double IndicatorValue would become CD1234. It doesn't solve the problem, but sure makes it a headache for the person that decompiles the code. It makes it really hard to find where the licencing check is done to potentially change it.

Another solution would be to handle everything internally, every indicator, every expert advisor gets calculated only on the owners server. The user sends licencing, bar data, [server time](https://www.mql5.com/en/docs/dateandtime/timetradeserver "MQL5 documentation: TimeTradeServer function"), etc to the server. The server responds with an indicator value or EA action. This on itself would be quite expensive and time consuming to implement, and solves most of the issues.

I Hope this sparked some curiosity and could be of any help.

![Miguel Angel Vico Alba](https://c.mql5.com/avatar/2025/10/68e99f33-714e.jpg)

**[Miguel Angel Vico Alba](https://www.mql5.com/en/users/mike_explosion)**
\|
20 Aug 2025 at 21:58

**Winged Trading [#](https://www.mql5.com/en/forum/6204/page5#comment_57852941):**

Just as a small note: in MQL5 the compiler already strips out variable names, function names and comments when generating the .ex5.

The executable is therefore already quite "opaque", so obfuscation by renaming identifiers doesn't really add an extra layer of protection here.

![Winged Trading](https://c.mql5.com/avatar/2023/9/64f751e2-5777.png)

**[Winged Trading](https://www.mql5.com/en/users/wingedtrading)**
\|
21 Aug 2025 at 06:24

**Miguel Angel Vico Alba [#](https://www.mql5.com/en/forum/6204/page5#comment_57853378):**

Just as a small note: in MQL5 the compiler already strips out variable names, function names and comments when generating the .ex5.

The executable is therefore already quite "opaque", so obfuscation by renaming identifiers doesn't really add an extra layer of protection here.

I was not aware of this. Thank you for elaborating.


![Promote Your Development Projects Using EX5 Libraries](https://c.mql5.com/2/0/Use_ex5_libraries.png)[Promote Your Development Projects Using EX5 Libraries](https://www.mql5.com/en/articles/362)

Hiding of the implementation details of classes/functions in an .ex5 file will enable you to share your know-how algorithms with other developers, set up common projects and promote them in the Web. And while the MetaQuotes team spares no effort to bring about the possibility of direct inheritance of ex5 library classes, we are going to implement it right now.

![The All or Nothing Forex Strategy](https://c.mql5.com/2/0/allVSzero.png)[The All or Nothing Forex Strategy](https://www.mql5.com/en/articles/336)

The purpose of this article is to create the most simple trading strategy that implements the "All or Nothing" gaming principle. We don't want to create a profitable Expert Advisor - the goal is to increase the initial deposit several times with the highest possible probability. Is it possible to hit the jackpot on ForEx or lose everything without knowing anything about technical analysis and without using any indicators?

![Trademinator 3: Rise of the Trading Machines](https://c.mql5.com/2/0/Terminator_3_Rise_of_the_Machines.png)[Trademinator 3: Rise of the Trading Machines](https://www.mql5.com/en/articles/350)

In the article "Dr. Tradelove..." we created an Expert Advisor, which independently optimizes parameters of a pre-selected trading system. Moreover, we decided to create an Expert Advisor that can not only optimize parameters of one trading system underlying the EA, but also select the best one of several trading systems. Let's see what can come of it...

![Using Discriminant Analysis to Develop Trading Systems](https://c.mql5.com/2/0/Discriminant_Analysis_MQL5.png)[Using Discriminant Analysis to Develop Trading Systems](https://www.mql5.com/en/articles/335)

When developing a trading system, there usually arises a problem of selecting the best combination of indicators and their signals. Discriminant analysis is one of the methods to find such combinations. The article gives an example of developing an EA for market data collection and illustrates the use of the discriminant analysis for building prognostic models for the FOREX market in Statistica software.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=rgjbfzyuzbchuwhmhztnretcnhhxbkjw&ssn=1769192448711828717&ssn_dr=0&ssn_sr=0&fv_date=1769192448&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F359&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Securing%20MQL5%20code%3A%20Password%20Protection%2C%20Key%20Generators%2C%20Time-limits%2C%20Remote%20Licenses%20and%20Advanced%20EA%20License%20Key%20Encryption%20Techniques%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919244807557051&fz_uniq=5071762846804553215&sv=2552)

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