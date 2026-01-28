---
title: Native Twitter Client: Part 2
url: https://www.mql5.com/en/articles/8318
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:05:26.383796
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/8318&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083327016184322421)

MetaTrader 5 / Examples


### Introduction

As promised in my first article ["Native Twitter Client for MT4 and MT5 without DLL"](https://www.mql5.com/en/articles/8270) following article will try to explore the Twitter API to send tweets with photos. To keep this article easy and simple to be understood I will focus on uploading image only. By the end of this article, you should have a working Twitter client without using any external DLL, which allow you to tweet messages with up to four photos. [The limit of 4 photos](https://www.mql5.com/go?link=https://developer.twitter.com/en/docs/twitter-api/v1/tweets/post-and-engage/api-reference/post-statuses-update "https://developer.twitter.com/en/docs/twitter-api/v1/tweets/post-and-engage/api-reference/post-statuses-update") is set by Twitter API as explained in the parameter _media\_ids_.

### Uploading photos

There is a new method, [called chunked upload](https://www.mql5.com/go?link=https://developer.twitter.com/en/docs/twitter-api/v1/media/upload-media/uploading-media/chunked-media-upload "https://developer.twitter.com/en/docs/twitter-api/v1/media/upload-media/uploading-media/chunked-media-upload"), to upload media with better methods to upload large files, e.g. videos or animated GIFs. For our purpose I will focus on the [simple method](https://www.mql5.com/go?link=https://developer.twitter.com/en/docs/twitter-api/v1/media/upload-media/api-reference/post-media-upload "https://developer.twitter.com/en/docs/twitter-api/v1/media/upload-media/api-reference/post-media-upload") which is limited to upload image only.

Please make sure you're familiar with [Twitter's media types and sizes restrictions](https://www.mql5.com/go?link=https://developer.twitter.com/en/docs/twitter-api/v1/media/upload-media/overview "https://developer.twitter.com/en/docs/twitter-api/v1/media/upload-media/overview").

Uploading photo to Twitter is basically a simple OAuth authorized HTTP multipart/form-data POST which I will explain in the next paragraph. Each photo uploaded will return a media\_id which is valid only within a certain amount of time to allow it to be included in a tweet to be publish.

To publish up to four photos, all returned media\_ids are simple joined together as a comma-separated list.

The steps to tweet a message with photos are as following:

1. Upload photo and collect its returned media\_id
2. Repeat uploading further photos, up to max. 4 photos. Always collect its returned media\_id.
3. Prepare your tweet message
4. Specify media\_ids when sending your tweet with the comma-separated list of all media\_id to be included.

**NOTE:**

To keep the code simple and easier to follow, error handling is omitted.

### HTTP multipart/form-data

To upload photos to Twitter, the photos can be uploaded as raw binary data or as Base64 encoded string. It is recommended to upload photos as raw binary data, since Base64 encoded string is about three times larger in size.

The HTTP multipart/form-data method is well-defined in [RFC-2388](https://www.mql5.com/go?link=https://www.ietf.org/rfc/rfc2388.txt "https://www.ietf.org/rfc/rfc2388.txt") but it might be easier to understand when reading this [nice tutorial of curl](https://www.mql5.com/go?link=https://ec.haxx.se/http/http-multipart "https://ec.haxx.se/http/http-multipart"). Basically, quoted from the mentioned article, "it is an HTTP POST request sent with the request body specially formatted as a series of "parts", separated with MIME boundaries."

```
POST /submit.cgi HTTP/1.1
Host: example.com
User-Agent: curl/7.46.0
Accept: */*
Content-Length: 313
Expect: 100-continue
Content-Type: multipart/form-data; boundary=------------------------d74496d66958873e

--------------------------d74496d66958873e
Content-Disposition: form-data; name="person"

anonymous
--------------------------d74496d66958873e
Content-Disposition: form-data; name="secret"; filename="file.txt"
Content-Type: text/plain

contents of the file
--------------------------d74496d66958873e--
```

The implementation in the Twitter class, is as follows:

```
   void              appendPhoto(string filename, string hash, uchar &data[],
                                 bool common_flag=false)
     {
      int flags=FILE_READ|FILE_BIN|FILE_SHARE_WRITE|FILE_SHARE_READ;
      if(common_flag)
         flags|=FILE_COMMON;
      //---
      int handle=FileOpen(filename,flags);
      if(handle==INVALID_HANDLE)
         return;
      //---
      int size=(int)FileSize(handle);
      uchar img[];
      ArrayResize(img,size);
      FileReadArray(handle,img,0,size);
      FileClose(handle);
      int pos = ArraySize(data);
      int offset = pos + size;
      int hlen = StringLen(hash)+6;
      int newlen = offset + hlen;
      ArrayResize(data, newlen);
      ArrayCopy(data, img, pos);
      StringToCharArray("\r\n--"+hash+"\r\n", data, offset, hlen);
     }
```

The above code adds the raw binary data of an image file to the "parts" of the HTTP multipart/form-data post. The "envelope" of the POST is done in the following code, with the Twitter Upload-API parameter " **media**" specified.

```
   string              uploadPhoto(string filename)
     {
      // POST multipart/form-data
      string url = "https://upload.twitter.com/1.1/media/upload.json";
      //string url = "https://httpbin.org/anything";
      string params[][2];
      string query = oauthRequest(params, url, "POST");
      string o = getOauth(params);
      string custom_headers = "Content-Type: multipart/form-data;"
                              " boundary=";
      string boundary = getNonce();
      StringAdd(custom_headers, boundary); // use nonce as boundary string
      StringAdd(custom_headers, "\r\n");
      string headers = getHeaders(o, custom_headers, "upload.twitter.com");

      //string query = getQuery(params, url);
      //Print(query);
      uchar data[];
      string part = "\r\n--";
      StringAdd(part, boundary);
      StringAdd(part, "\r\nContent-Disposition: form-data;"
                " name=\"media\"\r\n\r\n");
      StringToCharArray(part, data, 0, StringLen(part));
      appendPhoto(filename, boundary, data);
      string resp = SendRequest("POST", url, data, headers);;
      if(m_verbose)
        {
         SaveToFile(filename + "_post.txt", data);
         Print(resp);
        }
      return (getTokenValue(resp, "media_id"));
     }
```

To inspect and to verify the build HTTP multipart/form-data, the HTTP request can be saved as file in MT terminal's data folder for further inspection.

### Tweet with photos

For simple purpose, I use a simple function getTokenValue() to retrieve media\_id  returned by the Twitter Upload-API. You might want to consider using the excellent [JSON library available on MQL5.com](https://www.mql5.com/en/code/13663).

Following code shows a very simple way to use the Twitter class. The function Screenshots() simply takes screenshots of currently opened charts and builds a simple tweet message. Up to four charts are selected.

Each screenshot is saved as a file, its filename is returned in the string array fnames.

The screenshots are uploaded one-by-one, with its returned media\_id collected and gathered together as a comma-separated list.

By specifying media\_ids parameter with this above comma-separated list, we post the tweet message with these screenshots attached to the tweet.

As simple as that.

```
   CTwitter tw(consumer_key, consumer_secret,
               access_token, access_secret, verbose);

   // Gather information
   string fnames[4];
   string msg;
   Screenshots(fnames, msg);

   // Upload screenshots
   int n = ArraySize(fnames);
   int i = n - 1;
   // create comma separated media_ids
   string medias = tw.uploadPhoto(fnames[i]);
   for(i= n - 2; i>=0; i--)
   {
      StringAdd(medias, ",");
      StringAdd(medias, tw.uploadPhoto(fnames[i]));
   }

   // Send Tweet with photos' ids
   string resp = tw.tweet(msg, medias);
   Print(resp);
```

### Twitter class

The Twitter class as you can find in the Twitter.mqh is developed with the goal to be self-contained, independent of other include files. To send a tweet with photos that single file is all you need.

First you instantiate a Twitter object, specifying your consumer and access token, with an optionally verbose flag to help in debugging during development.

```
   CTwitter tw(consumer_key, consumer_secret,
               access_token, access_secret, verbose);
```

Then you can try to call avalilable publc functions:

- **verifyCredentials()**

returns the Twitter ID for your access token

- **uploadPhoto()**

upload a photo and returns its media\_id

- **tweet()**

send a tweet with optional media\_ids

There are some helper functions:

- **getTokenValue()**

return the value of a token/parameter from a json string.

**NOTE**: This is a very simple string parsing, do not expect full json compatibility.

- **unquote()**

remove quotes from a string.

### Tweet your charts

Attached is a working MT5 script that takes screenshot of up to four charts and build a simple tweet message of chart's symbol and OHLCV value.

It is a simple example for you to get started to develop your own experts and/or scripts.

**NOTE:** You must specify your own consumer and access tokens and secrets.

Following are examples of the tweets sent by the script.

![Tweet with photos sent from MT5 ](https://c.mql5.com/2/40/tweet0.png)

Figure 1. Tweet with photos sent from MT5

![Full size MT5 chart on Twitter ](https://c.mql5.com/2/40/tweet1.png)

Figure 2. Full size MT5 chart on Twitter

And of course you can also attach any photos ;)

![Tweet with lucky cats](https://c.mql5.com/2/40/tweet2.png)

Figure 3. Tweet with lucky cats

### Conclusion

A simple and easy to use Twitter class as a self-contained include file is provided for you to publish your charts and signals easily. The relevant technical details are presented in the hope that they can be easily understood.

This Twitter class is far from complete, there are many other Twitter APIs that can be added to this class. Feel free to post your improvements in the comment for the benefit of MQL5 community.

I hope you enjoy reading this article, as I have enjoyed writing it.

I wish you can use the code provided for fun and profit too.

Enjoy.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8318.zip "Download all attachments in the single ZIP archive")

[Twitter.mqh](https://www.mql5.com/en/articles/download/8318/twitter.mqh "Download Twitter.mqh")(19.77 KB)

[TwitterDemo.mq5](https://www.mql5.com/en/articles/download/8318/twitterdemo.mq5 "Download TwitterDemo.mq5")(4.87 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Native Twitter Client for MT4 and MT5 without DLL](https://www.mql5.com/en/articles/8270)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/349273)**
(2)


![Bo Bang Xian](https://c.mql5.com/avatar/2022/1/61E42100-2519.jpg)

**[Bo Bang Xian](https://www.mql5.com/en/users/bocai)**
\|
27 Sep 2020 at 17:58

Hello, God!

Can you tell me how to get the MAC address of the computer, so that you can implement the computer encryption function of EA indicator!

Thanks.

![Minh Truong Pham](https://c.mql5.com/avatar/2025/7/687ede4c-04cb.png)

**[Minh Truong Pham](https://www.mql5.com/en/users/truongxx)**
\|
6 May 2024 at 10:30

**MetaQuotes:**

New article [Native Twitter Client: Part 2](https://www.mql5.com/en/articles/8318) has been published:

Author: [Soewono Effendi](https://www.mql5.com/en/users/seffx "seffx")

Hello sir,

Thank you for helpfull article.

Can you please update this  article for api v2.0?

![Multicurrency monitoring of trading signals (Part 5): Composite signals](https://c.mql5.com/2/39/Article_Logo.png)[Multicurrency monitoring of trading signals (Part 5): Composite signals](https://www.mql5.com/en/articles/7759)

In the fifth article related to the creation of a trading signal monitor, we will consider composite signals and will implement the necessary functionality. In earlier versions, we used simple signals, such as RSI, WPR and CCI, and we also introduced the possibility to use custom indicators.

![Timeseries in DoEasy library (part 42): Abstract indicator buffer object class](https://c.mql5.com/2/38/MQL5-avatar-doeasy-library__7.png)[Timeseries in DoEasy library (part 42): Abstract indicator buffer object class](https://www.mql5.com/en/articles/7821)

In this article, we start the development of the indicator buffer classes for the DoEasy library. We will create the base class of the abstract buffer which is to be used as a foundation for the development of different class types of indicator buffers.

![Timeseries in DoEasy library (part 43): Classes of indicator buffer objects](https://c.mql5.com/2/38/MQL5-avatar-doeasy-library__8.png)[Timeseries in DoEasy library (part 43): Classes of indicator buffer objects](https://www.mql5.com/en/articles/7868)

The article considers the development of indicator buffer object classes as descendants of the abstract buffer object simplifying declaration and working with indicator buffers, while creating custom indicator programs based on DoEasy library.

![Native Twitter Client for MT4 and MT5 without DLL](https://c.mql5.com/2/41/mql5_twitter__1.png)[Native Twitter Client for MT4 and MT5 without DLL](https://www.mql5.com/en/articles/8270)

Ever wanted to access tweets and/or post your trade signals on Twitter ? Search no more, these on-going article series will show you how to do it without using any DLL. Enjoy the journey of implementing Twitter API using MQL. In this first part, we will follow the glory path of authentication and authorization in accessing Twitter API.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=uhpplimpwlmroxsbnljocrjsluzzjftn&ssn=1769252725705267231&ssn_dr=0&ssn_sr=0&fv_date=1769252725&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8318&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Native%20Twitter%20Client%3A%20Part%202%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925272538942381&fz_uniq=5083327016184322421&sv=2552)

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