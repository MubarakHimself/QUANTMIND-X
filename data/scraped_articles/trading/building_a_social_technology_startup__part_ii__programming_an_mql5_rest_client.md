---
title: Building a Social Technology Startup, Part II: Programming an MQL5 REST Client
url: https://www.mql5.com/en/articles/1044
categories: Trading, Trading Systems, Integration, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:31:06.440573
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/1044&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049187845861648040)

MetaTrader 5 / Trading


### Introduction

In the [previous part](https://www.mql5.com/en/articles/925) of this article, we presented the architecture of a so-called Social Decision Support System. On the one hand, this system consists of a MetaTrader 5 terminal sending Expert Advisors' automatic decisions to the server side. On the other side of the communication, there is a Twitter application built on the Slim PHP framework which receives those trading signals, stores them into a MySQL database, and finally tweets them to people. The main goal of the SDSS is to record human actions performed on robotic signals and make human decisions accordingly. This is possible because robotic signals can be exposed this way to a very large audience of experts.

In this second part we are going to develop the client side of the SDSS with the MQL5 programming language. We are discussing some alternatives as well as identifying the pros and cons of each of them. Then later, we will put together all the pieces of the puzzle, and end up shaping the PHP REST API that receives trading signals from Expert Advisors. To accomplish this we must take into account some aspects involved in the client side programming.

![Now you can tweet your MQL5 trading signals!](https://c.mql5.com/2/10/tecla-twitter.jpg)

_Now you can tweet your MQL5 trading signals!_

### 1\. The SDSS's Client Side

**1.1. Tweeting Some Trading Signals in the OnTimer Event**

I have considered showing how trading signals are sent from the [OnTimer](https://www.mql5.com/en/docs/basis/function/events#ontimer) event for simplicity issues. After seeing how this simple example works, it will be very easy to extrapolate this core behavior to a regular Expert Advisor.

dummy\_ontimer.mq5:

```
#property copyright     "Author: laplacianlab, CC Attribution-Noncommercial-No Derivate 3.0"
#property link          "https://www.mql5.com/en/users/laplacianlab"
#property version       "1.00"
#property description   "Simple REST client built on the OnTimer event for learning purposes."

int OnInit()
  {
   EventSetTimer(10);
   return(0);
  }

void OnDeinit(const int reason)
  {
  }

void OnTimer()
  {
//--- REST client's HTTP vars
   string uri="http://api.laplacianlab.com/signal/add";
   char post[];
   char result[];
   string headers;
   int res;
   string signal = "id_ea=1&symbol=AUDUSD&operation=BUY&value=0.9281&";
   StringToCharArray(signal,post);
//--- reset last error
   ResetLastError();
//--- post data to REST API
   res=WebRequest("POST",uri,NULL,NULL,50,post,ArraySize(post),result,headers);
//--- check errors
   if(res==-1)
     {
      Print("Error code =",GetLastError());
      //--- maybe the URL is not added, show message to add it
      MessageBox("Add address '"+uri+"' in Expert Advisors tab of the Options window","Error",MB_ICONINFORMATION);
     }
   else
     {
      //--- successful
      Print("REST client's POST: ",signal);
      Print("Server response: ",CharArrayToString(result,0,-1));
     }
  }
```

As you can see, the central part of this client application is [the new MQL5's WebRequest function](https://www.mql5.com/en/docs/network/webrequest).

Programming a custom MQL5 component to deal with the HTTP communication would be an alternative to this solution, yet delegating this task to MetaQuotes through [this new language feature](https://www.metaquotes.net/en/metatrader5/news/4225 "https://www.metaquotes.net/en/metatrader5/news/4225") is safer.

The MQL5 program above outputs the following:

```
OR      0       15:43:45.363    RESTClient (EURUSD,H1)  REST client's POST: id_ea=1&symbol=AUDUSD&operation=BUY&value=0.9281&
KK      0       15:43:45.365    RESTClient (EURUSD,H1)  Server response: {"id_ea":"1","symbol":"AUDUSD","operation":"BUY","value":"0.9281","id":77}
PD      0       15:43:54.579    RESTClient (EURUSD,H1)  REST client's POST: id_ea=1&symbol=AUDUSD&operation=BUY&value=0.9281&
CE      0       15:43:54.579    RESTClient (EURUSD,H1)  Server response: {"status": "ok", "message": {"text": "Please wait until the time window has elapsed."}}
ME      0       15:44:04.172    RESTClient (EURUSD,H1)  REST client's POST: id_ea=1&symbol=AUDUSD&operation=BUY&value=0.9281&
JD      0       15:44:04.172    RESTClient (EURUSD,H1)  Server response: {"status": "ok", "message": {"text": "Please wait until the time window has elapsed."}}
NE      0       15:44:14.129    RESTClient (EURUSD,H1)  REST client's POST: id_ea=1&symbol=AUDUSD&operation=BUY&value=0.9281&
ID      0       15:44:14.129    RESTClient (EURUSD,H1)  Server response: {"status": "ok", "message": {"text": "Please wait until the time window has elapsed."}}
NR      0       15:44:24.175    RESTClient (EURUSD,H1)  REST client's POST: id_ea=1&symbol=AUDUSD&operation=BUY&value=0.9281&
IG      0       15:44:24.175    RESTClient (EURUSD,H1)  Server response: {"status": "ok", "message": {"text": "Please wait until the time window has elapsed."}}
MR      0       15:44:34.162    RESTClient (EURUSD,H1)  REST client's POST: id_ea=1&symbol=AUDUSD&operation=BUY&value=0.9281&
JG      0       15:44:34.162    RESTClient (EURUSD,H1)  Server response: {"status": "ok", "message": {"text": "Please wait until the time window has elapsed."}}
PR      0       15:44:44.179    RESTClient (EURUSD,H1)  REST client's POST: id_ea=1&symbol=AUDUSD&operation=BUY&value=0.9281&
CG      0       15:44:44.179    RESTClient (EURUSD,H1)  Server response: {"status": "ok", "message": {"text": "Please wait until the time window has elapsed."}}
HS      0       15:44:54.787    RESTClient (EURUSD,H1)  REST client's POST: id_ea=1&symbol=AUDUSD&operation=BUY&value=0.9281&
KJ      0       15:44:54.787    RESTClient (EURUSD,H1)  Server response: {"id_ea":"1","symbol":"AUDUSD","operation":"BUY","value":"0.9281","id":78}
DE      0       15:45:04.163    RESTClient (EURUSD,H1)  REST client's POST: id_ea=1&symbol=AUDUSD&operation=BUY&value=0.9281&
OD      0       15:45:04.163    RESTClient (EURUSD,H1)  Server response: {"status": "ok", "message": {"text": "Please wait until the time window has elapsed."}}
```

Please, note that the server responds with this message:

```
{"status": "ok", "message": {"text": "Please wait until the time window has elapsed."}}
```

This is because there is a small security mechanism implemented in the API method signal/add to prevent the SDSS from hyperactive scalper robots:

```
/**
 * REST method.
 * Adds and tweets a new trading signal.
 */
$app->post('/signal/add', function() {
    $tweeterer = new Tweeterer();
    // This condition is a simple mechanism to prevent hyperactive scalpers
    if ($tweeterer->canTweet($tweeterer->getLastSignal(1)->created_at, '1 minute'))
    {
        $signal = (object)($_POST);
        $signal->id = $tweeterer->addSignal(1, $signal);
        $tokens = $tweeterer->getTokens(1);
        $connection = new TwitterOAuth(
            API_KEY,
            API_SECRET,
            $tokens->access_token,
            $tokens->access_token_secret);
        $connection->host = "https://api.twitter.com/1.1/";
        $ea = new EA();
        $message = "{$ea->get($signal->id_ea)->name} on $signal->symbol. $signal->operation at $signal->value";
        $connection->post('statuses/update', array('status' => $message));
        echo '{"status": "ok", "message": {"text": "Signal processed."}}';
    }
});
```

The simple mechanism above comes into play within the web app, just after the web server has already checked that the incoming HTTP request is not malicious (e.g. the incoming signal is not any denial of service attack).

The web server can be responsible for preventing such attacks. As an example, Apache can prevent them by combining the modules evasive and security.

This is a typical Apache's mod\_evasive configuration where the server administrator can control the HTTP requests that the app can accept per second, etc.

```
<IfModule mod_evasive20.c>
DOSHashTableSize    3097
DOSPageCount        2
DOSSiteCount        50
DOSPageInterval     1
DOSSiteInterval     1
DOSBlockingPeriod   60
DOSEmailNotify someone@somewhere.com
</IfModule>
```

So, as we say, the goal of the PHP method canTweet is to block hyperactive scalpers that are not considered as HTTP attacks by the SDSS. The method canTweet is implemented in the Twetterer class (which will be discussed later):

```
/**
 * Checks if it's been long enough so that the tweeterer can tweet again
 * @param string $timeLastTweet e.g. 2014-07-05 15:26:49
 * @param string $timeWindow A time window, e.g. 1 hour
 * @return boolean
 */
public function canTweet($timeLastTweet=null, $timeWindow=null)
{
    if(!isset($timeLastTweet)) return true;
    $diff = time() - strtotime($timeLastTweet);
    switch($timeWindow)
    {
        case '1 minute';
            $diff <= 60 ? $canTweet = false : $canTweet = true;
            break;
        case '1 hour';
            $diff <= 3600 ? $canTweet = false : $canTweet = true;
            break;
        case '1 day':
            $diff <= 86400 ? $canTweet = false : $canTweet = true;
            break;
        default:
            $canTweet = false;
            break;
    }
    if($canTweet)
    {
        return true;
    }
    else
    {
        throw new Exception('Please wait until the time window has elapsed.');
    }
}
```

Let's now see some HTTP request header fields that WebRequest automatically builds for us:

```
Content-Type: application/x-www-form-urlencoded
Accept: image/gif, image/x-xbitmap, image/jpeg, image/pjpeg, */*
```

WebRequest's POST assumes programmers want to send some HTML form data, nevertheless in this scenario we would want to send the server the following HTTP request header fields:

```
Content-Type: application/json
Accept: application/json
```

As there are no silver bullets we must be consistent with our decision and thoroughly study how WebRequest suits our requirements in order to discover the pros and cons.

It would be more correct from a technical point of view to establish truly HTTP REST dialogs, but as we said it is a safer solution to delegate HTTP dialogs to MetaQuotes even though WebRequest() seems to be originally intended for web pages, not for web services. It is for this reason that we will end up url encoding the client's trading signal. The API will receive url encoded signals and then will convert them to PHP's stdClass format.

An alternative to using the [WebRequest()](https://www.mql5.com/en/docs/network/webrequest) function is to write a custom MQL5 component working at a level close to the operating system using the wininet.dll library. The articles [Using WinInet.dll for Data Exchange between Terminals via the Internet](https://www.mql5.com/en/articles/73) and [Using WinInet in MQL5. Part 2: POST Requests and Files](https://www.mql5.com/en/articles/276) explain the fundamentals of this approach. However, the experience of MQL5 developers and MQL5 Community has shown that this solution is not as easy as it may seem at first glance. It presents the drawback that calls to the WinINet functions might break when MetaTrader is upgraded.

**1.2. Tweeting an EA's Trading Signals**

Now let's extrapolate what we've recently explained. I have created the following dummy robot in order to illustrate the problem about controlled scalping and denial of service attacks.

Dummy.mq5:

```
//+------------------------------------------------------------------+
//|                                                        Dummy.mq5 |
//|                               Copyright © 2014, Jordi Bassagañas |
//+------------------------------------------------------------------+
#property copyright     "Author: laplacianlab, CC Attribution-Noncommercial-No Derivate 3.0"
#property link          "https://www.mql5.com/en/users/laplacianlab"
#property version       "1.00"
#property description   "Dummy REST client (for learning purposes)."
//+------------------------------------------------------------------+
//| Trade class                                                      |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
//+------------------------------------------------------------------+
//| Declaration of variables                                         |
//+------------------------------------------------------------------+
CPositionInfo PositionInfo;
CTrade trade;
MqlTick tick;
int stopLoss = 20;
int takeProfit = 20;
double size = 0.1;
//+------------------------------------------------------------------+
//| Tweet trading signal                                             |
//+------------------------------------------------------------------+
void Tweet(string uri, string signal)
  {
   char post[];
   char result[];
   string headers;
   int res;
   StringToCharArray(signal,post);
//--- reset last error
   ResetLastError();
//--- post data to REST API
   res=WebRequest("POST",uri,NULL,NULL,50,post,ArraySize(post),result,headers);
//--- check errors
   if(res==-1)
     {
      //--- error
      Print("Error code =",GetLastError());
     }
   else
     {
      //--- successful
      Print("REST client's POST: ",signal);
      Print("Server response: ",CharArrayToString(result,0,-1));
     }
  }
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- update tick
   SymbolInfoTick(_Symbol, tick);
//--- calculate Take Profit and Stop Loss levels
   double tp;
   double sl;
   sl = tick.ask + stopLoss * _Point;
   tp = tick.bid - takeProfit * _Point;
//--- open position
   trade.PositionOpen(_Symbol,ORDER_TYPE_SELL,size,tick.bid,sl,tp);
//--- trade URL-encoded signal "id_ea=1&symbol=AUDUSD&operation=BUY&value=0.9281&";
   string signal = "id_ea=1&symbol=" + _Symbol + "&operation=SELL&value=" + (string)tick.bid + "&";
   Tweet("http://api.laplacianlab.com/signal/add",signal);
}
```

The code above cannot be easier. This Expert Advisor only places one single short position on every tick. For this reason, it is very likely that this robot ends up placing many positions in a short interval of time, especially if you run it in a moment of time where there is a lot of volatility. There's no reason to worry. The server side controls the tweeting interval by both configuring the web server to prevent DoS attacks, and by defining a certain time window in the PHP application, as explained.

With all this clear, you can now take this EA's Tweet function and place it in your favorite Expert Advisor.

**1.3. How do users see their tweeted trading signals?**

In the following example, @laplacianlab gives permission to the SDSS to tweet the signals of the dummy EA which was posted in the previous section:

![Figure 1. @laplacianlab gives permission to the SDSS to tweet on his behalf](https://c.mql5.com/2/10/dummy-ea.png)

Figure 1. @laplacianlab has given permission to the SDSS to tweet on his behalf

By the way, the Bollinger Bands name appears in this example because this is the one that we stored in the MySQL database in the first part of this article. id\_ea=1 was associated to "Bollinger Bands", but we should have changed it to something like "Dummy" in order to fit well this explanation. In any case this is a secondary aspect but sorry for this little inconvenience.

The MySQL database is finally as follows:

```
# MySQL database creation...

CREATE DATABASE IF NOT EXISTS laplacianlab_com_sdss;

use laplacianlab_com_sdss;

CREATE TABLE IF NOT EXISTS twitterers (
    id mediumint UNSIGNED NOT NULL AUTO_INCREMENT,
    twitter_id VARCHAR(255),
    access_token TEXT,
    access_token_secret TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS eas (
    id mediumint UNSIGNED NOT NULL AUTO_INCREMENT,
    name VARCHAR(32),
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS signals (
    id int UNSIGNED NOT NULL AUTO_INCREMENT,
    id_ea mediumint UNSIGNED NOT NULL,
    id_twitterer mediumint UNSIGNED NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    operation VARCHAR(6) NOT NULL,
    value DECIMAL(9,5) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id),
    FOREIGN KEY (id_ea) REFERENCES eas(id),
    FOREIGN KEY (id_twitterer) REFERENCES twitterers(id)
) ENGINE=InnoDB;

# Dump some sample data...

# As explained in Part I, there's one single twitterer

INSERT INTO eas(name, description) VALUES
('Bollinger Bands', '<p>Robot based on Bollinger Bands. Works with H4 charts.</p>'),
('Two EMA', '<p>Robot based on the crossing of two MA. Works with H4 charts.</p>');
```

### 2\. The SDSS's Server Side

Before continue to shape the server side of our Social Decision Support System, let's briefly remember that we have the following directory structure at the moment:

![Figure 2. Directory structure of the PHP API based on Slim](https://c.mql5.com/2/10/figure-folder-structure__2.png)

Figure 2. Directory structure of the PHP API based on Slim

**2.1. PHP API Code**

According to what it has been explained the index.php file should now look like this:

```
<?php
/**
 * Laplacianlab's SDSS - A REST API for tweeting MQL5 trading signals
 *
 * @author      Jordi Bassagañas
 * @copyright   2014 Jordi Bassagañas
 * @link        https://www.mql5.com/en/users/laplacianlab
 */

/* Bootstrap logic */
require_once 'config/config.php';
set_include_path(get_include_path() . PATH_SEPARATOR . APPLICATION_PATH . '/vendor/');
set_include_path(get_include_path() . PATH_SEPARATOR . APPLICATION_PATH . '/model/');
require_once 'slim/slim/Slim/Slim.php';
require_once 'abraham/twitteroauth/twitteroauth/twitteroauth.php';
require_once 'Tweeterer.php';
require_once 'EA.php';
session_start();

/* Init Slim */
use \Slim\Slim;
Slim::registerAutoloader();
$app = new Slim(array('debug' => false));
$app->response->headers->set('Content-Type', 'application/json');

/**
 * Slim's exception handler
 */
$app->error(function(Exception $e) use ($app) {
    echo '{"status": "error", "message": {"text": "' . $e->getMessage() . '"}}';
});

/**
 * REST method.
 * Custom 404 error.
 */
$app->notFound(function () use ($app) {
    echo '{"status": "error 404", "message": {"text": "Not found."}}';
});

/**
 * REST method.
 * Home page.
 */
$app->get('/', function () {
    echo '{"status": "ok", "message": {"text": "Service available, please check API."}}';
});

/**
 * REST method.
 * Adds and tweets a new trading signal.
 */
$app->post('/signal/add', function() {
    $tweeterer = new Tweeterer();
    // This condition is a simple mechanism to prevent hyperactive scalpers
    if ($tweeterer->canTweet($tweeterer->getLastSignal(1)->created_at, '1 minute'))
    {
        $signal = (object)($_POST);
        $signal->id = $tweeterer->addSignal(1, $signal);
        $tokens = $tweeterer->getTokens(1);
        $connection = new TwitterOAuth(
            API_KEY,
            API_SECRET,
            $tokens->access_token,
            $tokens->access_token_secret);
        $connection->host = "https://api.twitter.com/1.1/";
        $ea = new EA();
        $message = "{$ea->get($signal->id_ea)->name} on $signal->symbol. $signal->operation at $signal->value";
        $connection->post('statuses/update', array('status' => $message));
        echo '{"status": "ok", "message": {"text": "Signal processed."}}';
    }
});

/**
 * REST implementation with TwitterOAuth.
 * Gives permissions to Laplacianlab's SDSS to tweet on the user's behalf.
 * Please, visit https://github.com/abraham/twitteroauth
 */
$app->get('/tweet-signals', function() use ($app) {
    if (empty($_SESSION['twitter']['access_token']) || empty($_SESSION['twitter']['access_token_secret']))
    {
        $connection = new TwitterOAuth(API_KEY, API_SECRET);
        $request_token = $connection->getRequestToken(OAUTH_CALLBACK);
        if ($request_token)
        {
            $_SESSION['twitter'] = array(
                'request_token' => $request_token['oauth_token'],
                'request_token_secret' => $request_token['oauth_token_secret']
            );
            switch ($connection->http_code)
            {
                case 200:
                    $url = $connection->getAuthorizeURL($request_token['oauth_token']);
                    // redirect to Twitter
                    $app->redirect($url);
                    break;
                default:
                    throw new Exception('Connection with Twitter failed.');
                break;
            }
        }
        else
        {
            throw new Exception('Error Receiving Request Token.');
        }
    }
    else
    {
        echo '{"status": "ok", "message": {"text": "Laplacianlab\'s SDSS can '
        . 'now access your Twitter account on your behalf. Please, if you no '
        . 'longer want this, log in your Twitter account and revoke access."}}';
    }
});

/**
 * REST implementation with TwitterOAuth.
 * This is the OAuth callback of the method above.
 * Stores the access tokens into the database.
 * Please, visit https://github.com/abraham/twitteroauth
 */
$app->get('/twitter/oauth_callback', function() use ($app) {
    if(isset($_GET['oauth_token']))
    {
        $connection = new TwitterOAuth(
            API_KEY,
            API_SECRET,
            $_SESSION['twitter']['request_token'],
            $_SESSION['twitter']['request_token_secret']);
        $access_token = $connection->getAccessToken($_REQUEST['oauth_verifier']);
        if($access_token)
        {
            $connection = new TwitterOAuth(
                API_KEY,
                API_SECRET,
                $access_token['oauth_token'],
                $access_token['oauth_token_secret']);
            // Set Twitter API version to 1.1.
            $connection->host = "https://api.twitter.com/1.1/";
            $params = array('include_entities' => 'false');
            $content = $connection->get('account/verify_credentials', $params);
            if($content && isset($content->screen_name) && isset($content->name))
            {
                $tweeterer = new Tweeterer();
                $data = (object)array(
                    'twitter_id' => $content->id,
                    'access_token' => $access_token['oauth_token'],
                    'access_token_secret' => $access_token['oauth_token_secret']);
                $tweeterer->exists($content->id)
                        ? $tweeterer->update($data)
                        : $tweeterer->create($data);
                echo '{"status": "ok", "message": {"text": "Laplacianlab\'s SDSS can '
                . 'now access your Twitter account on your behalf. Please, if you no '
                . 'longer want this, log in your Twitter account and revoke access."}}';
                session_destroy();
            }
            else
            {
                throw new Exception('Login error.');
            }
        }
    }
    else
    {
        throw new Exception('Login error.');
    }
});

/**
 * Run Slim!
 */
$app->run();
```

**2.2. MySQL OOP Wrappers**

We now must create the PHP classes Tweeterer.php and EA.php in the Slim application's model directory. Please, note that rather than developing an actual model layer what we do is wrapping the MySQL tables in simple object-oriented classes.

model\\Tweeterer.php:

```
<?php
require_once 'DBConnection.php';
/**
 * Tweeterer's simple OOP wrapper
 *
 * @author      Jordi Bassagañas
 * @copyright   2014 Jordi Bassagañas
 * @link        https://www.mql5.com/en/users/laplacianlab
 */
class Tweeterer
{
    /**
     * @var string MySQL table
     */
    protected $table = 'twitterers';
    /**
     * Gets the user's OAuth tokens
     * @param integer $id
     * @return stdClass OAuth tokens: access_token and access_token_secret
     */
    public function getTokens($id)
    {
        $sql = "SELECT access_token, access_token_secret FROM $this->table WHERE id=$id";
        return DBConnection::getInstance()->query($sql)->fetch_object();
    }
    /**
     * Checks if it's been long enough so that the tweeterer can tweet again
     * @param string $timeLastTweet e.g. 2014-07-05 15:26:49
     * @param string $timeWindow A time window, e.g. 1 hour
     * @return boolean
     */
    public function canTweet($timeLastTweet=null, $timeWindow=null)
    {
        if(!isset($timeLastTweet)) return true;
        $diff = time() - strtotime($timeLastTweet);
        switch($timeWindow)
        {
            case '1 minute';
                $diff <= 60 ? $canTweet = false : $canTweet = true;
                break;
            case '1 hour';
                $diff <= 3600 ? $canTweet = false : $canTweet = true;
                break;
            case '1 day':
                $diff <= 86400 ? $canTweet = false : $canTweet = true;
                break;
            default:
                $canTweet = false;
                break;
        }
        if($canTweet)
        {
            return true;
        }
        else
        {
            throw new Exception('Please wait until the time window has elapsed.');
        }
    }
    /**
     * Adds a new signal
     * @param type $id_twitterer
     * @param stdClass $data
     * @return integer The new row id
     */
    public function addSignal($id_twitterer, stdClass $data)
    {
        $sql = 'INSERT INTO signals(id_ea, id_twitterer, symbol, operation, value) VALUES ('
            . $data->id_ea . ","
            . $id_twitterer . ",'"
            . $data->symbol . "','"
            . $data->operation . "',"
            . $data->value . ')';
        DBConnection::getInstance()->query($sql);
        return DBConnection::getInstance()->getHandler()->insert_id;
    }
    /**
     * Checks whether the given twitterer exists
     * @param string $id
     * @return boolean
     */
    public function exists($id)
    {
        $sql = "SELECT * FROM $this->table WHERE twitter_id='$id'";
        $result = DBConnection::getInstance()->query($sql);
        return (boolean)$result->num_rows;
    }
    /**
     * Creates a new twitterer
     * @param stdClass $data
     * @return integer The new row id
     */
    public function create(stdClass $data)
    {
        $sql = "INSERT INTO $this->table(twitter_id, access_token, access_token_secret) "
            . "VALUES ('"
            . $data->twitter_id . "','"
            . $data->access_token . "','"
            . $data->access_token_secret . "')";
        DBConnection::getInstance()->query($sql);
        return DBConnection::getInstance()->getHandler()->insert_id;
    }
    /**
     * Updates the twitterer's data
     * @param stdClass $data
     * @return Mysqli object
     */
    public function update(stdClass $data)
    {
        $sql = "UPDATE $this->table SET "
            . "access_token = '" . $data->access_token . "', "
            . "access_token_secret = '" . $data->access_token_secret . "' "
            . "WHERE twitter_id ='" . $data->twitter_id . "'";
        return DBConnection::getInstance()->query($sql);
    }
    /**
     * Gets the last trading signal sent by the twitterer
     * @param type $id The twitterer id
     * @return mixed The last trading signal
     */
    public function getLastSignal($id)
    {
        $sql = "SELECT * FROM signals WHERE id_twitterer=$id ORDER BY id DESC LIMIT 1";
        $result = DBConnection::getInstance()->query($sql);
        if($result->num_rows == 1)
        {
            return $result->fetch_object();
        }
        else
        {
            $signal = new stdClass;
            $signal->created_at = null;
            return $signal;
        }
    }
}
```

model\\EA.php:

```
<?php
require_once 'DBConnection.php';
/**
 * EA's simple OOP wrapper
 *
 * @author      Jordi Bassagañas
 * @copyright   2014 Jordi Bassagañas
 * @link        https://www.mql5.com/en/users/laplacianlab
 */
class EA
{
    /**
     * @var string MySQL table
     */
    protected $table = 'eas';
    /**
     * Gets an EA by id
     * @param integer $id
     * @return stdClass
     */
    public function get($id)
    {
        $sql = "SELECT * FROM $this->table WHERE id=$id";
        return DBConnection::getInstance()->query($sql)->fetch_object();
    }
}
```

model\\DBConnection.php:

```
<?php
/**
 * DBConnection class
 *
 * @author      Jordi Bassagañas
 * @copyright   2014 Jordi Bassagañas
 * @link        https://www.mql5.com/en/users/laplacianlab
 */
class DBConnection
{
    /**
     * @var DBConnection Singleton instance
     */
    private static $instance;
    /**
     * @var mysqli Database handler
     */
    private $mysqli;
    /**
     *  Opens a new connection to the MySQL server
     */
    private function __construct()
    {
        mysqli_report(MYSQLI_REPORT_STRICT);
        try {
            $this->mysqli = new MySQLI(DB_SERVER, DB_USER, DB_PASSWORD, DB_NAME);
        } catch (Exception $e) {
            throw new Exception('Unable to connect to the database, please, try again later.');
        }
    }
    /**
     * Gets the singleton instance
     * @return type
     */
    public static function getInstance()
    {
        if (!self::$instance instanceof self) self::$instance = new self;
        return self::$instance;
    }
    /**
     * Gets the database handler
     * @return mysqli
     */
    public function getHandler()
    {
        return $this->mysqli;
    }
    /**
     * Runs the given query
     * @param string $sql
     * @return mixed
     */
    public function query($sql)
    {
        $result = $this->mysqli->query($sql);
        if ($result === false)
        {
            throw new Exception('Unable to run query, please, try again later.');
        }
        else
        {
            return $result;
        }
    }
}
```

### Conclusion

We have developed the client side of the SDSS which was introduced in the first part of this article, and have ended up shaping the server side according to this decision. We have finally used the new MQL5's native function [WebRequest()](https://www.mql5.com/en/docs/network/webrequest). Regarding the pros and cons of this specific solution, we have seen that WebRequest() is not originally intended to consume web services, but to make GET and POST requests to web pages. However, at the same time, we have decided to use this new feature because it is safer than developing a custom component from scratch.

It would have been more elegant to establish truly REST dialogues between the MQL5 client and the PHP server, but it has been much more easier to adapt WebRequest() to our specific need. Thus, the web service receives URL-encoded data and converts them into a manageable format for PHP.

I am currently working on this system. For now I can tweet my personal trading signals. It is functional, it works for a single user, but there are some missing pieces in order for it to completely work in a real production environment. As an example, Slim is a database-agnostic framework, so you should care about SQL injections. Nor have we explained how to secure the communication between the MetaTrader 5 terminal and the PHP application, so please don't run this app in a real environment as it is presented in this article.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1044.zip "Download all attachments in the single ZIP archive")

[database.txt](https://www.mql5.com/en/articles/download/1044/database.txt "Download database.txt")(1.32 KB)

[dummy\_ontimer.mq5](https://www.mql5.com/en/articles/download/1044/dummy_ontimer.mq5 "Download dummy_ontimer.mq5")(1.36 KB)

[Dummy.mq5](https://www.mql5.com/en/articles/download/1044/dummy.mq5 "Download Dummy.mq5")(3.27 KB)

[index.txt](https://www.mql5.com/en/articles/download/1044/index.txt "Download index.txt")(6.08 KB)

[Tweeterer.txt](https://www.mql5.com/en/articles/download/1044/tweeterer.txt "Download Tweeterer.txt")(4.59 KB)

[EA.txt](https://www.mql5.com/en/articles/download/1044/ea.txt "Download EA.txt")(0.58 KB)

[DBConnection.txt](https://www.mql5.com/en/articles/download/1044/dbconnection.txt "Download DBConnection.txt")(1.54 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Building a Social Technology Startup, Part I: Tweet Your MetaTrader 5 Signals](https://www.mql5.com/en/articles/925)
- [Raise Your Linear Trading Systems to the Power](https://www.mql5.com/en/articles/734)
- [Marvel Your MQL5 Customers with a Usable Cocktail of Technologies!](https://www.mql5.com/en/articles/728)
- [Extending MQL5 Standard Library and Reusing Code](https://www.mql5.com/en/articles/741)
- [Building an Automatic News Trader](https://www.mql5.com/en/articles/719)
- [Another MQL5 OOP Class](https://www.mql5.com/en/articles/703)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/34780)**
(2)


![okwh](https://c.mql5.com/avatar/2011/9/4E7F67FD-3C19.jpg)

**[okwh](https://www.mql5.com/en/users/dxdcn)**
\|
14 Oct 2014 at 06:14

**MetaQuotes:**

New article [Building Emerging Social Technologies, Part 2: Preparing a REST Client for MQL5](https://www.mql5.com/en/articles/1044) has been released:

By Jordi Bassagan

How to bookmark an article to forum members?


![Nono Momo](https://c.mql5.com/avatar/avatar_na2.png)

**[Nono Momo](https://www.mql5.com/en/users/cubeer)**
\|
8 Mar 2015 at 16:46

Good!


![Freelance Jobs on MQL5.com - Developer's Favorite Place](https://c.mql5.com/2/10/ava_freelance-mql5.png)[Freelance Jobs on MQL5.com - Developer's Favorite Place](https://www.mql5.com/en/articles/1022)

Developers of trading robots no longer need to market their services to traders that require Expert Advisors - as now they will find you. Already, thousands of traders place orders to MQL5 freelance developers, and pay for work in on MQL5.com. For 4 years, this service facilitated three thousand traders to pay for more than 10 000 jobs performed. And the activity of traders and developers is constantly growing!

![Outline of MetaTrader Market (Infographics)](https://c.mql5.com/2/10/infographic_market_av__1.png)[Outline of MetaTrader Market (Infographics)](https://www.mql5.com/en/articles/1077)

A few weeks ago we published the infographic on Freelance service. We also promised to reveal some statistics of the MetaTrader Market. Now, we invite you to examine the data we have gathered.

![Tips for an Effective Product Presentation on the Market](https://c.mql5.com/2/11/ava_paint-market2.png)[Tips for an Effective Product Presentation on the Market](https://www.mql5.com/en/articles/999)

Selling programs to traders effectively does not only require writing an efficient and useful product and then publishing it on the Market. It is vital to provide a comprehensive, detailed description and good illustrations. A quality logo and correct screenshots are equally as important as the "true coding". Bear in mind a simple formula: no downloads = no sales.

![Johnpaul77 Signal Providers: "Our Strategy Remains Profitable for More Than Three Years Now. So Why Would We Change It?"](https://c.mql5.com/2/10/4-Photo-for-Interview-Avatar-Tito.png)[Johnpaul77 Signal Providers: "Our Strategy Remains Profitable for More Than Three Years Now. So Why Would We Change It?"](https://www.mql5.com/en/articles/1045)

Let us reveal a little secret: MQL5.com website visitors spend most of their time on Johnpaul77 signal's page. It is a leader of our signal rating having about 900 subscribers with the total funds of $5.7 million on real accounts. We have interviewed the signal's providers. As it turned out, there are four of them! How are duties distributed among the team members? What technical tools do they use? Why do they call themselves John Paul? And finally, how have common gamers from Indonesia become providers of the top signal on MQL5.com? Find out all that in the article.

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ppfzjbpgzvonorehvovclvdofaikornk&ssn=1769092265765426011&ssn_dr=0&ssn_sr=0&fv_date=1769092265&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1044&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Building%20a%20Social%20Technology%20Startup%2C%20Part%20II%3A%20Programming%20an%20MQL5%20REST%20Client%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909226519036117&fz_uniq=5049187845861648040&sv=2552)

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