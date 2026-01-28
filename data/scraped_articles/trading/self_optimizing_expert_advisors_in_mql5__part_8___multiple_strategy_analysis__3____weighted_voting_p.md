---
title: Self Optimizing Expert Advisors in MQL5 (Part 8): Multiple Strategy Analysis (3) — Weighted Voting Policy
url: https://www.mql5.com/en/articles/18770
categories: Trading, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T17:55:56.305452
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/18770&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049492685460450426)

MetaTrader 5 / Examples


We will now add the final component to our Multiple Strategy Expert Advisor: the Williams Percentage Reversal Strategy. As we’ve already done in the previous articles, we first hard-coded a manual implementation of the strategy to benchmark its performance against the class we are about to build for our trading application. However, to avoid boring readers who have been following the series, we will omit the test results we used to validate the class. For now, it should suffice to say that readers can trust that adequate testing was done to verify the integrity of the class we are presenting today.

When building an ensemble of strategies, naturally, a question that follows is, how can we prove that all the strategies we've selected are necessary? How can we be reasonably sure that we wouldn’t perform better with just a few of them? How can we prove any of this to ourselves?

Fortunately for us, the genetic optimizer can help answer such challenging questions, provided we carefully frame the question for it.

To achieve this, we will allow our strategies to collaborate through a democracy, where each strategy is allowed only one vote. The weight of the vote that each strategy casts can be a tuning parameter, adjusted by the genetic optimizer. If the optimizer determines that one of our strategies isn’t positively contributing to overall performance, it will set the weight of that strategy’s vote close to zero. Likewise, it will add more weight to the strategies that are profitable.

Therefore, we present this framework as a weighted voting policy, in which we initially set a benchmark performance level by giving each of our strategies uniformly distributed vote weights. In our example, we start with each strategy having a vote weight of 0.5, on a scale that ranges from 0 to 1.

From there, we allow the genetic optimizer to adjust these weights to maximize profitability and determine whether all three strategies are truly helpful.

It turns out this procedure returns a wide array of different configurations, each showing how the usefulness of a strategy can change depending on the particular strategy settings. In each unique configuration, the weight of each strategy shifts. So, there may be a setup where only one strategy proves useful, while in another configuration, all three are contributing positively.

This makes the question "Are all three strategies necessary?" a genuinely challenging one to answer. Our findings suggest that the answer is sensitive to the configuration the application employed in the first place. Let us begin.

### Getting Started in MQL5

By the end of our discussion, our inheritance tree for trading strategies can be visualized as Figure 1 below. We will have 3 individual trading strategies:

1. Relative Strength Index Momentum Strategy
2. Moving Average Crossover Strategy
3. Williams Percent Range Reversal Strategy


Each with common functionality, such as the ability to signal a long or short entry. Our three strategies each share a common parent in their ancestry. This is important to ensure that we maintain uniform utility across our classes.

In today's discussion, we will focus on implementing the last of the three strategies depicted in Figure 1: the Williams Percent Range Strategy. From there, our genetic optimizer will adjust the weights assigned to each strategy to ensure that the least profitable one does not degrade the performance of our application.

![](https://c.mql5.com/2/155/5033048682773.png)

Figure 1: The current state of our inheritance tree shared by our trading strategies

Additionally, in Figure 2, we’ve provided visual aids to help illustrate the core ideology behind our strategy. Note that in the illustration shown in Figure 2, the total weights of all votes do not add up to one. While it’s common to impose that constraint, we’ve chosen not to do so here. We may explore that variation in the future, as it would require a slightly different algorithm than the one implemented today.

For now, we simply allow the genetic optimizer to assign a value from and including both 0 and 1 for each of the three strategies. Our genetic optimizer will find it easier to generate profitable strategies if it does not have to account for the least profitable strategy. This is the reasoning behind our discussion: We want to demonstrate that the genetic optimizer can also prune our tree of trading strategies while tuning other important parameters of our strategy.

![](https://c.mql5.com/2/155/5513778767471.png)

Figure 2: Visualizing the weights attributed to each strategy by the genetic optimizer

The first step in implementing our strategy is to load the dependencies. The first dependency, as we did before for our Williams Percent Range (WPR) strategy, is to load the single-buffer Williams Percentage Indicator class that we previously created. Afterward, we load the parent strategy class, which we also developed in a separate discussion.

```
//+------------------------------------------------------------------+
//|                                                  WPRReversal.mqh |
//|                                               Gamuchirai Ndawana |
//|                    https://www.mql5.com/en/users/gamuchiraindawa |
//+------------------------------------------------------------------+
#property copyright "Gamuchirai Ndawana"
#property link      "https://www.mql5.com/en/users/gamuchiraindawa"
#property version   "1.00"

//+------------------------------------------------------------------+
//| Dependencies                                                     |
//+------------------------------------------------------------------+
#include <VolatilityDoctor\Indicators\WPR.mqh>
#include <VolatilityDoctor\Strategies\Parent\Strategy.mqh>
```

Once our dependencies have been loaded, we can begin defining the members of our class. The first member will point to the Williams Percent Range indicator we’re going to use. This will be a private member—the only private member of our class. The remaining members are public class members. Specifically, we include the constructor, destructor, and the virtual methods inherited from the parent.

```
class WPRReversal : public Strategy
  {
private:
                     //--- The instance of the RSI used in this strategy
                     WPR *my_wpr;

public:
                     //--- Class constructor
                     WPRReversal(string user_symbol,ENUM_TIMEFRAMES user_timeframe,int user_period);

                     //--- Class destructor
                    ~WPRReversal();

                    //--- Class overrides
                    virtual bool Update(void);
                    virtual bool BuySignal(void);
                    virtual bool SellSignal(void);
  };
```

We begin by overriding the update method. The update method simply calls the set\_indicator\_values method, which is a function present in all of our indicator classes. This function fills in the WPR readings from the terminal into our indicator buffer. It performs a precautionary check to ensure the count is not zero before handing control back to the calling context.

```
//+------------------------------------------------------------------+
//| Our strategy update method                                       |
//+------------------------------------------------------------------+
bool WPRReversal::Update(void)
   {
      //--- Set the indicator value
      my_wpr.SetIndicatorValues(Strategy::GetIndicatorBufferSize(),true);

      //--- Check readings are valid
      if(my_wpr.GetCurrentReading() != 0) return(true);

      //--- Something went wrong
      return(false);
   }
```

From there, we define two methods used to signal our buy and sell entries. These methods simply return a result of true if their respective conditions are met.

```
//+------------------------------------------------------------------+
//| Check for our buy signal                                         |
//+------------------------------------------------------------------+
bool WPRReversal::BuySignal(void)
   {
      //--- Buy signals when the RSI is above 50
      return(my_wpr.GetCurrentReading()>50);
   }

//+------------------------------------------------------------------+
//| Check for our sell signal                                        |
//+------------------------------------------------------------------+
bool WPRReversal::SellSignal(void)
   {
      //--- Sell signals when the RSI is below 50
      return(my_wpr.GetCurrentReading()<50);
   }
```

Lastly, we define our parametric constructor, which takes in the symbol, timeframe, and period that the WPR indicator should be initialized with. The destructor then simply deletes the pointer we created to the new instance of our WPR class object

```
//+------------------------------------------------------------------+
//| Our class constructor                                            |
//+------------------------------------------------------------------+
WPRReversal::WPRReversal(string user_symbol,ENUM_TIMEFRAMES user_timeframe,int user_period)
  {
   my_wpr = new WPR(user_symbol,user_timeframe,user_period);
   Print("WPRReversal Strategy Loaded.");
  }

//+------------------------------------------------------------------+
//| Our class destructor                                             |
//+------------------------------------------------------------------+
WPRReversal::~WPRReversal()
  {
   delete my_wpr;
  }
//+------------------------------------------------------------------+
```

### Building The Expert Advisor

We will now begin defining the expert advisor that we’ll use in our current setup. The first part of our expert advisor will be the system constants, which we’ll keep fixed for the sake of reproducibility in our tests. Simple parameters—such as the shift in the moving average and the type of moving average we want to use—must be fixed. These will be set to values that are easy to remember, such as a shift of zero.

```
//+------------------------------------------------------------------+
//|                                                   MSA Test 1.mq5 |
//|                                               Gamuchirai Ndawana |
//|                    https://www.mql5.com/en/users/gamuchiraindawa |
//+------------------------------------------------------------------+
#property copyright "Gamuchirai Ndawana"
#property link      "https://www.mql5.com/en/users/gamuchiraindawa"
#property version   "1.00"

//+------------------------------------------------------------------+
//| System constants                                                 |
//+------------------------------------------------------------------+
//--- Fix any parameters that can afford to remain fixed
#define MA_SHIFT         0
#define MA_TYPE          MODE_EMA
#define RSI_PRICE        PRICE_CLOSE
```

Additionally, we must accept certain user inputs. Remember our analogy: we intend to accept these input suggestions from the genetic optimizer. The first three input groups should be quite familiar to the reader. These are simply the periods we are going to use with our technical indicators.

The input group we are particularly interested in for this discussion is the last one: the global strategy parameters group. That’s where the weights we are setting today will be stored. Settings such as the holding period and the timeframe of the strategy should already be familiar to our returning readers. However, new readers should know that the holding period refers to how long we will wait before considering that a position has reached maturity and should be closed. Naturally, this holding period is sensitive to the strategy’s timeframe. For example, a holding period of 5 on a timeframe of M10 means we will hold the position for 50 minutes before closing.

```
//+------------------------------------------------------------------+
//| User Inputs                                                      |
//+------------------------------------------------------------------+
input   group          "Moving Average Strategy Parameters"
input   int             MA_PERIOD                       =        10;//Moving Average Period

input   group          "RSI Strategy Parameters"
input   int             RSI_PERIOD                      =         15;//RSI Period

input   group          "WPR Strategy Parameters"
input   int             WPR_PERIOD                      =         30;//WPR Period

input   group          "Global Strategy Parameters"
input   ENUM_TIMEFRAMES STRATEGY_TIME_FRAME             = PERIOD_D1;//Strategy Timeframe
input   int             HOLDING_PERIOD                  =         5;//Position Maturity Period
input   double          weight_1                        =       0.5;//Strategy 1 vote weight
input   double          weight_2                        =       0.5;//Strategy 2 vote weight
input   double          weight_3                        =       0.5;//Strategy 3 vote weight
```

Next are the dependencies that our trading application will require. The first dependency is the trade library, which is our base dependency. It helps us handle position management. From there, we have other custom-built dependencies such as TimeInfo and TradeInfo, which help us know when we can act on market information, as well as provide access to minimum trade levels, the ask price, and the minimum tradable value, respectively.

The remaining three dependencies come from the strategy classes we’ve been building together throughout this series. These should already be familiar to you—and if you are a new reader, then at least the very last dependency should be recognizable, because that is what we built together today

```
//+------------------------------------------------------------------+
//| Dependencies                                                     |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
#include <VolatilityDoctor\Time\Time.mqh>
#include <VolatilityDoctor\Trade\TradeInfo.mqh>
#include <VolatilityDoctor\Strategies\OpenCloseMACrossover.mqh>
#include <VolatilityDoctor\Strategies\RSIMidPoint.mqh>
#include <VolatilityDoctor\Strategies\WPRReversal.mqh>
```

We will also need a few global variables, such as handlers for our custom objects, and a timer to help us keep track of how close we are to the maturity of our positions.

```
//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+

//--- Custom Types
CTrade               Trade;
Time                 *TradeTime;
TradeInfo            *TradeInformation;
RSIMidPoint          *RSIMid;
OpenCloseMACrossover *MACross;
WPRReversal          *WPRR;

//--- System Types
int                  position_timer;
```

When our application is first initialized, we will create new instances of our custom-defined classes—such as the strategies and the TradeInfo class.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Create dynamic instances of our custom types
   TradeTime        = new Time(Symbol(),STRATEGY_TIME_FRAME);
   TradeInformation = new TradeInfo(Symbol(),STRATEGY_TIME_FRAME);
   MACross          = new OpenCloseMACrossover(Symbol(),STRATEGY_TIME_FRAME,MA_PERIOD,MA_SHIFT,MA_TYPE);
   RSIMid           = new RSIMidPoint(Symbol(),STRATEGY_TIME_FRAME,RSI_PERIOD,RSI_PRICE);
   WPRR             = new WPRReversal(Symbol(),STRATEGY_TIME_FRAME,WPR_PERIOD);
//--- Everything was fine
   return(INIT_SUCCEEDED);
  }
//--- End of OnInit Scope
```

When the application is no longer in use, we’ll delete these custom-defined objects to ensure we’re not leaking any memory.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Delete the dynamic objects
   delete TradeTime;
   delete TradeInformation;
   delete MACross;
   delete RSIMid;
  }
//--- End of Deinit Scope
```

Whenever new price data is received, we will first check whether a new candle has been formed. If that is the case, we will then update the parameters and indicator values in our strategies. Lastly, if we have no open positions, we’ll reset our position timer and check for signal conditions. Otherwise, if positions are already open, we’ll track how close we are to maturity as we prepare to wind down the position.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Check if a new daily candle has formed
   if(TradeTime.NewCandle())
     {
      //--- Update strategy
      Update();

      //--- If we have no open positions
      if(PositionsTotal() == 0)
        {
         //--- Reset the position timer
         position_timer = 0;

         //--- Check for a trading signal
         CheckSignal();
        }
      //--- Otherwise
      else
        {
         //--- The position has reached maturity
         if(position_timer == HOLDING_PERIOD)
            Trade.PositionClose(Symbol());
         //--- Otherwise keep holding
         else
            position_timer++;
        }
     }
  }
//--- End of OnTick Scope
```

The update method is implemented by simply calling the update function associated with each of our strategies.

```
//+------------------------------------------------------------------+
//| Update our technical indicators                                  |
//+------------------------------------------------------------------+
void Update(void)
  {
//--- Update the strategy
   RSIMid.Update();
   MACross.Update();
   WPRR.Update();
  }
//--- End of Update Scope
```

The check\_signal method is interesting in how it’s set up. That is to say, we begin by initializing the total vote to zero. If by the end of the process the total vote is positive, we go long. Otherwise, if the total vote is negative, we sell. From there, we check which signal is being generated by each strategy. If a strategy is generating a long signal, we add that strategy’s weight to the total vote. If it’s generating a short signal, we subtract the strategy’s weight from the total vote. Each strategy is given one turn to vote. At the end, we evaluate the total vote according to the rules just described.

```
//+------------------------------------------------------------------+
//| Check for a trading signal using our cross-over strategy         |
//+------------------------------------------------------------------+
void CheckSignal(void)
  {
   double vote = 0;

   if(MACross.BuySignal())
      vote += weight_1;
   else
      if(MACross.SellSignal())
         vote -= weight_1;

   if(RSIMid.BuySignal())
      vote += weight_2;
   else
      if(RSIMid.SellSignal())
         vote -= weight_2;

   if(WPRR.BuySignal())
      vote += weight_3;
   else
      if(WPRR.SellSignal())
         vote -= weight_3;

//--- Long positions when the close moving average is above the open
   if(vote > 0)
     {
      Trade.Buy(TradeInformation.MinVolume(),Symbol(),TradeInformation.GetAsk(),0,0,"");
      return;
     }

//--- Otherwise short
   else
      if(vote < 0)
        {
         Trade.Sell(TradeInformation.MinVolume(),Symbol(),TradeInformation.GetBid(),0,0,"");
         return;
        }
  }
//--- End of CheckSignal Scope
```

As with any application, we must end by unifying all system constants we’ve created.

```
//+------------------------------------------------------------------+
//| Undefine system constants                                        |
//+------------------------------------------------------------------+
#undef MA_SHIFT
#undef RSI_PRICE
#undef MA_TYPE
//+------------------------------------------------------------------+
```

We are now ready to start testing and optimizing our training strategy. We begin by selecting the expert advisor we’ve just built together. From there, we specify the symbol that we’re going to be testing our application on. We’ve been using the EURUSD throughout our discussion on the daily timeframe, as previously specified. The testing dates will be selected using a custom interval, and we’ve been running tests from February 2023 up until May 2025.

![](https://c.mql5.com/2/155/1265054292465.png)

Figure 3: The settings and dates that we will use for our genetic optimization

For forward testing, we select half of the data—meaning the first half will be used for backtesting and the latter for the forward test. The forward test is intended to show which strategies are stable and which ones are likely overfitting to the back test. We always select a random delay for the most authentic simulation of market events, and our modeling should be based on real ticks.

![](https://c.mql5.com/2/155/4469348943006.png)

Figure 4: Instructing the genetic optimizer, on which values and which intervals it should search our strategies parameters over

Lastly, for optimization, select the fast genetic-based algorithm. The total number of parameters in our strategy is a particular dimension we’ve tried our best to control and limit. But as we can see, the total number of steps required to optimize the strategy—even with modest settings—has grown considerable. It has indeed grown remarkably large in just one step. Therefore, I deemed it necessary for us to offload some of the work to the MQL5 Cloud. To follow along, you must first log into your MQL5 account and have a positive balance.

![](https://c.mql5.com/2/155/6402465015014.png)

Figure 5: Logging into your MQL5 user account through the MetaTrader 5 Terminal

Simply start the optimization procedure, then right-click on the number of cores available on your machine and select Use MQL5 Cloud Network. All of this is done on the Agents tab of your Strategy Tester.

![](https://c.mql5.com/2/155/5447257538142.png)

Figure 6: Enabling the MQL5 cloud, to accelerate backtesting

Once you enable the MQL5 Cloud, some of the tasks being performed on your machine will be offloaded to the cloud. This will help accelerate the optimization procedure, and hopefully, we’ll get our results faster—provided the network is secure and reliable.

![](https://c.mql5.com/2/155/4890241693690.png)

Figure 7: Connecting to any of the available data centers near you

The optimization results represent the tests being done on historical data accessible to the genetic optimizer. This allows it to evaluate how well the strategy is performing and adjust the parameters accordingly to improve performance. However, the genetic optimizer does not have access to the forward test results—these reflect how the selected strategy settings perform out-of-sample.

From the back test results, we can see that the profit levels are in line with those achieved in previous versions of our trading application. When we look at the top-performing strategies, we observe that the weights attributed to each sub-strategy fall within the range of 0.4 to 0.8. These top-performing weights are all quite close to each other, suggesting that the best-performing configuration in the back test employed all strategies.

![](https://c.mql5.com/2/155/61349902473.png)

Figure 8: The back test results of our genetic optimization process

However, when we turn to the forward test, we see that the best-performing strategies mostly relied on only two strategies. In fact, the top strategies had minimal weights for Strategy Three—some even assigned it a weight of zero.

What’s disheartening is that only a few of the top-performing strategies in the forward test were also profitable in the backtest. However, among the strategies that performed well in both tests, we again observed that Strategy Three had small weights—even in the most stable configurations we could find.

This, therefore, beckons us toward the idea of dropping Strategy Three, as it did not contribute meaningfully to the top-performing strategies from the forward test. However, this conclusion is based on the most profitable configuration found—and drawing conclusions this way is not always advisable, as it may lead us to overfit our decisions to the data at hand.

Yet, when we look across all strategies that were profitable in both tests, we generally find that in most instances, all three strategies had weights relatively close to each other. It’s only in this one standout instance where Strategy Three had the smallest weight and the best performance. It is challenging to make decisions under such uncertainty. However, this is the nature of the challenge set before us.

Therefore, believing that our actions are aligned with the best performance possible, we will conclude that Strategy Three is possibly not that important, and we will continue using only the first two strategies.

![](https://c.mql5.com/2/155/4627760241745.png)

Figure 9: The forward results from our genetic optimization process suggests to us that, strategy 3 may not be all that important for our success

### Conclusion

As you’ve seen from our discussion, determining the optimal number of strategies to be used in an ensemble application of multiple strategies can be a materially challenging task. We do not always know from the onset whether we will need one, five, or ten different strategies.

However, the main takeaway is that our genetic optimizer can help us tackle such difficult questions with ease. It is also worth noting that the genetic optimizer can be seen as a far more powerful tool than the popular ChatGPT and other LLMs that developers may rely on to answer some of the questions they face in their endeavor to build algorithmic applications.

As we’ve covered previously in our sister series of articles, Overcoming the Limitations of AI. We noticed that algorithms that are domain-aware are intrinsically more valuable to us than algorithms that are general-purpose. ChatGPT and other such LLMs are general-purpose algorithms, while the genetic optimizer embedded into your copy of MetaTrader 5 is a domain-aware algorithm, making it far superior to trying to pose the same question to ChatGPT.

We also demonstrated how you can start using the MQL5 Cloud to speed up your backtesting and optimization processes. This article possibly would not have been completed in time without the use of the MQL5 Cloud. It is easy to get started, and the rates are very affordable.

As a matter of fact, you have 24-hour provision of cloud computing with multiple redundant data centers. Just in case you lose connection due to network issues, you will remain connected almost always. All in all, the MQL5 Cloud and Genetic Optimizer are indispensable tools in the modern algorithmic developer’s toolkit.

This exercise will guide our decision-making processes as we move forward in developing statistical models for our trading strategy.

In our follow-up discussions, we will have already learned that the first two strategies may be enough for us to develop a binary classification task in which either Strategy One is the most profitable or Strategy Two is. We will then contrast the performance of our statistical modeling application against the initial strategy that we developed together.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18770.zip "Download all attachments in the single ZIP archive")

[MSA\_Test\_3.mq5](https://www.mql5.com/en/articles/download/18770/msa_test_3.mq5 "Download MSA_Test_3.mq5")(6.82 KB)

[WPRReversal.mqh](https://www.mql5.com/en/articles/download/18770/wprreversal.mqh "Download WPRReversal.mqh")(3.44 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Reimagining Classic Strategies (Part 21): Bollinger Bands And RSI Ensemble Strategy Discovery](https://www.mql5.com/en/articles/20933)
- [Reimagining Classic Strategies (Part 20): Modern Stochastic Oscillators](https://www.mql5.com/en/articles/20530)
- [Overcoming The Limitation of Machine Learning (Part 9): Correlation-Based Feature Learning in Self-Supervised Finance](https://www.mql5.com/en/articles/20514)
- [Reimagining Classic Strategies (Part 19): Deep Dive Into Moving Average Crossovers](https://www.mql5.com/en/articles/20488)
- [Overcoming The Limitation of Machine Learning (Part 8): Nonparametric Strategy Selection](https://www.mql5.com/en/articles/20317)
- [Overcoming The Limitation of Machine Learning (Part 7): Automatic Strategy Selection](https://www.mql5.com/en/articles/20256)
- [Self Optimizing Expert Advisors in MQL5 (Part 17): Ensemble Intelligence](https://www.mql5.com/en/articles/20238)

**[Go to discussion](https://www.mql5.com/en/forum/490792)**

![Introduction to MQL5 (Part 18): Introduction to Wolfe Wave Pattern](https://c.mql5.com/2/155/18555-introduction-to-mql5-part-18-logo.png)[Introduction to MQL5 (Part 18): Introduction to Wolfe Wave Pattern](https://www.mql5.com/en/articles/18555)

This article explains the Wolfe Wave pattern in detail, covering both the bearish and bullish variations. It also breaks down the step-by-step logic used to identify valid buy and sell setups based on this advanced chart pattern.

![Singular Spectrum Analysis in MQL5](https://c.mql5.com/2/155/18777-singular-spectrum-analysis-logo.png)[Singular Spectrum Analysis in MQL5](https://www.mql5.com/en/articles/18777)

This article is meant as a guide for those unfamiliar with the concept of Singular Spectrum Analysis and who wish to gain enough understanding to be able to apply the built-in tools available in MQL5.

![From Novice to Expert: Animated News Headline Using MQL5 (V)—Event Reminder System](https://c.mql5.com/2/156/18750-from-novice-to-expert-animated-logo.png)[From Novice to Expert: Animated News Headline Using MQL5 (V)—Event Reminder System](https://www.mql5.com/en/articles/18750)

In this discussion, we’ll explore additional advancements as we integrate refined event‑alerting logic for the economic calendar events displayed by the News Headline EA. This enhancement is critical—it ensures users receive timely notifications a short time before key upcoming events. Join this discussion to discover more.

![Master MQL5 from Beginner to Pro (Part VI): Basics of Developing Expert Advisors](https://c.mql5.com/2/102/Learning_MQL5_-_From_Beginner_to_Pro_Part_VI___LOGO.png)[Master MQL5 from Beginner to Pro (Part VI): Basics of Developing Expert Advisors](https://www.mql5.com/en/articles/15727)

This article continues the series for beginners. Here we will discuss the basic principles of developing Expert Advisors (EAs). We will create two EAs: the first one will trade without indicators, using pending orders, and the second one will be based on the standard MA indicator, opening deals at the current price. Here I assume that you are no longer a complete beginner and have a relatively good command of the material from the previous articles.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/18770&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049492685460450426)

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