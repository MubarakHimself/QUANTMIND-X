---
title: Developing a trading Expert Advisor from scratch (Part 13): Time and Trade (II)
url: https://www.mql5.com/en/articles/10412
categories: Trading Systems, Indicators
relevance_score: 6
scraped_at: 2026-01-23T11:47:13.543397
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=qxtibtzhnsakghdqdqnwmznhwvenyqco&ssn=1769158032809178326&ssn_dr=0&ssn_sr=0&fv_date=1769158032&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10412&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20trading%20Expert%20Advisor%20from%20scratch%20(Part%2013)%3A%20Time%20and%20Trade%20(II)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915803212572820&fz_uniq=5062716202570131343&sv=2552)

MetaTrader 5 / Trading systems


### Introduction

In the previous article ["Times & Trade (I)"](https://www.mql5.com/en/articles/10410) we discussed an alternative chart organization system, which is a prerequisite for creating an indicator enabling the quickest possible interpretation of deals executed in the market. But we have not completed this system: it still lacks the ability to show how you can access certain information, and such access would help to better understand what is happening. Such information cannot be presented directly on the chart. Actually, such presentation could be implemented, however the interpretation would be very confusing. Therefore, it is best to have the data represented in a classical way, i.e. values in text format. Our Expert Advisor does not have a system capable of performing this task. So, we need to implement it.

In order not to complicate the previous article by adding information that some readers may not need at all (as the system can be used without diving deep into such details), I decided to expand the system here and make it more complete, but I didn't include some of the things into the system that was proposed in the previous article. But this information may sometimes be necessary to understand what is actually happening in the market.

### Planning

It is important here to understand one thing. It's just details, but as saying goes _**the devil lives in the details**_. So, please have a look at the following image:

![](https://c.mql5.com/2/45/001__1.jpg)

Do you notice anything strange in this image? Something that might not make much sense, but it is here, so look very closely.

If you still haven't noticed anything strange, see the highlighted area below.

![](https://c.mql5.com/2/45/002__1.jpg)

No do you see what is happening? There were changes in the BID and ASK values at this point, but only one trade was performed here. Even if there were changes in the BID Or ASK value, it still makes no sense to have only one trade. But such things are actually more common than you might think. The problem is that such thing usually cannot be seen when you use the reading mode shown below:

![](https://c.mql5.com/2/45/003__1.jpg)

When using this market analyzing method, we cannot see the movement of BID and ASK values. It seems that the market is always working, that everyone is trying to close the deal, but this is not true. Actually, market _players_ place positions at certain points and wait for the market movement. When the position is hit, they try to take advantage and to profit from the move — because of this, the BID or ASK values move without any trades. This is a real fact that can be seen in the platform, which is however ignored by most people who believe that this information is not very important.

The figure below shows how our Times & Trade system will look like:

![](https://c.mql5.com/2/45/004.1.jpg)

If you look closely, you will see that there are four candlestick configurations on the chart. There should be five of them, but direct orders are excluded from the system because they do not actually move the market. Therefore, we actually have four configurations. These are the following formations:

![](https://c.mql5.com/2/45/002__2.jpg)![](https://c.mql5.com/2/45/004__1.jpg)

![](https://c.mql5.com/2/45/003__2.jpg)![](https://c.mql5.com/2/45/005__1.jpg)

The shadow sometimes does not touch the body of the candlestick. Why is this happening? The shadow is formed by the value of the spread, which in turn is the difference between BID and ASK, but if an operation takes place within this spread, then what the candlestick will look like? This will be the fifth type shown below:

![](https://c.mql5.com/2/45/005__2.jpg)

According to the formation type it is [DOJI](https://en.wikipedia.org/wiki/Doji "https://en.wikipedia.org/wiki/Doji"). This is the reason why direct orders are not shown in the system. But it does not explain why the body sometimes does not touch the shadow. Such behavior is connected with situations when something happens that causes the price to move too quickly, do to which there is a distance between the body and the shadow. One might think that this is a system failure, because it makes no sense for the price to do this. But here it does make sense, as this happens exactly when stop orders are triggered. To see this, take a look at the image below:

![](https://c.mql5.com/2/45/006.jpg)

There is a series of cases when there are orders but neither BID nor Ask is touched. All these points represent triggered stop orders. When this happens, the price usually jumps, which can be seen on the chart. The same fact can be visible on Times & Trade only if you are using the chart mode to evaluate the movement. Without this, you do not see the triggering of stops and may think that the movement has gained strength while actually it can get back quickly, and you will be hit by the stop.

Now that you know this, you will understand that a big series of candlesticks not touching the shadow represents triggered stop orders. In fact, it is impossible to capture this movement exactly when it occurs, since everything happens very quickly. But you can use the interpretation of the BID and ASK values to find out why this happened. This is up to you and your market experience. I won't go into details, but this is something that you should focus on if you really want to use Tape Reading as an indicator.

Now comes the detail: if this information can only be seen using candlesticks and they themselves are enough to learn some information, then why is it so necessary to have more data?

The great detail is that there are times when the market is slower, waiting for some information that can come out at the moment, but we can't know this by simply looking at the Times & Trade with candlesticks. We need something more than that. This information exists in the system itself, but it is difficult to interpret it as it comes. Data should be modeled so that it can be easier analyzed.

This modeling is the reason for writing this article: after this modeling is done, Times & Trade will change to look like this:

![](https://c.mql5.com/2/45/007.jpg)

In other words, we will have a complete picture of what is happening. Furthermore, everything will be fast, which is important for those who want to use tape reading as a way to trade.

### Implementation

To implement the system, we need to add several new variables to the C\_TimesAndTrade class. They are shown in the code below:

```
#include <NanoEA-SIMD\Auxiliar\C_FnSubWin.mqh>
#include <NanoEA-SIMD\Auxiliar\C_Canvas.mqh>
//+------------------------------------------------------------------+
class C_TimesAndTrade : private C_FnSubWin
{
//+------------------------------------------------------------------+
#define def_SizeBuff
2048
#define macro_Limits(A) (A & 0xFF)
#define def_MaxInfos 257
//+------------------------------------------------------------------+
        private :
                string          m_szCustomSymbol,
                                m_szObjName;
                char            m_ConnectionStatus;
                datetime        m_LastTime;
                ulong           m_MemTickTime;
                int             m_CountStrings;
                struct st0
                {
                        string  szTime;
                        int     flag;
                }m_InfoTrades[def_MaxInfos];
                struct st1
                {
                        C_Canvas Canvas;
                        int      WidthRegion,
                                 PosXRegion,
                                 MaxY;
                        string   szNameCanvas;
                }m_InfoCanvas;
```

Highlighting shows the parts that have been added to the source code. As you can see, we need to use the C\_Canvas class, but it does not have all the elements we need. In fact, we have to add four subroutines to this C\_Canvas class. These subroutines are shown in the code below:

```
// ... C_Canvas class code

inline void FontSet(const string name, const int size, const uint flags = 0, const uint angle = 0)
{
        if(!TextSetFont(name, size, flags, angle)) return;
        TextGetSize("M", m_TextInfos.width, m_TextInfos.height);
}
//+------------------------------------------------------------------+
inline void TextOutFast(int x, int y, string text, const uint clr, uint alignment = 0)
{
        TextOut(text, x, y, alignment, m_Pixel, m_width, m_height, clr, COLOR_FORMAT_ARGB_NORMALIZE);
}
//+------------------------------------------------------------------+
inline int TextWidth(void) const { return m_TextInfos.width; }
//+------------------------------------------------------------------+
inline int TextHeight(void) const { return m_TextInfos.height; }
//+------------------------------------------------------------------+

// ... The rest of the code ...
```

These lines create text. Very simple, nothing extremely elegant.

The next function in this class which is worth mentioning is C\_TimesAndTrade:

```
void PrintTimeTrade(void)
{
        int ui1;

        m_InfoCanvas.Canvas.Erase(clrBlack, 220);
        for (int c0 = 0, c1 = m_CountStrings - 1, y = 2; (c0 <= 255) && (y < m_InfoCanvas.MaxY); c0++, c1--, y += m_InfoCanvas.Canvas.TextHeight())
        if (m_InfoTrades[macro_Limits(c1)].szTime == NULL) break; else
        {
                ui1 = m_InfoTrades[macro_Limits(c1)].flag;
                m_InfoCanvas.Canvas.TextOutFast(2, y, m_InfoTrades[macro_Limits(c1)].szTime, macroColorRGBA((ui1 == 0 ? clrLightSkyBlue : (ui1 > 0 ? clrForestGreen : clrFireBrick)), 220));
        }
        m_InfoCanvas.Canvas.Update();
}
```

This function will display values in the special area reserved for this. In addition, the initialization procedure has also undergone minor changes, which can be seen below in the highlighted part:

```
void Init(const int iScale = 2)
{
        if (!ExistSubWin())
        {
                m_InfoCanvas.Canvas.FontSet("Lucida Console", 13);
                m_InfoCanvas.WidthRegion = (18 * m_InfoCanvas.Canvas.TextWidth()) + 4;
                CreateCustomSymbol();
                CreateChart();
                m_InfoCanvas.Canvas.Create(m_InfoCanvas.szNameCanvas, m_InfoCanvas.PosXRegion, 0, m_InfoCanvas.WidthRegion, TerminalInfoInteger(TERMINAL_SCREEN_HEIGHT), GetIdSubWinEA());
                Resize();
                m_ConnectionStatus = 0;
        }
        ObjectSetInteger(Terminal.Get_ID(), m_szObjName, OBJPROP_CHART_SCALE, (iScale > 5 ? 5 : (iScale < 0 ? 0 : iScale)));
}
```

Additional changes were also required in the substitution routine in the Times & Trade. The changes are as follows:

```
void Resize(void)
{
        static int MaxX = 0;
        int x = (int) ChartGetInteger(Terminal.Get_ID(), CHART_WIDTH_IN_PIXELS, GetIdSubWinEA());

        m_InfoCanvas.MaxY = (int) ChartGetInteger(Terminal.Get_ID(), CHART_HEIGHT_IN_PIXELS, GetIdSubWinEA());
        ObjectSetInteger(Terminal.Get_ID(), m_szObjName, OBJPROP_YSIZE, m_InfoCanvas.MaxY);
        if (MaxX != x)
        {
                MaxX = x;
                x -= m_InfoCanvas.WidthRegion;
                ObjectSetInteger(Terminal.Get_ID(), m_szObjName, OBJPROP_XSIZE, x);
                ObjectSetInteger(Terminal.Get_ID(), m_InfoCanvas.szNameCanvas, OBJPROP_XDISTANCE, x);
        }
        PrintTimeTrade();
}
```

The system is almost ready, but we still need the subroutine that is in the heart if the system. It has also been modified:

```
inline void Update(void)
{
        MqlTick Tick[];
        MqlRates Rates[def_SizeBuff];
        int i0, p1, p2 = 0;
        int iflag;
        long lg1;
        static int nSwap = 0;
        static long lTime = 0;

        if (m_ConnectionStatus < 3) return;
        if ((i0 = CopyTicks(Terminal.GetFullSymbol(), Tick, COPY_TICKS_ALL, m_MemTickTime, def_SizeBuff)) > 0)
        {
                for (p1 = 0, p2 = 0; (p1 < i0) && (Tick[p1].time_msc == m_MemTickTime); p1++);
                for (int c0 = p1, c1 = 0; c0 < i0; c0++)
                {
                        lg1 = Tick[c0].time_msc - lTime;
                        nSwap++;
                        if (Tick[c0].volume == 0) continue;
                        iflag = 0;
                        iflag += ((Tick[c0].flags & TICK_FLAG_BUY) == TICK_FLAG_BUY ? 1 : 0);
                        iflag -= ((Tick[c0].flags & TICK_FLAG_SELL) == TICK_FLAG_SELL ? 1 : 0);
                        if (iflag == 0) continue;
                        Rates[c1].high = Tick[c0].ask;
                        Rates[c1].low = Tick[c0].bid;
                        Rates[c1].open = Tick[c0].last;
                        Rates[c1].close = Tick[c0].last + ((Tick[c0].volume > 200 ? 200 : Tick[c0].volume) * (Terminal.GetTypeSymbol() == C_Terminal::WDO ? 0.02 : 1.0) * iflag);
                        Rates[c1].time = m_LastTime;
                        m_InfoTrades[macro_Limits(m_CountStrings)].szTime = StringFormat("%02.d.%03d ~ %02.d <>%04.d", ((lg1 - (lg1 % 1000)) / 1000) % 60 , lg1 % 1000, nSwap, Tick[c0].volume);
                        m_InfoTrades[macro_Limits(m_CountStrings)].flag = iflag;
                        m_CountStrings++;
                        nSwap = 0;
			lTime = Tick[c0].time_msc;
                        p2++;
                        c1++;
                        m_LastTime += 60;
                }
                CustomRatesUpdate(m_szCustomSymbol, Rates, p2);
                m_MemTickTime = Tick[i0 - 1].time_msc;
        }
        PrintTimeTrade();
}
```

The highlighted lines represent the code added to the subroutine to model the data we need. The following code

```
lg1 = Tick[c0].time_msc - lTime;
nSwap++;
```

checks how much time has passed between trades in milliseconds, and how many trades that did not cause a price change occurred. If these numbers are large, you can understand that the turnover is decreasing. With this feature you will notice this earlier than others.

The following part

```
m_InfoTrades[macro_Limits(m_CountStrings)].szTime = StringFormat("%02.d.%03d ~ %02.d <>%04.d", ((lg1 - (lg1 % 1000)) / 1000) % 60 , lg1 % 1000, nSwap, Tick[c0].volume);
m_InfoTrades[macro_Limits(m_CountStrings)].flag = iflag;
m_CountStrings++;
nSwap = 0;
lTime = Tick[c0].time_msc;
```

will model the values that will be presented. Please note that we will not test the **_m\_CountStrings_** counter due to its limited use. We will simply increase its values as new information becomes available. This is a trick that can sometimes be used. I myself use it when possible, as it is efficient in terms of processing, which is important as the trading system is designed to be used in real time. You should always try to optimize the system whenever possible, even if only a little — in the end it makes a big difference.

After everything has been implemented, compile the Expert Advisor and get something like this:

![](https://c.mql5.com/2/45/ScreenRecorderProject51.gif)

Watching the movements that described above on the Times & Trade chart, you can see that microstructures start appearing in the Times & Trade itself. However, even after studying these microstructures, I could not take any advantage of the fact that they exist. However, I'm not that experienced a trader, so who knows, maybe someone with more experience can do it.

This indicator is so powerful and so informative that I decided to make a video showing a small comparison between its values and real data shown by the asset at the time of writing. I want to demonstrate that it filters a lot of information, allowing you to read data much faster and understand better what is happening. I hope you enjoy and take advantage of this fantastic and powerful indicator.

Times And Trade Demonstração - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10412)

MQL5.community

1.91K subscribers

[Times And Trade Demonstração](https://www.youtube.com/watch?v=Ljbtck2ZOXo)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=Ljbtck2ZOXo&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F10412)

0:00

0:00 / 2:56

•Live

•

### Conclusion

The system proposed here is simply a modification of the charting system available in the MetaTrader 5 platform itself. What has been modified is the data modeling method. It can be interesting to see how closed trades affect price direction by forming microstructures in the lowest timeframe available on the platform, which is 1 minute. Many people like to say that they trade on the minute timeframe as if that means they have a high level of market knowledge. But if you look closer and get to understand trading processes, it becomes clear that a lot of things happen within 1 minute. So, although this seems like a short amount of time, this can cause us to miss many potentially profitable trades. Remember that in this Times & Trade system we are not looking at what happens within 1 minute — the values that appear on the screen are quoted in milliseconds.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/10412](https://www.mql5.com/pt/articles/10412)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10412.zip "Download all attachments in the single ZIP archive")

[EA\_-\_Times\_m\_Trade.zip](https://www.mql5.com/en/articles/download/10412/ea_-_times_m_trade.zip "Download EA_-_Times_m_Trade.zip")(5983.76 KB)

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

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/428104)**
(1)


![Sergei Poliukhov](https://c.mql5.com/avatar/avatar_na2.png)

**[Sergei Poliukhov](https://www.mql5.com/en/users/operlay)**
\|
3 Jul 2022 at 16:03

what broker do you use metatrader?


![Learn how to design a trading system by Williams PR](https://c.mql5.com/2/47/why-and-how__4.png)[Learn how to design a trading system by Williams PR](https://www.mql5.com/en/articles/11142)

A new article in our series about learning how to design a trading system by the most popular technical indicators by MQL5 to be used in the MetaTrader 5. In this article, we will learn how to design a trading system by the Williams' %R indicator.

![Developing a trading Expert Advisor from scratch (Part 12): Times and Trade (I)](https://c.mql5.com/2/46/development__3.png)[Developing a trading Expert Advisor from scratch (Part 12): Times and Trade (I)](https://www.mql5.com/en/articles/10410)

Today we will create Times & Trade with fast interpretation to read the order flow. It is the first part in which we will build the system. In the next article, we will complete the system with the missing information. To implement this new functionality, we will need to add several new things to the code of our Expert Advisor.

![Indicators with on-chart interactive controls](https://c.mql5.com/2/46/interactive-control.png)[Indicators with on-chart interactive controls](https://www.mql5.com/en/articles/10770)

The article offers a new perspective on indicator interfaces. I am going to focus on convenience. Having tried dozens of different trading strategies over the years, as well as having tested hundreds of different indicators, I have come to some conclusions I want to share with you in this article.

![Developing a trading Expert Advisor from scratch (Part 11): Cross order system](https://c.mql5.com/2/46/development__2.png)[Developing a trading Expert Advisor from scratch (Part 11): Cross order system](https://www.mql5.com/en/articles/10383)

In this article we will create a system of cross orders. There is one type of assets that makes traders' life very difficult for traders — futures contracts. But why do they make life difficult?

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/10412&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062716202570131343)

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