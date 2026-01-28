---
title: Developing a Replay System (Part 43): Chart Trade Project (II)
url: https://www.mql5.com/en/articles/11664
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:11:36.316923
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/11664&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070085743679836086)

MetaTrader 5 / Examples


### Introduction

In the previous article [Developing a Replay System (Part 42): Chart Trade Project (I)](https://www.mql5.com/en/articles/11652), I showed how you can arrange interaction between the mouse indicator and other indicators.

We began writing code to ensure that the Chart Trade indicator exists in perfect harmony with the mouse indicator. However, unlike what was done in the first versions of this Chart Trade indicator, described in the articles:

- [Developing a trading Expert Advisor from scratch (Part 30): CHART TRADE as an indicator?!](https://www.mql5.com/en/articles/10653)
- [Multiple indicators on one chart (Part 06): Turning MetaTrader 5 into a RAD system (II)](https://www.mql5.com/en/articles/10301)
- [Multiple indicators on one chart (Part 05): Turning MetaTrader 5 into a RAD system (I)](https://www.mql5.com/en/articles/10277)

here we will be doing something more advanced and therefore different. But in any case, the result will be the same as in video 01. I suggest you watch this video before you start reading the article so that you have an idea of what exactly we will be doing. This is not something that can be understood just by reading the code. They say an image is worth a thousand words, so watch the video to better understand what will be explained in the article.

Video from the article "Desenvolvendo um sistema de Replay (Parte 43): Projeto do Chart Trade (II)" - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11664)

MQL5.community

1.91K subscribers

[Video from the article "Desenvolvendo um sistema de Replay (Parte 43): Projeto do Chart Trade (II)"](https://www.youtube.com/watch?v=Z-tbTlurnOI)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

0:00

0:00 / 2:49

•Live

•

Video 01 - Demo video

The video shows that data is displayed in the Chart Trade window. You may have noticed that we do everything exactly as shown in the previous article, but the information is updated without actually using the objects. And you're probably wondering: how is this possible?

"This guy must be running some kind of trick. I've never seen anyone do things like this. This doesn't make any sense..." Others probably think that I am some kind of sorcerer or magician, with powers beyond the imagination. No. Nothing like this. I simply use both the MetaTrader 5 platform and the MQL5 language at a level that many people do not strive to understand. They always continue to say and do the same thing, without exploring the real potential and capabilities of either the MQL5 language or the MetaTrader 5 platform.

I hope everyone watched the video. Because in this article I'll show you how to do something that will fundamentally change the way you think about what's possible and what's impossible. Important: I will only explain what is shown in the video. The things not shown in the video will be discussed later.

### Updating the indicator code

The changes that need to be made will not be that big. But we will still move forward gradually, otherwise everyone will be left without a clear understanding of what is happening.

Let's start by looking at the changes made to the indicator code. The changes are shown in the code below.

**Chart Trade indicator source code:**

```
01. //+------------------------------------------------------------------+
02. #property copyright "Daniel Jose"
03. #property description "Base version for Chart Trade (DEMO version)"
04. #property version   "1.43"
05. #property icon "/Images/Market Replay/Icons/Indicators.ico"
06. #property link "https://www.mql5.com/es/articles/11664"
07. #property indicator_chart_window
08. #property indicator_plots 0
09. //+------------------------------------------------------------------+
10. #include <Market Replay\Chart Trader\C_ChartFloatingRAD.mqh>
11. //+------------------------------------------------------------------+
12. C_ChartFloatingRAD *chart = NULL;
13. //+------------------------------------------------------------------+
14. input int           user01 = 1;                     //Leverage
15. input double        user02 = 100.1;                 //Finance Take
16. input double        user03 = 75.4;                  //Finance Stop
17. //+------------------------------------------------------------------+
18. int OnInit()
19. {
20.     chart = new C_ChartFloatingRAD("Indicator Chart Trade", new C_Mouse("Indicator Mouse Study"), user01, user02, user03);
21.     if (_LastError != ERR_SUCCESS)
22.     {
23.             Print("Error number:", _LastError);
24.             return INIT_FAILED;
25.     }
26.
27.     return INIT_SUCCEEDED;
28. }
29. //+------------------------------------------------------------------+
30. int OnCalculate(const int rates_total, const int prev_calculated, const int begin, const double &price[])
31. {
32.     return rates_total;
33. }
34. //+------------------------------------------------------------------+
35. void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
36. {
37.     (*chart).DispatchMessage(id, lparam, dparam, sparam);
38.
39.     ChartRedraw();
40. }
41. //+------------------------------------------------------------------+
42. void OnDeinit(const int reason)
43. {
44.     delete chart;
45.
46.     ChartRedraw();
47. }
48. //+------------------------------------------------------------------+
```

If you compare it with the source code of the indicator, which was given in the article [Developing a Replay System (Part 42): Chart Trade Project (I)](https://www.mql5.com/en/articles/11652), you can see that it has undergone some changes. We have added a way to allow the user to specify the value that will initially be placed in the indicator. Although we have this form of interaction between lines 14 and 16, these points are not entirely necessary, although they are interesting for testing. Remember that Chart Trade is a system of interaction with the user.

Instead of lines 14 to 16, you can simply enter the default value into the indicator. This can be done on line 20, where instead of using values obtained from user interaction, we could put the values directly into the call. Thus, we will have Chart Trade with default values that can be changed by the user after the indicator appears on the chart.

In fact, this article will not cover that. The reason is that we will use a slightly different process to do this. But as you can see in the demo video, the values reported by the user are transferred to Chart Trade and displayed clearly. As already mentioned, the presence of such interaction is interesting for testing purposes.

We have seen that the indicator is capable of establishing communication. Now, let's figure out how to make changes reflected in the indicator, as shown in the video, albeit without actually using objects. The only object present on the graph, as seen in the video, is **OBJ\_CHART**. Any ideas how I managed to do this? How did I change data and values using only **OBJ\_CHART**?

If you have no idea how this is possible, you will have to study how the MetaTrader 5 platform and the MQL5 language actually work. To make the explanation clearer, let's start a new topic.

### Objects and objectives

Most people who want or dream of learning to program don't actually have a clue what they're doing. Their activity consists of trying to create things in a certain way. However, programming is not about tailoring suitable solutions Doing it this way can create more problems than solutions.

Perhaps this is not entirely clear. Well, I will try to convey a little of my many years of programming experience in various languages, platforms and systems.

Whenever we start using a system, whatever it may be, the first thing we should do is evaluate its capabilities, understand what the system offers us. This is in general terms. That is, we must try to understand why everything was created this way and not otherwise. Having understood how the most basic functions work, you can move on and begin to delve deeper.

Going deeper is not looking for ways to improve these functions or simply programming with them. The depth lies in trying to use those resources that have not yet been explored by others. Many people limit themselves to using only what everyone else usually uses, and that's not a bad thing. But how can you suggest improvements without only superficially familiarizing yourself with the capabilities of the system? This type of thinking makes no sense. It's like a child wanting to suggest a new production method just by looking at a running assembly line. This kind of thing has no future and is doomed to create more problems than solutions.

Perhaps I'm still not being clear enough. When seem MetaTrader 5, some people think they know enough about the platform to say what is possible or impossible to do with it. And then they turn to MQL5, trying to solve some of their issues. This is a mistake. You should not consider MQL5, like any other language, as a magical way to satisfy your needs or desires. MQL5 **is not** a magic language. This language extends the capabilities of MetaTrader 5. However, it is not capable of making the platform work differently than intended.

The right approach is to take a deep dive into how MetaTrader 5 works and then look for ways to adapt it to your own way of viewing and analyzing the market. As you explore MetaTrader 5, you'll see what you need to do to turn it into a tool that better suits your view of the market. To do this, you will have to use MQL5, which will open the doors of MetaTrader 5 in a way that makes life easier for you or other traders.

With this in mind, we can consider one question. Many users underestimate MetaTrader 5 due to ignorance of the basic details of its operation. One of these details, and not only in MetaTrader 5, is the concept of templates. They allow us to customize, adjust and organize certain things in a simple and practical way. From annotations to simplifying the process of viewing the market in a certain way at a certain moment.

Templates can contain a variety of things, and I've shown before how to explore some of them. In the article [Multiple indicators on one chart (Part 03): Developing custom definitions](https://www.mql5.com/en/articles/10239), I showed how different indicators can be placed next to each other, as shown in Figure 01.

![Figure 01](https://c.mql5.com/2/50/001__1.gif)

Figure 01 – Using several indicators in one subwindow

The concept shown above is only possible using templates. No matter how good a programmer you are and how extensive your knowledge of MQL5 is, you will never be able to achieve the result shown in Figure 01 without understanding how MetaTrader 5 functions. This is because not everything comes down to writing code. Programming only helps create a solution that would otherwise not be possible. However, it should not be the first attempt when you need something new, but rather a tool to achieve the desired result.

Why am I saying this? The reason is exactly what is shown in video 01 at the beginning of this article. No matter how well you can program or understand how MQL5 works, you will not be able to do what is shown in the video without understanding how MetaTrader 5 functions.

An important thing of this story is that when I published the article [Multiple indicators on one chart (Part 06): Turning MetaTrader 5 into a RAD system (II)](https://www.mql5.com/en/articles/10301), at that time I was not using certain MetaTrader 5 concepts. I was still stuck with some ideas and concepts that eventually turned out to be unsuitable. I'm not saying they were wrong, just less appropriate. There are much better solutions. This is due precisely to what was shown in that article, as well as in the previous one [Multiple indicators on one chart (Part 05): Turning MetaTrader 5 into a RAD system (I)](https://www.mql5.com/en/articles/10277).

In both cases, we opened the template and tried to work on it. But all this work was nothing more than a mere scratch on the surface, since much of what you see is reprogramming of the same template. This was done in such a way as to emulate the RAD system, which can be implemented by programming in MQL5.

Although this worked, you will notice that in fact the template created as in MetaTrader 5 was later recreated by the indicator itself when the indicator was placed on the chart. Consequently, all objects that existed in the template were recreated by the indicator. In this way, MetaTrader 5 gained access to the objects located there, which allowed it to configure and change the values present in each of them.

However, as time passed, it became clear that the idea could be improved. So, I found a more suitable Chart Trade model, which uses a minimum of objects. Well, we still need to do something more besides what will be shown in this article. However, video 01 shows that there have been changes in the values. At the same time, we only have one object on the screen.

This is exactly what programming allows us to do. Without it, we would be limited to what is available in MetaTrader 5. Through programming, we expand the capabilities of the platform. this is achieved by using what already exists in MetaTrader 5. we create a template with exactly what we need and place it in OBJ\_CHART. But without proper programming, it would be impossible to change the values present in the objects inside the template. By using programming correctly, we solve this problem, thereby expanding the capabilities of MetaTrader 5.

To make this clearer, open the template file used in Chart Trade. It is presented in full below.

**Chart Trade file (template)**:

```
001. <chart>
002. fore=0
003. grid=0
004. volume=0
005. ticker=0
006. ohlc=0
007. one_click=0
008. one_click_btn=0
009. bidline=0
010. askline=0
011. lastline=0
012. descriptions=0
013. tradelines=0
014. tradehistory=0
015. background_color=16777215
016. foreground_color=0
017. barup_color=16777215
018. bardown_color=16777215
019. bullcandle_color=16777215
020. bearcandle_color=16777215
021. chartline_color=16777215
022. volumes_color=16777215
023. grid_color=16777215
024. bidline_color=16777215
025. askline_color=16777215
026. lastline_color=16777215
027. stops_color=16777215
028. windows_total=1
029.
030. <window>
031. height=100.000000
032. objects=18
033.
034. <indicator>
035. name=Main
036. path=
037. apply=1
038. show_data=1
039. scale_inherit=0
040. scale_line=0
041. scale_line_percent=50
042. scale_line_value=0.000000
043. scale_fix_min=0
044. scale_fix_min_val=0.000000
045. scale_fix_max=0
046. scale_fix_max_val=0.000000
047. expertmode=0
048. fixed_height=-1
049. </indicator>
050.
051. <object>
052. type=110
053. name=MSG_NULL_000
054. color=0
055. pos_x=0
056. pos_y=0
057. size_x=170
058. size_y=210
059. bgcolor=16777215
060. refpoint=0
061. border_type=1
062. </object>
063.
064. <object>
065. type=110
066. name=MSG_NULL_001
067. color=0
068. pos_x=5
069. pos_y=30
070. size_x=152
071. size_y=26
072. bgcolor=12632256
073. refpoint=0
074. border_type=1
075. </object>
076.
077. <object>
078. type=110
079. name=MSG_NULL_002
080. color=0
081. pos_x=5
082. pos_y=58
083. size_x=152
084. size_y=26
085. bgcolor=15130800
086. refpoint=0
087. border_type=1
088. </object>
089.
090. <object>
091. type=110
092. name=MSG_NULL_003
093. color=0
094. pos_x=5
095. pos_y=86
096. size_x=152
097. size_y=26
098. bgcolor=10025880
099. refpoint=0
100. border_type=1
101. </object>
102.
103. <object>
104. type=110
105. name=MSG_NULL_004
106. color=0
107. pos_x=5
108. pos_y=114
109. size_x=152
110. size_y=26
111. bgcolor=8036607
112. refpoint=0
113. border_type=1
114. </object>
115.
116. <object>
117. type=102
118. name=MSG_NULL_007
119. descr=Lever ( x )
120. color=0
121. style=1
122. angle=0
123. pos_x=10
124. pos_y=63
125. fontsz=16
126. fontnm=System
127. anchorpos=0
128. refpoint=0
129. </object>
130.
131. <object>
132. type=102
133. name=MSG_NULL_008
134. descr=Take ( $ )
135. color=0
136. style=1
137. angle=0
138. pos_x=10
139. pos_y=91
140. fontsz=16
141. fontnm=System
142. anchorpos=0
143. refpoint=0
144. </object>
145.
146. <object>
147. type=102
148. name=MSG_NULL_009
149. descr=Stop ( $ )
150. color=0
151. style=1
152. angle=0
153. pos_x=10
154. pos_y=119
155. fontsz=16
156. fontnm=System
157. anchorpos=0
158. refpoint=0
159. </object>
160.
161. <object>
162. type=107
163. name=MSG_TITLE_IDE
164. descr=Chart Trade
165. color=16777215
166. style=1
167. pos_x=0
168. pos_y=0
169. size_x=164
170. size_y=28
171. bgcolor=16748574
172. fontsz=16
173. fontnm=System
174. refpoint=0
175. readonly=1
176. align=0
177. </object>
178.
179. <object>
180. type=106
181. name=MSG_BUY_MARKET
182. size_x=70
183. size_y=25
184. offset_x=0
185. offset_y=0
186. pos_x=5
187. pos_y=145
188. bmpfile_on=\Images\Market Replay\Chart Trade\BUY.bmp
189. bmpfile_off=\Images\Market Replay\Chart Trade\BUY.bmp
190. state=0
191. refpoint=0
192. anchorpos=0
193. </object>
194.
195. <object>
196. type=106
197. name=MSG_SELL_MARKET
198. size_x=70
199. size_y=25
200. offset_x=0
201. offset_y=0
202. pos_x=85
203. pos_y=145
204. bmpfile_on=\Images\Market Replay\Chart Trade\SELL.bmp
205. bmpfile_off=\Images\Market Replay\Chart Trade\SELL.bmp
206. state=0
207. refpoint=0
208. anchorpos=0
209. </object>
210.
211. <object>
212. type=103
213. name=MSG_CLOSE_POSITION
214. descr=Close Position
215. color=0
216. style=1
217. pos_x=5
218. pos_y=173
219. fontsz=16
220. fontnm=System
221. state=0
222. size_x=152
223. size_y=26
224. bgcolor=1993170
225. frcolor=-1
226. refpoint=0
227. </object>
228.
229. <object>
230. type=107
231. name=MSG_NAME_SYMBOL
232. descr=?
233. color=0
234. style=1
235. pos_x=7
236. pos_y=34
237. size_x=116
238. size_y=20
239. bgcolor=12632256
240. fontsz=14
241. fontnm=Tahoma
242. refpoint=0
243. readonly=1
244. align=1
245. </object>
246.
247. <object>
248. type=107
249. name=MSG_LEVERAGE_VALUE
250. descr=?
251. color=0
252. style=1
253. pos_x=80
254. pos_y=61
255. size_x=70
256. size_y=18
257. bgcolor=15130800
258. fontsz=12
259. fontnm=Tahoma
260. refpoint=0
261. readonly=0
262. align=1
263. </object>
264.
265. <object>
266. type=107
267. name=MSG_TAKE_VALUE
268. descr=?
269. color=0
270. style=1
271. pos_x=80
272. pos_y=91
273. size_x=70
274. size_y=18
275. bgcolor=10025880
276. fontsz=12
277. fontnm=Tahoma
278. refpoint=0
279. readonly=0
280. align=1
281. </object>
282.
283. <object>
284. type=107
285. name=MSG_STOP_VALUE
286. descr=?
287. color=0
288. style=1
289. pos_x=80
290. pos_y=119
291. size_x=70
292. size_y=18
293. bgcolor=8036607
294. fontsz=12
295. fontnm=Tahoma
296. refpoint=0
297. readonly=0
298. align=1
299. </object>
300.
301. <object>
302. type=106
303. name=MSG_DAY_TRADE
304. size_x=32
305. size_y=22
306. offset_x=0
307. offset_y=0
308. pos_x=123
309. pos_y=33
310. bmpfile_on=\Images\Market Replay\Chart Trade\DT.bmp
311. bmpfile_off=\Images\Market Replay\Chart Trade\SW.bmp
312. state=0
313. refpoint=0
314. anchorpos=0
315. </object>
316.
317. <object>
318. type=106
319. name=MSG_MAX_MIN
320. size_x=20
321. size_y=20
322. offset_x=0
323. offset_y=0
324. pos_x=140
325. pos_y=3
326. bmpfile_on=\Images\Market Replay\Chart Trade\Maximize.bmp
327. bmpfile_off=\Images\Market Replay\Chart Trade\Minimize.bmp
328. state=1
329. refpoint=0
330. anchorpos=0
331. </object>
332.
333. </window>
334. </chart>
```

I know its content may seem completely unnecessary. Please note **THERE WILL BE NO FILES IN THE ATTACHMENT**, the entire system will be published in such a way that the files can be obtained by rewriting the codes into the corresponding files. I do this to ensure that you read and understand the article. I don't want you to just download attached files and use the system without knowing what you are doing. While it may seem like this makes the article significantly longer, it actually isn't, it just makes the explanation much more detailed.

Now let's take a closer look at the Chart Trade template file. Pay attention to the lines containing the following:

**descr=?**

It occurs in several places in the template file, and is done intentionally. Why? Although **_descr_** appears in different places, these places, as shown above, mainly refer to type 107 objects. What are objects of type 107?

A description of these types of objects first appeared in the article [Multiple indicators on one chart (Part 05): Turning MetaTrader 5 into a RAD (I) system](https://www.mql5.com/en/articles/10277). For convenience, I will provide the table presented in this article below:

| The value of the TYPE variable | Referenced object |
| --- | --- |
| 102 | [OBJ\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_label) |
| 103 | [OBJ\_BUTTON](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_button) |
| 106 | [OBJ\_BITMAP\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_bitmap_label) |
| 107 | [OBJ\_EDIT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_edit) |
| 110 | [OBJ\_RECTANGLE\_LABEL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_rectangle_label) |

So, object 107 is OBJ\_EDIT. These are objects into which the user can enter some information.

But it's not possible to directly access or enter a value into an object that is in a template. It is a fact. How did I solve this problem so that object 107 present in the template can receive values from the indicator? To answer this question, you need to look at the code of the class responsible for using the template.

### **Updating the C\_ChartFloatingRAD class**

To make Chart Trade display user-entered values, you need to perform some basic operations immediately after launching the indicator on the chart. These operations are aimed at solving some problems that cannot be solved without the use of programming. For this reason, it is very important to have a good understanding of how MetaTrader 5 works. Without knowing platform operation principles, you will not be able to create the code necessary to overcome such obstacles that cannot be solved without programming.

The complete code for the C\_ChartFloatingRAD class can be seen below. This code, in its current form, allows the indicator to work as shown in video 01. Can you understand how this happens?

**C\_ChartFloatingRAD class source code**:

```
001. //+------------------------------------------------------------------+
002. #property copyright "Daniel Jose"
003. //+------------------------------------------------------------------+
004. #include "../Auxiliar/C_Terminal.mqh"
005. #include "../Auxiliar/C_Mouse.mqh"
006. //+------------------------------------------------------------------+
007. class C_ChartFloatingRAD : private C_Terminal
008. {
009.    private :
010.            struct st00
011.            {
012.                    int     x, y, cx, cy;
013.                    string  szObj_Chart,
014.                            szFileNameTemplate;
015.                    long    WinHandle;
016.                    double  FinanceTake,
017.                            FinanceStop;
018.                    int     Leverage;
019.            }m_Info;
020. //+------------------------------------------------------------------+
021.            C_Mouse *m_Mouse;
022. //+------------------------------------------------------------------+
023.            void CreateWindowRAD(int x, int y, int w, int h)
024.                    {
025.                            m_Info.szObj_Chart = (string)ObjectsTotal(GetInfoTerminal().ID);
026.                            ObjectCreate(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJ_CHART, 0, 0, 0);
027.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XDISTANCE, m_Info.x = x);
028.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YDISTANCE, m_Info.y = y);
029.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XSIZE, w);
030.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YSIZE, h);
031.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_DATE_SCALE, false);
032.                            ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_PRICE_SCALE, false);
033.                            m_Info.WinHandle = ObjectGetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_CHART_ID);
034.                            m_Info.cx = w;
035.                            m_Info.cy = 26;
036.                    };
037. //+------------------------------------------------------------------+
038.            void Level_1(int &fpIn, int &fpOut, const string szFind, const string szWhat, const string szValue)
039.                    {
040.                            string sz0 = "";
041.                            int i0 = 0;
042.                            string res[];
043.
044.                            while ((!FileIsEnding(fpIn)) && (sz0 != "</object>"))
045.                            {
046.                                    sz0 = FileReadString(fpIn);
047.                                    if (StringSplit(sz0, '=', res) > 1)
048.                                    {
049.                                            i0 = (res[1] == szFind ? 1 : i0);
050.                                            if ((i0 == 1) && (res[0] == szWhat))
051.                                            {
052.                                                    FileWriteString(fpOut, szWhat + "=" + szValue + "\r\n");
053.                                                    return;
054.                                            }
055.                                    }
056.                                    FileWriteString(fpOut, sz0 + "\r\n");
057.                            };
058.                    }
059. //+------------------------------------------------------------------+
060.            void SwapValueInTemplate(const string szFind, const string szWhat, const string szValue)
061.                    {
062.                            int fpIn, fpOut;
063.                            string sz0;
064.
065.                            if (_LastError != ERR_SUCCESS) return;
066.                            if ((fpIn = FileOpen(m_Info.szFileNameTemplate, FILE_READ | FILE_TXT)) == INVALID_HANDLE)
067.                            {
068.                                    SetUserError(C_Terminal::ERR_FileAcess);
069.                                    return;
070.                            }
071.                            if ((fpOut = FileOpen(m_Info.szFileNameTemplate + "_T", FILE_WRITE | FILE_TXT)) == INVALID_HANDLE)
072.                            {
073.                                    FileClose(fpIn);
074.                                    SetUserError(C_Terminal::ERR_FileAcess);
075.                                    return;
076.                            }
077.                            while (!FileIsEnding(fpIn))
078.                            {
079.                                    sz0 = FileReadString(fpIn);
080.                                    FileWriteString(fpOut, sz0 + "\r\n");
081.                                    if (sz0 == "<object>") Level_1(fpIn, fpOut, szFind, szWhat, szValue);
082.                            };
083.                            FileClose(fpIn);
084.                            FileClose(fpOut);
085.                            if (!FileMove(m_Info.szFileNameTemplate + "_T", 0, m_Info.szFileNameTemplate, FILE_REWRITE))
086.                            {
087.                                    FileDelete(m_Info.szFileNameTemplate + "_T");
088.                                    SetUserError(C_Terminal::ERR_FileAcess);
089.                            }
090.                    }
091. //+------------------------------------------------------------------+
092. inline void UpdateChartTemplate(void)
093.                    {
094.                            ChartApplyTemplate(m_Info.WinHandle, "/Files/" + m_Info.szFileNameTemplate);
095.                            ChartRedraw(m_Info.WinHandle);
096.                    }
097. //+------------------------------------------------------------------+
098. inline double PointsToFinance(const double Points)
099.                    {
100.                            return Points * (GetInfoTerminal().VolumeMinimal + (GetInfoTerminal().VolumeMinimal * (m_Info.Leverage - 1))) * GetInfoTerminal().AdjustToTrade;
101.                    };
102. //+------------------------------------------------------------------+
103.    public  :
104. //+------------------------------------------------------------------+
105.            C_ChartFloatingRAD(string szShortName, C_Mouse *MousePtr, const int Leverage, const double FinanceTake, const double FinanceStop)
106.                    :C_Terminal()
107.                    {
108.                            m_Mouse = MousePtr;
109.                            m_Info.Leverage = (Leverage < 0 ? 1 : Leverage);
110.                            m_Info.FinanceTake = PointsToFinance(FinanceToPoints(MathAbs(FinanceTake), m_Info.Leverage));
111.                            m_Info.FinanceStop = PointsToFinance(FinanceToPoints(MathAbs(FinanceStop), m_Info.Leverage));
112.                            if (!IndicatorCheckPass(szShortName)) SetUserError(C_Terminal::ERR_Unknown);
113.                            m_Info.szFileNameTemplate = StringFormat("Chart Trade/%u.tpl", GetInfoTerminal().ID);
114.                            if (!FileCopy("Chart Trade/IDE_RAD.tpl", 0, m_Info.szFileNameTemplate, FILE_REWRITE)) SetUserError(C_Terminal::ERR_FileAcess);
115.                            if (_LastError != ERR_SUCCESS) return;
116.                            SwapValueInTemplate("MSG_NAME_SYMBOL", "descr", GetInfoTerminal().szSymbol);
117.                            SwapValueInTemplate("MSG_LEVERAGE_VALUE", "descr", (string)m_Info.Leverage);
118.                            SwapValueInTemplate("MSG_TAKE_VALUE", "descr", (string)m_Info.FinanceTake);
119.                            SwapValueInTemplate("MSG_STOP_VALUE", "descr", (string)m_Info.FinanceStop);
120.                            if (_LastError != ERR_SUCCESS) return;
121.                            CreateWindowRAD(0, 0, 170, 210);
122.                            UpdateChartTemplate();
123.                    }
124. //+------------------------------------------------------------------+
125.            ~C_ChartFloatingRAD()
126.                    {
127.                            ObjectDelete(GetInfoTerminal().ID, m_Info.szObj_Chart);
128.                            FileDelete(m_Info.szFileNameTemplate);
129.
130.                            delete m_Mouse;
131.                    }
132. //+------------------------------------------------------------------+
133.            void DispatchMessage(const int id, const long &lparam, const double &dparam, const string &sparam)
134.                    {
135.                            static int sx = -1, sy = -1;
136.                            int x, y, mx, my;
137.
138.                            switch (id)
139.                            {
140.                                    case CHARTEVENT_MOUSE_MOVE:
141.                                            if ((*m_Mouse).CheckClick(C_Mouse::eClickLeft))
142.                                            {
143.                                                    x = (int)lparam;
144.                                                    y = (int)dparam;
145.                                                    if ((x > m_Info.x) && (x < (m_Info.x + m_Info.cx)) && (y > m_Info.y) && (y < (m_Info.y + m_Info.cy)))
146.                                                    {
147.                                                            if (sx < 0)
148.                                                            {
149.                                                                    ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, false);
150.                                                                    sx = x - m_Info.x;
151.                                                                    sy = y - m_Info.y;
152.                                                            }
153.                                                            if ((mx = x - sx) > 0) ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_XDISTANCE, m_Info.x = mx);
154.                                                            if ((my = y - sy) > 0) ObjectSetInteger(GetInfoTerminal().ID, m_Info.szObj_Chart, OBJPROP_YDISTANCE, m_Info.y = my);
155.                                                    }
156.                                            }else if (sx > 0)
157.                                            {
158.                                                    ChartSetInteger(GetInfoTerminal().ID, CHART_MOUSE_SCROLL, true);
159.                                                    sx = sy = -1;
160.                                            }
161.                                            break;
162.                            }
163.                    }
164. //+------------------------------------------------------------------+
165. };
166. //+------------------------------------------------------------------+
```

Please note that the C\_ChartFloatingRAD file has received virtually no significant additions since the previous article. This is intentional. If I showed the final version of this file, you would not be able to understand how and why MetaTrader 5 manages to modify the values in the template. This is done so that when opening the object list window with **CTRL+B**, you only see **OBJ\_CHART** in the list, but the values and window would still be changed.

Now let's try to understand how and why MetaTrader 5 can inform us about what is happening through a template. Remember this point: we do not stop using the template, and it will be applied to the **OBJ\_CHART** object. This happens when line 94 is executed and line 95 tells MetaTrader 5 to update **OBJ\_CHART**. If you do not consider this fact, you can spend a long time looking for something in the code that is not there. For example, for the things that are actually in the template file.

The added code can be seen between lines 38 and 123. It is structured in such a way that the further steps that will be performed in the next article will not require a lot of changes, which is very good. So, pay attention to the explanations from this article and you will understand the next one very easily. You might even be able to predict what will be done in the next article from a programming perspective.

Let's start with the class constructor, as this makes the other parts much easier to understand. This constructor starts on line 105. But the real change starts on line 109. In this line, we guarantee that the leverage value will always be greater than or equal to 1. For this, we use the ternary operator. So, there is nothing particularly complicated or confusing here.

Now let's move on to line 110. This line, like line 111, implements an adjustment to the value displayed in the Chart Trade window. The reason for this is that the user may enter a value that does is not valid for the asset on which the Chart Trade indicator is running. For example, in the case of dollar futures, movements always occur every $5.00 per contract. In this case, specifying take profit or stop loss as a value that is not a multiple of 5, for example, take profit=$76.00, does not make sense, since this value will never be reached.

So, lines 110 and 111 make two calls. The first one converts the financial value into points, and the second one does the opposite, converting the point value into a financial value. You might think that this returns the value to its original value. This is true, but here mathematical methods will be used to convert and adjust the values. The first call will use the function present in the C\_Terminal class. In the second one, we execute the function present in line 98.

This function at 98 uses only one line. Line 100 actually performs a calculation that converts the given number of pips into a financial value.

This was the easy part. Now let's look at the most difficult part of the current system.

Once the constructor reaches line 113, we get something very interesting. This line 113 will create the name of the temporary file. Remember, this file will be temporary. It will be present in the area defined in **MQL5\\Files**, along with other information created specifically on this line 113.

Once we have the filename created on line 113, we move on to line 114. In this line, we will completely copy the contents of the template file (which can be seen in the previous topic) under a new name. If we succeed, we will continue further. But if copying fails, we will report this to the caller.

The real magic happens between lines 116 and 119. Here we make the Chart Trade indicator receive values provided by MetaTrader 5 and/or the user. These lines call line 60 of the class code. From now on, we will work on this part of the code as the rest of the constructor was explained in the previous article. But before we finally move on to line 60, let me draw your attention to another line that was added to the code: line 128. It deletes the file created in the constructor. That's why I emphasized that the file created is temporary.

Let's move on to line 60. I need to repeat something here (I'm still explaining the basics, so be patient). The first step is to open the "original" file (note the quotes in the word), this is done on line 66. If we succeed, we will continue. If we fail, we will return to the caller.

On line 71, we will create a temporary file from a temporary file (sounds strange, but there's no other way to say it). If this fails, we will close the open file and return to the caller. Until line 77, all we did was open one file and create another. If execution reaches line 77, we will enter a loop to copy the entire file.

Wait, copying the entire file? Again? Yes, but this time we'll test one condition on line 81. This condition will check whether an object in the template file was found during the copy process. When this happens, we will move to line 38, where we will deal with the found object. In any case, if the pattern being analyzed is a pattern from the previous topic, we will jump to line 38. So, even if you want to use your own template, you can do so as long as you define the names used for the objects correctly.

After calling the function presented on line 38, we will move until execution reaches line 44. Here we will continue the process of copying the file in the same way as we did earlier. However, when the line that closes the object is found on this very line 44, we will return to the calling object, that is, line 81. Let's stop and take a moment to see the function that appears on line 38.

During the copying process, we will read the source file at line 46. After that, we will expand the contents of this line. If line 47, which performs this decomposition, reports that we have two instructions available, a new execution thread will be generated. Otherwise, the read line will be written, and this happens on line 56.

Now pay attention to the following point: during the decomposition performed on line 47, the **i0** variable has the initial value of 0. Pay special attention to this. Once the object name is found, the **i0** variable will be set to 1. This is where the danger lies. If you manually edit a template file, you must ensure that the object name appears before any of its parameters. The only parameter that can appear before the name is the type of the object. No other parameters should come before the name. If this happens, the entire process will fail.

On line 50, we check if **i0** indicates that the desired object has been found. In this case, the value of this variable is 1. But we have a second condition, which concerns the parameter we are looking for. In this case, we will always search for **descr**. If these two conditions occur, we write the desired value during the copy process. This type of modification occurs on line 52. After that, we return to the caller, that is, line 81.

Watch how this process happens. We copy the entire file, except for one specific line. It is this line that will be changed, as a result of which new data will appear in the template or it will take on a different appearance.

When the entire file has been copied and modified, we will close both files, which is done on lines 83 and 84. After that, on line 85, we will try to rename the temporary file to the file that the class expects to use. The operation will be exactly the same as shown in video 01.

### Conclusion

In this material, I explained the first steps that you need to follow in order to be able to manage the template. This way it adapts to what we need. This will allow us to better use the capabilities of MetaTrader 5 using fairly simple programming. However, the important thing is not to use more or fewer objects on the chart. What's really important, is to understand that MQL5 allows everyone to turn MetaTrader 5 into a platform suitable for their needs, and not to create a different competing platform.

Translated from Portuguese by MetaQuotes Ltd.

Original article: [https://www.mql5.com/pt/articles/11664](https://www.mql5.com/pt/articles/11664)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11664.zip "Download all attachments in the single ZIP archive")

[Anexo.zip](https://www.mql5.com/en/articles/download/11664/anexo.zip "Download Anexo.zip")(420.65 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/470996)**

![Price-Driven CGI Model: Advanced Data Post-Processing and Implementation](https://c.mql5.com/2/87/Price-Driven_CGI_Model__2__LOGO__2.png)[Price-Driven CGI Model: Advanced Data Post-Processing and Implementation](https://www.mql5.com/en/articles/15319)

In this article, we will explore the development of a fully customizable Price Data export script using MQL5, marking new advancements in the simulation of the Price Man CGI Model. We have implemented advanced refinement techniques to ensure that the data is user-friendly and optimized for animation purposes. Additionally, we will uncover the capabilities of Blender 3D in effectively working with and visualizing price data, demonstrating its potential for creating dynamic and engaging animations.

![Integrating MQL5 with data processing packages (Part 1): Advanced Data analysis and Statistical Processing](https://c.mql5.com/2/87/Integrating_MQL5_with_data_processing_packages_Part_1___LOGO.png)[Integrating MQL5 with data processing packages (Part 1): Advanced Data analysis and Statistical Processing](https://www.mql5.com/en/articles/15155)

Integration enables seamless workflow where raw financial data from MQL5 can be imported into data processing packages like Jupyter Lab for advanced analysis including statistical testing.

![MQL5 Wizard Techniques you should know (Part 30): Spotlight on Batch-Normalization in Machine Learning](https://c.mql5.com/2/87/MQL5_Wizard_Techniques_you_should_know_Part_30___LOGO.png)[MQL5 Wizard Techniques you should know (Part 30): Spotlight on Batch-Normalization in Machine Learning](https://www.mql5.com/en/articles/15466)

Batch normalization is the pre-processing of data before it is fed into a machine learning algorithm, like a neural network. This is always done while being mindful of the type of Activation to be used by the algorithm. We therefore explore the different approaches that one can take in reaping the benefits of this, with the help of a wizard assembled Expert Advisor.

![Build Self Optimizing Expert Advisors With MQL5 And Python (Part II): Tuning Deep Neural Networks](https://c.mql5.com/2/87/Build_Self_Optimizing_Expert_Advisors_With_MQL5_And_Python_Part_II___LOGO__2.png)[Build Self Optimizing Expert Advisors With MQL5 And Python (Part II): Tuning Deep Neural Networks](https://www.mql5.com/en/articles/15413)

Machine learning models come with various adjustable parameters. In this series of articles, we will explore how to customize your AI models to fit your specific market using the SciPy library.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=kyylseijqjnucsqalspckgzinwrkvivy&ssn=1769184694406335304&ssn_dr=0&ssn_sr=0&fv_date=1769184694&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11664&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20Replay%20System%20(Part%2043)%3A%20Chart%20Trade%20Project%20(II)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918469492478071&fz_uniq=5070085743679836086&sv=2552)

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