---
title: Analyzing binary code of prices on the exchange (Part II): Converting to BIP39 and writing GPT model
url: https://www.mql5.com/en/articles/17110
categories: Integration, Machine Learning
relevance_score: 0
scraped_at: 2026-01-24T14:01:12.327621
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=nirzbutuqquagkkqfiyaeewxgbddaxcl&ssn=1769252471419978645&ssn_dr=0&ssn_sr=0&fv_date=1769252471&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F17110&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Analyzing%20binary%20code%20of%20prices%20on%20the%20exchange%20(Part%20II)%3A%20Converting%20to%20BIP39%20and%20writing%20GPT%20model%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925247119641281&fz_uniq=5083284388633909366&sv=2552)

MetaTrader 5 / Examples


## Introduction

In our constant quest to understand the market language, we often forget that all our technical indicators, candlestick patterns, and wave theories are attempts to translate its messages into a language we understand. In the [first part](https://www.mql5.com/en/articles/16741) of the research, we took a radical step — we presented price movements as a binary code, turning a complex dance of charts into a simple sequence of zeros and ones. But what if we go even further?

Imagine for a minute: what if the market could speak to us in words? Not metaphorically, through charts and indicators, but literally — using human language? It is this idea that we develop in the second part of our study, using BIP39 protocol, the one that is used in cryptocurrency wallets to create mnemonic phrases.

Why choose BIP39? This protocol was created to turn random sequences of bits into memorable English words. In cryptocurrencies, it is used to create seed phrases, but we saw something more in it - an opportunity to turn the "digital whispers" of the market into meaningful sentences.

But simply translating binary code into words is not enough. We need an "artificial intelligence" capable of understanding these words and finding hidden patterns in them. Here, a transformer architecture similar to the one used in GPT comes to the rescue. Imagine it as an artificial brain that reads the "book of the market" written in the BIP39 language and learns to understand its deeper meaning.

In a sense, we are not just creating another technical indicator - we are developing a real translator from the language of the market to that of the human and vice versa. This translator not only mechanically converts numbers into words, but also tries to capture the very essence of market movements, their internal logic and hidden patterns.

Do you remember the "Arrival" movie where a linguist tried to decrypt aliens' language? Our task is somewhat similar. We are also trying to decrypt someone else's language, the language of the market. And as in that movie, understanding this language can give us not only practical benefits, but also a completely new perspective on the nature of what we are working with.

This article will consider in detail how to implement such a "translator" using modern machine learning tools, and more importantly, how to interpret its "translations". We will see that some words and phrases appear more often than others in certain market situations, as if the market really uses its own vocabulary to describe its conditions.

### The main components of the system: digital alchemy in action

Do you know what is the hardest thing about creating something new? Choosing the right building blocks for the entire system. In our case, there are three such bricks, and each of them is unique in its own way. Let me tell you about them the way I would about old friends, because over the months of work they have really become almost my family.

The first and most important one is PriceToBinaryConverter. I call it the "digital alchemist." Its task seems simple — to turn price movements into sequences of zeros and ones. But behind this simplicity lies the real magic. Imagine that you are looking at the chart not through trader’s eyes, but through the eyes of the computer. What can you see? That's right — only "up" and "down", "one" and "zero". This is exactly what our first component is doing.

```
class PriceToBinaryConverter:
    def __init__(self, sequence_length: int = 32):
        self.sequence_length = sequence_length

    def convert_prices_to_binary(self, prices: pd.Series) -> List[str]:
        binary_sequence = []
        for i in range(1, len(prices)):
            binary_digit = '1' if prices.iloc[i] > prices.iloc[i-1] else '0'
            binary_sequence.append(binary_digit)
        return binary_sequence

    def get_binary_chunks(self, binary_sequence: List[str]) -> List[str]:
        chunks = []
        for i in range(0, len(binary_sequence), self.sequence_length):
            chunk = ''.join(binary_sequence[i:i + self.sequence_length])
            if len(chunk) < self.sequence_length:
                chunk = chunk.ljust(self.sequence_length, '0')
            chunks.append(chunk)
        return chunks
```

The second component is BIP39Converter — a real polyglot translator. He takes these boring strings of zeros and ones and turns them into meaningful English words. Do you remember BIP39 protocol from the world of cryptocurrencies? The one that is used to create mnemonic phrases for wallets? We took this idea and applied it to market analysis. Now every price movement is not just a set of bits, but a part of a meaningful phrase in English.

```
def binary_to_bip39(self, binary_sequence: str) -> List[str]:
    words = []
    for i in range(0, len(binary_sequence), 11):
        binary_chunk = binary_sequence[i:i+11]
        if len(binary_chunk) == 11:
            word = self.binary_to_word.get(binary_chunk, 'unknown')
            words.append(word)
    return words
```

And finally, PriceTransformer which is our "artificial intelligence". If the first two components can be compared to translators, then this one is more like a writer. He studies all these translated phrases and tries to figure out what will happen next. As a writer who has read thousands of books and can now predict how a story will end just by reading its beginning.

It is funny, but precisely such a three-stage system turned out to be incredibly effective. Each component does its job perfectly, like musicians in an orchestra — individually they are good, but together they create a real symphony.

In the following sections, we will analyze each of these components in detail. For now, just imagine this chain of transformations: graph → bits → words → forecast. It is beautiful, isn't it? It is like we have created a machine that can read the book of the market and retell it in human language.

They say that the beauty of mathematics lies in its simplicity. That is probably why our system turned out to be so elegant — we just let mathematics do what it does best: making order in chaos.

### Neural network architecture: Teaching a machine to read the language of the market

When I started working on the neural network architecture for our project, I had a strange feeling of deja vu. Remember it in that scene from “The Matrix” where Neo sees the code for the first time? I was looking at the exchange data threads and thinking: "What if we approach this as a natural language processing problem?"

And then it dawned on me — after all, the price movements are very similar to the text! They have their own grammar (patterns), their own syntax (trends and corrections), and even their own punctuation (key levels). Why not use an architecture that works great with texts?

This is how PriceTransformer was born — our "translator" from the market language. Here is its heart:

```
class PriceTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Sequential(
            nn.Embedding(1024, d_model),
            nn.Dropout(0.1)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
```

Does it look complicated? In fact, everything ingenious is simple. Imagine a translator who does not just look at each word individually, but tries to understand the context. This is exactly what the self-attention mechanism does in our transformer.

The most interesting thing started when we launched this model on real USD/JPY data. I remember the moment when, after a week of learning, the model began to produce the first meaningful predictions. It was like the moment when a child utters the first words — seemingly simple phrases, but there is a very complex learning process behind them.

An accuracy of 73% may not seem very impressive until you remember that we predict not just the direction of movement, but entire sequences of words! It is as if you are trying to guess not just the next word in a sentence, but the whole next paragraph.

But what really surprised me is that the model began to find its "favorite" words for different market situations. For example, before strong upward movements, it often generated words starting with certain letter combinations. As if the market really has its own vocabulary!

We have developed a special conveyor for data processing:

```
def prepare_data(self, df: pd.DataFrame) -> tuple:
    binary_sequence = self.price_converter.convert_prices_to_binary(df['close'])
    binary_chunks = self.price_converter.get_binary_chunks(binary_sequence)
    sequences = []
    for chunk in binary_chunks:
        words = self.bip39_converter.binary_to_bip39(chunk)
        indices = [self.bip39_converter.wordlist.index(word) for word in words]
        sequences.append(indices)
    return sequences
```

This code may seem simple, but it takes months of experimentation. We tried dozens of data preprocessing options before we found this optimal one. It turned out that even such a small thing as the size of the chunk can greatly affect the quality of predictions.

To improve model performance we had to apply several smart techniques. Butch normalization has helped to stabilize learning - as a good mentor who does not let the student lose his way. Gradient clipping prevented "explosions of gradients",  — imagine this as a safety rope for a climber. And the dynamic learning speed control acted like cruise control in the car - fast on straight sections, slow on turns.

![](https://c.mql5.com/2/117/6vg65sqz_10-02-2025_100402.jpg)

But the most interesting thing turned out to be watching how the model learns to detect long-term dependencies. Sometimes it would find connections between events separated by dozens of candles on the chart. It is as if it has learned to see the "forest behind trees", catching not only short-term fluctuations, but also global trends.

At some point, I caught myself thinking that our model reminds me of an experienced trader. It also patiently studies the market, looks for patterns, and learns from its mistakes. It only does it with the speed of a computer and without emotions, which often prevent people from making the right decisions.

Of course, our PriceTransformer is not a magic wand or a philosopher's stone. This is a tool that is quite difficult to set up. But when it is set up correctly, the results are amazing. Its ability to generate long—term forecasts in the form of readable sequences of words is particularly impressive - it is as if the market has finally spoken to us in an understandable language.

### In search of the Grail: results of experiments with the language of the market

Do you know what the most exciting thing about scientific experiments is? The moment when, after months of work, you finally see the first results. I remember when we started testing on the USD/JPY pair. Three years of data, hourly charts, hundreds of thousands of candles... To be honest, I did not expect much — rather, I was hoping for at least some kind of signal in this noise of market data.

And then the most interesting thing began. The first surprise is that the accuracy of predicting the next word has reached 73%. For those who do not work with language models, I'll explain: this is a very good result. Imagine that you are reading a book and trying to guess every next word — how many of them will you manage to guess correctly?

But it is not even about numbers. The most surprising thing was how the model began to "speak" in its own language. Do you know how children sometimes invent their own words to describe things? So our model did something similar. It started having its "favorite" words for different market situations.

I remember one particularly vivid case. I was analyzing the strong upward movement on USD/JPY, and the model began to generate sequences of words which start with certain bigrams. At first I thought it was a coincidence. But when the same pattern repeated itself the next time in a similar situation, it became clear  that we had found something interesting.

I analyzed the distribution of repeating bigrams, and this is what we see:

![](https://c.mql5.com/2/117/frequency_analysis.png)

And here is what linguistic sequence analysis shows us:

![](https://c.mql5.com/2/117/1433861302346.png)

### Linguistic analysis: when words report more than numbers

When I started analyzing the vocabulary of our model, discoveries began to pour in one after another. Do you remember the phrase about bigrams? That was just the beginning. The real treasures came to light when we looked at the frequency analysis of words in different market situations.

For example, before strong bullish movements, words with positive connotations appeared most often: "victory", "joy", "success". Interestingly, these words were found 32% more often than in normal periods. While before bearish movements, the vocabulary became more "technical": "system", "analyze", "process". It is as if the market starts to "think" more rationally before a fall.

The correlation between volatility and vocabulary diversity was particularly strong. During quiet periods, the model used a relatively small set of words, repeating them more often. But as the volatility increased, the vocabulary would expand 2-3 times! Just like a person who, in a stressful situation, starts talking more and using more complex constructions.

An interesting phenomenon of “vocabulary clusters" was also discovered. Some words almost always appeared in groups. For example, if the word "bridge" appeared in the sequence, then with a probability of 80% it was followed by words related to movement: "swift", "climb", "advance". Those clusters turned out to be so stable that we started using them as additional indicators.

### Conclusion

Summing up our research, I would like to mention several key points. First, we proved that the market does have its own "language", and this language can be translated into human words not just metaphorically, but literally, using advanced technology.

Second, the accuracy of predictions at 73% is not just statistics. This confirms that there is a structure, patterns, and grammar in the apparent chaos of market movements. And now we have a tool to decrypt them.

But the most important thing is the prospects. Imagine what will happen when we apply this approach to other markets, to other timeframes. We may find that different markets "speak" different dialects of the same language. Or that the market uses different "intonations" at different time periods of the day.

Of course, our research is only the first step. There is still a lot of work ahead: optimizing the architecture, experimenting with different parameters, and searching new patterns. But one thing is already clear — we have discovered a new way to listen to the market. And it definitely has something to say.

After all, maybe the secret to successful trading is not to find the perfect strategy, but to learn to truly understand the language of the market. And now we have not only metaphors and charts for this, but also a real translator.

| Script name | What the script does |
| --- | --- |
| GPT Model | It creates and trains a model based on price language sequences, performs a trend forecast for 100 bars ahead |
| GPT Model Plot | It builds up a histogram of distribution of the words repeated on bullish and bearish movements |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/17110](https://www.mql5.com/ru/articles/17110)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17110.zip "Download all attachments in the single ZIP archive")

[GPT\_Model.py](https://www.mql5.com/en/articles/download/17110/gpt_model.py "Download GPT_Model.py")(9.91 KB)

[GPT\_Model\_Plot.py](https://www.mql5.com/en/articles/download/17110/gpt_model_plot.py "Download GPT_Model_Plot.py")(4.97 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Forex arbitrage trading: Analyzing synthetic currencies movements and their mean reversion](https://www.mql5.com/en/articles/17512)
- [Forex arbitrage trading: A simple synthetic market maker bot to get started](https://www.mql5.com/en/articles/17424)
- [Forex Arbitrage Trading: Relationship Assessment Panel](https://www.mql5.com/en/articles/17422)
- [Build a Remote Forex Risk Management System in Python](https://www.mql5.com/en/articles/17410)
- [Currency pair strength indicator in pure MQL5](https://www.mql5.com/en/articles/17303)
- [Capital management in trading and the trader's home accounting program with a database](https://www.mql5.com/en/articles/17282)
- [Analyzing all price movement options on the IBM quantum computer](https://www.mql5.com/en/articles/17171)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/494067)**
(1)


![linfo2](https://c.mql5.com/avatar/2023/4/6438c14d-e2f0.png)

**[linfo2](https://www.mql5.com/en/users/neilhazelwood)**
\|
27 Aug 2025 at 05:21

I really appreciate your ideas thank you seems to be a massive pattern indicator much appreciated thanks for sharing


![Automating Trading Strategies in MQL5 (Part 28): Creating a Price Action Bat Harmonic Pattern with Visual Feedback](https://c.mql5.com/2/165/19105-automating-trading-strategies-logo.png)[Automating Trading Strategies in MQL5 (Part 28): Creating a Price Action Bat Harmonic Pattern with Visual Feedback](https://www.mql5.com/en/articles/19105)

In this article, we develop a Bat Pattern system in MQL5 that identifies bullish and bearish Bat harmonic patterns using pivot points and Fibonacci ratios, triggering trades with precise entry, stop loss, and take-profit levels, enhanced with visual feedback through chart objects

![Neural Networks in Trading: A Multi-Agent Self-Adaptive Model (MASA)](https://c.mql5.com/2/104/Multi-agent_adaptive_model_MASA___LOGO.png)[Neural Networks in Trading: A Multi-Agent Self-Adaptive Model (MASA)](https://www.mql5.com/en/articles/16537)

I invite you to get acquainted with the Multi-Agent Self-Adaptive (MASA) framework, which combines reinforcement learning and adaptive strategies, providing a harmonious balance between profitability and risk management in turbulent market conditions.

![Getting Started with MQL5 Algo Forge](https://c.mql5.com/2/152/18518-kak-nachat-rabotu-s-mql5-algo-logo.png)[Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)

We are introducing MQL5 Algo Forge — a dedicated portal for algorithmic trading developers. It combines the power of Git with an intuitive interface for managing and organizing projects within the MQL5 ecosystem. Here, you can follow interesting authors, form teams, and collaborate on algorithmic trading projects.

![Artificial Tribe Algorithm (ATA)](https://c.mql5.com/2/106/Artificial_Tribe_Algorithm_LOGO.png)[Artificial Tribe Algorithm (ATA)](https://www.mql5.com/en/articles/16588)

The article provides a detailed discussion of the key components and innovations of the ATA optimization algorithm, which is an evolutionary method with a unique dual behavior system that adapts depending on the situation. ATA combines individual and social learning while using crossover for explorations and migration to find solutions when stuck in local optima.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/17110&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083284388633909366)

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