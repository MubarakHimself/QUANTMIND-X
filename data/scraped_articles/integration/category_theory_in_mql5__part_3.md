---
title: Category Theory in MQL5 (Part 3)
url: https://www.mql5.com/en/articles/12085
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:11:55.445070
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ntywsayvzrguztjglmminpnmofbqvzdl&ssn=1769191913023819782&ssn_dr=0&ssn_sr=0&fv_date=1769191913&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12085&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Category%20Theory%20in%20MQL5%20(Part%203)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919191382490086&fz_uniq=5071648123933109225&sv=2552)

MetaTrader 5 / Integration


### Introduction

Following the previous [article](https://www.mql5.com/en/articles/11958) where we covered the definition of a category by focusing on its axioms as well as introducing ontology logs. We’ll continue this series on category theory by examining [Limits & Colimits](https://en.wikipedia.org/wiki/Limit_(category_theory)#:~:text=In%20category%20theory%2C%20a%20branch,coproducts%2C%20pushouts%20and%20direct%20limits. "https://en.wikipedia.org/wiki/Limit_(category_theory)#:~:text=In%20category%20theory%2C%20a%20branch,coproducts%2C%20pushouts%20and%20direct%20limits."); focus in their respective types of [Products](https://en.wikipedia.org/wiki/Product_(category_theory)#:~:text=In%20category%20theory%2C%20the%20product,the%20product%20of%20topological%20spaces. "https://en.wikipedia.org/wiki/Product_(category_theory)#:~:text=In%20category%20theory%2C%20the%20product,the%20product%20of%20topological%20spaces.") & [Coproducts](https://en.wikipedia.org/wiki/Coproduct "https://en.wikipedia.org/wiki/Coproduct"); and conclude with their respective takes on [universal-property](https://en.wikipedia.org/wiki/Universal_property "https://en.wikipedia.org/wiki/Universal_property"). However, before we delve into how these concepts can be developed in MQL5 it may be encouraging to share some ideas, based on what has been covered up to this article, on how category theory can be applied and used in a trading system. The system shared here is very rudimentary and is only meant to highlight the potential of the subject to a trader. This is covered first in the prologue.

### Prologue

In this article we’ll look at products which in category theory are a way of enumerating domains’ element pairings without losing prior constituent information. Using this within an MQL5 wizard signal file, we’ll create an expert advisor. So, our product will be between 2 domains namely indicator values of De-Marker and indicator values of Williams Percent Range. Each of these domains will then have morphisms with the domains of ‘Long Condition’ and ‘Short Condition’. The net result from summing the output of the two domains, which are synonymous with the buy and sell functions of a typical signal file, will determine whether the expert goes long or short. Here is the listing of our signal file.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CSignalCT : public CExpertSignal
  {
protected:
   CiDeMarker        m_dmk;            // object-oscillator (first corner)
   CiWPR             m_wpr;            // object-indicator (second corner)
   //--- adjusted parameters
   int               m_periods;        // the "period of calculation" parameter of the oscillator & indicator
   ENUM_APPLIED_PRICE m_applied;       // the "prices series" parameter of the oscillator & indicator
   double            m_longdmk;        // long dmk weight
   double            m_shortdmk;       // short dmk weight
   double            m_longwpr;        // long wpr weight
   double            m_shortwpr;       // short wpr weight

public:
   //--- methods of setting adjustable parameters
   void              Periods(int value)                { m_periods=value;  }
   void              Applied(ENUM_APPLIED_PRICE value) { m_applied=value;  }

   void              LongDMK(double value) { m_longdmk=value;  }
   void              ShortDMK(double value) { m_shortdmk=value;  }
   void              LongWPR(double value) { m_longwpr=value;  }
   void              ShortWPR(double value) { m_shortwpr=value;  }
   //--- method of verification of settings
   virtual bool      ValidationSettings(void);
   //--- method of creating the indicator and timeseries
   virtual bool      InitIndicators(CIndicators *indicators);
   //--- methods of checking if the market models are formed
   virtual int       LongCondition(void);
   virtual int       ShortCondition(void);
                     CSignalCT(void);
                    ~CSignalCT(void);

protected:

   virtual void      LongMorphism(void);
   virtual void      ShortMorphism(void);

   virtual double    Product(ENUM_POSITION_TYPE Position);

   NCT::
   CDomain<double>   long_product,short_product;
   //--- method of initialization of the oscillator
   bool              InitDMK(CIndicators *indicators);
   bool              InitWPR(CIndicators *indicators);
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CSignalCT::CSignalCT(void)
  {
//--- initialization of protected data
   m_used_series=USE_SERIES_HIGH+USE_SERIES_LOW+USE_SERIES_CLOSE+USE_SERIES_TIME;

  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CSignalCT::~CSignalCT(void)
  {
  }
//+------------------------------------------------------------------+
//| "Voting" that price will grow.                                   |
//+------------------------------------------------------------------+
int CSignalCT::LongCondition(void)
  {
      int result=0;

      //Using Domains Indicator biases (long or short)
      //e.g. an DMK reading of 75 => long-25, short-75
      //or price at upper WPR => long-0, short-100
      LongMorphism();

      result=int(round(Product(POSITION_TYPE_BUY)));

      return(result);
  }
//+------------------------------------------------------------------+
//| "Voting" that price will fall.                                   |
//+------------------------------------------------------------------+
int CSignalCT::ShortCondition(void)
  {
      int result=0;

      ShortMorphism();

      result=int(round(Product(POSITION_TYPE_SELL)));

      return(result);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CSignalCT::LongMorphism(void)
   {
      int _index=StartIndex();

      m_wpr.Refresh(-1);
      m_dmk.Refresh(-1);
      m_close.Refresh(-1);

      double _wpr=-1.0*(m_dmk.GetData(0,_index)/100.0);
      double _dmk=(1.0-m_dmk.GetData(0,_index));

      NCT::CElement<double> _e;
      _e.Cardinality(2);
      _e.Set(0,_dmk);_e.Set(1,_wpr);

      long_product.Cardinality(1);
      long_product.Set(0,_e,true);
   }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CSignalCT::ShortMorphism(void)
   {
      int _index=StartIndex();

      m_wpr.Refresh(-1);
      m_dmk.Refresh(-1);
      m_close.Refresh(-1);

      double _wpr=-1.0+((m_dmk.GetData(0,_index))/100.0);
      double _dmk=(m_dmk.GetData(0,_index));

      NCT::CElement<double> _e;
      _e.Cardinality(2);
      _e.Set(0,_dmk);_e.Set(1,_wpr);

      short_product.Cardinality(1);
      short_product.Set(0,_e,true);
   }
//+------------------------------------------------------------------+
//| Morphisms at Product                                             |
//+------------------------------------------------------------------+
double CSignalCT::Product(ENUM_POSITION_TYPE Position)
   {
      double _product=0.0;

      NCT::CElement<double> _e;

      if(Position==POSITION_TYPE_BUY)
      {
         if(long_product.Cardinality()>=1 && long_product.Get(0,_e))
         {
            _product=100.0*((m_longdmk*_e.Get(0))+(m_longwpr*_e.Get(1)))/(m_longdmk+m_longwpr);
         }

         return(_product);
      }

      if(short_product.Cardinality()>=1 && short_product.Get(0,_e))
      {
         _product=100.0*((m_shortdmk*_e.Get(0))+(m_shortwpr*_e.Get(1)))/(m_shortdmk+m_shortwpr);
      }

      return(_product);
   }
```

A tester report based on real-ticks over most of 2022 for the pair EURJPY gives us the following curve.

[![ct_3_curve](https://c.mql5.com/2/51/ct_3_curve.png)](https://c.mql5.com/2/51/ct_3_curve.png "https://c.mql5.com/2/51/ct_3_curve.png")

With these report details.

[![ct_3_report](https://c.mql5.com/2/51/ct_3_report.png)](https://c.mql5.com/2/51/ct_3_report.png "https://c.mql5.com/2/51/ct_3_report.png")

Clearly it is not a perfect system but it does present some ideas which can further be developed into something more wholistic. Please see attachments for full source code.

As per wikipedia, Limits and colimits, like the strongly related notions of universal properties, exist at a high level of abstraction and in order to understand them, it is helpful to first study the specific examples these concepts are meant to generalise like products and coproducts respectively.

### Products

[Product](https://en.wikipedia.org/wiki/Product_(category_theory) "https://en.wikipedia.org/wiki/Product_(category_theory)") of two domains A and B is represented as A x B, and is defined as the set of ordered pairs (a, b), where a ∈ A and b ∈ B. Symbolically,

> A x B = {(a, b) \| a ∈ A, b ∈ B}

The product is a generalisation of the Cartesian product of domains, and it captures the idea of a "pair" or "tuple" of domains in a category. Intuitively, the product C represents the "joint" or "composite" behavior of A and B.

In addition to the universal property examined below, the product satisfies the following properties:

- The product is associative: (A × B) × C is isomorphic to A × (B × C).
- The product is commutative: A × B is isomorphic to B × A.
- The product of a domain with the initial domain is isomorphic to the domain itself: A × 1 is isomorphic to A.
- The product of a domain with the terminal domain is isomorphic to the domain itself: A × 0 is isomorphic to 0.

The concept of products can be applied to various mathematical structures, including sets, groups, rings, and vector spaces, as well as to more abstract structures such as topological spaces and categories themselves. In each case, the product captures the idea of a "joint" or "composite" behaviour of two domains or structures, and it provides a way of reasoning about the behaviour of the system as a whole. One example of where the use of category theory in finance provides a better alternative than what is commonly applied, is in risk management.

Category theory allows for a more abstract and general approach to modelling financial risk, which can be applied to a wide range of financial instruments and markets. This approach can provide a more unified and flexible framework for understanding risk, compared to the traditional approach of using specific models for each instrument or market. Additionally, the use of category theory in risk management can lead to more robust and scalable risk management strategies, which are better suited for the complex and interconnected nature of modern financial markets.

For instance, in traditional risk management, risk is often modelled using specific mathematical models for each financial instrument or market. For example, the Black-Scholes model is commonly used to model the risk of options, while the VaR (Value at Risk) model is often used to model the risk of a portfolio of assets. These models can be effective, but they are limited in their scope and can be difficult to generalise to new financial instruments or markets.

In contrast, category theory provides a more abstract and general approach to modelling risk. For instance, financial instruments in markets could be represented as domains in a category, and risk represented as morphism from these domains to a position domain that has only 2 elements, long-position and short-position. This would allow for a more unified and flexible framework for understanding risk, which can be applied to a wide range of financial instruments and markets. A similar approach was used with our signal file in the prologue.

For example, suppose we have two financial instruments, A and B, and we want to model the risk of moving from A to B. We would represent A and B as domains, and the risk of moving from A to B as their respective morphisms. These morphisms would capture the risk associated with the transitioning from A to B, while considering factors like volatility, liquidity, and market conditions. Alternatively as shown in the prologue we could have one instrument and want to know what position we need to have at a certain time. We would take financial indicators on this instrument and find their weighted product. The value of this will determine our decision. Here is a half listing that only shows the combination of these indicator values. Firstly we need a class to handle and store information of the product domains.

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CComposition
   {
      protected:

      int                           projectors;
      string                        projector[];

      bool                          Projectors(int Value)
                                    {
                                       if(Value>=0 && Value<INT_MAX)
                                       {
                                          projectors=Value;
                                          ArrayResize(projector,projectors);
                                          return(true);
                                       }

                                       return(false);
                                    };

      int                           Projectors(){ return(projectors); };

      public:

      string                        Get(int ProjectorIndex) { string _projector=""; if(ProjectorIndex>=0 && ProjectorIndex<Projectors()) { _projector=projector[ProjectorIndex]; } return(_projector); }
      bool                          Set(int ValueIndex,string Value) { if(ValueIndex>=0 && ValueIndex<Projectors()) { projector[ValueIndex]=Value; return(true); } return(false); }

      CDomain<string>               property;
      CDomain<string>               cone;

                                    CComposition(void){ projectors=0;ArrayFree(projector); };
                                    ~CComposition(void){};
   };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CProduct                      :public CComposition
   {
      protected:

      CDomain<string>               surjector[];

      public:

      bool                          Surjectors(int Value)
                                    {
                                       if(Value>=0 && Value<INT_MAX)
                                       {
                                          CComposition::Projectors(Value);
                                          ArrayResize(surjector,Value);
                                          return(true);
                                       }

                                       return(false);
                                    };

      int                           Surjectors(){ return(CComposition::projectors); };

      bool                          Get(int SurjectorIndex,CDomain<string> &Surjector) { if(SurjectorIndex>=0 && SurjectorIndex<CComposition::Projectors()) { Surjector=surjector[SurjectorIndex]; return(true); } return(false); }
      bool                          Set(int ValueIndex,CDomain<string> &Value) { if(ValueIndex>=0 && ValueIndex<CComposition::Projectors()) { surjector[ValueIndex]=Value; return(true); } return(false); }

                                    CProduct(void){ ArrayFree(surjector); };
                                    ~CProduct(void){};
   };
```

Once we have those defined we can perform a product(s) as follows.

```
      //////////
      //PRODUCTS
      //////////

      CDomain<double> _d_p_a,_d_p_b,_d_p_c;
      _d_p_a.Cardinality(__product_size);_d_p_b.Cardinality(__product_size);_d_p_c.Cardinality(__product_size);

      int _rsi_handle=iRSI(_Symbol,_Period,__product_size,__product_price);
      int _cci_handle=iCCI(_Symbol,_Period,__product_size,__product_price);
      int _dmk_handle=iDeMarker(_Symbol,_Period,__product_size);
      int _wpr_handle=iWPR(_Symbol,_Period,__product_size);
      int _stc_handle=iStochastic(_Symbol,_Period,8,4,4,MODE_SMA,STO_LOWHIGH);
      int _trx_handle=iTriX(_Symbol,_Period,__product_size,__product_price);

      if
      (
      FillDomain(_d_p_a,0,__product_size,_rsi_handle)
      &&
      FillDomain(_d_p_a,1,__product_size,_cci_handle)
      &&
      FillDomain(_d_p_b,0,__product_size,_dmk_handle)
      &&
      FillDomain(_d_p_b,1,__product_size,_wpr_handle)
      &&
      FillDomain(_d_p_c,0,__product_size,_stc_handle)
      &&
      FillDomain(_d_p_c,1,__product_size,_trx_handle)
      )
      {
         printf(__FUNCSIG__+" domain A: "+PrintDomain(_d_p_a,2));
         printf(__FUNCSIG__+" domain B: "+PrintDomain(_d_p_b,2));
         printf(__FUNCSIG__+" domain C: "+PrintDomain(_d_p_c,5));

         CProduct _product;

         GetProduct(_d_p_a,_d_p_b,_product,2);
         printf(__FUNCSIG__+" A & B product: "+PrintDomain(_product.cone,2));

         GetProduct(_product.cone,_d_p_c,_product,5);
         printf(__FUNCSIG__+" A & B & C product: "+PrintDomain(_product.cone,5));
      }
```

These indicator values though not normalised like in the signal file in the prologue, can none the less be thought of or used as risk indicators for a trade decision. Running the script does yield these logs.

```
2023.02.17 17:31:33.199 ct_3_1 (USDCHF.ln,W1)   void OnStart() domain A: {(-66.67),(66.67)}
2023.02.17 17:31:33.199 ct_3_1 (USDCHF.ln,W1)
2023.02.17 17:31:33.199 ct_3_1 (USDCHF.ln,W1)   void OnStart() domain B: {(-61.99),(-68.45)}
2023.02.17 17:31:33.199 ct_3_1 (USDCHF.ln,W1)
2023.02.17 17:31:33.199 ct_3_1 (USDCHF.ln,W1)   void OnStart() domain C: {(-0.00996),(-0.00628)}
2023.02.17 17:31:33.199 ct_3_1 (USDCHF.ln,W1)
2023.02.17 17:31:33.199 ct_3_1 (USDCHF.ln,W1)   void OnStart() A & B product: {((66.67),(-68.45)),((66.67),(-61.99)),((-66.67),(-68.45)),((-66.67),(-61.99))}
2023.02.17 17:31:33.199 ct_3_1 (USDCHF.ln,W1)
2023.02.17 17:31:33.200 ct_3_1 (USDCHF.ln,W1)   void OnStart() A & B & C product: {(((-66.67),(-61.99)),(-0.00628)),(((-66.67),(-61.99)),(-0.00996)),(((-66.67),(-68.45)),(-0.00628)),(((-66.67),(-68.45)),(-0.00996)),(((66.67),(-61.99)),(-0.00628)),(((66.67),(-61.99)),(-0.00996)),(((66.67),(-68.45)),(-0.00628)),(((66.67),(-68.45)),(-0.00996))}
```

The ability to consider multiple factors would be more sensitive to the macro than value at risk, for instance, which only considers a risk confidence interval over past performance. The notion of embedding multiple factors into a morphism could be taken a step further to identify and quantify the risks associated with complex financial instruments such as derivatives, which can be difficult to model using traditional approaches. Again, this is thanks to the systematic approach of separating data in domains and linking by integrity preserving morphisms. In addition, we could develop risk management strategies that are better suited for the complex and interconnected nature of modern financial markets, such as strategies that consider the correlations between different financial instruments and markets.

The concept of products in category theory can also be useful in finance for risk management across multiple assets as an example. The product of two domains in a category provides a way to combine them into a single domain, which can be used to model the joint risk associated with both domains (assets). For example, let's say we have two financial instruments, A and B, and we want to model the joint risk associated with holding both of them in a portfolio. We can represent A and B as domains in a category, and their joint risk as the product of the two domains. This product domains would capture the combined risk associated with both A and B, considering their individual risks as well as any interactions between them. This approach preserves the input risk of each financial instrument such that when training a model, the separate inputs can both be considered for better accuracy and insight rather than averaging the two into one. In essence we better understand the joint risk associated with different financial instruments, and are thus in a position to develop more effective risk management strategies. To illustrate, we could use products to model the risk associated with complex financial structures such as collateralized debt obligations (CDOs), which are made up of multiple underlying assets.

Products in category theory tend to lean towards _abstraction_ and general frameworks which when modelling risk, for instance, can lead to a wide range of application in financial instruments and markets. This allows for a more unified and flexible approach to risk management, which can help us to better understand the interactions and dependencies between different financial instruments and markets as opposed to the specificity of VaR/ Black-Scholes. Suppose we have an S&P 500 portfolio consisting of various stocks, and we want to model the risk of this portfolio. We could do this by VaR, which estimates the maximum potential loss of the portfolio with a given probability level.

However, VaR has some limitations, such as its assumption of normal distribution, which may not hold in practice, its inability to capture complex risk relationships among the stocks in the portfolio, plus its inability to assess the magnitude of loss when a VaR breach occurs. Category theory on the other hand provides a more abstract and flexible way to model risk by focusing on the structure and relationships among the various elements of the portfolio, rather than just their individual properties. In particular, category theory allows us to represent the S&P 500 portfolio as a category, where the stocks are the domains of the category, and the risk relationships among them are captured by the morphisms or arrows of the category.

For example, we could define a morphism between two stocks as representing the risk relationship between them, such as the degree to which their prices move in the same direction or the opposite direction (e.g. correlation). By defining such morphisms, we can build a more detailed and nuanced picture of the risk relationships among the stocks in the portfolio, which may be more accurate than just using VaR. Moreover, by abstracting the risk relationships among the stocks as morphisms, we can apply powerful tools and concepts from category theory, such as composition, duality, and universal properties, to analyse and manage risk in the portfolio. For example, we could use composition of morphisms to combine the risk relationships among multiple stocks and derive the overall risk of the portfolio, or we could use [duality](https://en.wikipedia.org/wiki/Dual_(category_theory) "https://en.wikipedia.org/wiki/Dual_(category_theory)") (to be covered) to study the inverse risk relationships among the stocks.

Here's a numerical example to further illustrate how using category theory, including products of category theory, can offer advantages over afore mentioned traditional risk metrics. Suppose we have a portfolio consisting of two financial instruments, A and B, each with a current value of $100, and we want to estimate the joint risk associated with holding both of them. For simplicity, let's assume that the returns of A and B are normally distributed, with a mean of 10% and a standard deviation of 20%. Using traditional models like VaR we could estimate the individual risks of A and B, as well as their joint risk. For example, using VaR with a 95% confidence level, we would estimate that the one-day VaR for A and B individually is approximately $25.46, and the joint VaR is approximately $36.03.

VaR = portfolio value x volatility x z-score

VaR = $100 x 20% x 1.645

VaR = $36.03.

Alternatively, we could use category theory and products of category theory to model the joint risk of A and B. In this approach, we would represent A and B as domains in a category, and their joint risk as the product of the two domains. Using the standard formula for the product of two normally distributed random variables, we can calculate that the joint distribution of A and B has a mean of 10%, and a standard deviation of approximately 0.28.

VaR = $100 x √ ((10%)2 x (28%)2 \+ (10%)2 x (28%)2) x 1.645

VaR = $29.15.

Using this approach, we could estimate the one-day VaR for the joint risk of A and B using the standard formula for VaR. With a 95% confidence level, the one-day VaR for the joint risk of A and B would be approximately $29.15. In this example, using category theory and products model the joint risk of A and B produces a different estimate of the joint VaR than traditional model above. However, category theory could provide a more general and flexible approach to risk management, which can be applied to a wider range of financial instruments and markets, and can capture the joint risk associated with multiple financial instruments more effectively. This features, typical output of a domain product would a matrix that enumerates risk outcomes for each iteration. This continuum of data serves as a basis for evaluating the mean and standard deviation, subject to investors objectives and targets.

Products are a type of Limit so perhaps it would be helpful to provide some general notes on how limits as a whole can be resourceful to traders in system development. In the context of finance, the use of limits can help us to approximate complex financial structures more effectively, which can aid in the development of more accurate risk management strategies. For example, we can use the concept of limits to estimate the value of a portfolio that contains a large number of different financial instruments. By representing the portfolio as a limit of simpler domains, we can estimate its value more accurately and efficiently.

As an example, suppose we have a portfolio of 1000 financial instruments, each with a current value of $100. We want to estimate the value of the entire portfolio, as well as the joint risk associated with holding all of the instruments. Using traditional methods, such as VaR or Black-Scholes, could be difficult to compute due to the high-dimensional nature of the portfolio. However, using the concept of limits in category theory, we can represent the portfolio as a limit of simpler domains, such as the sum of the first n financial instruments, where n ranges from 1 to 1000. We can then estimate the value of the portfolio by taking the limit of the sum as n approaches 1000. By using this approach, we can estimate the value of the portfolio more accurately and efficiently.

Moreover, the use of limits in category theory can allow us to reason about the behaviour of financial instruments and markets as they approach certain limits. For example, we can use the concept of limits to analyse the behaviour of financial markets during times of extreme volatility or stress, and to develop risk management strategies that are better suited to these conditions. To wrap up, the use of limits in category theory can bring added benefits to the use of category theory and products in finance, by allowing us to _approximate_ complex financial structures more effectively, and to reason about the behaviour of financial instruments and markets as they approach certain limits.

### Product Universal Property

A universal property is a way of characterising a domain in a category based on the way it interacts with other domains in the category. This property can be used to define domains and operations more abstractly which can lead to more powerful and flexible models. In the context of finance, the use of universal properties can allow us to define financial instruments and markets more abstractly, which can make it easier to reason about them and develop more general models. For example, we can define an exchange-traded derivative as a domain in a category that satisfies certain universal properties like identifying who the regulators are, besides stating whether its an option or a futures contract. This identification can help in estimating compliance costs and counter party risk. Here is a listing portion of the attached script that demonstrates this.

```
      //////////////////////////////
      //PRODUCT UNIVERSAL-PROPERTY
      //////////////////////////////

      CDomain<string> _d_security,_d_exchanges,_d_optioncycle,_d_strikewidth,_d_property;
      //
      CElement<string> _e;_e.Cardinality(1);
      //
      _d_security.Cardinality(2);
      _e.Set(0,"EURUSD");_d_security.Set(0,_e,true);
      _e.Set(0,"USDJPY");_d_security.Set(1,_e,true);
      //
      _d_exchanges.Cardinality(7);
      _e.Set(0,"Chicago Board Options Exchange (CBOE)");_d_exchanges.Set(0,_e,true);
      _e.Set(0,"Shanghai Stock Exchange (SSE)");_d_exchanges.Set(1,_e,true);
      _e.Set(0,"Shenzhen Stock Exchange (SZSE)");_d_exchanges.Set(2,_e,true);
      _e.Set(0,"Tokyo Stock Exchange (TSE)");_d_exchanges.Set(3,_e,true);
      _e.Set(0,"Osaka Exchange (OSE)");_d_exchanges.Set(4,_e,true);
      _e.Set(0,"Eurex Exchange");_d_exchanges.Set(5,_e,true);
      _e.Set(0,"London Stock Exchange (LSE)");_d_exchanges.Set(6,_e,true);
      //
      _d_optioncycle.Cardinality(3);
      _e.Set(0,"JAJO - January, April, July, and October");_d_optioncycle.Set(0,_e,true);
      _e.Set(0,"FMAN - February, May, August, and November");_d_optioncycle.Set(1,_e,true);
      _e.Set(0,"MJSD - March, June, September, and December");_d_optioncycle.Set(2,_e,true);
      //
      _d_strikewidth.Cardinality(2);
      _e.Set(0,"1000 points");_d_strikewidth.Set(0,_e,true);
      _e.Set(0,"1250 points");_d_strikewidth.Set(1,_e,true);
      //
      printf(__FUNCSIG__+" securities domain: "+PrintDomain(_d_security,0));
      printf(__FUNCSIG__+" exchanges domain: "+PrintDomain(_d_exchanges,0));
      printf(__FUNCSIG__+" option cycle domain: "+PrintDomain(_d_optioncycle,0));
      printf(__FUNCSIG__+" strike width domain: "+PrintDomain(_d_strikewidth,0));

      CProduct _product_1;

      GetProduct(_d_security,_d_exchanges,_product_1,0);
      printf(__FUNCSIG__+" securities & exchanges product: "+PrintDomain(_product_1.cone,0));

      CProduct _product_2;

      GetProduct(_d_optioncycle,_d_strikewidth,_product_2,0);
      printf(__FUNCSIG__+" securities & exchanges & optioncycle product: "+PrintDomain(_product_2.cone,0));

      CProduct _product_all;

      GetProduct(_product_1.cone,_product_2.cone,_product_all,0);
      printf(__FUNCSIG__+" securities & exchanges & optioncycle & strikewidth product: "+PrintDomain(_product_all.cone,0));

      _d_property.Cardinality(5);
      _e.Set(0,"Commodity Futures Trading Commission (CFTC)");_d_property.Set(0,_e,true);
      _e.Set(0,"China Securities Regulatory Commission (CSRC)");_d_property.Set(1,_e,true);
      _e.Set(0,"Financial Services Agency (FSA)");_d_property.Set(2,_e,true);
      _e.Set(0,"Federal Financial Supervisory Authority (BaFin)");_d_property.Set(3,_e,true);
      _e.Set(0,"Financial Conduct Authority (FCA)");_d_property.Set(4,_e,true);

      //
      _product_all.property=_d_property;
      //
      _product_all.universality.domain=_product_all.property;
      _product_all.universality.codomain=_product_all.cone;
      //
      CMorphism<string,string> _mm;
      _mm.domain=_product_all.property;
      _mm.codomain=_product_all.cone;
      //

      for(int c=0;c<_product_all.property.Cardinality();c++)
      {
         CElement<string> _e_property;_e_property.Cardinality(1);
         if(_product_all.property.Get(c,_e_property) && _e_property.Get(0)!="")
         {
            for(int cc=0;cc<_product_all.cone.Cardinality();cc++)
            {
               CElement<string> _e_cone;_e_cone.Cardinality(1);
               if(_product_all.cone.Get(cc,_e_cone) && _e_cone.Get(0)!="")
               {
                  if(_e_property.Get(0)=="Commodity Futures Trading Commission (CFTC)")
                  {
                     if(StringFind(_e_cone.Get(0),"Chicago Board Options Exchange (CBOE)")>=0)
                     {
                        if(_product_all.universality.Morphisms(_product_all.universality.Morphisms()+1))
                        {
                           if(_mm.Morph(_product_all.property,_product_all.cone,_e_property,_e_cone))
                           {
                              if(!_product_all.universality.Set(_product_all.universality.Morphisms()-1,_mm))
                              {
                              }
                           }
                        }
                     }
                  }
                  else if(_e_property.Get(0)=="China Securities Regulatory Commission (CSRC)")
                  {
                     if(StringFind(_e_cone.Get(0),"Shanghai Stock Exchange (SSE)")>=0||StringFind(_e_cone.Get(0),"Shenzhen Stock Exchange (SZSE)")>=0)
                     {
                        if(_product_all.universality.Morphisms(_product_all.universality.Morphisms()+1))
                        {
                           if(_mm.Morph(_product_all.property,_product_all.cone,_e_property,_e_cone))
                           {
                              if(!_product_all.universality.Set(_product_all.universality.Morphisms()-1,_mm))
                              {
                              }
                           }
                        }
                     }
                  }
                  else if(_e_property.Get(0)=="Financial Services Agency (FSA)")
                  {
                     if(StringFind(_e_cone.Get(0),"Tokyo Stock Exchange (TSE)")>=0||StringFind(_e_cone.Get(0),"Osaka Exchange (OSE)")>=0)
                     {
                        if(_product_all.universality.Morphisms(_product_all.universality.Morphisms()+1))
                        {
                           if(_mm.Morph(_product_all.property,_product_all.cone,_e_property,_e_cone))
                           {
                              if(!_product_all.universality.Set(_product_all.universality.Morphisms()-1,_mm))
                              {
                              }
                           }
                        }
                     }
                  }
                  else if(_e_property.Get(0)=="Federal Financial Supervisory Authority (BaFin)")
                  {
                     if(StringFind(_e_cone.Get(0),"Eurex Exchange")>=0)
                     {
                        if(_product_all.universality.Morphisms(_product_all.universality.Morphisms()+1))
                        {
                           if(_mm.Morph(_product_all.property,_product_all.cone,_e_property,_e_cone))
                           {
                              if(!_product_all.universality.Set(_product_all.universality.Morphisms()-1,_mm))
                              {
                              }
                           }
                        }
                     }
                  }
                  else if(_e_property.Get(0)=="Financial Conduct Authority (FCA)")
                  {
                     if(StringFind(_e_cone.Get(0),"London Stock Exchange (LSE)")>=0)
                     {
                        if(_product_all.universality.Morphisms(_product_all.universality.Morphisms()+1))
                        {
                           if(_mm.Morph(_product_all.property,_product_all.cone,_e_property,_e_cone))
                           {
                              if(!_product_all.universality.Set(_product_all.universality.Morphisms()-1,_mm))
                              {
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
```

On running this script we should produce the logs below that print the homomorphism between the cone domain and the property domain. This homomorphism marks the universal property.

```
/*
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   void OnStart() universal property:
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   {(Commodity Futures Trading Commission (CFTC)),(China Securities Regulatory Commission (CSRC)),(Financial Services Agency (FSA)),(Federal Financial Supervisory Authority (BaFin)),(Financial Conduct Authority (FCA))}
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   |
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (Commodity Futures Trading Commission (CFTC))|----->(((EURUSD),(Chicago Board Options Exchange (CBOE))),((JAJO - January, April, July, and October),(1000 points)))
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (Commodity Futures Trading Commission (CFTC))|----->(((EURUSD),(Chicago Board Options Exchange (CBOE))),((JAJO - January, April, July, and October),(1250 points)))
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (Commodity Futures Trading Commission (CFTC))|----->(((EURUSD),(Chicago Board Options Exchange (CBOE))),((FMAN - February, May, August, and November),(1000 points)))
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (Commodity Futures Trading Commission (CFTC))|----->(((EURUSD),(Chicago Board Options Exchange (CBOE))),((FMAN - February, May, August, and November),(1250 points)))
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (Commodity Futures Trading Commission (CFTC))|----->(((EURUSD),(Chicago Board Options Exchange (CBOE))),((MJSD - March, June, September, and December),(1000 points)))
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (Commodity Futures Trading Commission (CFTC))|----->(((EURUSD),(Chicago Board Options Exchange (CBOE))),((MJSD - March, June, September, and December),(1250 points)))
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (Commodity Futures Trading Commission (CFTC))|----->(((USDJPY),(Chicago Board Options Exchange (CBOE))),((JAJO - January, April, July, and October),(1000 points)))
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (Commodity Futures Trading Commission (CFTC))|----->(((USDJPY),(Chicago Board Options Exchange (CBOE))),((JAJO - January, April, July, and October),(1250 points)))
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (Commodity Futures Trading Commission (CFTC))|----->(((USDJPY),(Chicago Board Options Exchange (CBOE))),((FMAN - February, May, August, and November),(1000 points)))
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (Commodity Futures Trading Commission (CFTC))|----->(((USDJPY),(Chicago Board Options Exchange (CBOE))),((FMAN - February, May, August, and November),(1250 points)))
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (Commodity Futures Trading Commission (CFTC))|----->(((USDJPY),(Chicago Board Options Exchange (CBOE))),((MJSD - March, June, September, and December),(1000 points)))
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (Commodity Futures Trading Commission (CFTC))|----->(((USDJPY),(Chicago Board Options Exchange (CBOE))),((MJSD - March, June, September, and December),(1250 points)))
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((EURUSD),(Shanghai Stock Exchange (SSE))),((JAJO - January, April, July, and October),(1000 points)))
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((EURUSD),(Shanghai Stock Exchange (SSE))),((JAJO - January, April, July, and October),(1250 points)))
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((EURUSD),(Shanghai Stock Exchange (SSE))),((FMAN - February, May, August, and November),(1000 points)))
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((EURUSD),(Shanghai Stock Exchange (SSE))),((FMAN - February, May, August, and November),(1250 points)))
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((EURUSD),(Shanghai Stock Exchange (SSE))),((MJSD - March, June, September, and December),(1000 points)))
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((EURUSD),(Shanghai Stock Exchange (SSE))),((MJSD - March, June, September, and December),(1250 points)))
2023.02.21 09:57:41.077 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((EURUSD),(Shenzhen Stock Exchange (SZSE))),((JAJO - January, April, July, and October),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((EURUSD),(Shenzhen Stock Exchange (SZSE))),((JAJO - January, April, July, and October),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((EURUSD),(Shenzhen Stock Exchange (SZSE))),((FMAN - February, May, August, and November),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((EURUSD),(Shenzhen Stock Exchange (SZSE))),((FMAN - February, May, August, and November),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((EURUSD),(Shenzhen Stock Exchange (SZSE))),((MJSD - March, June, September, and December),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((EURUSD),(Shenzhen Stock Exchange (SZSE))),((MJSD - March, June, September, and December),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((USDJPY),(Shanghai Stock Exchange (SSE))),((JAJO - January, April, July, and October),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((USDJPY),(Shanghai Stock Exchange (SSE))),((JAJO - January, April, July, and October),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((USDJPY),(Shanghai Stock Exchange (SSE))),((FMAN - February, May, August, and November),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((USDJPY),(Shanghai Stock Exchange (SSE))),((FMAN - February, May, August, and November),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((USDJPY),(Shanghai Stock Exchange (SSE))),((MJSD - March, June, September, and December),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((USDJPY),(Shanghai Stock Exchange (SSE))),((MJSD - March, June, September, and December),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((USDJPY),(Shenzhen Stock Exchange (SZSE))),((JAJO - January, April, July, and October),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((USDJPY),(Shenzhen Stock Exchange (SZSE))),((JAJO - January, April, July, and October),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((USDJPY),(Shenzhen Stock Exchange (SZSE))),((FMAN - February, May, August, and November),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((USDJPY),(Shenzhen Stock Exchange (SZSE))),((FMAN - February, May, August, and November),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((USDJPY),(Shenzhen Stock Exchange (SZSE))),((MJSD - March, June, September, and December),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (China Securities Regulatory Commission (CSRC))|----->(((USDJPY),(Shenzhen Stock Exchange (SZSE))),((MJSD - March, June, September, and December),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((EURUSD),(Tokyo Stock Exchange (TSE))),((JAJO - January, April, July, and October),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((EURUSD),(Tokyo Stock Exchange (TSE))),((JAJO - January, April, July, and October),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((EURUSD),(Tokyo Stock Exchange (TSE))),((FMAN - February, May, August, and November),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((EURUSD),(Tokyo Stock Exchange (TSE))),((FMAN - February, May, August, and November),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((EURUSD),(Tokyo Stock Exchange (TSE))),((MJSD - March, June, September, and December),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((EURUSD),(Tokyo Stock Exchange (TSE))),((MJSD - March, June, September, and December),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((EURUSD),(Osaka Exchange (OSE))),((JAJO - January, April, July, and October),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((EURUSD),(Osaka Exchange (OSE))),((JAJO - January, April, July, and October),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((EURUSD),(Osaka Exchange (OSE))),((FMAN - February, May, August, and November),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((EURUSD),(Osaka Exchange (OSE))),((FMAN - February, May, August, and November),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((EURUSD),(Osaka Exchange (OSE))),((MJSD - March, June, September, and December),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((EURUSD),(Osaka Exchange (OSE))),((MJSD - March, June, September, and December),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((USDJPY),(Tokyo Stock Exchange (TSE))),((JAJO - January, April, July, and October),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((USDJPY),(Tokyo Stock Exchange (TSE))),((JAJO - January, April, July, and October),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((USDJPY),(Tokyo Stock Exchange (TSE))),((FMAN - February, May, August, and November),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((USDJPY),(Tokyo Stock Exchange (TSE))),((FMAN - February, May, August, and November),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((USDJPY),(Tokyo Stock Exchange (TSE))),((MJSD - March, June, September, and December),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((USDJPY),(Tokyo Stock Exchange (TSE))),((MJSD - March, June, September, and December),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((USDJPY),(Osaka Exchange (OSE))),((JAJO - January, April, July, and October),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((USDJPY),(Osaka Exchange (OSE))),((JAJO - January, April, July, and October),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((USDJPY),(Osaka Exchange (OSE))),((FMAN - February, May, August, and November),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((USDJPY),(Osaka Exchange (OSE))),((FMAN - February, May, August, and November),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((USDJPY),(Osaka Exchange (OSE))),((MJSD - March, June, September, and December),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Financial Services Agency (FSA))|----->(((USDJPY),(Osaka Exchange (OSE))),((MJSD - March, June, September, and December),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Federal Financial Supervisory Authority (BaFin))|----->(((EURUSD),(Eurex Exchange)),((JAJO - January, April, July, and October),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Federal Financial Supervisory Authority (BaFin))|----->(((EURUSD),(Eurex Exchange)),((JAJO - January, April, July, and October),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Federal Financial Supervisory Authority (BaFin))|----->(((EURUSD),(Eurex Exchange)),((FMAN - February, May, August, and November),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Federal Financial Supervisory Authority (BaFin))|----->(((EURUSD),(Eurex Exchange)),((FMAN - February, May, August, and November),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Federal Financial Supervisory Authority (BaFin))|----->(((EURUSD),(Eurex Exchange)),((MJSD - March, June, September, and December),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Federal Financial Supervisory Authority (BaFin))|----->(((EURUSD),(Eurex Exchange)),((MJSD - March, June, September, and December),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Federal Financial Supervisory Authority (BaFin))|----->(((USDJPY),(Eurex Exchange)),((JAJO - January, April, July, and October),(1000 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Federal Financial Supervisory Authority (BaFin))|----->(((USDJPY),(Eurex Exchange)),((JAJO - January, April, July, and October),(1250 points)))
2023.02.21 09:57:41.078 ct_3_1 (USDCHF.ln,H1)   (Federal Financial Supervisory Authority (BaFin))|----->(((USDJPY),(Eurex Exchange)),((FMAN - February, May, August, and November),(1000 points)))
2023.02.21 09:57:41.079 ct_3_1 (USDCHF.ln,H1)   (Federal Financial Supervisory Authority (BaFin))|----->(((USDJPY),(Eurex Exchange)),((FMAN - February, May, August, and November),(1250 points)))
2023.02.21 09:57:41.079 ct_3_1 (USDCHF.ln,H1)   (Federal Financial Supervisory Authority (BaFin))|----->(((USDJPY),(Eurex Exchange)),((MJSD - March, June, September, and December),(1000 points)))
2023.02.21 09:57:41.079 ct_3_1 (USDCHF.ln,H1)   (Federal Financial Supervisory Authority (BaFin))|----->(((USDJPY),(Eurex Exchange)),((MJSD - March, June, September, and December),(1250 points)))
2023.02.21 09:57:41.079 ct_3_1 (USDCHF.ln,H1)   (Financial Conduct Authority (FCA))|----->(((EURUSD),(London Stock Exchange (LSE))),((JAJO - January, April, July, and October),(1000 points)))
2023.02.21 09:57:41.079 ct_3_1 (USDCHF.ln,H1)   (Financial Conduct Authority (FCA))|----->(((EURUSD),(London Stock Exchange (LSE))),((JAJO - January, April, July, and October),(1250 points)))
2023.02.21 09:57:41.079 ct_3_1 (USDCHF.ln,H1)   (Financial Conduct Authority (FCA))|----->(((EURUSD),(London Stock Exchange (LSE))),((FMAN - February, May, August, and November),(1000 points)))
2023.02.21 09:57:41.079 ct_3_1 (USDCHF.ln,H1)   (Financial Conduct Authority (FCA))|----->(((EURUSD),(London Stock Exchange (LSE))),((FMAN - February, May, August, and November),(1250 points)))
2023.02.21 09:57:41.079 ct_3_1 (USDCHF.ln,H1)   (Financial Conduct Authority (FCA))|----->(((EURUSD),(London Stock Exchange (LSE))),((MJSD - March, June, September, and December),(1000 points)))
2023.02.21 09:57:41.079 ct_3_1 (USDCHF.ln,H1)   (Financial Conduct Authority (FCA))|----->(((EURUSD),(London Stock Exchange (LSE))),((MJSD - March, June, September, and December),(1250 points)))
2023.02.21 09:57:41.079 ct_3_1 (USDCHF.ln,H1)   (Financial Conduct Authority (FCA))|----->(((USDJPY),(London Stock Exchange (LSE))),((JAJO - January, April, July, and October),(1000 points)))
2023.02.21 09:57:41.079 ct_3_1 (USDCHF.ln,H1)   (Financial Conduct Authority (FCA))|----->(((USDJPY),(London Stock Exchange (LSE))),((JAJO - January, April, July, and October),(1250 points)))
2023.02.21 09:57:41.079 ct_3_1 (USDCHF.ln,H1)   (Financial Conduct Authority (FCA))|----->(((USDJPY),(London Stock Exchange (LSE))),((FMAN - February, May, August, and November),(1000 points)))
2023.02.21 09:57:41.079 ct_3_1 (USDCHF.ln,H1)   (Financial Conduct Authority (FCA))|----->(((USDJPY),(London Stock Exchange (LSE))),((FMAN - February, May, August, and November),(1250 points)))
2023.02.21 09:57:41.079 ct_3_1 (USDCHF.ln,H1)   (Financial Conduct Authority (FCA))|----->(((USDJPY),(London Stock Exchange (LSE))),((MJSD - March, June, September, and December),(1000 points)))
2023.02.21 09:57:41.079 ct_3_1 (USDCHF.ln,H1)   (Financial Conduct Authority (FCA))|----->(((USDJPY),(London Stock Exchange (LSE))),((MJSD - March, June, September, and December),(1250 points)))
2023.02.21 09:57:41.079 ct_3_1 (USDCHF.ln,H1)   |
2023.02.21 09:57:41.079 ct_3_1 (USDCHF.ln,H1)   {(((EURUSD),(Chicago Board Options Exchange (CBOE))),((JAJO - January, April, July, and October),(1000 points))),(((EURUSD),(Chicago Board Options Exchange (CBOE))),((JAJO - January, April, July, and October),(1250 points))),(((EURUSD),(Chicago Board Options Exchange (CBOE))),((FMAN - February, May, August, and November),(1000 points))),(((EURUSD),(Chicago Board Options Exchange (CBOE))),((FMAN - February, May, August, and November),(1250 points))),(((EURUSD),(Chicago Board Options Exchange (CBOE))),((MJSD
*/
```

Using universal properties can also make it easier to reason about complex financial structures like collateralised debt obligations (CDOs). These structures are typically composed of multiple underlying assets, each with its own risk profile. By defining the underlying assets as domains in a category, and using universal properties to define the structure of the CDO, we can develop more abstract and general models that capture the joint risk associated with the underlying assets more effectively. Furthermore, the concept of universal property can be used to develop new financial instruments or products by identifying their desired universal properties, rather than trying to create a specific financial instrument from scratch. This can lead to more innovative and powerful financial products that are better suited to the needs of investors and the marketplace.

```
      ////////////////////////////
      //PRODUCT UNIVERSAL-PROPERTY
      ////////////////////////////

      //EX no.2

      CDomain<string> _d_hedge,_d_cover,_d_postion,_d_p2_property;
      //
      CElement<string> _ep2;_ep2.Cardinality(1);
      //
      _d_hedge.Cardinality(2);
      _ep2.Set(0,"EURUSD");_d_hedge.Set(0,_ep2,true);
      _ep2.Set(0,"GBPUSD");_d_hedge.Set(1,_ep2,true);
      //
      _d_cover.Cardinality(2);
      _ep2.Set(0,"USDCHF");_d_cover.Set(0,_ep2,true);
      _ep2.Set(0,"USDJPY");_d_cover.Set(1,_ep2,true);
      //
      _d_postion.Cardinality(4);
      _ep2.Set(0,"EURCHF");_d_postion.Set(0,_ep2,true);
      _ep2.Set(0,"EURJPY");_d_postion.Set(1,_ep2,true);
      _ep2.Set(0,"GBPCHF");_d_postion.Set(2,_ep2,true);
      _ep2.Set(0,"GBPJPY");_d_postion.Set(3,_ep2,true);
      //
      printf(__FUNCSIG__+" hedge domain: "+PrintDomain(_d_hedge,0));
      printf(__FUNCSIG__+" cover domain: "+PrintDomain(_d_cover,0));
      printf(__FUNCSIG__+" postion domain: "+PrintDomain(_d_postion,0));

      CProduct _product_hc;

      GetProduct(_d_hedge,_d_cover,_product_hc,0);
      printf(__FUNCSIG__+" hedge & cover product: "+PrintDomain(_product_hc.cone,0));

      CProduct _product_hcp;

      GetProduct(_product_hc.cone,_d_postion,_product_hcp,0);
      printf(__FUNCSIG__+" hedge & cover & postion product: "+PrintDomain(_product_hcp.cone,0));
      //

      CDomain<double> _d_p2_eu,_d_p2_gu,_d_p2_uc,_d_p2_uj,_d_p2_ec,_d_p2_ej,_d_p2_gc,_d_p2_gj;
      _d_p2_eu.Cardinality(1);_d_p2_gu.Cardinality(1);_d_p2_uc.Cardinality(1);_d_p2_uj.Cardinality(1);
      _d_p2_ec.Cardinality(1);_d_p2_ej.Cardinality(1);_d_p2_gc.Cardinality(1);_d_p2_gj.Cardinality(1);

      int _eu_handle=iATR("EURUSD",_Period,__product_size);
      int _gu_handle=iATR("GBPUSD",_Period,__product_size);
      int _uc_handle=iATR("USDCHF",_Period,__product_size);
      int _uj_handle=iATR("USDJPY",_Period,__product_size);
      int _ec_handle=iATR("EURCHF",_Period,__product_size);
      int _ej_handle=iATR("EURJPY",_Period,__product_size);
      int _gc_handle=iATR("GBPCHF",_Period,__product_size);
      int _gj_handle=iATR("GBPJPY",_Period,__product_size);

      if
      (
      FillDomain(_d_p2_eu,0,1,_eu_handle)
      &&
      FillDomain(_d_p2_gu,0,1,_gu_handle)
      &&
      FillDomain(_d_p2_uc,0,1,_uc_handle)
      &&
      FillDomain(_d_p2_uj,0,1,_uj_handle)
      &&
      FillDomain(_d_p2_ec,0,1,_ec_handle)
      &&
      FillDomain(_d_p2_ej,0,1,_ej_handle)
      &&
      FillDomain(_d_p2_gc,0,1,_gc_handle)
      &&
      FillDomain(_d_p2_gj,0,1,_gj_handle)
      )
      {
         CElement<double> _e_eu,_e_gu,_e_uc,_e_uj,_e_ec,_e_ej,_e_gc,_e_gj;
         //
         if
         (
         _d_p2_eu.Get(0,_e_eu) && _d_p2_gu.Get(0,_e_gu) &&
         _d_p2_uc.Get(0,_e_uc) && _d_p2_uj.Get(0,_e_uj) &&
         _d_p2_ec.Get(0,_e_ec) && _d_p2_ej.Get(0,_e_ej) &&
         _d_p2_gc.Get(0,_e_gc) && _d_p2_gj.Get(0,_e_gj)
         )
         {
            _d_p2_property.Cardinality(4);
            _ep2.Set(0,DoubleToString(_e_eu.Get(0),3)+","+DoubleToString(_e_uc.Get(0),3)+","+DoubleToString(_e_ec.Get(0),3));_d_p2_property.Set(0,_ep2,true);
            _ep2.Set(0,DoubleToString(_e_gu.Get(0),3)+","+DoubleToString(_e_uc.Get(0),3)+","+DoubleToString(_e_gc.Get(0),3));_d_p2_property.Set(1,_ep2,true);
            _ep2.Set(0,DoubleToString(_e_eu.Get(0),3)+","+DoubleToString(_e_uj.Get(0),3)+","+DoubleToString(_e_ej.Get(0),3));_d_p2_property.Set(2,_ep2,true);
            _ep2.Set(0,DoubleToString(_e_gu.Get(0),3)+","+DoubleToString(_e_uj.Get(0),3)+","+DoubleToString(_e_gj.Get(0),3));_d_p2_property.Set(3,_ep2,true);

            //
            _product_hcp.property=_d_p2_property;
            //
            _product_hcp.universality.domain=_product_hcp.property;
            _product_hcp.universality.codomain=_product_hcp.cone;
            //
            CMorphism<string,string> _m_p2;
            _m_p2.domain=_product_hcp.property;
            _m_p2.codomain=_product_hcp.cone;
            //

            for(int c=0;c<_product_hcp.property.Cardinality();c++)
            {
               CElement<string> _e_property;_e_property.Cardinality(1);
               if(_product_hcp.property.Get(c,_e_property) && _e_property.Get(0)!="")
               {
                  for(int cc=0;cc<_product_hcp.cone.Cardinality();cc++)
                  {
                     CElement<string> _e_cone;_e_cone.Cardinality(1);
                     if(_product_hcp.cone.Get(cc,_e_cone) && _e_cone.Get(0)!="")
                     {
                        if(c==0)
                        {
                           if(StringFind(_e_cone.Get(0),"EURUSD")>=0&&StringFind(_e_cone.Get(0),"USDCHF")>=0&&StringFind(_e_cone.Get(0),"EURCHF")>=0)
                           {
                              if(_product_hcp.universality.Morphisms(_product_hcp.universality.Morphisms()+1))
                              {
                                 if(_m_p2.Morph(_product_hcp.property,_product_hcp.cone,_e_property,_e_cone))
                                 {
                                    if(!_product_hcp.universality.Set(_product_hcp.universality.Morphisms()-1,_m_p2))
                                    {
                                    }
                                 }
                              }
                           }
                        }
                        else if(c==1)
                        {
                           if(StringFind(_e_cone.Get(0),"GBPUSD")>=0&&StringFind(_e_cone.Get(0),"USDCHF")>=0&&StringFind(_e_cone.Get(0),"GBPCHF")>=0)
                           {
                              if(_product_hcp.universality.Morphisms(_product_hcp.universality.Morphisms()+1))
                              {
                                 if(_m_p2.Morph(_product_hcp.property,_product_hcp.cone,_e_property,_e_cone))
                                 {
                                    if(!_product_hcp.universality.Set(_product_hcp.universality.Morphisms()-1,_m_p2))
                                    {
                                    }
                                 }
                              }
                           }
                        }
                        else if(c==2)
                        {
                           if(StringFind(_e_cone.Get(0),"EURUSD")>=0&&StringFind(_e_cone.Get(0),"USDJPY")>=0&&StringFind(_e_cone.Get(0),"EURJPY")>=0)
                           {
                              if(_product_hcp.universality.Morphisms(_product_hcp.universality.Morphisms()+1))
                              {
                                 if(_m_p2.Morph(_product_hcp.property,_product_hcp.cone,_e_property,_e_cone))
                                 {
                                    if(!_product_hcp.universality.Set(_product_hcp.universality.Morphisms()-1,_m_p2))
                                    {
                                    }
                                 }
                              }
                           }
                        }
                        else if(c==3)
                        {
                           if(StringFind(_e_cone.Get(0),"GBPUSD")>=0&&StringFind(_e_cone.Get(0),"USDJPY")>=0&&StringFind(_e_cone.Get(0),"GBPJPY")>=0)
                           {
                              if(_product_hcp.universality.Morphisms(_product_hcp.universality.Morphisms()+1))
                              {
                                 if(_m_p2.Morph(_product_hcp.property,_product_hcp.cone,_e_property,_e_cone))
                                 {
                                    if(!_product_hcp.universality.Set(_product_hcp.universality.Morphisms()-1,_m_p2))
                                    {
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }

            printf(__FUNCSIG__+" universal property hcp: "+PrintHomomorphism(_product_hcp.universality,0));
         }
      }

```

On running this script again we get the homomorphism print of the universal property that in essence maps risk levels (in this case ATR values) to a portfolio in the cone domain which was formed as a result of products of various securities.

```
2023.02.21 20:21:53.614 ct_3_1 (USDCHF.ln,H1)   void OnStart() universal property hcp:
2023.02.21 20:21:53.614 ct_3_1 (USDCHF.ln,H1)
2023.02.21 20:21:53.614 ct_3_1 (USDCHF.ln,H1)   {(0.002,0.001,0.001),(0.002,0.001,0.002),(0.002,0.145,0.189),(0.002,0.145,0.258)}
2023.02.21 20:21:53.614 ct_3_1 (USDCHF.ln,H1)   |
2023.02.21 20:21:53.614 ct_3_1 (USDCHF.ln,H1)   (0.002,0.001,0.001)|----->(((EURUSD),(USDCHF)),(EURCHF))
2023.02.21 20:21:53.614 ct_3_1 (USDCHF.ln,H1)   (0.002,0.001,0.002)|----->(((GBPUSD),(USDCHF)),(GBPCHF))
2023.02.21 20:21:53.614 ct_3_1 (USDCHF.ln,H1)   (0.002,0.145,0.189)|----->(((EURUSD),(USDJPY)),(EURJPY))
2023.02.21 20:21:53.614 ct_3_1 (USDCHF.ln,H1)   (0.002,0.145,0.258)|----->(((GBPUSD),(USDJPY)),(GBPJPY))
2023.02.21 20:21:53.614 ct_3_1 (USDCHF.ln,H1)   |
2023.02.21 20:21:53.614 ct_3_1 (USDCHF.ln,H1)   {(((EURUSD),(USDCHF)),(GBPJPY)),(((EURUSD),(USDCHF)),(GBPCHF)),(((EURUSD),(USDCHF)),(EURJPY)),(((EURUSD),(USDCHF)),(EURCHF)),(((EURUSD),(USDJPY)),(GBPJPY)),(((EURUSD),(USDJPY)),(GBPCHF)),(((EURUSD),(USDJPY)),(EURJPY)),(((EURUSD),(USDJPY)),(EURCHF)),(((GBPUSD),(USDCHF)),(GBPJPY)),(((GBPUSD),(USDCHF)),(GBPCHF)),(((GBPUSD),(USDCHF)),(EURJPY)),(((GBPUSD),(USDCHF)),(EURCHF)),(((GBPUSD),(USDJPY)),(GBPJPY)),(((GBPUSD),(USDJPY)),(GBPCHF)),(((GBPUSD),(USDJPY)),(EURJPY)),(((GBPUSD),(USDJPY)),(EURCHF))}
```

### Coproducts

In algebra, the [coproduct](https://en.wikipedia.org/wiki/Coproduct "https://en.wikipedia.org/wiki/Coproduct") of two groups (G and H) is a domain that contains all the elements of both G and H, with the domain operation defined in such a way that the resulting structure is the "smallest" domain that contains both G and H. This is different from the product of two domains, which is a domain that contains all pairs of elements (g,h) where g is in G and h is in H, with the domain operation defined component-wise. The coproduct is a [colimit](https://en.wikipedia.org/wiki/Limit_(category_theory) "https://en.wikipedia.org/wiki/Limit_(category_theory)") because it is the "least upper bound" of G and H, in the sense that any other domain that contains both G and H must also contain the coproduct.

A coproduct of two domains A and B is also a third domain C, along with two injection morphisms from A and B to C respectively, such that for any other domain D with two morphisms from A and B respectively, there exists a unique morphism from C to D that makes certain diagrams commute. The coproduct is also referred to as the "disjoint union" or "sum" of A and B, as it captures the idea of a choice between two different domains. Intuitively, the coproduct C represents the "alternative" or "divergent" behaviour of A and B.

Coproduct typically satisfy the following properties:

- They are associative: (A + B) + C is isomorphic to A + (B + C).
- They are commutative: A + B is isomorphic to B + A.
- The coproduct of a domain with the initial domain is isomorphic to the domain itself: A + 0 is isomorphic to A.
- The coproduct of a domain with the terminal domain is isomorphic to the domain itself: A + 1 is isomorphic to 1.

The concept of coproducts can also be applied to various mathematical structures, including sets, groups, rings, and vector spaces, as well as to more abstract structures such as topological spaces and categories themselves. In each case, the coproduct captures the idea of a "choice" or "alternative" between two domains or structures, and it provides a way of reasoning about the behavior of the system as a whole.

From a trader's viewpoint, coproducts in category theory can be used to model and analyse investment portfolios, and can provide a better alternative to traditional methods for portfolio valuation. This is because they are a way of combining domains in a category, where the resulting domain represents a choice between the original domains. For example, the coproduct of two numbers is their maximum. This property can be used to model investment portfolios as a choice between different assets, and to analyse the valuation of these portfolios in a more flexible and general way.

As an illustration, suppose we have an investment portfolio that contains two assets: a stock and a bond. We want to analyse the value of the portfolio, and how it changes under different market conditions. Traditionally, this would be done using methods such as discounted cash flow analysis or the capital asset pricing model (CAPM). However, using coproducts in category theory, we can model the portfolio as a choice between the stock and the bond. The resulting coproduct domain would represent the value of the portfolio under different market conditions, and would capture the joint risk associated with holding both assets.

To show this, for simplicity, let's assume that the stock has a current value of $50 and the bond has a current value of $100. We can define the coproduct of the stock and the bond as their maximum, which is $100. This means that the value of the portfolio is at least $100, regardless of the market conditions. Using this approach, we can analyse the behaviour of the portfolio under different market scenarios. For instance, if the stock price increases to $60 and the bond price remains the same, the value of the portfolio would be $110, which is greater than $60 or $100, captures the price increment, but remains less than the overall sum $160. Similarly, if the bond price decreases to $90 and the stock price remains the same, the value of the portfolio would still be taken at $100, which is greater than $50 or $90 but is still below the overall sum of $140. On the whole coproducts provide a conservative less volatile approach to valuation that is less pegged to market price gyrations. We therefore can model investment portfolios as a choice between different assets, and analyse their behaviour in a more flexible and general way. This could lead to more accurate and effective _long-term_ portfolio valuation strategies, and can better capture the joint risk associated with holding multiple assets.

### Coproducts and Universal Property

The concept of [universal property](https://en.wikipedia.org/wiki/Universal_property "https://en.wikipedia.org/wiki/Universal_property"), as shown with products above, is a powerful tool in category theory that can also be used to further enhance the analysis of coproducts in finance. It provides a formal way of characterising the way in which a coproduct is unique up to a certain kind of isomorphism, and this can lead to more efficient and precise reasoning about investment portfolios. As an example, consider a portfolio that consists of three assets: a stock, a bond, and a real estate investment trust (REIT). We can model this portfolio using the coproduct of the stock, the bond, and the REIT in a category of investment assets. The coproduct of these three assets can be thought of as the choice between owning the stock, the bond, or the REIT. We can use the universal property of coproducts to understand how this choice is unique up to isomorphism. Specifically, the universal property of the coproduct states that for any other domain Z and morphisms from the three assets to Z, there exists a unique morphism from the coproduct to Z that makes certain diagrams commute.

Using the universal property, we can reason about the behaviour of the coproduct in a more precise and efficient way. For example, suppose we want to calculate the joint risk associated with holding the stock, the bond, and the REIT. We can use the universal property of the coproduct to show that any valuation that satisfies the diagrams required by the universal property must also satisfy certain properties, such as the subadditivity of risk. As an illustration, suppose the stock has a value of $50, the bond has a value of $100, and the REIT has a value of $150. Using the coproduct, we can model the portfolio as a choice between these three assets, and calculate the joint risk associated with holding them.

Specifically, the joint risk can be calculated as the maximum of the individual risks associated with each asset. Suppose the risk associated with the stock is 10%, the risk associated with the bond is 5%, and the risk associated with the REIT is 8%. Using the coproduct, we can calculate the joint risk as the maximum of these three risks, which is 10%. This is the risk associated with the portfolio under the assumption that the risks associated with each asset are independent. Using the universal property of the coproduct, we can reason about the behaviour of the portfolio under different assumptions about the correlation between the risks associated with each asset. For example, we can use the universal property to show that if the risks associated with the stock and the REIT are positively correlated, then the joint risk associated with holding them will be higher than the maximum of their individual risks.

In summary, the use of the universal property in conjunction with coproducts can provide a more precise and efficient way of reasoning about investment portfolios. It allows us to understand the unique properties of a coproduct up to isomorphism, and to reason about the behavior of a portfolio under different assumptions about correlation and risk.

### Conclusion

In conclusion, category theory provides a powerful set of tools for reasoning about complex systems, and its concepts of products, coproducts, and the universal property have important applications in finance, particularly in the realm of algorithmic trading. By implementing these concepts in MQL5, traders can gain deeper insights into the behaviour of financial markets and develop more effective trading strategies. The use of products and coproducts allows traders to reason about the joint or divergent behaviour of financial instruments, and to construct more sophisticated portfolios that take into account the interdependence between assets.

The universal property ensures that these constructions are unique and that they satisfy certain desirable properties. Limits and colimits provide a more abstract and broad way to reason about the behaviour of sequences of domains, and they allow traders to develop more sophisticated risk management strategies. Overall, the application of category theory in finance has the potential to revolutionize the way we think about financial markets and to enable traders to make more informed decisions. By incorporating these concepts into MQL5, traders can take advantage of the full power of category theory and develop more effective trading strategies that are based on a deeper understanding of the underlying structure of financial markets.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12085.zip "Download all attachments in the single ZIP archive")

[SignalCT3.mqh](https://www.mql5.com/en/articles/download/12085/signalct3.mqh "Download SignalCT3.mqh")(10.76 KB)

[ct\_3\_1.mq5](https://www.mql5.com/en/articles/download/12085/ct_3_1.mq5 "Download ct_3_1.mq5")(96.41 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Codex Pipelines, from Python to MQL5, for Indicator Selection: A Multi-Quarter Analysis of the XLF ETF with Machine Learning](https://www.mql5.com/en/articles/20595)
- [Codex Pipelines: From Python to MQL5 for Indicator Selection — A Multi-Quarter Analysis of the FXI ETF](https://www.mql5.com/en/articles/20550)
- [Market Positioning Codex for VGT with Kendall's Tau and Distance Correlation](https://www.mql5.com/en/articles/20271)
- [Markets Positioning Codex in MQL5 (Part 2): Bitwise Learning, with Multi-Patterns for Nvidia](https://www.mql5.com/en/articles/20045)
- [Markets Positioning Codex in MQL5 (Part 1): Bitwise Learning for Nvidia](https://www.mql5.com/en/articles/20020)
- [MQL5 Wizard Techniques you should know (Part 85): Using Patterns of Stochastic-Oscillator and the FrAMA with Beta VAE Inference Learning](https://www.mql5.com/en/articles/19948)
- [MQL5 Wizard Techniques you should know (Part 84): Using Patterns of Stochastic Oscillator and the FrAMA - Conclusion](https://www.mql5.com/en/articles/19890)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/442761)**
(3)


![Stephen Njuki](https://c.mql5.com/avatar/avatar_na2.png)

**[Stephen Njuki](https://www.mql5.com/en/users/ssn)**
\|
14 Mar 2023 at 16:07

Missed a file attachment.


![Yevgeniy Koshtenko](https://c.mql5.com/avatar/2025/3/67e0f73d-60b1.jpg)

**[Yevgeniy Koshtenko](https://www.mql5.com/en/users/koshtenko)**
\|
8 May 2023 at 08:16

I was very interested in the article. But I cannot [create an Expert Advisor](https://www.metatrader5.com/en/terminal/help/algotrading/autotrading "MetaTrader 5 Help: Create an Expert Advisor in the MetaTrader 5 Client Terminal") based on a signal in the Wizard. It asks for this: <Expert\\Signal\\My\\Cct.mqh>


![Stephen Njuki](https://c.mql5.com/avatar/avatar_na2.png)

**[Stephen Njuki](https://www.mql5.com/en/users/ssn)**
\|
25 May 2023 at 17:36

Pse see attachment.


![Learn how to design a trading system by Bill Williams' MFI](https://c.mql5.com/2/52/bw_mfi_avatar.png)[Learn how to design a trading system by Bill Williams' MFI](https://www.mql5.com/en/articles/12172)

This is a new article in the series in which we learn how to design a trading system based on popular technical indicators. This time we will cover Bill Williams' Market Facilitation Index (BW MFI).

![Revisiting Murray system](https://c.mql5.com/2/51/murrey_system_avatar.png)[Revisiting Murray system](https://www.mql5.com/en/articles/11998)

Graphical price analysis systems are deservedly popular among traders. In this article, I am going to describe the complete Murray system, including its famous levels, as well as some other useful techniques for assessing the current price position and making a trading decision.

![Data Science and Machine Learning (Part 11): Naïve Bayes, Probability theory in Trading](https://c.mql5.com/2/52/naive_bayes_avatar.png)[Data Science and Machine Learning (Part 11): Naïve Bayes, Probability theory in Trading](https://www.mql5.com/en/articles/12184)

Trading with probability is like walking on a tightrope - it requires precision, balance, and a keen understanding of risk. In the world of trading, the probability is everything. It's the difference between success and failure, profit and loss. By leveraging the power of probability, traders can make informed decisions, manage risk effectively, and achieve their financial goals. So, whether you're a seasoned investor or a novice trader, understanding probability is the key to unlocking your trading potential. In this article, we'll explore the exciting world of trading with probability and show you how to take your trading game to the next level.

![Creating an EA that works automatically (Part 05): Manual triggers (II)](https://c.mql5.com/2/50/Aprendendo_construindo_005_avatar.png)[Creating an EA that works automatically (Part 05): Manual triggers (II)](https://www.mql5.com/en/articles/11237)

Today we'll see how to create an Expert Advisor that simply and safely works in automatic mode. At the end of the previous article, I suggested that it would be appropriate to allow manual use of the EA, at least for a while.

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/12085&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071648123933109225)

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