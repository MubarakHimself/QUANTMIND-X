---
title: Category Theory in MQL5 (Part 2)
url: https://www.mql5.com/en/articles/11958
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:12:06.500846
---

[![](https://www.mql5.com/ff/sh/7h2yc16rtqsn2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Market analysis\\
\\
Dozens of channels, thousands of subscribers and daily updates. Learn more about trading.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=glufvbpblsoxonicqfngsyuzwfebnilr&s=103cc3ab372a16872ca1698fc86368ffe3b3eaa21b59b4006d5c6c10f48ad545&uid=&ref=https://www.mql5.com/en/articles/11958&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071650086733163507)

MetaTrader 5 / Integration


### Introduction

This is a continuation from a previous [article](https://www.mql5.com/en/articles/11849) on Category Theory use in MQL5. Just to recap we introduced Category theory as a branch of mathematics that is particularly useful for organising and classifying information. In that context, we discussed some basic concepts of category theory and how it can be applied to the analysis of financial time series data. Specifically, we examined elements, domains, and morphisms. Elements are the basic units of information within a category theory system, and they are typically thought of as members of a set, referred to in these series, as a domain. A domain can have relationships with other domains through a process called a morphism. For this article we will start by dwelling on what constitutes a category by looking at the axioms of identity, and association, commutative diagram. In doing so, and throughout the article, we will examine examples and also highlight examples of non-categories. Finally we'll conclude by introducing ontology logs.

Before we delve into the axiom definitions of a category it may be helpful to look at a few day to day examples of categories 'in action'. We've already defined domains and the elements they contain. So if to begin we take a category to be a collection of related domains then the following could serve as illustrative examples.

- A category of modes of transportation, with domains representing different modes of transportation (e.g. cars, planes, and trains etc). The elements of each domain would be the respective kinds transport vehicles. For example in cars we could have ride-sharing, car-rental, personal-car, taxi; while in planes we could have private jets, commercial, leased planes; with trains we could have trams, bullet trains, steam engines etc. The morphism between these could then serve to define a complete travel itinerary. For example if you took a lyft to the airport, flew commercial and then took a bullet train from that airport to your destination, each of those choices can be mapped easily within this category.

- A category of courses of a meal, with domains representing menus for each course. Lets say we have 5 courses namely: appetiser, soup, main-course, dessert, and cheese. In this case the elements of each domain would be the food items on the menu for that respective course. For example the appetiser course could have in its domain (menu) consisting choices: Candied carrots with honey, cumin, and paprika; OR Mushrooms stuffed with Pecorino Romano, garlic, and bread crumbs; OR Charred broccoli with shishito peppers and pickled onions;  Like wise the soup domain could have: Tuscan white bean and roasted garlic soup; OR Pumpkin sage bisque; OR Cold melon and basil soup. Each of these menu items represents an element in their domain. Similarly the morphism between these will then serve to define a complete meal chosen by a restaurant customer. For example if the customer had candied carrots then pumpkin and so on for all the other courses, each of those choices like above can be mapped easily within the category.

- Another category example can be types of weekly entertainment. Here our domains can be live-sports, tv-streaming, and park-visits. The elements of each domain could be: NFL, or MLB, or NBA for the live-sports domain; Netflix or HBO or AMC for tv-streaming domain; and Disney or Zoo or nature park for park-visits. Once again the audience can get to select from each domain a sports game to attend, a tv show to watch and a park to visit. And the string of selections across these domains (morphisms) easily allow this information to be mapped.

So all of this, and obviously much more can be logged by Category Theory. But to what end?  There is a saying "a bad workman blames his tools" and I feel this is the point here. For the use is really determined by the user. In my opinion Category Theory critically serves to quantify [equivalence](https://en.wikipedia.org/wiki/Equivalence_of_categories "https://en.wikipedia.org/wiki/Equivalence_of_categories"). This is a topic we will broach in further articles after we've covered some ground on the basic concepts. Suffice it to say it is the reason two separate categories on different topics that may appear unrelated on the surface, but on deeper inspection are found to be identical or mirror opposites, are discovered. These insights are vital in decision making because you seldom have enough information at the time of making decisions. If however you have correlatable categories then the information gap can be bridged.

### Identity

[Isomorphism](https://en.wikipedia.org/wiki/Isomorphism_of_categories "https://en.wikipedia.org/wiki/Isomorphism_of_categories") is a crucial property of homomorphisms in category theory because it ensures that the structure of the domains in the target category is preserved under the mapping. It also guarantees the preservation of the algebraic operations of the domains in the source category. For instance, let's consider a clothing category where the domains are shirts and pants, and the morphisms are the functions that map the size of a shirt to the size of a pant. A homomorphism in this category would be a function that preserves the pairing of the sizes of the shirts to respective sizes of pants. An isomorphism in this category would be a function that not only preserves the algebraic pairing of the sizes, but also establishes a one-to-one correspondence between the sizes of the shirts and pants. This means that for any shirt size, there is exactly one corresponding pant size, and **vice versa**. For instance, consider the function that maps the size of a shirt (e.g. "small", "medium", "large") to the size of a pant (e.g. "26", "28", "30", "32"). This function is a homomorphism because it preserves and defines a pairing of the sizes (e.g. "small" can be paired with "26" ). But it's not an isomorphism because it doesn't establish a one-to-one correspondence between the sizes of the shirts and pants given that "small" can also be worn with "28" or "26". There is no reversibility.

![non_iso](https://c.mql5.com/2/51/non_iso.png)

On the other hand, consider the function that maps the size of a shirt (e.g. "small", "medium", "large") to only the sizes of a pant (e.g. "28", "30", "32"). This function is not only a homomorphism but also an isomorphism because it also allows reversibility.

![iso](https://c.mql5.com/2/51/iso.png)

In the example of a clothing category, where the domains are shirts and pants, and the morphisms are the functions that map the size of a shirt to the size of a pant, a single morphism cannot be isomorphic because all individual morphisms are inherently reversible and there 'isomorphic'. The property of isomorphism deals with a group of morphisms (aka homomorphism set) from one domain to a co-domain and it can only be deemed present if all those morphisms as a group can be reversed while preserving the algebraic pairing properties of the homomorphism. This is why it is important to enumerate all morphisms from a domain before confirming isomorphism. Without considering all possible morphisms, it may appear that two objects are isomorphic when in fact they are not. A precursor for isomorphism is the [cardinality](https://en.wikipedia.org/wiki/Cardinality "https://en.wikipedia.org/wiki/Cardinality") of the domains. This value needs to be the same for both domains. If the domains have mismatched cardinality then the homomorphism between them cannot be reversed because from one domain you will have more than one element mapping to the same co-element. Therefore, we need a group of morphisms between domains to define isomorphism. Typically we need two morphisms, one from the shirts to pants and the other from pants to shirts, that establishes a one-to-one correspondence between the sizes of the shirts and pants. These morphisms should be inverse of each other, meaning that if one morphism takes a shirt of size "small" to a pant of size "28", the other morphism should take a pant of size "28" to a shirt of size "small". And when we compose these two morphisms, it should give us the identity morphism for the domain we started.

This axioms requirement in category theory has inferred a need for self-mapping morphisms. There is however still [debate](https://www.mql5.com/go?link=https://math.stackexchange.com/questions/1464533/on-the-identity-map-requirement-in-the-definition-of-category "https://math.stackexchange.com/questions/1464533/on-the-identity-map-requirement-in-the-definition-of-category") as to whether they are relevant in defining a category.

Moving on though let us illustrate an isomorphism in MQL5. I have re-written most of the script shared in article one to now include templating. All classes from the element class up to the category class incorporate the use of template data types for flexibility. I have however 'enumerated' the types available within the category class. These are: 'datetime', 'string', 'double', and int. The 'int' type is the default and will be used if a data type like 'color', for instance, is being used. All code is attached at the end of the article. Here is our new homomorphism class.

```
//+------------------------------------------------------------------+
//| HOMO-MORPHISM CLASS                                              |
//+------------------------------------------------------------------+
template <typename TD,typename TC>
class CHomomorphism                 : public CObject
   {
      protected:

      int                           morphisms;

      public:

      CDomain<TD>                   domain;
      CDomain<TC>                   codomain;

      CMorphism<TD,TC>              morphism[];

      int                           Morphisms() { return(morphisms); }
      bool                          Morphisms(int Value) { if(Value>=0 && Value<INT_MAX) { morphisms=Value; ArrayResize(morphism,morphisms); return(true); } return(false); }

      bool                          Get(int MorphismIndex,CMorphism<TD,TC> &Morphism) { if(MorphismIndex>=0 && MorphismIndex<Morphisms()) { Morphism=morphism[MorphismIndex]; Morphism.domain=domain; Morphism.codomain=codomain; return(true); } return(false); }

                                    template <typename TDD,typename TDC>
      bool                          Set(int ValueIndex,CMorphism<TDD,TDC> &Value)
                                    {
                                       if
                                       (
                                       (string(typename(TD))!=string(typename(TDD)))
                                       ||
                                       (string(typename(TC))!=string(typename(TDC)))
                                       ||
                                       )
                                       {
                                          return(false);
                                       }
                                       //
                                       if(ValueIndex>=0 && ValueIndex<Morphisms())
                                       {
                                          Value.domain=domain;
                                          Value.codomain=codomain;

                                          if(Index(Value)==-1)
                                          {
                                             morphism[ValueIndex]=Value;

                                             return(true);
                                          }
                                       }

                                       return(false);
                                    };


                                    template <typename TDD,typename TDC>
      int                           Index(CMorphism<TDD,TDC> &Value)
                                    {
                                       int _index=-1;
                                       //
                                       if
                                       (
                                       (string(typename(TD))!=string(typename(TDD)))
                                       ||
                                       (string(typename(TC))!=string(typename(TDC)))
                                       ||
                                       )
                                       {
                                          return(_index);
                                       }
                                       //
                                       for(int m=0; m<morphisms; m++)
                                       {
                                          if(MorphismMatch(Value,morphism[m]))
                                          {
                                             _index=m; break;
                                          }
                                       }

                                       return(_index);
                                    }

                                    CHomomorphism(void){  Morphisms(0); };
                                    ~CHomomorphism(void){};
   };
```

To test for isomorphism, we will use the same domains in the previous article and we will run them through the 'IsIsomorphic' function. This function returns a boolean value with true indicating success and false failure.

```
//+------------------------------------------------------------------+
//| Get Isomorphisms function                                        |
//+------------------------------------------------------------------+
template <typename TD,typename TC>
bool IsIsomorphic(CDomain<TD> &A,CDomain<TC> &B,CHomomorphism<TD,TC> &Output[])
   {
      if(A.Cardinality()!=B.Cardinality())
      {
         return(false);
      }

      int _cardinal=A.Cardinality();

      uint _factorial=MathFactorial(_cardinal);

      ArrayResize(Output,_factorial);

      for(uint f=0;f<_factorial;f++)
      {
         ArrayResize(Output[f].morphism,_cardinal);
         //
         for(int c=0;c<_cardinal;c++)
         {
            Output[f].morphism[c].domain=A;
            Output[f].morphism[c].codomain=B;
         }
      }

      int _index=0;
      CDomain<TC> _output[];ArrayResize(_output,_factorial);
      GetIsomorphisms(B, 0, _cardinal-1, _cardinal, _index, _output);

      for(uint f=0;f<_factorial;f++)
      {
         for(int c=0;c<_cardinal;c++)
         {
            CElement<TC> _ec;
            if(_output[f].Get(c,_ec))
            {
               for(int cc=0;cc<_cardinal;cc++)
               {
                  CElement<TC> _ecc;
                  if(B.Get(cc,_ecc))
                  {
                     if(ElementMatch(_ec,_ecc))
                     {
                        if(Output[f].morphism[c].Codomain(cc))
                        {
                           break;
                        }
                     }
                  }
               }
            }

            if(Output[f].morphism[c].Domain(c))
            {
            }
         }
      }

      return(true);
   }
```

The output homomorphism though needs to be specified at the on set and in our case we are using the variable '\_h\_i'. This output value which is an array of the homomorphism class will include an enumeration of all the possible isomorphic homomorphisms between the two input domains.

```
      //IDENTITY
      CHomomorphism<int,int> _h_i[];
      //is evens isomorphic to odds?
      if(IsIsomorphic(_evens,_odds,_h_i))
      {
         printf(__FUNCSIG__+" evens can be isomorphic to odds by up to: "+IntegerToString(ArraySize(_h_i))+" homomorphisms. These could be... ");
         for(int s=0; s<ArraySize(_h_i); s++)
         {
            printf(__FUNCSIG__);

            string _print="";
            for(int ss=0; ss<ArraySize(_h_i[s].morphism); ss++)
            {
               _print+=PrintMorphism(_h_i[s].morphism[ss],0);
            }

            printf(_print+" at: "+IntegerToString(s));
         }
      }
```

If we run this code we should have the logs below.

```
2023.01.26 10:42:56.909 ct_2 (EURGBP.ln,H1)     void OnStart() evens can be isomorphic to odds by up to: 6 homomorphisms. These could be...
2023.01.26 10:42:56.909 ct_2 (EURGBP.ln,H1)     void OnStart()
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     (0)|----->(1)
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     (2)|----->(3)
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     (4)|----->(5)
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)      at: 0
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     void OnStart()
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     (0)|----->(1)

2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     (2)|----->(5)
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     (4)|----->(3)
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)      at: 1
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     void OnStart()
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     (0)|----->(3)
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     (2)|----->(1)
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     (4)|----->(5)
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)      at: 2
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     void OnStart()
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     (0)|----->(3)
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     (2)|----->(5)
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     (4)|----->(1)
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)      at: 3
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     void OnStart()
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     (0)|----->(5)
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     (2)|----->(3)
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     (4)|----->(1)
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)      at: 4
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     void OnStart()
2023.01.26 10:42:56.910 ct_2 (EURGBP.ln,H1)     (0)|----->(5)
2023.01.26 10:42:56.911 ct_2 (EURGBP.ln,H1)     (2)|----->(1)
2023.01.26 10:42:56.911 ct_2 (EURGBP.ln,H1)     (4)|----->(3)
2023.01.26 10:42:56.911 ct_2 (EURGBP.ln,H1)      at: 5
```

This is with an evens domain whose print is:

```
2023.01.26 10:42:56.899 ct_2 (EURGBP.ln,H1)     void OnStart() evens are... {(0),(2),(4)}
```

And an odds domain with a print of:

```
2023.01.26 10:42:56.899 ct_2 (EURGBP.ln,H1)     void OnStart() odds are... {(1),(3),(5)}
```

### Association

In category theory, the [association](https://en.wikipedia.org/wiki/Associative_property "https://en.wikipedia.org/wiki/Associative_property") axiom is one of the basic properties that a category must satisfy. Using an example of a category of clothing where the domains are specific clothing items, such as shirts, pants, and shoes, then the morphisms would be the functions that pair the clothings depending on suitability. The axiom of association, also known as the "associativity law," states that the composition of morphisms is associative. This means that when composing multiple morphisms, the order in which they are applied does not affect the final result.

For example, consider the morphism " is worn with " across all 3 domains of clothing (shirts, pants and shoes). Let's say we have a morphism f: t-shirt -> button-up shirt, g: button-up shirt -> jeans and h: jeans -> sneakers. Using the axiom of association, we can define the composition of these morphisms as h o (g o f) = (h o g) o f. This means that the order of the morphisms does not matter and we can group them together in any way we want. This makes the definition of a category simpler, as it allows us to avoid having to compute the parentheses of multiple morphisms. Instead, we can group morphisms together, regardless of their order, and the final result will be the same.

Lets look at this in action in MQL5. We will need to re-compose the category class I had shared in the previous article.

```
//+------------------------------------------------------------------+
//| CATEGORY CLASS                                                   |
//+------------------------------------------------------------------+
class CCategory
   {
      protected:

      int                           domains_datetime;
      int                           domains_string;
      int                           domains_double;
      int                           domains_int;

      int                           ontologies;

      CDomain<datetime>             domain_datetime[];
      CDomain<string>               domain_string[];
      CDomain<double>               domain_double[];
      CDomain<int>                  domain_int[];

      COntology                     ontology[];

      public:

      int                           Domain(string T)
                                    {
                                       if(T=="datetime"){ return(domains_datetime); }
                                       else if(T=="string"){ return(domains_string); }
                                       else if(T=="double"){ return(domains_double); }

                                       return(domains_int);
                                    };

      bool                          Domain(string T,int Value)
                                    {
                                       if(Value>=0 && Value<INT_MAX)
                                       {
                                          if(T=="datetime")
                                          {
                                             if(ArrayResize(domain_datetime,Value)>=Value)
                                             {
                                                domains_datetime=Value;
                                                return(true);
                                             }
                                          }
                                          else if(T=="string")
                                          {
                                             if(ArrayResize(domain_string,Value)>=Value)
                                             {
                                                domains_string=Value;
                                                return(true);
                                             }
                                          }
                                          else if(T=="double")
                                          {
                                             if(ArrayResize(domain_double,Value)>=Value)
                                             {
                                                domains_double=Value;
                                                return(true);
                                             }
                                          }
                                          else //if(T=="int")
                                          {
                                             if(ArrayResize(domain_int,Value)>=Value)
                                             {
                                                domains_int=Value;
                                                return(true);
                                             }
                                          }
                                       }

                                       return(false);
                                    };


      int                           Ontology(){ return(ontologies); };
      bool                          Ontology(int Value){ if(Value>=0 && Value<INT_MAX){ ontologies=Value; ArrayResize(ontology,ontologies); return(true); } return(false); };


                                    template <typename T>
      bool                          Set(int ValueIndex,CDomain<T> &Value)
                                    {
                                       if(Index(Value)==-1 && ValueIndex>=0)
                                       {
                                          if
                                          (
                                          ValueIndex<Domain(string(typename(T)))
                                          ||
                                          (ValueIndex>=Domain(string(typename(T))) && Domain(string(typename(T)),ValueIndex+1))
                                          )
                                          {
                                             if(string(typename(T))=="datetime")
                                             {
                                                domain_datetime[ValueIndex]=Value;
                                                return(true);
                                             }
                                             else if(string(typename(T))=="string")
                                             {
                                                domain_string[ValueIndex]=Value;

                                                return(true);
                                             }
                                             else if(string(typename(T))=="double")
                                             {
                                                domain_double[ValueIndex]=Value;
                                                return(true);
                                             }
                                             else //if(string(typename(T))=="int")
                                             {
                                                domain_int[ValueIndex]=Value;
                                                return(true);
                                             }
                                          }
                                       }
                                       //
                                       return(false);
                                    };

                                    template <typename T>
      bool                          Get(int DomainIndex,CDomain<T> &D)
                                    {
                                       if(DomainIndex>=0 && DomainIndex<Domain(string(typename(T))))
                                       {
                                          if(string(typename(T))=="datetime")
                                          {
                                             D=domain_datetime[DomainIndex];

                                             return(true);
                                          }
                                          else if(string(typename(T))=="string")
                                          {
                                             D=domain_string[DomainIndex];

                                             return(true);
                                          }
                                          else if(string(typename(T))=="double")
                                          {
                                             D=domain_double[DomainIndex];

                                             return(true);
                                          }
                                          else //if(string(typename(T))=="int")
                                          {
                                             D=domain_int[DomainIndex];

                                             return(true);
                                          }
                                       }

                                       return(false);
                                    };

      bool                          Set(int ValueIndex,COntology &Value)
                                    {
                                       if
                                       (
                                       ValueIndex>=0 && ValueIndex<Ontology()
                                       )
                                       {
                                          ontology[ValueIndex]=Value;
                                          return(true);
                                       }
                                       else if(ValueIndex>=Ontology())
                                       {
                                          if(Ontology(Ontology()+1))
                                          {
                                             ontology[Ontology()-1]=Value;
                                             return(true);
                                          }
                                       }
                                       //
                                       return(false);
                                    };

      bool                          Get(int OntologyIndex,COntology &O)
                                    {
                                       if(OntologyIndex>=0 && OntologyIndex<Ontology())
                                       {
                                          O=ontology[OntologyIndex];

                                          return(true);
                                       }

                                       return(false);
                                    };


                                    template <typename T>
      int                           Index(CDomain<T> &Value)
                                    {
                                       int _index=-1;
                                       //
                                       for(int d=0; d<Domain(string(typename(T))); d++)
                                       {
                                          if(string(typename(T))=="string")
                                          {
                                             if(DomainMatch(Value,domain_string[d]))
                                             {
                                                _index=d; break;
                                             }
                                          }
                                          else if(string(typename(T))=="datetime")
                                          {
                                             if(DomainMatch(Value,domain_int[d]))
                                             {
                                                _index=d; break;
                                             }
                                          }
                                          else if(string(typename(T))=="double")
                                          {
                                             if(DomainMatch(Value,domain_double[d]))
                                             {
                                                _index=d; break;
                                             }
                                          }
                                          else if(string(typename(T))=="int")
                                          {
                                             if(DomainMatch(Value,domain_int[d]))
                                             {
                                                _index=d; break;
                                             }
                                          }
                                       }

                                       return(_index);
                                    }


      int                           Index(COntology &Value)
                                    {
                                       int _index=-1;
                                       //
                                       for(int o=0; o<Ontology(); o++)
                                       {
                                          if(!OntologyMatch(Value,ontology[o]))
                                          {
                                             _index=o; break;
                                          }
                                       }

                                       return(_index);
                                    }

                                    CCategory()
                                    {
                                       domains_datetime=0;
                                       domains_string=0;
                                       domains_double=0;
                                       domains_int=0;

                                       ontologies=0;
                                    };
                                    ~CCategory()
                                    {
                                    };
   };
```

Notice the 'enumerated' data types that are allowed. I will try to make this neater by integrating them as an object into a single array in future articles. A 'FillDomain' function, besides the one in previous article that populated a domain with natural numbers, has also been added.

```
//+------------------------------------------------------------------+
//| Fill Domain(Set) with one-cardinal elements from input E array.  |
//+------------------------------------------------------------------+
template <typename TD,typename TE>
void FillDomain(CDomain<TD> &D,CElement<TE> &E[])
   {
      if(string(typename(TD))!=string(typename(TE)))
      {
         return;
      }

      int _cardinal=ArraySize(E);
      //
      if(_cardinal<0||INT_MAX<=_cardinal)
      {
         return;
      }

      //Set its cardinal to input array size
      if(D.Cardinality(_cardinal))
      {
         for(int c=0;c<_cardinal;c++)
         {
            D.Set(c,E[c],true);
         }
      }
   }
```

So to check for association, we'll create a category '\_ca'. We then declare 3 simple sets of string type and these are each filled with a clothing type. We copy these sets (arrays) into new element arrays ('\_et', '\_ep', '\_es') then add each of these elements to its own domain('\_dt', '\_dp', '\_ds'). Having done that we set the number of domains to our category to 3 and then proceed to set the domain at each index with the newly created domains that were filled with the clothing type elements.

```
      //ASSOCIATION
      CCategory _ca;

      string _tops[__EA]={"T-shirt","button-up","polo","sweatshirt","tank top"};          //domain 0
      string _pants[__EA]={"jeans","slacks","khakis","sweatpants","shorts"};              //domain 1

      string _shoes[__EA]={"sneakers","dress shoes","loafers","running shoes","sandals"}; //domain 2

      CElement<string> _et[];ArrayResize(_et,__EA);
      CElement<string> _ep[];ArrayResize(_ep,__EA);
      CElement<string> _es[];ArrayResize(_es,__EA);

      for(int e=0;e<__EA;e++)
      {
         _et[e].Cardinality(1); _et[e].Set(0,_tops[e]);
         _ep[e].Cardinality(1); _ep[e].Set(0,_pants[e]);
         _es[e].Cardinality(1); _es[e].Set(0,_shoes[e]);
      }

      CDomain<string> _dt,_dp,_ds;
      FillDomain(_dt,_et);FillDomain(_dp,_ep);FillDomain(_ds,_es);

      //
      if(_ca.Domain("string",__DA))//resize domains array to 3
      {
         if(_ca.Set(0,_dt) && _ca.Set(1,_dp) && _ca.Set(2,_ds))//assign each filled domain above to a spot (index) within the category
         {
            if(_ca.Domain("string")==__DA)//check domains count
            {
               for(int e=0;e<__EA;e++)
               {
                  COntology _o_01_2;
                  CMorphism<string,string> _m1_01_2,_m2_01_2;
                  SetCategory(_ca,0,1,e,e,_o_01_2," is worn with ",_m1_01_2);
                  SetCategory(_ca,1,2,e,e,_o_01_2," is worn with ",_m2_01_2,ONTOLOGY_POST);printf(__FUNCSIG__+" (0 & 1) followed by 2 Log is: "+_o_01_2.ontology);

                  COntology _o_0_12;
                  CMorphism<string,string> _m1_0_12,_m2_0_12;
                  SetCategory(_ca,1,2,e,e,_o_0_12," is worn with ",_m1_0_12);
                  SetCategory(_ca,0,2,e,e,_o_0_12," is worn with ",_m2_0_12,ONTOLOGY_PRE);printf(__FUNCSIG__+" 0 following (1 & 2) Log is: "+_o_0_12.ontology);
               }
            }
         }
      }
```

To check for associativity we will output the results of morphisms in ontology logs. For this we will create a class 'COntology'. This is outlined below with an enumeration and struct of the same name.

```
//+------------------------------------------------------------------+
//| ONTOLOGY ENUM                                                    |
//+------------------------------------------------------------------+
enum EOntology
  {
      ONTOLOGY_PRE=-1,
      ONTOLOGY_NEW=0,
      ONTOLOGY_POST=1
  };
//+------------------------------------------------------------------+
//| ONTOLOGY STRUCT                                                  |
//+------------------------------------------------------------------+
struct SOntology
  {
      int                           in;
      int                           out;

                                    SOntology()
                                    {
                                       in=-1;
                                       out=-1;
                                    };
                                    ~SOntology(){};
  };
//+------------------------------------------------------------------+
//| ONTOLOGY CLASS                                                   |
//+------------------------------------------------------------------+
class COntology
  {
      protected:

      int                           facts;

      SOntology                     types[];
      SOntology                     universe[];

      public:

      string                        ontology;

      int                           Facts() { return(facts); }
      bool                          Facts(int Value) { if(Value>=0 && Value<INT_MAX) { facts=Value; ArrayResize(types,facts); ArrayResize(universe,facts); return(true); } return(false); }

      bool                          GetType(int TypeIndex,int &TypeIn,int &TypeOut) { if(TypeIndex>=0 && TypeIndex<Facts()) { TypeIn=types[TypeIndex].in; TypeOut=types[TypeIndex].out; return(true); } return(false); }
      bool                          SetType(int ValueIndex,int ValueIn,int ValueOut)
                                    {
                                       if(ValueIndex>=0 && ValueIndex<Facts())
                                       {
                                          types[ValueIndex].in=ValueIn; types[ValueIndex].out=ValueOut;
                                          return(true);
                                       }
                                       else if(ValueIndex>=0 && ValueIndex>=Facts() && ValueIndex<INT_MAX-1)
                                       {
                                          if(Facts(ValueIndex+1))
                                          {
                                             types[ValueIndex].in=ValueIn; types[ValueIndex].out=ValueOut;
                                             return(true);
                                          }
                                       }

                                       return(false);
                                    }

      bool                          GetUniverse(int UniverseIndex,int &UniverseIn,int &UniverseOut) { if(UniverseIndex>=0 && UniverseIndex<Facts()) { UniverseIn=universe[UniverseIndex].in; UniverseOut=universe[UniverseIndex].out; return(true); } return(false); }
      bool                          SetUniverse(int ValueIndex,int ValueIn,int ValueOut)
                                    {
                                       if(ValueIndex>=0 && ValueIndex<Facts())
                                       {
                                          universe[ValueIndex].in=ValueIn; universe[ValueIndex].out=ValueOut;
                                          return(true);
                                       }
                                       else if(ValueIndex>=0 && ValueIndex>=Facts() && ValueIndex<INT_MAX-1)
                                       {
                                          if(Facts(ValueIndex+1))
                                          {
                                             universe[ValueIndex].in=ValueIn; universe[ValueIndex].out=ValueOut;
                                             return(true);
                                          }
                                       }

                                       return(false);
                                    }

      string                        old_hash;
      string                        new_hash;

                                    COntology()
                                    {
                                       ontology="";

                                       facts=0;

                                       ArrayResize(types,facts);
                                       ArrayResize(universe,facts);

                                       old_hash="";
                                       new_hash="";
                                    };
                                    ~COntology(){};
  };
```

From the ontology output, running the script should give us the logs below.

```
2023.01.26 10:42:56.911 ct_2 (EURGBP.ln,H1)     void OnStart() (0 & 1) followed by 2 Log is: T-shirt is worn with jeans is worn with sneakers
2023.01.26 10:42:56.912 ct_2 (EURGBP.ln,H1)     void OnStart() 0 following (1 & 2) Log is: T-shirt is worn with jeans is worn with sneakers
2023.01.26 10:42:56.912 ct_2 (EURGBP.ln,H1)     void OnStart() (0 & 1) followed by 2 Log is: button-up is worn with slacks is worn with dress shoes
2023.01.26 10:42:56.912 ct_2 (EURGBP.ln,H1)     void OnStart() 0 following (1 & 2) Log is: button-up is worn with slacks is worn with dress shoes
2023.01.26 10:42:56.912 ct_2 (EURGBP.ln,H1)     void OnStart() (0 & 1) followed by 2 Log is: polo is worn with khakis is worn with loafers
2023.01.26 10:42:56.912 ct_2 (EURGBP.ln,H1)     void OnStart() 0 following (1 & 2) Log is: polo is worn with khakis is worn with loafers
2023.01.26 10:42:56.912 ct_2 (EURGBP.ln,H1)     void OnStart() (0 & 1) followed by 2 Log is: sweatshirt is worn with sweatpants is worn with running shoes
2023.01.26 10:42:56.912 ct_2 (EURGBP.ln,H1)     void OnStart() 0 following (1 & 2) Log is: sweatshirt is worn with sweatpants is worn with running shoes
2023.01.26 10:42:56.912 ct_2 (EURGBP.ln,H1)     void OnStart() (0 & 1) followed by 2 Log is: tank top is worn with shorts is worn with sandals
2023.01.26 10:42:56.913 ct_2 (EURGBP.ln,H1)     void OnStart() 0 following (1 & 2) Log is: tank top is worn with shorts is worn with sandals
```

In summary, the axiom of association allows for easy definition of a category by allowing the composition of morphisms to be associative, which eliminates the need to compute parentheses when defining multiple morphisms. This makes it simpler to understand the relationships between objects in a category, in this case clothing items in all three domains; shirts, pants and shoes.

### Commutative Diagrams

A [commutative diagram](https://en.wikipedia.org/wiki/Commutative_diagram "https://en.wikipedia.org/wiki/Commutative_diagram") is a diagram that represents the relationship between two morphisms in a category. It consists of a set of domains, morphisms, and their compositions, arranged in a specific way to show that the morphisms commute, meaning that the order in which they are composed does not matter.This implies that for any three domains A, B, and C in a category, and for their morphisms f: A -> B, g: B -> C, and h: A -> C; the composition of these morphisms must satisfy:

> f o g = h

An example of commutation using forex prices could be arbitrage; the relationship between the price of say EURUSD, EURJPY, and USDJPY. In this example, The domains could be the currencies themselves (EUR, USD, JPY) while the morphisms could represent the process of converting one currency to another based on the exchange rate. For instance, let f: EUR -> USD be the morphism that represents the process of converting EUR to USD at the exchange rate of EURUSD; let g: USD -> JPY be the morphism that represents the process of converting USD to JPY at the exchange rate of USDJPY; and let h: EUR -> JPY be the morphism that represents converting EUR to JPY. Following commutation rule would above: f o g = h, this would translate to:

converting EUR to USD at the EURUSD rate, then converting that USD to JPY at the USDJPY rate = converting EUR to JPY at the EURJPY rate.

Let's illustrate this with MQL5. We'll declare a category '\_cc' and populate it with 3 domains each with 2 currencies. The domains can be thought of as portfolios in asset allocation. Like with the association axiom we will fill and check our category values using ontology logs as below.

```
      //COMMUTATION
      CCategory _cc;

      string _a[__EC]={"EUR","GBP"};       //domain 0
      string _b[__EC]={"USD","CAD"};       //domain 1
      string _c[__EC]={"CHF","JPY"};       //domain 2

      CElement<string> _e_a[];ArrayResize(_e_a,__EC);
      CElement<string> _e_b[];ArrayResize(_e_b,__EC);
      CElement<string> _e_c[];ArrayResize(_e_c,__EC);

      for(int e=0;e<__EC;e++)
      {
         _e_a[e].Cardinality(1); _e_a[e].Set(0,_a[e]);
         _e_b[e].Cardinality(1); _e_b[e].Set(0,_b[e]);
         _e_c[e].Cardinality(1); _e_c[e].Set(0,_c[e]);
      }

      CDomain<string> _d_a,_d_b,_d_c;
      FillDomain(_d_a,_e_a);FillDomain(_d_b,_e_b);FillDomain(_d_c,_e_c);

      //
      if(_cc.Domain("string",__DC))//resize domains array to 3
      {
         if(_cc.Set(0,_d_a) && _cc.Set(1,_d_b) && _cc.Set(2,_d_c))//assign each filled domain above to a spot (index) within the category
         {
            if(_cc.Domain("string")==__DC)//check domains count
            {
               for(int e=0;e<__EC;e++)
               {
                  COntology _o_ab_bc;
                  string _ab=_a[e]+_b[e],_bc=_b[e]+_c[e];
                  double _ab_bid=SymbolInfoDouble(_ab,SYMBOL_BID),_bc_bid=SymbolInfoDouble(_bc,SYMBOL_BID);
                  string _aspect_ab=" is exchanged at: "+DoubleToString(_ab_bid,(int)SymbolInfoInteger(_ab,SYMBOL_DIGITS))+", for: ";
                  string _aspect_bc=" is exchanged at: "+DoubleToString(_bc_bid,(int)SymbolInfoInteger(_bc,SYMBOL_DIGITS))+", for: ";
                  CMorphism<string,string> _m_ab,_m_bc;
                  SetCategory(_cc,0,1,e,e,_o_ab_bc,_aspect_ab,_m_ab);
                  SetCategory(_cc,1,2,e,e,_o_ab_bc,_aspect_bc,_m_bc,ONTOLOGY_POST);printf(__FUNCSIG__+" a to b then b to c logs: "+_o_ab_bc.ontology);

                  COntology _o_ac;
                  string _ac=_a[e]+_c[e];
                  string _aspect_ac=" is exchanged at: "+DoubleToString(SymbolInfoDouble(_ac,SYMBOL_BID),(int)SymbolInfoInteger(_ac,SYMBOL_DIGITS))+", for: ";
                  CMorphism<string,string> _m_ac;
                  SetCategory(_cc,0,2,e,e,_o_ac,_aspect_ac,_m_ac);printf(__FUNCSIG__+" a to c logs: "+_o_ac.ontology+" vs product of bid rate for ab and bc of: "+DoubleToString(_ab_bid*_bc_bid,(int)SymbolInfoInteger(_ac,SYMBOL_DIGITS)));//ontology
               }
            }
         }
      }
```

We check for commutation by comparing the rate from the commute morphism '\_m\_ac' with the product of the two rates got from the morphisms '\_m\_ab', and '\_m\_bc'. There are discrepancies but these are mostly due to quality of history broker data and account of the spread (only bids are used). Running this script should give us these logs.

```
2023.01.26 17:27:19.093 ct_2 (EURGBP.ln,H1)     void OnStart() a to b then b to c logs: EUR is exchanged at: 1.08966, for: USD is exchanged at: 0.91723, for: CHF
2023.01.26 17:27:19.093 ct_2 (EURGBP.ln,H1)     void OnStart() a to c logs: EUR is exchanged at: 0.99945, for: CHF vs product of bid rate for ab and bc of: 0.99947
2023.01.26 17:27:19.093 ct_2 (EURGBP.ln,H1)     void OnStart() a to b then b to c logs: GBP is exchanged at: 1.65097, for: CAD is exchanged at: 97.663, for: JPY
2023.01.26 17:27:19.093 ct_2 (EURGBP.ln,H1)     void OnStart() a to c logs: GBP is exchanged at: 161.250, for: JPY vs product of bid rate for ab and bc of: 161.239
```

### Ontology Logs

Ontologies are _subjective summaries of a category_. They serve an important tool in category theory because they provide a way to _interpret_ the structure and relationships of the domains via morphisms in a given category. Put differently they are a mathematical structure (or data model, equivalent to a database schema) that describes the types of domains and their morphisms in the domain. There are several types of ontologies, these include:

- Taxonomic ontologies describe a set of objects and their sub-classes, and define the relationships between them in terms of "is-a" relationships.
- Hierarchical ontologies describe a set of objects and their properties, and define the relationships between them in terms of "part-of" relationships.
- Relational ontologies describe a set of objects and their relationships, and define the relationships between them in terms of arbitrary binary relationships.

For a forex trading system, a relational ontology may be most appropriate. This would allow for the modelling of various types of trading strategies and the relationships between them, such as which strategies are most effective under certain market conditions. Additionally, relational ontologies can be easily extended to include additional information such as historical data, which could be useful in developing a trading system.

Relational ontologies in action when used across domains in category theory can be demonstrated using various examples in the financial markets. Here are five examples:

1. Currency Pairs: The forex market is comprised of currency pairs, which are sets of two currencies that are traded against each other. The relationship between these currencies can be modelled using a relational ontology, where the currency pairs are considered as domains and the exchange rate is considered as the morphism that connects them. An example of this has already been shared above with the commutative diagram.
2. Time Series Data: Forex prices are often represented as time series data, which can be modelled using a relational ontology where the time series data is considered as a domain and the price changes over time are considered as morphisms.
3. Technical Indicators: Technical indicators, such as moving averages or relative strength indices, can be used to analyse forex prices. These indicators can be modelled using a relational ontology where the indicator is considered as a domain and the indicator values are considered as morphisms.
4. Trading Strategies: Traders often use various trading strategies, such as trend-following or momentum-based strategies. These strategies can be modelled using a relational ontology where the trading strategy is considered as a domain and the trades that are executed based on the strategy are considered as morphisms.
5. Order Book: The order book is a record of all buy and sell orders that have been placed in the forex market. This can be modelled (if provided by a broker) using a relational ontology where the order book is considered as a domain and the orders that are placed are considered as morphisms.

Overall, relational ontologies are useful in category theory when used across domains in trading because they allow for the modelling and analysis of complex systems and relationships in a clear and structured way. Ontologies employ 'types' and 'aspects' in presenting a subjective view of a category. A type can be thought of as a group of domains within a particular category. It can be simple or compound. When simple it contains just the one domain. When compound it brings together more than one domain. Aspects are to types what morphisms are to domains within a category.  A relational ontology based on currency pairs in the foreign exchange (forex) market can be modelled as follows:

Types:

1. Currency Pair: The currency pair is the main type in this ontology, representing the two currencies that are being traded against each other. For example, the currency pair "EUR/USD" represents the exchange rate between the Euro and the US Dollar.
2. Exchange Rate: The exchange rate is the rate at which one currency can be exchanged for another. It represents the morphism between the currency pair objects. For example, the exchange rate for the EUR/USD currency pair might be 1.20, meaning 1 Euro can be exchanged for 1.20 US Dollars.
3. Time: Time is a type that can be used to represent the time at which the exchange rate applies. It can be used to connect exchange rate morphism with time series data.

Aspects:

1. Historical Exchange Rates: Historical exchange rates are the exchange rates that have been recorded at different points in time. They can be used to analyze trends and patterns in the forex market.
2. Volatility: Volatility is a measure of how much the exchange rate changes over time. It can be used to assess the risk associated with a particular currency pair.
3. Correlation: Correlation is a measure of the relationship between two currency pairs. It can be used to identify opportunities for diversification or hedging in the forex market.

Overall, this relational ontology allows for a clear and structured representation of the relationships between currency pairs, exchange rates, time and other aspects in the forex market. It can be used to analyse  the market for dynamics and make informed decisions. [Grothendieck universes](https://en.wikipedia.org/wiki/Grothendieck_universe "https://en.wikipedia.org/wiki/Grothendieck_universe") are used to define the concept of a universe of types, which is key within category theory ontologies, a universe of types is a set of types within a given category. It allows for sorting of the types and addresses the problem of "sets of sets".

To see ontologies in action lets look at an example in the financial industry. Different entities, such as the Securities and Exchange Commission (SEC) and a hedge fund, may use ontologies to organise and categorise financial data, in the same category, in different ways. This can lead to different types and aspects of the data being represented in the ontologies used by each entity. From the perspective of the SEC, an ontology might be used to organise data related to securities and trading activities, with a focus on compliance and regulatory issues. This could include types  such as "securities," "trading activity," "insider trading," and "violations of securities laws." The ontology might also include aspects related to compliance and enforcement, such as "investigations" and "penalties." From the perspective of a hedge fund, an ontology, on the same category, might be used to organise data related to investment strategies, portfolio management, and performance metrics. This could include types such as "investment strategies," "portfolio management," "performance metrics," and "risk management." The ontology might also include aspects related to the fund's operations and management, such as "assets under management" and "fund managers." As we can see, while both entities may be using ontologies to organise financial data, the types and aspects represented in their ontologies are different, reflecting the different goals, perspectives, and concerns of the SEC versus a hedge fund. So, the SEC's ontology would be focused on the compliance, regulations, and violations in the securities trading, while the hedge fund's ontology would be focused on investments, risk management, and portfolio management.

Moving on to illustrate types further withIn forex, simple and compound types refer to the types of data that can be used to represent different aspects of the market. Simple types are basic data types that can be represented by a single value. For example, in forex, the exchange rate between two currencies can be considered a simple type, represented by a single value such as 1.20 (meaning 1 unit of the base currency can be exchanged for 1.20 units of the quote currency). Compound types, on the other hand, are data types that are composed of multiple values or other data types. For example, a candle stick chart can be considered a compound type, as it is composed of multiple values such as the opening price, closing price, highest price, and lowest price for a given time period. In terms of aspects, simple aspects are the ones that connect two simple types, for example, the exchange rate between two currencies can be considered a simple aspect. Compound aspects are the ones that connect two compound types or a compound type and a simple type, for example, a trading strategy can be considered a compound aspect as it connects the strategy with the trades that are executed based on it. Overall, simple types and aspects are basic building blocks that can be used to represent basic pieces of information in ontologies, while compound types and aspects are more complex data types that are composed of multiple values or other data types, that can be used to represent more complex aspects of the system being modelled.

Ontology-logs are a way of organising and representing knowledge by providing a logical structure to the domain of interest. They help to define concepts, properties, and relationships within a specific domain, sub-domain, task, instance or process. An ontology-log is a formal representation of a set of concepts, properties, and relationships that exist within a specific domain. It is used to define the structure of the knowledge within that domain and to provide a consistent and logical representation of the knowledge. This makes it easier for machines and humans to understand and use the information within the domain. Ontology-logs can be used in a variety of fields, such as artificial intelligence, natural language processing, knowledge representation, and information science. They are commonly used in conjunction with other types of ontologies, such as upper ontologies and domain ontologies, to provide a comprehensive view of a domain. Ontology-logs are typically represented in a machine-readable format, such as OWL (Web Ontology Language) or RDF (Resource Description Framework), which allows them to be easily processed and used by machines. They are also human-readable, which makes it easier for people to understand and use the information represented in the ontology log.

In the context of category theory, ontology-logs are used to organise and represent knowledge by providing a logical structure to the domain of interest. They help to define concepts, properties, and relationships within a specific domain, sub-domain, task, instance or process. Category theory is a branch of mathematics that deals with the structure of mathematical systems, and it provides a framework for modelling the relationships between objects and morphisms. Ontology logs, when used in the context of category theory, provide a way to organise and represent the knowledge within a domain in a structured and logical way.

There are several types of ontology-logs that are commonly used in category theory, such as:

1. [Upper ontology](https://en.wikipedia.org/wiki/Upper_ontology "https://en.wikipedia.org/wiki/Upper_ontology"): An upper ontology is a general ontology that provides a high-level view of a domain, and defines the most general concepts and relationships that apply across multiple sub-domains. It is a common way to provide a unifying structure across different ontologies.
2. [Domain ontology](https://en.wikipedia.org/wiki/Domain_model "https://en.wikipedia.org/wiki/Domain_model"): A domain ontology is a specific ontology that focuses on a particular sub-domain within a larger domain. It defines the concepts and relationships specific to that sub-domain.
3. Task ontology: A task ontology is a specific ontology that focuses on a particular task or problem within a domain. It defines the concepts and relationships specific to that task or problem.
4. Instance ontology: An instance ontology is a specific ontology that focuses on the instances of a particular concept or class. It defines the properties and relationships specific to those instances.
5. [Process ontology](https://en.wikipedia.org/wiki/Process_ontology "https://en.wikipedia.org/wiki/Process_ontology"): A process ontology is a specific ontology that focuses on the processes or actions that occur within a domain. It defines the concepts and relationships specific to those processes or actions.

These types of ontology-logs are used to structure the knowledge and relationships of a domain and to facilitate the understanding, representation and manipulation of the information. They help to provide a consistent and logical representation of the knowledge within a domain, and can be used in conjunction with relational ontologies to provide a comprehensive view of a domain.

A process ontology-log is a specific type of ontology-log that focuses on the processes or actions that occur within a domain. It defines the concepts and relationships specific to those processes or actions. As such, the types of aspects that can be used in a process ontology-log will depend on the specific domain and the processes or actions being modelled. Here are a few examples of aspects that could be used in a process ontology-log:

1. Inputs and Outputs: The input and output aspects of a process can be used to define the resources, materials, or information that are required to initiate the process and the final products, services or information generated by the process.
2. Steps or Phases: The process can be broken down into smaller steps or phases and each step can be defined as a concept, representing the actions and objectives of that step. The relationships between the steps can also be modelled.
3. Actors: Actors are the entities or agents involved in the process, such as people, machines or organizations. Relationships between actors and their roles in the process can be defined.
4. Constraints: Constraints are the rules, regulations or limitations that the process must adhere to. This aspect can be used to define the conditions and requirements that must be met for the process to be completed successfully.
5. Metrics: Metrics are the measurements or indicators that are used to evaluate the performance of the process, such as efficiency, quality, or cost. This aspect can be used to define the measurements and indicators used to evaluate the process, and how they are calculated.
6. Temporal aspects: Temporal aspects refer to the timing of the process, such as the start and end time, duration, and frequency of the process. This aspect can be used to model the temporal aspects of the process.

Overall, a process ontology-log can be used to model the various aspects of a process, including inputs and outputs, steps or phases, actors, constraints, metrics, and temporal aspects, and the relationships between them, in a structured and logical way, which can facilitate the understanding, representation and manipulation of the information. An example of a process ontology in the category of foreign exchange (forex) market could be the process of executing a trade. The process ontology would define the concepts and relationships specific to the process of executing a trade. Here is an example of how the aspects of this process ontology could be modelled:

1. Inputs and Outputs: The inputs of the process include the trading strategy, the currency pair to be traded, and the trading account balance. The output of the process is the executed trade.
2. Steps or Phases: The process can be broken down into smaller steps or phases, such as selecting the currency pair, setting the trade parameters, placing the order and monitoring the trade.
3. Actors: Actors involved in the process include the trader, the trading platform and the market. The trader is the main actor who initiates the process, the trading platform is the tool used to execute the trade and the market is the environment where the trade takes place.
4. Constraints: Constraints include regulations on margin, leverage and risk management. These constraints must be adhered to in order for the trade to be completed successfully.
5. Metrics: Metrics used to evaluate the performance of the process include the profit or loss on the trade, the risk-reward ratio, and the percentage of successful trades.
6. Temporal aspects: Temporal aspects refer to the timing of the process, such as the start and end time of the trade, the duration of the trade and the frequency of the trades.

This process ontology-log can be used to model the various aspects of the process of executing a trade in the forex market in a structured and logical way. It can be used to facilitate the understanding, representation and manipulation of the information related to the process of executing a trade, and it can be used in conjunction with other types of ontologies to provide a comprehensive view of financial markets.

### Conclusion

In conclusion, category theory provides a powerful framework for modelling and analysing complex systems, such as the financial markets. The implementation of category theory in MQL5 can help traders to better understand and navigate the market by providing a structured and logical representation of the relationships between domains and morphisms. The article has highlighted the importance of the axiom definitions of a category in category theory and how they provide a foundation for modelling the relationships between domains and morphisms. The article also discussed the concept of ontology logs and how they can be used to provide a logical structure to a subject of interest, in this case the financial markets. Overall, the implementation of category theory in MQL5 can be a valuable tool for traders looking to gain a deeper understanding of the market and make more informed trading decisions. It provides a structured and logical approach to analysing the market, which can help traders to identify patterns and opportunities that may otherwise be difficult to detect.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11958.zip "Download all attachments in the single ZIP archive")

[ct\_2.mq5](https://www.mql5.com/en/articles/download/11958/ct_2.mq5 "Download ct_2.mq5")(62.16 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/440816)**

![DoEasy. Controls (Part 29): ScrollBar auxiliary control](https://c.mql5.com/2/50/MQL5-avatar-doeasy-library-2__6.png)[DoEasy. Controls (Part 29): ScrollBar auxiliary control](https://www.mql5.com/en/articles/11847)

In this article, I will start developing the ScrollBar auxiliary control element and its derivative objects — vertical and horizontal scrollbars. A scrollbar is used to scroll the content of the form if it goes beyond the container. Scrollbars are usually located at the bottom and to the right of the form. The horizontal one at the bottom scrolls content left and right, while the vertical one scrolls up and down.

![Develop a Proof-of-Concept DLL with C++ multi-threading support for MetaTrader 5 on Linux](https://c.mql5.com/2/51/proof-of-concept-dll-avatar.png)[Develop a Proof-of-Concept DLL with C++ multi-threading support for MetaTrader 5 on Linux](https://www.mql5.com/en/articles/12042)

We will begin the journey to explore the steps and workflow on how to base development for MetaTrader 5 platform solely on Linux system in which the final product works seamlessly on both Windows and Linux system. We will get to know Wine, and Mingw; both are the essential tools to make cross-platform development works. Especially Mingw for its threading implementations (POSIX, and Win32) that we need to consider in choosing which one to go with. We then build a proof-of-concept DLL and consume it in MQL5 code, finally compare the performance of both threading implementations. All for your foundation to expand further on your own. You should be comfortable building MT related tools on Linux after reading this article.

![Population optimization algorithms: Fish School Search (FSS)](https://c.mql5.com/2/50/Fish_School_avatar.png)[Population optimization algorithms: Fish School Search (FSS)](https://www.mql5.com/en/articles/11841)

Fish School Search (FSS) is a new optimization algorithm inspired by the behavior of fish in a school, most of which (up to 80%) swim in an organized community of relatives. It has been proven that fish aggregations play an important role in the efficiency of foraging and protection from predators.

![MQL5 Cookbook — Services](https://c.mql5.com/2/50/mql5-recipes-Services.png)[MQL5 Cookbook — Services](https://www.mql5.com/en/articles/11826)

The article describes the versatile capabilities of services — MQL5 programs that do not require binding graphs. I will also highlight the differences of services from other MQL5 programs and emphasize the nuances of the developer's work with services. As examples, the reader is offered various tasks covering a wide range of functionality that can be implemented as a service.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ygpqgistryltazdgtsozqjkxqppcgyxy&ssn=1769191924296116697&ssn_dr=0&ssn_sr=0&fv_date=1769191924&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11958&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Category%20Theory%20in%20MQL5%20(Part%202)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919192456484669&fz_uniq=5071650086733163507&sv=2552)

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