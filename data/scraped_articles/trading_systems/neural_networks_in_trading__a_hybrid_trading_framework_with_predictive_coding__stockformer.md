---
title: Neural Networks in Trading: A Hybrid Trading Framework with Predictive Coding (StockFormer)
url: https://www.mql5.com/en/articles/16686
categories: Trading Systems, Expert Advisors, Machine Learning
relevance_score: 6
scraped_at: 2026-01-23T11:33:26.968582
---

[![](https://www.mql5.com/ff/sh/7h2yc16rtqsn2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Market analysis\\
\\
Dozens of channels, thousands of subscribers and daily updates. Learn more about trading.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=glufvbpblsoxonicqfngsyuzwfebnilr&s=103cc3ab372a16872ca1698fc86368ffe3b3eaa21b59b4006d5c6c10f48ad545&uid=&ref=https://www.mql5.com/en/articles/16686&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062551597948511317)

MetaTrader 5 / Trading systems


### Introduction

Reinforcement Learning ( _RL_) is increasingly being applied to complex problems in finance, including the development of trading strategies and portfolio management. Models are trained to analyze historical data on asset price movements, trading volumes, and technical indicators. However, most existing methods assume that the analyzed data fully capture all interdependencies between assets. In practice, this is rarely the case, especially in noisy and highly volatile market environments.

Traditional approaches often fail to account for both short- and long-term return forecasts as well as correlations across assets. Yet, successful investment strategies typically rely on a deep understanding of these factors. To address this, the paper " _[StockFormer: Learning Hybrid Trading Machines with Predictive Coding](https://www.mql5.com/go?link=https://www.ijcai.org/proceedings/2023/0530.pdf "https://www.ijcai.org/proceedings/2023/0530.pdf")_" introduces _StockFormer_, a hybrid trading system that combines _predictive_ coding with the flexibility of _RL_ agents. Predictive coding that is widely used in natural language processing and computer vision, enables the extraction of informative hidden states from noisy input data, a capability that is particularly valuable in financial applications.

The _StockFormer_ integrates three modified _Transformer_ branches, each responsible for capturing different aspects of market dynamics:

- Long-term trends
- Short-term trends
- Cross-asset dependencies

Each branch incorporates a _Diversified Multi-Head Attention_ ( _DMH-Attn_) mechanism, which extends the vanilla _Transformer_ by replacing a single _FeedForward_ block with multiple parallel blocks. This enables the model to capture diverse temporal patterns across subspaces while preserving critical information.

To optimize trading strategies, the latent states produced by these three branches are adaptively fused using multi-head attention into a unified state space, which is then used by the _RL_ agent.

Policy learning is carried out using the _[Actor–Critic](https://www.mql5.com/en/articles/12941)_ method. Crucially, gradient feedback from the _Critic_ is propagated back to improve the predictive coding module, ensuring tight integration between predictive modeling and policy optimization.

Experiments conducted on three public datasets demonstrated that _StockFormer_ significantly outperforms existing methods in both predictive accuracy and investment returns.

### The StockFormer algorithm

_StockFormer_ addresses forecasting and trading decision-making in financial markets through _RL_. A key limitation of conventional methods lies in their inability to effectively model dynamic dependencies between assets and their future trends. This is especially important in markets where conditions change rapidly and unpredictably. _StockFormer_ resolves this challenge through two core stages: _predictive_ coding and _trading strategy learning_.

In the first stage, _StockFormer_ leverages self-supervised learning to extract hidden patterns from noisy market data. This allows the model to capture short- and long-term dynamics as well as cross-asset dependencies. Using this approach, the model extracts important hidden states, which are then used in the next step to make trading decisions.

Financial markets exhibit highly diverse temporal patterns across multiple assets, complicating the extraction of effective representations from raw data. To address this, _StockFormer_ modifies the vanilla _Transformer's_ multi-head attention mechanism by replacing the single _FeedForward_ network ( _FFN_) with a group of parallel FFNs. Without increasing the parameter count, this design strengthens the ability of multi-head attention to decompose features, thereby improving the modeling of heterogeneous temporal patterns across subspaces.

This enhanced module is called _Diversified Multi-Head Attention_ ( _DMH-Attn_). For _Query_, _Key_, and _Value_ entities of dimension _dmodel_, the process begins by splitting the output features _Z_ of multi-head attention into _h_ groups along the channel dimension, where _h_ is the number of attention heads. A dedicated _FFN_ is then applied to each group in _Z_:

![](https://c.mql5.com/2/169/3029561935185__2.png)

Here _MH-Attn_ indicates multi-head attention. 𝑓𝑖 are output features of each _FFN_ head, containing two linear projections with _ReLU_ activation between them.

Each branch in the modernized _Transformer_ in _StockFormer_ is divided into two modules: an encoder and a decoder. Both are used during predictive coding training, but only the encoder is employed during strategy optimization. The model comprises _L_ encoder layers and _M_ decoder layers. The final encoder output _XLenc_ is fed into each decoder layer. The process of calculations on the _l_-th encoder layer and _m_-th decoder layer can be written as follows:

- encoder layer:

![](https://c.mql5.com/2/169/5930258664114__2.png)

- decoder layer:

![](https://c.mql5.com/2/169/1737851567560__2.png)

Here _Xl,enc_ and _Xm,dec_ are encoder and decoder outputs, respectively. Inputs to the first encoder and decoder layers consist of raw data with positional embeddings. The final decoder output is passed through a projection layer to generate predictive coding results.

The cross-asset dependency module identifies dynamic correlations across time series. At each time step _t_, it processes identical inputs in both encoder and decoder. For stock market data, the framework authors used technical indicators such as _MACD_, _RSI_, and _SMA_.

During training, data are split into two parts:

1. _Covariance matrix._ The covariance matrix is computed across daily closing prices of all assets over a fixed window before time t.
2. _Masked statistics._ This part contains half of the time series are randomly masked with zeros, while the remainder serve as visible features. At test time, we use full (unmasked) data.

The goal is to reconstruct masked statistics based on the covariance matrix and the remaining features. This predictive coding task forces the _Transformer_ encoder to learn interdependencies across assets.

Short-term and long-term forecasting modules in _StockFormer_ aim at predicting the return rates for each asset over different time horizons.

The short-term forecasting module predicts next-day returns for the asset ( _H_ = 1). For this, we feed analyzed statistics for _T_ days into the encoder. The decoder receives the same statistics but for the analyzed moment in time.

The long-term module operates similarly but returns forecasts over a longer horizon. This encourages the model to capture extended market dynamics.

For training the short-term and long-term forecasting modules, a combined loss function is used, incorporating both regression error and stock ranking error. The regression error minimizes the gap between predicted and actual returns, while the ranking error ensures that assets with higher returns are prioritized.

Thus, the two branches of the model enable _StockFormer_ to capture market dynamics across different time horizons, allowing the RL agent to make more accurate and better-informed trading decisions.

In the second training stage, _StockFormer_ integrates three types of latent representations: _srelat,t_, _slong,t_, and _sshort,t_ into a unified state space _St_ using a cascade of multi-head attention blocks. The process begins with the fusion of short-term and long-term forecasts. Here, the long-term forecast representation serves as the _Query_, as it is less affected by short-term noise. The outcome of this step is then aligned with the latent representation of asset interdependencies, which is used as the _Key_ and _Value_ in the subsequent attention module.

The model is then trained to determine the optimal trading strategy using the _Actor–Critic_ approach. One of the key advantages of _StockFormer_ is the integration of predictive coding and policy optimization phases. The _Critic's_ evaluations help refine the quality of latent representation extraction, enabling the model to analyze inter-asset relationships more effectively and to better handle noise in the input data.

The original visualization of the _StockFormer_ framework is provided below.

![](https://c.mql5.com/2/169/2203917964723__2.png)

### Implementation in MQL5

After covering the theoretical foundation of _StockFormer_, we turn to implementing the proposed methods in _MQL5_. As highlighted in the theoretical description, the key architectural modification lies in introducing a multi-head _FeedForward_ block. Implementing this block is the first step of our work.

In the implementation of the multi-head _FeedForward_ block proposed by the authors of the _StockFormer_ framework, the outputs of the multi-head _Self-Attention_ block for each sequence element are divided into _h_ equal groups, with each group processed by its own _MLP_ with unique trainable parameters.

It is important to note that the approach to forming heads here differs from the one used in the conventional multi-head attention block. In _Multi-Head Self-Attention_, multiple versions of _Query, Key_, and _Value_ entities were generated from a single sequence element embedding. In this case, however, the authors of _StockFormer_ propose splitting the representation vector of a sequence element directly into several equal groups. Each group is then processed by its own _MLP_. This approach, of course, enables the creation of multiple heads without increasing the number of trainable parameters. Moreover, the output tensor preserves the same dimensionality, eliminating the need for a projection layer as in _MH Self-Attention_. However, because of this, we cannot use existing convolutional layers, as was possible before. This means we must find an alternative solution.

On the one hand, we could consider transposing the three-dimensional tensor to adapt the solution for convolutional layers with independent analysis of unidimensional sequences. But _StockFormer_ contains a substantial number of such layers. Therefore, transposing data before and after the _FeedForward_ block at each layer would significantly increase both training and inference time. Therefore, the decision was made to design a multi-head variant of the convolutional layer. Before implementing this new component in the main program, however, some adjustments are required on the _OpenCL_ side.

#### Extending the OpenCL Program

We begin by constructing the feed-forward-pass kernel for the new multi-head convolutional _FeedForwardMHConv_ layer. It should be noted that the parameter structure and part of the algorithm were borrowed from the kernel of the existing convolutional layer. The convolutional head identifier and the total number of heads were introduced as an additional dimension in the task space.

```
__kernel void FeedForwardMHConv(__global float *matrix_w,
                                __global float *matrix_i,
                                __global float *matrix_o,
                                const int inputs,
                                const int step,
                                const int window_in,
                                const int window_out,
                                const int activation
                               )
  {
   const size_t i = get_global_id(0);
   const size_t h = get_global_id(1);
   const size_t v = get_global_id(2);
   const size_t total = get_global_size(0);
   const size_t heads = get_global_size(1);
```

In the kernel body we identify the thread across all dimensions of the task space. We then determine the input and output dimensions for each convolution head, as well as the offsets in the global data buffers corresponding to the elements under analysis.

```
   const int window_in_h = (window_in + heads - 1) / heads;
   const int window_out_h = (window_out + heads - 1) / heads;
   const int shift_out = window_out * i + window_out_h * h;
   const int shift_in = step * i + window_in_h * h;
   const int shift_var_in = v * inputs;
   const int shift_var_out = v * window_out * total;
   const int shift_var_w = v * window_out * (window_in_h + 1);
   const int shift_w_h = h * window_out_h * (window_in_h + 1);
```

Once this preparatory work is complete, we move on to constructing the convolution operations between the input data and the trainable filter. Within a single thread, we perform convolution for one head of the input data with its corresponding filter. To achieve this, we organize a system of nested loops. The outer loop iterates over the elements of the output layer corresponding to the given convolution head.

```
   float sum = 0;
   float4 inp, weight;
   int stop = (window_in_h <= (inputs - shift_in) ? window_in_h : (inputs - shift_in));
//---
   for(int out = 0; (out < window_out_h && (window_out_h * h + out) < window_out); out++)
     {
      int shift = (window_in_h + 1) * out + shift_w_h;
```

Within the outer loop, we first calculate the offset in the buffer of trainable parameters. We then initiate an inner loop to traverse the elements of the convolution window applied to the input data.

```
      for(int k = 0; k <= stop; k += 4)
        {
         switch(stop - k)
           {
            case 0:
               inp = (float4)(1, 0, 0, 0);
               weight = (float4)(matrix_w[shift_var_w + shift + window_in_h], 0, 0, 0);
               break;
            case 1:
               inp = (float4)(matrix_i[shift_var_in + shift_in + k], 1, 0, 0);
               weight = (float4)(matrix_w[shift_var_w + shift + k], matrix_w[shift_var_w + shift + window_in_h], 0, 0);
               break;
            case 2:
               inp = (float4)(matrix_i[shift_var_in + shift_in + k],
                              matrix_i[shift_var_in + shift_in + k + 1], 1, 0);
               weight = (float4)(matrix_w[shift_var_w + shift + k], matrix_w[shift_var_w + shift + k + 1],
                                 matrix_w[shift_var_w + shift + window_in_h], 0);
               break;
            case 3:
               inp = (float4)(matrix_i[shift_var_in + shift_in + k], matrix_i[shift_var_in + shift_in + k + 1],
                              matrix_i[shift_var_in + shift_in + k + 2], 1);
               weight = (float4)(matrix_w[shift_var_w + shift + k], matrix_w[shift_var_w + shift + k + 1],
                                 matrix_w[shift_var_w + shift + k + 2], matrix_w[shift_var_w + shift + shift_w_h]);
               break;
            default:
               inp = (float4)(matrix_i[shift_var_in + shift_in + k], matrix_i[shift_var_in + shift_in + k + 1],
                              matrix_i[shift_var_in + shift_in + k + 2], matrix_i[shift_var_in + shift_in + k + 3]);
               weight = (float4)(matrix_w[shift_var_w + shift + k], matrix_w[shift_var_w + shift + k + 1],
                                 matrix_w[shift_var_w + shift + k + 2], matrix_w[shift_var_w + shift + k + 3]);
               break;
           }
```

To optimize computation, we use built-in vector multiplication functions, which allow for more efficient use of processor resources. Accordingly, we first load the necessary values from external buffers into local vector variables, then perform vector multiplications before moving on to the next iteration of the inner loop.

```
         sum += IsNaNOrInf(dot(inp, weight), 0);
        }
```

After completing all iterations of the inner loop, we apply the activation function and store the result in the corresponding output buffer element. The process then continues with the next iteration of the outer loop.

```
      sum = IsNaNOrInf(sum, 0);
      //---
      matrix_o[shift_var_out + out + shift_out] = Activation(sum, activation);;
     }
  }
```

By the end of all loop iterations, the output buffer contains all required values, and the kernel execution is complete.

We now proceed to constructing the backpropagation algorithms. Here, it must be noted that unlike in the feed-forward pass, we cannot introduce the convolution head identifier as a dimension in the task space.

Recall that during the gradient distribution process, we accumulate the influence values of each input element on the output. In cases where the stride of the convolution window is smaller than its size, a single input element may affect result tensor elements across multiple convolution heads.

For this reason, in this case, we introduce the number of attention heads as an additional external parameter of the _CalcHiddenGradientMHConv_ kernel. The identifier of the specific convolution head is then determined during the process of accumulating the error gradients.

```
__kernel void CalcHiddenGradientMHConv(__global float *matrix_w,
                                       __global float *matrix_g,
                                       __global float *matrix_o,
                                       __global float *matrix_ig,
                                       const int outputs,
                                       const int step,
                                       const int window_in,
                                       const int window_out,
                                       const int activation,
                                       const int shift_out,
                                       const int heads
                                      )
  {
   const size_t i = get_global_id(0);
   const size_t inputs = get_global_size(0);
   const size_t v = get_global_id(1);
```

In the kernel body, we identify the current thread in the two-dimensional task space, which points to the input data element and the univariate sequence identifier. After that, we determine the values of the constants, including shift in the data buffers, as well as the window dimensions and the number of filters for one convolution head.

```
   const int shift_var_in = v * inputs;
   const int shift_var_out = v * outputs;
   const int shift_var_w = v * window_out * (window_in + 1);
   const int window_in_h = (window_in + heads - 1) / heads;
   const int window_out_h = (window_out + heads - 1) / heads;
```

Here we also define the range of the output window that is influenced by the analyzed element of the source data.

```
   float sum = 0;
   float out = matrix_o[shift_var_in + i];
   const int w_start = i % step;
   const int start = max((int)((i - window_in + step) / step), 0);
   int stop = (w_start + step - 1) / step;
   stop = min((int)((i + step - 1) / step + 1), stop) + start;
   if(stop > (outputs / window_out))
      stop = outputs / window_out;
```

After the preparatory work, we proceed to collecting error gradients from all dependent elements of the result tensor. To achieve this, we organize a system of loops. The outer loop will iterate over dependent elements within the previously defined window.

```
   for(int k = start; k < stop; k++)
     {
      int head = (k % window_out) / window_out_h;
```

In the body of the outer loop, we first define the convolution head for a single element of the result tensor, and then organize a nested loop for iterating over filters.

```
      for(int h = 0; h < window_out_h; h ++)
        {
         int shift_g = k * window_out + head * window_out_h + h;
         int shift_w = (stop - k - 1) * step + (i % step) / window_in_h +
                       head * (window_in_h + 1) + h * (window_in_h + 1);
         if(shift_g >= outputs || shift_w >= (window_in_h + 1) * window_out)
            break;
         float grad = matrix_g[shift_out + shift_g + shift_var_out];
         sum += grad * matrix_w[shift_w + shift_var_w];
        }
     }
```

It is within the body of the nested loop that we accumulate the error gradient across all filters of a single convolution head before moving on to the next iteration of the loop system.

Once the gradients from all dependent elements have been collected, the accumulated value is adjusted by the derivative of the activation function and the result is stored in the corresponding element of the data buffer:

```
   matrix_ig[shift_var_in + i] = Deactivation(sum, out, activation);
  }
```

At this point, the operations of the gradient distribution kernel are complete. As for the parameter update kernel, I suggest that you review it independently. The complete _OpenCL_ program code is provided in the appendix to this article. We will now move on to the next stage of our work: implementing the multi-head convolutional neural layer in the main program.

#### Multi-Head Convolutional Layer

To implement the convolutional functionality on the main program side, we introduce a new object, _CNeuronMHConvOCL_. As one might expect, the existing convolutional layer was used as the parent class. The structure of the new object is presented below.

```
class CNeuronMHConvOCL  :  public CNeuronConvOCL
  {
protected:
   uint              iHeads;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMHConvOCL(void)  :  iHeads(1)   {};
                    ~CNeuronMHConvOCL(void)  {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint step, uint window_out,
                          uint units_count, uint variables, uint heads,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const override   {  return defNeuronMHConvOCL;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
  };
```

In this structure, only one internal variable is introduced to store the specified number of convolution heads. All other objects and variables required for processing are inherited from the parent class. In addition, the forward and backward pass methods are overridden, serving as wrappers for calling the kernels described earlier. The kernel scheduling algorithm remains unchanged and therefore does not require further explanation. In this article, we will focus exclusively on the initialization method _Init_, which was implemented almost entirely from scratch.

Within the method's parameter structure, only a single new element was added, enabling the number of convolution heads to be passed from the calling program.

```
bool CNeuronMHConvOCL::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                            uint window, uint step, uint window_out,
                            uint units_count, uint variables, uint heads,
                            ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronProofOCL::Init(numOutputs, myIndex, open_cl, window, step,
                             units_count * window_out * variables, ADAM, batch))
      return false;
```

Inside the method, we first invoke the corresponding initialization method of the parent subsampling layer, which in this case acts as the ancestral object. We then assign the values of the external parameters to local variables.

```
   iWindowOut = window_out;
   iVariables = variables;
   iHeads = MathMax(MathMin(heads, window), 1);
```

Next, we must initialize the tensor of trainable parameters with random values. Before doing so, however, we define the dimensionality of this tensor. Its size depends on the number of univariate series in the multimodal sequence under analysis, the total number of filters, and the size of the convolution window for a single head.

```
   const int window_h = int((iWindow + heads - 1) / heads);
   const int count = int((window_h + 1) * iWindowOut * iVariables);
```

Note that we are referring to the total number of filters across all convolution heads while using only the convolution window of a single head. It is straightforward to deduce that the number of trainable parameters for a single convolution head equals the product of the number of filters per head and the size of its input window, plus one bias term ( _Fi \\*_( _Wi_ \+ 1)). To obtain the total number of parameters for a single univariate sequence, we simply multiply this value by the number of heads ( _Fi_ \\* (Wi + 1) \* _H_). It is also evident that the number of filters per head, multiplied by the number of heads, yields the total number of filters specified by the user.

The next step is to check the validity of the pointer to the buffer object containing the trainable parameters and, if necessary, create a new object.

```
   if(!WeightsConv)
     {
      WeightsConv = new CBufferFloat();
      if(!WeightsConv)
         return false;
     }
```

We reserve the required number of elements in the buffer and organize a loop to populate the buffer with random values.

```
   if(!WeightsConv.Reserve(count))
      return false;
   float k = (float)(1 / sqrt(window_h + 1));
   for(int i = 0; i < count; i++)
     {
      if(!WeightsConv.Add((GenerateWeight() * 2 * k - k) * WeightsMultiplier))
         return false;
     }
   if(!WeightsConv.BufferCreate(OpenCL))
      return false;
```

After successfully filling the buffer with random values, we transfer it to the OpenCL context memory. Next, we create momentum buffers, filling them with zero values.

```
   if(!FirstMomentumConv)
     {
      FirstMomentumConv = new CBufferFloat();
      if(!FirstMomentumConv)
         return false;
     }
   if(!FirstMomentumConv.BufferInit(count, 0.0))
      return false;
   if(!FirstMomentumConv.BufferCreate(OpenCL))
      return false;
//---
   if(!SecondMomentumConv)
     {
      SecondMomentumConv = new CBufferFloat();
      if(!SecondMomentumConv)
         return false;
     }
   if(!SecondMomentumConv.BufferInit(count, 0.0))
      return false;
   if(!SecondMomentumConv.BufferCreate(OpenCL))
      return false;
   if(!!DeltaWeightsConv)
      delete DeltaWeightsConv;
//---
   return true;
  }
```

At this point, we conclude our discussion of the methods of the multi-head convolutional layer object _CNeuronMHConvOCL_. The full implementation of this class and all of its methods can be found in the appendix.

#### Multi-Head _FeedForward_ Block

We have now created the first building block in the construction of the _StockFormer_ framework. Next, we will use it to implement the multi-head FeedForward block within the new object _CNeuronMHFeedForward_, whose structure is shown below.

```
class CNeuronMHFeedForward   :  public CNeuronBaseOCL
  {
protected:
   CNeuronMHConvOCL  acConvolutions[2];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMHFeedForward(void) {};
                    ~CNeuronMHFeedForward(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_out,
                          uint units_count, uint variables, uint heads,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const override   {  return defNeuronMHFeedForward;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual void      SetOpenCL(COpenCLMy *obj) override;
  };
```

In the structure of the new object, we declare an array consisting of two internal multi-head convolutional layers and override the familiar set of virtual methods. These internal objects are declared statically, which allows us to keep the constructor and destructor empty. Initialization of all declared and inherited objects is performed in the _Init_ method.

```
bool CNeuronMHFeedForward::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                uint window, uint window_out,
                                uint units_count, uint variables, uint heads,
                                ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units_count * variables, optimization_type, batch))
      return false;
```

The initialization method receives constants that define the architecture of the object being created. Some of these parameters are immediately passed to the corresponding initialization method of the parent class to set up the inherited base interfaces.

We then initialize the first convolutional layer, specifying GELU as its activation function.

```
   if(!acConvolutions[0].Init(0, 0, OpenCL, window, window, window_out, units_count, variables, heads,
                                                                                  optimization, iBatch))
      return false;
   acConvolutions[0].SetActivationFunction(GELU);
```

After that, we initialize the second convolutional layer, this time without an activation function.

```
   if(!acConvolutions[1].Init(0, 1, OpenCL, window_out, window_out, window, units_count, variables, heads,
                                                                                     optimization, iBatch))
      return false;
   acConvolutions[1].SetActivationFunction(None);
```

It should be noted that when calling the initialization method of the second convolutional layer, we swap the parameters corresponding to the number of filters and the input window size.

At the output of the FeedForward block, residual connections with normalization are applied. For this reason, we do not override the result buffer of the block's interface. However, we do override the error gradient buffer, enabling direct transfer of gradients from the interfaces into the corresponding buffer of the second convolutional layer.

```
   if(!SetGradient(acConvolutions[1].getGradient(), true))
      return false;
   SetActivationFunction(None);
//---
   return true;
  }
```

We also disable the activation function for the block itself and complete the initialization method by returning the logical result of execution to the calling program.

Once initialization is complete, we move on to implementing the forward-pass algorithm within the _feedForward_ method. In this case, the implementation is straightforward. We simply invoke the forward-pass methods of the internal convolutional layers in sequence.

```
bool CNeuronMHFeedForward::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   CObject *prev = NeuronOCL;
   for(uint i = 0; i < acConvolutions.Size(); i++)
     {
      if(!acConvolutions[i].FeedForward(prev))
         return false;
      prev = GetPointer(acConvolutions[i]);
     }
```

The outputs are then summed with the original inputs, followed by normalization within the elements of the multimodal sequence under analysis.

```
   if(!SumAndNormilize(NeuronOCL.getOutput(), acConvolutions[acConvolutions.Size() - 1].getOutput(),
                       Output, acConvolutions[0].GetWindow(), true, 0, 0, 0, 1))
      return false;
//---
   return true;
  }
```

The method concludes by returning the logical result of the operation to the calling program.

The _calcInputGradients_ algorithm of the error gradient distribution method looks a little more complicated since we need to propagate gradients along two data streams.

```
bool CNeuronMHFeedForward::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      return false;
```

The parameters of this method include a pointer to the source data object; into its buffer we WILL pass the error gradient, distributed in accordance with the influence of the input data on the final model output. And in the method body, we immediately check the relevance of the received pointer.

After successfully passing the control block, we organize a loop of the reverse iteration through the internal convolutional layers while sequentially calling the relevant methods.

```
   for(int i = (int)acConvolutions.Size() - 2; i >= 0; i--)
     {
      if(!acConvolutions[i].calcHiddenGradients(acConvolutions[i + 1].AsObject()))
         return false;
     }
```

After the error gradient is distributed across the internal object pipeline, it is then propagated to the source data level. This operation concludes the main workflow.

```
   if(!NeuronOCL.calcHiddenGradients(acConvolutions[0].AsObject()))
      return false;
```

Next, we need to propagate the error gradient along the second information stream. The algorithm here is split into two branches of operations, depending on the presence of the activation function of the source data. Since we don't have an activation function, we simply sum the accumulated error gradient at the source data level with similar values at the output of our block.

```
   if(NeuronOCL.Activation() == None)
     {
      if(!SumAndNormilize(NeuronOCL.getGradient(), Gradient, NeuronOCL.getGradient(),
                          acConvolutions[0].GetWindow(), false, 0, 0, 0, 1))
         return false;
     }
   else
     {
      if(!DeActivation(NeuronOCL.getOutput(), NeuronOCL.getPrevOutput(), Gradient, NeuronOCL.Activation()) ||
         !SumAndNormilize(NeuronOCL.getGradient(), NeuronOCL.getPrevOutput(), NeuronOCL.getGradient(),
                          acConvolutions[0].GetWindow(), false, 0, 0, 0, 1))
         return false;
     }
//---
   return true;
  }
```

Otherwise, we would need to first adjust the error gradient of our block's output level by the derivative of the activation function of the source data. Only then would we sum the data from the two information streams.

All that remains is to return the result of the operations to the calling program and exit the method.

The method for adjusting the trainable parameters of the block toward reducing the overall model error, _updateInputWeights_, is left for independent study. Its algorithm is quite straightforward: we simply call the corresponding methods of the internal objects in sequence. The complete implementation of the multi-head _FeedForward_ block object _CNeuronMHFeedForward_, along with all its methods, can be found in the attachment to this article.

#### Decoder of Diversified Multi-Head Attention

After building the multi-head _FeedForward_ block, we now proceed to the construction of encoder and decoder objects for diversified multi-head attention. To implement the algorithms of these modules, we introduce new objects: _CNeuronDMHAttention_ and _CNeuronCrossDMHAttention_, respectively. The construction of these objects follows a largely similar structure. The latter, however, differs in that it contains an internal cross-attention block and operates with two sources of input data. Within the scope of this article, I propose focusing on the decoder as the more complex object. Once its algorithms are clear, understanding the encoder will not pose significant difficulty.

As the parent class for both objects, we use CNeuronRMAT, which provides the underlying algorithm of the sequential model.

```
class CNeuronCrossDMHAttention   :  public CNeuronRMAT
  {
protected:
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { return false; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer) override { return false; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput,
                                                                   CBufferFloat *SecondGradient,
                                                        ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override { return false; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;

public:
                     CNeuronCrossDMHAttention(void)  {};
                    ~CNeuronCrossDMHAttention(void)  {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint window_cross, uint units_cross,
                          uint heads, uint layers,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronCrossDMHAttention; }
  };
```

In the structure of the decoder object, we can observe only the overriding of virtual methods. The internal object structure is defined in the initialization method _Init_, whose parameters include the key constants that determine the architecture of the object.

```
bool CNeuronCrossDMHAttention::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                    uint window, uint window_key, uint units_count,
                                    uint window_cross, uint units_cross, uint heads,
                                    uint layers, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units_count, optimization_type, batch))
      return false;
```

In the body of the method, we first invoke the parent class’s initialization method of the fully connected layer to set up the inherited interfaces.

Next, we clear the dynamic array used for storing pointers to the internal objects of the module and create several local variables for temporary data storage.

```
   cLayers.Clear();
   cLayers.SetOpenCL(OpenCL);
   CNeuronRelativeSelfAttention *attention = NULL;
   CNeuronRelativeCrossAttention *cross = NULL;
   CNeuronMHFeedForward *conv = NULL;
   bool use_self = units_count > 0;
   int layer = 0;
```

Once this preparatory stage is complete, we organize a loop with the number of iterations equal to the specified number of internal layers of the diversified multi-head attention decoder.

```
   for(uint i = 0; i < layers; i++)
     {
      if(use_self)
        {
         attention = new CNeuronRelativeSelfAttention();
         if(!attention ||
            !attention.Init(0, layer, OpenCL, window, window_key, units_count, heads, optimization, iBatch) ||
            !cLayers.Add(attention)
           )
           {
            delete attention;
            return false;
           }
         layer++;
        }
```

Inside the loop, we first create a block of relative _Self-Attention_ to analyze dependencies in the primary input data stream. It is important to note that the _Self\_Attention_ block is instantiated only when the sequence length of the primary input stream is greater than "1". Otherwise, there is no data available for dependency analysis.

We then add a relative cross-attention module.

```
      cross = new CNeuronRelativeCrossAttention();
      if(!cross ||
         !cross.Init(0, layer, OpenCL, window, window_key, units_count, heads,
                              window_cross, units_cross, optimization, iBatch) ||
         !cLayers.Add(cross)
        )
        {
         delete cross;
         return false;
        }
      layer++;
```

Each internal layer of the decoder is completed with a multi-head _FeedForward_ block, after which we move on to the next iteration of the loop.

```
      conv = new CNeuronMHFeedForward();
      if(!conv ||
         !conv.Init(0, layer, OpenCL, window, 2 * window, units_count, 1, heads, optimization, iBatch) ||
         !cLayers.Add(conv)
        )
        {
         delete conv;
         return false;
        }
      layer++;
     }
```

After initializing the full set of internal objects, we replace the interface pointers with references to these objects and conclude the method, returning a logical result to the calling program.

```
   SetOutput(conv.getOutput(), true);
   SetGradient(conv.getGradient(), true);
//---
   return true;
  }
```

The algorithm of the _feedForward_ method consists of sequential calls to the respective methods of the internal objects. I suggest leaving its study as an independent exercise. Instead, let's dedicate some attention to the error gradient distribution algorithm _calcInputGradients_.

```
bool CNeuronCrossDMHAttention::calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput,
                                   CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = -1)
  {
   if(!NeuronOCL || !SecondInput || !SecondGradient)
      return false;
```

The parameters of this method include pointers to the input data objects and their corresponding error gradients, into which the results of the operations will be written. Therefore, within the method body, we first validate the pointers received.

It is important to highlight that during the feed-forward pass, the second input data source is equally utilized by the cross-attention modules of all decoder layers. Consequently, we must aggregate the error gradients from all information flows. As usual in such cases, we need an internal buffer for data storage. Since **_no such buffer_** was defined in the new object, we use one of the unused buffers inherited from the parent class.

First, we check the size of the inherited buffer and, if necessary, adjust it.

```
   if(PrevOutput.Total() != SecondGradient.Total())
     {
      PrevOutput.BufferFree();
      if(!PrevOutput.BufferInit(SecondGradient.Total(), 0) ||
         !PrevOutput.BufferCreate(OpenCL))
         return false;
     }
```

Next, we initialize the error gradient buffer for the second data source with zero values. This step ensures that gradients from the current pass are not accumulated with values from previous ones.

```
   if(!SecondGradient.Fill(0))
      return false;
```

We then create local variables for temporary data storage.

```
   CObject *next = cLayers[-1];
   CNeuronBaseOCL *current = NULL;
```

At this point, the preparatory stage is complete, and we initiate a reverse iteration loop over the internal objects.

```
   for(int i = cLayers.Total() - 2; i >= 0; i--)
     {
      current = cLayers[i];
      if(!current ||
         !current.calcHiddenGradients(next, SecondInput, PrevOutput, SecondActivation))
         return false;
      if(next.Type() == defNeuronCrossDMHAttention)
         if(!SumAndNormilize(SecondGradient, PrevOutput, SecondGradient, 1, false, 0, 0, 0, 1))
            return false;
      next = current;
     }
```

Within this loop, we sequentially call the respective methods of the internal objects, while continuously checking the type of the object responsible for distributing the error gradients. If a cross-attention block is encountered, we add the error gradient of the second data source to the previously accumulated values.

After successfully completing all iterations of the loop, we propagate the error gradient back to the input data of the main stream.

```
   if(!NeuronOCL.calcHiddenGradients(next, SecondInput, PrevOutput, SecondActivation))
      return false;
   if(next.Type() == defNeuronCrossDMHAttention)
      if(!SumAndNormilize(SecondGradient, PrevOutput, SecondGradient, 1, false, 0, 0, 0, 1))
         return false;
//---
   return true;
  }
```

At this stage, we again verify the type of the object performing the gradient distribution and, if necessary, add the error gradient from the second information stream to the accumulated values. Finally, we conclude the method by returning a logical result to the calling program.

This completes our review of the construction algorithms for the decoder of diversified multi-head attention. The complete implementation of this object and all its methods can be found in the attachment. There you will also find the full code for all other objects presented in this article.

We have now implemented the core architectural unit of the _StockFormer_ framework - the diversified multi-head attention module in the form of both the encoder and decoder of the _Transformer_ architecture. However, the authors of _StockFormer_ also propose a two-level training process with a sophisticated mechanism of interaction between trainable models. This will be discussed in the upcoming article.

### Conclusion

We have become familiar with the _StockFormer_ framework, whose authors propose an innovative approach to training trading strategies in financial markets. _StockFormer_ combines methods of predictive coding with deep reinforcement learning. Its primary advantage lies in its ability to train flexible policies that account for dynamic dependencies between multiple assets, while simultaneously predicting their behavior in both short-term and long-term horizons.

The three-branch predictive coding mechanism extracts latent representations tied to short-term trends, long-term dynamics, and inter-asset dependencies. The cascading multi-head attention mechanism allows for efficient integration of these diverse representations into a unified state space.

In the practical section of this article, we implemented in _MQL5_ the modification of the vanilla Transformer algorithm proposed by the authors and incorporated it into the encoder and decoder modules of diversified multi-head attention. In the next article, we will continue this work by discussing the architecture of the trainable models and the process of their training.

#### References

- [StockFormer: Learning Hybrid Trading Machines with Predictive Coding](https://www.mql5.com/go?link=https://www.ijcai.org/proceedings/2023/0530.pdf "StockFormer: Learning Hybrid Trading Machines with Predictive Coding")
- [Other articles from this series](https://www.mql5.com/en/search#!keyword=Neural%20networks%20made%20easy&module=mql5_module_articles "https://www.mql5.com/en/search#!keyword=Neural%20networks%20made%20easy&module=mql5_module_articles")

#### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | Research.mq5 | Expert Advisor | Expert Advisor for collecting samples |
| 2 | ResearchRealORL.mq5 | Expert Advisor | Expert Advisor for collecting samples using the Real-ORL method |
| 3 | Study1.mq5 | Expert Advisor | Predictive Learning Expert Advisor |
| 4 | Study2.mq5 | Expert Advisor | Policy Training Expert Advisor |
| 5 | Test.mq5 | Expert Advisor | Model testing Expert Advisor |
| 6 | Trajectory.mqh | Class library | System state description structure |
| 7 | NeuroNet.mqh | Class library | A library of classes for creating a neural network |
| 8 | NeuroNet.cl | Code Base | OpenCL program code library |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/16686](https://www.mql5.com/ru/articles/16686)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16686.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/16686/mql5.zip "Download MQL5.zip")(2253.87 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Final Part)](https://www.mql5.com/en/articles/17241)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Chimera)](https://www.mql5.com/en/articles/17210)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model (Final Part)](https://www.mql5.com/en/articles/17157)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model](https://www.mql5.com/en/articles/17142)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Final Part)](https://www.mql5.com/en/articles/17104)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Hidformer)](https://www.mql5.com/en/articles/17069)
- [Neural Networks in Trading: Memory Augmented Context-Aware Learning for Cryptocurrency Markets (Final Part)](https://www.mql5.com/en/articles/16993)

**[Go to discussion](https://www.mql5.com/en/forum/495433)**

![Quantum computing and trading: A fresh approach to price forecasts](https://c.mql5.com/2/110/Quantum_Computing_and_Trading_A_New_Look_at_Price_Forecasts____LOGO.png)[Quantum computing and trading: A fresh approach to price forecasts](https://www.mql5.com/en/articles/16879)

The article describes an innovative approach to forecasting price movements in financial markets using quantum computing. The main focus is on the application of the Quantum Phase Estimation (QPE) algorithm to find prototypes of price patterns allowing traders to significantly speed up the market data analysis.

![Automating Trading Strategies in MQL5 (Part 32): Creating a Price Action 5 Drives Harmonic Pattern System](https://c.mql5.com/2/169/19463-automating-trading-strategies-logo__1.png)[Automating Trading Strategies in MQL5 (Part 32): Creating a Price Action 5 Drives Harmonic Pattern System](https://www.mql5.com/en/articles/19463)

In this article, we develop a 5 Drives pattern system in MQL5 that identifies bullish and bearish 5 Drives harmonic patterns using pivot points and Fibonacci ratios, executing trades with customizable entry, stop loss, and take-profit levels based on user-selected options. We enhance trader insight with visual feedback through chart objects like triangles, trendlines, and labels to clearly display the A-B-C-D-E-F pattern structure.

![Price Action Analysis Toolkit Development (Part 40): Market DNA Passport](https://c.mql5.com/2/169/19460-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 40): Market DNA Passport](https://www.mql5.com/en/articles/19460)

This article explores the unique identity of each currency pair through the lens of its historical price action. Inspired by the concept of genetic DNA, which encodes the distinct blueprint of every living being, we apply a similar framework to the markets, treating price action as the “DNA” of each pair. By breaking down structural behaviors such as volatility, swings, retracements, spikes, and session characteristics, the tool reveals the underlying profile that distinguishes one pair from another. This approach provides more profound insight into market behavior and equips traders with a structured way to align strategies with the natural tendencies of each instrument.

![Developing a multi-currency Expert Advisor (Part 21): Preparing for an important experiment and optimizing the code](https://c.mql5.com/2/110/Developing_a_Multicurrency_Advisor_Part_21____LOGO.png)[Developing a multi-currency Expert Advisor (Part 21): Preparing for an important experiment and optimizing the code](https://www.mql5.com/en/articles/16373)

For further progress it would be good to see if we can improve the results by periodically re-running the automatic optimization and generating a new EA. The stumbling block in many debates about the use of parameter optimization is the question of how long the obtained parameters can be used for trading in the future period while maintaining the profitability and drawdown at the specified levels. And is it even possible to do this?

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/16686&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062551597948511317)

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