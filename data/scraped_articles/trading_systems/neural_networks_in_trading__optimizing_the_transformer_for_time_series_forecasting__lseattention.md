---
title: Neural Networks in Trading: Optimizing the Transformer for Time Series Forecasting (LSEAttention)
url: https://www.mql5.com/en/articles/16360
categories: Trading Systems, Expert Advisors, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T18:34:48.134337
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/16360&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069565511471138582)

MetaTrader 5 / Trading systems


### Introduction

Multivariate time series forecasting plays a critical role across a wide range of domains (finance, healthcare, and more) where the objective is to predict future values based on historical data. This task becomes particularly challenging in long-term forecasting, which demands models capable of effectively capturing feature correlations and long-range dependencies in multivariate time series data. Recent research has increasingly focused on leveraging the _Transformer_ architecture for time series forecasting due to its powerful _Self-Attention_ mechanism, which excels at modeling complex temporal interactions. However, despite its potential, many contemporary methods for multivariate time series forecasting still rely heavily on linear models, raising concerns about the true effectiveness of _Transformers_ in this context.

The _Self-Attention_ mechanism at the core of the _Transformer_ architecture is defined as follows:

![](https://c.mql5.com/2/156/Att__1.png)

where _Q_, _K_, and _V_ represent the _Query_, _Key_, and _Value_ matrices respectively, and _dk_ denotes the dimensionality of the vectors describing each sequence element. This formulation enables the _Transformer_ to dynamically assess the relevance of different elements in the input sequence, thereby facilitating the modeling of complex dependencies within the data.

Various adaptations of the _Transformer_ architecture have been proposed to improve its performance on long-term time series forecasting tasks. For example, _[FEDformer](https://www.mql5.com/en/articles/14858)_ incorporates an advanced Fourier module that achieves linear complexity in both time and space, significantly enhancing scalability and efficiency for long input sequences.

_[PatchTST](https://www.mql5.com/en/articles/14798)_ on the other hand, abandons pointwise attention in favor of patch-level representation, focusing on contiguous segments rather than individual time steps. This approach allows the model to capture more extensive semantic information in multivariate time series, which is crucial for effective long-term forecasting.

In domains such as computer vision and natural language processing, attention matrices can suffer from entropy collapse or rank collapse. This problem is further exacerbated in time series forecasting due to the frequent fluctuations inherent in time-based data, often resulting in substantial degradation of model performance. The underlying causes of entropy collapse remain poorly understood, highlighting the need for further investigation into its mechanisms and effects on model generalization. These challenges are the focus of the paper titled " _[LSEAttention is All You Need for Time Series Forecasting](https://www.mql5.com/go?link=https://arxiv.org/abs/2410.23749 "https://arxiv.org/abs/2410.23749")_".

### 1\. The LSEAttention Algorithm

The goal of multivariate time series forecasting is to estimate the most probable future values _P_ for each of the _C_ channels, represented as a tensor _Y_ ∈ _RC×P_. This prediction is based on historical time series data of length _L_ with _C_ channels, encapsulated in the input matrix _X_ ∈ _RC×L_. The task involves training a predictive model _fω_ _RC×L_ → _RC×P_ parametrized by _ω_, to minimize the Mean Squared Error ( _MSE_) between predicted and actual values.

_Transformer_ rely heavily on pointwise _Self-Attention_ mechanisms to capture temporal associations. However, this reliance can lead to a phenomenon known as attention collapse, where attention matrices converge to nearly identical values across different input sequences. This results in poor generalization of the data by the model.

The authors of the _LSEAttention_ method draw an analogy between the dependency coefficients calculated via the _Softmax_ function and the _Log-Sum-Exp_ ( _LSE_) operation. They hypothesize that numerical instability in this formulation may be the root cause of attention collapse.

The condition number of a function reflects its sensitivity to small input variations. A high condition number indicates that even minor perturbations in the input can cause significant output deviations.

In the context of attention mechanisms, such instability can manifest as over-attention or entropy collapse, characterized by attention matrices with extremely high diagonal values (indicating overflow) and very low off-diagonal values (indicating underflow).

To address these issues, the authors propose the _LSEAttention_ module, which integrates the _Log-Sum-Exp_ ( _LSE_) trick with the _GELU_ (Gaussian Error Linear Unit) activation function. The _LSE_ trick mitigates numerical instability caused by overflow and underflow through normalization. The _Softmax_ function can be reformulated using _LSE_ as follows:

![](https://c.mql5.com/2/156/399519497270__1.png)

where the exponent of _LSE_( _x_) denotes the exponential values of the _log-sum-exp_ function, increasing numerical stability.

![](https://c.mql5.com/2/156/4694530218920__1.png)

By using exponent properties, any exponential term can be expressed as the product of two exponential terms.

![](https://c.mql5.com/2/156/124035487663__1.png)

where _a_ is a constant used for normalization. In practice, the maximum value is usually used as a constant. Substituting the product of the exponents into the LSE formula and taking the total value outside the sum sign, we have:

![](https://c.mql5.com/2/156/1574021125614__1.png)

The logarithm of a product becomes a sum of logarithms, and the natural logarithm of an exponential equals the exponent. This allows us to simplify the expression presented:

![](https://c.mql5.com/2/156/1845916829593__1.png)

Let's substituting the resulting expression into the _Softmax_ function and use the exponential property:

![](https://c.mql5.com/2/156/220910014967__1.png)

As you can notice, the exponential value of the constant common to the numerator and denominator is canceled out. The exponent of the natural logarithm is equal to the logarithmic expression. Thus, we obtain a numerically stable _Softmax_ expression.

![](https://c.mql5.com/2/156/6547421614718__1.png)

When using the maximum value as a constant ( _a = max(x)_), we always get _x-a_ less than or equal to 0. In this case, the exponential value from _x-a_ lies in the range from 0 to 1, not including 0. Accordingly, the denominator of the function is in the range (1, n\].

![](https://c.mql5.com/2/156/4915643002771__1.png)

In addition, the authors of the _LSEAttention_ framework propose using the _GELU_ activation function, which provides smoother probabilistic activation. This helps stabilize extreme values in the logarithmic probability prior to applying the exponential function, thereby softening abrupt transitions in attention scores. By approximating the _ReLU_ function through a smooth curve involving the cumulative distribution function ( _CDF_) of the standard normal distribution, _GELU_ reduces the sharp shifts in activations that can occur with traditional _ReLU_. This property is particularly beneficial for stabilizing _Transformer_-based attention mechanisms, where sudden activation spikes can lead to numerical instability and gradient explosions.

The _GELU_ function is formally defined as follows:

![](https://c.mql5.com/2/156/3257695254633__1.png)

where _Φ(x)_ represents the _CDF_ of the standard normal distribution. This formulation ensures that _GELU_ applies varying degrees of scaling to input values depending on their magnitude, thereby suppressing the amplification of extreme values. The smooth, probabilistic nature of _GELU_ enables a gradual transition of input activations, which in turn mitigates large gradient fluctuations during training.

This property becomes particularly valuable when combined with the _Log-Sum-Exp_ ( _LSE_) trick, which normalizes the _Softmax_ function in a numerically stable manner. Together, _LSE_ and _GELU_ effectively prevent overflow and underflow in the exponential operations of _Softmax_, resulting in a stabilized range of attention weights. This synergy enhances the robustness of _Transformer_ models by ensuring a well-distributed allocation of attention coefficients across tokens. Ultimately, this leads to more stable gradients and improved convergence during training.

In traditional _Transformer_ architectures, the _ReLU_ ( _Rectified Linear Unit_) activation function used in the _Feed-Forward Network_ ( _FFN_) block is prone to the " _dying ReLU_" problem, where neurons can become inactive by outputting zero for all negative input values. This results in zero gradients for those neurons, effectively halting their learning and contributing to training instability.

To address this issue, the _Parametric ReLU_ ( _PReLU_) function is used as an alternative. _PReLU_ introduces a learnable slope for negative inputs, allowing non-zero output even when the input is negative. This adaptation not only mitigates the _dying ReLU_ problem but also enables a smoother transition between negative and positive activations, thereby enhancing the model's capacity to learn across the entire input space. The presence of non-zero gradients for negative values supports better gradient flow, which is essential for training deeper architectures. Consequently, the use of _PReLU_ contributes to overall training stability and helps maintain active representations, ultimately leading to improved model performance.

In the _LSEAttention_ Time Series _Transformer_ ( _LATST_) architecture, the authors also incorporate [invertible data normalization](https://www.mql5.com/en/articles/14673), which proves particularly effective in addressing distributional discrepancies between training and test data in time series forecasting tasks.

The architecture retains the traditional temporal _Self-Attention_ mechanism, embedded within the _LSEAttention_ module.

Overall, the _LATST_ architecture consists of a single-layer _Transformer_ structure augmented with substitution modules, enabling adaptive learning while maintaining the reliability of attention mechanisms. This design facilitates efficient modeling of temporal dependencies and boosts performance in time series forecasting tasks. The original visualization of the framework is provided below.

![](https://c.mql5.com/2/156/4159680839954__1.png)

### 2\. Implementation in MQL5

Having reviewed the theoretical aspects of the _LSEAttention_ framework, we now turn to the practical part of our work, where we explore one possible implementation of the proposed techniques using _MQL5_. It's important to note that this implementation will differ significantly from previous ones. Specifically, we will not create a new object to implement the proposed methods. Instead, we will integrate them into previously developed classes.

#### 2.1 Adjusting the Softmax layer

Let us consider the _[CNeuronSoftMaxOCL](https://www.mql5.com/en/articles/11716#para3)_ class, which handles the _Softmax_ function layer. This class is extensively used both as a standalone component of our model and as part of various frameworks. For instance, we employed the _CNeuronSoftMaxOCL_ object in building a pooling module based on dependency patterns ( _[CNeuronMHAttentionPooling](https://www.mql5.com/en/articles/16130#para31)_), which we have applied in several recent studies. Therefore, it is logical to incorporate numerically stable _Softmax_ computations into this class algorithm.

To achieve this, we will modify the behavior of the _SoftMax\_FeedForward_ kernel. The kernel receives pointers to two data buffers as parameters: one for the input values and another for the output results.

```
__kernel void SoftMax_FeedForward(__global float *inputs,
                                  __global float *outputs)
  {
   const uint total = (uint)get_local_size(0);
   const uint l = (uint)get_local_id(0);
   const uint h = (uint)get_global_id(1);
```

We plan the execution of the kernel in a two-dimensional task space. The first dimension corresponds to the number of values to be normalized within a single unit sequence. The second dimension represents the number of such unit sequences (or normalization heads). We group threads into workgroups within each individual unit sequence.

Within the kernel body, we first identify the current thread in the task space across all dimensions.

We then declare a local memory array that will be used to facilitate data exchange within the workgroup.

```
   __local float temp[LOCAL_ARRAY_SIZE];
```

Next, we define constant offsets into the global data buffers pointing to the relevant elements.

```
   const uint ls = min(total, (uint)LOCAL_ARRAY_SIZE);
   uint shift_head = h * total;
```

To minimize accesses to global memory, we copy the input values into local variables and validate the resulting values.

```
   float inp = inputs[shift_head + l];
   if(isnan(inp) || isinf(inp) || inp<-120.0f)
      inp = -120.0f;
```

It is worth noting that we limit the input values to a lower threshold of -120, which approximates the smallest exponent value representable in _float_ format. This serves as an additional measure to prevent underflow. We do not impose an upper limit on values, as potential overflow will be addressed by subtracting the maximum value.

Next, we determine the maximum value within the current unit sequence. This is achieved through a loop that collects the maximums of each subgroup in the workgroup and stores them in elements of the local memory array.

```
   for(int i = 0; i < total; i += ls)
     {
      if(l >= i && l < (i + ls))
         temp[l] = (i > 0 ? fmax(inp, temp[l]) : inp);
      barrier(CLK_LOCAL_MEM_FENCE);
     }
```

We then iterate over the local array to identify the global maximum of the current workgroup.

```
   uint count = min(ls, (uint)total);
   do
     {
      count = (count + 1) / 2;
      if(l < ls)
         temp[l] = (l < count && (l + count) < total ? fmax(temp[l + count],temp[l]) : temp[l]);
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
   float max_value = temp[0];
   barrier(CLK_LOCAL_MEM_FENCE);
```

The obtained maximum value is stored in a local variable, and we ensure thread synchronization at this stage. It is critical that all threads in the workgroup retain the correct maximum value before any modification of the local memory array elements occurs.

Now, we subtract the maximum value from each original input. Again, we check for the lower bound. Since subtracting a positive maximum may push the result beyond the valid range. We then compute the exponential of the adjusted value.

```
   inp = fmax(inp - max_value, -120);
   float inp_exp = exp(inp);
   if(isinf(inp_exp) || isnan(inp_exp))
      inp_exp = 0;
```

With two subsequent loops, we sum the resulting exponentials across the workgroup. The loop structure is similar to the one used to compute the maximum value. We just change the operation in the body of the loops accordingly.

```
   for(int i = 0; i < total; i += ls)
     {
      if(l >= i && l < (i + ls))
         temp[l] = (i > 0 ? temp[l] : 0) + inp_exp;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
//---
   count = min(ls, (uint)total);
   do
     {
      count = (count + 1) / 2;
      if(l < ls)
         temp[l] += (l < count && (l + count) < total ? temp[l + count] : 0);
      if(l + count < ls)
         temp[l + count] = 0;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
```

Having obtained all required values, we can now compute the final _Softmax_ values by dividing each exponential by the sum of exponentials within the workgroup.

```
//---
   float sum = temp[0];
   outputs[shift_head+l] = inp_exp / (sum + 1.2e-7f);
  }
```

The result of this operation is written to the appropriate element in the global result buffer.

It is important to highlight that the modifications made to the _Softmax_ computation during the forward pass do not require changes to the backward pass algorithms. As shown in the mathematical derivations presented earlier in this article, the use of the _LSE_ trick does not alter the final output of the Softmax function. Consequently, the influence of input data on the final result remains unchanged. Allowing us to continue using the existing gradient error distribution algorithm without modification.

#### 2.2 Modifying the Relative Attention Module

It is important to note that the _Softmax_ algorithm is not always used as a standalone layer. In nearly all versions of our implementations involving different _Self-Attention_ block designs, its logic is embedded directly within a unified attention kernel. Let us examine the _[CNeuronRelativeSelfAttention](https://www.mql5.com/en/articles/16097#para31)_ module. Here, the entire algorithm for the modified Self-Attention mechanism is implemented within the _MHRelativeAttentionOut_ kernel. And of course, we aim to ensure a stable training process across all model architectures. Therefore, we must implement numerically stable _Softmax_ in all such kernels. Whenever possible, we retain the existing kernel parameters and task space configuration. This same approach was used in upgrading the _MHRelativeAttentionOut_ kernel.

However, please note that any changes made to the kernel parameters or task space layout must be reflected in all wrapper methods of the main program that enqueue this kernel for execution. Failing to do so can result in critical runtime errors during kernel dispatch. This applies not only to modifications of the global task space but also to changes in workgroup sizes.

```
__kernel void MHRelativeAttentionOut(__global const float *q,         ///<[in] Matrix of Querys
                                     __global const float *k,         ///<[in] Matrix of Keys
                                     __global const float *v,         ///<[in] Matrix of Values
                                     __global const float *bk,        ///<[in] Matrix of Positional Bias Keys
                                     __global const float *bv,        ///<[in] Matrix of Positional Bias Values
                                     __global const float *gc,        ///<[in] Global content bias vector
                                     __global const float *gp,        ///<[in] Global positional bias vector
                                     __global float *score,           ///<[out] Matrix of Scores
                                     __global float *out,             ///<[out] Matrix of attention
                                     const int dimension              ///< Dimension of Key
                                    )
  {
//--- init
   const int q_id = get_global_id(0);
   const int k_id = get_local_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int kunits = get_local_size(1);
   const int heads = get_global_size(2);
```

Within the kernel body, as before, we identify the current thread within the task space and define all necessary dimensions.

Next, we declare a set of required constants, including both offsets into the global data buffers and auxiliary values.

```
   const int shift_q = dimension * (q_id * heads + h);
   const int shift_kv = dimension * (heads * k_id + h);
   const int shift_gc = dimension * h;
   const int shift_s = kunits * (q_id *  heads + h) + k_id;
   const int shift_pb = q_id * kunits + k_id;
   const uint ls = min((uint)get_local_size(1), (uint)LOCAL_ARRAY_SIZE);
   float koef = sqrt((float)dimension);
```

We also define a local memory array for inter-thread data exchange within each workgroup.

```
   __local float temp[LOCAL_ARRAY_SIZE];
```

To compute attention scores according to the vanilla _Self-Attention_ algorithm, we begin by performing a dot product between the corresponding vectors from the _Query_ and _Key_ tensors. However, the authors of the R-MAT framework add context-dependent and global bias terms. Since all vectors are of equal length, these operations can be carried out in a single loop, where the number of iterations equals the vector size. Within the loop body, we perform element-wise multiplication followed by summation.

```
//--- score
   float sc = 0;
   for(int d = 0; d < dimension; d++)
     {
      float val_q = q[shift_q + d];
      float val_k = k[shift_kv + d];
      float val_bk = bk[shift_kv + d];
      sc += val_q * val_k + val_q * val_bk + val_k * val_bk + gc[shift_q + d] * val_k + gp[shift_q + d] * val_bk;
     }
   sc = sc / koef;
```

The resulting score is scaled by the square root of the vector dimensionality. According to the authors of the vanilla Transformer, this operation improves model stability. We adhere to this practice.

The resulting values are then converted into probabilities using the _Softmax_ function. Here, we insert operations to ensure numerical stability. First, we determine the maximum value among attention scores within each workgroup. To do this, we divide the threads into subgroups, each of which writes its local maximum to an element in the local memory array.

```
//--- max value
   for(int cur_k = 0; cur_k < kunits; cur_k += ls)
     {
      if(k_id >= cur_k && k_id < (cur_k + ls))
        {
         int shift_local = k_id % ls;
         temp[shift_local] = (cur_k == 0 ? sc : fmax(temp[shift_local], sc));
        }
      barrier(CLK_LOCAL_MEM_FENCE);
     }
```

We then loop over the array to find the global maximum value.

```
   uint count = min(ls, (uint)kunits);
//---
   do
     {
      count = (count + 1) / 2;
      if(k_id < ls)
         temp[k_id] = (k_id < count && (k_id + count) < kunits ?
                          fmax(temp[k_id + count], temp[k_id]) :
                                                    temp[k_id]);
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
```

The current attention score is then adjusted by subtracting this maximum value before applying the exponential function. Here, we must also synchronize threads. Because in the next step, we will be changing the values of the local array elements and risk overwriting the value of the maximum element before it is used by all the threads of the workgroup.

```
   sc = exp(fmax(sc - temp[0], -120));
   if(isnan(sc) || isinf(sc))
      sc = 0;
   barrier(CLK_LOCAL_MEM_FENCE);
```

Next, we compute the sum of all exponentials within the workgroup. As before, we use a two-pass reduction algorithm consisting of sequential loops.

```
//--- sum of exp
   for(int cur_k = 0; cur_k < kunits; cur_k += ls)
     {
      if(k_id >= cur_k && k_id < (cur_k + ls))
        {
         int shift_local = k_id % ls;
         temp[shift_local] = (cur_k == 0 ? 0 : temp[shift_local]) + sc;
        }
      barrier(CLK_LOCAL_MEM_FENCE);
     }
//---
   count = min(ls, (uint)kunits);
   do
     {
      count = (count + 1) / 2;
      if(k_id < ls)
         temp[k_id] += (k_id < count && (k_id + count) < kunits ? temp[k_id + count] : 0);
      if(k_id + count < ls)
         temp[k_id + count] = 0;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
```

Now, we can convert the attention scores into probabilities by dividing each value by the total sum.

```
//--- score
   float sum = temp[0];
   if(isnan(sum) || isinf(sum) || sum <= 1.2e-7f)
      sum = 1;
   sc /= sum;
   score[shift_s] = sc;
   barrier(CLK_LOCAL_MEM_FENCE);
```

The resulting probabilities are written to the corresponding elements of the global output buffer, and we synchronize thread execution within the workgroup.

Finally, we compute the weighted sum of the _Value_ tensor elements for each item in the input sequence. We will weigh the values based on the attention coefficients calculated above. Within one element of the sequence, this operation is represented by multiplying the vector of attention coefficients by the _Value_ tensor, to which the authors of the _R-MAT_ framework added global bias tensor.

This is implemented using a loop system, where the outer loop iterates over the last dimension of the _Value_ tensor.

```
//--- out
   for(int d = 0; d < dimension; d++)
     {
      float val_v = v[shift_kv + d];
      float val_bv = bv[shift_kv + d];
      float val = sc * (val_v + val_bv);
      if(isnan(val) || isinf(val))
         val = 0;
```

Inside the loop, each thread computes its contribution to the corresponding element, and these contributions are aggregated using nested sequential reduction loops within the workgroup.

```
      //--- sum of value
      for(int cur_v = 0; cur_v < kunits; cur_v += ls)
        {
         if(k_id >= cur_v && k_id < (cur_v + ls))
           {
            int shift_local = k_id % ls;
            temp[shift_local] = (cur_v == 0 ? 0 : temp[shift_local]) + val;
           }
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      //---
      count = min(ls, (uint)kunits);
      do
        {
         count = (count + 1) / 2;
         if(k_id < count && (k_id + count) < kunits)
            temp[k_id] += temp[k_id + count];
         if(k_id + count < ls)
            temp[k_id + count] = 0;
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      while(count > 1);
```

The sum is then written to the appropriate element of the global result buffer by one of the threads.

```
      //---
      if(k_id == 0)
         out[shift_q + d] = (isnan(temp[0]) || isinf(temp[0]) ? 0 : temp[0]);
      barrier(CLK_LOCAL_MEM_FENCE);
     }
  }
```

Afterwards, we synchronize the threads again before moving on to the next loop iteration.

As discussed earlier, changes made to the _Softmax_ function do not affect the dependence of results on the input data. Therefore, we are able to reuse the existing backpropagation algorithms without any modifications.

#### 2.3 GELU Activation Function

In addition to numerical stabilization of the _Softmax_ function, the authors of the _LSEAttention_ framework also recommend using the _GELU_ activation function. The authors proposed two versions of this function. One of them is presented below.

![](https://c.mql5.com/2/156/3989336493215__1.png)

Implementing this activation function is quite simple. We just add the new variant to our existing activation function handler.

```
float Activation(const float value, const int function)
  {
   if(isnan(value) || isinf(value))
      return 0;
//---
   float result = value;
   switch(function)
     {
      case 0:
         result = tanh(clamp(value, -20.0f, 20.0f));
         break;
      case 1:  //Sigmoid
         result = 1 / (1 + exp(clamp(-value, -20.0f, 20.0f)));
         break;
      case 2:  //LReLU
         if(value < 0)
            result *= 0.01f;
         break;
      case 3:  //SoftPlus
         result = (value >= 20.0f ? 1.0f : (value <= -20.0f ? 0.0f : log(1 + exp(value))));
         break;
      case 4:  //GELU
         result = value / (1 + exp(clamp(-1.702f * value, -20.0f, 20.0f)));
         break;
      default:
         break;
     }
//---
   return result;
  }
```

However, behind the apparent simplicity of the feed-forward pass, there is a more complex task of implementing the backpropagation pass. This is because the derivative of GELU depends on the original input and the sigmoid function. Neither of them is available in our standard implementation.

![](https://c.mql5.com/2/156/1058039140204__1.png)

Moreover, it is not possible to accurately express the derivative of the GELU function based solely on the result of the feed-forward pass. Therefore, we had to resort to certain heuristics and approximations.

Let\\s begin by recalling the shape of the sigmoid function.

![](https://c.mql5.com/2/156/4573797182336__1.png)

For input values greater than 5, the sigmoid approaches 1, and for inputs less than –5, it approaches 0. Therefore, for sufficiently negative values of _X_, the derivative of _GELU_ tends toward 0, as the left-hand factor of the derivative equation approaches zero. For large positive values of _X_, the derivative tends toward 1, as both multiplicative factors converge to 1. This is confirmed by the graph shown below.

![](https://c.mql5.com/2/156/5847461492021__1.png)

Guided by this understanding, we approximate the derivative as the sigmoid of the feed-forward pass result multiplied by 5. This method offers fast computation and produces a good approximation for _GELU_ outputs greater than or equal to 0. However, for negative output values, the derivative is fixed at 0.5, due to which further training of the model cannot be continued. In reality, the derivative should approach 0, effectively blocking the propagation of the error gradient.

![](https://c.mql5.com/2/156/6396699166719__1.png)

![](https://c.mql5.com/2/156/830665518012__1.png)

The decision has been made. Let's get started with implementation. To do this, we added another case to the derivative computation function.

```
float Deactivation(const float grad, const float inp_value, const int function)
  {
   float result = grad;
//---
   if(isnan(inp_value) || isinf(inp_value) ||
      isnan(grad) || isinf(grad))
      result = 0;
   else
      switch(function)
        {
         case 0: //TANH
            result = clamp(grad + inp_value, -1.0f, 1.0f) - inp_value;
            result *= 1.0f - pow(inp_value, 2.0f);
            break;
         case 1:  //Sigmoid
            result = clamp(grad + inp_value, 0.0f, 1.0f) - inp_value;
            result *= inp_value * (1.0f - inp_value);
            break;
         case 2: //LReLU
            if(inp_value < 0)
               result *= 0.01f;
            break;
         case 3:  //SoftPlus
            result *= Activation(inp_value, 1);
            break;
         case 4:  //GELU
            if(inp_value < 0.9f)
               result *= Activation(5 * inp_value, 1);
            break;
         default:
            break;
        }
//---
   return clamp(result, -MAX_GRAD, MAX_GRAD);
  }
```

Note that we compute the activation derivative only if the result of the feed-forward pass is less than 0.9. In all other cases, the derivative is assumed to be 1, which is accurate. This allows us to reduce the number of operations during gradient propagation.

The authors of the framework suggest using the _GELU_ function as the non-linearity between layers in the _FeedForward_ block. In our _CNeuronRMAT_ class, this block is implemented using a feedback convolutional module _[CResidualConv](https://www.mql5.com/en/articles/14505#para31)_. We modify the activation function used between layers within this module. This operation is done in the class initialization method. The specific update is underlined in the code.

```
bool CResidualConv::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                         uint window, uint window_out, uint count,
                         ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window_out * count, optimization_type, batch))
      return false;
//---
   if(!cConvs[0].Init(0, 0, OpenCL, window, window, window_out, count, optimization, iBatch))
      return false;
   if(!cNorm[0].Init(0, 1, OpenCL, window_out * count, iBatch, optimization))
      return false;
   cNorm[0].SetActivationFunction(GELU);
   if(!cConvs[1].Init(0, 2, OpenCL, window_out, window_out, window_out, count, optimization, iBatch))
      return false;
   if(!cNorm[1].Init(0, 3, OpenCL, window_out * count, iBatch, optimization))
      return false;
   cNorm[1].SetActivationFunction(None);
//---

........
........
........
//---
   return true;
  }
```

With this, we complete the implementation of the techniques proposed by the authors of the _LSEAttention_ framework. The full code of all modification can be found in the attachment, along with the full code for all programs used in preparing this article.

It should be noted that all environment interaction and model training programs were fully reused from the previous article. Similarly, the model architecture was left unchanged. This makes it all the more interesting to assess the impact of the introduced optimizations, since the only difference lies in the algorithmic improvements.

### 3\. Testing

In this article, we implemented optimization techniques for the vanilla Transformer algorithm, as proposed by the authors of the _LSEAttention_ framework, for time series forecasting. As previously stated, this work differs from our earlier studies. We did not create new neural layers, as done before. Instead, we integrated the proposed improvements into previously implemented components. In essence, we took the _HypDiff_ framework implemented in the previous article and incorporated algorithmic optimizations that did not alter the model architecture. We also changed the activation function in the _FeedForward_ block. These adjustments primarily affected the internal computation mechanisms by enhancing numerical stability. Naturally, we are interested in how these changes impact model training outcomes.

To ensure a fair comparison, we replicated the _HypDiff_ model training algorithm in full. The same training dataset was used. However, this time we did not perform iterative updates to the training set. While this might slightly degrade training performance, it allows for an accurate comparison of the model before and after algorithm optimization.

The models were evaluated using real historical data from Q1 of 2024. The test results are presented below.

![](https://c.mql5.com/2/156/793670743582__1.png)![](https://c.mql5.com/2/156/4415699954322__1.png)

It should be noted that the model performance before and after modification was quite similar. During the test period, the updated model executed 24 trades. It deviated from the baseline model by only one trade, which falls within the margin of error. Both models made 13 profitable trades. The only visible improvement was the absence of a drawdown in February.

### Conclusion

The _LSEAttention_ method represents an evolution of attention mechanisms, particularly effective in tasks that demand high resilience to noise and data variability. The main advantage of _LSEAttention_ lies in the use of logarithmic smoothing, implemented via the _Log-Sum-Exp_ function. This allows the model to avoid issues of numerical overflow and vanishing gradients, which are critical in deep neural networks.

In the practical section, we implemented the proposed approaches in _MQL5_, integrating them into previously developed modules. We trained and tested the models using real historical data. Based on the test results, we can conclude that these methods improve the stability of the model training process.

#### References

- [LSEAttention is All You Need for Time Series Forecasting](https://www.mql5.com/go?link=https://arxiv.org/abs/2410.23749 "LSEAttention is All You Need for Time Series Forecasting")
- [Other articles from this series](https://www.mql5.com/en/search#!keyword=neural%20networks&author=DNG&method=2)

#### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | Research.mq5 | Expert Advisor | Expert Advisor for collecting examples |
| 2 | ResearchRealORL.mq5 | Expert Advisor | Expert Advisor for collecting examples using the Real-ORL method |
| 3 | Study.mq5 | Expert Advisor | Model training Expert Advisor |
| 4 | Test.mq5 | Expert Advisor | Model testing Expert Advisor |
| 5 | Trajectory.mqh | Class library | System state description structure |
| 6 | NeuroNet.mqh | Class library | A library of classes for creating a neural network |
| 7 | NeuroNet.cl | Library | OpenCL program code library |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/16360](https://www.mql5.com/ru/articles/16360)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16360.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/16360/mql5.zip "Download MQL5.zip")(2123.05 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Neural Networks in Trading: Hybrid Graph Sequence Models (GSM++)](https://www.mql5.com/en/articles/17279)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Final Part)](https://www.mql5.com/en/articles/17241)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Chimera)](https://www.mql5.com/en/articles/17210)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model (Final Part)](https://www.mql5.com/en/articles/17157)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model](https://www.mql5.com/en/articles/17142)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Final Part)](https://www.mql5.com/en/articles/17104)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Hidformer)](https://www.mql5.com/en/articles/17069)

**[Go to discussion](https://www.mql5.com/en/forum/491017)**

![Price Action Analysis Toolkit Development (Part 31): Python Candlestick Recognition Engine (I) — Manual Detection](https://c.mql5.com/2/156/18789-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 31): Python Candlestick Recognition Engine (I) — Manual Detection](https://www.mql5.com/en/articles/18789)

Candlestick patterns are fundamental to price-action trading, offering valuable insights into potential market reversals or continuations. Envision a reliable tool that continuously monitors each new price bar, identifies key formations such as engulfing patterns, hammers, dojis, and stars, and promptly notifies you when a significant trading setup is detected. This is precisely the functionality we have developed. Whether you are new to trading or an experienced professional, this system provides real-time alerts for candlestick patterns, enabling you to focus on executing trades with greater confidence and efficiency. Continue reading to learn how it operates and how it can enhance your trading strategy.

![Non-linear regression models on the stock exchange](https://c.mql5.com/2/103/Nonlinear_regression_models_on_the_stock_exchange___LOGO.png)[Non-linear regression models on the stock exchange](https://www.mql5.com/en/articles/16473)

Non-linear regression models on the stock exchange: Is it possible to predict financial markets? Let's consider creating a model for forecasting prices for EURUSD, and make two robots based on it - in Python and MQL5.

![Market Profile indicator](https://c.mql5.com/2/103/Learning_about_the_Market_Profile_indicator___LOGO.png)[Market Profile indicator](https://www.mql5.com/en/articles/16461)

In this article, we will consider Market Profile indicator. We will find out what lies behind this name, try to understand its operation principles and have a look at its terminal version (MarketProfile).

![From Basic to Intermediate: Union (II)](https://c.mql5.com/2/101/Do_bwsico_ao_intermedisrio_Uniho_II.png)[From Basic to Intermediate: Union (II)](https://www.mql5.com/en/articles/15503)

Today we have a very funny and quite interesting article. We will look at Union and will try to solve the problem discussed earlier. We'll also explore some unusual situations that can arise when using union in applications. The materials presented here are intended for didactic purposes only. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/16360&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069565511471138582)

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