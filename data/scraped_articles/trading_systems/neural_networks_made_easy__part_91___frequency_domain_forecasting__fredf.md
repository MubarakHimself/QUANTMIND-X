---
title: Neural Networks Made Easy (Part 91): Frequency Domain Forecasting (FreDF)
url: https://www.mql5.com/en/articles/14944
categories: Trading Systems, Expert Advisors, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T19:07:22.057397
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=eityuweaikmzxtuxnminhpbqplqopipd&ssn=1769184440171433915&ssn_dr=0&ssn_sr=0&fv_date=1769184440&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14944&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Neural%20Networks%20Made%20Easy%20(Part%2091)%3A%20Frequency%20Domain%20Forecasting%20(FreDF)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918444073925189&fz_uniq=5070025974914944696&sv=2552)

MetaTrader 5 / Examples


### Introduction

Forecasting time series of future prices is critical in various financial market scenarios. Most of the methods that currently exist are based on certain autocorrelation in the data. In other words, we exploit the presence of correlation between time steps that exists both in the input data and in the predicted values.

Among the models gaining popularity are those based on the _Transformer_ architecture that use _Self-Attention_ mechanisms for dynamic autocorrelation estimation. Also, we see an increasing interest in the use of frequency analysis in forecasting models. The representation of the sequence of input data in the frequency domain helps avoid the complexity of describing autocorrelation and improves the efficiency of various models.

Another important aspect is the autocorrelation in the sequence of predicted values. Obviously, the predicted values are part of a larger time series, which includes the analyzed and predicted sequences. Therefore, the predicted values preserve the correlation of the analyzed data. But this phenomenon is often ignored in modern forecasting methods. In particular, modern methods predominantly use the _Direct Forecast_ ( _DF_) paradigm, which generates multi-stage forecasts simultaneously. This implicitly assumes the independence of the steps in the sequence of predicted values. This mismatch between model assumptions and data features results in suboptimal forecast quality.

One of the solutions to this problem was proposed in the paper " _[FreDF: Learning to Forecast in Frequency Domain](https://www.mql5.com/go?link=https://arxiv.org/abs/2402.02399 "https://arxiv.org/abs/2402.02399")_". The authors of the paper proposed a direct forecast method with frequency gain ( _FreDF_). It clarifies the _DF_ paradigm by aligning the predicted values and the sequence of labels in the frequency domain. When moving to the frequency domain, where the bases are orthogonal and independent, the influence of autocorrelation is effectively reduced. Thus, _FreDF_ combats the inconsistency between the assumption about _DF_ and the existence of autocorrelation of labels, while maintaining the advantages of DF.

The authors of the method tested its effectiveness in a series of experiments, which demonstrated the significant superiority of the proposed approach over modern methods.

### 1\. FreDF Algorithm

The _DF_ paradigm uses a multiple-output model _ɡ_ _θ_ for generating _T_-step forecasts _Ŷ_ = _ɡθ_( _X_). Let _Yt_ be the _t_-th step of _Y_, and _Yt_( _n_) be the _n_-th sample observation. The model parameters _θ_ are optimized by minimizing the mean squared error (MSE):

![](https://c.mql5.com/2/78/3655556439308.png)

The _DF_ paradigm computes the forecast error at each step independently, treating each element of the sequence as a separate task. However, this approach oversights the autocorrelation present within _Y_, which contradicts the presence of autocorrelation of labels. As a consequence, this results in a biased likelihood and a deviation from the maximum likelihood principle during model training.

One strategy to overcome this limitation is to represent the sequence of labels in a transformed domain formed by orthogonal bases. In particular, this can be effectively implemented using the Fourier transform, which projects the sequence onto orthogonal bases associated with different frequencies. By transforming the label sequence into the orthogonal frequency domain, it is possible to effectively reduce the dependence on label autocorrelation.

![](https://c.mql5.com/2/78/2028757067137.png)

where _i_ is the imaginary unit defined as √(-1),

_exp_(•) is the Fourier basis associated with the frequency _k_ which is orthogonal for different _k_ values.

Due to the orthogonality of the basis, the frequency domain representation of the label sequence bypasses the dependence arising from autocorrelation in the time domain. This highlights the potential of frequency-domain prediction learning.

With the classical use of _DF_ approaches, at a given time stamp _n_, the historical sequence _X_ _n_ is input into the model to generate _T_-step forecasts, denoted as _Ŷ_ _n_ = _ɡ_ _θ_( _Xn_). The forecast error in the time domain _Ltmp_ is calculated.

In addition to the classical approach, the authors of the _FreDF_ method propose to transform predicted values and label sequences into the frequency domain. Then the prediction error in the frequency domain is calculated using the following formula:

![](https://c.mql5.com/2/78/3113317412142.png)

Here each term of the summation is a matrix of complex numbers _A_; \| _A_ \| denotes the operation of computing and summing the modulus of each element in the matrix. In this case, the modulus of a complex number _a = ar \+ i ai_ is computed as √(ar^2 + ai^2).

Please note that due to the different numerical characteristics of the label sequence in the frequency domain, the authors of the _FreDF_ method do not use the squared loss form ( _MSE_), as is typical for time domain loss error calculations. Specifically, different frequency components often have very different magnitudes, with lower frequencies having higher volumes by several orders of magnitude compared to higher frequencies, which makes squared loss methods unstable.

The prediction errors in the time and frequency domains are combined using the coefficient _α_ in the range of values \[0,1\], which controls the relative strength of the frequency domain equalization:

![](https://c.mql5.com/2/78/2847899706842.png)

_FreDF_ bypasses the effect of autocorrelation of target values by aligning the generated predicted values and the sequence of labels in the frequency domain. It also preserves the advantages of _D.F._, such as efficient output and multitasking capabilities. The notable feature of _FreDF_ is its compatibility with various forecasting models and transformations. This flexibility significantly expands the potential scope of _FreDF_ application.

The author's visualization of the method is presented below.

![](https://c.mql5.com/2/78/5260010595231.png)

### 2\. Implementing in MQL5

After considering the theoretical aspects of the proposed _FreDF_ method, let's move on to the practical part of our article, in which we will implement our vision of the approach. From the theoretical description presented above, it can be concluded that the proposed approach does not introduce any specific design features into the model architecture. Moreover, it does not affect the actual operation of the model. Its effect can only be seen during the model training process. Probably the proposed _FreDF_ method can be compared to some complex loss function. So, we will use it to train the model whose target labels, according to our a priori knowledge, have autocorrelation dependencies.

Before we begin to build a new object for implementing the proposed approaches, it is worth noting that the authors of the method used the Fourier transform to transform data from a time series to a frequency domain. It must be said that the _FreDF_ method is quite flexible. It also works well in combination with other methods transforming data into the orthogonal domain. The authors of the method conducted a series of experiments to prove its effectiveness when using other transformations. The results of these experiments are presented below.

![](https://c.mql5.com/2/78/5206036690760.png)

As can be seen, the models using the Fourier transform show better results.

I would like to draw your attention to the coefficient _α_. Its value of about 0.8 seems optimal based on the results of the experiments. It should be noted that if forecasting is only performed in the frequency domain (using _α_ equal to 1), according to the results of the same experiments, the model accuracy decreases.

Thus, we can conclude that in order to obtain an optimal time series forecasting model, the training process should include both the time and frequency domains of the signal under study. Different representations allow us to obtain more information about the signal and, as a result, train a more efficient model.

But let's get back to our implementation. According to the results of the experiments conducted by the method authors, the Fourier transform allows training models with a smaller forecast error. In the previous [article](https://www.mql5.com/en/articles/14913#para31), we have already implemented the direct and reverse fast Fourier transform. We can use these developments in our new implementation.

To implement the _FreDF_ approaches, we will create a new class _CNeuronFreDFOCL_, which will inherit the main functionality from the neural layer base class _CNeuronBaseOCL_. The structure of the new class is shown below.

```
class CNeuronFreDFOCL   :  public CNeuronBaseOCL
  {
protected:
   uint              iWindow;
   uint              iCount;
   uint              iFFTin;
   bool              bTranspose;
   float             fAlpha;
   //---
   CBufferFloat      cForecastFreRe;
   CBufferFloat      cForecastFreIm;
   CBufferFloat      cTargetFreRe;
   CBufferFloat      cTargetFreIm;
   CBufferFloat      cLossFreRe;
   CBufferFloat      cLossFreIm;
   CBufferFloat      cGradientFreRe;
   CBufferFloat      cGradientFreIm;
   CBufferFloat      cTranspose;
   //---
   virtual bool      FFT(CBufferFloat *inp_re, CBufferFloat *inp_im,
                         CBufferFloat *out_re, CBufferFloat *out_im, bool reverse = false);
   virtual bool      Transpose(CBufferFloat *inputs, CBufferFloat *outputs, uint rows, uint cols);
   virtual bool      FreqMSA(CBufferFloat *target, CBufferFloat *forecast, CBufferFloat *gradient);
   virtual bool      CumulativeGradient(CBufferFloat *gradient1, CBufferFloat *gradient2,
                                        CBufferFloat *cummulative, float alpha);
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL)   {  return true;   }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronFreDFOCL(void)   {};
                    ~CNeuronFreDFOCL(void)   {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint count, float alpha, bool need_transpose = true,
                          ENUM_OPTIMIZATION optimization_type = ADAM, uint batch = 1);
   virtual bool      calcOutputGradients(CArrayFloat *Target, float &error);
   //---
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual int       Type(void)        const                      {  return defNeuronFreDFOCL; }
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
```

The presented structure of the new class has two notable features:

- Internal objects are represented only by data buffers and there are no internal layers
- The _calcOutputGradients_ method is overridden

One more implicit feature can be mentioned here: this object does not contain trainable parameters, which is quite rare. All these features are related to the purpose of the class: we are creating a class of a complex loss function, not a trainable neural layer. And the _calcOutputGradients_ method in our neural layer architecture is responsible for calculating the deviations of predicted values from the target ones. We will get acquainted with the purpose of the internal objects and variables while implementing the methods.

All objects are declared statically, allowing us to leave the class constructor and destructor "empty". All operations related to freeing the memory will be performed by the system itself.

Class objects are initialized in the _Init_ method. As usual, in the parameters of this method, we pass the main constants that define the architecture of the class. Here we have:

- a _window_ describing one element of the input data,
- _count_ for the number of elements in the sequence,
- _alpha_ coefficient for the force of equalization of the frequency and time domains,
- the _need\_transpose_ flag indicating the need to transpose data for frequency conversion.

This object will be used at the output of the model. Therefore, the input is the predicted values generated by our model. Data must be provided in a format consistent with the target results. The _window_ and _count_ parameters correspond to both predicted and target values. We also provide the user with the ability to transform data into the frequency domain in a different plane. This is why we introduced the _need\_transpose_ flag.

I would like to cite here the results of some other [experiments](https://www.mql5.com/go?link=https://arxiv.org/pdf/2402.02399 "https://arxiv.org/pdf/2402.02399") conducted by the authors of the method. They tested the performance of the models when comparing frequency characteristics in unitary time series of a multivariate sequence ( _T_), in terms of individual time steps ( _D_) and for the total sequence ( _2D_).

![](https://c.mql5.com/2/78/5404870816567.png)

The best results were demonstrated by the model with representing frequency characteristics of the general aggregate sequence. The comparison of frequency characteristics of individual time steps turned out to be the outsider of the experiment. The analysis of frequency characteristics of unitary time series was second best, slightly behind the leader.

In our implementation, we provide the user with the ability to select the measurement for frequency conversion by specifying the corresponding _need\_transpose_ flag value. To compare 2-dimensional frequency characteristics, specify the size of the entire sequence in the _window_ parameter and use the following values for the remaining parameters:

- _count:_ 1,
- _need\_transpose: false._

```
bool CNeuronFreDFOCL::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                           uint window, uint count, float Alpha, bool need_transpose = true,
                           ENUM_OPTIMIZATION optimization_type = ADAM, uint batch = 1)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * count, optimization_type, batch))
      return false;
```

In the method body, we first call the relevant parent class method that has the same name and check the result of the operations. Again, the parent class implements the necessary set of controls, including the size of the neural layer being created. For the layer size, we specify the product of variables _window_ and _count_. Obviously, if you specify a zero value in just one of them, the entire product will be equal to "0", and the parent class method will fail.

After successful execution of the parent class method, we save the obtained values in local variables.

```
   bTranspose = need_transpose;
   iWindow = window;
   iCount = count;
   fAlpha = MathMax(0, MathMin(Alpha, 1));
   activation = None;
```

As we have seen earlier, for the fast Fourier transform, we need buffers with a size of a power of 2. Let's calculate the sizes of data buffers:

```
//--- Calculate FFTsize
   uint size = (bTranspose ? count : window);
   int power = int(MathLog(size) / M_LN2);
   if(MathPow(2, power) != size)
      power++;
   iFFTin = uint(MathPow(2, power));
```

The next step is to initialize the internal data buffers. First, we initialize the frequency response buffers of the predicted values. We use a 2 data buffer design. One buffer is used for recording data of the real component, and the second one is used for the imaginary one.

```
//---
   uint n = (bTranspose ? iWindow : iCount);
   if(!cForecastFreRe.BufferInit(iFFTin * n, 0) || !cForecastFreRe.BufferCreate(OpenCL))
      return false;
   if(!cForecastFreIm.BufferInit(iFFTin * n, 0) || !cForecastFreIm.BufferCreate(OpenCL))
      return false;
```

Next, we create similar buffers for the frequency characteristics of the target values:

```
   if(!cTargetFreRe.BufferInit(iFFTin * n, 0) || !cTargetFreRe.BufferCreate(OpenCL))
      return false;
   if(!cTargetFreIm.BufferInit(iFFTin * n, 0) || !cTargetFreIm.BufferCreate(OpenCL))
      return false;
```

We write the prediction error into buffers _cLossFreeRe_ and _cLossFree:_

```
   if(!cLossFreRe.BufferInit(iFFTin * n, 0) || !cLossFreRe.BufferCreate(OpenCL))
      return false;
   if(!cLossFreIm.BufferInit(iFFTin * n, 0) || !cLossFreIm.BufferCreate(OpenCL))
      return false;
```

Please note the importance of comparing both components of the frequency characteristics. For correct forecasting of time series, both the amplitudes and phases of the frequency characteristics of the time series are important.

It is also necessary to create buffers for recording error gradients at the level of predicted time series values:

```
   if(!cGradientFreRe.BufferInit(iFFTin * n, 0) || !cGradientFreRe.BufferCreate(OpenCL))
      return false;
   if(!cGradientFreIm.BufferInit(iFFTin * n, 0) || !cGradientFreIm.BufferCreate(OpenCL))
      return false;
```

In order to save memory, we can exclude buffers _cGradientFreeRe_ and _cGradientFreeIm_. They can be easily replaced, for example, with buffers _cForecastFreeRe_ and _cForecastFreeIm_. But their presence makes the code more readable. Also, the amount of memory they use in our case is not critical.

Finally, we will create a temporary buffer to write the transposed values, if required:

```
      if(!cTranspose.BufferInit(iWindow * iCount, 0) || !cTranspose.BufferCreate(OpenCL))
         return false;
//---
   return true;
  }
```

After data initialization, we usually create a feed-forward pass method. It was already said above that an object of this class does not perform operations with data during operation. As you know, the feed-forward method describes the model's operating mode. We could redefine the feed-forward pass method with a "dummy", but then how would we transfer data? As always, we would like to minimize the data copying process, because the data volume can be different, and process organization adds "overhead costs". In this context, we make the feed-forward pass method as simple as possible. In this method, we only check the correspondence of pointers to result buffers in the current and previous layers. If necessary, we replace the pointer in the current layer with the result buffer of the previous layer.

```
bool CNeuronFreDFOCL::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || !NeuronOCL.getOutput())
      return false;
   if(NeuronOCL.getOutput() != Output)
     {
      Output.BufferFree();
      delete Output;
      Output = NeuronOCL.getOutput();
     }
//---
   return true;
  }
```

Thus, we replace the pointer to one buffer instead of transferring data regardless of its volume. Please note that control is performed on each pass, and the data buffers are replaced only on the first one.

We implement the main functionality of the class for the backpropagation pass. Let's first do a little preparatory work. To fully implement the required functionality, we will create 2 small kernels on the _OpenCL_ program side.

Authors of the _FreDF_ method recommend using _MAE_ as a loss function when estimating deviations in the frequency domain. They also note a decrease in the training stability when using _MSE_. Let me remind you that our basic neural layer class _CNeuronBaseOCL_ uses exactly _MSE_ to determine the error gradient. So, we need to create a kernel to determine the forecast error gradient using _MAE_. From a mathematical point of view, this is quite simple: we just need to subtract the vector of predicted values from the vector of target labels.

```
__kernel void GradientMSA(__global float *matrix_t,
                          __global float *matrix_o,
                          __global float *matrix_g
                         )
  {
   int i = get_global_id(0);
   matrix_g[i] = matrix_t[i] - matrix_o[i];
  }
```

After determining the error gradient in the frequency and time domains, we need to combine the error gradients using the temperature coefficient. Let's implement this functionality in the _CumulativeGradient_ kernel, which should be quite easy to understand, I think.

```
__kernel void CumulativeGradient(__global float *gradient_freq,
                                 __global float *gradient_tmp,
                                 __global float *gradient_out,
                                 float alpha
                                )
  {
   int i = get_global_id(0);
   gradient_out[i] = alpha * gradient_freq[i] + (1 - alpha) * gradient_tmp[i];
  }
```

Let me remind you that to transform data from the time domain to the frequency domain and back, we will use the fast Fourier transform algorithm, which we implemented in the previous [article](https://www.mql5.com/en/articles/14913#para31). That article provides a description of the algorithm used and the method for placing the kernel in the execution queue.

Now we will not consider the algorithms for methods of placing kernels in the execution queue. They all follow the same procedure, which has already been presented several times in the articles within this series, including the [previous](https://www.mql5.com/en/articles/14913#para31) one.

Let's consider the _CNeuronFreDFOCL::calcOutputGradients_ method, which implements the main functionality of our class. As you know, according to the structure of our models, this method determines the deviation of the predicted values from the target labels. In the method parameters, we receive a pointer to the buffer with target values. After performing the method operations, we need to save the error gradient into the corresponding buffer of the current layer.

```
bool CNeuronFreDFOCL::calcOutputGradients(CArrayFloat *Target, float &error)
  {
   if(!Target)
      return false;
   if(Target.Total() < Output.Total())
      return false;
```

In the method body, we check the correctness of the received pointer to the target value buffer. Also, its size must be no less than the model's result tensor.

Since the received buffer may not have a copy of itself on the _OpenCL_ context side, we have to create it there for subsequent calculations. However, for more economical use of _OpenCL_ context resources, we will transfer the obtained data to the already created gradient buffer.

```
   if(Target.Total() == Output.Total())
     {
      if(!Gradient.AssignArray(Target))
         return false;
     }
   else
     {
      for(int i = 0; i < Output.Total(); i++)
        {
         if(!Gradient.Update(i, Target.At(i)))
            return false;
        }
     }
   if(!Gradient.BufferWrite())
      return false;
```

Here there are 2 possible developments. If the sizes of the target label and predicted value buffers are equal, then we use the existing copy method. Otherwise, we use a loop to transfer the required number of values. In any case, after copying the data, we transfer it to the _OpenCL_ context memory.

The obtained data is then used to calculate deviations in both the time and frequency domains. Please paying attention that when calculating deviations in the time domain, the error gradient buffer of our layer will be overwritten by the calculated deviations, while the obtained target values will be completely lost. Therefore, before calculating the deviations in the time domain, we at least need to decompose the obtained time series of target labels into frequency components.

A time series can be decomposed into frequency characteristics in two dimensions. The one to be used is determined by the value of the _bTranspose_ flag. If the flag is set to _true_, we first transpose the model's result buffer and then decompose it into frequency responses:

```
   if(bTranspose)
     {
      if(!Transpose(Output, GetPointer(cTranspose), iWindow, iCount))
         return false;
      if(!FFT(GetPointer(cTranspose), NULL, GetPointer(cForecastFreRe), GetPointer(cForecastFreIm), false))
         return false;
```

We perform similar operations for the target label tensor:

```
      if(!Transpose(Gradient, GetPointer(cTranspose), iWindow, iCount))
         return false;
      if(!FFT(GetPointer(cTranspose), NULL, GetPointer(cTargetFreRe), GetPointer(cTargetFreIm), false))
         return false;
     }
```

If the _bTranspose_ flag value is _false_, then we perform the decomposition of the target and predicted values into the corresponding frequency characteristics without preliminary transposition:

```
   else
     {
      if(!FFT(Output, NULL, GetPointer(cForecastFreRe), GetPointer(cForecastFreIm), false))
         return false;
      if(!FFT(Gradient, NULL, GetPointer(cTargetFreRe), GetPointer(cTargetFreIm), false))
         return false;
     }
```

Once the frequency characteristics are determined, we can calculate deviations in both the time and frequency domains without worrying about losing target values.

```
   if(!FreqMSA(GetPointer(cTargetFreRe), GetPointer(cForecastFreRe), GetPointer(cLossFreRe)))
      return false;
   if(!FreqMSA(GetPointer(cTargetFreIm), GetPointer(cForecastFreIm), GetPointer(cLossFreIm)))
      return false;
   if(!FreqMSA(Gradient, Output, Gradient))
      return false;
```

Note that in the frequency domain, we determine deviations in both the real and imaginary parts of the frequency response. Because the value of the phase shift is no less important than the signal amplitude. However, we cannot directly approximate the gradients of time and frequency domain errors. Obviously, the data is incomparable. Therefore, we first need to return the gradients of the frequency response error to the time domain. For this, we will use the inverse Fourier transform.

```
   if(!FFT(GetPointer(cLossFreRe), GetPointer(cLossFreIm), GetPointer(cGradientFreRe), GetPointer(cGradientFreIm), true))
      return false;
```

The error gradients of the time and frequency domains have been brought into a comparable form. Now, the measurement of frequency characteristic extraction depends on the value of the _bTranspose_ flag. Therefore, we need to transform the frequency domain error gradient according to the flag value. Only then can we determine the cumulative error gradient of our model.

```
   if(bTranspose)
     {
      if(!Transpose(GetPointer(cGradientFreRe), GetPointer(cTranspose), iCount, iWindow))
         return false;
      if(!CumulativeGradient(GetPointer(cTranspose), Gradient, Gradient, fAlpha))
         return false;
     }
   else
      if(!CumulativeGradient(GetPointer(cGradientFreRe), Gradient, Gradient, fAlpha))
         return false;
//---
   return true;
  }
```

Do not forget to control the results at each step. The logical value of the operations is returned to the caller.

After determining the error gradient at the model output, we need to pass it to the previous layer. We implement this functionality in the _CNeuronFreDFOCL::calcInputGradients_ method, which receives a pointer to the object of the previous neural layer in its parameters.

Remember that our layer does not contain trainable parameters. During the feed-forward pass, we have replaced the data buffer and are showing the values from the previous layer as the results. What is the purpose of this method? It is very simple. We just need to adjust the cumulative error gradient calculated above to the activation function of the previous layer.

```
bool CNeuronFreDFOCL::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      return false;
//---
   return DeActivation(NeuronOCL.getOutput(), NeuronOCL.getGradient(), Gradient, NeuronOCL.Activation());
  }
```

Since our class does not contain trainable parameters, we redefine the _updateInputWeights_ method with an "empty stub".

The absence of tranable parameters in the class also influence file operation methods. Because we don't need to store irrelevant internal objects. Therefore, when saving data, we only call the parent class method of the same name.

```
bool CNeuronFreDFOCL::Save(const int file_handle)
  {
   if(!CNeuronBaseOCL::Save(file_handle))
      return false;
```

We save the values of the variables describing the design features of the object:

```
   if(FileWriteInteger(file_handle, int(iWindow)) < INT_VALUE)
      return false;
   if(FileWriteInteger(file_handle, int(iCount)) < INT_VALUE)
      return false;
   if(FileWriteInteger(file_handle, int(iFFTin)) < INT_VALUE)
      return false;
   if(FileWriteInteger(file_handle, int(bTranspose)) < INT_VALUE)
      return false;
   if(FileWriteFloat(file_handle, fAlpha) < sizeof(float))
      return false;
//---
   return true;
  }
```

The _Load_ algorithm for restoring an object from a data file looks a little more complicated. Here we first restore the elements of the parent class:

```
bool CNeuronFreDFOCL::Load(const int file_handle)
  {
   if(!CNeuronBaseOCL::Load(file_handle))
      return false;
```

Then we load the variable data in the order they are saved, remembering to check when the end of the data file is reached:

```
   if(FileIsEnding(file_handle))
      return false;
   iWindow = uint(FileReadInteger(file_handle));
   if(FileIsEnding(file_handle))
      return false;
   iCount = uint(FileReadInteger(file_handle));
   if(FileIsEnding(file_handle))
      return false;
   iFFTin = uint(FileReadInteger(file_handle));
   if(FileIsEnding(file_handle))
      return false;
   bTranspose = bool(FileReadInteger(file_handle));
   if(FileIsEnding(file_handle))
      return false;
   fAlpha = FileReadFloat(file_handle);
```

Then we need to initialize the nested objects in accordance with the loaded parameters of the class architecture. Object are initialized similarly to the algorithm for initializing a new class instance:

```
   uint n = (bTranspose ? iWindow : iCount);
   if(!cForecastFreRe.BufferInit(iFFTin * n, 0) || !cForecastFreRe.BufferCreate(OpenCL))
      return false;
   if(!cForecastFreIm.BufferInit(iFFTin * n, 0) || !cForecastFreIm.BufferCreate(OpenCL))
      return false;
   if(!cTargetFreRe.BufferInit(iFFTin * n, 0) || !cTargetFreRe.BufferCreate(OpenCL))
      return false;
   if(!cTargetFreIm.BufferInit(iFFTin * n, 0) || !cTargetFreIm.BufferCreate(OpenCL))
      return false;
   if(!cLossFreRe.BufferInit(iFFTin * n, 0) || !cLossFreRe.BufferCreate(OpenCL))
      return false;
   if(!cLossFreIm.BufferInit(iFFTin * n, 0) || !cLossFreIm.BufferCreate(OpenCL))
      return false;
   if(!cGradientFreRe.BufferInit(iFFTin * n, 0) || !cGradientFreRe.BufferCreate(OpenCL))
      return false;
   if(!cGradientFreIm.BufferInit(iFFTin * n, 0) || !cGradientFreIm.BufferCreate(OpenCL))
      return false;
   if(bTranspose)
     {
      if(!cTranspose.BufferInit(iWindow * iCount, 0) || !cTranspose.BufferCreate(OpenCL))
         return false;
     }
   else
     {
      cTranspose.BufferFree();
      cTranspose.Clear();
     }
//---
   return true;
  }
```

This concludes the description of the methods of our new _CNeuronFreDFOCL_ class. You can see the full code of this class in the attachment.

After constructing the methods of the new class, we usually move on to describing the trainable model architecture. But in this article, we have built a rather unusual neural layer. We have implemented a complex loss function in the form of a neural layer. So, we can add the above created object to one of the models we trained earlier, retrain it and see how the results change. For my experiments I chose the _FEDformer_ model; its architecture is described _[here](https://www.mql5.com/en/articles/14858#para33)_. Let's add a new layer to it.

```
bool CreateEncoderDescriptions(CArrayObj *encoder)
  {
//---
........
........
//--- layer 17
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronFreDFOCL;
   descr.window = BarDescr;
   descr.count =  NForecast;
   descr.step = int(true);
   descr.probability = 0.8f;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
//---
   return true;
  }
```

After thinking about it for a while, I decided to expand the experiment. The authors of the _FreDF_ method proposed their own algorithm for using the dependencies in predicted results. Actually, there is also a dependence between the individual parameters of our Actor's results. For example, the volumes of buy and sell trades are mutually exclusive, because at any given time we only have an open position in one direction. Stop loss and take profit parameters determine the strength of the most likely upcoming move. Therefore, the take profit of a long position should be correlated to some extent with the stop loss of a short position and vice versa. Similar reasoning can be used to suggest dependencies in the predicted Critic values. So why not extend the experiment to the models mentioned? No sooner said than done. Adding a new layer to the Actor and Critic models:

```
bool CreateDescriptions(CArrayObj *actor, CArrayObj *critic)
  {
//---
   CLayerDescription *descr;
//---
   if(!actor)
     {
      actor = new CArrayObj();
      if(!actor)
         return false;
     }
   if(!critic)
     {
      critic = new CArrayObj();
      if(!critic)
         return false;
     }
//--- Actor
.........
.........
//--- layer 17
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronFreDFOCL;
   descr.window = NActions;
   descr.count =  1;
   descr.step = int(false);
   descr.probability = 0.8f;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- Critic
.........
.........
//--- layer 17
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronFreDFOCL;
   descr.window = NRewards;
   descr.count =  1;
   descr.step = int(false);
   descr.probability = 0.8f;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//---
   return true;
  }
```

Please note that in this case we are analyzing the frequency characteristics of the entire sequence of results, and not of individual unitary series.

Our implementation of the approaches proposed by the _FreDF_ method does not require any adjustments to the Expert Advisors used for model training and interaction with the environment. This means that to test the obtained results, we can use previously prepared Expert Advisors and training datasets.

### 3\. Testing

We have done quite a lot of work to implement the approaches proposed by the authors of _FreDF_ using _MQL5_. Now we move on to the final stage of our work: training and testing.

As mentioned above, we will train the models using the previously created Expert Advisor and pre-collected training data. In our articles, we train models on the historical data of the _EURUSD_ instrument with the _H1_ timeframe for year 2023.

First we train the model of the environment state _Encoder_. The model is trained to predict future states of the environment over a planning horizon determined by the _NForecast_ constant. In my experiment, I used 12 subsequent candles. The forecast is generated in the context of all analyzed parameters describing the state of the environment.

```
#define        NForecast               12            //Number of forecast
```

In the process of _Encoder_ training, we can see a reduction in the forecast error compared to a similar model without using _FreDF_ approaches. However, we did not perform a graphical comparison of the forecast results. Therefore, it is difficult to judge the actual quality of the forecast values. It should be noted here that, as strange as it may seem, our goal is not to obtain the most accurate forecasts of all the analyzed indicators. The _Actor_ model uses _Encoder's_ latent space to decide on optimal actions. The goal of the first stage of training is to obtain the most informative latent space of the _Encoder_, which would encode the most likely upcoming price movement.

As before, the _Encoder_ model analyzes only price movement, so during the first stage of training we do not need to update the training set.

In the second stage of our learning process, we search for the most optimal _Actor_ action policy. Here we run iterative training of _Actor_ and _Critic_ models, which alternates with updating the training dataset.

As a result of several iterations of _Actor_ policy training, we managed to get a model that can generate profit. We tested the performance of the trained model in the MetaTrader 5 strategy tester using real historical data for January 2024. The testing parameters fully corresponded to the parameters of the training dataset, including the instrument, timeframe and parameters of the analyzed indicators. The test results are presented in the screenshots below.

![](https://c.mql5.com/2/78/2118573809759.png)![](https://c.mql5.com/2/78/5657498958597.png)

Based on the testing results, we can notice a clear trend towards an increase in the account balance. During the testing period, the model executed 49 trades, 21 of which were closed with a profit. Yes, less than half of the positions were profitable. However, the average profitable trade is almost 2 times larger than the average losing trade. As a result, the profit factor of the model on the test dataset is 1.43 and the total income for the month is about 19%.

### Conclusion

In this article, we discussed the _FreDF_ method, which aims to improve time series forecasting. The authors of the method empirically substantiated that ignoring autocorrelation in the labeled sequence leads to a bias in the likelihood and a deterioration in the quality of forecasts in the current _DF_ paradigm. They presented a simple but effective modification of the current _DF_ paradigm, which takes into account autocorrelation by aligning forecast and label sequences in the frequency domain. The _FreDF_ method is compatible with various forecasting models and transformations, making it flexible and versatile.

In the practical part of the article, we implemented our vision of the proposed approaches in the _MQL5_ language. We supplemented the previously created _FEDformer_ model with proposed approaches and conducted training. Then we tested the trained model. The testing results suggest the effectiveness of the proposed approaches, since the addition of _FreDF_ had increased the efficiency of the model, all other things being equal.

I would like to note the flexibility of the _FreDF_ method, which allows it to be used effectively with a wide range of existing models.

### References

- [FreDF: Learning to Forecast in Frequency Domain](https://www.mql5.com/go?link=https://arxiv.org/abs/2402.02399 "https://arxiv.org/abs/2402.02399")
- [Other articles from this series](https://www.mql5.com/en/search#!keyword=Neural%20networks%20made%20easy&module=mql5_module_articles "https://www.mql5.com/en/search#!keyword=Neural%20networks%20made%20easy&module=mql5_module_articles")

### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | Research.mq5 | Expert Advisor | Example collection EA |
| 2 | ResearchRealORL.mq5 | Expert Advisor | EA for collecting examples using the Real-ORL method |
| 3 | Study.mq5 | Expert Advisor | Model training EA |
| 4 | StudyEncoder.mq5 | Expert Advisor | Encode Training EA |
| 5 | Test.mq5 | Expert Advisor | Model testing EA |
| 6 | Trajectory.mqh | Class library | System state description structure |
| 7 | NeuroNet.mqh | Class library | A library of classes for creating a neural network |
| 8 | NeuroNet.cl | Code Base | OpenCL program code library |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14944](https://www.mql5.com/ru/articles/14944)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14944.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/14944/mql5.zip "Download MQL5.zip")(1240.5 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/475734)**

![Elements of correlation analysis in MQL5: Pearson chi-square test of independence and correlation ratio](https://c.mql5.com/2/80/Pearson_chi-square_independence_test_and_correlation_ratio____LOGO.png)[Elements of correlation analysis in MQL5: Pearson chi-square test of independence and correlation ratio](https://www.mql5.com/en/articles/15042)

The article observes classical tools of correlation analysis. An emphasis is made on brief theoretical background, as well as on the practical implementation of the Pearson chi-square test of independence and the correlation ratio.

![Feature Engineering With Python And MQL5 (Part I): Forecasting Moving Averages For Long-Range AI Models](https://c.mql5.com/2/99/Feature_Engineering_With_Python_And_MQL5_Part_II__LOGO2.png)[Feature Engineering With Python And MQL5 (Part I): Forecasting Moving Averages For Long-Range AI Models](https://www.mql5.com/en/articles/16230)

The moving averages are by far the best indicators for our AI models to predict. However, we can improve our accuracy even further by carefully transforming our data. This article will demonstrate, how you can build AI Models capable of forecasting further into the future than you may currently be practicing without significant drops to your accuracy levels. It is truly remarkable, how useful the moving averages are.

![Self Optimizing Expert Advisor With MQL5 And Python (Part VI): Taking Advantage of Deep Double Descent](https://c.mql5.com/2/100/Self_Optimizing_Expert_Advisor_With_MQL5_And_Python_Part_VI__LOGO.png)[Self Optimizing Expert Advisor With MQL5 And Python (Part VI): Taking Advantage of Deep Double Descent](https://www.mql5.com/en/articles/15971)

Traditional machine learning teaches practitioners to be vigilant not to overfit their models. However, this ideology is being challenged by new insights published by diligent researches from Harvard, who have discovered that what appears to be overfitting may in some circumstances be the results of terminating your training procedures prematurely. We will demonstrate how we can use the ideas published in the research paper, to improve our use of AI in forecasting market returns.

![MQL5 Wizard Techniques you should know (Part 45): Reinforcement Learning with Monte-Carlo](https://c.mql5.com/2/99/MQL5_Wizard_Techniques_you_should_know_Part_45___LOGO.png)[MQL5 Wizard Techniques you should know (Part 45): Reinforcement Learning with Monte-Carlo](https://www.mql5.com/en/articles/16254)

Monte-Carlo is the fourth different algorithm in reinforcement learning that we are considering with the aim of exploring its implementation in wizard assembled Expert Advisors. Though anchored in random sampling, it does present vast ways of simulation which we can look to exploit.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=pczzaltbdgvnnnmrhbcylnwjdpzndkox&ssn=1769184440171433915&ssn_dr=0&ssn_sr=0&fv_date=1769184440&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14944&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Neural%20Networks%20Made%20Easy%20(Part%2091)%3A%20Frequency%20Domain%20Forecasting%20(FreDF)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918444073897922&fz_uniq=5070025974914944696&sv=2552)

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