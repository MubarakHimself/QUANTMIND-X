---
title: Training a multilayer perceptron using the Levenberg-Marquardt algorithm
url: https://www.mql5.com/en/articles/16296
categories: Trading, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T17:55:09.861993
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/16296&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068797146116849099)

MetaTrader 5 / Trading


### **Introduction**

The purpose of this article is to provide practicing traders with a very effective neural network training algorithm - a variant of the Newtonian optimization method known as the Levenberg-Marquardt algorithm. It is one of the fastest algorithms for training feed-forward neural networks, rivaled only by the Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) algorithm.

Stochastic optimization methods such as stochastic gradient descent (SGD) and Adam are well suited for offline training when the neural network overfits over long periods of time. If a trader using neural networks wants the model to quickly adapt to constantly changing trading conditions, he needs to retrain the network online at each new bar, or after a short period of time. In this case, the best algorithms are those that, in addition to information about the gradient of the loss function, also use additional information about the second partial derivatives, which allows finding a local minimum of the loss function in just a few training epochs.

At the moment, as far as I know, there is no publicly available implementation of the Levenberg-Marquardt algorithm in MQL5. It is time to fill this gap, and also, at the same time, briefly go over the well-known and simplest optimization algorithms, such as gradient descent, gradient descent with momentum, and stochastic gradient descent. At the end of the article, we will conduct a small test of the efficiency of the Levenberg-Marquardt algorithm and algorithms from the scikit-learn machine learning library.

### Dataset

All subsequent examples use synthetic data for ease of presentation. Time is used as the only predictor variable, and the target variable that we want to predict using the neural network is the function:

1 + sin(pi/4\*time) + NormDistr(0,sigma)

This function consists of a deterministic part, represented by a periodic component in the form of a sine, and a stochastic part - Gaussian white noise. Total 81 data points. Below is a graph of this function and its approximation by a three-layer perceptron.

![SyntheticData](https://c.mql5.com/2/150/SyntheticData__1.png)

Fig. 1. The target function and its approximation by a three-layer perceptron

### Gradient descent

Let's start with the implementation of the regular gradient descent, as the simplest method for training neural networks. I will use a very good example from the MQL5 Reference book as a template ( [Matrix and Vector Methods/Machine learning](https://www.mql5.com/en/docs/matrix/matrix_machine_learning "https://www.mql5.com/en/docs/matrix/matrix_machine_learning")). I modified it a bit, adding the ability to select the activation function for the last layer of the network and making the implementation of gradient descent universal, capable of learning not only on the quadratic loss function, as implicitly assumed in the example from the reference book, but on all available loss functions in MQL5. The loss function is central to training neural networks, and it is sometimes worth experimenting with different functions beyond just the quadratic loss. Here is the general equation for calculating the output layer error (delta):

![delta last layer](https://c.mql5.com/2/150/Delta-last__1.png)

here

- delta\_k — output layer error,
- E — loss function,
- g'(a\_k) — derivative of the activation function,
- a\_k — pre-activation of the last layer,
- y\_k — predicted value of the network.

```
//--- Derivative of the loss function with respect to the predicted value
matrix DerivLoss_wrt_y = result_.LossGradient(target,loss_func);
matrix deriv_act;
  if(!result_.Derivative(deriv_act, ac_func_last))
     return false;
 matrix  loss = deriv_act*DerivLoss_wrt_y;  // loss = delta_k
```

The partial derivatives of the loss function with respect to the predicted value of the network are calculated by the **LossGradient** function, while the derivative of the activation function is calculated by the **Derivative** function. In the reference example, the difference between the target and predicted value of the network multiplied by 2 is used as the error of the output layer.

```
matrix loss = (target - result_)*2;
```

In the machine learning literature, the error of each layer of the network is usually called delta(D2,D1 etc.) rather than loss (for example, see Bishop(1995)). From now on, I will use exactly this notation in the code.

How did we get this result? Here, it is implicitly assumed that the loss function is the sum of squared differences between the target and predicted values, rather than the mean squared error (MSE), which is additionally normalized by the size of the training sample. The derivative of this loss function is exactly equal to (target - result)\*2. Since the last layer of the network uses the identical activation function, whose derivative is equal to one, we arrive at this result. Therefore, those who want to use arbitrary loss functions and output layer activation functions to train the network need to use the above general equation.

Let's now train our network on the mean square loss function. Here, have displayed the graph on a logarithmic scale for clarity.

![Loss SD](https://c.mql5.com/2/150/SD_Loss__1.png)

Fig. 2. MSE loss function, gradient descent

On average, the gradient descent algorithm requires 1500-2000 epochs (i.e. passes over the entire training data set) to reach the minimum loss function threshold. In this case, I used two hidden layers with 5 neurons in each.

The red line on the graph indicates the minimum threshold of the loss function. It is defined as the variance of white Gaussian noise. Here I used noise with variance equal to 0.01 (0.1 sigma\* 0.1 sigma).

What happens if we allow the neural network model to learn below this minimum threshold? Then we will encounter such an undesirable phenomenon as network overfitting. It is pointless to try to get the loss function error on the training dataset below the minimum threshold, as this will affect the predictive power of the model on the test dataset. Here we are faced with the fact that it is impossible to predict a series more accurately than the statistical spread of that series allows. If we stop training above the minimum threshold, we will face another problem - the network will be undertrained. That is, one that was unable to fully capture the predictable component of the series.

As you can see, gradient descent needs to go through quite a few iterations to reach the optimal set of parameters. Note that our dataset is pretty simple. For real practical problems, the training time for gradient descent turns out to be unacceptable. One of the simplest ways to improve the convergence and speed of gradient descent is the momentum method.

### Gradient descent with momentum

The idea behind gradient descent with momentum is to smooth out the trajectory of the network parameters during training by averaging the parameters like a simple exponential average. Just as we smooth the time series of prices of financial instruments with an average in order to highlight the main direction, we also smooth the trajectory of a parametric vector that moves toward the point of a local minimum of our loss function. To better visualize this, let's look at a graph that shows how the values of the two parameters changed - from the beginning of training to the minimum point of the loss function. Fig. 3 shows the trajectory without using momentum.

![SD without Momentum](https://c.mql5.com/2/150/SD_without_M__2.png)

Fig. 3. Gradient descent without momentum

We see that as the minimum approaches, the parameter vector begins to oscillate chaotically, which does not allow us to reach the optimum point. To get rid of this phenomenon, we will have to reduce the learning rate. Then the algorithm will, of course, begin to converge, but the time spent on the search may increase significantly.

Fig. 4 shows the trajectory of the parameter vector using the momentum (with the value of 0.9). This time the trajectory is smoother, and we easily reach the optimum point. Now we can even increase the learning rate. This is, in fact, the main idea of gradient descent with momentum - to speed up the convergence process.

![SD with Momentum](https://c.mql5.com/2/150/SD_with_M__2.png)

Fig. 4. Gradient descent, momentum (0.9)

The Momentum\_SD script implements the gradient descent algorithm with momentum. In this algorithm, I decided to get rid of one hidden layer and separate the weights and biases of the network, for clarity of perception. Now we have only one hidden layer with 20 neurons instead of two hidden layers with 5 neurons each, as in the previous example.

```
//+------------------------------------------------------------------+
//|                                                  Momentum_SD.mq5 |
//|                                                           Eugene |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Eugene"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property script_show_inputs
#include <Graphics\Graphic.mqh>
#include <Math\Stat\Math.mqh>
#include <Math\Stat\Normal.mqh>
enum Plots
  {
   LossFunction_plot,
   target_netpredict_plot
  };

matrix weights1, weights2,bias1,bias2;                      // network parameter matrices
matrix dW1,db1,dW2,db2;                                     // weight increment matrices
matrix n1,n2,act1,act2;                                     // neural layer output matrices
input int    layer1                  = 20;                  // neurons Layer 1
input int    Epochs                  = 1000;                // Epochs
input double lr                      = 0.1;                 // learning rate coefficient
input double sigma_                  = 0.1;                 // standard deviation synthetic data
input double gamma_                  = 0.9;                 // momentum
input Plots  plot_                   = LossFunction_plot;   // display graph
input bool   plot_log                = false;               // Plot Log graph
input ENUM_ACTIVATION_FUNCTION ac_func      = AF_TANH;      // Activation Layer1
input ENUM_ACTIVATION_FUNCTION ac_func_last = AF_LINEAR;    // Activation Layer2
input ENUM_LOSS_FUNCTION       loss_func    = LOSS_MSE;     // Loss function

double LossPlot[],target_Plot[],NetOutput[];
matrix ones_;
int Sample_,Features;

//+------------------------------------------------------------------+
//| Script start function                                            |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- generate a training sample
   matrix data, target;
   Func(data,target);
   StandartScaler(data);
   Sample_= (int)data.Rows();
   Features = (int)data.Cols();
   ArrayResize(target_Plot,Sample_);
   for(int i=0; i< (int)target.Rows(); i++)
     {
      target_Plot[i] =target[i,0];
     }
   ones_ = matrix::Ones(1,Sample_);

   ulong start=GetMicrosecondCount();
//--- train the model
   if(!Train(data, target, Epochs))
      return;
   ulong end = (GetMicrosecondCount()-start)/1000;
   Print("Learning time = " + (string)end + " msc");
//--- generate a test sample
    Func(data,target);
    StandartScaler(data);
//--- test the model
   Test(data, target);
//--- display graphs
   PlotGraphic(15,plot_log);
  }
//+------------------------------------------------------------------+
//| Model training method                                            |
//+------------------------------------------------------------------+
bool Train(matrix &data, matrix &target, const int epochs)
  {
//--- create the model
   if(!CreateNet())
      return false;
   ArrayResize(LossPlot,Epochs);
//--- train the model
   for(int ep = 0; ep < epochs; ep++)
     {
      //--- feed forward
      if(!FeedForward(data))
         return false;
      PrintFormat("Epoch %d, loss %.5f", ep, act2.Loss(target, loss_func));
      LossPlot[ep] = act2.Loss(target, loss_func);
      //--- backpropagation and update of weight matrix
      if(!Backprop(data, target))
         return false;
     }
//---
   double rmse=act2.RegressionMetric(target.Transpose(),REGRESSION_RMSE);
   PrintFormat("rmse %.3f / sigma %.2f ",rmse,sigma_);
   ArrayResize(NetOutput,Sample_);
   for(int i=0; i< (int)act2.Cols(); i++)
     {
      NetOutput[i]  =act2.Transpose()[i,0];
     }
//--- return result
   return true;
  }
//+------------------------------------------------------------------+
//| Model creation method                                            |
//+------------------------------------------------------------------+
bool CreateNet()
  {
//--- initialize weight matrices
   if(!weights1.Init(layer1,Features)  || !weights2.Init(1,layer1))
      return false;
//--- initialize offset matrices
   if(!bias1.Init(layer1,1)  || !bias2.Init(1,1))
      return false;
//--- initialize the matrix of parameter increments
   dW1.Init(layer1,Features);
   dW2.Init(1, layer1);
   db1.Init(layer1,1);
   db2.Init(1,1);
   dW1.Fill(0);
   dW2.Fill(0);
   db1.Fill(0);
   db2.Fill(0);
//--- fill the parameter matrices with random values
   weights1.Random(-0.1, 0.1);
   weights2.Random(-0.1, 0.1);
   bias1.Random(-0.1,0.1);
   bias2.Random(-0.1,0.1);
//--- return result
   return true;
  }
//+------------------------------------------------------------------+
//| Feed-forward method                                              |
//+------------------------------------------------------------------+
bool FeedForward(matrix &data)
  {
//--- calculate the first neural layer
//--- n1 pre-activation of the first layer
   n1 = weights1.MatMul(data.Transpose()) + bias1.MatMul(ones_);
//--- calculate the activation function of the act1 first layer
   n1.Activation(act1, ac_func);
//--- calculate the second neural layer
//--- n2 pre-activation of the second layer
   n2 = weights2.MatMul(act1) + bias2.MatMul(ones_);
//--- calculate the activation function of the act2 second layer
   n2.Activation(act2, ac_func_last);
//--- return result
   return true;
  }
//+------------------------------------------------------------------+
//| Backpropagation method                                           |
//+------------------------------------------------------------------+
bool Backprop(matrix &data, matrix &target)
  {
//--- Derivative of the loss function with respect to the predicted value
   matrix DerivLoss_wrt_y = act2.LossGradient(target.Transpose(),loss_func);
   matrix deriv_act2;
   n2.Derivative(deriv_act2, ac_func_last);
//--- D2
   matrix  D2 = deriv_act2*DerivLoss_wrt_y; // error(delta) of the network output layer
//--- D1
   matrix deriv_act1;
   n1.Derivative(deriv_act1, ac_func);
   matrix D1 = weights2.Transpose().MatMul(D2);
   D1 = D1*deriv_act1; // error (delta) of the first layer of the network
//--- update network parameters
   matrix  ones = matrix::Ones(data.Rows(),1);
   dW1 = gamma_*dW1 + (1-gamma_)*(D1.MatMul(data)) * lr;
   db1 = gamma_*db1 + (1-gamma_)*(D1.MatMul(ones)) * lr;
   dW2 = gamma_*dW2 + (1-gamma_)*(D2.MatMul(act1.Transpose())) * lr;
   db2 = gamma_*db2 + (1-gamma_)*(D2.MatMul(ones)) * lr;
   weights1 =  weights1 - dW1;
   weights2 =  weights2 - dW2;
   bias1    =  bias1    - db1;
   bias2    =  bias2    - db2;
//--- return result
   return true;
  }
```

Thanks to the momentum, I was able to increase the learning speed from 0.1 to 0.5. Now the algorithm converges in 150-200 iterations instead of 500 for regular gradient descent.

![Loss SD with Momentum](https://c.mql5.com/2/150/Loss_SD_Momentum__1.png)

Fig. 5. MSE loss function, MLP(1-20-1) SD\_Momentum

### Stochastic gradient descent

Momentum is good, but when the data set is not 81 data points, as in our example, but tens of thousands of data instances, then it makes sense to talk about such a well-proven (and simple) algorithm as SGD. SGD is the same gradient descent, but the gradient is calculated not over the entire training set, but only on some very small part of this set (mini-batch), or even on one data point. After that, the network weights are updated, a new data point is randomly selected, and the process is repeated until the algorithm converges. That is why the algorithm is called stochastic. In conventional gradient descent, we updated the network weights only after we calculated the gradient on the entire data set. This is the so-called batch method.

We implement a variant of SGD where only one data point is used as a mini-batch.

![Loss SGD](https://c.mql5.com/2/150/Loss_sgd__1.png)

Fig. 6. Loss function in logarithmic scale, SGD

The SGD algorithm (batch\_size = 1) converges to the minimum boundary in 4-6 thousand iterations, but let's remember that we are using only one training example out of 81 to update the parameter vector. Therefore, the algorithm on this dataset converges in approximately 50-75 epochs. Not a bad improvement over the previous algorithm, right? I also used momentum here, but since only one data point is used, it doesn't have much of an impact on the convergence speed.

### Levenberg-Marquardt algorithm

This good old algorithm is for some reason completely forgotten nowadays, although if your network has up to a couple of hundred parameters, there is simply no equal to it together with L-BFGS.

But there is one important point. The LM algorithm is designed to minimize functions that are sums of squares of other non-linear functions. Therefore, for this method, we will be limited to only a quadratic or root mean square loss function. All things being equal, this loss function does its job perfectly, and there is no big problem here, but we need to know that we will not be able to train the network using this algorithm on other functions.

Let's now take a detailed look at how this algorithm appeared. Let's start with Newton's method:

![Newton’s method](https://c.mql5.com/2/150/Newton__2.png)

here

**A** – inverse Hessian matrix of the F(x) loss function,

**g** – F(x) loss function gradient,

**x** – vector of parameters

Now let's look at our quadratic loss function:

![SSE](https://c.mql5.com/2/150/SSE__2.png)

here, **v** is a network error (predicted value minus target), while **x** is a vector of network parameters that includes all the weights and biases for each layer.

Let's find the gradient of this loss function:

![gradient SSE Loss function](https://c.mql5.com/2/150/gradient_SSE_Loss_function__1.png)

In matrix form, this can be written as:

![matrix notation gradient](https://c.mql5.com/2/150/matrix_notation_gradient___1.png)

The key point is the Jacobian matrix:

![Jacobian](https://c.mql5.com/2/150/Jacobian__2.png)

In the Jacobian matrix, each row contains all partial derivatives of the network error with respect to all parameters. Each line corresponds to one example of the training set.

Now let's look at the Hessian matrix. This is the matrix of second partial derivatives of the loss function. Calculating the Hessian is a difficult and expensive task, so an approximation of the Hessian by the Jacobian matrix is used:

![Hessian](https://c.mql5.com/2/150/Hessian__1.png)

If we substitute the Hessian and gradient equations into the Newton's method equation, we obtain the Gauss-Newton method:

![Gauss-Newton method](https://c.mql5.com/2/150/Gauss-Newton_method__1.png)

But the problem with the Gauss-Newton method is that the **\[J'J\]** matrix may not be reversible. To solve this issue, the identity matrix multiplied by the **mu\*I** positive scalar is added to the matrix. In this case, we get the Levenberg-Marquardt algorithm:

![Levenberg-Marquardt method](https://c.mql5.com/2/150/Levenberg-Marquardt_method___1.png)

The peculiarity of this algorithm is that when the **mu** parameter takes on large positive values, the algorithm is reduced to the usual gradient descent, which we discussed at the beginning of the article. If the **mu** parameter tends to zero, we return to the Gauss-Newton method.

Usually training starts with a small **mu** value. If the loss function value does not become smaller, then the **mu** parameter is increased (for example, multiplied by 10). Since this brings us closer to the gradient descent method, sooner or later we will achieve a reduction in the loss function. If the loss function has decreased, we decrease the value of the **mu** parameter by connecting the Gauss-Newton method for faster convergence to the minimum point. This is the main idea of the Levenberg-Marquardt method - to constantly switch between the gradient descent method and the Gauss-Newton method.

The implementation of the backpropagation method for the Levenberg-Marquardt algorithm has its own characteristics. Since the elements of the Jacobian matrix are partial derivatives of the network errors, and not the squares of these errors, the equation for calculating the delta of the last layer of the network, which I gave at the beginning of the article, is simplified. Now delta is simply equal to the derivative of the last layer's activation function. This result is obtained if we find the derivative of the network error (y – target) with respect to y, which is obviously equal to one.

Here is the neural network code itself with detailed comments.

```
//+------------------------------------------------------------------+
//|                                                           LM.mq5 |
//|                                                           Eugene |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Eugene"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property script_show_inputs
#include <Graphics\Graphic.mqh>
#include <Math\Stat\Math.mqh>
#include <Math\Stat\Normal.mqh>
enum Plots
  {
   LossFunction_plot,
   mu_plot,
   gradient_plot,
   target_netpredict_plot
  };

matrix weights1,weights2,bias1,bias2;                    // network parameter matrices
matrix n1,n2,act1,act2,new_n1,new_n2,new_act1,new_act2;  // neural layer output matrices
input int    layer1     = 20;                            // neurons Layer 1
input int    Epochs     = 10;                            // Epochs
input double Initial_mu = 0.001;                         // mu
input double Incr_Rate  = 10;                            // increase mu
input double Decr_Rate  = 0.1;                           // decrease mu
input double Min_grad   = 0.000001;                      // min gradient norm
input double Loss_goal  = 0.001;                         // Loss goal
input double sigma_     = 0.1;                           // standard deviation synthetic data
input Plots  plot_      = LossFunction_plot;             // display graph
input bool   plot_log   = false;                         // logarithmic function graph
input ENUM_ACTIVATION_FUNCTION ac_func      = AF_TANH;   // first layer activation function
input ENUM_ACTIVATION_FUNCTION ac_func_last = AF_LINEAR; // last layer activation function
input ENUM_LOSS_FUNCTION       loss_func    = LOSS_MSE;  // Loss function

double LossPlot[],NetOutput[],mu_Plot[],gradient_Plot[],target_Plot[];
matrix ones_;
double old_error,gradient_NormP2;
double mu_ = Initial_mu;
bool break_forloop = false;
int Sample_,Features;

//+------------------------------------------------------------------+
//| Script start function                                            |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- generate a training sample
   matrix data, target;
   Func(data,target);
   StandartScaler(data);
   Sample_= (int)data.Rows();
   Features = (int)data.Cols();
   ArrayResize(target_Plot,Sample_);
   for(int i=0; i< (int)target.Rows(); i++)
     {
      target_Plot[i] =target[i,0];
     }

   ones_ = matrix::Ones(1,Sample_);
//--- train the model
   ulong start=GetMicrosecondCount();
   Train(data, target, Epochs);
   ulong end = (GetMicrosecondCount()-start)/1000 ;
   Print("Learning time = " + (string)end + " msc");
   int NumberParameters = layer1*(Features+1) + 1*(layer1+1);
   Print("Number Parameters of NN = ",NumberParameters);

//--- generate a test sample
   Func(data,target);
   StandartScaler(data);
//--- test the model
   Test(data,target);

//--- display graphs
   PlotGraphic(15,plot_log);
  }
//+------------------------------------------------------------------+
//| Model training method                                            |
//+------------------------------------------------------------------+
bool Train(matrix &data, matrix &target, const int epochs)
  {
//--- create the model
   if(!CreateNet())
      return false;
//--- train the model
   for(int ep = 0; ep < epochs; ep++)
     {
      //--- feed forward
      if(!FeedForward(data))
         return false;
      PrintFormat("Epoch %d, loss %.5f", ep, act2.Loss(target, loss_func));
      //--- arrays for graphs
      ArrayResize(LossPlot,ep+1,10000);
      ArrayResize(mu_Plot,ep+1,10000);
      ArrayResize(gradient_Plot,ep+1,10000);
      LossPlot[ep]       = act2.Loss(target, loss_func);
      mu_Plot [ep]       = mu_;
      gradient_Plot[ep]  = gradient_NormP2;
      //--- Stop training if the target value of the loss function is reached
      if(break_forloop == true){break;}
      //--- backpropagation and update of weight matrix
      if(!Backprop(data, target))
         return false;
     }
//--- Euclidean norm of gradient, mu parameter, RMSE metric
   Print("gradient_normP2 =  ", gradient_NormP2);
   Print(" mu_ = ", mu_);
   double rmse=act2.RegressionMetric(target.Transpose(),REGRESSION_RMSE);
   PrintFormat("rmse %.3f / sigma %.2f ",rmse,sigma_);
//--- array of network output for graph
   ArrayResize(NetOutput,Sample_);
   for(int i=0; i< (int)act2.Transpose().Rows(); i++)
     {
      NetOutput[i]  = act2.Transpose()[i,0];
     }
//--- return result
   return true;
  }
//+------------------------------------------------------------------+
//| Model creation method                                            |
//+------------------------------------------------------------------+
bool CreateNet()
  {
//--- initialize weight matrices
   if(!weights1.Init(layer1,Features) || !weights2.Init(1,layer1))
      return false;
//--- initialize offset matrices
   if(!bias1.Init(layer1,1)  || !bias2.Init(1,1))
      return false;
//--- fill the weight matrices with random values
   weights1.Random(-0.1, 0.1);
   weights2.Random(-0.1, 0.1);
   bias1.Random(-0.1, 0.1);
   bias2.Random(-0.1, 0.1);
//--- return result
   return true;
  }
//+------------------------------------------------------------------+
//| Feed-forward method                                              |
//+------------------------------------------------------------------+
bool FeedForward(matrix &data)
  {
//--- calculate the first neural layer
//--- n1 pre-activation of the first layer
   n1 = weights1.MatMul(data.Transpose()) + bias1.MatMul(ones_);
//--- calculate the activation function of the act1 first layer
   n1.Activation(act1, ac_func);
//--- calculate the second neural layer
//--- n2 pre-activation of the second layer
   n2 = weights2.MatMul(act1) + bias2.MatMul(ones_);
//--- calculate the activation function of the act2 second layer
   n2.Activation(act2, ac_func_last);
//--- return result
   return true;
  }
//+------------------------------------------------------------------+
//| Backpropagation method                                           |
//+------------------------------------------------------------------+
bool Backprop(matrix &data, matrix &target)
  {
//--- current value of the loss function
   old_error = act2.Loss(target, loss_func);
//--- network error (quadratic loss function)
   matrix loss = act2.Transpose() - target ;
//--- derivative of the activation function of the last layer
   matrix D2;
   n2.Derivative(D2, ac_func_last);
//--- derivative of the first layer activation function
   matrix deriv_act1;
   n1.Derivative(deriv_act1, ac_func);
//--- first layer network error
   matrix D1 = weights2.Transpose().MatMul(D2);
   D1 = deriv_act1 * D1;
//--- first partial derivatives of network errors with respect to the first layer weights
   matrix jac1;
   partjacobian(data.Transpose(),D1,jac1);
//--- first partial derivatives of network errors with respect to the second layer weights
   matrix jac2;
   partjacobian(act1,D2,jac2);
//--- Jacobian
   matrix j1_D1 = Matrixconcatenate(jac1,D1.Transpose(),1);
   matrix j2_D2 = Matrixconcatenate(jac2,D2.Transpose(),1);
   matrix jac   = Matrixconcatenate(j1_D1,j2_D2,1);
// --- Loss function gradient
   matrix je = (jac.Transpose().MatMul(loss));
//--- Euclidean norm of gradient normalized to sample size
   gradient_NormP2 = je.Norm(MATRIX_NORM_FROBENIUS)/Sample_;
   if(gradient_NormP2 < Min_grad)
     {
      Print("Local minimum. The gradient is less than the specified value.");
      break_forloop = true; // stop training
      return true;
     }
//--- Hessian
   matrix Hessian = (jac.Transpose().MatMul(jac));
   matrix I=matrix::Eye(Hessian.Rows(), Hessian.Rows());
//---
   break_forloop = true;
   while(mu_ <= 1e10 && mu_ > 1e-20)
     {
      matrix H_I = (Hessian + mu_*I);
      //--- solution via Solve
      vector v_je = je.Col(0);
      vector Updatelinsolve = -1* H_I.Solve(v_je);
      matrix Update = matrix::Zeros(Hessian.Rows(),1);
      Update.Col(Updatelinsolve,0); // increment of the parameter vector

      //--- inefficient calculation of inverse matrix
      //   matrix Update = H_I.Inv();
      //   Update = -1*Update.MatMul(je);
      //---

      //--- save the current parameters
      matrix  Prev_weights1 = weights1;
      matrix  Prev_bias1    = bias1;
      matrix  Prev_weights2 = weights2;
      matrix  Prev_bias2    = bias2;
      //---

      //--- update the parameters
      //--- first layer
      matrix updWeight1 = matrix::Zeros(layer1,Features);
      int count =0;
      for(int j=0; j <Features; j++)
        {
         for(int i=0 ; i <layer1; i++)
           {
            updWeight1[i,j] = Update[count,0];
            count = count+1;
           }
        }

      matrix updbias1 = matrix::Zeros(layer1,1);
      for(int i =0 ; i <layer1; i++)
        {
         updbias1[i,0] = Update[count,0];
         count = count +1;
        }

      weights1 = weights1 + updWeight1;
      bias1 = bias1 + updbias1;

      //--- second layer
      matrix updWeight2 = matrix::Zeros(1,layer1);
      for(int i =0 ; i <layer1; i++)
        {
         updWeight2[0,i] = Update[count,0];
         count = count +1;
        }
      matrix updbias2 = matrix::Zeros(1,1);
      updbias2[0,0] = Update[count,0];

      weights2 = weights2 + updWeight2;
      bias2 = bias2 + updbias2;

      //--- calculate the loss function for the new parameters
      new_n1 = weights1.MatMul(data.Transpose()) + bias1.MatMul(ones_);
      new_n1.Activation(new_act1, ac_func);
      new_n2 = weights2.MatMul(new_act1) + bias2.MatMul(ones_);
      new_n2.Activation(new_act2, ac_func_last);
      //--- loss function taking into account new parameters
      double new_error = new_act2.Loss(target, loss_func);
      //--- if the loss function is less than the specified threshold, terminate training
      if(new_error < Loss_goal)
        {
         break_forloop = true;
         Print("Training complete. The desired loss function value achieved");
         return true;
        }
      break_forloop = false;
      //--- correct the mu parameter
      if(new_error >= old_error)
        {
         weights1 = Prev_weights1;
         bias1    = Prev_bias1;
         weights2 = Prev_weights2;
         bias2    =  Prev_bias2;
         mu_ = mu_*Incr_Rate;
        }
      else
        {
         mu_ = mu_*Decr_Rate;
         break;
        }

     }
//--- return result
   return true;
  }
```

The algorithm converges if the gradient norm is less than a predetermined number, or if the desired level of the loss function is reached. The algorithm stops if the mu parameter is less than or greater than a predetermined number, or after a predetermined number of epochs have been completed.

![LM parameters](https://c.mql5.com/2/150/LM_parameters__2.png)

Fig. 7. LM script parameters

Let's look at the result of all this math:

![Loss LM](https://c.mql5.com/2/150/Loss_LM__1.png)

Fig. 8. Loss function in logarithmic scale, LM

It is a completely different picture now. The algorithm reached the minimum boundary in 6 iterations. What if we trained the network on one thousand epochs? We would get a typical overfitting. The picture below demonstrates this well. The network simply starts to memorize Gaussian noise.

![Overfitting LM](https://c.mql5.com/2/150/overfit_LM__1.png)

Fig. 9. Typical overfitting, LM, 1000 epochs

Let's look at the metrics on the training and test sets.

![performance LM](https://c.mql5.com/2/150/overfit_rmse_LM__2.png)

Fig. 10. Performance statistics, LM, 1000 epochs

We see RMSE 0.168 with a lower limit of 0.20 and then there is immediate retribution for overfitting on the test of 0.267.

### Testing on big data and comparison with Python sklearn library

It is time to test our algorithm on a more realistic example. Now I took two features with 1000 data points. You can download this data along with the LM\_BigData script at the end of the article. LM will compete with algorithms from the Python library: SGD, Adam and L-BFGS.

Here is a test script in Python

```
# Eugene
# https://www.mql5.com

import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPRegressor

# here is your path to the data
df = pd.read_csv(r'C:\Users\Evgeniy\AppData\Local\Programs\Python\Python39\Data.csv',delimiter=';')
X = df.to_numpy()
df1 = pd.read_csv(r'C:\Users\Evgeniy\AppData\Local\Programs\Python\Python39\Target.csv')
y = df1.to_numpy()
y = y.reshape(-1)

start = time.time()

'''
clf = MLPRegressor(solver='sgd', alpha=0.0,
                    hidden_layer_sizes=(20),
                    activation='tanh',
                    max_iter=700,batch_size=10,
                    learning_rate_init=0.01,momentum=0.9,
                    shuffle = False,n_iter_no_change = 2000, tol = 0.000001)
'''

'''
clf = MLPRegressor(solver='adam', alpha=0.0,
                    hidden_layer_sizes=(20),
                    activation='tanh',
                    max_iter=3000,batch_size=100,
                    learning_rate_init=0.01,
                    n_iter_no_change = 2000, tol = 0.000001)
'''

#'''
clf = MLPRegressor(solver='lbfgs', alpha=0.0,
                    hidden_layer_sizes=(100),
                    activation='tanh',max_iter=300,
                    tol = 0.000001)
#'''

clf.fit(X, y)
end = time.time() - start          # training time

print("learning time  =",end*1000)
print("solver = ",clf.solver);
print("loss = ",clf.loss_*2)
print("iter = ",clf.n_iter_)
#print("n_layers_ = ",clf.n_layers_)
#print("n_outputs_ = ",clf.n_outputs_)
#print("out_activation_ = ",clf.out_activation_)

coef = clf.coefs_
#print("coefs_ = ",coef)
inter = clf.intercepts_
#print("intercepts_ = ",inter)
plt.plot(np.log(pd.DataFrame(clf.loss_curve_)))
plt.title(clf.solver)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
```

To compare the algorithms correctly, I multiplied the loss function in Python by 2, since in this library it is calculated like this:

return ((y\_true - y\_pred) \*\* 2).mean() / 2

In other words, the developers additionally divided MSE into 2. Below are typical results from optimizers. I tried to find the best hyperparameter settings for these algorithms. Unfortunately, this library does not provide the ability to initialize the values of the starting parameters so that all algorithms start from the same point in the parametric space. There is also no way to set a target threshold for the loss function. For LM the loss function target is set to 0.01, for Python algorithms I tried to set the number of iterations that approximately achieves the same level.

Test results of MLP with one hidden layer, 20 neurons:

1) **Stochastic gradient descent**

- loss mse – 0,00278
- training time  - **11459 msc**

![SGD Loss Python](https://c.mql5.com/2/150/sgd_loss__1.png)

Fig. 11. SGD, 20 neurons, loss = 0.00278

2) **Adam**

-  loss mse – 0.03363
-  training time  - **8581 msc**

![Adam loss](https://c.mql5.com/2/150/adam_loss__1.png)

Fig. 12. Adam, 20 neurons, loss = 0.03363

3) **L-BFGS**

- loss mse – 0.02770
- training time - **277 msc**

Unfortunately, it is not possible to display the loss function graph for L-BFGS.

4) **LM MQL5**

-  loss – 0.00846
-  training time — **117 msc**

![LM Loss](https://c.mql5.com/2/150/LM_loss__1.png)

Fig. 13. LM, 20 neurons, loss = 0.00846

![performance LM](https://c.mql5.com/2/150/perfLM___2.png)

As for me, the algorithm easily competes with L-BFGS and can even give it a head start. But nothing is perfect. As the number of parameters increases, the Levenberg-Marquardt method begins to lose to L-BFGS.

100 neurons **L-BFGS**:

- loss mse – 0.00847
- training time - **671 msc**

100 neurons **LM**:

- loss mse – 0.00206
- training time  -  **1253 msc**

100 neurons correspond to 401 network parameters. It is up to you to decide whether this is a lot or a little, but in my humble opinion, this is excess power. In cases up to 100 neurons, LM clearly has an advantage.

Conclusion

In this article, we discussed and implemented the basic and simplest neural network training algorithms:

- gradient descent
- gradient descent with momentum
- stochastic gradient descent

At the same time, we briefly touched on the issues of convergence and retraining of neural networks.

But most importantly, we built a very fast Levenberg-Marquardt algorithm that is ideal for online training of small networks.

We compared the performance of neural network training algorithms used in the scikit-learn machine learning library, and our algorithm turned out to be the fastest when the number of neural network parameters does not exceed 400 or 100 neurons in the hidden layer. Further, as the number of neurons increases, L-BFGS begins to dominate.

For each algorithm, separate scripts have been created, with detailed comments:

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | SD.mq5 | Script | Gradient descent |
| 2 | Momentum\_SD.mq5 | Script | Gradient descent with momentum |
| 3 | SGD.mq5 | Script | Stochastic gradient descent |
| 4 | LM.mq5 | Script | Levenberg-Marquardt algorithm |
| 5 | LM\_BigData.mq5 | Script | LM algorithm, test on two-dimensional features |
| 6 | SklearnMLP.py | Script | Python algorithms test script |
| 7 | FileCSV.mqh | Include | Reading text files with data |
| 8 | Data.csv, Target.csv | Csv | Python script features and targets |
| 9 | X1.txt, X2.txt, Target.txt | Txt | LM\_BigData.mq5 script features and targets |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/16296](https://www.mql5.com/ru/articles/16296)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16296.zip "Download all attachments in the single ZIP archive")

[SD.mq5](https://www.mql5.com/en/articles/download/16296/sd.mq5 "Download SD.mq5")(26.15 KB)

[SGD.mq5](https://www.mql5.com/en/articles/download/16296/sgd.mq5 "Download SGD.mq5")(26.16 KB)

[Momentum\_SD.mq5](https://www.mql5.com/en/articles/download/16296/momentum_sd.mq5 "Download Momentum_SD.mq5")(22.59 KB)

[LM.mq5](https://www.mql5.com/en/articles/download/16296/lm.mq5 "Download LM.mq5")(36.21 KB)

[LM\_BigData.mq5](https://www.mql5.com/en/articles/download/16296/lm_bigdata.mq5 "Download LM_BigData.mq5")(36.15 KB)

[FileCSV.mqh](https://www.mql5.com/en/articles/download/16296/filecsv.mqh "Download FileCSV.mqh")(10.45 KB)

[X1.txt](https://www.mql5.com/en/articles/download/16296/x1.txt "Download X1.txt")(19.26 KB)

[X2.txt](https://www.mql5.com/en/articles/download/16296/x2.txt "Download X2.txt")(19.26 KB)

[Target.txt](https://www.mql5.com/en/articles/download/16296/target.txt "Download Target.txt")(17.62 KB)

[Data.csv](https://www.mql5.com/en/articles/download/16296/data.csv "Download Data.csv")(37.56 KB)

[Target.csv](https://www.mql5.com/en/articles/download/16296/target.csv "Download Target.csv")(17.62 KB)

[SklearnMLP.py](https://www.mql5.com/en/articles/download/16296/sklearnmlp.py "Download SklearnMLP.py")(1.83 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Forecasting exchange rates using classic machine learning methods: Logit and Probit models](https://www.mql5.com/en/articles/16029)
- [Econometric tools for forecasting volatility: GARCH model](https://www.mql5.com/en/articles/15223)
- [Elements of correlation analysis in MQL5: Pearson chi-square test of independence and correlation ratio](https://www.mql5.com/en/articles/15042)
- [Two-sample Kolmogorov-Smirnov test as an indicator of time series non-stationarity](https://www.mql5.com/en/articles/14813)
- [Non-stationary processes and spurious regression](https://www.mql5.com/en/articles/14412)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/489375)**
(6)


![Evgeniy Chernish](https://c.mql5.com/avatar/2024/3/65eac9b5-9233.png)

**[Evgeniy Chernish](https://www.mql5.com/en/users/vp999369)**
\|
8 Nov 2024 at 08:52

Thanks for the feedback.

On python. It is not an error, it warns that the algorithm has stopped because we have reached the iteration limit. That is, the algorithm stopped before the value tol = 0.000001 was reached. And then it warns that the lbfgs optimiser does not have a "loss\_curve" attribute, i.e. [loss function](https://www.mql5.com/en/docs/matrix/matrix_machine_learning/matrix_loss "MQL5 Documentation: function Loss") data. For adam and sgd they do, but for lbfgs for some reason they don't. I probably should have made a script so that when lbfgs is started it would not ask for this property so that it would not confuse people.

On SD. Since we start each time from different points in the parameter space, the paths to the solution will also be different. I have done a lot of testing, sometimes it really takes more iterations to converge. I tried to give an average number of iterations. You can increase the number of iterations and you will see that the algorithm converges in the end.

![Andrey Dik](https://c.mql5.com/avatar/2024/8/66be0662-3c24.png)

**[Andrey Dik](https://www.mql5.com/en/users/joo)**
\|
8 Nov 2024 at 09:32

**Evgeniy Chernish [#](https://www.mql5.com/ru/forum/476062#comment_55065437):**

On SD. Since we start each time from a different point in the parameter space, the paths to converge to a solution will also be different. I have done a lot of testing, sometimes it really takes more iterations to converge. I tried to give an average number of iterations. You can increase the number of iterations and you will see that the algorithm converges in the end.

That's what I'm talking about. It's the robustness, or, reproducibility of the results. The greater the scatter of results, the closer the algorithm is to RND for a given problem.

[![](https://c.mql5.com/3/448/3598558418981__1.png)](https://c.mql5.com/3/448/3598558418981.png "https://c.mql5.com/3/448/3598558418981.png")

Here's an example of how three different algorithms work. Which one is the best? Unless you run a series of independent tests and calculate the average results (ideally, calculate and compare the variance of the final results), it is impossible to compare.

![Evgeniy Chernish](https://c.mql5.com/avatar/2024/3/65eac9b5-9233.png)

**[Evgeniy Chernish](https://www.mql5.com/en/users/vp999369)**
\|
8 Nov 2024 at 09:52

**Andrey Dik [#](https://www.mql5.com/ru/forum/476062#comment_55065612):**

That's what I'm talking about. It is the stability, or, reproducibility of the results. The greater the scatter of results, the closer the algorithm is to RND for a given problem.

Here's an example of how three different algorithms work. Which one is the best? Unless you run a series of independent tests and calculate the average results (ideally, calculate and compare the variance of the final results), it is impossible to compare.

Then it is necessary to define the evaluation criterion.

You can take time and maximum result (or minimum if you want to find the minimum of the function) as a criterion.

Set the number of restarts.

Record the maximum achieved for this number of restarts and the time spent on it.

Conduct a series of such tests, let's say 1000.

And calculate the mean and variance for this series, i.e. the mean and variance for the maximum.

I have not done it just so thoroughly almost to the construction of the distribution density of the results, it is impossible to cover everything in one article.

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
8 Nov 2024 at 11:43

The article is great without additional tests and is in line with common conclusions regarding algorithms :) This makes it possible to quickly agree on something and move on to the next topic.


![Andrey Dik](https://c.mql5.com/avatar/2024/8/66be0662-3c24.png)

**[Andrey Dik](https://www.mql5.com/en/users/joo)**
\|
8 Nov 2024 at 11:44

**Evgeniy Chernish [#](https://www.mql5.com/ru/forum/476062#comment_55065955):**

Then it is necessary to define the evaluation criterion.

We can take time and maximum result as a criterion (or minimum if we need to find the minimum of the function).

Set the number of restarts.

Record the maximum achieved for this number of restarts and the time spent on it.

Conduct a series of such tests, let's say 1000.

And calculate the mean and variance for this series, i.e. the mean and variance for the maximum.

I have not done it just so thoroughly almost to the construction of the distribution density of the results, it is impossible to cover everything in one article.

No, in this case you don't have to go to such trouble, but if you are comparing different methods, you could add one more cycle (independent tests) and plot the graphs of individual tests. Everything would be very clear, who converges, how stable it is and how many iterations it takes. And so it turned out to be "like last time", when the result is great, but only once in a million.

Anyway, thanks, the article gave me some interesting thoughts.

![MQL5 Wizard Techniques you should know (Part 71): Using Patterns of MACD and the OBV](https://c.mql5.com/2/150/18462-mql5-wizard-techniques-you-logo__1.png)[MQL5 Wizard Techniques you should know (Part 71): Using Patterns of MACD and the OBV](https://www.mql5.com/en/articles/18462)

The Moving-Average-Convergence-Divergence (MACD) oscillator and the On-Balance-Volume (OBV) oscillator are another pair of indicators that could be used in conjunction within an MQL5 Expert Advisor. This pairing, as is practice in these article series, is complementary with the MACD affirming trends while OBV checks volume. As usual, we use the MQL5 wizard to build and test any potential these two may possess.

![Data Science and ML (Part 43): Hidden Patterns Detection in Indicators Data Using Latent Gaussian Mixture Models (LGMM)](https://c.mql5.com/2/150/18497-data-science-and-ml-part-44-logo.png)[Data Science and ML (Part 43): Hidden Patterns Detection in Indicators Data Using Latent Gaussian Mixture Models (LGMM)](https://www.mql5.com/en/articles/18497)

Have you ever looked at the chart and felt that strange sensation… that there’s a pattern hidden just beneath the surface? A secret code that might reveal where prices are headed if only you could crack it? Meet LGMM, the Market’s Hidden Pattern Detector. A machine learning model that helps identify those hidden patterns in the market.

![From Novice to Expert: Animated News Headline Using MQL5 (II)](https://c.mql5.com/2/150/18465-from-novice-to-expert-animated-logo.png)[From Novice to Expert: Animated News Headline Using MQL5 (II)](https://www.mql5.com/en/articles/18465)

Today, we take another step forward by integrating an external news API as the source of headlines for our News Headline EA. In this phase, we’ll explore various news sources—both established and emerging—and learn how to access their APIs effectively. We'll also cover methods for parsing the retrieved data into a format optimized for display within our Expert Advisor. Join the discussion as we explore the benefits of accessing news headlines and the economic calendar directly on the chart, all within a compact, non-intrusive interface.

![Analyzing weather impact on currencies of agricultural countries using Python](https://c.mql5.com/2/100/Analysis_of_the_impact_of_weather_on_the_currencies_of_agricultural_countries_using_Python___LOGO.png)[Analyzing weather impact on currencies of agricultural countries using Python](https://www.mql5.com/en/articles/16060)

What is the relationship between weather and Forex? Classical economic theory has long ignored the influence of such factors as weather on market behavior. But everything has changed. Let's try to find connections between the weather conditions and the position of agricultural currencies on the market.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/16296&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068797146116849099)

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