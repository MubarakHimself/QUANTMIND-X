---
title: Pipelines in MQL5
url: https://www.mql5.com/en/articles/19544
categories: Integration, Expert Advisors, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T21:05:19.628433
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=eyxcdxnfbldbvupmydkzzmtsoiudclqg&ssn=1769191518531969529&ssn_dr=0&ssn_sr=0&fv_date=1769191518&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19544&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Pipelines%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919151817476395&fz_uniq=5071563263969274461&sv=2552)

MetaTrader 5 / Integration


### Introduction: Why Data Pre-Processing matters

In designing and building forecasting systems and models that are AI powered, it is tempting to focus on just the architecture of the deep learning model or the complexity of the trading strategy. However, one of the most vital determinants of the model performance often lies not in the neural networks themselves, but also in the quality and consistency of the data that is fed to the model. In practice, raw financial data such as OHLC bars, tick volumes or even spreads are often far from ‘model-ready’. These ‘raw’ values may exist at different scales, may contain outliers caused by sudden market shocks, or may include categorical features like trading sessions; all of which can not, off-the-bat, be processed by mathematical models.

What our summary above suggests, is if for instance we are training a simple neural network on normalized price returns and volatility indicators, then the Standard-Scaler would probably be the suitable choice given that it ensures comparability across features. Alternatively, if, say, the model is an ONNX LSTM trained in Python with sigmoid activations, then the min max scaler could be more appropriate, since it tends to align with the activation’s bounded range. On the flip side, if we were modelling trading behaviors during a high-impact news event, where spreads and volumes can spike dramatically, then the robust scaler can be used to help preserve the central structure of the data while minimizing this anomaly influence.

The main takeaway here is that preprocessing shouldn’t be treated as an afterthought. The choice of scaler directly affects how features interact within the learning steps, and can make all the difference between a model that generalizes well and one that is biased or unstable.

If no proper preprocessing is in place, even the most sophisticated network is bound to struggle to learn patterns effectively. For example, a feature like spread, when measured in fractions of a pp, may be overshadowed by volume, in magnitude, which could be in the thousands. Similarly, assigning integers to market sessions such as Asia a 0, Europe a 1 and US a 2; does risk bringing categorical data which can be misleading to the model.

That's why preprocessing can be considered the bedrock of robust machine learning workflows. In the Python ecosystem, the SCIKIT-LEARN library has been known for its tools on standardization, normalization, robust scaling, as well as one hot encoding that all serve to prepare data for ‘downstream’ models. Every transformation step does mean that the input features end up contributing fairly to the learning process. In trading, however, operating with MQL5 libraries as is does present this lack of preprocessing. To bridge this, we can implement MQL5 classes that, to some degree, mirror the functionality of SCIKIT-LEARN and end up with pipelines tailored for financial tine series forecasting.

### Bridging the Ecosystem Gap

The SCIKIT-LEARN library of Python has, for all intents and purposes, become the go-to industry standard in machine learning pre-processing. By writing a minimal amount of code, developers get to apply a Standard-Scaler to center features; a Min-Max-Scaler to compress features to within a fixed range; a Robust-Scaler that serves to reduce the undue confluence of outliers; or a One-Hot-Encoder that helps expand features into binary representations. More than that, SCIKIT-LEARN’s pipeline class does allow these steps to be chained seamlessly, which does mean that all datasets that are passed to the model enjoy the same transformation sequences. This modular plug-and-play mechanism has brought about rapid adoption of machine learning in a vast array of industries.

In contrast, MQL5 developers are daunted by a different reality. While MQL5 is relatively efficient in handling trading data, it still fails to natively provide preprocessing methods comparable to SCIKIT-LEARN. For every transformation - whether scaling, encoding or even imputing missing values, the coding has to be done manually and often in a fragmented way. Not only does this raise the odds of introducing errors, but it also makes it harder to reproduce test results or maintain consistency across training and testing data.

The solution, in my opinion, could be in designing a preprocessing pipeline class in MQL5 that emulates this SCIKIT-LEARN philosophy. If we can implement reusable modules such as CStandardScaler, CMinMaxScaler, CRobustScaler, and COneHotEncoder, we could chain a preprocessing pipeline into a container. This structure would ensure that the raw financial time series data does undergo systematic preparation before entering deep learning models. This would be the case whether the models are coded natively in MQL5 or they are imported via ONNX. By using this, MQL5 developers get to adopt a familiar Python workflow with MQL5, unlocking cleaner experimentation, faster development, and presumably having more reliable AI systems.

### Anatomy of a Preprocessing Pipeline

In defining a preprocessing pipeline, one can think of it as a conveyor belt for data. Inputs of raw data get in on one end and by the time they leave, they are transformed into a suitable format for model consumption. Every stage of the conveyor belt performs a defined task like filling missing values, encoding categories or rescaling features that are numeric. Usually, in Python, this is all encapsulated in SCIKIT-LEARN’s pipeline object. In MQL5, we need to design a similar structure using custom classes. Our such class is going to be named CPreprocessingPipeline. We implement it in MQL5 as follows:

```
// Preprocessing Pipeline
class CPreprocessingPipeline
{
private:
   SPreprocessorStep m_steps[];
   IPreprocessor    *m_preprocessors[];
   int              m_step_count;

public:
   CPreprocessingPipeline() : m_step_count(0) {}

   ~CPreprocessingPipeline()
   {  for(int i = 0; i < m_step_count; i++)
         delete m_preprocessors[i];
   }

   void AddImputeMedian(int column)
   {  ArrayResize(m_steps, m_step_count + 1);
      ArrayResize(m_preprocessors, m_step_count + 1);
      m_steps[m_step_count].type = PREPROCESSOR_IMPUTE_MEDIAN;
      m_steps[m_step_count].column = column;
      m_preprocessors[m_step_count] = new CImputeMedian(column);
      m_step_count++;
   }

   void AddImputeMode(int column)
   {  ArrayResize(m_steps, m_step_count + 1);
      ArrayResize(m_preprocessors, m_step_count + 1);
      m_steps[m_step_count].type = PREPROCESSOR_IMPUTE_MODE;
      m_steps[m_step_count].column = column;
      m_preprocessors[m_step_count] = new CImputeMode(column);
      m_step_count++;
   }

   void AddStandardScaler()
   {  ArrayResize(m_steps, m_step_count + 1);
      ArrayResize(m_preprocessors, m_step_count + 1);
      m_steps[m_step_count].type = PREPROCESSOR_STANDARD_SCALER;
      m_steps[m_step_count].column = -1;
      m_preprocessors[m_step_count] = new CStandardScaler();
      m_step_count++;
   }

   void AddRobustScaler()
   {  ArrayResize(m_steps, m_step_count + 1);
      ArrayResize(m_preprocessors, m_step_count + 1);
      m_steps[m_step_count].type = PREPROCESSOR_ROBUST_SCALER;
      m_steps[m_step_count].column = -1;
      m_preprocessors[m_step_count] = new CRobustScaler();
      m_step_count++;
   }

   void AddMinMaxScaler(double new_min = 0.0, double new_max = 1.0)
   {  ArrayResize(m_steps, m_step_count + 1);
      ArrayResize(m_preprocessors, m_step_count + 1);
      m_steps[m_step_count].type = PREPROCESSOR_MINMAX_SCALER;
      m_steps[m_step_count].column = -1;
      m_preprocessors[m_step_count] = new CMinMaxScaler(new_min, new_max);
      m_step_count++;
   }

   void AddOneHotEncoder(int column)
   {  ArrayResize(m_steps, m_step_count + 1);
      ArrayResize(m_preprocessors, m_step_count + 1);
      m_steps[m_step_count].type = PREPROCESSOR_ONEHOT_ENCODER;
      m_steps[m_step_count].column = column;
      m_preprocessors[m_step_count] = new COneHotEncoder(column);
      m_step_count++;
   }

   bool FitPipeline(matrix &data)
   {  matrix temp;
      temp.Copy(data);
      for(int i = 0; i < m_step_count; i++)
      {  matrix out;
         if(!m_preprocessors[i].Fit(temp)) return false;
         if(!m_preprocessors[i].Transform(temp, out)) return false;
         temp.Copy(out);
      }
      return true;
   }

   bool TransformPipeline(matrix &data, matrix &out)
   {  out.Copy(data);
      for(int i = 0; i < m_step_count; i++)
      {  matrix temp;
         if(!m_preprocessors[i].Transform(out, temp)) return false;
         out.Copy(temp);
      }
      return true;
   }

   bool FitTransformPipeline(matrix &data, matrix &out)
   {  if(!FitPipeline(data)) return false;
      return TransformPipeline(data, out);
   }
};
```

This class serves in effect as a container for the transformation steps. Developers get to add steps using methods such as AddStandardScaler(), AddRobustScaler(), AddMinMaxScaler(), or AddOneHotEncoder(). Every one of these steps gets represented by a separate class. For example, we have the CStandardScaler class. These classes go on to implement a consistent interface by engaging functions/ methods such as Fit(), Transform(), and FitTransform(). Once the pipeline is assembled, it then gets ‘fitted’ to a training dataset, where it learns parameters such as means, medians, modes or even category mappings. Once the fitting is performed to a satisfactory training and model testing performance, it can be applied to a new dataset with similar transformation steps. This consistency ensures we avoid some common pitfalls in testing, such as [data leakage](https://en.wikipedia.org/wiki/Leakage_(machine_learning) "https://en.wikipedia.org/wiki/Leakage_(machine_learning)").

This design modularity does present a few benefits. Firstly, it does encourage reusability, given that the same pipeline object can be applied across many experiments. Secondly, it improves maintainability given that every transformation class is less dependent and therefore easier to debug. Finally, it tends to promote consistency in the sense that both the training, validation, and even testing all follow the exact same transformation path. In other words, the preprocessing pipeline introduces a level of discipline into MQL5 machine learning workflows that arguably mirror the robustness of the Python ecosystem.

### Standard Scaler

The standard scaler is one of the probably most widely used preprocessing tools in machine learning. Its purpose is to center each feature around zero and scale it by its standard deviation. In mathematics, we define this transformation with the following formula:

![f1](https://c.mql5.com/2/169/from_standard.png)

Where:

- μ or mu is the mean of the feature, and
- σ or sigma is its standard deviation.
- x is a data point in the data set.

The end result of this is that every feature has a mean of zero and a standard deviation of one. This tends to make the dataset more uniform by reducing the risk of particular features dominating the learning process. We code our CStandardScaler class in MQL5 as follows:

```
// Standard Scaler
class CStandardScaler : public IPreprocessor
{
private:
   double m_means[];
   double m_stds[];
   bool   m_is_fitted;

public:
   CStandardScaler() : m_is_fitted(false) {}

   bool Fit(matrix &data)
   {  int rows = int(data.Rows());
      int cols = int(data.Cols());
      ArrayResize(m_means, cols);
      ArrayResize(m_stds, cols);
      for(int j = 0; j < cols; j++)
      {  vector column(rows);
         for(int i = 0; i < rows; i++) column[i] = data[i][j];
         m_means[j] = column.Mean();
         m_stds[j] = column.Std();
         if(m_stds[j] == 0.0) m_stds[j] = EPSILON;
      }
      m_is_fitted = true;
      return true;
   }

   bool Transform(matrix &data, matrix &out)
   {  if(!m_is_fitted) return false;
      int rows = int(data.Rows());
      int cols = int(data.Cols());
      out.Init(rows, cols);
      for(int j = 0; j < cols; j++)
         for(int i = 0; i < rows; i++)
            out[i][j] = (!MathIsValidNumber(data[i][j]) ? DBL_MIN : (data[i][j] - m_means[j]) / m_stds[j]);
      return true;
   }

   bool FitTransform(matrix &data, matrix &out)
   {  if(!Fit(data)) return false;
      return Transform(data, out);
   }
};
```

In our MQL5 class above, we achieve the standard scaler objectives, by computing column-wise means and standard deviations within the Fit() phase. These values are stored internally to then get applied in the Transform() phase so that every data point gets adjusted accordingly. Once a feature has a zero variance, a small none zero value, an epsilon, is added to avoid division by zero. This ensures numeric stability.

In trading applications, the Standard Scaler is rather useful when handling data features that exist at different scales or timeframes. For instance, consider spreads. These are usually recorded as a pip fraction whereas, on the other hand, feature data such as tick volume is logged in the thousands. Without any ‘standardization’ the model may pay disproportionate attention to the larger-scale variable simply because of its size. When we ‘standardize’ both, the model evaluates the features on a relatively equal footing.

A simple use case can also involve preparing inputs such as log returns, moving averages, and volatility measures for a neural network. In applying this scaler, the features get normalized, improving the potential of models whether MLPs or even SVMs to converge efficiently when training.

### Min Max Scaler

Whereas the Standard Scaler normalizes features to a zero mean and unit variance, the min max scaler rescales features to a specific range, that is typically between 0 and 1. This transformation’s formula is as follows:

![f2](https://c.mql5.com/2/169/Form_minmax.png)

This transformation sees to it that all values are bounded within a desired range, which can be rather useful when using models that are sensitive to input magnitudes. Usually, this includes neural networks that have sigmoid or tanh activations. Our CMinMaxScaler class is implemented as follows in MQL5:

```
// MinMax Scaler
class CMinMaxScaler : public IPreprocessor
{
private:
   double m_mins[];
   double m_maxs[];
   double m_new_min;
   double m_new_max;
   bool   m_is_fitted;

public:
   CMinMaxScaler(double new_min = 0.0, double new_max = 1.0) : m_new_min(new_min), m_new_max(new_max), m_is_fitted(false) {}

   bool Fit(matrix &data)
   {  int rows = int(data.Rows());
      int cols = int(data.Cols());
      ArrayResize(m_mins, cols);
      ArrayResize(m_maxs, cols);
      for(int j = 0; j < cols; j++)
      {  vector column(rows);
         for(int i = 0; i < rows; i++) column[i] = data[i][j];
         m_mins[j] = column.Min();
         m_maxs[j] = column.Max();
         if(m_maxs[j] - m_mins[j] == 0.0) m_maxs[j] += EPSILON;
      }
      m_is_fitted = true;
      return true;
   }

   bool Transform(matrix &data, matrix &out)
   {  if(!m_is_fitted) return false;
      int rows = int(data.Rows());
      int cols = int(data.Cols());
      out.Init(rows, cols);
      for(int j = 0; j < cols; j++)
         for(int i = 0; i < rows; i++)
         {  if(!MathIsValidNumber(data[i][j])) out[i][j] = DBL_MIN;
            else
            {  double scale = (m_new_max - m_new_min) / (m_maxs[j] - m_mins[j]);
               out[i][j] = (data[i][j] - m_mins[j]) * scale + m_new_min;
            }
         }
      return true;
   }

   bool FitTransform(matrix &data, matrix &out)
   {  if(!Fit(data)) return false;
      return Transform(data, out);
   }
};
```

This class implements the behavior firstly by determining the minimum and maximum of each column during the Fit() phase. These determined values are then incorporated to rescale the data during the Transform() phase. Developers can specify custom bounds, such as -1 to +1 instead of the typical \[0,1\] in order to match expectations of the engaged model. As with the standard scaler, an epsilon ensures we do not get zero divisions in the even that all values are identical.

A practical trading example can involve rescaling closing prices before feeding them into an ONNX-based LSTM model. Given that neural networks tend to shine when inputs are bounded within a narrow range, the min max normalization does ensure smooth gradients and faster convergence. Likewise, when working with momentum indicators or oscillators with large absolute values, the min max scaler does bring them into a consistent and predictable range.

However, arguably, the main advantage of the min max scaler lies in its simplicity and ability to preserve the shape of the prior distribution. Unlike standardization, that changes the variance of the dataset, the min max scaler simply compresses values into a fixed interval. Nonetheless, this method can be sensitive to outliers, since a single extreme value can distort the scaling of the whole input feature data set. For datasets that are stable or when using this scaler in combination with the removal of wild outliers, it can be the best choice when preparing features/ input data to deep learning models.

### Robust Scaler

The markets are renown for their unpredictability and nurturing of outliers. Sudden news events often cause bid-ask spreads to widen dramatically or volume to spike way above historical averages. In cases such as these, preprocessing methods like the standard scaler or the min max scaler can become distorted, given that both disproportionately rely on the mean and extreme values of the data. It is in these situations therefore where the robust scaler proves invaluable.

The Robust Scaler centers data by its median, then scales it by its interquartile range, aka IQR. This IQR is defined as the difference between the 75th percentile, Q3, and the 25th percentile - Q1. This transformation is therefore expressed as follows:

![f3](https://c.mql5.com/2/169/form_robust__1.png)

Where:

- IQR is the gap between the 75th percentile Q3 and the 25th Q1;
- Median is the dataset’s median and;
- x is a data point in the dataset.


Given that this scaler ignores the influence of extreme values, it tends to be resistant to outliers. With trading data, this implies that even if a rare market shock happens, most of the features will remain suitably scaled for modelling. Our MQL5 implementation of this class is therefore as follows:

```
// Robust Scaler
class CRobustScaler : public IPreprocessor
{
private:
   double m_medians[];
   double m_iqrs[];
   bool   m_is_fitted;

public:
   CRobustScaler() : m_is_fitted(false) {}

   bool Fit(matrix &data)
   {  int rows = int(data.Rows());
      int cols = int(data.Cols());
      ArrayResize(m_medians, cols);
      ArrayResize(m_iqrs, cols);
      for(int j = 0; j < cols; j++)
      {  vector column(rows);
         for(int i = 0; i < rows; i++) column[i] = data[i][j];
         m_medians[j] = column.Median();
         double q25 = column.Quantile(0.25);
         double q75 = column.Quantile(0.75);
         m_iqrs[j] = q75 - q25;
         if(m_iqrs[j] == 0.0) m_iqrs[j] = EPSILON;
      }
      m_is_fitted = true;
      return true;
   }

   bool Transform(matrix &data, matrix &out)
   {  if(!m_is_fitted) return false;
      int rows = int(data.Rows());
      int cols = int(data.Cols());
      out.Init(rows, cols);
      for(int j = 0; j < cols; j++)
         for(int i = 0; i < rows; i++)
            out[i][j] = (!MathIsValidNumber(data[i][j]) ? DBL_MIN : (data[i][j] - m_medians[j]) / m_iqrs[j]);
      return true;
   }

   bool FitTransform(matrix &data, matrix &out)
   {  if(!Fit(data)) return false;
      return Transform(data, out);
   }
};
```

Our MQL5 CRobustScaler class above calculates medians and quartiles during the Fit() phase and then applies the scaling within the Transform() step. An epsilon safeguard is also used to guarantee stability in the event that the IQR happens to be zero. Our implementation makes it possible to process datasets that have the potential to mislead models because of irregular spikes in the market.

To illustrate, imagine training a model while using tick volume. In ‘normal’ sessions, volumes could hover around a stable range, however when a news release hits the wires, they can easily multiply by 10. Standardization or min-max scaling would stretch the feature distribution and compress the ‘normal’ values to a very tight and almost insignificant band. The Robust Scaler, however, does focus on the central 50 percent of the data, which tends to ensure that the majority of the input feature patterns to a model remain intact. This therefore makes it adept in deep learning, where models can be tasked with handling volatile, heavy tailed distributions that are commonly seen on forex or some crypto markets.

### One-Hot Encoding

In trading, not all features are numeric. Some are often discrete or categorical in nature, since they tend to map to specific forms of the market. For instance, we might classify time into trading sessions of Asia, Europe, and the US. Or we may want to group market regimes into bullish, bearish and flat. These examples are not exhaustive, however, the point here is that machine learning models cannot natively interpret such categorical, non-numeric values. And to emphasize this further, simply assigning integers to each category such as Asia = 0, Europe = 1, and US = 2, introduces a false sense of order and can bias the model.

The solution to this problem is often one-hot encoding. This is where each category gets transformed into a binary vector - for instance our market session classifications for Asia, Europe, and the US become \[1,0,0\], \[0,1,0\], and \[0,0,1\] respectively. This tends to allow models to distinguish categories without necessarily assuming any ordinal relationship. Our implementation of this in MQL5 is as follows:

```
// One-Hot Encoder
class COneHotEncoder : public IPreprocessor
{
private:
   int    m_column;
   double m_categories[];
   bool   m_is_fitted;

public:
   COneHotEncoder(int column) : m_column(column), m_is_fitted(false) {}

   bool Fit(matrix &data)
   {  int rows = int(data.Rows());
      vector values;
      int unique = 0;
      for(int i = 0; i < rows; i++)
      {  if(!MathIsValidNumber(data[i][m_column])) continue;
         int idx = CVectorUtils::BinarySearch(values, data[i][m_column]);
         if(idx == -1)
         {  values.Resize(unique + 1);
            values[unique] = data[i][m_column];
            unique++;
         }
      }
      values.Swap(m_categories);
      //ArrayCopy(m_categories, values);
      m_is_fitted = true;
      return true;
   }

   bool Transform(matrix &data, matrix &out)
   {  if(!m_is_fitted) return false;
      int rows = int(data.Rows());
      int cols = int(data.Cols());
      int cat_count = ArraySize(m_categories);
      if(data.Cols() == cols - 1 + cat_count) return false;
      out.Resize( rows, cols - 1 + cat_count);
      out.Fill(0.0);
      for(int i = 0; i < rows; i++)
      {  int out_col = 0;
         for(int j = 0; j < cols; j++)
         {  if(j == m_column) continue;
            out[i][out_col] = data[i][j];
            out_col++;
         }
         for(int k = 0; k < cat_count; k++)
            if(data[i][m_column] == m_categories[k])
            {  out[i][out_col + k] = 1.0;
               break;
            }
      }
      m_is_fitted = true;
      return true;
   }

   bool FitTransform(matrix &data, matrix &out)
   {  if(!Fit(data)) return false;
      return Transform(data, out);
   }
};
```

The COneHotEncoder class above implements this transformation. During the Fit() phase, identification of unique categories is done in the selected feature column. During the Transform() phase, the categorical column then gets replaced with multiple binary columns whose number represents the number of categories. The result is an expanded feature matrix where categorical information gets embedded in a model friendly format.

To demonstrate how this could be resourceful, we can stick to the examples above of encoding trading sessions. If we use raw session numbers of 0 for Asia, 1 for Europe, and 2 for US; a neural network may misinterpret the difference between Asia and the US as being greater than that between Asia and Europe in metrics that are not necessarily time related. By adopting one hot encoding, every session gets independent representation and the model becomes more free to learn distinct behaviors for each category. This is crucial for trading models given that different sessions, for instance, can exhibit unique liquidity, volatility and directional signals.

### Putting It All Together

Separately, scalers and encoders are powerful, however their true potential lies in being harnessed together into one workflow. This is the preprocessing pipeline. By augmenting together several transformations, we ensure that every dataset undergoes the exact treatment before it gets to the model.

Consider this scenario. Supposing we were preparing a dataset for which we have identified 4 key features:

1. Closing price
2. Tick volume
3. Spread
4. Trading session bucket (categorical)

The first step would be, usually, to handle missing values. This is something that some MQL5 developers could skimp over since data is often taken as is from the broker, however one would be advised to never assume all required data is present. For instance, if tick volume data is incomplete, we can apply the AddImputeMedian(1) function, where 1 represents the tick volume feature index within our 4 used features. This would replace missing entries with the median of the column. Likewise, if the trading session bucket is missing some data, we can apply the AddImputeMode(3) function to fill in the most common trading session as a compromise. These choices can be tuned by the developer depending on his system/ model, what is shared here is purely for illustration.

With the missing data addressed, our next step would be to transform the categorical data into an ‘impartial’ binary format. In accomplishing this, we would apply the AddOneHotEncoder(3) function to expand the session column into binary vectors, while also ensuring that each session is distinctly represented.

The next step, or step 3, could be to apply a scaler. Given our dataset, we can choose between AddStandardScaler(), AddRobustScaler(), or AddminMaxScaler() functions. Ultimately, this step is set up to ensure all numeric features are adjusted to be comparable in scale. Once these steps are added to the pipeline, we would then have to call the FitPipeline() function on the training dataset. This function would learn all the necessary parameters such as the Mean, Median, Mode, and Category Mappings. Later, we then get to call the TransformPipeline() function on both training and test datasets (or optimization and forward walk) and this would ensure consistency without leaking future information into the process of training.

In the end, our output is a clean, scaled, and encoded matrix of features that is immediately ready for use in a deep learning model - whether custom MQL5 coded or ONNX imported. This pipeline makes the preprocessing transparent, reproducible, and modular; which allows developers to put more emphasis and focus on signals or strategy and less on data wrangling. Running a demonstration of our pipeline classes to prepare data for a model can be done with the script that is attached at the end of the article. A test run on the symbol USD JPY, presents us with the following logs:

```
2025.09.12 17:05:50.150 Pipeline_Illustration (USDJPY,H4)       RAW (first 6 rows) [rows=2999, cols=4]
2025.09.12 17:05:50.150 Pipeline_Illustration (USDJPY,H4)         147.625000, 6894.000000, 20.000000, 2.000000
2025.09.12 17:05:50.150 Pipeline_Illustration (USDJPY,H4)         147.837000, 14153.000000, 20.000000, 1.000000
2025.09.12 17:05:50.150 Pipeline_Illustration (USDJPY,H4)         147.885000, 16794.000000, 20.000000, 1.000000
2025.09.12 17:05:50.150 Pipeline_Illustration (USDJPY,H4)         147.489000, 8010.000000, 20.000000, 0.000000
2025.09.12 17:05:50.150 Pipeline_Illustration (USDJPY,H4)         147.219000, 6710.000000, 20.000000, 0.000000
2025.09.12 17:05:50.150 Pipeline_Illustration (USDJPY,H4)         147.194000, 13686.000000, 20.000000, 3.000000
2025.09.12 17:05:50.163 Pipeline_Illustration (USDJPY,H4)       TRANSFORMED (FitTransform on all) [rows=2999, cols=32]
2025.09.12 17:05:50.163 Pipeline_Illustration (USDJPY,H4)         0.353976, 0.081606, 0.052632, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ...
2025.09.12 17:05:50.163 Pipeline_Illustration (USDJPY,H4)         0.363616, 0.167533, 0.052632, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ...
2025.09.12 17:05:50.163 Pipeline_Illustration (USDJPY,H4)         0.365798, 0.198795, 0.052632, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ...
2025.09.12 17:05:50.163 Pipeline_Illustration (USDJPY,H4)         0.347792, 0.094816, 0.052632, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ...
2025.09.12 17:05:50.163 Pipeline_Illustration (USDJPY,H4)         0.335516, 0.079428, 0.052632, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ...
2025.09.12 17:05:50.163 Pipeline_Illustration (USDJPY,H4)         0.334379, 0.162005, 0.052632, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, ...
2025.09.12 17:05:50.170 Pipeline_Illustration (USDJPY,H4)       TRAIN (after TransformPipeline) [rows=2249, cols=32]
2025.09.12 17:05:50.170 Pipeline_Illustration (USDJPY,H4)         0.353976, 0.081606, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ...
2025.09.12 17:05:50.170 Pipeline_Illustration (USDJPY,H4)         0.363616, 0.167533, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ...
2025.09.12 17:05:50.170 Pipeline_Illustration (USDJPY,H4)         0.365798, 0.198795, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ...
2025.09.12 17:05:50.170 Pipeline_Illustration (USDJPY,H4)         0.347792, 0.094816, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ...
2025.09.12 17:05:50.170 Pipeline_Illustration (USDJPY,H4)         0.335516, 0.079428, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ...
2025.09.12 17:05:50.170 Pipeline_Illustration (USDJPY,H4)         0.334379, 0.162005, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, ...
2025.09.12 17:05:50.170 Pipeline_Illustration (USDJPY,H4)       TEST  (after TransformPipeline) [rows=750, cols=32]
2025.09.12 17:05:50.170 Pipeline_Illustration (USDJPY,H4)         0.538217, 0.098806, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, ...
2025.09.12 17:05:50.170 Pipeline_Illustration (USDJPY,H4)         0.536307, 0.280804, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ...
2025.09.12 17:05:50.170 Pipeline_Illustration (USDJPY,H4)         0.545628, 0.163082, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ...
2025.09.12 17:05:50.170 Pipeline_Illustration (USDJPY,H4)         0.540217, 0.121817, -0.028571, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ...
2025.09.12 17:05:50.170 Pipeline_Illustration (USDJPY,H4)         0.533852, 0.093858, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ...
2025.09.12 17:05:50.170 Pipeline_Illustration (USDJPY,H4)         0.532215, 0.071675, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ...
2025.09.12 17:05:50.170 Pipeline_Illustration (USDJPY,H4)       Preprocessing pipeline demo complete.
```

### Comparison of Scalers in Trading

Selecting the correct scaler is less often a one-size-fits-all proposition. Every scaler does present its strengths and weaknesses, and the optimal choice usually is dependent on the type of data and behavior of the target market. The side by side tabulated comparison, below, could help clarify where each scaler excels:

| Scaler | Strengths | Weaknesses | Best for |
| --- | --- | --- | --- |
| Standard-Scaler | Suitable for Gaussian like data. Produces features with zero mean and unit variance. | Sensitive to outliers. | Good at handling technical indicators or volatility based solutions. |
| Min-Max-Scaler | Appropriate for bounded activations. Produces data in a fixed range, typically \[0,1\]. | It is very sensitive to outliers, as extreme values distort the scaling. | Ideal for price values and features, feeding a sigmoid/tanh network |
| Robust-Scaler | The use of medians and IQR, makes it less impervious to outliers. | Has potential to often lose the absolute scaling. | Suited for heavy tailed data, tick volumes and spreads during volatile market situations. |

What our summary above suggests, is if for instance we are training a simple neural network on normalized price returns and volatility indicators, then the Standard-Scaler would probably be the suitable choice given that it ensures comparability across features. Alternatively, if, say, the model is an ONNX LSTM trained in Python with sigmoid activations, then the min max scaler could be more appropriate, since it tends to align with the activation’s bounded range. On the flip side, if we were modelling trading behaviors during a high-impact news event, where spreads and volumes can spike dramatically, then the robust scaler can be used to help preserve the central structure of the data while minimizing this anomaly influence.

The main takeaway here is that preprocessing shouldn’t be treated as an afterthought. The choice of scaler directly affects how features interact within the learning steps, and can make all the difference between a model that generalizes well and one that is biased or unstable.

### Preparing Data for Deep Learning

What we refer to as deep-learning-models, or neural networks, whether coded in MQL5 or imported via ONNX tend to have strict expectations about their inputs. A model trained in Python on normalized data, for instance, will underperform or fail entirely if presented with raw and unscaled features of MQL5 data. This emphasizes the importance of preprocessing pipelines as something that is strictly speaking, not optional, but essential as far as serious trading and workflows are concerned.

Consider an ONNX model trained in TensorFlow on min-max normalized prices. If the same model is later deployed in MQL5 but with raw OHLC data, it goes without saying that the weights will be misaligned with the scaling of these inputs and this will result in poor forecasts. Applying the CMinMaxScaler() pipeline step with identical min/max settings as in the Python training will guarantee consistency between the two environments.

Beyond scaling, categorical encoding also serves a vital purpose.  A neural network trained with one hot encoding, such as the trading sessions example we considered above, will expect the same binary format at inference. If the encoding is inconsistent, where for example Asia was mapped to \[0,1,0\] in training but is now at \[1,0,0\] then the forecasts of the model will become meaning less. By engaging a pipeline that logs learned categories, we mitigate such risks.

Persistence is another key aspect. Parameters such as means, minimums, maximums, as well as category mappings need to be stored after training and then re-applied during inference. Without this constancy, retraining or inference does run the risk of drifting apart. In Python, SCIKIT-LEARN does provide serialization with modules such as joblib or pickle. In MQL5, developers could get a similar effect by saving the pipeline state variables from their arrays format into bin or csv files and have them reloaded during Expert Advisor initialization.

### Challenges & Best Practices

While we do have rigor in the MQL5 workflows thanks to these preprocessing pipelines, this does come with a few challenges that need to be addressed to ensure reliability. Of these, we consider five. First up is the handling of NaNs and dealing with missing data. Financial datasets usually contain missing values because of market closures, incomplete tick data or irregular broker data feeds. In MQL5 pipelines, placeholders such as DBL\_MIN are often engaged to flag for these missing entries. It is therefore vital to impute these values consistently by using alternatives such as medians if they are numeric or the mode if the data is discrete/ categorical. This prevents feeding invalid data into the model.

The second hiccup with these pipelines in MQL5 could be preventing [data leakage](https://en.wikipedia.org/wiki/Leakage_(machine_learning) "https://en.wikipedia.org/wiki/Leakage_(machine_learning)"). A common error is the fitting of scalers or encoders on the whole dataset before splitting into training, validation, and testing datasets is done. This practice can leak future information into the training process and artificially inflate performance metrics. MQL5’s strategy tester is inherently built to run by strictly reading current or old price information, but in the event that supplemental data is used and sourced to be fed to a model, then extra checks need to be in place to ensure such leakages do not occur. Best practice would be to always Fit() the pipeline on training data only, then Transform() both training and test sets separately.

Thirdly, scaling mixed data types can be challenging. When datasets contain both numeric and categorical features, transformations need to be applied with caution. Encoders need to be used before scalers, in order to ensure that newly created binary columns are appropriately integrated in to the new numeric matrix.

Fourthly, debugging transformations needs to be paid attention to. In order to verify the correctness of the transformation outputs, it is useful to print the first few rows of transformed data. Tools for debugging such as the Print() can reveal quickly whether encoding and scaling is behaving as expected.

Finally, ensuring reproducibility can not always be assumed. For consistency across experiments, pipeline parameters, minimums, medians, and category mappings need to be stored alongside the model that has been trained. This serves as a guarantee that the exact same preprocessing can be applied in back testing, live trading or even retraining.

Adherence to these five notes can help developers avoid common pitfalls by seeing to it that pipelines in MQL5 remain as robust as their Python counterparts.

### Conclusion

Preprocessing is certainly not a glamorous part of machine learning, but it is arguably one of the capstones. Without careful preparation, even the most advanced deep learning model will stumble when confronted with raw, unscaled, or inconsistently encoded trading data. For developers adept with MetaTrader 5, this challenge has traditionally been a hindrance to fully leveraging machine learning workflows. Unlike Python that offers SCIKIT-LEARN’s robust toolkit, MQL5 provides no built-in preprocessing pipelines. The solution, as demonstrated, could lie in building modular reusable preprocessing pipelines within MQL5.

In doing so, where these are not treated as optional utilities but as essential components of the AI workflow, developers will align their trading systems with the rigor of machine learning practice such that most models - whether native or ONNX based - has the best possible basis for success.

| name | description |
| --- | --- |
| PipeLine.mqh | Base class for pipeline functionality |
| Pipeline\_Illustration.mq5 | Script that references and shows use of the base class |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19544.zip "Download all attachments in the single ZIP archive")

[Pipeline\_Illustration.mq5](https://www.mql5.com/en/articles/download/19544/Pipeline_Illustration.mq5 "Download Pipeline_Illustration.mq5")(14.01 KB)

[PipeLine.mqh](https://www.mql5.com/en/articles/download/19544/PipeLine.mqh "Download PipeLine.mqh")(14.96 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/495650)**

![Introduction to MQL5 (Part 21): Automating Harmonic Pattern Detection](https://c.mql5.com/2/170/19331-introduction-to-mql5-part-21-logo.png)[Introduction to MQL5 (Part 21): Automating Harmonic Pattern Detection](https://www.mql5.com/en/articles/19331)

Learn how to detect and display the Gartley harmonic pattern in MetaTrader 5 using MQL5. This article explains each step of the process, from identifying swing points to applying Fibonacci ratios and plotting the full pattern on the chart for clear visual confirmation.

![Developing A Custom Account Performace Matrix Indicator](https://c.mql5.com/2/170/19508-developing-a-custom-account-logo.png)[Developing A Custom Account Performace Matrix Indicator](https://www.mql5.com/en/articles/19508)

This indicator acts as a discipline enforcer by tracking account equity, profit/loss, and drawdown in real-time while displaying a performance dashboard. It can help traders stay consistent, avoid overtrading, and comply with prop-firm challenge rules.

![Building AI-Powered Trading Systems in MQL5 (Part 1): Implementing JSON Handling for AI APIs](https://c.mql5.com/2/170/19562-building-ai-powered-trading-logo.png)[Building AI-Powered Trading Systems in MQL5 (Part 1): Implementing JSON Handling for AI APIs](https://www.mql5.com/en/articles/19562)

In this article, we develop a JSON parsing framework in MQL5 to handle data exchange for AI API integration, focusing on a JSON class for processing JSON structures. We implement methods to serialize and deserialize JSON data, supporting various data types like strings, numbers, and objects, essential for communicating with AI services like ChatGPT, enabling future AI-driven trading systems by ensuring accurate data handling and manipulation.

![The Parafrac V2 Oscillator: Integrating Parabolic SAR with Average True Range](https://c.mql5.com/2/170/19354-the-parafrac-v2-oscillator-logo.png)[The Parafrac V2 Oscillator: Integrating Parabolic SAR with Average True Range](https://www.mql5.com/en/articles/19354)

The Parafrac V2 Oscillator is an advanced technical analysis tool that integrates the Parabolic SAR with the Average True Range (ATR) to overcome limitations of its predecessor, which relied on fractals and was prone to signal spikes overshadowing previous and current signals. By leveraging ATR’s volatility measure, the version 2 offers a smoother, more reliable method for detecting trends, reversals, and divergences, helping traders reduce chart congestion and analysis paralysis.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/19544&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071563263969274461)

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