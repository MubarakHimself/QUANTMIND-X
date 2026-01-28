---
title: Data Science and ML (Part 26): The Ultimate Battle in Time Series Forecasting — LSTM vs GRU Neural Networks
url: https://www.mql5.com/en/articles/15182
categories: Expert Advisors, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T21:24:23.482823
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=wganknmnalsayhoiyrbdcjqadntmuxtn&ssn=1769192662544849439&ssn_dr=0&ssn_sr=0&fv_date=1769192662&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15182&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Data%20Science%20and%20ML%20(Part%2026)%3A%20The%20Ultimate%20Battle%20in%20Time%20Series%20Forecasting%20%E2%80%94%20LSTM%20vs%20GRU%20Neural%20Networks%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919266226336199&fz_uniq=5071808588206255830&sv=2552)

MetaTrader 5 / Expert Advisors


**Contents**

- [What is a Long Short-Term Memory (LSTM) neural network?](https://www.mql5.com/en/articles/15182#what-is-lstm)
- [Mathematics behind the Long-Short-Term Memory(LSTM) network](https://www.mql5.com/en/articles/15182#the-problem-with-simple-rnn)
- [What is a gated recurrent unit(GRU) neural network?](https://www.mql5.com/en/articles/15182#what-is-gru)
- [Mathematics behind the gated recurrent unit (GRU) network](https://www.mql5.com/en/articles/15182#maths-behind-gru)
- [Building the parent class for LSTM and GRU networks](https://www.mql5.com/en/articles/15182#building-gru-and-lstm-parent-class)
- [LSTM and GRU neural network child classes](https://www.mql5.com/en/articles/15182#lstm-gru-nn-child-classes)
- [Training both models](https://www.mql5.com/en/articles/15182#training-lstm-and-gru)
- [Checking feature importance for both models](https://www.mql5.com/en/articles/15182#feature-importance-gru-lstm)
- [LSTM versus GRU classifiers on the strategy tester](https://www.mql5.com/en/articles/15182#lstm-versus-gru-on-strategy-tester)
- [The differences between LSTM and GRU neural network models](https://www.mql5.com/en/articles/15182#differences-between-lstm-and-gru)
- [Conclusion](https://www.mql5.com/en/articles/15182#conclusion)

### What is a Long Short-Term Memory (LSTM) Neural Network?

Long Short-Term Memory(LSTM), is a type of [recurrent neural network](https://www.mql5.com/en/articles/15114#understanding-rnns) designed for sequence tasks, excelling in capturing and utilizing long-term dependencies in data. Unlike vanilla Recurrent Neural Networks(simple RNNs) discussed in the [previous article](https://www.mql5.com/en/articles/15114) of this series (a must-read). Which can't capture long-term dependencies in the data.

_LSTMs were introduced to fix the short-term memory which is prevalent in simple RNNs._

**The Problem with Simple Recurrent Neural Networks**

Simple Recurrent Neural Networks (RNNs) are designed to handle sequential data by using their [internal hidden state](https://www.mql5.com/en/articles/15114#mathematics-behind-rnn) (memory) to capture information about previous inputs in the sequence. Despite their conceptual simplicity and initial success in modeling sequential data, they have several limitations.

One significant issue is the vanishing gradient problem. During backpropagation, gradients are used to update the weights of the network. In simple RNNs, these gradients can diminish exponentially as they are propagated backward through time, especially for long sequences. This results in the network's inability to learn long-term dependencies, as the gradients become too small to make effective updates to the weights, making it difficult for simple RNNs to capture patterns that span over many time steps.

Another challenge is the exploding gradient problem, which is the opposite of the vanishing gradient problem. In this case, gradients grow exponentially during backpropagation. This can cause numerical instability and make the training process very challenging. Although less common than vanishing gradients, exploding gradients can lead to wildly large updates to the network's weights, effectively causing the learning process to fail.

Simple RNNs are also difficult to train due to their susceptibility to both vanishing and exploding gradient problems, which can make the training process inefficient and slow. Training simple RNNs can be more computationally expensive and may require careful tuning of hyperparameters.

Furthermore, simple RNNs are unable to handle complex temporal dependencies in data. Due to their limited memory capacity, they often struggle to understand and capture complex sequential patterns.

For tasks that involve an understanding of long-range dependencies in the data, simple RNNs may fail to capture the necessary context, leading to suboptimal performance.

### Mathematics Behind the Long Short-Term Memory(LSTM) Network

To understand the nitty-gritty behind LSTM, firstly let us look at the LSTM cell.

![lstm cell illustration](https://c.mql5.com/2/82/lstm_cell_illustration.png)

**01: Forget Gate**

Given by the equation.

![forget gate equation](https://c.mql5.com/2/82/forget_gate_formula.gif)

A sigmoid function  ![](https://c.mql5.com/2/82/sigma.gif) takes as input the previous hidden state  ![](https://c.mql5.com/2/82/previous_hidden_state.gif) and the current input ![](https://c.mql5.com/2/82/current_input.gif). The output  ![](https://c.mql5.com/2/82/current_output.gif) is a value between 0 and 1, indicating how much of each component in  ![](https://c.mql5.com/2/82/previous_cell_state.gif) (previous cell state) should be retained.

![](https://c.mql5.com/2/82/weight_forget_gate.gif) \- weight of the forget gate.

![](https://c.mql5.com/2/82/bias_forget_gate.gif) \- bias of the forget gate.

The forget gate determines which information from the previous cell state should be carried forward. It outputs a number between 0 and 1 for each number in the cell state  ![](https://c.mql5.com/2/82/previous_cell_state__1.gif), where **0 means _completely forget_** and **1 means _completely retain_.**

**02: Input Gate**

Given by the formula.

![](https://c.mql5.com/2/82/input_gate.gif)

A sigmoid function  ![](https://c.mql5.com/2/82/sigma__1.gif)determines which values to update. This gate controls the input of new data into the memory cell.

![](https://c.mql5.com/2/82/weight_input_gate.gif) \- weight input gate.

![](https://c.mql5.com/2/82/bias_input_gate.gif) \- bias input gate.

This gate decides which values from the new input  ![](https://c.mql5.com/2/82/current_input__1.gif) are used to update the cell state. It regulates the flow of new information into the cell.

**03: Candidate Memory Cell**

Given by the equation.

![](https://c.mql5.com/2/82/candidate_memory_cell_formula__1.gif)

A tanh function generates potential new information that could be stored in the cell state.

![](https://c.mql5.com/2/82/weight_candidate_memory_cell.gif) -  Weight of the candidate memory cell.

![](https://c.mql5.com/2/82/bias_candidate_memory_cell.gif) -  Bias of the candidate memory cell.

This component generates the new candidate values that can be added to the cell state. It uses the tanh activation function to ensure the values are between -1 and 1.

**04: Cell State Update**

Given by the equation.

![](https://c.mql5.com/2/82/cell_state_update_formula.gif)

The previous cell state  ![](https://c.mql5.com/2/82/previous_cell_state__2.gif)  is multiplied by  ![](https://c.mql5.com/2/82/current_output__1.gif) (forget gate output) to discard unimportant information. Then,  ![](https://c.mql5.com/2/82/it.gif) (input gate output) is multiplied by  ![](https://c.mql5.com/2/82/candidate_cell_state.gif) ​(candidate cell state), and the results are summed to form the new cell state  ![](https://c.mql5.com/2/82/ct.gif).

The cell state is updated by combining the old cell state and the candidate values. The forget gate output controls the previous cell state contribution, and the input gate output controls the new candidate values' contribution.

**05: Output Gate**

Given by the equation.

![](https://c.mql5.com/2/82/output_gate_formula.gif)

A sigmoid function determines which parts of the cell state to output. This gate controls the output of information from the memory cell.

![](https://c.mql5.com/2/82/weight_output_layer.gif) \- Weight of the output layer

![](https://c.mql5.com/2/82/bias_output_layer.gif) \- Bias of the output layer

This gate determines the final output for the current cell state. It decides which parts of the cell state should be output based on the input  ![](https://c.mql5.com/2/82/current_input__2.gif) and the previous hidden state ![](https://c.mql5.com/2/82/previous_hidden_state__1.gif).

**06: Hidden State Update**

Given by the equation.

![](https://c.mql5.com/2/82/hidden_state_update_formula.gif)

The new hidden state  ![](https://c.mql5.com/2/82/ht.gif) is obtained by multiplying the output gate  ![](https://c.mql5.com/2/82/ot.gif) with the tanh of the updated cell state ![](https://c.mql5.com/2/82/ct__1.gif) .

The hidden state is updated based on the cell state and the output gate's decision. It is used as the output for the current time step and as an input for the next time step

### What is a Gated Recurrent Unit(GRU) Neural Network?

The Gated Recurrent Unit (GRU) is a type of Recurrent Neural Network (RNN) that, in certain cases, has advantages over long short-term memory (LSTM). GRU uses less memory and is faster than LSTM, however, LSTM is more accurate when using datasets with longer sequences.

LSTMs and GRUs were introduced to mitigate short-term memory prevalent in simple recurrent neural networks. Both have long-term memory enabled by using the gates in their cells.

Despite working similarly to simple RNNs in many ways, LSTMs and GRUs address the vanishing gradient problem from which simple recurrent neural networks suffer.

### Mathematics Behind the Gated Recurrent Unit (GRU) Network

The image below illustrates how the GRU cell looks when dissected.

![GRU cell illustration](https://c.mql5.com/2/82/gru_cell_illustration.png)

**01: The Update Gate**

Given by the formula.

![](https://c.mql5.com/2/82/update_gate.gif)

This gate determines how much of the previous hidden state  ![](https://c.mql5.com/2/82/previous_hidden_state__2.gif) should be retained and how much of the candidate hidden state  ![](https://c.mql5.com/2/82/ht__1.gif) should be used to update the hidden state.

The update gate controls how much of the previous hidden state  ![](https://c.mql5.com/2/82/previous_hidden_state__3.gif) should be carried forward to the next time step. It effectively decides the balance between keeping the old information and incorporating new information.

**02: Reset Gate**

Given by the formula.

![](https://c.mql5.com/2/82/reset_gate.gif)

The sigmoid function  ![](https://c.mql5.com/2/82/sigma__2.gif) in this gate, determines which parts of the previous hidden state should be reset before combining with the current input to create the candidate activation.

The reset gate determines how much of the previous hidden state should be forgotten before computing the new candidate activation. It allows the model to drop irrelevant information from the past

**03: Candidate Activation**

Given by the formula.

![](https://c.mql5.com/2/82/candidate_activation.gif)

The candidate activation is computed using the current input  ![](https://c.mql5.com/2/82/current_input__3.gif) and the reset hidden state ​ ![](https://c.mql5.com/2/82/rt.gif).

This component generates new potential values for the hidden state that can be incorporated based on the update gate's decision.

**04: Hidden State Update**

Given by the formula.

![](https://c.mql5.com/2/82/hidden_state_update_gru.gif)

The update gate output  ![](https://c.mql5.com/2/82/zt.gif) controls how much of the candidate hidden state ![](https://c.mql5.com/2/82/ht__2.gif) is used to form the new hidden state ![](https://c.mql5.com/2/82/ht__3.gif).

The hidden state is updated by combining the previous hidden state and the candidate hidden state. The update gate  ![](https://c.mql5.com/2/82/zt__1.gif) controls this combination, ensuring that the relevant information from the past is retained while incorporating new information.

### Building the Parent Class for LSTM and GRU Networks

Since LSTM and GRU work similarly in many ways and they take the same parameters, it might be a good idea to have a **base(parent) class** for the functions necessary for building, compiling, optimizing, checking feature importance, and saving the models. This class will be inherited in the subsequent LSTM and GRU child classes.

```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import tf2onnx
import optuna
import shap
from sklearn.metrics import accuracy_score

class RNNClassifier():
    def __init__(self, time_step, x_train, y_train, x_test, y_test):
        self.model = None
        self.time_step = time_step
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    # a crucial function that all the subclasses must implement

    def build_compile_and_train(self, params, verbose=0):
        raise NotImplementedError("Subclasses should implement this method")

    # a function for saving the RNN model to onnx & the Standard scaler parameters

    def save_onnx_model(self, onnx_file_name):

    # optuna objective function to oprtimize
    def optimize_objective(self, trial):

    # optimize for 50 trials by default
    def optimize(self, n_trials=50):


    def _rnn_predict(self, data):


    def check_feature_importance(self, feature_names):
```

**Optimizing the LSTM and GRU using [Optuna](https://www.mql5.com/go?link=https://optuna.readthedocs.io/en/stable/ "https://optuna.readthedocs.io/en/stable/")**

As [said once](https://www.mql5.com/en/articles/14926#:~:text=Classifier%20Models%20like%20neural%20networks%20are%20very%20sensitive%20to%20hyperparameters.%C2%A0Default%20parameters%20won%27t%20help.), Neural networks are very sensitive to hyperparameters. Without the right tuning and not having the optimal parameters, neural networks could be ineffective.

Python

```
def optimize_objective(self, trial):
    params = {
        "neurons": trial.suggest_int('neurons', 10, 100),
        "n_hidden_layers": trial.suggest_int('n_hidden_layers', 1, 5),
        "dropout_rate": trial.suggest_float('dropout_rate', 0.1, 0.5),
        "learning_rate": trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        "hidden_activation_function": trial.suggest_categorical('hidden_activation_function', ['relu', 'tanh', 'sigmoid']),
        "loss_function": trial.suggest_categorical('loss_function', ['categorical_crossentropy', 'binary_crossentropy', 'mean_squared_error', 'mean_absolute_error'])
    }

    val_accuracy = self.build_compile_and_train(params, verbose=0) # we build a model with different parameters and train it, just to return a validation accuracy value

    return val_accuracy

# optimize for 50 trials by default
def optimize(self, n_trials=50):
    study = optuna.create_study(direction='maximize') # we want to find the model with the highest validation accuracy value
    study.optimize(self.optimize_objective, n_trials=n_trials)

    return study.best_params # returns the parameters that produced the best performing model
```

The method **optimize\_objective** defines the objective function for hyperparameter optimization using the Optuna framework. It guides the optimization process to find the best set of hyperparameters that maximize the model's performance.

The method **Optimize** uses Optuna to perform hyperparameter optimization by repeatedly calling the **optimize\_objective** method.

**Checking Feature Importance using SHAP**

Measuring how impactful features are to the model's predictions is important to a data scientist, It could not only help us understand the areas for key improvements but also, sharpen our understanding of a particular dataset about a model.

```
def check_feature_importance(self, feature_names):

    # Sample a subset of training data for SHAP explainer
    sampled_idx = np.random.choice(len(self.x_train), size=100, replace=False)
    explainer = shap.KernelExplainer(self._rnn_predict, self.x_train[sampled_idx].reshape(100, -1))

    # Get SHAP values for the test set
    shap_values = explainer.shap_values(self.x_test[:100].reshape(100, -1), nsamples=100)

    # Update feature names for SHAP
    feature_names = [f'{feature}_t{t}' for t in range(self.time_step) for feature in feature_names]

    # Plot the SHAP values
    shap.summary_plot(shap_values, self.x_test[:100].reshape(100, -1), feature_names=feature_names, max_display=len(feature_names), show=False)

    # Adjust layout and set figure size
    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.9, top=0.9)
    plt.gcf().set_size_inches(7.5, 14)
    plt.tight_layout()

    # Get the class name of the current instance
    class_name = self.__class__.__name__

    # Create the file name using the class name
    file_name = f"{class_name.lower()}_feature_importance.png"

    plt.savefig(file_name)
    plt.show()
```

**Saving the LSTM and GRU classifiers to ONNX model formats**

Finally, after we have built the models, we have to save them in [ONNX format](https://www.mql5.com/en/articles/13394) which is compatible with MQL5.

```
def save_onnx_model(self, onnx_file_name):
    # Convert the Keras model to ONNX
    spec = (tf.TensorSpec((None, self.time_step, self.x_train.shape[2]), tf.float16, name="input"),)
    self.model.output_names = ['outputs']

    onnx_model, _ = tf2onnx.convert.from_keras(self.model, input_signature=spec, opset=13)

    # Save the ONNX model to a file
    with open(onnx_file_name, "wb") as f:
        f.write(onnx_model.SerializeToString())
    # Save the mean and scale parameters to binary files
    scaler.mean_.tofile(f"{onnx_file_name.replace('.onnx','')}.standard_scaler_mean.bin")
    scaler.scale_.tofile(f"{onnx_file_name.replace('.onnx','')}.standard_scaler_scale.bin")
```

### LSTM and GRU Neural Network Child Classes

Recurrent neural networks work similarly in many ways, even their implementation using [Keras](https://www.mql5.com/go?link=https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM "https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM") follows a similar approach and parameters. Their major difference is the type of the model, everything else remains the same.

**LSTM classifier**

Python

```
class LSTMClassifier(RNNClassifier):
    def build_compile_and_train(self, params, verbose=0):
        self.model = Sequential()
        self.model.add(Input(shape=(self.time_step, self.x_train.shape[2])))
        self.model.add(LSTM(units=params["neurons"], activation='relu', kernel_initializer='he_uniform')) # input layer

        for layer in range(params["n_hidden_layers"]): # dynamically adjusting the number of hidden layers
            self.model.add(Dense(units=params["neurons"], activation=params["hidden_activation_function"], kernel_initializer='he_uniform'))
            self.model.add(Dropout(params["dropout_rate"]))

        self.model.add(Dense(units=len(classes_in_y), activation='softmax', name='output_layer', kernel_initializer='he_uniform')) # the output layer

        # Compile the model
        adam_optimizer = Adam(learning_rate=params["learning_rate"])
        self.model.compile(optimizer=adam_optimizer, loss=params["loss_function"], metrics=['accuracy'])

        if verbose != 0:
            self.model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = self.model.fit(self.x_train, self.y_train, epochs=100, batch_size=32,
                                 validation_data=(self.x_test, self.y_test),
                                 callbacks=[early_stopping], verbose=verbose)

        val_loss, val_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=verbose)
        return val_accuracy
```

**GRU Classifier**

Python

```
class GRUClassifier(RNNClassifier):
    def build_compile_and_train(self, params, verbose=0):
        self.model = Sequential()
        self.model.add(Input(shape=(self.time_step, self.x_train.shape[2])))
        self.model.add(GRU(units=params["neurons"], activation='relu', kernel_initializer='he_uniform')) # input layer

        for layer in range(params["n_hidden_layers"]): # dynamically adjusting the number of hidden layers
            self.model.add(Dense(units=params["neurons"], activation=params["hidden_activation_function"], kernel_initializer='he_uniform'))
            self.model.add(Dropout(params["dropout_rate"]))

        self.model.add(Dense(units=len(classes_in_y), activation='softmax', name='output_layer', kernel_initializer='he_uniform')) # the output layer

        # Compile the model
        adam_optimizer = Adam(learning_rate=params["learning_rate"])
        self.model.compile(optimizer=adam_optimizer, loss=params["loss_function"], metrics=['accuracy'])

        if verbose != 0:
            self.model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = self.model.fit(self.x_train, self.y_train, epochs=100, batch_size=32,
                                 validation_data=(self.x_test, self.y_test),
                                 callbacks=[early_stopping], verbose=verbose)

        val_loss, val_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=verbose)
        return val_accuracy
```

_As can be seen in the child classes classifiers, the only difference is the type of the model, both LSTMs and GRUs take a similar approach._

### Training Both Models

Firstly, we have to initialize the class instances for both models. Starting with the LSTM model.

```
lstm_clf = LSTMClassifier(time_step=time_step,
                          x_train= x_train_seq,
                          y_train= y_train_encoded,
                          x_test= x_test_seq,
                          y_test= y_test_encoded
                         )
```

Then we initialize the GRU model.

```
gru_clf = GRUClassifier(time_step=time_step,
                          x_train= x_train_seq,
                          y_train= y_train_encoded,
                          x_test= x_test_seq,
                          y_test= y_test_encoded
                         )
```

After optimizing both models for 20 trials;

```
best_params = lstm_clf.optimize(n_trials=20)
```

```
best_params = gru_clf.optimize(n_trials=20)
```

The LSTM classifier model at trial 19 was the best.

```
[I 2024-07-01 11:14:40,588] Trial 19 finished with value: 0.5597269535064697 and parameters: {'neurons': 79, 'n_hidden_layers': 4, 'dropout_rate': 0.335909076638275, 'learning_rate': 3.0704319088493336e-05, 'hidden_activation_function': 'relu', 'loss_function': 'categorical_crossentropy'}.
Best is trial 19 with value: 0.5597269535064697.
```

Yielding an accuracy of approximately 55.97% on validation data meanwhile, the GRU classifier model found at trial 3 was the best of all models.

```
[I 2024-07-01 11:18:52,190] Trial 3 finished with value: 0.532423198223114 and parameters: {'neurons': 55, 'n_hidden_layers': 5, 'dropout_rate': 0.2729838602302831, 'learning_rate': 0.009626688728041802, 'hidden_activation_function': 'sigmoid', 'loss_function': 'mean_squared_error'}.
Best is trial 3 with value: 0.532423198223114.
```

It provided an accuracy of approximately 53.24% on validation data.

### Checking Feature Importance for Both Models

| LSTM classifier | GRU classifier |
| --- | --- |
| ```<br>feature_importance = lstm_clf.check_feature_importance(X.columns)<br>```<br>Outcome.<br>![](https://c.mql5.com/2/82/lstmclassifier_feature_importance.png) | ```<br>feature_importance = gru_clf.check_feature_importance(X.columns)<br>```<br>Outcome.<br>![gru classifier feature importance](https://c.mql5.com/2/82/gruclassifier_feature_importance.png) |

The LSTM classifier feature importance looks somehow similar to the one we [obtained with the simple RNN model](https://www.mql5.com/en/articles/15114#feature-importance-rnn). The least important variables are from far time steps meanwhile the most important features are ones from closer timesteps.

_**This is like saying, that the variables that contribute the most to what happens to the current bar are the recent closed bar information.**_

The GRU classifier had a diverse opinion that doesn't seem to make much sense. _This could be because its model had a lower accuracy._

It said the most impactful variable was the **day of the week** 7 days prior. Features such as Open, High, Low,, and Close from the time step value of 6 which is the very recent information, were placed in the middle indicating they had an average contribution to the final prediction outcome.

### LSTM Versus GRU classifiers on the Strategy Tester

Shortly after training, both LSTM and GRU classifier models were saved to ONNX format.

LSTM \| Python

```
lstm_clf.build_compile_and_train(best_params, verbose=1) # best_params = best parameters obtained after optimization

lstm_clf.save_onnx_model("lstm.EURUSD.D1.onnx")
```

GRU \| Python

```
gru_clf.build_compile_and_train(best_params, verbose=1)

gru_clf.save_onnx_model("gru.EURUSD.D1.onnx") # best_params = best parameters obtained after optimization
```

After saving the ONNX model and its scaler files under the MQL5\\Files directory, we can add the files to both Expert Advisors as resource files.

| LSTM | GRU |
| --- | --- |
| ```<br>#resource "\\Files\\lstm.EURUSD.D1.onnx" as uchar onnx_model[]; //lstm model in onnx format<br>#resource "\\Files\\lstm.EURUSD.D1.standard_scaler_mean.bin" as double standardization_mean[];<br>#resource "\\Files\\lstm.EURUSD.D1.standard_scaler_scale.bin" as double standardization_std[];<br>#include <MALE5\Recurrent Neural Networks(RNNs)\LSTM.mqh><br>CLSTM lstm;<br>#include <MALE5\preprocessing.mqh><br>StandardizationScaler *scaler; //For loading the scaling technique<br>``` | ```<br>#resource "\\Files\\gru.EURUSD.D1.onnx" as uchar onnx_model[]; //gru model in onnx format<br>#resource "\\Files\\gru.EURUSD.D1.standard_scaler_mean.bin" as double standardization_mean[];<br>#resource "\\Files\\gru.EURUSD.D1.standard_scaler_scale.bin" as double standardization_std[];<br>#include <MALE5\Recurrent Neural Networks(RNNs)\GRU.mqh><br>CGRU gru;<br>#include <MALE5\preprocessing.mqh><br>StandardizationScaler *scaler; //For loading the scaling technique<br>``` |

The code for the rest of the Expert Advisors remains the same [as we discussed](https://www.mql5.com/en/articles/15114#making-rnn-expert-advisor).

Using default settings we have used since [Part 24 of this article series](https://www.mql5.com/en/articles/15013), where we started with Timeseries forecasting.

**Stop loss: 500, Take profit: 700, Slippage: 50.**

Again, since the data was collected on a daily timeframe it might be a good idea to test it on a lower timeframe to avoid errors when " _market closed errors_" since we are looking for trading signals at the opening of a new bar. We can also set the Modelling type to open prices for faster testing.

![tester settings](https://c.mql5.com/2/82/bandicam_2024-06-13_18-12-52-463.png)

**LSTM Expert Advisor results**

![lstm ea tester report](https://c.mql5.com/2/82/lstm_report.png)

![lstm EA tester report](https://c.mql5.com/2/82/lstm_graph.png)

**GRU Expert Advisor results**

![gru EA tester report ](https://c.mql5.com/2/82/gru_report.png)

![gru expert advisor tester graph](https://c.mql5.com/2/82/gru_graph.png)

**What can we learn from the Strategy Tester outcomes**

Despite being the least accurate model with 44.98%, LSTM-based Expert Advisor was the most profitable with a net profit of 138 $, followed by the GRU-based Expert Advisor which was profitable 45.25% of the time, despite giving a total net profit of 120 $.

LSTM is a clear winner in this case _profits-wise_. Despite LSTM being technically smarter than other RNNs of its kind, there could be a lot of factors leading to this all recurrent models are good and can outperform others in certain situations, feel free to use any of the models discussed in this and the previous article.

### The Differences Between LSTM and GRU Neural Network Models

Understanding these models in comparison helps when deciding what each model offers in contrast to the other. When one should be used, and when it shouldn't. Below are their tabulated differences.

| Aspect | LSTM | GRU |
| --- | --- | --- |
| Architecture Complexity | **LSTMs have a more complex design** with three gates (input, output, forget) and a cell state, providing detailed control over what information is kept or discarded at each time step. | **GRUs have a simpler design** with only two gates (reset and update). This simple  architecture makes them easier to implement. |
| Training Speed | Having additional gates and a cell state in LSTMs means there is more process to be done and parameters to optimize. **They are slower during training.** | Due to having fewer with fewer gates and simpler operations, **they typically train faster than LSTMs**. |
| Performance | In complex problems where capturing long-term dependencies is crucial LSTMs tend to perform slightly better than their counterparts. | GRUs usually deliver comparable performance to LSTMs for many tasks. |
| Handling of Long-Term Dependencies | LSTMs are explicitly designed to retain long-term dependencies in the data, thanks to the cell state and the gating mechanisms that control information flow over time. | While GRUs also handle long-term dependencies well, they may not be as effective as LSTMs in capturing very long-term dependencies due to their simpler structure. |
| Memory Usage | Due to their complex structure and additional parameters, LSTMs consume more memory, which can be a limitation in resource-constrained environments. | GRUs on the other hand are simpler, have fewer parameters, and uses less memory. Making them more suitable for applications with limited computational resources. |

### Final Thoughts

Both LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) neural networks are powerful tools for traders seeking to leverage advanced time-series forecasting models. While LSTMs provide a more intricate architecture that excels at capturing long-term dependencies in market data, GRUs offer a simpler and more efficient alternative that can often match the performance of LSTMs with less computational costs.

These Timeseries deep learning models(LSTM and GRU), have been utilized in various domains outside forex trading such as weather forecasting, energy consumption modeling, anomaly detection, and speech recognition with great success as usually hyped however, In the forever-changing forex market I can not guarantee such promises.

This article aimed only to provide an understanding of these models in depth and how they can be deployed in MQL5 for trading. Feel free to explore and play with the models and datasets discussed in this article and share your results in the discussion section.

Best regards.

Track development of machine learning models and much more discussed in this article series on this [GitHub repo](https://www.mql5.com/go?link=https://github.com/MegaJoctan/MALE5 "https://github.com/MegaJoctan/MALE5").

**Attachments Table**

| File name | File type | Description & Usage |
| --- | --- | --- |
| GRU EA.mq5<br>LSTM EA.mq5 | Expert Advisors | GRU based Expert Advisor.<br>LSTM based Expert Advisor. |
| gru.EURUSD.D1.onnx<br>lstm.EURUSD.D1.onnx | ONNX files | GRU model in ONNX format.<br>LSTM model in ONNX format. |
| lstm.EURUSD.D1.standard\_scaler\_mean.bin <br>lstm.EURUSD.D1.standard\_scaler\_scale.bin | Binary files | Binary files for the Standardization scaler used for the LSTM model. |
| gru.EURUSD.D1.standard\_scaler\_mean.bin <br>gru.EURUSD.D1.standard\_scaler\_scale.bin | Binary files | Binary files for the Standardization scaler used for the GRU model. |
| preprocessing.mqh | An Include file | A library which consists of the Standardization Scaler. |
| [lstm-gru-for-forex-trading-tutorial.ipynb](https://www.mql5.com/go?link=https://www.kaggle.com/code/omegajoctan/lstm-gru-for-forex-trading-tutorial "https://www.kaggle.com/code/omegajoctan/rnns-for-forex-forecasting-tutorial/notebook") | Python Script/Jupyter Notebook | Consists all the python code discussed in this article |

**Sources & References**

- [Illustrated Guide to LSTM's and GRU's: A step by step explanation](https://www.mql5.com/go?link=https://youtu.be/8HyCNIVRbSU "https://youtu.be/8HyCNIVRbSU")
- [Designing neural network based decoders for surface codes](https://www.mql5.com/go?link=https://www.researchgate.net/publication/329362532_Designing_neural_network_based_decoders_for_surface_codes "https://www.researchgate.net/publication/329362532_Designing_neural_network_based_decoders_for_surface_codes")
- [An Adaptive Anti-Noise Neural Network for Bearing Fault Diagnosis Under Noise and Varying Load Conditions](https://www.mql5.com/go?link=https://www.researchgate.net/publication/340821036_An_Adaptive_Anti-Noise_Neural_Network_for_Bearing_Fault_Diagnosis_Under_Noise_and_Varying_Load_Conditions "https://www.researchgate.net/publication/340821036_An_Adaptive_Anti-Noise_Neural_Network_for_Bearing_Fault_Diagnosis_Under_Noise_and_Varying_Load_Conditions")

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15182.zip "Download all attachments in the single ZIP archive")

[Attachments.zip](https://www.mql5.com/en/articles/download/15182/attachments.zip "Download Attachments.zip")(634.47 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Python-MetaTrader 5 Strategy Tester (Part 04): Tester 101](https://www.mql5.com/en/articles/20917)
- [Python-MetaTrader 5 Strategy Tester (Part 03): MT5-Like Trading Operations — Handling and Managing](https://www.mql5.com/en/articles/20782)
- [Python-MetaTrader 5 Strategy Tester (Part 02): Dealing with Bars, Ticks, and Overloading Built-in Functions in a Simulator](https://www.mql5.com/en/articles/20455)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 06): Python-Like File IO operations in MQL5](https://www.mql5.com/en/articles/20695)
- [Data Science and ML (Part 47): Forecasting the Market Using the DeepAR model in Python](https://www.mql5.com/en/articles/20571)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 05): The Logging module from Python, Log Like a Pro](https://www.mql5.com/en/articles/20458)
- [Implementing Practical Modules from Other Languages in MQL5 (Part 04): time, date, and datetime modules from Python](https://www.mql5.com/en/articles/19035)

**[Go to discussion](https://www.mql5.com/en/forum/470090)**

![GIT: What is it?](https://c.mql5.com/2/69/GIT__Mas_que_coisa_2_esta___LOGO.png)[GIT: What is it?](https://www.mql5.com/en/articles/12516)

In this article, I will introduce a very important tool for developers. If you are not familiar with GIT, read this article to get an idea of what it is and how to use it with MQL5.

![DoEasy. Service functions (Part 1): Price patterns](https://c.mql5.com/2/71/DoEasy._Service_functions_Part_1___LOGO.png)[DoEasy. Service functions (Part 1): Price patterns](https://www.mql5.com/en/articles/14339)

In this article, we will start developing methods for searching for price patterns using timeseries data. A pattern has a certain set of parameters, common to any type of patterns. All data of this kind will be concentrated in the object class of the base abstract pattern. In the current article, we will create an abstract pattern class and a Pin Bar pattern class.

![Population optimization algorithms: Resistance to getting stuck in local extrema (Part II)](https://c.mql5.com/2/72/Population_optimization_algorithms__Resistance_to_getting_stuck_in_local_extrema__LOGO__1.png)[Population optimization algorithms: Resistance to getting stuck in local extrema (Part II)](https://www.mql5.com/en/articles/14212)

We continue our experiment that aims to examine the behavior of population optimization algorithms in the context of their ability to efficiently escape local minima when population diversity is low and reach global maxima. Research results are provided.

![Introduction to MQL5 (Part 8): Beginner's Guide to Building Expert Advisors (II)](https://c.mql5.com/2/84/Introduction_to_MQL5_Part_8_Beginners_Guide_to_Building_Expert_Advisors___LOGO.png)[Introduction to MQL5 (Part 8): Beginner's Guide to Building Expert Advisors (II)](https://www.mql5.com/en/articles/15299)

This article addresses common beginner questions from MQL5 forums and demonstrates practical solutions. Learn to perform essential tasks like buying and selling, obtaining candlestick prices, and managing automated trading aspects such as trade limits, trading periods, and profit/loss thresholds. Get step-by-step guidance to enhance your understanding and implementation of these concepts in MQL5.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/15182&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071808588206255830)

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