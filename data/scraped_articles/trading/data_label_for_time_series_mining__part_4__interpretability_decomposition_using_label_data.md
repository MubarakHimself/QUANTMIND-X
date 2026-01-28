---
title: Data label for time series mining (Part 4)：Interpretability Decomposition Using Label Data
url: https://www.mql5.com/en/articles/13218
categories: Trading, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:29:51.927192
---

[![](https://www.mql5.com/ff/sh/7h2yc16rtqsn2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Market analysis\\
\\
Dozens of channels, thousands of subscribers and daily updates. Learn more about trading.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=glufvbpblsoxonicqfngsyuzwfebnilr&s=103cc3ab372a16872ca1698fc86368ffe3b3eaa21b59b4006d5c6c10f48ad545&uid=&ref=https://www.mql5.com/en/articles/13218&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5082904442942001438)

MetaTrader 5 / Examples


### Introduction

In the previous article of this series, we mentioned the NHITS model, where we only validated the prediction of closing prices for a single variable input. In this article, we will discuss the Interpretability of the model and about using multiple covariates to predict closing prices. We will use a different model, NBEATS, for demonstration, to provide more possibilities. However, it should be noted that the focus of this article should be on the Interpretability of the model, and the answer to why the topic of covariates is also introduced will be given. So that you can use different models to verify your ideas at any time. Of course, these two models are essentially high-quality interpretable models, and you can also extend to other models to verify your ideas with the libraries mentioned in my article. It is worth mentioning that this series of articles aims to provide solutions to problems, please consider carefully before directly applying it to your real trading, the real trading implementation may require more parameter adjustments and more optimization methods to provide reliable and stable results.

The links to the preceding three articles are:

- [Data label for time series mining (Part 1)：Make a dataset with trend markers through the EA operation chart](https://www.mql5.com/en/articles/13225)
- [Data label for timeseries mining (Part 2) ：Make datasets with trend markers using Python](https://www.mql5.com/en/articles/13253)
- [Data label for time series mining (Part 3)：Example for using label data](https://www.mql5.com/en/articles/13255)

Table of contents:

- [Introduction](https://www.mql5.com/en/articles/13218#para1)
- [About NBEATS](https://www.mql5.com/en/articles/13218#para2)
- [Import libraries](https://www.mql5.com/en/articles/13218#para3)
- [Rewrite The TimeSeriesDataSet Class](https://www.mql5.com/en/articles/13218#para4)
- [Data Processing](https://www.mql5.com/en/articles/13218#para5)
- [Get Learning Rate](https://www.mql5.com/en/articles/13218#para6)
- [Define The Training Function](https://www.mql5.com/en/articles/13218#para7)
- [Model Training and Testing](https://www.mql5.com/en/articles/13218#para8)
- [Interpret Model](https://www.mql5.com/en/articles/13218#para9)
- [Conclusion](https://www.mql5.com/en/articles/13218#para10)

### About NBEATS

This model has been extensively cited and explained in various journals and websites. However, to save you from constantly shuttling between different websites, I decided to give a simple introduction to this model. NBEATS can handle input and output sequences of any length and does not depend on specific feature engineering or input scaling for time series. It can also use polynomials and Fourier series as base functions for interpretable configurations to simulate trend and seasonal decomposition. Moreover, this model adopts a dual residual stacking topology, so that each building block has two residual branches, one along the reverse prediction and the other along the forward prediction, greatly improving the trainability and interpretability of the model. Wow, it looks very impressive!

The specific paper address is: [https://arxiv.org/pdf/1905.10437.pdf](https://www.mql5.com/go?link=https://arxiv.org/pdf/1905.10437.pdf "https://arxiv.org/pdf/1905.10437.pdf")

**1.Model Architecture**![f1](https://c.mql5.com/2/61/modelpng.png)

**2.Model Implementation Process**

The input time series (dimension is length) is mapped to a low-dimensional vector (dimension is dim), and the second part maps it back to the time series (length is length). This step is also similar to AutoEncoder, which maps the time series to a low-dimensional vector to save core information, and then restores it. This process can be simply represented as:

![f2](https://c.mql5.com/2/61/f2.png)

The module will generate two sets of expansion coefficients, one for predicting the future (forecast), and the other for predicting the past (backcast). This process can be represented by the following formula:

![f3](https://c.mql5.com/2/61/f3.png)

**3.Interpretability**

Specifically, the decomposition of the model is interpretable. The NBEATS model introduces some prior knowledge in each layer, forcing some layers to learn certain types of time series characteristics, and achieving interpretable time series decomposition. The implementation method is to constrain the expansion coefficients to the function form of the output sequence. For example, if you want a certain layer block to mainly predict the seasonality of the time series, you can use the following formula to force the output to be seasonal:

![f4](https://c.mql5.com/2/61/f4.png)

![f5](https://c.mql5.com/2/61/f5.png)

**4.Covariates**

In this article, we will introduce covariates to help us predict the target value. The following is the definition of covariates:

- **static\_categoricals:** A list of categorical variables that do not change with time.
- **static\_reals:** A list of continuous variables that do not change with time.
- **time\_varying\_known\_categoricals:** A list of categorical variables that change with time and are known in the future, such as holiday information.
- **time\_varying\_known\_reals:** A list of continuous variables that change with time and are known in the future, such as dates.
- **time\_varying\_unknown\_categoricals:** A list of categorical variables that change with time and are unknown in the future, such as trends.
- **time\_varying\_unknown\_reals:** A list of continuous variables that change with time and are unknown in the future, such as rises or falls.

**5.External Variables**

The NBEATS model can also introduce external variables, which in layman's terms are unrelated to our dataset, but the model changes accordingly. The paper team named it NBEATSx, and we will not discuss it in this article.

### Import libraries

There's nothing to say. Just do it!

```
import lightning.pytorch as pl
import os
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet,NBeats
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import MQF2DistributionLoss
from pytorch_forecasting.data.samplers import TimeSynchronizedBatchSampler
from lightning.pytorch.tuner import Tuner
import MetaTrader5 as mt
import warnings
import json
```

### Rewrite TimeSeriesDataSet Class

There's nothing to say. Just do it! As for why you do this, see the articles earlier in this series.

```
class New_TmSrDt(TimeSeriesDataSet):
    '''
    rewrite dataset class
    '''
    def to_dataloader(self, train: bool = True,
                      batch_size: int = 64,
                      batch_sampler: Sampler | str = None,
                      shuffle:bool=False,
                      drop_last:bool=False,
                      **kwargs) -> DataLoader:

        default_kwargs = dict(
            shuffle=shuffle,
            # drop_last=train and len(self) > batch_size,
            drop_last=drop_last, #
            collate_fn=self._collate_fn,
            batch_size=batch_size,
            batch_sampler=batch_sampler,
        )
        default_kwargs.update(kwargs)
        kwargs = default_kwargs
        # print(kwargs['drop_last'])
        if kwargs["batch_sampler"] is not None:
            sampler = kwargs["batch_sampler"]
            if isinstance(sampler, str):
                if sampler == "synchronized":
                    kwargs["batch_sampler"] = TimeSynchronizedBatchSampler(
                        SequentialSampler(self),
                        batch_size=kwargs["batch_size"],
                        shuffle=kwargs["shuffle"],
                        drop_last=kwargs["drop_last"],
                    )
                else:
                    raise ValueError(f"batch_sampler {sampler} unknown - see docstring for valid batch_sampler")
            del kwargs["batch_size"]
            del kwargs["shuffle"]
            del kwargs["drop_last"]

        return DataLoader(self,**kwargs)
```

### Data Processing

We will not repeat the loading data and data preprocessing here, please refer to the relevant content of my previous three articles for specific explanations, this article only explains the corresponding changes in the places.

**1.Data Acquisition**

```
def get_data(mt_data_len:int):
    if not mt.initialize():
        print('initialize() failed!')
    else:
        print(mt.version())
        sb=mt.symbols_total()
        rts=None
        if sb > 0:
            rts=mt.copy_rates_from_pos("GOLD_micro",mt.TIMEFRAME_M15,0,mt_data_len)
        mt.shutdown()
        # print(len(rts))
    rts_fm=pd.DataFrame(rts)
    rts_fm['time']=pd.to_datetime(rts_fm['time'], unit='s')

    rts_fm['time_idx']= rts_fm.index%(max_encoder_length+2*max_prediction_length)
    rts_fm['series']=rts_fm.index//(max_encoder_length+2*max_prediction_length)
    return rts_fm
```

**2.Pretreatment**

Unlike before, here we are going to talk about covariates, why do we do this? In fact, there are other variants of this model, NBEATSx and GAGA. If you're interested in them, or other models included in the pytorch-forecasting library that we're using, it's important to understand the covariates. We're here to talk about it briefly.

Features against our forex data bar, use the "open", "high", and "low" data columns as covariates. Of course, you can also freely expand other data as covariates, such as MACD, ADX, RSI, and other related indicators, but please remember that it must be related to our data. You cannot add external variables such as Federal Reserve meeting minutes, interest rate decisions, non-farm data, etc. as covariate inputs, as this model does not have the function to parse these data. Perhaps one day I will write an article focusing on discussing how to add external variables to our model.

Now let’s discuss how to add covariates in the New\_TmSrDt() class. The following variable definitions are provided in this class:

- static\_categoricals (List\[str\])
- static\_reals (List\[str\])
- timevaryingknown\_categoricals (List\[str\])
- timevaryingknown\_reals (List\[str\])
- timevaryingunknown\_categoricals (List\[str\])
- timevaryingunknown\_reals (List\[str\])

I have already explained the specific meanings of these variables before. Now we are thinking about which category the variables “open”, “high”, “low” belong to? The others are easy to distinguish, the confusing ones are:

- timevaryingknown\_categoricals
- timevaryingknown\_reals
- timevaryingunknown\_categoricals
- timevaryingunknown\_reals

Because the variables “open”, “high”, “low” are not categories at all, only time\_varying\_known\_reals and time\_varying\_unknown\_reals are left to choose from. You might say that we want to predict “close”, then every bar’s quote “open”, “high”, “low” can be obtained in real time, why can’t it be added to time\_varying\_known\_reals? Please think carefully, if you only predict the value of one bar, then this idea is established, you can completely classify them as time\_varying\_known\_reals, but what if we want to predict the values of multiple bars? You may only know the data of the current bar, and the values behind are completely unknown, so it is not suitable for the environment discussed in our article, so we should add it to time\_varying\_unknown\_reals. But if you only predict the “close” value of one bar, you can definitely add it to time\_varying\_known\_reals, so it is important to carefully consider our use case. There is also a special case about time\_varying\_known\_reals. In fact, each of our bars has a fixed cycle, such as M15, H1, H4, D1, etc., so we can completely calculate the time to which the bars to be predicted belong. So you can completely add time as time\_varying\_known\_reals, we will not discuss this article, interested readers can add it themselves. If you want to use covariates, you can change the "time\_varying\_unknown\_reals=\["close"\]" to "time\_varying\_unknown\_reals=\["close","high","open","low"\]". Of course, our version of the NBEATS doesn't support this feature!

So, the code is:

```
def spilt_data(data:pd.DataFrame,
               t_drop_last:bool,
               t_shuffle:bool,
               v_drop_last:bool,
               v_shuffle:bool):
    training_cutoff = data["time_idx"].max() - max_prediction_length #max:95
    context_length = max_encoder_length
    prediction_length = max_prediction_length
    training = New_TmSrDt(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="close",
        categorical_encoders={"series":NaNLabelEncoder().fit(data.series)},
        group_ids=["series"],
        time_varying_unknown_reals=["close"],
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
    )

    validation = New_TmSrDt.from_dataset(training,
                                         data,
                                         min_prediction_idx=training_cutoff + 1)

    train_dataloader = training.to_dataloader(train=True,
                                              shuffle=t_shuffle,
                                              drop_last=t_drop_last,
                                              batch_size=batch_size,
                                              num_workers=0,)
    val_dataloader = validation.to_dataloader(train=False,
                                              shuffle=v_shuffle,
                                              drop_last=v_drop_last,
                                              batch_size=batch_size,
                                              num_workers=0)
    return train_dataloader,val_dataloader,training
```

### Get Learning Rate

There's nothing to say. Just do it! As for why you do this, see the articles earlier in this series.

```
def get_learning_rate():

    pl.seed_everything(42)
    trainer = pl.Trainer(accelerator="cpu", gradient_clip_val=0.1,logger=False)
    net = NBeats.from_dataset(
        training,
        learning_rate=3e-2,
        weight_decay=1e-2,
        backcast_loss_ratio=0.0,
        optimizer="AdamW",
    )
    res = Tuner(trainer).lr_find(
        net, train_dataloaders=t_loader, val_dataloaders=v_loader, min_lr=1e-5, max_lr=1e-1
    )
    # print(f"suggested learning rate: {res.suggestion()}")
    lr_=res.suggestion()
    return lr_
```

**Note:** There are a few differences between this function and Nbits in that the NBeats.from\_dataset() function does not have hidden\_size parameters. And the loss parameter can't use the MQF2DistributionLoss() method.

### Define The Training Function

There's nothing to say. Just do it! As for why you do this, see the articles earlier in this series.

```
def train():
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        min_delta=1e-4,
                                        patience=10,
                                        verbose=True,
                                        mode="min")
    ck_callback=ModelCheckpoint(monitor='val_loss',
                                mode="min",
                                save_top_k=1,
                                filename='{epoch}-{val_loss:.2f}')
    trainer = pl.Trainer(
        max_epochs=ep,
        accelerator="cpu",
        enable_model_summary=True,
        gradient_clip_val=1.0,
        callbacks=[early_stop_callback,ck_callback],
        limit_train_batches=30,
        enable_checkpointing=True,
    )
    net = NBeats.from_dataset(
        training,
        learning_rate=lr,
        log_interval=10,
        log_val_interval=1,
        weight_decay=1e-2,
        backcast_loss_ratio=0.0,
        optimizer="AdamW",
        stack_types = ["trend", "seasonality"],
    )
    trainer.fit(
        net,
        train_dataloaders=t_loader,
        val_dataloaders=v_loader,
        # ckpt_path='best'
    )
    return trainer
```

**Note:** The NBeats.from\_dataset() in this function requires us to add an interpretable decomposition type variable stack\_types. So, we use the default value. In addition to these two defaults, there is also a "generic" option.

### Model Training and Testing

Now we are implementing the training and prediction logic of the model, which has been explained in the previous article, and there is no change, so I will not discuss it too much.

```
if __name__=='__main__':
    ep=200
    __train=False
    mt_data_len=200000
    max_encoder_length = 2*96
    max_prediction_length = 30
    batch_size = 128
    info_file='results.json'
    warnings.filterwarnings("ignore")
    dt=get_data(mt_data_len=mt_data_len)
    if __train:
        # print(dt)
        # dt=get_data(mt_data_len=mt_data_len)
        t_loader,v_loader,training=spilt_data(dt,
                                              t_shuffle=False,t_drop_last=True,
                                              v_shuffle=False,v_drop_last=True)
        lr=get_learning_rate()
        trainer__=train()
        m_c_back=trainer__.checkpoint_callback
        m_l_back=trainer__.early_stopping_callback
        best_m_p=m_c_back.best_model_path
        best_m_l=m_l_back.best_score.item()

        # print(best_m_p)

        if os.path.exists(info_file):
            with open(info_file,'r+') as f1:
                last=json.load(fp=f1)
                last_best_model=last['last_best_model']
                last_best_score=last['last_best_score']
                if last_best_score > best_m_l:
                    last['last_best_model']=best_m_p
                    last['last_best_score']=best_m_l
                    json.dump(last,fp=f1)
        else:
            with open(info_file,'w') as f2:
                json.dump(dict(last_best_model=best_m_p,last_best_score=best_m_l),fp=f2)

        best_model = NHiTS.load_from_checkpoint(best_m_p)
        predictions = best_model.predict(v_loader, trainer_kwargs=dict(accelerator="cpu",logger=False), return_y=True)
        raw_predictions = best_model.predict(v_loader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu",logger=False))

        for idx in range(10):  # plot 10 examples
            best_model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)
        plt.show()
    else:
        with open(info_file) as f:
            best_m_p=json.load(fp=f)['last_best_model']
        print('model path is:',best_m_p)

        best_model = NHiTS.load_from_checkpoint(best_m_p)

        offset=1
        dt=dt.iloc[-max_encoder_length-offset:-offset,:]
        last_=dt.iloc[-1]
        # print(len(dt))
        for i in range(1,max_prediction_length+1):
            dt.loc[dt.index[-1]+1]=last_
        dt['series']=0
        # dt['time_idx']=dt.apply(lambda x:x.index,args=1)
        dt['time_idx']=dt.index-dt.index[0]
        # dt=get_data(mt_data_len=max_encoder_length)
        predictions = best_model.predict(dt, mode='raw',trainer_kwargs=dict(accelerator="cpu",logger=False),return_x=True)
        best_model.plot_prediction(predictions.x,predictions.output,show_future_observed=False)
        plt.show()
```

**Note:** Make sure you have TensorBoard installed before you run it! This is important, otherwise some inexplicable mistakes will happen.

Training result (There are 10 images that will appear when you run the code, and here is a random one as an example) ：

![f7](https://c.mql5.com/2/61/train.png)

Test result：

![f8](https://c.mql5.com/2/61/f8.png)

### Interpret Model

There are many ways to interpret the data, but the NBEATS model is unique in breaking down the forecasts into seasonality and trends (Of course, because these two are chosen in this article, the results can only be broken down into these two, but there can be many other combinations).

If you're finish the training and want to decompose the prediction, you can add the code:

```
for idx in range(10):  # plot 10 examples
    best_model.plot_interpretation(x, raw_predictions, idx=idx)
```

If you want to decompose the prediction when you run a forecast, you can add the following code：

```
best_model.plot_interpretation(predictions.x,predictions.output,idx=0)
```

The result like this:

![](https://c.mql5.com/2/61/f9.png)

From the fig, our results look like not good enough, because we only show a rough example, and we have not carefully optimized our model, and our data key indicators are still not scientifically specified. Moreover, most of the model parameters are only used by defaults and not tuned, so we have a lot of room for optimization.

### Conclusion

In this article, we discuss how to use our labeled data to predict future prices using the NBEATS model. At the same time, we also demonstrate the special interpretability decomposition function of the NBEATS model. Although our code changes are not significant, please pay attention to our discussion about covariates in the text. If you really understand the usage of different covariates, you can extend this model to other different application scenarios! I believe this will greatly help you improve the accuracy of EA and accurately complete the tasks you need to complete. Of course, this article is just an example. If you want to apply it to actual trading, it is still a bit rough. There are many places that need further optimization, so don’t use it directly in trading! At the same time, we also mentioned some information about external variables. I don’t know if anyone is interested in this direction. If I can get enough information, I may discuss how to implement it in this series of articles in the future.

So, this article ends here, I hope you can gain something!

**All the code:**

```
# Copyright 2021, MetaQuotes Ltd.
# https://www.mql5.com

import lightning.pytorch as pl
import os
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet,NBeats
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.samplers import TimeSynchronizedBatchSampler
from lightning.pytorch.tuner import Tuner
import MetaTrader5 as mt
import warnings
import json

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler,SequentialSampler

class New_TmSrDt(TimeSeriesDataSet):
    '''
    rewrite dataset class
    '''
    def to_dataloader(self, train: bool = True,
                      batch_size: int = 64,
                      batch_sampler: Sampler | str = None,
                      shuffle:bool=False,
                      drop_last:bool=False,
                      **kwargs) -> DataLoader:

        default_kwargs = dict(
            shuffle=shuffle,
            # drop_last=train and len(self) > batch_size,
            drop_last=drop_last, #
            collate_fn=self._collate_fn,
            batch_size=batch_size,
            batch_sampler=batch_sampler,
        )
        default_kwargs.update(kwargs)
        kwargs = default_kwargs
        # print(kwargs['drop_last'])
        if kwargs["batch_sampler"] is not None:
            sampler = kwargs["batch_sampler"]
            if isinstance(sampler, str):
                if sampler == "synchronized":
                    kwargs["batch_sampler"] = TimeSynchronizedBatchSampler(
                        SequentialSampler(self),
                        batch_size=kwargs["batch_size"],
                        shuffle=kwargs["shuffle"],
                        drop_last=kwargs["drop_last"],
                    )
                else:
                    raise ValueError(f"batch_sampler {sampler} unknown - see docstring for valid batch_sampler")
            del kwargs["batch_size"]
            del kwargs["shuffle"]
            del kwargs["drop_last"]

        return DataLoader(self,**kwargs)

def get_data(mt_data_len:int):
    if not mt.initialize():
        print('initialize() failed!')
    else:
        print(mt.version())
        sb=mt.symbols_total()
        rts=None
        if sb > 0:
            rts=mt.copy_rates_from_pos("GOLD_micro",mt.TIMEFRAME_M15,0,mt_data_len)
        mt.shutdown()
        # print(len(rts))
    rts_fm=pd.DataFrame(rts)
    rts_fm['time']=pd.to_datetime(rts_fm['time'], unit='s')

    rts_fm['time_idx']= rts_fm.index%(max_encoder_length+2*max_prediction_length)
    rts_fm['series']=rts_fm.index//(max_encoder_length+2*max_prediction_length)
    return rts_fm

def spilt_data(data:pd.DataFrame,
               t_drop_last:bool,
               t_shuffle:bool,
               v_drop_last:bool,
               v_shuffle:bool):
    training_cutoff = data["time_idx"].max() - max_prediction_length #max:95
    context_length = max_encoder_length
    prediction_length = max_prediction_length
    training = New_TmSrDt(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="close",
        categorical_encoders={"series":NaNLabelEncoder().fit(data.series)},
        group_ids=["series"],
        time_varying_unknown_reals=["close"],
        max_encoder_length=context_length,
        # min_encoder_length=max_encoder_length//2,
        max_prediction_length=prediction_length,
        # min_prediction_length=1,

    )

    validation = New_TmSrDt.from_dataset(training,
                                         data,
                                         min_prediction_idx=training_cutoff + 1)

    train_dataloader = training.to_dataloader(train=True,
                                              shuffle=t_shuffle,
                                              drop_last=t_drop_last,
                                              batch_size=batch_size,
                                              num_workers=0,)
    val_dataloader = validation.to_dataloader(train=False,
                                              shuffle=v_shuffle,
                                              drop_last=v_drop_last,
                                              batch_size=batch_size,
                                              num_workers=0)
    return train_dataloader,val_dataloader,training

def get_learning_rate():

    pl.seed_everything(42)
    trainer = pl.Trainer(accelerator="cpu", gradient_clip_val=0.1,logger=False)
    net = NBeats.from_dataset(
        training,
        learning_rate=3e-2,
        weight_decay=1e-2,
        backcast_loss_ratio=0.1,
        optimizer="AdamW",
    )
    res = Tuner(trainer).lr_find(
        net, train_dataloaders=t_loader, val_dataloaders=v_loader, min_lr=1e-5, max_lr=1e-1
    )
    # print(f"suggested learning rate: {res.suggestion()}")
    lr_=res.suggestion()
    return lr_
def train():
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        min_delta=1e-4,
                                        patience=10,
                                        verbose=True,
                                        mode="min")
    ck_callback=ModelCheckpoint(monitor='val_loss',
                                mode="min",
                                save_top_k=1,
                                filename='{epoch}-{val_loss:.2f}')
    trainer = pl.Trainer(
        max_epochs=ep,
        accelerator="cpu",
        enable_model_summary=True,
        gradient_clip_val=1.0,
        callbacks=[early_stop_callback,ck_callback],
        limit_train_batches=30,
        enable_checkpointing=True,
    )
    net = NBeats.from_dataset(
        training,
        learning_rate=lr,
        log_interval=10,
        log_val_interval=1,
        weight_decay=1e-2,
        backcast_loss_ratio=0.0,
        optimizer="AdamW",
        stack_types=["trend", "seasonality"],
    )
    trainer.fit(
        net,
        train_dataloaders=t_loader,
        val_dataloaders=v_loader,
        # ckpt_path='best'
    )
    return trainer

if __name__=='__main__':
    ep=200
    __train=False
    mt_data_len=80000
    max_encoder_length = 96
    max_prediction_length = 20
    # context_length = max_encoder_length
    # prediction_length = max_prediction_length
    batch_size = 128
    info_file='results.json'
    warnings.filterwarnings("ignore")
    dt=get_data(mt_data_len=mt_data_len)
    if __train:
        # print(dt)
        # dt=get_data(mt_data_len=mt_data_len)
        t_loader,v_loader,training=spilt_data(dt,
                                              t_shuffle=False,t_drop_last=True,
                                              v_shuffle=False,v_drop_last=True)
        lr=get_learning_rate()
        # lr=3e-3
        trainer__=train()
        m_c_back=trainer__.checkpoint_callback
        m_l_back=trainer__.early_stopping_callback
        best_m_p=m_c_back.best_model_path
        best_m_l=m_l_back.best_score.item()

        # print(best_m_p)

        if os.path.exists(info_file):
            with open(info_file,'r+') as f1:
                last=json.load(fp=f1)
                last_best_model=last['last_best_model']
                last_best_score=last['last_best_score']
                if last_best_score > best_m_l:
                    last['last_best_model']=best_m_p
                    last['last_best_score']=best_m_l
                    json.dump(last,fp=f1)
        else:
            with open(info_file,'w') as f2:
                json.dump(dict(last_best_model=best_m_p,last_best_score=best_m_l),fp=f2)

        best_model = NBeats.load_from_checkpoint(best_m_p)
        predictions = best_model.predict(v_loader, trainer_kwargs=dict(accelerator="cpu",logger=False), return_y=True)
        raw_predictions = best_model.predict(v_loader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu",logger=False))

        for idx in range(10):  # plot 10 examples
            best_model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)
        plt.show()
    else:
        with open(info_file) as f:
            best_m_p=json.load(fp=f)['last_best_model']
        print('model path is:',best_m_p)

        best_model = NBeats.load_from_checkpoint(best_m_p)

        offset=1
        dt=dt.iloc[-max_encoder_length-offset:-offset,:]
        last_=dt.iloc[-1]
        # print(len(dt))
        for i in range(1,max_prediction_length+1):
            dt.loc[dt.index[-1]+1]=last_
        dt['series']=0
        # dt['time_idx']=dt.apply(lambda x:x.index,args=1)
        dt['time_idx']=dt.index-dt.index[0]
        # dt=get_data(mt_data_len=max_encoder_length)
        predictions = best_model.predict(dt, mode='raw',trainer_kwargs=dict(accelerator="cpu",logger=False),return_x=True)
        # best_model.plot_prediction(predictions.x,predictions.output,show_future_observed=False)
        best_model.plot_interpretation(predictions.x,predictions.output,idx=0)
        plt.show()
```

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13218.zip "Download all attachments in the single ZIP archive")

[n\_beats.py](https://www.mql5.com/en/articles/download/13218/n_beats.py "Download n_beats.py")(9.16 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Integrate Your Own LLM into EA (Part 5): Develop and Test Trading Strategy with LLMs(IV) — Test Trading Strategy](https://www.mql5.com/en/articles/13506)
- [Integrate Your Own LLM into EA (Part 5): Develop and Test Trading Strategy with LLMs (III) – Adapter-Tuning](https://www.mql5.com/en/articles/13500)
- [Integrate Your Own LLM into EA (Part 5): Develop and Test Trading Strategy with LLMs (II)-LoRA-Tuning](https://www.mql5.com/en/articles/13499)
- [Integrate Your Own LLM into EA (Part 5): Develop and Test Trading Strategy with LLMs(I)-Fine-tuning](https://www.mql5.com/en/articles/13497)
- [Integrate Your Own LLM into EA (Part 4): Training Your Own LLM with GPU](https://www.mql5.com/en/articles/13498)
- [Integrate Your Own LLM into EA (Part 3): Training Your Own LLM with CPU](https://www.mql5.com/en/articles/13920)
- [Data label for time series mining (Part 6)：Apply and Test in EA Using ONNX](https://www.mql5.com/en/articles/13919)

**[Go to discussion](https://www.mql5.com/en/forum/458510)**

![Design Patterns in software development and MQL5 (Part 3): Behavioral Patterns 1](https://c.mql5.com/2/61/Design_Patterns_yPart_39_Behavioral_Patterns_1__LOGO.png)[Design Patterns in software development and MQL5 (Part 3): Behavioral Patterns 1](https://www.mql5.com/en/articles/13796)

A new article from Design Patterns articles and we will take a look at one of its types which is behavioral patterns to understand how we can build communication methods between created objects effectively. By completing these Behavior patterns we will be able to understand how we can create and build a reusable, extendable, tested software.

![Market Reactions and Trading Strategies in Response to Dividend Announcements: Evaluating the Efficient Market Hypothesis in Stock Trading](https://c.mql5.com/2/61/Evaluating_the_Efficient_Market_Hypothesis_in_Stock_Trading_LOGO.png)[Market Reactions and Trading Strategies in Response to Dividend Announcements: Evaluating the Efficient Market Hypothesis in Stock Trading](https://www.mql5.com/en/articles/13850)

In this article, we will analyse the impact of dividend announcements on stock market returns and see how investors can earn more returns than those offered by the market when they expect a company to announce dividends. In doing so, we will also check the validity of the Efficient Market Hypothesis in the context of the Indian Stock Market.

![Developing a Replay System — Market simulation (Part 18): Ticks and more ticks (II)](https://c.mql5.com/2/56/replay-p18-avatar.png)[Developing a Replay System — Market simulation (Part 18): Ticks and more ticks (II)](https://www.mql5.com/en/articles/11113)

Obviously the current metrics are very far from the ideal time for creating a 1-minute bar. That's the first thing we are going to fix. Fixing the synchronization problem is not difficult. This may seem hard, but it's actually quite simple. We did not make the required correction in the previous article since its purpose was to explain how to transfer the tick data that was used to create the 1-minute bars on the chart into the Market Watch window.

![MQL5 Wizard Techniques you should know (Part 08): Perceptrons](https://c.mql5.com/2/61/MQL5_Wizard_Techniques_you_should_know_xPart_08c_Perceptrons_LOGO.png)[MQL5 Wizard Techniques you should know (Part 08): Perceptrons](https://www.mql5.com/en/articles/13832)

Perceptrons, single hidden layer networks, can be a good segue for anyone familiar with basic automated trading and is looking to dip into neural networks. We take a step by step look at how this could be realized in a signal class assembly that is part of the MQL5 Wizard classes for expert advisors.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/13218&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082904442942001438)

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