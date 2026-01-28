---
title: MQL5 Wizard Techniques you should know (Part 59): Reinforcement Learning (DDPG) with Moving Average and Stochastic Oscillator Patterns
url: https://www.mql5.com/en/articles/17684
categories: Trading Systems, Integration, Expert Advisors, Machine Learning
relevance_score: 13
scraped_at: 2026-01-22T17:11:11.114712
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=kkpyjmsgxrvgwzbqscdebzkoucbeutze&ssn=1769090453765128249&ssn_dr=0&ssn_sr=0&fv_date=1769090453&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F17684&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Wizard%20Techniques%20you%20should%20know%20(Part%2059)%3A%20Reinforcement%20Learning%20(DDPG)%20with%20Moving%20Average%20and%20Stochastic%20Oscillator%20Patterns%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909045375529693&fz_uniq=5048834322808545173&sv=2552)

MetaTrader 5 / Trading systems


### Introduction

In the [last article](https://www.mql5.com/en/articles/17668) we had introduced DDPG, a reinforcement-learning algorithm and had looked at 3 of its crucial classes as implemented in Python. The replay buffer class, the actor-network class, and the critic-network class. What was not covered was the DDPG-agent class; importing MetaTrader 5 price data to Python; functions for the MA, & Stochastic-Oscillator;  a get-pattern function for bringing together data from the two indicators into a binary input vector for the supervised learning network (implemented in the earlier supervised-learning article via MQL5); and finally an environment simulation loop for training the actor & critic networks.

All these are part of Reinforcement-Learning (RL) which we are looking at as a segue from Supervised-Learning (SL) to Inference-Learning (IL) (or unsupervised learning). Any one of these modes can be used unilaterally to train and use a model, however, these articles are trying to make the case they could be used together to build something more interesting. So we continue our look at RL and by dealing with the very important DDPG-Agent class.

### DDPG-Agent

The core architecture and initialization of this class can be defined as follows:

```
def __init__(self, state_dim, action_dim):

    # Actor networks
    self.actor = Actor(state_dim, action_dim, HIDDEN_DIM).to(device)
    self.actor_target = Actor(state_dim, action_dim, HIDDEN_DIM).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

    # Critic networks
    self.critic = Critic(state_dim, action_dim, HIDDEN_DIM).to(device)
    self.critic_target = Critic(state_dim, action_dim, HIDDEN_DIM).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

    self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
```

The critical components here are the dual network architecture, optimizer setup, and experience management. The dual architecture maintains a separate policy (actor network) and a separate value (critic network) from the two main policy and value networks. These implement target networks for both, which is important for stability when training. The initialization of the respective targets is done with same weights as their main networks.

The optimizer setup features separate Adam optimizers for the actor and critic networks. Also, as is typically the case, we use separate learning rates for the policy and value networks. Finally, for experience management, we ensure that the replay buffer stores transitions for off-policy learning and by fixing buffer size we prevent unbounded memory usage. We select actions by incorporating exploration as follows:

```
def select_action(self, state, noise_scale=0.1):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = self.actor(state).cpu().data.numpy().flatten()
    action += noise_scale * np.random.randn(self.action_dim)
    return np.clip(action, -1, 1)
```

The key mechanisms here are state-processing, exploration-strategy, and device-management. State-processing oversees the conversion of NumPy array to proper tensor format, the addition of a batch dimension (by un squeezing), and finally by ensuring computation is on the correct device.

The exploration-strategy adds Gaussian noise to the deterministic policy output. The noise scale controls exploration magnitude and clipping maintains a valid action range. Device-management ensures an efficient movement between GPU/CPU if applicable. Also, a final output as NumPy array is returned by the function for environment compatibility. The learning update mechanism is as follows:

```
def update(self):

    if len(self.replay_buffer) < BATCH_SIZE:

        return
```

This if-clause serves as an update gate where updates are skipped until sufficient experiences, numbering the batch size, are collected. This ensures meaningful batch statistics. Updating of the 2 critic networks is as follows:

```
# Sample batch

states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

# Target Q calculation
next_actions = self.actor_target(next_states)
target_q = self.critic_target(next_states, next_actions)
target_q = rewards + (1 - dones) * GAMMA * target_q

# Current Q estimation
current_q = self.critic(states, actions)

# Loss computation and backpropagation
critic_loss = nn.MSELoss()(current_q, target_q.detach())
self.critic_optimizer.zero_grad()
critic_loss.backward()
self.critic_optimizer.step()
```

The key aspects addressed by this code are target value calculation, loss-computation, and gradient-management. The target-value calculation uses target networks to get stable Q-targets. It implements the Bellman equation with termination handling as set by the experience parameter ‘dones’. A discount factor of GAMMA controls future reward importance.

For the loss-computation, mean squared error between the current and target Q-values are determined. The detach() method prevents target gradients from flowing (or being carried by the tensor for transferability). And, the standard temporal-difference learning is applied. Gradient management simply ensures all gradients are reset to zero, and the optimization of the critic network is a separate step. Actor network updates are also performed as follows:

```
actor_loss = -self.critic(states, self.actor(states)).mean()
self.actor_optimizer.zero_grad()
actor_loss.backward()
self.actor_optimizer.step()
```

The policy-gradient specifics dealt with here are maximizing Q-values through the minimization of negative Q, differentiating through both the actor and critic networks, as well as applying a pure policy gradient approach with no log probabilities (being deterministic). The target-network updates are as follows:

```
for target, param in zip(self.actor_target.parameters(), self.actor.parameters()):

    target.data.copy_(TAU * param.data + (1 - TAU) * target.data)

for target, param in zip(self.critic_target.parameters(), self.critic.parameters()):

    target.data.copy_(TAU * param.data + (1 - TAU) * target.data)
```

This soft update mechanism features polyak-averaging with a TAU value that is typically less than 1. The network weights are tracked slowly, as this provides an alternative to periodic hard updates. This process on the whole maintains stability while allowing learning. Our model needs to be persistent. It should be able to load previously saved network weights and also save them after training. We accomplish this as follows:

```
def save(self, filename):

    torch.save({
        'actor': self.actor.state_dict(),

        'critic': self.critic.state_dict(),

        'actor_target': self.actor_target.state_dict(),

        'critic_target': self.critic_target.state_dict(),
    }, filename)

def load(self, filename):

    checkpoint = torch.load(filename)
    self.actor.load_state_dict(checkpoint['actor'])
    self.critic.load_state_dict(checkpoint['critic'])
    self.actor_target.load_state_dict(checkpoint['actor_target'])
    self.critic_target.load_state_dict(checkpoint['critic_target'])
```

Key features with our listing above are: we save/load all network states; we maintain target network consistency; allow training continuation; and support model evaluation. To sum up the agent-class therefore, when implementing a DDPG-agent, there are a few critical design choices that need to be made. These could broadly fall under three categories, namely: Choosing DDPG-specific components, harnessing strengths of the implementation, and making potential enhancements.

The DDPG-components used are mainly the target networks, the deterministic-policy, and separate learning-rates. The target-networks are very important for stable learning of the rewards from actions taken (Q-Learning) when dealing with continuous action spaces. The use of continuous spaces makes this paramount. This deterministic policy does then necessitate external exploration with noise in order to build robustness. The use of separate learning rates is also a typical application where the policy (actor-network) has a slower learning rate than the value network.

Choices made that make this a relatively strong implementation are clear ‘separation of concerns’ where we have well-defined methods for action selection and also updates. Also, there is a device-awareness that ensures consistent handling of GPU/CPU transitions. Batch-processing is also used in order to make the tensor operations more efficient, and finally shape-safety is checked at multiple points to ensure consistent tensor-dimensionality.

Potential-enhancements, though, could be: gradient clipping in order to avert exploding-gradients; using a learning-rate-schedule in order to refine and better control the learning process; using a priority replay for efficient sampling though this ties into the already mentioned replay-buffer in the last article; and finally parallel exploration where multiple instances of the actor can be used for faster data collection.

There are also a few training-dynamics that are noteworthy and these are to do with the update-sequence, and hyperparameter considerations. The update-sequence features the critic networks being updated first. This is because more accurate Q-values do guide the policy improvement. In order to introduce some extra stability, the delaying of policy updates can also be implemented. Finally, frequent target updates are performed to slow-track the parameters (network-weights) that have been learnt.

Hyperparameter considerations should include a focus on TAU, since it controls the target network speed and therefore is a key driver for the stability of the overall learning process. There should be use of a noise scale that allows decay over time. Buffer-sizing is also critical since it affects the learning efficiency and the batch size has an impact on the variance of updates

### MA and Stochastic Functions

These two functions are implemented in python for reinforcement-learning (RL) unlike in supervised-learning [article](https://www.mql5.com/en/articles/17479) where we did this in MQL5 and simply exported the network input data to Python for training. With our implementation here, we are using MetaTrader’s MetaTrader 5 Python-module to connect to a running terminal instance and then retrieve price data. There are guides in the documentation [here](https://www.mql5.com/en/docs/python_metatrader5) on how to do this. Our indicator functions below transform raw price data into technical indicator data that served as inputs into our supervised learning model after being converted/ normalized to a binary pattern vector.

The outputs of the supervised learning model are what we cast as states, since in essence they are forecasting changes in price action. These states are then used as inputs for the DDPG RL agent. Our MA function takes as input a panda data frame of prices from the MetaTrader 5 Python-module. This data frame needs to be validated and prepared as follows:

```
p = np.asarray(p).flatten()  # Convert to 1D array if not already

if len(p) < window:

    raise ValueError("Window size cannot be larger than the number of prices.")
```

What we are doing here is standardizing the array to ensure consistent 1D input format regardless of input shape. We also have error handling that averts invalid window sizes that would cause computation errors. Data integrity also maintains clean data flow through the processing pipeline. The calculation mechanism is as follows:

```
return np.convolve(p, np.ones(window), 'valid') / window
```

This implementation uses convolution for efficient rolling of the average computation. Use of the ‘valid’ input parameter ensures only fully computed windows are returned. Normalization is also done by window size to produce a true average. The entire operation is vectorized for optimal performance. The financial significance of this is that it smooths price data to help identify trends, and the window size used (aka averaging period) determines the sensitivity to price changes. The stochastic oscillator function validates its inputs as follows:

```
p = np.asarray(p).flatten()

if len(p) < k_window:
    raise ValueError("Window size for %K cannot be larger than the number of prices.")
```

Design considerations here are consistent input formatting with the MA function. There needs to be a separate validation for the %K calculation window, with the raising of an error as an early failure for invalid parameters. The %K calculation is as follows:

```
for i in range(k_window - 1, len(p)):

    current_close = p[i]
    lowest_low = min(p[i - k_window + 1:i + 1])
    highest_high = max(p[i - k_window + 1:i + 1])
    K = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
    K_values.append(K)
```

Important components here rolling-window-analysis, market-context, and overall implementation. The rolling window analysis needs to examine price range over a look-back period. This helps identify the relative position of the current close price, with a standard scaling of 0 to 100 being applied. The market context helps us assess the overbought/oversold conditions. Values near 100 suggest potential reversal down, while values near 0 would suggest an upward turn. The overall implementation uses an explicit loop for clarity, uses proper window indexing to handle edge cases, and preserves temporal order of the results. The %D calculation is as follows:

```
D_values = MA(K_values, d_window)
```

Essentially, this is a signal refinement where a smoothed version of %K moving average is used. Typical assignment for this averaging period is 3 and that is what we are using. This additional buffer provides confirmation for swings in %K and thus helps reduce false signals from a raw %K.

### Get Pattern Function

This function is used to integrate data from our two indicator buffers above into the learning pipeline. It plays a feature engineering role. This is because: it helps in dimensionality-reduction, given that it transforms raw prices into more meaningful signals; it helps with stationary-improvement, since indicators are often more stable than raw prices; and finally, it allows temporal-context-capture given that widowed calculations maintain time dependencies (a generated input vector of say \[1,0,0,1\] can be associated with the time it is produced just like any indicator value or raw prices are also tagged with the time they are generated).

Primarily, though, it is used for supervised-learning preparation. The features it outputs in a binary vector of 0s and 1s train the model to forecast the next price changes. The MA provides trend information, while the STO function gives us momentum and reversal information. We covered the combined complementary patterns from both indicators in article 57. The forecast price change outputs then serve as RL state representation.

This implies our supervised-learning model predictions become state inputs for DDPG. Our used indicators of MA and STO therefore end up helping the DDPG agent by providing market context to help in understanding a given market regime. This reduces the need for raw history prices when defining the state.

Implementation strengths include robustness from validating inputs to prevent silent failures, dimension handling to ensure consistent array shapes, and error messages for clear messaging in the even of improper usage. Also, pro-performance considerations of using vectorized operations where possible, explicit looping and memory efficiency from streaming-friendly design. It does remain relevant to traders, and not lost in technicalities. This is because industry standard indicators are being used to generate states. The indicators are complementary since they bring together trend and momentum metrics, and the state outputs are in a normalized range, which is important for consistency.

Potential enhancements are computation optimization by using vectorized implementation of %K calculation, using numba acceleration (imported from JIT) to speed up loops in STO function, and caching of intermediate calculations. Extended functionality can be added through additional validation for NaN/inf values can also be added. This code implementing reinforcement learning with DDPG is sizable and whereas it would be appropriate to provide comment on its key sections, I will simply attach the uncovered portions of it at the endo of this article. Chief among will be this get pattern function.

### Testing

Of the 10 patterns we tested in the supervised learning article, #57, only 7 were able to walk forward profitably for a year, having been trained on a year prior. Because each pattern amounted to its own network, we have to generate reinforcement-learning networks and environments also for each pattern. We follow a similar methodology in article 57 where we train on the pair EUR USD for the year 2023 on the daily time frame. In this case, we are training our reinforcement-learning networks by simulating the year 2023 as a ‘live-market’ environment. As argued in the last 2 articles, reinforcement-learning is a system in place for supporting and protecting an already established and trained model, which in our case is the network we trained by supervised-learning in article 57.

This it does by back-propagating when in production or live environments and not on historical data. Because back propagating an ONNX network from MQL5 is not feasible, we are ‘simulating’ a live environment, which in our case is still the year 2023.

Rather than ask we posed in supervised learning, what is price going to do next?, we ask the question given this incoming price changes, what actions should the trader take. We thus perform simulation trainings as outlined above for the year 2023 and then do a forward walk for the year 2024 where our entry conditions get modified slightly.

Rather than solely basing our long or short positions on what price is going to do next, we also consider what actions we really need to take in light of what price will do next. We also factor in whether the rewards will be profitable. Of the 7 patterns that walked forward in article 57, only 3 walk forward meaningfully when reinforcement-learning is used. Using our indexing of the 10 that runs from 0 to 9, these patterns are 1, 2, and 5. Their reports are presented below:

For pattern 1:

![r1](https://c.mql5.com/2/129/r1_r.jpg)

![c1](https://c.mql5.com/2/129/c1_r.jpg)

For pattern 2:

![r2](https://c.mql5.com/2/129/r2_r.jpg)

![c2](https://c.mql5.com/2/129/c2_r.jpg)

For pattern 5:

![r5](https://c.mql5.com/2/129/r5_r.jpg)

![c5](https://c.mql5.com/2/129/c5_r.jpg)

The tested Expert Advisor as always is built with a custom signal class whose code is attached below. We make changes to the signal class file we had in article 57, by renaming the function 'IsPattern' to 'Supervise'. Also, we introduce a new function 'Reinforce'. The code of both of these is shared below:

```
//+------------------------------------------------------------------+
//| Supervised Learning Model Forward Pass.                          |
//+------------------------------------------------------------------+
double CSignal_DDPG::Supervise(int Index, ENUM_POSITION_TYPE T)
{  vectorf _x = Get(Index, m_time.GetData(X()), m_close, m_ma, m_ma_lag, m_sto);
   vectorf _y(1);
   _y.Fill(0.0);
   int _i=Index;
   if(_i==8)
   {  _i -= 2;
   }
   ResetLastError();
   if(!OnnxRun(m_handles[_i], ONNX_NO_CONVERSION, _x, _y))
   {  printf(__FUNCSIG__ + " failed to get y forecast, err: %i", GetLastError());
      return(double(_y[0]));
   }
   if(T == POSITION_TYPE_BUY && _y[0] > 0.5f)
   {  _y[0] = 2.0f * (_y[0] - 0.5f);
   }
   else if(T == POSITION_TYPE_SELL && _y[0] < 0.5f)
   {  _y[0] = 2.0f * (0.5f - _y[0]);
   }
   return(double(_y[0]));
}
//+------------------------------------------------------------------+
//| Reinforcement Learning Model Forward Pass.                       |
//+------------------------------------------------------------------+
double CSignal_DDPG::Reinforce(int Index, ENUM_POSITION_TYPE T, double State)
{  vectorf _x(1);
   _x.Fill(float(State));
   vectorf _y(1);
   _y.Fill(0.0);
   vectorf _y_state(1);
   _y_state.Fill(float(State));
   vectorf _y_action(1);
   _y_action.Fill(0.0);
   vectorf _z(1);
   _z.Fill(0.0);
   int _i=Index;
   if(_i==8)
   {  _i -= 2;
   }
   ResetLastError();
   if(!OnnxRun(m_handles_a[_i], ONNX_NO_CONVERSION, _x, _y))
   {  printf(__FUNCSIG__ + " failed to get y action forecast, err: %i", GetLastError());
   }
   _y_action[0] = _y[0];
   ResetLastError();
   if(!OnnxRun(m_handles_c[_i], ONNX_NO_CONVERSION, _y_state, _y_action, _z))
   {  printf(__FUNCSIG__ + " failed to get z reward forecast, err: %i", GetLastError());
   }
   //normalize action output & check for state-action alignment
   if(T == POSITION_TYPE_BUY && _y[0] > 0.5f)
   {  _y[0] = 2.0f * (_y[0] - 0.5f);
   }
   else if(T == POSITION_TYPE_SELL && _y[0] < 0.5f)
   {  _y[0] = 2.0f * (0.5f - _y[0]);
   }
   else
   {  _y[0] = 0.0f;
   }
   return(double(_y[0]*_z[0]));
}
```

This custom signal class file is meant to be assembled into an Expert Advisor via the MQL5 wizard and for readers who are new they can find guidance [here](https://www.mql5.com/en/articles/171) and [here](https://www.mql5.com/en/articles/275) on how to do this.

### Conclusion

We have looked at the case for applying reinforcement-learning when models are in deployment/ production. Our reinforcement-learning was using the deep deterministic policy gradient algorithm and this implementation had the classes: replay buffer, actor, critic and agent as covered in this and the last article.  Reinforcement-learning in deployment/production serves as a means of keeping the model focused on what was learnt in the supervised-learning phase (exploitation) while also looking out for new unknown changes in the environment/market-conditions that should be considered when making decisions going forward (exploitation). In doing this correctly, we inherently have to back propagate and train a model while it is being used.

However, since training an ONNX model in MQL5 is unsupported, we opted for a simulation of live trading conditions on historical data. Post simulation, we tested the trained reinforcement-learning models on the subsequent year from the training year and only 3 of the 7 were able to forward walk, albeit with skewed trade results since positions were mostly held in long only or short only. This as we argued in article 57 is most likely down to a small test-window, meaning extensive training and testing over more data should rectify this. We now look at inference next.

| Type | Description |
| --- | --- |
| \*.\*onnx files | ONNX model files in the Python Subfolder within location of custom signal class file |
| \*.\*mqh files | Custom signal class file in and file with function for processing input network data (57\_X) |
| \*.\*mq5 files | Wizard Assembled Expert Advisor whose header shows files used. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17684.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/17684/mql5.zip "Download MQL5.zip")(1106.07 KB)

[Experts.zip](https://www.mql5.com/en/articles/download/17684/experts.zip "Download Experts.zip")(1.67 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/484373)**

![Neural Networks in Trading: Hierarchical Vector Transformer (Final Part)](https://c.mql5.com/2/91/Hierarchical_Vector_Transformer__LOGO.png)[Neural Networks in Trading: Hierarchical Vector Transformer (Final Part)](https://www.mql5.com/en/articles/15713)

We continue studying the Hierarchical Vector Transformer method. In this article, we will complete the construction of the model. We will also train and test it on real historical data.

![From Basic to Intermediate: BREAK and CONTINUE Statements](https://c.mql5.com/2/91/Comandos_BREAK_e_CONTINUE___LOGO_2.png)[From Basic to Intermediate: BREAK and CONTINUE Statements](https://www.mql5.com/en/articles/15376)

In this article, we will look at how to use the RETURN, BREAK, and CONTINUE statements in a loop. Understanding what each of these statements does in the loop execution flow is very important for working with more complex applications. The content presented here is intended solely for educational purposes. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

![Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (IV): Trade Management Panel class](https://c.mql5.com/2/131/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_X_CODEIV___LOGO.png)[Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (IV): Trade Management Panel class](https://www.mql5.com/en/articles/17396)

This discussion covers the updated TradeManagementPanel in our New\_Admin\_Panel EA. The update enhances the panel by using built-in classes to offer a user-friendly trade management interface. It includes trading buttons for opening positions and controls for managing existing trades and pending orders. A key feature is the integrated risk management that allows setting stop loss and take profit values directly in the interface. This update improves code organization for large programs and simplifies access to order management tools, which are often complex in the terminal.

![Advanced Memory Management and Optimization Techniques in MQL5](https://c.mql5.com/2/130/Advanced_Memory_Management_and_Optimization_Techniques_in_MQL5____LOGO.png)[Advanced Memory Management and Optimization Techniques in MQL5](https://www.mql5.com/en/articles/17693)

Discover practical techniques to optimize memory usage in MQL5 trading systems. Learn to build efficient, stable, and fast-performing Expert Advisors and indicators. We’ll explore how memory really works in MQL5, the common traps that slow your systems down or cause them to fail, and — most importantly — how to fix them.

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/17684&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5048834322808545173)

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