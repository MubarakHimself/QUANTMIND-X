---
title: MQL5 Wizard Techniques you should know (Part 47): Reinforcement Learning with Temporal Difference
url: https://www.mql5.com/en/articles/16303
categories: Trading Systems, Expert Advisors, Machine Learning
relevance_score: 6
scraped_at: 2026-01-23T11:38:00.270635
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=czgobtrxequxykzeegbgdsthdqcsbeqv&ssn=1769157478096720301&ssn_dr=0&ssn_sr=0&fv_date=1769157478&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16303&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Wizard%20Techniques%20you%20should%20know%20(Part%2047)%3A%20Reinforcement%20Learning%20with%20Temporal%20Difference%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915747882879266&fz_uniq=5062605851975394662&sv=2552)

MetaTrader 5 / Trading systems


### Introduction

The introduction to temporal difference (TD) learning in reinforcement learning serves as a gateway to understand how TD distinguishes itself from other algorithms, such as Monte Carlo, Q-Learning, and SARSA. This article aims to unravel the complexities surrounding TD learning by highlighting its unique ability to update value estimates incrementally based on partial information from episodes, rather than waiting for episodes to complete as seen in Monte Carlo methods. This distinction makes TD learning a powerful tool, especially where environments are dynamic and require prompt updates to the learning policy.

In the [last reinforcement-learning](https://www.mql5.com/en/articles/16254) article, we looked at the Monte Carlo algorithm that gathered reward information over multiple cycles before performing a single update for each episode. [Temporal difference](https://en.wikipedia.org/wiki/Temporal_difference_learning "https://en.wikipedia.org/wiki/Temporal_difference_learning") (TD) though, is all about learning from partial and incomplete episodes much like the algorithms of Q-Learning and SARSA that we tackled earlier on [here](https://www.mql5.com/en/articles/15743) and [here](https://www.mql5.com/en/articles/16143).

Below is a tabulated summary of the primary distinctions between TD, Q-Learning, and SARSA.

And to recap, the formula for state-action pairs with on-policy update, like SARSA for instance, is as follows:

|  | **TD Learning** | **Q-Learning** | **SARSA** |
| --- | --- | --- | --- |
| Types of Values | State values V(s) | Action values Q(s,a) | Action values Q(s,a) |
| Learning Approach | Estimates of future state values | Off-policy | On-policy |
| Policy Type | Not dependent on specific policy | Learns optimal policy | Learns current behaviour policy |
| Update Target | Next state value V(s′) | Max Q(s′,a′) | Actual Q(s′,a′) |
| Exploration | Often requires separate policy | Assumes agent seeks optimal | More conservative |
| Behaviour | Moves towards next state’s value | Greedy; favors optimal path | Follows actual exploration path |

To recap some of the lingo we covered, on policy updates means that the state-action pairs that were getting updated were the current pairs and not necessarily the optimal ones or the ones with the highest Q-Values. If we are to update the state-action pairs with the highest Q-Values, then this would be an off-policy approach. We perform these updates in MQL5 as follows:

```
//+------------------------------------------------------------------+
// Update using On-policy
//+------------------------------------------------------------------+
void Cql::SetOnPolicy(double Reward, vector &E)
{  Action(E);
//where 'act' index 1 represents the current Q_SA-action from Q_SA-Map
   double _sa = Q_SA[transition_act][e_row[1]][e_col[1]];
   double _v = Q_V[e_row[1]][e_col[1]];
   if(THIS.use_markov)
   {  int _old_index = GetMarkov(e_row[1], e_col[1]);
      int _new_index = GetMarkov(e_row[0], e_col[0]);
      _sa *= markov[_old_index][_new_index];
      _v *= markov[_old_index][_new_index];
   }
   for (int i = 0; i < THIS.actions; i++)
   {  Q_SA[i][e_row[1]][e_col[1]] += THIS.alpha * ((Reward + (THIS.gamma * _sa)) - Q_SA[i][e_row[1]][e_col[1]]);
   }
   Q_V[e_row[1]][e_col[1]] += THIS.alpha * ((Reward + (THIS.gamma * _v)) - Q_V[e_row[1]][e_col[1]]);
}
```

```
//+------------------------------------------------------------------+
// Update using Off-policy
//+------------------------------------------------------------------+
void Cql::SetOffPolicy(double Reward, vector &E)
{  Action(E);
//where 'act' index 0 represents highest valued Q_SA-action from Q_SA-Map
//as determined from Action() function above.
   double _sa = Q_SA[transition_act][e_row[0]][e_col[0]];
   double _v = Q_V[e_row[0]][e_col[0]];
   if(THIS.use_markov)
   {  int _old_index = GetMarkov(e_row[1], e_col[1]);
      int _new_index = GetMarkov(e_row[0], e_col[0]);
      _sa *= markov[_old_index][_new_index];
      _v *= markov[_old_index][_new_index];
   }
   for (int i = 0; i < THIS.actions; i++)
   {  Q_SA[i][e_row[0]][e_col[0]] += THIS.alpha * ((Reward + (THIS.gamma * _sa)) - Q_SA[i][e_row[0]][e_col[0]]);
   }
   Q_V[e_row[0]][e_col[0]] += THIS.alpha * ((Reward + (THIS.gamma * _v)) - Q_V[e_row[0]][e_col[0]]);
}
```

Included in our modified and revised functions are updates with the addition of a new ‘Q\_V’ object that we have represented as a matrix for clarity in mapping to the respective environment states, but we could easily have had this as a vector since the environment state coordinates can be mapped into a single index integer. The old Q map is renamed ‘Q\_SA’. This new naming is in line with the new object tracking Q-Map values independent of actions which is what TD focuses on, while the renaming of the old Q-Map to Q\_SA emphasizes its state-action pair values that get updated whenever the function is called. Our MQL5 implementations above are derived from the following formula for TD (which can be either on or off policy):

### ![](https://c.mql5.com/2/100/4439184024072.png)

Where:

- V (s) : Value of the current state s
- V (s′) : Value of the next state s′
- α: Learning rate (controls how much we adjust the current value)
- r: Reward received after taking the action
- γ: Discount factor (determines the importance of future rewards)
- This formula updates the value estimate of a state V (s)  based on the reward received and the estimated value of the next state V (s′)

And to recap, the formula for state-action pairs with on-policy update, like SARSA for instance, is as follows:

![](https://c.mql5.com/2/100/1066198237530.png)

Where:

- Q (s, a) : Q-value of the current state-action pair (s, a)
- Q (s′, a′) : Q-value of the next state-action pair (s′, a′) where a′ is the action chosen by the current policy in the next state s′
- α: Learning rate
- r: Reward received after taking action a
- γ: Discount factor

Likewise, the formula for off-policy updates is as shown below:

![](https://c.mql5.com/2/100/3605541603744.png)

Where:

- Q (s, a) : Q-value of the current state-action pair (s, a)
- max a′ ​Q (s′, a′) : Maximum Q-value of the next state s′ over all possible actions a′ (assumes the best action in s′ will be taken)
- α: Learning rate
- r: Reward received after taking action a
- γ: Discount factor

From the last two formulae, it is clear the updates that are action specific do provide some input into the selection of the next action. However, for the TD update since it aggregates values across all actions and simply assigns them to their respective environment state, the influence of this process on the action to be selected is not well-defined.

That’s why in using TD an additional model, which in our case is a policy-neural-network, it is necessary to guide the selection of the next action for the actor. This model can take on a number of forms but for our purposes it simply is a 3-layer MLP sized as 3-9-3, where the output layer provides a probability distribution for each of the possible 3-actions which in our instance is still buy, sell & hold. This MLP will therefore be a classifier and not a regressor. We indicate the declaration code of this class in the custom signal class interface as shown below:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CSignalTD   : public CExpertSignal
{
protected:

   int                           m_actions;           // LetMarkov possible actions
   int                           m_environments;      // Environments, per matrix axis
   int                           m_scale;             // Environments, row-to-col scale
   bool                          m_use_markov;        // Use Markov
   double                        m_epsilon;           // Epsilon
   bool                          m_policy;            // On Policy

public:
   void                          CSignalTD(void);
   void                          ~CSignalTD(void);

   //--- methods of setting adjustable parameters
   void                          QL_Scale(int value)
   {  m_scale = value;
   }
   void                          QL_Markov(bool value)
   {  m_use_markov = value;
   }
   void                          QL_Epsilon(bool value)
   {  m_epsilon = value;
   }
   void                          QL_Policy(bool value)
   {  m_policy = value;
   }

   //--- method of verification of arch
   virtual bool      ValidationSettings(void);
   //--- method of creating the indicator and timeseries
   virtual bool      InitIndicators(CIndicators *indicators);
   //--- methods of checking if the market models are formed
   virtual int       LongCondition(void);
   virtual int       ShortCondition(void);

protected:
   int               GetOutput(Cql *QL, CNeuralNetwork &PN);
   Sql               RL;
   Cql               *QL_BUY, *QL_SELL;
   CNeuralNetwork    POLICY_NETWORK_BUY,POLICY_NETWORK_SELL;
};
```

The complete source code of our custom signal class is attached at the end of this article, and it is meant to be used via the MQL5 wizard to generate an Expert Advisor. There are guides [here](https://www.mql5.com/en/articles/171) and [here](https://www.mql5.com/en/articles/275) for readers who are new on how to do this. With our source code attached, the reader can easily make changes to the MLP design by changing no only the number of layers, but also their size. From our design though, the input layer being 3 is meant to take as inputs, the x-axis environment coordinate, the transition value or the current Q\_V matrix reading, and the environment y-axis reading. The transition value is taken from the Q\_V matrix that we update in on & off policy settings, as already shared in the source code above. This selection of the transition value is handled in the revised Action function as indicated below:

```
//+------------------------------------------------------------------+
// Choose an action using epsilon-greedy approach
//+------------------------------------------------------------------+
void Cql::Action(vector &E)
{  int _best_act = 0;
   if (double((rand() % SHORT_MAX) / SHORT_MAX) < THIS.epsilon)
   {  // Explore: Choose random action
      _best_act = (rand() % THIS.actions);
   }
   else
   {  // Exploit: Choose best action
      double _best_value = Q_SA[0][e_row[0]][e_col[0]];
      for (int i = 1; i < THIS.actions; i++)
      {  if (Q_SA[i][e_row[0]][e_col[0]] > _best_value)
         {  _best_value = Q_SA[i][e_row[0]][e_col[0]];
            _best_act = i;
         }
      }
   }
//update last action
   act[1] = act[0];
   act[0] = _best_act;
//markov decision process
   e_row[1] = e_row[0];
   e_col[1] = e_col[0];
   LetMarkov(e_row[1], e_col[1], E);
   int _next_state = 0;
   for (int i = 0; i < int(markov.Cols()); i++)
   {  if(markov[int(E[0])][i] > markov[int(E[0])][_next_state])
      {  _next_state = i;
      }
   }
   int _next_row = 0, _next_col = 0;
   SetMarkov(_next_state, _next_row, _next_col);
   e_row[0] = _next_row;
   e_col[0] = _next_col;
   transition_value = Q_V[_next_row][_next_col];
   policy_history[1][0] = policy_history[0][0];
   policy_history[1][1] = policy_history[0][1];
   policy_history[1][2] = policy_history[0][2];
   policy_history[0][0] = _next_row;
   policy_history[0][1] = transition_value;
   policy_history[0][2] = _next_col;
   transition_act = 1;
   for (int i = 0; i < THIS.actions; i++)
   {  if(Q_SA[i][_next_row][_next_col] > Q_SA[transition_act][_next_row][_next_col])
      {  transition_act = i;
      }
   }
   //if(transition_act!=1)printf(__FUNCSIG__+ " act is : %i ",transition_act);
}
```

The overriding thesis therefore in our policy network, the afore mentioned MLP, is that the suitable action to be selected next is a function of the current environment state and its Q-Value, only. This differs from what we have been using leading up to here, where the Markov Decision Process (MDP) was used to select the suitable action from the Q-Map that we renamed in this article (attached code) to Q\_SA. In all cases, we have used the memoryless MDP by computing a buffer of recent sequences of environment states. These environment sequences, thanks to the let Markov function, which is shared again below, gives us a projection for the next environment state.

```
//+------------------------------------------------------------------+
// Function to update markov matrix
//+------------------------------------------------------------------+
void Cql::LetMarkov(int OldRow, int OldCol, vector &E)  //
{  matrix _transitions;  // Count the transitions
   _transitions.Init(markov.Rows(), markov.Cols());
   _transitions.Fill(0.0);
   vector _states;  // Count the occurrences of each state
   _states.Init(markov.Rows());
   _states.Fill(0.0);
// Count transitions from state i to state ii
   for (int i = 0; i < int(E.Size()) - 1; i++)
   {  int _old_state = int(E[i]);
      int _new_state = int(E[i + 1]);
      _transitions[_old_state][_new_state]++;
      _states[_old_state]++;
   }
// Reset prior values to zero.
   markov.Fill(0.0);
// Compute probabilities by normalizing transition counts
   for (int i = 0; i < int(markov.Rows()); i++)
   {  for (int ii = 0; ii < int(markov.Cols()); ii++)
      {  if (_states[i] > 0)
         {  markov[i][ii] = double(_transitions[i][ii] / _states[i]);
         }
         else
         {  markov[i][ii] = 0.0;  // No transitions from this state
         }
      }
   }
}
```

This process of determining the next states thanks to MDP is still performed with TD, the difference here though being that we cannot use these projected next state coordinates by themselves to determine the next action. Previously, when we were using Q\_SA, it was a matter of reading off the action with the highest probability weighting off the next state coordinates to know what the actor was supposed to do. Now though, our equivalent matrix, the Q\_V only gives us values for any provided state coordinates and yet determining the actor’s next action is critical in the reinforcement learning loop.

That’s why we therefore introduce a policy network, an MLP, that for our purposes we have simply designed as a 3-9-3, featuring a single hidden layer of 9, taking in the 3 afore mentioned inputs of the two environment coordinates and their Q-Value, and outputting a vector of 3 that is meant to capture the probability distribution across the 3 possible actions of buy, sell, & hold with the value scoring the highest in the vector being the recommended action, is adopted.

### Macro Benefits and Purpose of TD

TD updates its Q-Values more frequently than in Monte Carlo and the advantages of this in rapidly changing, and fluid financial markets is obvious. However, what sets it apart from say SARSA or Q-Learning from an advantage standpoint? We try to answer this question by looking at a few day-to-day examples.

By definition, the main difference between TD and SARSA/ Q-Learning is that TD focuses more on value-based learning where only state values are leaned and updated while the other two algorithms focus on state-action pairing to perform similar updates.

Scenario A

Supposing there is an inventory management system in a warehouse that purely keeps track of the amount of stock levels across a multitude of products. The goal of this system would only be to manage the inventory stock levels and ensure there is no under or over stocking of any product.

In this situation, TD would be advantageous over the likes of SARSA or Q-Learning simply because of its focus on state-values as opposed to state-action pairs. This is because in this case, the system might only need to predict the "value" of each state (e.g., overall stock levels) without evaluating every specific action (e.g., ordering for each SKU). In this situation therefore, without the need for a policy network MLP, TD learning can update the value function for the state (inventory levels) without calculating every possible ordering decision for each product.

Furthermore, inventory management may have gradual changes instead of abrupt state-action pairs that have distinct reward feedback. Since TD learning deals with incremental feedback, this makes it suitable for environments with smooth transitions where knowing the overall state is more important than knowing every state-action outcome. Finally, environments that have several actions and large state-action spaces, such as inventory management that is more complex (where we would need to map a course of action to each inventory level state), Q-Learning and SARSA though applicable are bound to come at a compute cost, costs that TD never runs into because of its simpler wholistic application.

Scenario B

Consider a smart building system that adjusts heating, ventilation, and air conditioning (HVAC) settings so as to minimize energy consumption while keeping its occupants comfortable. Its goals therefore of balancing short-term rewards with long-term goals align with lowering energy bills and keeping the building at optimal temperature and air quality respectively.

TD would be better suited than SARSA or Q-Learning again in this case because the energy consumption levels or user-comfort are absolute metrics that do not have any action pegged to them depending on their values. In this particular case in order to balance short-term and long-term rewards two reinforcement learning cycles can be trained to forecast both in parallel. TD’s incremental per cycle updates (rather than per episode as we saw with Monte Carlo), also make it ideal for this smart building system since environmental conditions like temperature, occupancy, and air quality change gradually. This allows TD to provide a seamless adjustment mechanism.

Finally, as mentioned above, the SARSA or Q-Learning equivalent will come with additional compute and memory requirements since specific actions would be required to ‘remedy’ any short falls or excesses to the environment state values.

Scenario C

A traffic flow prediction and control system, with the goal of minimizing congestion at multiple intersections through the forecasting of traffic flow and respective adjustment of traffic light signals, provides our 3rd possible TD use case. TD is advantageous in this situation as well, when compared to SARSA or Q-Learning because the main concern is understanding and predicting the overall traffic state (like congestion levels) rather than optimizing each individual traffic signal action. TD learning enables the system to learn the overall "value" of a traffic state, rather than the impact of each specific signal change.

Traffic also being inherently dynamic and changing continuously throughout the day does play into TD’s incremental update MO. A very adaptable algorithm, unlike Monte Carlo that waits for episode completion before performing updates. Reduction in compute and memory overloads for TD also apply in this case, especially if one considers how intertwined and interconnected traffic junctions can be in even a relatively small city. Compute and memory costs would certainly be a huge factor when implementing a traffic flow prediction system, and TD would go a long way in addressing this.

Scenario D

Predictive maintenance in manufacturing, presents our final use case for TD as a preferred algorithm when compared to other algorithms in reinforcement learning we have covered this far. Consider a case where a manufacturing plant aims to predict when machines need maintenance so as to avoid downtime. And just like with the smart building system, this system would need to balance short-term rewards (saving maintenance costs by delaying checks) with long-term gains (preventing breakdowns). TD learning would be suitable here because it can update the machine’s overall health value over time based on partial feedback, rather than tracking specific actions (repair or replace) as SARSA or Q-learning would.

TD would also be a better algorithm because machine degradation happens gradually, and TD can easily update the health value continuously based on an incremental sensor data rather than waiting for longer periods which could run into additional risks. Also, in a plant that has multiple machines, TD learning would be able to scale well, as the number of machines is decreased or increased because it focuses on tracking just the state/ health of each machine and has no need to store and update specific state-actions pairs for each machine in the plant.

These are a few cases, outside of financial trading and the markets, so let us now consider how we can specifically apply this as a custom signal class.

### Structuring the Custom Signal Class Using TD Algorithm

The custom signal class that we build to implement TD relies on two additional classes. Since it is an algorithm for reinforcement learning, one of those classes is the CQL class. We have already used or referred to this class in all the previous articles; however, its interface is shared again below:

```
//+------------------------------------------------------------------+
//| Q_SA-Learning Class Interface.                                      |
//+------------------------------------------------------------------+
class Cql
{
protected:
   matrix            markov;
   void              LetMarkov(int OldRow, int OldCol, vector &E);

   vector            acts;
   matrix            environment;
   matrix            Q_SA[];
   matrix            Q_V;

public:
   void              Action(vector &E);
   void              Environment(vector &E_Row, vector &E_Col, vector &E);

   void              SetOffPolicy(double Reward, vector &E);
   void              SetOnPolicy(double Reward, vector &E);

   double            GetReward(double MaxProfit, double MaxLoss, double Float);
   vector            SetTarget(vector &Rewards, vector &TargetOutput);

   void              SetMarkov(int Index, int &Row, int &Col);
   int               GetMarkov(int Row, int Col);

   Sql               THIS;

   int               act[2];

   int               e_row[2];
   int               e_col[2];

   int               transition_act;
   double            transition_value;

   matrix            policy_history;

   vector            Q_Loss()
   {  vector _loss;
      _loss.Init(THIS.actions);
      _loss.Fill(0.0);
      for(int i = 0; i < THIS.actions; i++)
      {  _loss[i] = Q_SA[e_row[0]][e_col[0]][i];
      }
      return(_loss);
   }

   void              Cql(Sql &RL)
   {  //
      if(RL.actions > 0 && RL.environments > 0)
      {  policy_history.Init(2,2+1);
         policy_history.Fill(0.0);
         acts.Init(RL.actions);
         ArrayResize(Q_SA, RL.actions);
         for(int i = 0; i < RL.actions; i++)
         {  acts[i] = i + 1;
            Q_SA[i].Init(RL.environments, RL.environments);
         }
         Q_V.Init(RL.environments, RL.environments);
         environment.Init(RL.environments, RL.environments);
         for(int i = 0; i < RL.environments; i++)
         {  for(int ii = 0; ii < RL.environments; ii++)
            {  environment[i][ii] = ii + (i * RL.environments) + 1;
            }
         }
         markov.Init(RL.environments * RL.environments, RL.environments * RL.environments);
         markov.Fill(0.0);
         THIS = RL;
         ArrayFill(e_row, 0, 2, 0);
         ArrayFill(e_col, 0, 2, 0);
         ArrayFill(act, 0, 2, 1);
         transition_act = 1;
      }
   };
   void              ~Cql(void) {};
};
```

This is the main reinforcement learning class and some of its classes defining on and off policy updates have already been shared above. Also, this class is often sufficient to implement reinforcement learning for a custom signal class however for TD since we are not using or solely relying on environment values to make trade decisions but are electing to continue with the forecasting of suitable actions as we have been with past, we need a model to make these projections.

That’s why an additional class, the CNeuralNetwork class is used to define this model as a neural network or MLP. Similarly, its class interface is shared below:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CNeuralNetwork
{
protected:

   matrix               TransposeToCol(vector &V);
   matrix               TransposeToRow(vector &V);

public:
   CLayer               *layers[];
   double               m_learning_rate;
   ENUM_LOSS_FUNCTION   m_loss;
   void                 AddDenseLayer(ulong Neurons, ENUM_ACTIVATION_FUNCTION AF = AF_RELU, ulong LastNeurons = 0)
   {  ArrayResize(layers, layers.Size() + 1);
      layers[layers.Size() - 1] = new CLayer(Neurons, AF);
      if(LastNeurons  != 0)
      {  layers[layers.Size() - 1].AddWeights(LastNeurons);
      }
      else if(layers.Size() - 1 > 0)
      {  layers[layers.Size() - 1].AddWeights(layers[layers.Size() - 2].activations.Size());
      }
   };
   void                 Init(double LearningRate, ENUM_LOSS_FUNCTION LF)
   {  m_loss = LF;
      m_learning_rate = LearningRate;
   };

   vector               Forward(vector& Data);
   void                 Backward(vector& LabelAnswer);

   void                 CNeuralNetwork(){};
   void                 ~CNeuralNetwork()
   {  if(layers.Size() > 0)
      {  for(int i = 0; i < int(layers.Size()); i++)
         {  delete layers[i];
         }
      }
   };
};
```

We have made some noticeable changes from the last CMLP class that was performing a similar function. Overall emphasis was on efficiency; however, this is still a beta even though for our purposes it was able to give us some results. Besides the efficiency changes that were mostly in the backpropagation (Backward) function, and are a work in progress, we introduced a layer class, and also changed the way the network is constructed. The initialization of the custom signal class now looks as follows:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
void CSignalTD::CSignalTD(void) :    m_scale(5),
   m_use_markov(true),
   m_policy(true)

{
//--- initialization of protected data
   m_used_series = USE_SERIES_OPEN + USE_SERIES_HIGH + USE_SERIES_LOW + USE_SERIES_CLOSE + USE_SERIES_SPREAD + USE_SERIES_TIME;
   //
   RL.actions  = 3;//buy, sell, do nothing
   RL.environments = 3;//bullish, bearish, flat
   RL.use_markov = m_use_markov;
   RL.epsilon = m_epsilon;
   QL_BUY = new Cql(RL);
   QL_SELL = new Cql(RL);
   //
   POLICY_NETWORK_BUY.AddDenseLayer(9, AF_SIGMOID, 3);
   POLICY_NETWORK_BUY.AddDenseLayer(3, AF_SOFTMAX);
   POLICY_NETWORK_BUY.Init(0.0004,LOSS_BCE);
   //
   POLICY_NETWORK_SELL.AddDenseLayer(9, AF_SIGMOID, 3);
   POLICY_NETWORK_SELL.AddDenseLayer(3, AF_SOFTMAX);
   POLICY_NETWORK_SELL.Init(0.0004,LOSS_BCE);
}
```

Besides having 2 CQL classes to handle reinforcement learning on the buy and sell side respectively, we now have 2 policy networks to also make action forecasts again on the buy and sell side respectively. The get output function still runs much the same as it has been in past articles, with the main change being that an instance of the neural network class is one of its inputs as a policy network. Its new listing is as follows:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CSignalTD::GetOutput(Cql *QL, CNeuralNetwork &PN)
{  int _td_act = 1;
   vector _in, _in_row, _in_row_old, _in_col, _in_col_old;
   if
   (
      _in_row.Init(m_scale) &&
      _in_row.CopyRates(m_symbol.Name(), m_period, 8, 0, m_scale+1) &&
      _in_row.Size() == m_scale+1
      &&
      _in_row_old.Init(m_scale) &&
      _in_row_old.CopyRates(m_symbol.Name(), m_period, 8, 1, m_scale+1) &&
      _in_row_old.Size() == m_scale+1
      &&
      _in_col.Init(m_scale) &&
      _in_col.CopyRates(m_symbol.Name(), m_period, 8, 0, m_scale+1) &&
      _in_col.Size() == m_scale+1
      &&
      _in_col_old.Init(m_scale) &&
      _in_col_old.CopyRates(m_symbol.Name(), m_period, 8, m_scale, m_scale+1) &&
      _in_col_old.Size() == m_scale+1
   )
   {  _in_row -= _in_row_old;
      _in_col -= _in_col_old;
      _in_row.Resize(m_scale);
      _in_col.Resize(m_scale);
      vector _in_e;
      _in_e.Init(m_scale);
      QL.Environment(_in_row, _in_col, _in_e);
      int _row = 0, _col = 0;
      QL.SetMarkov(int(_in_e[m_scale - 1]), _row, _col);
      double _reward_float = _in_row[m_scale - 1];
      double _reward_max = _in_row.Max();
      double _reward_min = _in_row.Min();
      double _reward = QL.GetReward(_reward_max, _reward_min, _reward_float);
      if(m_policy)
      {  QL.SetOnPolicy(_reward, _in_e);
      }
      else if(!m_policy)
      {  QL.SetOffPolicy(_reward, _in_e);
      }
      PN.Forward(QL.policy_history.Row(1));
      vector _label;
      _label.Init(3);
      _label.Fill(0.0);
      if(_in_row[m_scale-1] > 0.0)
      {  _label[0] = 1.0;
      }
      else if(_in_row[m_scale-1] < 0.0)
      {  _label[2] = 1.0;
      }
      else if(_in_row[m_scale-1] == 0.0)
      {  _label[1] = 1.0;
      }
      PN.Backward(_label);
      vector _td_output = PN.Forward(QL.policy_history.Row(0));
      if(_td_output[0] >= _td_output[1] && _td_output[0] >= _td_output[2])
      {  _td_act = 0;
      }
      else if(_td_output[2] >= _td_output[0] && _td_output[2] >= _td_output[1])
      {  _td_act = 2;
      }
   }
   return(_td_act);
}
```

With a policy network as one of its inputs, it’s now able to perform online or incremental learning since we do not train on a batch of data before forecasting but on just the latest single bar data. There should only be a backpropagation run and one forward feed run but because we need to load the current price bar information into our neural network before doing the Backward or backpropagation run, we make a single Forward run with the previous bar information before it, to load this information. Our Forward function is also modified to return the output classification vector from the forward feed pass. It is sized three where each value at the respective index provides a ‘probability’ or threshold of the likelihood of either selling, holding or buying if we are to follow the indexing order 0, 1, 2 respectively.

So, to do a recap of what was mentioned above, we are sticking with the same simple reinforcement learning environment and actions setup of 3 states and 3 actions. Policy update choice is optimizable, as in the last reinforcement learning article that covered Monte Carlo. When updates are performed here though, they only update the Q-Values of a new introduced matrix the Q\_V. This contrasts with what we have done in the past of updating Q-Values across an environment map, _for each action_. The updating of values for each action was constructive in selecting the next action because once the Markov Decision Process is used to determine the coordinates of the next environment state, then it simply becomes a matter of reading off the action with the highest Q-Value and selecting this as our next action.

With TD, our environment matrix Q\_V does not have actions with Q-Values but rather only has values assigned to a particular environment. This means that in order to select or determine the next course of action, a policy network (an MLP in our case) is used to make these forecasts and its inputs would be the value for the current environment (which is a defacto sum of all the Q-Values of applicable actions at that state) as well as the environment state coordinates. The output also as mentioned is a ‘probability distribution’ across the 3 possible actions on which action would be best given the current environment and its value.

This custom signal class is thus assembled with the MQL5 wizard into an Expert Advisor, and after a brief optimization stint, for the year 2022 on the 1-hour time frame that uses the symbol GBP USD, one of the favourable settings from that stint provide us with the following report:

![r1](https://c.mql5.com/2/100/r1__2.png)

![c1](https://c.mql5.com/2/100/c1.png)

These results perhaps speak to the potential of our custom signal class, but without denigrating them, it always recommended to cross-validate any Expert Advisor’s settings before choosing to deploy it in a live environment. The cross-validation and extra testing diligence over a longer time window than what we have done here is left to the reader to look into as he may see fit.

### Optimizing and Tuning the Signal Class with TD Learning

From our test runs above, we were solely optimizing for epsilon, whether to use Markov weighting in the update process, and whether to perform on policy updates or off policy updates. There of course were additional typical non-TD parameters like the open and close thresholds for the long and short conditions, the take profit level, as well as entry price threshold in points.

However, within TD and reinforcement learning for that matter, there are quite a few parameters that we glossed over and used them with preset values. These are alpha and gamma that are assigned the values 0.1 and 0.5 respectively. These two parameters are key in the policy updates and arguably could be very sensitive to the overall performance of the signal class. The other key parameters that we have overlooked in implementing our signal class by effectively assigning them constant values are the settings of the policy networks. We stuck with a 3-9-3 network where the activation functions on each layer were all preset, as well as the learning rate. Each one of these and probably all of them when adjusted concurrently can have large effects on the results and performance of the custom signal class.

### Conclusion

We have looked at the Temporal Difference algorithm of reinforcement learning and tried to highlight its use cases that set it apart from other algorithms that we have already covered. One aspect we have not looked at that is interesting to reinforcement learning in general is the decaying of exploration over time. The thesis or argument behind this is that as a reinforcement learning model learns over time, the need to explore new territory or keep learning greatly diminishes; therefore exploitation would be more important. This is something that readers can look into when looking to customize the attached code for further use.

Another aspect could be making epsilon variable, and not just by decreasing it, which is what the decaying above implies. The thesis of this is reinforcement learning is meant to be very dynamic and adaptable, unlike supervised learning models, which rely on static data. TD’s reinforcement learning framework therefore can actively engage with a changing environment. We have considered dynamic learning rate methods in a [past article](https://www.mql5.com/en/articles/15405), and it could be argued that the same could be considered for epsilon so that the presumption of decaying alone does not serve as the only way of keeping the reinforcement learning to its roots.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16303.zip "Download all attachments in the single ZIP archive")

[Cmlpw.mqh](https://www.mql5.com/en/articles/download/16303/cmlpw.mqh "Download Cmlpw.mqh")(5.56 KB)

[Cql.mqh](https://www.mql5.com/en/articles/download/16303/cql.mqh "Download Cql.mqh")(11.63 KB)

[SignalWZ\_47.mqh](https://www.mql5.com/en/articles/download/16303/signalwz_47.mqh "Download SignalWZ_47.mqh")(9.12 KB)

[wz\_47.mq5](https://www.mql5.com/en/articles/download/16303/wz_47.mq5 "Download wz_47.mq5")(6.82 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/476534)**

![Creating a Trading Administrator Panel in MQL5 (Part VI): Multiple Functions Interface (I)](https://c.mql5.com/2/101/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_VI___LOGO.png)[Creating a Trading Administrator Panel in MQL5 (Part VI): Multiple Functions Interface (I)](https://www.mql5.com/en/articles/16240)

The Trading Administrator's role goes beyond just Telegram communications; they can also engage in various control activities, including order management, position tracking, and interface customization. In this article, we’ll share practical insights on expanding our program to support multiple functionalities in MQL5. This update aims to overcome the current Admin Panel's limitation of focusing primarily on communication, enabling it to handle a broader range of tasks.

![Price Action Analysis Toolkit Development (Part 1): Chart Projector](https://c.mql5.com/2/101/Price_Action_Analysis_Toolkit_Development_Part_1____LOGO__2.png)[Price Action Analysis Toolkit Development (Part 1): Chart Projector](https://www.mql5.com/en/articles/16014)

This project aims to leverage the MQL5 algorithm to develop a comprehensive set of analysis tools for MetaTrader 5. These tools—ranging from scripts and indicators to AI models and expert advisors—will automate the market analysis process. At times, this development will yield tools capable of performing advanced analyses with no human involvement and forecasting outcomes to appropriate platforms. No opportunity will ever be missed. Join me as we explore the process of building a robust market analysis custom tools' chest. We will begin by developing a simple MQL5 program that I have named, Chart Projector.

![Client in Connexus (Part 7): Adding the Client Layer](https://c.mql5.com/2/101/http60x60.png)[Client in Connexus (Part 7): Adding the Client Layer](https://www.mql5.com/en/articles/16324)

In this article we continue the development of the connexus library. In this chapter we build the CHttpClient class responsible for sending a request and receiving an order. We also cover the concept of mocks, leaving the library decoupled from the WebRequest function, which allows greater flexibility for users.

![Trading with the MQL5 Economic Calendar (Part 2): Creating a News Dashboard Panel](https://c.mql5.com/2/101/Trading_with_the_MQL5_Economic_Calendar_Part_2___LOGO__1.png)[Trading with the MQL5 Economic Calendar (Part 2): Creating a News Dashboard Panel](https://www.mql5.com/en/articles/16301)

In this article, we create a practical news dashboard panel using the MQL5 Economic Calendar to enhance our trading strategy. We begin by designing the layout, focusing on key elements like event names, importance, and timing, before moving into the setup within MQL5. Finally, we implement a filtering system to display only the most relevant news, giving traders quick access to impactful economic events.

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/16303&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062605851975394662)

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