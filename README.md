# Building an A3C agent using TensorFlow.

## Update 2018-07-29

- I wasn't able to fix the "Illegal Instruction!" errors.
- I tried reproducing these errors in a minimal example but didn't succeed.
- I trained an agent using just one thread on 4 million frames, but it's hard to tell for me if it's working properly.

![Value estimate and max prob of policy](https://github.com/maxmdaniel/a3c/blob/master/Max-prob_and_value.png)

![Gradient norms and losses](https://github.com/maxmdaniel/a3c/blob/master/grad_norms_and_loss.png)

## Next steps

- Reproduce "Illegal Instruction!" errors in minimal example and find out how to fix.
- Replace Atari environment with [Catch](https://github.com/OpenMined/CampX/) for faster experiments. Only try Atari once succeeded at Catch.

## Current status

**I do not recommend reviewing the current code.** It currently doesn't work and isn't properly commented.

- Running just a single thread works in principle and doesn't seem to immediately suffer from vanishing or exploding gradients, but I haven't yet tested the agent's performance after substantial training.
- Runs so far exhibit intermittent periods of several ten-thousand frames where gradients are nearly zero and the agent constantly uses a uniformly random policy.

![Value estimate and max probability of the policy](https://github.com/maxmdaniel/a3c/blob/master/random_periods.png)

- When using multiple threads currently throws errors like "Illegal Instruction! 84"

## Goal

Replicate performance of [Mnih et al. (2016)](https://arxiv.org/pdf/1602.01783.pdf).

## Rationale

I chose this particular project because (i) it was among
  the suggestions for things to build I had received from people
  working in the field, (ii) it seemed like one of few opportunities
  that involve interesting engineering challenges relative to my level
  of experience (e.g. the need for a deep neural net as a function
  approximator, and how to create a TensorFlow graph that can be run
  from multiple threads) but can be run with just my laptop's CPU,
  (iii) I had some ideas for more difficult follow-up experiments
  (e.g. try to build a version of asynchronous n-step Q-learning using
  eligibility traces). 
