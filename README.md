# Building an A3C agent using TensorFlow.

## Current status

**I do not recommend reviewing the current code.** It currently doesn't work, isn't properly commented, and I expect to make major updates over the coming 2 weeks.

- Running just a single thread works in principle and doesn't seem to immediately suffer from vanishing or exploding gradients, but I haven't yet tested the agent's performance after substantial training.
- Runs so far exhibit intermittent periods of several ten-thousand frames where gradients are nearly zero and the agent constantly uses a uniformly random policy.
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
