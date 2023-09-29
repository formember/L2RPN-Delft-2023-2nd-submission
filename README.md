# L2RPN Delft 2023 competition

This is a repository of 2nd submission in [L2RPN Delft 2023](https://codalab.lisn.upsaclay.fr/competitions/12420). 

## Our solution

- For discrete actions:
  - We first reduce more than 70k actions into 314 actions following  [L2RPN_NIPS_2020_a_PPO_Solution](https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution) 
  - Then we explore ~800,000 scenarios on 32-core CPU with 32 processes for ~28h. For each scenario we search for a greedy action from the reduced action space to minimize the maximum load rate of all lines. In the end we get a 14 Gigabytes dataset.
  - Train a neural network on above dataset on a machine with 32-core CPU and 4*nvidia-V100 for ~13h.
  - We train the PPO model distributedly.
- For continuous actions:
  - We follow the [OptimCVXPY](https://github.com/rte-france/l2rpn-baselines/tree/master/l2rpn_baselines/OptimCVXPY) in L2RPN baselines.

- For the final submission, we are inspired by [l2rpn wcci 2022](https://github.com/AlibabaResearch/l2rpn-wcci-2022).

## Description of the files

- max_array.npy, min_array.npy: used for normalisation
- acttions_space.npz: reduced action space
- ppo_ckpt: ppo model

