## Summary

This is the code for the ICML 2023 paper, "Learning to acquire novel cognitive tasks with evolution, plasticity and meta-meta-learning".

We evolve a recurrent  network, endowed with a reward-modulated Hebbian
plasticity rule, that can  automatically learn simple cognitive tasks from
stimuli and rewards alone. The network is tested on a new task, never seen
during evolution (delayed match-to-sample).

## How to use

1- Run  `code.py`. A full  run  of 1000 generations will take about half a day on a machine with a standard GPU, but you can stop it before that.

2- This will  generate several log files. The most important  is
`blosses_onerun.txt`, which records the main evaluatioin metric (mean success
rate over the last 100 trials of a  block) for the currrent candidate (i.e.
batch element 0, the unmutated genome). Every 10th value in this file is obtained on the withheld test task; others are on various training-set tasks. 

3- Run `curves.py', which  will automatically generate curves for training and testing loss, as in Figure 2 of the paper. These curves will only include the one run you just rran, so there will only be one line for each curve with no error interval.

4- Repeat the same process as many times as you like, each time saving `blosses_onerun.txt` under a diff

