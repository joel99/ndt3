- We will repeat the analysis from `arch/base` across 5 sessions.
- Not all these sessions can afford 50% test set while having calibration = 300 trials, so we reduce to 10% for a consistent.

Each test dataset needs its own probe, etc, the logistics of tracking more variants is blowing up. To simplify:
- It would be too much to sweep single session HPs, so we will use the same HPs as `arch/base` (which served fine there).
- We also discard the other parameter variants, maintaining only the core relevant ones.

`tune` folder adapts multi-context models to each test session.
`probe` folder velocity decodes from adapted or single-session models.

Some other notes:
- We assume TAPT performs on par with in-pretraining (seemed to be the case in pilots)
- We also cannot run single-session baselines on the 20-40 trial Pitt datasets.
- We do not outperform NLB models, but that is not explored in this particular experiment.

Note that with running these many experiments, the wandb service will sometimes not initialize in time. I'm not sure how to set timeouts, so I relaunch the desired runs manually (you can leave `frag` settings on, and just update the datasets that need to be relaunched.)
TODO figure out timeout setting.