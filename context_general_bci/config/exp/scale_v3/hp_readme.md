As we scale, there are two obvious hyperparameters to be sensitive to:
1. Batch size
   1. We use a heuristic for this. In experiments to this point, we did not observe drastic effects of batch size (i.e. they should be dwarfed by effects of scale.)
   2. So heuristic is that batch size should be roughly 1-2 order of magnitude smaller than the number of trials.
2. Model capacity
   1. We'll loosely manually tune this. We also haven't observed _huge_ effects but we should be careful. Overlap each scale with the higher capacity's lower scale.

Equivalent experiments (that we can save running):
- `session/s20k_h256`, use `exp/arch/robust_unsort/f32`
- `subject/s20k_h256`, use `exp/arch/robust_unsort/subject_f32`
- `task/s20k_h256`, use `exp/arch/robust_unsort/task_f32`

The size incremental pretrain shows minimal benefit from scaling model size, so we jointly increase...
- And we take the _smaller_ size for any given data scale.