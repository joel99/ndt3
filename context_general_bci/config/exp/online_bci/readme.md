Configs specified in this directory are top level training configurations. Configurations in subdirs are recommended at test time, and reference the training configuration that is pulled via `config.tag`.

Though we have config to freeze backbone, doesn't seem necessary past 100-ish calibration trials.

On-rig training uses the non-cluster configs. There is no enforcing, but please keep the cluster and non-cluster configs (which use gdrive links specified in `decoder_module` outside this repo) pointing to the same checkpoints.

We currently track test time configuration like crop bin in these yamls only; even if using an identical checkpoint, only way to hot swap is to use NDTProxy to pull via the yaml, which will pull an identical ckpt but use yaml's test-time config. Using NDTProxy to directly select ckpt will maintain the training config.

Closed loop crop bin and task embed ckpt are test time config, swappable without retraining. HOWEVER, closed loop crop bin is read from the config, while task embed ckpt is baked into the onnx file. Which means that closed loop crop bin must require a new onnx load, while task embed can be toggled in Exec.
Threshold or not is tuning time. Key difference is whether a new tag is assigned.