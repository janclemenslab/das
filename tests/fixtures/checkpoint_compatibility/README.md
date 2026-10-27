Small synthetic checkpoints and reference outputs saved before removing DAS SSL
pretraining. They exercise the existing legacy H5/YAML TCN and STFT formats and
a native Torch checkpoint whose training metadata includes the former
`pretrained_checkpoint` and `freeze_pretrained` fields.

The legacy weights use NumPy seed 1234; reference inputs use NumPy seed 42.
The native model uses Torch seed 42. Outputs were captured in evaluation mode
with the unchanged model implementations. The regression check also converts
both legacy fixtures and compares their native checkpoint outputs.
