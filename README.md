# Softmax Recurrent Unit (SMRU)

This repository contains the implementation and experiments of the SMRU as reported in the paper [Softmax Recurrent Unit: A new type of gated RNN cell](https://arxiv.org/abs/XXXX.YYYYY) by Lucas Vos and Twan van Laarhoven.


## Reference
If you use this code or research, please cite the following [paper](https://arxiv.org/abs/XXXX.YYYYY): 

```
@article{VosSMRU2019,
	author    = {{Vos}, Lucas and {van Laarhoven}, Twan},
	title     = {Softmax Recurrent Unit: A new type of gated RNN cell},
	journal   = {arXiv:XXXX:YYYYY},
	year      = {2019},
}
```

## SMRU Implementation

We used an older PyTorch version, before the cuDNN implementations, to code the SMRU as a PyTorch compatible RNN cell. This implementation  only provides a CPU based version of the SMRU. 

