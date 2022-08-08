# Residual Shuffle-Exchange Networks: Unofficial PyTorch Implementation

This repository contains the unofficial _PyTorch_ implementation of the following paper:

>**Residual Shuffle-Exchange Networks for Fast Processing of Long Sequences**
>
> by Andis Draguns, Emīls Ozoliņš, Agris Šostaks, Matīss Apinis, Kārlis Freivalds
>
> [[arXiv](https://arxiv.org/abs/2004.04662)]
>
>Abstract: _Attention is a commonly used mechanism in sequence processing, but it is of O(n²) complexity which prevents its application to long sequences. The recently introduced neural Shuffle-Exchange network offers a computation-efficient alternative, enabling the modelling of long-range dependencies in O(n log n) time. The model, however, is quite complex, involving a sophisticated gating mechanism derived from the Gated Recurrent Unit._
>
>_In this paper, we present a simple and lightweight variant of the Shuffle-Exchange network, which is based on a residual network employing GELU and Layer Normalization. The proposed architecture not only scales to longer sequences but also converges faster and provides better accuracy. It surpasses the Shuffle-Exchange network on the LAMBADA language modelling task and achieves state-of-the-art performance on the MusicNet dataset for music transcription while being efficient in the number of parameters._
>
>_We show how to combine the improved Shuffle-Exchange network with convolutional layers, establishing it as a useful building block in long sequence processing applications._

# Current Status

Algorithmic benchmarks are close to the paper and generalization to longer sequences can be observed.

Results on MusicNet task are much worse, so far I only managed to achieve ~12% APS.

# Heavy weight-sharing

In the paper authors propose separate sets of weights for "forward" and "reversed" switch units.
I studied a variant with single set of weights used for all instances of switch unit in a single block.
You can find my results in notebooks/Algorithmic.ipynb. In short: it's probably not worth it.