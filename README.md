# vramfit

Will this model fit in this card, and roughly how fast will it be.

One script so far. You tell it the parameter count and the quant, it
tells you how much VRAM the weights want.

    python fit.py --params 8.03e9 --quant q4_k_m

The point is to stop loading a model to find out.
