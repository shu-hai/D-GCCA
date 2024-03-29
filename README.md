# D-GCCA: Decomposition-based Generalized Canonical Correlation Analysis for Multi-view High-dimensional Data 

This python package implements the D-GCCA method proposed in [1]. See [example.py](https://github.com/shu-hai/D-GCCA/blob/master/example.py) for details, with Python 3.

D-GCCA decomposes the observed multi-view data <img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{Y}_k\in \mathbb{R}^{p_k\times n}, k=1,\dots, K\ge 2"> into

<img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{Y}_k=\boldsymbol{X}_k %2B \boldsymbol{E}_k=\boldsymbol{C}_k %2B \boldsymbol{D}_k %2B \boldsymbol{E}_k">

where <img src="https://render.githubusercontent.com/render/math?math=\{\boldsymbol{C}_k\}_{k=1}^K"> are low-rank common-source matrices that represent the signal data coming from the common latent factors shared across all data views, and <img src="https://render.githubusercontent.com/render/math?math=\{\boldsymbol{D}_k\}_{k=1}^K"> are low-rank distinctive-source matrices each from the distinctive latent factors of the corresponding view, and <img src="https://render.githubusercontent.com/render/math?math=\{\boldsymbol{E}_k\}_{k=1}^K"> are noise matrices.
In other words, the common-source and distinctive-source matrices contain the variation information in each view, respectively, explained by the common and distinctive latent factors of the K views.


Please cite the article [1] for this package.

For K=2 data views, please use the [D-CCA](https://github.com/shu-hai/D-CCA) package, instead.

[1] Shu, H., Qu, Z., & Zhu, H. ["D-GCCA: Decomposition-based Generalized Canonical Correlation Analysis for Multi-view High-dimensional Data"](https://www.jmlr.org/papers/volume23/20-021/20-021.pdf). Journal of Machine Learning Research, 23(169):1−64
