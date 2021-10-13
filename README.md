# D-GCCA: Decomposition-based Generalized Canonical Correlation Analysis for Multiple High-dimensional Datasets 

This python package implements the D-GCCA method proposed in [1]. See [example.py](https://github.com/shu-hai/D-GCCA/blob/master/example.py) for details, with Python 3.

D-GCCA decomposes the observed multi-block data <img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{Y}_k\in \mathbb{R}^{p_k\times n}, k=1,\dots, K\ge 2"> into

<img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{Y}_k=\boldsymbol{X}_k %2B \boldsymbol{E}_k=\boldsymbol{C}_k %2B \boldsymbol{D}_k %2B \boldsymbol{E}_k">

where <img src="https://render.githubusercontent.com/render/math?math=\{\boldsymbol{C}_k\}_{k=1}^K"> are low-rank common-source matrices that represent the signal data coming from the common latent factors shared across all datasets, and <img src="https://render.githubusercontent.com/render/math?math=\{\boldsymbol{D}_k\}_{k=1}^K"> are low-rank distinctive-source matrices each from the distinctive latent factors of the corresponding dataset, and <img src="https://render.githubusercontent.com/render/math?math=\{\boldsymbol{E}_k\}_{k=1}^K"> are noise matrices.
In other words, the common-source and distinctive-source matrices contain the variation information of each dataset, respectively, explained by the common and distinctive latent factors of the K datasets.


Please cite the article [1] for this package, which is available [here](https://arxiv.org/abs/2001.02856).

[1] Shu, H., Qu, Z., & Zhu, H. D-GCCA: Decomposition-based Generalized Canonical Correlation Analysis for Multiple High-dimensional Datasets. arXiv preprint arXiv:2001.02856.
