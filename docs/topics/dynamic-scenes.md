# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-01-31 06:56 UTC_

Total papers shown: **2**


---

- **Latent Temporal Discrepancy as Motion Prior: A Loss-Weighting Strategy for Dynamic Fidelity in T2V**  
  Meiqi Wu, Bingze Song, Ruimin Lin, Chen Zhu, Xiaokun Feng, Jiahong Wu, Xiangxiang Chu, Kaiqi Huang  
  _2026-01-28_ · https://arxiv.org/abs/2601.20504v1  
  <details><summary>Abstract</summary>

  Video generation models have achieved notable progress in static scenarios, yet their performance in motion video generation remains limited, with quality degrading under drastic dynamic changes. This is due to noise disrupting temporal coherence and increasing the difficulty of learning dynamic regions. {Unfortunately, existing diffusion models rely on static loss for all scenarios, constraining their ability to capture complex dynamics.} To address this issue, we introduce Latent Temporal Discrepancy (LTD) as a motion prior to guide loss weighting. LTD measures frame-to-frame variation in the latent space, assigning larger penalties to regions with higher discrepancy while maintaining regular optimization for stable regions. This motion-aware strategy stabilizes training and enables the model to better reconstruct high-frequency dynamics. Extensive experiments on the general benchmark VBench and the motion-focused VMBench show consistent gains, with our method outperforming strong baselines by 3.31% on VBench and 3.58% on VMBench, achieving significant improvements in motion quality.

  </details>



- **CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting**  
  Jiyuan Xu, Wenyu Zhang, Xin Jing, Shuai Chen, Shuai Zhang, Jiahao Nie  
  _2026-01-28_ · https://arxiv.org/abs/2601.20318v1  
  <details><summary>Abstract</summary>

  Current methods for multivariate time series forecasting can be classified into channel-dependent and channel-independent models. Channel-dependent models learn cross-channel features but often overfit the channel ordering, which hampers adaptation when channels are added or reordered. Channel-independent models treat each channel in isolation to increase flexibility, yet this neglects inter-channel dependencies and limits performance. To address these limitations, we propose \textbf{CPiRi}, a \textbf{channel permutation invariant (CPI)} framework that infers cross-channel structure from data rather than memorizing a fixed ordering, enabling deployment in settings with structural and distributional co-drift without retraining. CPiRi couples \textbf{spatio-temporal decoupling architecture} with \textbf{permutation-invariant regularization training strategy}: a frozen pretrained temporal encoder extracts high-quality temporal features, a lightweight spatial module learns content-driven inter-channel relations, while a channel shuffling strategy enforces CPI during training. We further \textbf{ground CPiRi in theory} by analyzing permutation equivariance in multivariate time series forecasting. Experiments on multiple benchmarks show state-of-the-art results. CPiRi remains stable when channel orders are shuffled and exhibits strong \textbf{inductive generalization} to unseen channels even when trained on \textbf{only half} of the channels, while maintaining \textbf{practical efficiency} on large-scale datasets. The source code is released at https://github.com/JasonStraka/CPiRi.

  </details>


