# Neural Rendering & View Synthesis

_Updated: 2026-03-23 07:38 UTC_

Total papers shown: **3**


---

- **LagerNVS: Latent Geometry for Fully Neural Real-time Novel View Synthesis**  
  Stanislaw Szymanowicz, Minghao Chen, Jianyuan Wang, Christian Rupprecht, Andrea Vedaldi  
  _2026-03-20_ · https://arxiv.org/abs/2603.20176v1  
  <details><summary>Abstract</summary>

  Recent work has shown that neural networks can perform 3D tasks such as Novel View Synthesis (NVS) without explicit 3D reconstruction. Even so, we argue that strong 3D inductive biases are still helpful in the design of such networks. We show this point by introducing LagerNVS, an encoder-decoder neural network for NVS that builds on `3D-aware' latent features. The encoder is initialized from a 3D reconstruction network pre-trained using explicit 3D supervision. This is paired with a lightweight decoder, and trained end-to-end with photometric losses. LagerNVS achieves state-of-the-art deterministic feed-forward Novel View Synthesis (including 31.4 PSNR on Re10k), with and without known cameras, renders in real time, generalizes to in-the-wild data, and can be paired with a diffusion decoder for generative extrapolation.

  </details>



- **Generalizable NGP-SR: Generalizable Neural Radiance Fields Super-Resolution via Neural Graph Primitives**  
  Wanqi Yuan, Omkar Sharad Mayekar, Connor Pennington, Nianyi Li  
  _2026-03-20_ · https://arxiv.org/abs/2603.20128v1  
  <details><summary>Abstract</summary>

  Neural Radiance Fields (NeRF) achieve photorealistic novel view synthesis but become costly when high-resolution (HR) rendering is required, as HR outputs demand dense sampling and higher-capacity models. Moreover, naively super-resolving per-view renderings in 2D often breaks multi-view consistency. We propose Generalizable NGP-SR, a 3D-aware super-resolution framework that reconstructs an HR radiance field directly from low-resolution (LR) posed images. Built on Neural Graphics Primitives (NGP), NGP-SR conditions radiance prediction on 3D coordinates and learned local texture tokens, enabling recovery of high-frequency details within the radiance field and producing view-consistent HR novel views without external HR references or post-hoc 2D upsampling. Importantly, our model is generalizable: once trained, it can be applied to unseen scenes and rendered from novel viewpoints without per-scene optimization. Experiments on multiple datasets show that NGP-SR consistently improves both reconstruction quality and runtime efficiency over prior NeRF-based super-resolution methods, offering a practical solution for scalable high-resolution novel view synthesis.

  </details>



- **Fourier Splatting: Generalized Fourier encoded primitives for scalable radiance fields**  
  Mihnea-Bogdan Jurca, Bert Van hauwermeiren, Adrian Munteanu  
  _2026-03-20_ · https://arxiv.org/abs/2603.19834v1  
  <details><summary>Abstract</summary>

  Novel view synthesis has recently been revolutionized by 3D Gaussian Splatting (3DGS), which enables real-time rendering through explicit primitive rasterization. However, existing methods tie visual fidelity strictly to the number of primitives: quality downscaling is achieved only through pruning primitives. We propose the first inherently scalable primitive for radiance field rendering. Fourier Splatting employs scalable primitives with arbitrary closed shapes obtained by parameterizing planar surfels with Fourier encoded descriptors. This formulation allows a single trained model to be rendered at varying levels of detail simply by truncating Fourier coefficients at runtime. To facilitate stable optimization, we employ a straight-through estimator for gradient extension beyond the primitive boundary, and introduce HYDRA, a densification strategy that decomposes complex primitives into simpler constituents within the MCMC framework. Our method achieves state-of-the-art rendering quality among planar-primitive frameworks and comparable perceptual metrics compared to leading volumetric representations on standard benchmarks, providing a versatile solution for bandwidth-constrained high-fidelity rendering.

  </details>


