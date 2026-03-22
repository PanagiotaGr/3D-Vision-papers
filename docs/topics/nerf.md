# NeRF & Neural Radiance Fields

_Updated: 2026-03-22 07:06 UTC_

Total papers shown: **2**


---

- **Rethinking Vector Field Learning for Generative Segmentation**  
  Chaoyang Wang, Yaobo Liang, Boci Peng, Fan Duan, Jingdong Wang, Yunhai Tong  
  _2026-03-19_ · https://arxiv.org/abs/2603.19218v1  
  <details><summary>Abstract</summary>

  Taming diffusion models for generative segmentation has attracted increasing attention. While existing approaches primarily focus on architectural tweaks or training heuristics, there remains a limited understanding of the intrinsic mismatch between continuous flow matching objectives and discrete perception tasks. In this work, we revisit diffusion segmentation from the perspective of vector field learning. We identify two key limitations of the commonly used flow matching objective: gradient vanishing and trajectory traversing, which result in slow convergence and poor class separation. To tackle these issues, we propose a principled vector field reshaping strategy that augments the learned velocity field with a detached distance-aware correction term. This correction introduces both attractive and repulsive interactions, enhancing gradient magnitudes near centroids while preserving the original diffusion training framework. Furthermore, we design a computationally efficient, quasi-random category encoding scheme inspired by Kronecker sequences, which integrates seamlessly with an end-to-end pixel neural field framework for pixel-level semantic alignment. Extensive experiments consistently demonstrate significant improvements over vanilla flow matching approaches, substantially narrowing the performance gap between generative segmentation and strong discriminative specialists.

  </details>



- **VesselTok: Tokenizing Vessel-like 3D Biomedical Graph Representations for Reconstruction and Generation**  
  Chinmay Prabhakar, Bastian Wittmann, Tamaz Amiranashvili, Paul Büschl, Ezequiel de la Rosa, Julian McGinnis, Benedikt Wiestler, Bjoern Menze, Suprosanna Shit  
  _2026-03-19_ · https://arxiv.org/abs/2603.18797v1  
  <details><summary>Abstract</summary>

  Spatial graphs provide a lightweight and elegant representation of curvilinear anatomical structures such as blood vessels, lung airways, and neuronal networks. Accurately modeling these graphs is crucial in clinical and (bio-)medical research. However, the high spatial resolution of large networks drastically increases their complexity, resulting in significant computational challenges. In this work, we aim to tackle these challenges by proposing VesselTok, a framework that approaches spatially dense graphs from a parametric shape perspective to learn latent representations (tokens). VesselTok leverages centerline points with a pseudo radius to effectively encode tubular geometry. Specifically, we learn a novel latent representation conditioned on centerline points to encode neural implicit representations of vessel-like, tubular structures. We demonstrate VesselTok's performance across diverse anatomies, including lung airways, lung vessels, and brain vessels, highlighting its ability to robustly encode complex topologies. To prove the effectiveness of VesselTok's learnt latent representations, we show that they (i) generalize to unseen anatomies, (ii) support generative modeling of plausible anatomical graphs, and (iii) transfer effectively to downstream inverse problems, such as link prediction.

  </details>


