# Gaussian Splatting & 3DGS

_Updated: 2026-03-16 07:46 UTC_

Total papers shown: **2**


---

- **Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting**  
  Yang Chen, Yi Yu, Jiaming He, Yueqi Duan, Zheng Zhu, Yap-Peng Tan  
  _2026-03-13_ · https://arxiv.org/abs/2603.12796v1  
  <details><summary>Abstract</summary>

  Recent advances in 3D Gaussian Splatting (3DGS) deliver high-quality rendering, yet the Gaussian representation exposes a new attack surface, the resource-targeting attack. This attack poisons training images, excessively inducing Gaussian growth to cause resource exhaustion. Although efficiency-oriented methods such as smoothing, thresholding, and pruning have been explored, these spatial-domain strategies operate on visible structures but overlook how stealthy perturbations distort the underlying spectral behaviors of training data. As a result, poisoned inputs introduce abnormal high-frequency amplifications that mislead 3DGS into interpreting noisy patterns as detailed structures, ultimately causing unstable Gaussian overgrowth and degraded scene fidelity. To address this, we propose \textbf{Spectral Defense} in Gaussian and image fields. We first design a 3D frequency filter to selectively prune Gaussians exhibiting abnormally high frequencies. Since natural scenes also contain legitimate high-frequency structures, directly suppressing high frequencies is insufficient, and we further develop a 2D spectral regularization on renderings, distinguishing naturally isotropic frequencies while penalizing anisotropic angular energy to constrain noisy patterns. Experiments show that our defense builds robust, accurate, and secure 3DGS, suppressing overgrowth by up to $5.92\times$, reducing memory by up to $3.66\times$, and improving speed by up to $4.34\times$ under attacks.

  </details>



- **Catalyst4D: High-Fidelity 3D-to-4D Scene Editing via Dynamic Propagation**  
  Shifeng Chen, Yihui Li, Jun Liao, Hongyu Yang, Di Huang  
  _2026-03-13_ · https://arxiv.org/abs/2603.12766v1  
  <details><summary>Abstract</summary>

  Recent advances in 3D scene editing using NeRF and 3DGS enable high-quality static scene editing. In contrast, dynamic scene editing remains challenging, as methods that directly extend 2D diffusion models to 4D often produce motion artifacts, temporal flickering, and inconsistent style propagation. We introduce Catalyst4D, a framework that transfers high-quality 3D edits to dynamic 4D Gaussian scenes while maintaining spatial and temporal coherence. At its core, Anchor-based Motion Guidance (AMG) builds a set of structurally stable and spatially representative anchors from both original and edited Gaussians. These anchors serve as robust region-level references, and their correspondences are established via optimal transport to enable consistent deformation propagation without cross-region interference or motion drift. Complementarily, Color Uncertainty-guided Appearance Refinement (CUAR) preserves temporal appearance consistency by estimating per-Gaussian color uncertainty and selectively refining regions prone to occlusion-induced artifacts. Extensive experiments demonstrate that Catalyst4D achieves temporally stable, high-fidelity dynamic scene editing and outperforms existing methods in both visual quality and motion coherence.

  </details>


