# NeRF & Neural Radiance Fields

_Updated: 2026-03-16 07:46 UTC_

Total papers shown: **3**


---

- **NOIR: Neural Operator mapping for Implicit Representations**  
  Sidaty El Hadramy, Nazim Haouchine, Michael Wehrli, Philippe C. Cattin  
  _2026-03-13_ · https://arxiv.org/abs/2603.13118v1  
  <details><summary>Abstract</summary>

  This paper presents NOIR, a framework that reframes core medical imaging tasks as operator learning between continuous function spaces, challenging the prevailing paradigm of discrete grid-based deep learning. Instead of operating on fixed pixel or voxel grids, NOIR embeds discrete medical signals into shared Implicit Neural Representations and learns a Neural Operator that maps between their latent modulations, enabling resolution-independent function-to-function transformations. We evaluate NOIR across multiple 2D and 3D downstream tasks, including segmentation, shape completion, image-to-image translation, and image synthesis, on several public datasets such as Shenzhen, OASIS-4, SkullBreak, fastMRI, as well as an in-house clinical dataset. It achieves competitive performance at native resolution while demonstrating strong robustness to unseen discretizations, and empirically satisfies key theoretical properties of neural operators. The project page is available here: https://github.com/Sidaty1/NOIR-io.

  </details>



- **Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis**  
  Yinuo Jiang, Jun Cheng, Yiran Wang, Cheng Cheng  
  _2026-03-13_ · https://arxiv.org/abs/2603.12903v1  
  <details><summary>Abstract</summary>

  Neural Radiance Fields (NeRF) have shown remarkable success in image novel view synthesis (NVS), inspiring extensions to LiDAR NVS. However, most methods heavily rely on accurate camera poses for scene reconstruction. The sparsity and textureless nature of LiDAR data also present distinct challenges, leading to geometric holes and discontinuous surfaces. To address these issues, we propose SG-NLF, a pose-free LiDAR NeRF framework that integrates spectral information with geometric consistency. Specifically, we design a hybrid representation based on spectral priors to reconstruct smooth geometry. For pose optimization, we construct a confidence-aware graph based on feature compatibility to achieve global alignment. In addition, an adversarial learning strategy is introduced to enforce cross-frame consistency, thereby enhancing reconstruction quality. Comprehensive experiments demonstrate the effectiveness of our framework, especially in challenging low-frequency scenarios. Compared to previous state-of-the-art methods, SG-NLF improves reconstruction quality and pose accuracy by over 35.8% and 68.8%. Our work can provide a novel perspective for LiDAR view synthesis.

  </details>



- **Catalyst4D: High-Fidelity 3D-to-4D Scene Editing via Dynamic Propagation**  
  Shifeng Chen, Yihui Li, Jun Liao, Hongyu Yang, Di Huang  
  _2026-03-13_ · https://arxiv.org/abs/2603.12766v1  
  <details><summary>Abstract</summary>

  Recent advances in 3D scene editing using NeRF and 3DGS enable high-quality static scene editing. In contrast, dynamic scene editing remains challenging, as methods that directly extend 2D diffusion models to 4D often produce motion artifacts, temporal flickering, and inconsistent style propagation. We introduce Catalyst4D, a framework that transfers high-quality 3D edits to dynamic 4D Gaussian scenes while maintaining spatial and temporal coherence. At its core, Anchor-based Motion Guidance (AMG) builds a set of structurally stable and spatially representative anchors from both original and edited Gaussians. These anchors serve as robust region-level references, and their correspondences are established via optimal transport to enable consistent deformation propagation without cross-region interference or motion drift. Complementarily, Color Uncertainty-guided Appearance Refinement (CUAR) preserves temporal appearance consistency by estimating per-Gaussian color uncertainty and selectively refining regions prone to occlusion-induced artifacts. Extensive experiments demonstrate that Catalyst4D achieves temporally stable, high-fidelity dynamic scene editing and outperforms existing methods in both visual quality and motion coherence.

  </details>


