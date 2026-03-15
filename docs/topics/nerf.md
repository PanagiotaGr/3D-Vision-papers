# NeRF & Neural Radiance Fields

_Updated: 2026-03-15 07:12 UTC_

Total papers shown: **4**


---

- **Node-RF: Learning Generalized Continuous Space-Time Scene Dynamics with Neural ODE-based NeRFs**  
  Hiran Sarkar, Liming Kuang, Yordanka Velikova, Benjamin Busam  
  _2026-03-12_ · https://arxiv.org/abs/2603.12078v1  
  <details><summary>Abstract</summary>

  Predicting scene dynamics from visual observations is challenging. Existing methods capture dynamics only within observed boundaries failing to extrapolate far beyond the training sequence. Node-RF (Neural ODE-based NeRF) overcomes this limitation by integrating Neural Ordinary Differential Equations (NODEs) with dynamic Neural Radiance Fields (NeRFs), enabling a continuous-time, spatiotemporal representation that generalizes beyond observed trajectories at constant memory cost. From visual input, Node-RF learns an implicit scene state that evolves over time via an ODE solver, propagating feature embeddings via differential calculus. A NeRF-based renderer interprets calculated embeddings to synthesize arbitrary views for long-range extrapolation. Training on multiple motion sequences with shared dynamics allows for generalization to unseen conditions. Our experiments demonstrate that Node-RF can characterize abstract system behavior without explicit model to identify critical points for future predictions.

  </details>



- **NBAvatar: Neural Billboards Avatars with Realistic Hand-Face Interaction**  
  David Svitov, Mahtab Dahaghin  
  _2026-03-12_ · https://arxiv.org/abs/2603.12063v1  
  <details><summary>Abstract</summary>

  We present NBAvatar - a method for realistic rendering of head avatars handling non-rigid deformations caused by hand-face interaction. We introduce a novel representation for animated avatars by combining the training of oriented planar primitives with neural rendering. Such a combination of explicit and implicit representations enables NBAvatar to handle temporally and pose-consistent geometry, along with fine-grained appearance details provided by the neural rendering technique. In our experiments, we demonstrate that NBAvatar implicitly learns color transformations caused by face-hand interactions and surpasses existing approaches in terms of novel-view and novel-pose rendering quality. Specifically, NBAvatar achieves up to 30% LPIPS reduction under high-resolution megapixel rendering compared to Gaussian-based avatar methods, while also improving PSNR and SSIM, and achieves higher structural similarity compared to the state-of-the-art hand-face interaction method InteractAvatar.

  </details>



- **AstroSplat: Physics-Based Gaussian Splatting for Rendering and Reconstruction of Small Celestial Bodies**  
  Jennifer Nolan, Travis Driver, John Christian  
  _2026-03-12_ · https://arxiv.org/abs/2603.11969v1  
  <details><summary>Abstract</summary>

  Image-based surface reconstruction and characterization are crucial for missions to small celestial bodies (e.g., asteroids), as it informs mission planning, navigation, and scientific analysis. Recent advances in Gaussian splatting enable high-fidelity neural scene representations but typically rely on a spherical harmonic intensity parameterization that is strictly appearance-based and does not explicitly model material properties or light-surface interactions. We introduce AstroSplat, a physics-based Gaussian splatting framework that integrates planetary reflectance models to improve the autonomous reconstruction and photometric characterization of small-body surfaces from in-situ imagery. The proposed framework is validated on real imagery taken by NASA's Dawn mission, where we demonstrate superior rendering performance and surface reconstruction accuracy compared to the typical spherical harmonic parameterization.

  </details>



- **CEI-3D: Collaborative Explicit-Implicit 3D Reconstruction for Realistic and Fine-Grained Object Editing**  
  Yue Shi, Rui Shi, Yuxuan Xiong, Bingbing Ni, Wenjun Zhang  
  _2026-03-12_ · https://arxiv.org/abs/2603.11810v1  
  <details><summary>Abstract</summary>

  Existing 3D editing methods often produce unrealistic and unrefined results due to the deeply integrated nature of their reconstruction networks. To address the challenge, this paper introduces CEI-3D, an editing-oriented reconstruction pipeline designed to facilitate realistic and fine-grained editing. Specifically, we propose a collaborative explicit-implicit reconstruction approach, which represents the target object using an implicit SDF network and a differentially sampled, locally controllable set of handler points. The implicit network provides a smooth and continuous geometry prior, while the explicit handler points offer localized control, enabling mutual guidance between the global 3D structure and user-specified local editing regions. To independently control each attribute of the handler points, we design a physical properties disentangling module to decouple the color of the handler points into separate physical properties. We also propose a dual-diffuse-albedo network in this module to process the edited and non-edited regions through separate branches, thereby preventing undesired interference from editing operations. Building on the reconstructed collaborative explicit-implicit representation with disentangled properties, we introduce a spatial-aware editing module that enables part-wise adjustment of relevant handler points. This module employs a cross-view propagation-based 3D segmentation strategy, which helps users to edit the specified physical attributes of a target part efficiently. Extensive experiments on both real and synthetic datasets demonstrate that our approach achieves more realistic and fine-grained editing results than the state-of-the-art (SOTA) methods while requiring less editing time. Our code is available on https://github.com/shiyue001/CEI-3D.

  </details>


