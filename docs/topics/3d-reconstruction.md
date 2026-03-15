# 3D Reconstruction

_Updated: 2026-03-15 07:12 UTC_

Total papers shown: **5**


---

- **DVD: Deterministic Video Depth Estimation with Generative Priors**  
  Hongfei Zhang, Harold Haodong Chen, Chenfei Liao, Jing He, Zixin Zhang, Haodong Li, Yihao Liang, Kanghao Chen, Bin Ren, Xu Zheng, et al.  
  _2026-03-12_ · https://arxiv.org/abs/2603.12250v1  
  <details><summary>Abstract</summary>

  Existing video depth estimation faces a fundamental trade-off: generative models suffer from stochastic geometric hallucinations and scale drift, while discriminative models demand massive labeled datasets to resolve semantic ambiguities. To break this impasse, we present DVD, the first framework to deterministically adapt pre-trained video diffusion models into single-pass depth regressors. Specifically, DVD features three core designs: (i) repurposing the diffusion timestep as a structural anchor to balance global stability with high-frequency details; (ii) latent manifold rectification (LMR) to mitigate regression-induced over-smoothing, enforcing differential constraints to restore sharp boundaries and coherent motion; and (iii) global affine coherence, an inherent property bounding inter-window divergence, which enables seamless long-video inference without requiring complex temporal alignment. Extensive experiments demonstrate that DVD achieves state-of-the-art zero-shot performance across benchmarks. Furthermore, DVD successfully unlocks the profound geometric priors implicit in video foundation models using 163x less task-specific data than leading baselines. Notably, we fully release our pipeline, providing the whole training suite for SOTA video depth estimation to benefit the open-source community.

  </details>



- **Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation**  
  Chongyang Xu, Yixian Zou, Ziliang Feng, Fanman Meng, Shuaicheng Liu  
  _2026-03-12_ · https://arxiv.org/abs/2603.11984v1  
  <details><summary>Abstract</summary>

  Diffusion-based visuomotor policies effectively capture multimodal action distributions through iterative denoising, but their high inference latency limits real-time robotic control. Recent flow matching and consistency-based methods achieve single-step generation, yet sacrifice the ability to preserve distinct action modes, collapsing multimodal behaviors into averaged, often physically infeasible trajectories. We observe that the compute budget asymmetry in robotics (offline training vs.\ real-time inference) naturally motivates recovering this multimodal fidelity by shifting iterative refinement from inference time to training time. Building on this insight, we propose Ada3Drift, which learns a training-time drifting field that attracts predicted actions toward expert demonstration modes while repelling them from other generated samples, enabling high-fidelity single-step generation (1 NFE) from 3D point cloud observations. To handle the few-shot robotic regime, Ada3Drift further introduces a sigmoid-scheduled loss transition from coarse distribution learning to mode-sharpening refinement, and multi-scale field aggregation that captures action modes at varying spatial granularities. Experiments on three simulation benchmarks (Adroit, Meta-World, and RoboTwin) and real-world robotic manipulation tasks demonstrate that Ada3Drift achieves state-of-the-art performance while requiring $10\times$ fewer function evaluations than diffusion-based alternatives.

  </details>



- **AstroSplat: Physics-Based Gaussian Splatting for Rendering and Reconstruction of Small Celestial Bodies**  
  Jennifer Nolan, Travis Driver, John Christian  
  _2026-03-12_ · https://arxiv.org/abs/2603.11969v1  
  <details><summary>Abstract</summary>

  Image-based surface reconstruction and characterization are crucial for missions to small celestial bodies (e.g., asteroids), as it informs mission planning, navigation, and scientific analysis. Recent advances in Gaussian splatting enable high-fidelity neural scene representations but typically rely on a spherical harmonic intensity parameterization that is strictly appearance-based and does not explicitly model material properties or light-surface interactions. We introduce AstroSplat, a physics-based Gaussian splatting framework that integrates planetary reflectance models to improve the autonomous reconstruction and photometric characterization of small-body surfaces from in-situ imagery. The proposed framework is validated on real imagery taken by NASA's Dawn mission, where we demonstrate superior rendering performance and surface reconstruction accuracy compared to the typical spherical harmonic parameterization.

  </details>



- **Single-View Rolling-Shutter SfM**  
  Sofía Errázuriz Muñoz, Kim Kiehn, Petr Hruby, Kathlén Kohn  
  _2026-03-12_ · https://arxiv.org/abs/2603.11888v1  
  <details><summary>Abstract</summary>

  Rolling-shutter (RS) cameras are ubiquitous, but RS SfM (structure-from-motion) has not been fully solved yet. This work suggests an approach to remedy this: We characterize RS single-view geometry of observed world points or lines. Exploiting this geometry, we describe which motion and scene parameters can be recovered from a single RS image and systematically derive minimal reconstruction problems. We evaluate several representative cases with proof-of-concept solvers, highlighting both feasibility and practical limitations.

  </details>



- **CEI-3D: Collaborative Explicit-Implicit 3D Reconstruction for Realistic and Fine-Grained Object Editing**  
  Yue Shi, Rui Shi, Yuxuan Xiong, Bingbing Ni, Wenjun Zhang  
  _2026-03-12_ · https://arxiv.org/abs/2603.11810v1  
  <details><summary>Abstract</summary>

  Existing 3D editing methods often produce unrealistic and unrefined results due to the deeply integrated nature of their reconstruction networks. To address the challenge, this paper introduces CEI-3D, an editing-oriented reconstruction pipeline designed to facilitate realistic and fine-grained editing. Specifically, we propose a collaborative explicit-implicit reconstruction approach, which represents the target object using an implicit SDF network and a differentially sampled, locally controllable set of handler points. The implicit network provides a smooth and continuous geometry prior, while the explicit handler points offer localized control, enabling mutual guidance between the global 3D structure and user-specified local editing regions. To independently control each attribute of the handler points, we design a physical properties disentangling module to decouple the color of the handler points into separate physical properties. We also propose a dual-diffuse-albedo network in this module to process the edited and non-edited regions through separate branches, thereby preventing undesired interference from editing operations. Building on the reconstructed collaborative explicit-implicit representation with disentangled properties, we introduce a spatial-aware editing module that enables part-wise adjustment of relevant handler points. This module employs a cross-view propagation-based 3D segmentation strategy, which helps users to edit the specified physical attributes of a target part efficiently. Extensive experiments on both real and synthetic datasets demonstrate that our approach achieves more realistic and fine-grained editing results than the state-of-the-art (SOTA) methods while requiring less editing time. Our code is available on https://github.com/shiyue001/CEI-3D.

  </details>


