# NeRF & Neural Radiance Fields

_Updated: 2026-03-26 07:34 UTC_

Total papers shown: **9**


---

- **SpectralSplats: Robust Differentiable Tracking via Spectral Moment Supervision**  
  Avigail Cohen Rimon, Amir Mann, Mirela Ben Chen, Or Litany  
  _2026-03-25_ · https://arxiv.org/abs/2603.24036v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) enables real-time, photorealistic novel view synthesis, making it a highly attractive representation for model-based video tracking. However, leveraging the differentiability of the 3DGS renderer "in the wild" remains notoriously fragile. A fundamental bottleneck lies in the compact, local support of the Gaussian primitives. Standard photometric objectives implicitly rely on spatial overlap; if severe camera misalignment places the rendered object outside the target's local footprint, gradients strictly vanish, leaving the optimizer stranded. We introduce SpectralSplats, a robust tracking framework that resolves this "vanishing gradient" problem by shifting the optimization objective from the spatial to the frequency domain. By supervising the rendered image via a set of global complex sinusoidal features (Spectral Moments), we construct a global basin of attraction, ensuring that a valid, directional gradient toward the target exists across the entire image domain, even when pixel overlap is completely nonexistent. To harness this global basin without introducing periodic local minima associated with high frequencies, we derive a principled Frequency Annealing schedule from first principles, gracefully transitioning the optimizer from global convexity to precise spatial alignment. We demonstrate that SpectralSplats acts as a seamless, drop-in replacement for spatial losses across diverse deformation parameterizations (from MLPs to sparse control points), successfully recovering complex deformations even from severely misaligned initializations where standard appearance-based tracking catastrophically fails.

  </details>



- **One View Is Enough! Monocular Training for In-the-Wild Novel View Generation**  
  Adrien Ramanana Rahary, Nicolas Dufour, Patrick Perez, David Picard  
  _2026-03-24_ · https://arxiv.org/abs/2603.23488v1  
  <details><summary>Abstract</summary>

  Monocular novel-view synthesis has long required multi-view image pairs for supervision, limiting training data scale and diversity. We argue it is not necessary: one view is enough. We present OVIE, trained entirely on unpaired internet images. We leverage a monocular depth estimator as a geometric scaffold at training time: we lift a source image into 3D, apply a sampled camera transformation, and project to obtain a pseudo-target view. To handle disocclusions, we introduce a masked training formulation that restricts geometric, perceptual, and textural losses to valid regions, enabling training on 30 million uncurated images. At inference, OVIE is geometry-free, requiring no depth estimator or 3D representation. Trained exclusively on in-the-wild images, OVIE outperforms prior methods in a zero-shot setting, while being 600x faster than the second-best baseline. Code and models are publicly available at https://github.com/AdrienRR/ovie.

  </details>



- **I3DM: Implicit 3D-aware Memory Retrieval and Injection for Consistent Video Scene Generation**  
  Jia Li, Han Yan, Yihang Chen, Siqi Li, Xibin Song, Yifu Wang, Jianfei Cai, Tien-Tsin Wong, Pan Ji  
  _2026-03-24_ · https://arxiv.org/abs/2603.23413v1  
  <details><summary>Abstract</summary>

  Despite remarkable progress in video generation, maintaining long-term scene consistency upon revisiting previously explored areas remains challenging. Existing solutions rely either on explicitly constructing 3D geometry, which suffers from error accumulation and scale ambiguity, or on naive camera Field-of-View (FoV) retrieval, which typically fails under complex occlusions. To overcome these limitations, we propose I3DM, a novel implicit 3D-aware memory mechanism for consistent video scene generation that bypasses explicit 3D reconstruction. At the core of our approach is a 3D-aware memory retrieval strategy, which leverages the intermediate features of a pre-trained Feed-Forward Novel View Synthesis (FF-NVS) model to score view relevance, enabling robust retrieval even in highly occluded scenarios. Furthermore, to fully utilize the retrieved historical frames, we introduce a 3D-aligned memory injection module. This module implicitly warps historical content to the target view and adaptively conditions the generation on reliable warping regions, leading to improved revisit consistency and accurate camera control. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches, achieving superior revisit consistency, generation fidelity, and camera control precision.

  </details>



- **Pose-Free Omnidirectional Gaussian Splatting for 360-Degree Videos with Consistent Depth Priors**  
  Chuanqing Zhuang, Xin Lu, Zehui Deng, Zhengda Lu, Yiqun Wang, Junqi Diao, Jun Xiao  
  _2026-03-24_ · https://arxiv.org/abs/2603.23324v1  
  <details><summary>Abstract</summary>

  Omnidirectional 3D Gaussian Splatting with panoramas is a key technique for 3D scene representation, and existing methods typically rely on slow SfM to provide camera poses and sparse points priors. In this work, we propose a pose-free omnidirectional 3DGS method, named PFGS360, that reconstructs 3D Gaussians from unposed omnidirectional videos. To achieve accurate camera pose estimation, we first construct a spherical consistency-aware pose estimation module, which recovers poses by establishing consistent 2D-3D correspondences between the reconstructed Gaussians and the unposed images using Gaussians' internal depth priors. Besides, to enhance the fidelity of novel view synthesis, we introduce a depth-inlier-aware densification module to extract depth inliers and Gaussian outliers with consistent monocular depth priors, enabling efficient Gaussian densification and achieving photorealistic novel view synthesis. The experiments show significant outperformance over existing pose-free and pose-aware 3DGS methods on both real-world and synthetic 360-degree videos. Code is available at https://github.com/zcq15/PFGS360.

  </details>



- **UniQueR: Unified Query-based Feedforward 3D Reconstruction**  
  Chensheng Peng, Quentin Herau, Jiezhi Yang, Yichen Xie, Yihan Hu, Wenzhao Zheng, Matthew Strong, Masayoshi Tomizuka, Wei Zhan  
  _2026-03-24_ · https://arxiv.org/abs/2603.22851v1  
  <details><summary>Abstract</summary>

  We present UniQueR, a unified query-based feedforward framework for efficient and accurate 3D reconstruction from unposed images. Existing feedforward models such as DUSt3R, VGGT, and AnySplat typically predict per-pixel point maps or pixel-aligned Gaussians, which remain fundamentally 2.5D and limited to visible surfaces. In contrast, UniQueR formulates reconstruction as a sparse 3D query inference problem. Our model learns a compact set of 3D anchor points that act as explicit geometric queries, enabling the network to infer scene structure, including geometry in occluded regions--in a single forward pass. Each query encodes spatial and appearance priors directly in global 3D space (instead of per-frame camera space) and spawns a set of 3D Gaussians for differentiable rendering. By leveraging unified query interactions across multi-view features and a decoupled cross-attention design, UniQueR achieves strong geometric expressiveness while substantially reducing memory and computational cost. Experiments on Mip-NeRF 360 and VR-NeRF demonstrate that UniQueR surpasses state-of-the-art feedforward methods in both rendering quality and geometric accuracy, using an order of magnitude fewer primitives than dense alternatives.

  </details>



- **Predictive Photometric Uncertainty in Gaussian Splatting for Novel View Synthesis**  
  Chamuditha Jayanga Galappaththige, Thomas Gottwald, Peter Stehr, Edgar Heinert, Niko Suenderhauf, Dimity Miller, Matthias Rottmann  
  _2026-03-24_ · https://arxiv.org/abs/2603.22786v1  
  <details><summary>Abstract</summary>

  Recent advances in 3D Gaussian Splatting have enabled impressive photorealistic novel view synthesis. However, to transition from a pure rendering engine to a reliable spatial map for autonomous agents and safety-critical applications, knowing where the representation is uncertain is as important as the rendering fidelity itself. We bridge this critical gap by introducing a lightweight, plug-and-play framework for pixel-wise, view-dependent predictive uncertainty estimation. Our post-hoc method formulates uncertainty as a Bayesian-regularized linear least-squares optimization over reconstruction residuals. This architecture-agnostic approach extracts a per-primitive uncertainty channel without modifying the underlying scene representation or degrading baseline visual fidelity. Crucially, we demonstrate that providing this actionable reliability signal successfully translates 3D Gaussian splatting into a trustworthy spatial map, further improving state-of-the-art performance across three critical downstream perception tasks: active view selection, pose-agnostic scene change detection, and pose-agnostic anomaly detection.

  </details>



- **Single-Subject Multi-View MRI Super-Resolution via Implicit Neural Representations**  
  Heejong Kim, Abhishek Thanki, Roel van Herten, Daniel Margolis, Mert R Sabuncu  
  _2026-03-23_ · https://arxiv.org/abs/2603.22627v1  
  <details><summary>Abstract</summary>

  Clinical MRI frequently acquires anisotropic volumes with high in-plane resolution and low through-plane resolution to reduce acquisition time. Multiple orientations are therefore acquired to provide complementary anatomical information. Conventional integration of these views relies on registration followed by interpolation, which can degrade fine structural details. Recent deep learning-based super-resolution (SR) approaches have demonstrated strong performance in enhancing single-view images. However, their clinical reliability is often limited by the need for large-scale training datasets, resulting in increased dependence on cohort-level priors. Self-supervised strategies offer an alternative by learning directly from the target scans. Prior work either neglects the existence of multi-view information or assumes that in-plane information can supervise through-plane reconstruction under the assumption of pre-alignment between images. However, this assumption is rarely satisfied in clinical settings. In this work, we introduce Single-Subject Implicit Multi-View Super-Resolution for MRI (SIMS-MRI), a framework that operates solely on anisotropic multi-view scans from a single patient without requiring pre- or post-processing. Our method combines a multi-resolution hash-encoded implicit representation with learned inter-view alignment to generate a spatially consistent isotropic reconstruction. We validate the SIMS-MRI pipeline on both simulated brain and clinical prostate MRI datasets. Code will be made publicly available for reproducibility: https://github.com/abhshkt/SIMS-MRI

  </details>



- **FullCircle: Effortless 3D Reconstruction from Casual 360$^\circ$ Captures**  
  Yalda Foroutan, Ipek Oztas, Daniel Rebain, Aysegul Dundar, Kwang Moo Yi, Lily Goli, Andrea Tagliasacchi  
  _2026-03-23_ · https://arxiv.org/abs/2603.22572v1  
  <details><summary>Abstract</summary>

  Radiance fields have emerged as powerful tools for 3D scene reconstruction. However, casual capture remains challenging due to the narrow field of view of perspective cameras, which limits viewpoint coverage and feature correspondences necessary for reliable camera calibration and reconstruction. While commercially available 360$^\circ$ cameras offer significantly broader coverage than perspective cameras for the same capture effort, existing 360$^\circ$ reconstruction methods require special capture protocols and pre-processing steps that undermine the promise of radiance fields: effortless workflows to capture and reconstruct 3D scenes. We propose a practical pipeline for reconstructing 3D scenes directly from raw 360$^\circ$ camera captures. We require no special capture protocols or pre-processing, and exhibit robustness to a prevalent source of reconstruction errors: the human operator that is visible in all 360$^\circ$ imagery. To facilitate evaluation, we introduce a multi-tiered dataset of scenes captured as raw dual-fisheye images, establishing a benchmark for robust casual 360$^\circ$ reconstruction. Our method significantly outperforms not only vanilla 3DGS for 360$^\circ$ cameras but also robust perspective baselines when perspective cameras are simulated from the same capture, demonstrating the advantages of 360$^\circ$ capture for casual reconstruction. Additional results are available at: https://theialab.github.io/fullcircle

  </details>



- **Repurposing Geometric Foundation Models for Multi-view Diffusion**  
  Wooseok Jang, Seonghu Jeon, Jisang Han, Jinhyeok Choi, Minkyung Kwon, Seungryong Kim, Saining Xie, Sainan Liu  
  _2026-03-23_ · https://arxiv.org/abs/2603.22275v1  
  <details><summary>Abstract</summary>

  While recent advances in generative latent spaces have driven substantial progress in single-image generation, the optimal latent space for novel view synthesis (NVS) remains largely unexplored. In particular, NVS requires geometrically consistent generation across viewpoints, but existing approaches typically operate in a view-independent VAE latent space. In this paper, we propose Geometric Latent Diffusion (GLD), a framework that repurposes the geometrically consistent feature space of geometric foundation models as the latent space for multi-view diffusion. We show that these features not only support high-fidelity RGB reconstruction but also encode strong cross-view geometric correspondences, providing a well-suited latent space for NVS. Our experiments demonstrate that GLD outperforms both VAE and RAE on 2D image quality and 3D consistency metrics, while accelerating training by more than 4.4x compared to the VAE latent space. Notably, GLD remains competitive with state-of-the-art methods that leverage large-scale text-to-image pretraining, despite training its diffusion model from scratch without such generative pretraining.

  </details>


