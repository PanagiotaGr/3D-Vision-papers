# NeRF & Neural Radiance Fields

_Updated: 2026-03-25 07:16 UTC_

Total papers shown: **14**


---

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



- **GTSR: Subsurface Scattering Awared 3D Gaussians for Translucent Surface Reconstruction**  
  Youwen Yuan, Xi Zhao  
  _2026-03-23_ · https://arxiv.org/abs/2603.22036v1  
  <details><summary>Abstract</summary>

  Reconstructing translucent objects from multi-view images is a difficult problem. Previously, researchers have used differentiable path tracing and the neural implicit field, which require relatively large computational costs. Recently, many works have achieved good reconstruction results for opaque objects based on a 3DGS pipeline with much higher efficiency. However, such methods have difficulty dealing with translucent objects, because they do not consider the optical properties of translucent objects. In this paper, we propose a novel 3DGS-based pipeline (GTSR) to reconstruct the surface geometry of translucent objects. GTSR combines two sets of Gaussians, surface and interior Gaussians, which are used to model the surface and scattering color when lights pass translucent objects. To render the appearance of translucent objects, we introduce a method that uses the Fresnel term to blend two sets of Gaussians. Furthermore, to improve the reconstructed details of non-contour areas, we introduce the Disney BSDF model with deferred rendering to enhance constraints of the normal and depth. Experimental results demonstrate that our method outperforms baseline reconstruction methods on the NeuralTO Syn dataset while showing great real-time rendering performance. We also extend the dataset with new translucent objects of varying material properties and demonstrate our method can adapt to different translucent materials.

  </details>



- **SatGeo-NeRF: Geometrically Regularized NeRF for Satellite Imagery**  
  Valentin Wagner, Sebastian Bullinger, Michael Arens, Rainer Stiefelhagen  
  _2026-03-23_ · https://arxiv.org/abs/2603.21931v1  
  <details><summary>Abstract</summary>

  We present SatGeo-NeRF, a geometrically regularized NeRF for satellite imagery that mitigates overfitting-induced geometric artifacts observed in current state-of-the-art models using three model-agnostic regularizers. Gravity-Aligned Planarity Regularization aligns depth-inferred, approximated surface normals with the gravity axis to promote local planarity, coupling adjacent rays via a corresponding surface approximation to facilitate cross-ray gradient flow. Granularity Regularization enforces a coarse-to-fine geometry-learning scheme, and Depth-Supervised Regularization stabilizes early training for improved geometric accuracy. On the DFC2019 satellite reconstruction benchmark, SatGeo-NeRF improves the Mean Altitude Error by 13.9% and 11.7% relative to state-of-the-art baselines such as EO-NeRF and EO-GS.

  </details>



- **RefracGS: Novel View Synthesis Through Refractive Water Surfaces with 3D Gaussian Ray Tracing**  
  Yiming Shao, Qiyu Dai, Chong Gao, Guanbin Li, Yeqiang Wang, He Sun, Qiong Zeng, Baoquan Chen, Wenzheng Chen  
  _2026-03-23_ · https://arxiv.org/abs/2603.21695v1  
  <details><summary>Abstract</summary>

  Novel view synthesis (NVS) through non-planar refractive surfaces presents fundamental challenges due to severe, spatially varying optical distortions. While recent representations like NeRF and 3D Gaussian Splatting (3DGS) excel at NVS, their assumption of straight-line ray propagation fails under these conditions, leading to significant artifacts. To overcome this limitation, we introduce RefracGS, a framework that jointly reconstructs the refractive water surface and the scene beneath the interface. Our key insight is to explicitly decouple the refractive boundary from the target objects: the refractive surface is modeled via a neural height field, capturing wave geometry, while the underlying scene is represented as a 3D Gaussian field. We formulate a refraction-aware Gaussian ray tracing approach that accurately computes non-linear ray trajectories using Snell's law and efficiently renders the underlying Gaussian field while backpropagating the loss gradients to the parameterized refractive surface. Through end-to-end joint optimization of both representations, our method ensures high-fidelity NVS and view-consistent surface recovery. Experiments on both synthetic and real-world scenes with complex waves demonstrate that RefracGS outperforms prior refractive methods in visual quality, while achieving 15x faster training and real-time rendering at 200 FPS. The project page for RefracGS is available at https://yimgshao.github.io/refracgs/.

  </details>



- **FluidGaussian: Propagating Simulation-Based Uncertainty Toward Functionally-Intelligent 3D Reconstruction**  
  Yuqiu Liu, Jialin Song, Marissa Ramirez de Chanlatte, Rochishnu Chowdhury, Rushil Paresh Desai, Wuyang Chen, Daniel Martin, Michael Mahoney  
  _2026-03-22_ · https://arxiv.org/abs/2603.21356v1  
  <details><summary>Abstract</summary>

  Real objects that inhabit the physical world follow physical laws and thus behave plausibly during interaction with other physical objects. However, current methods that perform 3D reconstructions of real-world scenes from multi-view 2D images optimize primarily for visual fidelity, i.e., they train with photometric losses and reason about uncertainty in the image or representation space. This appearance-centric view overlooks body contacts and couplings, conflates function-critical regions (e.g., aerodynamic or hydrodynamic surfaces) with ornamentation, and reconstructs structures suboptimally, even when physical regularizers are added. All these can lead to unphysical and implausible interactions. To address this, we consider the question: How can 3D reconstruction become aware of real-world interactions and underlying object functionality, beyond visual cues? To answer this question, we propose FluidGaussian, a plug-and-play method that tightly couples geometry reconstruction with ubiquitous fluid-structure interactions to assess surface quality at high granularity. We define a simulation-based uncertainty metric induced by fluid simulations and integrate it with active learning to prioritize views that improve both visual and physical fidelity. In an empirical evaluation on NeRF Synthetic (Blender), Mip-NeRF 360, and DrivAerNet++, our FluidGaussian method yields up to +8.6% visual PSNR (Peak Signal-to-Noise Ratio) and -62.3% velocity divergence during fluid simulations. Our code is available at https://github.com/delta-lab-ai/FluidGaussian.

  </details>



- **EmoTaG: Emotion-Aware Talking Head Synthesis on Gaussian Splatting with Few-Shot Personalization**  
  Haolan Xu, Keli Cheng, Lei Wang, Ning Bi, Xiaoming Liu  
  _2026-03-22_ · https://arxiv.org/abs/2603.21332v1  
  <details><summary>Abstract</summary>

  Audio-driven 3D talking head synthesis has advanced rapidly with Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). By leveraging rich pre-trained priors, few-shot methods enable instant personalization from just a few seconds of video. However, under expressive facial motion, existing few-shot approaches often suffer from geometric instability and audio-emotion mismatch, highlighting the need for more effective emotion-aware motion modeling. In this work, we present EmoTaG, a few-shot emotion-aware 3D talking head synthesis framework built on the Pretrain-and-Adapt paradigm. Our key insight is to reformulate motion prediction in a structured FLAME parameter space rather than directly deforming 3D Gaussians, thereby introducing explicit geometric priors that improve motion stability. Building upon this, we propose a Gated Residual Motion Network (GRMN), which captures emotional prosody from audio while supplementing head pose and upper-face cues absent from audio, enabling expressive and coherent motion generation. Extensive experiments demonstrate that EmoTaG achieves state-of-the-art performance in emotional expressiveness, lip synchronization, visual realism, and motion stability.

  </details>



- **F4Splat: Feed-Forward Predictive Densification for Feed-Forward 3D Gaussian Splatting**  
  Injae Kim, Chaehyeon Kim, Minseong Bae, Minseok Joo, Hyunwoo J. Kim  
  _2026-03-22_ · https://arxiv.org/abs/2603.21304v1  
  <details><summary>Abstract</summary>

  Feed-forward 3D Gaussian Splatting methods enable single-pass reconstruction and real-time rendering. However, they typically adopt rigid pixel-to-Gaussian or voxel-to-Gaussian pipelines that uniformly allocate Gaussians, leading to redundant Gaussians across views. Moreover, they lack an effective mechanism to control the total number of Gaussians while maintaining reconstruction fidelity. To address these limitations, we present F4Splat, which performs Feed-Forward predictive densification for Feed-Forward 3D Gaussian Splatting, introducing a densification-score-guided allocation strategy that adaptively distributes Gaussians according to spatial complexity and multi-view overlap. Our model predicts per-region densification scores to estimate the required Gaussian density and allows explicit control over the final Gaussian budget without retraining. This spatially adaptive allocation reduces redundancy in simple regions and minimizes duplicate Gaussians across overlapping views, producing compact yet high-quality 3D representations. Extensive experiments demonstrate that our model achieves superior novel-view synthesis performance compared to prior uncalibrated feed-forward methods, while using significantly fewer Gaussians.

  </details>


