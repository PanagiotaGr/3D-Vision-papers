# NeRF & Neural Radiance Fields

_Updated: 2026-03-24 07:17 UTC_

Total papers shown: **11**


---

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



- **Training-Free Instance-Aware 3D Scene Reconstruction and Diffusion-Based View Synthesis from Sparse Images**  
  Jiatong Xia, Lingqiao Liu  
  _2026-03-22_ · https://arxiv.org/abs/2603.21166v1  
  <details><summary>Abstract</summary>

  We introduce a novel, training-free system for reconstructing, understanding, and rendering 3D indoor scenes from a sparse set of unposed RGB images. Unlike traditional radiance field approaches that require dense views and per-scene optimization, our pipeline achieves high-fidelity results without any training or pose preprocessing. The system integrates three key innovations: (1) A robust point cloud reconstruction module that filters unreliable geometry using a warping-based anomaly removal strategy; (2) A warping-guided 2D-to-3D instance lifting mechanism that propagates 2D segmentation masks into a consistent, instance-aware 3D representation; and (3) A novel rendering approach that projects the point cloud into new views and refines the renderings with a 3D-aware diffusion model. Our method leverages the generative power of diffusion to compensate for missing geometry and enhances realism, especially under sparse input conditions. We further demonstrate that object-level scene editing such as instance removal can be naturally supported in our pipeline by modifying only the point cloud, enabling the synthesis of consistent, edited views without retraining. Our results establish a new direction for efficient, editable 3D content generation without relying on scene-specific optimization. Project page: https://jiatongxia.github.io/TID3R/

  </details>



- **SGAD-SLAM: Splatting Gaussians at Adjusted Depth for Better Radiance Fields in RGBD SLAM**  
  Pengchong Hu, Zhizhong Han  
  _2026-03-22_ · https://arxiv.org/abs/2603.21055v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has made remarkable progress in RGBD SLAM. Current methods usually use 3D Gaussians or view-tied 3D Gaussians to represent radiance fields in tracking and mapping. However, these Gaussians are either too flexible or too limited in movements, resulting in slow convergence or limited rendering quality. To resolve this issue, we adopt pixel-aligned Gaussians but allow each Gaussian to adjust its position along its ray to maximize the rendering quality, even if Gaussians are simplified to improve system scalability. To speed up the tracking, we model the depth distribution around each pixel as a Gaussian distribution, and then use these distributions to align each frame to the 3D scene quickly. We report our evaluations on widely used benchmarks, justify our designs, and show advantages over the latest methods in view rendering, camera tracking, runtime, and storage complexity. Please see our project page for code and videos at https://machineperceptionlab.github.io/SGAD-SLAM-Project .

  </details>



- **Fast and Robust Deformable 3D Gaussian Splatting**  
  Han Jiao, Jiakai Sun, Lei Zhao, Zhanjie Zhang, Wei Xing, Huaizhong Lin  
  _2026-03-21_ · https://arxiv.org/abs/2603.20857v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting has demonstrated remarkable real-time rendering capabilities and superior visual quality in novel view synthesis for static scenes. Building upon these advantages, researchers have progressively extended 3D Gaussians to dynamic scene reconstruction. Deformation field-based methods have emerged as a promising approach among various techniques. These methods maintain 3D Gaussian attributes in a canonical field and employ the deformation field to transform this field across temporal sequences. Nevertheless, these approaches frequently encounter challenges such as suboptimal rendering speeds, significant dependence on initial point clouds, and vulnerability to local optima in dim scenes. To overcome these limitations, we present FRoG, an efficient and robust framework for high-quality dynamic scene reconstruction. FRoG integrates per-Gaussian embedding with a coarse-to-fine temporal embedding strategy, accelerating rendering through the early fusion of temporal embeddings. Moreover, to enhance robustness against sparse initializations, we introduce a novel depth- and error-guided sampling strategy. This strategy populates the canonical field with new 3D Gaussians at low-deviation initial positions, significantly reducing the optimization burden on the deformation field and improving detail reconstruction in both static and dynamic regions. Furthermore, by modulating opacity variations, we mitigate the local optima problem in dim scenes, improving color fidelity. Comprehensive experimental results validate that our method achieves accelerated rendering speeds while maintaining state-of-the-art visual quality.

  </details>



- **Satellite-to-Street: Synthesizing Post-Disaster Views from Satellite Imagery via Generative Vision Models**  
  Yifan Yang, Lei Zou, Wendy Jepson  
  _2026-03-21_ · https://arxiv.org/abs/2603.20697v1  
  <details><summary>Abstract</summary>

  In the immediate aftermath of natural disasters, rapid situational awareness is critical. Traditionally, satellite observations are widely used to estimate damage extent. However, they lack the ground-level perspective essential for characterizing specific structural failures and impacts. Meanwhile, ground-level data (e.g., street-view imagery) remains largely inaccessible during time-sensitive events. This study investigates Satellite-to-Street View Synthesis to bridge this data gap. We introduce two generative strategies to synthesize post-disaster street views from satellite imagery: a Vision-Language Model (VLM)-guided approach and a damage-sensitive Mixture-of-Experts (MoE) method. We benchmark these against general-purpose baselines (Pix2Pix, ControlNet) using a proposed Structure-Aware Evaluation Framework. This multi-tier protocol integrates (1) pixel-level quality assessment, (2) ResNet-based semantic consistency verification, and (3) a novel VLM-as-a-Judge for perceptual alignment. Experiments on 300 disaster scenarios reveal a critical realism--fidelity trade-off: while diffusion-based approaches (e.g., ControlNet) achieve high perceptual realism, they often hallucinate structural details. Quantitative results show that standard ControlNet achieves the highest semantic accuracy, 0.71, whereas VLM-enhanced and MoE models excel in textural plausibility but struggle with semantic clarity. This work establishes a baseline for trustworthy cross-view synthesis, emphasizing that visually realistic generations may still fail to preserve critical structural information required for reliable disaster assessment.

  </details>


