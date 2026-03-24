# Neural Rendering & View Synthesis

_Updated: 2026-03-24 07:17 UTC_

Total papers shown: **8**


---

- **Repurposing Geometric Foundation Models for Multi-view Diffusion**  
  Wooseok Jang, Seonghu Jeon, Jisang Han, Jinhyeok Choi, Minkyung Kwon, Seungryong Kim, Saining Xie, Sainan Liu  
  _2026-03-23_ · https://arxiv.org/abs/2603.22275v1  
  <details><summary>Abstract</summary>

  While recent advances in generative latent spaces have driven substantial progress in single-image generation, the optimal latent space for novel view synthesis (NVS) remains largely unexplored. In particular, NVS requires geometrically consistent generation across viewpoints, but existing approaches typically operate in a view-independent VAE latent space. In this paper, we propose Geometric Latent Diffusion (GLD), a framework that repurposes the geometrically consistent feature space of geometric foundation models as the latent space for multi-view diffusion. We show that these features not only support high-fidelity RGB reconstruction but also encode strong cross-view geometric correspondences, providing a well-suited latent space for NVS. Our experiments demonstrate that GLD outperforms both VAE and RAE on 2D image quality and 3D consistency metrics, while accelerating training by more than 4.4x compared to the VAE latent space. Notably, GLD remains competitive with state-of-the-art methods that leverage large-scale text-to-image pretraining, despite training its diffusion model from scratch without such generative pretraining.

  </details>



- **RefracGS: Novel View Synthesis Through Refractive Water Surfaces with 3D Gaussian Ray Tracing**  
  Yiming Shao, Qiyu Dai, Chong Gao, Guanbin Li, Yeqiang Wang, He Sun, Qiong Zeng, Baoquan Chen, Wenzheng Chen  
  _2026-03-23_ · https://arxiv.org/abs/2603.21695v1  
  <details><summary>Abstract</summary>

  Novel view synthesis (NVS) through non-planar refractive surfaces presents fundamental challenges due to severe, spatially varying optical distortions. While recent representations like NeRF and 3D Gaussian Splatting (3DGS) excel at NVS, their assumption of straight-line ray propagation fails under these conditions, leading to significant artifacts. To overcome this limitation, we introduce RefracGS, a framework that jointly reconstructs the refractive water surface and the scene beneath the interface. Our key insight is to explicitly decouple the refractive boundary from the target objects: the refractive surface is modeled via a neural height field, capturing wave geometry, while the underlying scene is represented as a 3D Gaussian field. We formulate a refraction-aware Gaussian ray tracing approach that accurately computes non-linear ray trajectories using Snell's law and efficiently renders the underlying Gaussian field while backpropagating the loss gradients to the parameterized refractive surface. Through end-to-end joint optimization of both representations, our method ensures high-fidelity NVS and view-consistent surface recovery. Experiments on both synthetic and real-world scenes with complex waves demonstrate that RefracGS outperforms prior refractive methods in visual quality, while achieving 15x faster training and real-time rendering at 200 FPS. The project page for RefracGS is available at https://yimgshao.github.io/refracgs/.

  </details>



- **PAS3R: Pose-Adaptive Streaming 3D Reconstruction for Long Video Sequences**  
  Lanbo Xu, Liang Guo, Caigui Jiang, Cheng Wang  
  _2026-03-22_ · https://arxiv.org/abs/2603.21436v1  
  <details><summary>Abstract</summary>

  Online monocular 3D reconstruction enables dense scene recovery from streaming video but remains fundamentally limited by the stability-adaptation dilemma: the reconstruction model must rapidly incorporate novel viewpoints while preserving previously accumulated scene structure. Existing streaming approaches rely on uniform or attention-based update mechanisms that often fail to account for abrupt viewpoint transitions, leading to trajectory drift and geometric inconsistencies over long sequences. We introduce PAS3R, a pose-adaptive streaming reconstruction framework that dynamically modulates state updates according to camera motion and scene structure. Our key insight is that frames contributing significant geometric novelty should exert stronger influence on the reconstruction state, while frames with minor viewpoint variation should prioritize preserving historical context. PAS3R operationalizes this principle through a motion-aware update mechanism that jointly leverages inter-frame pose variation and image frequency cues to estimate frame importance. To further stabilize long-horizon reconstruction, we introduce trajectory-consistent training objectives that incorporate relative pose constraints and acceleration regularization. A lightweight online stabilization module further suppresses high-frequency trajectory jitter and geometric artifacts without increasing memory consumption. Extensive experiments across multiple benchmarks demonstrate that PAS3R significantly improves trajectory accuracy, depth estimation, and point cloud reconstruction quality in long video sequences while maintaining competitive performance on shorter sequences.

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



- **Two Experts Are Better Than One Generalist: Decoupling Geometry and Appearance for Feed-Forward 3D Gaussian Splatting**  
  Hwasik Jeong, Seungryong Lee, Gyeongjin Kang, Seungkwon Yang, Xiangyu Sun, Seungtae Nam, Eunbyung Park  
  _2026-03-22_ · https://arxiv.org/abs/2603.21064v1  
  <details><summary>Abstract</summary>

  Pose-free feed-forward 3D Gaussian Splatting (3DGS) has opened a new frontier for rapid 3D modeling, enabling high-quality Gaussian representations to be generated from uncalibrated multi-view images in a single forward pass. The dominant approach in this space adopts unified monolithic architectures, often built on geometry-centric 3D foundation models, to jointly estimate camera poses and synthesize 3DGS representations within a single network. While architecturally streamlined, such "all-in-one" designs may be suboptimal for high-fidelity 3DGS generation, as they entangle geometric reasoning and appearance modeling within a shared representation. In this work, we introduce 2Xplat, a pose-free feed-forward 3DGS framework based on a two-expert design that explicitly separates geometry estimation from Gaussian generation. A dedicated geometry expert first predicts camera poses, which are then explicitly passed to a powerful appearance expert that synthesizes 3D Gaussians. Despite its conceptual simplicity, being largely underexplored in prior works, the proposed approach proves highly effective. In fewer than 5K training iterations, the proposed two-experts pipeline substantially outperforms prior pose-free feed-forward 3DGS approaches and achieves performance on par with state-of-the-art posed methods. These results challenge the prevailing unified paradigm and suggest the potential advantages of modular design principles for complex 3D geometric estimation and appearance synthesis tasks.

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


