# Gaussian Splatting & 3DGS

_Updated: 2026-03-25 07:16 UTC_

Total papers shown: **17**


---

- **FHAvatar: Fast and High-Fidelity Reconstruction of Face-and-Hair Composable 3D Head Avatar from Few Casual Captures**  
  Yujie Sun, Zhuoqiang Cai, Chaoyue Niu, Jianchuan Chen, Zhiwen Chen, Chengfei Lv, Fan Wu  
  _2026-03-24_ · https://arxiv.org/abs/2603.23345v1  
  <details><summary>Abstract</summary>

  We present FHAvatar, a novel framework for reconstructing 3D Gaussian avatars with composable face and hair components from an arbitrary number of views. Unlike previous approaches that couple facial and hair representations within a unified modeling process, we explicitly decouple two components in texture space by representing the face with planar Gaussians and the hair with strand-based Gaussians. To overcome the limitations of existing methods that rely on dense multi-view captures or costly per-identity optimization, we propose an aggregated transformer backbone to learn geometry-aware cross-view priors and head-hair structural coherence from multi-view datasets, enabling effective and efficient feature extraction and fusion from few casual captures. Extensive quantitative and qualitative experiments demonstrate that FHAvatar achieves state-of-the-art reconstruction quality from only a few observations of new identities within minutes, while supporting real-time animation, convenient hairstyle transfer, and stylized editing, broadening the accessibility and applicability of digital avatar creation.

  </details>



- **Pose-Free Omnidirectional Gaussian Splatting for 360-Degree Videos with Consistent Depth Priors**  
  Chuanqing Zhuang, Xin Lu, Zehui Deng, Zhengda Lu, Yiqun Wang, Junqi Diao, Jun Xiao  
  _2026-03-24_ · https://arxiv.org/abs/2603.23324v1  
  <details><summary>Abstract</summary>

  Omnidirectional 3D Gaussian Splatting with panoramas is a key technique for 3D scene representation, and existing methods typically rely on slow SfM to provide camera poses and sparse points priors. In this work, we propose a pose-free omnidirectional 3DGS method, named PFGS360, that reconstructs 3D Gaussians from unposed omnidirectional videos. To achieve accurate camera pose estimation, we first construct a spherical consistency-aware pose estimation module, which recovers poses by establishing consistent 2D-3D correspondences between the reconstructed Gaussians and the unposed images using Gaussians' internal depth priors. Besides, to enhance the fidelity of novel view synthesis, we introduce a depth-inlier-aware densification module to extract depth inliers and Gaussian outliers with consistent monocular depth priors, enabling efficient Gaussian densification and achieving photorealistic novel view synthesis. The experiments show significant outperformance over existing pose-free and pose-aware 3DGS methods on both real-world and synthetic 360-degree videos. Code is available at https://github.com/zcq15/PFGS360.

  </details>



- **GTLR-GS: Geometry-Texture Aware LiDAR-Regularized 3D Gaussian Splatting for Realistic Scene Reconstruction**  
  Yan Fang, Jianfei Ge, Jiangjian Xiao  
  _2026-03-24_ · https://arxiv.org/abs/2603.23192v1  
  <details><summary>Abstract</summary>

  Recent advances in 3D Gaussian Splatting (3DGS) have enabled real-time, photorealistic scene reconstruction. However, conventional 3DGS frameworks typically rely on sparse point clouds derived from Structure-from-Motion (SfM), which inherently suffer from scale ambiguity, limited geometric consistency, and strong view dependency due to the lack of geometric priors. In this work, a LiDAR-centric 3D Gaussian Splatting framework is proposed that explicitly incorporates metric geometric priors into the entire Gaussian optimization process. Instead of treating LiDAR data as a passive initialization source, 3DGS optimization is reformulated as a geometry-conditioned allocation and refinement problem under a fixed representational budget. Specifically, this work introduces (i) a geometry-texture-aware allocation strategy that selectively assigns Gaussian primitives to regions with high structural or appearance complexity, (ii) a curvature-adaptive refinement mechanism that dynamically guides Gaussian splitting toward geometrically complex areas during training, and (iii) a confidence-aware metric depth regularization that anchors the reconstructed geometry to absolute scale using LiDAR measurements while maintaining optimization stability. Extensive experiments on the ScanNet++ dataset and a custom real-world dataset validate the proposed approach. The results demonstrate state-of-the-art performance in metric-scale reconstruction with high geometric fidelity.

  </details>



- **GSwap: Realistic Head Swapping with Dynamic Neural Gaussian Field**  
  Jingtao Zhou, Xuan Gao, Dongyu Liu, Junhui Hou, Yudong Guo, Juyong Zhang  
  _2026-03-24_ · https://arxiv.org/abs/2603.23168v1  
  <details><summary>Abstract</summary>

  We present GSwap, a novel consistent and realistic video head-swapping system empowered by dynamic neural Gaussian portrait priors, which significantly advances the state of the art in face and head replacement. Unlike previous methods that rely primarily on 2D generative models or 3D Morphable Face Models (3DMM), our approach overcomes their inherent limitations, including poor 3D consistency, unnatural facial expressions, and restricted synthesis quality. Moreover, existing techniques struggle with full head-swapping tasks due to insufficient holistic head modeling and ineffective background blending, often resulting in visible artifacts and misalignments. To address these challenges, GSwap introduces an intrinsic 3D Gaussian feature field embedded within a full-body SMPL-X surface, effectively elevating 2D portrait videos into a dynamic neural Gaussian field. This innovation ensures high-fidelity, 3D-consistent portrait rendering while preserving natural head-torso relationships and seamless motion dynamics. To facilitate training, we adapt a pretrained 2D portrait generative model to the source head domain using only a few reference images, enabling efficient domain adaptation. Furthermore, we propose a neural re-rendering strategy that harmoniously integrates the synthesized foreground with the original background, eliminating blending artifacts and enhancing realism. Extensive experiments demonstrate that GSwap surpasses existing methods in multiple aspects, including visual quality, temporal coherence, identity preservation, and 3D consistency.

  </details>



- **Gau-Occ: Geometry-Completed Gaussians for Multi-Modal 3D Occupancy Prediction**  
  Chengxin Lv, Yihui Li, Hongyu Yang, YunHong Wang  
  _2026-03-24_ · https://arxiv.org/abs/2603.22852v1  
  <details><summary>Abstract</summary>

  3D semantic occupancy prediction is crucial for autonomous driving. While multi-modal fusion improves accuracy over vision-only methods, it typically relies on computationally expensive dense voxel or BEV tensors. We present Gau-Occ, a multi-modal framework that bypasses dense volumetric processing by modeling the scene as a compact collection of semantic 3D Gaussians. To ensure geometric completeness, we propose a LiDAR Completion Diffuser (LCD) that recovers missing structures from sparse LiDAR to initialize robust Gaussian anchors. Furthermore, we introduce Gaussian Anchor Fusion (GAF), which efficiently integrates multi-view image semantics via geometry-aligned 2D sampling and cross-modal alignment. By refining these compact Gaussian descriptors, Gau-Occ captures both spatial consistency and semantic discriminability. Extensive experiments across challenging benchmarks demonstrate that Gau-Occ achieves state-of-the-art performance with significant computational efficiency.

  </details>



- **UniQueR: Unified Query-based Feedforward 3D Reconstruction**  
  Chensheng Peng, Quentin Herau, Jiezhi Yang, Yichen Xie, Yihan Hu, Wenzhao Zheng, Matthew Strong, Masayoshi Tomizuka, Wei Zhan  
  _2026-03-24_ · https://arxiv.org/abs/2603.22851v1  
  <details><summary>Abstract</summary>

  We present UniQueR, a unified query-based feedforward framework for efficient and accurate 3D reconstruction from unposed images. Existing feedforward models such as DUSt3R, VGGT, and AnySplat typically predict per-pixel point maps or pixel-aligned Gaussians, which remain fundamentally 2.5D and limited to visible surfaces. In contrast, UniQueR formulates reconstruction as a sparse 3D query inference problem. Our model learns a compact set of 3D anchor points that act as explicit geometric queries, enabling the network to infer scene structure, including geometry in occluded regions--in a single forward pass. Each query encodes spatial and appearance priors directly in global 3D space (instead of per-frame camera space) and spawns a set of 3D Gaussians for differentiable rendering. By leveraging unified query interactions across multi-view features and a decoupled cross-attention design, UniQueR achieves strong geometric expressiveness while substantially reducing memory and computational cost. Experiments on Mip-NeRF 360 and VR-NeRF demonstrate that UniQueR surpasses state-of-the-art feedforward methods in both rendering quality and geometric accuracy, using an order of magnitude fewer primitives than dense alternatives.

  </details>



- **PhotoAgent: A Robotic Photographer with Spatial and Aesthetic Understanding**  
  Lirong Che, Zhenfeng Gan, Yanbo Chen, Junbo Tan, Xueqian Wang  
  _2026-03-24_ · https://arxiv.org/abs/2603.22796v1  
  <details><summary>Abstract</summary>

  Embodied agents for creative tasks like photography must bridge the semantic gap between high-level language commands and geometric control. We introduce PhotoAgent, an agent that achieves this by integrating Large Multimodal Models (LMMs) reasoning with a novel control paradigm. PhotoAgent first translates subjective aesthetic goals into solvable geometric constraints via LMM-driven, chain-of-thought (CoT) reasoning, allowing an analytical solver to compute a high-quality initial viewpoint. This initial pose is then iteratively refined through visual reflection within a photorealistic internal world model built with 3D Gaussian Splatting (3DGS). This ``mental simulation'' replaces costly and slow physical trial-and-error, enabling rapid convergence to aesthetically superior results. Evaluations confirm that PhotoAgent excels in spatial reasoning and achieves superior final image quality.

  </details>



- **Predictive Photometric Uncertainty in Gaussian Splatting for Novel View Synthesis**  
  Chamuditha Jayanga Galappaththige, Thomas Gottwald, Peter Stehr, Edgar Heinert, Niko Suenderhauf, Dimity Miller, Matthias Rottmann  
  _2026-03-24_ · https://arxiv.org/abs/2603.22786v1  
  <details><summary>Abstract</summary>

  Recent advances in 3D Gaussian Splatting have enabled impressive photorealistic novel view synthesis. However, to transition from a pure rendering engine to a reliable spatial map for autonomous agents and safety-critical applications, knowing where the representation is uncertain is as important as the rendering fidelity itself. We bridge this critical gap by introducing a lightweight, plug-and-play framework for pixel-wise, view-dependent predictive uncertainty estimation. Our post-hoc method formulates uncertainty as a Bayesian-regularized linear least-squares optimization over reconstruction residuals. This architecture-agnostic approach extracts a per-primitive uncertainty channel without modifying the underlying scene representation or degrading baseline visual fidelity. Crucially, we demonstrate that providing this actionable reliability signal successfully translates 3D Gaussian splatting into a trustworthy spatial map, further improving state-of-the-art performance across three critical downstream perception tasks: active view selection, pose-agnostic scene change detection, and pose-agnostic anomaly detection.

  </details>



- **FullCircle: Effortless 3D Reconstruction from Casual 360$^\circ$ Captures**  
  Yalda Foroutan, Ipek Oztas, Daniel Rebain, Aysegul Dundar, Kwang Moo Yi, Lily Goli, Andrea Tagliasacchi  
  _2026-03-23_ · https://arxiv.org/abs/2603.22572v1  
  <details><summary>Abstract</summary>

  Radiance fields have emerged as powerful tools for 3D scene reconstruction. However, casual capture remains challenging due to the narrow field of view of perspective cameras, which limits viewpoint coverage and feature correspondences necessary for reliable camera calibration and reconstruction. While commercially available 360$^\circ$ cameras offer significantly broader coverage than perspective cameras for the same capture effort, existing 360$^\circ$ reconstruction methods require special capture protocols and pre-processing steps that undermine the promise of radiance fields: effortless workflows to capture and reconstruct 3D scenes. We propose a practical pipeline for reconstructing 3D scenes directly from raw 360$^\circ$ camera captures. We require no special capture protocols or pre-processing, and exhibit robustness to a prevalent source of reconstruction errors: the human operator that is visible in all 360$^\circ$ imagery. To facilitate evaluation, we introduce a multi-tiered dataset of scenes captured as raw dual-fisheye images, establishing a benchmark for robust casual 360$^\circ$ reconstruction. Our method significantly outperforms not only vanilla 3DGS for 360$^\circ$ cameras but also robust perspective baselines when perspective cameras are simulated from the same capture, demonstrating the advantages of 360$^\circ$ capture for casual reconstruction. Additional results are available at: https://theialab.github.io/fullcircle

  </details>



- **Drop-In Perceptual Optimization for 3D Gaussian Splatting**  
  Ezgi Ozyilkan, Zhiqi Chen, Oren Rippel, Jona Ballé, Kedar Tatwawadi  
  _2026-03-23_ · https://arxiv.org/abs/2603.23297v1  
  <details><summary>Abstract</summary>

  Despite their output being ultimately consumed by human viewers, 3D Gaussian Splatting (3DGS) methods often rely on ad-hoc combinations of pixel-level losses, resulting in blurry renderings. To address this, we systematically explore perceptual optimization strategies for 3DGS by searching over a diverse set of distortion losses. We conduct the first-of-its-kind large-scale human subjective study on 3DGS, involving 39,320 pairwise ratings across several datasets and 3DGS frameworks. A regularized version of Wasserstein Distortion, which we call WD-R, emerges as the clear winner, excelling at recovering fine textures without incurring a higher splat count. WD-R is preferred by raters more than $2.3\times$ over the original 3DGS loss, and $1.5\times$ over current best method Perceptual-GS. WD-R also consistently achieves state-of-the-art LPIPS, DISTS, and FID scores across various datasets, and generalizes across recent frameworks, such as Mip-Splatting and Scaffold-GS, where replacing the original loss with WD-R consistently enhances perceptual quality within a similar resource budget (number of splats for Mip-Splatting, model size for Scaffold-GS), and leads to reconstructions being preferred by human raters $1.8\times$ and $3.6\times$, respectively. We also find that this carries over to the task of 3DGS scene compression, with $\approx 50\%$ bitrate savings for comparable perceptual metric performance.

  </details>



- **FreeArtGS: Articulated Gaussian Splatting Under Free-moving Scenario**  
  Hang Dai, Hongwei Fan, Han Zhang, Duojin Wu, Jiyao Zhang, Hao Dong  
  _2026-03-23_ · https://arxiv.org/abs/2603.22102v1  
  <details><summary>Abstract</summary>

  The increasing demand for augmented reality and robotics is driving the need for articulated object reconstruction with high scalability. However, existing settings for reconstructing from discrete articulation states or casual monocular videos require non-trivial axis alignment or suffer from insufficient coverage, limiting their applicability. In this paper, we introduce FreeArtGS, a novel method for reconstructing articulated objects under free-moving scenario, a new setting with a simple setup and high scalability. FreeArtGS combines free-moving part segmentation with joint estimation and end-to-end optimization, taking only a monocular RGB-D video as input. By optimizing with the priors from off-the-shelf point-tracking and feature models, the free-moving part segmentation module identifies rigid parts from relative motion under unconstrained capture. The joint estimation module calibrates the unified object-to-camera poses and recovers joint type and axis robustly from part segmentation. Finally, 3DGS-based end-to-end optimization is implemented to jointly reconstruct visual textures, geometry, and joint angles of the articulated object. We conduct experiments on two benchmarks and real-world free-moving articulated objects. Experimental results demonstrate that FreeArtGS consistently excels in reconstructing free-moving articulated objects and remains highly competitive in previous reconstruction settings, proving itself a practical and effective solution for realistic asset generation. The project page is available at: https://freeartgs.github.io/

  </details>



- **GTSR: Subsurface Scattering Awared 3D Gaussians for Translucent Surface Reconstruction**  
  Youwen Yuan, Xi Zhao  
  _2026-03-23_ · https://arxiv.org/abs/2603.22036v1  
  <details><summary>Abstract</summary>

  Reconstructing translucent objects from multi-view images is a difficult problem. Previously, researchers have used differentiable path tracing and the neural implicit field, which require relatively large computational costs. Recently, many works have achieved good reconstruction results for opaque objects based on a 3DGS pipeline with much higher efficiency. However, such methods have difficulty dealing with translucent objects, because they do not consider the optical properties of translucent objects. In this paper, we propose a novel 3DGS-based pipeline (GTSR) to reconstruct the surface geometry of translucent objects. GTSR combines two sets of Gaussians, surface and interior Gaussians, which are used to model the surface and scattering color when lights pass translucent objects. To render the appearance of translucent objects, we introduce a method that uses the Fresnel term to blend two sets of Gaussians. Furthermore, to improve the reconstructed details of non-contour areas, we introduce the Disney BSDF model with deferred rendering to enhance constraints of the normal and depth. Experimental results demonstrate that our method outperforms baseline reconstruction methods on the NeuralTO Syn dataset while showing great real-time rendering performance. We also extend the dataset with new translucent objects of varying material properties and demonstrate our method can adapt to different translucent materials.

  </details>



- **Cross-Instance Gaussian Splatting Registration via Geometry-Aware Feature-Guided Alignment**  
  Roy Amoyal, Oren Freifeld, Chaim Baskin  
  _2026-03-23_ · https://arxiv.org/abs/2603.21936v1  
  <details><summary>Abstract</summary>

  We present Gaussian Splatting Alignment (GSA), a novel method for aligning two independent 3D Gaussian Splatting (3DGS) models via a similarity transformation (rotation, translation, and scale), even when they are of different objects in the same category (e.g., different cars). In contrast, existing methods can only align 3DGS models of the same object (e.g., the same car) and often must be given true scale as input, while we estimate it successfully. GSA leverages viewpoint-guided spherical map features to obtain robust correspondences and introduces a two-step optimization framework that aligns 3DGS models while keeping them fixed. First, we apply an iterative feature-guided absolute orientation solver as our coarse registration, which is robust to poor initialization (e.g., 180 degrees misalignment or a 10x scale gap). Next, we use a fine registration step that enforces multi-view feature consistency, inspired by inverse radiance-field formulations. The first step already achieves state-of-the-art performance, and the second further improves results. In the same-object case, GSA outperforms prior works, often by a large margin, even when the other methods are given the true scale. In the harder case of different objects in the same category, GSA vastly surpasses them, providing the first effective solution for category-level 3DGS registration and unlocking new applications. Project webpage: https://bgu-cs-vil.github.io/GSA-project/

  </details>



- **Camera-Agnostic Pruning of 3D Gaussian Splats via Descriptor-Based Beta Evidence**  
  Peter Fasogbon, Ugurcan Budak, Patrice Rondao Alface, Hamed Rezazadegan Tavakoli  
  _2026-03-23_ · https://arxiv.org/abs/2603.21933v1  
  <details><summary>Abstract</summary>

  The pruning of 3D Gaussian splats is essential for reducing their complexity to enable efficient storage, transmission, and downstream processing. However, most of the existing pruning strategies depend on camera parameters, rendered images, or view-dependent measures. This dependency becomes a hindrance in emerging camera-agnostic exchange settings, where splats are shared directly as point-based representations (e.g., .ply). In this paper, we propose a camera-agnostic, one-shot, post-training pruning method for 3D Gaussian splats that relies solely on attribute-derived neighbourhood descriptors. As our primary contribution, we introduce a hybrid descriptor framework that captures structural and appearance consistency directly from the splat representation. Building on these descriptors, we formulate pruning as a statistical evidence estimation problem and introduce a Beta evidence model that quantifies per-splat reliability through a probabilistic confidence score. Experiments conducted on standardized test sequences defined by the ISO/IEC MPEG Common Test Conditions (CTC) demonstrate that our approach achieves substantial pruning while preserving reconstruction quality, establishing a practical and generalizable alternative to existing camera-dependent pruning strategies.

  </details>



- **RefracGS: Novel View Synthesis Through Refractive Water Surfaces with 3D Gaussian Ray Tracing**  
  Yiming Shao, Qiyu Dai, Chong Gao, Guanbin Li, Yeqiang Wang, He Sun, Qiong Zeng, Baoquan Chen, Wenzheng Chen  
  _2026-03-23_ · https://arxiv.org/abs/2603.21695v1  
  <details><summary>Abstract</summary>

  Novel view synthesis (NVS) through non-planar refractive surfaces presents fundamental challenges due to severe, spatially varying optical distortions. While recent representations like NeRF and 3D Gaussian Splatting (3DGS) excel at NVS, their assumption of straight-line ray propagation fails under these conditions, leading to significant artifacts. To overcome this limitation, we introduce RefracGS, a framework that jointly reconstructs the refractive water surface and the scene beneath the interface. Our key insight is to explicitly decouple the refractive boundary from the target objects: the refractive surface is modeled via a neural height field, capturing wave geometry, while the underlying scene is represented as a 3D Gaussian field. We formulate a refraction-aware Gaussian ray tracing approach that accurately computes non-linear ray trajectories using Snell's law and efficiently renders the underlying Gaussian field while backpropagating the loss gradients to the parameterized refractive surface. Through end-to-end joint optimization of both representations, our method ensures high-fidelity NVS and view-consistent surface recovery. Experiments on both synthetic and real-world scenes with complex waves demonstrate that RefracGS outperforms prior refractive methods in visual quality, while achieving 15x faster training and real-time rendering at 200 FPS. The project page for RefracGS is available at https://yimgshao.github.io/refracgs/.

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


