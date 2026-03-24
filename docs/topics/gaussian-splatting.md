# Gaussian Splatting & 3DGS

_Updated: 2026-03-24 07:17 UTC_

Total papers shown: **12**


---

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



- **Two Experts Are Better Than One Generalist: Decoupling Geometry and Appearance for Feed-Forward 3D Gaussian Splatting**  
  Hwasik Jeong, Seungryong Lee, Gyeongjin Kang, Seungkwon Yang, Xiangyu Sun, Seungtae Nam, Eunbyung Park  
  _2026-03-22_ · https://arxiv.org/abs/2603.21064v1  
  <details><summary>Abstract</summary>

  Pose-free feed-forward 3D Gaussian Splatting (3DGS) has opened a new frontier for rapid 3D modeling, enabling high-quality Gaussian representations to be generated from uncalibrated multi-view images in a single forward pass. The dominant approach in this space adopts unified monolithic architectures, often built on geometry-centric 3D foundation models, to jointly estimate camera poses and synthesize 3DGS representations within a single network. While architecturally streamlined, such "all-in-one" designs may be suboptimal for high-fidelity 3DGS generation, as they entangle geometric reasoning and appearance modeling within a shared representation. In this work, we introduce 2Xplat, a pose-free feed-forward 3DGS framework based on a two-expert design that explicitly separates geometry estimation from Gaussian generation. A dedicated geometry expert first predicts camera poses, which are then explicitly passed to a powerful appearance expert that synthesizes 3D Gaussians. Despite its conceptual simplicity, being largely underexplored in prior works, the proposed approach proves highly effective. In fewer than 5K training iterations, the proposed two-experts pipeline substantially outperforms prior pose-free feed-forward 3DGS approaches and achieves performance on par with state-of-the-art posed methods. These results challenge the prevailing unified paradigm and suggest the potential advantages of modular design principles for complex 3D geometric estimation and appearance synthesis tasks.

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



- **Glove2Hand: Synthesizing Natural Hand-Object Interaction from Multi-Modal Sensing Gloves**  
  Xinyu Zhang, Ziyi Kou, Chuan Qin, Mia Huang, Ergys Ristani, Ankit Kumar, Lele Chen, Kun He, Abdeslam Boularias, Li Guan  
  _2026-03-21_ · https://arxiv.org/abs/2603.20850v1  
  <details><summary>Abstract</summary>

  Understanding hand-object interaction (HOI) is fundamental to computer vision, robotics, and AR/VR. However, conventional hand videos often lack essential physical information such as contact forces and motion signals, and are prone to frequent occlusions. To address the challenges, we present Glove2Hand, a framework that translates multi-modal sensing glove HOI videos into photorealistic bare hands, while faithfully preserving the underlying physical interaction dynamics. We introduce a novel 3D Gaussian hand model that ensures temporal rendering consistency. The rendered hand is seamlessly integrated into the scene using a diffusion-based hand restorer, which effectively handles complex hand-object interactions and non-rigid deformations. Leveraging Glove2Hand, we create HandSense, the first multi-modal HOI dataset featuring glove-to-hand videos with synchronized tactile and IMU signals. We demonstrate that HandSense significantly enhances downstream bare-hand applications, including video-based contact estimation and hand tracking under severe occlusion.

  </details>



- **The Role and Relationship of Initialization and Densification in 3D Gaussian Splatting**  
  Ivan Desiatov, Torsten Sattler  
  _2026-03-21_ · https://arxiv.org/abs/2603.20714v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has become the method of choice for photo-realistic 3D reconstruction of scenes, due to being able to efficiently and accurately recover the scene appearance and geometry from images. 3DGS represents the scene through a set of 3D Gaussians, parameterized by their position, spatial extent, and view-dependent color. Starting from an initial point cloud, 3DGS refines the Gaussians' parameters as to reconstruct a set of training images as accurately as possible. Typically, a sparse Structure-from-Motion point cloud is used as initialization. In order to obtain dense Gaussian clouds, 3DGS methods thus rely on a densification stage. In this paper, we systematically study the relation between densification and initialization. Proposing a new benchmark, we study combinations of different types of initializations (dense laser scans, dense (multi-view) stereo point clouds, dense monocular depth estimates, sparse SfM point clouds) and different densification schemes. We show that current densification approaches are not able to take full advantage of dense initialization as they are often unable to (significantly) improve over sparse SfM-based initialization. We will make our benchmark publicly available.

  </details>


