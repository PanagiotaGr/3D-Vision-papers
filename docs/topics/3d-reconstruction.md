# 3D Reconstruction

_Updated: 2026-01-15 07:17 UTC_

Total papers shown: **8**


---

- **Efficient Camera-Controlled Video Generation of Static Scenes via Sparse Diffusion and 3D Rendering**  
  Jieying Chen, Jeffrey Hu, Joan Lasenby, Ayush Tewari  
  _2026-01-14_ · https://arxiv.org/abs/2601.09697v1  
  <details><summary>Abstract</summary>

  Modern video generative models based on diffusion models can produce very realistic clips, but they are computationally inefficient, often requiring minutes of GPU time for just a few seconds of video. This inefficiency poses a critical barrier to deploying generative video in applications that require real-time interactions, such as embodied AI and VR/AR. This paper explores a new strategy for camera-conditioned video generation of static scenes: using diffusion-based generative models to generate a sparse set of keyframes, and then synthesizing the full video through 3D reconstruction and rendering. By lifting keyframes into a 3D representation and rendering intermediate views, our approach amortizes the generation cost across hundreds of frames while enforcing geometric consistency. We further introduce a model that predicts the optimal number of keyframes for a given camera trajectory, allowing the system to adaptively allocate computation. Our final method, SRENDER, uses very sparse keyframes for simple trajectories and denser ones for complex camera motion. This results in video generation that is more than 40 times faster than the diffusion-based baseline in generating 20 seconds of video, while maintaining high visual fidelity and temporal stability, offering a practical path toward efficient and controllable video synthesis.

  </details>



- **V-DPM: 4D Video Reconstruction with Dynamic Point Maps**  
  Edgar Sucar, Eldar Insafutdinov, Zihang Lai, Andrea Vedaldi  
  _2026-01-14_ · https://arxiv.org/abs/2601.09499v1  
  <details><summary>Abstract</summary>

  Powerful 3D representations such as DUSt3R invariant point maps, which encode 3D shape and camera parameters, have significantly advanced feed forward 3D reconstruction. While point maps assume static scenes, Dynamic Point Maps (DPMs) extend this concept to dynamic 3D content by additionally representing scene motion. However, existing DPMs are limited to image pairs and, like DUSt3R, require post processing via optimization when more than two views are involved. We argue that DPMs are more useful when applied to videos and introduce V-DPM to demonstrate this. First, we show how to formulate DPMs for video input in a way that maximizes representational power, facilitates neural prediction, and enables reuse of pretrained models. Second, we implement these ideas on top of VGGT, a recent and powerful 3D reconstructor. Although VGGT was trained on static scenes, we show that a modest amount of synthetic data is sufficient to adapt it into an effective V-DPM predictor. Our approach achieves state of the art performance in 3D and 4D reconstruction for dynamic scenes. In particular, unlike recent dynamic extensions of VGGT such as P3, DPMs recover not only dynamic depth but also the full 3D motion of every point in the scene.

  </details>



- **Affostruction: 3D Affordance Grounding with Generative Reconstruction**  
  Chunghyun Park, Seunghyeon Lee, Minsu Cho  
  _2026-01-14_ · https://arxiv.org/abs/2601.09211v1  
  <details><summary>Abstract</summary>

  This paper addresses the problem of affordance grounding from RGBD images of an object, which aims to localize surface regions corresponding to a text query that describes an action on the object. While existing methods predict affordance regions only on visible surfaces, we propose Affostruction, a generative framework that reconstructs complete geometry from partial observations and grounds affordances on the full shape including unobserved regions. We make three core contributions: generative multi-view reconstruction via sparse voxel fusion that extrapolates unseen geometry while maintaining constant token complexity, flow-based affordance grounding that captures inherent ambiguity in affordance distributions, and affordance-driven active view selection that leverages predicted affordances for intelligent viewpoint sampling. Affostruction achieves 19.1 aIoU on affordance grounding (40.4\% improvement) and 32.67 IoU for 3D reconstruction (67.7\% improvement), enabling accurate affordance prediction on complete shapes.

  </details>



- **SPARK: Scalable Real-Time Point Cloud Aggregation with Multi-View Self-Calibration**  
  Chentian Sun  
  _2026-01-13_ · https://arxiv.org/abs/2601.08414v1  
  <details><summary>Abstract</summary>

  Real-time multi-camera 3D reconstruction is crucial for 3D perception, immersive interaction, and robotics. Existing methods struggle with multi-view fusion, camera extrinsic uncertainty, and scalability for large camera setups. We propose SPARK, a self-calibrating real-time multi-camera point cloud reconstruction framework that jointly handles point cloud fusion and extrinsic uncertainty. SPARK consists of: (1) a geometry-aware online extrinsic estimation module leveraging multi-view priors and enforcing cross-view and temporal consistency for stable self-calibration, and (2) a confidence-driven point cloud fusion strategy modeling depth reliability and visibility at pixel and point levels to suppress noise and view-dependent inconsistencies. By performing frame-wise fusion without accumulation, SPARK produces stable point clouds in dynamic scenes while scaling linearly with the number of cameras. Extensive experiments on real-world multi-camera systems show that SPARK outperforms existing approaches in extrinsic accuracy, geometric consistency, temporal stability, and real-time performance, demonstrating its effectiveness and scalability for large-scale multi-camera 3D reconstruction.

  </details>



- **Second-order Gaussian directional derivative representations for image high-resolution corner detection**  
  Dongbo Xie, Junjie Qiu, Changming Sun, Weichuan Zhang  
  _2026-01-13_ · https://arxiv.org/abs/2601.08182v1  
  <details><summary>Abstract</summary>

  Corner detection is widely used in various computer vision tasks, such as image matching and 3D reconstruction. Our research indicates that there are theoretical flaws in Zhang et al.'s use of a simple corner model to obtain a series of corner characteristics, as the grayscale information of two adjacent corners can affect each other. In order to address the above issues, a second-order Gaussian directional derivative (SOGDD) filter is used in this work to smooth two typical high-resolution angle models (i.e. END-type and L-type models). Then, the SOGDD representations of these two corner models were derived separately, and many characteristics of high-resolution corners were discovered, which enabled us to demonstrate how to select Gaussian filtering scales to obtain intensity variation information from images, accurately depicting adjacent corners. In addition, a new high-resolution corner detection method for images has been proposed for the first time, which can accurately detect adjacent corner points. The experimental results have verified that the proposed method outperforms state-of-the-art methods in terms of localization error, robustness to image blur transformation, image matching, and 3D reconstruction.

  </details>



- **PRISM: Color-Stratified Point Cloud Sampling**  
  Hansol Lim, Minhyeok Im, Jongseong Brad Choi  
  _2026-01-11_ · https://arxiv.org/abs/2601.06839v1  
  <details><summary>Abstract</summary>

  We present PRISM, a novel color-guided stratified sampling method for RGB-LiDAR point clouds. Our approach is motivated by the observation that unique scene features often exhibit chromatic diversity while repetitive, redundant features are homogeneous in color. Conventional downsampling methods (Random Sampling, Voxel Grid, Normal Space Sampling) enforce spatial uniformity while ignoring this photometric content. In contrast, PRISM allocates sampling density proportional to chormatic diversity. By treating RGB color space as the stratification domain and imposing a maximum capacity k per color bin, the method preserves texture-rich regions with high color variation while substantially reducing visually homogeneous surfaces. This shifts the sampling space from spatial coverage to visual complexity to produce sparser point clouds that retain essential features for 3D reconstruction tasks.

  </details>



- **MoE3D: A Mixture-of-Experts Module for 3D Reconstruction**  
  Zichen Wang, Ang Cao, Liam J. Wang, Jeong Joon Park  
  _2026-01-08_ · https://arxiv.org/abs/2601.05208v2  
  <details><summary>Abstract</summary>

  We propose a simple yet effective approach to enhance the performance of feed-forward 3D reconstruction models. Existing methods often struggle near depth discontinuities, where standard regression losses encourage spatial averaging and thus blur sharp boundaries. To address this issue, we introduce a mixture-of-experts formulation that handles uncertainty at depth boundaries by combining multiple smooth depth predictions. A softmax weighting head dynamically selects among these hypotheses on a per-pixel basis. By integrating our mixture model into a pre-trained state-of-the-art 3D model, we achieve a substantial reduction of boundary artifacts and gains in overall reconstruction accuracy. Notably, our approach is highly compute efficient, delivering generalizable improvements even when fine-tuned on a small subset of training data while incurring only negligible additional inference computation, suggesting a promising direction for lightweight and accurate 3D reconstruction.

  </details>



- **Segmentation-Driven Monocular Shape from Polarization based on Physical Model**  
  Jinyu Zhang, Xu Ma, Weili Chen, Gonzalo R. Arce  
  _2026-01-08_ · https://arxiv.org/abs/2601.04776v1  
  <details><summary>Abstract</summary>

  Monocular shape-from-polarization (SfP) leverages the intrinsic relationship between light polarization properties and surface geometry to recover surface normals from single-view polarized images, providing a compact and robust approach for three-dimensional (3D) reconstruction. Despite its potential, existing monocular SfP methods suffer from azimuth angle ambiguity, an inherent limitation of polarization analysis, that severely compromises reconstruction accuracy and stability. This paper introduces a novel segmentation-driven monocular SfP (SMSfP) framework that reformulates global shape recovery into a set of local reconstructions over adaptively segmented convex sub-regions. Specifically, a polarization-aided adaptive region growing (PARG) segmentation strategy is proposed to decompose the global convexity assumption into locally convex regions, effectively suppressing azimuth ambiguities and preserving surface continuity. Furthermore, a multi-scale fusion convexity prior (MFCP) constraint is developed to ensure local surface consistency and enhance the recovery of fine textural and structural details. Extensive experiments on both synthetic and real-world datasets validate the proposed approach, showing significant improvements in disambiguation accuracy and geometric fidelity compared with existing physics-based monocular SfP techniques.

  </details>


