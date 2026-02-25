# NeRF & Neural Radiance Fields

_Updated: 2026-02-25 07:15 UTC_

Total papers shown: **10**


---

- **Event-Aided Sharp Radiance Field Reconstruction for Fast-Flying Drones**  
  Rong Zou, Marco Cannici, Davide Scaramuzza  
  _2026-02-24_ · https://arxiv.org/abs/2602.21101v1  
  <details><summary>Abstract</summary>

  Fast-flying aerial robots promise rapid inspection under limited battery constraints, with direct applications in infrastructure inspection, terrain exploration, and search and rescue. However, high speeds lead to severe motion blur in images and induce significant drift and noise in pose estimates, making dense 3D reconstruction with Neural Radiance Fields (NeRFs) particularly challenging due to their high sensitivity to such degradations. In this work, we present a unified framework that leverages asynchronous event streams alongside motion-blurred frames to reconstruct high-fidelity radiance fields from agile drone flights. By embedding event-image fusion into NeRF optimization and jointly refining event-based visual-inertial odometry priors using both event and frame modalities, our method recovers sharp radiance fields and accurate camera trajectories without ground-truth supervision. We validate our approach on both synthetic data and real-world sequences captured by a fast-flying drone. Despite highly dynamic drone flights, where RGB frames are severely degraded by motion blur and pose priors become unreliable, our method reconstructs high-fidelity radiance fields and preserves fine scene details, delivering a performance gain of over 50% on real-world data compared to state-of-the-art methods.

  </details>



- **Monocular Endoscopic Tissue 3D Reconstruction with Multi-Level Geometry Regularization**  
  Yangsen Chen, Hao Wang  
  _2026-02-24_ · https://arxiv.org/abs/2602.20718v1  
  <details><summary>Abstract</summary>

  Reconstructing deformable endoscopic tissues is crucial for achieving robot-assisted surgery. However, 3D Gaussian Splatting-based approaches encounter challenges in achieving consistent tissue surface reconstruction, while existing NeRF-based methods lack real-time rendering capabilities. In pursuit of both smooth deformable surfaces and real-time rendering, we introduce a novel approach based on 3D Gaussian Splatting. Specifically, we introduce surface-aware reconstruction, initially employing a Sign Distance Field-based method to construct a mesh, subsequently utilizing this mesh to constrain the Gaussian Splatting reconstruction process. Furthermore, to ensure the generation of physically plausible deformations, we incorporate local rigidity and global non-rigidity restrictions to guide Gaussian deformation, tailored for the highly deformable nature of soft endoscopic tissue. Based on 3D Gaussian Splatting, our proposed method delivers a fast rendering process and smooth surface appearances. Quantitative and qualitative analysis against alternative methodologies shows that our approach achieves solid reconstruction quality in both textures and geometries.

  </details>



- **Large-scale Photorealistic Outdoor 3D Scene Reconstruction from UAV Imagery Using Gaussian Splatting Techniques**  
  Christos Maikos, Georgios Angelidis, Georgios Th. Papadopoulos  
  _2026-02-23_ · https://arxiv.org/abs/2602.20342v1  
  <details><summary>Abstract</summary>

  In this study, we present an end-to-end pipeline capable of converting drone-captured video streams into high-fidelity 3D reconstructions with minimal latency. Unmanned aerial vehicles (UAVs) are extensively used in aerial real-time perception applications. Moreover, recent advances in 3D Gaussian Splatting (3DGS) have demonstrated significant potential for real-time neural rendering. However, their integration into end-to-end UAV-based reconstruction and visualization systems remains underexplored. Our goal is to propose an efficient architecture that combines live video acquisition via RTMP streaming, synchronized sensor fusion, camera pose estimation, and 3DGS optimization, achieving continuous model updates and low-latency deployment within interactive visualization environments that supports immersive augmented and virtual reality (AR/VR) applications. Experimental results demonstrate that the proposed method achieves competitive visual fidelity, while delivering significantly higher rendering performance and substantially reduced end-to-end latency, compared to NeRF-based approaches. Reconstruction quality remains within 4-7\% of high-fidelity offline references, confirming the suitability of the proposed system for real-time, scalable augmented perception from aerial platforms.

  </details>



- **tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction**  
  Chen Wang, Hao Tan, Wang Yifan, Zhiqin Chen, Yuheng Liu, Kalyan Sunkavalli, Sai Bi, Lingjie Liu, Yiwei Hu  
  _2026-02-23_ · https://arxiv.org/abs/2602.20160v1  
  <details><summary>Abstract</summary>

  We propose tttLRM, a novel large 3D reconstruction model that leverages a Test-Time Training (TTT) layer to enable long-context, autoregressive 3D reconstruction with linear computational complexity, further scaling the model's capability. Our framework efficiently compresses multiple image observations into the fast weights of the TTT layer, forming an implicit 3D representation in the latent space that can be decoded into various explicit formats, such as Gaussian Splats (GS) for downstream applications. The online learning variant of our model supports progressive 3D reconstruction and refinement from streaming observations. We demonstrate that pretraining on novel view synthesis tasks effectively transfers to explicit 3D modeling, resulting in improved reconstruction quality and faster convergence. Extensive experiments show that our method achieves superior performance in feedforward 3D Gaussian reconstruction compared to state-of-the-art approaches on both objects and scenes.

  </details>



- **SemanticNVS: Improving Semantic Scene Understanding in Generative Novel View Synthesis**  
  Xinya Chen, Christopher Wewer, Jiahao Xie, Xinting Hu, Jan Eric Lenssen  
  _2026-02-23_ · https://arxiv.org/abs/2602.20079v1  
  <details><summary>Abstract</summary>

  We present SemanticNVS, a camera-conditioned multi-view diffusion model for novel view synthesis (NVS), which improves generation quality and consistency by integrating pre-trained semantic feature extractors. Existing NVS methods perform well for views near the input view, however, they tend to generate semantically implausible and distorted images under long-range camera motion, revealing severe degradation. We speculate that this degradation is due to current models failing to fully understand their conditioning or intermediate generated scene content. Here, we propose to integrate pre-trained semantic feature extractors to incorporate stronger scene semantics as conditioning to achieve high-quality generation even at distant viewpoints. We investigate two different strategies, (1) warped semantic features and (2) an alternating scheme of understanding and generation at each denoising step. Experimental results on multiple datasets demonstrate the clear qualitative and quantitative (4.69%-15.26% in FID) improvement over state-of-the-art alternatives.

  </details>



- **Spherical Hermite Maps**  
  Mohamed Abouagour, Eleftherios Garyfallidis  
  _2026-02-23_ · https://arxiv.org/abs/2602.20063v1  
  <details><summary>Abstract</summary>

  Spherical functions appear throughout computer graphics, from spherical harmonic lighting and precomputed radiance transfer to neural radiance fields and procedural planet rendering. Efficient evaluation is critical for real-time applications, yet existing approaches face a quality-performance trade-off: bilinear LUT sampling is fast but produces faceting, while bicubic filtering requires 16 texture samples. Most implementations use finite differences for normals, requiring extra samples and introducing noise. This paper presents Spherical Hermite Maps, a derivative-augmented LUT representation that resolves this trade-off. By storing function values alongside scaled partial derivatives at each texel of a padded cubemap, bicubic-Hermite reconstruction is enabled from only four texture samples (a 2x2 footprint) while providing continuous gradients from the same samples. The key insight is that Hermite interpolation reconstructs smooth derivatives as a byproduct of value reconstruction, making surface normals effectively free. In controlled experiments, Spherical Hermite Maps improve PSNR by 8-41 dB over bilinear interpolation and match 16-tap bicubic quality at one-quarter the cost. Analytic normals reduce mean angular error by 9-13% on complex surfaces while yielding stable specular highlights. Three applications demonstrate versatility: spherical harmonic glyph visualization, radial depth-map impostors for mesh level-of-detail, and procedural planet/asteroid rendering with spherical heightfields.

  </details>



- **Learning Positive-Incentive Point Sampling in Neural Implicit Fields for Object Pose Estimation**  
  Yifei Shi, Boyan Wan, Xin Xu, Kai Xu  
  _2026-02-23_ · https://arxiv.org/abs/2602.19937v1  
  <details><summary>Abstract</summary>

  Learning neural implicit fields of 3D shapes is a rapidly emerging field that enables shape representation at arbitrary resolutions. Due to the flexibility, neural implicit fields have succeeded in many research areas, including shape reconstruction, novel view image synthesis, and more recently, object pose estimation. Neural implicit fields enable learning dense correspondences between the camera space and the object's canonical space-including unobserved regions in camera space-significantly boosting object pose estimation performance in challenging scenarios like highly occluded objects and novel shapes. Despite progress, predicting canonical coordinates for unobserved camera-space regions remains challenging due to the lack of direct observational signals. This necessitates heavy reliance on the model's generalization ability, resulting in high uncertainty. Consequently, densely sampling points across the entire camera space may yield inaccurate estimations that hinder the learning process and compromise performance. To alleviate this problem, we propose a method combining an SO(3)-equivariant convolutional implicit network and a positive-incentive point sampling (PIPS) strategy. The SO(3)-equivariant convolutional implicit network estimates point-level attributes with SO(3)-equivariance at arbitrary query locations, demonstrating superior performance compared to most existing baselines. The PIPS strategy dynamically determines sampling locations based on the input, thereby boosting the network's accuracy and training efficiency. Our method outperforms the state-of-the-art on three pose estimation datasets. Notably, it demonstrates significant improvements in challenging scenarios, such as objects captured with unseen pose, high occlusion, novel geometry, and severe noise.

  </details>



- **Augmented Radiance Field: A General Framework for Enhanced Gaussian Splatting**  
  Yixin Yang, Bojian Wu, Yang Zhou, Hui Huang  
  _2026-02-23_ · https://arxiv.org/abs/2602.19916v1  
  <details><summary>Abstract</summary>

  Due to the real-time rendering performance, 3D Gaussian Splatting (3DGS) has emerged as the leading method for radiance field reconstruction. However, its reliance on spherical harmonics for color encoding inherently limits its ability to separate diffuse and specular components, making it challenging to accurately represent complex reflections. To address this, we propose a novel enhanced Gaussian kernel that explicitly models specular effects through view-dependent opacity. Meanwhile, we introduce an error-driven compensation strategy to improve rendering quality in existing 3DGS scenes. Our method begins with 2D Gaussian initialization and then adaptively inserts and optimizes enhanced Gaussian kernels, ultimately producing an augmented radiance field. Experiments demonstrate that our method not only surpasses state-of-the-art NeRF methods in rendering performance but also achieves greater parameter efficiency. Project page at: https://xiaoxinyyx.github.io/augs.

  </details>



- **BayesFusion-SDF: Probabilistic Signed Distance Fusion with View Planning on CPU**  
  Soumya Mazumdar, Vineet Kumar Rakesh, Tapas Samanta  
  _2026-02-23_ · https://arxiv.org/abs/2602.19697v1  
  <details><summary>Abstract</summary>

  Key part of robotics, augmented reality, and digital inspection is dense 3D reconstruction from depth observations. Traditional volumetric fusion techniques, including truncated signed distance functions (TSDF), enable efficient and deterministic geometry reconstruction; however, they depend on heuristic weighting and fail to transparently convey uncertainty in a systematic way. Recent neural implicit methods, on the other hand, get very high fidelity but usually need a lot of GPU power for optimization and aren't very easy to understand for making decisions later on. This work presents BayesFusion-SDF, a CPU-centric probabilistic signed distance fusion framework that conceptualizes geometry as a sparse Gaussian random field with a defined posterior distribution over voxel distances. First, a rough TSDF reconstruction is used to create an adaptive narrow-band domain. Then, depth observations are combined using a heteroscedastic Bayesian formulation that is solved using sparse linear algebra and preconditioned conjugate gradients. Randomized diagonal estimators are a quick way to get an idea of posterior uncertainty. This makes it possible to extract surfaces and plan the next best view while taking into account uncertainty. Tests on a controlled ablation scene and a CO3D object sequence show that the new method is more accurate geometrically than TSDF baselines and gives useful estimates of uncertainty for active sensing. The proposed formulation provides a clear and easy-to-use alternative to GPU-heavy neural reconstruction methods while still being able to be understood in a probabilistic way and acting in a predictable way. GitHub: https://mazumdarsoumya.github.io/BayesFusionSDF

  </details>



- **PoseCraft: Tokenized 3D Body Landmark and Camera Conditioning for Photorealistic Human Image Synthesis**  
  Zhilin Guo, Jing Yang, Kyle Fogarty, Jingyi Wan, Boqiao Zhang, Tianhao Wu, Weihao Xia, Chenliang Zhou, Sakar Khattar, Fangcheng Zhong, et al.  
  _2026-02-22_ · https://arxiv.org/abs/2602.19350v1  
  <details><summary>Abstract</summary>

  Digitizing humans and synthesizing photorealistic avatars with explicit 3D pose and camera controls are central to VR, telepresence, and entertainment. Existing skinning-based workflows require laborious manual rigging or template-based fittings, while neural volumetric methods rely on canonical templates and re-optimization for each unseen pose. We present PoseCraft, a diffusion framework built around tokenized 3D interface: instead of relying only on rasterized geometry as 2D control images, we encode sparse 3D landmarks and camera extrinsics as discrete conditioning tokens and inject them into diffusion via cross-attention. Our approach preserves 3D semantics by avoiding 2D re-projection ambiguity under large pose and viewpoint changes, and produces photorealistic imagery that faithfully captures identity and appearance. To train and evaluate at scale, we also implement GenHumanRF, a data generation workflow that renders diverse supervision from volumetric reconstructions. Our experiments show that PoseCraft achieves significant perceptual quality improvement over diffusion-centric methods, and attains better or comparable metrics to latest volumetric rendering SOTA while better preserving fabric and hair details.

  </details>


