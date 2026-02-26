# NeRF & Neural Radiance Fields

_Updated: 2026-02-26 07:14 UTC_

Total papers shown: **10**


---

- **Lie Flow: Video Dynamic Fields Modeling and Predicting with Lie Algebra as Geometric Physics Principle**  
  Weidong Qiao, Wangmeng Zuo, Hui Li  
  _2026-02-25_ · https://arxiv.org/abs/2602.21645v1  
  <details><summary>Abstract</summary>

  Modeling 4D scenes requires capturing both spatial structure and temporal motion, which is challenging due to the need for physically consistent representations of complex rigid and non-rigid motions. Existing approaches mainly rely on translational displacements, which struggle to represent rotations, articulated transformations, often leading to spatial inconsistency and physically implausible motion. LieFlow, a dynamic radiance representation framework that explicitly models motion within the SE(3) Lie group, enabling coherent learning of translation and rotation in a unified geometric space. The SE(3) transformation field enforces physically inspired constraints to maintain motion continuity and geometric consistency. The evaluation includes a synthetic dataset with rigid-body trajectories and two real-world datasets capturing complex motion under natural lighting and occlusions. Across all datasets, LieFlow consistently improves view-synthesis fidelity, temporal coherence, and physical realism over NeRF-based baselines. These results confirm that SE(3)-based motion modeling offers a robust and physically grounded framework for representing dynamic 4D scenes.

  </details>



- **Scaling View Synthesis Transformers**  
  Evan Kim, Hyunwoo Ryu, Thomas W. Mitchel, Vincent Sitzmann  
  _2026-02-24_ · https://arxiv.org/abs/2602.21341v1  
  <details><summary>Abstract</summary>

  Geometry-free view synthesis transformers have recently achieved state-of-the-art performance in Novel View Synthesis (NVS), outperforming traditional approaches that rely on explicit geometry modeling. Yet the factors governing their scaling with compute remain unclear. We present a systematic study of scaling laws for view synthesis transformers and derive design principles for training compute-optimal NVS models. Contrary to prior findings, we show that encoder-decoder architectures can be compute-optimal; we trace earlier negative results to suboptimal architectural choices and comparisons across unequal training compute budgets. Across several compute levels, we demonstrate that our encoder-decoder architecture, which we call the Scalable View Synthesis Model (SVSM), scales as effectively as decoder-only models, achieves a superior performance-compute Pareto frontier, and surpasses the previous state-of-the-art on real-world NVS benchmarks with substantially reduced training compute.

  </details>



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


