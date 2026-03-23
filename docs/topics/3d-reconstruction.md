# 3D Reconstruction

_Updated: 2026-03-23 07:38 UTC_

Total papers shown: **9**


---

- **LagerNVS: Latent Geometry for Fully Neural Real-time Novel View Synthesis**  
  Stanislaw Szymanowicz, Minghao Chen, Jianyuan Wang, Christian Rupprecht, Andrea Vedaldi  
  _2026-03-20_ · https://arxiv.org/abs/2603.20176v1  
  <details><summary>Abstract</summary>

  Recent work has shown that neural networks can perform 3D tasks such as Novel View Synthesis (NVS) without explicit 3D reconstruction. Even so, we argue that strong 3D inductive biases are still helpful in the design of such networks. We show this point by introducing LagerNVS, an encoder-decoder neural network for NVS that builds on `3D-aware' latent features. The encoder is initialized from a 3D reconstruction network pre-trained using explicit 3D supervision. This is paired with a lightweight decoder, and trained end-to-end with photometric losses. LagerNVS achieves state-of-the-art deterministic feed-forward Novel View Synthesis (including 31.4 PSNR on Re10k), with and without known cameras, renders in real time, generalizes to in-the-wild data, and can be paired with a diffusion decoder for generative extrapolation.

  </details>



- **A Unified Platform and Quality Assurance Framework for 3D Ultrasound Reconstruction with Robotic, Optical, and Electromagnetic Tracking**  
  Lewis Howell, Manisha Waterston, Tze Min Wah, James H. Chandler, James R. McLaughlan  
  _2026-03-20_ · https://arxiv.org/abs/2603.20077v1  
  <details><summary>Abstract</summary>

  Three-dimensional (3D) Ultrasound (US) can facilitate diagnosis, treatment planning, and image-guided therapy. However, current studies rarely provide a comprehensive evaluation of volumetric accuracy and reproducibility, highlighting the need for robust Quality Assurance (QA) frameworks, particularly for tracked 3D US reconstruction using freehand or robotic acquisition. This study presents a QA framework for 3D US reconstruction and a flexible open source platform for tracked US research. A custom phantom containing geometric inclusions with varying symmetry properties enables straightforward evaluation of optical, electromagnetic, and robotic kinematic tracking for 3D US at different scanning speeds and insonation angles. A standardised pipeline performs real-time segmentation and 3D reconstruction of geometric targets (DSC = 0.97, FPS = 46) without GPU acceleration, followed by automated registration and comparison with ground-truth geometries. Applying this framework showed that our robotic 3D US achieves state-of-the-art reconstruction performance (DSC-3D = 0.94 +- 0.01, HD95 = 1.17 +- 0.12), approaching the spatial resolution limit imposed by the transducer. This work establishes a flexible experimental platform and a reproducible validation methodology for 3D US reconstruction. The proposed framework enables robust cross-platform comparisons and improved reporting practices, supporting the safe and effective clinical translation of 3D ultrasound in diagnostic and image-guided therapy applications.

  </details>



- **Layered Quantum Architecture Search for 3D Point Cloud Classification**  
  Natacha Kuete Meli, Jovita Lukasik, Vladislav Golyanik, Michael Moeller  
  _2026-03-20_ · https://arxiv.org/abs/2603.20024v1  
  <details><summary>Abstract</summary>

  We introduce layered Quantum Architecture Search (layered-QAS), a strategy inspired by classical network morphism that designs Parametrised Quantum Circuit (PQC) architectures by progressively growing and adapting them. PQCs offer strong expressiveness with relatively few parameters, yet they lack standard architectural layers (e.g., convolution, attention) that encode inductive biases for a given learning task. To assess the effectiveness of our method, we focus on 3D point cloud classification as a challenging yet highly structured problem. Whereas prior work on this task has used PQCs only as feature extractors for classical classifiers, our approach uses the PQC as the main building block of the classification model. Simulations show that our layered-QAS mitigates barren plateau, outperforms quantum-adapted local and evolutionary QAS baselines, and achieves state-of-the-art results among PQC-based methods on the ModelNet dataset.

  </details>



- **LIORNet: Self-Supervised LiDAR Snow Removal Framework for Autonomous Driving under Adverse Weather Conditions**  
  Ji-il Park, Inwook Shim  
  _2026-03-20_ · https://arxiv.org/abs/2603.19936v1  
  <details><summary>Abstract</summary>

  LiDAR sensors provide high-resolution 3D perception and long-range detection, making them indispensable for autonomous driving and robotics. However, their performance significantly degrades under adverse weather conditions such as snow, rain, and fog, where spurious noise points dominate the point cloud and lead to false perception. To address this problem, various approaches have been proposed: distance-based filters exploiting spatial sparsity, intensity-based filters leveraging reflectance distributions, and learning-based methods that adapt to complex environments. Nevertheless, distance-based methods struggle to distinguish valid object points from noise, intensity-based methods often rely on fixed thresholds that lack adaptability to changing conditions, and learning-based methods suffer from the high cost of annotation, limited generalization, and computational overhead. In this study, we propose LIORNet, which eliminates these drawbacks and integrates the strengths of all three paradigms. LIORNet is built upon a U-Net++ backbone and employs a self-supervised learning strategy guided by pseudo-labels generated from multiple physical and statistical cues, including range-dependent intensity thresholds, snow reflectivity, point sparsity, and sensing range constraints. This design enables LIORNet to distinguish noise points from environmental structures without requiring manual annotations, thereby overcoming the difficulty of snow labeling and the limitations of single-principle approaches. Extensive experiments on the WADS and CADC datasets demonstrate that LIORNet outperforms state-of-the-art filtering algorithms in both accuracy and runtime while preserving critical environmental features. These results highlight LIORNet as a practical and robust solution for LiDAR perception in extreme weather, with strong potential for real-time deployment in autonomous driving systems.

  </details>



- **SegVGGT: Joint 3D Reconstruction and Instance Segmentation from Multi-View Images**  
  Jinyuan Qu, Hongyang Li, Lei Zhang  
  _2026-03-20_ · https://arxiv.org/abs/2603.19926v1  
  <details><summary>Abstract</summary>

  3D instance segmentation methods typically rely on high-quality point clouds or posed RGB-D scans, requiring complex multi-stage processing pipelines, and are highly sensitive to reconstruction noise. While recent feed-forward transformers have revolutionized multi-view 3D reconstruction, they remain decoupled from high-level semantic understanding. In this work, we present SegVGGT, a unified end-to-end framework that simultaneously performs feed-forward 3D reconstruction and instance segmentation directly from multi-view RGB images. By introducing object queries that interact with multi-level geometric features, our method deeply integrates instance identification into the visual geometry grounded transformer. To address the severe attention dispersion problem caused by the massive number of global image tokens, we propose the Frame-level Attention Distribution Alignment (FADA) strategy. FADA explicitly guides object queries to attend to instance-relevant frames during training, providing structured supervision without extra inference overhead. Extensive experiments demonstrate that SegVGGT achieves the state-of-the-art performance on ScanNetv2 and ScanNet200, outperforming both recent joint models and RGB-D-based approaches, while exhibiting strong generalization capabilities on ScanNet++.

  </details>



- **Learning Hierarchical Orthogonal Prototypes for Generalized Few-Shot 3D Point Cloud Segmentation**  
  Yifei Zhao, Fanyu Zhao, Zhongyuan Zhang, Shengtang Wu, Yixuan Lin, Yinsheng Li  
  _2026-03-20_ · https://arxiv.org/abs/2603.19788v1  
  <details><summary>Abstract</summary>

  Generalized few-shot 3D point cloud segmentation aims to adapt to novel classes from only a few annotations while maintaining strong performance on base classes, but this remains challenging due to the inherent stability-plasticity trade-off: adapting to novel classes can interfere with shared representations and cause base-class forgetting. We present HOP3D, a unified framework that learns hierarchical orthogonal prototypes with an entropy-based few-shot regularizer to enable robust novel-class adaptation without degrading base-class performance. HOP3D introduces hierarchical orthogonalization that decouples base and novel learning at both the gradient and representation levels, effectively mitigating base-novel interference. To further enhance adaptation under sparse supervision, we incorporate an entropy-based regularizer that leverages predictive uncertainty to refine prototype learning and promote balanced predictions. Extensive experiments on ScanNet200 and ScanNet++ demonstrate that HOP3D consistently outperforms state-of-the-art baselines under both 1-shot and 5-shot settings. The code is available at https://fdueblab-hop3d.github.io/.

  </details>



- **PCSTracker: Long-Term Scene Flow Estimation for Point Cloud Sequences**  
  Min Lin, Gangwei Xu, Xianqi Wang, Yuyi Peng, Xin Yang  
  _2026-03-20_ · https://arxiv.org/abs/2603.19762v1  
  <details><summary>Abstract</summary>

  Point cloud scene flow estimation is fundamental to long-term and fine-grained 3D motion analysis. However, existing methods are typically limited to pairwise settings and struggle to maintain temporal consistency over long sequences as geometry evolves, occlusions emerge, and errors accumulate. In this work, we propose PCSTracker, the first end-to-end framework specifically designed for consistent scene flow estimation in point cloud sequences. Specifically, we introduce an iterative geometry motion joint optimization module (IGMO) that explicitly models the temporal evolution of point features to alleviate correspondence inconsistencies caused by dynamic geometric changes. In addition, a spatio-temporal point trajectory update module (STTU) is proposed to leverage broad temporal context to infer plausible positions for occluded points, ensuring coherent motion estimation. To further handle long sequences, we employ an overlapping sliding-window inference strategy that alternates cross-window propagation and in-window refinement, effectively suppressing error accumulation and maintaining stable long-term motion consistency. Extensive experiments on the synthetic PointOdyssey3D and real-world ADT3D datasets show that PCSTracker achieves the best accuracy in long-term scene flow estimation and maintains real-time performance at 32.5 FPS, while demonstrating superior 3D motion understanding compared to RGB-D-based approaches.

  </details>



- **Uncertainty-aware Prototype Learning with Variational Inference for Few-shot Point Cloud Segmentation**  
  Yifei Zhao, Fanyu Zhao, Yinsheng Li  
  _2026-03-20_ · https://arxiv.org/abs/2603.19757v1  
  <details><summary>Abstract</summary>

  Few-shot 3D semantic segmentation aims to generate accurate semantic masks for query point clouds with only a few annotated support examples. Existing prototype-based methods typically construct compact and deterministic prototypes from the support set to guide query segmentation. However, such rigid representations are unable to capture the intrinsic uncertainty introduced by scarce supervision, which often results in degraded robustness and limited generalization. In this work, we propose UPL (Uncertainty-aware Prototype Learning), a probabilistic approach designed to incorporate uncertainty modeling into prototype learning for few-shot 3D segmentation. Our framework introduces two key components. First, UPL introduces a dual-stream prototype refinement module that enriches prototype representations by jointly leveraging limited information from both support and query samples. Second, we formulate prototype learning as a variational inference problem, regarding class prototypes as latent variables. This probabilistic formulation enables explicit uncertainty modeling, providing robust and interpretable mask predictions. Extensive experiments on the widely used ScanNet and S3DIS benchmarks show that our UPL achieves consistent state-of-the-art performance under different settings while providing reliable uncertainty estimation. The code is available at https://fdueblab-upl.github.io/.

  </details>



- **ReLi3D: Relightable Multi-view 3D Reconstruction with Disentangled Illumination**  
  Jan-Niklas Dihlmann, Mark Boss, Simon Donne, Andreas Engelhardt, Hendrik P. A. Lensch, Varun Jampani  
  _2026-03-20_ · https://arxiv.org/abs/2603.19753v1  
  <details><summary>Abstract</summary>

  Reconstructing 3D assets from images has long required separate pipelines for geometry reconstruction, material estimation, and illumination recovery, each with distinct limitations and computational overhead. We present ReLi3D, the first unified end-to-end pipeline that simultaneously reconstructs complete 3D geometry, spatially-varying physically-based materials, and environment illumination from sparse multi-view images in under one second. Our key insight is that multi-view constraints can dramatically improve material and illumination disentanglement, a problem that remains fundamentally ill-posed for single-image methods. Key to our approach is the fusion of the multi-view input via a transformer cross-conditioning architecture, followed by a novel unified two-path prediction strategy. The first path predicts the object's structure and appearance, while the second path predicts the environment illumination from image background or object reflections. This, combined with a differentiable Monte Carlo multiple importance sampling renderer, creates an optimal illumination disentanglement training pipeline. In addition, with our mixed domain training protocol, which combines synthetic PBR datasets with real-world RGB captures, we establish generalizable results in geometry, material accuracy, and illumination quality. By unifying previously separate reconstruction tasks into a single feed-forward pass, we enable near-instantaneous generation of complete, relightable 3D assets. Project Page: https://reli3d.jdihlmann.com/

  </details>


