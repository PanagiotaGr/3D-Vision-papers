# 3D Reconstruction

_Updated: 2026-04-05 07:21 UTC_

Total papers shown: **5**


---

- **CompassAD: Intent-Driven 3D Affordance Grounding in Functionally Competing Objects**  
  Jingliang Li, Jindou Jia, Tuo An, Chuhao Zhou, Xiangyu Chen, Shilin Shan, Boyu Ma, Bofan Lyu, Gen Li, Jianfei Yang  
  _2026-04-02_ · https://arxiv.org/abs/2604.02060v1  
  <details><summary>Abstract</summary>

  When told to "cut the apple," a robot must choose the knife over nearby scissors, despite both objects affording the same cutting function. In real-world scenes, multiple objects may share identical affordances, yet only one is appropriate under the given task context. We call such cases confusing pairs. However, existing 3D affordance methods largely sidestep this challenge by evaluating isolated single objects, often with explicit category names provided in the query. We formalize Multi-Object Affordance Grounding under Intent-Driven Instructions, a new 3D affordance setting that requires predicting a per-point affordance mask on the correct object within a cluttered multi-object point cloud, conditioned on implicit natural language intent. To study this problem, we construct CompassAD, the first benchmark centered on implicit intent in confusable multi-object scenes. It comprises 30 confusing object pairs spanning 16 affordance types, 6,422 scenes, and 88K+ query-answer pairs. Furthermore, we propose CompassNet, a framework that incorporates two dedicated modules tailored to this task. Instance-bounded Cross Injection (ICI) constrains language-geometry alignment within object boundaries to prevent cross-object semantic leakage. Bi-level Contrastive Refinement (BCR) enforces discrimination at both geometric-group and point levels, sharpening distinctions between target and confusable surfaces. Extensive experiments demonstrate state-of-the-art results on both seen and unseen queries, and deployment on a robotic manipulator confirms effective transfer to real-world grasping in confusing multi-object scenes.

  </details>



- **Test-Time Adaptation for Height Completion via Self-Supervised ViT Features and Monocular Foundation Models**  
  Osher Rafaeli, Tal Svoray, Ariel Nahlieli  
  _2026-04-02_ · https://arxiv.org/abs/2604.02009v1  
  <details><summary>Abstract</summary>

  Accurate digital surface models (DSMs) are essential for many geospatial applications, including urban monitoring, environmental analyses, infrastructure management, and change detection. However, large-scale DSMs frequently contain incomplete or outdated regions due to acquisition limitations, reconstruction artifacts, or changes in the built environment. Traditional height completion approaches primarily rely on spatial interpolation or which assume spatial continuity and therefore fail when objects are missing. Recent learning-based approaches improve reconstruction quality but typically require supervised training on sensor-specific datasets, limiting their generalization across domains and sensing conditions. We propose Prior2DSM, a training-free framework for metric DSM completion that operates entirely at test time by leveraging foundation models. Unlike previous height completion approaches that require task-specific training, the proposed method combines self-supervised Vision Transformer (ViT) features from DINOv3 with monocular depth foundation models to propagate metric information from incomplete height priors through semantic feature-space correspondence. Test-time adaptation (TTA) is performed using parameter-efficient low-rank adaptation (LoRA) together with a lightweight multilayer perceptron (MLP), which predicts spatially varying scale and shift parameters to convert relative depth estimates into metric heights. Experiments demonstrate consistent improvements over interpolation based methods, prior-based rescaling height approaches, and state-of-the-art monocular depth estimation models. Prior2DSM reduces reconstruction error while preserving structural fidelity, achieving up to a 46% reduction in RMSE compared to linear fitting of MDE, and further enables DSM updating and coupled RGB-DSM generation.

  </details>



- **Enhanced Polarization Locking in VCSELs**  
  Zifeng Yuan, Dewen Zhang, Lei Shi, Yutong Liu, Aaron Danner  
  _2026-04-02_ · https://arxiv.org/abs/2604.01857v1  
  <details><summary>Abstract</summary>

  While optical injection locking (OIL) of vertical-cavity surface-emitting lasers (VCSELs) has been widely studied in the past, the polarization dynamics of OIL have received far less attention. Recent studies suggest that polarization locking via OIL could enable novel computational applications such as polarization-encoded Ising computers. However, the inherent polarization preference and limited polarization switchability of VCSELs hinder their use for such purposes. To address these challenges, we fabricate VCSELs with tailored oxide aperture designs and combine these with bias current tuning to study the overall impact on polarization locking. Experimental results demonstrate that this approach reduces the required injection power (to as low as 3.6 μW) and expands the locking range. To investigate the impact of the approach, the spin-flip model (SFM) is used to analyze the effects of amplitude anisotropy and bias current on polarization locking, demonstrating strong coherence with experimental results.

  </details>



- **PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency**  
  Leezy Han, Seunggyu Kim, Dongseok Shim, Hyeonbeom Lee  
  _2026-04-02_ · https://arxiv.org/abs/2604.01791v1  
  <details><summary>Abstract</summary>

  Monocular depth estimation (MDE) has been widely adopted in the perception systems of autonomous vehicles and mobile robots. However, existing approaches often struggle to maintain temporal consistency in depth estimation across consecutive frames. This inconsistency not only causes jitter but can also lead to estimation failures when the depth range changes abruptly. To address these challenges, this paper proposes a consistency-aware monocular depth estimation framework that leverages wheel odometry from a mobile robot to achieve stable and coherent depth predictions over time. Specifically, we estimate camera pose and sparse depth from triangulation using optical flow between consecutive frames. The sparse depth estimates are used to update a recursive Bayesian estimate of the metric scale, which is then applied to rescale the relative depth predicted by a pre-trained depth estimation foundation model. The proposed method is evaluated on the KITTI, TartanAir, MS2, and our own dataset, demonstrating robust and accurate depth estimation performance.

  </details>



- **Control-DINO: Feature Space Conditioning for Controllable Image-to-Video Diffusion**  
  Edoardo A. Dominici, Thomas Deixelberger, Konstantinos Vardis, Markus Steinberger  
  _2026-04-02_ · https://arxiv.org/abs/2604.01761v1  
  <details><summary>Abstract</summary>

  Video models have recently been applied with success to problems in content generation, novel view synthesis, and, more broadly, world simulation. Many applications in generation and transfer rely on conditioning these models, typically through perceptual, geometric, or simple semantic signals, fundamentally using them as generative renderers. At the same time, high-dimensional features obtained from large-scale self-supervised learning on images or point clouds are increasingly used as a general-purpose interface for vision models. The connection between the two has been explored for subject specific editing, aligning and training video diffusion models, but not in the role of a more general conditioning signal for pretrained video diffusion models. Features obtained through self-supervised learning like DINO, contain a lot of entangled information about style, lighting and semantics of the scene. This makes them great at reconstruction tasks but limits their generative capabilities. In this paper, we show how we can use the features for tasks such as video domain transfer and video-from-3D generation. We introduce a lightweight architecture and training strategy that decouples appearance from other features that we wish to preserve, enabling robust control for appearance changes such as stylization and relighting. Furthermore, we show that low spatial resolution can be compensated by higher feature dimensionality, improving controllability in generative rendering from explicit spatial representations.

  </details>


