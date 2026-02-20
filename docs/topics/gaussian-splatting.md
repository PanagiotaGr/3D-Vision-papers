# Gaussian Splatting & 3DGS

_Updated: 2026-02-20 07:12 UTC_

Total papers shown: **5**


---

- **4D Monocular Surgical Reconstruction under Arbitrary Camera Motions**  
  Jiwei Shan, Zeyu Cai, Cheng-Tai Hsieh, Yirui Li, Hao Liu, Lijun Han, Hesheng Wang, Shing Shin Cheng  
  _2026-02-19_ · https://arxiv.org/abs/2602.17473v1  
  <details><summary>Abstract</summary>

  Reconstructing deformable surgical scenes from endoscopic videos is challenging and clinically important. Recent state-of-the-art methods based on implicit neural representations or 3D Gaussian splatting have made notable progress. However, most are designed for deformable scenes with fixed endoscope viewpoints and rely on stereo depth priors or accurate structure-from-motion for initialization and optimization, limiting their ability to handle monocular sequences with large camera motion in real clinical settings. To address this, we propose Local-EndoGS, a high-quality 4D reconstruction framework for monocular endoscopic sequences with arbitrary camera motion. Local-EndoGS introduces a progressive, window-based global representation that allocates local deformable scene models to each observed window, enabling scalability to long sequences with substantial motion. To overcome unreliable initialization without stereo depth or accurate structure-from-motion, we design a coarse-to-fine strategy integrating multi-view geometry, cross-window information, and monocular depth priors, providing a robust foundation for optimization. We further incorporate long-range 2D pixel trajectory constraints and physical motion priors to improve deformation plausibility. Experiments on three public endoscopic datasets with deformable scenes and varying camera motions show that Local-EndoGS consistently outperforms state-of-the-art methods in appearance quality and geometry. Ablation studies validate the effectiveness of our key designs. Code will be released upon acceptance at: https://github.com/IRMVLab/Local-EndoGS.

  </details>



- **NRGS-SLAM: Monocular Non-Rigid SLAM for Endoscopy via Deformation-Aware 3D Gaussian Splatting**  
  Jiwei Shan, Zeyu Cai, Yirui Li, Yongbo Chen, Lijun Han, Yun-hui Liu, Hesheng Wang, Shing Shin Cheng  
  _2026-02-19_ · https://arxiv.org/abs/2602.17182v1  
  <details><summary>Abstract</summary>

  Visual simultaneous localization and mapping (V-SLAM) is a fundamental capability for autonomous perception and navigation. However, endoscopic scenes violate the rigidity assumption due to persistent soft-tissue deformations, creating a strong coupling ambiguity between camera ego-motion and intrinsic deformation. Although recent monocular non-rigid SLAM methods have made notable progress, they often lack effective decoupling mechanisms and rely on sparse or low-fidelity scene representations, which leads to tracking drift and limited reconstruction quality. To address these limitations, we propose NRGS-SLAM, a monocular non-rigid SLAM system for endoscopy based on 3D Gaussian Splatting. To resolve the coupling ambiguity, we introduce a deformation-aware 3D Gaussian map that augments each Gaussian primitive with a learnable deformation probability, optimized via a Bayesian self-supervision strategy without requiring external non-rigidity labels. Building on this representation, we design a deformable tracking module that performs robust coarse-to-fine pose estimation by prioritizing low-deformation regions, followed by efficient per-frame deformation updates. A carefully designed deformable mapping module progressively expands and refines the map, balancing representational capacity and computational efficiency. In addition, a unified robust geometric loss incorporates external geometric priors to mitigate the inherent ill-posedness of monocular non-rigid SLAM. Extensive experiments on multiple public endoscopic datasets demonstrate that NRGS-SLAM achieves more accurate camera pose estimation (up to 50\% reduction in RMSE) and higher-quality photo-realistic reconstructions than state-of-the-art methods. Comprehensive ablation studies further validate the effectiveness of our key design choices. Source code will be publicly available upon paper acceptance.

  </details>



- **B$^3$-Seg: Camera-Free, Training-Free 3DGS Segmentation via Analytic EIG and Beta-Bernoulli Bayesian Updates**  
  Hiromichi Kamata, Samuel Arthur Munro, Fuminori Homma  
  _2026-02-19_ · https://arxiv.org/abs/2602.17134v1  
  <details><summary>Abstract</summary>

  Interactive 3D Gaussian Splatting (3DGS) segmentation is essential for real-time editing of pre-reconstructed assets in film and game production. However, existing methods rely on predefined camera viewpoints, ground-truth labels, or costly retraining, making them impractical for low-latency use. We propose B$^3$-Seg (Beta-Bernoulli Bayesian Segmentation for 3DGS), a fast and theoretically grounded method for open-vocabulary 3DGS segmentation under camera-free and training-free conditions. Our approach reformulates segmentation as sequential Beta-Bernoulli Bayesian updates and actively selects the next view via analytic Expected Information Gain (EIG). This Bayesian formulation guarantees the adaptive monotonicity and submodularity of EIG, which produces a greedy $(1{-}1/e)$ approximation to the optimal view sampling policy. Experiments on multiple datasets show that B$^3$-Seg achieves competitive results to high-cost supervised methods while operating end-to-end segmentation within a few seconds. The results demonstrate that B$^3$-Seg enables practical, interactive 3DGS segmentation with provable information efficiency.

  </details>



- **3D Scene Rendering with Multimodal Gaussian Splatting**  
  Chi-Shiang Gau, Konstantinos D. Polyzos, Athanasios Bacharis, Saketh Madhuvarasu, Tara Javidi  
  _2026-02-19_ · https://arxiv.org/abs/2602.17124v1  
  <details><summary>Abstract</summary>

  3D scene reconstruction and rendering are core tasks in computer vision, with applications spanning industrial monitoring, robotics, and autonomous driving. Recent advances in 3D Gaussian Splatting (GS) and its variants have achieved impressive rendering fidelity while maintaining high computational and memory efficiency. However, conventional vision-based GS pipelines typically rely on a sufficient number of camera views to initialize the Gaussian primitives and train their parameters, typically incurring additional processing cost during initialization while falling short in conditions where visual cues are unreliable, such as adverse weather, low illumination, or partial occlusions. To cope with these challenges, and motivated by the robustness of radio-frequency (RF) signals to weather, lighting, and occlusions, we introduce a multimodal framework that integrates RF sensing, such as automotive radar, with GS-based rendering as a more efficient and robust alternative to vision-only GS rendering. The proposed approach enables efficient depth prediction from only sparse RF-based depth measurements, yielding a high-quality 3D point cloud for initializing Gaussian functions across diverse GS architectures. Numerical tests demonstrate the merits of judiciously incorporating RF sensing into GS pipelines, achieving high-fidelity 3D scene rendering driven by RF-informed structural accuracy.

  </details>



- **Semantic-Guided 3D Gaussian Splatting for Transient Object Removal**  
  Aditi Prabakaran, Priyesh Shukla  
  _2026-02-17_ · https://arxiv.org/abs/2602.15516v1  
  <details><summary>Abstract</summary>

  Transient objects in casual multi-view captures cause ghosting artifacts in 3D Gaussian Splatting (3DGS) reconstruction. Existing solutions relied on scene decomposition at significant memory cost or on motion-based heuristics that were vulnerable to parallax ambiguity. A semantic filtering framework was proposed for category-aware transient removal using vision-language models. CLIP similarity scores between rendered views and distractor text prompts were accumulated per-Gaussian across training iterations. Gaussians exceeding a calibrated threshold underwent opacity regularization and periodic pruning. Unlike motion-based approaches, semantic classification resolved parallax ambiguity by identifying object categories independently of motion patterns. Experiments on the RobustNeRF benchmark demonstrated consistent improvement in reconstruction quality over vanilla 3DGS across four sequences, while maintaining minimal memory overhead and real-time rendering performance. Threshold calibration and comparisons with baselines validated semantic guidance as a practical strategy for transient removal in scenarios with predictable distractor categories.

  </details>


