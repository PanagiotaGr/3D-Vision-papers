# SLAM & Localization

_Updated: 2026-02-08 07:05 UTC_

Total papers shown: **7**


---

- **Predicting Camera Pose from Perspective Descriptions for Spatial Reasoning**  
  Xuejun Zhang, Aditi Tiwari, Zhenhailong Wang, Heng Ji  
  _2026-02-05_ · https://arxiv.org/abs/2602.06041v1  
  <details><summary>Abstract</summary>

  Multi-image spatial reasoning remains challenging for current multimodal large language models (MLLMs). While single-view perception is inherently 2D, reasoning over multiple views requires building a coherent scene understanding across viewpoints. In particular, we study perspective taking, where a model must build a coherent 3D understanding from multi-view observations and use it to reason from a new, language-specified viewpoint. We introduce CAMCUE, a pose-aware multi-image framework that uses camera pose as an explicit geometric anchor for cross-view fusion and novel-view reasoning. CAMCUE injects per-view pose into visual tokens, grounds natural-language viewpoint descriptions to a target camera pose, and synthesizes a pose-conditioned imagined target view to support answering. To support this setting, we curate CAMCUE-DATA with 27,668 training and 508 test instances pairing multi-view images and poses with diverse target-viewpoint descriptions and perspective-shift questions. We also include human-annotated viewpoint descriptions in the test split to evaluate generalization to human language. CAMCUE improves overall accuracy by 9.06% and predicts target poses from natural-language viewpoint descriptions with over 90% rotation accuracy within 20° and translation accuracy within a 0.5 error threshold. This direct grounding avoids expensive test-time search-and-match, reducing inference time from 256.6s to 1.45s per example and enabling fast, interactive use in real-world scenarios.

  </details>



- **NVS-HO: A Benchmark for Novel View Synthesis of Handheld Objects**  
  Musawar Ali, Manuel Carranza-García, Nicola Fioraio, Samuele Salti, Luigi Di Stefano  
  _2026-02-05_ · https://arxiv.org/abs/2602.05822v1  
  <details><summary>Abstract</summary>

  We propose NVS-HO, the first benchmark designed for novel view synthesis of handheld objects in real-world environments using only RGB inputs. Each object is recorded in two complementary RGB sequences: (1) a handheld sequence, where the object is manipulated in front of a static camera, and (2) a board sequence, where the object is fixed on a ChArUco board to provide accurate camera poses via marker detection. The goal of NVS-HO is to learn a NVS model that captures the full appearance of an object from (1), whereas (2) provides the ground-truth images used for evaluation. To establish baselines, we consider both a classical SfM pipeline and a state-of-the-art pre-trained feed-forward neural network (VGGT) as pose estimators, and train NVS models based on NeRF and Gaussian Splatting. Our experiments reveal significant performance gaps in current methods under unconstrained handheld conditions, highlighting the need for more robust approaches. NVS-HO thus offers a challenging real-world benchmark to drive progress in RGB-based novel view synthesis of handheld objects.

  </details>



- **Geometric Observability Index: An Operator-Theoretic Framework for Per-Feature Sensitivity, Weak Observability, and Dynamic Effects in SE(3) Pose Estimation**  
  Joe-Mei Feng, Sheng-Wei Yu  
  _2026-02-05_ · https://arxiv.org/abs/2602.05582v1  
  <details><summary>Abstract</summary>

  We present a unified operator-theoretic framework for analyzing per-feature sensitivity in camera pose estimation on the Lie group SE(3). Classical sensitivity tools - conditioning analyses, Euclidean perturbation arguments, and Fisher information bounds - do not explain how individual image features influence the pose estimate, nor why dynamic or inconsistent observations can disproportionately distort modern SLAM and structure-from-motion systems. To address this gap, we extend influence function theory to matrix Lie groups and derive an intrinsic perturbation operator for left-trivialized M-estimators on SE(3). The resulting Geometric Observability Index (GOI) quantifies the contribution of a single measurement through the curvature operator and the Lie algebraic structure of the observable subspace. GOI admits a spectral decomposition along the principal directions of the observable curvature, revealing a direct correspondence between weak observability and amplified sensitivity. In the population regime, GOI coincides with the Fisher information geometry on SE(3), yielding a single-measurement analogue of the Cramer-Rao bound. The same spectral mechanism explains classical degeneracies such as pure rotation and vanishing parallax, as well as dynamic feature amplification along weak curvature directions. Overall, GOI provides a geometrically consistent description of measurement influence that unifies conditioning analysis, Fisher information geometry, influence function theory, and dynamic scene detectability through the spectral geometry of the curvature operator. Because these quantities arise directly within Gauss-Newton pipelines, the curvature spectrum and GOI also yield lightweight, training-free diagnostic signals for identifying dynamic features and detecting weak observability configurations without modifying existing SLAM architectures.

  </details>



- **A Comparative Study of 3D Person Detection: Sensor Modalities and Robustness in Diverse Indoor and Outdoor Environments**  
  Malaz Tamim, Andrea Matic-Flierl, Karsten Roscher  
  _2026-02-05_ · https://arxiv.org/abs/2602.05538v1  
  <details><summary>Abstract</summary>

  Accurate 3D person detection is critical for safety in applications such as robotics, industrial monitoring, and surveillance. This work presents a systematic evaluation of 3D person detection using camera-only, LiDAR-only, and camera-LiDAR fusion. While most existing research focuses on autonomous driving, we explore detection performance and robustness in diverse indoor and outdoor scenes using the JRDB dataset. We compare three representative models - BEVDepth (camera), PointPillars (LiDAR), and DAL (camera-LiDAR fusion) - and analyze their behavior under varying occlusion and distance levels. Our results show that the fusion-based approach consistently outperforms single-modality models, particularly in challenging scenarios. We further investigate robustness against sensor corruptions and misalignments, revealing that while DAL offers improved resilience, it remains sensitive to sensor misalignment and certain LiDAR-based corruptions. In contrast, the camera-based BEVDepth model showed the lowest performance and was most affected by occlusion, distance, and noise. Our findings highlight the importance of utilizing sensor fusion for enhanced 3D person detection, while also underscoring the need for ongoing research to address the vulnerabilities inherent in these systems.

  </details>



- **VGGT-Motion: Motion-Aware Calibration-Free Monocular SLAM for Long-Range Consistency**  
  Zhuang Xiong, Chen Zhang, Qingshan Xu, Wenbing Tao  
  _2026-02-05_ · https://arxiv.org/abs/2602.05508v1  
  <details><summary>Abstract</summary>

  Despite recent progress in calibration-free monocular SLAM via 3D vision foundation models, scale drift remains severe on long sequences. Motion-agnostic partitioning breaks contextual coherence and causes zero-motion drift, while conventional geometric alignment is computationally expensive. To address these issues, we propose VGGT-Motion, a calibration-free SLAM system for efficient and robust global consistency over kilometer-scale trajectories. Specifically, we first propose a motion-aware submap construction mechanism that uses optical flow to guide adaptive partitioning, prune static redundancy, and encapsulate turns for stable local geometry. We then design an anchor-driven direct Sim(3) registration strategy. By exploiting context-balanced anchors, it achieves search-free, pixel-wise dense alignment and efficient loop closure without costly feature matching. Finally, a lightweight submap-level pose graph optimization enforces global consistency with linear complexity, enabling scalable long-range operation. Experiments show that VGGT-Motion markedly improves trajectory accuracy and efficiency, achieving state-of-the-art performance in zero-shot, long-range calibration-free monocular SLAM.

  </details>



- **Feature points evaluation on omnidirectional vision with a photorealistic fisheye sequence -- A report on experiments done in 2014**  
  Julien Moreau, S. Ambellouis, Yassine Ruichek  
  _2026-02-05_ · https://arxiv.org/abs/2602.05487v1  
  <details><summary>Abstract</summary>

  What is this report: This is a scientific report, contributing with a detailed bibliography, a dataset which we will call now PFSeq for ''Photorealistic Fisheye Sequence'' and make available at https://doi.org/10. 57745/DYIVVU, and comprehensive experiments. This work should be considered as a draft, and has been done during my PhD thesis ''Construction of 3D models from fisheye video data-Application to the localisation in urban area'' in 2014 [Mor16]. These results have never been published. The aim was to find the best features detector and descriptor for fisheye images, in the context of selfcalibration, with cameras mounted on the top of a car and aiming at the zenith (to proceed then fisheye visual odometry and stereovision in urban scenes). We face a chicken and egg problem, because we can not take advantage of an accurate projection model for an optimal features detection and description, and we rightly need good features to perform the calibration (i.e. to compute the accurate projection model of the camera). What is not this report: It does not contribute with new features algorithm. It does not compare standard features algorithms to algorithms designed for omnidirectional images (unfortunately). It has not been peer-reviewed. Discussions have been translated and enhanced but the experiments have not been run again and the report has not been updated accordingly to the evolution of the state-of-the-art (read this as a 2014 report).

  </details>



- **NeVStereo: A NeRF-Driven NVS-Stereo Architecture for High-Fidelity 3D Tasks**  
  Pengcheng Chen, Yue Hu, Wenhao Li, Nicole M Gunderson, Andrew Feng, Zhenglong Sun, Peter Beerel, Eric J Seibel  
  _2026-02-05_ · https://arxiv.org/abs/2602.05423v1  
  <details><summary>Abstract</summary>

  In modern dense 3D reconstruction, feed-forward systems (e.g., VGGT, pi3) focus on end-to-end matching and geometry prediction but do not explicitly output the novel view synthesis (NVS). Neural rendering-based approaches offer high-fidelity NVS and detailed geometry from posed images, yet they typically assume fixed camera poses and can be sensitive to pose errors. As a result, it remains non-trivial to obtain a single framework that can offer accurate poses, reliable depth, high-quality rendering, and accurate 3D surfaces from casually captured views. We present NeVStereo, a NeRF-driven NVS-stereo architecture that aims to jointly deliver camera poses, multi-view depth, novel view synthesis, and surface reconstruction from multi-view RGB-only inputs. NeVStereo combines NeRF-based NVS for stereo-friendly renderings, confidence-guided multi-view depth estimation, NeRF-coupled bundle adjustment for pose refinement, and an iterative refinement stage that updates both depth and the radiance field to improve geometric consistency. This design mitigated the common NeRF-based issues such as surface stacking, artifacts, and pose-depth coupling. Across indoor, outdoor, tabletop, and aerial benchmarks, our experiments indicate that NeVStereo achieves consistently strong zero-shot performance, with up to 36% lower depth error, 10.4% improved pose accuracy, 4.5% higher NVS fidelity, and state-of-the-art mesh quality (F1 91.93%, Chamfer 4.35 mm) compared to existing prestigious methods.

  </details>


