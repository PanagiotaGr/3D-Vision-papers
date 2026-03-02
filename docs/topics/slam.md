# SLAM & Localization

_Updated: 2026-03-02 07:14 UTC_

Total papers shown: **4**


---

- **UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images**  
  Junhwa Hur, Charles Herrmann, Songyou Peng, Philipp Henzler, Zeyu Ma, Todd Zickler, Deqing Sun  
  _2026-02-27_ · https://arxiv.org/abs/2602.24290v1  
  <details><summary>Abstract</summary>

  Dense 4D reconstruction from unposed images remains a critical challenge, with current methods relying on slow test-time optimization or fragmented, task-specific feedforward models. We introduce UFO-4D, a unified feedforward framework to reconstruct a dense, explicit 4D representation from just a pair of unposed images. UFO-4D directly estimates dynamic 3D Gaussian Splats, enabling the joint and consistent estimation of 3D geometry, 3D motion, and camera pose in a feedforward manner. Our core insight is that differentiably rendering multiple signals from a single Dynamic 3D Gaussian representation offers major training advantages. This approach enables a self-supervised image synthesis loss while tightly coupling appearance, depth, and motion. Since all modalities share the same geometric primitives, supervising one inherently regularizes and improves the others. This synergy overcomes data scarcity, allowing UFO-4D to outperform prior work by up to 3 times in joint geometry, motion, and camera pose estimation. Our representation also enables high-fidelity 4D interpolation across novel views and time. Please visit our project page for visual results: https://ufo-4d.github.io/

  </details>



- **Autonomous Inspection of Power Line Insulators with UAV on an Unmapped Transmission Tower**  
  Václav Riss, Vít Krátký, Robert Pěnička, Martin Saska  
  _2026-02-27_ · https://arxiv.org/abs/2602.24011v1  
  <details><summary>Abstract</summary>

  This paper introduces an online inspection algorithm that enables an autonomous UAV to fly around a transmission tower and obtain detailed inspection images without a prior map of the tower. Our algorithm relies on camera-LiDAR sensor fusion for online detection and localization of insulators. In particular, the algorithm is based on insulator detection using a convolutional neural network, projection of LiDAR points onto the image, and filtering them using the bounding boxes. The detection pipeline is coupled with several proposed insulator localization methods based on DBSCAN, RANSAC, and PCA algorithms. The performance of the proposed online inspection algorithm and camera-LiDAR sensor fusion pipeline is demonstrated through simulation and real-world flights. In simulation, we showed that our single-flight inspection strategy can save up to 24 % of total inspection time, compared to the two-flight strategy of scanning the tower and afterwards visiting the inspection waypoints in the optimal way. In a real-world experiment, the best performing proposed method achieves a mean horizontal and vertical localization error for the insulator of 0.16 +- 0.08 m and 0.16 +- 0.11 m, respectively. Compared to the most relevant approach, the proposed method achieves more than an order of magnitude lower variance in horizontal insulator localization error.

  </details>



- **AHAP: Reconstructing Arbitrary Humans from Arbitrary Perspectives with Geometric Priors**  
  Xiaozhen Qiao, Wenjia Wang, Zhiyuan Zhao, Jiacheng Sun, Ping Luo, Hongyuan Zhang, Xuelong Li  
  _2026-02-27_ · https://arxiv.org/abs/2602.23951v1  
  <details><summary>Abstract</summary>

  Reconstructing 3D humans from images captured at multiple perspectives typically requires pre-calibration, like using checkerboards or MVS algorithms, which limits scalability and applicability in diverse real-world scenarios. In this work, we present \textbf{AHAP} (Reconstructing \textbf{A}rbitrary \textbf{H}umans from \textbf{A}rbitrary \textbf{P}erspectives), a feed-forward framework for reconstructing arbitrary humans from arbitrary camera perspectives without requiring camera calibration. Our core lies in the effective fusion of multi-view geometry to assist human association, reconstruction and localization. Specifically, we use a Cross-View Identity Association module through learnable person queries and soft assignment, supervised by contrastive learning to resolve cross-view human identity association. A Human Head fuses cross-view features and scene context for SMPL prediction, guided by cross-view reprojection losses to enforce body pose consistency. Additionally, multi-view geometry eliminates the depth ambiguity inherent in monocular methods, providing more precise 3D human localization through multi-view triangulation. Experiments on EgoHumans and EgoExo4D demonstrate that AHAP achieves competitive performance on both world-space human reconstruction and camera pose estimation, while being 180$\times$ faster than optimization-based approaches.

  </details>



- **Altitude-Aware Visual Place Recognition in Top-Down View**  
  Xingyu Shao, Mengfan He, Chunyu Li, Liangzheng Sun, Ziyang Meng  
  _2026-02-27_ · https://arxiv.org/abs/2602.23872v1  
  <details><summary>Abstract</summary>

  To address the challenge of aerial visual place recognition (VPR) problem under significant altitude variations, this study proposes an altitude-adaptive VPR approach that integrates ground feature density analysis with image classification techniques. The proposed method estimates airborne platforms' relative altitude by analyzing the density of ground features in images, then applies relative altitude-based cropping to generate canonical query images, which are subsequently used in a classification-based VPR strategy for localization. Extensive experiments across diverse terrains and altitude conditions demonstrate that the proposed approach achieves high accuracy and robustness in both altitude estimation and VPR under significant altitude changes. Compared to conventional methods relying on barometric altimeters or Time-of-Flight (ToF) sensors, this solution requires no additional hardware and offers a plug-and-play solution for downstream applications, {making it suitable for small- and medium-sized airborne platforms operating in diverse environments, including rural and urban areas.} Under significant altitude variations, incorporating our relative altitude estimation module into the VPR retrieval pipeline boosts average R@1 and R@5 by 29.85\% and 60.20\%, respectively, compared with applying VPR retrieval alone. Furthermore, compared to traditional {Monocular Metric Depth Estimation (MMDE) methods}, the proposed method reduces the mean error by 202.1 m, yielding average additional improvements of 31.4\% in R@1 and 44\% in R@5. These results demonstrate that our method establishes a robust, vision-only framework for three-dimensional visual place recognition, offering a practical and scalable solution for accurate airborne platforms localization under large altitude variations and limited sensor availability.

  </details>


