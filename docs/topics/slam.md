# SLAM & Localization

_Updated: 2026-03-26 07:34 UTC_

Total papers shown: **10**


---

- **Comparative analysis of dual-form networks for live land monitoring using multi-modal satellite image time series**  
  Iris Dumeur, Jérémy Anger, Gabriele Facciolo  
  _2026-03-25_ · https://arxiv.org/abs/2603.24109v1  
  <details><summary>Abstract</summary>

  Multi-modal Satellite Image Time Series (SITS) analysis faces significant computational challenges for live land monitoring applications. While Transformer architectures excel at capturing temporal dependencies and fusing multi-modal data, their quadratic computational complexity and the need to reprocess entire sequences for each new acquisition limit their deployment for regular, large-area monitoring. This paper studies various dual-form attention mechanisms for efficient multi-modal SITS analysis, that enable parallel training while supporting recurrent inference for incremental processing. We compare linear attention and retention mechanisms within a multi-modal spectro-temporal encoder. To address SITS-specific challenges of temporal irregularity and unalignment, we develop temporal adaptations of dual-form mechanisms that compute token distances based on actual acquisition dates rather than sequence indices. Our approach is evaluated on two tasks using Sentinel-1 and Sentinel-2 data: multi-modal SITS forecasting as a proxy task, and real-world solar panel construction monitoring. Experimental results demonstrate that dual-form mechanisms achieve performance comparable to standard Transformers while enabling efficient recurrent inference. The multimodal framework consistently outperforms mono-modal approaches across both tasks, demonstrating the effectiveness of dual mechanisms for sensor fusion. The results presented in this work open new opportunities for operational land monitoring systems requiring regular updates over large geographic areas.

  </details>



- **HGGT: Robust and Flexible 3D Hand Mesh Reconstruction from Uncalibrated Images**  
  Yumeng Liu, Xiao-Xiao Long, Marc Habermann, Xuanze Yang, Cheng Lin, Yuan Liu, Yuexin Ma, Wenping Wang, Ligang Liu  
  _2026-03-25_ · https://arxiv.org/abs/2603.23997v1  
  <details><summary>Abstract</summary>

  Recovering high-fidelity 3D hand geometry from images is a critical task in computer vision, holding significant value for domains such as robotics, animation and VR/AR. Crucially, scalable applications demand both accuracy and deployment flexibility, requiring the ability to leverage massive amounts of unstructured image data from the internet or enable deployment on consumer-grade RGB cameras without complex calibration. However, current methods face a dilemma. While single-view approaches are easy to deploy, they suffer from depth ambiguity and occlusion. Conversely, multi-view systems resolve these uncertainties but typically demand fixed, calibrated setups, limiting their real-world utility. To bridge this gap, we draw inspiration from 3D foundation models that learn explicit geometry directly from visual data. By reformulating hand reconstruction from arbitrary views as a visual-geometry grounded task, we propose a feed-forward architecture that, for the first time in literature, jointly infers 3D hand meshes and camera poses from uncalibrated views. Extensive evaluations show that our approach outperforms state-of-the-art benchmarks and demonstrates strong generalization to uncalibrated, in-the-wild scenarios. Here is the link of our project page: https://lym29.github.io/HGGT/.

  </details>



- **HyDRA: Hybrid Domain-Aware Robust Architecture for Heterogeneous Collaborative Perception**  
  Minwoo Song, Minhee Kang, Heejin Ahn  
  _2026-03-25_ · https://arxiv.org/abs/2603.23975v1  
  <details><summary>Abstract</summary>

  In collaborative perception, an agent's performance can be degraded by heterogeneity arising from differences in model architecture or training data distributions. To address this challenge, we propose HyDRA (Hybrid Domain-Aware Robust Architecture), a unified pipeline that integrates intermediate and late fusion within a domain-aware framework. We introduce a lightweight domain classifier that dynamically identifies heterogeneous agents and assigns them to the late-fusion branch. Furthermore, we propose anchor-guided pose graph optimization to mitigate localization errors inherent in late fusion, leveraging reliable detections from intermediate fusion as fixed spatial anchors. Extensive experiments demonstrate that, despite requiring no additional training, HyDRA achieves performance comparable to state-of-the-art heterogeneity-aware CP methods. Importantly, this performance is maintained as the number of collaborating agents increases, enabling zero-cost scaling without retraining.

  </details>



- **Bio-Inspired Event-Based Visual Servoing for Ground Robots**  
  Maral Mordad, Kian Behzad, Debojyoti Biswas, Noah J. Cowan, Milad Siami  
  _2026-03-24_ · https://arxiv.org/abs/2603.23672v1  
  <details><summary>Abstract</summary>

  Biological sensory systems are inherently adaptive, filtering out constant stimuli and prioritizing relative changes, likely enhancing computational and metabolic efficiency. Inspired by active sensing behaviors across a wide range of animals, this paper presents a novel event-based visual servoing framework for ground robots. Utilizing a Dynamic Vision Sensor (DVS), we demonstrate that by applying a fixed spatial kernel to the asynchronous event stream generated from structured logarithmic intensity-change patterns, the resulting net event flux analytically isolates specific kinematic states. We establish a generalized theoretical bound for this event rate estimator and show that linear and quadratic spatial profiles isolate the robot's velocity and position-velocity product, respectively. Leveraging these properties, we employ a multi-pattern stimulus to directly synthesize a nonlinear state-feedback term entirely without traditional state estimation. To overcome the inescapable loss of linear observability at equilibrium inherent in event sensing, we propose a bio-inspired active sensing limit-cycle controller. Experimental validation on a 1/10-scale autonomous ground vehicle confirms the efficacy, extreme low-latency, and computational efficiency of the proposed direct-sensing approach.

  </details>



- **WildWorld: A Large-Scale Dataset for Dynamic World Modeling with Actions and Explicit State toward Generative ARPG**  
  Zhen Li, Zian Meng, Shuwei Shi, Wenshuo Peng, Yuwei Wu, Bo Zheng, Chuanhao Li, Kaipeng Zhang  
  _2026-03-24_ · https://arxiv.org/abs/2603.23497v1  
  <details><summary>Abstract</summary>

  Dynamical systems theory and reinforcement learning view world evolution as latent-state dynamics driven by actions, with visual observations providing partial information about the state. Recent video world models attempt to learn this action-conditioned dynamics from data. However, existing datasets rarely match the requirement: they typically lack diverse and semantically meaningful action spaces, and actions are directly tied to visual observations rather than mediated by underlying states. As a result, actions are often entangled with pixel-level changes, making it difficult for models to learn structured world dynamics and maintain consistent evolution over long horizons. In this paper, we propose WildWorld, a large-scale action-conditioned world modeling dataset with explicit state annotations, automatically collected from a photorealistic AAA action role-playing game (Monster Hunter: Wilds). WildWorld contains over 108 million frames and features more than 450 actions, including movement, attacks, and skill casting, together with synchronized per-frame annotations of character skeletons, world states, camera poses, and depth maps. We further derive WildBench to evaluate models through Action Following and State Alignment. Extensive experiments reveal persistent challenges in modeling semantically rich actions and maintaining long-horizon state consistency, highlighting the need for state-aware video generation. The project page is https://shandaai.github.io/wildworld-project/.

  </details>



- **Pose-Free Omnidirectional Gaussian Splatting for 360-Degree Videos with Consistent Depth Priors**  
  Chuanqing Zhuang, Xin Lu, Zehui Deng, Zhengda Lu, Yiqun Wang, Junqi Diao, Jun Xiao  
  _2026-03-24_ · https://arxiv.org/abs/2603.23324v1  
  <details><summary>Abstract</summary>

  Omnidirectional 3D Gaussian Splatting with panoramas is a key technique for 3D scene representation, and existing methods typically rely on slow SfM to provide camera poses and sparse points priors. In this work, we propose a pose-free omnidirectional 3DGS method, named PFGS360, that reconstructs 3D Gaussians from unposed omnidirectional videos. To achieve accurate camera pose estimation, we first construct a spherical consistency-aware pose estimation module, which recovers poses by establishing consistent 2D-3D correspondences between the reconstructed Gaussians and the unposed images using Gaussians' internal depth priors. Besides, to enhance the fidelity of novel view synthesis, we introduce a depth-inlier-aware densification module to extract depth inliers and Gaussian outliers with consistent monocular depth priors, enabling efficient Gaussian densification and achieving photorealistic novel view synthesis. The experiments show significant outperformance over existing pose-free and pose-aware 3DGS methods on both real-world and synthetic 360-degree videos. Code is available at https://github.com/zcq15/PFGS360.

  </details>



- **Tightly-Coupled Radar-Visual-Inertial Odometry**  
  Morten Nissov, Mohit Singh, Kostas Alexis  
  _2026-03-24_ · https://arxiv.org/abs/2603.23052v1  
  <details><summary>Abstract</summary>

  Visual-Inertial Odometry (VIO) is a staple for reliable state estimation on constrained and lightweight platforms due to its versatility and demonstrated performance. However, pertinent challenges regarding robust operation in dark, low-texture, obscured environments complicate the use of such methods. Alternatively, Frequency Modulated Continuous Wave (FMCW) radars, and by extension Radar-Inertial Odometry (RIO), offer robustness to these visual challenges, albeit at the cost of reduced information density and worse long-term accuracy. To address these limitations, this work combines the two in a tightly coupled manner, enabling the resulting method to operate robustly regardless of environmental conditions or trajectory dynamics. The proposed method fuses image features, radar Doppler measurements, and Inertial Measurement Unit (IMU) measurements within an Iterated Extended Kalman Filter (IEKF) in real-time, with radar range data augmenting the visual feature depth initialization. The method is evaluated through flight experiments conducted in both indoor and outdoor environments, as well as through challenges to both exteroceptive modalities (such as darkness, fog, or fast flight), thoroughly demonstrating its robustness. The implementation of the proposed method is available at: https://github.com/ntnu-arl/radvio

  </details>



- **Design Guidelines for Nonlinear Kalman Filters via Covariance Compensation**  
  Shida Jiang, Jaewoong Lee, Shengyu Tao, Scott Moura  
  _2026-03-24_ · https://arxiv.org/abs/2603.22992v1  
  <details><summary>Abstract</summary>

  Nonlinear extensions of the Kalman filter (KF), such as the extended Kalman filter (EKF) and the unscented Kalman filter (UKF), are indispensable for state estimation in complex dynamical systems, yet the conditions for a nonlinear KF to provide robust and accurate estimations remain poorly understood. This work proposes a theoretical framework that identifies the causes of failure and success in certain nonlinear KFs and establishes guidelines for their improvement. Central to our framework is the concept of covariance compensation: the deviation between the covariance predicted by a nonlinear KF and that of the EKF. With this definition and detailed theoretical analysis, we derive three design guidelines for nonlinear KFs: (i) invariance under orthogonal transformations, (ii) sufficient covariance compensation beyond the EKF baseline, and (iii) selection of compensation magnitude that favors underconfidence. Both theoretical analysis and empirical validation confirm that adherence to these principles significantly improves estimation accuracy, whereas fixed parameter choices commonly adopted in the literature are often suboptimal. The codes and the proofs for all the theorems in this paper are available at https://github.com/Shida-Jiang/Guidelines-for-Nonlinear-Kalman-Filters.

  </details>



- **MultiCam: On-the-fly Multi-Camera Pose Estimation Using Spatiotemporal Overlaps of Known Objects**  
  Shiyu Li, Hannah Schieber, Kristoffer Waldow, Benjamin Busam, Julian Kreimeier, Daniel Roth  
  _2026-03-24_ · https://arxiv.org/abs/2603.22839v1  
  <details><summary>Abstract</summary>

  Multi-camera dynamic Augmented Reality (AR) applications require a camera pose estimation to leverage individual information from each camera in one common system. This can be achieved by combining contextual information, such as markers or objects, across multiple views. While commonly cameras are calibrated in an initial step or updated through the constant use of markers, another option is to leverage information already present in the scene, like known objects. Another downside of marker-based tracking is that markers have to be tracked inside the field-of-view (FoV) of the cameras. To overcome these limitations, we propose a constant dynamic camera pose estimation leveraging spatiotemporal FoV overlaps of known objects on the fly. To achieve that, we enhance the state-of-the-art object pose estimator to update our spatiotemporal scene graph, enabling a relation even among non-overlapping FoV cameras. To evaluate our approach, we introduce a multi-camera, multi-object pose estimation dataset with temporal FoV overlap, including static and dynamic cameras. Furthermore, in FoV overlapping scenarios, we outperform the state-of-the-art on the widely used YCB-V and T-LESS dataset in camera pose accuracy. Our performance on both previous and our proposed datasets validates the effectiveness of our marker-less approach for AR applications. The code and dataset are available on https://github.com/roth-hex-lab/IEEE-VR-2026-MultiCam.

  </details>



- **Typography-Based Monocular Distance Estimation Framework for Vehicle Safety Systems**  
  Manognya Lokesh Reddy, Zheng Liu  
  _2026-03-24_ · https://arxiv.org/abs/2603.22781v1  
  <details><summary>Abstract</summary>

  Accurate inter-vehicle distance estimation is a cornerstone of advanced driver assistance systems and autonomous driving. While LiDAR and radar provide high precision, their cost prohibits widespread adoption in mass-market vehicles. Monocular vision offers a low-cost alternative but suffers from scale ambiguity and sensitivity to environmental disturbances. This paper introduces a typography-based monocular distance estimation framework, which exploits the standardized typography of license plates as passive fiducial markers for metric distance estimation. The core geometric module uses robust plate detection and character segmentation to measure character height and computes distance via the pinhole camera model. The system incorporates interactive calibration, adaptive detection with strict and permissive modes, and multi-method character segmentation leveraging both adaptive and global thresholding. To enhance robustness, the framework further includes camera pose compensation using lane-based horizon estimation, hybrid deep-learning fusion, temporal Kalman filtering for velocity estimation, and multi-feature fusion that exploits additional typographic cues such as stroke width, character spacing, and plate border thickness. Experimental validation with a calibrated monocular camera in a controlled indoor setup achieved a coefficient of variation of 2.3% in character height across consecutive frames and a mean absolute error of 7.7%. The framework operates without GPU acceleration, demonstrating real-time feasibility. A comprehensive comparison with a plate-width based method shows that character-based ranging reduces the standard deviation of estimates by 35%, translating to smoother, more consistent distance readings in practice, where erratic estimates could trigger unnecessary braking or acceleration.

  </details>


