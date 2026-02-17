# SLAM & Localization

_Updated: 2026-02-17 07:13 UTC_

Total papers shown: **6**


---

- **Learning Proposes, Geometry Disposes: A Modular Framework for Efficient Spatial Reasoning**  
  Haichao Zhu, Zhaorui Yang, Qian Zhang  
  _2026-02-16_ · https://arxiv.org/abs/2602.14409v1  
  <details><summary>Abstract</summary>

  Spatial perception aims to estimate camera motion and scene structure from visual observations, a problem traditionally addressed through geometric modeling and physical consistency constraints. Recent learning-based methods have demonstrated strong representational capacity for geometric perception and are increasingly used to augment classical geometry-centric systems in practice. However, whether learning components should directly replace geometric estimation or instead serve as intermediate modules within such pipelines remains an open question. In this work, we address this gap and investigate an end-to-end modular framework for effective spatial reasoning, where learning proposes geometric hypotheses, while geometric algorithms dispose estimation decisions. In particular, we study this principle in the context of relative camera pose estimation on RGB-D sequences. Using VGGT as a representative learning model, we evaluate learning-based pose and depth proposals under varying motion magnitudes and scene dynamics, followed by a classical point-to-plane RGB-D ICP as the geometric backend. Our experiments on the TUM RGB-D benchmark reveal three consistent findings: (1) learning-based pose proposals alone are unreliable; (2) learning-proposed geometry, when improperly aligned with camera intrinsics, can degrade performance; and (3) when learning-proposed depth is geometrically aligned and followed by a geometric disposal stage, consistent improvements emerge in moderately challenging rigid settings. These results demonstrate that geometry is not merely a refinement component, but an essential arbiter that validates and absorbs learning-based geometric observations. Our study highlights the importance of modular, geometry-aware system design for robust spatial perception.

  </details>



- **AdaptManip: Learning Adaptive Whole-Body Object Lifting and Delivery with Online Recurrent State Estimation**  
  Morgan Byrd, Donghoon Baek, Kartik Garg, Hyunyoung Jung, Daesol Cho, Maks Sorokin, Robert Wright, Sehoon Ha  
  _2026-02-16_ · https://arxiv.org/abs/2602.14363v1  
  <details><summary>Abstract</summary>

  This paper presents Adaptive Whole-body Loco-Manipulation, AdaptManip, a fully autonomous framework for humanoid robots to perform integrated navigation, object lifting, and delivery. Unlike prior imitation learning-based approaches that rely on human demonstrations and are often brittle to disturbances, AdaptManip aims to train a robust loco-manipulation policy via reinforcement learning without human demonstrations or teleoperation data. The proposed framework consists of three coupled components: (1) a recurrent object state estimator that tracks the manipulated object in real time under limited field-of-view and occlusions; (2) a whole-body base policy for robust locomotion with residual manipulation control for stable object lifting and delivery; and (3) a LiDAR-based robot global position estimator that provides drift-robust localization. All components are trained in simulation using reinforcement learning and deployed on real hardware in a zero-shot manner. Experimental results show that AdaptManip significantly outperforms baseline methods, including imitation learning-based approaches, in adaptability and overall success rate, while accurate object state estimation improves manipulation performance even under occlusion. We further demonstrate fully autonomous real-world navigation, object lifting, and delivery on a humanoid robot.

  </details>



- **Simultaneous State Estimation and Online Model Learning in a Soft Robotic System**  
  Jan-Hendrik Ewering, Max Bartholdt, Simon F. G. Ehlers, Niklas Wahlström, Thomas B. Schön, Thomas Seel  
  _2026-02-15_ · https://arxiv.org/abs/2602.14092v1  
  <details><summary>Abstract</summary>

  Operating complex real-world systems, such as soft robots, can benefit from precise predictive control schemes that require accurate state and model knowledge. This knowledge is typically not available in practical settings and must be inferred from noisy measurements. In particular, it is challenging to simultaneously estimate unknown states and learn a model online from sequentially arriving measurements. In this paper, we show how a recently proposed gray-box system identification tool enables the estimation of a soft robot's current pose while at the same time learning a bending stiffness model. For estimation and learning, we rely solely on a nominal constant-curvature robot model and measurements of the robot's base reactions (e.g., base forces). The estimation scheme -- relying on a marginalized particle filter -- allows us to conveniently interface nominal constant-curvature equations with a Gaussian Process (GP) bending stiffness model to be learned. This, in contrast to estimation via a random walk over stiffness values, enables prediction of bending stiffness and improves overall model quality. We demonstrate, using real-world soft-robot data, that the method learns a bending stiffness model online while accurately estimating the robot's pose. Notably, reduced multi-step forward-prediction errors indicate that the learned bending-stiffness GP improves overall model quality.

  </details>



- **Flow4R: Unifying 4D Reconstruction and Tracking with Scene Flow**  
  Shenhan Qian, Ganlin Zhang, Shangzhe Wu, Daniel Cremers  
  _2026-02-15_ · https://arxiv.org/abs/2602.14021v1  
  <details><summary>Abstract</summary>

  Reconstructing and tracking dynamic 3D scenes remains a fundamental challenge in computer vision. Existing approaches often decouple geometry from motion: multi-view reconstruction methods assume static scenes, while dynamic tracking frameworks rely on explicit camera pose estimation or separate motion models. We propose Flow4R, a unified framework that treats camera-space scene flow as the central representation linking 3D structure, object motion, and camera motion. Flow4R predicts a minimal per-pixel property set-3D point position, scene flow, pose weight, and confidence-from two-view inputs using a Vision Transformer. This flow-centric formulation allows local geometry and bidirectional motion to be inferred symmetrically with a shared decoder in a single forward pass, without requiring explicit pose regressors or bundle adjustment. Trained jointly on static and dynamic datasets, Flow4R achieves state-of-the-art performance on 4D reconstruction and tracking tasks, demonstrating the effectiveness of the flow-central representation for spatiotemporal scene understanding.

  </details>



- **High-fidelity 3D reconstruction for planetary exploration**  
  Alfonso Martínez-Petersen, Levin Gerdes, David Rodríguez-Martínez, C. J. Pérez-del-Pulgar  
  _2026-02-14_ · https://arxiv.org/abs/2602.13909v1  
  <details><summary>Abstract</summary>

  Planetary exploration increasingly relies on autonomous robotic systems capable of perceiving, interpreting, and reconstructing their surroundings in the absence of global positioning or real-time communication with Earth. Rovers operating on planetary surfaces must navigate under sever environmental constraints, limited visual redundancy, and communication delays, making onboard spatial awareness and visual localization key components for mission success. Traditional techniques based on Structure-from-Motion (SfM) and Simultaneous Localization and Mapping (SLAM) provide geometric consistency but struggle to capture radiometric detail or to scale efficiently in unstructured, low-texture terrains typical of extraterrestrial environments. This work explores the integration of radiance field-based methods - specifically Neural Radiance Fields (NeRF) and Gaussian Splatting - into a unified, automated environment reconstruction pipeline for planetary robotics. Our system combines the Nerfstudio and COLMAP frameworks with a ROS2-compatible workflow capable of processing raw rover data directly from rosbag recordings. This approach enables the generation of dense, photorealistic, and metrically consistent 3D representations from minimal visual input, supporting improved perception and planning for autonomous systems operating in planetary-like conditions. The resulting pipeline established a foundation for future research in radiance field-based mapping, bridging the gap between geometric and neural representations in planetary exploration.

  </details>



- **UAV-SEAD: State Estimation Anomaly Dataset for UAVs**  
  Aykut Kabaoglu, Sanem Sariel  
  _2026-02-14_ · https://arxiv.org/abs/2602.13900v1  
  <details><summary>Abstract</summary>

  Accurate state estimation in Unmanned Aerial Vehicles (UAVs) is crucial for ensuring reliable and safe operation, as anomalies occurring during mission execution may induce discrepancies between expected and observed system behaviors, thereby compromising mission success or posing potential safety hazards. It is essential to continuously monitor and detect such conditions in order to ensure a timely response and maintain system reliability. In this work, we focus on UAV state estimation anomalies and provide a large-scale real-world UAV dataset to facilitate research aimed at improving the development of anomaly detection. Unlike existing datasets that primarily rely on injected faults into simulated data, this dataset comprises 1396 real flight logs totaling over 52 hours of flight time, collected across diverse indoor and outdoor environments using a collection of PX4-based UAVs equipped with a variety of sensor configurations. The dataset comprises both normal and anomalous flights without synthetic manipulation, making it uniquely suitable for realistic anomaly detection tasks. A structured classification is proposed that categorizes UAV state estimation anomalies into four classes: mechanical and electrical, external position, global position, and altitude anomalies. These classifications reflect collective, contextual, and outlier anomalies observed in multivariate sensor data streams, including IMU, GPS, barometer, magnetometer, distance sensors, visual odometry, and optical flow, that can be found in the PX4 logging mechanism. It is anticipated that this dataset will play a key role in the development, training, and evaluation of anomaly detection and isolation systems to address the critical gap in UAV reliability research.

  </details>


