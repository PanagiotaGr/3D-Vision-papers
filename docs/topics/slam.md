# SLAM

_Updated: 2026-01-15 07:17 UTC_

Total papers shown: **3**


---

- **SCE-SLAM: Scale-Consistent Monocular SLAM via Scene Coordinate Embeddings**  
  Yuchen Wu, Jiahe Li, Xiaohan Yu, Lina Yu, Jin Zheng, Xiao Bai  
  _2026-01-14_ · https://arxiv.org/abs/2601.09665v1  
  <details><summary>Abstract</summary>

  Monocular visual SLAM enables 3D reconstruction from internet video and autonomous navigation on resource-constrained platforms, yet suffers from scale drift, i.e., the gradual divergence of estimated scale over long sequences. Existing frame-to-frame methods achieve real-time performance through local optimization but accumulate scale drift due to the lack of global constraints among independent windows. To address this, we propose SCE-SLAM, an end-to-end SLAM system that maintains scale consistency through scene coordinate embeddings, which are learned patch-level representations encoding 3D geometric relationships under a canonical scale reference. The framework consists of two key modules: geometry-guided aggregation that leverages 3D spatial proximity to propagate scale information from historical observations through geometry-modulated attention, and scene coordinate bundle adjustment that anchors current estimates to the reference scale through explicit 3D coordinate constraints decoded from the scene coordinate embeddings. Experiments on KITTI, Waymo, and vKITTI demonstrate substantial improvements: our method reduces absolute trajectory error by 8.36m on KITTI compared to the best prior approach, while maintaining 36 FPS and achieving scale consistency across large-scale scenes.

  </details>



- **Multimodal Signal Processing For Thermo-Visible-Lidar Fusion In Real-time 3D Semantic Mapping**  
  Jiajun Sun, Yangyi Ou, Haoyuan Zheng, Chao yang, Yue Ma  
  _2026-01-14_ · https://arxiv.org/abs/2601.09578v1  
  <details><summary>Abstract</summary>

  In complex environments, autonomous robot navigation and environmental perception pose higher requirements for SLAM technology. This paper presents a novel method for semantically enhancing 3D point cloud maps with thermal information. By first performing pixel-level fusion of visible and infrared images, the system projects real-time LiDAR point clouds onto this fused image stream. It then segments heat source features in the thermal channel to instantly identify high temperature targets and applies this temperature information as a semantic layer on the final 3D map. This approach generates maps that not only have accurate geometry but also possess a critical semantic understanding of the environment, making it highly valuable for specific applications like rapid disaster assessment and industrial preventive maintenance.

  </details>



- **InsSo3D: Inertial Navigation System and 3D Sonar SLAM for turbid environment inspection**  
  Simon Archieri, Ahmet Cinar, Shu Pan, Jonatan Scharff Willners, Michele Grimald, Ignacio Carlucho, Yvan Petillot  
  _2026-01-09_ · https://arxiv.org/abs/2601.05805v1  
  <details><summary>Abstract</summary>

  This paper presents InsSo3D, an accurate and efficient method for large-scale 3D Simultaneous Localisation and Mapping (SLAM) using a 3D Sonar and an Inertial Navigation System (INS). Unlike traditional sonar, which produces 2D images containing range and azimuth information but lacks elevation information, 3D Sonar produces a 3D point cloud, which therefore does not suffer from elevation ambiguity. We introduce a robust and modern SLAM framework adapted to the 3D Sonar data using INS as prior, detecting loop closure and performing pose graph optimisation. We evaluated InsSo3D performance inside a test tank with access to ground truth data and in an outdoor flooded quarry. Comparisons to reference trajectories and maps obtained from an underwater motion tracking system and visual Structure From Motion (SFM) demonstrate that InsSo3D efficiently corrects odometry drift. The average trajectory error is below 21cm during a 50-minute-long mission, producing a map of 10m by 20m with a 9cm average reconstruction error, enabling safe inspection of natural or artificial underwater structures even in murky water conditions.

  </details>


