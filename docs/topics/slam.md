# SLAM & Localization

_Updated: 2026-02-23 07:19 UTC_

Total papers shown: **4**


---

- **RoEL: Robust Event-based 3D Line Reconstruction**  
  Gwangtak Bae, Jaeho Shin, Seunggu Kang, Junho Kim, Ayoung Kim, Young Min Kim  
  _2026-02-20_ · https://arxiv.org/abs/2602.18258v1  
  <details><summary>Abstract</summary>

  Event cameras in motion tend to detect object boundaries or texture edges, which produce lines of brightness changes, especially in man-made environments. While lines can constitute a robust intermediate representation that is consistently observed, the sparse nature of lines may lead to drastic deterioration with minor estimation errors. Only a few previous works, often accompanied by additional sensors, utilize lines to compensate for the severe domain discrepancies of event sensors along with unpredictable noise characteristics. We propose a method that can stably extract tracks of varying appearances of lines using a clever algorithmic process that observes multiple representations from various time slices of events, compensating for potential adversaries within the event data. We then propose geometric cost functions that can refine the 3D line maps and camera poses, eliminating projective distortions and depth ambiguities. The 3D line maps are highly compact and can be equipped with our proposed cost function, which can be adapted for any observations that can detect and extract line structures or projections of them, including 3D point cloud maps or image observations. We demonstrate that our formulation is powerful enough to exhibit a significant performance boost in event-based mapping and pose refinement across diverse datasets, and can be flexibly applied to multimodal scenarios. Our results confirm that the proposed line-based formulation is a robust and effective approach for the practical deployment of event-based perceptual modules. Project page: https://gwangtak.github.io/roel/

  </details>



- **Have We Mastered Scale in Deep Monocular Visual SLAM? The ScaleMaster Dataset and Benchmark**  
  Hyoseok Ju, Bokeon Suh, Giseop Kim  
  _2026-02-20_ · https://arxiv.org/abs/2602.18174v1  
  <details><summary>Abstract</summary>

  Recent advances in deep monocular visual Simultaneous Localization and Mapping (SLAM) have achieved impressive accuracy and dense reconstruction capabilities, yet their robustness to scale inconsistency in large-scale indoor environments remains largely unexplored. Existing benchmarks are limited to room-scale or structurally simple settings, leaving critical issues of intra-session scale drift and inter-session scale ambiguity insufficiently addressed. To fill this gap, we introduce the ScaleMaster Dataset, the first benchmark explicitly designed to evaluate scale consistency under challenging scenarios such as multi-floor structures, long trajectories, repetitive views, and low-texture regions. We systematically analyze the vulnerability of state-of-the-art deep monocular visual SLAM systems to scale inconsistency, providing both quantitative and qualitative evaluations. Crucially, our analysis extends beyond traditional trajectory metrics to include a direct map-to-map quality assessment using metrics like Chamfer distance against high-fidelity 3D ground truth. Our results reveal that while recent deep monocular visual SLAM systems demonstrate strong performance on existing benchmarks, they suffer from severe scale-related failures in realistic, large-scale indoor environments. By releasing the ScaleMaster dataset and baseline results, we aim to establish a foundation for future research toward developing scale-consistent and reliable visual SLAM systems.

  </details>



- **GrandTour: A Legged Robotics Dataset in the Wild for Multi-Modal Perception and State Estimation**  
  Jonas Frey, Turcan Tuna, Frank Fu, Katharine Patterson, Tianao Xu, Maurice Fallon, Cesar Cadena, Marco Hutter  
  _2026-02-20_ · https://arxiv.org/abs/2602.18164v1  
  <details><summary>Abstract</summary>

  Accurate state estimation and multi-modal perception are prerequisites for autonomous legged robots in complex, large-scale environments. To date, no large-scale public legged-robot dataset captures the real-world conditions needed to develop and benchmark algorithms for legged-robot state estimation, perception, and navigation. To address this, we introduce the GrandTour dataset, a multi-modal legged-robotics dataset collected across challenging outdoor and indoor environments, featuring an ANYbotics ANYmal-D quadruped equipped with the \boxi multi-modal sensor payload. GrandTour spans a broad range of environments and operational scenarios across distinct test sites, ranging from alpine scenery and forests to demolished buildings and urban areas, and covers a wide variation in scale, complexity, illumination, and weather conditions. The dataset provides time-synchronized sensor data from spinning LiDARs, multiple RGB cameras with complementary characteristics, proprioceptive sensors, and stereo depth cameras. Moreover, it includes high-precision ground-truth trajectories from satellite-based RTK-GNSS and a Leica Geosystems total station. This dataset supports research in SLAM, high-precision state estimation, and multi-modal learning, enabling rigorous evaluation and development of new approaches to sensor fusion in legged robotic systems. With its extensive scope, GrandTour represents the largest open-access legged-robotics dataset to date. The dataset is available at https://grand-tour.leggedrobotics.com, on HuggingFace (ROS-independent), and in ROS formats, along with tools and demo resources.

  </details>



- **EgoPush: Learning End-to-End Egocentric Multi-Object Rearrangement for Mobile Robots**  
  Boyuan An, Zhexiong Wang, Yipeng Wang, Jiaqi Li, Sihang Li, Jing Zhang, Chen Feng  
  _2026-02-20_ · https://arxiv.org/abs/2602.18071v1  
  <details><summary>Abstract</summary>

  Humans can rearrange objects in cluttered environments using egocentric perception, navigating occlusions without global coordinates. Inspired by this capability, we study long-horizon multi-object non-prehensile rearrangement for mobile robots using a single egocentric camera. We introduce EgoPush, a policy learning framework that enables egocentric, perception-driven rearrangement without relying on explicit global state estimation that often fails in dynamic scenes. EgoPush designs an object-centric latent space to encode relative spatial relations among objects, rather than absolute poses. This design enables a privileged reinforcement-learning (RL) teacher to jointly learn latent states and mobile actions from sparse keypoints, which is then distilled into a purely visual student policy. To reduce the supervision gap between the omniscient teacher and the partially observed student, we restrict the teacher's observations to visually accessible cues. This induces active perception behaviors that are recoverable from the student's viewpoint. To address long-horizon credit assignment, we decompose rearrangement into stage-level subproblems using temporally decayed, stage-local completion rewards. Extensive simulation experiments demonstrate that EgoPush significantly outperforms end-to-end RL baselines in success rate, with ablation studies validating each design choice. We further demonstrate zero-shot sim-to-real transfer on a mobile platform in the real world. Code and videos are available at https://ai4ce.github.io/EgoPush/.

  </details>


