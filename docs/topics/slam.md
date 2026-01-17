# SLAM & Localization

_Updated: 2026-01-17 06:45 UTC_

Total papers shown: **6**


---

- **A Unified Framework for Kinematic Simulation of Rigid Foldable Structures**  
  Dongwook Kwak, Geonhee Cho, Jiook Chung, Jinkyu Yang  
  _2026-01-15_ · https://arxiv.org/abs/2601.10225v1  
  <details><summary>Abstract</summary>

  Origami-inspired structures with rigid panels now span thick, kirigami, and multi-sheet realizations, making unified kinematic analysis essential. Yet a general method that consolidates their loop constraints has been lacking. We present an automated approach that generates the Pfaffian constraint matrix for arbitrary rigid foldable structures (RFS). From a minimally extended data schema, the tool constructs the facet-hinge graph, extracts a minimum cycle basis that captures all constraints, and assembles a velocity-level constraint matrix via screw theory that encodes coupled rotation and translation loop closure. The framework computes and visualizes deploy and fold motions across diverse RFS while eliminating tedious and error-prone constraint calculations.

  </details>



- **Terrain-Adaptive Mobile 3D Printing with Hierarchical Control**  
  Shuangshan Nors Li, J. Nathan Kutz  
  _2026-01-15_ · https://arxiv.org/abs/2601.10208v1  
  <details><summary>Abstract</summary>

  Mobile 3D printing on unstructured terrain remains challenging due to the conflict between platform mobility and deposition precision. Existing gantry-based systems achieve high accuracy but lack mobility, while mobile platforms struggle to maintain print quality on uneven ground. We present a framework that tightly integrates AI-driven disturbance prediction with multi-modal sensor fusion and hierarchical hardware control, forming a closed-loop perception-learning-actuation system. The AI module learns terrain-to-perturbation mappings from IMU, vision, and depth sensors, enabling proactive compensation rather than reactive correction. This intelligence is embedded into a three-layer control architecture: path planning, predictive chassis-manipulator coordination, and precision hardware execution. Through outdoor experiments on terrain with slopes and surface irregularities, we demonstrate sub-centimeter printing accuracy while maintaining full platform mobility. This AI-hardware integration establishes a practical foundation for autonomous construction in unstructured environments.

  </details>



- **LCF3D: A Robust and Real-Time Late-Cascade Fusion Framework for 3D Object Detection in Autonomous Driving**  
  Carlo Sgaravatti, Riccardo Pieroni, Matteo Corno, Sergio M. Savaresi, Luca Magri, Giacomo Boracchi  
  _2026-01-14_ · https://arxiv.org/abs/2601.09812v1  
  <details><summary>Abstract</summary>

  Accurately localizing 3D objects like pedestrians, cyclists, and other vehicles is essential in Autonomous Driving. To ensure high detection performance, Autonomous Vehicles complement RGB cameras with LiDAR sensors, but effectively combining these data sources for 3D object detection remains challenging. We propose LCF3D, a novel sensor fusion framework that combines a 2D object detector on RGB images with a 3D object detector on LiDAR point clouds. By leveraging multimodal fusion principles, we compensate for inaccuracies in the LiDAR object detection network. Our solution combines two key principles: (i) late fusion, to reduce LiDAR False Positives by matching LiDAR 3D detections with RGB 2D detections and filtering out unmatched LiDAR detections; and (ii) cascade fusion, to recover missed objects from LiDAR by generating new 3D frustum proposals corresponding to unmatched RGB detections. Experiments show that LCF3D is beneficial for domain generalization, as it turns out to be successful in handling different sensor configurations between training and testing domains. LCF3D achieves significant improvements over LiDAR-based methods, particularly for challenging categories like pedestrians and cyclists in the KITTI dataset, as well as motorcycles and bicycles in nuScenes. Code can be downloaded from: https://github.com/CarloSgaravatti/LCF3D.

  </details>



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



- **Hybrid guided variational autoencoder for visual place recognition**  
  Ni Wang, Zihan You, Emre Neftci, Thorben Schoepe  
  _2026-01-14_ · https://arxiv.org/abs/2601.09248v1  
  <details><summary>Abstract</summary>

  Autonomous agents such as cars, robots and drones need to precisely localize themselves in diverse environments, including in GPS-denied indoor environments. One approach for precise localization is visual place recognition (VPR), which estimates the place of an image based on previously seen places. State-of-the-art VPR models require high amounts of memory, making them unwieldy for mobile deployment, while more compact models lack robustness and generalization capabilities. This work overcomes these limitations for robotics using a combination of event-based vision sensors and an event-based novel guided variational autoencoder (VAE). The encoder part of our model is based on a spiking neural network model which is compatible with power-efficient low latency neuromorphic hardware. The VAE successfully disentangles the visual features of 16 distinct places in our new indoor VPR dataset with a classification performance comparable to other state-of-the-art approaches while, showing robust performance also under various illumination conditions. When tested with novel visual inputs from unknown scenes, our model can distinguish between these places, which demonstrates a high generalization capability by learning the essential features of location. Our compact and robust guided VAE with generalization capabilities poses a promising model for visual place recognition that can significantly enhance mobile robot navigation in known and unknown indoor environments.

  </details>


