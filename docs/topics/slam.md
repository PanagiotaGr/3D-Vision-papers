# SLAM & Localization

_Updated: 2026-01-21 06:54 UTC_

Total papers shown: **8**


---

- **SUNSET -- A Sensor-fUsioN based semantic SegmEnTation exemplar for ROS-based self-adaptation**  
  Andreas Wiedholz, Rafael Paintner, Julian Gleißner, Alwin Hoffmann, Tobias Huber  
  _2026-01-20_ · https://arxiv.org/abs/2601.13732v1  
  <details><summary>Abstract</summary>

  The fact that robots are getting deployed more often in dynamic environments, together with the increasing complexity of their software systems, raises the need for self-adaptive approaches. In these environments robotic software systems increasingly operate amid (1) uncertainties, where symptoms are easy to observe but root causes are ambiguous, or (2) multiple uncertainties appear concurrently. We present SUNSET, a ROS2-based exemplar that enables rigorous, repeatable evaluation of architecture-based self-adaptation in such conditions. It implements a sensor fusion semantic-segmentation pipeline driven by a trained Machine Learning (ML) model whose input preprocessing can be perturbed to induce realistic performance degradations. The exemplar exposes five observable symptoms, where each can be caused by different root causes and supports concurrent uncertainties spanning self-healing and self-optimisation. SUNSET includes the segmentation pipeline, a trained ML model, uncertainty-injection scripts, a baseline controller, and step-by-step integration and evaluation documentation to facilitate reproducible studies and fair comparison.

  </details>



- **A General One-Shot Multimodal Active Perception Framework for Robotic Manipulation: Learning to Predict Optimal Viewpoint**  
  Deyun Qin, Zezhi Liu, Hanqian Luo, Xiao Liang, Yongchun Fang  
  _2026-01-20_ · https://arxiv.org/abs/2601.13639v1  
  <details><summary>Abstract</summary>

  Active perception in vision-based robotic manipulation aims to move the camera toward more informative observation viewpoints, thereby providing high-quality perceptual inputs for downstream tasks. Most existing active perception methods rely on iterative optimization, leading to high time and motion costs, and are tightly coupled with task-specific objectives, which limits their transferability. In this paper, we propose a general one-shot multimodal active perception framework for robotic manipulation. The framework enables direct inference of optimal viewpoints and comprises a data collection pipeline and an optimal viewpoint prediction network. Specifically, the framework decouples viewpoint quality evaluation from the overall architecture, supporting heterogeneous task requirements. Optimal viewpoints are defined through systematic sampling and evaluation of candidate viewpoints, after which large-scale training datasets are constructed via domain randomization. Moreover, a multimodal optimal viewpoint prediction network is developed, leveraging cross-attention to align and fuse multimodal features and directly predict camera pose adjustments. The proposed framework is instantiated in robotic grasping under viewpoint-constrained environments. Experimental results demonstrate that active perception guided by the framework significantly improves grasp success rates. Notably, real-world evaluations achieve nearly double the grasp success rate and enable seamless sim-to-real transfer without additional fine-tuning, demonstrating the effectiveness of the proposed framework.

  </details>



- **Event-based Heterogeneous Information Processing for Online Vision-based Obstacle Detection and Localization**  
  Reza Ahmadvand, Sarah Safura Sharif, Yaser Mike Banad  
  _2026-01-19_ · https://arxiv.org/abs/2601.13451v1  
  <details><summary>Abstract</summary>

  This paper introduces a novel framework for robotic vision-based navigation that integrates Hybrid Neural Networks (HNNs) with Spiking Neural Network (SNN)-based filtering to enhance situational awareness for unmodeled obstacle detection and localization. By leveraging the complementary strengths of Artificial Neural Networks (ANNs) and SNNs, the system achieves both accurate environmental understanding and fast, energy-efficient processing. The proposed architecture employs a dual-pathway approach: an ANN component processes static spatial features at low frequency, while an SNN component handles dynamic, event-based sensor data in real time. Unlike conventional hybrid architectures that rely on domain conversion mechanisms, our system incorporates a pre-developed SNN-based filter that directly utilizes spike-encoded inputs for localization and state estimation. Detected anomalies are validated using contextual information from the ANN pathway and continuously tracked to support anticipatory navigation strategies. Simulation results demonstrate that the proposed method offers acceptable detection accuracy while maintaining computational efficiency close to SNN-only implementations, which operate at a fraction of the resource cost. This framework represents a significant advancement in neuromorphic navigation systems for robots operating in unpredictable and dynamic environments.

  </details>



- **Think3D: Thinking with Space for Spatial Reasoning**  
  Zaibin Zhang, Yuhan Wu, Lianjie Jia, Yifan Wang, Zhongbo Zhang, Yijiang Li, Binghao Ran, Fuxi Zhang, Zhuohan Sun, Zhenfei Yin, et al.  
  _2026-01-19_ · https://arxiv.org/abs/2601.13029v1  
  <details><summary>Abstract</summary>

  Understanding and reasoning about the physical world requires spatial intelligence: the ability to interpret geometry, perspective, and spatial relations beyond 2D perception. While recent vision large models (VLMs) excel at visual understanding, they remain fundamentally 2D perceivers and struggle with genuine 3D reasoning. We introduce Think3D, a framework that enables VLM agents to think with 3D space. By leveraging 3D reconstruction models that recover point clouds and camera poses from images or videos, Think3D allows the agent to actively manipulate space through camera-based operations and ego/global-view switching, transforming spatial reasoning into an interactive 3D chain-of-thought process. Without additional training, Think3D significantly improves the spatial reasoning performance of advanced models such as GPT-4.1 and Gemini 2.5 Pro, yielding average gains of +7.8% on BLINK Multi-view and MindCube, and +4.7% on VSI-Bench. We further show that smaller models, which struggle with spatial exploration, benefit significantly from a reinforcement learning policy that enables the model to select informative viewpoints and operations. With RL, the benefit from tool usage increases from +0.7% to +6.8%. Our findings demonstrate that training-free, tool-augmented spatial exploration is a viable path toward more flexible and human-like 3D reasoning in multimodal agents, establishing a new dimension of multimodal intelligence. Code and weights are released at https://github.com/zhangzaibin/spagent.

  </details>



- **Generalizable and Animatable 3D Full-Head Gaussian Avatar from a Single Image**  
  Shuling Zhao, Dan Xu  
  _2026-01-19_ · https://arxiv.org/abs/2601.12770v1  
  <details><summary>Abstract</summary>

  Building 3D animatable head avatars from a single image is an important yet challenging problem. Existing methods generally collapse under large camera pose variations, compromising the realism of 3D avatars. In this work, we propose a new framework to tackle the novel setting of one-shot 3D full-head animatable avatar reconstruction in a single feed-forward pass, enabling real-time animation and simultaneous 360$^\circ$ rendering views. To facilitate efficient animation control, we model 3D head avatars with Gaussian primitives embedded on the surface of a parametric face model within the UV space. To obtain knowledge of full-head geometry and textures, we leverage rich 3D full-head priors within a pretrained 3D generative adversarial network (GAN) for global full-head feature extraction and multi-view supervision. To increase the fidelity of the 3D reconstruction of the input image, we take advantage of the symmetric nature of the UV space and human faces to fuse local fine-grained input image features with the global full-head textures. Extensive experiments demonstrate the effectiveness of our method, achieving high-quality 3D full-head modeling as well as real-time animation, thereby improving the realism of 3D talking avatars.

  </details>



- **DC-VLAQ: Query-Residual Aggregation for Robust Visual Place Recognition**  
  Hanyu Zhu, Zhihao Zhan, Yuhang Ming, Liang Li, Dibo Hou, Javier Civera, Wanzeng Kong  
  _2026-01-19_ · https://arxiv.org/abs/2601.12729v1  
  <details><summary>Abstract</summary>

  One of the central challenges in visual place recognition (VPR) is learning a robust global representation that remains discriminative under large viewpoint changes, illumination variations, and severe domain shifts. While visual foundation models (VFMs) provide strong local features, most existing methods rely on a single model, overlooking the complementary cues offered by different VFMs. However, exploiting such complementary information inevitably alters token distributions, which challenges the stability of existing query-based global aggregation schemes. To address these challenges, we propose DC-VLAQ, a representation-centric framework that integrates the fusion of complementary VFMs and robust global aggregation. Specifically, we first introduce a lightweight residual-guided complementary fusion that anchors representations in the DINOv2 feature space while injecting complementary semantics from CLIP through a learned residual correction. In addition, we propose the Vector of Local Aggregated Queries (VLAQ), a query--residual global aggregation scheme that encodes local tokens by their residual responses to learnable queries, resulting in improved stability and the preservation of fine-grained discriminative cues. Extensive experiments on standard VPR benchmarks, including Pitts30k, Tokyo24/7, MSLS, Nordland, SPED, and AmsterTime, demonstrate that DC-VLAQ consistently outperforms strong baselines and achieves state-of-the-art performance, particularly under challenging domain shifts and long-term appearance changes.

  </details>



- **Camera Pose Revisited**  
  Władysław Skarbek, Michał Salomonowicz, Michał Król  
  _2026-01-18_ · https://arxiv.org/abs/2601.12567v1  
  <details><summary>Abstract</summary>

  Estimating the position and orientation of a camera with respect to an observed scene is one of the central problems in computer vision, particularly in the context of camera calibration and multi-sensor systems. This paper addresses the planar Perspective--$n$--Point problem, with special emphasis on the initial estimation of the pose of a calibration object. As a solution, we propose the \texttt{PnP-ProCay78} algorithm, which combines the classical quadratic formulation of the reconstruction error with a Cayley parameterization of rotations and least-squares optimization. The key component of the method is a deterministic selection of starting points based on an analysis of the reconstruction error for two canonical vectors, allowing costly solution-space search procedures to be avoided. Experimental validation is performed using data acquired also from high-resolution RGB cameras and very low-resolution thermal cameras in an integrated RGB--IR setup. The results demonstrate that the proposed algorithm achieves practically the same projection accuracy as optimal \texttt{SQPnP} and slightly higher than \texttt{IPPE}, both prominent \texttt{PnP-OpenCV} procedures. However, \texttt{PnP-ProCay78} maintains a significantly simpler algorithmic structure. Moreover, the analysis of optimization trajectories in Cayley space provides an intuitive insight into the convergence process, making the method attractive also from a didactic perspective. Unlike existing PnP solvers, the proposed \texttt{PnP-ProCay78} algorithm combines projection error minimization with an analytically eliminated reconstruction-error surrogate for translation, yielding a hybrid cost formulation that is both geometrically transparent and computationally efficient.

  </details>



- **R-VoxelMap: Accurate Voxel Mapping with Recursive Plane Fitting for Online LiDAR Odometry**  
  Haobo Xi, Shiyong Zhang, Qianli Dong, Yunze Tong, Songyang Wu, Jing Yuan, Xuebo Zhang  
  _2026-01-18_ · https://arxiv.org/abs/2601.12377v1  
  <details><summary>Abstract</summary>

  This paper proposes R-VoxelMap, a novel voxel mapping method that constructs accurate voxel maps using a geometry-driven recursive plane fitting strategy to enhance the localization accuracy of online LiDAR odometry. VoxelMap and its variants typically fit and check planes using all points in a voxel, which may lead to plane parameter deviation caused by outliers, over segmentation of large planes, and incorrect merging across different physical planes. To address these issues, R-VoxelMap utilizes a geometry-driven recursive construction strategy based on an outlier detect-and-reuse pipeline. Specifically, for each voxel, accurate planes are first fitted while separating outliers using random sample consensus (RANSAC). The remaining outliers are then propagated to deeper octree levels for recursive processing, ensuring a detailed representation of the environment. In addition, a point distribution-based validity check algorithm is devised to prevent erroneous plane merging. Extensive experiments on diverse open-source LiDAR(-inertial) simultaneous localization and mapping (SLAM) datasets validate that our method achieves higher accuracy than other state-of-the-art approaches, with comparable efficiency and memory usage. Code will be available on GitHub.

  </details>


