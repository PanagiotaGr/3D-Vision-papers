# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-24 07:14 UTC_

Total papers shown: **5**


---

- **Flow3r: Factored Flow Prediction for Scalable Visual Geometry Learning**  
  Zhongxiao Cong, Qitao Zhao, Minsik Jeon, Shubham Tulsiani  
  _2026-02-23_ · https://arxiv.org/abs/2602.20157v1  
  <details><summary>Abstract</summary>

  Current feed-forward 3D/4D reconstruction systems rely on dense geometry and pose supervision -- expensive to obtain at scale and particularly scarce for dynamic real-world scenes. We present Flow3r, a framework that augments visual geometry learning with dense 2D correspondences (`flow') as supervision, enabling scalable training from unlabeled monocular videos. Our key insight is that the flow prediction module should be factored: predicting flow between two images using geometry latents from one and pose latents from the other. This factorization directly guides the learning of both scene geometry and camera motion, and naturally extends to dynamic scenes. In controlled experiments, we show that factored flow prediction outperforms alternative designs and that performance scales consistently with unlabeled data. Integrating factored flow into existing visual geometry architectures and training with ${\sim}800$K unlabeled videos, Flow3r achieves state-of-the-art results across eight benchmarks spanning static and dynamic scenes, with its largest gains on in-the-wild dynamic videos where labeled data is most scarce.

  </details>



- **Learning Positive-Incentive Point Sampling in Neural Implicit Fields for Object Pose Estimation**  
  Yifei Shi, Boyan Wan, Xin Xu, Kai Xu  
  _2026-02-23_ · https://arxiv.org/abs/2602.19937v1  
  <details><summary>Abstract</summary>

  Learning neural implicit fields of 3D shapes is a rapidly emerging field that enables shape representation at arbitrary resolutions. Due to the flexibility, neural implicit fields have succeeded in many research areas, including shape reconstruction, novel view image synthesis, and more recently, object pose estimation. Neural implicit fields enable learning dense correspondences between the camera space and the object's canonical space-including unobserved regions in camera space-significantly boosting object pose estimation performance in challenging scenarios like highly occluded objects and novel shapes. Despite progress, predicting canonical coordinates for unobserved camera-space regions remains challenging due to the lack of direct observational signals. This necessitates heavy reliance on the model's generalization ability, resulting in high uncertainty. Consequently, densely sampling points across the entire camera space may yield inaccurate estimations that hinder the learning process and compromise performance. To alleviate this problem, we propose a method combining an SO(3)-equivariant convolutional implicit network and a positive-incentive point sampling (PIPS) strategy. The SO(3)-equivariant convolutional implicit network estimates point-level attributes with SO(3)-equivariance at arbitrary query locations, demonstrating superior performance compared to most existing baselines. The PIPS strategy dynamically determines sampling locations based on the input, thereby boosting the network's accuracy and training efficiency. Our method outperforms the state-of-the-art on three pose estimation datasets. Notably, it demonstrates significant improvements in challenging scenarios, such as objects captured with unseen pose, high occlusion, novel geometry, and severe noise.

  </details>



- **No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection**  
  Zunkai Dai, Ke Li, Jiajia Liu, Jie Yang, Yuanyuan Qiao  
  _2026-02-22_ · https://arxiv.org/abs/2602.19248v1  
  <details><summary>Abstract</summary>

  The collection and detection of video anomaly data has long been a challenging problem due to its rare occurrence and spatio-temporal scarcity. Existing video anomaly detection (VAD) methods under perform in open-world scenarios. Key contributing factors include limited dataset diversity, and inadequate understanding of context-dependent anomalous semantics. To address these issues, i) we propose LAVIDA, an end-to-end zero-shot video anomaly detection framework. ii) LAVIDA employs an Anomaly Exposure Sampler that transforms segmented objects into pseudo-anomalies to enhance model adaptability to unseen anomaly categories. It further integrates a Multimodal Large Language Model (MLLM) to bolster semantic comprehension capabilities. Additionally, iii) we design a token compression approach based on reverse attention to handle the spatio-temporal scarcity of anomalous patterns and decrease computational cost. The training process is conducted solely on pseudo anomalies without any VAD data. Evaluations across four benchmark VAD datasets demonstrate that LAVIDA achieves SOTA performance in both frame-level and pixel-level anomaly detection under the zero-shot setting. Our code is available in https://github.com/VitaminCreed/LAVIDA.

  </details>



- **TeFlow: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation**  
  Qingwen Zhang, Chenhan Jiang, Xiaomeng Zhu, Yunqi Miao, Yushan Zhang, Olov Andersson, Patric Jensfelt  
  _2026-02-22_ · https://arxiv.org/abs/2602.19053v1  
  <details><summary>Abstract</summary>

  Self-supervised feed-forward methods for scene flow estimation offer real-time efficiency, but their supervision from two-frame point correspondences is unreliable and often breaks down under occlusions. Multi-frame supervision has the potential to provide more stable guidance by incorporating motion cues from past frames, yet naive extensions of two-frame objectives are ineffective because point correspondences vary abruptly across frames, producing inconsistent signals. In the paper, we present TeFlow, enabling multi-frame supervision for feed-forward models by mining temporally consistent supervision. TeFlow introduces a temporal ensembling strategy that forms reliable supervisory signals by aggregating the most temporally consistent motion cues from a candidate pool built across multiple frames. Extensive evaluations demonstrate that TeFlow establishes a new state-of-the-art for self-supervised feed-forward methods, achieving performance gains of up to 33\% on the challenging Argoverse 2 and nuScenes datasets. Our method performs on par with leading optimization-based methods, yet speeds up 150 times. The code is open-sourced at https://github.com/KTH-RPL/OpenSceneFlow along with trained model weights.

  </details>



- **PhysConvex: Physics-Informed 3D Dynamic Convex Radiance Fields for Reconstruction and Simulation**  
  Dan Wang, Xinrui Cui, Serge Belongie, Ravi Ramamoorthi  
  _2026-02-21_ · https://arxiv.org/abs/2602.18886v1  
  <details><summary>Abstract</summary>

  Reconstructing and simulating dynamic 3D scenes with both visual realism and physical consistency remains a fundamental challenge. Existing neural representations, such as NeRFs and 3DGS, excel in appearance reconstruction but struggle to capture complex material deformation and dynamics. We propose PhysConvex, a Physics-informed 3D Dynamic Convex Radiance Field that unifies visual rendering and physical simulation. PhysConvex represents deformable radiance fields using physically grounded convex primitives governed by continuum mechanics. We introduce a boundary-driven dynamic convex representation that models deformation through vertex and surface dynamics, capturing spatially adaptive, non-uniform deformation, and evolving boundaries. To efficiently simulate complex geometries and heterogeneous materials, we further develop a reduced-order convex simulation that advects dynamic convex fields using neural skinning eigenmodes as shape- and material-aware deformation bases with time-varying reduced DOFs under Newtonian dynamics. Convex dynamics also offers compact, gap-free volumetric coverage, enhancing both geometric efficiency and simulation fidelity. Experiments demonstrate that PhysConvex achieves high-fidelity reconstruction of geometry, appearance, and physical properties from videos, outperforming existing methods.

  </details>


