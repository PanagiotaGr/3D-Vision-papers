# SLAM & Localization

_Updated: 2026-02-15 07:05 UTC_

Total papers shown: **3**


---

- **DiffPlace: Street View Generation via Place-Controllable Diffusion Model Enhancing Place Recognition**  
  Ji Li, Zhiwei Li, Shihao Li, Zhenjiang Yu, Boyang Wang, Haiou Liu  
  _2026-02-12_ · https://arxiv.org/abs/2602.11875v1  
  <details><summary>Abstract</summary>

  Generative models have advanced significantly in realistic image synthesis, with diffusion models excelling in quality and stability. Recent multi-view diffusion models improve 3D-aware street view generation, but they struggle to produce place-aware and background-consistent urban scenes from text, BEV maps, and object bounding boxes. This limits their effectiveness in generating realistic samples for place recognition tasks. To address these challenges, we propose DiffPlace, a novel framework that introduces a place-ID controller to enable place-controllable multi-view image generation. The place-ID controller employs linear projection, perceiver transformer, and contrastive learning to map place-ID embeddings into a fixed CLIP space, allowing the model to synthesize images with consistent background buildings while flexibly modifying foreground objects and weather conditions. Extensive experiments, including quantitative comparisons and augmented training evaluations, demonstrate that DiffPlace outperforms existing methods in both generation quality and training support for visual place recognition. Our results highlight the potential of generative models in enhancing scene-level and place-aware synthesis, providing a valuable approach for improving place recognition in autonomous driving

  </details>



- **HAIC: Humanoid Agile Object Interaction Control via Dynamics-Aware World Model**  
  Dongting Li, Xingyu Chen, Qianyang Wu, Bo Chen, Sikai Wu, Hanyu Wu, Guoyao Zhang, Liang Li, Mingliang Zhou, Diyun Xiang, et al.  
  _2026-02-12_ · https://arxiv.org/abs/2602.11758v1  
  <details><summary>Abstract</summary>

  Humanoid robots show promise for complex whole-body tasks in unstructured environments. Although Human-Object Interaction (HOI) has advanced, most methods focus on fully actuated objects rigidly coupled to the robot, ignoring underactuated objects with independent dynamics and non-holonomic constraints. These introduce control challenges from coupling forces and occlusions. We present HAIC, a unified framework for robust interaction across diverse object dynamics without external state estimation. Our key contribution is a dynamics predictor that estimates high-order object states (velocity, acceleration) solely from proprioceptive history. These predictions are projected onto static geometric priors to form a spatially grounded dynamic occupancy map, enabling the policy to infer collision boundaries and contact affordances in blind spots. We use asymmetric fine-tuning, where a world model continuously adapts to the student policy's exploration, ensuring robust state estimation under distribution shifts. Experiments on a humanoid robot show HAIC achieves high success rates in agile tasks (skateboarding, cart pushing/pulling under various loads) by proactively compensating for inertial perturbations, and also masters multi-object long-horizon tasks like carrying a box across varied terrain by predicting the dynamics of multiple objects.

  </details>



- **GSO-SLAM: Bidirectionally Coupled Gaussian Splatting and Direct Visual Odometry**  
  Jiung Yeon, Seongbo Ha, Hyeonwoo Yu  
  _2026-02-12_ · https://arxiv.org/abs/2602.11714v1  
  <details><summary>Abstract</summary>

  We propose GSO-SLAM, a real-time monocular dense SLAM system that leverages Gaussian scene representation. Unlike existing methods that couple tracking and mapping with a unified scene, incurring computational costs, or loosely integrate them with well-structured tracking frameworks, introducing redundancies, our method bidirectionally couples Visual Odometry (VO) and Gaussian Splatting (GS). Specifically, our approach formulates joint optimization within an Expectation-Maximization (EM) framework, enabling the simultaneous refinement of VO-derived semi-dense depth estimates and the GS representation without additional computational overhead. Moreover, we present Gaussian Splat Initialization, which utilizes image information, keyframe poses, and pixel associations from VO to produce close approximations to the final Gaussian scene, thereby eliminating the need for heuristic methods. Through extensive experiments, we validate the effectiveness of our method, showing that it not only operates in real time but also achieves state-of-the-art geometric/photometric fidelity of the reconstructed scene and tracking accuracy.

  </details>


