# SLAM & Localization

_Updated: 2026-02-20 07:12 UTC_

Total papers shown: **8**


---

- **GraphThinker: Reinforcing Video Reasoning with Event Graph Thinking**  
  Zixu Cheng, Da Li, Jian Hu, Ziquan Liu, Wei Li, Shaogang Gong  
  _2026-02-19_ · https://arxiv.org/abs/2602.17555v1  
  <details><summary>Abstract</summary>

  Video reasoning requires understanding the causal relationships between events in a video. However, such relationships are often implicit and costly to annotate manually. While existing multimodal large language models (MLLMs) often infer event relations through dense captions or video summaries for video reasoning, such modeling still lacks causal understanding. Without explicit causal structure modeling within and across video events, these models suffer from hallucinations during the video reasoning. In this work, we propose GraphThinker, a reinforcement finetuning-based method that constructs structural event-level scene graphs and enhances visual grounding to jointly reduce hallucinations in video reasoning. Specifically, we first employ an MLLM to construct an event-based video scene graph (EVSG) that explicitly models both intra- and inter-event relations, and incorporate these formed scene graphs into the MLLM as an intermediate thinking process. We also introduce a visual attention reward during reinforcement finetuning, which strengthens video grounding and further mitigates hallucinations. We evaluate GraphThinker on two datasets, RexTime and VidHalluc, where it shows superior ability to capture object and event relations with more precise event localization, reducing hallucinations in video reasoning compared to prior methods.

  </details>



- **NRGS-SLAM: Monocular Non-Rigid SLAM for Endoscopy via Deformation-Aware 3D Gaussian Splatting**  
  Jiwei Shan, Zeyu Cai, Yirui Li, Yongbo Chen, Lijun Han, Yun-hui Liu, Hesheng Wang, Shing Shin Cheng  
  _2026-02-19_ · https://arxiv.org/abs/2602.17182v1  
  <details><summary>Abstract</summary>

  Visual simultaneous localization and mapping (V-SLAM) is a fundamental capability for autonomous perception and navigation. However, endoscopic scenes violate the rigidity assumption due to persistent soft-tissue deformations, creating a strong coupling ambiguity between camera ego-motion and intrinsic deformation. Although recent monocular non-rigid SLAM methods have made notable progress, they often lack effective decoupling mechanisms and rely on sparse or low-fidelity scene representations, which leads to tracking drift and limited reconstruction quality. To address these limitations, we propose NRGS-SLAM, a monocular non-rigid SLAM system for endoscopy based on 3D Gaussian Splatting. To resolve the coupling ambiguity, we introduce a deformation-aware 3D Gaussian map that augments each Gaussian primitive with a learnable deformation probability, optimized via a Bayesian self-supervision strategy without requiring external non-rigidity labels. Building on this representation, we design a deformable tracking module that performs robust coarse-to-fine pose estimation by prioritizing low-deformation regions, followed by efficient per-frame deformation updates. A carefully designed deformable mapping module progressively expands and refines the map, balancing representational capacity and computational efficiency. In addition, a unified robust geometric loss incorporates external geometric priors to mitigate the inherent ill-posedness of monocular non-rigid SLAM. Extensive experiments on multiple public endoscopic datasets demonstrate that NRGS-SLAM achieves more accurate camera pose estimation (up to 50\% reduction in RMSE) and higher-quality photo-realistic reconstructions than state-of-the-art methods. Comprehensive ablation studies further validate the effectiveness of our key design choices. Source code will be publicly available upon paper acceptance.

  </details>



- **Cholec80-port: A Geometrically Consistent Trocar Port Segmentation Dataset for Robust Surgical Scene Understanding**  
  Shunsuke Kikuchi, Atsushi Kouno, Hiroki Matsuzaki  
  _2026-02-19_ · https://arxiv.org/abs/2602.17060v1  
  <details><summary>Abstract</summary>

  Trocar ports are camera-fixed, pseudo-static structures that can persistently occlude laparoscopic views and attract disproportionate feature points due to specular, textured surfaces. This makes ports particularly detrimental to geometry-based downstream pipelines such as image stitching, 3D reconstruction, and visual SLAM, where dynamic or non-anatomical outliers degrade alignment and tracking stability. Despite this practical importance, explicit port labels are rare in public surgical datasets, and existing annotations often violate geometric consistency by masking the central lumen (opening), even when anatomical regions are visible through it. We present Cholec80-port, a high-fidelity trocar port segmentation dataset derived from Cholec80, together with a rigorous standard operating procedure (SOP) that defines a port-sleeve mask excluding the central opening. We additionally cleanse and unify existing public datasets under the same SOP. Experiments demonstrate that geometrically consistent annotations substantially improve cross-dataset robustness beyond what dataset size alone provides.

  </details>



- **Decentralized and Fully Onboard: Range-Aided Cooperative Localization and Navigation on Micro Aerial Vehicles**  
  Abhishek Goudar, Angela P. Schoellig  
  _2026-02-18_ · https://arxiv.org/abs/2602.16594v2  
  <details><summary>Abstract</summary>

  Controlling a team of robots in a coordinated manner is challenging because centralized approaches (where all computation is performed on a central machine) scale poorly, and globally referenced external localization systems may not always be available. In this work, we consider the problem of range-aided decentralized localization and formation control. In such a setting, each robot estimates its relative pose by combining data only from onboard odometry sensors and distance measurements to other robots in the team. Additionally, each robot calculates the control inputs necessary to collaboratively navigate an environment to accomplish a specific task, for example, moving in a desired formation while monitoring an area. We present a block coordinate descent approach to localization that does not require strict coordination between the robots. We present a novel formulation for formation control as inference on factor graphs that takes into account the state estimation uncertainty and can be solved efficiently. Our approach to range-aided localization and formation-based navigation is completely decentralized, does not require specialized trajectories to maintain formation, and achieves decimeter-level positioning and formation control accuracy. We demonstrate our approach through multiple real experiments involving formation flights in diverse indoor and outdoor environments.

  </details>



- **Markerless Robot Detection and 6D Pose Estimation for Multi-Agent SLAM**  
  Markus Rueggeberg, Maximilian Ulmer, Maximilian Durner, Wout Boerdijk, Marcus Gerhard Mueller, Rudolph Triebel, Riccardo Giubilato  
  _2026-02-18_ · https://arxiv.org/abs/2602.16308v1  
  <details><summary>Abstract</summary>

  The capability of multi-robot SLAM approaches to merge localization history and maps from different observers is often challenged by the difficulty in establishing data association. Loop closure detection between perceptual inputs of different robotic agents is easily compromised in the context of perceptual aliasing, or when perspectives differ significantly. For this reason, direct mutual observation among robots is a powerful way to connect partial SLAM graphs, but often relies on the presence of calibrated arrays of fiducial markers (e.g., AprilTag arrays), which severely limits the range of observations and frequently fails under sharp lighting conditions, e.g., reflections or overexposure. In this work, we propose a novel solution to this problem leveraging recent advances in Deep-Learning-based 6D pose estimation. We feature markerless pose estimation as part of a decentralized multi-robot SLAM system and demonstrate the benefit to the relative localization accuracy among the robotic team. The solution is validated experimentally on data recorded in a test field campaign on a planetary analogous environment.

  </details>



- **Automated Assessment of Kidney Ureteroscopy Exploration for Training**  
  Fangjie Li, Nicholas Kavoussi, Charan Mohan, Matthieu Chabanas, Jie Ying Wu  
  _2026-02-17_ · https://arxiv.org/abs/2602.15988v1  
  <details><summary>Abstract</summary>

  Purpose: Kidney ureteroscopic navigation is challenging with a steep learning curve. However, current clinical training has major deficiencies, as it requires one-on-one feedback from experts and occurs in the operating room (OR). Therefore, there is a need for a phantom training system with automated feedback to greatly \revision{expand} training opportunities. Methods: We propose a novel, purely ureteroscope video-based scope localization framework that automatically identifies calyces missed by the trainee in a phantom kidney exploration. We use a slow, thorough, prior exploration video of the kidney to generate a reference reconstruction. Then, this reference reconstruction can be used to localize any exploration video of the same phantom. Results: In 15 exploration videos, a total of 69 out of 74 calyces were correctly classified. We achieve < 4mm camera pose localization error. Given the reference reconstruction, the system takes 10 minutes to generate the results for a typical exploration (1-2 minute long). Conclusion: We demonstrate a novel camera localization framework that can provide accurate and automatic feedback for kidney phantom explorations. We show its ability as a valid tool that enables out-of-OR training without requiring supervision from an expert.

  </details>



- **SpecFuse: A Spectral-Temporal Fusion Predictive Control Framework for UAV Landing on Oscillating Marine Platforms**  
  Haichao Liu, Yufeng Hu, Shuang Wang, Kangjun Guo, Jun Ma, Jinni Zhou  
  _2026-02-17_ · https://arxiv.org/abs/2602.15633v1  
  <details><summary>Abstract</summary>

  Autonomous landing of Uncrewed Aerial Vehicles (UAVs) on oscillating marine platforms is severely constrained by wave-induced multi-frequency oscillations, wind disturbances, and prediction phase lags in motion prediction. Existing methods either treat platform motion as a general random process or lack explicit modeling of wave spectral characteristics, leading to suboptimal performance under dynamic sea conditions. To address these limitations, we propose SpecFuse: a novel spectral-temporal fusion predictive control framework that integrates frequency-domain wave decomposition with time-domain recursive state estimation for high-precision 6-DoF motion forecasting of Uncrewed Surface Vehicles (USVs). The framework explicitly models dominant wave harmonics to mitigate phase lags, refining predictions in real time via IMU data without relying on complex calibration. Additionally, we design a hierarchical control architecture featuring a sampling-based HPO-RRT* algorithm for dynamic trajectory planning under non-convex constraints and a learning-augmented predictive controller that fuses data-driven disturbance compensation with optimization-based execution. Extensive validations (2,000 simulations + 8 lake experiments) show our approach achieves a 3.2 cm prediction error, 4.46 cm landing deviation, 98.7% / 87.5% success rates (simulation / real-world), and 82 ms latency on embedded hardware, outperforming state-of-the-art methods by 44%-48% in accuracy. Its robustness to wave-wind coupling disturbances supports critical maritime missions such as search and rescue and environmental monitoring. All code, experimental configurations, and datasets will be released as open-source to facilitate reproducibility.

  </details>



- **One Agent to Guide Them All: Empowering MLLMs for Vision-and-Language Navigation via Explicit World Representation**  
  Zerui Li, Hongpei Zheng, Fangguo Zhao, Aidan Chan, Jian Zhou, Sihao Lin, Shijie Li, Qi Wu  
  _2026-02-17_ · https://arxiv.org/abs/2602.15400v1  
  <details><summary>Abstract</summary>

  A navigable agent needs to understand both high-level semantic instructions and precise spatial perceptions. Building navigation agents centered on Multimodal Large Language Models (MLLMs) demonstrates a promising solution due to their powerful generalization ability. However, the current tightly coupled design dramatically limits system performance. In this work, we propose a decoupled design that separates low-level spatial state estimation from high-level semantic planning. Unlike previous methods that rely on predefined, oversimplified textual maps, we introduce an interactive metric world representation that maintains rich and consistent information, allowing MLLMs to interact with and reason on it for decision-making. Furthermore, counterfactual reasoning is introduced to further elicit MLLMs' capacity, while the metric world representation ensures the physical validity of the produced actions. We conduct comprehensive experiments in both simulated and real-world environments. Our method establishes a new zero-shot state-of-the-art, achieving 48.8\% Success Rate (SR) in R2R-CE and 42.2\% in RxR-CE benchmarks. Furthermore, to validate the versatility of our metric representation, we demonstrate zero-shot sim-to-real transfer across diverse embodiments, including a wheeled TurtleBot 4 and a custom-built aerial drone. These real-world deployments verify that our decoupled framework serves as a robust, domain-invariant interface for embodied Vision-and-Language navigation.

  </details>


