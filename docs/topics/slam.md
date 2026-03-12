# SLAM & Localization

_Updated: 2026-03-12 07:11 UTC_

Total papers shown: **10**


---

- **Semantic Landmark Particle Filter for Robot Localisation in Vineyards**  
  Rajitha de Silva, Jonathan Cox, James R. Heselden, Marija Popović, Cesar Cadena, Riccardo Polvara  
  _2026-03-11_ · https://arxiv.org/abs/2603.10847v1  
  <details><summary>Abstract</summary>

  Reliable localisation in vineyards is hindered by row-level perceptual aliasing: parallel crop rows produce nearly identical LiDAR observations, causing geometry-only and vision-based SLAM systems to converge towards incorrect corridors, particularly during headland transitions. We present a Semantic Landmark Particle Filter (SLPF) that integrates trunk and pole landmark detections with 2D LiDAR within a probabilistic localisation framework. Detected trunks are converted into semantic walls, forming structural row boundaries embedded in the measurement model to improve discrimination between adjacent rows. GNSS is incorporated as a lightweight prior that stabilises localisation when semantic observations are sparse. Field experiments in a 10-row vineyard demonstrate consistent improvements over geometry-only (AMCL), vision-based (RTAB-Map), and GNSS baselines. Compared to AMCL, SLPF reduces Absolute Pose Error by 22% and 65% across two traversal directions; relative to a NoisyGNSS baseline, APE decreases by 65% and 61%. Row correctness improves from 0.67 to 0.73, while mean cross-track error decreases from 1.40 m to 1.26 m. These results show that embedding row-level structural semantics within the measurement model enables robust localisation in highly repetitive outdoor agricultural environments.

  </details>



- **Adaptive Manipulation Potential and Haptic Estimation for Tool-Mediated Interaction**  
  Lin Yang, Anirvan Dutta, Yuan Ji, Yanxin Zhou, Shilin Shan, Lv Chen, Etienne Burdet, Domenico Campolo  
  _2026-03-11_ · https://arxiv.org/abs/2603.10352v1  
  <details><summary>Abstract</summary>

  Achieving human-level dexterity in contact-rich, tool-mediated manipulation remains a significant challenge due to visual occlusion and the underdetermined nature of haptic sensing. This paper introduces a parameterized Equilibrium Manifold (EM) as a unified representation for tool-mediated interaction, and develops a closed-loop framework that integrates haptic estimation, online planning, and adaptive stiffness control. We establish a physical-geometric duality using an adaptive manipulation potential incorporating a differentiable contact model, which induces the manifold's geometric structure and ensures that complex physical interactions are encapsulated as continuous operations on the EM. Within this framework, we reformulate haptic estimation as a manifold parameter estimation problem. Specifically, a hybrid inference strategy (haptic SLAM) is employed in which discrete object shapes are classified via particle filtering, while the continuous object pose is estimated using analytical gradients for efficient optimization. By continuously updating the parameters of the manipulation potential, the framework dynamically reshapes the induced EM to guide online trajectory replanning and implement uncertainty-aware impedance control, thereby closing the perception-action loop. The system is validated through simulation and over 260 real-world screw-loosening trials. Experimental results demonstrate robust identification and manipulation success in standard scenarios while maintaining accurate tracking. Furthermore, ablation studies confirm that haptic SLAM and uncertainty-aware stiffness modulation outperform fixed impedance baselines, effectively preventing jamming during tight tolerance interactions.

  </details>



- **ReCoSplat: Autoregressive Feed-Forward Gaussian Splatting Using Render-and-Compare**  
  Freeman Cheng, Botao Ye, Xueting Li, Junqi You, Fangneng Zhan, Ming-Hsuan Yang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09968v1  
  <details><summary>Abstract</summary>

  Online novel view synthesis remains challenging, requiring robust scene reconstruction from sequential, often unposed, observations. We present ReCoSplat, an autoregressive feed-forward Gaussian Splatting model supporting posed or unposed inputs, with or without camera intrinsics. While assembling local Gaussians using camera poses scales better than canonical-space prediction, it creates a dilemma during training: using ground-truth poses ensures stability but causes a distribution mismatch when predicted poses are used at inference. To address this, we introduce a Render-and-Compare (ReCo) module. ReCo renders the current reconstruction from the predicted viewpoint and compares it with the incoming observation, providing a stable conditioning signal that compensates for pose errors. To support long sequences, we propose a hybrid KV cache compression strategy combining early-layer truncation with chunk-level selective retention, reducing the KV cache size by over 90% for 100+ frames. ReCoSplat achieves state-of-the-art performance across different input settings on both in- and out-of-distribution benchmarks. Code and pretrained models will be released. Our project page is at https://freemancheng.com/ReCoSplat .

  </details>



- **NanoBench: A Multi-Task Benchmark Dataset for Nano-Quadrotor System Identification, Control, and State Estimation**  
  Syed Izzat Ullah, Jose Baca  
  _2026-03-10_ · https://arxiv.org/abs/2603.09908v1  
  <details><summary>Abstract</summary>

  Existing aerial-robotics benchmarks target vehicles from hundreds of grams to several kilograms and typically expose only high-level state data. They omit the actuator-level signals required to study nano-scale quadrotors, where low-Reynolds number aerodynamics, coreless DC motor nonlinearities, and severe computational constraints invalidate models and controllers developed for larger vehicles. We introduce NanoBench, an open-source multi-task benchmark collected on the commercially available Crazyflie 2.1 nano-quadrotor (takeoff weight 27 g) in a Vicon motion capture arena. The dataset contains over 170 flight trajectories spanning hover, multi-frequency excitation, standard tracking, and aggressive maneuvers across multiple speed regimes. Each trajectory provides synchronized Vicon ground truth, raw IMU data, onboard extended Kalman filter estimates, PID controller internals, and motor PWM commands at 100 Hz, alongside battery telemetry at 10 Hz, aligned with sub-0.5 ms consistency. NanoBench defines standardized evaluation protocols, train/test splits, and open-source baselines for three tasks: nonlinear system identification, closed-loop controller benchmarking, and onboard state estimation assessment. To our knowledge, it is the first public dataset to jointly provide actuator commands, controller internals, and estimator outputs with millimeter-accurate ground truth on a commercially available nano-scale aerial platform.

  </details>



- **ConfCtrl: Enabling Precise Camera Control in Video Diffusion via Confidence-Aware Interpolation**  
  Liudi Yang, George Eskandar, Fengyi Shen, Mohammad Altillawi, Yang Bai, Chi Zhang, Ziyuan Liu, Abhinav Valada  
  _2026-03-10_ · https://arxiv.org/abs/2603.09819v1  
  <details><summary>Abstract</summary>

  We address the challenge of novel view synthesis from only two input images under large viewpoint changes. Existing regression-based methods lack the capacity to reconstruct unseen regions, while camera-guided diffusion models often deviate from intended trajectories due to noisy point cloud projections or insufficient conditioning from camera poses. To address these issues, we propose ConfCtrl, a confidence-aware video interpolation framework that enables diffusion models to follow prescribed camera poses while completing unseen regions. ConfCtrl initializes the diffusion process by combining a confidence-weighted projected point cloud latent with noise as the conditioning input. It then applies a Kalman-inspired predict-update mechanism, treating the projected point cloud as a noisy measurement and using learned residual corrections to balance pose-driven predictions with noisy geometric observations. This allows the model to rely on reliable projections while down-weighting uncertain regions, yielding stable, geometry-aware generation. Experiments on multiple datasets show that ConfCtrl produces geometrically consistent and visually plausible novel views, effectively reconstructing occluded regions under large viewpoint changes.

  </details>



- **Efficient and robust control with spikes that constrain free energy**  
  André Urbano, Pablo Lanillos, Sander Keemink  
  _2026-03-10_ · https://arxiv.org/abs/2603.09729v1  
  <details><summary>Abstract</summary>

  Animal brains exhibit remarkable efficiency in perception and action, while being robust to both external and internal perturbations. The means by which brains accomplish this remains, for now, poorly understood, hindering our understanding of animal and human cognition, as well as our own implementation of efficient algorithms for control of dynamical systems.A potential candidate for a robust mechanism of state estimation and action computation is the free energy principle, but existing implementations of this principle have largely relied on conventional, biologically implausible approaches without spikes. We propose a novel, efficient, and robust spiking control framework with realistic biological characteristics. The resulting networks function as free energy constrainers, in which neurons only fire if they reduce the free energy of their internal representation. The networks offer efficient operation through highly sparse activity while matching performance with other similar spiking frameworks, and have high resilience against both external (e.g. sensory noise or collisions) and internal perturbations (e.g. synaptic noise and delays or neuron silencing) that such a network would be faced with when deployed by either an organism or an engineer. Overall, our work provides a novel mathematical account for spiking control through constraining free energy, providing both better insight into how brain networks might leverage their spiking substrate and a new route for implementing efficient control algorithms in neuromorphic hardware.

  </details>



- **VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM**  
  Anh Thuan Tran, Jana Kosecka  
  _2026-03-10_ · https://arxiv.org/abs/2603.09673v1  
  <details><summary>Abstract</summary>

  Simultaneous Localization and Mapping (SLAM) with 3D Gaussian Splatting (3DGS) enables fast, differentiable rendering and high-fidelity reconstruction across diverse real-world scenes. However, existing 3DGS-SLAM approaches handle measurement reliability implicitly, making pose estimation and global alignment susceptible to drift in low-texture regions, transparent surfaces, or areas with complex reflectance properties. To this end, we introduce VarSplat, an uncertainty-aware 3DGS-SLAM system that explicitly learns per-splat appearance variance. By using the law of total variance with alpha compositing, we then render differentiable per-pixel uncertainty map via efficient, single-pass rasterization. This map guides tracking, submap registration, and loop detection toward focusing on reliable regions and contributes to more stable optimization. Experimental results on Replica (synthetic) and TUM-RGBD, ScanNet, and ScanNet++ (real-world) show that VarSplat improves robustness and achieves competitive or superior tracking, mapping, and novel view synthesis rendering compared to existing studies for dense RGB-D SLAM.

  </details>



- **X-GS: An Extensible Open Framework Unifying 3DGS Architectures with Downstream Multimodal Models**  
  Yueen Ma, Irwin King  
  _2026-03-10_ · https://arxiv.org/abs/2603.09632v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, subsequently extending into numerous spatial AI applications. However, most existing 3DGS methods are isolated, focusing on specific domains such as online SLAM, semantic enrichment, or 3DGS for unposed images. In this paper, we introduce X-GS, an extensible open framework that unifies a broad range of techniques to enable real-time 3DGS-based online SLAM enriched with semantics, bridging the gap to downstream multimodal models. At the core of X-GS is a highly efficient pipeline called X-GS-Perceiver, capable of taking unposed RGB (or optionally RGB-D) video streams as input to co-optimize geometry and poses, and distill high-dimensional semantic features from vision foundation models into the 3D Gaussians. We achieve real-time performance through a novel online Vector Quantization (VQ) module, a GPU-accelerated grid-sampling scheme, and a highly parallelized pipeline design. The semantic 3D Gaussians can then be utilized by vision-language models within the X-GS-Thinker component, enabling downstream tasks such as object detection, zero-shot caption generation, and potentially embodied tasks. Experimental results on real-world datasets showcase the efficacy, efficiency, and newly unlocked multimodal capabilities of the X-GS framework.

  </details>



- **Towards Terrain-Aware Safe Locomotion for Quadrupedal Robots Using Proprioceptive Sensing**  
  Peiyu Yang, Jiatao Ding, Wei Pan, Claudio Semini, Cosimo Della Santina  
  _2026-03-10_ · https://arxiv.org/abs/2603.09585v1  
  <details><summary>Abstract</summary>

  Achieving safe quadrupedal locomotion in real-world environments has attracted much attention in recent years. When walking over uneven terrain, achieving reliable estimation and realising safety-critical control based on the obtained information is still an open question. To address this challenge, especially for low-cost robots equipped solely with proprioceptive sensors (e.g., IMUs, joint encoders, and contact force sensors), this work first presents an estimation framework that generates a 2.5-D terrain map and extracts support plane parameters, which are then integrated into contact and state estimation. Then, we integrate this estimation framework into a safety-critical control pipeline by formulating control barrier functions that provide rigorous safety guarantees. Experiments demonstrate that the proposed terrain estimation method provides smooth terrain representations. Moreover, the coupled estimation framework of terrain, state, and contact reduces the mean absolute error of base position estimation by 64.8%, decreases the estimation variance by 47.2%, and improves the robustness of contact estimation compared to a decoupled framework. The terrain-informed CBFs integrate historical terrain information and current proprioceptive measurements to ensure global safety by keeping the robot out of hazardous areas and local safety by preventing body-terrain collision, relying solely on proprioceptive sensing.

  </details>



- **SCDP: Learning Humanoid Locomotion from Partial Observations via Mixed-Observation Distillation**  
  Milo Carroll, Tianhu Peng, Lingfan Bao, Chengxu Zhou, Zhibin Li  
  _2026-03-10_ · https://arxiv.org/abs/2603.09574v1  
  <details><summary>Abstract</summary>

  Distilling humanoid locomotion control from offline datasets into deployable policies remains a challenge, as existing methods rely on privileged full-body states that require complex and often unreliable state estimation. We present Sensor-Conditioned Diffusion Policies (SCDP) that enables humanoid locomotion using only onboard sensors, eliminating the need for explicit state estimation. SCDP decouples sensing from supervision through mixed-observation training: diffusion model conditions on sensor histories while being supervised to predict privileged future state-action trajectories, enforcing the model to infer the motion dynamics under partial observability. We further develop restricted denoising, context distribution alignment, and context-aware attention masking to encourage implicit state estimation within the model and to prevent train-deploy mismatch. We validate SCDP on velocity-commanded locomotion and motion reference tracking tasks. In simulation, SCDP achieves near-perfect success on velocity control (99-100%) and 93% tracking success in AMASS test set, performing comparable to privileged baselines while using only onboard sensors. Finally, we deploy the trained policy on a real G1 humanoid at 50 Hz, demonstrating robust real robot locomotion without external sensing or state estimation.

  </details>


