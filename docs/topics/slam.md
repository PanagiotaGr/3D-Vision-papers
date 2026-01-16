# SLAM & Localization

_Updated: 2026-01-16 06:49 UTC_

Total papers shown: **8**


---

- **SurgGoal: Rethinking Surgical Planning Evaluation via Goal-Satisfiability**  
  Ruochen Li, Kun Yuan, Yufei Xia, Yue Zhou, Qingyu Lu, Weihang Li, Youxiang Zhu, Nassir Navab  
  _2026-01-15_ · https://arxiv.org/abs/2601.10455v1  
  <details><summary>Abstract</summary>

  Surgical planning integrates visual perception, long-horizon reasoning, and procedural knowledge, yet it remains unclear whether current evaluation protocols reliably assess vision-language models (VLMs) in safety-critical settings. Motivated by a goal-oriented view of surgical planning, we define planning correctness via phase-goal satisfiability, where plan validity is determined by expert-defined surgical rules. Based on this definition, we introduce a multicentric meta-evaluation benchmark with valid procedural variations and invalid plans containing order and content errors. Using this benchmark, we show that sequence similarity metrics systematically misjudge planning quality, penalizing valid plans while failing to identify invalid ones. We therefore adopt a rule-based goal-satisfiability metric as a high-precision meta-evaluation reference to assess Video-LLMs under progressively constrained settings, revealing failures due to perception errors and under-constrained reasoning. Structural knowledge consistently improves performance, whereas semantic guidance alone is unreliable and benefits larger models only when combined with structural constraints.

  </details>



- **Online identification of nonlinear time-varying systems with uncertain information**  
  He Ren, Gaowei Yan, Hang Liu, Lifeng Cao, Zhijun Zhao, Gang Dang  
  _2026-01-15_ · https://arxiv.org/abs/2601.10379v1  
  <details><summary>Abstract</summary>

  Digital twins (DTs), serving as the core enablers for real-time monitoring and predictive maintenance of complex cyber-physical systems, impose critical requirements on their virtual models: high predictive accuracy, strong interpretability, and online adaptive capability. However, existing techniques struggle to meet these demands simultaneously: Bayesian methods excel in uncertainty quantification but lack model interpretability, while interpretable symbolic identification methods (e.g., SINDy) are constrained by their offline, batch-processing nature, which make real-time updates challenging. To bridge this semantic and computational gap, this paper proposes a novel Bayesian Regression-based Symbolic Learning (BRSL) framework. The framework formulates online symbolic discovery as a unified probabilistic state-space model. By incorporating sparse horseshoe priors, model selection is transformed into a Bayesian inference task, enabling simultaneous system identification and uncertainty quantification. Furthermore, we derive an online recursive algorithm with a forgetting factor and establish precise recursive conditions that guarantee the well-posedness of the posterior distribution. These conditions also function as real-time monitors for data utility, enhancing algorithmic robustness. Additionally, a rigorous convergence analysis is provided, demonstrating the convergence of parameter estimates under persistent excitation conditions. Case studies validate the effectiveness of the proposed framework in achieving interpretable, probabilistic prediction and online learning.

  </details>



- **FastStair: Learning to Run Up Stairs with Humanoid Robots**  
  Yan Liu, Tao Yu, Haolin Song, Hongbo Zhu, Nianzong Hu, Yuzhi Hao, Xiuyong Yao, Xizhe Zang, Hua Chen, Jie Zhao  
  _2026-01-15_ · https://arxiv.org/abs/2601.10365v1  
  <details><summary>Abstract</summary>

  Running up stairs is effortless for humans but remains extremely challenging for humanoid robots due to the simultaneous requirements of high agility and strict stability. Model-free reinforcement learning (RL) can generate dynamic locomotion, yet implicit stability rewards and heavy reliance on task-specific reward shaping tend to result in unsafe behaviors, especially on stairs; conversely, model-based foothold planners encode contact feasibility and stability structure, but enforcing their hard constraints often induces conservative motion that limits speed. We present FastStair, a planner-guided, multi-stage learning framework that reconciles these complementary strengths to achieve fast and stable stair ascent. FastStair integrates a parallel model-based foothold planner into the RL training loop to bias exploration toward dynamically feasible contacts and to pretrain a safety-focused base policy. To mitigate planner-induced conservatism and the discrepancy between low- and high-speed action distributions, the base policy was fine-tuned into speed-specialized experts and then integrated via Low-Rank Adaptation (LoRA) to enable smooth operation across the full commanded-speed range. We deploy the resulting controller on the Oli humanoid robot, achieving stable stair ascent at commanded speeds up to 1.65 m/s and traversing a 33-step spiral staircase (17 cm rise per step) in 12 s, demonstrating robust high-speed performance on long staircases. Notably, the proposed approach served as the champion solution in the Canton Tower Robot Run Up Competition.

  </details>



- **CHORAL: Traversal-Aware Planning for Safe and Efficient Heterogeneous Multi-Robot Routing**  
  David Morilla-Cabello, Eduardo Montijano  
  _2026-01-15_ · https://arxiv.org/abs/2601.10340v1  
  <details><summary>Abstract</summary>

  Monitoring large, unknown, and complex environments with autonomous robots poses significant navigation challenges, where deploying teams of heterogeneous robots with complementary capabilities can substantially improve both mission performance and feasibility. However, effectively modeling how different robotic platforms interact with the environment requires rich, semantic scene understanding. Despite this, existing approaches often assume homogeneous robot teams or focus on discrete task compatibility rather than continuous routing. Consequently, scene understanding is not fully integrated into routing decisions, limiting their ability to adapt to the environment and to leverage each robot's strengths. In this paper, we propose an integrated semantic-aware framework for coordinating heterogeneous robots. Starting from a reconnaissance flight, we build a metric-semantic map using open-vocabulary vision models and use it to identify regions requiring closer inspection and capability-aware paths for each platform to reach them. These are then incorporated into a heterogeneous vehicle routing formulation that jointly assigns inspection tasks and computes robot trajectories. Experiments in simulation and in a real inspection mission with three robotic platforms demonstrate the effectiveness of our approach in planning safer and more efficient routes by explicitly accounting for each platform's navigation capabilities. We release our framework, CHORAL, as open source to support reproducibility and deployment of diverse robot teams.

  </details>



- **The impact of tactile sensor configurations on grasp learning efficiency -- a comparative evaluation in simulation**  
  Eszter Birtalan, Miklós Koller  
  _2026-01-15_ · https://arxiv.org/abs/2601.10268v1  
  <details><summary>Abstract</summary>

  Tactile sensors are breaking into the field of robotics to provide direct information related to contact surfaces, including contact events, slip events and even texture identification. These events are especially important for robotic hand designs, including prosthetics, as they can greatly improve grasp stability. Most presently published robotic hand designs, however, implement them in vastly different densities and layouts on the hand surface, often reserving the majority of the available space. We used simulations to evaluate 6 different tactile sensor configurations with different densities and layouts, based on their impact on reinforcement learning. Our two-setup system allows for robust results that are not dependent on the use of a given physics simulator, robotic hand model or machine learning algorithm. Our results show setup-specific, as well as generalized effects across the 6 sensorized simulations, and we identify one configuration as consistently yielding the best performance across both setups. These results could help future research aimed at robotic hand designs, including prostheses.

  </details>



- **Proactive Local-Minima-Free Robot Navigation: Blending Motion Prediction with Safe Control**  
  Yifan Xue, Ze Zhang, Knut Åkesson, Nadia Figueroa  
  _2026-01-15_ · https://arxiv.org/abs/2601.10233v1  
  <details><summary>Abstract</summary>

  This work addresses the challenge of safe and efficient mobile robot navigation in complex dynamic environments with concave moving obstacles. Reactive safe controllers like Control Barrier Functions (CBFs) design obstacle avoidance strategies based only on the current states of the obstacles, risking future collisions. To alleviate this problem, we use Gaussian processes to learn barrier functions online from multimodal motion predictions of obstacles generated by neural networks trained with energy-based learning. The learned barrier functions are then fed into quadratic programs using modulated CBFs (MCBFs), a local-minimum-free version of CBFs, to achieve safe and efficient navigation. The proposed framework makes two key contributions. First, it develops a prediction-to-barrier function online learning pipeline. Second, it introduces an autonomous parameter tuning algorithm that adapts MCBFs to deforming, prediction-based barrier functions. The framework is evaluated in both simulations and real-world experiments, consistently outperforming baselines and demonstrating superior safety and efficiency in crowded dynamic environments.

  </details>



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


