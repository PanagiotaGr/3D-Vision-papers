# SLAM & Localization

_Updated: 2026-01-15 10:49 UTC_

Total papers shown: **3**


---

- **CLARE: Continual Learning for Vision-Language-Action Models via Autonomous Adapter Routing and Expansion**  
  Ralf Römer, Yi Zhang, Angela P. Schoellig  
  _2026-01-14_ · https://arxiv.org/abs/2601.09512v1  
  <details><summary>Abstract</summary>

  To teach robots complex manipulation tasks, it is now a common practice to fine-tune a pre-trained vision-language-action model (VLA) on task-specific data. However, since this recipe updates existing representations, it is unsuitable for long-term operation in the real world, where robots must continually adapt to new tasks and environments while retaining the knowledge they have already acquired. Existing continual learning methods for robotics commonly require storing previous data (exemplars), struggle with long task sequences, or rely on task identifiers for deployment. To address these limitations, we propose CLARE, a general, parameter-efficient framework for exemplar-free continual learning with VLAs. CLARE introduces lightweight modular adapters into selected feedforward layers and autonomously expands the model only where necessary when learning a new task, guided by layer-wise feature similarity. During deployment, an autoencoder-based routing mechanism dynamically activates the most relevant adapters without requiring task labels. Through extensive experiments on the LIBERO benchmark, we show that CLARE achieves high performance on new tasks without catastrophic forgetting of earlier tasks, significantly outperforming even exemplar-based methods. Code and data are available at https://tum-lsy.github.io/clare.

  </details>



- **Data Scaling for Navigation in Unknown Environments**  
  Lauri Suomela, Naoki Takahata, Sasanka Kuruppu Arachchige, Harry Edelman, Joni-Kristian Kämäräinen  
  _2026-01-14_ · https://arxiv.org/abs/2601.09444v1  
  <details><summary>Abstract</summary>

  Generalization of imitation-learned navigation policies to environments unseen in training remains a major challenge. We address this by conducting the first large-scale study of how data quantity and data diversity affect real-world generalization in end-to-end, map-free visual navigation. Using a curated 4,565-hour crowd-sourced dataset collected across 161 locations in 35 countries, we train policies for point goal navigation and evaluate their closed-loop control performance on sidewalk robots operating in four countries, covering 125 km of autonomous driving. Our results show that large-scale training data enables zero-shot navigation in unknown environments, approaching the performance of policies trained with environment-specific demonstrations. Critically, we find that data diversity is far more important than data quantity. Doubling the number of geographical locations in a training set decreases navigation errors by ~15%, while performance benefit from adding data from existing locations saturates with very little data. We also observe that, with noisy crowd-sourced data, simple regression-based models outperform generative and sequence-based architectures. We release our policies, evaluation setup and example videos on the project page.

  </details>



- **ReflexDiffusion: Reflection-Enhanced Trajectory Planning for High-lateral-acceleration Scenarios in Autonomous Driving**  
  Xuemei Yao, Xiao Yang, Jianbin Sun, Liuwei Xie, Xuebin Shao, Xiyu Fang, Hang Su, Kewei Yang  
  _2026-01-14_ · https://arxiv.org/abs/2601.09377v1  
  <details><summary>Abstract</summary>

  Generating safe and reliable trajectories for autonomous vehicles in long-tail scenarios remains a significant challenge, particularly for high-lateral-acceleration maneuvers such as sharp turns, which represent critical safety situations. Existing trajectory planners exhibit systematic failures in these scenarios due to data imbalance. This results in insufficient modelling of vehicle dynamics, road geometry, and environmental constraints in high-risk situations, leading to suboptimal or unsafe trajectory prediction when vehicles operate near their physical limits. In this paper, we introduce ReflexDiffusion, a novel inference-stage framework that enhances diffusion-based trajectory planners through reflective adjustment. Our method introduces a gradient-based adjustment mechanism during the iterative denoising process: after each standard trajectory update, we compute the gradient between the conditional and unconditional noise predictions to explicitly amplify critical conditioning signals, including road curvature and lateral vehicle dynamics. This amplification enforces strict adherence to physical constraints, particularly improving stability during high-lateral-acceleration maneuvers where precise vehicle-road interaction is paramount. Evaluated on the nuPlan Test14-hard benchmark, ReflexDiffusion achieves a 14.1% improvement in driving score for high-lateral-acceleration scenarios over the state-of-the-art (SOTA) methods. This demonstrates that inference-time trajectory optimization can effectively compensate for training data sparsity by dynamically reinforcing safety-critical constraints near handling limits. The framework's architecture-agnostic design enables direct deployment to existing diffusion-based planners, offering a practical solution for improving autonomous vehicle safety in challenging driving conditions.

  </details>


