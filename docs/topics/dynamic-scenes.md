# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-23 07:19 UTC_

Total papers shown: **2**


---

- **Learning Smooth Time-Varying Linear Policies with an Action Jacobian Penalty**  
  Zhaoming Xie, Kevin Karol, Jessica Hodgins  
  _2026-02-20_ · https://arxiv.org/abs/2602.18312v1  
  <details><summary>Abstract</summary>

  Reinforcement learning provides a framework for learning control policies that can reproduce diverse motions for simulated characters. However, such policies often exploit unnatural high-frequency signals that are unachievable by humans or physical robots, making them poor representations of real-world behaviors. Existing work addresses this issue by adding a reward term that penalizes a large change in actions over time. This term often requires substantial tuning efforts. We propose to use the action Jacobian penalty, which penalizes changes in action with respect to the changes in simulated state directly through auto differentiation. This effectively eliminates unrealistic high-frequency control signals without task specific tuning. While effective, the action Jacobian penalty introduces significant computational overhead when used with traditional fully connected neural network architectures. To mitigate this, we introduce a new architecture called a Linear Policy Net (LPN) that significantly reduces the computational burden for calculating the action Jacobian penalty during training. In addition, a LPN requires no parameter tuning, exhibits faster learning convergence compared to baseline methods, and can be more efficiently queried during inference time compared to a fully connected neural network. We demonstrate that a Linear Policy Net, combined with the action Jacobian penalty, is able to learn policies that generate smooth signals while solving a number of motion imitation tasks with different characteristics, including dynamic motions such as a backflip and various challenging parkour skills. Finally, we apply this approach to create policies for dynamic motions on a physical quadrupedal robot equipped with an arm.

  </details>



- **Spatio-temporal Decoupled Knowledge Compensator for Few-Shot Action Recognition**  
  Hongyu Qu, Xiangbo Shu, Rui Yan, Hailiang Gao, Wenguan Wang, Jinhui Tang  
  _2026-02-20_ · https://arxiv.org/abs/2602.18043v1  
  <details><summary>Abstract</summary>

  Few-Shot Action Recognition (FSAR) is a challenging task that requires recognizing novel action categories with a few labeled videos. Recent works typically apply semantically coarse category names as auxiliary contexts to guide the learning of discriminative visual features. However, such context provided by the action names is too limited to provide sufficient background knowledge for capturing novel spatial and temporal concepts in actions. In this paper, we propose DiST, an innovative Decomposition-incorporation framework for FSAR that makes use of decoupled Spatial and Temporal knowledge provided by large language models to learn expressive multi-granularity prototypes. In the decomposition stage, we decouple vanilla action names into diverse spatio-temporal attribute descriptions (action-related knowledge). Such commonsense knowledge complements semantic contexts from spatial and temporal perspectives. In the incorporation stage, we propose Spatial/Temporal Knowledge Compensators (SKC/TKC) to discover discriminative object-level and frame-level prototypes, respectively. In SKC, object-level prototypes adaptively aggregate important patch tokens under the guidance of spatial knowledge. Moreover, in TKC, frame-level prototypes utilize temporal attributes to assist in inter-frame temporal relation modeling. These learned prototypes thus provide transparency in capturing fine-grained spatial details and diverse temporal patterns. Experimental results show DiST achieves state-of-the-art results on five standard FSAR datasets.

  </details>


