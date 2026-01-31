# SLAM & Localization

_Updated: 2026-01-31 06:56 UTC_

Total papers shown: **3**


---

- **Disentangling perception and reasoning for improving data efficiency in learning cloth manipulation without demonstrations**  
  Donatien Delehelle, Fei Chen, Darwin Caldwell  
  _2026-01-29_ · https://arxiv.org/abs/2601.21713v1  
  <details><summary>Abstract</summary>

  Cloth manipulation is a ubiquitous task in everyday life, but it remains an open challenge for robotics. The difficulties in developing cloth manipulation policies are attributed to the high-dimensional state space, complex dynamics, and high propensity to self-occlusion exhibited by fabrics. As analytical methods have not been able to provide robust and general manipulation policies, reinforcement learning (RL) is considered a promising approach to these problems. However, to address the large state space and complex dynamics, data-based methods usually rely on large models and long training times. The resulting computational cost significantly hampers the development and adoption of these methods. Additionally, due to the challenge of robust state estimation, garment manipulation policies often adopt an end-to-end learning approach with workspace images as input. While this approach enables a conceptually straightforward sim-to-real transfer via real-world fine-tuning, it also incurs a significant computational cost by training agents on a highly lossy representation of the environment state. This paper questions this common design choice by exploring an efficient and modular approach to RL for cloth manipulation. We show that, through careful design choices, model size and training time can be significantly reduced when learning in simulation. Furthermore, we demonstrate how the resulting simulation-trained model can be transferred to the real world. We evaluate our approach on the SoftGym benchmark and achieve significant performance improvements over available baselines on our task, while using a substantially smaller model.

  </details>



- **IROS: A Dual-Process Architecture for Real-Time VLM-Based Indoor Navigation**  
  Joonhee Lee, Hyunseung Shin, Jeonggil Ko  
  _2026-01-29_ · https://arxiv.org/abs/2601.21506v1  
  <details><summary>Abstract</summary>

  Indoor mobile robot navigation requires fast responsiveness and robust semantic understanding, yet existing methods struggle to provide both. Classical geometric approaches such as SLAM offer reliable localization but depend on detailed maps and cannot interpret human-targeted cues (e.g., signs, room numbers) essential for indoor reasoning. Vision-Language-Action (VLA) models introduce semantic grounding but remain strictly reactive, basing decisions only on visible frames and failing to anticipate unseen intersections or reason about distant textual cues. Vision-Language Models (VLMs) provide richer contextual inference but suffer from high computational latency, making them unsuitable for real-time operation on embedded platforms. In this work, we present IROS, a real-time navigation framework that combines VLM-level contextual reasoning with the efficiency of lightweight perceptual modules on low-cost, on-device hardware. Inspired by Dual Process Theory, IROS separates fast reflexive decisions (System One) from slow deliberative reasoning (System Two), invoking the VLM only when necessary. Furthermore, by augmenting compact VLMs with spatial and textual cues, IROS delivers robust, human-like navigation with minimal latency. Across five real-world buildings, IROS improves decision accuracy and reduces latency by 66% compared to continuous VLM-based navigation.

  </details>



- **Multi-Robot Decentralized Collaborative SLAM in Planetary Analogue Environments: Dataset, Challenges, and Lessons Learned**  
  Pierre-Yves Lajoie, Karthik Soma, Haechan Mark Bong, Alice Lemieux-Bourque, Rongge Zhang, Vivek Shankar Varadharajan, Giovanni Beltrame  
  _2026-01-28_ · https://arxiv.org/abs/2601.21063v1  
  <details><summary>Abstract</summary>

  Decentralized collaborative simultaneous localization and mapping (C-SLAM) is essential to enable multirobot missions in unknown environments without relying on preexisting localization and communication infrastructure. This technology is anticipated to play a key role in the exploration of the Moon, Mars, and other planets. In this article, we share insights and lessons learned from C-SLAM experiments involving three robots operating on a Mars analogue terrain and communicating over an ad hoc network. We examine the impact of limited and intermittent communication on C-SLAM performance, as well as the unique localization challenges posed by planetary-like environments. Additionally, we introduce a novel dataset collected during our experiments, which includes real-time peer-to-peer inter-robot throughput and latency measurements. This dataset aims to support future research on communication-constrained, decentralized multirobot operations.

  </details>


