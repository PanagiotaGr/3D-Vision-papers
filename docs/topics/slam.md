# SLAM & Localization

_Updated: 2026-01-27 06:53 UTC_

Total papers shown: **3**


---

- **Low Cost, High Efficiency: LiDAR Place Recognition in Vineyards with Matryoshka Representation Learning**  
  Judith Vilella-Cantos, Mauro Martini, Marcello Chiaberge, Mónica Ballesta, David Valiente  
  _2026-01-26_ · https://arxiv.org/abs/2601.18714v1  
  <details><summary>Abstract</summary>

  Localization in agricultural environments is challenging due to their unstructured nature and lack of distinctive landmarks. Although agricultural settings have been studied in the context of object classification and segmentation, the place recognition task for mobile robots is not trivial in the current state of the art. In this study, we propose MinkUNeXt-VINE, a lightweight, deep-learning-based method that surpasses state-of-the-art methods in vineyard environments thanks to its pre-processing and Matryoshka Representation Learning multi-loss approach. Our method prioritizes enhanced performance with low-cost, sparse LiDAR inputs and lower-dimensionality outputs to ensure high efficiency in real-time scenarios. Additionally, we present a comprehensive ablation study of the results on various evaluation cases and two extensive long-term vineyard datasets employing different LiDAR sensors. The results demonstrate the efficiency of the trade-off output produced by this approach, as well as its robust performance on low-cost and low-resolution input data. The code is publicly available for reproduction.

  </details>



- **Attention-Based Neural-Augmented Kalman Filter for Legged Robot State Estimation**  
  Seokju Lee, Kyung-Soo Kim  
  _2026-01-26_ · https://arxiv.org/abs/2601.18569v1  
  <details><summary>Abstract</summary>

  In this letter, we propose an Attention-Based Neural-Augmented Kalman Filter (AttenNKF) for state estimation in legged robots. Foot slip is a major source of estimation error: when slip occurs, kinematic measurements violate the no-slip assumption and inject bias during the update step. Our objective is to estimate this slip-induced error and compensate for it. To this end, we augment an Invariant Extended Kalman Filter (InEKF) with a neural compensator that uses an attention mechanism to infer error conditioned on foot-slip severity and then applies this estimate as a post-update compensation to the InEKF state (i.e., after the filter update). The compensator is trained in a latent space, which aims to reduce sensitivity to raw input scales and encourages structured slip-conditioned compensations, while preserving the InEKF recursion. Experiments demonstrate improved performance compared to existing legged-robot state estimators, particularly under slip-prone conditions.

  </details>



- **Co-PLNet: A Collaborative Point-Line Network for Prompt-Guided Wireframe Parsing**  
  Chao Wang, Xuanying Li, Cheng Dai, Jinglei Feng, Yuxiang Luo, Yuqi Ouyang, Hao Qin  
  _2026-01-26_ · https://arxiv.org/abs/2601.18252v1  
  <details><summary>Abstract</summary>

  Wireframe parsing aims to recover line segments and their junctions to form a structured geometric representation useful for downstream tasks such as Simultaneous Localization and Mapping (SLAM). Existing methods predict lines and junctions separately and reconcile them post-hoc, causing mismatches and reduced robustness. We present Co-PLNet, a point-line collaborative framework that exchanges spatial cues between the two tasks, where early detections are converted into spatial prompts via a Point-Line Prompt Encoder (PLP-Encoder), which encodes geometric attributes into compact and spatially aligned maps. A Cross-Guidance Line Decoder (CGL-Decoder) then refines predictions with sparse attention conditioned on complementary prompts, enforcing point-line consistency and efficiency. Experiments on Wireframe and YorkUrban show consistent improvements in accuracy and robustness, together with favorable real-time efficiency, demonstrating our effectiveness for structured geometry perception.

  </details>


