# SLAM & Localization

_Updated: 2026-01-28 06:53 UTC_

Total papers shown: **6**


---

- **VGGT-SLAM 2.0: Real time Dense Feed-forward Scene Reconstruction**  
  Dominic Maggio, Luca Carlone  
  _2026-01-27_ · https://arxiv.org/abs/2601.19887v1  
  <details><summary>Abstract</summary>

  We present VGGT-SLAM 2.0, a real time RGB feed-forward SLAM system which substantially improves upon VGGT-SLAM for incrementally aligning submaps created from VGGT. Firstly, we remove high-dimensional 15-degree-of-freedom drift and planar degeneracy from VGGT-SLAM by creating a new factor graph design while still addressing the reconstruction ambiguity of VGGT given unknown camera intrinsics. Secondly, by studying the attention layers of VGGT, we show that one of the layers is well suited to assist in image retrieval verification for free without additional training, which enables both rejecting false positive matches and allows for completing more loop closures. Finally, we conduct a suite of experiments which includes showing VGGT-SLAM 2.0 can easily be adapted for open-set object detection and demonstrating real time performance while running online onboard a ground robot using a Jetson Thor. We also test in environments ranging from cluttered indoor apartments and office scenes to a 4,200 square foot barn, and we also demonstrate VGGT-SLAM 2.0 achieves the highest accuracy on the TUM dataset with about 23 percent less pose error than VGGT-SLAM. Code will be released upon publication.

  </details>



- **The S3LI Vulcano Dataset: A Dataset for Multi-Modal SLAM in Unstructured Planetary Environments**  
  Riccardo Giubilato, Marcus Gerhard Müller, Marco Sewtz, Laura Alejandra Encinar Gonzalez, John Folkesson, Rudolph Triebel  
  _2026-01-27_ · https://arxiv.org/abs/2601.19557v1  
  <details><summary>Abstract</summary>

  We release the S3LI Vulcano dataset, a multi-modal dataset towards development and benchmarking of Simultaneous Localization and Mapping (SLAM) and place recognition algorithms that rely on visual and LiDAR modalities. Several sequences are recorded on the volcanic island of Vulcano, from the Aeolian Islands in Sicily, Italy. The sequences provide users with data from a variety of environments, textures and terrains, including basaltic or iron-rich rocks, geological formations from old lava channels, as well as dry vegetation and water. The data (rmc.dlr.de/s3li_dataset) is accompanied by an open source toolkit (github.com/DLR-RM/s3li-toolkit) providing tools for generating ground truth poses as well as preparation of labelled samples for place recognition tasks.

  </details>



- **Fast Converging 3D Gaussian Splatting for 1-Minute Reconstruction**  
  Ziyu Zhang, Tianle Liu, Diantao Tu, Shuhan Shen  
  _2026-01-27_ · https://arxiv.org/abs/2601.19489v1  
  <details><summary>Abstract</summary>

  We present a fast 3DGS reconstruction pipeline designed to converge within one minute, developed for the SIGGRAPH Asia 3DGS Fast Reconstruction Challenge. The challenge consists of an initial round using SLAM-generated camera poses (with noisy trajectories) and a final round using COLMAP poses (highly accurate). To robustly handle these heterogeneous settings, we develop a two-stage solution. In the first round, we use reverse per-Gaussian parallel optimization and compact forward splatting based on Taming-GS and Speedy-splat, load-balanced tiling, an anchor-based Neural-Gaussian representation enabling rapid convergence with fewer learnable parameters, initialization from monocular depth and partially from feed-forward 3DGS models, and a global pose refinement module for noisy SLAM trajectories. In the final round, the accurate COLMAP poses change the optimization landscape; we disable pose refinement, revert from Neural-Gaussians back to standard 3DGS to eliminate MLP inference overhead, introduce multi-view consistency-guided Gaussian splitting inspired by Fast-GS, and introduce a depth estimator to supervise the rendered depth. Together, these techniques enable high-fidelity reconstruction under a strict one-minute budget. Our method achieved the top performance with a PSNR of 28.43 and ranked first in the competition.

  </details>



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


