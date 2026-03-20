# NeRF & Neural Radiance Fields

_Updated: 2026-03-20 07:11 UTC_

Total papers shown: **6**


---

- **Rethinking Vector Field Learning for Generative Segmentation**  
  Chaoyang Wang, Yaobo Liang, Boci Peng, Fan Duan, Jingdong Wang, Yunhai Tong  
  _2026-03-19_ · https://arxiv.org/abs/2603.19218v1  
  <details><summary>Abstract</summary>

  Taming diffusion models for generative segmentation has attracted increasing attention. While existing approaches primarily focus on architectural tweaks or training heuristics, there remains a limited understanding of the intrinsic mismatch between continuous flow matching objectives and discrete perception tasks. In this work, we revisit diffusion segmentation from the perspective of vector field learning. We identify two key limitations of the commonly used flow matching objective: gradient vanishing and trajectory traversing, which result in slow convergence and poor class separation. To tackle these issues, we propose a principled vector field reshaping strategy that augments the learned velocity field with a detached distance-aware correction term. This correction introduces both attractive and repulsive interactions, enhancing gradient magnitudes near centroids while preserving the original diffusion training framework. Furthermore, we design a computationally efficient, quasi-random category encoding scheme inspired by Kronecker sequences, which integrates seamlessly with an end-to-end pixel neural field framework for pixel-level semantic alignment. Extensive experiments consistently demonstrate significant improvements over vanilla flow matching approaches, substantially narrowing the performance gap between generative segmentation and strong discriminative specialists.

  </details>



- **VesselTok: Tokenizing Vessel-like 3D Biomedical Graph Representations for Reconstruction and Generation**  
  Chinmay Prabhakar, Bastian Wittmann, Tamaz Amiranashvili, Paul Büschl, Ezequiel de la Rosa, Julian McGinnis, Benedikt Wiestler, Bjoern Menze, Suprosanna Shit  
  _2026-03-19_ · https://arxiv.org/abs/2603.18797v1  
  <details><summary>Abstract</summary>

  Spatial graphs provide a lightweight and elegant representation of curvilinear anatomical structures such as blood vessels, lung airways, and neuronal networks. Accurately modeling these graphs is crucial in clinical and (bio-)medical research. However, the high spatial resolution of large networks drastically increases their complexity, resulting in significant computational challenges. In this work, we aim to tackle these challenges by proposing VesselTok, a framework that approaches spatially dense graphs from a parametric shape perspective to learn latent representations (tokens). VesselTok leverages centerline points with a pseudo radius to effectively encode tubular geometry. Specifically, we learn a novel latent representation conditioned on centerline points to encode neural implicit representations of vessel-like, tubular structures. We demonstrate VesselTok's performance across diverse anatomies, including lung airways, lung vessels, and brain vessels, highlighting its ability to robustly encode complex topologies. To prove the effectiveness of VesselTok's learnt latent representations, we show that they (i) generalize to unseen anatomies, (ii) support generative modeling of plausible anatomical graphs, and (iii) transfer effectively to downstream inverse problems, such as link prediction.

  </details>



- **Fast and Generalizable NeRF Architecture Selection for Satellite Scene Reconstruction**  
  Devjyoti Chakraborty, Zaki Sukma, Rakandhiya D. Rachmanto, Kriti Ghosh, In Kee Kim, Suchendra M. Bhandarkar, Lakshmish Ramaswamy, Nancy K. O'Hare, Deepak Mishra  
  _2026-03-18_ · https://arxiv.org/abs/2603.18306v1  
  <details><summary>Abstract</summary>

  Neural Radiance Fields (NeRF) have emerged as a powerful approach for photorealistic 3D reconstruction from multi-view images. However, deploying NeRF for satellite imagery remains challenging. Each scene requires individual training, and optimizing architectures via Neural Architecture Search (NAS) demands hours to days of GPU time. While existing approaches focus on architectural improvements, our SHAP analysis reveals that multi-view consistency, rather than model architecture, determines reconstruction quality. Based on this insight, we develop PreSCAN, a predictive framework that estimates NeRF quality prior to training using lightweight geometric and photometric descriptors. PreSCAN selects suitable architectures in < 30 seconds with < 1 dB prediction error, achieving 1000$\times$ speedup over NAS. We further demonstrate PreSCAN's deployment utility on edge platforms (Jetson Orin), where combining its predictions with offline cost profiling reduces inference power by 26% and latency by 43% with minimal quality loss. Experiments on DFC2019 datasets confirm that PreSCAN generalizes across diverse satellite scenes without retraining.

  </details>



- **Semantic Segmentation and Depth Estimation for Real-Time Lunar Surface Mapping Using 3D Gaussian Splatting**  
  Guillem Casadesus Vila, Adam Dai, Grace Gao  
  _2026-03-18_ · https://arxiv.org/abs/2603.18218v1  
  <details><summary>Abstract</summary>

  Navigation and mapping on the lunar surface require robust perception under challenging conditions, including poorly textured environments, high-contrast lighting, and limited computational resources. This paper presents a real-time mapping framework that integrates dense perception models with a 3D Gaussian Splatting (3DGS) representation. We first benchmark several models on synthetic datasets generated with the LuPNT simulator, selecting a stereo dense depth estimation model based on Gated Recurrent Units for its balance of speed and accuracy in depth estimation, and a convolutional neural network for its superior performance in detecting semantic segments. Using ground truth poses to decouple the local scene understanding from the global state estimation, our pipeline reconstructs a 120-meter traverse with a geometric height accuracy of approximately 3 cm, outperforming a traditional point cloud baseline without LiDAR. The resulting 3DGS map enables novel view synthesis and serves as a foundation for a full SLAM system, where its capacity for joint map and pose optimization would offer significant advantages. Our results demonstrate that combining semantic segmentation and dense depth estimation with learned map representations is an effective approach for creating detailed, large-scale maps to support future lunar surface missions.

  </details>



- **VisionNVS: Self-Supervised Inpainting for Novel View Synthesis under the Virtual-Shift Paradigm**  
  Hongbo Lu, Liang Yao, Chenghao He, Fan Liu, Wenlong Liao, Tao He, Pai Peng  
  _2026-03-18_ · https://arxiv.org/abs/2603.17382v1  
  <details><summary>Abstract</summary>

  A fundamental bottleneck in Novel View Synthesis (NVS) for autonomous driving is the inherent supervision gap on novel trajectories: models are tasked with synthesizing unseen views during inference, yet lack ground truth images for these shifted poses during training. In this paper, we propose VisionNVS, a camera-only framework that fundamentally reformulates view synthesis from an ill-posed extrapolation problem into a self-supervised inpainting task. By introducing a ``Virtual-Shift'' strategy, we use monocular depth proxies to simulate occlusion patterns and map them onto the original view. This paradigm shift allows the use of raw, recorded images as pixel-perfect supervision, effectively eliminating the domain gap inherent in previous approaches. Furthermore, we address spatial consistency through a Pseudo-3D Seam Synthesis strategy, which integrates visual data from adjacent cameras during training to explicitly model real-world photometric discrepancies and calibration errors. Experiments demonstrate that VisionNVS achieves superior geometric fidelity and visual quality compared to LiDAR-dependent baselines, offering a robust solution for scalable driving simulation.

  </details>



- **Neural Radiance Maps for Extraterrestrial Navigation and Path Planning**  
  Adam Dai, Shubh Gupta, Grace Gao  
  _2026-03-18_ · https://arxiv.org/abs/2603.17236v1  
  <details><summary>Abstract</summary>

  Autonomous vehicles such as the Mars rovers currently lead the vanguard of surface exploration on extraterrestrial planets and moons. In order to accelerate the pace of exploration and science objectives, it is critical to plan safe and efficient paths for these vehicles. However, current rover autonomy is limited by a lack of global maps which can be easily constructed and stored for onboard re-planning. Recently, Neural Radiance Fields (NeRFs) have been introduced as a detailed 3D scene representation which can be trained from sparse 2D images and efficiently stored. We propose to use NeRFs to construct maps for online use in autonomous navigation, and present a planning framework which leverages the NeRF map to integrate local and global information. Our approach interpolates local cost observations across global regions using kernel ridge regression over terrain features extracted from the NeRF map, allowing the rover to re-route itself around untraversable areas discovered during online operation. We validate our approach in high-fidelity simulation and demonstrate lower cost and higher percentage success rate path planning compared to various baselines.

  </details>


