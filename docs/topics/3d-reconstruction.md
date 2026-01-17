# 3D Reconstruction

_Updated: 2026-01-17 06:45 UTC_

Total papers shown: **7**


---

- **NanoSD: Edge Efficient Foundation Model for Real Time Image Restoration**  
  Subhajit Sanyal, Srinivas Soumitri Miriyala, Akshay Janardan Bankar, Sravanth Kodavanti, Harshit, Abhishek Ameta, Shreyas Pandith, Amit Satish Unde  
  _2026-01-14_ · https://arxiv.org/abs/2601.09823v1  
  <details><summary>Abstract</summary>

  Latent diffusion models such as Stable Diffusion 1.5 offer strong generative priors that are highly valuable for image restoration, yet their full pipelines remain too computationally heavy for deployment on edge devices. Existing lightweight variants predominantly compress the denoising U-Net or reduce the diffusion trajectory, which disrupts the underlying latent manifold and limits generalization beyond a single task. We introduce NanoSD, a family of Pareto-optimal diffusion foundation models distilled from Stable Diffusion 1.5 through network surgery, feature-wise generative distillation, and structured architectural scaling jointly applied to the U-Net and the VAE encoder-decoder. This full-pipeline co-design preserves the generative prior while producing models that occupy distinct operating points along the accuracy-latency-size frontier (e.g., 130M-315M parameters, achieving real-time inference down to 20ms on mobile-class NPUs). We show that parameter reduction alone does not correlate with hardware efficiency, and we provide an analysis revealing how architectural balance, feature routing, and latent-space preservation jointly shape true on-device latency. When used as a drop-in backbone, NanoSD enables state-of-the-art performance across image super-resolution, image deblurring, face restoration, and monocular depth estimation, outperforming prior lightweight diffusion models in both perceptual quality and practical deployability. NanoSD establishes a general-purpose diffusion foundation model family suitable for real-time visual generation and restoration on edge devices.

  </details>



- **LCF3D: A Robust and Real-Time Late-Cascade Fusion Framework for 3D Object Detection in Autonomous Driving**  
  Carlo Sgaravatti, Riccardo Pieroni, Matteo Corno, Sergio M. Savaresi, Luca Magri, Giacomo Boracchi  
  _2026-01-14_ · https://arxiv.org/abs/2601.09812v1  
  <details><summary>Abstract</summary>

  Accurately localizing 3D objects like pedestrians, cyclists, and other vehicles is essential in Autonomous Driving. To ensure high detection performance, Autonomous Vehicles complement RGB cameras with LiDAR sensors, but effectively combining these data sources for 3D object detection remains challenging. We propose LCF3D, a novel sensor fusion framework that combines a 2D object detector on RGB images with a 3D object detector on LiDAR point clouds. By leveraging multimodal fusion principles, we compensate for inaccuracies in the LiDAR object detection network. Our solution combines two key principles: (i) late fusion, to reduce LiDAR False Positives by matching LiDAR 3D detections with RGB 2D detections and filtering out unmatched LiDAR detections; and (ii) cascade fusion, to recover missed objects from LiDAR by generating new 3D frustum proposals corresponding to unmatched RGB detections. Experiments show that LCF3D is beneficial for domain generalization, as it turns out to be successful in handling different sensor configurations between training and testing domains. LCF3D achieves significant improvements over LiDAR-based methods, particularly for challenging categories like pedestrians and cyclists in the KITTI dataset, as well as motorcycles and bicycles in nuScenes. Code can be downloaded from: https://github.com/CarloSgaravatti/LCF3D.

  </details>



- **Efficient Camera-Controlled Video Generation of Static Scenes via Sparse Diffusion and 3D Rendering**  
  Jieying Chen, Jeffrey Hu, Joan Lasenby, Ayush Tewari  
  _2026-01-14_ · https://arxiv.org/abs/2601.09697v1  
  <details><summary>Abstract</summary>

  Modern video generative models based on diffusion models can produce very realistic clips, but they are computationally inefficient, often requiring minutes of GPU time for just a few seconds of video. This inefficiency poses a critical barrier to deploying generative video in applications that require real-time interactions, such as embodied AI and VR/AR. This paper explores a new strategy for camera-conditioned video generation of static scenes: using diffusion-based generative models to generate a sparse set of keyframes, and then synthesizing the full video through 3D reconstruction and rendering. By lifting keyframes into a 3D representation and rendering intermediate views, our approach amortizes the generation cost across hundreds of frames while enforcing geometric consistency. We further introduce a model that predicts the optimal number of keyframes for a given camera trajectory, allowing the system to adaptively allocate computation. Our final method, SRENDER, uses very sparse keyframes for simple trajectories and denser ones for complex camera motion. This results in video generation that is more than 40 times faster than the diffusion-based baseline in generating 20 seconds of video, while maintaining high visual fidelity and temporal stability, offering a practical path toward efficient and controllable video synthesis.

  </details>



- **SCE-SLAM: Scale-Consistent Monocular SLAM via Scene Coordinate Embeddings**  
  Yuchen Wu, Jiahe Li, Xiaohan Yu, Lina Yu, Jin Zheng, Xiao Bai  
  _2026-01-14_ · https://arxiv.org/abs/2601.09665v1  
  <details><summary>Abstract</summary>

  Monocular visual SLAM enables 3D reconstruction from internet video and autonomous navigation on resource-constrained platforms, yet suffers from scale drift, i.e., the gradual divergence of estimated scale over long sequences. Existing frame-to-frame methods achieve real-time performance through local optimization but accumulate scale drift due to the lack of global constraints among independent windows. To address this, we propose SCE-SLAM, an end-to-end SLAM system that maintains scale consistency through scene coordinate embeddings, which are learned patch-level representations encoding 3D geometric relationships under a canonical scale reference. The framework consists of two key modules: geometry-guided aggregation that leverages 3D spatial proximity to propagate scale information from historical observations through geometry-modulated attention, and scene coordinate bundle adjustment that anchors current estimates to the reference scale through explicit 3D coordinate constraints decoded from the scene coordinate embeddings. Experiments on KITTI, Waymo, and vKITTI demonstrate substantial improvements: our method reduces absolute trajectory error by 8.36m on KITTI compared to the best prior approach, while maintaining 36 FPS and achieving scale consistency across large-scale scenes.

  </details>



- **Iterative Differential Entropy Minimization (IDEM) method for fine rigid pairwise 3D Point Cloud Registration: A Focus on the Metric**  
  Emmanuele Barberi, Felice Sfravara, Filippo Cucinotta  
  _2026-01-14_ · https://arxiv.org/abs/2601.09601v1  
  <details><summary>Abstract</summary>

  Point cloud registration is a central theme in computer vision, with alignment algorithms continuously improving for greater robustness. Commonly used methods evaluate Euclidean distances between point clouds and minimize an objective function, such as Root Mean Square Error (RMSE). However, these approaches are most effective when the point clouds are well-prealigned and issues such as differences in density, noise, holes, and limited overlap can compromise the results. Traditional methods, such as Iterative Closest Point (ICP), require choosing one point cloud as fixed, since Euclidean distances lack commutativity. When only one point cloud has issues, adjustments can be made, but in real scenarios, both point clouds may be affected, often necessitating preprocessing. The authors introduce a novel differential entropy-based metric, designed to serve as the objective function within an optimization framework for fine rigid pairwise 3D point cloud registration, denoted as Iterative Differential Entropy Minimization (IDEM). This metric does not depend on the choice of a fixed point cloud and, during transformations, reveals a clear minimum corresponding to the best alignment. Multiple case studies are conducted, and the results are compared with those obtained using RMSE, Chamfer distance, and Hausdorff distance. The proposed metric proves effective even with density differences, noise, holes, and partial overlap, where RMSE does not always yield optimal alignment.

  </details>



- **Multimodal Signal Processing For Thermo-Visible-Lidar Fusion In Real-time 3D Semantic Mapping**  
  Jiajun Sun, Yangyi Ou, Haoyuan Zheng, Chao yang, Yue Ma  
  _2026-01-14_ · https://arxiv.org/abs/2601.09578v1  
  <details><summary>Abstract</summary>

  In complex environments, autonomous robot navigation and environmental perception pose higher requirements for SLAM technology. This paper presents a novel method for semantically enhancing 3D point cloud maps with thermal information. By first performing pixel-level fusion of visible and infrared images, the system projects real-time LiDAR point clouds onto this fused image stream. It then segments heat source features in the thermal channel to instantly identify high temperature targets and applies this temperature information as a semantic layer on the final 3D map. This approach generates maps that not only have accurate geometry but also possess a critical semantic understanding of the environment, making it highly valuable for specific applications like rapid disaster assessment and industrial preventive maintenance.

  </details>



- **V-DPM: 4D Video Reconstruction with Dynamic Point Maps**  
  Edgar Sucar, Eldar Insafutdinov, Zihang Lai, Andrea Vedaldi  
  _2026-01-14_ · https://arxiv.org/abs/2601.09499v1  
  <details><summary>Abstract</summary>

  Powerful 3D representations such as DUSt3R invariant point maps, which encode 3D shape and camera parameters, have significantly advanced feed forward 3D reconstruction. While point maps assume static scenes, Dynamic Point Maps (DPMs) extend this concept to dynamic 3D content by additionally representing scene motion. However, existing DPMs are limited to image pairs and, like DUSt3R, require post processing via optimization when more than two views are involved. We argue that DPMs are more useful when applied to videos and introduce V-DPM to demonstrate this. First, we show how to formulate DPMs for video input in a way that maximizes representational power, facilitates neural prediction, and enables reuse of pretrained models. Second, we implement these ideas on top of VGGT, a recent and powerful 3D reconstructor. Although VGGT was trained on static scenes, we show that a modest amount of synthetic data is sufficient to adapt it into an effective V-DPM predictor. Our approach achieves state of the art performance in 3D and 4D reconstruction for dynamic scenes. In particular, unlike recent dynamic extensions of VGGT such as P3, DPMs recover not only dynamic depth but also the full 3D motion of every point in the scene.

  </details>


