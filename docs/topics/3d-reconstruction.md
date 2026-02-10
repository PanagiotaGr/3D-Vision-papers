# 3D Reconstruction

_Updated: 2026-02-10 07:20 UTC_

Total papers shown: **11**


---

- **Analysis of Converged 3D Gaussian Splatting Solutions: Density Effects and Prediction Limit**  
  Zhendong Wang, Cihan Ruan, Jingchuan Xiao, Chuqing Shi, Wei Jiang, Wei Wang, Wenjie Liu, Nam Ling  
  _2026-02-09_ · https://arxiv.org/abs/2602.08909v1  
  <details><summary>Abstract</summary>

  We investigate what structure emerges in 3D Gaussian Splatting (3DGS) solutions from standard multi-view optimization. We term these Rendering-Optimal References (RORs) and analyze their statistical properties, revealing stable patterns: mixture-structured scales and bimodal radiance across diverse scenes. To understand what determines these parameters, we apply learnability probes by training predictors to reconstruct RORs from point clouds without rendering supervision. Our analysis uncovers fundamental density-stratification. Dense regions exhibit geometry-correlated parameters amenable to render-free prediction, while sparse regions show systematic failure across architectures. We formalize this through variance decomposition, demonstrating that visibility heterogeneity creates covariance-dominated coupling between geometric and appearance parameters in sparse regions. This reveals the dual character of RORs: geometric primitives where point clouds suffice, and view synthesis primitives where multi-view constraints are essential. We provide density-aware strategies that improve training robustness and discuss architectural implications for systems that adaptively balance feed-forward prediction and rendering-based refinement.

  </details>



- **Overview and Comparison of AVS Point Cloud Compression Standard**  
  Wei Gao, Wenxu Gao, Xingming Mu, Changhao Peng, Ge Li  
  _2026-02-09_ · https://arxiv.org/abs/2602.08613v1  
  <details><summary>Abstract</summary>

  Point cloud is a prevalent 3D data representation format with significant application values in immersive media, autonomous driving, digital heritage protection, etc. However, the large data size of point clouds poses challenges to transmission and storage, which influences the wide deployments. Therefore, point cloud compression plays a crucial role in practical applications for both human and machine perception optimization. To this end, the Moving Picture Experts Group (MPEG) has established two standards for point cloud compression, including Geometry-based Point Cloud Compression (G-PCC) and Video-based Point Cloud Compression (V-PCC). In the meantime, the Audio Video coding Standard (AVS) Workgroup of China also have launched and completed the development for its first generation point cloud compression standard, namely AVS PCC. This new standardization effort has adopted many new coding tools and techniques, which are different from the other counterpart standards. This paper reviews the AVS PCC standard from two perspectives, i.e., the related technologies and performance comparisons.

  </details>



- **TIBR4D: Tracing-Guided Iterative Boundary Refinement for Efficient 4D Gaussian Segmentation**  
  He Wu, Xia Yan, Yanghui Xu, Liegang Xia, Jiazhou Chen  
  _2026-02-09_ · https://arxiv.org/abs/2602.08540v1  
  <details><summary>Abstract</summary>

  Object-level segmentation in dynamic 4D Gaussian scenes remains challenging due to complex motion, occlusions, and ambiguous boundaries. In this paper, we present an efficient learning-free 4D Gaussian segmentation framework that lifts video segmentation masks to 4D spaces, whose core is a two-stage iterative boundary refinement, TIBR4D. The first stage is an Iterative Gaussian Instance Tracing (IGIT) at the temporal segment level. It progressively refines Gaussian-to-instance probabilities through iterative tracing, and extracts corresponding Gaussian point clouds that better handle occlusions and preserve completeness of object structures compared to existing one-shot threshold-based methods. The second stage is a frame-wise Gaussian Rendering Range Control (RCC) via suppressing highly uncertain Gaussians near object boundaries while retaining their core contributions for more accurate boundaries. Furthermore, a temporal segmentation merging strategy is proposed for IGIT to balance identity consistency and dynamic awareness. Longer segments enforce stronger multi-frame constraints for stable identities, while shorter segments allow identity changes to be captured promptly. Experiments on HyperNeRF and Neu3D demonstrate that our method produces accurate object Gaussian point clouds with clearer boundaries and higher efficiency compared to SOTA methods.

  </details>



- **RealSynCol: a high-fidelity synthetic colon dataset for 3D reconstruction applications**  
  Chiara Lena, Davide Milesi, Alessandro Casella, Luca Carlini, Joseph C. Norton, James Martin, Bruno Scaglioni, Keith L. Obstein, Roberto De Sire, Marco Spadaccini, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.08397v1  
  <details><summary>Abstract</summary>

  Deep learning has the potential to improve colonoscopy by enabling 3D reconstruction of the colon, providing a comprehensive view of mucosal surfaces and lesions, and facilitating the identification of unexplored areas. However, the development of robust methods is limited by the scarcity of large-scale ground truth data. We propose RealSynCol, a highly realistic synthetic dataset designed to replicate the endoscopic environment. Colon geometries extracted from 10 CT scans were imported into a virtual environment that closely mimics intraoperative conditions and rendered with realistic vascular textures. The resulting dataset comprises 28\,130 frames, paired with ground truth depth maps, optical flow, 3D meshes, and camera trajectories. A benchmark study was conducted to evaluate the available synthetic colon datasets for the tasks of depth and pose estimation. Results demonstrate that the high realism and variability of RealSynCol significantly enhance generalization performance on clinical images, proving it to be a powerful tool for developing deep learning algorithms to support endoscopic diagnosis.

  </details>



- **Generating Adversarial Events: A Motion-Aware Point Cloud Framework**  
  Hongwei Ren, Youxin Jiang, Qifei Gu, Xiangqian Wu  
  _2026-02-09_ · https://arxiv.org/abs/2602.08230v1  
  <details><summary>Abstract</summary>

  Event cameras have been widely adopted in safety-critical domains such as autonomous driving, robotics, and human-computer interaction. A pressing challenge arises from the vulnerability of deep neural networks to adversarial examples, which poses a significant threat to the reliability of event-based systems. Nevertheless, research into adversarial attacks on events is scarce. This is primarily due to the non-differentiable nature of mainstream event representations, which hinders the extension of gradient-based attack methods. In this paper, we propose MA-ADV, a novel \textbf{M}otion-\textbf{A}ware \textbf{Adv}ersarial framework. To the best of our knowledge, this is the first work to generate adversarial events by leveraging point cloud representations. MA-ADV accounts for high-frequency noise in events and employs a diffusion-based approach to smooth perturbations, while fully leveraging the spatial and temporal relationships among events. Finally, MA-ADV identifies the minimal-cost perturbation through a combination of sample-wise Adam optimization, iterative refinement, and binary search. Extensive experimental results validate that MA-ADV ensures a 100\% attack success rate with minimal perturbation cost, and also demonstrate enhanced robustness against defenses, underscoring the critical security challenges facing future event-based perception systems.

  </details>



- **Dynamic Black-hole Emission Tomography with Physics-informed Neural Fields**  
  Berthy T. Feng, Andrew A. Chael, David Bromley, Aviad Levis, William T. Freeman, Katherine L. Bouman  
  _2026-02-08_ · https://arxiv.org/abs/2602.08029v1  
  <details><summary>Abstract</summary>

  With the success of static black-hole imaging, the next frontier is the dynamic and 3D imaging of black holes. Recovering the dynamic 3D gas near a black hole would reveal previously-unseen parts of the universe and inform new physics models. However, only sparse radio measurements from a single viewpoint are possible, making the dynamic 3D reconstruction problem significantly ill-posed. Previously, BH-NeRF addressed the ill-posed problem by assuming Keplerian dynamics of the gas, but this assumption breaks down near the black hole, where the strong gravitational pull of the black hole and increased electromagnetic activity complicate fluid dynamics. To overcome the restrictive assumptions of BH-NeRF, we propose PI-DEF, a physics-informed approach that uses differentiable neural rendering to fit a 4D (time + 3D) emissivity field given EHT measurements. Our approach jointly reconstructs the 3D velocity field with the 4D emissivity field and enforces the velocity as a soft constraint on the dynamics of the emissivity. In experiments on simulated data, we find significantly improved reconstruction accuracy over both BH-NeRF and a physics-agnostic approach. We demonstrate how our method may be used to estimate other physics parameters of the black hole, such as its spin.

  </details>



- **Continuity-driven Synergistic Diffusion with Neural Priors for Ultra-Sparse-View CBCT Reconstruction**  
  Junlin Wang, Jiancheng Fang, Peng Peng, Shaoyu Wang, Qiegen Liu  
  _2026-02-08_ · https://arxiv.org/abs/2602.07980v1  
  <details><summary>Abstract</summary>

  The clinical application of cone-beam computed tomography (CBCT) is constrained by the inherent trade-off between radiation exposure and image quality. Ultra-sparse angular sampling, employed to reduce dose, introduces severe undersampling artifacts and inter-slice inconsistencies, compromising diagnostic reliability. Existing reconstruction methods often struggle to balance angular continuity with spatial detail fidelity. To address these challenges, we propose a Continuity-driven Synergistic Diffusion with Neural priors (CSDN) for ultra-sparse-view CBCT reconstruction. Neural priors are introduced as a structural foundation to encode a continuous threedimensional attenuation representation, enabling the synthesis of physically consistent dense projections from ultra-sparse measurements. Building upon this neural-prior-based initialization, a synergistic diffusion strategy is developed, consisting of two collaborative refinement paths: a Sinogram Refinement Diffusion (Sino-RD) process that restores angular continuity and a Digital Radiography Refinement Diffusion (DR-RD) process that enforces inter-slice consistency from the projection image perspective. The outputs of the two diffusion paths are adaptively fused by the Dual-Projection Reconstruction Fusion (DPRF) module to achieve coherent volumetric reconstruction. Extensive experiments demonstrate that the proposed CSDN effectively suppresses artifacts and recovers fine textures under ultra-sparse-view conditions, outperforming existing state-of-the-art techniques.

  </details>



- **Scalable Adaptation of 3D Geometric Foundation Models via Weak Supervision from Internet Video**  
  Zihui Gao, Ke Liu, Donny Y. Chen, Duochao Shi, Guosheng Lin, Hao Chen, Chunhua Shen  
  _2026-02-08_ · https://arxiv.org/abs/2602.07891v1  
  <details><summary>Abstract</summary>

  Geometric foundation models show promise in 3D reconstruction, yet their progress is severely constrained by the scarcity of diverse, large-scale 3D annotations. While Internet videos offer virtually unlimited raw data, utilizing them as a scaling source for geometric learning is challenging due to the absence of ground-truth geometry and the presence of observational noise. To address this, we propose SAGE, a framework for Scalable Adaptation of GEometric foundation models from raw video streams. SAGE leverages a hierarchical mining pipeline to transform videos into training trajectories and hybrid supervision: (1) Informative training trajectory selection; (2) Sparse Geometric Anchoring via SfM point clouds for global structural guidance; and (3) Dense Differentiable Consistency via 3D Gaussian rendering for multi-view constraints. To prevent catastrophic forgetting, we introduce a regularization strategy using anchor data. Extensive experiments show that SAGE significantly enhances zero-shot generalization, reducing Chamfer Distance by 20-42% on unseen benchmarks (7Scenes, TUM-RGBD, Matterport3D) compared to state-of-the-art baselines. To our knowledge, SAGE pioneers the adaptation of geometric foundation models via Internet video, establishing a scalable paradigm for general-purpose 3D learning.

  </details>



- **Recovering 3D Shapes from Ultra-Fast Motion-Blurred Images**  
  Fei Yu, Shudan Guo, Shiqing Xin, Beibei Wang, Haisen Zhao, Wenzheng Chen  
  _2026-02-08_ · https://arxiv.org/abs/2602.07860v1  
  <details><summary>Abstract</summary>

  We consider the problem of 3D shape recovery from ultra-fast motion-blurred images. While 3D reconstruction from static images has been extensively studied, recovering geometry from extreme motion-blurred images remains challenging. Such scenarios frequently occur in both natural and industrial settings, such as fast-moving objects in sports (e.g., balls) or rotating machinery, where rapid motion distorts object appearance and makes traditional 3D reconstruction techniques like Multi-View Stereo (MVS) ineffective. In this paper, we propose a novel inverse rendering approach for shape recovery from ultra-fast motion-blurred images. While conventional rendering techniques typically synthesize blur by averaging across multiple frames, we identify a major computational bottleneck in the repeated computation of barycentric weights. To address this, we propose a fast barycentric coordinate solver, which significantly reduces computational overhead and achieves a speedup of up to 4.57x, enabling efficient and photorealistic simulation of high-speed motion. Crucially, our method is fully differentiable, allowing gradients to propagate from rendered images to the underlying 3D shape, thereby facilitating shape recovery through inverse rendering. We validate our approach on two representative motion types: rapid translation and rotation. Experimental results demonstrate that our method enables efficient and realistic modeling of ultra-fast moving objects in the forward simulation. Moreover, it successfully recovers 3D shapes from 2D imagery of objects undergoing extreme translational and rotational motion, advancing the boundaries of vision-based 3D reconstruction. Project page: https://maxmilite.github.io/rec-from-ultrafast-blur/

  </details>



- **Influence of Geometry, Class Imbalance and Alignment on Reconstruction Accuracy -- A Micro-CT Phantom-Based Evaluation**  
  Avinash Kumar K M, Samarth S. Raut  
  _2026-02-07_ · https://arxiv.org/abs/2602.07658v1  
  <details><summary>Abstract</summary>

  The accuracy of the 3D models created from medical scans depends on imaging hardware, segmentation methods and mesh processing techniques etc. The effects of geometry type, class imbalance, voxel and point cloud alignment on accuracy remain to be thoroughly explored. This work evaluates the errors across the reconstruction pipeline and explores the use of voxel and surface-based accuracy metrics for different segmentation algorithms and geometry types. A sphere, a facemask, and an AAA were printed using the SLA technique and scanned using a micro-CT machine. Segmentation was performed using GMM, Otsu and RG based methods. Segmented and reference models aligned using the KU algorithm, were quantitatively compared to evaluate metrics like Dice and Jaccard scores, precision. Surface meshes were registered with reference meshes using an ICP-based alignment process. Metrics like chamfer distance, and average Hausdorff distance were evaluated. The Otsu method was found to be the most suitable method for all the geometries. AAA yielded low overlap scores due to its small wall thickness and misalignment. The effect of class imbalance on specificity was observed the most for AAA. Surface-based accuracy metrics differed from the voxel-based trends. The RG method performed best for sphere, while GMM and Otsu perform better for AAA. The facemask surface was most error-prone, possibly due to misalignment during the ICP process. Segmentation accuracy is a cumulative sum of errors across different stages of the reconstruction process. High voxel-based accuracy metrics may be misleading in cases of high class imbalance and sensitivity to alignment. The Jaccard index is found to be more stringent than the Dice and more suitable for accuracy assessment for thin-walled structures. Voxel and point cloud alignment should be ensured to make any reliable assessment of the reconstruction pipeline.

  </details>



- **Perspective-aware fusion of incomplete depth maps and surface normals for accurate 3D reconstruction**  
  Ondrej Hlinka, Georg Kaniak, Christian Kapeller  
  _2026-02-07_ · https://arxiv.org/abs/2602.07444v1  
  <details><summary>Abstract</summary>

  We address the problem of reconstructing 3D surfaces from depth and surface normal maps acquired by a sensor system based on a single perspective camera. Depth and normal maps can be obtained through techniques such as structured-light scanning and photometric stereo, respectively. We propose a perspective-aware log-depth fusion approach that extends existing orthographic gradient-based depth-normals fusion methods by explicitly accounting for perspective projection, leading to metrically accurate 3D reconstructions. Additionally, the method handles missing depth measurements by leveraging available surface normal information to inpaint gaps. Experiments on the DiLiGenT-MV data set demonstrate the effectiveness of our approach and highlight the importance of perspective-aware depth-normals fusion.

  </details>


