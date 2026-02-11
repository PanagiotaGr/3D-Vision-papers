# 3D Reconstruction

_Updated: 2026-02-11 07:17 UTC_

Total papers shown: **15**


---

- **VersaViT: Enhancing MLLM Vision Backbones via Task-Guided Optimization**  
  Yikun Liu, Yuan Liu, Shangzhe Di, Haicheng Wang, Zhongyin Zhao, Le Tian, Xiao Zhou, Jie Zhou, Jiangchao Yao, Yanfeng Wang, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.09934v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have recently achieved remarkable success in visual-language understanding, demonstrating superior high-level semantic alignment within their vision encoders. An important question thus arises: Can these encoders serve as versatile vision backbones, capable of reliably performing classic vision-centric tasks as well? To address the question, we make the following contributions: (i) we identify that the vision encoders within MLLMs exhibit deficiencies in their dense feature representations, as evidenced by their suboptimal performance on dense prediction tasks (e.g., semantic segmentation, depth estimation); (ii) we propose VersaViT, a well-rounded vision transformer that instantiates a novel multi-task framework for collaborative post-training. This framework facilitates the optimization of the vision backbone via lightweight task heads with multi-granularity supervision; (iii) extensive experiments across various downstream tasks demonstrate the effectiveness of our method, yielding a versatile vision backbone suited for both language-mediated reasoning and pixel-level understanding.

  </details>



- **SARS: A Novel Face and Body Shape and Appearance Aware 3D Reconstruction System extends Morphable Models**  
  Gulraiz Khan, Kenneth Y. Wertheim, Kevin Pimbblet, Waqas Ahmed  
  _2026-02-10_ · https://arxiv.org/abs/2602.09918v1  
  <details><summary>Abstract</summary>

  Morphable Models (3DMMs) are a type of morphable model that takes 2D images as inputs and recreates the structure and physical appearance of 3D objects, especially human faces and bodies. 3DMM combines identity and expression blendshapes with a basic face mesh to create a detailed 3D model. The variability in the 3D Morphable models can be controlled by tuning diverse parameters. They are high-level image descriptors, such as shape, texture, illumination, and camera parameters. Previous research in 3D human reconstruction concentrated solely on global face structure or geometry, ignoring face semantic features such as age, gender, and facial landmarks characterizing facial boundaries, curves, dips, and wrinkles. In order to accommodate changes in these high-level facial characteristics, this work introduces a shape and appearance-aware 3D reconstruction system (named SARS by us), a c modular pipeline that extracts body and face information from a single image to properly rebuild the 3D model of the human full body.

  </details>



- **VideoAfford: Grounding 3D Affordance from Human-Object-Interaction Videos via Multimodal Large Language Model**  
  Hanqing Wang, Mingyu Liu, Xiaoyu Chen, Chengwei MA, Yiming Zhong, Wenti Yin, Yuhao Liu, Zhiqing Cui, Jiahao Yuan, Lu Dai, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.09638v1  
  <details><summary>Abstract</summary>

  3D affordance grounding aims to highlight the actionable regions on 3D objects, which is crucial for robotic manipulation. Previous research primarily focused on learning affordance knowledge from static cues such as language and images, which struggle to provide sufficient dynamic interaction context that can reveal temporal and causal cues. To alleviate this predicament, we collect a comprehensive video-based 3D affordance dataset, \textit{VIDA}, which contains 38K human-object-interaction videos covering 16 affordance types, 38 object categories, and 22K point clouds. Based on \textit{VIDA}, we propose a strong baseline: VideoAfford, which activates multimodal large language models with additional affordance segmentation capabilities, enabling both world knowledge reasoning and fine-grained affordance grounding within a unified framework. To enhance action understanding capability, we leverage a latent action encoder to extract dynamic interaction priors from HOI videos. Moreover, we introduce a \textit{spatial-aware} loss function to enable VideoAfford to obtain comprehensive 3D spatial knowledge. Extensive experimental evaluations demonstrate that our model significantly outperforms well-established methods and exhibits strong open-world generalization with affordance reasoning abilities. All datasets and code will be publicly released to advance research in this area.

  </details>



- **RAD: Retrieval-Augmented Monocular Metric Depth Estimation for Underrepresented Classes**  
  Michael Baltaxe, Dan Levi, Sagie Benaim  
  _2026-02-10_ · https://arxiv.org/abs/2602.09532v1  
  <details><summary>Abstract</summary>

  Monocular Metric Depth Estimation (MMDE) is essential for physically intelligent systems, yet accurate depth estimation for underrepresented classes in complex scenes remains a persistent challenge. To address this, we propose RAD, a retrieval-augmented framework that approximates the benefits of multi-view stereo by utilizing retrieved neighbors as structural geometric proxies. Our method first employs an uncertainty-aware retrieval mechanism to identify low-confidence regions in the input and retrieve RGB-D context samples containing semantically similar content. We then process both the input and retrieved context via a dual-stream network and fuse them using a matched cross-attention module, which transfers geometric information only at reliable point correspondences. Evaluations on NYU Depth v2, KITTI, and Cityscapes demonstrate that RAD significantly outperforms state-of-the-art baselines on underrepresented classes, reducing relative absolute error by 29.2% on NYU Depth v2, 13.3% on KITTI, and 7.2% on Cityscapes, while maintaining competitive performance on standard in-domain benchmarks.

  </details>



- **Bridging the Modality Gap in Roadside LiDAR: A Training-Free Vision-Language Model Framework for Vehicle Classification**  
  Yiqiao Li, Bo Shang, Jie Wei  
  _2026-02-10_ · https://arxiv.org/abs/2602.09425v1  
  <details><summary>Abstract</summary>

  Fine-grained truck classification is critical for intelligent transportation systems (ITS), yet current LiDAR-based methods face scalability challenges due to their reliance on supervised deep learning and labor-intensive manual annotation. Vision-Language Models (VLMs) offer promising few-shot generalization, but their application to roadside LiDAR is limited by a modality gap between sparse 3D point clouds and dense 2D imagery. We propose a framework that bridges this gap by adapting off-the-shelf VLMs for fine-grained truck classification without parameter fine-tuning. Our new depth-aware image generation pipeline applies noise removal, spatial and temporal registration, orientation rectification, morphological operations, and anisotropic smoothing to transform sparse, occluded LiDAR scans into depth-encoded 2D visual proxies. Validated on a real-world dataset of 20 vehicle classes, our approach achieves competitive classification accuracy with as few as 16-30 examples per class, offering a scalable alternative to data-intensive supervised baselines. We further observe a "Semantic Anchor" effect: text-based guidance regularizes performance in ultra-low-shot regimes $k < 4$, but degrades accuracy in more-shot settings due to semantic mismatch. Furthermore, we demonstrate the efficacy of this framework as a Cold Start strategy, using VLM-generated labels to bootstrap lightweight supervised models. Notably, the few-shot VLM-based model achieves over correct classification rate of 75 percent for specific drayage categories (20ft, 40ft, and 53ft containers) entirely without the costly training or fine-tuning, significantly reducing the intensive demands of initial manual labeling, thus achieving a method of practical use in ITS applications.

  </details>



- **Single-Slice-to-3D Reconstruction in Medical Imaging and Natural Objects: A Comparative Benchmark with SAM 3D**  
  Yan Luo, Advaith Ravishankar, Serena Liu, Yutong Yang, Mengyu Wang  
  _2026-02-10_ · https://arxiv.org/abs/2602.09407v1  
  <details><summary>Abstract</summary>

  A 3D understanding of anatomy is central to diagnosis and treatment planning, yet volumetric imaging remains costly with long wait times. Image-to-3D foundations models can solve this issue by reconstructing 3D data from 2D modalites. Current foundation models are trained on natural image distributions to reconstruct naturalistic objects from a single image by leveraging geometric priors across pixels. However, it is unclear whether these learned geometric priors transfer to medical data. In this study, we present a controlled zero-shot benchmark of single slice medical image-to-3D reconstruction across five state-of-the-art image-to-3D models: SAM3D, Hunyuan3D-2.1, Direct3D, Hi3DGen, and TripoSG. These are evaluated across six medical datasets spanning anatomical and pathological structures and two natrual datasets, using voxel based metrics and point cloud distance metrics. Across medical datasets, voxel based overlap remains moderate for all models, consistent with a depth reconstruction failure mode when inferring volume from a single slice. In contrast, global distance metrics show more separation between methods: SAM3D achieves the strongest overall topological similarity to ground truth medical 3D data, while alternative models are more prone to over-simplication of reconstruction. Our results quantify the limits of single-slice medical reconstruction and highlight depth ambiguity caused by the planar nature of 2D medical data, motivating multi-view image-to-3D reconstruction to enable reliable medical 3D inference.

  </details>



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


