# Gaussian Splatting & 3DGS

_Updated: 2026-03-21 07:01 UTC_

Total papers shown: **14**


---

- **Matryoshka Gaussian Splatting**  
  Zhilin Guo, Boqiao Zhang, Hakan Aktas, Kyle Fogarty, Jeffrey Hu, Nursena Koprucu Aslan, Wenzhao Li, Canberk Baykal, Albert Miao, Josef Bengtson, et al.  
  _2026-03-19_ · https://arxiv.org/abs/2603.19234v1  
  <details><summary>Abstract</summary>

  The ability to render scenes at adjustable fidelity from a single model, known as level of detail (LoD), is crucial for practical deployment of 3D Gaussian Splatting (3DGS). Existing discrete LoD methods expose only a limited set of operating points, while concurrent continuous LoD approaches enable smoother scaling but often suffer noticeable quality degradation at full capacity, making LoD a costly design decision. We introduce Matryoshka Gaussian Splatting (MGS), a training framework that enables continuous LoD for standard 3DGS pipelines without sacrificing full-capacity rendering quality. MGS learns a single ordered set of Gaussians such that rendering any prefix, the first k splats, produces a coherent reconstruction whose fidelity improves smoothly with increasing budget. Our key idea is stochastic budget training: each iteration samples a random splat budget and optimises both the corresponding prefix and the full set. This strategy requires only two forward passes and introduces no architectural modifications. Experiments across four benchmarks and six baselines show that MGS matches the full-capacity performance of its backbone while enabling a continuous speed-quality trade-off from a single model. Extensive ablations on ordering strategies, training objectives, and model capacity further validate the designs.

  </details>



- **Reconstruction Matters: Learning Geometry-Aligned BEV Representation through 3D Gaussian Splatting**  
  Yiren Lu, Xin Ye, Burhaneddin Yaman, Jingru Luo, Zhexiao Xiong, Liu Ren, Yu Yin  
  _2026-03-19_ · https://arxiv.org/abs/2603.19193v1  
  <details><summary>Abstract</summary>

  Bird's-Eye-View (BEV) perception serves as a cornerstone for autonomous driving, offering a unified spatial representation that fuses surrounding-view images to enable reasoning for various downstream tasks, such as semantic segmentation, 3D object detection, and motion prediction. However, most existing BEV perception frameworks adopt an end-to-end training paradigm, where image features are directly transformed into the BEV space and optimized solely through downstream task supervision. This formulation treats the entire perception process as a black box, often lacking explicit 3D geometric understanding and interpretability, leading to suboptimal performance. In this paper, we claim that an explicit 3D representation matters for accurate BEV perception, and we propose Splat2BEV, a Gaussian Splatting-assisted framework for BEV tasks. Splat2BEV aims to learn BEV feature representations that are both semantically rich and geometrically precise. We first pre-train a Gaussian generator that explicitly reconstructs 3D scenes from multi-view inputs, enabling the generation of geometry-aligned feature representations. These representations are then projected into the BEV space to serve as inputs for downstream tasks. Extensive experiments on nuScenes and argoverse dataset demonstrate that Splat2BEV achieves state-of-the-art performance and validate the effectiveness of incorporating explicit 3D reconstruction into BEV perception.

  </details>



- **GSMem: 3D Gaussian Splatting as Persistent Spatial Memory for Zero-Shot Embodied Exploration and Reasoning**  
  Yiren Lu, Yi Du, Disheng Liu, Yunlai Zhou, Chen Wang, Yu Yin  
  _2026-03-19_ · https://arxiv.org/abs/2603.19137v1  
  <details><summary>Abstract</summary>

  Effective embodied exploration requires agents to accumulate and retain spatial knowledge over time. However, existing scene representations, such as discrete scene graphs or static view-based snapshots, lack \textit{post-hoc re-observability}. If an initial observation misses a target, the resulting memory omission is often irrecoverable. To bridge this gap, we propose \textbf{GSMem}, a zero-shot embodied exploration and reasoning framework built upon 3D Gaussian Splatting (3DGS). By explicitly parameterizing continuous geometry and dense appearance, 3DGS serves as a persistent spatial memory that endows the agent with \textit{Spatial Recollection}: the ability to render photorealistic novel views from optimal, previously unoccupied viewpoints. To operationalize this, GSMem employs a retrieval mechanism that simultaneously leverages parallel object-level scene graphs and semantic-level language fields. This complementary design robustly localizes target regions, enabling the agent to ``hallucinate'' optimal views for high-fidelity Vision-Language Model (VLM) reasoning. Furthermore, we introduce a hybrid exploration strategy that combines VLM-driven semantic scoring with a 3DGS-based coverage objective, balancing task-aware exploration with geometric coverage. Extensive experiments on embodied question answering and lifelong navigation demonstrate the robustness and effectiveness of our framework

  </details>



- **GHOST: Fast Category-agnostic Hand-Object Interaction Reconstruction from RGB Videos using Gaussian Splatting**  
  Ahmed Tawfik Aboukhadra, Marcel Rogge, Nadia Robertini, Abdalla Arafa, Jameel Malik, Ahmed Elhayek, Didier Stricker  
  _2026-03-19_ · https://arxiv.org/abs/2603.18912v1  
  <details><summary>Abstract</summary>

  Understanding realistic hand-object interactions from monocular RGB videos is essential for AR/VR, robotics, and embodied AI. Existing methods rely on category-specific templates or heavy computation, yet still produce physically inconsistent hand-object alignment in 3D. We introduce GHOST (Gaussian Hand-Object Splatting), a fast, category-agnostic framework for reconstructing dynamic hand-object interactions using 2D Gaussian Splatting. GHOST represents both hands and objects as dense, view-consistent Gaussian discs and introduces three key innovations: (1) a geometric-prior retrieval and consistency loss that completes occluded object regions, (2) a grasp-aware alignment that refines hand translations and object scale to ensure realistic contact, and (3) a hand-aware background loss that prevents penalizing hand-occluded object regions. GHOST achieves complete, physically consistent, and animatable reconstructions from a single RGB video while running an order of magnitude faster than prior category-agnostic methods. Extensive experiments on ARCTIC, HO3D, and in-the-wild datasets demonstrate state-of-the-art accuracy in 3D reconstruction and 2D rendering quality, establishing GHOST as an efficient and robust solution for realistic hand-object interaction modeling. Code is available at https://github.com/ATAboukhadra/GHOST.

  </details>



- **From ex(p) to poly: Gaussian Splatting with Polynomial Kernels**  
  Joerg H. Mueller, Martin Winter, Markus Steinberger  
  _2026-03-19_ · https://arxiv.org/abs/2603.18707v1  
  <details><summary>Abstract</summary>

  Recent advancements in Gaussian Splatting (3DGS) have introduced various modifications to the original kernel, resulting in significant performance improvements. However, many of these kernel changes are incompatible with existing datasets optimized for the original Gaussian kernel, presenting a challenge for widespread adoption. In this work, we address this challenge by proposing an alternative kernel that maintains compatibility with existing datasets while improving computational efficiency. Specifically, we replace the original exponential kernel with a polynomial approximation combined with a ReLU function. This modification allows for more aggressive culling of Gaussians, leading to enhanced performance across different 3DGS implementations. Our results show a notable performance improvement of 4 to 15% with negligible impact on image quality. We also provide a detailed mathematical analysis of the new kernel and discuss its potential benefits for 3DGS implementations on NPU hardware.

  </details>



- **SwiftGS: Episodic Priors for Immediate Satellite Surface Recovery**  
  Rong Fu, Jiekai Wu, Haiyun Wei, Xiaowen Ma, Shiyin Lin, Kangan Qian, Chuang Liu, Jianyuan Ni, Simon James Fong  
  _2026-03-19_ · https://arxiv.org/abs/2603.18634v1  
  <details><summary>Abstract</summary>

  Rapid, large-scale 3D reconstruction from multi-date satellite imagery is vital for environmental monitoring, urban planning, and disaster response, yet remains difficult due to illumination changes, sensor heterogeneity, and the cost of per-scene optimization. We introduce SwiftGS, a meta-learned system that reconstructs 3D surfaces in a single forward pass by predicting geometry-radiation-decoupled Gaussian primitives together with a lightweight SDF, replacing expensive per-scene fitting with episodic training that captures transferable priors. The model couples a differentiable physics graph for projection, illumination, and sensor response with spatial gating that blends sparse Gaussian detail and global SDF structure, and incorporates semantic-geometric fusion, conditional lightweight task heads, and multi-view supervision from a frozen geometric teacher under an uncertainty-aware multi-task loss. At inference, SwiftGS operates zero-shot with optional compact calibration and achieves accurate DSM reconstruction and view-consistent rendering at significantly reduced computational cost, with ablations highlighting the benefits of the hybrid representation, physics-aware rendering, and episodic meta-training.

  </details>



- **OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting**  
  Hongjia Zhai, Qi Zhang, Xiaokun Pan, Xiyu Zhang, Yitong Dong, Huaqi Zhang, Dan Xu, Guofeng Zhang  
  _2026-03-19_ · https://arxiv.org/abs/2603.18510v1  
  <details><summary>Abstract</summary>

  Open-vocabulary scene understanding with online panoptic mapping is essential for embodied applications to perceive and interact with environments. However, existing methods are predominantly offline or lack instance-level understanding, limiting their applicability to real-world robotic tasks. In this paper, we propose OnlinePG, a novel and effective system that integrates geometric reconstruction and open-vocabulary perception using 3D Gaussian Splatting in an online setting. Technically, to achieve online panoptic mapping, we employ an efficient local-to-global paradigm with a sliding window. To build local consistency map, we construct a 3D segment clustering graph that jointly leverages geometric and semantic cues, fusing inconsistent segments within sliding window into complete instances. Subsequently, to update the global map, we construct explicit grids with spatial attributes for the local 3D Gaussian map and fuse them into the global map via robust bidirectional bipartite 3D Gaussian instance matching. Finally, we utilize the fused VLM features inside the 3D spatial attribute grids to achieve open-vocabulary scene understanding. Extensive experiments on widely used datasets demonstrate that our method achieves better performance among online approaches, while maintaining real-time efficiency.

  </details>



- **Inst4DGS: Instance-Decomposed 4D Gaussian Splatting with Multi-Video Label Permutation Learning**  
  Yonghan Lee, Dinesh Manocha  
  _2026-03-19_ · https://arxiv.org/abs/2603.18402v1  
  <details><summary>Abstract</summary>

  We present Inst4DGS, an instance-decomposed 4D Gaussian Splatting (4DGS) approach with long-horizon per-Gaussian trajectories. While dynamic 4DGS has advanced rapidly, instance-decomposed 4DGS remains underexplored, largely due to the difficulty of associating inconsistent instance labels across independently segmented multi-view videos. We address this challenge by introducing per-video label-permutation latents that learn cross-video instance matches through a differentiable Sinkhorn layer, enabling direct multi-view supervision with consistent identity preservation. This explicit label alignment yields sharp decision boundaries and temporally stable identities without identity drift. To further improve efficiency, we propose instance-decomposed motion scaffolds that provide low-dimensional motion bases per object for long-horizon trajectory optimization. Experiments on Panoptic Studio and Neural3DV show that Inst4DGS jointly supports tracking and instance decomposition while achieving state-of-the-art rendering and segmentation quality. On the Panoptic Studio dataset, Inst4DGS improves PSNR from 26.10 to 28.36, and instance mIoU from 0.6310 to 0.9129, over the strongest baseline.

  </details>



- **Semantic Segmentation and Depth Estimation for Real-Time Lunar Surface Mapping Using 3D Gaussian Splatting**  
  Guillem Casadesus Vila, Adam Dai, Grace Gao  
  _2026-03-18_ · https://arxiv.org/abs/2603.18218v1  
  <details><summary>Abstract</summary>

  Navigation and mapping on the lunar surface require robust perception under challenging conditions, including poorly textured environments, high-contrast lighting, and limited computational resources. This paper presents a real-time mapping framework that integrates dense perception models with a 3D Gaussian Splatting (3DGS) representation. We first benchmark several models on synthetic datasets generated with the LuPNT simulator, selecting a stereo dense depth estimation model based on Gated Recurrent Units for its balance of speed and accuracy in depth estimation, and a convolutional neural network for its superior performance in detecting semantic segments. Using ground truth poses to decouple the local scene understanding from the global state estimation, our pipeline reconstructs a 120-meter traverse with a geometric height accuracy of approximately 3 cm, outperforming a traditional point cloud baseline without LiDAR. The resulting 3DGS map enables novel view synthesis and serves as a foundation for a full SLAM system, where its capacity for joint map and pose optimization would offer significant advantages. Our results demonstrate that combining semantic segmentation and dense depth estimation with learned map representations is an effective approach for creating detailed, large-scale maps to support future lunar surface missions.

  </details>



- **AHOY! Animatable Humans under Occlusion from YouTube Videos with Gaussian Splatting and Video Diffusion Priors**  
  Aymen Mir, Riza Alp Guler, Xiangjun Tang, Peter Wonka, Gerard Pons-Moll  
  _2026-03-18_ · https://arxiv.org/abs/2603.17975v1  
  <details><summary>Abstract</summary>

  We present AHOY, a method for reconstructing complete, animatable 3D Gaussian avatars from in-the-wild monocular video despite heavy occlusion. Existing methods assume unoccluded input-a fully visible subject, often in a canonical pose-excluding the vast majority of real-world footage where people are routinely occluded by furniture, objects, or other people. Reconstructing from such footage poses fundamental challenges: large body regions may never be observed, and multi-view supervision per pose is unavailable. We address these challenges with four contributions: (i) a hallucination-as-supervision pipeline that uses identity-finetuned diffusion models to generate dense supervision for previously unobserved body regions; (ii) a two-stage canonical-to-pose-dependent architecture that bootstraps from sparse observations to full pose-dependent Gaussian maps; (iii) a map-pose/LBS-pose decoupling that absorbs multi-view inconsistencies from the generated data; (iv) a head/body split supervision strategy that preserves facial identity. We evaluate on YouTube videos and on multi-view capture data with significant occlusion and demonstrate state-of-the-art reconstruction quality. We also demonstrate that the resulting avatars are robust enough to be animated with novel poses and composited into 3DGS scenes captured using cell-phone video. Our project page is available at https://miraymen.github.io/ahoy/

  </details>



- **CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image**  
  Yizheng Song, Yiyu Zhuang, Qipeng Xu, Haixiang Wang, Jiahe Zhu, Jing Tian, Siyu Zhu, Hao Zhu  
  _2026-03-18_ · https://arxiv.org/abs/2603.17779v1  
  <details><summary>Abstract</summary>

  Single-view 3D human reconstruction has garnered significant attention in recent years. Despite numerous advancements, prior research has concentrated on reconstructing 3D models from clear, close-up images of individual subjects, often yielding subpar results in the more prevalent multi-person scenarios. Reconstructing 3D human crowd models is a highly intricate task, laden with challenges such as: 1) extensive occlusions, 2) low clarity, and 3) numerous and various appearances. To address this task, we propose CrowdGaussian, a unified framework that directly reconstructs multi-person 3D Gaussian Splatting (3DGS) representations from single-image inputs. To handle occlusions, we devise a self-supervised adaptation pipeline that enables the pretrained large human model to reconstruct complete 3D humans with plausible geometry and appearance from heavily occluded inputs. Furthermore, we introduce Self-Calibrated Learning (SCL). This training strategy enables single-step diffusion models to adaptively refine coarse renderings to optimal quality by blending identity-preserving samples with clean/corrupted image pairs. The outputs can be distilled back to enhance the quality of multi-person 3DGS representations. Extensive experiments demonstrate that CrowdGaussian generates photorealistic, geometrically coherent reconstructions of multi-person scenes.

  </details>



- **TAPESTRY: From Geometry to Appearance via Consistent Turntable Videos**  
  Yan Zeng, Haoran Jiang, Kaixin Yao, Qixuan Zhang, Longwen Zhang, Lan Xu, Jingyi Yu  
  _2026-03-18_ · https://arxiv.org/abs/2603.17735v1  
  <details><summary>Abstract</summary>

  Automatically generating photorealistic and self-consistent appearances for untextured 3D models is a critical challenge in digital content creation. The advancement of large-scale video generation models offers a natural approach: directly synthesizing 360-degree turntable videos (TTVs), which can serve not only as high-quality dynamic previews but also as an intermediate representation to drive texture synthesis and neural rendering. However, existing general-purpose video diffusion models struggle to maintain strict geometric consistency and appearance stability across the full range of views, making their outputs ill-suited for high-quality 3D reconstruction. To this end, we introduce TAPESTRY, a framework for generating high-fidelity TTVs conditioned on explicit 3D geometry. We reframe the 3D appearance generation task as a geometry-conditioned video diffusion problem: given a 3D mesh, we first render and encode multi-modal geometric features to constrain the video generation process with pixel-level precision, thereby enabling the creation of high-quality and consistent TTVs. Building upon this, we also design a method for downstream reconstruction tasks from the TTV input, featuring a multi-stage pipeline with 3D-Aware Inpainting. By rotating the model and performing a context-aware secondary generation, this pipeline effectively completes self-occluded regions to achieve full surface coverage. The videos generated by TAPESTRY are not only high-quality dynamic previews but also serve as a reliable, 3D-aware intermediate representation that can be seamlessly back-projected into UV textures or used to supervise neural rendering methods like 3DGS. This enables the automated creation of production-ready, complete 3D assets from untextured meshes. Experimental results demonstrate that our method outperforms existing approaches in both video consistency and final reconstruction quality.

  </details>



- **ReLaGS: Relational Language Gaussian Splatting**  
  Yaxu Xie, Abdalla Arafa, Alireza Javanmardi, Christen Millerdurai, Jia Cheng Hu, Shaoxiang Wang, Alain Pagani, Didier Stricker  
  _2026-03-18_ · https://arxiv.org/abs/2603.17605v1  
  <details><summary>Abstract</summary>

  Achieving unified 3D perception and reasoning across tasks such as segmentation, retrieval, and relation understanding remains challenging, as existing methods are either object-centric or rely on costly training for inter-object reasoning. We present a novel framework that constructs a hierarchical language-distilled Gaussian scene and its 3D semantic scene graph without scene-specific training. A Gaussian pruning mechanism refines scene geometry, while a robust multi-view language alignment strategy aggregates noisy 2D features into accurate 3D object embeddings. On top of this hierarchy, we build an open-vocabulary 3D scene graph with Vision Language derived annotations and Graph Neural Network-based relational reasoning. Our approach enables efficient and scalable open-vocabulary 3D reasoning by jointly modeling hierarchical semantics and inter/intra-object relationships, validated across tasks including open-vocabulary segmentation, scene graph generation, and relation-guided retrieval. Project page: https://dfki-av.github.io/ReLaGS/

  </details>



- **UniSem: Generalizable Semantic 3D Reconstruction from Sparse Unposed Images**  
  Guibiao Liao, Qian Ren, Kaimin Liao, Hua Wang, Zhi Chen, Luchao Wang, Yaohua Tang  
  _2026-03-18_ · https://arxiv.org/abs/2603.17519v1  
  <details><summary>Abstract</summary>

  Semantic-aware 3D reconstruction from sparse, unposed images remains challenging for feed-forward 3D Gaussian Splatting (3DGS). Existing methods often predict an over-complete set of Gaussian primitives under sparse-view supervision, leading to unstable geometry and inferior depth quality. Meanwhile, they rely solely on 2D segmenter features for semantic lifting, which provides weak 3D-level and limited generalizable supervision, resulting in incomplete 3D semantics in novel scenes. To address these issues, we propose UniSem, a unified framework that jointly improves depth accuracy and semantic generalization via two key components. First, Error-aware Gaussian Dropout (EGD) performs error-guided capacity control by suppressing redundancy-prone Gaussians using rendering error cues, producing meaningful, geometrically stable Gaussian representations for improved depth estimation. Second, we introduce a Mix-training Curriculum (MTC) that progressively blends 2D segmenter-lifted semantics with the model's own emergent 3D semantic priors, implemented with object-level prototype alignment to enhance semantic coherence and completeness. Extensive experiments on ScanNet and Replica show that UniSem achieves superior performance in depth prediction and open-vocabulary 3D segmentation across varying numbers of input views. Notably, with 16-view inputs, UniSem reduces depth Rel by 15.2% and improves open-vocabulary segmentation mAcc by 3.7% over strong baselines.

  </details>


