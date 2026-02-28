# 3D Reconstruction

_Updated: 2026-02-28 06:55 UTC_

Total papers shown: **20**


---

- **VGG-T$^3$: Offline Feed-Forward 3D Reconstruction at Scale**  
  Sven Elflein, Ruilong Li, Sérgio Agostinho, Zan Gojcic, Laura Leal-Taixé, Qunjie Zhou, Aljosa Osep  
  _2026-02-26_ · https://arxiv.org/abs/2602.23361v1  
  <details><summary>Abstract</summary>

  We present a scalable 3D reconstruction model that addresses a critical limitation in offline feed-forward methods: their computational and memory requirements grow quadratically w.r.t. the number of input images. Our approach is built on the key insight that this bottleneck stems from the varying-length Key-Value (KV) space representation of scene geometry, which we distill into a fixed-size Multi-Layer Perceptron (MLP) via test-time training. VGG-T$^3$ (Visual Geometry Grounded Test Time Training) scales linearly w.r.t. the number of input views, similar to online models, and reconstructs a $1k$ image collection in just $54$ seconds, achieving a $11.6\times$ speed-up over baselines that rely on softmax attention. Since our method retains global scene aggregation capability, our point map reconstruction error outperforming other linear-time methods by large margins. Finally, we demonstrate visual localization capabilities of our model by querying the scene representation with unseen images.

  </details>



- **UniScale: Unified Scale-Aware 3D Reconstruction for Multi-View Understanding via Prior Injection for Robotic Perception**  
  Mohammad Mahdavian, Gordon Tan, Binbin Xu, Yuan Ren, Dongfeng Bai, Bingbing Liu  
  _2026-02-26_ · https://arxiv.org/abs/2602.23224v1  
  <details><summary>Abstract</summary>

  We present UniScale, a unified, scale-aware multi-view 3D reconstruction framework for robotic applications that flexibly integrates geometric priors through a modular, semantically informed design. In vision-based robotic navigation, the accurate extraction of environmental structure from raw image sequences is critical for downstream tasks. UniScale addresses this challenge with a single feed-forward network that jointly estimates camera intrinsics and extrinsics, scale-invariant depth and point maps, and the metric scale of a scene from multi-view images, while optionally incorporating auxiliary geometric priors when available. By combining global contextual reasoning with camera-aware feature representations, UniScale is able to recover the metric-scale of the scene. In robotic settings where camera intrinsics are known, they can be easily incorporated to improve performance, with additional gains obtained when camera poses are also available. This co-design enables robust, metric-aware 3D reconstruction within a single unified model. Importantly, UniScale does not require training from scratch, and leverages world priors exhibited in pre-existing models without geometric encoding strategies, making it particularly suitable for resource-constrained robotic teams. We evaluate UniScale on multiple benchmarks, demonstrating strong generalization and consistent performance across diverse environments. We will release our implementation upon acceptance.

  </details>



- **Efficient Encoder-Free Fourier-based 3D Large Multimodal Model**  
  Guofeng Mei, Wei Lin, Luigi Riz, Yujiao Wu, Yiming Wang, Fabio Poiesi  
  _2026-02-26_ · https://arxiv.org/abs/2602.23153v1  
  <details><summary>Abstract</summary>

  Large Multimodal Models (LMMs) that process 3D data typically rely on heavy, pre-trained visual encoders to extract geometric features. While recent 2D LMMs have begun to eliminate such encoders for efficiency and scalability, extending this paradigm to 3D remains challenging due to the unordered and large-scale nature of point clouds. This leaves a critical unanswered question: How can we design an LMM that tokenizes unordered 3D data effectively and efficiently without a cumbersome encoder? We propose Fase3D, the first efficient encoder-free Fourier-based 3D scene LMM. Fase3D tackles the challenges of scalability and permutation invariance with a novel tokenizer that combines point cloud serialization and the Fast Fourier Transform (FFT) to approximate self-attention. This design enables an effective and computationally minimal architecture, built upon three key innovations: First, we represent large scenes compactly via structured superpoints. Second, our space-filling curve serialization followed by an FFT enables efficient global context modeling and graph-based token merging. Lastly, our Fourier-augmented LoRA adapters inject global frequency-aware interactions into the LLMs at a negligible cost. Fase3D achieves performance comparable to encoder-based 3D LMMs while being significantly more efficient in computation and parameters. Project website: https://tev-fbk.github.io/Fase3D.

  </details>



- **Align then Adapt: Rethinking Parameter-Efficient Transfer Learning in 4D Perception**  
  Yiding Sun, Jihua Zhu, Haozhe Cheng, Chaoyi Lu, Zhichuan Yang, Lin Chen, Yaonan Wang  
  _2026-02-26_ · https://arxiv.org/abs/2602.23069v1  
  <details><summary>Abstract</summary>

  Point cloud video understanding is critical for robotics as it accurately encodes motion and scene interaction. We recognize that 4D datasets are far scarcer than 3D ones, which hampers the scalability of self-supervised 4D models. A promising alternative is to transfer 3D pre-trained models to 4D perception tasks. However, rigorous empirical analysis reveals two critical limitations that impede transfer capability: overfitting and the modality gap. To overcome these challenges, we develop a novel "Align then Adapt" (PointATA) paradigm that decomposes parameter-efficient transfer learning into two sequential stages. Optimal-transport theory is employed to quantify the distributional discrepancy between 3D and 4D datasets, enabling our proposed point align embedder to be trained in Stage 1 to alleviate the underlying modality gap. To mitigate overfitting, an efficient point-video adapter and a spatial-context encoder are integrated into the frozen 3D backbone to enhance temporal modeling capacity in Stage 2. Notably, with the above engineering-oriented designs, PointATA enables a pre-trained 3D model without temporal knowledge to reason about dynamic video content at a smaller parameter cost compared to previous work. Extensive experiments show that PointATA can match or even outperform strong full fine-tuning models, whilst enjoying the advantage of parameter efficiency, e.g. 97.21 \% accuracy on 3D action recognition, $+8.7 \%$ on 4 D action segmentation, and 84.06\% on 4D semantic segmentation.

  </details>



- **UCM: Unifying Camera Control and Memory with Time-aware Positional Encoding Warping for World Models**  
  Tianxing Xu, Zixuan Wang, Guangyuan Wang, Li Hu, Zhongyi Zhang, Peng Zhang, Bang Zhang, Song-Hai Zhang  
  _2026-02-26_ · https://arxiv.org/abs/2602.22960v1  
  <details><summary>Abstract</summary>

  World models based on video generation demonstrate remarkable potential for simulating interactive environments but face persistent difficulties in two key areas: maintaining long-term content consistency when scenes are revisited and enabling precise camera control from user-provided inputs. Existing methods based on explicit 3D reconstruction often compromise flexibility in unbounded scenarios and fine-grained structures. Alternative methods rely directly on previously generated frames without establishing explicit spatial correspondence, thereby constraining controllability and consistency. To address these limitations, we present UCM, a novel framework that unifies long-term memory and precise camera control via a time-aware positional encoding warping mechanism. To reduce computational overhead, we design an efficient dual-stream diffusion transformer for high-fidelity generation. Moreover, we introduce a scalable data curation strategy utilizing point-cloud-based rendering to simulate scene revisiting, facilitating training on over 500K monocular videos. Extensive experiments on real-world and synthetic benchmarks demonstrate that UCM significantly outperforms state-of-the-art methods in long-term scene consistency, while also achieving precise camera controllability in high-fidelity video generation.

  </details>



- **Sapling-NeRF: Geo-Localised Sapling Reconstruction in Forests for Ecological Monitoring**  
  Miguel Ángel Muñoz-Bañón, Nived Chebrolu, Sruthi M. Krishna Moorthy, Yifu Tao, Fernando Torres, Roberto Salguero-Gómez, Maurice Fallon  
  _2026-02-26_ · https://arxiv.org/abs/2602.22731v1  
  <details><summary>Abstract</summary>

  Saplings are key indicators of forest regeneration and overall forest health. However, their fine-scale architectural traits are difficult to capture with existing 3D sensing methods, which make quantitative evaluation difficult. Terrestrial Laser Scanners (TLS), Mobile Laser Scanners (MLS), or traditional photogrammetry approaches poorly reconstruct thin branches, dense foliage, and lack the scale consistency needed for long-term monitoring. Implicit 3D reconstruction methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) are promising alternatives, but cannot recover the true scale of a scene and lack any means to be accurately geo-localised. In this paper, we present a pipeline which fuses NeRF, LiDAR SLAM, and GNSS to enable repeatable, geo-localised ecological monitoring of saplings. Our system proposes a three-level representation: (i) coarse Earth-frame localisation using GNSS, (ii) LiDAR-based SLAM for centimetre-accurate localisation and reconstruction, and (iii) NeRF-derived object-centric dense reconstruction of individual saplings. This approach enables repeatable quantitative evaluation and long-term monitoring of sapling traits. Our experiments in forest plots in Wytham Woods (Oxford, UK) and Evo (Finland) show that stem height, branching patterns, and leaf-to-wood ratios can be captured with increased accuracy as compared to TLS. We demonstrate that accurate stem skeletons and leaf distributions can be measured for saplings with heights between 0.5m and 2m in situ, giving ecologists access to richer structural and quantitative data for analysing forest dynamics.

  </details>



- **QuadSync: Quadrifocal Tensor Synchronization via Tucker Decomposition**  
  Daniel Miao, Gilad Lerman, Joe Kileel  
  _2026-02-26_ · https://arxiv.org/abs/2602.22639v1  
  <details><summary>Abstract</summary>

  In structure from motion, quadrifocal tensors capture more information than their pairwise counterparts (essential matrices), yet they have often been thought of as impractical and only of theoretical interest. In this work, we challenge such beliefs by providing a new framework to recover $n$ cameras from the corresponding collection of quadrifocal tensors. We form the block quadrifocal tensor and show that it admits a Tucker decomposition whose factor matrices are the stacked camera matrices, and which thus has a multilinear rank of (4,~4,~4,~4) independent of $n$. We develop the first synchronization algorithm for quadrifocal tensors, using Tucker decomposition, alternating direction method of multipliers, and iteratively reweighted least squares. We further establish relationships between the block quadrifocal, trifocal, and bifocal tensors, and introduce an algorithm that jointly synchronizes these three entities. Numerical experiments demonstrate the effectiveness of our methods on modern datasets, indicating the potential and importance of using higher-order information in synchronization.

  </details>



- **GIFSplat: Generative Prior-Guided Iterative Feed-Forward 3D Gaussian Splatting from Sparse Views**  
  Tianyu Chen, Wei Xiang, Kang Han, Yu Lu, Di Wu, Gaowen Liu, Ramana Rao Kompella  
  _2026-02-26_ · https://arxiv.org/abs/2602.22571v1  
  <details><summary>Abstract</summary>

  Feed-forward 3D reconstruction offers substantial runtime advantages over per-scene optimization, which remains slow at inference and often fragile under sparse views. However, existing feed-forward methods still have potential for further performance gains, especially for out-of-domain data, and struggle to retain second-level inference time once a generative prior is introduced. These limitations stem from the one-shot prediction paradigm in existing feed-forward pipeline: models are strictly bounded by capacity, lack inference-time refinement, and are ill-suited for continuously injecting generative priors. We introduce GIFSplat, a purely feed-forward iterative refinement framework for 3D Gaussian Splatting from sparse unposed views. A small number of forward-only residual updates progressively refine current 3D scene using rendering evidence, achieve favorable balance between efficiency and quality. Furthermore, we distill a frozen diffusion prior into Gaussian-level cues from enhanced novel renderings without gradient backpropagation or ever-increasing view-set expansion, thereby enabling per-scene adaptation with generative prior while preserving feed-forward efficiency. Across DL3DV, RealEstate10K, and DTU, GIFSplat consistently outperforms state-of-the-art feed-forward baselines, improving PSNR by up to +2.1 dB, and it maintains second-scale inference time without requiring camera poses or any test-time gradient optimization.

  </details>



- **SwiftNDC: Fast Neural Depth Correction for High-Fidelity 3D Reconstruction**  
  Kang Han, Wei Xiang, Lu Yu, Mathew Wyatt, Gaowen Liu, Ramana Rao Kompella  
  _2026-02-26_ · https://arxiv.org/abs/2602.22565v1  
  <details><summary>Abstract</summary>

  Depth-guided 3D reconstruction has gained popularity as a fast alternative to optimization-heavy approaches, yet existing methods still suffer from scale drift, multi-view inconsistencies, and the need for substantial refinement to achieve high-fidelity geometry. Here, we propose SwiftNDC, a fast and general framework built around a Neural Depth Correction field that produces cross-view consistent depth maps. From these refined depths, we generate a dense point cloud through back-projection and robust reprojection-error filtering, obtaining a clean and uniformly distributed geometric initialization for downstream reconstruction. This reliable dense geometry substantially accelerates 3D Gaussian Splatting (3DGS) for mesh reconstruction, enabling high-quality surfaces with significantly fewer optimization iterations. For novel-view synthesis, SwiftNDC can also improve 3DGS rendering quality, highlighting the benefits of strong geometric initialization. We conduct a comprehensive study across five datasets, including two for mesh reconstruction, as well as three for novel-view synthesis. SwiftNDC consistently reduces running time for accurate mesh reconstruction and boosts rendering fidelity for view synthesis, demonstrating the effectiveness of combining neural depth refinement with robust geometric initialization for high-fidelity and efficient 3D reconstruction.

  </details>



- **Neu-PiG: Neural Preconditioned Grids for Fast Dynamic Surface Reconstruction on Long Sequences**  
  Julian Kaltheuner, Hannah Dröge, Markus Plack, Patrick Stotko, Reinhard Klein  
  _2026-02-25_ · https://arxiv.org/abs/2602.22212v1  
  <details><summary>Abstract</summary>

  Temporally consistent surface reconstruction of dynamic 3D objects from unstructured point cloud data remains challenging, especially for very long sequences. Existing methods either optimize deformations incrementally, risking drift and requiring long runtimes, or rely on complex learned models that demand category-specific training. We present Neu-PiG, a fast deformation optimization method based on a novel preconditioned latent-grid encoding that distributes spatial features parameterized on the position and normal direction of a keyframe surface. Our method encodes entire deformations across all time steps at various spatial scales into a multi-resolution latent grid, parameterized by the position and normal direction of a reference surface from a single keyframe. This latent representation is then augmented for time modulation and decoded into per-frame 6-DoF deformations via a lightweight multilayer perceptron (MLP). To achieve high-fidelity, drift-free surface reconstructions in seconds, we employ Sobolev preconditioning during gradient-based training of the latent space, completely avoiding the need for any explicit correspondences or further priors. Experiments across diverse human and animal datasets demonstrate that Neu-PiG outperforms state-the-art approaches, offering both superior accuracy and scalability to long sequences while running at least 60x faster than existing training-free methods and achieving inference speeds on the same order as heavy pretrained models.

  </details>



- **Global-Aware Edge Prioritization for Pose Graph Initialization**  
  Tong Wei, Giorgos Tolias, Jiri Matas, Daniel Barath  
  _2026-02-25_ · https://arxiv.org/abs/2602.21963v1  
  <details><summary>Abstract</summary>

  The pose graph is a core component of Structure-from-Motion (SfM), where images act as nodes and edges encode relative poses. Since geometric verification is expensive, SfM pipelines restrict the pose graph to a sparse set of candidate edges, making initialization critical. Existing methods rely on image retrieval to connect each image to its $k$ nearest neighbors, treating pairs independently and ignoring global consistency. We address this limitation through the concept of edge prioritization, ranking candidate edges by their utility for SfM. Our approach has three components: (1) a GNN trained with SfM-derived supervision to predict globally consistent edge reliability; (2) multi-minimal-spanning-tree-based pose graph construction guided by these ranks; and (3) connectivity-aware score modulation that reinforces weak regions and reduces graph diameter. This globally informed initialization yields more reliable and compact pose graphs, improving reconstruction accuracy in sparse and high-speed settings and outperforming SOTA retrieval methods on ambiguous scenes. The ode and trained models are available at https://github.com/weitong8591/global_edge_prior.

  </details>



- **Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context**  
  JiaKui Hu, Jialun Liu, Liying Yang, Xinliang Zhang, Kaiwen Li, Shuang Zeng, Yuanwei Li, Haibin Huang, Chi Zhang, Yanye Lu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21929v1  
  <details><summary>Abstract</summary>

  Scene-consistent video generation aims to create videos that explore 3D scenes based on a camera trajectory. Previous methods rely on video generation models with external memory for consistency, or iterative 3D reconstruction and inpainting, which accumulate errors during inference due to incorrect intermediary outputs, non-differentiable processes, and separate models. To overcome these limitations, we introduce ``geometry-as-context". It iteratively completes the following steps using an autoregressive camera-controlled video generation model: (1) estimates the geometry of the current view necessary for 3D reconstruction, and (2) simulates and restores novel view images rendered by the 3D scene. Under this multi-task framework, we develop the camera gated attention module to enhance the model's capability to effectively leverage camera poses. During the training phase, text contexts are utilized to ascertain whether geometric or RGB images should be generated. To ensure that the model can generate RGB-only outputs during inference, the geometry context is randomly dropped from the interleaved text-image-geometry training sequence. The method has been tested on scene video generation with one-direction and forth-and-back trajectories. The results show its superiority over previous approaches in maintaining scene consistency and camera control.

  </details>



- **EndoDDC: Learning Sparse to Dense Reconstruction for Endoscopic Robotic Navigation via Diffusion Depth Completion**  
  Yinheng Lin, Yiming Huang, Beilei Cui, Long Bai, Huxin Gao, Hongliang Ren, Jiewen Lai  
  _2026-02-25_ · https://arxiv.org/abs/2602.21893v2  
  <details><summary>Abstract</summary>

  Accurate depth estimation plays a critical role in the navigation of endoscopic surgical robots, forming the foundation for 3D reconstruction and safe instrument guidance. Fine-tuning pretrained models heavily relies on endoscopic surgical datasets with precise depth annotations. While existing self-supervised depth estimation techniques eliminate the need for accurate depth annotations, their performance degrades in environments with weak textures and variable lighting, leading to sparse reconstruction with invalid depth estimation. Depth completion using sparse depth maps can mitigate these issues and improve accuracy. Despite the advances in depth completion techniques in general fields, their application in endoscopy remains limited. To overcome these limitations, we propose EndoDDC, an endoscopy depth completion method that integrates images, sparse depth information with depth gradient features, and optimizes depth maps through a diffusion model, addressing the issues of weak texture and light reflection in endoscopic environments. Extensive experiments on two publicly available endoscopy datasets show that our approach outperforms state-of-the-art models in both depth accuracy and robustness. This demonstrates the potential of our method to reduce visual errors in complex endoscopic environments. Our code will be released at https://github.com/yinheng-lin/EndoDDC.

  </details>



- **Joint Shadow Generation and Relighting via Light-Geometry Interaction Maps**  
  Shan Wang, Peixia Li, Chenchen Xu, Ziang Cheng, Jiayu Yang, Hongdong Li, Pulak Purkait  
  _2026-02-25_ · https://arxiv.org/abs/2602.21820v1  
  <details><summary>Abstract</summary>

  We propose Light-Geometry Interaction (LGI) maps, a novel representation that encodes light-aware occlusion from monocular depth. Unlike ray tracing, which requires full 3D reconstruction, LGI captures essential light-shadow interactions reliably and accurately, computed from off-the-shelf 2.5D depth map predictions. LGI explicitly ties illumination direction to geometry, providing a physics-inspired prior that constrains generative models. Without such prior, these models often produce floating shadows, inconsistent illumination, and implausible shadow geometry. Building on this representation, we propose a unified pipeline for joint shadow generation and relighting - unlike prior methods that treat them as disjoint tasks - capturing the intrinsic coupling of illumination and shadowing essential for modeling indirect effects. By embedding LGI into a bridge-matching generative backbone, we reduce ambiguity and enforce physically consistent light-shadow reasoning. To enable effective training, we curated the first large-scale benchmark dataset for joint shadow and relighting, covering reflections, transparency, and complex interreflections. Experiments show significant gains in realism and consistency across synthetic and real images. LGI thus bridges geometry-inspired rendering with generative modeling, enabling efficient, physically consistent shadow generation and relighting.

  </details>



- **XStreamVGGT: Extremely Memory-Efficient Streaming Vision Geometry Grounded Transformer with KV Cache Compression**  
  Zunhai Su, Weihao Ye, Hansen Feng, Keyu Fan, Jing Zhang, Dahai Yu, Zhengwu Liu, Ngai Wong  
  _2026-02-25_ · https://arxiv.org/abs/2602.21780v1  
  <details><summary>Abstract</summary>

  Learning-based 3D visual geometry models have significantly advanced with the advent of large-scale transformers. Among these, StreamVGGT leverages frame-wise causal attention to deliver robust and efficient streaming 3D reconstruction. However, it suffers from unbounded growth in the Key-Value (KV) cache due to the massive influx of vision tokens from multi-image and long-video inputs, leading to increased memory consumption and inference latency as input frames accumulate. This ultimately limits its scalability for long-horizon applications. To address this gap, we propose XStreamVGGT, a tuning-free approach that seamlessly integrates pruning and quantization to systematically compress the KV cache, enabling extremely memory-efficient streaming inference. Specifically, redundant KVs generated from multi-frame inputs are initially pruned to conform to a fixed KV memory budget using an efficient token-importance identification mechanism that maintains full compatibility with high-performance attention kernels (e.g., FlashAttention). Additionally, leveraging the inherent distribution patterns of KV tensors, we apply dimension-adaptive KV quantization within the pruning pipeline to further minimize memory overhead while preserving numerical accuracy. Extensive evaluations show that XStreamVGGT achieves mostly negligible performance degradation while substantially reducing memory usage by 4.42$\times$ and accelerating inference by 5.48$\times$, enabling practical and scalable streaming 3D applications. The code is available at https://github.com/ywh187/XStreamVGGT/.

  </details>



- **Structure-to-Image: Zero-Shot Depth Estimation in Colonoscopy via High-Fidelity Sim-to-Real Adaptation**  
  Juan Yang, Yuyan Zhang, Han Jia, Bing Hu, Wanzhong Song  
  _2026-02-25_ · https://arxiv.org/abs/2602.21740v1  
  <details><summary>Abstract</summary>

  Monocular depth estimation (MDE) for colonoscopy is hampered by the domain gap between simulated and real-world images. Existing image-to-image translation methods, which use depth as a posterior constraint, often produce structural distortions and specular highlights by failing to balance realism with structure consistency. To address this, we propose a Structure-to-Image paradigm that transforms the depth map from a passive constraint into an active generative foundation. We are the first to introduce phase congruency to colonoscopic domain adaptation and design a cross-level structure constraint to co-optimize geometric structures and fine-grained details like vascular textures. In zero-shot evaluations conducted on a publicly available phantom dataset, the MDE model that was fine-tuned on our generated data achieved a maximum reduction of 44.18% in RMSE compared to competing methods. Our code is available at https://github.com/YyangJJuan/PC-S2I.git.

  </details>



- **Assessing airborne laser scanning and aerial photogrammetry for deep learning-based stand delineation**  
  Håkon Næss Sandum, Hans Ole Ørka, Oliver Tomic, Terje Gobakken  
  _2026-02-25_ · https://arxiv.org/abs/2602.21709v1  
  <details><summary>Abstract</summary>

  Accurate forest stand delineation is essential for forest inventory and management but remains a largely manual and subjective process. A recent study has shown that deep learning can produce stand delineations comparable to expert interpreters when combining aerial imagery and airborne laser scanning (ALS) data. However, temporal misalignment between data sources limits operational scalability. Canopy height models (CHMs) derived from digital photogrammetry (DAP) offer better temporal alignment but may smoothen canopy surface and canopy gaps, raising the question of whether they can reliably replace ALS-derived CHMs. Similarly, the inclusion of a digital terrain model (DTM) has been suggested to improve delineation performance, but has remained untested in published literature. Using expert-delineated forest stands as reference data, we assessed a U-Net-based semantic segmentation framework with municipality-level cross-validation across six municipalities in southeastern Norway. We compared multispectral aerial imagery combined with (i) an ALS-derived CHM, (ii) a DAP-derived CHM, and (iii) a DAP-derived CHM in combination with a DTM. Results showed comparable performance across all data combinations, reaching overall accuracy values between 0.90-0.91. Agreement between model predictions was substantially larger than agreement with the reference data, highlighting both model consistency and the inherent subjectivity of stand delineation. The similar performance of DAP-CHMs, despite the reduced structural detail, and the lack of improvements of the DTM indicate that the framework is resilient to variations in input data. These findings indicate that large datasets for deep learning-based stand delineations can be assembled using projects including temporally aligned ALS data and DAP point clouds.

  </details>



- **SF3D-RGB: Scene Flow Estimation from Monocular Camera and Sparse LiDAR**  
  Rajai Alhimdiat, Ramy Battrawy, René Schuster, Didier Stricker, Wesam Ashour  
  _2026-02-25_ · https://arxiv.org/abs/2602.21699v1  
  <details><summary>Abstract</summary>

  Scene flow estimation is an extremely important task in computer vision to support the perception of dynamic changes in the scene. For robust scene flow, learning-based approaches have recently achieved impressive results using either image-based or LiDAR-based modalities. However, these methods have tended to focus on the use of a single modality. To tackle these problems, we present a deep learning architecture, SF3D-RGB, that enables sparse scene flow estimation using 2D monocular images and 3D point clouds (e.g., acquired by LiDAR) as inputs. Our architecture is an end-to-end model that first encodes information from each modality into features and fuses them together. Then, the fused features enhance a graph matching module for better and more robust mapping matrix computation to generate an initial scene flow. Finally, a residual scene flow module further refines the initial scene flow. Our model is designed to strike a balance between accuracy and efficiency. Furthermore, experiments show that our proposed method outperforms single-modality methods and achieves better scene flow accuracy on real-world datasets while using fewer parameters compared to other state-of-the-art methods with fusion.

  </details>



- **Send Less, Perceive More: Masked Quantized Point Cloud Communication for Loss-Tolerant Collaborative Perception**  
  Sheng Xu, Enshu Wang, Hongfei Xue, Jian Teng, Bingyi Liu, Yi Zhu, Pu Wang, Libing Wu, Chunming Qiao  
  _2026-02-25_ · https://arxiv.org/abs/2602.21667v1  
  <details><summary>Abstract</summary>

  Collaborative perception allows connected vehicles to overcome occlusions and limited viewpoints by sharing sensory information. However, existing approaches struggle to achieve high accuracy under strict bandwidth constraints and remain highly vulnerable to random transmission packet loss. We introduce QPoint2Comm, a quantized point-cloud communication framework that dramatically reduces bandwidth while preserving high-fidelity 3D information. Instead of transmitting intermediate features, QPoint2Comm directly communicates quantized point-cloud indices using a shared codebook, enabling efficient reconstruction with lower bandwidth than feature-based methods. To ensure robustness to possible communication packet loss, we employ a masked training strategy that simulates random packet loss, allowing the model to maintain strong performance even under severe transmission failures. In addition, a cascade attention fusion module is proposed to enhance multi-vehicle information integration. Extensive experiments on both simulated and real-world datasets demonstrate that QPoint2Comm sets a new state of the art in accuracy, communication efficiency, and resilience to packet loss.

  </details>



- **HybridINR-PCGC: Hybrid Lossless Point Cloud Geometry Compression Bridging Pretrained Model and Implicit Neural Representation**  
  Wenjie Huang, Qi Yang, Shuting Xia, He Huang, Zhu Li, Yiling Xu  
  _2026-02-25_ · https://arxiv.org/abs/2602.21662v1  
  <details><summary>Abstract</summary>

  Learning-based point cloud compression presents superior performance to handcrafted codecs. However, pretrained-based methods, which are based on end-to-end training and expected to generalize to all the potential samples, suffer from training data dependency. Implicit neural representation (INR) based methods are distribution-agnostic and more robust, but they require time-consuming online training and suffer from the bitstream overhead from the overfitted model. To address these limitations, we propose HybridINR-PCGC, a novel hybrid framework that bridges the pretrained model and INR. Our framework retains distribution-agnostic properties while leveraging a pretrained network to accelerate convergence and reduce model overhead, which consists of two parts: the Pretrained Prior Network (PPN) and the Distribution Agnostic Refiner (DAR). We leverage the PPN, designed for fast inference and stable performance, to generate a robust prior for accelerating the DAR's convergence. The DAR is decomposed into a base layer and an enhancement layer, and only the enhancement layer needed to be packed into the bitstream. Finally, we propose a supervised model compression module to further supervise and minimize the bitrate of the enhancement layer parameters. Based on experiment results, HybridINR-PCGC achieves a significantly improved compression rate and encoding efficiency. Specifically, our method achieves a Bpp reduction of approximately 20.43% compared to G-PCC on 8iVFB. In the challenging out-of-distribution scenario Cat1B, our method achieves a Bpp reduction of approximately 57.85% compared to UniPCGC. And our method exhibits a superior time-rate trade-off, achieving an average Bpp reduction of 15.193% relative to the LINR-PCGC on 8iVFB.

  </details>


