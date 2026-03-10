# 3D Reconstruction

_Updated: 2026-03-10 07:06 UTC_

Total papers shown: **15**


---

- **mmGAT: Pose Estimation by Graph Attention with Mutual Features from mmWave Radar Point Cloud**  
  Abdullah Al Masud, Shi Xintong, Mondher Bouazizi, Ohtsuki Tomoaki  
  _2026-03-09_ · https://arxiv.org/abs/2603.08551v1  
  <details><summary>Abstract</summary>

  Pose estimation and human action recognition (HAR) are pivotal technologies spanning various domains. While the image-based pose estimation and HAR are widely admired for their superior performance, they lack in privacy protection and suboptimal performance in low-light and dark environments. This paper exploits the capabilities of millimeter-wave (mmWave) radar technology for human pose estimation by processing radar data with Graph Neural Network (GNN) architecture, coupled with the attention mechanism. Our goal is to capture the finer details of the radar point cloud to improve the pose estimation performance. To this end, we present a unique feature extraction technique that exploits the full potential of the GNN processing method for pose estimation. Our model mmGAT demonstrates remarkable performance on two publicly available benchmark mmWave datasets and establishes new state of the art results in most scenarios in terms of human pose estimation. Our approach achieves a noteworthy reduction of pose estimation mean per joint position error (MPJPE) by 35.6% and PA-MPJPE by 14.1% from the current state of the art benchmark within this domain.

  </details>



- **PCFEx: Point Cloud Feature Extraction for Graph Neural Networks**  
  Abdullah Al Masud, Shi Xintong, Mondher Bouazizi, Ohtsuki Tomoaki  
  _2026-03-09_ · https://arxiv.org/abs/2603.08540v1  
  <details><summary>Abstract</summary>

  Graph neural networks (GNNs) have gained significant attention for their effectiveness across various domains. This study focuses on applying GNN to process 3D point cloud data for human pose estimation (HPE) and human activity recognition (HAR). We propose novel point cloud feature extraction (PCFEx) techniques to capture meaningful information at the point, edge, and graph levels of the point cloud by considering point cloud as a graph. Moreover, we introduce a GNN architecture designed to efficiently process these features. Our approach is evaluated on four most popular publicly available millimeter wave radar datasets, three for HPE and one for HAR. The results show substantial improvements, with significantly reduced errors in all three HPE benchmarks, and an overall accuracy of 98.8% in mmWave-based HAR, outperforming the existing state of the art models. This work demonstrates the great potential of feature extraction incorporated with GNN modeling approach to enhance the precision of point cloud processing.

  </details>



- **OSCAR: Occupancy-based Shape Completion via Acoustic Neural Implicit Representations**  
  Magdalena Wysocki, Kadir Burak Buldu, Miruna-Alexandra Gafencu, Mohammad Farid Azampour, Nassir Navab  
  _2026-03-09_ · https://arxiv.org/abs/2603.08279v1  
  <details><summary>Abstract</summary>

  Accurate 3D reconstruction of vertebral anatomy from ultrasound is important for guiding minimally invasive spine interventions, but it remains challenging due to acoustic shadowing and view-dependent signal variations. We propose an occupancy-based shape completion method that reconstructs complete 3D anatomical geometry from partial ultrasound observations. Crucially for intra-operative applications, our approach extracts the anatomical surface directly from the image, avoiding the need for anatomical labels during inference. This label-free completion relies on a coupled latent space representing both the image appearance and the underlying anatomical shape. By leveraging a Neural Implicit Representation (NIR) that jointly models both spatial occupancy and acoustic interactions, the method uses acoustic parameters to become implicitly aware of the unseen regions without explicit shadowing labels through tracking acoustic signal transmission. We show that this method outperforms state-of-the-art shape completion for B-mode ultrasound by 80% in HD95 score. We validate our approach both in-silico and on phantom US images with registered mesh models from CT labels, demonstrating accurate reconstruction of occluded anatomy and robust generalization across diverse imaging conditions. Code and data will be released on publication.

  </details>



- **Topologically Stable Hough Transform**  
  Stefan Huber, Kristóf Huszár, Michael Kerber, Martin Uray  
  _2026-03-09_ · https://arxiv.org/abs/2603.08245v1  
  <details><summary>Abstract</summary>

  We propose an alternative formulation of the well-known Hough transform to detect lines in point clouds. Replacing the discretized voting scheme of the classical Hough transform by a continuous score function, its persistent features in the sense of persistent homology give a set of candidate lines. We also devise and implement an algorithm to efficiently compute these candidate lines.

  </details>



- **MV-Fashion: Towards Enabling Virtual Try-On and Size Estimation with Multi-View Paired Data**  
  Hunor Laczkó, Libang Jia, Loc-Phat Truong, Diego Hernández, Sergio Escalera, Jordi Gonzalez, Meysam Madadi  
  _2026-03-09_ · https://arxiv.org/abs/2603.08147v1  
  <details><summary>Abstract</summary>

  Existing 4D human datasets fall short for fashion-specific research, lacking either realistic garment dynamics or task-specific annotations. Synthetic datasets suffer from a realism gap, whereas real-world captures lack the detailed annotations and paired data required for virtual try-on (VTON) and size estimation tasks. To bridge this gap, we introduce MV-Fashion, a large-scale, multi-view video dataset engineered for domain-specific fashion analysis. MV-Fashion features 3,273 sequences (72.5 million frames) from 80 diverse subjects wearing 3-10 outfits each. It is designed to capture complex, real-world garment dynamics, including multiple layers and varied styling (e.g. rolled sleeves, tucked shirt). A core contribution is a rich data representation that includes pixel-level semantic annotations, ground-truth material properties like elasticity, and 3D point clouds. Crucially for VTON applications, MV-Fashion provides paired data: multi-view synchronized captures of worn garments alongside their corresponding flat, catalogue images. We leverage this dataset to establish baselines for fashion-centric tasks, including virtual try-on, clothing size estimation, and novel view synthesis. The dataset is available at https://hunorlaczko.github.io/MV-Fashion .

  </details>



- **Speed3R: Sparse Feed-forward 3D Reconstruction Models**  
  Weining Ren, Xiao Tan, Kai Han  
  _2026-03-09_ · https://arxiv.org/abs/2603.08055v1  
  <details><summary>Abstract</summary>

  While recent feed-forward 3D reconstruction models accelerate 3D reconstruction by jointly inferring dense geometry and camera poses in a single pass, their reliance on dense attention imposes a quadratic complexity, creating a prohibitive computational bottleneck that severely limits inference speed. To resolve this, we introduce Speed3R, an end-to-end trainable model inspired by the core principle of Structure-from-Motion: that a sparse set of keypoints is sufficient for robust pose estimation. Speed3R features a dual-branch attention mechanism where a compression branch creates a coarse contextual prior to guide a selection branch, which performs fine-grained attention only on the most informative image tokens. This strategy mimics the efficiency of traditional keypoint matching, achieving a remarkable 12.4x inference speedup on 1000-view sequences, while introducing a minimal, controlled trade-off in geometric accuracy. Validated on standard benchmarks with both VGGT and $π^3$ backbones, our method delivers high-quality reconstructions at a fraction of computational cost, paving the way for efficient large-scale scene modeling.

  </details>



- **$L^3$:Scene-agnostic Visual Localization in the Wild**  
  Yu Zhang, Muhua Zhu, Yifei Xue, Tie Ji, Yizhen Lao  
  _2026-03-09_ · https://arxiv.org/abs/2603.07937v1  
  <details><summary>Abstract</summary>

  Standard visual localization methods typically require offline pre-processing of scenes to obtain 3D structural information for better performance. This inevitably introduces additional computational and time costs, as well as the overhead of storing scene representations. Can we visually localize in a wild scene without any off-line preprocessing step? In this paper, we leverage the online inference capabilities of feed-forward 3D reconstruction networks to propose a novel map-free visual localization framework $L^3$. Specifically, by performing direct online 3D reconstruction on RGB images, followed by two-stage metric scale recovery and pose refinement based on 2D-3D correspondences, $L^3$ achieves high accuracy without the need to pre-build or store any offline scene representations. Extensive experiments demonstrate $L^3$ not only that the performance is comparable to state-of-the-art solutions on various benchmarks, but also that it exhibits significantly superior robustness in sparse scenes (fewer reference images per scene).

  </details>



- **Toward Unified Multimodal Representation Learning for Autonomous Driving**  
  Ximeng Tao, Dimitar Filev, Gaurav Pandey  
  _2026-03-09_ · https://arxiv.org/abs/2603.07874v1  
  <details><summary>Abstract</summary>

  Contrastive Language-Image Pre-training (CLIP) has shown impressive performance in aligning visual and textual representations. Recent studies have extended this paradigm to 3D vision to improve scene understanding for autonomous driving. A common strategy is to employ pairwise cosine similarity between modalities to guide the training of a 3D encoder. However, considering the similarity between individual modality pairs rather than all modalities jointly fails to ensure consistent and unified alignment across the entire multimodal space. In this paper, we propose a Contrastive Tensor Pre-training (CTP) framework that simultaneously aligns multiple modalities in a unified embedding space to enhance end-to-end autonomous driving. Compared with pairwise cosine similarity alignment, our method extends the 2D similarity matrix into a multimodal similarity tensor. Furthermore, we introduce a tensor loss to enable joint contrastive learning across all modalities. For experimental validation of our framework, we construct a text-image-point cloud triplet dataset derived from existing autonomous driving datasets. The results show that our proposed unified multimodal alignment framework achieves favorable performance for both scenarios: (i) aligning a 3D encoder with pretrained CLIP encoders, and (ii) pretraining all encoders from scratch.

  </details>



- **FrameVGGT: Frame Evidence Rolling Memory for streaming VGGT**  
  Zhisong Xu, Takeshi Oishi  
  _2026-03-08_ · https://arxiv.org/abs/2603.07690v1  
  <details><summary>Abstract</summary>

  Streaming Visual Geometry Transformers such as StreamVGGT enable strong online 3D perception but suffer from unbounded KV-cache growth, which limits deployment over long streams. We revisit bounded-memory streaming from the perspective of geometric support. In geometry-driven reasoning, memory quality depends not only on how many tokens are retained, but also on whether the retained memory still preserves sufficiently coherent local support. This suggests that token-level retention may become less suitable under fixed budgets, as it can thin the evidence available within each contributing frame and make subsequent fusion more sensitive to weakly aligned history. Motivated by this observation, we propose FrameVGGT, a frame-driven rolling explicit-memory framework that treats each frame's incremental KV contribution as a coherent evidence block. FrameVGGT summarizes each block into a compact prototype and maintains a fixed-capacity mid-term bank of complementary frame blocks under strict budgets, with an optional lightweight anchor tier for rare prolonged degradation. Across long-sequence 3D reconstruction, video depth estimation, and camera pose benchmarks, FrameVGGT achieves favorable accuracy--memory trade-offs under bounded memory, while maintaining more stable geometry over long streams.

  </details>



- **Ref-DGS: Reflective Dual Gaussian Splatting**  
  Ningjing Fan, Yiqun Wang, Dongming Yan, Peter Wonka  
  _2026-03-08_ · https://arxiv.org/abs/2603.07664v1  
  <details><summary>Abstract</summary>

  Reflective appearance, especially strong and typically near-field specular reflections, poses a fundamental challenge for accurate surface reconstruction and novel view synthesis. Existing Gaussian splatting methods either fail to model near-field specular reflections or rely on explicit ray tracing at substantial computational cost. We present Ref-DGS, a reflective dual Gaussian splatting framework that addresses this trade-off by decoupling surface reconstruction from specular reflection within an efficient rasterization-based pipeline. Ref-DGS introduces a dual Gaussian scene representation consisting of geometry Gaussians and complementary local reflection Gaussians that capture near-field specular interactions without explicit ray tracing, along with a global environment reflection field for modeling far-field specular reflections. To predict specular radiance, we further propose a lightweight, physically-aware adaptive mixing shader that fuses global and local reflection features. Experiments demonstrate that Ref-DGS achieves state-of-the-art performance on reflective scenes while training substantially faster than ray-based Gaussian methods.

  </details>



- **Fast Attention-Based Simplification of LiDAR Point Clouds for Object Detection and Classification**  
  Z. Rozsa, Á. Madaras, Q. Wei, X. Lu, M. Golarits, H. Yuan, T. Sziranyi, R. Hamzaoui  
  _2026-03-08_ · https://arxiv.org/abs/2603.07593v1  
  <details><summary>Abstract</summary>

  LiDAR point clouds are widely used in autonomous driving and consist of large numbers of 3D points captured at high frequency to represent surrounding objects such as vehicles, pedestrians, and traffic signs. While this dense data enables accurate perception, it also increases computational cost and power consumption, which can limit real-time deployment. Existing point cloud sampling methods typically face a trade-off: very fast approaches tend to reduce accuracy, while more accurate methods are computationally expensive. To address this limitation, we propose an efficient learned point cloud simplification method for LiDAR data. The method combines a feature embedding module with an attention-based sampling module to prioritize task-relevant regions and is trained end-to-end. We evaluate the method against farthest point sampling (FPS) and random sampling (RS) on 3D object detection on the KITTI dataset and on object classification across four datasets. The method was consistently faster than FPS and achieved similar, and in some settings better, accuracy, with the largest gains under aggressive downsampling. It was slower than RS, but it typically preserved accuracy more reliably at high sampling ratios.

  </details>



- **ACCURATE: Arbitrary-shaped Continuum Reconstruction Under Robust Adaptive Two-view Estimation**  
  Yaozhi Zhang, Shun Yu, Yugang Zhang, Yang Liu  
  _2026-03-08_ · https://arxiv.org/abs/2603.07533v1  
  <details><summary>Abstract</summary>

  Accurate reconstruction of arbitrary-shaped long slender continuum bodies, such as guidewires, catheters and other soft continuum manipulators, is essential for accurate mechanical simulation. However, existing image-based reconstruction approaches often suffer from limited accuracy because they often underutilize camera geometry, or lack generality as they rely on rigid geometric assumptions that may fail for continuum robots with complex and highly deformable shapes. To address these limitations, we propose ACCURATE, a 3D reconstruction framework integrating an image segmentation neural network with a geometry-constrained topology traversal and dynamic programming algorithm that enforces global biplanar geometric consistency, minimizes the cumulative point-to-epipolar-line distance, and remains robust to occlusions and epipolar ambiguities cases caused by noise and discretization. Our method achieves high reconstruction accuracy on both simulated and real phantom datasets acquired using a clinical X-ray C-arm system, with mean absolute errors below 1.0 mm.

  </details>



- **High-Fidelity Medical Shape Generation via Skeletal Latent Diffusion**  
  Guoqing Zhang, Jingyun Yang, Siqi Chen, Anping Zhang, Yang Li  
  _2026-03-08_ · https://arxiv.org/abs/2603.07504v1  
  <details><summary>Abstract</summary>

  Anatomy shape modeling is a fundamental problem in medical data analysis. However, the geometric complexity and topological variability of anatomical structures pose significant challenges to accurate anatomical shape generation. In this work, we propose a skeletal latent diffusion framework that explicitly incorporates structural priors for efficient and high-fidelity medical shape generation. We introduce a shape auto-encoder in which the encoder captures global geometric information through a differentiable skeletonization module and aggregates local surface features into shape latents, while the decoder predicts the corresponding implicit fields over sparsely sampled coordinates. New shapes are generated via a latent-space diffusion model, followed by neural implicit decoding and mesh extraction. To address the limited availability of medical shape data, we construct a large-scale dataset, \textit{MedSDF}, comprising surface point clouds and corresponding signed distance fields across multiple anatomical categories. Extensive experiments on MedSDF and vessel datasets demonstrate that the proposed method achieves superior reconstruction and generation quality while maintaining a higher computational efficiency compared with existing approaches. Code is available at: https://github.com/wlsdzyzl/meshage.

  </details>



- **SLNet: A Super-Lightweight Geometry-Adaptive Network for 3D Point Cloud Recognition**  
  Mohammad Saeid, Amir Salarpour, Pedram MohajerAnsari, Mert D. Pesé  
  _2026-03-08_ · https://arxiv.org/abs/2603.07454v1  
  <details><summary>Abstract</summary>

  We present SLNet, a lightweight backbone for 3D point cloud recognition designed to achieve strong performance without the computational cost of many recent attention, graph, and deep MLP based models. The model is built on two simple ideas: NAPE (Nonparametric Adaptive Point Embedding), which captures spatial structure using a combination of Gaussian RBF and cosine bases with input adaptive bandwidth and blending, and GMU (Geometric Modulation Unit), a per channel affine modulator that adds only 2D learnable parameters. These components are used within a four stage hierarchical encoder with FPS+kNN grouping, nonparametric normalization, and shared residual MLPs. In experiments, SLNet shows that a very small model can still remain highly competitive across several 3D recognition tasks. On ModelNet40, SLNet-S with 0.14M parameters and 0.31 GFLOPs achieves 93.64% overall accuracy, outperforming PointMLP-elite with 5x fewer parameters, while SLNet-M with 0.55M parameters and 1.22 GFLOPs reaches 93.92%, exceeding PointMLP with 24x fewer parameters. On ScanObjectNN, SLNet-M achieves 84.25% overall accuracy within 1.2 percentage points of PointMLP while using 28x fewer parameters. For large scale scene segmentation, SLNet-T extends the backbone with local Point Transformer attention and reaches 58.2% mIoU on S3DIS Area 5 with only 2.5M parameters, more than 17x fewer than Point Transformer V3. We also introduce NetScore+, which extends NetScore by incorporating latency and peak memory so that efficiency can be evaluated in a more deployment oriented way. Across multiple benchmarks and hardware settings, SLNet delivers a strong overall balance between accuracy and efficiency. Code is available at: https://github.com/m-saeid/SLNet.

  </details>



- **DogWeave: High-Fidelity 3D Canine Reconstruction from a Single Image via Normal Fusion and Conditional Inpainting**  
  Shufan Sun, Chenchen Wang, Zongfu Yu  
  _2026-03-08_ · https://arxiv.org/abs/2603.07441v1  
  <details><summary>Abstract</summary>

  Monocular 3D animal reconstruction is challenging due to complex articulation, self-occlusion, and fine-scale details such as fur. Existing methods often produce distorted geometry and inconsistent textures due to the lack of articulated 3D supervision and limited availability of back-view images in 2D datasets, which makes reconstructing unobserved regions particularly difficult. To address these limitations, we propose DogWeave, a model-based framework for reconstructing high-fidelity 3D canine models from a single RGB image. DogWeave improves geometry by refining a coarsely-initiated parametric mesh into a detailed SDF representation through multi-view normal field optimization using diffusion-enhanced normals. It then generates view-consistent textures through conditional partial inpainting guided by structure and style cues, enabling realistic reconstruction of unobserved regions. Using only about 7,000 dog images processed via our 2D pipeline for training, DogWeave produces complete, realistic 3D models and outperforms state-of-the-art single image to 3d reconstruction methods in both shape accuracy and texture realism for canines.

  </details>


