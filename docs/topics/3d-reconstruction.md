# 3D Reconstruction

_Updated: 2026-01-27 06:53 UTC_

Total papers shown: **9**


---

- **Splat-Portrait: Generalizing Talking Heads with Gaussian Splatting**  
  Tong Shi, Melonie de Almeida, Daniela Ivanova, Nicolas Pugeault, Paul Henderson  
  _2026-01-26_ · https://arxiv.org/abs/2601.18633v1  
  <details><summary>Abstract</summary>

  Talking Head Generation aims at synthesizing natural-looking talking videos from speech and a single portrait image. Previous 3D talking head generation methods have relied on domain-specific heuristics such as warping-based facial motion representation priors to animate talking motions, yet still produce inaccurate 3D avatar reconstructions, thus undermining the realism of generated animations. We introduce Splat-Portrait, a Gaussian-splatting-based method that addresses the challenges of 3D head reconstruction and lip motion synthesis. Our approach automatically learns to disentangle a single portrait image into a static 3D reconstruction represented as static Gaussian Splatting, and a predicted whole-image 2D background. It then generates natural lip motion conditioned on input audio, without any motion driven priors. Training is driven purely by 2D reconstruction and score-distillation losses, without 3D supervision nor landmarks. Experimental results demonstrate that Splat-Portrait exhibits superior performance on talking head generation and novel view synthesis, achieving better visual quality compared to previous works. Our project code and supplementary documents are public available at https://github.com/stonewalking/Splat-portrait.

  </details>



- **PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction**  
  Isaac Deutsch, Nicolas Moënne-Loccoz, Gavriel State, Zan Gojcic  
  _2026-01-26_ · https://arxiv.org/abs/2601.18336v1  
  <details><summary>Abstract</summary>

  Multi-view 3D reconstruction methods remain highly sensitive to photometric inconsistencies arising from camera optical characteristics and variations in image signal processing (ISP). Existing mitigation strategies such as per-frame latent variables or affine color corrections lack physical grounding and generalize poorly to novel views. We propose the Physically-Plausible ISP (PPISP) correction module, which disentangles camera-intrinsic and capture-dependent effects through physically based and interpretable transformations. A dedicated PPISP controller, trained on the input views, predicts ISP parameters for novel viewpoints, analogous to auto exposure and auto white balance in real cameras. This design enables realistic and fair evaluation on novel views without access to ground-truth images. PPISP achieves SoTA performance on standard benchmarks, while providing intuitive control and supporting the integration of metadata when available. The source code is available at: https://github.com/nv-tlabs/ppisp

  </details>



- **Contextual Range-View Projection for 3D LiDAR Point Clouds**  
  Seyedali Mousavi, Seyedhamidreza Mousavi, Masoud Daneshtalab  
  _2026-01-26_ · https://arxiv.org/abs/2601.18301v1  
  <details><summary>Abstract</summary>

  Range-view projection provides an efficient method for transforming 3D LiDAR point clouds into 2D range image representations, enabling effective processing with 2D deep learning models. However, a major challenge in this projection is the many-to-one conflict, where multiple 3D points are mapped onto the same pixel in the range image, requiring a selection strategy. Existing approaches typically retain the point with the smallest depth (closest to the LiDAR), disregarding semantic relevance and object structure, which leads to the loss of important contextual information. In this paper, we extend the depth-based selection rule by incorporating contextual information from both instance centers and class labels, introducing two mechanisms: \textit{Centerness-Aware Projection (CAP)} and \textit{Class-Weighted-Aware Projection (CWAP)}. In CAP, point depths are adjusted according to their distance from the instance center, thereby prioritizing central instance points over noisy boundary and background points. In CWAP, object classes are prioritized through user-defined weights, offering flexibility in the projection strategy. Our evaluations on the SemanticKITTI dataset show that CAP preserves more instance points during projection, achieving up to a 3.1\% mIoU improvement compared to the baseline. Furthermore, CWAP enhances the performance of targeted classes while having a negligible impact on the performance of other classes

  </details>



- **Depth to Anatomy: Learning Internal Organ Locations from Surface Depth Images**  
  Eytan Kats, Kai Geissler, Daniel Mensing, Jochen G. Hirsch, Stefan Heldman, Mattias P. Heinrich  
  _2026-01-26_ · https://arxiv.org/abs/2601.18260v1  
  <details><summary>Abstract</summary>

  Automated patient positioning plays an important role in optimizing scanning procedure and improving patient throughput. Leveraging depth information captured by RGB-D cameras presents a promising approach for estimating internal organ positions, thereby enabling more accurate and efficient positioning. In this work, we propose a learning-based framework that directly predicts the 3D locations and shapes of multiple internal organs from single 2D depth images of the body surface. Utilizing a large-scale dataset of full-body MRI scans, we synthesize depth images paired with corresponding anatomical segmentations to train a unified convolutional neural network architecture. Our method accurately localizes a diverse set of anatomical structures, including bones and soft tissues, without requiring explicit surface reconstruction. Experimental results demonstrate the potential of integrating depth sensors into radiology workflows to streamline scanning procedures and enhance patient experience through automated patient positioning.

  </details>



- **Revisiting 3D Reconstruction Kernels as Low-Pass Filters**  
  Shengjun Zhang, Min Chen, Yibo Wei, Mingyu Dong, Yueqi Duan  
  _2026-01-25_ · https://arxiv.org/abs/2601.17900v1  
  <details><summary>Abstract</summary>

  3D reconstruction is to recover 3D signals from the sampled discrete 2D pixels, with the goal to converge continuous 3D spaces. In this paper, we revisit 3D reconstruction from the perspective of signal processing, identifying the periodic spectral extension induced by discrete sampling as the fundamental challenge. Previous 3D reconstruction kernels, such as Gaussians, Exponential functions, and Student's t distributions, serve as the low pass filters to isolate the baseband spectrum. However, their unideal low-pass property results in the overlap of high-frequency components with low-frequency components in the discrete-time signal's spectrum. To this end, we introduce Jinc kernel with an instantaneous drop to zero magnitude exactly at the cutoff frequency, which is corresponding to the ideal low pass filters. As Jinc kernel suffers from low decay speed in the spatial domain, we further propose modulated kernels to strick an effective balance, and achieves superior rendering performance by reconciling spatial efficiency and frequency-domain fidelity. Experimental results have demonstrated the effectiveness of our Jinc and modulated kernels.

  </details>



- **Agreement-Driven Multi-View 3D Reconstruction for Live Cattle Weight Estimation**  
  Rabin Dulal, Wenfeng Jia, Lihong Zheng, Jane Quinn  
  _2026-01-25_ · https://arxiv.org/abs/2601.17791v1  
  <details><summary>Abstract</summary>

  Accurate cattle live weight estimation is vital for livestock management, welfare, and productivity. Traditional methods, such as manual weighing using a walk-over weighing system or proximate measurements using body condition scoring, involve manual handling of stock and can impact productivity from both a stock and economic perspective. To address these issues, this study investigated a cost-effective, non-contact method for live weight calculation in cattle using 3D reconstruction. The proposed pipeline utilized multi-view RGB images with SAM 3D-based agreement-guided fusion, followed by ensemble regression. Our approach generates a single 3D point cloud per animal and compares classical ensemble models with deep learning models under low-data conditions. Results show that SAM 3D with multi-view agreement fusion outperforms other 3D generation methods, while classical ensemble models provide the most consistent performance for practical farm scenarios (R$^2$ = 0.69 $\pm$ 0.10, MAPE = 2.22 $\pm$ 0.56 \%), making this practical for on-farm implementation. These findings demonstrate that improving reconstruction quality is more critical than increasing model complexity for scalable deployment on farms where producing a large volume of 3D data is challenging.

  </details>



- **Flatten The Complex: Joint B-Rep Generation via Compositional $k$-Cell Particles**  
  Junran Lu, Yuanqi Li, Hengji Li, Jie Guo, Yanwen Guo  
  _2026-01-25_ · https://arxiv.org/abs/2601.17733v1  
  <details><summary>Abstract</summary>

  Boundary Representation (B-Rep) is the widely adopted standard in Computer-Aided Design (CAD) and manufacturing. However, generative modeling of B-Reps remains a formidable challenge due to their inherent heterogeneity as geometric cell complexes, which entangles topology with geometry across cells of varying orders (i.e., $k$-cells such as vertices, edges, faces). Previous methods typically rely on cascaded sequences to handle this hierarchy, which fails to fully exploit the geometric relationships between cells, such as adjacency and sharing, limiting context awareness and error recovery. To fill this gap, we introduce a novel paradigm that reformulates B-Reps into sets of compositional $k$-cell particles. Our approach encodes each topological entity as a composition of particles, where adjacent cells share identical latents at their interfaces, thereby promoting geometric coupling along shared boundaries. By decoupling the rigid hierarchy, our representation unifies vertices, edges, and faces, enabling the joint generation of topology and geometry with global context awareness. We synthesize these particle sets using a multi-modal flow matching framework to handle unconditional generation as well as precise conditional tasks, such as 3D reconstruction from single-view or point cloud. Furthermore, the explicit and localized nature of our representation naturally extends to downstream tasks like local in-painting and enables the direct synthesis of non-manifold structures (e.g., wireframes). Extensive experiments demonstrate that our method produces high-fidelity CAD models with superior validity and editability compared to state-of-the-art methods.

  </details>



- **Advancing Structured Priors for Sparse-Voxel Surface Reconstruction**  
  Ting-Hsun Chi, Chu-Rong Chen, Chi-Tun Hsu, Hsuan-Ting Lin, Sheng-Yu Huang, Cheng Sun, Yu-Chiang Frank Wang  
  _2026-01-25_ · https://arxiv.org/abs/2601.17720v1  
  <details><summary>Abstract</summary>

  Reconstructing accurate surfaces with radiance fields has progressed rapidly, yet two promising explicit representations, 3D Gaussian Splatting and sparse-voxel rasterization, exhibit complementary strengths and weaknesses. 3D Gaussian Splatting converges quickly and carries useful geometric priors, but surface fidelity is limited by its point-like parameterization. Sparse-voxel rasterization provides continuous opacity fields and crisp geometry, but its typical uniform dense-grid initialization slows convergence and underutilizes scene structure. We combine the advantages of both by introducing a voxel initialization method that places voxels at plausible locations and with appropriate levels of detail, yielding a strong starting point for per-scene optimization. To further enhance depth consistency without blurring edges, we propose refined depth geometry supervision that converts multi-view cues into direct per-ray depth regularization. Experiments on standard benchmarks demonstrate improvements over prior methods in geometric accuracy, better fine-structure recovery, and more complete surfaces, while maintaining fast convergence.

  </details>



- **SPACE-CLIP: Spatial Perception via Adaptive CLIP Embeddings for Monocular Depth Estimation**  
  Taewan Cho, Taeryang Kim, Andrew Jaeyong Choi  
  _2026-01-25_ · https://arxiv.org/abs/2601.17657v1  
  <details><summary>Abstract</summary>

  Contrastive Language-Image Pre-training (CLIP) has accomplished extraordinary success for semantic understanding but inherently struggles to perceive geometric structure. Existing methods attempt to bridge this gap by querying CLIP with textual prompts, a process that is often indirect and inefficient. This paper introduces a fundamentally different approach using a dual-pathway decoder. We present SPACE-CLIP, an architecture that unlocks and interprets latent geometric knowledge directly from a frozen CLIP vision encoder, completely bypassing the text encoder and its associated textual prompts. A semantic pathway interprets high-level features, dynamically conditioned on global context using feature-wise linear modulation (FiLM). In addition, a structural pathway extracts fine-grained spatial details from early layers. These complementary streams are hierarchically fused, enabling a robust synthesis of semantic context and precise geometry. Extensive experiments on the KITTI benchmark show that SPACE-CLIP dramatically outperforms previous CLIP-based methods. Our ablation studies validate that the synergistic fusion of our dual pathways is critical to this success. SPACE-CLIP offers a new, efficient, and architecturally elegant blueprint for repurposing large-scale vision models. The proposed method is not just a standalone depth estimator, but a readily integrable spatial perception module for the next generation of embodied AI systems, such as vision-language-action (VLA) models. Our model is available at https://github.com/taewan2002/space-clip

  </details>


