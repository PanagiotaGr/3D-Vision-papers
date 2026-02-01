# 3D Reconstruction

_Updated: 2026-02-01 07:04 UTC_

Total papers shown: **7**


---

- **MetricAnything: Scaling Metric Depth Pretraining with Noisy Heterogeneous Sources**  
  Baorui Ma, Jiahui Yang, Donglin Di, Xuancheng Zhang, Jianxun Cui, Hao Li, Yan Xie, Wei Chen  
  _2026-01-29_ · https://arxiv.org/abs/2601.22054v1  
  <details><summary>Abstract</summary>

  Scaling has powered recent advances in vision foundation models, yet extending this paradigm to metric depth estimation remains challenging due to heterogeneous sensor noise, camera-dependent biases, and metric ambiguity in noisy cross-source 3D data. We introduce Metric Anything, a simple and scalable pretraining framework that learns metric depth from noisy, diverse 3D sources without manually engineered prompts, camera-specific modeling, or task-specific architectures. Central to our approach is the Sparse Metric Prompt, created by randomly masking depth maps, which serves as a universal interface that decouples spatial reasoning from sensor and camera biases. Using about 20M image-depth pairs spanning reconstructed, captured, and rendered 3D data across 10000 camera models, we demonstrate-for the first time-a clear scaling trend in the metric depth track. The pretrained model excels at prompt-driven tasks such as depth completion, super-resolution and Radar-camera fusion, while its distilled prompt-free student achieves state-of-the-art results on monocular depth estimation, camera intrinsics recovery, single/multi-view metric 3D reconstruction, and VLA planning. We also show that using pretrained ViT of Metric Anything as a visual encoder significantly boosts Multimodal Large Language Model capabilities in spatial intelligence. These results show that metric depth estimation can benefit from the same scaling laws that drive modern foundation models, establishing a new path toward scalable and efficient real-world metric perception. We open-source MetricAnything at http://metric-anything.github.io/metric-anything-io/ to support community research.

  </details>



- **PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction**  
  Changjian Jiang, Kerui Ren, Xudong Li, Kaiwen Song, Linning Xu, Tao Lu, Junting Dong, Yu Zhang, Bo Dai, Mulin Yu  
  _2026-01-29_ · https://arxiv.org/abs/2601.22046v1  
  <details><summary>Abstract</summary>

  Streaming reconstruction from monocular image sequences remains challenging, as existing methods typically favor either high-quality rendering or accurate geometry, but rarely both. We present PLANING, an efficient on-the-fly reconstruction framework built on a hybrid representation that loosely couples explicit geometric primitives with neural Gaussians, enabling geometry and appearance to be modeled in a decoupled manner. This decoupling supports an online initialization and optimization strategy that separates geometry and appearance updates, yielding stable streaming reconstruction with substantially reduced structural redundancy. PLANING improves dense mesh Chamfer-L2 by 18.52% over PGSR, surpasses ARTDECO by 1.31 dB PSNR, and reconstructs ScanNetV2 scenes in under 100 seconds, over 5x faster than 2D Gaussian Splatting, while matching the quality of offline per-scene optimization. Beyond reconstruction quality, the structural clarity and computational efficiency of \modelname~make it well suited for a broad range of downstream applications, such as enabling large-scale scene modeling and simulation-ready environments for embodied AI. Project page: https://city-super.github.io/PLANING/ .

  </details>



- **Urban Neural Surface Reconstruction from Constrained Sparse Aerial Imagery with 3D SAR Fusion**  
  Da Li, Chen Yao, Tong Mao, Jiacheng Bao, Houjun Sun  
  _2026-01-29_ · https://arxiv.org/abs/2601.22045v1  
  <details><summary>Abstract</summary>

  Neural surface reconstruction (NSR) has recently shown strong potential for urban 3D reconstruction from multi-view aerial imagery. However, existing NSR methods often suffer from geometric ambiguity and instability, particularly under sparse-view conditions. This issue is critical in large-scale urban remote sensing, where aerial image acquisition is limited by flight paths, terrain, and cost. To address this challenge, we present the first urban NSR framework that fuses 3D synthetic aperture radar (SAR) point clouds with aerial imagery for high-fidelity reconstruction under constrained, sparse-view settings. 3D SAR can efficiently capture large-scale geometry even from a single side-looking flight path, providing robust priors that complement photometric cues from images. Our framework integrates radar-derived spatial constraints into an SDF-based NSR backbone, guiding structure-aware ray selection and adaptive sampling for stable and efficient optimization. We also construct the first benchmark dataset with co-registered 3D SAR point clouds and aerial imagery, facilitating systematic evaluation of cross-modal 3D reconstruction. Extensive experiments show that incorporating 3D SAR markedly enhances reconstruction accuracy, completeness, and robustness compared with single-modality baselines under highly sparse and oblique-view conditions, highlighting a viable route toward scalable high-fidelity urban reconstruction with advanced airborne and spaceborne optical-SAR sensing.

  </details>



- **Synthetic-to-Real Domain Bridging for Single-View 3D Reconstruction of Ships for Maritime Monitoring**  
  Borja Carrillo-Perez, Felix Sattler, Angel Bueno Rodriguez, Maurice Stephan, Sarah Barnes  
  _2026-01-29_ · https://arxiv.org/abs/2601.21786v1  
  <details><summary>Abstract</summary>

  Three-dimensional (3D) reconstruction of ships is an important part of maritime monitoring, allowing improved visualization, inspection, and decision-making in real-world monitoring environments. However, most state-ofthe-art 3D reconstruction methods require multi-view supervision, annotated 3D ground truth, or are computationally intensive, making them impractical for real-time maritime deployment. In this work, we present an efficient pipeline for single-view 3D reconstruction of real ships by training entirely on synthetic data and requiring only a single view at inference. Our approach uses the Splatter Image network, which represents objects as sparse sets of 3D Gaussians for rapid and accurate reconstruction from single images. The model is first fine-tuned on synthetic ShapeNet vessels and further refined with a diverse custom dataset of 3D ships, bridging the domain gap between synthetic and real-world imagery. We integrate a state-of-the-art segmentation module based on YOLOv8 and custom preprocessing to ensure compatibility with the reconstruction network. Postprocessing steps include real-world scaling, centering, and orientation alignment, followed by georeferenced placement on an interactive web map using AIS metadata and homography-based mapping. Quantitative evaluation on synthetic validation data demonstrates strong reconstruction fidelity, while qualitative results on real maritime images from the ShipSG dataset confirm the potential for transfer to operational maritime settings. The final system provides interactive 3D inspection of real ships without requiring real-world 3D annotations. This pipeline provides an efficient, scalable solution for maritime monitoring and highlights a path toward real-time 3D ship visualization in practical applications. Interactive demo: https://dlr-mi.github.io/ship3d-demo/.

  </details>



- **4D-CAAL: 4D Radar-Camera Calibration and Auto-Labeling for Autonomous Driving**  
  Shanliang Yao, Zhuoxiao Li, Runwei Guan, Kebin Cao, Meng Xia, Fuping Hu, Sen Xu, Yong Yue, Xiaohui Zhu, Weiping Ding, et al.  
  _2026-01-29_ · https://arxiv.org/abs/2601.21454v1  
  <details><summary>Abstract</summary>

  4D radar has emerged as a critical sensor for autonomous driving, primarily due to its enhanced capabilities in elevation measurement and higher resolution compared to traditional 3D radar. Effective integration of 4D radar with cameras requires accurate extrinsic calibration, and the development of radar-based perception algorithms demands large-scale annotated datasets. However, existing calibration methods often employ separate targets optimized for either visual or radar modalities, complicating correspondence establishment. Furthermore, manually labeling sparse radar data is labor-intensive and unreliable. To address these challenges, we propose 4D-CAAL, a unified framework for 4D radar-camera calibration and auto-labeling. Our approach introduces a novel dual-purpose calibration target design, integrating a checkerboard pattern on the front surface for camera detection and a corner reflector at the center of the back surface for radar detection. We develop a robust correspondence matching algorithm that aligns the checkerboard center with the strongest radar reflection point, enabling accurate extrinsic calibration. Subsequently, we present an auto-labeling pipeline that leverages the calibrated sensor relationship to transfer annotations from camera-based segmentations to radar point clouds through geometric projection and multi-feature optimization. Extensive experiments demonstrate that our method achieves high calibration accuracy while significantly reducing manual annotation effort, thereby accelerating the development of robust multi-modal perception systems for autonomous driving.

  </details>



- **From Implicit Ambiguity to Explicit Solidity: Diagnosing Interior Geometric Degradation in Neural Radiance Fields for Dense 3D Scene Understanding**  
  Jiangsan Zhao, Jakob Geipel, Kryzysztof Kusnierek  
  _2026-01-29_ · https://arxiv.org/abs/2601.21421v1  
  <details><summary>Abstract</summary>

  Neural Radiance Fields (NeRFs) have emerged as a powerful paradigm for multi-view reconstruction, complementing classical photogrammetric pipelines based on Structure-from-Motion (SfM) and Multi-View Stereo (MVS). However, their reliability for quantitative 3D analysis in dense, self-occluding scenes remains poorly understood. In this study, we identify a fundamental failure mode of implicit density fields under heavy occlusion, which we term Interior Geometric Degradation (IGD). We show that transmittance-based volumetric optimization satisfies photometric supervision by reconstructing hollow or fragmented structures rather than solid interiors, leading to systematic instance undercounting. Through controlled experiments on synthetic datasets with increasing occlusion, we demonstrate that state-of-the-art mask-supervised NeRFs saturate at approximately 89% instance recovery in dense scenes, despite improved surface coherence and mask quality. To overcome this limitation, we introduce an explicit geometric pipeline based on Sparse Voxel Rasterization (SVRaster), initialized from SfM feature geometry. By projecting 2D instance masks onto an explicit voxel grid and enforcing geometric separation via recursive splitting, our approach preserves physical solidity and achieves a 95.8% recovery rate in dense clusters. A sensitivity analysis using degraded segmentation masks further shows that explicit SfM-based geometry is substantially more robust to supervision failure, recovering 43% more instances than implicit baselines. These results demonstrate that explicit geometric priors are a prerequisite for reliable quantitative analysis in highly self-occluding 3D scenes.

  </details>



- **Mesh Splatting for End-to-end Multiview Surface Reconstruction**  
  Ruiqi Zhang, Jiacheng Wu, Jie Chen  
  _2026-01-29_ · https://arxiv.org/abs/2601.21400v1  
  <details><summary>Abstract</summary>

  Surfaces are typically represented as meshes, which can be extracted from volumetric fields via meshing or optimized directly as surface parameterizations. Volumetric representations occupy 3D space and have a large effective receptive field along rays, enabling stable and efficient optimization via volumetric rendering; however, subsequent meshing often produces overly dense meshes and introduces accumulated errors. In contrast, pure surface methods avoid meshing but capture only boundary geometry with a single-layer receptive field, making it difficult to learn intricate geometric details and increasing reliance on priors (e.g., shading or normals). We bridge this gap by differentiably turning a surface representation into a volumetric one, enabling end-to-end surface reconstruction via volumetric rendering to model complex geometries. Specifically, we soften a mesh into multiple semi-transparent layers that remain differentiable with respect to the base mesh, endowing it with a controllable 3D receptive field. Combined with a splatting-based renderer and a topology-control strategy, our method can be optimized in about 20 minutes to achieve accurate surface reconstruction while substantially improving mesh quality.

  </details>


