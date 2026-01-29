# 3D Reconstruction

_Updated: 2026-01-29 07:04 UTC_

Total papers shown: **19**


---

- **Quartet of Diffusions: Structure-Aware Point Cloud Generation through Part and Symmetry Guidance**  
  Chenliang Zhou, Fangcheng Zhong, Weihao Xia, Albert Miao, Canberk Baykal, Cengiz Oztireli  
  _2026-01-28_ · https://arxiv.org/abs/2601.20425v1  
  <details><summary>Abstract</summary>

  We introduce the Quartet of Diffusions, a structure-aware point cloud generation framework that explicitly models part composition and symmetry. Unlike prior methods that treat shape generation as a holistic process or only support part composition, our approach leverages four coordinated diffusion models to learn distributions of global shape latents, symmetries, semantic parts, and their spatial assembly. This structured pipeline ensures guaranteed symmetry, coherent part placement, and diverse, high-quality outputs. By disentangling the generative process into interpretable components, our method supports fine-grained control over shape attributes, enabling targeted manipulation of individual parts while preserving global consistency. A central global latent further reinforces structural coherence across assembled parts. Our experiments show that the Quartet achieves state-of-the-art performance. To our best knowledge, this is the first 3D point cloud generation framework that fully integrates and enforces both symmetry and part priors throughout the generative process.

  </details>



- **GVGS: Gaussian Visibility-Aware Multi-View Geometry for Accurate Surface Reconstruction**  
  Mai Su, Qihan Yu, Zhongtao Wang, Yilong Li, Chengwei Pan, Yisong Chen, Guoping Wang  
  _2026-01-28_ · https://arxiv.org/abs/2601.20331v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting enables efficient optimization and high-quality rendering, yet accurate surface reconstruction remains challenging. Prior methods improve surface reconstruction by refining Gaussian depth estimates, either via multi-view geometric consistency or through monocular depth priors. However, multi-view constraints become unreliable under large geometric discrepancies, while monocular priors suffer from scale ambiguity and local inconsistency, ultimately leading to inaccurate Gaussian depth supervision. To address these limitations, we introduce a Gaussian visibility-aware multi-view geometric consistency constraint that aggregates the visibility of shared Gaussian primitives across views, enabling more accurate and stable geometric supervision. In addition, we propose a progressive quadtree-calibrated Monocular depth constraint that performs block-wise affine calibration from coarse to fine spatial scales, mitigating the scale ambiguity of depth priors while preserving fine-grained surface details. Extensive experiments on DTU and TNT datasets demonstrate consistent improvements in geometric accuracy over prior Gaussian-based and implicit surface reconstruction methods. Codes are available at an anonymous repository: https://github.com/GVGScode/GVGS.

  </details>



- **Physically Guided Visual Mass Estimation from a Single RGB Image**  
  Sungjae Lee, Junhan Jeong, Yeonjoo Hong, Kwang In Kim  
  _2026-01-28_ · https://arxiv.org/abs/2601.20303v1  
  <details><summary>Abstract</summary>

  Estimating object mass from visual input is challenging because mass depends jointly on geometric volume and material-dependent density, neither of which is directly observable from RGB appearance. Consequently, mass prediction from pixels is ill-posed and therefore benefits from physically meaningful representations to constrain the space of plausible solutions. We propose a physically structured framework for single-image mass estimation that addresses this ambiguity by aligning visual cues with the physical factors governing mass. From a single RGB image, we recover object-centric three-dimensional geometry via monocular depth estimation to inform volume and extract coarse material semantics using a vision-language model to guide density-related reasoning. These geometry, semantic, and appearance representations are fused through an instance-adaptive gating mechanism, and two physically guided latent factors (volume- and density-related) are predicted through separate regression heads under mass-only supervision. Experiments on image2mass and ABO-500 show that the proposed method consistently outperforms state-of-the-art methods.

  </details>



- **Size Matters: Reconstructing Real-Scale 3D Models from Monocular Images for Food Portion Estimation**  
  Gautham Vinod, Bruce Coburn, Siddeshwar Raghavan, Jiangpeng He, Fengqing Zhu  
  _2026-01-27_ · https://arxiv.org/abs/2601.20051v1  
  <details><summary>Abstract</summary>

  The rise of chronic diseases related to diet, such as obesity and diabetes, emphasizes the need for accurate monitoring of food intake. While AI-driven dietary assessment has made strides in recent years, the ill-posed nature of recovering size (portion) information from monocular images for accurate estimation of ``how much did you eat?'' is a pressing challenge. Some 3D reconstruction methods have achieved impressive geometric reconstruction but fail to recover the crucial real-world scale of the reconstructed object, limiting its usage in precision nutrition. In this paper, we bridge the gap between 3D computer vision and digital health by proposing a method that recovers a true-to-scale 3D reconstructed object from a monocular image. Our approach leverages rich visual features extracted from models trained on large-scale datasets to estimate the scale of the reconstructed object. This learned scale enables us to convert single-view 3D reconstructions into true-to-life, physically meaningful models. Extensive experiments and ablation studies on two publicly available datasets show that our method consistently outperforms existing techniques, achieving nearly a 30% reduction in mean absolute volume-estimation error, showcasing its potential to enhance the domain of precision nutrition. Code: https://gitlab.com/viper-purdue/size-matters

  </details>



- **GeoDiff3D: Self-Supervised 3D Scene Generation with Geometry-Constrained 2D Diffusion Guidance**  
  Haozhi Zhu, Miaomiao Zhao, Dingyao Liu, Runze Tian, Yan Zhang, Jie Guo, Fenggen Yu  
  _2026-01-27_ · https://arxiv.org/abs/2601.19785v2  
  <details><summary>Abstract</summary>

  3D scene generation is a core technology for gaming, film/VFX, and VR/AR. Growing demand for rapid iteration, high-fidelity detail, and accessible content creation has further increased interest in this area. Existing methods broadly follow two paradigms - indirect 2D-to-3D reconstruction and direct 3D generation - but both are limited by weak structural modeling and heavy reliance on large-scale ground-truth supervision, often producing structural artifacts, geometric inconsistencies, and degraded high-frequency details in complex scenes. We propose GeoDiff3D, an efficient self-supervised framework that uses coarse geometry as a structural anchor and a geometry-constrained 2D diffusion model to provide texture-rich reference images. Importantly, GeoDiff3D does not require strict multi-view consistency of the diffusion-generated references and remains robust to the resulting noisy, inconsistent guidance. We further introduce voxel-aligned 3D feature aggregation and dual self-supervision to maintain scene coherence and fine details while substantially reducing dependence on labeled data. GeoDiff3D also trains with low computational cost and enables fast, high-quality 3D scene generation. Extensive experiments on challenging scenes show improved generalization and generation quality over existing baselines, offering a practical solution for accessible and efficient 3D scene construction.

  </details>



- **WaterClear-GS: Optical-Aware Gaussian Splatting for Underwater Reconstruction and Restoration**  
  Xinrui Zhang, Yufeng Wang, Shuangkang Fang, Zesheng Wang, Dacheng Qi, Wenrui Ding  
  _2026-01-27_ · https://arxiv.org/abs/2601.19753v1  
  <details><summary>Abstract</summary>

  Underwater 3D reconstruction and appearance restoration are hindered by the complex optical properties of water, such as wavelength-dependent attenuation and scattering. Existing Neural Radiance Fields (NeRF)-based methods struggle with slow rendering speeds and suboptimal color restoration, while 3D Gaussian Splatting (3DGS) inherently lacks the capability to model complex volumetric scattering effects. To address these issues, we introduce WaterClear-GS, the first pure 3DGS-based framework that explicitly integrates underwater optical properties of local attenuation and scattering into Gaussian primitives, eliminating the need for an auxiliary medium network. Our method employs a dual-branch optimization strategy to ensure underwater photometric consistency while naturally recovering water-free appearances. This strategy is enhanced by depth-guided geometry regularization and perception-driven image loss, together with exposure constraints, spatially-adaptive regularization, and physically guided spectral regularization, which collectively enforce local 3D coherence and maintain natural visual perception. Experiments on standard benchmarks and our newly collected dataset demonstrate that WaterClear-GS achieves outstanding performance on both novel view synthesis (NVS) and underwater image restoration (UIR) tasks, while maintaining real-time rendering. The code will be available at https://buaaxrzhang.github.io/WaterClear-GS/.

  </details>



- **Cortex-Grounded Diffusion Models for Brain Image Generation**  
  Fabian Bongratz, Yitong Li, Sama Elbaroudy, Christian Wachinger  
  _2026-01-27_ · https://arxiv.org/abs/2601.19498v1  
  <details><summary>Abstract</summary>

  Synthetic neuroimaging data can mitigate critical limitations of real-world datasets, including the scarcity of rare phenotypes, domain shifts across scanners, and insufficient longitudinal coverage. However, existing generative models largely rely on weak conditioning signals, such as labels or text, which lack anatomical grounding and often produce biologically implausible outputs. To this end, we introduce Cor2Vox, a cortex-grounded generative framework for brain magnetic resonance image (MRI) synthesis that ties image generation to continuous structural priors of the cerebral cortex. It leverages high-resolution cortical surfaces to guide a 3D shape-to-image Brownian bridge diffusion process, enabling topologically faithful synthesis and precise control over underlying anatomies. To support the generation of new, realistic brain shapes, we developed a large-scale statistical shape model of cortical morphology derived from over 33,000 UK Biobank scans. We validated the fidelity of Cor2Vox based on traditional image quality metrics, advanced cortical surface reconstruction, and whole-brain segmentation quality, outperforming many baseline methods. Across three applications, namely (i) anatomically consistent synthesis, (ii) simulation of progressive gray matter atrophy, and (iii) harmonization of in-house frontotemporal dementia scans with public datasets, Cor2Vox preserved fine-grained cortical morphology at the sub-voxel level, exhibiting remarkable robustness to variations in cortical geometry and disease phenotype without retraining.

  </details>



- **Towards Gold-Standard Depth Estimation for Tree Branches in UAV Forestry: Benchmarking Deep Stereo Matching Methods**  
  Yida Lin, Bing Xue, Mengjie Zhang, Sam Schofield, Richard Green  
  _2026-01-27_ · https://arxiv.org/abs/2601.19461v1  
  <details><summary>Abstract</summary>

  Autonomous UAV forestry operations require robust depth estimation with strong cross-domain generalization, yet existing evaluations focus on urban and indoor scenarios, leaving a critical gap for vegetation-dense environments. We present the first systematic zero-shot evaluation of eight stereo methods spanning iterative refinement, foundation model, diffusion-based, and 3D CNN paradigms. All methods use officially released pretrained weights (trained on Scene Flow) and are evaluated on four standard benchmarks (ETH3D, KITTI 2012/2015, Middlebury) plus a novel 5,313-pair Canterbury Tree Branches dataset ($1920 \times 1080$). Results reveal scene-dependent patterns: foundation models excel on structured scenes (BridgeDepth: 0.23 px on ETH3D; DEFOM: 4.65 px on Middlebury), while iterative methods show variable cross-benchmark performance (IGEV++: 0.36 px on ETH3D but 6.77 px on Middlebury; IGEV: 0.33 px on ETH3D but 4.99 px on Middlebury). Qualitative evaluation on the Tree Branches dataset establishes DEFOM as the gold-standard baseline for vegetation depth estimation, with superior cross-domain consistency (consistently ranking 1st-2nd across benchmarks, average rank 1.75). DEFOM predictions will serve as pseudo-ground-truth for future benchmarking.

  </details>



- **Instance-Guided Radar Depth Estimation for 3D Object Detection**  
  Chen-Chou Lo, Patrick Vandewalle  
  _2026-01-27_ · https://arxiv.org/abs/2601.19314v1  
  <details><summary>Abstract</summary>

  Accurate depth estimation is fundamental to 3D perception in autonomous driving, supporting tasks such as detection, tracking, and motion planning. However, monocular camera-based 3D detection suffers from depth ambiguity and reduced robustness under challenging conditions. Radar provides complementary advantages such as resilience to poor lighting and adverse weather, but its sparsity and low resolution limit its direct use in detection frameworks. This motivates the need for effective Radar-camera fusion with improved preprocessing and depth estimation strategies. We propose an end-to-end framework that enhances monocular 3D object detection through two key components. First, we introduce InstaRadar, an instance segmentation-guided expansion method that leverages pre-trained segmentation masks to enhance Radar density and semantic alignment, producing a more structured representation. InstaRadar achieves state-of-the-art results in Radar-guided depth estimation, showing its effectiveness in generating high-quality depth features. Second, we integrate the pre-trained RCDPT into the BEVDepth framework as a replacement for its depth module. With InstaRadar-enhanced inputs, the RCDPT integration consistently improves 3D detection performance. Overall, these components yield steady gains over the baseline BEVDepth model, demonstrating the effectiveness of InstaRadar and the advantage of explicit depth supervision in 3D object detection. Although the framework lags behind Radar-camera fusion models that directly extract BEV features, since Radar serves only as guidance rather than an independent feature stream, this limitation highlights potential for improvement. Future work will extend InstaRadar to point cloud-like representations and integrate a dedicated Radar branch with temporal cues for enhanced BEV fusion.

  </details>



- **TIGaussian: Disentangle Gaussians for Spatial-Awared Text-Image-3D Alignment**  
  Jiarun Liu, Qifeng Chen, Yiru Zhao, Minghua Liu, Baorui Ma, Sheng Yang  
  _2026-01-27_ · https://arxiv.org/abs/2601.19247v1  
  <details><summary>Abstract</summary>

  While visual-language models have profoundly linked features between texts and images, the incorporation of 3D modality data, such as point clouds and 3D Gaussians, further enables pretraining for 3D-related tasks, e.g., cross-modal retrieval, zero-shot classification, and scene recognition. As challenges remain in extracting 3D modal features and bridging the gap between different modalities, we propose TIGaussian, a framework that harnesses 3D Gaussian Splatting (3DGS) characteristics to strengthen cross-modality alignment through multi-branch 3DGS tokenizer and modality-specific 3D feature alignment strategies. Specifically, our multi-branch 3DGS tokenizer decouples the intrinsic properties of 3DGS structures into compact latent representations, enabling more generalizable feature extraction. To further bridge the modality gap, we develop a bidirectional cross-modal alignment strategies: a multi-view feature fusion mechanism that leverages diffusion priors to resolve perspective ambiguity in image-3D alignment, while a text-3D projection module adaptively maps 3D features to text embedding space for better text-3D alignment. Extensive experiments on various datasets demonstrate the state-of-the-art performance of TIGaussian in multiple tasks.

  </details>



- **Resolving Primitive-Sharing Ambiguity in Long-Tailed Industrial Point Cloud Segmentation via Spatial Context Constraints**  
  Chao Yin, Qing Han, Zhiwei Hou, Yue Liu, Anjin Dai, Hongda Hu, Ji Yang, Wei Yao  
  _2026-01-27_ · https://arxiv.org/abs/2601.19128v1  
  <details><summary>Abstract</summary>

  Industrial point cloud segmentation for Digital Twin construction faces a persistent challenge: safety-critical components such as reducers and valves are systematically misclassified. These failures stem from two compounding factors: such components are rare in training data, yet they share identical local geometry with dominant structures like pipes. This work identifies a dual crisis unique to industrial 3D data extreme class imbalance 215:1 ratio compounded by geometric ambiguity where most tail classes share cylindrical primitives with head classes. Existing frequency-based re-weighting methods address statistical imbalance but cannot resolve geometric ambiguity. We propose spatial context constraints that leverage neighborhood prediction consistency to disambiguate locally similar structures. Our approach extends the Class-Balanced (CB) Loss framework with two architecture-agnostic mechanisms: (1) Boundary-CB, an entropy-based constraint that emphasizes ambiguous boundaries, and (2) Density-CB, a density-based constraint that compensates for scan-dependent variations. Both integrate as plug-and-play modules without network modifications, requiring only loss function replacement. On the Industrial3D dataset (610M points from water treatment facilities), our method achieves 55.74% mIoU with 21.7% relative improvement on tail-class performance (29.59% vs. 24.32% baseline) while preserving head-class accuracy (88.14%). Components with primitive-sharing ambiguity show dramatic gains: reducer improves from 0% to 21.12% IoU; valve improves by 24.3% relative. This resolves geometric ambiguity without the typical head-tail trade-off, enabling reliable identification of safety-critical components for automated knowledge extraction in Digital Twin applications.

  </details>



- **NuiWorld: Exploring a Scalable Framework for End-to-End Controllable World Generation**  
  Han-Hung Lee, Cheng-Yu Yang, Yu-Lun Liu, Angel X. Chang  
  _2026-01-27_ · https://arxiv.org/abs/2601.19048v1  
  <details><summary>Abstract</summary>

  World generation is a fundamental capability for applications like video games, simulation, and robotics. However, existing approaches face three main obstacles: controllability, scalability, and efficiency. End-to-end scene generation models have been limited by data scarcity. While object-centric generation approaches rely on fixed resolution representations, degrading fidelity for larger scenes. Training-free approaches, while flexible, are often slow and computationally expensive at inference time. We present NuiWorld, a framework that attempts to address these challenges. To overcome data scarcity, we propose a generative bootstrapping strategy that starts from a few input images. Leveraging recent 3D reconstruction and expandable scene generation techniques, we synthesize scenes of varying sizes and layouts, producing enough data to train an end-to-end model. Furthermore, our framework enables controllability through pseudo sketch labels, and demonstrates a degree of generalization to previously unseen sketches. Our approach represents scenes as a collection of variable scene chunks, which are compressed into a flattened vector-set representation. This significantly reduces the token length for large scenes, enabling consistent geometric fidelity across scenes sizes while improving training and inference efficiency.

  </details>



- **Non-Invasive 3D Wound Measurement with RGB-D Imaging**  
  Lena Harkämper, Leo Lebrat, David Ahmedt-Aristizabal, Olivier Salvado, Mattias Heinrich, Rodrigo Santa Cruz  
  _2026-01-26_ · https://arxiv.org/abs/2601.19014v1  
  <details><summary>Abstract</summary>

  Chronic wound monitoring and management require accurate and efficient wound measurement methods. This paper presents a fast, non-invasive 3D wound measurement algorithm based on RGB-D imaging. The method combines RGB-D odometry with B-spline surface reconstruction to generate detailed 3D wound meshes, enabling automatic computation of clinically relevant wound measurements such as perimeter, surface area, and dimensions. We evaluated our system on realistic silicone wound phantoms and measured sub-millimetre 3D reconstruction accuracy compared with high-resolution ground-truth scans. The extracted measurements demonstrated low variability across repeated captures and strong agreement with manual assessments. The proposed pipeline also outperformed a state-of-the-art object-centric RGB-D reconstruction method while maintaining runtimes suitable for real-time clinical deployment. Our approach offers a promising tool for automated wound assessment in both clinical and remote healthcare settings.

  </details>



- **FreeOrbit4D: Training-Free Arbitrary Camera Redirection for Monocular Videos via Geometry-Complete 4D Reconstruction**  
  Wei Cao, Hao Zhang, Fengrui Tian, Yulun Wu, Yingying Li, Shenlong Wang, Ning Yu, Yaoyao Liu  
  _2026-01-26_ · https://arxiv.org/abs/2601.18993v1  
  <details><summary>Abstract</summary>

  Camera redirection aims to replay a dynamic scene from a single monocular video under a user-specified camera trajectory. However, large-angle redirection is inherently ill-posed: a monocular video captures only a narrow spatio-temporal view of a dynamic 3D scene, providing highly partial observations of the underlying 4D world. The key challenge is therefore to recover a complete and coherent representation from this limited input, with consistent geometry and motion. While recent diffusion-based methods achieve impressive results, they often break down under large-angle viewpoint changes far from the original trajectory, where missing visual grounding leads to severe geometric ambiguity and temporal inconsistency. To address this, we present FreeOrbit4D, an effective training-free framework that tackles this geometric ambiguity by recovering a geometry-complete 4D proxy as structural grounding for video generation. We obtain this proxy by decoupling foreground and background reconstructions: we unproject the monocular video into a static background and geometry-incomplete foreground point clouds in a unified global space, then leverage an object-centric multi-view diffusion model to synthesize multi-view images and reconstruct geometry-complete foreground point clouds in canonical object space. By aligning the canonical foreground point cloud to the global scene space via dense pixel-synchronized 3D--3D correspondences and projecting the geometry-complete 4D proxy onto target camera viewpoints, we provide geometric scaffolds that guide a conditional video diffusion model. Extensive experiments show that FreeOrbit4D produces more faithful redirected videos under challenging large-angle trajectories, and our geometry-complete 4D proxy further opens a potential avenue for practical applications such as edit propagation and 4D data generation. Project page and code will be released soon.

  </details>



- **On the Role of Depth in Surgical Vision Foundation Models: An Empirical Study of RGB-D Pre-training**  
  John J. Han, Adam Schmidt, Muhammad Abdullah Jamal, Chinedu Nwoye, Anita Rau, Jie Ying Wu, Omid Mohareri  
  _2026-01-26_ · https://arxiv.org/abs/2601.18929v1  
  <details><summary>Abstract</summary>

  Vision foundation models (VFMs) have emerged as powerful tools for surgical scene understanding. However, current approaches predominantly rely on unimodal RGB pre-training, overlooking the complex 3D geometry inherent to surgical environments. Although several architectures support multimodal or geometry-aware inputs in general computer vision, the benefits of incorporating depth information in surgical settings remain underexplored. We conduct a large-scale empirical study comparing eight ViT-based VFMs that differ in pre-training domain, learning objective, and input modality (RGB vs. RGB-D). For pre-training, we use a curated dataset of 1.4 million robotic surgical images paired with depth maps generated from an off-the-shelf network. We evaluate these models under both frozen-backbone and end-to-end fine-tuning protocols across eight surgical datasets spanning object detection, segmentation, depth estimation, and pose estimation. Our experiments yield several consistent findings. Models incorporating explicit geometric tokenization, such as MultiMAE, substantially outperform unimodal baselines across all tasks. Notably, geometric-aware pre-training enables remarkable data efficiency: models fine-tuned on just 25% of labeled data consistently surpass RGB-only models trained on the full dataset. Importantly, these gains require no architectural or runtime changes at inference; depth is used only during pre-training, making adoption straightforward. These findings suggest that multimodal pre-training offers a viable path towards building more capable surgical vision systems.

  </details>



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


