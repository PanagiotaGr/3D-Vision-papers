# 3D Reconstruction

_Updated: 2026-02-02 07:16 UTC_

Total papers shown: **7**


---

- **Segment Any Events with Language**  
  Seungjun Lee, Gim Hee Lee  
  _2026-01-30_ · https://arxiv.org/abs/2601.23159v1  
  <details><summary>Abstract</summary>

  Scene understanding with free-form language has been widely explored within diverse modalities such as images, point clouds, and LiDAR. However, related studies on event sensors are scarce or narrowly centered on semantic-level understanding. We introduce SEAL, the first Semantic-aware Segment Any Events framework that addresses Open-Vocabulary Event Instance Segmentation (OV-EIS). Given the visual prompt, our model presents a unified framework to support both event segmentation and open-vocabulary mask classification at multiple levels of granularity, including instance-level and part-level. To enable thorough evaluation on OV-EIS, we curate four benchmarks that cover label granularity from coarse to fine class configurations and semantic granularity from instance-level to part-level understanding. Extensive experiments show that our SEAL largely outperforms proposed baselines in terms of performance and inference speed with a parameter-efficient architecture. In the Appendix, we further present a simple variant of our SEAL achieving generic spatiotemporal OV-EIS that does not require any visual prompts from users in the inference. Check out our project page in https://0nandon.github.io/SEAL

  </details>



- **FlowCalib: LiDAR-to-Vehicle Miscalibration Detection using Scene Flows**  
  Ilir Tahiraj, Peter Wittal, Markus Lienkamp  
  _2026-01-30_ · https://arxiv.org/abs/2601.23107v1  
  <details><summary>Abstract</summary>

  Accurate sensor-to-vehicle calibration is essential for safe autonomous driving. Angular misalignments of LiDAR sensors can lead to safety-critical issues during autonomous operation. However, current methods primarily focus on correcting sensor-to-sensor errors without considering the miscalibration of individual sensors that cause these errors in the first place. We introduce FlowCalib, the first framework that detects LiDAR-to-vehicle miscalibration using motion cues from the scene flow of static objects. Our approach leverages the systematic bias induced by rotational misalignment in the flow field generated from sequential 3D point clouds, eliminating the need for additional sensors. The architecture integrates a neural scene flow prior for flow estimation and incorporates a dual-branch detection network that fuses learned global flow features with handcrafted geometric descriptors. These combined representations allow the system to perform two complementary binary classification tasks: a global binary decision indicating whether misalignment is present and separate, axis-specific binary decisions indicating whether each rotational axis is misaligned. Experiments on the nuScenes dataset demonstrate FlowCalib's ability to robustly detect miscalibration, establishing a benchmark for sensor-to-vehicle miscalibration detection.

  </details>



- **Rethinking Transferable Adversarial Attacks on Point Clouds from a Compact Subspace Perspective**  
  Keke Tang, Xianheng Liu, Weilong Peng, Xiaofei Wang, Daizong Liu, Peican Zhu, Can Lu, Zhihong Tian  
  _2026-01-30_ · https://arxiv.org/abs/2601.23102v1  
  <details><summary>Abstract</summary>

  Transferable adversarial attacks on point clouds remain challenging, as existing methods often rely on model-specific gradients or heuristics that limit generalization to unseen architectures. In this paper, we rethink adversarial transferability from a compact subspace perspective and propose CoSA, a transferable attack framework that operates within a shared low-dimensional semantic space. Specifically, each point cloud is represented as a compact combination of class-specific prototypes that capture shared semantic structure, while adversarial perturbations are optimized within a low-rank subspace to induce coherent and architecture-agnostic variations. This design suppresses model-dependent noise and constrains perturbations to semantically meaningful directions, thereby improving cross-model transferability without relying on surrogate-specific artifacts. Extensive experiments on multiple datasets and network architectures demonstrate that CoSA consistently outperforms state-of-the-art transferable attacks, while maintaining competitive imperceptibility and robustness under common defense strategies. Codes will be made public upon paper acceptance.

  </details>



- **Self-Supervised Slice-to-Volume Reconstruction with Gaussian Representations for Fetal MRI**  
  Yinsong Wang, Thomas Fletcher, Xinzhe Luo, Aine Travers Dineen, Rhodri Cusack, Chen Qin  
  _2026-01-30_ · https://arxiv.org/abs/2601.22990v1  
  <details><summary>Abstract</summary>

  Reconstructing 3D fetal MR volumes from motion-corrupted stacks of 2D slices is a crucial and challenging task. Conventional slice-to-volume reconstruction (SVR) methods are time-consuming and require multiple orthogonal stacks for reconstruction. While learning-based SVR approaches have significantly reduced the time required at the inference stage, they heavily rely on ground truth information for training, which is inaccessible in practice. To address these challenges, we propose GaussianSVR, a self-supervised framework for slice-to-volume reconstruction. GaussianSVR represents the target volume using 3D Gaussian representations to achieve high-fidelity reconstruction. It leverages a simulated forward slice acquisition model to enable self-supervised training, alleviating the need for ground-truth volumes. Furthermore, to enhance both accuracy and efficiency, we introduce a multi-resolution training strategy that jointly optimizes Gaussian parameters and spatial transformations across different resolution levels. Experiments show that GaussianSVR outperforms the baseline methods on fetal MR volumetric reconstruction. Code will be available upon acceptance.

  </details>



- **Deep in the Jungle: Towards Automating Chimpanzee Population Estimation**  
  Tom Raynes, Otto Brookes, Timm Haucke, Lukas Bösch, Anne-Sophie Crunchant, Hjalmar Kühl, Sara Beery, Majid Mirmehdi, Tilo Burghardt  
  _2026-01-30_ · https://arxiv.org/abs/2601.22917v1  
  <details><summary>Abstract</summary>

  The estimation of abundance and density in unmarked populations of great apes relies on statistical frameworks that require animal-to-camera distance measurements. In practice, acquiring these distances depends on labour-intensive manual interpretation of animal observations across large camera trap video corpora. This study introduces and evaluates an only sparsely explored alternative: the integration of computer vision-based monocular depth estimation (MDE) pipelines directly into ecological camera trap workflows for great ape conservation. Using a real-world dataset of 220 camera trap videos documenting a wild chimpanzee population, we combine two MDE models, Dense Prediction Transformers and Depth Anything, with multiple distance sampling strategies. These components are used to generate detection distance estimates, from which population density and abundance are inferred. Comparative analysis against manually derived ground-truth distances shows that calibrated DPT consistently outperforms Depth Anything. This advantage is observed in both distance estimation accuracy and downstream density and abundance inference. Nevertheless, both models exhibit systematic biases. We show that, given complex forest environments, they tend to overestimate detection distances and consequently underestimate density and abundance relative to conventional manual approaches. We further find that failures in animal detection across distance ranges are a primary factor limiting estimation accuracy. Overall, this work provides a case study that shows MDE-driven camera trap distance sampling is a viable and practical alternative to manual distance estimation. The proposed approach yields population estimates within 22% of those obtained using traditional methods.

  </details>



- **Under-Canopy Terrain Reconstruction in Dense Forests Using RGB Imaging and Neural 3D Reconstruction**  
  Refael Sheffer, Chen Pinchover, Haim Zisman, Dror Ozeri, Roee Litman  
  _2026-01-30_ · https://arxiv.org/abs/2601.22861v1  
  <details><summary>Abstract</summary>

  Mapping the terrain and understory hidden beneath dense forest canopies is of great interest for numerous applications such as search and rescue, trail mapping, forest inventory tasks, and more. Existing solutions rely on specialized sensors: either heavy, costly airborne LiDAR, or Airborne Optical Sectioning (AOS), which uses thermal synthetic aperture photography and is tailored for person detection. We introduce a novel approach for the reconstruction of canopy-free, photorealistic ground views using only conventional RGB images. Our solution is based on the celebrated Neural Radiance Fields (NeRF), a recent 3D reconstruction method. Additionally, we include specific image capture considerations, which dictate the needed illumination to successfully expose the scene beneath the canopy. To better cope with the poorly lit understory, we employ a low light loss. Finally, we propose two complementary approaches to remove occluding canopy elements by controlling per-ray integration procedure. To validate the value of our approach, we present two possible downstream tasks. For the task of search and rescue (SAR), we demonstrate that our method enables person detection which achieves promising results compared to thermal AOS (using only RGB images). Additionally, we show the potential of our approach for forest inventory tasks like tree counting. These results position our approach as a cost-effective, high-resolution alternative to specialized sensors for SAR, trail mapping, and forest-inventory tasks.

  </details>



- **Diachronic Stereo Matching for Multi-Date Satellite Imagery**  
  Elías Masquil, Luca Savant Aira, Roger Marí, Thibaud Ehret, Pablo Musé, Gabriele Facciolo  
  _2026-01-30_ · https://arxiv.org/abs/2601.22808v1  
  <details><summary>Abstract</summary>

  Recent advances in image-based satellite 3D reconstruction have progressed along two complementary directions. On one hand, multi-date approaches using NeRF or Gaussian-splatting jointly model appearance and geometry across many acquisitions, achieving accurate reconstructions on opportunistic imagery with numerous observations. On the other hand, classical stereoscopic reconstruction pipelines deliver robust and scalable results for simultaneous or quasi-simultaneous image pairs. However, when the two images are captured months apart, strong seasonal, illumination, and shadow changes violate standard stereoscopic assumptions, causing existing pipelines to fail. This work presents the first Diachronic Stereo Matching method for satellite imagery, enabling reliable 3D reconstruction from temporally distant pairs. Two advances make this possible: (1) fine-tuning a state-of-the-art deep stereo network that leverages monocular depth priors, and (2) exposing it to a dataset specifically curated to include a diverse set of diachronic image pairs. In particular, we start from a pretrained MonSter model, trained initially on a mix of synthetic and real datasets such as SceneFlow and KITTI, and fine-tune it on a set of stereo pairs derived from the DFC2019 remote sensing challenge. This dataset contains both synchronic and diachronic pairs under diverse seasonal and illumination conditions. Experiments on multi-date WorldView-3 imagery demonstrate that our approach consistently surpasses classical pipelines and unadapted deep stereo models on both synchronic and diachronic settings. Fine-tuning on temporally diverse images, together with monocular priors, proves essential for enabling 3D reconstruction from previously incompatible acquisition dates. Left image (winter) Right image (autumn) DSM geometry Ours (1.23 m) Zero-shot (3.99 m) LiDAR GT Figure 1. Output geometry for a winter-autumn image pair from Omaha (OMA 331 test scene). Our method recovers accurate geometry despite the diachronic nature of the pair, exhibiting strong appearance changes, which cause existing zero-shot methods to fail. Missing values due to perspective shown in black. Mean altitude error in parentheses; lower is better.

  </details>


