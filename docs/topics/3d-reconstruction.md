# 3D Reconstruction

_Updated: 2026-04-12 07:42 UTC_

Total papers shown: **8**


---

- **ETCH-X: Robustify Expressive Body Fitting to Clothed Humans with Composable Datasets**  
  Xiaoben Li, Jingyi Wu, Zeyu Cai, Yu Siyuan, Boqian Li, Yuliang Xiu  
  _2026-04-09_ · https://arxiv.org/abs/2604.08548v1  
  <details><summary>Abstract</summary>

  Human body fitting, which aligns parametric body models such as SMPL to raw 3D point clouds of clothed humans, serves as a crucial first step for downstream tasks like animation and texturing. An effective fitting method should be both locally expressive-capturing fine details such as hands and facial features-and globally robust to handle real-world challenges, including clothing dynamics, pose variations, and noisy or partial inputs. Existing approaches typically excel in only one aspect, lacking an all-in-one solution.We upgrade ETCH to ETCH-X, which leverages a tightness-aware fitting paradigm to filter out clothing dynamics ("undress"), extends expressiveness with SMPL-X, and replaces explicit sparse markers (which are highly sensitive to partial data) with implicit dense correspondences ("dense fit") for more robust and fine-grained body fitting. Our disentangled "undress" and "dense fit" modular stages enable separate and scalable training on composable data sources, including diverse simulated garments (CLOTH3D), large-scale full-body motions (AMASS), and fine-grained hand gestures (InterHand2.6M), improving outfit generalization and pose robustness of both bodies and hands. Our approach achieves robust and expressive fitting across diverse clothing, poses, and levels of input completeness, delivering a substantial performance improvement over ETCH on both: 1) seen data, such as 4D-Dress (MPJPE-All, 33.0% ) and CAPE (V2V-Hands, 35.8% ), and 2) unseen data, such as BEDLAM2.0 (MPJPE-All, 80.8% ; V2V-All, 80.5% ). Code and models will be released at https://xiaobenli00.github.io/ETCH-X/.

  </details>



- **Scal3R: Scalable Test-Time Training for Large-Scale 3D Reconstruction**  
  Tao Xie, Peishan Yang, Yudong Jin, Yingfeng Cai, Wei Yin, Weiqiang Ren, Qian Zhang, Wei Hua, Sida Peng, Xiaoyang Guo, et al.  
  _2026-04-09_ · https://arxiv.org/abs/2604.08542v1  
  <details><summary>Abstract</summary>

  This paper addresses the task of large-scale 3D scene reconstruction from long video sequences. Recent feed-forward reconstruction models have shown promising results by directly regressing 3D geometry from RGB images without explicit 3D priors or geometric constraints. However, these methods often struggle to maintain reconstruction accuracy and consistency over long sequences due to limited memory capacity and the inability to effectively capture global contextual cues. In contrast, humans can naturally exploit the global understanding of the scene to inform local perception. Motivated by this, we propose a novel neural global context representation that efficiently compresses and retains long-range scene information, enabling the model to leverage extensive contextual cues for enhanced reconstruction accuracy and consistency. The context representation is realized through a set of lightweight neural sub-networks that are rapidly adapted during test time via self-supervised objectives, which substantially increases memory capacity without incurring significant computational overhead. The experiments on multiple large-scale benchmarks, including the KITTI Odometry~\cite{Geiger2012CVPR} and Oxford Spires~\cite{tao2025spires} datasets, demonstrate the effectiveness of our approach in handling ultra-large scenes, achieving leading pose accuracy and state-of-the-art 3D reconstruction accuracy while maintaining efficiency. Code is available at https://zju3dv.github.io/scal3r.

  </details>



- **Self-Improving 4D Perception via Self-Distillation**  
  Nan Huang, Pengcheng Yu, Weijia Zeng, James M. Rehg, Angjoo Kanazawa, Haiwen Feng, Qianqian Wang  
  _2026-04-09_ · https://arxiv.org/abs/2604.08532v1  
  <details><summary>Abstract</summary>

  Large-scale multi-view reconstruction models have made remarkable progress, but most existing approaches still rely on fully supervised training with ground-truth 3D/4D annotations. Such annotations are expensive and particularly scarce for dynamic scenes, limiting scalability. We propose SelfEvo, a self-improving framework that continually improves pretrained multi-view reconstruction models using unlabeled videos. SelfEvo introduces a self-distillation scheme using spatiotemporal context asymmetry, enabling self-improvement for learning-based 4D perception without external annotations. We systematically study design choices that make self-improvement effective, including loss signals, forms of asymmetry, and other training strategies. Across eight benchmarks spanning diverse datasets and domains, SelfEvo consistently improves pretrained baselines and generalizes across base models (e.g. VGGT and $π^3$), with significant gains on dynamic scenes. Overall, SelfEvo achieves up to 36.5% relative improvement in video depth estimation and 20.1% in camera estimation, without using any labeled data. Project Page: https://self-evo.github.io/.

  </details>



- **SurfelSplat: Learning Efficient and Generalizable Gaussian Surfel Representations for Sparse-View Surface Reconstruction**  
  Chensheng Dai, Shengjun Zhang, Min Chen, Yueqi Duan  
  _2026-04-09_ · https://arxiv.org/abs/2604.08370v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has demonstrated impressive performance in 3D scene reconstruction. Beyond novel view synthesis, it shows great potential for multi-view surface reconstruction. Existing methods employ optimization-based reconstruction pipelines that achieve precise and complete surface extractions. However, these approaches typically require dense input views and high time consumption for per-scene optimization. To address these limitations, we propose SurfelSplat, a feed-forward framework that generates efficient and generalizable pixel-aligned Gaussian surfel representations from sparse-view images. We observe that conventional feed-forward structures struggle to recover accurate geometric attributes of Gaussian surfels because the spatial frequency of pixel-aligned primitives exceeds Nyquist sampling rates. Therefore, we propose a cross-view feature aggregation module based on the Nyquist sampling theorem. Specifically, we first adapt the geometric forms of Gaussian surfels with spatial sampling rate-guided low-pass filters. We then project the filtered surfels across all input views to obtain cross-view feature correlations. By processing these correlations through a specially designed feature fusion network, we can finally regress Gaussian surfels with precise geometry. Extensive experiments on DTU reconstruction benchmarks demonstrate that our model achieves comparable results with state-of-the-art methods, and predict Gaussian surfels within 1 second, offering a 100x speedup without costly per-scene training.

  </details>



- **Revisiting Radar Perception With Spectral Point Clouds**  
  Hamza Alsharif, Jing Gu, Pavol Jancura, Satish Ravindran, Gijs Dubbelman  
  _2026-04-09_ · https://arxiv.org/abs/2604.08282v1  
  <details><summary>Abstract</summary>

  Radar perception models are trained with different inputs, from range-Doppler spectra to sparse point clouds. Dense spectra are assumed to outperform sparse point clouds, yet they can vary considerably across sensors and configurations, which hinders transfer. In this paper, we provide alternatives for incorporating spectral information into radar point clouds and show that, point clouds need not underperform compared to spectra. We introduce the spectral point cloud paradigm, where point clouds are treated as sparse, compressed representations of the radar spectra, and argue that, when enriched with spectral information, they serve as strong candidates for a unified input representation that is more robust against sensor-specific differences. We develop an experimental framework that compares spectral point cloud (PC) models at varying densities against a dense range-Doppler (RD) benchmark, and report the density levels where the PC configurations meet the performance of the RD benchmark. Furthermore, we experiment with two basic spectral enrichment approaches, that inject additional target-relevant information into the point clouds. Contrary to the common belief that the dense RD approach is superior, we show that point clouds can do just as well, and can surpass the RD benchmark when enrichment is applied. Spectral point clouds can therefore serve as strong candidates for unified radar perception, paving the way for future radar foundation models.

  </details>



- **Brain3D: EEG-to-3D Decoding of Visual Representations via Multimodal Reasoning**  
  Emanuele Balloni, Emanuele Frontoni, Chiara Matti, Marina Paolanti, Roberto Pierdicca, Emiliano Santarnecchi  
  _2026-04-09_ · https://arxiv.org/abs/2604.08068v1  
  <details><summary>Abstract</summary>

  Decoding visual information from electroencephalography (EEG) has recently achieved promising results, primarily focusing on reconstructing two-dimensional (2D) images from brain activity. However, the reconstruction of three-dimensional (3D) representations remains largely unexplored. This limits the geometric understanding and reduces the applicability of neural decoding in different contexts. To address this gap, we propose Brain3D, a multimodal architecture for EEG-to-3D reconstruction based on EEG-to-image decoding. It progressively transforms neural representations into the 3D domain using geometry-aware generative reasoning. Our pipeline first produces visually grounded images from EEG signals, then employs a multimodal large language model to extract structured 3D-aware descriptions, which guide a diffusion-based generation stage whose outputs are finally converted into coherent 3D meshes via a single-image-to-3D model. By decomposing the problem into structured stages, the proposed approach avoids direct EEG-to-3D mappings and enables scalable brain-driven 3D generation. We conduct a comprehensive evaluation comparing the reconstructed 3D outputs against the original visual stimuli, assessing both semantic alignment and geometric fidelity. Experimental results demonstrate strong performance of the proposed architecture, achieving up to 85.4% 10-way Top-1 EEG decoding accuracy and 0.648 CLIPScore, supporting the feasibility of multimodal EEG-driven 3D reconstruction.

  </details>



- **SceneScribe-1M: A Large-Scale Video Dataset with Comprehensive Geometric and Semantic Annotations**  
  Yunnan Wang, Kecheng Zheng, Jianyuan Wang, Minghao Chen, David Novotny, Christian Rupprecht, Yinghao Xu, Xing Zhu, Wenjun Zeng, Xin Jin, et al.  
  _2026-04-09_ · https://arxiv.org/abs/2604.07990v1  
  <details><summary>Abstract</summary>

  The convergence of 3D geometric perception and video synthesis has created an unprecedented demand for large-scale video data that is rich in both semantic and spatio-temporal information. While existing datasets have advanced either 3D understanding or video generation, a significant gap remains in providing a unified resource that supports both domains at scale. To bridge this chasm, we introduce SceneScribe-1M, a new large-scale, multi-modal video dataset. It comprises one million in-the-wild videos, each meticulously annotated with detailed textual descriptions, precise camera parameters, dense depth maps, and consistent 3D point tracks. We demonstrate the versatility and value of SceneScribe-1M by establishing benchmarks across a wide array of downstream tasks, including monocular depth estimation, scene reconstruction, and dynamic point tracking, as well as generative tasks such as text-to-video synthesis, with or without camera control. By open-sourcing SceneScribe-1M, we aim to provide a comprehensive benchmark and a catalyst for research, fostering the development of models that can both perceive the dynamic 3D world and generate controllable, realistic video content.

  </details>



- **Object-Centric Stereo Ranging for Autonomous Driving: From Dense Disparity to Census-Based Template Matching**  
  Qihao Huang  
  _2026-04-09_ · https://arxiv.org/abs/2604.07980v1  
  <details><summary>Abstract</summary>

  Accurate depth estimation is critical for autonomous driving perception systems, particularly for long range vehicle detection on highways. Traditional dense stereo matching methods such as Block Matching (BM) and Semi Global Matching (SGM) produce per pixel disparity maps but suffer from high computational cost, sensitivity to radiometric differences between stereo cameras, and poor accuracy at long range where disparity values are small. In this report, we present a comprehensive stereo ranging system that integrates three complementary depth estimation approaches: dense BM/SGM disparity, object centric Census based template matching, and monocular geometric priors, within a unified detection ranging tracking pipeline. Our key contribution is a novel object centric Census based template matching algorithm that performs GPU accelerated sparse stereo matching directly within detected bounding boxes, employing a far close divide and conquer strategy, forward backward verification, occlusion aware sampling, and robust multi block aggregation. We further describe an online calibration refinement framework that combines auto rectification offset search, radar stereo voting based disparity correction, and object level radar stereo association for continuous extrinsic drift compensation. The complete system achieves real time performance through asynchronous GPU pipeline design and delivers robust ranging across diverse driving conditions including nighttime, rain, and varying illumination.

  </details>


