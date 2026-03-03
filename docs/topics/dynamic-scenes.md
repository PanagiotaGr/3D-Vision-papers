# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-03 07:08 UTC_

Total papers shown: **8**


---

- **FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding**  
  Yiweng Xie, Bo He, Junke Wang, Xiangyu Zheng, Ziyi Ye, Zuxuan Wu  
  _2026-03-02_ · https://arxiv.org/abs/2603.02096v1  
  <details><summary>Abstract</summary>

  This paper presents FluxMem, a training-free framework for efficient streaming video understanding. FluxMem adaptively compresses redundant visual memory through a hierarchical, two-stage design: (1) a Temporal Adjacency Selection (TAS) module removes redundant visual tokens across adjacent frames, and (2) a Spatial Domain Consolidation (SDC) module further merges spatially repetitive regions within each frame into compact representations. To adapt effectively to dynamic scenes, we introduce a self-adaptive token compression mechanism in both TAS and SDC, which automatically determines the compression rate based on intrinsic scene statistics rather than manual tuning. Extensive experiments demonstrate that FluxMem achieves new state-of-the-art results on existing online video benchmarks, reaching 76.4 on StreamingBench and 67.2 on OVO-Bench under real-time settings, while reducing latency by 69.9% and peak GPU memory by 34.5% on OVO-Bench. Furthermore, it maintains strong offline performance, achieving 73.1 on MLVU while using 65% fewer visual tokens.

  </details>



- **LaST-VLA: Thinking in Latent Spatio-Temporal Space for Vision-Language-Action in Autonomous Driving**  
  Yuechen Luo, Fang Li, Shaoqing Xu, Yang Ji, Zehan Zhang, Bing Wang, Yuannan Shen, Jianwei Cui, Long Chen, Guang Chen, et al.  
  _2026-03-02_ · https://arxiv.org/abs/2603.01928v1  
  <details><summary>Abstract</summary>

  While Vision-Language-Action (VLA) models have revolutionized autonomous driving by unifying perception and planning, their reliance on explicit textual Chain-of-Thought (CoT) leads to semantic-perceptual decoupling and perceptual-symbolic conflicts. Recent shifts toward latent reasoning attempt to bypass these bottlenecks by thinking in continuous hidden space. However, without explicit intermediate constraints, standard latent CoT often operates as a physics-agnostic representation. To address this, we propose the Latent Spatio-Temporal VLA (LaST-VLA), a framework shifting the reasoning paradigm from discrete symbolic processing into a physically grounded Latent Spatio-Temporal CoT. By implementing a dual-feature alignment mechanism, we distill geometric constraints from 3D foundation models and dynamic foresight from world models directly into the latent space. Coupled with a progressive SFT training strategy that transitions from feature alignment to trajectory generation, and refined via Reinforcement Learning with Group Relative Policy Optimization (GRPO) to ensure safety and rule compliance. \method~setting a new record on NAVSIM v1 (91.3 PDMS) and NAVSIM v2 (87.1 EPDMS), while excelling in spatial-temporal reasoning on SURDS and NuDynamics benchmarks.

  </details>



- **Training-Free Spatio-temporal Decoupled Reasoning Video Segmentation with Adaptive Object Memory**  
  Zhengtong Zhu, Jiaqing Fan, Zhixuan Liu, Fanzhang Li  
  _2026-03-02_ · https://arxiv.org/abs/2603.01545v1  
  <details><summary>Abstract</summary>

  Reasoning Video Object Segmentation (ReasonVOS) is a challenging task that requires stable object segmentation across video sequences using implicit and complex textual inputs. Previous methods fine-tune Multimodal Large Language Models (MLLMs) to produce segmentation outputs, which demand substantial resources. Additionally, some existing methods are coupled in the processing of spatio-temporal information, which affects the temporal stability of the model to some extent. To address these issues, we propose Training-Free \textbf{S}patio-temporal \textbf{D}ecoupled Reasoning Video Segmentation with \textbf{A}daptive Object \textbf{M}emory (SDAM). We aim to design a training-free reasoning video segmentation framework that outperforms existing methods requiring fine-tuning, using only pre-trained models. Meanwhile, we propose an Adaptive Object Memory module that selects and memorizes key objects based on motion cues in different video sequences. Finally, we propose Spatio-temporal Decoupling for stable temporal propagation. In the spatial domain, we achieve precise localization and segmentation of target objects, while in the temporal domain, we leverage key object temporal information to drive stable cross-frame propagation. Our method achieves excellent results on five benchmark datasets, including Ref-YouTubeVOS, Ref-DAVIS17, MeViS, ReasonVOS, and ReVOS.

  </details>



- **CoSMo3D: Open-World Promptable 3D Semantic Part Segmentation through LLM-Guided Canonical Spatial Modeling**  
  Li Jin, Weikai Chen, Yujie Wang, Yingda Yin, Zeyu Hu, Runze Zhang, Keyang Luo, Shengju Qian, Xin Wang, Xueying Qin  
  _2026-03-01_ · https://arxiv.org/abs/2603.01205v1  
  <details><summary>Abstract</summary>

  Open-world promptable 3D semantic segmentation remains brittle as semantics are inferred in the input sensor coordinates. Yet, humans, in contrast, interpret parts via functional roles in a canonical space -- wings extend laterally, handles protrude to the side, and legs support from below. Psychophysical evidence shows that we mentally rotate objects into canonical frames to reveal these roles. To fill this gap, we propose \methodName{}, which attains canonical space perception by inducing a latent canonical reference frame learned directly from data. By construction, we create a unified canonical dataset through LLM-guided intra- and cross-category alignment, exposing canonical spatial regularities across 200 categories. By induction, we realize canonicality inside the model through a dual-branch architecture with canonical map anchoring and canonical box calibration, collapsing pose variation and symmetry into a stable canonical embedding. This shift from input pose space to canonical embedding yields far more stable and transferable part semantics. Experimental results show that \methodName{} establishes new state of the art in open-world promptable 3D segmentation.

  </details>



- **Decoupling Motion and Geometry in 4D Gaussian Splatting**  
  Yi Zhang, Yulei Kang, Jian-Fang Hu  
  _2026-03-01_ · https://arxiv.org/abs/2603.00952v1  
  <details><summary>Abstract</summary>

  High-fidelity reconstruction of dynamic scenes is an important yet challenging problem. While recent 4D Gaussian Splatting (4DGS) has demonstrated the ability to model temporal dynamics, it couples Gaussian motion and geometric attributes within a single covariance formulation, which limits its expressiveness for complex motions and often leads to visual artifacts. To address this, we propose VeGaS, a novel velocity-based 4D Gaussian Splatting framework that decouples Gaussian motion and geometry. Specifically, we introduce a Galilean shearing matrix that explicitly incorporates time-varying velocity to flexibly model complex non-linear motions, while strictly isolating the effects of Gaussian motion from the geometry-related conditional Gaussian covariance. Furthermore, a Geometric Deformation Network is introduced to refine Gaussian shapes and orientations using spatio-temporal context and velocity cues, enhancing temporal geometric modeling. Extensive experiments on public datasets demonstrate that VeGaS achieves state-of-the-art performance.

  </details>



- **Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos**  
  Yu Luo, Guangyu Wei, Yangfan Li, Jieyu He, Yueming Lyu  
  _2026-03-01_ · https://arxiv.org/abs/2603.00881v1  
  <details><summary>Abstract</summary>

  Segmentation of the main coronary artery from X-ray coronary angiography (XCA) sequences is crucial for the diagnosis of coronary artery diseases. However, this task is challenging due to issues such as blurred boundaries, inconsistent radiation contrast, complex motion patterns, and a lack of annotated images for training. Although Semi-Supervised Learning (SSL) can alleviate the annotation burden, conventional methods struggle with complicated temporal dynamics and unreliable uncertainty quantification. To address these challenges, we propose SAM3-based Teacher-student framework with Motion-Aware consistency and Progressive Confidence Regularization (SMART), a semi-supervised vessel segmentation approach for X-ray angiography videos. First, our method utilizes SAM3's unique promptable concept segmentation design and innovates a SAM3-based teacher-student framework to maximize the performance potential of both the teacher and the student. Second, we enhance segmentation by integrating the vessel mask warping technique and motion consistency loss to model complex vessel dynamics. To address the issue of unreliable teacher predictions caused by blurred boundaries and minimal contrast, we further propose a progressive confidence-aware consistency regularization to mitigate the risk of unreliable outputs. Extensive experiments on three datasets of XCA sequences from different institutions demonstrate that SMART achieves state-of-the-art performance while requiring significantly fewer annotations, making it particularly valuable for real-world clinical applications where labeled data is scarce. Our code is available at: https://github.com/qimingfan10/SMART.

  </details>



- **Efficient Conformal Volumetry for Template-Based Segmentation**  
  Matt Y. Cheung, Ashok Veeraraghavan, Guha Balakrishnan  
  _2026-02-28_ · https://arxiv.org/abs/2603.00798v1  
  <details><summary>Abstract</summary>

  Template-based segmentation, a widely used paradigm in medical imaging, propagates anatomical labels via deformable registration from a labeled atlas to a target image, and is often used to compute volumetric biomarkers for downstream decision-making. While conformal prediction (CP) provides finite-sample valid intervals for scalar metrics, existing segmentation-based uncertainty quantification (UQ) approaches either rely on learned model features, often unavailable in classic template-based pipelines, or treat the registration process as a black box, resulting in overly conservative intervals when applied directly in output space. We introduce ConVOLT, a CP framework that achieves efficient volumetric UQ by conditioning calibration on properties of the estimated deformation field from template-based segmentation. ConVOLT calibrates a learned volumetric scaling factor from deformation space features. We evaluate ConVOLT on template-based segmentation tasks involving global, regional, and label volumetry across multiple datasets and registration methods. ConVOLT achieves target coverage while producing substantially tighter intervals than output-space conformal baselines. Our work paves way to exploit the registration process for efficient UQ in medical imaging pipelines.

  </details>



- **Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction: Dataset, Model and Benchmark**  
  Lijing Cai, Zhan Shi, Chenglong Huang, Jinyao Wu, Qiping Li, Zikang Huo, Linsen Chen, Chongde Zi, Xun Cao  
  _2026-02-28_ · https://arxiv.org/abs/2603.00611v1  
  <details><summary>Abstract</summary>

  Recently, Spectral Compressive Imaging (SCI) has achieved remarkable success, unlocking significant potential for dynamic spectral vision. However, existing reconstruction methods, primarily image-based, suffer from two limitations: (i) Encoding process masks spatial-spectral features, leading to uncertainty in reconstructing missing information from single compressed measurements, and (ii) The frame-by-frame reconstruction paradigm fails to ensure temporal consistency, which is crucial in the video perception. To address these challenges, this paper seeks to advance spectral reconstruction from the image level to the video level, leveraging the complementary features and temporal continuity across adjacent frames in dynamic scenes. Initially, we construct the first high-quality dynamic hyperspectral image dataset (DynaSpec), comprising 30 sequences obtained through frame-scanning acquisition. Subsequently, we propose the Propagation-Guided Spectral Video Reconstruction Transformer (PG-SVRT), which employs a spatial-then-temporal attention to effectively reconstruct spectral features from abundant video information, while using a bridged token to reduce computational complexity. Finally, we conduct simulation experiments to assess the performance of four SCI systems, and construct a DD-CASSI prototype for real-world data collection and benchmarking. Extensive experiments demonstrate that PG-SVRT achieves superior performance in reconstruction quality, spectral fidelity, and temporal consistency, while maintaining minimal FLOPs. Project page: https://github.com/nju-cite/DynaSpec

  </details>


