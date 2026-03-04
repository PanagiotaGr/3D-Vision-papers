# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-04 07:04 UTC_

Total papers shown: **8**


---

- **The Dresden Dataset for 4D Reconstruction of Non-Rigid Abdominal Surgical Scenes**  
  Reuben Docea, Rayan Younis, Yonghao Long, Maxime Fleury, Jinjing Xu, Chenyang Li, André Schulze, Ann Wierick, Johannes Bender, Micha Pfeiffer, et al.  
  _2026-03-03_ · https://arxiv.org/abs/2603.02985v1  
  <details><summary>Abstract</summary>

  The D4D Dataset provides paired endoscopic video and high-quality structured-light geometry for evaluating 3D reconstruction of deforming abdominal soft tissue in realistic surgical conditions. Data were acquired from six porcine cadaver sessions using a da Vinci Xi stereo endoscope and a Zivid structured-light camera, registered via optical tracking and manually curated iterative alignment methods. Three sequence types - whole deformations, incremental deformations, and moved-camera clips - probe algorithm robustness to non-rigid motion, deformation magnitude, and out-of-view updates. Each clip provides rectified stereo images, per-frame instrument masks, stereo depth, start/end structured-light point clouds, curated camera poses and camera intrinsics. In postprocessing, ICP and semi-automatic registration techniques are used to register data, and instrument masks are created. The dataset enables quantitative geometric evaluation in both visible and occluded regions, alongside photometric view-synthesis baselines. Comprising over 300,000 frames and 369 point clouds across 98 curated recordings, this resource can serve as a comprehensive benchmark for developing and evaluating non-rigid SLAM, 4D reconstruction, and depth estimation methods.

  </details>



- **Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers**  
  Youngjun Jun, Seil Kang, Woojung Han, Seong Jae Hwang  
  _2026-03-03_ · https://arxiv.org/abs/2603.02919v1  
  <details><summary>Abstract</summary>

  Video Diffusion Transformers (DiTs) have been synthesizing high-quality video with high fidelity from given text descriptions involving motion. However, understanding how Video DiTs convert motion words into video remains insufficient. Furthermore, while prior studies on interpretable saliency maps primarily target objects, motion-related behavior in Video DiTs remains largely unexplored. In this paper, we investigate concrete motion features that specify when and which object moves for a given motion concept. First, to spatially localize, we introduce GramCol, which adaptively produces per-frame saliency maps for any text concept, including both motion and non-motion. Second, we propose a motion-feature selection algorithm to obtain an Interpretable Motion-Attentive Map (IMAP) that localizes motion spatially and temporally. Our method discovers concept saliency maps without the need for any gradient calculation or parameter update. Experimentally, our method shows outstanding localization capability on the motion localization task and zero-shot video semantic segmentation, providing interpretable and clearer saliency maps for both motion and non-motion concepts.

  </details>



- **Track4World: Feedforward World-centric Dense 3D Tracking of All Pixels**  
  Jiahao Lu, Jiayi Xu, Wenbo Hu, Ruijie Zhu, Chengfeng Zhao, Sai-Kit Yeung, Ying Shan, Yuan Liu  
  _2026-03-03_ · https://arxiv.org/abs/2603.02573v1  
  <details><summary>Abstract</summary>

  Estimating the 3D trajectory of every pixel from a monocular video is crucial and promising for a comprehensive understanding of the 3D dynamics of videos. Recent monocular 3D tracking works demonstrate impressive performance, but are limited to either tracking sparse points on the first frame or a slow optimization-based framework for dense tracking. In this paper, we propose a feedforward model, called Track4World, enabling an efficient holistic 3D tracking of every pixel in the world-centric coordinate system. Built on the global 3D scene representation encoded by a VGGT-style ViT, Track4World applies a novel 3D correlation scheme to simultaneously estimate the pixel-wise 2D and 3D dense flow between arbitrary frame pairs. The estimated scene flow, along with the reconstructed 3D geometry, enables subsequent efficient 3D tracking of every pixel of this video. Extensive experiments on multiple benchmarks demonstrate that our approach consistently outperforms existing methods in 2D/3D flow estimation and 3D tracking, highlighting its robustness and scalability for real-world 4D reconstruction tasks.

  </details>



- **Aligning Fetal Anatomy with Kinematic Tree Log-Euclidean PolyRigid Transforms**  
  Yingcheng Liu, Athena Taymourtash, Yang Liu, Esra Abaci Turk, William M. Wells, Leo Joskowicz, P. Ellen Grant, Polina Golland  
  _2026-03-02_ · https://arxiv.org/abs/2603.02371v1  
  <details><summary>Abstract</summary>

  Automated analysis of articulated bodies is crucial in medical imaging. Existing surface-based models often ignore internal volumetric structures and rely on deformation methods that lack anatomical consistency guarantees. To address this problem, we introduce a differentiable volumetric body model based on the Skinned Multi-Person Linear (SMPL) formulation, driven by a new Kinematic Tree-based Log-Euclidean PolyRigid (KTPolyRigid) transform. KTPolyRigid resolves Lie algebra ambiguities associated with large, non-local articulated motions, and encourages smooth, bijective volumetric mappings. Evaluated on 53 fetal MRI volumes, KTPolyRigid yields deformation fields with significantly fewer folding artifacts. Furthermore, our framework enables robust groupwise image registration and a label-efficient, template-based segmentation of fetal organs. It provides a robust foundation for standardized volumetric analysis of articulated bodies in medical imaging.

  </details>



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


