# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-13 07:13 UTC_

Total papers shown: **9**


---

- **WorldTree: Towards 4D Dynamic Worlds from Monocular Video using Tree-Chains**  
  Qisen Wang, Yifan Zhao, Jia Li  
  _2026-02-12_ · https://arxiv.org/abs/2602.11845v1  
  <details><summary>Abstract</summary>

  Dynamic reconstruction has achieved remarkable progress, but there remain challenges in monocular input for more practical applications. The prevailing works attempt to construct efficient motion representations, but lack a unified spatiotemporal decomposition framework, suffering from either holistic temporal optimization or coupled hierarchical spatial composition. To this end, we propose WorldTree, a unified framework comprising Temporal Partition Tree (TPT) that enables coarse-to-fine optimization based on the inheritance-based partition tree structure for hierarchical temporal decomposition, and Spatial Ancestral Chains (SAC) that recursively query ancestral hierarchical structure to provide complementary spatial dynamics while specializing motion representations across ancestral nodes. Experimental results on different datasets indicate that our proposed method achieves 8.26% improvement of LPIPS on NVIDIA-LS and 9.09% improvement of mLPIPS on DyCheck compared to the second-best method. Code: https://github.com/iCVTEAM/WorldTree.

  </details>



- **TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction**  
  Yuxiang Zhong, Jun Wei, Chaoqi Chen, Senyou An, Hui Huang  
  _2026-02-12_ · https://arxiv.org/abs/2602.11705v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has revolutionized 3D scene representation with superior efficiency and quality. While recent adaptations for computed tomography (CT) show promise, they struggle with severe artifacts under highly sparse-view projections and dynamic motions. To address these challenges, we propose Tomographic Geometry Field (TG-Field), a geometry-aware Gaussian deformation framework tailored for both static and dynamic CT reconstruction. A multi-resolution hash encoder is employed to capture local spatial priors, regularizing primitive parameters under ultra-sparse settings. We further extend the framework to dynamic reconstruction by introducing time-conditioned representations and a spatiotemporal attention block to adaptively aggregate features, thereby resolving spatiotemporal ambiguities and enforcing temporal coherence. In addition, a motion-flow network models fine-grained respiratory motion to track local anatomical deformations. Extensive experiments on synthetic and real-world datasets demonstrate that TG-Field consistently outperforms existing methods, achieving state-of-the-art reconstruction accuracy under highly sparse-view conditions.

  </details>



- **EmoSpace: Fine-Grained Emotion Prototype Learning for Immersive Affective Content Generation**  
  Bingyuan Wang, Xingbei Chen, Zongyang Qiu, Linping Yuan, Zeyu Wang  
  _2026-02-12_ · https://arxiv.org/abs/2602.11658v1  
  <details><summary>Abstract</summary>

  Emotion is important for creating compelling virtual reality (VR) content. Although some generative methods have been applied to lower the barrier to creating emotionally rich content, they fail to capture the nuanced emotional semantics and the fine-grained control essential for immersive experiences. To address these limitations, we introduce EmoSpace, a novel framework for emotion-aware content generation that learns dynamic, interpretable emotion prototypes through vision-language alignment. We employ a hierarchical emotion representation with rich learnable prototypes that evolve during training, enabling fine-grained emotional control without requiring explicit emotion labels. We develop a controllable generation pipeline featuring multi-prototype guidance, temporal blending, and attention reweighting that supports diverse applications, including emotional image outpainting, stylized generation, and emotional panorama generation for VR environments. Our experiments demonstrate the superior performance of EmoSpace over existing methods in both qualitative and quantitative evaluations. Additionally, we present a comprehensive user study investigating how VR environments affect emotional perception compared to desktop settings. Our work facilitates immersive visual content generation with fine-grained emotion control and supports applications like therapy, education, storytelling, artistic creation, and cultural preservation. Code and models will be made publicly available.

  </details>



- **From Steering to Pedalling: Do Autonomous Driving VLMs Generalize to Cyclist-Assistive Spatial Perception and Planning?**  
  Krishna Kanth Nakka, Vedasri Nakka  
  _2026-02-11_ · https://arxiv.org/abs/2602.10771v1  
  <details><summary>Abstract</summary>

  Cyclists often encounter safety-critical situations in urban traffic, highlighting the need for assistive systems that support safe and informed decision-making. Recently, vision-language models (VLMs) have demonstrated strong performance on autonomous driving benchmarks, suggesting their potential for general traffic understanding and navigation-related reasoning. However, existing evaluations are predominantly vehicle-centric and fail to assess perception and reasoning from a cyclist-centric viewpoint. To address this gap, we introduce CyclingVQA, a diagnostic benchmark designed to probe perception, spatio-temporal understanding, and traffic-rule-to-lane reasoning from a cyclist's perspective. Evaluating 31+ recent VLMs spanning general-purpose, spatially enhanced, and autonomous-driving-specialized models, we find that current models demonstrate encouraging capabilities, while also revealing clear areas for improvement in cyclist-centric perception and reasoning, particularly in interpreting cyclist-specific traffic cues and associating signs with the correct navigational lanes. Notably, several driving-specialized models underperform strong generalist VLMs, indicating limited transfer from vehicle-centric training to cyclist-assistive scenarios. Finally, through systematic error analysis, we identify recurring failure modes to guide the development of more effective cyclist-assistive intelligent systems.

  </details>



- **C^2ROPE: Causal Continuous Rotary Positional Encoding for 3D Large Multimodal-Models Reasoning**  
  Guanting Ye, Qiyan Zhao, Wenhao Yu, Xiaofeng Zhang, Jianmin Ji, Yanyong Zhang, Ka-Veng Yuen  
  _2026-02-11_ · https://arxiv.org/abs/2602.10551v1  
  <details><summary>Abstract</summary>

  Recent advances in 3D Large Multimodal Models (LMMs) built on Large Language Models (LLMs) have established the alignment of 3D visual features with LLM representations as the dominant paradigm. However, the inherited Rotary Position Embedding (RoPE) introduces limitations for multimodal processing. Specifically, applying 1D temporal positional indices disrupts the continuity of visual features along the column dimension, resulting in spatial locality loss. Moreover, RoPE follows the prior that temporally closer image tokens are more causally related, leading to long-term decay in attention allocation and causing the model to progressively neglect earlier visual tokens as the sequence length increases. To address these issues, we propose C^2RoPE, an improved RoPE that explicitly models local spatial Continuity and spatial Causal relationships for visual processing. C^2RoPE introduces a spatio-temporal continuous positional embedding mechanism for visual tokens. It first integrates 1D temporal positions with Cartesian-based spatial coordinates to construct a triplet hybrid positional index, and then employs a frequency allocation strategy to encode spatio-temporal positional information across the three index components. Additionally, we introduce Chebyshev Causal Masking, which determines causal dependencies by computing the Chebyshev distance of image tokens in 2D space. Evaluation results across various benchmarks, including 3D scene reasoning and 3D visual question answering, demonstrate C^2RoPE's effectiveness. The code is be available at https://github.com/ErikZ719/C2RoPE.

  </details>



- **ENIGMA: EEG-to-Image in 15 Minutes Using Less Than 1% of the Parameters**  
  Reese Kneeland, Wangshu Jiang, Ugo Bruzadin Nunes, Paul Steven Scotti, Arnaud Delorme, Jonathan Xu  
  _2026-02-10_ · https://arxiv.org/abs/2602.10361v1  
  <details><summary>Abstract</summary>

  To be practical for real-life applications, models for brain-computer interfaces must be easily and quickly deployable on new subjects, effective on affordable scanning hardware, and small enough to run locally on accessible computing resources. To directly address these current limitations, we introduce ENIGMA, a multi-subject electroencephalography (EEG)-to-Image decoding model that reconstructs seen images from EEG recordings and achieves state-of-the-art (SOTA) performance on the research-grade THINGS-EEG2 and consumer-grade AllJoined-1.6M benchmarks, while fine-tuning effectively on new subjects with as little as 15 minutes of data. ENIGMA boasts a simpler architecture and requires less than 1% of the trainable parameters necessary for previous approaches. Our approach integrates a subject-unified spatio-temporal backbone along with a set of multi-subject latent alignment layers and an MLP projector to map raw EEG signals to a rich visual latent space. We evaluate our approach using a broad suite of image reconstruction metrics that have been standardized in the adjacent field of fMRI-to-Image research, and we describe the first EEG-to-Image study to conduct extensive behavioral evaluations of our reconstructions using human raters. Our simple and robust architecture provides a significant performance boost across both research-grade and consumer-grade EEG hardware, and a substantial improvement in fine-tuning efficiency and inference cost. Finally, we provide extensive ablations to determine the architectural choices most responsible for our performance gains in both single and multi-subject cases across multiple benchmark datasets. Collectively, our work provides a substantial step towards the development of practical brain-computer interface applications.

  </details>



- **4RC: 4D Reconstruction via Conditional Querying Anytime and Anywhere**  
  Yihang Luo, Shangchen Zhou, Yushi Lan, Xingang Pan, Chen Change Loy  
  _2026-02-10_ · https://arxiv.org/abs/2602.10094v1  
  <details><summary>Abstract</summary>

  We present 4RC, a unified feed-forward framework for 4D reconstruction from monocular videos. Unlike existing approaches that typically decouple motion from geometry or produce limited 4D attributes such as sparse trajectories or two-view scene flow, 4RC learns a holistic 4D representation that jointly captures dense scene geometry and motion dynamics. At its core, 4RC introduces a novel encode-once, query-anywhere and anytime paradigm: a transformer backbone encodes the entire video into a compact spatio-temporal latent space, from which a conditional decoder can efficiently query 3D geometry and motion for any query frame at any target timestamp. To facilitate learning, we represent per-view 4D attributes in a minimally factorized form by decomposing them into base geometry and time-dependent relative motion. Extensive experiments demonstrate that 4RC outperforms prior and concurrent methods across a wide range of 4D reconstruction tasks.

  </details>



- **Spatio-Temporal Attention for Consistent Video Semantic Segmentation in Automated Driving**  
  Serin Varghese, Kevin Ross, Fabian Hueger, Kira Maag  
  _2026-02-10_ · https://arxiv.org/abs/2602.10052v1  
  <details><summary>Abstract</summary>

  Deep neural networks, especially transformer-based architectures, have achieved remarkable success in semantic segmentation for environmental perception. However, existing models process video frames independently, thus failing to leverage temporal consistency, which could significantly improve both accuracy and stability in dynamic scenes. In this work, we propose a Spatio-Temporal Attention (STA) mechanism that extends transformer attention blocks to incorporate multi-frame context, enabling robust temporal feature representations for video semantic segmentation. Our approach modifies standard self-attention to process spatio-temporal feature sequences while maintaining computational efficiency and requiring minimal changes to existing architectures. STA demonstrates broad applicability across diverse transformer architectures and remains effective across both lightweight and larger-scale models. A comprehensive evaluation on the Cityscapes and BDD100k datasets shows substantial improvements of 9.20 percentage points in temporal consistency metrics and up to 1.76 percentage points in mean intersection over union compared to single-frame baselines. These results demonstrate STA as an effective architectural enhancement for video-based semantic segmentation applications.

  </details>



- **Time2General: Learning Spatiotemporal Invariant Representations for Domain-Generalization Video Semantic Segmentation**  
  Siyu Chen, Ting Han, Haoling Huang, Chaolei Wang, Chengzheng Fu, Duxin Zhu, Guorong Cai, Jinhe Su  
  _2026-02-10_ · https://arxiv.org/abs/2602.09648v1  
  <details><summary>Abstract</summary>

  Domain Generalized Video Semantic Segmentation (DGVSS) is trained on a single labeled driving domain and is directly deployed on unseen domains without target labels and test-time adaptation while maintaining temporally consistent predictions over video streams. In practice, both domain shift and temporal-sampling shift break correspondence-based propagation and fixed-stride temporal aggregation, causing severe frame-to-frame flicker even in label-stable regions. We propose Time2General, a DGVSS framework built on Stability Queries. Time2General introduces a Spatio-Temporal Memory Decoder that aggregates multi-frame context into a clip-level spatio-temporal memory and decodes temporally consistent per-frame masks without explicit correspondence propagation. To further suppress flicker and improve robustness to varying sampling rates, the Masked Temporal Consistency Loss is proposed to regularize temporal prediction discrepancies across different strides, and randomize training strides to expose the model to diverse temporal gaps. Extensive experiments on multiple driving benchmarks show that Time2General achieves a substantial improvement in cross-domain accuracy and temporal stability over prior DGSS and VSS baselines while running at up to 18 FPS. Code will be released after the review process.

  </details>


