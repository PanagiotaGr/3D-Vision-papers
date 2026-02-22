# SLAM & Localization

_Updated: 2026-02-22 07:04 UTC_

Total papers shown: **2**


---

- **GraphThinker: Reinforcing Video Reasoning with Event Graph Thinking**  
  Zixu Cheng, Da Li, Jian Hu, Ziquan Liu, Wei Li, Shaogang Gong  
  _2026-02-19_ · https://arxiv.org/abs/2602.17555v1  
  <details><summary>Abstract</summary>

  Video reasoning requires understanding the causal relationships between events in a video. However, such relationships are often implicit and costly to annotate manually. While existing multimodal large language models (MLLMs) often infer event relations through dense captions or video summaries for video reasoning, such modeling still lacks causal understanding. Without explicit causal structure modeling within and across video events, these models suffer from hallucinations during the video reasoning. In this work, we propose GraphThinker, a reinforcement finetuning-based method that constructs structural event-level scene graphs and enhances visual grounding to jointly reduce hallucinations in video reasoning. Specifically, we first employ an MLLM to construct an event-based video scene graph (EVSG) that explicitly models both intra- and inter-event relations, and incorporate these formed scene graphs into the MLLM as an intermediate thinking process. We also introduce a visual attention reward during reinforcement finetuning, which strengthens video grounding and further mitigates hallucinations. We evaluate GraphThinker on two datasets, RexTime and VidHalluc, where it shows superior ability to capture object and event relations with more precise event localization, reducing hallucinations in video reasoning compared to prior methods.

  </details>



- **NRGS-SLAM: Monocular Non-Rigid SLAM for Endoscopy via Deformation-Aware 3D Gaussian Splatting**  
  Jiwei Shan, Zeyu Cai, Yirui Li, Yongbo Chen, Lijun Han, Yun-hui Liu, Hesheng Wang, Shing Shin Cheng  
  _2026-02-19_ · https://arxiv.org/abs/2602.17182v1  
  <details><summary>Abstract</summary>

  Visual simultaneous localization and mapping (V-SLAM) is a fundamental capability for autonomous perception and navigation. However, endoscopic scenes violate the rigidity assumption due to persistent soft-tissue deformations, creating a strong coupling ambiguity between camera ego-motion and intrinsic deformation. Although recent monocular non-rigid SLAM methods have made notable progress, they often lack effective decoupling mechanisms and rely on sparse or low-fidelity scene representations, which leads to tracking drift and limited reconstruction quality. To address these limitations, we propose NRGS-SLAM, a monocular non-rigid SLAM system for endoscopy based on 3D Gaussian Splatting. To resolve the coupling ambiguity, we introduce a deformation-aware 3D Gaussian map that augments each Gaussian primitive with a learnable deformation probability, optimized via a Bayesian self-supervision strategy without requiring external non-rigidity labels. Building on this representation, we design a deformable tracking module that performs robust coarse-to-fine pose estimation by prioritizing low-deformation regions, followed by efficient per-frame deformation updates. A carefully designed deformable mapping module progressively expands and refines the map, balancing representational capacity and computational efficiency. In addition, a unified robust geometric loss incorporates external geometric priors to mitigate the inherent ill-posedness of monocular non-rigid SLAM. Extensive experiments on multiple public endoscopic datasets demonstrate that NRGS-SLAM achieves more accurate camera pose estimation (up to 50\% reduction in RMSE) and higher-quality photo-realistic reconstructions than state-of-the-art methods. Comprehensive ablation studies further validate the effectiveness of our key design choices. Source code will be publicly available upon paper acceptance.

  </details>


