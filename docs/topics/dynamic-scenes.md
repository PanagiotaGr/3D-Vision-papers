# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-01-26 06:54 UTC_

Total papers shown: **2**


---

- **AnyView: Synthesizing Any Novel View in Dynamic Scenes**  
  Basile Van Hoorick, Dian Chen, Shun Iwase, Pavel Tokmakov, Muhammad Zubair Irshad, Igor Vasiljevic, Swati Gupta, Fangzhou Cheng, Sergey Zakharov, Vitor Campagnolo Guizilini  
  _2026-01-23_ · https://arxiv.org/abs/2601.16982v1  
  <details><summary>Abstract</summary>

  Modern generative video models excel at producing convincing, high-quality outputs, but struggle to maintain multi-view and spatiotemporal consistency in highly dynamic real-world environments. In this work, we introduce \textbf{AnyView}, a diffusion-based video generation framework for \emph{dynamic view synthesis} with minimal inductive biases or geometric assumptions. We leverage multiple data sources with various levels of supervision, including monocular (2D), multi-view static (3D) and multi-view dynamic (4D) datasets, to train a generalist spatiotemporal implicit representation capable of producing zero-shot novel videos from arbitrary camera locations and trajectories. We evaluate AnyView on standard benchmarks, showing competitive results with the current state of the art, and propose \textbf{AnyViewBench}, a challenging new benchmark tailored towards \emph{extreme} dynamic view synthesis in diverse real-world scenarios. In this more dramatic setting, we find that most baselines drastically degrade in performance, as they require significant overlap between viewpoints, while AnyView maintains the ability to produce realistic, plausible, and spatiotemporally consistent videos when prompted from \emph{any} viewpoint. Results, data, code, and models can be viewed at: https://tri-ml.github.io/AnyView/

  </details>



- **HA2F: Dual-module Collaboration-Guided Hierarchical Adaptive Aggregation Framework for Remote Sensing Change Detection**  
  Shuying Li, Yuchen Wang, San Zhang, Chuang Yang  
  _2026-01-23_ · https://arxiv.org/abs/2601.16573v1  
  <details><summary>Abstract</summary>

  Remote sensing change detection (RSCD) aims to identify the spatio-temporal changes of land cover, providing critical support for multi-disciplinary applications (e.g., environmental monitoring, disaster assessment, and climate change studies). Existing methods focus either on extracting features from localized patches, or pursue processing entire images holistically, which leads to the cross temporal feature matching deviation and exhibiting sensitivity to radiometric and geometric noise. Following the above issues, we propose a dual-module collaboration guided hierarchical adaptive aggregation framework, namely HA2F, which consists of dynamic hierarchical feature calibration module (DHFCM) and noise-adaptive feature refinement module (NAFRM). The former dynamically fuses adjacent-level features through perceptual feature selection, suppressing irrelevant discrepancies to address multi-temporal feature alignment deviations. The NAFRM utilizes the dual feature selection mechanism to highlight the change sensitive regions and generate spatial masks, suppressing the interference of irrelevant regions or shadows. Extensive experiments verify the effectiveness of the proposed HA2F, which achieves state-of-the-art performance on LEVIR-CD, WHU-CD, and SYSU-CD datasets, surpassing existing comparative methods in terms of both precision metrics and computational efficiency. In addition, ablation experiments show that DHFCM and NAFRM are effective. \href{https://huggingface.co/InPeerReview/RemoteSensingChangeDetection-RSCD.HA2F}{HA2F Official Code is Available Here!}

  </details>


