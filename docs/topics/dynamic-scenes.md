# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-14 07:02 UTC_

Total papers shown: **4**


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


