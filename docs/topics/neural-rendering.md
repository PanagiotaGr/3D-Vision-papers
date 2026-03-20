# Neural Rendering & View Synthesis

_Updated: 2026-03-20 07:11 UTC_

Total papers shown: **6**


---

- **GSMem: 3D Gaussian Splatting as Persistent Spatial Memory for Zero-Shot Embodied Exploration and Reasoning**  
  Yiren Lu, Yi Du, Disheng Liu, Yunlai Zhou, Chen Wang, Yu Yin  
  _2026-03-19_ · https://arxiv.org/abs/2603.19137v1  
  <details><summary>Abstract</summary>

  Effective embodied exploration requires agents to accumulate and retain spatial knowledge over time. However, existing scene representations, such as discrete scene graphs or static view-based snapshots, lack \textit{post-hoc re-observability}. If an initial observation misses a target, the resulting memory omission is often irrecoverable. To bridge this gap, we propose \textbf{GSMem}, a zero-shot embodied exploration and reasoning framework built upon 3D Gaussian Splatting (3DGS). By explicitly parameterizing continuous geometry and dense appearance, 3DGS serves as a persistent spatial memory that endows the agent with \textit{Spatial Recollection}: the ability to render photorealistic novel views from optimal, previously unoccupied viewpoints. To operationalize this, GSMem employs a retrieval mechanism that simultaneously leverages parallel object-level scene graphs and semantic-level language fields. This complementary design robustly localizes target regions, enabling the agent to ``hallucinate'' optimal views for high-fidelity Vision-Language Model (VLM) reasoning. Furthermore, we introduce a hybrid exploration strategy that combines VLM-driven semantic scoring with a 3DGS-based coverage objective, balancing task-aware exploration with geometric coverage. Extensive experiments on embodied question answering and lifelong navigation demonstrate the robustness and effectiveness of our framework

  </details>



- **3DreamBooth: High-Fidelity 3D Subject-Driven Video Generation Model**  
  Hyun-kyu Ko, Jihyeon Park, Younghyun Kim, Dongheok Park, Eunbyung Park  
  _2026-03-19_ · https://arxiv.org/abs/2603.18524v1  
  <details><summary>Abstract</summary>

  Creating dynamic, view-consistent videos of customized subjects is highly sought after for a wide range of emerging applications, including immersive VR/AR, virtual production, and next-generation e-commerce. However, despite rapid progress in subject-driven video generation, existing methods predominantly treat subjects as 2D entities, focusing on transferring identity through single-view visual features or textual prompts. Because real-world subjects are inherently 3D, applying these 2D-centric approaches to 3D object customization reveals a fundamental limitation: they lack the comprehensive spatial priors necessary to reconstruct the 3D geometry. Consequently, when synthesizing novel views, they must rely on generating plausible but arbitrary details for unseen regions, rather than preserving the true 3D identity. Achieving genuine 3D-aware customization remains challenging due to the scarcity of multi-view video datasets. While one might attempt to fine-tune models on limited video sequences, this often leads to temporal overfitting. To resolve these issues, we introduce a novel framework for 3D-aware video customization, comprising 3DreamBooth and 3Dapter. 3DreamBooth decouples spatial geometry from temporal motion through a 1-frame optimization paradigm. By restricting updates to spatial representations, it effectively bakes a robust 3D prior into the model without the need for exhaustive video-based training. To enhance fine-grained textures and accelerate convergence, we incorporate 3Dapter, a visual conditioning module. Following single-view pre-training, 3Dapter undergoes multi-view joint optimization with the main generation branch via an asymmetrical conditioning strategy. This design allows the module to act as a dynamic selective router, querying view-specific geometric hints from a minimal reference set. Project page: https://ko-lani.github.io/3DreamBooth/

  </details>



- **Semantic Segmentation and Depth Estimation for Real-Time Lunar Surface Mapping Using 3D Gaussian Splatting**  
  Guillem Casadesus Vila, Adam Dai, Grace Gao  
  _2026-03-18_ · https://arxiv.org/abs/2603.18218v1  
  <details><summary>Abstract</summary>

  Navigation and mapping on the lunar surface require robust perception under challenging conditions, including poorly textured environments, high-contrast lighting, and limited computational resources. This paper presents a real-time mapping framework that integrates dense perception models with a 3D Gaussian Splatting (3DGS) representation. We first benchmark several models on synthetic datasets generated with the LuPNT simulator, selecting a stereo dense depth estimation model based on Gated Recurrent Units for its balance of speed and accuracy in depth estimation, and a convolutional neural network for its superior performance in detecting semantic segments. Using ground truth poses to decouple the local scene understanding from the global state estimation, our pipeline reconstructs a 120-meter traverse with a geometric height accuracy of approximately 3 cm, outperforming a traditional point cloud baseline without LiDAR. The resulting 3DGS map enables novel view synthesis and serves as a foundation for a full SLAM system, where its capacity for joint map and pose optimization would offer significant advantages. Our results demonstrate that combining semantic segmentation and dense depth estimation with learned map representations is an effective approach for creating detailed, large-scale maps to support future lunar surface missions.

  </details>



- **Universal Skeleton Understanding via Differentiable Rendering and MLLMs**  
  Ziyi Wang, Peiming Li, Xinshun Wang, Yang Tang, Kai-Kuang Ma, Mengyuan Liu  
  _2026-03-18_ · https://arxiv.org/abs/2603.18003v1  
  <details><summary>Abstract</summary>

  Multimodal large language models (MLLMs) exhibit strong visual-language reasoning, yet remain confined to their native modalities and cannot directly process structured, non-visual data such as human skeletons. Existing methods either compress skeleton dynamics into lossy feature vectors for text alignment, or quantize motion into discrete tokens that generalize poorly across heterogeneous skeleton formats. We present SkeletonLLM, which achieves universal skeleton understanding by translating arbitrary skeleton sequences into the MLLM's native visual modality. At its core is DrAction, a differentiable, format-agnostic renderer that converts skeletal kinematics into compact image sequences. Because the pipeline is end-to-end differentiable, MLLM gradients can directly guide the rendering to produce task-informative visual tokens. To further enhance reasoning capabilities, we introduce a cooperative training strategy: Causal Reasoning Distillation transfers structured, step-by-step reasoning from a teacher model, while Discriminative Finetuning sharpens decision boundaries between confusable actions. SkeletonLLM demonstrates strong generalization on diverse tasks including recognition, captioning, reasoning, and cross-format transfer -- suggesting a viable path for applying MLLMs to non-native modalities. Code will be released upon acceptance.

  </details>



- **TAPESTRY: From Geometry to Appearance via Consistent Turntable Videos**  
  Yan Zeng, Haoran Jiang, Kaixin Yao, Qixuan Zhang, Longwen Zhang, Lan Xu, Jingyi Yu  
  _2026-03-18_ · https://arxiv.org/abs/2603.17735v1  
  <details><summary>Abstract</summary>

  Automatically generating photorealistic and self-consistent appearances for untextured 3D models is a critical challenge in digital content creation. The advancement of large-scale video generation models offers a natural approach: directly synthesizing 360-degree turntable videos (TTVs), which can serve not only as high-quality dynamic previews but also as an intermediate representation to drive texture synthesis and neural rendering. However, existing general-purpose video diffusion models struggle to maintain strict geometric consistency and appearance stability across the full range of views, making their outputs ill-suited for high-quality 3D reconstruction. To this end, we introduce TAPESTRY, a framework for generating high-fidelity TTVs conditioned on explicit 3D geometry. We reframe the 3D appearance generation task as a geometry-conditioned video diffusion problem: given a 3D mesh, we first render and encode multi-modal geometric features to constrain the video generation process with pixel-level precision, thereby enabling the creation of high-quality and consistent TTVs. Building upon this, we also design a method for downstream reconstruction tasks from the TTV input, featuring a multi-stage pipeline with 3D-Aware Inpainting. By rotating the model and performing a context-aware secondary generation, this pipeline effectively completes self-occluded regions to achieve full surface coverage. The videos generated by TAPESTRY are not only high-quality dynamic previews but also serve as a reliable, 3D-aware intermediate representation that can be seamlessly back-projected into UV textures or used to supervise neural rendering methods like 3DGS. This enables the automated creation of production-ready, complete 3D assets from untextured meshes. Experimental results demonstrate that our method outperforms existing approaches in both video consistency and final reconstruction quality.

  </details>



- **VisionNVS: Self-Supervised Inpainting for Novel View Synthesis under the Virtual-Shift Paradigm**  
  Hongbo Lu, Liang Yao, Chenghao He, Fan Liu, Wenlong Liao, Tao He, Pai Peng  
  _2026-03-18_ · https://arxiv.org/abs/2603.17382v1  
  <details><summary>Abstract</summary>

  A fundamental bottleneck in Novel View Synthesis (NVS) for autonomous driving is the inherent supervision gap on novel trajectories: models are tasked with synthesizing unseen views during inference, yet lack ground truth images for these shifted poses during training. In this paper, we propose VisionNVS, a camera-only framework that fundamentally reformulates view synthesis from an ill-posed extrapolation problem into a self-supervised inpainting task. By introducing a ``Virtual-Shift'' strategy, we use monocular depth proxies to simulate occlusion patterns and map them onto the original view. This paradigm shift allows the use of raw, recorded images as pixel-perfect supervision, effectively eliminating the domain gap inherent in previous approaches. Furthermore, we address spatial consistency through a Pseudo-3D Seam Synthesis strategy, which integrates visual data from adjacent cameras during training to explicitly model real-world photometric discrepancies and calibration errors. Experiments demonstrate that VisionNVS achieves superior geometric fidelity and visual quality compared to LiDAR-dependent baselines, offering a robust solution for scalable driving simulation.

  </details>


