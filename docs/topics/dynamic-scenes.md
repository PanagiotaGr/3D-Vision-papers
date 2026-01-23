# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-01-23 06:52 UTC_

Total papers shown: **9**


---

- **ActionMesh: Animated 3D Mesh Generation with Temporal 3D Diffusion**  
  Remy Sabathier, David Novotny, Niloy J. Mitra, Tom Monnier  
  _2026-01-22_ · https://arxiv.org/abs/2601.16148v1  
  <details><summary>Abstract</summary>

  Generating animated 3D objects is at the heart of many applications, yet most advanced works are typically difficult to apply in practice because of their limited setup, their long runtime, or their limited quality. We introduce ActionMesh, a generative model that predicts production-ready 3D meshes "in action" in a feed-forward manner. Drawing inspiration from early video models, our key insight is to modify existing 3D diffusion models to include a temporal axis, resulting in a framework we dubbed "temporal 3D diffusion". Specifically, we first adapt the 3D diffusion stage to generate a sequence of synchronized latents representing time-varying and independent 3D shapes. Second, we design a temporal 3D autoencoder that translates a sequence of independent shapes into the corresponding deformations of a pre-defined reference shape, allowing us to build an animation. Combining these two components, ActionMesh generates animated 3D meshes from different inputs like a monocular video, a text description, or even a 3D mesh with a text prompt describing its animation. Besides, compared to previous approaches, our method is fast and produces results that are rig-free and topology consistent, hence enabling rapid iteration and seamless applications like texturing and retargeting. We evaluate our model on standard video-to-4D benchmarks (Consistent4D, Objaverse) and report state-of-the-art performances on both geometric accuracy and temporal consistency, demonstrating that our model can deliver animated 3D meshes with unprecedented speed and quality.

  </details>



- **EVolSplat4D: Efficient Volume-based Gaussian Splatting for 4D Urban Scene Synthesis**  
  Sheng Miao, Sijin Li, Pan Wang, Dongfeng Bai, Bingbing Liu, Yue Wang, Andreas Geiger, Yiyi Liao  
  _2026-01-22_ · https://arxiv.org/abs/2601.15951v1  
  <details><summary>Abstract</summary>

  Novel view synthesis (NVS) of static and dynamic urban scenes is essential for autonomous driving simulation, yet existing methods often struggle to balance reconstruction time with quality. While state-of-the-art neural radiance fields and 3D Gaussian Splatting approaches achieve photorealism, they often rely on time-consuming per-scene optimization. Conversely, emerging feed-forward methods frequently adopt per-pixel Gaussian representations, which lead to 3D inconsistencies when aggregating multi-view predictions in complex, dynamic environments. We propose EvolSplat4D, a feed-forward framework that moves beyond existing per-pixel paradigms by unifying volume-based and pixel-based Gaussian prediction across three specialized branches. For close-range static regions, we predict consistent geometry of 3D Gaussians over multiple frames directly from a 3D feature volume, complemented by a semantically-enhanced image-based rendering module for predicting their appearance. For dynamic actors, we utilize object-centric canonical spaces and a motion-adjusted rendering module to aggregate temporal features, ensuring stable 4D reconstruction despite noisy motion priors. Far-Field scenery is handled by an efficient per-pixel Gaussian branch to ensure full-scene coverage. Experimental results on the KITTI-360, KITTI, Waymo, and PandaSet datasets show that EvolSplat4D reconstructs both static and dynamic environments with superior accuracy and consistency, outperforming both per-scene optimization and state-of-the-art feed-forward baselines.

  </details>



- **UBATrack: Spatio-Temporal State Space Model for General Multi-Modal Tracking**  
  Qihua Liang, Liang Chen, Yaozong Zheng, Jian Nong, Zhiyi Mo, Bineng Zhong  
  _2026-01-21_ · https://arxiv.org/abs/2601.14799v1  
  <details><summary>Abstract</summary>

  Multi-modal object tracking has attracted considerable attention by integrating multiple complementary inputs (e.g., thermal, depth, and event data) to achieve outstanding performance. Although current general-purpose multi-modal trackers primarily unify various modal tracking tasks (i.e., RGB-Thermal infrared, RGB-Depth or RGB-Event tracking) through prompt learning, they still overlook the effective capture of spatio-temporal cues. In this work, we introduce a novel multi-modal tracking framework based on a mamba-style state space model, termed UBATrack. Our UBATrack comprises two simple yet effective modules: a Spatio-temporal Mamba Adapter (STMA) and a Dynamic Multi-modal Feature Mixer. The former leverages Mamba's long-sequence modeling capability to jointly model cross-modal dependencies and spatio-temporal visual cues in an adapter-tuning manner. The latter further enhances multi-modal representation capacity across multiple feature dimensions to improve tracking robustness. In this way, UBATrack eliminates the need for costly full-parameter fine-tuning, thereby improving the training efficiency of multi-modal tracking algorithms. Experiments show that UBATrack outperforms state-of-the-art methods on RGB-T, RGB-D, and RGB-E tracking benchmarks, achieving outstanding results on the LasHeR, RGBT234, RGBT210, DepthTrack, VOT-RGBD22, and VisEvent datasets.

  </details>



- **FeedbackSTS-Det: Sparse Frames-Based Spatio-Temporal Semantic Feedback Network for Infrared Small Target Detection**  
  Yian Huang, Qing Qin, Aji Mao, Xiangyu Qiu, Liang Xu, Xian Zhang, Zhenming Peng  
  _2026-01-21_ · https://arxiv.org/abs/2601.14690v1  
  <details><summary>Abstract</summary>

  Infrared small target detection (ISTD) under complex backgrounds remains a critical yet challenging task, primarily due to the extremely low signal-to-clutter ratio, persistent dynamic interference, and the lack of distinct target features. While multi-frame detection methods leverages temporal cues to improve upon single-frame approaches, existing methods still struggle with inefficient long-range dependency modeling and insufficient robustness. To overcome these issues, we propose a novel scheme for ISTD, realized through a sparse frames-based spatio-temporal semantic feedback network named FeedbackSTS-Det. The core of our approach is a novel spatio-temporal semantic feedback strategy with a closed-loop semantic association mechanism, which consists of paired forward and backward refinement modules that work cooperatively across the encoder and decoder. Moreover, both modules incorporate an embedded sparse semantic module (SSM), which performs structured sparse temporal modeling to capture long-range dependencies with low computational cost. This integrated design facilitates robust implicit inter-frame registration and continuous semantic refinement, effectively suppressing false alarms. Furthermore, our overall procedure maintains a consistent training-inference pipeline, which ensures reliable performance transfer and increases model robustness. Extensive experiments on multiple benchmark datasets confirm the effectiveness of FeedbackSTS-Det. Code and models are available at: https://github.com/IDIP-Lab/FeedbackSTS-Det.

  </details>



- **LaVR: Scene Latent Conditioned Generative Video Trajectory Re-Rendering using Large 4D Reconstruction Models**  
  Mingyang Xie, Numair Khan, Tianfu Wang, Naina Dhingra, Seonghyeon Nam, Haitao Yang, Zhuo Hui, Christopher Metzler, Andrea Vedaldi, Hamed Pirsiavash, et al.  
  _2026-01-21_ · https://arxiv.org/abs/2601.14674v1  
  <details><summary>Abstract</summary>

  Given a monocular video, the goal of video re-rendering is to generate views of the scene from a novel camera trajectory. Existing methods face two distinct challenges. Geometrically unconditioned models lack spatial awareness, leading to drift and deformation under viewpoint changes. On the other hand, geometrically-conditioned models depend on estimated depth and explicit reconstruction, making them susceptible to depth inaccuracies and calibration errors. We propose to address these challenges by using the implicit geometric knowledge embedded in the latent space of a large 4D reconstruction model to condition the video generation process. These latents capture scene structure in a continuous space without explicit reconstruction. Therefore, they provide a flexible representation that allows the pretrained diffusion prior to regularize errors more effectively. By jointly conditioning on these latents and source camera poses, we demonstrate that our model achieves state-of-the-art results on the video re-rendering task. Project webpage is https://lavr-4d-scene-rerender.github.io/

  </details>



- **OmniTransfer: All-in-one Framework for Spatio-temporal Video Transfer**  
  Pengze Zhang, Yanze Wu, Mengtian Li, Xu Bai, Songtao Zhao, Fulong Ye, Chong Mou, Xinghui Li, Zhuowei Chen, Qian He, et al.  
  _2026-01-20_ · https://arxiv.org/abs/2601.14250v1  
  <details><summary>Abstract</summary>

  Videos convey richer information than images or text, capturing both spatial and temporal dynamics. However, most existing video customization methods rely on reference images or task-specific temporal priors, failing to fully exploit the rich spatio-temporal information inherent in videos, thereby limiting flexibility and generalization in video generation. To address these limitations, we propose OmniTransfer, a unified framework for spatio-temporal video transfer. It leverages multi-view information across frames to enhance appearance consistency and exploits temporal cues to enable fine-grained temporal control. To unify various video transfer tasks, OmniTransfer incorporates three key designs: Task-aware Positional Bias that adaptively leverages reference video information to improve temporal alignment or appearance consistency; Reference-decoupled Causal Learning separating reference and target branches to enable precise reference transfer while improving efficiency; and Task-adaptive Multimodal Alignment using multimodal semantic guidance to dynamically distinguish and tackle different tasks. Extensive experiments show that OmniTransfer outperforms existing methods in appearance (ID and style) and temporal transfer (camera movement and video effects), while matching pose-guided methods in motion transfer without using pose, establishing a new paradigm for flexible, high-fidelity video generation.

  </details>



- **Two-Stream temporal transformer for video action classification**  
  Nattapong Kurpukdee, Adrian G. Bors  
  _2026-01-20_ · https://arxiv.org/abs/2601.14086v1  
  <details><summary>Abstract</summary>

  Motion representation plays an important role in video understanding and has many applications including action recognition, robot and autonomous guidance or others. Lately, transformer networks, through their self-attention mechanism capabilities, have proved their efficiency in many applications. In this study, we introduce a new two-stream transformer video classifier, which extracts spatio-temporal information from content and optical flow representing movement information. The proposed model identifies self-attention features across the joint optical flow and temporal frame domain and represents their relationships within the transformer encoder mechanism. The experimental results show that our proposed methodology provides excellent classification results on three well-known video datasets of human activities.

  </details>



- **STEC: A Reference-Free Spatio-Temporal Entropy Coverage Metric for Evaluating Sampled Video Frames**  
  Shih-Yao Lin  
  _2026-01-20_ · https://arxiv.org/abs/2601.13974v1  
  <details><summary>Abstract</summary>

  Frame sampling is a fundamental component in video understanding and video--language model pipelines, yet evaluating the quality of sampled frames remains challenging. Existing evaluation metrics primarily focus on perceptual quality or reconstruction fidelity, and are not designed to assess whether a set of sampled frames adequately captures informative and representative video content. We propose Spatio-Temporal Entropy Coverage (STEC), a simple and non-reference metric for evaluating the effectiveness of video frame sampling. STEC builds upon Spatio-Temporal Frame Entropy (STFE), which measures per-frame spatial information via entropy-based structural complexity, and evaluates sampled frames based on their temporal coverage and redundancy. By jointly modeling spatial information strength, temporal dispersion, and non-redundancy, STEC provides a principled and lightweight measure of sampling quality. Experiments on the MSR-VTT test-1k benchmark demonstrate that STEC clearly differentiates common sampling strategies, including random, uniform, and content-aware methods. We further show that STEC reveals robustness patterns across individual videos that are not captured by average performance alone, highlighting its practical value as a general-purpose evaluation tool for efficient video understanding. We emphasize that STEC is not designed to predict downstream task accuracy, but to provide a task-agnostic diagnostic signal for analyzing frame sampling behavior under constrained budgets.

  </details>



- **MVGD-Net: A Novel Motion-aware Video Glass Surface Detection Network**  
  Yiwei Lu, Hao Huang, Tao Yan  
  _2026-01-20_ · https://arxiv.org/abs/2601.13715v1  
  <details><summary>Abstract</summary>

  Glass surface ubiquitous in both daily life and professional environments presents a potential threat to vision-based systems, such as robot and drone navigation. To solve this challenge, most recent studies have shown significant interest in Video Glass Surface Detection (VGSD). We observe that objects in the reflection (or transmission) layer appear farther from the glass surfaces. Consequently, in video motion scenarios, the notable reflected (or transmitted) objects on the glass surface move slower than objects in non-glass regions within the same spatial plane, and this motion inconsistency can effectively reveal the presence of glass surfaces. Based on this observation, we propose a novel network, named MVGD-Net, for detecting glass surfaces in videos by leveraging motion inconsistency cues. Our MVGD-Net features three novel modules: the Cross-scale Multimodal Fusion Module (CMFM) that integrates extracted spatial features and estimated optical flow maps, the History Guided Attention Module (HGAM) and Temporal Cross Attention Module (TCAM), both of which further enhances temporal features. A Temporal-Spatial Decoder (TSD) is also introduced to fuse the spatial and temporal features for generating the glass region mask. Furthermore, for learning our network, we also propose a large-scale dataset, which comprises 312 diverse glass scenarios with a total of 19,268 frames. Extensive experiments demonstrate that our MVGD-Net outperforms relevant state-of-the-art methods.

  </details>


