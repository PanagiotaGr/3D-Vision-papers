# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-27 07:10 UTC_

Total papers shown: **19**


---

- **Towards Long-Form Spatio-Temporal Video Grounding**  
  Xin Gu, Bing Fan, Jiali Yao, Zhipeng Zhang, Yan Huang, Cheng Han, Heng Fan, Libo Zhang  
  _2026-02-26_ · https://arxiv.org/abs/2602.23294v1  
  <details><summary>Abstract</summary>

  In real scenarios, videos can span several minutes or even hours. However, existing research on spatio-temporal video grounding (STVG), given a textual query, mainly focuses on localizing targets in short videos of tens of seconds, typically less than one minute, which limits real-world applications. In this paper, we explore Long-Form STVG (LF-STVG), which aims to locate targets in long-term videos. Compared with short videos, long-term videos contain much longer temporal spans and more irrelevant information, making it difficult for existing STVG methods that process all frames at once. To address this challenge, we propose an AutoRegressive Transformer architecture for LF-STVG, termed ART-STVG. Unlike conventional STVG methods that require the entire video sequence to make predictions at once, ART-STVG treats the video as streaming input and processes frames sequentially, enabling efficient handling of long videos. To model spatio-temporal context, we design spatial and temporal memory banks and apply them to the decoders. Since memories from different moments are not always relevant to the current frame, we introduce simple yet effective memory selection strategies to provide more relevant information to the decoders, significantly improving performance. Furthermore, instead of parallel spatial and temporal localization, we propose a cascaded spatio-temporal design that connects the spatial decoder to the temporal decoder, allowing fine-grained spatial cues to assist complex temporal localization in long videos. Experiments on newly extended LF-STVG datasets show that ART-STVG significantly outperforms state-of-the-art methods, while achieving competitive performance on conventional short-form STVG.

  </details>



- **Spatio-Temporal Token Pruning for Efficient High-Resolution GUI Agents**  
  Zhou Xu, Bowen Zhou, Qi Wang, Shuwen Feng, Jingyu Xiao  
  _2026-02-26_ · https://arxiv.org/abs/2602.23235v1  
  <details><summary>Abstract</summary>

  Pure-vision GUI agents provide universal interaction capabilities but suffer from severe efficiency bottlenecks due to the massive spatiotemporal redundancy inherent in high-resolution screenshots and historical trajectories. We identify two critical misalignments in existing compression paradigms: the temporal mismatch, where uniform history encoding diverges from the agent's "fading memory" attention pattern, and the spatial topology conflict, where unstructured pruning compromises the grid integrity required for precise coordinate grounding, inducing spatial hallucinations. To address these challenges, we introduce GUIPruner, a training-free framework tailored for high-resolution GUI navigation. It synergizes Temporal-Adaptive Resolution (TAR), which eliminates historical redundancy via decay-based resizing, and Stratified Structure-aware Pruning (SSP), which prioritizes interactive foregrounds and semantic anchors while safeguarding global layout. Extensive evaluations across diverse benchmarks demonstrate that GUIPruner consistently achieves state-of-the-art performance, effectively preventing the collapse observed in large-scale models under high compression. Notably, on Qwen2-VL-2B, our method delivers a 3.4x reduction in FLOPs and a 3.3x speedup in vision encoding latency while retaining over 94% of the original performance, enabling real-time, high-precision navigation with minimal resource consumption.

  </details>



- **Motion-aware Event Suppression for Event Cameras**  
  Roberto Pellerito, Nico Messikommer, Giovanni Cioffi, Marco Cannici, Davide Scaramuzza  
  _2026-02-26_ · https://arxiv.org/abs/2602.23204v1  
  <details><summary>Abstract</summary>

  In this work, we introduce the first framework for Motion-aware Event Suppression, which learns to filter events triggered by IMOs and ego-motion in real time. Our model jointly segments IMOs in the current event stream while predicting their future motion, enabling anticipatory suppression of dynamic events before they occur. Our lightweight architecture achieves 173 Hz inference on consumer-grade GPUs with less than 1 GB of memory usage, outperforming previous state-of-the-art methods on the challenging EVIMO benchmark by 67\% in segmentation accuracy while operating at a 53\% higher inference rate. Moreover, we demonstrate significant benefits for downstream applications: our method accelerates Vision Transformer inference by 83\% via token pruning and improves event-based visual odometry accuracy, reducing Absolute Trajectory Error (ATE) by 13\%.

  </details>



- **Uni-Animator: Towards Unified Visual Colorization**  
  Xinyuan Chen, Yao Xu, Shaowen Wang, Pengjie Song, Bowen Deng  
  _2026-02-26_ · https://arxiv.org/abs/2602.23191v1  
  <details><summary>Abstract</summary>

  We propose Uni-Animator, a novel Diffusion Transformer (DiT)-based framework for unified image and video sketch colorization. Existing sketch colorization methods struggle to unify image and video tasks, suffering from imprecise color transfer with single or multiple references, inadequate preservation of high-frequency physical details, and compromised temporal coherence with motion artifacts in large-motion scenes. To tackle imprecise color transfer, we introduce visual reference enhancement via instance patch embedding, enabling precise alignment and fusion of reference color information. To resolve insufficient physical detail preservation, we design physical detail reinforcement using physical features that effectively capture and retain high-frequency textures. To mitigate motion-induced temporal inconsistency, we propose sketch-based dynamic RoPE encoding that adaptively models motion-aware spatial-temporal dependencies. Extensive experimental results demonstrate that Uni-Animator achieves competitive performance on both image and video sketch colorization, matching that of task-specific methods while unlocking unified cross-domain capabilities with high detail fidelity and robust temporal consistency.

  </details>



- **DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis**  
  Xinglong Luo, Ao Luo, Zhengning Wang, Yueqi Yang, Chaoyu Feng, Lei Lei, Bing Zeng, Shuaicheng Liu  
  _2026-02-26_ · https://arxiv.org/abs/2602.23022v1  
  <details><summary>Abstract</summary>

  Image alignment is a fundamental task in computer vision with broad applications. Existing methods predominantly employ optical flow-based image warping. However, this technique is susceptible to common challenges such as occlusions and illumination variations, leading to degraded alignment visual quality and compromised accuracy in downstream tasks. In this paper, we present DMAligner, a diffusion-based framework for image alignment through alignment-oriented view synthesis. DMAligner is crafted to tackle the challenges in image alignment from a new perspective, employing a generation-based solution that showcases strong capabilities and avoids the problems associated with flow-based image warping. Specifically, we propose a Dynamics-aware Diffusion Training approach for learning conditional image generation, synthesizing a novel view for image alignment. This incorporates a Dynamics-aware Mask Producing (DMP) module to adaptively distinguish dynamic foreground regions from static backgrounds, enabling the diffusion model to more effectively handle challenges that classical methods struggle to solve. Furthermore, we develop the Dynamic Scene Image Alignment (DSIA) dataset using Blender, which includes 1,033 indoor and outdoor scenes with over 30K image pairs tailored for image alignment. Extensive experimental results demonstrate the superiority of the proposed approach on DSIA benchmarks, as well as on a series of widely-used video datasets for qualitative comparisons. Our code is available at https://github.com/boomluo02/DMAligner.

  </details>



- **OSDaR-AR: Enhancing Railway Perception Datasets via Multi-modal Augmented Reality**  
  Federico Nesti, Gianluca D'Amico, Mauro Marinoni, Giorgio Buttazzo  
  _2026-02-26_ · https://arxiv.org/abs/2602.22920v1  
  <details><summary>Abstract</summary>

  Although deep learning has significantly advanced the perception capabilities of intelligent transportation systems, railway applications continue to suffer from a scarcity of high-quality, annotated data for safety-critical tasks like obstacle detection. While photorealistic simulators offer a solution, they often struggle with the ``sim-to-real" gap; conversely, simple image-masking techniques lack the spatio-temporal coherence required to obtain augmented single- and multi-frame scenes with the correct appearance and dimensions. This paper introduces a multi-modal augmented reality framework designed to bridge this gap by integrating photorealistic virtual objects into real-world railway sequences from the OSDaR23 dataset. Utilizing Unreal Engine 5 features, our pipeline leverages LiDAR point-clouds and INS/GNSS data to ensure accurate object placement and temporal stability across RGB frames. This paper also proposes a segmentation-based refinement strategy for INS/GNSS data to significantly improve the realism of the augmented sequences, as confirmed by the comparative study presented in the paper. Carefully designed augmented sequences are collected to produce OSDaR-AR, a public dataset designed to support the development of next-generation railway perception systems. The dataset is available at the following page: https://syndra.retis.santannapisa.it/osdarar.html

  </details>



- **ProjFlow: Projection Sampling with Flow Matching for Zero-Shot Exact Spatial Motion Control**  
  Akihisa Watanabe, Qing Yu, Edgar Simo-Serra, Kent Fujiwara  
  _2026-02-26_ · https://arxiv.org/abs/2602.22742v1  
  <details><summary>Abstract</summary>

  Generating human motion with precise spatial control is a challenging problem. Existing approaches often require task-specific training or slow optimization, and enforcing hard constraints frequently disrupts motion naturalness. Building on the observation that many animation tasks can be formulated as a linear inverse problem, we introduce ProjFlow, a training-free sampler that achieves zero-shot, exact satisfaction of linear spatial constraints while preserving motion realism. Our key advance is a novel kinematics-aware metric that encodes skeletal topology. This metric allows the sampler to enforce hard constraints by distributing corrections coherently across the entire skeleton, avoiding the unnatural artifacts of naive projection. Furthermore, for sparse inputs, such as filling in long gaps between a few keyframes, we introduce a time-varying formulation using pseudo-observations that fade during sampling. Extensive experiments on representative applications, motion inpainting, and 2D-to-3D lifting, demonstrate that ProjFlow achieves exact constraint satisfaction and matches or improves realism over zero-shot baselines, while remaining competitive with training-based controllers.

  </details>



- **AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction**  
  Hanyang Liu, Rongjun Qin  
  _2026-02-25_ · https://arxiv.org/abs/2602.22376v1  
  <details><summary>Abstract</summary>

  Recent advances in 4D scene reconstruction have significantly improved dynamic modeling across various domains. However, existing approaches remain limited under aerial conditions with single-view capture, wide spatial range, and dynamic objects of limited spatial footprint and large motion disparity. These challenges cause severe depth ambiguity and unstable motion estimation, making monocular aerial reconstruction inherently ill-posed. To this end, we present AeroDGS, a physics-guided 4D Gaussian splatting framework for monocular UAV videos. AeroDGS introduces a Monocular Geometry Lifting module that reconstructs reliable static and dynamic geometry from a single aerial sequence, providing a robust basis for dynamic estimation. To further resolve monocular ambiguity, we propose a Physics-Guided Optimization module that incorporates differentiable ground-support, upright-stability, and trajectory-smoothness priors, transforming ambiguous image cues into physically consistent motion. The framework jointly refines static backgrounds and dynamic entities with stable geometry and coherent temporal evolution. We additionally build a real-world UAV dataset that spans various altitudes and motion conditions to evaluate dynamic aerial reconstruction. Experiments on synthetic and real UAV scenes demonstrate that AeroDGS outperforms state-of-the-art methods, achieving superior reconstruction fidelity in dynamic aerial environments.

  </details>



- **WeaveTime: Stream from Earlier Frames into Emergent Memory in VideoLLMs**  
  Yulin Zhang, Cheng Shi, Sibei Yang  
  _2026-02-25_ · https://arxiv.org/abs/2602.22142v1  
  <details><summary>Abstract</summary>

  Recent advances in Multimodal Large Language Models have greatly improved visual understanding and reasoning, yet their quadratic attention and offline training protocols make them ill-suited for streaming settings where frames arrive sequentially and future observations are inaccessible. We diagnose a core limitation of current Video-LLMs, namely Time-Agnosticism, in which videos are treated as an unordered bag of evidence rather than a causally ordered sequence, yielding two failures in streams: temporal order ambiguity, in which the model cannot follow or reason over the correct chronological order, and past-current focus blindness where it fails to distinguish present observations from accumulated history. We present WeaveTime, a simple, efficient, and model agnostic framework that first teaches order and then uses order. We introduce a lightweight Temporal Reconstruction objective-our Streaming Order Perception enhancement-that instills order aware representations with minimal finetuning and no specialized streaming data. At inference, a Past-Current Dynamic Focus Cache performs uncertainty triggered, coarse-to-fine retrieval, expanding history only when needed. Plugged into exsiting Video-LLM without architectural changes, WeaveTime delivers consistent gains on representative streaming benchmarks, improving accuracy while reducing latency. These results establish WeaveTime as a practical path toward time aware stream Video-LLMs under strict online, time causal constraints. Code and weights will be made publicly available. Project Page: https://zhangyl4.github.io/publications/weavetime/

  </details>



- **Lumosaic: Hyperspectral Video via Active Illumination and Coded-Exposure Pixels**  
  Dhruv Verma, Andrew Qiu, Roberto Rangel, Ayandev Barman, Hao Yang, Chenjia Hu, Fengqi Zhang, Roman Genov, David B. Lindell, Kiriakos N. Kutulakos, et al.  
  _2026-02-25_ · https://arxiv.org/abs/2602.22140v1  
  <details><summary>Abstract</summary>

  We present Lumosaic, a compact active hyperspectral video system designed for real-time capture of dynamic scenes. Our approach combines a narrowband LED array with a coded-exposure-pixel (CEP) camera capable of high-speed, per-pixel exposure control, enabling joint encoding of scene information across space, time, and wavelength within each video frame. Unlike passive snapshot systems that divide light across multiple spectral channels simultaneously and assume no motion during a frame's exposure, Lumosaic actively synchronizes illumination and pixel-wise exposure, improving photon utilization and preserving spectral fidelity under motion. A learning-based reconstruction pipeline then recovers 31-channel hyperspectral (400-700 nm) video at 30 fps and VGA resolution, producing temporally coherent and spectrally accurate reconstructions. Experiments on synthetic and real data demonstrate that Lumosaic significantly improves reconstruction fidelity and temporal stability over existing snapshot hyperspectral imaging systems, enabling robust hyperspectral video across diverse materials and motion conditions.

  </details>



- **WeatherCity: Urban Scene Reconstruction with Controllable Multi-Weather Transformation**  
  Wenhua Wu, Huai Guan, Zhe Liu, Hesheng Wang  
  _2026-02-25_ · https://arxiv.org/abs/2602.22096v1  
  <details><summary>Abstract</summary>

  Editable high-fidelity 4D scenes are crucial for autonomous driving, as they can be applied to end-to-end training and closed-loop simulation. However, existing reconstruction methods are primarily limited to replicating observed scenes and lack the capability for diverse weather simulation. While image-level weather editing methods tend to introduce scene artifacts and offer poor controllability over the weather effects. To address these limitations, we propose WeatherCity, a novel framework for 4D urban scene reconstruction and weather editing. Specifically, we leverage a text-guided image editing model to achieve flexible editing of image weather backgrounds. To tackle the challenge of multi-weather modeling, we introduce a novel weather Gaussian representation based on shared scene features and dedicated weather-specific decoders. This representation is further enhanced with a content consistency optimization, ensuring coherent modeling across different weather conditions. Additionally, we design a physics-driven model that simulates dynamic weather effects through particles and motion patterns. Extensive experiments on multiple datasets and various scenes demonstrate that WeatherCity achieves flexible controllability, high fidelity, and temporal consistency in 4D reconstruction and weather editing. Our framework not only enables fine-grained control over weather conditions (e.g., light rain and heavy snow) but also supports object-level manipulation within the scene.

  </details>



- **Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos**  
  Matthew Strong, Wei-Jer Chang, Quentin Herau, Jiezhi Yang, Yihan Hu, Chensheng Peng, Wei Zhan  
  _2026-02-25_ · https://arxiv.org/abs/2602.22091v1  
  <details><summary>Abstract</summary>

  Ego-centric driving videos available online provide an abundant source of visual data for autonomous driving, yet their lack of annotations makes it difficult to learn representations that capture both semantic structure and 3D geometry. Recent advances in large feedforward spatial models demonstrate that point maps and ego-motion can be inferred in a single forward pass, suggesting a promising direction for scalable driving perception. We therefore propose a label-free, teacher-guided framework for learning autonomous driving representations directly from unposed videos. Unlike prior self-supervised approaches that focus primarily on frame-to-frame consistency, we posit that safe and reactive driving depends critically on temporal context. To this end, we leverage a feedforward architecture equipped with a lightweight autoregressive module, trained using multi-modal supervisory signals that guide the model to jointly predict current and future point maps, camera poses, semantic segmentation, and motion masks. Multi-modal teachers provide sequence-level pseudo-supervision, enabling LFG to learn a unified pseudo-4D representation from raw YouTube videos without poses, labels, or LiDAR. The resulting encoder not only transfers effectively to downstream autonomous driving planning on the NAVSIM benchmark, surpassing multi-camera and LiDAR baselines with only a single monocular camera, but also yields strong performance when evaluated on a range of semantic, geometric, and qualitative motion prediction tasks. These geometry and motion-aware features position LFG as a compelling video-centric foundation model for autonomous driving.

  </details>



- **AdaSpot: Spend Resolution Where It Matters for Precise Event Spotting**  
  Artur Xarles, Sergio Escalera, Thomas B. Moeslund, Albert Clapés  
  _2026-02-25_ · https://arxiv.org/abs/2602.22073v1  
  <details><summary>Abstract</summary>

  Precise Event Spotting aims to localize fast-paced actions or events in videos with high temporal precision, a key task for applications in sports analytics, robotics, and autonomous systems. Existing methods typically process all frames uniformly, overlooking the inherent spatio-temporal redundancy in video data. This leads to redundant computation on non-informative regions while limiting overall efficiency. To remain tractable, they often spatially downsample inputs, losing fine-grained details crucial for precise localization. To address these limitations, we propose \textbf{AdaSpot}, a simple yet effective framework that processes low-resolution videos to extract global task-relevant features while adaptively selecting the most informative region-of-interest in each frame for high-resolution processing. The selection is performed via an unsupervised, task-aware strategy that maintains spatio-temporal consistency across frames and avoids the training instability of learnable alternatives. This design preserves essential fine-grained visual cues with a marginal computational overhead compared to low-resolution-only baselines, while remaining far more efficient than uniform high-resolution processing. Experiments on standard PES benchmarks demonstrate that \textbf{AdaSpot} achieves state-of-the-art performance under strict evaluation metrics (\eg, $+3.96$ and $+2.26$ mAP$@0$ frames on Tennis and FineDiving), while also maintaining strong results under looser metrics. Code is available at: \href{https://github.com/arturxe2/AdaSpot}{https://github.com/arturxe2/AdaSpot}.

  </details>



- **Dream-SLAM: Dreaming the Unseen for Active SLAM in Dynamic Environments**  
  Xiangqi Meng, Pengxu Hou, Zhenjun Zhao, Javier Civera, Daniel Cremers, Hesheng Wang, Haoang Li  
  _2026-02-25_ · https://arxiv.org/abs/2602.21967v1  
  <details><summary>Abstract</summary>

  In addition to the core tasks of simultaneous localization and mapping (SLAM), active SLAM additionally in- volves generating robot actions that enable effective and efficient exploration of unknown environments. However, existing active SLAM pipelines are limited by three main factors. First, they inherit the restrictions of the underlying SLAM modules that they may be using. Second, their motion planning strategies are typically shortsighted and lack long-term vision. Third, most approaches struggle to handle dynamic scenes. To address these limitations, we propose a novel monocular active SLAM method, Dream-SLAM, which is based on dreaming cross-spatio-temporal images and semantically plausible structures of partially observed dynamic environments. The generated cross-spatio-temporal im- ages are fused with real observations to mitigate noise and data incompleteness, leading to more accurate camera pose estimation and a more coherent 3D scene representation. Furthermore, we integrate dreamed and observed scene structures to enable long- horizon planning, producing farsighted trajectories that promote efficient and thorough exploration. Extensive experiments on both public and self-collected datasets demonstrate that Dream-SLAM outperforms state-of-the-art methods in localization accuracy, mapping quality, and exploration efficiency. Source code will be publicly available upon paper acceptance.

  </details>



- **GeoMotion: Rethinking Motion Segmentation via Latent 4D Geometry**  
  Xiankang He, Peile Lin, Ying Cui, Dongyan Guo, Chunhua Shen, Xiaoqin Zhang  
  _2026-02-25_ · https://arxiv.org/abs/2602.21810v1  
  <details><summary>Abstract</summary>

  Motion segmentation in dynamic scenes is highly challenging, as conventional methods heavily rely on estimating camera poses and point correspondences from inherently noisy motion cues. Existing statistical inference or iterative optimization techniques that struggle to mitigate the cumulative errors in multi-stage pipelines often lead to limited performance or high computational cost. In contrast, we propose a fully learning-based approach that directly infers moving objects from latent feature representations via attention mechanisms, thus enabling end-to-end feed-forward motion segmentation. Our key insight is to bypass explicit correspondence estimation and instead let the model learn to implicitly disentangle object and camera motion. Supported by recent advances in 4D scene geometry reconstruction (e.g., $π^3$), the proposed method leverages reliable camera poses and rich spatial-temporal priors, which ensure stable training and robust inference for the model. Extensive experiments demonstrate that by eliminating complex pre-processing and iterative refinement, our approach achieves state-of-the-art motion segmentation performance with high efficiency. The code is available at:https://github.com/zjutcvg/GeoMotion.

  </details>



- **SF3D-RGB: Scene Flow Estimation from Monocular Camera and Sparse LiDAR**  
  Rajai Alhimdiat, Ramy Battrawy, René Schuster, Didier Stricker, Wesam Ashour  
  _2026-02-25_ · https://arxiv.org/abs/2602.21699v1  
  <details><summary>Abstract</summary>

  Scene flow estimation is an extremely important task in computer vision to support the perception of dynamic changes in the scene. For robust scene flow, learning-based approaches have recently achieved impressive results using either image-based or LiDAR-based modalities. However, these methods have tended to focus on the use of a single modality. To tackle these problems, we present a deep learning architecture, SF3D-RGB, that enables sparse scene flow estimation using 2D monocular images and 3D point clouds (e.g., acquired by LiDAR) as inputs. Our architecture is an end-to-end model that first encodes information from each modality into features and fuses them together. Then, the fused features enhance a graph matching module for better and more robust mapping matrix computation to generate an initial scene flow. Finally, a residual scene flow module further refines the initial scene flow. Our model is designed to strike a balance between accuracy and efficiency. Furthermore, experiments show that our proposed method outperforms single-modality methods and achieves better scene flow accuracy on real-world datasets while using fewer parameters compared to other state-of-the-art methods with fusion.

  </details>



- **Space-Time Forecasting of Dynamic Scenes with Motion-aware Gaussian Grouping**  
  Junmyeong Lee, Hoseung Choi, Minsu Cho  
  _2026-02-25_ · https://arxiv.org/abs/2602.21668v1  
  <details><summary>Abstract</summary>

  Forecasting dynamic scenes remains a fundamental challenge in computer vision, as limited observations make it difficult to capture coherent object-level motion and long-term temporal evolution. We present Motion Group-aware Gaussian Forecasting (MoGaF), a framework for long-term scene extrapolation built upon the 4D Gaussian Splatting representation. MoGaF introduces motion-aware Gaussian grouping and group-wise optimization to enforce physically consistent motion across both rigid and non-rigid regions, yielding spatially coherent dynamic representations. Leveraging this structured space-time representation, a lightweight forecasting module predicts future motion, enabling realistic and temporally stable scene evolution. Experiments on synthetic and real-world datasets demonstrate that MoGaF consistently outperforms existing baselines in rendering quality, motion plausibility, and long-term forecasting stability. Our project page is available at https://slime0519.github.io/mogaf

  </details>



- **Human Video Generation from a Single Image with 3D Pose and View Control**  
  Tiantian Wang, Chun-Han Yao, Tao Hu, Mallikarjun Byrasandra Ramalinga Reddy, Ming-Hsuan Yang, Varun Jampani  
  _2026-02-24_ · https://arxiv.org/abs/2602.21188v1  
  <details><summary>Abstract</summary>

  Recent diffusion methods have made significant progress in generating videos from single images due to their powerful visual generation capabilities. However, challenges persist in image-to-video synthesis, particularly in human video generation, where inferring view-consistent, motion-dependent clothing wrinkles from a single image remains a formidable problem. In this paper, we present Human Video Generation in 4D (HVG), a latent video diffusion model capable of generating high-quality, multi-view, spatiotemporally coherent human videos from a single image with 3D pose and view control. HVG achieves this through three key designs: (i) Articulated Pose Modulation, which captures the anatomical relationships of 3D joints via a novel dual-dimensional bone map and resolves self-occlusions across views by introducing 3D information; (ii) View and Temporal Alignment, which ensures multi-view consistency and alignment between a reference image and pose sequences for frame-to-frame stability; and (iii) Progressive Spatio-Temporal Sampling with temporal alignment to maintain smooth transitions in long multi-view animations. Extensive experiments on image-to-video tasks demonstrate that HVG outperforms existing methods in generating high-quality 4D human videos from diverse human images and pose inputs.

  </details>



- **UDVideoQA: A Traffic Video Question Answering Dataset for Multi-Object Spatio-Temporal Reasoning in Urban Dynamics**  
  Joseph Raj Vishal, Nagasiri Poluri, Katha Naik, Rutuja Patil, Kashyap Hegde Kota, Krishna Vinod, Prithvi Jai Ramesh, Mohammad Farhadi, Yezhou Yang, Bharatesh Chakravarthi  
  _2026-02-24_ · https://arxiv.org/abs/2602.21137v1  
  <details><summary>Abstract</summary>

  Understanding the complex, multi-agent dynamics of urban traffic remains a fundamental challenge for video language models. This paper introduces Urban Dynamics VideoQA, a benchmark dataset that captures the unscripted real-world behavior of dynamic urban scenes. UDVideoQA is curated from 16 hours of traffic footage recorded at multiple city intersections under diverse traffic, weather, and lighting conditions. It employs an event-driven dynamic blur technique to ensure privacy preservation without compromising scene fidelity. Using a unified annotation pipeline, the dataset contains 28K question-answer pairs generated across 8 hours of densely annotated video, averaging one question per second. Its taxonomy follows a hierarchical reasoning level, spanning basic understanding and attribution to event reasoning, reverse reasoning, and counterfactual inference, enabling systematic evaluation of both visual grounding and causal reasoning. Comprehensive experiments benchmark 10 SOTA VideoLMs on UDVideoQA and 8 models on a complementary video question generation benchmark. Results reveal a persistent perception-reasoning gap, showing models that excel in abstract inference often fail with fundamental visual grounding. While models like Gemini Pro achieve the highest zero-shot accuracy, fine-tuning the smaller Qwen2.5-VL 7B model on UDVideoQA bridges this gap, achieving performance comparable to proprietary systems. In VideoQGen, Gemini 2.5 Pro, and Qwen3 Max generate the most relevant and complex questions, though all models exhibit limited linguistic diversity, underscoring the need for human-centric evaluation. The UDVideoQA suite, including the dataset, annotation tools, and benchmarks for both VideoQA and VideoQGen, provides a foundation for advancing robust, privacy-aware, and real-world multimodal reasoning. UDVideoQA is available at https://ud-videoqa.github.io/UD-VideoQA/UD-VideoQA/.

  </details>


