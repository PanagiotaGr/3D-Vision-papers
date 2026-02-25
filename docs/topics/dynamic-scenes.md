# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-25 07:15 UTC_

Total papers shown: **14**


---

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



- **UFO: Unifying Feed-Forward and Optimization-based Methods for Large Driving Scene Modeling**  
  Kaiyuan Tan, Yingying Shen, Mingfei Tu, Haohui Zhu, Bing Wang, Guang Chen, Hangjun Ye, Haiyang Sun  
  _2026-02-24_ · https://arxiv.org/abs/2602.20943v1  
  <details><summary>Abstract</summary>

  Dynamic driving scene reconstruction is critical for autonomous driving simulation and closed-loop learning. While recent feed-forward methods have shown promise for 3D reconstruction, they struggle with long-range driving sequences due to quadratic complexity in sequence length and challenges in modeling dynamic objects over extended durations. We propose UFO, a novel recurrent paradigm that combines the benefits of optimization-based and feed-forward methods for efficient long-range 4D reconstruction. Our approach maintains a 4D scene representation that is iteratively refined as new observations arrive, using a visibility-based filtering mechanism to select informative scene tokens and enable efficient processing of long sequences. For dynamic objects, we introduce an object pose-guided modeling approach that supports accurate long-range motion capture. Experiments on the Waymo Open Dataset demonstrate that our method significantly outperforms both per-scene optimization and existing feed-forward methods across various sequence lengths. Notably, our approach can reconstruct 16-second driving logs within 0.5 second while maintaining superior visual quality and geometric accuracy.

  </details>



- **LST-SLAM: A Stereo Thermal SLAM System for Kilometer-Scale Dynamic Environments**  
  Zeyu Jiang, Kuan Xu, Changhao Chen  
  _2026-02-24_ · https://arxiv.org/abs/2602.20925v1  
  <details><summary>Abstract</summary>

  Thermal cameras offer strong potential for robot perception under challenging illumination and weather conditions. However, thermal Simultaneous Localization and Mapping (SLAM) remains difficult due to unreliable feature extraction, unstable motion tracking, and inconsistent global pose and map construction, particularly in dynamic large-scale outdoor environments. To address these challenges, we propose LST-SLAM, a novel large-scale stereo thermal SLAM system that achieves robust performance in complex, dynamic scenes. Our approach combines self-supervised thermal feature learning, stereo dual-level motion tracking, and geometric pose optimization. We also introduce a semantic-geometric hybrid constraint that suppresses potentially dynamic features lacking strong inter-frame geometric consistency. Furthermore, we develop an online incremental bag-of-words model for loop closure detection, coupled with global pose optimization to mitigate accumulated drift. Extensive experiments on kilometer-scale dynamic thermal datasets show that LST-SLAM significantly outperforms recent representative SLAM systems, including AirSLAM and DROID-SLAM, in both robustness and accuracy.

  </details>



- **RU4D-SLAM: Reweighting Uncertainty in Gaussian Splatting SLAM for 4D Scene Reconstruction**  
  Yangfan Zhao, Hanwei Zhang, Ke Huang, Qiufeng Wang, Zhenzhou Shao, Dengyu Wu  
  _2026-02-24_ · https://arxiv.org/abs/2602.20807v1  
  <details><summary>Abstract</summary>

  Combining 3D Gaussian splatting with Simultaneous Localization and Mapping (SLAM) has gained popularity as it enables continuous 3D environment reconstruction during motion. However, existing methods struggle in dynamic environments, particularly moving objects complicate 3D reconstruction and, in turn, hinder reliable tracking. The emergence of 4D reconstruction, especially 4D Gaussian splatting, offers a promising direction for addressing these challenges, yet its potential for 4D-aware SLAM remains largely underexplored. Along this direction, we propose a robust and efficient framework, namely Reweighting Uncertainty in Gaussian Splatting SLAM (RU4D-SLAM) for 4D scene reconstruction, that introduces temporal factors into spatial 3D representation while incorporating uncertainty-aware perception of scene changes, blurred image synthesis, and dynamic scene reconstruction. We enhance dynamic scene representation by integrating motion blur rendering, and improve uncertainty-aware tracking by extending per-pixel uncertainty modeling, which is originally designed for static scenarios, to handle blurred images. Furthermore, we propose a semantic-guided reweighting mechanism for per-pixel uncertainty estimation in dynamic scenes, and introduce a learnable opacity weight to support adaptive 4D mapping. Extensive experiments on standard benchmarks demonstrate that our method substantially outperforms state-of-the-art approaches in both trajectory accuracy and 4D scene reconstruction, particularly in dynamic environments with moving objects and low-quality inputs. Code available: https://ru4d-slam.github.io

  </details>



- **Real-time Motion Segmentation with Event-based Normal Flow**  
  Sheng Zhong, Zhongyang Ren, Xiya Zhu, Dehao Yuan, Cornelia Fermuller, Yi Zhou  
  _2026-02-24_ · https://arxiv.org/abs/2602.20790v1  
  <details><summary>Abstract</summary>

  Event-based cameras are bio-inspired sensors with pixels that independently and asynchronously respond to brightness changes at microsecond resolution, offering the potential to handle visual tasks in challenging scenarios. However, due to the sparse information content in individual events, directly processing the raw event data to solve vision tasks is highly inefficient, which severely limits the applicability of state-of-the-art methods in real-time tasks, such as motion segmentation, a fundamental task for dynamic scene understanding. Incorporating normal flow as an intermediate representation to compress motion information from event clusters within a localized region provides a more effective solution. In this work, we propose a normal flow-based motion segmentation framework for event-based vision. Leveraging the dense normal flow directly learned from event neighborhoods as input, we formulate the motion segmentation task as an energy minimization problem solved via graph cuts, and optimize it iteratively with normal flow clustering and motion model fitting. By using a normal flow-based motion model initialization and fitting method, the proposed system is able to efficiently estimate the motion models of independently moving objects with only a limited number of candidate models, which significantly reduces the computational complexity and ensures real-time performance, achieving nearly a 800x speedup in comparison to the open-source state-of-the-art method. Extensive evaluations on multiple public datasets fully demonstrate the accuracy and efficiency of our framework.

  </details>



- **RAYNOVA: 3D-Geometry-Free Auto-Regressive Driving World Modeling with Unified Spatio-Temporal Representation**  
  Yichen Xie, Chensheng Peng, Mazen Abdelfattah, Yihan Hu, Jiezhi Yang, Eric Higgins, Ryan Brigden, Masayoshi Tomizuka, Wei Zhan  
  _2026-02-24_ · https://arxiv.org/abs/2602.20685v1  
  <details><summary>Abstract</summary>

  World foundation models aim to simulate the evolution of the real world with physically plausible behavior. Unlike prior methods that handle spatial and temporal correlations separately, we propose RAYNOVA, a geometry-free world model that employs a dual-causal autoregressive framework. It follows both scale-wise and temporal topological orders in the autoregressive process, and leverages global attention for unified 4D spatio-temporal reasoning. Different from existing works that impose strong 3D geometric priors, RAYNOVA constructs an isotropic spatio-temporal representation across views, frames, and scales based on relative Plücker-ray positional encoding, enabling robust generalization to diverse camera setups and ego motions. We further introduce a recurrent training paradigm to alleviate distribution drift in long-horizon video generation. RAYNOVA achieves state-of-the-art multi-view video generation results on nuScenes, while offering higher throughput and strong controllability under diverse input conditions, generalizing to novel views and camera configurations without explicit 3D scene representation. Our code will be released at http://yichen928.github.io/raynova.

  </details>



- **SurgAtt-Tracker: Online Surgical Attention Tracking via Temporal Proposal Reranking and Motion-Aware Refinement**  
  Rulin Zhou, Guankun Wang, An Wang, Yujie Ma, Lixin Ouyang, Bolin Cui, Junyan Li, Chaowei Zhu, Mingyang Li, Ming Chen, et al.  
  _2026-02-24_ · https://arxiv.org/abs/2602.20636v1  
  <details><summary>Abstract</summary>

  Accurate and stable field-of-view (FoV) guidance is critical for safe and efficient minimally invasive surgery, yet existing approaches often conflate visual attention estimation with downstream camera control or rely on direct object-centric assumptions. In this work, we formulate surgical attention tracking as a spatio-temporal learning problem and model surgeon focus as a dense attention heatmap, enabling continuous and interpretable frame-wise FoV guidance. We propose SurgAtt-Tracker, a holistic framework that robustly tracks surgical attention by exploiting temporal coherence through proposal-level reranking and motion-aware refinement, rather than direct regression. To support systematic training and evaluation, we introduce SurgAtt-1.16M, a large-scale benchmark with a clinically grounded annotation protocol that enables comprehensive heatmap-based attention analysis across procedures and institutions. Extensive experiments on multiple surgical datasets demonstrate that SurgAtt-Tracker consistently achieves state-of-the-art performance and strong robustness under occlusion, multi-instrument interference, and cross-domain settings. Beyond attention tracking, our approach provides a frame-wise FoV guidance signal that can directly support downstream robotic FoV planning and automatic camera control.

  </details>



- **WildGHand: Learning Anti-Perturbation Gaussian Hand Avatars from Monocular In-the-Wild Videos**  
  Hanhui Li, Xuan Huang, Wanquan Liu, Yuhao Cheng, Long Chen, Yiqiang Yan, Xiaodan Liang, Chenqiang Gao  
  _2026-02-24_ · https://arxiv.org/abs/2602.20556v1  
  <details><summary>Abstract</summary>

  Despite recent progress in 3D hand reconstruction from monocular videos, most existing methods rely on data captured in well-controlled environments and therefore degrade in real-world settings with severe perturbations, such as hand-object interactions, extreme poses, illumination changes, and motion blur. To tackle these issues, we introduce WildGHand, an optimization-based framework that enables self-adaptive 3D Gaussian splatting on in-the-wild videos and produces high-fidelity hand avatars. WildGHand incorporates two key components: (i) a dynamic perturbation disentanglement module that explicitly represents perturbations as time-varying biases on 3D Gaussian attributes during optimization, and (ii) a perturbation-aware optimization strategy that generates per-frame anisotropic weighted masks to guide optimization. Together, these components allow the framework to identify and suppress perturbations across both spatial and temporal dimensions. We further curate a dataset of monocular hand videos captured under diverse perturbations to benchmark in-the-wild hand avatar reconstruction. Extensive experiments on this dataset and two public datasets demonstrate that WildGHand achieves state-of-the-art performance and substantially improves over its base model across multiple metrics (e.g., up to a $15.8\%$ relative gain in PSNR and a $23.1\%$ relative reduction in LPIPS). Our implementation and dataset are available at https://github.com/XuanHuang0/WildGHand.

  </details>



- **gQIR: Generative Quanta Image Reconstruction**  
  Aryan Garg, Sizhuo Ma, Mohit Gupta  
  _2026-02-23_ · https://arxiv.org/abs/2602.20417v1  
  <details><summary>Abstract</summary>

  Capturing high-quality images from only a few detected photons is a fundamental challenge in computational imaging. Single-photon avalanche diode (SPAD) sensors promise high-quality imaging in regimes where conventional cameras fail, but raw \emph{quanta frames} contain only sparse, noisy, binary photon detections. Recovering a coherent image from a burst of such frames requires handling alignment, denoising, and demosaicing (for color) under noise statistics far outside those assumed by standard restoration pipelines or modern generative models. We present an approach that adapts large text-to-image latent diffusion models to the photon-limited domain of quanta burst imaging. Our method leverages the structural and semantic priors of internet-scale diffusion models while introducing mechanisms to handle Bernoulli photon statistics. By integrating latent-space restoration with burst-level spatio-temporal reasoning, our approach produces reconstructions that are both photometrically faithful and perceptually pleasing, even under high-speed motion. We evaluate the method on synthetic benchmarks and new real-world datasets, including the first color SPAD burst dataset and a challenging \textit{Deforming (XD)} video benchmark. Across all settings, the approach substantially improves perceptual quality over classical and modern learning-based baselines, demonstrating the promise of adapting large generative priors to extreme photon-limited sensing. Code at \href{https://github.com/Aryan-Garg/gQIR}{https://github.com/Aryan-Garg/gQIR}.

  </details>



- **N4MC: Neural 4D Mesh Compression**  
  Guodong Chen, Huanshuo Dong, Mallesham Dasari  
  _2026-02-23_ · https://arxiv.org/abs/2602.20312v1  
  <details><summary>Abstract</summary>

  We present N4MC, the first 4D neural compression framework to efficiently compress time-varying mesh sequences by exploiting their temporal redundancy. Unlike prior neural mesh compression methods that treat each mesh frame independently, N4MC takes inspiration from inter-frame compression in 2D video codecs, and learns motion compensation in long mesh sequences. Specifically, N4MC converts consecutive irregular mesh frames into regular 4D tensors to provide a uniform and compact representation. These tensors are then condensed using an auto-decoder, which captures both spatial and temporal correlations for redundancy removal. To enhance temporal coherence, we introduce a transformer-based interpolation model that predicts intermediate mesh frames conditioned on latent embeddings derived from tracked volume centers, eliminating motion ambiguities. Extensive evaluations show that N4MC outperforms state-of-the-art in rate-distortion performance, while enabling real-time decoding of 4D mesh sequences. The implementation of our method is available at: https://github.com/frozzzen3/N4MC.

  </details>



- **Flow3r: Factored Flow Prediction for Scalable Visual Geometry Learning**  
  Zhongxiao Cong, Qitao Zhao, Minsik Jeon, Shubham Tulsiani  
  _2026-02-23_ · https://arxiv.org/abs/2602.20157v1  
  <details><summary>Abstract</summary>

  Current feed-forward 3D/4D reconstruction systems rely on dense geometry and pose supervision -- expensive to obtain at scale and particularly scarce for dynamic real-world scenes. We present Flow3r, a framework that augments visual geometry learning with dense 2D correspondences (`flow') as supervision, enabling scalable training from unlabeled monocular videos. Our key insight is that the flow prediction module should be factored: predicting flow between two images using geometry latents from one and pose latents from the other. This factorization directly guides the learning of both scene geometry and camera motion, and naturally extends to dynamic scenes. In controlled experiments, we show that factored flow prediction outperforms alternative designs and that performance scales consistently with unlabeled data. Integrating factored flow into existing visual geometry architectures and training with ${\sim}800$K unlabeled videos, Flow3r achieves state-of-the-art results across eight benchmarks spanning static and dynamic scenes, with its largest gains on in-the-wild dynamic videos where labeled data is most scarce.

  </details>



- **Learning Positive-Incentive Point Sampling in Neural Implicit Fields for Object Pose Estimation**  
  Yifei Shi, Boyan Wan, Xin Xu, Kai Xu  
  _2026-02-23_ · https://arxiv.org/abs/2602.19937v1  
  <details><summary>Abstract</summary>

  Learning neural implicit fields of 3D shapes is a rapidly emerging field that enables shape representation at arbitrary resolutions. Due to the flexibility, neural implicit fields have succeeded in many research areas, including shape reconstruction, novel view image synthesis, and more recently, object pose estimation. Neural implicit fields enable learning dense correspondences between the camera space and the object's canonical space-including unobserved regions in camera space-significantly boosting object pose estimation performance in challenging scenarios like highly occluded objects and novel shapes. Despite progress, predicting canonical coordinates for unobserved camera-space regions remains challenging due to the lack of direct observational signals. This necessitates heavy reliance on the model's generalization ability, resulting in high uncertainty. Consequently, densely sampling points across the entire camera space may yield inaccurate estimations that hinder the learning process and compromise performance. To alleviate this problem, we propose a method combining an SO(3)-equivariant convolutional implicit network and a positive-incentive point sampling (PIPS) strategy. The SO(3)-equivariant convolutional implicit network estimates point-level attributes with SO(3)-equivariance at arbitrary query locations, demonstrating superior performance compared to most existing baselines. The PIPS strategy dynamically determines sampling locations based on the input, thereby boosting the network's accuracy and training efficiency. Our method outperforms the state-of-the-art on three pose estimation datasets. Notably, it demonstrates significant improvements in challenging scenarios, such as objects captured with unseen pose, high occlusion, novel geometry, and severe noise.

  </details>



- **No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection**  
  Zunkai Dai, Ke Li, Jiajia Liu, Jie Yang, Yuanyuan Qiao  
  _2026-02-22_ · https://arxiv.org/abs/2602.19248v1  
  <details><summary>Abstract</summary>

  The collection and detection of video anomaly data has long been a challenging problem due to its rare occurrence and spatio-temporal scarcity. Existing video anomaly detection (VAD) methods under perform in open-world scenarios. Key contributing factors include limited dataset diversity, and inadequate understanding of context-dependent anomalous semantics. To address these issues, i) we propose LAVIDA, an end-to-end zero-shot video anomaly detection framework. ii) LAVIDA employs an Anomaly Exposure Sampler that transforms segmented objects into pseudo-anomalies to enhance model adaptability to unseen anomaly categories. It further integrates a Multimodal Large Language Model (MLLM) to bolster semantic comprehension capabilities. Additionally, iii) we design a token compression approach based on reverse attention to handle the spatio-temporal scarcity of anomalous patterns and decrease computational cost. The training process is conducted solely on pseudo anomalies without any VAD data. Evaluations across four benchmark VAD datasets demonstrate that LAVIDA achieves SOTA performance in both frame-level and pixel-level anomaly detection under the zero-shot setting. Our code is available in https://github.com/VitaminCreed/LAVIDA.

  </details>


