# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-23 07:19 UTC_

Total papers shown: **16**


---

- **SARAH: Spatially Aware Real-time Agentic Humans**  
  Evonne Ng, Siwei Zhang, Zhang Chen, Michael Zollhoefer, Alexander Richard  
  _2026-02-20_ · https://arxiv.org/abs/2602.18432v1  
  <details><summary>Abstract</summary>

  As embodied agents become central to VR, telepresence, and digital human applications, their motion must go beyond speech-aligned gestures: agents should turn toward users, respond to their movement, and maintain natural gaze. Current methods lack this spatial awareness. We close this gap with the first real-time, fully causal method for spatially-aware conversational motion, deployable on a streaming VR headset. Given a user's position and dyadic audio, our approach produces full-body motion that aligns gestures with speech while orienting the agent according to the user. Our architecture combines a causal transformer-based VAE with interleaved latent tokens for streaming inference and a flow matching model conditioned on user trajectory and audio. To support varying gaze preferences, we introduce a gaze scoring mechanism with classifier-free guidance to decouple learning from control: the model captures natural spatial alignment from data, while users can adjust eye contact intensity at inference time. On the Embody 3D dataset, our method achieves state-of-the-art motion quality at over 300 FPS -- 3x faster than non-causal baselines -- while capturing the subtle spatial dynamics of natural conversation. We validate our approach on a live VR system, bringing spatially-aware conversational agents to real-time deployment. Please see https://evonneng.github.io/sarah/ for details.

  </details>



- **CapNav: Benchmarking Vision Language Models on Capability-conditioned Indoor Navigation**  
  Xia Su, Ruiqi Chen, Benlin Liu, Jingwei Ma, Zonglin Di, Ranjay Krishna, Jon Froehlich  
  _2026-02-20_ · https://arxiv.org/abs/2602.18424v1  
  <details><summary>Abstract</summary>

  Vision-Language Models (VLMs) have shown remarkable progress in Vision-Language Navigation (VLN), offering new possibilities for navigation decision-making that could benefit both robotic platforms and human users. However, real-world navigation is inherently conditioned by the agent's mobility constraints. For example, a sweeping robot cannot traverse stairs, while a quadruped can. We introduce Capability-Conditioned Navigation (CapNav), a benchmark designed to evaluate how well VLMs can navigate complex indoor spaces given an agent's specific physical and operational capabilities. CapNav defines five representative human and robot agents, each described with physical dimensions, mobility capabilities, and environmental interaction abilities. CapNav provides 45 real-world indoor scenes, 473 navigation tasks, and 2365 QA pairs to test if VLMs can traverse indoor environments based on agent capabilities. We evaluate 13 modern VLMs and find that current VLM's navigation performance drops sharply as mobility constraints tighten, and that even state-of-the-art models struggle with obstacle types that require reasoning on spatial dimensions. We conclude by discussing the implications for capability-aware navigation and the opportunities for advancing embodied spatial reasoning in future VLMs. The benchmark is available at https://github.com/makeabilitylab/CapNav

  </details>



- **Self-Aware Object Detection via Degradation Manifolds**  
  Stefan Becker, Simon Weiss, Wolfgang Hübner, Michael Arens  
  _2026-02-20_ · https://arxiv.org/abs/2602.18394v1  
  <details><summary>Abstract</summary>

  Object detectors achieve strong performance under nominal imaging conditions but can fail silently when exposed to blur, noise, compression, adverse weather, or resolution changes. In safety-critical settings, it is therefore insufficient to produce predictions without assessing whether the input remains within the detector's nominal operating regime. We refer to this capability as self-aware object detection. We introduce a degradation-aware self-awareness framework based on degradation manifolds, which explicitly structure a detector's feature space according to image degradation rather than semantic content. Our method augments a standard detection backbone with a lightweight embedding head trained via multi-layer contrastive learning. Images sharing the same degradation composition are pulled together, while differing degradation configurations are pushed apart, yielding a geometrically organized representation that captures degradation type and severity without requiring degradation labels or explicit density modeling. To anchor the learned geometry, we estimate a pristine prototype from clean training embeddings, defining a nominal operating point in representation space. Self-awareness emerges as geometric deviation from this reference, providing an intrinsic, image-level signal of degradation-induced shift that is independent of detection confidence. Extensive experiments on synthetic corruption benchmarks, cross-dataset zero-shot transfer, and natural weather-induced distribution shifts demonstrate strong pristine-degraded separability, consistent behavior across multiple detector architectures, and robust generalization under semantic shift. These results suggest that degradation-aware representation geometry provides a practical and detector-agnostic foundation.

  </details>



- **G-LoG Bi-filtration for Medical Image Classification**  
  Qingsong Wang, Jiaxing He, Bingzhe Hou, Tieru Wu, Yang Cao, Cailing Yao  
  _2026-02-20_ · https://arxiv.org/abs/2602.18329v1  
  <details><summary>Abstract</summary>

  Building practical filtrations on objects to detect topological and geometric features is an important task in the field of Topological Data Analysis (TDA). In this paper, leveraging the ability of the Laplacian of Gaussian operator to enhance the boundaries of medical images, we define the G-LoG (Gaussian-Laplacian of Gaussian) bi-filtration to generate the features more suitable for multi-parameter persistence module. By modeling volumetric images as bounded functions, then we prove the interleaving distance on the persistence modules obtained from our bi-filtrations on the bounded functions is stable with respect to the maximum norm of the bounded functions. Finally, we conduct experiments on the MedMNIST dataset, comparing our bi-filtration against single-parameter filtration and the established deep learning baselines, including Google AutoML Vision, ResNet, AutoKeras and auto-sklearn. Experiments results demonstrate that our bi-filtration significantly outperforms single-parameter filtration. Notably, a simple Multi-Layer Perceptron (MLP) trained on the topological features generated by our bi-filtration achieves performance comparable to complex deep learning models trained on the original dataset.

  </details>



- **Diff2DGS: Reliable Reconstruction of Occluded Surgical Scenes via 2D Gaussian Splatting**  
  Tianyi Song, Danail Stoyanov, Evangelos Mazomenos, Francisco Vasconcelos  
  _2026-02-20_ · https://arxiv.org/abs/2602.18314v1  
  <details><summary>Abstract</summary>

  Real-time reconstruction of deformable surgical scenes is vital for advancing robotic surgery, improving surgeon guidance, and enabling automation. Recent methods achieve dense reconstructions from da Vinci robotic surgery videos, with Gaussian Splatting (GS) offering real-time performance via graphics acceleration. However, reconstruction quality in occluded regions remains limited, and depth accuracy has not been fully assessed, as benchmarks like EndoNeRF and StereoMIS lack 3D ground truth. We propose Diff2DGS, a novel two-stage framework for reliable 3D reconstruction of occluded surgical scenes. In the first stage, a diffusion-based video module with temporal priors inpaints tissue occluded by instruments with high spatial-temporal consistency. In the second stage, we adapt 2D Gaussian Splatting (2DGS) with a Learnable Deformation Model (LDM) to capture dynamic tissue deformation and anatomical geometry. We also extend evaluation beyond prior image-quality metrics by performing quantitative depth accuracy analysis on the SCARED dataset. Diff2DGS outperforms state-of-the-art approaches in both appearance and geometry, reaching 38.02 dB PSNR on EndoNeRF and 34.40 dB on StereoMIS. Furthermore, our experiments demonstrate that optimizing for image quality alone does not necessarily translate into optimal 3D reconstruction accuracy. To address this, we further optimize the depth quality of the reconstructed 3D results, ensuring more faithful geometry in addition to high-fidelity appearance.

  </details>



- **Multi-Level Conditioning by Pairing Localized Text and Sketch for Fashion Image Generation**  
  Ziyue Liu, Davide Talon, Federico Girella, Zanxi Ruan, Mattia Mondo, Loris Bazzani, Yiming Wang, Marco Cristani  
  _2026-02-20_ · https://arxiv.org/abs/2602.18309v1  
  <details><summary>Abstract</summary>

  Sketches offer designers a concise yet expressive medium for early-stage fashion ideation by specifying structure, silhouette, and spatial relationships, while textual descriptions complement sketches to convey material, color, and stylistic details. Effectively combining textual and visual modalities requires adherence to the sketch visual structure when leveraging the guidance of localized attributes from text. We present LOcalized Text and Sketch with multi-level guidance (LOTS), a framework that enhances fashion image generation by combining global sketch guidance with multiple localized sketch-text pairs. LOTS employs a Multi-level Conditioning Stage to independently encode local features within a shared latent space while maintaining global structural coordination. Then, the Diffusion Pair Guidance stage integrates both local and global conditioning via attention-based guidance within the diffusion model's multi-step denoising process. To validate our method, we develop Sketchy, the first fashion dataset where multiple text-sketch pairs are provided per image. Sketchy provides high-quality, clean sketches with a professional look and consistent structure. To assess robustness beyond this setting, we also include an "in the wild" split with non-expert sketches, featuring higher variability and imperfections. Experiments demonstrate that our method strengthens global structural adherence while leveraging richer localized semantic guidance, achieving improvement over state-of-the-art. The dataset, platform, and code are publicly available.

  </details>



- **DEIG: Detail-Enhanced Instance Generation with Fine-Grained Semantic Control**  
  Shiyan Du, Conghan Yue, Xinyu Cheng, Dongyu Zhang  
  _2026-02-20_ · https://arxiv.org/abs/2602.18282v1  
  <details><summary>Abstract</summary>

  Multi-Instance Generation has advanced significantly in spatial placement and attribute binding. However, existing approaches still face challenges in fine-grained semantic understanding, particularly when dealing with complex textual descriptions. To overcome these limitations, we propose DEIG, a novel framework for fine-grained and controllable multi-instance generation. DEIG integrates an Instance Detail Extractor (IDE) that transforms text encoder embeddings into compact, instance-aware representations, and a Detail Fusion Module (DFM) that applies instance-based masked attention to prevent attribute leakage across instances. These components enable DEIG to generate visually coherent multi-instance scenes that precisely match rich, localized textual descriptions. To support fine-grained supervision, we construct a high-quality dataset with detailed, compositional instance captions generated by VLMs. We also introduce DEIG-Bench, a new benchmark with region-level annotations and multi-attribute prompts for both humans and objects. Experiments demonstrate that DEIG consistently outperforms existing approaches across multiple benchmarks in spatial consistency, semantic accuracy, and compositional generalization. Moreover, DEIG functions as a plug-and-play module, making it easily integrable into standard diffusion-based pipelines.

  </details>



- **BLM-Guard: Explainable Multimodal Ad Moderation with Chain-of-Thought and Policy-Aligned Rewards**  
  Yiran Yang, Zhaowei Liu, Yuan Yuan, Yukun Song, Xiong Ma, Yinghao Song, Xiangji Zeng, Lu Sun, Yulu Wang, Hai Zhou, et al.  
  _2026-02-20_ · https://arxiv.org/abs/2602.18193v1  
  <details><summary>Abstract</summary>

  Short-video platforms now host vast multimodal ads whose deceptive visuals, speech and subtitles demand finer-grained, policy-driven moderation than community safety filters. We present BLM-Guard, a content-audit framework for commercial ads that fuses Chain-of-Thought reasoning with rule-based policy principles and a critic-guided reward. A rule-driven ICoT data-synthesis pipeline jump-starts training by generating structured scene descriptions, reasoning chains and labels, cutting annotation costs. Reinforcement learning then refines the model using a composite reward balancing causal coherence with policy adherence. A multitask architecture models intra-modal manipulations (e.g., exaggerated imagery) and cross-modal mismatches (e.g., subtitle-speech drift), boosting robustness. Experiments on real short-video ads show BLM-Guard surpasses strong baselines in accuracy, consistency and generalization.

  </details>



- **Evaluating Graphical Perception Capabilities of Vision Transformers**  
  Poonam Poonam, Pere-Pau Vázquez, Timo Ropinski  
  _2026-02-20_ · https://arxiv.org/abs/2602.18178v1  
  <details><summary>Abstract</summary>

  Vision Transformers, ViTs, have emerged as a powerful alternative to convolutional neural networks, CNNs, in a variety of image-based tasks. While CNNs have previously been evaluated for their ability to perform graphical perception tasks, which are essential for interpreting visualizations, the perceptual capabilities of ViTs remain largely unexplored. In this work, we investigate the performance of ViTs in elementary visual judgment tasks inspired by the foundational studies of Cleveland and McGill, which quantified the accuracy of human perception across different visual encodings. Inspired by their study, we benchmark ViTs against CNNs and human participants in a series of controlled graphical perception tasks. Our results reveal that, although ViTs demonstrate strong performance in general vision tasks, their alignment with human-like graphical perception in the visualization domain is limited. This study highlights key perceptual gaps and points to important considerations for the application of ViTs in visualization systems and graphical perceptual modeling.

  </details>



- **Have We Mastered Scale in Deep Monocular Visual SLAM? The ScaleMaster Dataset and Benchmark**  
  Hyoseok Ju, Bokeon Suh, Giseop Kim  
  _2026-02-20_ · https://arxiv.org/abs/2602.18174v1  
  <details><summary>Abstract</summary>

  Recent advances in deep monocular visual Simultaneous Localization and Mapping (SLAM) have achieved impressive accuracy and dense reconstruction capabilities, yet their robustness to scale inconsistency in large-scale indoor environments remains largely unexplored. Existing benchmarks are limited to room-scale or structurally simple settings, leaving critical issues of intra-session scale drift and inter-session scale ambiguity insufficiently addressed. To fill this gap, we introduce the ScaleMaster Dataset, the first benchmark explicitly designed to evaluate scale consistency under challenging scenarios such as multi-floor structures, long trajectories, repetitive views, and low-texture regions. We systematically analyze the vulnerability of state-of-the-art deep monocular visual SLAM systems to scale inconsistency, providing both quantitative and qualitative evaluations. Crucially, our analysis extends beyond traditional trajectory metrics to include a direct map-to-map quality assessment using metrics like Chamfer distance against high-fidelity 3D ground truth. Our results reveal that while recent deep monocular visual SLAM systems demonstrate strong performance on existing benchmarks, they suffer from severe scale-related failures in realistic, large-scale indoor environments. By releasing the ScaleMaster dataset and baseline results, we aim to establish a foundation for future research toward developing scale-consistent and reliable visual SLAM systems.

  </details>



- **GrandTour: A Legged Robotics Dataset in the Wild for Multi-Modal Perception and State Estimation**  
  Jonas Frey, Turcan Tuna, Frank Fu, Katharine Patterson, Tianao Xu, Maurice Fallon, Cesar Cadena, Marco Hutter  
  _2026-02-20_ · https://arxiv.org/abs/2602.18164v1  
  <details><summary>Abstract</summary>

  Accurate state estimation and multi-modal perception are prerequisites for autonomous legged robots in complex, large-scale environments. To date, no large-scale public legged-robot dataset captures the real-world conditions needed to develop and benchmark algorithms for legged-robot state estimation, perception, and navigation. To address this, we introduce the GrandTour dataset, a multi-modal legged-robotics dataset collected across challenging outdoor and indoor environments, featuring an ANYbotics ANYmal-D quadruped equipped with the \boxi multi-modal sensor payload. GrandTour spans a broad range of environments and operational scenarios across distinct test sites, ranging from alpine scenery and forests to demolished buildings and urban areas, and covers a wide variation in scale, complexity, illumination, and weather conditions. The dataset provides time-synchronized sensor data from spinning LiDARs, multiple RGB cameras with complementary characteristics, proprioceptive sensors, and stereo depth cameras. Moreover, it includes high-precision ground-truth trajectories from satellite-based RTK-GNSS and a Leica Geosystems total station. This dataset supports research in SLAM, high-precision state estimation, and multi-modal learning, enabling rigorous evaluation and development of new approaches to sensor fusion in legged robotic systems. With its extensive scope, GrandTour represents the largest open-access legged-robotics dataset to date. The dataset is available at https://grand-tour.leggedrobotics.com, on HuggingFace (ROS-independent), and in ROS formats, along with tools and demo resources.

  </details>



- **RamanSeg: Interpretability-driven Deep Learning on Raman Spectra for Cancer Diagnosis**  
  Chris Tomy, Mo Vali, David Pertzborn, Tammam Alamatouri, Anna Mühlig, Orlando Guntinas-Lichius, Anna Xylander, Eric Michele Fantuzzi, Matteo Negro, Francesco Crisafi, et al.  
  _2026-02-20_ · https://arxiv.org/abs/2602.18119v1  
  <details><summary>Abstract</summary>

  Histopathology, the current gold standard for cancer diagnosis, involves the manual examination of tissue samples after chemical staining, a time-consuming process requiring expert analysis. Raman spectroscopy is an alternative, stain-free method of extracting information from samples. Using nnU-Net, we trained a segmentation model on a novel dataset of spatial Raman spectra aligned with tumour annotations, achieving a mean foreground Dice score of 80.9%, surpassing previous work. Furthermore, we propose a novel, interpretable, prototype-based architecture called RamanSeg. RamanSeg classifies pixels based on discovered regions of the training set, generating a segmentation mask. Two variants of RamanSeg allow a trade-off between interpretability and performance: one with prototype projection and another projection-free version. The projection-free RamanSeg outperformed a U-Net baseline with a mean foreground Dice score of 67.3%, offering a meaningful improvement over a black-box training approach.

  </details>



- **OODBench: Out-of-Distribution Benchmark for Large Vision-Language Models**  
  Ling Lin, Yang Bai, Heng Su, Congcong Zhu, Yaoxing Wang, Yang Zhou, Huazhu Fu, Jingrun Chen  
  _2026-02-20_ · https://arxiv.org/abs/2602.18094v1  
  <details><summary>Abstract</summary>

  Existing Visual-Language Models (VLMs) have achieved significant progress by being trained on massive-scale datasets, typically under the assumption that data are independent and identically distributed (IID). However, in real-world scenarios, it is often impractical to expect that all data processed by an AI system satisfy this assumption. Furthermore, failure to appropriately handle out-of-distribution (OOD) objects may introduce safety risks in real-world applications (e.g., autonomous driving or medical assistance). Unfortunately, current research has not yet provided valid benchmarks that can comprehensively assess the performance of VLMs in response to OOD data. Therefore, we propose OODBench, a predominantly automated method with minimal human verification, for constructing new benchmarks and evaluating the ability of VLMs to process OOD data. OODBench contains 40K instance-level OOD instance-category pairs, and we show that current VLMs still exhibit notable performance degradation on OODBench, even when the underlying image categories are common. In addition, we propose a reliable automated assessment metric that employs a Basic-to-Advanced Progression of prompted questions to assess the impact of OOD data on questions of varying difficulty more fully. Lastly, we summarize substantial findings and insights to facilitate future research in the acquisition and evaluation of OOD data.

  </details>



- **DohaScript: A Large-Scale Multi-Writer Dataset for Continuous Handwritten Hindi Text**  
  Kunwar Arpit Singh, Ankush Prakash, Haroon R Lone  
  _2026-02-20_ · https://arxiv.org/abs/2602.18089v1  
  <details><summary>Abstract</summary>

  Despite having hundreds of millions of speakers, handwritten Devanagari text remains severely underrepresented in publicly available benchmark datasets. Existing resources are limited in scale, focus primarily on isolated characters or short words, and lack controlled lexical content and writer level diversity, which restricts their utility for modern data driven handwriting analysis. As a result, they fail to capture the continuous, fused, and structurally complex nature of Devanagari handwriting, where characters are connected through a shared shirorekha (horizontal headline) and exhibit rich ligature formations. We introduce DohaScript, a large scale, multi writer dataset of handwritten Hindi text collected from 531 unique contributors. The dataset is designed as a parallel stylistic corpus, in which all writers transcribe the same fixed set of six traditional Hindi dohas (couplets). This controlled design enables systematic analysis of writer specific variation independent of linguistic content, and supports tasks such as handwriting recognition, writer identification, style analysis, and generative modeling. The dataset is accompanied by non identifiable demographic metadata, rigorous quality curation based on objective sharpness and resolution criteria, and page level layout difficulty annotations that facilitate stratified benchmarking. Baseline experiments demonstrate clear quality separation and strong generalization to unseen writers, highlighting the dataset's reliability and practical value. DohaScript is intended to serve as a standardized and reproducible benchmark for advancing research on continuous handwritten Devanagari text in low resource script settings.

  </details>



- **Faster Training, Fewer Labels: Self-Supervised Pretraining for Fine-Grained BEV Segmentation**  
  Daniel Busch, Christian Bohn, Thomas Kurbiel, Klaus Friedrichs, Richard Meyes, Tobias Meisen  
  _2026-02-20_ · https://arxiv.org/abs/2602.18066v1  
  <details><summary>Abstract</summary>

  Dense Bird's Eye View (BEV) semantic maps are central to autonomous driving, yet current multi-camera methods depend on costly, inconsistently annotated BEV ground truth. We address this limitation with a two-phase training strategy for fine-grained road marking segmentation that removes full supervision during pretraining and halves the amount of training data during fine-tuning while still outperforming the comparable supervised baseline model. During the self-supervised pretraining, BEVFormer predictions are differentiably reprojected into the image plane and trained against multi-view semantic pseudo-labels generated by the widely used semantic segmentation model Mask2Former. A temporal loss encourages consistency across frames. The subsequent supervised fine-tuning phase requires only 50% of the dataset and significantly less training time. With our method, the fine-tuning benefits from rich priors learned during pretraining boosting the performance and BEV segmentation quality (up to +2.5pp mIoU over the fully supervised baseline) on nuScenes. It simultaneously halves the usage of annotation data and reduces total training time by up to two thirds. The results demonstrate that differentiable reprojection plus camera perspective pseudo labels yields transferable BEV features and a scalable path toward reduced-label autonomous perception.

  </details>



- **3DMedAgent: Unified Perception-to-Understanding for 3D Medical Analysis**  
  Ziyue Wang, Linghan Cai, Chang Han Low, Haofeng Liu, Junde Wu, Jingyu Wang, Rui Wang, Lei Song, Jiang Bian, Jingjing Fu, et al.  
  _2026-02-20_ · https://arxiv.org/abs/2602.18064v1  
  <details><summary>Abstract</summary>

  3D CT analysis spans a continuum from low-level perception to high-level clinical understanding. Existing 3D-oriented analysis methods adopt either isolated task-specific modeling or task-agnostic end-to-end paradigms to produce one-hop outputs, impeding the systematic accumulation of perceptual evidence for downstream reasoning. In parallel, recent multimodal large language models (MLLMs) exhibit improved visual perception and can integrate visual and textual information effectively, yet their predominantly 2D-oriented designs fundamentally limit their ability to perceive and analyze volumetric medical data. To bridge this gap, we propose 3DMedAgent, a unified agent that enables 2D MLLMs to perform general 3D CT analysis without 3D-specific fine-tuning. 3DMedAgent coordinates heterogeneous visual and textual tools through a flexible MLLM agent, progressively decomposing complex 3D analysis into tractable subtasks that transition from global to regional views, from 3D volumes to informative 2D slices, and from visual evidence to structured textual representations. Central to this design, 3DMedAgent maintains a long-term structured memory that aggregates intermediate tool outputs and supports query-adaptive, evidence-driven multi-step reasoning. We further introduce the DeepChestVQA benchmark for evaluating unified perception-to-understanding capabilities in 3D thoracic imaging. Experiments across over 40 tasks demonstrate that 3DMedAgent consistently outperforms general, medical, and 3D-specific MLLMs, highlighting a scalable path toward general-purpose 3D clinical assistants.Code and data are available at \href{https://github.com/jinlab-imvr/3DMedAgent}{https://github.com/jinlab-imvr/3DMedAgent}.

  </details>


