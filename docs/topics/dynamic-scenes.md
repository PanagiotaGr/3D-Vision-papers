# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-06 07:06 UTC_

Total papers shown: **13**


---

- **Transformer-Based Inpainting for Real-Time 3D Streaming in Sparse Multi-Camera Setups**  
  Leif Van Holland, Domenic Zingsheim, Mana Takhsha, Hannah Dröge, Patrick Stotko, Markus Plack, Reinhard Klein  
  _2026-03-05_ · https://arxiv.org/abs/2603.05507v1  
  <details><summary>Abstract</summary>

  High-quality 3D streaming from multiple cameras is crucial for immersive experiences in many AR/VR applications. The limited number of views - often due to real-time constraints - leads to missing information and incomplete surfaces in the rendered images. Existing approaches typically rely on simple heuristics for the hole filling, which can result in inconsistencies or visual artifacts. We propose to complete the missing textures using a novel, application-targeted inpainting method independent of the underlying representation as an image-based post-processing step after the novel view rendering. The method is designed as a standalone module compatible with any calibrated multi-camera system. For this we introduce a multi-view aware, transformer-based network architecture using spatio-temporal embeddings to ensure consistency across frames while preserving fine details. Additionally, our resolution-independent design allows adaptation to different camera setups, while an adaptive patch selection strategy balances inference speed and quality, allowing real-time performance. We evaluate our approach against state-of-the-art inpainting techniques under the same real-time constraints and demonstrate that our model achieves the best trade-off between quality and speed, outperforming competitors in both image and video-based metrics.

  </details>



- **CATNet: Collaborative Alignment and Transformation Network for Cooperative Perception**  
  Gong Chen, Chaokun Zhang, Tao Tang, Pengcheng Lv, Feng Li, Xin Xie  
  _2026-03-05_ · https://arxiv.org/abs/2603.05255v1  
  <details><summary>Abstract</summary>

  Cooperative perception significantly enhances scene understanding by integrating complementary information from diverse agents. However, existing research often overlooks critical challenges inherent in real-world multi-source data integration, specifically high temporal latency and multi-source noise. To address these practical limitations, we propose Collaborative Alignment and Transformation Network (CATNet), an adaptive compensation framework that resolves temporal latency and noise interference in multi-agent systems. Our key innovations can be summarized in three aspects. First, we introduce a Spatio-Temporal Recurrent Synchronization (STSync) that aligns asynchronous feature streams via adjacent-frame differential modeling, establishing a temporal-spatially unified representation space. Second, we design a Dual-Branch Wavelet Enhanced Denoiser (WTDen) that suppresses global noise and reconstructs localized feature distortions within aligned representations. Third, we construct an Adaptive Feature Selector (AdpSel) that dynamically focuses on critical perceptual features for robust fusion. Extensive experiments on multiple datasets demonstrate that CATNet consistently outperforms existing methods under complex traffic conditions, proving its superior robustness and adaptability.

  </details>



- **MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer**  
  Juntong Fang, Zequn Chen, Weiqi Zhang, Donglin Di, Xuancheng Zhang, Chengmin Yang, Yu-Shen Liu  
  _2026-03-05_ · https://arxiv.org/abs/2603.05078v1  
  <details><summary>Abstract</summary>

  Reconstructing dynamic 4D scenes remains challenging due to the presence of moving objects that corrupt camera pose estimation. Existing optimization methods alleviate this issue with additional supervision, but they are mostly computationally expensive and impractical in real-time applications. To address these limitations, we propose MoRe, a feedforward 4D reconstruction network that efficiently recovers dynamic 3D scenes from monocular videos. Built upon a strong static reconstruction backbone, MoRe employs an attention-forcing strategy to disentangle dynamic motion from static structure. To further enhance robustness, we fine-tune the model on large-scale, diverse datasets encompassing both dynamic and static scenes. Moreover, our grouped causal attention captures temporal dependencies and adapts to varying token lengths across frames, ensuring temporally coherent geometry reconstruction. Extensive experiments on multiple benchmarks demonstrate that MoRe achieves high-quality dynamic reconstructions with exceptional efficiency.

  </details>



- **DSA-SRGS: Super-Resolution Gaussian Splatting for Dynamic Sparse-View DSA Reconstruction**  
  Shiyu Zhang, Zhicong Wu, Huangxuan Zhao, Zhentao Liu, Lei Chen, Yong Luo, Lefei Zhang, Zhiming Cui, Ziwen Ke, Bo Du  
  _2026-03-05_ · https://arxiv.org/abs/2603.04770v1  
  <details><summary>Abstract</summary>

  Digital subtraction angiography (DSA) is a key imaging technique for the auxiliary diagnosis and treatment of cerebrovascular diseases. Recent advancements in gaussian splatting and dynamic neural representations have enabled robust 3D vessel reconstruction from sparse dynamic inputs. However, these methods are fundamentally constrained by the resolution of input projections, where performing naive upsampling to enhance rendering resolution inevitably results in severe blurring and aliasing artifacts. Such lack of super-resolution capability prevents the reconstructed 4D models from recovering fine-grained vascular details and intricate branching structures, which restricts their application in precision diagnosis and treatment. To solve this problem, this paper proposes DSA-SRGS, the first super-resolution gaussian splatting framework for dynamic sparse-view DSA reconstruction. Specifically, we introduce a Multi-Fidelity Texture Learning Module that integrates high-quality priors from a fine-tuned DSA-specific super-resolution model, into the 4D reconstruction optimization. To mitigate potential hallucination artifacts from pseudo-labels, this module employs a Confidence-Aware Strategy to adaptively weight supervision signals between the original low-resolution projections and the generated high-resolution pseudo-labels. Furthermore, we develop Radiative Sub-Pixel Densification, an adaptive strategy that leverages gradient accumulation from high-resolution sub-pixel sampling to refine the 4D radiative gaussian kernels. Extensive experiments on two clinical DSA datasets demonstrate that DSA-SRGS significantly outperforms state-of-the-art methods in both quantitative metrics and qualitative visual fidelity.

  </details>



- **Evaluating and Correcting Human Annotation Bias in Dynamic Micro-Expression Recognition**  
  Feng Liu, Bingyu Nan, Xuezhong Qian, Xiaolan Fu  
  _2026-03-05_ · https://arxiv.org/abs/2603.04766v1  
  <details><summary>Abstract</summary>

  Existing manual labeling of micro-expressions is subject to errors in accuracy, especially in cross-cultural scenarios where deviation in labeling of key frames is more prominent. To address this issue, this paper presents a novel Global Anti-Monotonic Differential Selection Strategy (GAMDSS) architecture for enhancing the effectiveness of spatio-temporal modeling of micro-expressions through keyframe re-selection. Specifically, the method identifies Onset and Apex frames, which are characterized by significant micro-expression variation, from complete micro-expression action sequences via a dynamic frame reselection mechanism. It then uses these to determine Offset frames and construct a rich spatio-temporal dynamic representation. A two-branch structure with shared parameters is then used to efficiently extract spatio-temporal features. Extensive experiments are conducted on seven widely recognized micro-expression datasets. The results demonstrate that GAMDSS effectively reduces subjective errors caused by human factors in multicultural datasets such as SAMM and 4DME. Furthermore, quantitative analyses confirm that offset-frame annotations in multicultural datasets are more uncertain, providing theoretical justification for standardizing micro-expression annotations. These findings directly support our argument for reconsidering the validity and generalizability of dataset annotation paradigms. Notably, this design can be integrated into existing models without increasing the number of parameters, offering a new approach to enhancing micro-expression recognition performance. The source code is available on GitHub[https://github.com/Cross-Innovation-Lab/GAMDSS].

  </details>



- **ArtHOI: Articulated Human-Object Interaction Synthesis by 4D Reconstruction from Video Priors**  
  Zihao Huang, Tianqi Liu, Zhaoxi Chen, Shaocong Xu, Saining Zhang, Lixing Xiao, Zhiguo Cao, Wei Li, Hao Zhao, Ziwei Liu  
  _2026-03-04_ · https://arxiv.org/abs/2603.04338v1  
  <details><summary>Abstract</summary>

  Synthesizing physically plausible articulated human-object interactions (HOI) without 3D/4D supervision remains a fundamental challenge. While recent zero-shot approaches leverage video diffusion models to synthesize human-object interactions, they are largely confined to rigid-object manipulation and lack explicit 4D geometric reasoning. To bridge this gap, we formulate articulated HOI synthesis as a 4D reconstruction problem from monocular video priors: given only a video generated by a diffusion model, we reconstruct a full 4D articulated scene without any 3D supervision. This reconstruction-based approach treats the generated 2D video as supervision for an inverse rendering problem, recovering geometrically consistent and physically plausible 4D scenes that naturally respect contact, articulation, and temporal coherence. We introduce ArtHOI, the first zero-shot framework for articulated human-object interaction synthesis via 4D reconstruction from video priors. Our key designs are: 1) Flow-based part segmentation: leveraging optical flow as a geometric cue to disentangle dynamic from static regions in monocular video; 2) Decoupled reconstruction pipeline: joint optimization of human motion and object articulation is unstable under monocular ambiguity, so we first recover object articulation, then synthesize human motion conditioned on the reconstructed object states. ArtHOI bridges video-based generation and geometry-aware reconstruction, producing interactions that are both semantically aligned and physically grounded. Across diverse articulated scenes (e.g., opening fridges, cabinets, microwaves), ArtHOI significantly outperforms prior methods in contact accuracy, penetration reduction, and articulation fidelity, extending zero-shot interaction synthesis beyond rigid manipulation through reconstruction-informed synthesis.

  </details>



- **CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video**  
  Lingen Li, Guangzhi Wang, Xiaoyu Li, Zhaoyang Zhang, Qi Dou, Jinwei Gu, Tianfan Xue, Ying Shan  
  _2026-03-04_ · https://arxiv.org/abs/2603.04291v1  
  <details><summary>Abstract</summary>

  Generating high-quality 360° panoramic videos from perspective input is one of the crucial applications for virtual reality (VR), whereby high-resolution videos are especially important for immersive experience. Existing methods are constrained by computational limitations of vanilla diffusion models, only supporting $\leq$ 1K resolution native generation and relying on suboptimal post super-resolution to increase resolution. We introduce CubeComposer, a novel spatio-temporal autoregressive diffusion model that natively generates 4K-resolution 360° videos. By decomposing videos into cubemap representations with six faces, CubeComposer autoregressively synthesizes content in a well-planned spatio-temporal order, reducing memory demands while enabling high-resolution output. Specifically, to address challenges in multi-dimensional autoregression, we propose: (1) a spatio-temporal autoregressive strategy that orchestrates 360° video generation across cube faces and time windows for coherent synthesis; (2) a cube face context management mechanism, equipped with a sparse context attention design to improve efficiency; and (3) continuity-aware techniques, including cube-aware positional encoding, padding, and blending to eliminate boundary seams. Extensive experiments on benchmark datasets demonstrate that CubeComposer outperforms state-of-the-art methods in native resolution and visual quality, supporting practical VR application scenarios. Project page: https://lg-li.github.io/project/cubecomposer

  </details>



- **A Baseline Study and Benchmark for Few-Shot Open-Set Action Recognition with Feature Residual Discrimination**  
  Stefano Berti, Giulia Pasquale, Lorenzo Natale  
  _2026-03-04_ · https://arxiv.org/abs/2603.04125v1  
  <details><summary>Abstract</summary>

  Few-Shot Action Recognition (FS-AR) has shown promising results but is often limited by a closed-set assumption that fails in real-world open-set scenarios. While Few-Shot Open-Set (FSOS) recognition is well-established for images, its extension to spatio-temporal video data remains underexplored. To address this, we propose an architectural extension based on a Feature-Residual Discriminator (FR-Disc), adapting previous work on skeletal data to the more complex video domain. Extensive experiments on five datasets demonstrate that while common open-set techniques provide only marginal gains, our FR-Disc significantly enhances unknown rejection capabilities without compromising closed-set accuracy, setting a new state-of-the-art for FSOS-AR. The project website, code, and benchmark are available at: https://hsp-iit.github.io/fsosar/.

  </details>



- **Spatial Causal Prediction in Video**  
  Yanguang Zhao, Jie Yang, Shengqiong Wu, Shutong Hu, Hongbo Qiu, Yu Wang, Guijia Zhang, Tan Kai Ze, Hao Fei, Chia-Wen Lin, et al.  
  _2026-03-04_ · https://arxiv.org/abs/2603.03944v1  
  <details><summary>Abstract</summary>

  Spatial reasoning, the ability to understand spatial relations, causality, and dynamic evolution, is central to human intelligence and essential for real-world applications such as autonomous driving and robotics. Existing studies, however, primarily assess models on visible spatio-temporal understanding, overlooking their ability to infer unseen past or future spatial states. In this work, we introduce Spatial Causal Prediction (SCP), a new task paradigm that challenges models to reason beyond observation and predict spatial causal outcomes. We further construct SCP-Bench, a benchmark comprising 2,500 QA pairs across 1,181 videos spanning diverse viewpoints, scenes, and causal directions, to support systematic evaluation. Through comprehensive experiments on {23} state-of-the-art models, we reveal substantial gaps between human and model performance, limited temporal extrapolation, and weak causal grounding. We further analyze key factors influencing performance and propose perception-enhancement and reasoning-guided strategies toward advancing spatial causal intelligence. The project page is https://guangstrip.github.io/SCP-Bench.

  </details>



- **LiDAR Prompted Spatio-Temporal Multi-View Stereo for Autonomous Driving**  
  Qihao Sun, Jiarun Liu, Ziqian Ni, Jianyun Xu, Tao Xie, Lijun Zhao, Ruifeng Li, Sheng Yang  
  _2026-03-04_ · https://arxiv.org/abs/2603.03765v1  
  <details><summary>Abstract</summary>

  Accurate metric depth is critical for autonomous driving perception and simulation, yet current approaches struggle to achieve high metric accuracy, multi-view and temporal consistency, and cross-domain generalization. To address these challenges, we present DriveMVS, a novel multi-view stereo framework that reconciles these competing objectives through two key insights: (1) Sparse but metrically accurate LiDAR observations can serve as geometric prompts to anchor depth estimation in absolute scale, and (2) deep fusion of diverse cues is essential for resolving ambiguities and enhancing robustness, while a spatio-temporal decoder ensures consistency across frames. Built upon these principles, DriveMVS embeds the LiDAR prompt in two ways: as a hard geometric prior that anchors the cost volume, and as soft feature-wise guidance fused by a triple-cue combiner. Regarding temporal consistency, DriveMVS employs a spatio-temporal decoder that jointly leverages geometric cues from the MVS cost volume and temporal context from neighboring frames. Experiments show that DriveMVS achieves state-of-the-art performance on multiple benchmarks, excelling in metric accuracy, temporal stability, and zero-shot cross-domain transfer, demonstrating its practical value for scalable, reliable autonomous driving systems.

  </details>



- **InfinityStory: Unlimited Video Generation with World Consistency and Character-Aware Shot Transitions**  
  Mohamed Elmoghany, Liangbing Zhao, Xiaoqian Shen, Subhojyoti Mukherjee, Yang Zhou, Gang Wu, Viet Dac Lai, Seunghyun Yoon, Ryan Rossi, Abdullah Rashwan, et al.  
  _2026-03-04_ · https://arxiv.org/abs/2603.03646v1  
  <details><summary>Abstract</summary>

  Generating long-form storytelling videos with consistent visual narratives remains a significant challenge in video synthesis. We present a novel framework, dataset, and a model that address three critical limitations: background consistency across shots, seamless multi-subject shot-to-shot transitions, and scalability to hour-long narratives. Our approach introduces a background-consistent generation pipeline that maintains visual coherence across scenes while preserving character identity and spatial relationships. We further propose a transition-aware video synthesis module that generates smooth shot transitions for complex scenarios involving multiple subjects entering or exiting frames, going beyond the single-subject limitations of prior work. To support this, we contribute with a synthetic dataset of 10,000 multi-subject transition sequences covering underrepresented dynamic scene compositions. On VBench, InfinityStory achieves the highest Background Consistency (88.94), highest Subject Consistency (82.11), and the best overall average rank (2.80), showing improved stability, smoother transitions, and better temporal coherence.

  </details>



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


