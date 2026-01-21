# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-01-21 06:54 UTC_

Total papers shown: **8**


---

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



- **Optical Linear Systems Framework for Event Sensing and Computational Neuromorphic Imaging**  
  Nimrod Kruger, Nicholas Owen Ralph, Gregory Cohen, Paul Hurley  
  _2026-01-20_ · https://arxiv.org/abs/2601.13498v1  
  <details><summary>Abstract</summary>

  Event vision sensors (neuromorphic cameras) output sparse, asynchronous ON/OFF events triggered by log-intensity threshold crossings, enabling microsecond-scale sensing with high dynamic range and low data bandwidth. As a nonlinear system, this event representation does not readily integrate with the linear forward models that underpin most computational imaging and optical system design. We present a physics-grounded processing pipeline that maps event streams to estimates of per-pixel log-intensity and intensity derivatives, and embeds these measurements in a dynamic linear systems model with a time-varying point spread function. This enables inverse filtering directly from event data, using frequency-domain Wiener deconvolution with a known (or parameterised) dynamic transfer function. We validate the approach in simulation for single and overlapping point sources under modulated defocus, and on real event data from a tunable-focus telescope imaging a star field, demonstrating source localisation and separability. The proposed framework provides a practical bridge between event sensing and model-based computational imaging for dynamic optical systems.

  </details>



- **AsyncBEV: Cross-modal Flow Alignment in Asynchronous 3D Object Detection**  
  Shiming Wang, Holger Caesar, Liangliang Nan, Julian F. P. Kooij  
  _2026-01-19_ · https://arxiv.org/abs/2601.12994v1  
  <details><summary>Abstract</summary>

  In autonomous driving, multi-modal perception tasks like 3D object detection typically rely on well-synchronized sensors, both at training and inference. However, despite the use of hardware- or software-based synchronization algorithms, perfect synchrony is rarely guaranteed: Sensors may operate at different frequencies, and real-world factors such as network latency, hardware failures, or processing bottlenecks often introduce time offsets between sensors. Such asynchrony degrades perception performance, especially for dynamic objects. To address this challenge, we propose AsyncBEV, a trainable lightweight and generic module to improve the robustness of 3D Birds' Eye View (BEV) object detection models against sensor asynchrony. Inspired by scene flow estimation, AsyncBEV first estimates the 2D flow from the BEV features of two different sensor modalities, taking into account the known time offset between these sensor measurements. The predicted feature flow is then used to warp and spatially align the feature maps, which we show can easily be integrated into different current BEV detector architectures (e.g., BEV grid-based and token-based). Extensive experiments demonstrate AsyncBEV improves robustness against both small and large asynchrony between LiDAR or camera sensors in both the token-based CMT and grid-based UniBEV, especially for dynamic objects. We significantly outperform the ego motion compensated CMT and UniBEV baselines, notably by $16.6$ % and $11.9$ % NDS on dynamic objects in the worst-case scenario of a $0.5 s$ time offset. Code will be released upon acceptance.

  </details>



- **Linear Mechanisms for Spatiotemporal Reasoning in Vision Language Models**  
  Raphi Kang, Hongqiao Chen, Georgia Gkioxari, Pietro Perona  
  _2026-01-18_ · https://arxiv.org/abs/2601.12626v1  
  <details><summary>Abstract</summary>

  Spatio-temporal reasoning is a remarkable capability of Vision Language Models (VLMs), but the underlying mechanisms of such abilities remain largely opaque. We postulate that visual/geometrical and textual representations of spatial structure must be combined at some point in VLM computations. We search for such confluence, and ask whether the identified representation can causally explain aspects of input-output model behavior through a linear model. We show empirically that VLMs encode object locations by linearly binding \textit{spatial IDs} to textual activations, then perform reasoning via language tokens. Through rigorous causal interventions we demonstrate that these IDs, which are ubiquitous across the model, can systematically mediate model beliefs at intermediate VLM layers. Additionally, we find that spatial IDs serve as a diagnostic tool for identifying limitations in existing VLMs, and as a valuable learning signal. We extend our analysis to video VLMs and identify an analogous linear temporal ID mechanism. By characterizing our proposed spatiotemporal ID mechanism, we elucidate a previously underexplored internal reasoning process in VLMs, toward improved interpretability and the principled design of more aligned and capable models. We release our code for reproducibility: https://github.com/Raphoo/linear-mech-vlms.

  </details>



- **CurConMix+: A Unified Spatio-Temporal Framework for Hierarchical Surgical Workflow Understanding**  
  Yongjun Jeon, Jongmin Shin, Kanggil Park, Seonmin Park, Soyoung Lim, Jung Yong Kim, Jinsoo Rhu, Jongman Kim, Gyu-Seong Choi, Namkee Oh, et al.  
  _2026-01-18_ · https://arxiv.org/abs/2601.12312v1  
  <details><summary>Abstract</summary>

  Surgical action triplet recognition aims to understand fine-grained surgical behaviors by modeling the interactions among instruments, actions, and anatomical targets. Despite its clinical importance for workflow analysis and skill assessment, progress has been hindered by severe class imbalance, subtle visual variations, and the semantic interdependence among triplet components. Existing approaches often address only a subset of these challenges rather than tackling them jointly, which limits their ability to form a holistic understanding. This study builds upon CurConMix, a spatial representation framework. At its core, a curriculum-guided contrastive learning strategy learns discriminative and progressively correlated features, further enhanced by structured hard-pair sampling and feature-level mixup. Its temporal extension, CurConMix+, integrates a Multi-Resolution Temporal Transformer (MRTT) that achieves robust, context-aware understanding by adaptively fusing multi-scale temporal features and dynamically balancing spatio-temporal cues. Furthermore, we introduce LLS48, a new, hierarchically annotated benchmark for complex laparoscopic left lateral sectionectomy, providing step-, task-, and action-level annotations. Extensive experiments on CholecT45 and LLS48 demonstrate that CurConMix+ not only outperforms state-of-the-art approaches in triplet recognition, but also exhibits strong cross-level generalization, as its fine-grained features effectively transfer to higher-level phase and step recognition tasks. Together, the framework and dataset provide a unified foundation for hierarchy-aware, reproducible, and interpretable surgical workflow understanding. The code and dataset will be publicly released on GitHub to facilitate reproducibility and further research.

  </details>


