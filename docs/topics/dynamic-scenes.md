# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-28 07:11 UTC_

Total papers shown: **10**


---

- **UNIC: Neural Garment Deformation Field for Real-time Clothed Character Animation**  
  Chengfeng Zhao, Junbo Qi, Yulou Liu, Zhiyang Dou, Minchen Li, Taku Komura, Ziwei Liu, Wenping Wang, Yuan Liu  
  _2026-03-26_ · https://arxiv.org/abs/2603.25580v1  
  <details><summary>Abstract</summary>

  Simulating physically realistic garment deformations is an essential task for virtual immersive experience, which is often achieved by physics simulation methods. However, these methods are typically time-consuming, computationally demanding, and require costly hardware, which is not suitable for real-time applications. Recent learning-based methods tried to resolve this problem by training graph neural networks to learn the garment deformation on vertices, which, however, fail to capture the intricate deformation of complex garment meshes with complex topologies. In this paper, we introduce a novel neural deformation field-based method, named UNIC, to animate the garments of an avatar in real time, given the motion sequences. Our key idea is to learn the instance-specific neural deformation field to animate the garment meshes. Such an instance-specific learning scheme does not require UNIC to generalize to new garments but only to new motion sequences, which greatly reduces the difficulty in training and improves the deformation quality. Moreover, neural deformation fields map the 3D points to their deformation offsets, which not only avoids handling topologies of the complex garments but also injects a natural smoothness constraint in the deformation learning. Extensive experiments have been conducted on various kinds of garment meshes to demonstrate the effectiveness and efficiency of UNIC over baseline methods, making it potentially practical and useful in real-world interactive applications like video games.

  </details>



- **Challenges in Hyperspectral Imaging for Autonomous Driving: The HSI-Drive Case**  
  Koldo Basterretxea, Jon Gutiérrez-Zaballa, Javier Echanobe  
  _2026-03-26_ · https://arxiv.org/abs/2603.25510v1  
  <details><summary>Abstract</summary>

  The use of hyperspectral imaging (HSI) in autonomous driving (AD), while promising, faces many challenges related to the specifics and requirements of this application domain. On the one hand, non-controlled and variable lighting conditions, the wide depth-of-field ranges, and dynamic scenes with fast-moving objects. On the other hand, the requirements for real-time operation and the limited computational resources of embedded platforms. The combination of these factors determines both the criteria for selecting appropriate HSI technologies and the development of custom vision algorithms that leverage the spectral and spatial information obtained from the sensors. In this article, we analyse several techniques explored in the research of HSI-based vision systems with application to AD, using as an example results obtained from experiments using data from the most recent version of the HSI-Drive dataset.

  </details>



- **LaMP: Learning Vision-Language-Action Policies with 3D Scene Flow as Latent Motion Prior**  
  Xinkai Wang, Chenyi Wang, Yifu Xu, Mingzhe Ye, Fu-Cheng Zhang, Jialin Tian, Xinyu Zhan, Lifeng Zhu, Cewu Lu, Lixin Yang  
  _2026-03-26_ · https://arxiv.org/abs/2603.25399v1  
  <details><summary>Abstract</summary>

  We introduce \textbf{LaMP}, a dual-expert Vision-Language-Action framework that embeds dense 3D scene flow as a latent motion prior for robotic manipulation. Existing VLA models regress actions directly from 2D semantic visual features, forcing them to learn complex 3D physical interactions implicitly. This implicit learning strategy degrades under unfamiliar spatial dynamics. LaMP addresses this limitation by aligning a flow-matching \emph{Motion Expert} with a policy-predicting \emph{Action Expert} through gated cross-attention. Specifically, the Motion Expert generates a one-step partially denoised 3D scene flow, and its hidden states condition the Action Expert without full multi-step reconstruction. We evaluate LaMP on the LIBERO, LIBERO-Plus, and SimplerEnv-WidowX simulation benchmarks as well as real-world experiments. LaMP consistently outperforms evaluated VLA baselines across LIBERO, LIBERO-Plus, and SimplerEnv-WidowX benchmarks, achieving the highest reported average success rates under the same training budgets. On LIBERO-Plus OOD perturbations, LaMP shows improved robustness with an average 9.7% gain over the strongest prior baseline. Our project page is available at https://summerwxk.github.io/lamp-project-page/.

  </details>



- **MoRGS: Efficient Per-Gaussian Motion Reasoning for Streamable Dynamic 3D Scenes**  
  Wonjoon Lee, Sungmin Woo, Donghyeong Kim, Jungho Lee, Sangheon Park, Sangyoun Lee  
  _2026-03-26_ · https://arxiv.org/abs/2603.25042v1  
  <details><summary>Abstract</summary>

  Online reconstruction of dynamic scenes aims to learn from streaming multi-view inputs under low-latency constraints. The fast training and real-time rendering capabilities of 3D Gaussian Splatting have made on-the-fly reconstruction practically feasible, enabling online 4D reconstruction. However, existing online approaches, despite their efficiency and visual quality, fail to learn per-Gaussian motion that reflects true scene dynamics. Without explicit motion cues, appearance and motion are optimized solely under photometric loss, causing per-Gaussian motion to chase pixel residuals rather than true 3D motion. To address this, we propose MoRGS, an efficient online per-Gaussian motion reasoning framework that explicitly models per-Gaussian motion to improve 4D reconstruction quality. Specifically, we leverage optical flow on a sparse set of key views as lightweight motion cues that regularize per-Gaussian motion beyond photometric supervision. To compensate for the sparsity of flow supervision, we learn a per-Gaussian motion offset field that reconciles discrepancies between projected 3D motion and observed flow across views and time. In addition, we introduce a per-Gaussian motion confidence that separates dynamic from static Gaussians and weights Gaussian attribute residual updates, thereby suppressing redundant motion in static regions for better temporal consistency and accelerating the modeling of large motions. Extensive experiments demonstrate that MoRGS achieves state-of-the-art reconstruction quality and motion fidelity among online methods, while maintaining streamable performance.

  </details>



- **Infinite Gaze Generation for Videos with Autoregressive Diffusion**  
  Jenna Kang, Colin Groth, Tong Wu, Finley Torrens, Patsorn Sangkloy, Gordon Wetzstein, Qi Sun  
  _2026-03-26_ · https://arxiv.org/abs/2603.24938v1  
  <details><summary>Abstract</summary>

  Predicting human gaze in video is fundamental to advancing scene understanding and multimodal interaction. While traditional saliency maps provide spatial probability distributions and scanpaths offer ordered fixations, both abstractions often collapse the fine-grained temporal dynamics of raw gaze. Furthermore, existing models are typically constrained to short-term windows ($\approx$ 3-5s), failing to capture the long-range behavioral dependencies inherent in real-world content. We present a generative framework for infinite-horizon raw gaze prediction in videos of arbitrary length. By leveraging an autoregressive diffusion model, we synthesize gaze trajectories characterized by continuous spatial coordinates and high-resolution timestamps. Our model is conditioned on a saliency-aware visual latent space. Quantitative and qualitative evaluations demonstrate that our approach significantly outperforms existing approaches in long-range spatio-temporal accuracy and trajectory realism.

  </details>



- **TIGFlow-GRPO: Trajectory Forecasting via Interaction-Aware Flow Matching and Reward-Driven Optimization**  
  Xuepeng Jing, Wenhuan Lu, Hao Meng, Zhizhi Yu, Jianguo Wei  
  _2026-03-26_ · https://arxiv.org/abs/2603.24936v1  
  <details><summary>Abstract</summary>

  Human trajectory forecasting is important for intelligent multimedia systems operating in visually complex environments, such as autonomous driving and crowd surveillance. Although Conditional Flow Matching (CFM) has shown strong ability in modeling trajectory distributions from spatio-temporal observations, existing approaches still focus primarily on supervised fitting, which may leave social norms and scene constraints insufficiently reflected in generated trajectories. To address this issue, we propose TIGFlow-GRPO, a two-stage generative framework that aligns flow-based trajectory generation with behavioral rules. In the first stage, we build a CFM-based predictor with a Trajectory-Interaction-Graph (TIG) module to model fine-grained visual-spatial interactions and strengthen context encoding. This stage captures both agent-agent and agent-scene relations more effectively, providing more informative conditional features for subsequent alignment. In the second stage, we perform Flow-GRPO post-training,where deterministic flow rollout is reformulated as stochastic ODE-to-SDE sampling to enable trajectory exploration, and a composite reward combines view-aware social compliance with map-aware physical feasibility. By evaluating trajectories explored through SDE rollout, GRPO progressively steers multimodal predictions toward behaviorally plausible futures. Experiments on the ETH/UCY and SDD datasets show that TIGFlow-GRPO improves forecasting accuracy and long-horizon stability while generating trajectories that are more socially compliant and physically feasible. These results suggest that the proposed framework provides an effective way to connect flow-based trajectory modeling with behavior-aware alignment in dynamic multimedia environments.

  </details>



- **DRoPS: Dynamic 3D Reconstruction of Pre-Scanned Objects**  
  Narek Tumanyan, Samuel Rota Bulò, Denis Rozumny, Lorenzo Porzi, Adam Harley, Tali Dekel, Peter Kontschieder, Jonathon Luiten  
  _2026-03-25_ · https://arxiv.org/abs/2603.24770v1  
  <details><summary>Abstract</summary>

  Dynamic scene reconstruction from casual videos has seen recent remarkable progress. Numerous approaches have attempted to overcome the ill-posedness of the task by distilling priors from 2D foundational models and by imposing hand-crafted regularization on the optimized motion. However, these methods struggle to reconstruct scenes from extreme novel viewpoints, especially when highly articulated motions are present. In this paper, we present DRoPS, a novel approach that leverages a static pre-scan of the dynamic object as an explicit geometric and appearance prior. While existing state-of-the-art methods fail to fully exploit the pre-scan, DRoPS leverages our novel setup to effectively constrain the solution space and ensure geometrical consistency throughout the sequence. The core of our novelty is twofold: first, we establish a grid-structured and surface-aligned model by organizing Gaussian primitives into pixel grids anchored to the object surface. Second, by leveraging the grid structure of our primitives, we parameterize motion using a CNN conditioned on those grids, injecting strong implicit regularization and correlating the motion of nearby points. Extensive experiments demonstrate that our method significantly outperforms the current state of the art in rendering quality and 3D tracking accuracy.

  </details>



- **Accelerating Diffusion-based Video Editing via Heterogeneous Caching: Beyond Full Computing at Sampled Denoising Timestep**  
  Tianyi Liu, Ye Lu, Linfeng Zhang, Chen Cai, Jianjun Gao, Yi Wang, Kim-Hui Yap, Lap-Pui Chau  
  _2026-03-25_ · https://arxiv.org/abs/2603.24260v1  
  <details><summary>Abstract</summary>

  Diffusion-based video editing has emerged as an important paradigm for high-quality and flexible content generation. However, despite their generality and strong modeling capacity, Diffusion Transformers (DiT) remain computationally expensive due to the iterative denoising process, posing challenges for practical deployment. Existing video diffusion acceleration methods primarily exploit denoising timestep-level feature reuse, which mitigates the redundancy in denoising process, but overlooks the architectural redundancy within the DiT that many attention operations over spatio-temporal tokens are redundantly executed, offering little to no incremental contribution to the model output. This work introduces HetCache, a training-free diffusion acceleration framework designed to exploit the inherent heterogeneity in diffusion-based masked video-to-video (MV2V) generation and editing. Instead of uniformly reuse or randomly sampling tokens, HetCache assesses the contextual relevance and interaction strength among various types of tokens in designated computing steps. Guided by spatial priors, it divides the spatial-temporal tokens in DiT model into context and generative tokens, and selectively caches the context tokens that exhibit the strongest correlation and most representative semantics with generative ones. This strategy reduces redundant attention operations while maintaining editing consistency and fidelity. Experiments show that HetCache achieves a noticeable acceleration, including a 2.67$\times$ latency speedup and FLOPs reduction over commonly used foundation models, with negligible degradation in editing quality.

  </details>



- **Spectral Scalpel: Amplifying Adjacent Action Discrepancy via Frequency-Selective Filtering for Skeleton-Based Action Segmentation**  
  Haoyu Ji, Bowen Chen, Zhihao Yang, Wenze Huang, Yu Gao, Xueting Liu, Weihong Ren, Zhiyong Wang, Honghai Liu  
  _2026-03-25_ · https://arxiv.org/abs/2603.24134v1  
  <details><summary>Abstract</summary>

  Skeleton-based Temporal Action Segmentation (STAS) seeks to densely segment and classify diverse actions within long, untrimmed skeletal motion sequences. However, existing STAS methodologies face challenges of limited inter-class discriminability and blurred segmentation boundaries, primarily due to insufficient distinction of spatio-temporal patterns between adjacent actions. To address these limitations, we propose Spectral Scalpel, a frequency-selective filtering framework aimed at suppressing shared frequency components between adjacent distinct actions while amplifying their action-specific frequencies, thereby enhancing inter-action discrepancies and sharpening transition boundaries. Specifically, Spectral Scalpel employs adaptive multi-scale spectral filters as scalpels to edit frequency spectra, coupled with a discrepancy loss between adjacent actions serving as the surgical objective. This design amplifies representational disparities between neighboring actions, effectively mitigating boundary localization ambiguities and inter-class confusion. Furthermore, complementing long-term temporal modeling, we introduce a frequency-aware channel mixer to strengthen channel evolution by aggregating spectra across channels. This work presents a novel paradigm for STAS that extends conventional spatio-temporal modeling by incorporating frequency-domain analysis. Extensive experiments on five public datasets demonstrate that Spectral Scalpel achieves state-of-the-art performance. Code is available at https://github.com/HaoyuJi/SpecScalpel.

  </details>



- **LaDy: Lagrangian-Dynamic Informed Network for Skeleton-based Action Segmentation via Spatial-Temporal Modulation**  
  Haoyu Ji, Xueting Liu, Yu Gao, Wenze Huang, Zhihao Yang, Weihong Ren, Zhiyong Wang, Honghai Liu  
  _2026-03-25_ · https://arxiv.org/abs/2603.24097v1  
  <details><summary>Abstract</summary>

  Skeleton-based Temporal Action Segmentation (STAS) aims to densely parse untrimmed skeletal sequences into frame-level action categories. However, existing methods, while proficient at capturing spatio-temporal kinematics, neglect the underlying physical dynamics that govern human motion. This oversight limits inter-class discriminability between actions with similar kinematics but distinct dynamic intents, and hinders precise boundary localization where dynamic force profiles shift. To address these, we propose the Lagrangian-Dynamic Informed Network (LaDy), a framework integrating principles of Lagrangian dynamics into the segmentation process. Specifically, LaDy first computes generalized coordinates from joint positions and then estimates Lagrangian terms under physical constraints to explicitly synthesize the generalized forces. To further ensure physical coherence, our Energy Consistency Loss enforces the work-energy theorem, aligning kinetic energy change with the work done by the net force. The learned dynamics then drive a Spatio-Temporal Modulation module: Spatially, generalized forces are fused with spatial representations to provide more discriminative semantics. Temporally, salient dynamic signals are constructed for temporal gating, thereby significantly enhancing boundary awareness. Experiments on challenging datasets show that LaDy achieves state-of-the-art performance, validating the integration of physical dynamics for action segmentation. Code is available at https://github.com/HaoyuJi/LaDy.

  </details>


