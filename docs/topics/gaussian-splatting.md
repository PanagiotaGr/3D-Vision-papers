# Gaussian Splatting & 3DGS

_Updated: 2026-02-03 07:08 UTC_

Total papers shown: **13**


---

- **SoMA: A Real-to-Sim Neural Simulator for Robotic Soft-body Manipulation**  
  Mu Huang, Hui Wang, Kerui Ren, Linning Xu, Yunsong Zhou, Mulin Yu, Bo Dai, Jiangmiao Pang  
  _2026-02-02_ · https://arxiv.org/abs/2602.02402v1  
  <details><summary>Abstract</summary>

  Simulating deformable objects under rich interactions remains a fundamental challenge for real-to-sim robot manipulation, with dynamics jointly driven by environmental effects and robot actions. Existing simulators rely on predefined physics or data-driven dynamics without robot-conditioned control, limiting accuracy, stability, and generalization. This paper presents SoMA, a 3D Gaussian Splat simulator for soft-body manipulation. SoMA couples deformable dynamics, environmental forces, and robot joint actions in a unified latent neural space for end-to-end real-to-sim simulation. Modeling interactions over learned Gaussian splats enables controllable, stable long-horizon manipulation and generalization beyond observed trajectories without predefined physical models. SoMA improves resimulation accuracy and generalization on real-world robot manipulation by 20%, enabling stable simulation of complex tasks such as long-horizon cloth folding.

  </details>



- **Uncertainty-Aware Image Classification In Biomedical Imaging Using Spectral-normalized Neural Gaussian Processes**  
  Uma Meleti, Jeffrey J. Nirschl  
  _2026-02-02_ · https://arxiv.org/abs/2602.02370v1  
  <details><summary>Abstract</summary>

  Accurate histopathologic interpretation is key for clinical decision-making; however, current deep learning models for digital pathology are often overconfident and poorly calibrated in out-of-distribution (OOD) settings, which limit trust and clinical adoption. Safety-critical medical imaging workflows benefit from intrinsic uncertainty-aware properties that can accurately reject OOD input. We implement the Spectral-normalized Neural Gaussian Process (SNGP), a set of lightweight modifications that apply spectral normalization and replace the final dense layer with a Gaussian process layer to improve single-model uncertainty estimation and OOD detection. We evaluate SNGP vs. deterministic and MonteCarlo dropout on six datasets across three biomedical classification tasks: white blood cells, amyloid plaques, and colorectal histopathology. SNGP has comparable in-distribution performance while significantly improving uncertainty estimation and OOD detection. Thus, SNGP or related models offer a useful framework for uncertainty-aware classification in digital pathology, supporting safe deployment and building trust with pathologists.

  </details>



- **UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction**  
  Changbai Li, Haodong Zhu, Hanlin Chen, Xiuping Liang, Tongfei Chen, Shuwei Shao, Linlin Yang, Huobin Tan, Baochang Zhang  
  _2026-02-02_ · https://arxiv.org/abs/2602.02089v1  
  <details><summary>Abstract</summary>

  While 3D Gaussian Splatting (3DGS) enables high-quality, real-time rendering for bounded scenes, its extension to large-scale urban environments gives rise to critical challenges in terms of geometric consistency, memory efficiency, and computational scalability. To address these issues, we present UrbanGS, a scalable reconstruction framework that effectively tackles these challenges for city-scale applications. First, we propose a Depth-Consistent D-Normal Regularization module. Unlike existing approaches that rely solely on monocular normal estimators, which can effectively update rotation parameters yet struggle to update position parameters, our method integrates D-Normal constraints with external depth supervision. This allows for comprehensive updates of all geometric parameters. By further incorporating an adaptive confidence weighting mechanism based on gradient consistency and inverse depth deviation, our approach significantly enhances multi-view depth alignment and geometric coherence, which effectively resolves the issue of geometric accuracy in complex large-scale scenes. To improve scalability, we introduce a Spatially Adaptive Gaussian Pruning (SAGP) strategy, which dynamically adjusts Gaussian density based on local geometric complexity and visibility to reduce redundancy. Additionally, a unified partitioning and view assignment scheme is designed to eliminate boundary artifacts and optimize computational load. Extensive experiments on multiple urban datasets demonstrate that UrbanGS achieves superior performance in rendering quality, geometric accuracy, and memory efficiency, providing a systematic solution for high-fidelity large-scale scene reconstruction.

  </details>



- **SurfSplat: Conquering Feedforward 2D Gaussian Splatting with Surface Continuity Priors**  
  Bing He, Jingnan Gao, Yunuo Chen, Ning Cao, Gang Chen, Zhengxue Cheng, Li Song, Wenjun Zhang  
  _2026-02-02_ · https://arxiv.org/abs/2602.02000v1  
  <details><summary>Abstract</summary>

  Reconstructing 3D scenes from sparse images remains a challenging task due to the difficulty of recovering accurate geometry and texture without optimization. Recent approaches leverage generalizable models to generate 3D scenes using 3D Gaussian Splatting (3DGS) primitive. However, they often fail to produce continuous surfaces and instead yield discrete, color-biased point clouds that appear plausible at normal resolution but reveal severe artifacts under close-up views. To address this issue, we present SurfSplat, a feedforward framework based on 2D Gaussian Splatting (2DGS) primitive, which provides stronger anisotropy and higher geometric precision. By incorporating a surface continuity prior and a forced alpha blending strategy, SurfSplat reconstructs coherent geometry together with faithful textures. Furthermore, we introduce High-Resolution Rendering Consistency (HRRC), a new evaluation metric designed to evaluate high-resolution reconstruction quality. Extensive experiments on RealEstate10K, DL3DV, and ScanNet demonstrate that SurfSplat consistently outperforms prior methods on both standard metrics and HRRC, establishing a robust solution for high-fidelity 3D reconstruction from sparse inputs. Project page: https://hebing-sjtu.github.io/SurfSplat-website/

  </details>



- **ProxyImg: Towards Highly-Controllable Image Representation via Hierarchical Disentangled Proxy Embedding**  
  Ye Chen, Yupeng Zhu, Xiongzhen Zhang, Zhewen Wan, Yingzhe Li, Wenjun Zhang, Bingbing Ni  
  _2026-02-02_ · https://arxiv.org/abs/2602.01881v1  
  <details><summary>Abstract</summary>

  Prevailing image representation methods, including explicit representations such as raster images and Gaussian primitives, as well as implicit representations such as latent images, either suffer from representation redundancy that leads to heavy manual editing effort, or lack a direct mapping from latent variables to semantic instances or parts, making fine-grained manipulation difficult. These limitations hinder efficient and controllable image and video editing. To address these issues, we propose a hierarchical proxy-based parametric image representation that disentangles semantic, geometric, and textural attributes into independent and manipulable parameter spaces. Based on a semantic-aware decomposition of the input image, our representation constructs hierarchical proxy geometries through adaptive Bezier fitting and iterative internal region subdivision and meshing. Multi-scale implicit texture parameters are embedded into the resulting geometry-aware distributed proxy nodes, enabling continuous high-fidelity reconstruction in the pixel domain and instance- or part-independent semantic editing. In addition, we introduce a locality-adaptive feature indexing mechanism to ensure spatial texture coherence, which further supports high-quality background completion without relying on generative models. Extensive experiments on image reconstruction and editing benchmarks, including ImageNet, OIR-Bench, and HumanEdit, demonstrate that our method achieves state-of-the-art rendering fidelity with significantly fewer parameters, while enabling intuitive, interactive, and physically plausible manipulation. Moreover, by integrating proxy nodes with Position-Based Dynamics, our framework supports real-time physics-driven animation using lightweight implicit rendering, achieving superior temporal consistency and visual realism compared with generative approaches.

  </details>



- **CloDS: Visual-Only Unsupervised Cloth Dynamics Learning in Unknown Conditions**  
  Yuliang Zhan, Jian Li, Wenbing Huang, Wenbing Huang, Yang Liu, Hao Sun  
  _2026-02-02_ · https://arxiv.org/abs/2602.01844v1  
  <details><summary>Abstract</summary>

  Deep learning has demonstrated remarkable capabilities in simulating complex dynamic systems. However, existing methods require known physical properties as supervision or inputs, limiting their applicability under unknown conditions. To explore this challenge, we introduce Cloth Dynamics Grounding (CDG), a novel scenario for unsupervised learning of cloth dynamics from multi-view visual observations. We further propose Cloth Dynamics Splatting (CloDS), an unsupervised dynamic learning framework designed for CDG. CloDS adopts a three-stage pipeline that first performs video-to-geometry grounding and then trains a dynamics model on the grounded meshes. To cope with large non-linear deformations and severe self-occlusions during grounding, we introduce a dual-position opacity modulation that supports bidirectional mapping between 2D observations and 3D geometry via mesh-based Gaussian splatting in video-to-geometry grounding stage. It jointly considers the absolute and relative position of Gaussian components. Comprehensive experimental evaluations demonstrate that CloDS effectively learns cloth dynamics from visual data while maintaining strong generalization capabilities for unseen configurations. Our code is available at https://github.com/whynot-zyl/CloDS. Visualization results are available at https://github.com/whynot-zyl/CloDS_video}.%\footnote{As in this example.

  </details>



- **OFERA: Blendshape-driven 3D Gaussian Control for Occluded Facial Expression to Realistic Avatars in VR**  
  Seokhwan Yang, Boram Yoon, Seoyoung Kang, Hail Song, Woontack Woo  
  _2026-02-02_ · https://arxiv.org/abs/2602.01748v1  
  <details><summary>Abstract</summary>

  We propose OFERA, a novel framework for real-time expression control of photorealistic Gaussian head avatars for VR headset users. Existing approaches attempt to recover occluded facial expressions using additional sensors or internal cameras, but sensor-based methods increase device weight and discomfort, while camera-based methods raise privacy concerns and suffer from limited access to raw data. To overcome these limitations, we leverage the blendshape signals provided by commercial VR headsets as expression inputs. Our framework consists of three key components: (1) Blendshape Distribution Alignment (BDA), which applies linear regression to align the headset-provided blendshape distribution to a canonical input space; (2) an Expression Parameter Mapper (EPM) that maps the aligned blendshape signals into an expression parameter space for controlling Gaussian head avatars; and (3) a Mapper-integrated Avatar (MiA) that incorporates EPM into the avatar learning process to ensure distributional consistency. Furthermore, OFERA establishes an end-to-end pipeline that senses and maps expressions, updates Gaussian avatars, and renders them in real-time within VR environments. We show that EPM outperforms existing mapping methods on quantitative metrics, and we demonstrate through a user study that the full OFERA framework enhances expression fidelity while preserving avatar realism. By enabling real-time and photorealistic avatar expression control, OFERA significantly improves telepresence in VR communication. A project page is available at https://ysshwan147.github.io/projects/ofera/.

  </details>



- **FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization**  
  Yikun Ma, Yiqing Li, Jingwen Ye, Zhongkai Wu, Weidong Zhang, Lin Gao, Zhi Jin  
  _2026-02-02_ · https://arxiv.org/abs/2602.01723v1  
  <details><summary>Abstract</summary>

  Extending 3D Gaussian Splatting (3DGS) to 4D physical simulation remains challenging. Based on the Material Point Method (MPM), existing methods either rely on manual parameter tuning or distill dynamics from video diffusion models, limiting the generalization and optimization efficiency. Recent attempts using LLMs/VLMs suffer from a text/image-to-3D perceptual gap, yielding unstable physics behavior. In addition, they often ignore the surface structure of 3DGS, leading to implausible motion. We propose FastPhysGS, a fast and robust framework for physics-based dynamic 3DGS simulation:(1) Instance-aware Particle Filling (IPF) with Monte Carlo Importance Sampling (MCIS) to efficiently populate interior particles while preserving geometric fidelity; (2) Bidirectional Graph Decoupling Optimization (BGDO), an adaptive strategy that rapidly optimizes material parameters predicted from a VLM. Experiments show FastPhysGS achieves high-fidelity physical simulation in 1 minute using only 7 GB runtime memory, outperforming prior works with broad potential applications.

  </details>



- **VRGaussianAvatar: Integrating 3D Gaussian Avatars into VR**  
  Hail Song, Boram Yoon, Seokhwan Yang, Seoyoung Kang, Hyunjeong Kim, Henning Metzmacher, Woontack Woo  
  _2026-02-02_ · https://arxiv.org/abs/2602.01674v1  
  <details><summary>Abstract</summary>

  We present VRGaussianAvatar, an integrated system that enables real-time full-body 3D Gaussian Splatting (3DGS) avatars in virtual reality using only head-mounted display (HMD) tracking signals. The system adopts a parallel pipeline with a VR Frontend and a GA Backend. The VR Frontend uses inverse kinematics to estimate full-body pose and streams the resulting pose along with stereo camera parameters to the backend. The GA Backend stereoscopically renders a 3DGS avatar reconstructed from a single image. To improve stereo rendering efficiency, we introduce Binocular Batching, which jointly processes left and right eye views in a single batched pass to reduce redundant computation and support high-resolution VR displays. We evaluate VRGaussianAvatar with quantitative performance tests and a within-subject user study against image- and video-based mesh avatar baselines. Results show that VRGaussianAvatar sustains interactive VR performance and yields higher perceived appearance similarity, embodiment, and plausibility. Project page and source code are available at https://vrgaussianavatar.github.io.

  </details>



- **MarkCleaner: High-Fidelity Watermark Removal via Imperceptible Micro-Geometric Perturbation**  
  Xiaoxi Kong, Jieyu Yuan, Pengdi Chen, Yuanlin Zhang, Chongyi Li, Bin Li  
  _2026-02-02_ · https://arxiv.org/abs/2602.01513v1  
  <details><summary>Abstract</summary>

  Semantic watermarks exhibit strong robustness against conventional image-space attacks. In this work, we show that such robustness does not survive under micro-geometric perturbations: spatial displacements can remove watermarks by breaking the phase alignment. Motivated by this observation, we introduce MarkCleaner, a watermark removal framework that avoids semantic drift caused by regeneration-based watermark removal. Specifically, MarkCleaner is trained with micro-geometry-perturbed supervision, which encourages the model to separate semantic content from strict spatial alignment and enables robust reconstruction under subtle geometric displacements. The framework adopts a mask-guided encoder that learns explicit spatial representations and a 2D Gaussian Splatting-based decoder that explicitly parameterizes geometric perturbations while preserving semantic content. Extensive experiments demonstrate that MarkCleaner achieves superior performance in both watermark removal effectiveness and visual fidelity, while enabling efficient real-time inference. Our code will be made available upon acceptance.

  </details>



- **Radioactive 3D Gaussian Ray Tracing for Tomographic Reconstruction**  
  Ling Chen, Bao Yang  
  _2026-02-01_ · https://arxiv.org/abs/2602.01057v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has recently emerged in computer vision as a promising rendering technique. By adapting the principles of Elliptical Weighted Average (EWA) splatting to a modern differentiable pipeline, 3DGS enables real-time, high-quality novel view synthesis. Building upon this, R2-Gaussian extended the 3DGS paradigm to tomographic reconstruction by rectifying integration bias, achieving state-of-the-art performance in computed tomography (CT). To enable differentiability, R2-Gaussian adopts a local affine approximation: each 3D Gaussian is locally mapped to a 2D Gaussian on the detector and composed via alpha blending to form projections. However, the affine approximation can degrade reconstruction quantitative accuracy and complicate the incorporation of nonlinear geometric corrections. To address these limitations, we propose a tomographic reconstruction framework based on 3D Gaussian ray tracing. Our approach provides two key advantages over splatting-based models: (i) it computes the line integral through 3D Gaussian primitives analytically, avoiding the local affine collapse and thus yielding a more physically consistent forward projection model; and (ii) the ray-tracing formulation gives explicit control over ray origins and directions, which facilitates the precise application of nonlinear geometric corrections, e.g., arc-correction used in positron emission tomography (PET). These properties extend the applicability of Gaussian-based reconstruction to a wider range of realistic tomography systems while improving projection accuracy.

  </details>



- **HPC: Hierarchical Point-based Latent Representation for Streaming Dynamic Gaussian Splatting Compression**  
  Yangzhi Ma, Bojun Liu, Wenting Liao, Dong Liu, Zhu Li, Li Li  
  _2026-01-31_ · https://arxiv.org/abs/2602.00671v1  
  <details><summary>Abstract</summary>

  While dynamic Gaussian Splatting has driven significant advances in free-viewpoint video, maintaining its rendering quality with a small memory footprint for efficient streaming transmission still presents an ongoing challenge. Existing streaming dynamic Gaussian Splatting compression methods typically leverage a latent representation to drive the neural network for predicting Gaussian residuals between frames. Their core latent representations can be categorized into structured grid-based and unstructured point-based paradigms. However, the former incurs significant parameter redundancy by inevitably modeling unoccupied space, while the latter suffers from limited compactness as it fails to exploit local correlations. To relieve these limitations, we propose HPC, a novel streaming dynamic Gaussian Splatting compression framework. It employs a hierarchical point-based latent representation that operates on a per-Gaussian basis to avoid parameter redundancy in unoccupied space. Guided by a tailored aggregation scheme, these latent points achieve high compactness with low spatial redundancy. To improve compression efficiency, we further undertake the first investigation to compress neural networks for streaming dynamic Gaussian Splatting through mining and exploiting the inter-frame correlation of parameters. Combined with latent compression, this forms a fully end-to-end compression framework. Comprehensive experimental evaluations demonstrate that HPC substantially outperforms state-of-the-art methods. It achieves a storage reduction of 67% against its baseline while maintaining high reconstruction fidelity.

  </details>



- **Tune-Your-Style: Intensity-tunable 3D Style Transfer with Gaussian Splatting**  
  Yian Zhao, Rushi Ye, Ruochong Zheng, Zesen Cheng, Chaoran Feng, Jiashu Yang, Pengchong Qiao, Chang Liu, Jie Chen  
  _2026-01-31_ · https://arxiv.org/abs/2602.00618v1  
  <details><summary>Abstract</summary>

  3D style transfer refers to the artistic stylization of 3D assets based on reference style images. Recently, 3DGS-based stylization methods have drawn considerable attention, primarily due to their markedly enhanced training and rendering speeds. However, a vital challenge for 3D style transfer is to strike a balance between the content and the patterns and colors of the style. Although the existing methods strive to achieve relatively balanced outcomes, the fixed-output paradigm struggles to adapt to the diverse content-style balance requirements from different users. In this work, we introduce a creative intensity-tunable 3D style transfer paradigm, dubbed \textbf{Tune-Your-Style}, which allows users to flexibly adjust the style intensity injected into the scene to match their desired content-style balance, thus enhancing the customizability of 3D style transfer. To achieve this goal, we first introduce Gaussian neurons to explicitly model the style intensity and parameterize a learnable style tuner to achieve intensity-tunable style injection. To facilitate the learning of tunable stylization, we further propose the tunable stylization guidance, which obtains multi-view consistent stylized views from diffusion models through cross-view style alignment, and then employs a two-stage optimization strategy to provide stable and efficient guidance by modulating the balance between full-style guidance from the stylized views and zero-style guidance from the initial rendering. Extensive experiments demonstrate that our method not only delivers visually appealing results, but also exhibits flexible customizability for 3D style transfer. Project page is available at https://zhao-yian.github.io/TuneStyle.

  </details>


