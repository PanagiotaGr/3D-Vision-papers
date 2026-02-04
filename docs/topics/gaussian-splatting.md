# Gaussian Splatting & 3DGS

_Updated: 2026-02-04 07:08 UTC_

Total papers shown: **15**


---

- **Constrained Dynamic Gaussian Splatting**  
  Zihan Zheng, Zhenglong Wu, Xuanxuan Wang, Houqiang Zhong, Xiaoyun Zhang, Qiang Hu, Guangtao Zhai, Wenjun Zhang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03538v1  
  <details><summary>Abstract</summary>

  While Dynamic Gaussian Splatting enables high-fidelity 4D reconstruction, its deployment is severely hindered by a fundamental dilemma: unconstrained densification leads to excessive memory consumption incompatible with edge devices, whereas heuristic pruning fails to achieve optimal rendering quality under preset Gaussian budgets. In this work, we propose Constrained Dynamic Gaussian Splatting (CDGS), a novel framework that formulates dynamic scene reconstruction as a budget-constrained optimization problem to enforce a strict, user-defined Gaussian budget during training. Our key insight is to introduce a differentiable budget controller as the core optimization driver. Guided by a multi-modal unified importance score, this controller fuses geometric, motion, and perceptual cues for precise capacity regulation. To maximize the utility of this fixed budget, we further decouple the optimization of static and dynamic elements, employing an adaptive allocation mechanism that dynamically distributes capacity based on motion complexity. Furthermore, we implement a three-phase training strategy to seamlessly integrate these constraints, ensuring precise adherence to the target count. Coupled with a dual-mode hybrid compression scheme, CDGS not only strictly adheres to hardware constraints (error < 2%}) but also pushes the Pareto frontier of rate-distortion performance. Extensive experiments demonstrate that CDGS delivers optimal rendering quality under varying capacity limits, achieving over 3x compression compared to state-of-the-art methods.

  </details>



- **Pi-GS: Sparse-View Gaussian Splatting with Dense π^3 Initialization**  
  Manuel Hofer, Markus Steinberger, Thomas Köhler  
  _2026-02-03_ · https://arxiv.org/abs/2602.03327v1  
  <details><summary>Abstract</summary>

  Novel view synthesis has evolved rapidly, advancing from Neural Radiance Fields to 3D Gaussian Splatting (3DGS), which offers real-time rendering and rapid training without compromising visual fidelity. However, 3DGS relies heavily on accurate camera poses and high-quality point cloud initialization, which are difficult to obtain in sparse-view scenarios. While traditional Structure from Motion (SfM) pipelines often fail in these settings, existing learning-based point estimation alternatives typically require reliable reference views and remain sensitive to pose or depth errors. In this work, we propose a robust method utilizing π^3, a reference-free point cloud estimation network. We integrate dense initialization from π^3 with a regularization scheme designed to mitigate geometric inaccuracies. Specifically, we employ uncertainty-guided depth supervision, normal consistency loss, and depth warping. Experimental results demonstrate that our approach achieves state-of-the-art performance on the Tanks and Temples, LLFF, DTU, and MipNeRF360 datasets.

  </details>



- **WebSplatter: Enabling Cross-Device Efficient Gaussian Splatting in Web Browsers via WebGPU**  
  Yudong Han, Chao Xu, Xiaodan Ye, Weichen Bi, Zilong Dong, Yun Ma  
  _2026-02-03_ · https://arxiv.org/abs/2602.03207v1  
  <details><summary>Abstract</summary>

  We present WebSplatter, an end-to-end GPU rendering pipeline for the heterogeneous web ecosystem. Unlike naive ports, WebSplatter introduces a wait-free hierarchical radix sort that circumvents the lack of global atomics in WebGPU, ensuring deterministic execution across diverse hardware. Furthermore, we propose an opacity-aware geometry culling stage that dynamically prunes splats before rasterization, significantly reducing overdraw and peak memory footprint. Evaluation demonstrates that WebSplatter consistently achieves 1.2$\times$ to 4.5$\times$ speedups over state-of-the-art web viewers.

  </details>



- **SharpTimeGS: Sharp and Stable Dynamic Gaussian Splatting via Lifespan Modulation**  
  Zhanfeng Liao, Jiajun Zhang, Hanzhang Tu, Zhixi Wang, Yunqi Gao, Hongwen Zhang, Yebin Liu  
  _2026-02-03_ · https://arxiv.org/abs/2602.02989v1  
  <details><summary>Abstract</summary>

  Novel view synthesis of dynamic scenes is fundamental to achieving photorealistic 4D reconstruction and immersive visual experiences. Recent progress in Gaussian-based representations has significantly improved real-time rendering quality, yet existing methods still struggle to maintain a balance between long-term static and short-term dynamic regions in both representation and optimization. To address this, we present SharpTimeGS, a lifespan-aware 4D Gaussian framework that achieves temporally adaptive modeling of both static and dynamic regions under a unified representation. Specifically, we introduce a learnable lifespan parameter that reformulates temporal visibility from a Gaussian-shaped decay into a flat-top profile, allowing primitives to remain consistently active over their intended duration and avoiding redundant densification. In addition, the learned lifespan modulates each primitives' motion, reducing drift in long-lived static points while retaining unrestricted motion for short-lived dynamic ones. This effectively decouples motion magnitude from temporal duration, improving long-term stability without compromising dynamic fidelity. Moreover, we design a lifespan-velocity-aware densification strategy that mitigates optimization imbalance between static and dynamic regions by allocating more capacity to regions with pronounced motion while keeping static areas compact and stable. Extensive experiments on multiple benchmarks demonstrate that our method achieves state-of-the-art performance while supporting real-time rendering up to 4K resolution at 100 FPS on one RTX 4090.

  </details>



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
  _2026-02-02_ · https://arxiv.org/abs/2602.02000v2  
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



- **Split&Splat: Zero-Shot Panoptic Segmentation via Explicit Instance Modeling and 3D Gaussian Splatting**  
  Leonardo Monchieri, Elena Camuffo, Francesco Barbato, Pietro Zanuttigh, Simone Milani  
  _2026-02-01_ · https://arxiv.org/abs/2602.03809v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (GS) enables fast and high-quality scene reconstruction, but it lacks an object-consistent and semantically aware structure. We propose Split&Splat, a framework for panoptic scene reconstruction using 3DGS. Our approach explicitly models object instances. It first propagates instance masks across views using depth, thus producing view-consistent 2D masks. Each object is then reconstructed independently and merged back into the scene while refining its boundaries. Finally, instance-level semantic descriptors are embedded in the reconstructed objects, supporting various applications, including panoptic segmentation, object retrieval, and 3D editing. Unlike existing methods, Split&Splat tackles the problem by first segmenting the scene and then reconstructing each object individually. This design naturally supports downstream tasks and allows Split&Splat to achieve state-of-the-art performance on the ScanNetv2 segmentation benchmark.

  </details>


