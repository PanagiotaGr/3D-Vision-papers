# 3D Reconstruction

_Updated: 2026-01-22 06:52 UTC_

Total papers shown: **9**


---

- **RayRoPE: Projective Ray Positional Encoding for Multi-view Attention**  
  Yu Wu, Minsik Jeon, Jen-Hao Rick Chang, Oncel Tuzel, Shubham Tulsiani  
  _2026-01-21_ · https://arxiv.org/abs/2601.15275v1  
  <details><summary>Abstract</summary>

  We study positional encodings for multi-view transformers that process tokens from a set of posed input images, and seek a mechanism that encodes patches uniquely, allows SE(3)-invariant attention with multi-frequency similarity, and can be adaptive to the geometry of the underlying scene. We find that prior (absolute or relative) encoding schemes for multi-view attention do not meet the above desiderata, and present RayRoPE to address this gap. RayRoPE represents patch positions based on associated rays but leverages a predicted point along the ray instead of the direction for a geometry-aware encoding. To achieve SE(3) invariance, RayRoPE computes query-frame projective coordinates for computing multi-frequency similarity. Lastly, as the 'predicted' 3D point along a ray may not be precise, RayRoPE presents a mechanism to analytically compute the expected position encoding under uncertainty. We validate RayRoPE on the tasks of novel-view synthesis and stereo depth estimation and show that it consistently improves over alternate position encoding schemes (e.g. 15% relative improvement on LPIPS in CO3D). We also show that RayRoPE can seamlessly incorporate RGB-D input, resulting in even larger gains over alternatives that cannot positionally encode this information.

  </details>



- **Three-dimensional visualization of X-ray micro-CT with large-scale datasets: Efficiency and accuracy for real-time interaction**  
  Yipeng Yin, Rao Yao, Qingying Li, Dazhong Wang, Hong Zhou, Zhijun Fang, Jianing Chen, Longjie Qian, Mingyue Wu  
  _2026-01-21_ · https://arxiv.org/abs/2601.15098v1  
  <details><summary>Abstract</summary>

  As Micro-CT technology continues to refine its characterization of material microstructures, industrial CT ultra-precision inspection is generating increasingly large datasets, necessitating solutions to the trade-off between accuracy and efficiency in the 3D characterization of defects during ultra-precise detection. This article provides a unique perspective on recent advances in accurate and efficient 3D visualization using Micro-CT, tracing its evolution from medical imaging to industrial non-destructive testing (NDT). Among the numerous CT reconstruction and volume rendering methods, this article selectively reviews and analyzes approaches that balance accuracy and efficiency, offering a comprehensive analysis to help researchers quickly grasp highly efficient and accurate 3D reconstruction methods for microscopic features. By comparing the principles of computed tomography with advancements in microstructural technology, this article examines the evolution of CT reconstruction algorithms from analytical methods to deep learning techniques, as well as improvements in volume rendering algorithms, acceleration, and data reduction. Additionally, it explores advanced lighting models for high-accuracy, photorealistic, and efficient volume rendering. Furthermore, this article envisions potential directions in CT reconstruction and volume rendering. It aims to guide future research in quickly selecting efficient and precise methods and developing new ideas and approaches for real-time online monitoring of internal material defects through virtual-physical interaction, for applying digital twin model to structural health monitoring (SHM).

  </details>



- **Symmetry Informative and Agnostic Feature Disentanglement for 3D Shapes**  
  Tobias Weißberg, Weikang Wang, Paul Roetzer, Nafie El Amrani, Florian Bernard  
  _2026-01-21_ · https://arxiv.org/abs/2601.14804v1  
  <details><summary>Abstract</summary>

  Shape descriptors, i.e., per-vertex features of 3D meshes or point clouds, are fundamental to shape analysis. Historically, various handcrafted geometry-aware descriptors and feature refinement techniques have been proposed. Recently, several studies have initiated a new research direction by leveraging features from image foundation models to create semantics-aware descriptors, demonstrating advantages across tasks like shape matching, editing, and segmentation. Symmetry, another key concept in shape analysis, has also attracted increasing attention. Consequently, constructing symmetry-aware shape descriptors is a natural progression. Although the recent method $χ$ (Wang et al., 2025) successfully extracted symmetry-informative features from semantic-aware descriptors, its features are only one-dimensional, neglecting other valuable semantic information. Furthermore, the extracted symmetry-informative feature is usually noisy and yields small misclassified patches. To address these gaps, we propose a feature disentanglement approach which is simultaneously symmetry informative and symmetry agnostic. Further, we propose a feature refinement technique to improve the robustness of predicted symmetry informative features. Extensive experiments, including intrinsic symmetry detection, left/right classification, and shape matching, demonstrate the effectiveness of our proposed framework compared to various state-of-the-art methods, both qualitatively and quantitatively.

  </details>



- **Rig-Aware 3D Reconstruction of Vehicle Undercarriages using Gaussian Splatting**  
  Nitin Kulkarni, Akhil Devarashetti, Charlie Cluss, Livio Forte, Dan Buckmaster, Philip Schneider, Chunming Qiao, Alina Vereshchaka  
  _2026-01-20_ · https://arxiv.org/abs/2601.14208v1  
  <details><summary>Abstract</summary>

  Inspecting the undercarriage of used vehicles is a labor-intensive task that requires inspectors to crouch or crawl underneath each vehicle to thoroughly examine it. Additionally, online buyers rarely see undercarriage photos. We present an end-to-end pipeline that utilizes a three-camera rig to capture videos of the undercarriage as the vehicle drives over it, and produces an interactive 3D model of the undercarriage. The 3D model enables inspectors and customers to rotate, zoom, and slice through the undercarriage, allowing them to detect rust, leaks, or impact damage in seconds, thereby improving both workplace safety and buyer confidence. Our primary contribution is a rig-aware Structure-from-Motion (SfM) pipeline specifically designed to overcome the challenges of wide-angle lens distortion and low-parallax scenes. Our method overcomes the challenges of wide-angle lens distortion and low-parallax scenes by integrating precise camera calibration, synchronized video streams, and strong geometric priors from the camera rig. We use a constrained matching strategy with learned components, the DISK feature extractor, and the attention-based LightGlue matcher to generate high-quality sparse point clouds that are often unattainable with standard SfM pipelines. These point clouds seed the Gaussian splatting process to generate photorealistic undercarriage models that render in real-time. Our experiments and ablation studies demonstrate that our design choices are essential to achieve state-of-the-art quality.

  </details>



- **ParkingTwin: Training-Free Streaming 3D Reconstruction for Parking-Lot Digital Twins**  
  Xinhao Liu, Yu Wang, Xiansheng Guo, Gordon Owusu Boateng, Yu Cao, Haonan Si, Xingchen Guo, Nirwan Ansari  
  _2026-01-20_ · https://arxiv.org/abs/2601.13706v1  
  <details><summary>Abstract</summary>

  High-fidelity parking-lot digital twins provide essential priors for path planning, collision checking, and perception validation in Automated Valet Parking (AVP). Yet robot-oriented reconstruction faces a trilemma: sparse forward-facing views cause weak parallax and ill-posed geometry; dynamic occlusions and extreme lighting hinder stable texture fusion; and neural rendering typically needs expensive offline optimization, violating edge-side streaming constraints. We propose ParkingTwin, a training-free, lightweight system for online streaming 3D reconstruction. First, OSM-prior-driven geometric construction uses OpenStreetMap semantic topology to directly generate a metric-consistent TSDF, replacing blind geometric search with deterministic mapping and avoiding costly optimization. Second, geometry-aware dynamic filtering employs a quad-modal constraint field (normal/height/depth consistency) to reject moving vehicles and transient occlusions in real time. Third, illumination-robust fusion in CIELAB decouples luminance and chromaticity via adaptive L-channel weighting and depth-gradient suppression, reducing seams under abrupt lighting changes. ParkingTwin runs at 30+ FPS on an entry-level GTX 1660. On a 68,000 m^2 real-world dataset, it achieves SSIM 0.87 (+16.0%), delivers about 15x end-to-end speedup, and reduces GPU memory by 83.3% compared with state-of-the-art 3D Gaussian Splatting (3DGS) that typically requires high-end GPUs (RTX 4090D). The system outputs explicit triangle meshes compatible with Unity/Unreal digital-twin pipelines. Project page: https://mihoutao-liu.github.io/ParkingTwin/

  </details>



- **A Lightweight Model-Driven 4D Radar Framework for Pervasive Human Detection in Harsh Conditions**  
  Zhenan Liu, Amir Khajepour, George Shaker  
  _2026-01-19_ · https://arxiv.org/abs/2601.13373v1  
  <details><summary>Abstract</summary>

  Pervasive sensing in industrial and underground environments is severely constrained by airborne dust, smoke, confined geometry, and metallic structures, which rapidly degrade optical and LiDAR based perception. Elevation resolved 4D mmWave radar offers strong resilience to such conditions, yet there remains a limited understanding of how to process its sparse and anisotropic point clouds for reliable human detection in enclosed, visibility degraded spaces. This paper presents a fully model-driven 4D radar perception framework designed for real-time execution on embedded edge hardware. The system uses radar as its sole perception modality and integrates domain aware multi threshold filtering, ego motion compensated temporal accumulation, KD tree Euclidean clustering with Doppler aware refinement, and a rule based 3D classifier. The framework is evaluated in a dust filled enclosed trailer and in real underground mining tunnels, and in the tested scenarios the radar based detector maintains stable pedestrian identification as camera and LiDAR modalities fail under severe visibility degradation. These results suggest that the proposed model-driven approach provides robust, interpretable, and computationally efficient perception for safety-critical applications in harsh industrial and subterranean environments.

  </details>



- **Real-Time 4D Radar Perception for Robust Human Detection in Harsh Enclosed Environments**  
  Zhenan Liu, Yaodong Cui, Amir Khajepour, George Shaker  
  _2026-01-19_ · https://arxiv.org/abs/2601.13364v1  
  <details><summary>Abstract</summary>

  This paper introduces a novel methodology for generating controlled, multi-level dust concentrations in a highly cluttered environment representative of harsh, enclosed environments, such as underground mines, road tunnels, or collapsed buildings, enabling repeatable mm-wave propagation studies under severe electromagnetic constraints. We also present a new 4D mmWave radar dataset, augmented by camera and LiDAR, illustrating how dust particles and reflective surfaces jointly impact the sensing functionality. To address these challenges, we develop a threshold-based noise filtering framework leveraging key radar parameters (RCS, velocity, azimuth, elevation) to suppress ghost targets and mitigate strong multipath reflections at the raw data level. Building on the filtered point clouds, a cluster-level, rule-based classification pipeline exploits radar semantics-velocity, RCS, and volumetric spread-to achieve reliable, real-time pedestrian detection without extensive domainspecific training. Experimental results confirm that this integrated approach significantly enhances clutter mitigation, detection robustness, and overall system resilience in dust-laden mining environments.

  </details>



- **Think3D: Thinking with Space for Spatial Reasoning**  
  Zaibin Zhang, Yuhan Wu, Lianjie Jia, Yifan Wang, Zhongbo Zhang, Yijiang Li, Binghao Ran, Fuxi Zhang, Zhuohan Sun, Zhenfei Yin, et al.  
  _2026-01-19_ · https://arxiv.org/abs/2601.13029v1  
  <details><summary>Abstract</summary>

  Understanding and reasoning about the physical world requires spatial intelligence: the ability to interpret geometry, perspective, and spatial relations beyond 2D perception. While recent vision large models (VLMs) excel at visual understanding, they remain fundamentally 2D perceivers and struggle with genuine 3D reasoning. We introduce Think3D, a framework that enables VLM agents to think with 3D space. By leveraging 3D reconstruction models that recover point clouds and camera poses from images or videos, Think3D allows the agent to actively manipulate space through camera-based operations and ego/global-view switching, transforming spatial reasoning into an interactive 3D chain-of-thought process. Without additional training, Think3D significantly improves the spatial reasoning performance of advanced models such as GPT-4.1 and Gemini 2.5 Pro, yielding average gains of +7.8% on BLINK Multi-view and MindCube, and +4.7% on VSI-Bench. We further show that smaller models, which struggle with spatial exploration, benefit significantly from a reinforcement learning policy that enables the model to select informative viewpoints and operations. With RL, the benefit from tool usage increases from +0.7% to +6.8%. Our findings demonstrate that training-free, tool-augmented spatial exploration is a viable path toward more flexible and human-like 3D reasoning in multimodal agents, establishing a new dimension of multimodal intelligence. Code and weights are released at https://github.com/zhangzaibin/spagent.

  </details>



- **Generalizable and Animatable 3D Full-Head Gaussian Avatar from a Single Image**  
  Shuling Zhao, Dan Xu  
  _2026-01-19_ · https://arxiv.org/abs/2601.12770v1  
  <details><summary>Abstract</summary>

  Building 3D animatable head avatars from a single image is an important yet challenging problem. Existing methods generally collapse under large camera pose variations, compromising the realism of 3D avatars. In this work, we propose a new framework to tackle the novel setting of one-shot 3D full-head animatable avatar reconstruction in a single feed-forward pass, enabling real-time animation and simultaneous 360$^\circ$ rendering views. To facilitate efficient animation control, we model 3D head avatars with Gaussian primitives embedded on the surface of a parametric face model within the UV space. To obtain knowledge of full-head geometry and textures, we leverage rich 3D full-head priors within a pretrained 3D generative adversarial network (GAN) for global full-head feature extraction and multi-view supervision. To increase the fidelity of the 3D reconstruction of the input image, we take advantage of the symmetric nature of the UV space and human faces to fuse local fine-grained input image features with the global full-head textures. Extensive experiments demonstrate the effectiveness of our method, achieving high-quality 3D full-head modeling as well as real-time animation, thereby improving the realism of 3D talking avatars.

  </details>


