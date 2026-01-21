# Gaussian Splatting & 3DGS

_Updated: 2026-01-21 06:54 UTC_

Total papers shown: **10**


---

- **Rig-Aware 3D Reconstruction of Vehicle Undercarriages using Gaussian Splatting**  
  Nitin Kulkarni, Akhil Devarashetti, Charlie Cluss, Livio Forte, Dan Buckmaster, Philip Schneider, Chunming Qiao, Alina Vereshchaka  
  _2026-01-20_ · https://arxiv.org/abs/2601.14208v1  
  <details><summary>Abstract</summary>

  Inspecting the undercarriage of used vehicles is a labor-intensive task that requires inspectors to crouch or crawl underneath each vehicle to thoroughly examine it. Additionally, online buyers rarely see undercarriage photos. We present an end-to-end pipeline that utilizes a three-camera rig to capture videos of the undercarriage as the vehicle drives over it, and produces an interactive 3D model of the undercarriage. The 3D model enables inspectors and customers to rotate, zoom, and slice through the undercarriage, allowing them to detect rust, leaks, or impact damage in seconds, thereby improving both workplace safety and buyer confidence. Our primary contribution is a rig-aware Structure-from-Motion (SfM) pipeline specifically designed to overcome the challenges of wide-angle lens distortion and low-parallax scenes. Our method overcomes the challenges of wide-angle lens distortion and low-parallax scenes by integrating precise camera calibration, synchronized video streams, and strong geometric priors from the camera rig. We use a constrained matching strategy with learned components, the DISK feature extractor, and the attention-based LightGlue matcher to generate high-quality sparse point clouds that are often unattainable with standard SfM pipelines. These point clouds seed the Gaussian splatting process to generate photorealistic undercarriage models that render in real-time. Our experiments and ablation studies demonstrate that our design choices are essential to achieve state-of-the-art quality.

  </details>



- **One-Shot Refiner: Boosting Feed-forward Novel View Synthesis via One-Step Diffusion**  
  Yitong Dong, Qi Zhang, Minchao Jiang, Zhiqiang Wu, Qingnan Fan, Ying Feng, Huaqi Zhang, Hujun Bao, Guofeng Zhang  
  _2026-01-20_ · https://arxiv.org/abs/2601.14161v1  
  <details><summary>Abstract</summary>

  We present a novel framework for high-fidelity novel view synthesis (NVS) from sparse images, addressing key limitations in recent feed-forward 3D Gaussian Splatting (3DGS) methods built on Vision Transformer (ViT) backbones. While ViT-based pipelines offer strong geometric priors, they are often constrained by low-resolution inputs due to computational costs. Moreover, existing generative enhancement methods tend to be 3D-agnostic, resulting in inconsistent structures across views, especially in unseen regions. To overcome these challenges, we design a Dual-Domain Detail Perception Module, which enables handling high-resolution images without being limited by the ViT backbone, and endows Gaussians with additional features to store high-frequency details. We develop a feature-guided diffusion network, which can preserve high-frequency details during the restoration process. We introduce a unified training strategy that enables joint optimization of the ViT-based geometric backbone and the diffusion-based refinement module. Experiments demonstrate that our method can maintain superior generation quality across multiple datasets.

  </details>



- **FastGHA: Generalized Few-Shot 3D Gaussian Head Avatars with Real-Time Animation**  
  Xinya Ji, Sebastian Weiss, Manuel Kansy, Jacek Naruniec, Xun Cao, Barbara Solenthaler, Derek Bradley  
  _2026-01-20_ · https://arxiv.org/abs/2601.13837v1  
  <details><summary>Abstract</summary>

  Despite recent progress in 3D Gaussian-based head avatar modeling, efficiently generating high fidelity avatars remains a challenge. Current methods typically rely on extensive multi-view capture setups or monocular videos with per-identity optimization during inference, limiting their scalability and ease of use on unseen subjects. To overcome these efficiency drawbacks, we propose \OURS, a feed-forward method to generate high-quality Gaussian head avatars from only a few input images while supporting real-time animation. Our approach directly learns a per-pixel Gaussian representation from the input images, and aggregates multi-view information using a transformer-based encoder that fuses image features from both DINOv3 and Stable Diffusion VAE. For real-time animation, we extend the explicit Gaussian representations with per-Gaussian features and introduce a lightweight MLP-based dynamic network to predict 3D Gaussian deformations from expression codes. Furthermore, to enhance geometric smoothness of the 3D head, we employ point maps from a pre-trained large reconstruction model as geometry supervision. Experiments show that our approach significantly outperforms existing methods in both rendering quality and inference efficiency, while supporting real-time dynamic avatar animation.

  </details>



- **ParkingTwin: Training-Free Streaming 3D Reconstruction for Parking-Lot Digital Twins**  
  Xinhao Liu, Yu Wang, Xiansheng Guo, Gordon Owusu Boateng, Yu Cao, Haonan Si, Xingchen Guo, Nirwan Ansari  
  _2026-01-20_ · https://arxiv.org/abs/2601.13706v1  
  <details><summary>Abstract</summary>

  High-fidelity parking-lot digital twins provide essential priors for path planning, collision checking, and perception validation in Automated Valet Parking (AVP). Yet robot-oriented reconstruction faces a trilemma: sparse forward-facing views cause weak parallax and ill-posed geometry; dynamic occlusions and extreme lighting hinder stable texture fusion; and neural rendering typically needs expensive offline optimization, violating edge-side streaming constraints. We propose ParkingTwin, a training-free, lightweight system for online streaming 3D reconstruction. First, OSM-prior-driven geometric construction uses OpenStreetMap semantic topology to directly generate a metric-consistent TSDF, replacing blind geometric search with deterministic mapping and avoiding costly optimization. Second, geometry-aware dynamic filtering employs a quad-modal constraint field (normal/height/depth consistency) to reject moving vehicles and transient occlusions in real time. Third, illumination-robust fusion in CIELAB decouples luminance and chromaticity via adaptive L-channel weighting and depth-gradient suppression, reducing seams under abrupt lighting changes. ParkingTwin runs at 30+ FPS on an entry-level GTX 1660. On a 68,000 m^2 real-world dataset, it achieves SSIM 0.87 (+16.0%), delivers about 15x end-to-end speedup, and reduces GPU memory by 83.3% compared with state-of-the-art 3D Gaussian Splatting (3DGS) that typically requires high-end GPUs (RTX 4090D). The system outputs explicit triangle meshes compatible with Unity/Unreal digital-twin pipelines. Project page: https://mihoutao-liu.github.io/ParkingTwin/

  </details>



- **ICo3D: An Interactive Conversational 3D Virtual Human**  
  Richard Shaw, Youngkyoon Jang, Athanasios Papaioannou, Arthur Moreau, Helisa Dhamo, Zhensong Zhang, Eduardo Pérez-Pellitero  
  _2026-01-19_ · https://arxiv.org/abs/2601.13148v1  
  <details><summary>Abstract</summary>

  This work presents Interactive Conversational 3D Virtual Human (ICo3D), a method for generating an interactive, conversational, and photorealistic 3D human avatar. Based on multi-view captures of a subject, we create an animatable 3D face model and a dynamic 3D body model, both rendered by splatting Gaussian primitives. Once merged together, they represent a lifelike virtual human avatar suitable for real-time user interactions. We equip our avatar with an LLM for conversational ability. During conversation, the audio speech of the avatar is used as a driving signal to animate the face model, enabling precise synchronization. We describe improvements to our dynamic Gaussian models that enhance photorealism: SWinGS++ for body reconstruction and HeadGaS++ for face reconstruction, and provide as well a solution to merge the separate face and body models without artifacts. We also present a demo of the complete system, showcasing several use cases of real-time conversation with the 3D avatar. Our approach offers a fully integrated virtual avatar experience, supporting both oral and written form interactions in immersive environments. ICo3D is applicable to a wide range of fields, including gaming, virtual assistance, and personalized education, among others. Project page: https://ico3d.github.io/

  </details>



- **GaussExplorer: 3D Gaussian Splatting for Embodied Exploration and Reasoning**  
  Kim Yu-Ji, Dahye Lee, Kim Jun-Seong, GeonU Kim, Nam Hyeon-Woo, Yongjin Kwon, Yu-Chiang Frank Wang, Jaesung Choe, Tae-Hyun Oh  
  _2026-01-19_ · https://arxiv.org/abs/2601.13132v1  
  <details><summary>Abstract</summary>

  We present GaussExplorer, a framework for embodied exploration and reasoning built on 3D Gaussian Splatting (3DGS). While prior approaches to language-embedded 3DGS have made meaningful progress in aligning simple text queries with Gaussian embeddings, they are generally optimized for relatively simple queries and struggle to interpret more complex, compositional language queries. Alternative studies based on object-centric RGB-D structured memories provide spatial grounding but are constrained by pre-fixed viewpoints. To address these issues, GaussExplorer introduces Vision-Language Models (VLMs) on top of 3DGS to enable question-driven exploration and reasoning within 3D scenes. We first identify pre-captured images that are most correlated with the query question, and subsequently adjust them into novel viewpoints to more accurately capture visual information for better reasoning by VLMs. Experiments show that ours outperforms existing methods on several benchmarks, demonstrating the effectiveness of integrating VLM-based reasoning with 3DGS for embodied tasks.

  </details>



- **CSGaussian: Progressive Rate-Distortion Compression and Segmentation for 3D Gaussian Splatting**  
  Yu-Jen Tseng, Chia-Hao Kao, Jing-Zhong Chen, Alessandro Gnutti, Shao-Yuan Lo, Yen-Yu Lin, Wen-Hsiao Peng  
  _2026-01-19_ · https://arxiv.org/abs/2601.12814v1  
  <details><summary>Abstract</summary>

  We present the first unified framework for rate-distortion-optimized compression and segmentation of 3D Gaussian Splatting (3DGS). While 3DGS has proven effective for both real-time rendering and semantic scene understanding, prior works have largely treated these tasks independently, leaving their joint consideration unexplored. Inspired by recent advances in rate-distortion-optimized 3DGS compression, this work integrates semantic learning into the compression pipeline to support decoder-side applications--such as scene editing and manipulation--that extend beyond traditional scene reconstruction and view synthesis. Our scheme features a lightweight implicit neural representation-based hyperprior, enabling efficient entropy coding of both color and semantic attributes while avoiding costly grid-based hyperprior as seen in many prior works. To facilitate compression and segmentation, we further develop compression-guided segmentation learning, consisting of quantization-aware training to enhance feature separability and a quality-aware weighting mechanism to suppress unreliable Gaussian primitives. Extensive experiments on the LERF and 3D-OVS datasets demonstrate that our approach significantly reduces transmission cost while preserving high rendering quality and strong segmentation performance.

  </details>



- **Generalizable and Animatable 3D Full-Head Gaussian Avatar from a Single Image**  
  Shuling Zhao, Dan Xu  
  _2026-01-19_ · https://arxiv.org/abs/2601.12770v1  
  <details><summary>Abstract</summary>

  Building 3D animatable head avatars from a single image is an important yet challenging problem. Existing methods generally collapse under large camera pose variations, compromising the realism of 3D avatars. In this work, we propose a new framework to tackle the novel setting of one-shot 3D full-head animatable avatar reconstruction in a single feed-forward pass, enabling real-time animation and simultaneous 360$^\circ$ rendering views. To facilitate efficient animation control, we model 3D head avatars with Gaussian primitives embedded on the surface of a parametric face model within the UV space. To obtain knowledge of full-head geometry and textures, we leverage rich 3D full-head priors within a pretrained 3D generative adversarial network (GAN) for global full-head feature extraction and multi-view supervision. To increase the fidelity of the 3D reconstruction of the input image, we take advantage of the symmetric nature of the UV space and human faces to fuse local fine-grained input image features with the global full-head textures. Extensive experiments demonstrate the effectiveness of our method, achieving high-quality 3D full-head modeling as well as real-time animation, thereby improving the realism of 3D talking avatars.

  </details>



- **KaoLRM: Repurposing Pre-trained Large Reconstruction Models for Parametric 3D Face Reconstruction**  
  Qingtian Zhu, Xu Cao, Zhixiang Wang, Yinqiang Zheng, Takafumi Taketomi  
  _2026-01-19_ · https://arxiv.org/abs/2601.12736v1  
  <details><summary>Abstract</summary>

  We propose KaoLRM to re-target the learned prior of the Large Reconstruction Model (LRM) for parametric 3D face reconstruction from single-view images. Parametric 3D Morphable Models (3DMMs) have been widely used for facial reconstruction due to their compact and interpretable parameterization, yet existing 3DMM regressors often exhibit poor consistency across varying viewpoints. To address this, we harness the pre-trained 3D prior of LRM and incorporate FLAME-based 2D Gaussian Splatting into LRM's rendering pipeline. Specifically, KaoLRM projects LRM's pre-trained triplane features into the FLAME parameter space to recover geometry, and models appearance via 2D Gaussian primitives that are tightly coupled to the FLAME mesh. The rich prior enables the FLAME regressor to be aware of the 3D structure, leading to accurate and robust reconstructions under self-occlusions and diverse viewpoints. Experiments on both controlled and in-the-wild benchmarks demonstrate that KaoLRM achieves superior reconstruction accuracy and cross-view consistency, while existing methods remain sensitive to viewpoint variations. The code is released at https://github.com/CyberAgentAILab/KaoLRM.

  </details>



- **GaussianTrimmer: Online Trimming Boundaries for 3DGS Segmentation**  
  Liwei Liao, Ronggang Wang  
  _2026-01-19_ · https://arxiv.org/abs/2601.12683v1  
  <details><summary>Abstract</summary>

  With the widespread application of 3D Gaussians in 3D scene representation, 3D scene segmentation methods based on 3D Gaussians have also gradually emerged. However, existing 3D Gaussian segmentation methods basically segment on the basis of Gaussian primitives. Due to the large variation range of the scale of 3D Gaussians, large-sized Gaussians that often span the foreground and background lead to jagged boundaries of segmented objects. To this end, we propose an online boundary trimming method, GaussianTrimmer, which is an efficient and plug-and-play post-processing method capable of trimming coarse boundaries for existing 3D Gaussian segmentation methods. Our method consists of two core steps: 1. Generating uniformly and well-covered virtual cameras; 2. Trimming Gaussian at the primitive level based on 2D segmentation results on virtual cameras. Extensive quantitative and qualitative experiments demonstrate that our method can improve the segmentation quality of existing 3D Gaussian segmentation methods as a plug-and-play method.

  </details>


