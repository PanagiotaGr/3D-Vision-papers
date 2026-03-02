# NeRF & Neural Radiance Fields

_Updated: 2026-03-02 07:14 UTC_

Total papers shown: **3**


---

- **HumanOrbit: 3D Human Reconstruction as 360° Orbit Generation**  
  Keito Suzuki, Kunyao Chen, Lei Wang, Bang Du, Runfa Blark Li, Peng Liu, Ning Bi, Truong Nguyen  
  _2026-02-27_ · https://arxiv.org/abs/2602.24148v1  
  <details><summary>Abstract</summary>

  We present a method for generating a full 360° orbit video around a person from a single input image. Existing methods typically adapt image-based diffusion models for multi-view synthesis, but yield inconsistent results across views and with the original identity. In contrast, recent video diffusion models have demonstrated their ability in generating photorealistic results that align well with the given prompts. Inspired by these results, we propose HumanOrbit, a video diffusion model for multi-view human image generation. Our approach enables the model to synthesize continuous camera rotations around the subject, producing geometrically consistent novel views while preserving the appearance and identity of the person. Using the generated multi-view frames, we further propose a reconstruction pipeline that recovers a textured mesh of the subject. Experimental results validate the effectiveness of HumanOrbit for multi-view image generation and that the reconstructed 3D models exhibit superior completeness and fidelity compared to those from state-of-the-art baselines.

  </details>



- **DiffusionHarmonizer: Bridging Neural Reconstruction and Photorealistic Simulation with Online Diffusion Enhancer**  
  Yuxuan Zhang, Katarína Tóthová, Zian Wang, Kangxue Yin, Haithem Turki, Riccardo de Lutio, Yen-Yu Chang, Or Litany, Sanja Fidler, Zan Gojcic  
  _2026-02-27_ · https://arxiv.org/abs/2602.24096v1  
  <details><summary>Abstract</summary>

  Simulation is essential to the development and evaluation of autonomous robots such as self-driving vehicles. Neural reconstruction is emerging as a promising solution as it enables simulating a wide variety of scenarios from real-world data alone in an automated and scalable way. However, while methods such as NeRF and 3D Gaussian Splatting can produce visually compelling results, they often exhibit artifacts particularly when rendering novel views, and fail to realistically integrate inserted dynamic objects, especially when they were captured from different scenes. To overcome these limitations, we introduce DiffusionHarmonizer, an online generative enhancement framework that transforms renderings from such imperfect scenes into temporally consistent outputs while improving their realism. At its core is a single-step temporally-conditioned enhancer that is converted from a pretrained multi-step image diffusion model, capable of running in online simulators on a single GPU. The key to training it effectively is a custom data curation pipeline that constructs synthetic-real pairs emphasizing appearance harmonization, artifact correction, and lighting realism. The result is a scalable system that significantly elevates simulation fidelity in both research and production environments.

  </details>



- **Leveraging Geometric Prior Uncertainty and Complementary Constraints for High-Fidelity Neural Indoor Surface Reconstruction**  
  Qiyu Feng, Jiwei Shan, Shing Shin Cheng, Hesheng Wang  
  _2026-02-27_ · https://arxiv.org/abs/2602.23926v1  
  <details><summary>Abstract</summary>

  Neural implicit surface reconstruction with signed distance function has made significant progress, but recovering fine details such as thin structures and complex geometries remains challenging due to unreliable or noisy geometric priors. Existing approaches rely on implicit uncertainty that arises during optimization to filter these priors, which is indirect and inefficient, and masking supervision in high-uncertainty regions further leads to under-constrained optimization. To address these issues, we propose GPU-SDF, a neural implicit framework for indoor surface reconstruction that leverages geometric prior uncertainty and complementary constraints. We introduce a self-supervised module that explicitly estimates prior uncertainty without auxiliary networks. Based on this estimation, we design an uncertainty-guided loss that modulates prior influence rather than discarding it, thereby retaining weak but informative cues. To address regions with high prior uncertainty, GPU-SDF further incorporates two complementary constraints: an edge distance field that strengthens boundary supervision and a multi-view consistency regularization that enforces geometric coherence. Extensive experiments confirm that GPU-SDF improves the reconstruction of fine details and serves as a plug-and-play enhancement for existing frameworks. Source code will be available at https://github.com/IRMVLab/GPU-SDF

  </details>


