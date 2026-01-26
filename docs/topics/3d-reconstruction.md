# 3D Reconstruction

_Updated: 2026-01-26 06:54 UTC_

Total papers shown: **4**


---

- **GPA-VGGT:Adapting VGGT to Large scale Localization by self-Supervised learning with Geometry and Physics Aware loss**  
  Yangfan Xu, Lilian Zhang, Xiaofeng He, Pengdong Wu, Wenqi Wu, Jun Mao  
  _2026-01-23_ · https://arxiv.org/abs/2601.16885v1  
  <details><summary>Abstract</summary>

  Transformer-based general visual geometry frameworks have shown promising performance in camera pose estimation and 3D scene understanding. Recent advancements in Visual Geometry Grounded Transformer (VGGT) models have shown great promise in camera pose estimation and 3D reconstruction. However, these models typically rely on ground truth labels for training, posing challenges when adapting to unlabeled and unseen scenes. In this paper, we propose a self-supervised framework to train VGGT with unlabeled data, thereby enhancing its localization capability in large-scale environments. To achieve this, we extend conventional pair-wise relations to sequence-wise geometric constraints for self-supervised learning. Specifically, in each sequence, we sample multiple source frames and geometrically project them onto different target frames, which improves temporal feature consistency. We formulate physical photometric consistency and geometric constraints as a joint optimization loss to circumvent the requirement for hard labels. By training the model with this proposed method, not only the local and global cross-view attention layers but also the camera and depth heads can effectively capture the underlying multi-view geometry. Experiments demonstrate that the model converges within hundreds of iterations and achieves significant improvements in large-scale localization. Our code will be released at https://github.com/X-yangfan/GPA-VGGT.

  </details>



- **Using Shadows in Circular Synthetic Aperture Sonar Imaging for Target Analysis**  
  Yann Le Gall, Nicolas Burlet, Mathieu Simon, Fabien Novella, Samantha Dugelay, Jean-Philippe Malkasse  
  _2026-01-23_ · https://arxiv.org/abs/2601.16733v1  
  <details><summary>Abstract</summary>

  Circular Synthetic Aperture Sonar (CSAS) provides a 360° azimuth view of the seabed, surpassing the limited aperture and mono-view image of conventional side-scan SAS. This makes CSAS a valuable tool for target recognition in mine warfare where the diversity of point of view is essential for reducing false alarms. CSAS processing typically produces a very high-resolution two-dimensional image. However, the parallax introduced by the circular displacement of the illuminator fill-in the shadow regions, and the shadow cast by an object on the seafloor is lost in favor of azimuth coverage and resolution. Yet the shadows provide complementary information on target shape useful for target recognition. In this paper, we explore a way to retrieve shadow information from CSAS data to improve target analysis and carry 3D reconstruction. Sub-aperture filtering is used to get a collection of images at various points of view along the circular trajectory and fixed focus shadow enhancement (FFSE) is applied to obtain sharp shadows. An interactive interface is also proposed to allow human operators to visualize these shadows along the circular trajectory. A space-carving reconstruction method is applied to infer the 3D shape of the object from the segmented shadows. The results demonstrate the potential of shadows in circular SAS for improving target analysis and 3D reconstruction.

  </details>



- **OnlineSI: Taming Large Language Model for Online 3D Understanding and Grounding**  
  Zixian Liu, Zhaoxi Chen, Liang Pan, Ziwei Liu  
  _2026-01-23_ · https://arxiv.org/abs/2601.16538v1  
  <details><summary>Abstract</summary>

  In recent years, researchers have increasingly been interested in how to enable Multimodal Large Language Models (MLLM) to possess spatial understanding and reasoning capabilities. However, most existing methods overlook the importance of the ability to continuously work in an ever-changing world, and lack the possibility of deployment on embodied systems in real-world environments. In this work, we introduce OnlineSI, a framework that can continuously improve its spatial understanding of its surroundings given a video stream. Our core idea is to maintain a finite spatial memory to retain past observations, ensuring the computation required for each inference does not increase as the input accumulates. We further integrate 3D point cloud information with semantic information, helping MLLM to better locate and identify objects in the scene. To evaluate our method, we introduce the Fuzzy $F_1$-Score to mitigate ambiguity, and test our method on two representative datasets. Experiments demonstrate the effectiveness of our method, paving the way towards real-world embodied systems.

  </details>



- **AnchoredDream: Zero-Shot 360° Indoor Scene Generation from a Single View via Geometric Grounding**  
  Runmao Yao, Junsheng Zhou, Zhen Dong, Yu-Shen Liu  
  _2026-01-23_ · https://arxiv.org/abs/2601.16532v1  
  <details><summary>Abstract</summary>

  Single-view indoor scene generation plays a crucial role in a range of real-world applications. However, generating a complete 360° scene from a single image remains a highly ill-posed and challenging problem. Recent approaches have made progress by leveraging diffusion models and depth estimation networks, yet they still struggle to maintain appearance consistency and geometric plausibility under large viewpoint changes, limiting their effectiveness in full-scene generation. To address this, we propose AnchoredDream, a novel zero-shot pipeline that anchors 360° scene generation on high-fidelity geometry via an appearance-geometry mutual boosting mechanism. Given a single-view image, our method first performs appearance-guided geometry generation to construct a reliable 3D scene layout. Then, we progressively generate the complete scene through a series of modules: warp-and-inpaint, warp-and-refine, post-optimization, and a novel Grouting Block, which ensures seamless transitions between the input view and generated regions. Extensive experiments demonstrate that AnchoredDream outperforms existing methods by a large margin in both appearance consistency and geometric plausibility--all in a zero-shot manner. Our results highlight the potential of geometric grounding for high-quality, zero-shot single-view scene generation.

  </details>


