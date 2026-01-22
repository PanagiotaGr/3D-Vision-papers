# SLAM & Localization

_Updated: 2026-01-22 06:52 UTC_

Total papers shown: **6**


---

- **MonoRace: Winning Champion-Level Drone Racing with Robust Monocular AI**  
  Stavrow A. Bahnam, Robin Ferede, Till M. Blaha, Anton E. Lang, Erin Lucassen, Quentin Missinne, Aderik E. C. Verraest, Christophe De Wagter, Guido C. H. E. de Croon  
  _2026-01-21_ · https://arxiv.org/abs/2601.15222v1  
  <details><summary>Abstract</summary>

  Autonomous drone racing represents a major frontier in robotics research. It requires an Artificial Intelligence (AI) that can run on board light-weight flying robots under tight resource and time constraints, while pushing the physical system to its limits. The state of the art in this area consists of a system with a stereo camera and an inertial measurement unit (IMU) that beat human drone racing champions in a controlled indoor environment. Here, we present MonoRace: an onboard drone racing approach that uses a monocular, rolling-shutter camera and IMU that generalizes to a competition environment without any external motion tracking system. The approach features robust state estimation that combines neural-network-based gate segmentation with a drone model. Moreover, it includes an offline optimization procedure that leverages the known geometry of gates to refine any state estimation parameter. This offline optimization is based purely on onboard flight data and is important for fine-tuning the vital external camera calibration parameters. Furthermore, the guidance and control are performed by a neural network that foregoes inner loop controllers by directly sending motor commands. This small network runs on the flight controller at 500Hz. The proposed approach won the 2025 Abu Dhabi Autonomous Drone Racing Competition (A2RL), outperforming all competing AI teams and three human world champion pilots in a direct knockout tournament. It set a new milestone in autonomous drone racing research, reaching speeds up to 100 km/h on the competition track and successfully coping with problems such as camera interference and IMU saturation.

  </details>



- **LaVR: Scene Latent Conditioned Generative Video Trajectory Re-Rendering using Large 4D Reconstruction Models**  
  Mingyang Xie, Numair Khan, Tianfu Wang, Naina Dhingra, Seonghyeon Nam, Haitao Yang, Zhuo Hui, Christopher Metzler, Andrea Vedaldi, Hamed Pirsiavash, et al.  
  _2026-01-21_ · https://arxiv.org/abs/2601.14674v1  
  <details><summary>Abstract</summary>

  Given a monocular video, the goal of video re-rendering is to generate views of the scene from a novel camera trajectory. Existing methods face two distinct challenges. Geometrically unconditioned models lack spatial awareness, leading to drift and deformation under viewpoint changes. On the other hand, geometrically-conditioned models depend on estimated depth and explicit reconstruction, making them susceptible to depth inaccuracies and calibration errors. We propose to address these challenges by using the implicit geometric knowledge embedded in the latent space of a large 4D reconstruction model to condition the video generation process. These latents capture scene structure in a continuous space without explicit reconstruction. Therefore, they provide a flexible representation that allows the pretrained diffusion prior to regularize errors more effectively. By jointly conditioning on these latents and source camera poses, we demonstrate that our model achieves state-of-the-art results on the video re-rendering task. Project webpage is https://lavr-4d-scene-rerender.github.io/

  </details>



- **SUNSET -- A Sensor-fUsioN based semantic SegmEnTation exemplar for ROS-based self-adaptation**  
  Andreas Wiedholz, Rafael Paintner, Julian Gleißner, Alwin Hoffmann, Tobias Huber  
  _2026-01-20_ · https://arxiv.org/abs/2601.13732v1  
  <details><summary>Abstract</summary>

  The fact that robots are getting deployed more often in dynamic environments, together with the increasing complexity of their software systems, raises the need for self-adaptive approaches. In these environments robotic software systems increasingly operate amid (1) uncertainties, where symptoms are easy to observe but root causes are ambiguous, or (2) multiple uncertainties appear concurrently. We present SUNSET, a ROS2-based exemplar that enables rigorous, repeatable evaluation of architecture-based self-adaptation in such conditions. It implements a sensor fusion semantic-segmentation pipeline driven by a trained Machine Learning (ML) model whose input preprocessing can be perturbed to induce realistic performance degradations. The exemplar exposes five observable symptoms, where each can be caused by different root causes and supports concurrent uncertainties spanning self-healing and self-optimisation. SUNSET includes the segmentation pipeline, a trained ML model, uncertainty-injection scripts, a baseline controller, and step-by-step integration and evaluation documentation to facilitate reproducible studies and fair comparison.

  </details>



- **A General One-Shot Multimodal Active Perception Framework for Robotic Manipulation: Learning to Predict Optimal Viewpoint**  
  Deyun Qin, Zezhi Liu, Hanqian Luo, Xiao Liang, Yongchun Fang  
  _2026-01-20_ · https://arxiv.org/abs/2601.13639v1  
  <details><summary>Abstract</summary>

  Active perception in vision-based robotic manipulation aims to move the camera toward more informative observation viewpoints, thereby providing high-quality perceptual inputs for downstream tasks. Most existing active perception methods rely on iterative optimization, leading to high time and motion costs, and are tightly coupled with task-specific objectives, which limits their transferability. In this paper, we propose a general one-shot multimodal active perception framework for robotic manipulation. The framework enables direct inference of optimal viewpoints and comprises a data collection pipeline and an optimal viewpoint prediction network. Specifically, the framework decouples viewpoint quality evaluation from the overall architecture, supporting heterogeneous task requirements. Optimal viewpoints are defined through systematic sampling and evaluation of candidate viewpoints, after which large-scale training datasets are constructed via domain randomization. Moreover, a multimodal optimal viewpoint prediction network is developed, leveraging cross-attention to align and fuse multimodal features and directly predict camera pose adjustments. The proposed framework is instantiated in robotic grasping under viewpoint-constrained environments. Experimental results demonstrate that active perception guided by the framework significantly improves grasp success rates. Notably, real-world evaluations achieve nearly double the grasp success rate and enable seamless sim-to-real transfer without additional fine-tuning, demonstrating the effectiveness of the proposed framework.

  </details>



- **Event-based Heterogeneous Information Processing for Online Vision-based Obstacle Detection and Localization**  
  Reza Ahmadvand, Sarah Safura Sharif, Yaser Mike Banad  
  _2026-01-19_ · https://arxiv.org/abs/2601.13451v1  
  <details><summary>Abstract</summary>

  This paper introduces a novel framework for robotic vision-based navigation that integrates Hybrid Neural Networks (HNNs) with Spiking Neural Network (SNN)-based filtering to enhance situational awareness for unmodeled obstacle detection and localization. By leveraging the complementary strengths of Artificial Neural Networks (ANNs) and SNNs, the system achieves both accurate environmental understanding and fast, energy-efficient processing. The proposed architecture employs a dual-pathway approach: an ANN component processes static spatial features at low frequency, while an SNN component handles dynamic, event-based sensor data in real time. Unlike conventional hybrid architectures that rely on domain conversion mechanisms, our system incorporates a pre-developed SNN-based filter that directly utilizes spike-encoded inputs for localization and state estimation. Detected anomalies are validated using contextual information from the ANN pathway and continuously tracked to support anticipatory navigation strategies. Simulation results demonstrate that the proposed method offers acceptable detection accuracy while maintaining computational efficiency close to SNN-only implementations, which operate at a fraction of the resource cost. This framework represents a significant advancement in neuromorphic navigation systems for robots operating in unpredictable and dynamic environments.

  </details>



- **Think3D: Thinking with Space for Spatial Reasoning**  
  Zaibin Zhang, Yuhan Wu, Lianjie Jia, Yifan Wang, Zhongbo Zhang, Yijiang Li, Binghao Ran, Fuxi Zhang, Zhuohan Sun, Zhenfei Yin, et al.  
  _2026-01-19_ · https://arxiv.org/abs/2601.13029v1  
  <details><summary>Abstract</summary>

  Understanding and reasoning about the physical world requires spatial intelligence: the ability to interpret geometry, perspective, and spatial relations beyond 2D perception. While recent vision large models (VLMs) excel at visual understanding, they remain fundamentally 2D perceivers and struggle with genuine 3D reasoning. We introduce Think3D, a framework that enables VLM agents to think with 3D space. By leveraging 3D reconstruction models that recover point clouds and camera poses from images or videos, Think3D allows the agent to actively manipulate space through camera-based operations and ego/global-view switching, transforming spatial reasoning into an interactive 3D chain-of-thought process. Without additional training, Think3D significantly improves the spatial reasoning performance of advanced models such as GPT-4.1 and Gemini 2.5 Pro, yielding average gains of +7.8% on BLINK Multi-view and MindCube, and +4.7% on VSI-Bench. We further show that smaller models, which struggle with spatial exploration, benefit significantly from a reinforcement learning policy that enables the model to select informative viewpoints and operations. With RL, the benefit from tool usage increases from +0.7% to +6.8%. Our findings demonstrate that training-free, tool-augmented spatial exploration is a viable path toward more flexible and human-like 3D reasoning in multimodal agents, establishing a new dimension of multimodal intelligence. Code and weights are released at https://github.com/zhangzaibin/spagent.

  </details>


