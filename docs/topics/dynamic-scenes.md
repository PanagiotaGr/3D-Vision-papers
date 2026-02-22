# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-22 07:04 UTC_

Total papers shown: **3**


---

- **4D Monocular Surgical Reconstruction under Arbitrary Camera Motions**  
  Jiwei Shan, Zeyu Cai, Cheng-Tai Hsieh, Yirui Li, Hao Liu, Lijun Han, Hesheng Wang, Shing Shin Cheng  
  _2026-02-19_ · https://arxiv.org/abs/2602.17473v1  
  <details><summary>Abstract</summary>

  Reconstructing deformable surgical scenes from endoscopic videos is challenging and clinically important. Recent state-of-the-art methods based on implicit neural representations or 3D Gaussian splatting have made notable progress. However, most are designed for deformable scenes with fixed endoscope viewpoints and rely on stereo depth priors or accurate structure-from-motion for initialization and optimization, limiting their ability to handle monocular sequences with large camera motion in real clinical settings. To address this, we propose Local-EndoGS, a high-quality 4D reconstruction framework for monocular endoscopic sequences with arbitrary camera motion. Local-EndoGS introduces a progressive, window-based global representation that allocates local deformable scene models to each observed window, enabling scalability to long sequences with substantial motion. To overcome unreliable initialization without stereo depth or accurate structure-from-motion, we design a coarse-to-fine strategy integrating multi-view geometry, cross-window information, and monocular depth priors, providing a robust foundation for optimization. We further incorporate long-range 2D pixel trajectory constraints and physical motion priors to improve deformation plausibility. Experiments on three public endoscopic datasets with deformable scenes and varying camera motions show that Local-EndoGS consistently outperforms state-of-the-art methods in appearance quality and geometry. Ablation studies validate the effectiveness of our key designs. Code will be released upon acceptance at: https://github.com/IRMVLab/Local-EndoGS.

  </details>



- **Tree crop mapping of South America reveals links to deforestation and conservation**  
  Yuchang Jiang, Anton Raichuk, Xiaoye Tong, Vivien Sainte Fare Garnot, Daniel Ortiz-Gonzalo, Dan Morris, Konrad Schindler, Jan Dirk Wegner, Maxim Neumann  
  _2026-02-19_ · https://arxiv.org/abs/2602.17372v1  
  <details><summary>Abstract</summary>

  Monitoring tree crop expansion is vital for zero-deforestation policies like the European Union's Regulation on Deforestation-free Products (EUDR). However, these efforts are hindered by a lack of highresolution data distinguishing diverse agricultural systems from forests. Here, we present the first 10m-resolution tree crop map for South America, generated using a multi-modal, spatio-temporal deep learning model trained on Sentinel-1 and Sentinel-2 satellite imagery time series. The map identifies approximately 11 million hectares of tree crops, 23% of which is linked to 2000-2020 forest cover loss. Critically, our analysis reveals that existing regulatory maps supporting the EUDR often classify established agriculture, particularly smallholder agroforestry, as "forest". This discrepancy risks false deforestation alerts and unfair penalties for small-scale farmers. Our work mitigates this risk by providing a high-resolution baseline, supporting conservation policies that are effective, inclusive, and equitable.

  </details>



- **A Multi-modal Detection System for Infrastructure-based Freight Signal Priority**  
  Ziyan Zhang, Chuheng Wei, Xuanpeng Zhao, Siyan Li, Will Snyder, Mike Stas, Peng Hao, Kanok Boriboonsomsin, Guoyuan Wu  
  _2026-02-19_ · https://arxiv.org/abs/2602.17252v1  
  <details><summary>Abstract</summary>

  Freight vehicles approaching signalized intersections require reliable detection and motion estimation to support infrastructure-based Freight Signal Priority (FSP). Accurate and timely perception of vehicle type, position, and speed is essential for enabling effective priority control strategies. This paper presents the design, deployment, and evaluation of an infrastructure-based multi-modal freight vehicle detection system integrating LiDAR and camera sensors. A hybrid sensing architecture is adopted, consisting of an intersection-mounted subsystem and a midblock subsystem, connected via wireless communication for synchronized data transmission. The perception pipeline incorporates both clustering-based and deep learning-based detection methods with Kalman filter tracking to achieve stable real-time performance. LiDAR measurements are registered into geodetic reference frames to support lane-level localization and consistent vehicle tracking. Field evaluations demonstrate that the system can reliably monitor freight vehicle movements at high spatio-temporal resolution. The design and deployment provide practical insights for developing infrastructure-based sensing systems to support FSP applications.

  </details>


