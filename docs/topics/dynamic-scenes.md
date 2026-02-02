# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-02 07:16 UTC_

Total papers shown: **1**


---

- **FlowCalib: LiDAR-to-Vehicle Miscalibration Detection using Scene Flows**  
  Ilir Tahiraj, Peter Wittal, Markus Lienkamp  
  _2026-01-30_ · https://arxiv.org/abs/2601.23107v1  
  <details><summary>Abstract</summary>

  Accurate sensor-to-vehicle calibration is essential for safe autonomous driving. Angular misalignments of LiDAR sensors can lead to safety-critical issues during autonomous operation. However, current methods primarily focus on correcting sensor-to-sensor errors without considering the miscalibration of individual sensors that cause these errors in the first place. We introduce FlowCalib, the first framework that detects LiDAR-to-vehicle miscalibration using motion cues from the scene flow of static objects. Our approach leverages the systematic bias induced by rotational misalignment in the flow field generated from sequential 3D point clouds, eliminating the need for additional sensors. The architecture integrates a neural scene flow prior for flow estimation and incorporates a dual-branch detection network that fuses learned global flow features with handcrafted geometric descriptors. These combined representations allow the system to perform two complementary binary classification tasks: a global binary decision indicating whether misalignment is present and separate, axis-specific binary decisions indicating whether each rotational axis is misaligned. Experiments on the nuScenes dataset demonstrate FlowCalib's ability to robustly detect miscalibration, establishing a benchmark for sensor-to-vehicle miscalibration detection.

  </details>


