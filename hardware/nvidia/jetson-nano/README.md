# 1. Documents
[Hardware Specific](https://developer.download.nvidia.com/assets/embedded/secure/jetson/Nano/docs/NVIDIA_Jetson_Nano_Developer_Kit_User_Guide.pdf?HQGn5xLF_b3Qmey6SZjyqMDnlhRK1wi1O1mGJg4TdN20FisU5neA5UUIJe9oRXqr5BO6AQNISf1hIOJsK2y3Uz_hyKEtIG-Nr9J_AMYCGDPbEy8-hpLG8FpRHg27L9N5gZBob6H_bX6NGJBn2mZyZN7QtZDIVvYB8D6V3aC0R5L5dPfAe6R1sAy-wfoIXoCMtoVBC8Sn)  
**[NVIDIA Tegra Linux Driver Package Architecture](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%2520Linux%2520Driver%2520Package%2520Development%2520Guide%2Foverview.html%23)**  

# 1.1 Power
- 2 software-defined power mode: 10W (default) & 5W.
[Detail Power Mode](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%2520Linux%2520Driver%2520Package%2520Development%2520Guide%2Fpower_management_nano.html%23)  

| Component                   | Power (W)                           |
| :-------------------------: | :---------------------------------: |
| Carrier board               | 0.5 (2A) - 1.25 (4A) no peripherals |
| GPU (VDD_GPU)               | within power budget                 |
| CPU (VDD_CPU)               | within power budget                 |
| Core (VDD_SOC encode/decode)| not cover by power budger           |
| Attached peripherals        |                                     |

# 1.2 JetPack include:
- TensorRT & cuDNN
- CUDA
- Multimedia API package
- VisionWorks and OpenCV

# 1.3 Developer Tools
- [Nsight Eclipse Edition](https://developer.nvidia.com/nsight-eclipse-edition)  
- [CUDA-GDB](https://developer.nvidia.com/cuda-gdb)  
- [CUDA-MEMCHECK](https://developer.nvidia.com/cuda-memcheck)  
- [Nsight Systems](https://developer.nvidia.com/nsight-systems)  
- [nvprof](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview)  
- Nsight Graphics, Compute, Compute CLI
