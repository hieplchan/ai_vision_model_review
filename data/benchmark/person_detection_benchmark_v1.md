# 1. Result table
- Customize Resolution: 529 x 961  
- Hardware: Intel i7-8700 + GeForce GTX 1080
- **No Model & Hardware Optimized Yet**  
- Flops is only approximately
- Test on 1000 samples with free CPU and GPU

|             Name              |  Flops (MMac) |  Parameters (M)  | Latency (ms) | Throughput (FPS) | GPU Load (%) | GPU Mem (MB) |
| :---------------------------- | :----------:  | :--------------: | :----------: | :--------------: | :----------: | :----------: |
| 1.1 MobilenetV1 1.0 backbone  | 871.167       | 3.31             | 11           |                  | 100          | 695          |
| 1.2 MobilenetV1 0.75 backbone | 369.097       | 1.26             | 7            |                  | 100          | 629          |
| 1.3 MobilenetV1 0.5 backbone  | 172.03        | 0.577            | 4.5          |                  | 100          | 599          |
| 2. Detection only (1.0)       | 1086.882      | 6.57             | 15           | 65               | 100          | 713          |
| 3. Pose estimate only (1.0)   | 871.167       | 3.31             | 22           | 50               | 50 ???       | 697          |

## 1.1 Micronet v1 - Detection + Pose estimate + Track
### 1.1.1 Software estimate
- Backbone: 12 ms
- Detect: 6 ms (track replace detect)
- Pose estimate: 12 ms
- Total: 30 ms (~ 30 frame/s)
- GPU load: 50%
- Real-time demand: 10 frame/s
- **6 real-time camera**
### 1.1.2 Hardware
[GPU benchmark](https://timdettmers.com/2019/04/03/which-gpu-for-deep-learning/)  
[i7 8700 CPU: 7.900.000](https://xgear.vn/cpu-intel-core-i7-8700-processor-12m-cache-4-60-ghz/)  
**[RTX 2070 GPU: 15.000.000](https://xgear.vn/card-man-hinh-gigabyte-rtx-2070-8gb-windforce/)**  
~~[GTX 1080: 16.400.000](https://www.anphatpc.com.vn/vga-msi-geforce-gtx-1080-gaming-x-8g_id20304.html)~~  
[Main board: 2.350.000](https://xgear.vn/mainboard-gigabyte-b360m-aorus-gaming-3-lga-1151v2/)  
[Power 550W: 1.350.000](https://xgear.vn/nguon-may-tinh-cooler-master-mwe-bronze-550w-80-plus-bronze/)  
[Case: 500](https://xgear.vn/case-1st-player-r3-rainbow/)  
[SSD 120GB: 490.000](https://xgear.vn/ssd-apacer-panther-sata-iii-120gb/)  
[RAM 16GB BUS 2666: 2.350.000](https://xgear.vn/ram-laptop-kingston-hyperx-impact-ddr4-16gb-bus-2400-cl-14/)  
- **Total: 7.9 + 15 + 2.35 + 1.350 + 0.5 + 0.49 + 2.35= 29.94** (Basic PC Build - Not include fan + other stuff)  
- Price/camera: 29.94/6 = 4.99  

# 2. Features Extract only (backbone)  
- Features Extract Layer:  
  - Customize MobileNetV1  
  - Depth: 1, 0.75, 0.5

# 3. Detection only  
- Features Extract Layer:  
  - Customize MobileNetV1  
  - Depth: 1 (0.5, 0.75 not tested yet)
- Detect Layer:  
  - SSD  
  - Total Priors Box: 29226 (approximately, can be changed in future)  
  - Customize priors box aspect ratio: no more horizontal rectangle priors (not testing yet)
- **No Pre-Processing & Post-Processing** for benchmark purpose.

# 4. Pose estimate only
- Features Extract Layer:  
  - Customize MobileNetV1
  - Depth: 1, 0.75, 0.5

- **No Pre-Processing** for benchmark purpose.

- Human pose Post-Processing:
  - Greedy search (Not optimized)
  - Person Number Depenent (50 for benchmark purpose)
