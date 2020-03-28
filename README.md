# Pytorch Vision - Heavily Focus on Production Level  

# 1. Classification
## 1.1 Imagenet
- Model benchmark mainly for choosing Features Extraction backbone.  
  - Latency and Throughput benchmark on OpenVINO with Intel Core i7-4800MQ - batch = 2  - input: resize(256) + centercrop(244).
  - **MobileNetV3 & MnasNet & NASNetAMobile** is still on develop stage.  
  - Original Paper compare to Pytorch Implement (thanks to torchvision & awesome guys on opensource community).

|          Name          |   Madds   |  Parameters  |   Top1-acc   | Latency (ms) | Throughput (FPS) |
| :--------------------- | :------:  | :----------: | :----------: | :----------: | :--------------: |
| MobileNet V3 Large Org | 219   M   | 5.4  M       | 75.2   %     |              |                  |
| MobileNet V3 Large     | 273.6 M   | 3.96 M       | 70.29  %     | 14.9365      | 133.9002         |
| MobileNet V3 Small Org | 66    M   | 2.9  M       | 67.4   %     |              |                  |  
| MobileNet V3 Small     | 64.09 M   | 2.51 M       | 65.07  %     | 6.9940       | 285.9594         |
| MobileNet V2 Org       | 300   M   | 3.4  M       | 72     %     |              |                  |  
| MobileNet V2           | 320.2 M   | 3.5  M       | 71.88  %     | 15.6170      | 128.0656         |
| SqueezeNet 1.1 Org     | 352   M   | 1.2  M       | 57.5   %     |              |                  |  
| SqueezeNet 1.1         | 355.9 M   | 1.24 M       | 58.2   %     |              |                  |
| MnasNet A1 Org         | 312   M   | 3.9  M       | 75.2   %     |              |                  |  
| MnasNet A1             |           |              |              | 15.6155      | 128.0779         |
| NASNetAMobile Org      | 564   M   | 5.3  M       | 74.0   %     |              |                  |  
| NASNetAMobile          | 589.9 M   | 5.29 M       | 74.1   %     |              |                  |
| Restnet 50 Org         | 312   M   | 3.9  M       | 76.15  %     |              |                  |  
| Restnet 50 SIN         |           |              | 76.72  %     |              |                  |

# 2. Detection
## 2.1 Docs
[Object Detection Overview](https://machinethink.net/blog/object-detection/)  
[Good Overview](https://cv-tricks.com/object-detection/faster-r-cnn-yolo-ssd/)
[Non-Maximun Supressiion](https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/)    
**[Dilated Convolutions](https://www.inference.vc/dilated-convolutions-and-kronecker-factorisation/)**   

## 2.2 SSD Benchmark
- Customize Resolution: 529 x 961  
- Hardware: Intel i7-8700 + GeForce GTX 1080
- **No Model & Hardware Optimized Yet**  
- Flops is only approximately
- Test on 1000 samples with free CPU and GPU
- For detail please see **[person_detection_benchmark_v1.md](data/benchmark/person_detection_benchmark_v1.md)**

|             Name              |  Flops (MMac) |  Parameters (M)  | Latency (ms) | Throughput (FPS) | GPU Load (%) | GPU Mem (MB) |
| :---------------------------- | :----------:  | :--------------: | :----------: | :--------------: | :----------: | :----------: |
| 1.1 MobilenetV1 1.0 backbone  | 871.167       | 3.31             | 11           |                  | 100          | 695          |
| 1.2 MobilenetV1 0.75 backbone | 369.097       | 1.26             | 7            |                  | 100          | 629          |
| 1.3 MobilenetV1 0.5 backbone  | 172.03        | 0.577            | 4.5          |                  | 100          | 599          |
| 2. Detection only (1.0)       | 1086.882      | 6.57             | 15           | 65               | 100          | 713          |
| 3. Pose estimate only (1.0)   | 871.167       | 3.31             | 22           | 50               | 50 ???       | 697          |

# 3. Segmentation
## 3.1 Docs
[CSAIL MIT Segmentation](https://github.com/CSAILVision/semantic-segmentation-pytorch)  

# 4. Deep Model Visualization
## 4.1 Docs
**[Pytorch Goodwork](https://github.com/utkuozbulak/pytorch-cnn-visualizations)**  
[Features 1](https://nbviewer.jupyter.org/github/anhquan0412/animation-classification/blob/master/convolutional-feature-visualization.ipynb)  
[Features 2](https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030)    

# 5. TODO
1. Vision Deep Learning Models Inspection
    1. [ ] **Vision Framework**  
    2. [x] Mobilenetv3
    3. [x] Mobilenetv2
    4. [x] Deep Model Visualization
    5. [x] MnasNet  
    6. [x] SqueezeNet  
    7. [ ] **SSD Detection Layer**
    8. [ ] **Segmentation Layer**

2. Hardware Inference Acceleration
    1. Intel OpenVINO Ecosystem
        1. [x] Model Optimizer & Inference Engine
        2. [x] Heterogeneous Computing (UHD Graphics 630 - FP16)
        3. [ ] Intel Media & QuickSync - Suspended
        4. [ ] OpenVX - Suspended
        5. [ ] 8-bit Quantization - Suspended
    2. NVIDIA TensorRT Inference Server - Suspended
