# 1. Documents #
[OpenVINO](https://www.youtube.com/channel/UCkN8KINLvP1rMkL4trkNgTg)  

[Python TBB](https://www.youtube.com/watch?v=REWgzcIzSAA)  
[Python TBB Git](https://github.com/IntelPython/composability_bench/tree/master/scipy2018_demo)  

# 2. Coursera Intel #
[Fundamentals of Parallelism on Intel Architecture](https://www.coursera.org/learn/parallelism-ia/home/welcome)  


# 3. Utilities
```
watch -n0,5 gpustat -cp
htop -u hiep -d 5
setw -g mouse on
```

# 4. OpenVINO
[OpenVINO Model Optimizer Arguments](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html)  
[ImageNet Download](http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads)  

- Setup OpenVINO Enviroments
```
cd C:\Program Files (x86)\IntelSWTools\openvino_2019.1.133\bin
```

- ONNX OpenVINO Model Optimizer
```
cd C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer

python mo_onnx.py --input_model D:\Work_space\ai_learning\vision_models\models\_models\onnx\mobilenetv1_101_features.onnx --output_dir D:\Work_space\ai_learning\vision_models\models\_models\openvino\mobilenetv1 --input_shape [1,961,529,3] --log_level INFO

python mo_caffe.py --input_proto D:\Work_space\ai_learning\vision_models\models\_models\caffe\nasnetamobile_predict.pb --input_model D:\Work_space\ai_learning\vision_models\models\_models\caffe\nasnetamobile_init.pb --output_dir D:\Work_space\ai_learning\vision_models\models\_models\openvino\nasnetamobile --input_shape [2,3,224,224] --log_level INFO

python mo_tf.py --input_model D:\Work_space\micronet\tests\posenet_intel_test\posenet_tensorflow\_models\model-mobilenet_v1_101.pb --output_dir D:\Work_space\ai_learning\vision_models\models\_models\openvino\mobilenetv1 --input_shape [1,961,529,3] --log_level INFO
```

- OpenVINO Benchmark App  
```
cd C:\Program Files (x86)\IntelSWTools\openvino_2019.1.133\deployment_tools\inference_engine\samples\python_samples\benchmark_app  

python benchmark_app.py -i D:\Work_space\ai_learning\vision_models\data\image -m D:\Work_space\ai_learning\vision_models\models\_models\openvino\mobilenetv1\mobilenetv1_101_features.xml -api sync -d CPU -niter 1000 -nireq 1 -nthreads 1 -b 1 -pin YES  
```
