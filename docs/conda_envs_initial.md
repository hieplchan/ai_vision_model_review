# 1. Conda enviroments setup
```
conda create -n vision-kit python=3.7
conda activate vision-kit
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c intel mkl numpy tbb4py smp
conda install -c fastai fastai
conda install jupyter
pip uninstall pillow
pip install pillow-simd
pip install torchsummary
pip install pycocotools
```

```
conda env remove --name vision-kit
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
```
