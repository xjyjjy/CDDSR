# **Learning Continuous Degradation for Arbitrary Scale Blind Super-Resolution**  

Official Repository for **Learning Continuous Degradation for Arbitrary-Scale Blind
Super-Resolution**  

---

## 📝 Introduction

This repository provides the official implementation of:

**CDDSR**  
Jiyuan Xia, Yuanshen Guan, Ruikang Xu, Jiacheng Li, Mingde Yao, and Zhiwei Xiong

---

## 🖼️ Main Figure

<p align="center">
  <img src="assets/pipeline.pdf" width="100%">
</p>
<p align="center">
Overview of our proposed framework / pipeline.
</p>

---

## 📦 Installation

```bash
git clone https://github.com/xjyjjy/CDDSR
conda create -n CDDSR python=3.9
conda activate CDDSR
pip install -r requirements.txt
```

## 

## 📂 Dataset

To prepare data, download datasets [CDD](https://pan.baidu.com/s/1jAR2sTJY_KDYHSOEMW5v6w?pwd=1234) and place them in the './CDD' folder. Please refer to our paper for more detais.

---

## 🚀 Training

Stage1:

```bash
python train.py  --config config/nyu_s1.yaml  --name your_path --gpu 0,1,2,3
python train.py  --config config/ma_s1.yaml  --name your_path --gpu 0,1,2,3
```

Offline KE Inference:

```bash
python  KE_inference.py  --config config/KE_test_nyu.yaml --model stage1_weight_path --gpu 0
python  KE_inference.py  --config config/KE_train_nyu.yaml --model stage1_weight_path --gpu 0
python  KE_inference.py  --config config/KE_test_ma.yaml --model stage1_weight_path --gpu 0
python  KE_inference.py  --config config/KE_train_ma.yaml --model stage1_weight_path --gpu 0
```

Stage2:

```bash
python train.py  --config config/nyu_s2.yaml  --name your_path --gpu 0,1,2,3
python train.py  --config config/ma_s2.yaml  --name your_path --gpu 0,1,2,3
```

---

## 🧪 Testing / Inference

CDD Defocus Blur:

```bash
echo "x2_defocus"
python  test.py  --config config/test/test_nyu2.yaml  --model stage2_weight_path --gpu 0 &
echo "x2.5_defocus"
python  test.py  --config config/test/test_nyu2.5.yaml  --model stage2_weight_path --gpu 0 &
echo "x3_defocus"
python  test.py  --config config/test/test_nyu3.yaml  --model stage2_weight_path --gpu 0 &
echo "x3.5_defocus"
python  test.py  --config config/test/test_nyu3.5.yaml  --model stage2_weight_path --gpu 0 &
echo "x4_defocus"
python  test.py  --config config/test/test_nyu4.yaml  --model stage2_weight_path --gpu 0 
```

CDD Motion Blur:

```bash
echo "x2_motion"
python test.py  --config config/test/test_ma2.yaml  --model stage2_weight_path  --gpu 0 &
echo "x2.5_motion"
python test.py  --config configs/test/test_ma2.5.yaml  --model stage2_weight_path --gpu 0 &
echo "x3_motion"
python  test.py  --config configs/test/test_ma3.yaml  --model stage2_weight_path --gpu 0 &
echo "x3.5_motion"
python test.py  --config configs/test/test_ma3.5.yaml  --model stage2_weight_path --gpu 0 &
echo "x4_motion"
python test.py  --config configs/test/test_ma4.yaml  --model stage2_weight_path --gpu 0
```

Download pre-trained models from this [link](https://pan.baidu.com/s/1dbqlLU667yHqPa3TcOz7_g?pwd=1234), and place them in the ./checkpoint directory.

---

## 🤝 Acknowledgements

Built upon inspirations from [LIIF](https://github.com/yinboc/liif), [DAN](https://github.com/greatlog/DAN).

---

