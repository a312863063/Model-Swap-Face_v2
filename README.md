# Model-Swap-Face v2
<br />
&emsp;&emsp;这个项目是基于<a href='https://github.com/NVlabs/stylegan2'>stylegan2</a> <a href='https://github.com/eladrich/pixel2style2pixel'>pSp</a>制作的，比v1版本<a href='https://github.com/a312863063/Model-Swap-Face'>Model-Swap-Face</a>在推理速度和图像质量上有一定提升。主要的功能是将虚拟模特进行环球不同区域的风格转换，目前转换器提供<b>西欧模特</b>、<b>东亚模特</b>和<b>北非模特</b>三种主流的风格样式，可帮我们实现生产资料零成本化以及广告制作的高效化；此外项目还提供了去位姿编码器以获得模特的正脸肖像。投射器的版权所有：<a href='http://www.seeprettyface.com'>www.seeprettyface.com</a>。<br /><br /><br /><br />

# 效果预览
<p align="center">
	<img src="https://github.com/a312863063/Model-Swap-Face_v2/blob/main/docs/model_stylization.jpg" alt="Sample">
</p>
<br /><br /><br /><br />

# Inference框架
<p align="center">
	<img src="https://github.com/a312863063/Model-Swap-Face_v2/blob/main/docs/infer_arch.png" alt="Sample">
</p>
<br /><br /><br /><br />

# 使用方法
## 环境配置
&emsp;&emsp;参照<a href='https://github.com/eladrich/pixel2style2pixel'>pSp</a>文档进行环境配置。<br /><br />

## 下载模型
&emsp;&emsp;模型下载地址参见`pretrained_models`文件夹中的txt文档，主要包括的模型如下：<br />
&emsp;&emsp;```encoder.pt``` -- 含位姿编码器<br />
&emsp;&emsp;```encoder_without_pos.pt``` -- 去位姿编码器<br />
&emsp;&emsp;```projector_EastAsian.pt``` -- 东亚模特投射器<br />
&emsp;&emsp;```projector_WestEuropean.pt``` -- 西欧模特投射器<br />
&emsp;&emsp;```projector_NorthAfrican.pt``` -- 北非模特投射器（比较特殊转线下了，可自行训练或私联我）<br />
&emsp;&emsp;```shape_predictor_68_face_landmarks.dat``` -- 人脸检测<br />
&emsp;&emsp;```79999_iter.pth``` -- 人脸mask分割<br />


## 运行代码
&emsp;&emsp;将图片放在input文件夹下，然后编辑scripts/inference.py的路径，运行如下代码：<br />
&emsp;&emsp;```python scripts/inference.py```<br />
&emsp;&emsp;结果会保存在output文件夹下<br /><br />

## 贴回素材
&emsp;&emsp;按需选择（图片含位姿/图片/视频）：<br />
&emsp;&emsp;1. https://github.com/wuhuikai/FaceSwap<br />
&emsp;&emsp;2. https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels<br />
&emsp;&emsp;3. https://github.com/iperov/DeepFaceLive <br /><br /><br /><br />

# 了解更多
&emsp;&emsp;当然，模特只是很小的一种分类。我期待有一天我们所有人都能无差别的成为生产资料的制造者和使用者，在那种背景下，人们的社交关系会发生质的变化。欢迎您有空时来<a href='http://www.seeprettyface.com'>我的网站</a>逛逛一二，我平时很喜欢思考一些新的应用形式与模态。
 
