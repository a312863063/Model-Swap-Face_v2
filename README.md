# Model-Swap-Face_v2
<br />
&emsp;&emsp;这个项目是基于<a href='https://github.com/NVlabs/stylegan2'>stylegan2</a> <a href='https://github.com/eladrich/pixel2style2pixel'>pSp</a>制作的，比v1版本<a href='https://github.com/a312863063/Model-Swap-Face'>Model-Swap-Face</a>在推理速度和图像质量上有一定提升。主要的功能是将虚拟模特进行环球不同区域的风格转换，目前转换器提供<b>西欧模特</b>、<b>东亚模特</b>和<b>北非模特</b>三种主流的风格样式，可帮我们实现生产资料零成本化以及广告制作的高效化；此外项目还提供了去位姿编码器以获得模特的正脸肖像。转换器的版权所有：<a href='http://www.seeprettyface.com'>www.seeprettyface.com</a>。<br />
&emsp;&emsp;这个项目尚没有花很多精力去做，主要是提供了下列这些文件，怎么样去用就看各位的巧思了。。<br />
&emsp;&emsp;· ```encoder.pt``` -- 含位姿编码器<br />
&emsp;&emsp;· ```encoder_without_pos.pt``` -- 去位姿编码器<br />
&emsp;&emsp;· ```projector_EastAsian.pt``` -- 东亚模特投射器<br />
&emsp;&emsp;· ```projector_WestEuropean.pt``` -- 西欧模特投射器<br />
&emsp;&emsp;· ```projector_NorthAfrican.pt``` -- 北非模特投射器<br /><br /><br /><br /><br />

# 效果预览
<p align="center">
	<img src="https://github.com/a312863063/Model-Swap-Face_v2/blob/main/docs/model_stylization.jpg" alt="Sample">
</p>
<br /><br /><br /><br />

# 使用方法
## 环境配置
&emsp;&emsp;参照<a href='https://github.com/eladrich/pixel2style2pixel'>pSp</a>文档进行环境配置。<br /><br />

## 下载模型
&emsp;&emsp;模型下载地址参见`pretrained_models`文件夹中的txt文档。<br />

## 运行代码
&emsp;&emsp;将图片放在input文件夹下，然后编辑scripts/inference.py，运行如下代码：<br />
&emsp;&emsp;```python scripts/inference.py```<br />
&emsp;&emsp;结果会保存在output文件夹下<br /><br /><br /><br />

## 了解更多
&emsp;&emsp;我周末的时候喜欢思考一些新的应用场景和模式，欢迎访问<a href='http://www.seeprettyface.com'>我的网站</a>了解更多。

