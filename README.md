## 快速使用OCR模型检测图片中的文字
>基于paddle官方项目整理：https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/ocr/PP-OCR/cpu-gpu/python  
>FastDeploy中PPOCRv3图片文字识别
>### 项目介绍
>本项目将paddle官方相关代码进行了整理，方便快速简洁的使用  
>rec_label.txt中是识别的字符标签，实际应用中可根据情况替换空格等一些无用符号为空字符
### 1，image
>image文件夹中存放待检测文字的图片
### 2，inference.py
>使用模型检测图片中的文字
### 3，flask_start.py
>用flask将程序包装成一个服务，并在服务器上启动
### 4，flask_request.py
>以post请求传输数据调用服务