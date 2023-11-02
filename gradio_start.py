# pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
# 用gradio将程序包装成一个可视化的界面，可以在网页可视化的展示
import gradio
import argparse
from predict import PPOCRv3_class

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser('|在服务器上启动gradio服务|')
parser.add_argument('--det_model', default='PPOCRv3_det', type=str, help='paddle模型文件夹位置，包含.pdmodel、.pdiparams')
parser.add_argument('--cls_model', default='PPOCRv3_cls', type=str, help='paddle模型文件夹位置，包含.pdmodel、.pdiparams')
parser.add_argument('--rec_model', default='PPOCRv3_rec', type=str, help='paddle模型文件夹位置，包含.pdmodel、.pdiparams')
parser.add_argument('--rec_label', default='rec_label.txt', type=str, help='识别模型的标签')
parser.add_argument('--image_path', default='image/demo.jpg', type=str)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference', default='trt', type=str)
parser.add_argument('--float16', default=True, type=bool)
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def function(image):
    text = model.predict(image).text
    return text


if __name__ == '__main__':
    print('| 使用gradio启动服务 |')
    model = PPOCRv3_class(args)
    gradio_app = gradio.Interface(fn=function, inputs=['image'], outputs=['text'])
    gradio_app.launch(share=False)
