import cv2
import time
import argparse
import fastdeploy

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser()
parser.add_argument('--det_model', default='PPOCRv3_det', type=str, help='paddle模型文件夹位置，包含.pdmodel、.pdiparams')
parser.add_argument('--cls_model', default='PPOCRv3_cls', type=str, help='paddle模型文件夹位置，包含.pdmodel、.pdiparams')
parser.add_argument('--rec_model', default='PPOCRv3_rec', type=str, help='paddle模型文件夹位置，包含.pdmodel、.pdiparams')
parser.add_argument('--rec_label', default='rec_label.txt', type=str, help='识别模型的标签')
parser.add_argument('--image_path', default='image/006.jpg', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference', default='trt', type=str)
parser.add_argument('--float16', default=True, type=bool)
args = parser.parse_args()
args.n = int(input('请输入测试的轮次:'))


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
class PPOCRv3:
    def __init__(self, args):
        runtime_option = fastdeploy.RuntimeOption()
        if args.device.lower() in ['gpu', 'cuda']:
            runtime_option.use_gpu()
        else:
            runtime_option.use_cpu()
        if args.inference in ['trt', 'tensorrt']:
            runtime_option.use_trt_backend()
            if args.float16:
                runtime_option.enable_trt_fp16()
        else:
            runtime_option.use_ort_backend()
        cls_batch_size = 1
        rec_batch_size = 6
        print('| 使用{} | 模型加载中... |'.format(args.device))
        det_option = runtime_option
        det_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640], [1, 3, 960, 960])
        det_option.set_trt_cache_file(args.det_model + "/det_trt_cache.trt")  # 直接读取转换后的trt模型，如果没有则下次转换后会保存
        self.det_model = fastdeploy.vision.ocr.DBDetector(args.det_model + '/' + 'inference.pdmodel',
                                                          args.det_model + '/' + 'inference.pdiparams',
                                                          runtime_option=det_option)
        cls_option = runtime_option
        cls_option.set_trt_input_shape("x", [1, 3, 48, 10], [cls_batch_size, 3, 48, 320], [cls_batch_size, 3, 48, 1024])
        cls_option.set_trt_cache_file(args.cls_model + "/cls_trt_cache.trt")  # 直接读取转换后的trt模型，如果没有则下次转换后会保存
        self.cls_model = fastdeploy.vision.ocr.Classifier(args.cls_model + '/' + 'inference.pdmodel',
                                                          args.cls_model + '/' + 'inference.pdiparams',
                                                          runtime_option=cls_option)
        rec_option = runtime_option
        rec_option.set_trt_input_shape("x", [1, 3, 48, 10], [rec_batch_size, 3, 48, 320], [rec_batch_size, 3, 48, 2304])
        rec_option.set_trt_cache_file(args.rec_model + "/rec_trt_cache.trt")  # 直接读取转换后的trt模型，如果没有则下次转换后会保存
        self.rec_model = fastdeploy.vision.ocr.Recognizer(args.rec_model + '/' + 'inference.pdmodel',
                                                          args.rec_model + '/' + 'inference.pdiparams',
                                                          args.rec_label,
                                                          runtime_option=rec_option)
        self.model = fastdeploy.vision.ocr.PPOCRv3(det_model=self.det_model, cls_model=self.cls_model,
                                                   rec_model=self.rec_model)
        print('| 模型加载完毕! |')

    def predict(self, image):  # 结果可能为标点符号、空字符，或带有空格
        pred = self.model.predict(image)
        return pred


if __name__ == '__main__':
    image = cv2.imread(args.image_path)
    model = PPOCRv3(args)
    start_time = time.time()
    for i in range(args.n):
        pred = model.predict(image)
    end_time = time.time()
    text_list = pred.text
    print('| 轮次:{} | 平均每张耗时{:.4f} |'.format(args.n, (end_time - start_time) / args.n))
    input('>>>按回车结束(此时可以查看GPU占用量)<<<')
