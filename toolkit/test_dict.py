"""
Author: yida
Time is: 2022/8/19 19:45 
this Code: 分析最终原图识别结果
"""
import numpy as np

if __name__ == '__main__':
    path = r"/Users/yida/Desktop/实验结果/实验三/train/2234/test_dict.npy"        # 2058 2144 2231 2329
    test_dict = np.load(path, allow_pickle=True)
    all_test = test_dict[0]  # 全部预测信息
    single_test = test_dict[1]  # 错分预测信息
    total = 0
    # 分析全部土种信息
    for img, predict in all_test.items():
        img_label = img.split('_')[0]  # 图像标签
        true_label, true_value, predict_label, predict_value = 0, 0, 0, 0
        for label, value in predict.items():
            if label == img_label:
                true_value = value
                true_label = label
            if predict_value < value:
                predict_value = value
                predict_label = label

        if true_label == predict_label:
            # print("{}预测正确, 置信度为{}...".format(img, true_value))
            total += 1
        else:
            print("***{}本应该预测为{}置信度为{}...但错误预测为{}置信度为{}...***".format(img, true_label, true_value, predict_label,
                                                                      predict_value))
    print("一共预测正确的土壤图像数{}/{}...".format(total, len(single_test)))
    print(all_test)
    print(single_test)