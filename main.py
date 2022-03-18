import os

from smart_paint import SmartPaintTrain

if __name__ == '__main__':
    os.system("sh get_weights.sh && sh get_dataset.sh")
    content_dir = 'train_2014'
    content_targets = [os.path.join(content_dir, x) for x in os.listdir(content_dir)]

    smart_paint = SmartPaintTrain("weights/imagenet-vgg-verydeep-19.mat", "styles/style7.jpg")

    smart_paint.train('test_images/test1.jpg', 'results', 1000, 16, content_targets, 'checkpoints/')
