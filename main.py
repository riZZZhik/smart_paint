from smart_paint import SmartPaintTrain

if __name__ == '__main__':
    smart_paint = SmartPaintTrain("weights/imagenet-vgg-verydeep-19.mat", "styles/style7.jpg")
    smart_paint.train('test_images/test1.jpg', 'results', 1000, 16, 'train2014', 'checkpoints/')
