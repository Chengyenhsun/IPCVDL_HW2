from torchsummary import summary
import torchvision.models as models

# 建立一個帶有批量歸一化的 VGG19 模型
vgg19_bn = models.vgg19_bn(num_classes=10)


# 使用 torchsummary.summary 在終端中顯示模型結構
summary(vgg19_bn, (3, 32, 32))  # 輸入圖像維度 (3, 224, 224)
