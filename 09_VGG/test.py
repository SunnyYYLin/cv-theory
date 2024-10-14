from datasets import load_dataset
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

# transform = transforms.Compose([
#     transforms.Resize((64, 64)),  # 调整图像尺寸为 64x64
#     transforms.ToTensor(),  # 将图像转换为 PyTorch 的张量
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 归一化（ImageNet 统计的 RGB 均值和标准差）
#                          std=[0.229, 0.224, 0.225])
# ])

if __name__ == '__main__':
    tiny_imagenet = load_dataset('zh-plus/tiny-imagenet', split='train')
    # 检查哪张图片转换后不是[3, 64, 64]，并打印
    gray_count = 0
    for data in tqdm(tiny_imagenet):
        image = data['image']
        image = transforms.ToTensor()(image)
        if image.size() != (3, 64, 64):
            print(image.size())
            print(data['image'])
            gray_count += 1
    print(f"gray_count: {gray_count}")