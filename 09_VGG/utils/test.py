import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def test_model(model, 
               test_dataset, 
               criterion_class, 
               batch_size=32, 
               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
               model_path=None  # 加载的模型路径
               ):
    # 加载模型参数
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # 设置为评估模式

    # DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = criterion_class()
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # 禁用梯度计算，加速和节省显存
        for batch in tqdm(test_loader, desc="Testing"):
            # 将数据移至设备
            inputs, labels = batch['image'], batch['label']
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            
            # 统计正确率
            predicted = torch.argmax(outputs, dim=1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    # 计算测试损失和准确率
    avg_test_loss = test_loss / total_samples
    test_accuracy = 100 * correct_predictions / total_samples

    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    return avg_test_loss, test_accuracy
