import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tqdm import tqdm

def train_model(model, 
                train_dataset, 
                val_dataset, 
                criterion_class,
                optimizer_class=optim.Adam,  # 优化器类
                epochs=16, 
                batch_size=32, 
                learning_rate=1e-3, 
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                log_dir="logs",  # TensorBoard 日志目录
                save_dir="checkpoints",  # 保存模型的目录
                early_stopping_patience=None,  # 是否使用 early stopping
                use_scheduler=False,  # 是否使用学习率调度器
                clip_grad_value=None,  # 是否使用梯度裁剪
                verbose=True
                ):

    # Move model to device
    model.to(device)
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    criterion = criterion_class()

    # Optionally use learning rate scheduler
    scheduler = None
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

    # TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Create directory for saving models
    os.makedirs(save_dir, exist_ok=True)

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    global_step = 0
    
    for epoch in range(epochs):
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for batch in tqdm(train_loader, 
                            desc=f"Epoch {epoch+1}/{epochs}", 
                            total=len(train_loader), 
                            disable=not verbose
                        ):
            # Move data to device
            inputs, labels = batch['image'], batch['label']
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass with optional mixed precision
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            if clip_grad_value:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)
            optimizer.step()

            # Statistics
            predicted = torch.argmax(outputs, 1)
            correct = (predicted == labels).sum().item()  # Calculate correct predictions
            running_corrects += correct
            total_samples += labels.size(0)
            writer.add_scalar('Loss/train', loss.item(), global_step)
            if (global_step+1) % 100 == 0:
                writer.add_scalar('Accuracy/train', running_corrects / total_samples, global_step)
                total_samples = 0
                running_corrects = 0

            global_step += 1

        # Validation
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch['image'], batch['label']
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total

        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}')

        # Learning rate scheduler
        if scheduler:
            scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f'best_model_epoch_{epoch+1}.pth'))
        else:
            patience_counter += 1
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break

    writer.close()