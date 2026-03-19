import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
import time
import os
from torchvision.models import vgg16
from PerceptualLoss import LossNetwork
from model import Laplace, LANet
import argparse
from dataloader import TrainDataSet
from tqdm import tqdm

# ===== 新增: 尝试导入 thop 计算 FLOPs =====
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("[Warning] 未安装 thop，无法统计 FLOPs。安装命令： pip install thop")
# ======================================

T = 100000  # 默认=100000

log_dir = 'logs/'
print(log_dir)


class CharbonnierLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(CharbonnierLoss, self).__init__()
        self.delta = delta

    def forward(self, x):
        return torch.mean(self.delta**2 * (torch.sqrt(1 + (x/self.delta)**2) - 1))


# ===== 新增：统计参数量的函数 =====
def count_parameters(model):
    """统计模型中可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===== 新增：打印模型 Params / FLOPs / 推理速度 =====
def print_model_profile(model, img_size, device, name="LANet"):
    """
    打印模型的参数量（Params）、FLOPs（若安装 thop）、以及推理速度（单张 ms 和 FPS）
    默认只对 LANet 做统计（Laplace 一般不算在推理网络里）
    """
    model = model.to(device)
    model.eval()

    # 1) Params
    num_params = count_parameters(model)
    print("========== Model Profile [{}] ==========".format(name))
    print(f"Params: {num_params}  (~{num_params/1e6:.3f} M)")

    # 构造一个 dummy 输入（与训练时 Resize 一致）
    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    # 2) FLOPs（可选）
    if HAS_THOP:
        with torch.no_grad():
            # LANet 的 forward 返回 (generate_img, laplace_map)，thop 只用来计算图，不影响
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
        print(f"FLOPs per image: {flops}  (~{flops/1e9:.3f} GFLOPs)")
    else:
        print("FLOPs: 未安装 thop，跳过统计（pip install thop）。")

    # 3) 推理时间 / FPS（只测前向传播）
    n_warmup = 5   # 预热次数
    n_iter = 20    # 计时迭代次数
    with torch.no_grad():
        # 预热
        for _ in range(n_warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(n_iter):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

    avg_time = (t1 - t0) / n_iter  # 秒/每张
    print(f"Inference time: {avg_time * 1000:.3f} ms / image")
    print(f"FPS: {1.0 / avg_time:.2f}")
    print("========================================")


'''MFANet 训练部分'''
def trainB(config):
    device = torch.device(("cuda:") + str(config.cuda_id) if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    '''初始化'''
    criterion = []
    start_time = time.time()
    img_loss_lst = []
    laplace_loss_lst = []
    epoch = 0
    laplace = Laplace().to(device)
    model = LANet().to(device)

    # ========= 新增：打印参数量 =========
    lanet_params = count_parameters(model)
    laplace_params = count_parameters(laplace)

    print("======================================")
    print(f"LANet 可训练参数量: {lanet_params} ({lanet_params/1e6:.3f} M)")
    print(f"Laplace 可训练参数量: {laplace_params} ({laplace_params/1e6:.3f} M)")
    print(f"总可训练参数量: {lanet_params + laplace_params} ({(lanet_params + laplace_params)/1e6:.3f} M)")
    print("======================================")

    # ========= 新增：打印 LANet 的 FLOPs 和推理速度 =========
    print_model_profile(model, img_size=config.resize, device=device, name="LANet")
    # ===================================================

    '''创建文件夹'''
    os.makedirs(config.data_folder, exist_ok=True)
    os.makedirs(config.snapshots_folder, exist_ok=True)
    os.makedirs(config.loss_folder, exist_ok=True)


    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # 加载训练数据集
    transform_list = [transforms.Resize((config.resize, config.resize)), transforms.ToTensor()]
    tsfms = transforms.Compose(transform_list)
    train_dataset = TrainDataSet(config.input_images_path, config.label_images_path, tsfms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    '''训练过程'''
    accumulation_steps = 4  # 设置您需要的累积步数
    for epoch in range(epoch, config.num_epochs):
        img_loss_tmp = []
        laplace_loss_tmp = []
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch}/{config.num_epochs}', leave=False)

        for input_img, label_img in progress_bar:
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            for flag in range(2):
                model.zero_grad()
                generate_img, laplace_map = model(input_img)

                if flag == 0:
                    laplace_label = laplace(label_img)
                    laplace_loss = criterion[0](laplace_map, laplace_label).to(device)
                    laplace_loss_tmp.append(laplace_loss.item())
                    laplace_loss.backward()

                if flag == 1:
                    img_loss = criterion[1](generate_img - label_img)
                    vgg_loss = criterion[2](generate_img, label_img)
                    loss1 = img_loss + 0.5 * vgg_loss
                    img_loss_tmp.append(loss1.item())
                    loss1.backward()

            # 每 accumulation_steps 步更新一次梯度
            if (epoch * len(train_dataloader) + progress_bar.n) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({'训练损失': loss1.item(), '边缘损失': laplace_loss.item()})
            img_loss_lst.append(np.mean(img_loss_tmp))
            laplace_loss_lst.append(np.mean(laplace_loss_tmp))

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "img_loss": img_loss_lst,
                "laplace_loss": laplace_loss_lst,
                "model": model
            }, config.snapshots_folder + 'model_epoch_{}.pk'.format(epoch))

        if epoch % 10 == 0:
            np.save(config.loss_folder + f'{epoch}_imgloss.npy', img_loss_lst)
            np.save(config.loss_folder + f'{epoch}_laplace_loss.npy', laplace_loss_lst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 输入参数
    parser.add_argument('--input_images_path', type=str, default="./data-UFO/trainB/",
                        help='输入图像的路径（水下图像）默认: ./data-UFO/trainB/')
    parser.add_argument('--label_images_path', type=str, default="./data-UFO/trainA/",
                        help='标签图像的路径（清晰图像）默认: ./data-UFO/trainA/')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--decay_rate', type=float, default=0.7, help='学习率衰减 默认: 0.7')
    parser.add_argument('--num_epochs', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=2, help="默认: 1")
    parser.add_argument('--resize', type=int, default=256, help="调整图像大小，默认: 调整图像为256*256")
    parser.add_argument('--snapshots_folder', type=str, default="./checkpoints1/")
    parser.add_argument('--loss_folder', type=str, default="./loss_files/")
    parser.add_argument('--data_folder', type=str, default="./train_dataset/")
    parser.add_argument('--resume', type=bool, default=False)

    config = parser.parse_args()
    trainB(config)
