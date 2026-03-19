import torch
import torchvision
from torchvision import transforms
import os
import argparse
from dataloader import TestDataSet
import torch.nn.functional as F
from skimage.color import rgb2lab
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

from model import LANet


def rgb_to_lab(img):
    # 如果输入是四维张量（batch_size, channels, height, width）
    if img.dim() == 4:
        img = img.squeeze(0)  # 移除 batch_size 维度
    # 确保输入图像的形状为 (H, W, C)
    img = img.permute(1, 2, 0).cpu()  # 移动到 CPU

    # 将 RGB 转换为 Lab
    lab_img = rgb2lab(img.numpy())
    lab_img = torch.from_numpy(lab_img.transpose(2, 0, 1)).float()

    return lab_img


def calculate_uiconm(img1, img2):
    img2 = img2.to(img1.device)
    diff = torch.abs(img1 - img2)
    max_diff = torch.max(diff)
    min_diff = torch.min(diff)
    uiconm = max_diff - min_diff
    return uiconm.item()  # 返回 UICONM 值

def calculate_uism(img):
    # 将 generate_img 转换为单通道灰度图像
    img_gray = torch.mean(img, dim=1, keepdim=True)

    # 使用 Sobel 算子计算图像梯度
    gradient_x = F.conv2d(img_gray, torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                                 dtype=img.dtype, device=img.device).view(1, 1, 3, 3), padding=1)
    gradient_y = F.conv2d(img_gray, torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                                 dtype=img.dtype, device=img.device).view(1, 1, 3, 3), padding=1)

    # 计算梯度的均值作为 UISM 值
    uism = torch.mean(torch.abs(gradient_x) + torch.abs(gradient_y))

    return uism.item()

def calculate_color_moments(img):
    # 计算Lab颜色空间中的色彩矩
    mean_values = torch.mean(img, dim=(1, 2))
    std_values = torch.std(img, dim=(1, 2))
    color_moments = torch.cat((mean_values, std_values), dim=0)
    return color_moments.numpy()

# 添加计算 UICM 的函数
def calculate_uicm(img1, img2):
    # 将图像展平为一维张量
    flat_img1 = img1.view(-1)
    flat_img2 = img2.view(-1)

    # 计算图像的均值和方差
    mu1 = torch.mean(flat_img1)
    mu2 = torch.mean(flat_img2)
    sigma1_sq = torch.mean((flat_img1 - mu1) ** 2)
    sigma2_sq = torch.mean((flat_img2 - mu2) ** 2)

    # 计算亮度对比度
    contrast = 2 * sigma1_sq / (sigma1_sq + sigma2_sq)

    # 计算亮度对比度和亮度差的乘积
    brightness_diff = 2 * mu1 * mu2 / (mu1 ** 2 + mu2 ** 2)

    # 计算 UICM
    uicm = contrast * brightness_diff

    return uicm.item()  # 返回 UICM 值

def calculate_edge_loss(img):
    # 将 generate_img 转换为单通道
    img_gray = torch.mean(img, dim=1, keepdim=True)

    # 使用 Sobel 算子计算边缘损失
    gradient_x = F.conv2d(img_gray, torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                                 dtype=img.dtype, device=img.device).view(1, 1, 3, 3), padding=1)
    gradient_y = F.conv2d(img_gray, torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                                 dtype=img.dtype, device=img.device).view(1, 1, 3, 3), padding=1)
    edge_map = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
    return torch.mean(edge_map)

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)

    # 处理除零错误，避免PSNR为NaN
    if mse.item() == 0.0:
        return float('inf')

    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

def calculate_mse(img1, img2):
    mse = F.mse_loss(img1, img2)
    return mse.item()

def calculate_uiqm(uicm, uism, uiconm):
    # 根据给定的公式计算UIQM
    uiqm = 0.028 * uicm + 0.295 * uism + 3.375 * uiconm
    return uiqm


def calculate_ssim(img1, img2):
    # 将图像转换为灰度
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # 计算 SSIM
    ssim_index, _ = ssim(img1_gray, img2_gray, full=True)

    return ssim_index

def test(config):
    '''  ------测试流程------  '''
    device = torch.device("cuda:" + str(config.cuda_id))
    ckp = torch.load(config.snapshot_pth)
    test_model = ckp["model"]
    # test_model = torch.load(config.snapshot_pth)

    # 测试输入的大小：256*256
    # transform_list = [transforms.Resize((256, 256)), transforms.ToTensor()]
    transform_list = [transforms.ToTensor()]
    tsfm = transforms.Compose(transform_list)

    # 加载测试数据集
    testset = TestDataSet(config.test_pth, tsfm)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False)

    # 为输出创建文件夹
    os.makedirs(config.output_pth, exist_ok=True)

    total_psnr = 0.0
    total_edge_loss = 0.0
    total_mse = 0.0
    total_uicm = 0.0
    total_uism = 0.0
    total_uiconm = 0.0
    total_ssim = 0.0
    total_uiqm = 0.0

    max_psnr = 0.0
    max_edge_loss = 0.0
    max_mse = 0.0
    max_uicm = 0.0
    max_uiconm = 0.0
    max_ssim = 0.0
    max_uism = 0.0
    max_uiqm = 0.0

    for i, (img, name) in enumerate(test_dataloader):
        with torch.no_grad():
            img = img.to(device)
            generate_img, _ = test_model(img)


            # torchvision.utils.save_image(generate_img, config.output_pth + name[0])
            # # 将原始图像和增强后的图像合并为一个图像
            # combined_img = torch.cat([img, generate_img], dim=3)
            # # 保存合并后的图像
            # torchvision.utils.save_image(combined_img, os.path.join(config.output_pth, name[0]))

            # # 保存原始图像
            # torchvision.utils.save_image(img, os.path.join(config.output_pth, "original_" + name[0]))

            # 保存增强后的图像
            torchvision.utils.save_image(generate_img, os.path.join(config.output_pth, name[0]))

            # 计算PSNR
            original_img = img.cpu()
            generated_img = generate_img.cpu()
            psnr = calculate_psnr(original_img, generated_img)
            total_psnr += psnr
            max_psnr = max(max_psnr, psnr)

            # 计算边缘损失
            edge_loss = calculate_edge_loss(generate_img)
            total_edge_loss += edge_loss
            max_edge_loss = max(max_edge_loss, edge_loss.item())

            # 计算MSE
            mse = calculate_mse(original_img, generated_img)
            total_mse += mse
            max_mse = max(max_mse, mse)

            # 计算UICM
            uicm = calculate_uicm(original_img, generated_img)
            total_uicm += uicm
            max_uicm = max(max_uicm, uicm)

            # 计算UICONM
            uiconm = calculate_uiconm(original_img, generate_img)
            total_uiconm += uiconm
            max_uiconm = max(max_uiconm, uiconm)

            # 计算UISM
            uism = calculate_uism(generate_img)
            total_uism += uism
            max_uism = max(max_uism, uism)

            # 计算SSIM
            original_img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            generated_img = generate_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            ssim_value = calculate_ssim(original_img, generated_img)
            total_ssim += ssim_value
            max_ssim = max(max_ssim, ssim_value)

            # 计算 UIQM
            uiqm = calculate_uiqm(uicm, uism, uiconm)
            total_uiqm += uiqm
            max_uiqm = max(max_uiqm, uiqm)

            # 汇总信息字符串
            info_str = '处理图像 [{}/{}], UICONM: {:.4f}, UISM: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}, Edge Loss: {:.4f}, MSE: {:.4f}, UICM: {:.4f}, UIQM: {:.4f}'.format(
                str(i + 1), str(len(testset)), uiconm, uism, psnr, ssim_value, edge_loss, mse, uicm, uiqm)

            print(info_str)

    # 计算每个指标的总的平均值
    average_psnr = total_psnr / len(testset)
    average_edge_loss = total_edge_loss / len(testset)
    average_mse = total_mse / len(testset)
    average_uicm = total_uicm / len(testset)
    average_uiconm = total_uiconm / len(testset)
    average_uism = total_uism / len(testset)
    average_ssim = total_ssim / len(testset)
    average_uiqm = total_uiqm / len(testset)

    # 打印平均值和最大值
    print('平均PSNR: {:.4f}，最大PSNR: {:.4f}'.format(average_psnr, max_psnr))
    print('平均边缘损失: {:.4f}，最大边缘损失: {:.4f}'.format(average_edge_loss, max_edge_loss))
    print('平均MSE: {:.4f}，最大MSE: {:.4f}'.format(average_mse, max_mse))
    print('平均UICM: {:.4f}，最大UICM: {:.4f}'.format(average_uicm, max_uicm))
    print('平均UISM: {:.4f}，最大UISM: {:.4f}'.format(average_uism, max_uism))
    print('平均UICONM: {:.4f}，最大UICONM: {:.4f}'.format(average_uiconm, max_uiconm))
    print('平均UIQM: {:.4f}，最大UIQM: {:.4f}'.format(average_uiqm, max_uiqm))
    print('平均SSIM: {:.4f}，最大SSIM: {:.4f}'.format(average_ssim, max_ssim))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=int, default=0, help='默认: 0')
    parser.add_argument('--snapshot_pth', type=str, default="./checkpoints1/model_epoch_90.pk",
                        help='检查点路径，默认: ./checkpoints1/model_epoch_***.pk')
    parser.add_argument('--test_pth', type=str, default='./data-EUVP/test/',
                        help='测试图像的路径，默认: ./data-EUVP/test/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_pth', type=str, default='./results-EUVP(1)/',
                        help='保存生成图像的路径，默认: ./results-EUVP(1)/')

    config = parser.parse_args()
    test(config)
