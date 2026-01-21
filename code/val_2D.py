import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_fracture(image, label, net, classes, patch_size=[224, 224]):
    # 1. 转为 Numpy，去掉 Batch 维度
    # 假设输入是 (1, 1, H, W) -> squeeze 后变成 (H, W)
    image, label = image.squeeze(0).squeeze(0).cpu().detach().numpy(), label.squeeze(0).squeeze(0).cpu().detach().numpy()


    # 2. 这里的 prediction 原本是存 3D 的，现在直接存 2D 结果
    # prediction = np.zeros_like(label) # 这一行其实也可以不要了，直接赋值就行

    # ======================================================
    # ♻️ 核心改动：删掉了 for ind in range(image.shape[0]):
    # 直接把 image 当作 slice 处理
    # ======================================================
    slice = image
    x, y = slice.shape[0], slice.shape[1]

    # 缩放输入 (Order=0 适合 mask，Order=3 适合 image，原作者用的 0 也没问题)
    if x != patch_size[0] or y != patch_size[1]:
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3) 

    # 转 Tensor 送入网络
    input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()

    net.eval()
    with torch.no_grad():
        out = net(input)
        
        # ⚠️ 如果你的模型输出是 2 通道 (num_classes=2) -> 保持原作者逻辑
        out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
        
        # ⚠️ 如果你的模型输出是 1 通道 (BCE) -> 改成下面这行 (二选一)
        # out = (torch.sigmoid(out) > 0.5).float().squeeze(0).squeeze(0)

        out = out.cpu().detach().numpy()

        # debug_dir = '/data/yqh/Code/Segmentation/Semi-supervised/ABD-main/code'
        # out = (out * 255).astype(np.uint8)
        # cv2.imwrite(f"{debug_dir}/out.png", out)
        
        # 还原尺寸
        if x != patch_size[0] or y != patch_size[1]:
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            pred = out
            
    prediction = pred # 直接赋值

    # ⬇️⬇️⬇️ 在这里插入打印代码 (Debug) ⬇️⬇️⬇️
    # print(f"Debug - Prediction Max: {prediction.max()}, Unique: {np.unique(prediction)}")
    # print(f"Debug - GT Max: {label.max()}, Unique: {np.unique(label)}")

    # 3. 计算指标 (保持原样)
    metric_list = []
    metric_list.append(calculate_metric_percase(prediction, label))
    return metric_list

