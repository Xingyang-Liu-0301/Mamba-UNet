import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy.ndimage.interpolation import zoom
import random
from scipy.ndimage import binary_erosion

# ---------------------------------------------------------------------------
# 这是一个专门为你重写的 Dataset 类，用于读取 png 格式的弱监督数据
# ---------------------------------------------------------------------------
class ACDCDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="skeletonized_labelcol"):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type # 这里会接收你在 train.py 里写的 'skeletonized_labelcol'
        self.transform = transform

        # 读取 txt 文件列表 (和 AD-MT 逻辑保持一致)
        if self.split == 'train':
            # 确保你的数据集目录下有 train.txt
            with open(self._base_dir + '/train.txt', 'r') as f:
                self.sample_list = [line.strip() for line in f.readlines()]
        elif self.split == 'val':
            # 确保你的数据集目录下有 val.txt
            with open(self._base_dir + '/val.txt', 'r') as f:
                self.sample_list = [line.strip() for line in f.readlines()]
        
        print("Dataset split: {}, total {} samples".format(self.split, len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        
        # 1. 构建路径
        img_path = os.path.join(self._base_dir, "img", "{}.png".format(case))
        gt_path = os.path.join(self._base_dir, "labelcol", "{}.png".format(case))
        weak_label_path = os.path.join(self._base_dir, self.sup_type, "{}.png".format(case))

        # 2. 读取图片和真值
        # 注意：这里 image 保持 0-255 的 uint8 格式方便做形态学处理，后面再归一化
        image_uint8 = np.array(Image.open(img_path).convert('L'))
        gt = np.array(Image.open(gt_path).convert('L'))

        # 3. 归一化 (0-255 -> 0-1) 用于送入网络
        image = image_uint8.astype(np.float32) / 255.0
        
        # 处理 GT (验证用): 0背景, 1骨折
        if gt.max() > 1:
            gt = (gt > 128).astype(np.int64)
        else:
            gt = (gt > 0.5).astype(np.int64)

        # 4. 根据模式分流
        if self.split == "train":
            # [训练模式]: 必须构建包含 Ignore(-100) 的标签
            
            # A. 初始化 label 全为 -100 (代表未知/忽略)
            label = np.ones_like(gt) * -100
            
            if os.path.exists(weak_label_path):
                weak_label_img = np.array(Image.open(weak_label_path).convert('L'))
                
                # --- B. 标记前景 (骨折线) ---
                # 你的骨折线是白色的(255)，设为 1
                mask_fracture = weak_label_img > 128
                label[mask_fracture] = 1 

                # --- C. 自动生成背景 (针对超声散斑噪声优化) ---
                # 你的参数: Percentile=95, MaxCap=180, Erosion=1
                
                # 1. 计算自适应阈值
                # np.percentile(..., 95) 意味着比95%的像素都亮的值
                # 这在超声里通常意味着我们要把除了极亮(骨头)以外的所有东西都当背景
                bg_percentile = 95
                calculated_thresh = np.percentile(image_uint8, bg_percentile)
                
                # 2. 应用安全上限 (180)
                final_thresh = min(calculated_thresh, 180)
                
                # 3. 生成掩膜 (小于阈值的都是背景)
                mask_bg_heuristic = image_uint8 < final_thresh
                
                # 4. 腐蚀操作 (去除散斑)
                # iterations=1 是你调试出来的最佳值
                mask_bg_heuristic = binary_erosion(mask_bg_heuristic, iterations=1)

                # 5. 确保背景不覆盖前景
                mask_bg_final = mask_bg_heuristic & (~mask_fracture)
                
                # 6. 将确定的背景设为 0
                label[mask_bg_final] = 0
                
            else:
                # 只有当没有弱标签文件时，才勉强用 GT (仅调试)
                label = gt.copy()
            
            # D. 类型安全转换
            # 此时 label 里有: 1(骨折), 0(确定的暗背景), -100(散斑/模糊边界/骨头其他部分)
            # 我们不希望 RandomGenerator 修改这些数值，所以最好在这里先手动处理一下
            # 但为了配合你的框架，我们构造 sample
            
            sample = {'image': image, 'label': label, 'gt': gt}
            
            # ⚠️ 注意：你的 RandomGenerator 里的 zoom 插值可能会破坏 -100 这个整数
            # 所以在 transform 之后，我们需要把 label 重新修整回 long 类型
            if self.transform:
                sample = self.transform(sample)
                
            # 确保 label 是 Long 类型，且 -100 没有因为插值变成 -99 或 -101
            if isinstance(sample['label'], torch.Tensor):
                # 只要是负数，都强制归位为 -100 (防止插值产生 -50 这种数)
                mask_ignore = sample['label'] < 0
                sample['label'][mask_ignore] = -100
                sample['label'] = sample['label'].long()
            
        else:
            # [验证模式]
            image = torch.tensor(np.ascontiguousarray(image), dtype=torch.float32).unsqueeze(0)
            label = torch.tensor(np.ascontiguousarray(gt), dtype=torch.long)
            sample = {'image': image, 'label': label}

        sample["idx"] = case
        return sample



# 保持原来的随机增强类不变，因为 train.py 里用到了它
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    # def __call__(self, sample):
    #     image, label, gt = sample['image'], sample['label'], sample['gt']
        
    #     # 简单的随机翻转和旋转
    #     if random.random() > 0.5:
    #         image, label, gt = random_rot_flip(image, label, gt)
    #     elif random.random() > 0.5:
    #          image, label, gt = random_rotate(image, label, gt)
             
    #     # Resize 到网络需要的尺寸
    #     x, y = image.shape
    #     image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
    #     label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
    #     gt = zoom(gt, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
    #     # 转成 Tensor
    #     # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
    #     # label = torch.from_numpy(label.astype(np.uint8))
    #     # gt = torch.from_numpy(gt.astype(np.uint8))
    #     # 1. 先用 np.ascontiguousarray 强制把数组内存变连续，并指定类型
    #     image = np.ascontiguousarray(image, dtype=np.float32)
    #     label = np.ascontiguousarray(label, dtype=np.uint8)
    #     gt = np.ascontiguousarray(gt, dtype=np.uint8)

    #     # 2. 然后用 torch.from_numpy 安全转换
    #     image = torch.from_numpy(image).unsqueeze(0)
    #     label = torch.from_numpy(label)
    #     gt = torch.from_numpy(gt)
        
    #     sample['image'], sample['label'], sample['gt'] = image, label, gt
    #     return sample
    def __call__(self, sample):
        image, label, gt = sample['image'], sample['label'], sample['gt']
        
        # 1. 随机增强
        if random.random() > 0.5:
            image, label, gt = random_rot_flip(image, label, gt)
        elif random.random() > 0.5:
             image, label, gt = random_rotate(image, label, gt)
             
        # 2. 缩放 (保持 order=0 最近邻插值，防止 -100 被插值成其他奇怪的数)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        gt = zoom(gt, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # 3. 🛡️ 关键修复：类型转换
        # image 保持 float32 不变
        image = np.array(image).astype(np.float32).copy()
        
        # ❌ 原来的错误写法：label = np.array(label).astype(np.uint8).copy() 
        # ✅ 修复后：使用 int16 或 int64 来容纳 -100
        label = np.array(label).astype(np.int16).copy()
        gt = np.array(gt).astype(np.int16).copy()

        # 4. 转 Tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        # PyTorch 的标签需要 Long 类型，这里转换很安全
        label = torch.tensor(label, dtype=torch.long)
        gt = torch.tensor(gt, dtype=torch.long)
        
        sample['image'], sample['label'], sample['gt'] = image, label, gt
        return sample




# 辅助函数：翻转
def random_rot_flip(image, label, gt):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    gt = np.rot90(gt, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    gt = np.flip(gt, axis=axis).copy()
    return image, label, gt

# 辅助函数：旋转
from scipy import ndimage
def random_rotate(image, label, gt):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    gt = ndimage.rotate(gt, angle, order=0, reshape=False)
    return image, label, gt