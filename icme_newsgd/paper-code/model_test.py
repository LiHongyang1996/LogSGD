import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np

cuda = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)

# 定义 BasicBlock 和 ResNet 类（您已经提供的代码）
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if not use_batchnorm:
            self.bn1 = self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes) if use_batchnorm else nn.Sequential()
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_batchnorm=True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.use_batchnorm = use_batchnorm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if use_batchnorm else nn.Sequential()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.use_batchnorm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(use_batchnorm=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], use_batchnorm=use_batchnorm)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据预处理
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR-10 的均值和标准差
])

# 加载 CIFAR-10 测试集
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

# 实例化模型并加载权重
model = ResNet18(use_batchnorm=True).to(device)


# 检查权重文件是否存在
checkpoint_path = f'/code/fileofpower/final_checkpoint_cifar10_{sys.argv[2]}_{sys.argv[3]}.pth'

if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(f"权重文件未找到: {checkpoint_path}")

# 加载权重
checkpoint = torch.load(checkpoint_path, map_location=device)['runavg_state_dict']

# 如果保存的是 state_dict
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

# 设置模型为评估模式
model.eval()

# 定义评估函数
def evaluate(model, dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():  # 不需要计算梯度
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = 100. * correct / total
    return accuracy

# 计算准确率
accuracy = evaluate(model, test_loader, device)
# print(f"CIFAR-10_{sys.argv[2]}_{sys.argv[3]}_ 测试集准确率: {accuracy:.2f}%")



transform = transforms.Compose([
    transforms.ToTensor(),
    # 标准化（可选）
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
# 加载CIFAR-10训练和测试数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 初始化存储结果的列表
train_losses = []
train_predictions = []
train_labels = []

test_losses = []
test_predictions = []
test_labels = []

# 计算训练数据的loss、cross-entropy和prediction
with torch.no_grad():
    for inputs, labels in trainloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 计算预测概率（Softmax）
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        train_losses.append(loss.item())
        train_predictions.append(probabilities.cpu().numpy())
        train_labels.append(labels.cpu().numpy())

# 计算测试数据的loss、cross-entropy和prediction
with torch.no_grad():
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 计算预测概率（Softmax）
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        test_losses.append(loss.item())
        test_predictions.append(probabilities.cpu().numpy())
        test_labels.append(labels.cpu().numpy())

# 将列表转换为numpy数组
Ltrain = np.array(train_losses)
Ltest = np.array(test_losses)
Ptrain = np.vstack(train_predictions)
Ptest = np.vstack(test_predictions)
Ytrain = np.concatenate(train_labels)
Ytest = np.concatenate(test_labels)


# 定义计算修改的熵的函数
def compute_modified_entropy(p, y, epsilon=1e-5):
    """Computes label-informed entropy as per 'Systematic evaluation of privacy risks of machine learning models' USENIX21"""
    n = len(p)
    entropy = np.zeros(n)
    for i in range(n):
        pi = p[i]
        yi = y[i]
        p_yi = pi[yi]

        # 计算正确类别的项
        term1 = (1 - p_yi) * np.log(p_yi + epsilon)

        # 计算错误类别的项
        term2 = 0.0
        for j, p_j in enumerate(pi):
            if j != yi:
                term2 += p_j * np.log(1 - p_j + epsilon)

        # 计算总熵（注意符号）
        entropy[i] = - (term1 + term2)
    return entropy


# 定义阈值搜索空间函数
def ths_searching_space(nt, train, test):
    """ 定义阈值搜索空间，在给定的指标的最大值和最小值之间取nt个点 """
    thrs = np.linspace(
        min(train.min(), test.min()),
        max(train.max(), test.max()),
        nt
    )
    return thrs


# 定义搜索最佳阈值的函数
def search_optimal_threshold(train_scores, test_scores, thrs):
    """
    搜索最佳阈值以最大化攻击准确率。

    参数：
    - train_scores：训练集上的指标值
    - test_scores：测试集上的指标值
    - thrs：阈值列表

    返回：
    - max_acc：最大攻击准确率
    """
    n_train = len(train_scores)
    n_test = len(test_scores)
    labels = np.concatenate([np.ones(n_train), np.zeros(n_test)])  # 1表示成员，0表示非成员
    scores = np.concatenate([train_scores, test_scores])

    max_acc = 0.0
    for th in thrs:
        preds = (scores <= th).astype(int)  # 如果指标值小于等于阈值，则预测为成员
        acc = np.mean(preds == labels)
        if acc > max_acc:
            max_acc = acc
    return max_acc


# 定义成员推理攻击函数
def mia_attack(Ltrain, Ltest, Ptrain, Ptest, Ytrain, Ytest, nt=150):
    """
    执行基于阈值的成员推理攻击，并计算攻击成功率。

    参数：
    - Ltrain：训练集上的损失值数组
    - Ltest：测试集上的损失值数组
    - Ptrain：训练集上的预测概率数组
    - Ptest：测试集上的预测概率数组
    - Ytrain：训练集上的真实标签数组
    - Ytest：测试集上的真实标签数组
    - nt：阈值搜索的步数

    返回：
    - loss_attack_acc：基于损失的攻击准确率
    - entropy_attack_acc：基于熵的攻击准确率
    """
    # 确保测试集和训练集的大小相同
    n = min(len(Ltrain), len(Ltest))
    n=3000
    Ltrain = Ltrain[:n]
    Ltest = Ltest[:n]
    Ptrain = Ptrain[:n]
    Ptest = Ptest[:n]
    Ytrain = Ytrain[:n]
    Ytest = Ytest[:n]

    # 基于损失的成员推理攻击
    thrs = ths_searching_space(nt, Ltrain, Ltest)
    loss_attack_acc = search_optimal_threshold(Ltrain, Ltest, thrs)

    # 计算修改的熵
    Etrain = compute_modified_entropy(Ptrain, Ytrain)
    Etest = compute_modified_entropy(Ptest, Ytest)

    # 基于熵的成员推理攻击
    thrs = ths_searching_space(nt, Etrain, Etest)
    entropy_attack_acc = search_optimal_threshold(Etrain, Etest, thrs)

    return loss_attack_acc, entropy_attack_acc


# 执行成员推理攻击
loss_attack_acc, entropy_attack_acc = mia_attack(Ltrain, Ltest, Ptrain, Ptest, Ytrain, Ytest, nt=150)

print(f"cifar10_{sys.argv[2]}_{sys.argv[3]}_基于损失的攻击成功率：{loss_attack_acc * 100:.2f}%")
print(f"cifar10_{sys.argv[2]}_{sys.argv[3]}_基于熵的攻击成功率：{entropy_attack_acc * 100:.2f}%")
print(f"CIFAR-10_{sys.argv[2]}_{sys.argv[3]}_ 测试集准确率: {accuracy:.2f}%")

loss_result = f"cifar10_{sys.argv[2]}_{sys.argv[3]}_基于损失的攻击成功率：{loss_attack_acc * 100:.2f}%"
entropy_result = f"cifar10_{sys.argv[2]}_{sys.argv[3]}_基于熵的攻击成功率：{entropy_attack_acc * 100:.2f}%"
acc = f"CIFAR-10_{sys.argv[2]}_{sys.argv[3]}_ 测试集准确率: {accuracy:.2f}%"
file_name = f'cifar10_{sys.argv[2]}_{sys.argv[3]}.txt'
with open(file_name, 'w') as file:
    file.write(acc)
    file.write(loss_result)
    file.write(entropy_result)