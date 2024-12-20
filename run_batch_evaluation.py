import os
import sys
import csv
import re
from pathlib import Path
from tqdm import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import gdown
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TEMP_DIR = "./"

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out

class ResNet(nn.Module):
    def __init__(self, depth, num_filters, block_name="BasicBlock", num_classes=100):
        super(ResNet, self).__init__()
        n = (depth - 2) // 6
        block = BasicBlock
        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)
        self.stage_channels = num_filters

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, is_last=(blocks == 1))
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_last=(i == blocks - 1)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x, f1_pre = self.layer1(x)
        x, f2_pre = self.layer2(x)
        x, f3_pre = self.layer3(x)
        x = self.avgpool(x)
        avg = x.reshape(x.size(0), -1)
        out = self.fc(avg)
        return out

def resnet8x4(**kwargs):
    return ResNet(8, [32, 64, 128, 256], "basicblock", **kwargs)

def resnet32x4(**kwargs):
    return ResNet(32, [32, 64, 128, 256], "basicblock", **kwargs)


class BaseTrainer:
    pretrained_teacher_link = (
        "https://drive.google.com/uc?id=1Gh3Z8BZ62PGD7PQiFiwmU9vMwMpF5F46"
    )

    def __init__(self):
        self.teacher = resnet32x4(num_classes=100)
        self.student = resnet8x4(num_classes=100)
        os.makedirs(TEMP_DIR, exist_ok=True)
        try:
            gdown.download(
                self.pretrained_teacher_link, os.path.join(TEMP_DIR, "resnet_32x4.pth"), resume=True
            )
        except Exception:
            assert os.path.exists(os.path.join(TEMP_DIR, "resnet_32x4.pth"))

        self.teacher.load_state_dict(
            torch.load(os.path.join(TEMP_DIR, "resnet_32x4.pth"), map_location="cpu")["model"]
        )

        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

        self.train_set = datasets.CIFAR100(
            os.path.join(TEMP_DIR, "data/"), download=True, train=True, transform=self.train_transform
        )
        self.test_set = datasets.CIFAR100(
            os.path.join(TEMP_DIR, "data/"), download=False, train=False, transform=self.test_transform
        )
        self.test_dataloader = DataLoader(self.test_set, batch_size=64, shuffle=False)

    def save_student_checkpoint(self, ckpt_path):
        state_dict = self.student.state_dict()
        torch.save(state_dict, ckpt_path)

    def load_student_checkpoint(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        state_dict = {k.removeprefix("module."): v for k,v in state_dict.items()}
        self.student.load_state_dict(state_dict)

    @torch.no_grad()
    def evaluate_student(self):
        self.student.cuda().eval()
        n = 0
        correct = 0
        for image, target in tqdm(self.test_dataloader, desc="Evaluating Student", leave=False):
            image = image.cuda()
            target = target.cuda()
            output = self.student(image)
            n += image.size(0)
            correct += output.max(-1).indices.eq(target).sum().item()
        accuracy = 100 * correct / n
        return accuracy

    def train_student(self):
        pass


if __name__ == "__main__":
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
    root_path = Path(root_dir)

    pattern = re.compile(r"(\d+)\((.*?)\)_\d+_assignsubmission_file_")
    output_file = root_path / "student_accuracy_evaluation.csv"
    fieldnames = ["student_id", "student_name", "baseline_acc", "improved_KD"]

    trainer = BaseTrainer()

    # 학생 디렉토리에서 학번과 이름 추출, 정렬
    student_dirs = []
    for entry in root_path.iterdir():
        if entry.is_dir():
            match = pattern.search(entry.name)
            if match:
                s_id = match.group(1)
                s_name = match.group(2)
                # 학번 오름차순 정렬을 위해 int 변환
                student_dirs.append((int(s_id), s_name, entry))
    student_dirs.sort(key=lambda x: x[0])

    # CSV 파일을 UTF-8 BOM으로 인코딩
    with open(output_file, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for student_id, student_name, entry in tqdm(student_dirs, desc="Processing Students"):
            baseline_ckpt = entry / "student_checkpoint.pth"
            improved_ckpt = entry / "student_improved_checkpoint.pth"

            baseline_acc = ""
            improved_acc = ""

            # baseline checkpoint 평가
            if baseline_ckpt.exists():
                try:
                    trainer.load_student_checkpoint(str(baseline_ckpt))
                    baseline_acc = trainer.evaluate_student()
                except Exception:
                    baseline_acc = ""

            # improved checkpoint 평가
            if improved_ckpt.exists():
                try:
                    trainer.load_student_checkpoint(str(improved_ckpt))
                    improved_acc = trainer.evaluate_student()
                except Exception:
                    improved_acc = ""

            writer.writerow({
                "student_id": student_id,
                "student_name": student_name,
                "baseline_acc": baseline_acc,
                "improved_KD": improved_acc
            })
