import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from custom_dataset import VOCDataset
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from loss import Yolov1Loss
from tqdm import tqdm
import time
from model import Yolov1
import cv2
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)

LEARNING_RATE = 1E-4 # ResNet16 : 0.0001, DarkNet : 0.0001
BATCH_SIZE = 16 # 논문에서는 64
EPOCHS = 135
PIN_MEMORY = True
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
LOAD_MODEL = False
LOAD_MODEL_FILE = 'overfit.pth.tar'
IMG_DIR = '../data/images'
LABEL_DIR = '../data/labels'

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes):
        for t in self.transforms:
            img, boxes = t(img), boxes

        return img, boxes

train_transforms = Compose([transforms.Resize((448, 448)), # DarkNet : (448, 448), ResNet : (224, 224)
                            transforms.ToTensor(),
                            transforms.RandomRotation(degrees=3),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])
test_transforms = Compose([transforms.Resize((448, 448)), # DarkNet : (448, 448), ResNet : (224, 224)
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),])

def view_aug(train_dataset):
    fig=plt.figure(figsize=(10,2))
    for i in range(20):
        img = train_dataset[i]
        ax = fig.add_subplot(2, 10,i+1)
        ax.imshow(img[0].permute(1, 2, 0))
        ax.axis('off')
    plt.show()

def main():
    S = 7
    B = 2
    C = 20
    train_loss_list = []
    val_loss_list = []
    best_val_acc = 0.0
    mean_avg_test_prec_list = []
    mean_avg_train_prec_list = []

    # DarkNet
    model = Yolov1()
    model = model.to(DEVICE)

    """
    ResNet16 - Fine Tuning
    """
    # model = resnet18(weights='DEFAULT')
    # model.avgpool = nn.Identity() # resnet18에 있던 avgpool을 사용하지 않음
    # model.fc = nn.Linear(in_features=S*S*512, out_features=S*S*(C+(B*5)))


    model = model.to(DEVICE)

    """
    Torch.Size Test
    """
    # print(model)
    # x = torch.randn((2, 3, 224, 224)).to(DEVICE)
    # print(model(x))

    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = Yolov1Loss()

    train_dataset = VOCDataset('../data/100examples.csv', img_dir=IMG_DIR, label_dir=LABEL_DIR, transform=train_transforms)
    test_dataset = VOCDataset('../data/8examples.csv', img_dir=IMG_DIR, label_dir=LABEL_DIR, transform=test_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, shuffle=False)


    for epoch in range(EPOCHS):
        train_loss = 0.0
        val_loss = 0.0

        loop = tqdm(train_loader, leave=True)
        model.train()
        for batch_idx, (image, label) in enumerate(loop):
            loop.set_description(f'Training')

            image = image.to(DEVICE)
            label = label.to(DEVICE)

            output = model(image)

            loss = loss_fn(output, label)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=f'{loss.item(): .4f}')

        # get_bboxes : # 모델의 eval모드에서의 예측 결과와 실제 바운딩 박스를 가져오는 함수
        pred_boxes, target_boxes, _ = get_bboxes(train_loader, model, loss_fn=loss_fn, iou_threshold=0.5, threshold=0.4)
        mean_avg_prec_train = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5,
                                                    box_format='midpoint')  # True Positive, False Positive로 평균 정밀도를 구해 mAP 계산
        mean_avg_train_prec_list.append(mean_avg_prec_train)

        pred_boxes, target_boxes, val_loss = get_bboxes(test_loader, model, loss_fn=loss_fn, iou_threshold=0.5, threshold=0.4)
        mean_avg_prec_test = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5,
                                               box_format='midpoint')
        mean_avg_test_prec_list.append(mean_avg_prec_test)


        train_loss = train_loss/len(train_loader)
        val_loss = val_loss/len(test_loader)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(f"\nEPOCH : {epoch + 1} / {EPOCHS}, Train Loss : {train_loss: .4f}"
              f", Test Loss : {val_loss: .4f}, Train mAP : {mean_avg_prec_train: .4f}, Test mAP : {mean_avg_prec_test: .4f}")

        if mean_avg_prec_test > best_val_acc:
            torch.save(model.state_dict(), 'best.pt')
            best_val_acc = mean_avg_prec_test
            time.sleep(3)


    plt.plot(train_loss_list, label = 'train loss')
    plt.plot(val_loss_list, label = 'val loss')
    plt.legend()
    plt.savefig('DarkNet_loss_graph.png')
    plt.show()

    plt.plot(mean_avg_test_prec_list, label = 'Testing mAP')
    plt.plot(mean_avg_train_prec_list, label = 'Training mAP')
    plt.title('DarkNet mAP')
    plt.savefig('DarkNet_mAP_graph.png')
    plt.show()

if __name__ == '__main__':
    main()