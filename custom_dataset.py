from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

"""
Process 요악 - 최종편

VOC dataset의 label은 이미지의 크기가 (1 * 1)이라고 했을 때 object의 x, y, w, h를 이미지의 크기를 기준으로 작성한 것이다.
이 x, y, w, h가 VOCDataset을 통해 변환되는데, 먼저 image를 7 * 7 크기의 grid cell(각 grid cell은 1*1크기)로 나눈 상태로 생각을 해야한다. object의 중심좌표가 이미지의 49개의 grid cell중 
어느 grid cell에  있는지 먼저 찾는다. 그리고 중심좌표 (x, y)가 grid cell에 있다면 0~1크기의 grid cell기준으로 어느 좌표에 있는지를 다시 변환한다.(기존에는 원본 이미지 기준으로 중심좌표가 정해져있었음)
그리고 w, h는 7*7 grid cell 크기에 맞게 변환한다. 이렇게 Yolov1 format x, y, w, h가 만들어지게된다.
"""

class VOCDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file) # 100examples
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S # number of grid cell
        self.B = B # number of bounding box
        self.C = C # number of class
        self.transform = transform
        # print(self.annotations.iloc[0, 1])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # 행번호로 선택하는 iloc -> 즉 index를 따라 label.txt의 데이터를 선택, 즉 label_path는 index에 따른 label.txt의 경로가 됨.
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]) # 000007.jpeg 000007.txt
        boxes = []

        # dataset의 label format 데이터 전처리
        with open(label_path) as f:
            for labels in f.readlines(): # splitted_label = ['14', '0.7106666666666667', '0.644', '0.5786666666666667', '0.712\n']
                splitted_label = labels.split(' ')
                class_name, x, y, w, h,  = 0, 0, 0, 0, 0

                for idx, label in enumerate(splitted_label): # index명을 idx말고 index로 설정했다가 image_path부분에서의 index가 꼬여서 image의 tensor가 이상하게 출력됬음
                    if idx == 0:
                        class_name = int(label)
                    else:
                        label = float(label) if float(label) != int(float(label)) else int(label)

                        if idx == 1:
                            x = label
                        elif idx == 2:
                            y = label
                        elif idx == 3:
                            w = label
                        elif idx == 4:
                            h = label

                boxes.append([class_name, x, y, w, h])

        image_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(image_path)
        boxes = torch.tensor(boxes) # transformed된 이미지에 맞춰 boxes도 변형시키기위해 tensor로 변환


        if self.transform:
            image, boxes = self.transform(image, boxes)

        # model을 통해 나온 최종 feature map의 크기인 7 * 7 * 30(class 20개, 5개의 label * 각 grid cell이 예측하는 bounding box 수)
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        # Yolov1 논문에 제시딘 7 * 7 개의 grid cells의 형태로 데이터를 변환하는 과정(7 * 7 grid cells, 각 셀마다 예측하는 bounding box수 2개, class 수 20개
        # 즉 기존 dataset을 yolo format dataset으로 변형하는 과정
        for box in boxes:
            class_label, x, y, w, h = box.tolist()
            class_label = int(class_label)

            # label의 bounding box가 어느 grid cell에 있는지 찾아야함.
            # 우리가 사용하는 데이터셋의 label과 image를 보면 image를 (0, 0)부터 (1, 1)의 크기로 고정한 뒤
            # bounding box의 x, y, w, h를 0~1 사이의 값으로 둔것을 확인 할 수 있다.
            # https://velog.io/@jaeha0725/Object-Detection-Label-COCO-YOLO-KITTI
            i = int(self.S * y)
            j = int(self.S * x) # 중심좌표를 포함하는 grid cell의 index

            # 이 수식은 19.jpeg이미지와 label을 보면 이해 가능
            x_grid_cell = self.S * x - j
            y_grid_cell = self.S * y - i
            w_grid_cell = self.S * w
            h_grid_cell = self.S * h


            if label_matrix[i, j, 20] == 0: # 7 * 7 크기의 grid cell중 yolov1 format bounding box의 중심좌표가 속한 i, j 행렬의 grid cell에 1이라고 표시를함
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor([x_grid_cell, y_grid_cell, w_grid_cell, h_grid_cell])
                label_matrix[i, j, 21:25] = box_coordinates # yolov1 format bounding box coordinates를 대입
                label_matrix[i, j, class_label] = 1 # class_label 지정.

        # label_matrix[..., 20]은 이미지를 7 * 7 grid cell로 나누었을때 어느 grid cell에 bounding box의 중심좌표가 있는지를 0과 1(존재)로 나타낸 tensor다.
        # print(label_matrix[...].shape) # torch.Size([7, 7, 30])
        # print(label_matrix[..., 20].shape) # torch.Size([7, 7])
        # print(label_matrix[..., 20])
        #print(label_matrix[..., 21:25]) # bounding box의 중심 좌표가 포함된 grid cel의 x, y, w, h값이 적혀있음
        #print(label_matrix[..., 21:25].shape) # torch.Size([7, 7, 4])


        return image, label_matrix


if __name__ == '__main__':
    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, img, boxes):
            for t in self.transforms:
                img, boxes = t(img), boxes

            return img, boxes


    train_transforms = Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])
    test_transforms = Compose([transforms.Resize((224, 224)), transforms.ToTensor(), ])
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    b = VOCDataset('../data/8examples.csv', '../data/images', '../data/labels', transform=train_transforms, )
    a = DataLoader(b, batch_size=16, shuffle=False)
    torch.set_printoptions(profile='full')
    for idx, (x, y) in enumerate(a):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        print(x)
        break