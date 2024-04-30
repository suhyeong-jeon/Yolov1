import cv2
from PIL import Image
import glob
import pandas as pd
import torchvision.transforms as transforms
import torch
from model import Yolov1
from utils import cellboxes_to_boxes, non_max_suppression

torch.set_printoptions(profile="full")

classes = {
    '__background__': 0,
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
}
classes = {v:k for k,v in classes.items()}

test_csv_path = '../data/100examples.csv'
origin_image_path = '../data/images'
model = Yolov1()
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def csv_to_data(test_image_path):
    read_csv = pd.read_csv(test_image_path)
    image_list = read_csv.iloc[:,0]
    return image_list

def test(image_list, model):
    test_augmentation = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])

    model.load_state_dict(torch.load('./best.pt'))
    model = model.to(DEVICE)

    for image in image_list:
        image_path = f"{origin_image_path}/{image}"
        image = Image.open(image_path).convert('RGB')

        image_cv2 = cv2.imread(image_path)
        image_cv2 = cv2.resize(image_cv2, (500, 500))

        model.eval()
        with torch.no_grad():
            aug_image = test_augmentation(image)
            aug_image = aug_image.unsqueeze(0)
            aug_image = aug_image.to(DEVICE)

            output = model(aug_image)
            batch_size = output.shape[0]
            bboxes = cellboxes_to_boxes(output)

            for idx in range(batch_size):
                nms_boxes = non_max_suppression(  # nms를 적용시키고 남은 bounding boxes
                    bboxes[idx],
                    iou_threshold=0.5,
                    threshold=0.4,
                    box_format='midpoint',
                )

        print(nms_boxes)

        if len(nms_boxes) == 0:
            cv2.imshow('test', image_cv2)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27:
                break
            cv2.destroyAllWindows()
            continue
        else:
            x_min = int((nms_boxes[0][2] - nms_boxes[0][4] / 2) * 448)
            y_min = int((nms_boxes[0][3] - nms_boxes[0][5] / 2) * 448)
            x_max = int((nms_boxes[0][2] + nms_boxes[0][4] / 2) * 448)
            y_max = int((nms_boxes[0][3] + nms_boxes[0][5] / 2) * 448)
            image_cv2 = cv2.rectangle(image_cv2, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
            image_cv2 = cv2.putText(image_cv2, f'{str(classes[int(nms_boxes[0][0])])}', (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX
                                    , fontScale=0.8, color=(0, 0, 0), thickness=2)
            image_cv2 = cv2.putText(image_cv2, f'{(float(nms_boxes[0][1])): .2f}', (x_min-15, y_min-30),
                                    cv2.FONT_HERSHEY_SIMPLEX
                                    , fontScale=0.8, color=(0, 0, 0), thickness=2)
            cv2.imshow('test', image_cv2)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break
        cv2.destroyAllWindows()







if __name__ == '__main__':
    image_list = csv_to_data(test_csv_path)
    test(image_list, model)

