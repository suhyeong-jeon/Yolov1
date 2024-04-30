import torch
import torch.nn as nn
from utils import intersection_over_union
"""
fowward 초반 부분 final 요약
[batch크기, 1470]크기의 기존 predictions을 [batch크기, 7, 7, 30]의 모양으로 변환. 즉 학습에 사용한 사진의 갯수에서 각 사진의 grid cell당 class20개, bounding box2개에 대한 
confidence score, x, y, w, h 정보를 포함한게 새로운 predictions이다.
iou_b1, iou_b2는 모든 grid cell에 대한 x, y, w, h와 실제 target의 x, y, w, h의 iou 값
exists_box는 target[..., 20], 즉 학습에 사용한 모든 사진에서의 ground truth box의 중심좌표가 어디있는지 표시한 tensor의 모양을 predictions[..., 21:25],
predictions[..., 26:30]과 연산시키기 위해 변형시킨 값이다.
"""


class Yolov1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(Yolov1Loss, self).__init__()
        self.sse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    # predictions : darknet이 최종적으로 산출하는 7*7*(20+5*2)크기의 flatten된 feature map
    def forward(self, predictions, target):
        #print(predictions.shape)
        # [batch크기, 1470]크기의 기존 predictions을 self.S, self.S, self.C + self.B * 5개([batch크기, 7,7,30])의 모양으로 변환.
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        # print(predictions.shape)

        # predictions[..., 21:25]는 첫번째 bounding box의 좌표, 정답과 비교해 IoU 계산
        # predictions[..., 21:25].shape = torch.Size([16, 7, 7, 4])
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) # custom_dataset에 어떤 구조이고 어떤 정보를 포함하는지 적어둠
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) # dim=0 방향으로 텐서를 쌓음. dim=0은 세로, dim=1은 가로
        iou_maxes, bestbox = torch.max(ious, dim=0) # dim=0방향으로 각각의 tensor의 최대값을 반환. 즉 두 iou중 target과 더 많이 겹치는 iou를 선택함

        # exists_box는 논문에서의 1obj_ij임
        # torch.Size((16, 7, 7, 1]), 즉 7 * 7 크기의 grid cell중 어디가 ground truth box의 중심좌표가 있는지 적혀있는 tensor를 predictions[..., 21:25]와 연산시키기위해 unsqueeze(3)을 적용
        exists_box = target[..., 20].unsqueeze(3) # target데이터를 활용해 이미지에 ground truth box의 중심의 존재 여부를 확인. 존재한다면 exists_box는 1 아니면 0


        # Localization Loss 계산

        # 예측한 bounding box 좌표값
        box_predictions = exists_box * (
            (
                    bestbox * predictions[..., 26:30]
                    + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        # 실제 bounding box 좌표값
        box_targets = exists_box * target[..., 21:25]

        # w, h에 루트
        # torch.sign은 음수인경우 -1, 0이면 0, 양수면 1로 변환시켜줌.
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # 아직 lambda_coordj는 안곱해준 상태
        box_loss = self.sse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )


        # Confidence Loss

        # confidence score for the bounding box with highest IoU
        pred_box = bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]


        object_loss = self.sse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )


        # Confidence loss for No Object Loss(Object가 없을 경우의 confidence loss)

        no_object_loss = self.sse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.sse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )


        # For Class Loss

        class_loss = self.sse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )


        # 최종 Loss
        loss = self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_object_loss + class_loss

        return loss


if __name__ == '__main__':
    a = torch.tensor([0.5])
    b= torch.tensor([0.3])
    print(torch.cat([a.unsqueeze(0), b.unsqueeze(0)], dim=0))
    """
    tensor([[0.5000],
            [0.3000]])
    """