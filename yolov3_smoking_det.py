import cv2
import torchvision
from torchvision.transforms.transforms import Normalize
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
import torch
import os
import numpy as np
from .smoking_utils.yolov3 import Exp, DETECT_CLASSES
import torch.nn.functional as F
from .smoking_utils.visualize import vis
import xml.etree.ElementTree as ET
from xml.dom import minidom
from custom_nodes.comfy_saul_plugin.img_tools import get_folder_file_name
import hydra
from omegaconf import DictConfig, OmegaConf

SCORE_THRESHOLD=0.6
P_START=0.1
P_END=0.1
CONFIG_PATH = "/home/saul/comfy/ComfyUI/custom_nodes/comfy_saul_plugin/run/config/default.yaml"
PATH = f'/home/saul/AIGC/generated_data/'
REQUIRED_CLASSES = ["smoke", "cup"]

class SmokingAutoLabel(ComfyNodeABC):

    DESCRIPTION = 'GIven Images, use yolov3 to label them, and create label file.'
    CATEGORY = 'Saul'

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            'required': {
                'image_in': (IO.IMAGE, {}),
            }
        }
    
    RETURN_TYPES = (IO.IMAGE, )
    RETURN_NAMES = ('image_out', )
    FUNCTION = 'main'
    

    def main(self, image_in: torch.Tensor) -> tuple[torch.Tensor, ]:
        # RGB → BGR
        # image_in = image_in[..., [2, 1, 0]]

        # make dir to store generated data
        print(f"det node input shape: {image_in.shape}")
        if len(image_in.shape) == 3:
            image_in = image_in.unsqueeze(0)
        if isinstance(image_in, torch.Tensor):
            if image_in.dim() == 4:
                if image_in.shape[1] == 3:  # NCHW → NHWC
                    image_in = image_in.permute(0, 2, 3, 1).contiguous()
                elif image_in.shape[-1] != 3:
                    raise ValueError(f"Unexpected tensor shape: {image_in.shape}")
            else:
                raise ValueError(f"Expected 4D tensor, got {image_in.dim()}D")
        # print(f"In frames: {image_in.shape[0]}")
        # if image_in.shape[0] > 20:  # avoid tensor shape like [0, H, W, C]
        #     image_in = image_in[int(image_in.shape[0] * P_START):-int(image_in.shape[0] * P_END)]
        # print(f"Out frames: {image_in.shape[0]}")

        # load model
        exp = Exp()
        model = exp.get_model()
        model_path = '/home/saul/comfy/ComfyUI/models/saul_model/best_ckpt_pscch.pth'
        model.load_state_dict(torch.load(model_path)["model"], strict=False)
        model.eval()
        model.cuda()
        # image_in = image_in.cuda()
        _, H, W, _ = image_in.shape

        # preprocess the image
        img_padding = []
        for i in range(image_in.shape[0]):
            tmp, r = preproc(image_in[i].clone(),(640, 640))
            img_padding.append(tmp)
        img_padding = torch.stack(img_padding, dim=0)  # [N, H

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        mean = [v for v in mean]
        std = [v for v in std]
        img_padding = Normalize(mean, std)(img_padding.permute(0, 3, 1, 2))  # [N, C, H, W]
        img_padding = img_padding.permute(0, 2, 3, 1)  # [N, H, W, C]


        # avoid OOM
        # res = model(img_padding.clone().permute(0, 3, 1, 2).cuda())
        results = []
        # print(img_padding.squeeze().shape)
        model.eval()
        for i in range(img_padding.shape[0]):
            img = img_padding[i].clone().permute(2, 0, 1).unsqueeze(0).cuda()  # [1, 3, H, W]
            print(img.shape, img.dtype, img.min(), img.max())
            with torch.no_grad():  
                out = model(img)
            results.append(out.cpu()) 
        res = torch.cat(results, dim=0)

        outputs = postprocess(
            res, len(DETECT_CLASSES), 0.3, 0.3, class_agnostic=True
        )
        # print(outputs)
        x0, x1, y0, y1 = 0, H, 0, W
        res = []
        cfg = OmegaConf.load(CONFIG_PATH)
        root_path = f'{PATH}/{cfg.task}/{get_folder_file_name()}'
        os.makedirs(root_path, exist_ok=False)
        for i in range(len(outputs)):
            x = image_in[i]
            y = outputs[i]
            if y is None:
                res.append(x)
                continue
            y = y.cpu()
            bboxes = y[:, 0:4]
            bboxes /= r

            cls = y[:, 6]
            scores = y[:, 4] * y[:, 5]

            # filter
            cls_str = [DETECT_CLASSES[int(v)] for v in list(cls)]
            smoke_cup_flag = False
            for j in range(len(cls_str)):
                if cls_str[j] in REQUIRED_CLASSES:
                    print(f"Found {cls_str[j]} in {j}, score: {scores[j]}")
                    smoke_cup_flag = (scores[j] > SCORE_THRESHOLD) or (smoke_cup_flag)
            if smoke_cup_flag == False:
                print(f"Skip image {i} because of low score: {scores}")
                continue

            x = np.array(x)
            x = np.ascontiguousarray(x)
            # print(np.max(x), np.min(x), x.shape, type(x))

            vis_res = vis(x, bboxes, scores, cls, 0.3, DETECT_CLASSES)
            vis_res = torch.from_numpy(vis_res)
            # Save image and label (box)
            file_name = get_folder_file_name()
            img_original = image_in[i].permute(2, 0, 1)
            torchvision.utils.save_image(img_original, f'{root_path}/{file_name}.jpg')
            torchvision.utils.save_image(vis_res.permute(2, 0, 1), f'{root_path}/{file_name}_box.jpg')
            image_size = image_in[i].shape  # width, height, depth
            # print(cls)
            boxes = []
            for i in range(cls.shape[0]):
                boxes.append(
                {"name": DETECT_CLASSES[int(cls[i])], 
                 "xmin": int(bboxes[i][0]), "ymin": int(bboxes[i][1]), "xmax": int(bboxes[i][2]), "ymax": int(bboxes[i][3])}
                )
            xml_content = create_voc_xml(f'{file_name}.jpg', image_size, boxes)
            with open(f"{root_path}/{file_name}.xml", "w", encoding="utf-8") as f:
                f.write(xml_content)
            res.append(vis_res)
        res = torch.stack(res, dim=0)
        print(f"Final frames: {res.shape[0]}")
        return res,


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def preproc(image: torch.Tensor, input_size, swap=(2, 0, 1)):
    H, W, C = image.shape
    image_np = np.array(image * 255.).astype(np.uint8)
    image_padding_np = np.ones((input_size[0], input_size[1], C), dtype=np.uint8) * 114
    r = min(input_size[0] / H, input_size[1] / W)
    H_new, W_new = int(H * r), int(W * r)
    # print(f"Resize ratio: {r}, new size: ({H_new}, {W_new})")
    # print("image.shape:", image_np.shape, "image type", image_np.dtype)
    image_np_resize = cv2.resize(image_np.squeeze(), (W_new, H_new), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    image_padding_np[:H_new, :W_new, :] = image_np_resize
    image_padding_np = np.ascontiguousarray(image_padding_np)
    image_padding = torch.from_numpy(image_padding_np / 255.).float()
    return image_padding, r



def create_voc_xml(filename, image_size, boxes, folder="GENERATED"):
    """
    参数:
        filename: "000001.jpg"
        image_size: (width, height, depth)
        boxes: list of dicts, 每个 dict 含有 {name, xmin, ymin, xmax, ymax}
    返回:
        xml 字符串（可保存为 .xml 文件）
    """
    width, height, depth = image_size

    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    for box in boxes:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = box["name"]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(box["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(box["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(box["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(box["ymax"])

    # 美化格式
    xml_str = ET.tostring(annotation, encoding="utf-8")
    parsed = minidom.parseString(xml_str)
    return parsed.toprettyxml(indent="  ")
