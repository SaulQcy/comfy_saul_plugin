from comfy.comfy_types.node_typing import IO, InputTypeDict, TypedDict, ComfyNodeABC

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


import numpy as np

def select_main_person(pose, img_w=1280, img_h=720):
    center = np.array([img_w / 2, img_h / 2])
    min_dist = float('inf')
    main = None
    for person in pose[0]['people']:
        raw = person.get('face_keypoints_2d', [])
        if not isinstance(raw, list) or len(raw) % 3 != 0 or len(raw) < 3:
            continue  # 跳过无效数据
        keypoints = np.array(raw).reshape(-1, 3)
        visible = keypoints[:, 2] > 0.1
        if np.sum(visible) == 0:
            continue
        mean_pos = keypoints[visible, :2].mean(axis=0)
        dist = np.linalg.norm(mean_pos - center)
        if dist < min_dist:
            min_dist = dist
            main = person
    return main

def extract_keypoints(person, use=("face", "pose")):
    keypoints = []
    for part in use:
        key = {
            "pose": "pose_keypoints_2d",
            "face": "face_keypoints_2d",
            "hand_left": "hand_left_keypoints_2d",
            "hand_right": "hand_right_keypoints_2d"
        }.get(part)
        if key in person and isinstance(person[key], list) and len(person[key]) % 3 == 0:
            keypoints.append(np.array(person[key]))
    if not keypoints:
        return np.zeros((0, 3))
    return np.concatenate(keypoints).reshape(-1, 3)

def compute_pose_similarity(pose_A, pose_B):
    person_A = select_main_person(pose_A)
    person_B = select_main_person(pose_B)

    if person_A is None or person_B is None:
        return 0.0

    kp1 = extract_keypoints(person_A)
    kp2 = extract_keypoints(person_B)

    valid = (kp1[:, 2] > 0.1) & (kp2[:, 2] > 0.1)
    if np.sum(valid) < 3:
        return 0.0

    v1 = kp1[valid, :2] - kp1[valid, :2].mean(axis=0)
    v2 = kp2[valid, :2] - kp2[valid, :2].mean(axis=0)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0

    v1 /= norm1
    v2 /= norm2

    similarity = float(np.dot(v1.flatten(), v2.flatten()))
    return similarity

class KeypointsSimilarity(ComfyNodeABC):
    DESCRIPTION = "Given the keypoints, calculate the similarity of them, i.e., whether they are toward the same direction. "
    CATEGORY = 'Saul'

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            'required': {
                'kpts_1': (IO.ANY, {}),
                'kpts_2': (IO.ANY, {}),
            }
        }
    
    RETURN_NAMES = ('similarity', )
    RETURN_TYPES = (IO.ANY, )
    FUNCTION = 'main'

    def main(self, kpts_1, kpts_2):
        similarity = compute_pose_similarity(kpts_1, kpts_2)
        print(similarity)
        return similarity, 
    
